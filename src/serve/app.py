"""FastAPI 在线服务：完整四段漏斗（M3）。

    多路召回（ItemCF/双塔/SASRec/热门）→ 配额融合 → 粗排（双塔点积+口碑先验）
    → 精排（DeepFM-lite）→ 规则重排 + MMR 多样性

端点：
- GET  /api/recommend?user_id=&mode=full|popular   行式推荐（mode 即 O3 消融开关）
- POST /api/event                                   点击/负反馈回流（O1，下次请求生效）
- GET  /api/users                                   演示用高活跃样本用户列表

双塔 user tower、SASRec 序列编码、精排特征（序列相似度/类型偏好）均为请求时
在线计算（O1，§4.6），模型权重离线训练（D8）。popular 模式 = 非个性化路径
（纯热门 → 规则重排），与 full 模式构成消融对照（O3）。
"""
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src import config
from src.llm.service import LLMService
from src.rank.coarse import CoarseRank
from src.rank.fine import FineRank
from src.recall.hot import HotRecall
from src.recall.itemcf import ItemCFRecall
from src.recall.merge import merge
from src.recall.sasrec import SASRecRecall
from src.recall.twotower import TwoTowerRecall
from src.rerank.mmr import mmr_order
from src.rerank.rules import rule_rerank
from src.state.user_state import UserStateStore


class Recommender:
    """封装在线漏斗与全部内存态（启动时一次性加载产物）。

    state_db=None 用默认 demo.db；评测隔离可传 ":memory:"。
    """

    def __init__(self, state_db=None, ratings=None):
        """state_db=None 用默认 demo.db；评测隔离可传 ":memory:"；
        ratings 可注入训练期评分（评测协议：剔除测试交互，测试目标是"未来"）。"""
        if ratings is None:
            ratings = pd.read_parquet(config.ART_DIR / "ratings.parquet")
        movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
        hot_df = pd.read_parquet(config.ART_DIR / "hot.parquet")
        movie_ids = movies.movie_id.values

        self.movie_info = {
            int(r.movie_id): {
                "title": r.title,
                "genres": r.genres.split("|") if isinstance(r.genres, str) else [],
            }
            for r in movies.itertuples()
        }
        self.itemcf = ItemCFRecall(
            np.load(config.ART_DIR / "itemcf_sim.npy"), movie_ids)
        self.tt = TwoTowerRecall.load()
        self.sas = SASRecRecall.load()
        self.hot = HotRecall(hot_df)
        self.coarse = CoarseRank()
        self.fine = FineRank.load()
        self.llm = LLMService()
        # 海报（IMDb 源，scripts/enrich_posters.py 产出；缺图前端回退渐变占位）
        self.posters = {}
        posters_path = config.ART_DIR / "posters.parquet"
        if posters_path.exists():
            df = pd.read_parquet(posters_path)
            self.posters = dict(zip(df.movie_id.astype(int),
                                    df.poster_url.astype(str)))
        self.store = UserStateStore(ratings, db_path=state_db)
        train_pos = pd.read_parquet(config.ART_DIR / "train_pos.parquet")
        self.pos_count = (train_pos.groupby("user_id").size()
                          .sort_values(ascending=False))

    # ---- 四段漏斗 ----

    def funnel(self, user_id: int, mode: str = "full"):
        """返回 (stages, state)。stages: channels / recall / coarse / fine /
        rules / final 各阶段候选列表（popular 模式只走 非个性化路径）。"""
        state = self.store.get(user_id)
        exclude = state.seen | state.suppress

        # ① 多路召回 + ② 配额融合
        channels, quotas = {}, {}
        if mode == "full" and state.seeds:
            channels["itemcf"] = self.itemcf.recall(
                state.seeds, exclude, config.ITEMCF_QUOTA)
            quotas["itemcf"] = config.ITEMCF_QUOTA
            channels["twotower"] = self.tt.recall(
                user_id, state.recent_positives, exclude, config.TWOTOWER_QUOTA)
            quotas["twotower"] = config.TWOTOWER_QUOTA
            channels["sasrec"] = self.sas.recall(
                state.recent_positives, exclude, config.SASREC_QUOTA)
            quotas["sasrec"] = config.SASREC_QUOTA
        channels["hot"] = self.hot.top(exclude, config.HOT_QUOTA)
        quotas["hot"] = config.HOT_QUOTA
        merged = merge(channels, quotas)
        stages = {"channels": channels, "recall": merged}

        if mode == "full" and state.seeds:
            # ③ 粗排：通道保持式压缩（M3 评测驱动修订，见 coarse.py 注释）
            stages["coarse"] = self.coarse.rank(merged)
            # ④ 精排（DeepFM + 序列信号分数级融合，含交叉特征）
            stages["fine"] = self.fine.rank(
                user_id, state.recent_positives, len(state.seen),
                stages["coarse"], config.FINE_TOPN)
            # ⑤ 重排：规则（已看/打散/冷门保量）+ MMR 多样性
            ruled = rule_rerank(stages["fine"], self.movie_info, state,
                                self.hot.hot_set, config.ROW_K)
            stages["rules"] = ruled
            final = mmr_order(ruled, self.tt.item_embs,
                              self.tt.mid2row, config.MMR_LAMBDA)
            # 展示分：融合分数量纲大（γ·logit 过 sigmoid 饱和），
            # 列表内按秩归一（score 此后仅用于展示，不影响任何逻辑）
            for i, c in enumerate(final):
                c["score"] = round(1.0 - i / max(len(final) - 1, 1), 4)
            stages["final"] = final
        else:
            stages["final"] = rule_rerank(merged, self.movie_info, state,
                                          self.hot.hot_set, config.ROW_K)
        return stages, state

    # ---- 行式推荐 ----

    def recommend(self, user_id: int, mode: str = "full") -> dict:
        stages, state = self.funnel(user_id, mode)
        exclude = state.seen | state.suppress
        note = "" if (mode == "popular" or state.seeds) else "冷启动用户：热门兜底"
        rows = [{"key": "for_you", "title": "为你推荐", "note": note,
                 "movies": [self._card(c) for c in stages["final"]]}]

        # 次行：因为你看过 X（ItemCF 单种子邻居，种子来自实时状态层）
        if state.seeds:
            seed = state.seeds[0]
            title = self.movie_info.get(seed, {}).get("title", str(seed))
            nb = self.itemcf.neighbors(seed, exclude, config.SUB_ROW_K)
            rows.append({
                "key": "because", "title": f"因为你看过《{title}》", "note": "",
                "movies": [self._card({"movie_id": m, "score": s, "source": "itemcf"})
                           for m, s in nb]})

        # 次行：热门榜单 / 冷门佳片
        rows.append({
            "key": "hot", "title": "热门榜单", "note": "",
            "movies": [self._card({"movie_id": m, "score": s, "source": "hot"})
                       for m, s in self.hot.top(exclude, config.SUB_ROW_K)]})
        rows.append({
            "key": "cold", "title": "冷门佳片", "note": "高口碑小众",
            "movies": [self._card({"movie_id": m, "score": s, "source": "cold"})
                       for m, s in self.hot.cold_top(exclude, config.SUB_ROW_K)]})

        return {
            "user_id": user_id, "mode": mode, "rows": rows,
            "profile": {"recent": [
                self.movie_info.get(m, {}).get("title", str(m))
                for m in reversed(state.recent_positives[-8:])]},
        }

    def _card(self, c: dict) -> dict:
        info = self.movie_info.get(
            c["movie_id"], {"title": f"movie {c['movie_id']}", "genres": []})
        return {"movie_id": c["movie_id"], "title": info["title"],
                "genres": info["genres"], "score": round(c["score"], 4),
                "source": c.get("source", ""),
                "poster": self.posters.get(c["movie_id"], "")}


REC: Recommender | None = None


@asynccontextmanager
async def lifespan(_app):
    global REC
    # 服务进程 torch 单线程：本机 2 核，小模型在线前向若启用多线程，
    # 会与 uvicorn 线程池/事件循环争抢核心，实测请求延迟从 ~20ms 恶化到秒级。
    torch.set_num_threads(1)
    REC = Recommender()
    yield


app = FastAPI(title="长视频推荐 Demo (M3)", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/api/recommend")
def recommend(user_id: int, mode: str = "full"):
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    if mode not in ("full", "popular"):
        raise HTTPException(400, "mode must be full|popular")
    return REC.recommend(user_id, mode)


class Event(BaseModel):
    user_id: int
    movie_id: int
    action: str    # click | dislike


class LLMEnhanceReq(BaseModel):
    user_id: int
    movie_ids: list


class QueryReq(BaseModel):
    user_id: int
    text: str


@app.post("/api/query")
def query(req: QueryReq):
    """自然语言查询（§4.5，M5）：LLM 解析 → 类型过滤 + 双塔个性化检索。

    LLM 解析失败/未配置 → 退化为热度检索（degraded=True）。
    """
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    text = req.text.strip()[:60]
    if not text:
        return {"text": "", "genres": [], "mood": "", "degraded": True,
                "movies": []}
    parsed = REC.llm.parse_query(text) if REC.llm.available else None
    state = REC.store.get(req.user_id)
    exclude = state.seen | state.suppress
    genres = parsed["genres"] if parsed else []

    if genres:                                   # 类型过滤（catalog 白名单）
        gset = set(genres)
        pool = [mid for mid, info in REC.movie_info.items()
                if mid not in exclude and (set(info["genres"]) & gset)]
    else:                                        # 解析失败：热度池兜底
        pool = [m for m in REC.hot.pop_order if m not in exclude][:500]
    if not pool:
        return {"text": text, "genres": genres,
                "mood": parsed["mood"] if parsed else "",
                "degraded": parsed is None, "movies": []}

    rows = np.array([REC.tt.mid2row[m] for m in pool])
    wr_arr = np.array([REC.hot.wr.get(m, 3.0) for m in pool],
                      dtype=np.float32) / 5.0
    if state.seeds:                              # 双塔个性化相似度（O1 在线前向）
        u = REC.tt.user_embed(req.user_id, state.recent_positives)
        sims = REC.tt.item_embs[rows] @ u
        sims = (sims + 1.0) / 2.0                # [-1,1] → [0,1]
    else:
        sims = np.full(len(pool), 0.5, dtype=np.float32)
    scores = 0.6 * sims + 0.4 * wr_arr
    order = np.argsort(-scores)[:config.SUB_ROW_K + 2]
    return {
        "text": text, "genres": genres,
        "mood": parsed["mood"] if parsed else "",
        "degraded": parsed is None,
        "movies": [REC._card({"movie_id": pool[i], "score": float(scores[i]),
                              "source": "query"}) for i in order]}


@app.get("/api/compare")
def compare(user_id: int):
    """同屏对比（M5）：完整链路 vs 纯热门 + 实时列表指标（§6 消融叙事）。"""
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    full_stages, full_state = REC.funnel(user_id, "full")
    pop_stages, _ = REC.funnel(user_id, "popular")
    full, pop = full_stages["final"], pop_stages["final"]

    def metrics(lst):
        top10 = lst[:10]
        genres = {g for c in top10
                  for g in REC.movie_info.get(c["movie_id"], {}).get("genres", [])}
        wrs = [REC.hot.wr.get(c["movie_id"], 3.0) for c in lst]
        return {
            "genre_diversity": len(genres),
            "longtail": round(sum(1 for c in lst
                                  if c["movie_id"] not in REC.hot.hot_set)
                              / max(len(lst), 1), 4),
            "avg_wr": round(sum(wrs) / max(len(wrs), 1), 3),
        }

    overlap = len({c["movie_id"] for c in full} & {c["movie_id"] for c in pop})
    return {
        "full": [REC._card(c) for c in full],
        "popular": [REC._card(c) for c in pop],
        "metrics": {"full": metrics(full), "popular": metrics(pop),
                    "overlap": overlap},
        "profile": {"recent": [
            REC.movie_info.get(m, {}).get("title", str(m))
            for m in reversed(full_state.recent_positives[-8:])]},
    }


@app.get("/api/eval")
def eval_data():
    """评测展示页数据（M5）：读取 scripts/eval.py 产出的结构化指标。"""
    import json
    p = config.ART_DIR / "eval_report.json"
    if not p.exists():
        raise HTTPException(504, "评测数据未生成：请先运行 python scripts/eval.py")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/eval")
def eval_page():
    return FileResponse(config.WEB_DIR / "eval.html")


@app.post("/api/llm_enhance")
def llm_enhance(req: LLMEnhanceReq):
    """LLM 增强端点（§4.5）：对前端当前展示的 Top-20 做重排 + Top-12 理由。

    异步于主链路（前端渲染后调用）；LLM 不可用/超时/解析失败 → degraded=True，
    前端保持原序，主链路零依赖（R2/R4）。电影信息由服务端补全（含实时状态）。
    """
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    if not REC.llm.available:
        return {"order": None, "reasons": {}, "degraded": True}
    movies = []
    for mid in req.movie_ids[:20]:
        info = REC.movie_info.get(int(mid))
        if info:
            movies.append({"movie_id": int(mid), "title": info["title"],
                           "genres": info["genres"]})
    if not movies:
        return {"order": None, "reasons": {}, "degraded": True}
    state = REC.store.get(req.user_id)
    profile = [REC.movie_info.get(m, {}).get("title", str(m))
               for m in reversed(state.recent_positives[-8:])]
    order, reasons, degraded = REC.llm.enhance(
        req.user_id, movies, profile)
    return {"order": order, "reasons": reasons, "degraded": degraded}


@app.post("/api/event")
def event(e: Event):
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    if e.action not in ("click", "dislike"):
        raise HTTPException(400, "action must be click|dislike")
    REC.store.record(e.user_id, e.movie_id, e.action)
    return {"ok": True}


@app.get("/api/users")
def users():
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    return {"sample_users": [
        {"user_id": int(u), "positives": int(n)}
        for u, n in REC.pos_count.head(20).items()]}


@app.post("/api/reload_posters")
def reload_posters():
    """海报表热加载：后台增强脚本产出后无需重启服务即可生效。"""
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    posters_path = config.ART_DIR / "posters.parquet"
    if posters_path.exists():
        df = pd.read_parquet(posters_path)
        REC.posters = dict(zip(df.movie_id.astype(int),
                                df.poster_url.astype(str)))
    return {"posters": len(REC.posters)}
