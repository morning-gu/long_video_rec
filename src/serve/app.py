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
from src.recall.lightgcn import LightGCNRecall
from src.recall.merge import merge
from src.recall.sasrec import SASRecRecall
from src.recall.semantic import SemanticRecall
from src.recall.tiger import TigerRecall
from src.recall.twotower import TwoTowerRecall
from src.rerank.dpp import dpp_order
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
        self.itemcf = ItemCFRecall.load()
        self.tt = TwoTowerRecall.load()
        self.sas = SASRecRecall.load()
        self.hot = HotRecall(hot_df)
        self.coarse = CoarseRank(tt_item_embs=self.tt.item_embs, hot=self.hot,
                                 movie_ids=movie_ids)
        self.fine = FineRank.load("v1")
        # M6：精排 v2（+内容特征），产物存在才加载；v1 保留可切换
        self.fine2 = (FineRank.load("v2")
                      if (config.ART_DIR / "fine_v2.pt").exists() else None)
        self.rankers = ["v1"] + (["v2"] if self.fine2 else [])
        self.llm = LLMService()
        # M8：本地 LoRA ranker（LLM_RANKER_BACKEND=local 且 adapter 存在；
        # 加载失败静默回退 API，主链路零依赖不变）
        self.llm_local = None
        if config.LLM_RANKER_BACKEND == "local" and (
                ROOT_ADAPTER := config.ROOT / "data" / "lora-adapter"
        ).exists():
            try:
                from src.llm.local_ranker import LocalLLMRanker
                self.llm_local = LocalLLMRanker(config.LORA_BASE_MODEL,
                                                str(ROOT_ADAPTER))
                print("[llm] 本地 LoRA ranker 已加载（rerank 走本地，理由走 API）")
            except Exception as e:
                print(f"[llm] 本地 ranker 加载失败，回退 API：{e}")
        # M7：TIGER 生成式召回 + RQ-VAE（新片冷启动编码）+ 蒸馏粗排 student
        self.tiger = (TigerRecall.load()
                      if (config.ART_DIR / "tiger.pt").exists()
                      and (config.ART_DIR / "sem_ids.npy").exists() else None)
        self.rqvae = None
        if (config.ART_DIR / "rqvae.pt").exists():
            from src.data.semantic_id import RQVAE
            from src.device import get_device
            dim = self.tt.item_embs.shape[1] + (
                np.load(config.ART_DIR / "content_emb.npy").shape[1]
                if (config.ART_DIR / "content_emb.npy").exists() else 0)
            rq = RQVAE(dim)
            rq.load_state_dict(torch.load(
                config.ART_DIR / "rqvae.pt", map_location=str(get_device()),
                weights_only=True))
            self.rqvae = rq.eval().to(get_device())
        if (config.ART_DIR / "distill.pt").exists():
            from src.rank.distill import DistillStudent
            from src.device import get_device
            st = DistillStudent()
            st.load_state_dict(torch.load(
                config.ART_DIR / "distill.pt", map_location=str(get_device()),
                weights_only=True))
            self.coarse.student = st.eval().to(get_device())
        self._new_item_seq = 9000            # 新片 ID 段（避开 ML-1M 1..3952）
        # M6 新增通道（产物存在才加载；旧通道不受影响，可运行时切换对比）
        self.lightgcn = (LightGCNRecall.load()
                         if (config.ART_DIR / "lg_user_emb.npy").exists() else None)
        self.semantic = (SemanticRecall.load()
                         if (config.ART_DIR / "content_emb.npy").exists() else None)
        self.channels_available = ["itemcf", "twotower", "sasrec", "hot"] + (
            ["lightgcn"] if self.lightgcn else []) + (
            ["semantic"] if self.semantic else []) + (
            ["tiger"] if self.tiger else [])
        self.rerankers = list(config.RERANKERS)
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

    def funnel(self, user_id: int, mode: str = "full", channels=None,
               rerank: str = "mmr", ranker: str = "v1", coarse: str = "keep"):
        """返回 (stages, state)。

        channels: 启用的召回通道列表（None = 全部可用；多选）。
        coarse: "keep"（通道保持压缩）| "tt"（双塔点积+口碑先验，M3 初版）
        | "none"（跳过，融合候选直通精排）。
        ranker: "v1" | "v2" | "none"（跳过精排，粗排候选直通规则重排）。
        rerank: "mmr" | "dpp" | "none"（仅规则重排，无多样性算法）。
        （M6+：各阶段算法可运行时组合切换，旧实现全部保留。）
        """
        state = self.store.get(user_id)
        exclude = state.seen | state.suppress
        active = set(channels) if channels else set(self.channels_available)

        # ① 多路召回 + ② 配额融合
        ch, quotas = {}, {}
        if mode == "full" and state.seeds:
            if "itemcf" in active:
                ch["itemcf"] = self.itemcf.recall(
                    state.seeds, exclude, config.ITEMCF_QUOTA)
                quotas["itemcf"] = config.ITEMCF_QUOTA
            if "twotower" in active:
                ch["twotower"] = self.tt.recall(
                    user_id, state.recent_positives, exclude,
                    config.TWOTOWER_QUOTA)
                quotas["twotower"] = config.TWOTOWER_QUOTA
            if "sasrec" in active:
                ch["sasrec"] = self.sas.recall(
                    state.recent_positives, exclude, config.SASREC_QUOTA)
                quotas["sasrec"] = config.SASREC_QUOTA
            if self.lightgcn is not None and "lightgcn" in active:
                ch["lightgcn"] = self.lightgcn.recall(
                    user_id, exclude, config.LIGHTGCN_QUOTA)
                quotas["lightgcn"] = config.LIGHTGCN_QUOTA
            if self.semantic is not None and "semantic" in active:
                ch["semantic"] = self.semantic.recall(
                    state.recent_positives, exclude, config.SEMANTIC_QUOTA)
                quotas["semantic"] = config.SEMANTIC_QUOTA
            if self.tiger is not None and "tiger" in active:
                ch["tiger"] = self.tiger.recall(
                    state.recent_positives, exclude, config.TIGER_QUOTA)
                quotas["tiger"] = config.TIGER_QUOTA
        if "hot" in active or not ch:             # 热门兜底（通道全关时）
            ch["hot"] = self.hot.top(exclude, config.HOT_QUOTA)
            quotas["hot"] = config.HOT_QUOTA
        merged = merge(ch, quotas)
        stages = {"channels": ch, "recall": merged}

        if mode == "full" and state.seeds:
            # ③ 粗排（keep=通道压缩 | tt=双塔重排 | distill=蒸馏 | none=跳过）
            if coarse == "none":
                stages["coarse"] = list(merged)
            elif coarse in ("tt", "distill"):
                u_emb = self.tt.user_embed(user_id, state.recent_positives)
                stages["coarse"] = self.coarse.rank(merged, coarse, u_emb)
            else:
                stages["coarse"] = self.coarse.rank(merged)
            # ④ 精排（v1=M3；v2=M6 +内容特征；none=跳过）
            if ranker == "none":
                stages["fine"] = list(stages["coarse"])
            else:
                fine_model = self.fine2 if (ranker == "v2" and self.fine2) \
                    else self.fine
                stages["fine"] = fine_model.rank(
                    user_id, state.recent_positives, len(state.seen),
                    stages["coarse"], config.FINE_TOPN)
            # ⑤ 重排：规则（已看/打散/冷门保量）+ 多样性（MMR | DPP | 无）
            ruled = rule_rerank(stages["fine"], self.movie_info, state,
                                self.hot.hot_set, config.ROW_K)
            stages["rules"] = ruled
            if rerank == "dpp":
                final = dpp_order(ruled, self.tt.item_embs, self.tt.mid2row)
            elif rerank == "none":
                final = list(ruled)
            else:
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

    def recommend(self, user_id: int, mode: str = "full", channels=None,
                  rerank: str = "mmr", ranker: str = "v1",
                  coarse: str = "keep") -> dict:
        stages, state = self.funnel(user_id, mode, channels, rerank, ranker,
                                    coarse)
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


@app.get("/api/meta")
def meta():
    """可用算法组合（M6+ 前端切换面板数据源）：召回通道（多选）+
    粗排/精排/重排（各单选，均含"无"做完全消融）。"""
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    return {"channels": REC.channels_available,
            "rerankers": list(config.RERANKERS) + ["none"],
            "rankers": REC.rankers + ["none"],
            "coarse": ["keep", "tt", "distill", "none"]}


@app.get("/api/recommend")
def recommend(user_id: int, mode: str = "full", channels: str = "",
              rerank: str = "mmr", ranker: str = "v1", coarse: str = "keep"):
    """漏斗四阶段算法组合（M6+）：channels 多选；
    rerank: mmr|dpp|none；ranker: v1|v2|none；coarse: keep|none。"""
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    if mode not in ("full", "popular"):
        raise HTTPException(400, "mode must be full|popular")
    if rerank not in (*config.RERANKERS, "none"):
        raise HTTPException(400, f"rerank must be one of "
                                 f"{(*config.RERANKERS, 'none')}")
    if ranker not in (*REC.rankers, "none"):
        raise HTTPException(400, f"ranker must be one of "
                                 f"{(*REC.rankers, 'none')}")
    if coarse not in ("keep", "tt", "distill", "none"):
        raise HTTPException(400, "coarse must be keep|tt|distill|none")
    ch_list = [c.strip() for c in channels.split(",") if c.strip()]
    unknown = [c for c in ch_list if c not in REC.channels_available]
    if unknown:
        raise HTTPException(400, f"unknown channels {unknown}; "
                                 f"available: {REC.channels_available}")
    return REC.recommend(user_id, mode, ch_list or None, rerank, ranker,
                         coarse)


class Event(BaseModel):
    user_id: int
    movie_id: int
    action: str            # click | like | watched | dislike | rate
    value: float | None = None   # rate 动作的 1~5 评分


@app.post("/api/event")
def event(e: Event):
    """演示期动作回流（O1，动作语义注册表见 user_state.py）。

    扩展新动作（如"分享"）：user_state.ALLOWED_ACTIONS 加一行 + _apply 分支。
    """
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    from src.state.user_state import ALLOWED_ACTIONS
    if e.action not in ALLOWED_ACTIONS:
        raise HTTPException(400, f"action must be one of {sorted(ALLOWED_ACTIONS)}")
    if e.action == "rate" and (e.value is None or not 1.0 <= e.value <= 5.0):
        raise HTTPException(400, "rate requires value in [1, 5]")
    REC.store.record(e.user_id, e.movie_id, e.action, e.value)
    return {"ok": True}


@app.get("/api/movie/{movie_id}")
def movie_detail(movie_id: int, user_id: int = 0):
    """电影详情页：元数据 + 用户关系（归因）+ LLM 理由 + ItemCF 相似推荐。

    关系归因（消除"已看过"歧义）：
    - history_rating：MovieLens 历史评分（用户真的看过并评过分）；
    - marked：演示期最近一次动作（喜欢/看完/评分/屏蔽）。
    """
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    info = REC.movie_info.get(movie_id)
    if not info:
        raise HTTPException(404, "movie not found")
    state = REC.store.get(user_id) if user_id else None
    exclude = (state.seen | state.suppress) if state else set()
    exclude.add(movie_id)
    similar = [REC._card({"movie_id": m, "score": s, "source": "itemcf"})
               for m, s in REC.itemcf.neighbors(movie_id, exclude,
                                                config.SUB_ROW_K)]
    marked = REC.store.last_mark(user_id, movie_id) if user_id else None
    return {
        "movie_id": movie_id, "title": info["title"], "genres": info["genres"],
        "poster": REC.posters.get(movie_id, ""),
        "wr": round(REC.hot.wr.get(movie_id, 0.0), 2),
        "pos_count": REC.hot.pos_count.get(movie_id, 0),
        "pop_rank": REC.hot.pop_rank.get(movie_id, len(REC.hot.pop_order)) + 1,
        "history_rating": (REC.store.rating_of(user_id, movie_id)
                           if user_id else None),
        "marked": ({"action": marked[0], "value": marked[1]}
                   if marked else None),
        "reason": (REC.llm.reason_for(user_id, movie_id)
                   if user_id and REC.llm.available else None),
        "similar": similar,
    }


class SynopsisReq(BaseModel):
    movie_id: int


@app.get("/api/profile_card")
def profile_card(user_id: int):
    """用户口味画像卡（M7）：LLM 总结观影历史 → 标签 + 一句话。"""
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    if not REC.llm.available:
        return {"tags": [], "summary": "", "degraded": True}
    state = REC.store.get(user_id)
    hist = []
    for m in reversed(state.recent_positives[-10:]):
        info = REC.movie_info.get(m)
        if info:
            hist.append((info["title"], "/".join(info["genres"][:3])))
    if not hist:
        return {"tags": [], "summary": "", "degraded": True}
    card = REC.llm.profile_card(user_id, hist)
    return {"tags": card["tags"] if card else [],
            "summary": card["summary"] if card else "",
            "degraded": card is None}


class NewMovieReq(BaseModel):
    user_id: int
    title: str
    text: str = ""


@app.post("/api/new_movie")
def new_movie(req: NewMovieReq):
    """新片上架（M7 冷启动演示）：LLM 画像 → 内容向量 → RQ-VAE 语义 ID →
    注册进 TIGER 生成词表 + 语义召回空间 → 零交互检查能否被当前用户召回。

    语义 ID 的核心卖点：新片无需任何交互数据即可进入生成式召回词表
    （传统 ID embedding 需要交互才能学到表示）。
    """
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    title = req.title.strip()[:60]
    if not title:
        raise HTTPException(400, "title required")
    if REC.rqvae is None or REC.tiger is None:
        raise HTTPException(503, "TIGER/RQ-VAE 产物未构建")

    # ① LLM 内容画像（虚构片名走推断模式；用户描述作为线索并入，
    #    由 LLM 产出丰富 desc——thin 文本会导致语义向量质量差）
    prof = (REC.llm.movie_profile(title, "", fictional=True, hint=req.text)
            if REC.llm.available else None) or {}
    genres = prof.get("genres", [])
    mood, era = prof.get("mood", ""), prof.get("era", "")
    desc = prof.get("desc", "") or req.text.strip()[:60]

    # ② 内容向量（已拟合模型同空间投影）→ ③ RQ-VAE 语义 ID
    from src.data.content import embed_new_movie
    from src.data.semantic_id import encode_new
    cvec = embed_new_movie(title, genres, mood, era, desc)
    tt_block = REC.tt.item_embs.mean(0)          # 新片无协同向量，用均值先验
    codes = encode_new(REC.rqvae, np.concatenate([tt_block, cvec]))

    # 内容近邻（表示正确性的直接证据：近邻应全为同类型片）
    neighbors = []
    if REC.semantic is not None:
        sims = REC.semantic.vectors @ cvec
        for r in np.argsort(-sims)[:3]:
            m = int(REC.semantic.movie_ids[r])
            info = REC.movie_info.get(m, {})
            neighbors.append({"title": info.get("title", str(m)),
                              "sim": round(float(sims[r]), 3)})

    # ④ 注册（TIGER 生成词表 + 语义召回空间 + 元数据）
    REC._new_item_seq += 1
    mid = REC._new_item_seq
    REC.movie_info[mid] = {"title": title, "genres": genres}
    REC.tiger.register(mid, codes)
    if REC.semantic is not None:
        REC.semantic.register(mid, cvec)

    # ⑤ 零交互召回检查（阈值=通道召回配额；另附全库排名——语义通道判别力
    #    有限（HR 0.034），命中与否都如实展示，近邻证明表示正确性）
    state = REC.store.get(req.user_id)
    exclude = state.seen | state.suppress
    t_rec = REC.tiger.recall(state.recent_positives, exclude, 100)
    t_rank = next((i for i, (m, _) in enumerate(t_rec) if m == mid), -1)
    s_rank, s_full = -1, None
    if REC.semantic is not None:
        p = REC.semantic.user_profile(state.recent_positives)
        if p is not None:
            s_full = int((REC.semantic.vectors @ p > cvec @ p).sum()) + 1
        s_rec = REC.semantic.recall(state.recent_positives, exclude,
                                    config.SEMANTIC_QUOTA)
        s_rank = next((i for i, (m, _) in enumerate(s_rec) if m == mid), -1)
    return {
        "movie_id": mid, "title": title, "genres": genres, "mood": mood,
        "desc": desc, "codes": [int(c) for c in codes],
        "neighbors": neighbors,
        "tiger_rank": t_rank + 1 if t_rank >= 0 else None,
        "semantic_rank": s_rank + 1 if s_rank >= 0 else None,
        "semantic_full_rank": s_full,
        "recalled": t_rank >= 0 or s_rank >= 0,
        "tiger_top": [{"title": REC.movie_info[m]["title"], "score": round(s, 3)}
                      for m, s in t_rec[:5]],
    }


@app.post("/api/synopsis")
def synopsis(req: SynopsisReq):
    """AI 简介（详情页兜底槽位）：无推荐理由时展示；按电影缓存，
    不确定的影片返回 None（前端隐藏槽位，宁缺毋滥）。"""
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    info = REC.movie_info.get(req.movie_id)
    if not info:
        raise HTTPException(404, "movie not found")
    if not REC.llm.available:
        return {"synopsis": None}
    return {"synopsis": REC.llm.synopsis(req.movie_id, info["title"],
                                         info["genres"])}


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
def compare(user_id: int, channels: str = "", rerank: str = "mmr",
            ranker: str = "v1", coarse: str = "keep"):
    """同屏对比（M5）：完整链路 vs 纯热门 + 实时列表指标（§6 消融叙事）。

    channels/rerank/ranker/coarse 参数与 /api/recommend 一致（M6+）。
    """
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    ch_list = [c.strip() for c in channels.split(",") if c.strip()] or None
    full_stages, full_state = REC.funnel(user_id, "full", ch_list, rerank,
                                         ranker, coarse)
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
    # M8：rerank 走本地 LoRA（若已加载）——同一任务的"微调 vs 提示词"线上对照
    order = None
    degraded = True
    if REC.llm_local is not None:
        try:
            hist_titles = [REC.movie_info.get(m, {}).get("title", str(m))
                           for m in reversed(state.recent_positives[-10:])]
            order = REC.llm_local.rerank(hist_titles, movies)
            degraded = False
        except Exception:
            order = None
    if order is None and REC.llm.available:
        order, deg = REC.llm.enhance_order_only(req.user_id, movies, profile)
        degraded = deg
    reasons = {}
    if REC.llm.available:
        final_ids = order or [m["movie_id"] for m in movies]
        reasons, _ = REC.llm.reasons_only(req.user_id, movies, profile,
                                          final_ids[:12])
    return {"order": order, "reasons": reasons, "degraded": degraded}


@app.get("/api/llm_failures")
def llm_failures(limit: int = 20):
    """LLM 降级原因审计（R2）：最近 N 条失败记录（时间/环节/原因）。

    排查"AI 为什么降级"用；正常应为空。
    """
    if REC is None or not REC.llm.available:
        return {"failures": []}
    return {"failures": REC.llm.recent_failures(limit)}


@app.get("/api/users")
def users():
    """高活跃样本用户（下拉框数据源，默认前端只展示前 5 个，可搜索）。"""
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    return {"sample_users": [
        {"user_id": int(u), "positives": int(n)}
        for u, n in REC.pos_count.head(50).items()]}


@app.post("/api/users")
def create_user():
    """创建新用户（冷启动演示）：ID 从 9001 起，避开 ML-1M 存量段 1..6040。

    新用户无历史 → 热门兜底；经详情页喜欢/评分后由用户状态实时层即时个性化。
    """
    if REC is None:
        raise HTTPException(503, "artifacts loading")
    row = REC.store.db.execute(
        "SELECT MAX(user_id) FROM events WHERE user_id > 9000").fetchone()
    REC._next_new_user = max(getattr(REC, "_next_new_user", 9000),
                             row[0] or 9000) + 1
    return {"user_id": REC._next_new_user}


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
