"""LLM 增强层（设计文档 §4.5，O2 预算无上限）。

- LLM Rerank：Top-20 候选 + 用户画像摘要 → 结构化 JSON 重排；
  **候选集白名单约束**——order 必须是候选 id 的全排列，否则丢弃（防幻觉，
  对应 GenRec 的 catalog-aware scoring，文档 03 文档 3 §3.7）；
- 推荐理由：Top-12 每部一句中文推荐理由；
- 缓存：SQLite llm_cache（O2 后定位为延迟优化而非成本控制）；
- 降级：任何失败（超时/解析/校验）静默返回 degraded=True，主链路零依赖（R2/R4）；
- 线程安全：FastAPI 同步端点在线程池中调用，SQLite check_same_thread=False，
  LLM 客户端无状态。
"""
import hashlib
import json
import sqlite3
import threading
import time

from src import config

# MovieLens-1M 全部 18 个类型（查询解析的目标词表，LLM 须映射到此集合）
KNOWN_GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                "Thriller", "War", "Western"]


def _extract_json(text: str):
    """从模型输出中鲁棒提取第一个 JSON 对象（容忍思考文本/前后缀）。"""
    text = text.strip()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


class LLMService:
    def __init__(self, db_path=None):
        self.available = bool(config.LLM_API_KEY)
        self.client = None
        self.model = config.LLM_MODEL
        self._lock = threading.Lock()
        if not self.available:
            return
        from openai import OpenAI
        self.client = OpenAI(api_key=config.LLM_API_KEY,
                             base_url=config.LLM_BASE_URL,
                             timeout=config.LLM_TIMEOUT)
        self.db = sqlite3.connect(db_path or config.DB_PATH,
                                  check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS llm_cache ("
                        "k TEXT PRIMARY KEY, v TEXT, ts INTEGER)")
        self.db.commit()

    # ---- 基础调用 ----

    def _chat(self, prompt: str) -> str:
        """单次对话调用；qwen3 系列关闭思考模式以降低时延。

        仅当 enable_thinking 参数不被端点支持（4xx 参数错误/TypeError）时
        才去掉参数重试；超时/网络错误直接抛给上层降级——否则最坏 2×25s
        双超时放大（本机慢网络下曾被观测到）。
        """
        kwargs = dict(model=self.model,
                      messages=[{"role": "user", "content": prompt}],
                      temperature=0.3)
        try:
            r = self.client.chat.completions.create(
                **kwargs, extra_body={"enable_thinking": False})
        except TypeError:
            r = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) in (400, 404, 422):
                r = self.client.chat.completions.create(**kwargs)
            else:
                raise
        return r.choices[0].message.content or ""

    def _log_fail(self, purpose: str, err) -> None:
        """降级原因落库（llm_failures 表）：事后可查"AI 为什么降级"。"""
        try:
            with self._lock:
                self.db.execute("CREATE TABLE IF NOT EXISTS llm_failures ("
                                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                "ts INTEGER, purpose TEXT, err TEXT)")
                self.db.execute(
                    "INSERT INTO llm_failures (ts, purpose, err) VALUES (?,?,?)",
                    (int(time.time()), purpose, str(err)[:300]))
                self.db.commit()
        except Exception:
            pass

    def recent_failures(self, limit: int = 20) -> list:
        """最近的降级记录（新→旧）。"""
        import datetime
        try:
            with self._lock:
                rows = self.db.execute(
                    "SELECT ts, purpose, err FROM llm_failures "
                    "ORDER BY id DESC LIMIT ?",
                    (min(limit, 100),)).fetchall()
        except Exception:
            return []
        return [{"time": datetime.datetime.fromtimestamp(t).strftime(
                    "%m-%d %H:%M:%S"),
                 "purpose": p, "error": e} for t, p, e in rows]

    def _cached(self, key: str):
        with self._lock:
            row = self.db.execute(
                "SELECT v FROM llm_cache WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, key: str, value) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?)",
                (key, json.dumps(value, ensure_ascii=False), int(time.time())))
            self.db.commit()

    @staticmethod
    def _key(purpose: str, user_id: int, mids: list) -> str:
        return hashlib.sha256(
            f"{purpose}|{config.LLM_MODEL}|{user_id}|{','.join(map(str, mids))}"
            .encode()).hexdigest()

    # ---- 增强入口 ----

    def enhance(self, user_id: int, movies: list, profile_titles: list):
        """movies: [{movie_id, title, genres}] 按当前顺序（Top-20）。

        返回 (order, reasons, degraded)：
        - order: 重排后的 movie_id 列表（校验失败时为 None，沿用原序）；
        - reasons: {movie_id: 一句话理由}（仅 Top-12）；
        - degraded: 任一环节失败为 True。
        """
        if not self.available or not movies:
            return None, {}, True
        order, degraded = self._rerank(user_id, movies, profile_titles)
        final_ids = order or [m["movie_id"] for m in movies]
        reasons, deg2 = self._reasons(user_id, movies, profile_titles,
                                      final_ids[:12])
        return order, reasons, (degraded or deg2)

    # ---- 自然语言查询（§4.5，M5）----

    def parse_query(self, text: str):
        """「来部轻松的科幻片」→ {"genres": ["Sci-Fi","Comedy"], "mood": "轻松"}。

        类型映射到 KNOWN_GENRES 白名单（catalog 约束，防幻觉）；失败返回 None，
        由调用方退化为热度检索。
        """
        key = self._key("query", 0, [text])
        cached = self._cached(key)
        if cached is not None:
            return cached if cached.get("genres") else None
        prompt = (
            "把用户的观影需求解析为结构化 JSON。\n\n"
            "可用类型（只能从中选择）：\n" +
            "、".join(KNOWN_GENRES) + "\n\n"
            f"用户输入：「{text}」\n\n"
            '输出 JSON：{"genres": ["..."], "mood": "...", "era": ""}\n'
            "- genres：1~3 个最匹配的可用类型（英文原名）；\n"
            "- mood：用户情绪/氛围关键词（中文，≤6 字）；\n"
            "- era：年代偏好如 \"90年代\"，无则空字符串。\n"
            "只输出 JSON。")
        try:
            obj = _extract_json(self._chat(prompt))
            genres = [g for g in (obj.get("genres") or [])
                      if g in KNOWN_GENRES]
            mood = str(obj.get("mood", ""))[:12]
            if not genres:
                self._log_fail("query",
                               f"类型白名单过滤后为空: {str(obj.get('genres'))[:150]}")
            self._put(key, {"genres": genres, "mood": mood})
            return {"genres": genres, "mood": mood}
        except Exception as e:
            self._log_fail("query", f"{type(e).__name__}: {e}")
            return None

    def _rerank(self, user_id: int, movies: list, profile_titles: list):
        mids = [m["movie_id"] for m in movies]
        key = self._key("rerank", user_id, mids)
        cached = self._cached(key)
        if cached is not None:
            return cached["order"], False
        lines = [f"- 《{m['title']}》（{'/'.join(m['genres'][:3])}）"
                 for m in movies[:20]]
        cand = [f"- {m['movie_id']} | {m['title']} | "
                f"{'/'.join(m['genres'][:3])}" for m in movies[:20]]
        prompt = (
            "你是长视频平台首页推荐的重排助手，根据用户画像对候选电影重排。\n\n"
            "【用户最近喜欢】（新→旧）\n" +
            "\n".join(f"- {t}" for t in profile_titles[:8]) + "\n\n"
            "【候选电影】（当前排序）\n" + "\n".join(cand) + "\n\n"
            "【要求】\n"
            "1. 只能使用候选列表中已有的 id，不得编造；\n"
            "2. 最符合用户近期口味、且与其近期看过的内容形成互补的排前面；\n"
            "3. 适当打散：避免前 5 名全是同一类型；\n"
            '4. 只输出 JSON：{"order": [id1, id2, ...]}，'
            "必须包含全部候选 id 且每个恰好一次。")
        try:
            obj = _extract_json(self._chat(prompt))
            order = obj.get("order") if obj else None
            if (isinstance(order, list) and len(order) == len(movies)
                    and sorted(map(int, order)) == sorted(mids)):
                order = [int(x) for x in order]
                self._put(key, {"order": order})
                return order, False
            self._log_fail("rerank", "校验失败: " + (
                "响应无 JSON" if obj is None else f"order={str(order)[:150]}"))
        except Exception as e:
            self._log_fail("rerank", f"{type(e).__name__}: {e}")
        return None, True

    def _reasons(self, user_id: int, movies: list, profile_titles: list,
                 final_ids: list):
        by_id = {m["movie_id"]: m for m in movies}
        mids = [i for i in final_ids if i in by_id]
        if not mids:
            return {}, True
        key = self._key("reason", user_id, mids)
        cached = self._cached(key)
        if cached is not None:
            reasons = {int(k): v for k, v in cached["reasons"].items()}
            # 兼容旧缓存：回填按片理由（详情页 reason_for 取用）
            for mid, text in reasons.items():
                self._put(self._key("reason1", user_id, [mid]),
                          {"reason": text})
            return reasons, False
        cand = [f"- {i} | {by_id[i]['title']} | "
                f"{'/'.join(by_id[i]['genres'][:3])}" for i in mids]
        prompt = (
            "为推荐列表中的每部电影写一句中文推荐理由（15~25 字），"
            "自然具体，可引用用户最近看过的电影或其偏好类型。\n\n"
            "【用户最近看过】\n" +
            "\n".join(f"- {t}" for t in profile_titles[:8]) + "\n\n"
            "【电影列表】\n" + "\n".join(cand) + "\n\n"
            '只输出 JSON：{"reasons": {"<电影id>": "<理由>", ...}}，'
            "key 必须是上述电影 id。")
        try:
            obj = _extract_json(self._chat(prompt))
            raw = obj.get("reasons") if obj else None
            if isinstance(raw, dict):
                reasons = {int(k): str(v)[:60]
                           for k, v in raw.items()
                           if str(k).isdigit() and int(k) in set(mids)}
                if reasons:
                    self._put(key, {"reasons":
                                    {str(k): v for k, v in reasons.items()}})
                    # 额外按 (用户, 影片) 落一份：详情页可直接取用（§4.5）
                    for mid, text in reasons.items():
                        self._put(self._key("reason1", user_id, [mid]),
                                  {"reason": text})
                    return reasons, False
                self._log_fail("reason",
                               f"校验失败: 无有效 id，raw={str(raw)[:150]}")
            else:
                self._log_fail("reason",
                               f"响应无 reasons 字段: {str(obj)[:150]}")
        except Exception as e:
            self._log_fail("reason", f"{type(e).__name__}: {e}")
        return {}, True

    # ---- 详情页：理由取用 / AI 简介 ----

    def reason_for(self, user_id: int, movie_id: int):
        """该用户该影片的推荐理由（由 _reasons 生成时按片落库）。"""
        if not self.available:
            return None
        cached = self._cached(self._key("reason1", user_id, [movie_id]))
        return cached.get("reason") if cached else None

    def profile_card(self, user_id: int, history: list):
        """用户口味画像卡（M7）：观影历史 → 标签 + 一句话总结。

        history: [(title, genres_str)] 最近若干部（新→旧）；缓存键含最后一部，
        历史变化即失效。失败返回 None（前端隐藏）。
        """
        if not self.available or not history:
            return None
        key = self._key("pcard", user_id, [history[0][0], len(history)])
        cached = self._cached(key)
        if cached is not None:
            return cached if cached.get("tags") else None
        lines = [f"- 《{t}》（{g}）" for t, g in history[:10]]
        prompt = (
            "根据用户最近的观影记录，总结其口味画像。\n\n"
            "【最近看过】（新→旧）\n" + "\n".join(lines) + "\n\n"
            '只输出 JSON：{"tags": ["2~4 个中文标签，每个不超过 6 字"],'
            ' "summary": "一句话总结其观影口味（不超过 40 字）"}')
        try:
            obj = _extract_json(self._chat(prompt))
            tags = [str(t)[:8] for t in (obj.get("tags") or [])][:4]
            summary = str(obj.get("summary", ""))[:60]
            if not tags:
                return None
            out = {"tags": tags, "summary": summary}
            self._put(key, out)
            return out
        except Exception as e:
            self._log_fail("pcard", f"{type(e).__name__}: {e}")
            return None

    def movie_profile(self, title: str, genres: str, fictional: bool = False,
                      hint: str = ""):
        """新片内容画像（M7 冷启动）。

        fictional=True 为推断模式：虚构新片按片名/线索**推断**画像
        （真实电影模式则"不认识则拒绝"，避免编造）。
        hint：用户补充的一句话线索（并入推断提示，产出更丰富的画像）。
        返回 {"genres": [...], "mood", "era", "desc"} 或 None。
        """
        key = self._key("mprof", 1 if fictional else 0, [title, hint])
        cached = self._cached(key)
        if cached is not None:
            return cached if cached.get("genres") or cached.get("desc") else None
        g = genres.replace("|", "/") if isinstance(genres, str) else ""
        if fictional:
            head = (f"《{title}》是一部虚构的新电影（类型线索：{g}"
                    + (f"；补充线索：{hint}" if hint else "")
                    + "）。根据片名与线索推断其内容画像 JSON"
                      "（desc 务必具体丰富，包含题材、基调与看点）：\n")
            tail = ""
        else:
            head = f"根据电影《{title}》（类型：{g}）输出内容画像 JSON：\n"
            tail = ('如果你不认识这部电影，输出 {"unknown": true}，'
                    '不要编造。\n')
        prompt = (
            head
            + '{"genres": [从这18个类型选0~3个: ' + ", ".join(KNOWN_GENRES) + '],\n'
            '  "mood": 从[轻松,幽默,温馨,治愈,浪漫,热血,史诗,紧张,悬疑,惊悚,黑暗,沉重,科幻感,怀旧,现实,荒诞]中选1个,\n'
            '  "era": 年代感如"90年代"或"",\n'
            '  "desc": 一句话中文描述电影气质与题材(不超过40字)}\n'
            + tail + "只输出 JSON。")
        try:
            obj = _extract_json(self._chat(prompt))
            if obj is None or obj.get("unknown"):
                self._put(key, {})
                return None
            out = {"genres": [x for x in (obj.get("genres") or [])
                              if x in KNOWN_GENRES],
                   "mood": str(obj.get("mood", ""))[:6],
                   "era": str(obj.get("era", ""))[:8],
                   "desc": str(obj.get("desc", ""))[:60]}
            self._put(key, out)
            return out
        except Exception as e:
            self._log_fail("mprof", f"{type(e).__name__}: {e}")
            return None

    def synopsis(self, movie_id: int, title: str, genres: list):
        """AI 简介（按电影缓存，与用户无关）；不确定的影片返回 None。

        数据边界：无剧情简介数据源（TMDB 在本网络被阻断），由 LLM 生成并
        在前端显式标注"AI 简介"；不确定时宁缺毋滥（空串 → None → 槽位隐藏）。
        """
        if not self.available:
            return None
        key = self._key("synopsis", 0, [movie_id])
        cached = self._cached(key)
        if cached is not None:
            return cached.get("synopsis") or None
        prompt = (
            f"用一两句中文（不超过 60 字）介绍电影《{title}》"
            f"（类型：{'/'.join(genres[:3]) if genres else '未知'}）。\n"
            "如果你不认识这部电影，不要编造。\n"
            '只输出 JSON：{"synopsis": "..."}；不认识则输出 {"synopsis": ""}。')
        try:
            obj = _extract_json(self._chat(prompt))
            s = str(obj.get("synopsis", "")).strip()[:90] if obj else ""
            self._put(key, {"synopsis": s})
            return s or None
        except Exception as e:
            self._log_fail("synopsis", f"{type(e).__name__}: {e}")
            return None
