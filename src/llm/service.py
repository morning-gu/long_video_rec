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
        """单次对话调用；qwen3 系列尝试关闭思考模式以降低时延。"""
        kwargs = dict(model=self.model,
                      messages=[{"role": "user", "content": prompt}],
                      temperature=0.3)
        try:
            r = self.client.chat.completions.create(
                **kwargs, extra_body={"enable_thinking": False})
        except Exception:
            r = self.client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""

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
            self._put(key, {"genres": genres, "mood": mood})
            return {"genres": genres, "mood": mood}
        except Exception:
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
        except Exception:
            pass
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
            return cached["reasons"], False
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
                    return reasons, False
        except Exception:
            pass
        return {}, True
