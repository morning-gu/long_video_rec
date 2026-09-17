"""用户状态实时层（设计文档 §4.6，O1）：内存态 + 写透 SQLite events 表。

内存态：
- 全量评分 → 已看集合（评过分即看过）；
- 正反馈时间序列 → ItemCF 种子（最近 N 部，新→旧）；
- 演示期事件（喜欢/不感兴趣/看完/评分等）→ 即时叠加到上述两态。

动作语义注册表（便于扩展打分、看完等新动作）：
- click / watched / like  → 正反馈：进已看 + 成为最新种子；
- dislike                 → 负反馈：进已看 + 抑制集合；
- rate(value 1~5)         → 评分：进已看；>=POS_RATING 视为正反馈成为种子，
                            <=RATING_NEGATIVE 视为负反馈进抑制集合。

写透：每次演示事件同步落 SQLite；进程重启时按 id 顺序回放重建（R7）。
这是"实时化"趋势（Kafka/Flink 流式特征）的单机替身——在线更新输入状态，
模型权重（召回/排序模型）保持离线训练（决策 D8）。
"""
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd

from src import config

POSITIVE_ACTIONS = {"click", "watched", "like"}
ALLOWED_ACTIONS = POSITIVE_ACTIONS | {"dislike", "rate"}
RATING_NEGATIVE = 2        # 评分 <= 2 视为负反馈（与 POS_RATING 对称）


@dataclass
class UserState:
    user_id: int
    seen: set = field(default_factory=set)
    suppress: set = field(default_factory=set)          # 负反馈抑制集合
    recent_positives: list = field(default_factory=list)  # 正反馈，时间升序

    @property
    def seeds(self) -> list:
        """最近正反馈（新→旧），供 ItemCF 召回。"""
        return list(reversed(self.recent_positives[-config.SEED_TOPK:]))


class UserStateStore:
    def __init__(self, ratings: pd.DataFrame, db_path=None):
        """db_path=None 时使用默认 demo.db；评测隔离场景可传 ":memory:"。"""
        ratings = ratings.sort_values(["user_id", "timestamp"], kind="stable")
        # 历史评分（含分值，供详情页归因展示"你评过 X 分"）
        self._rated = {}
        for u, g in ratings.groupby("user_id"):
            self._rated[int(u)] = dict(zip(g.movie_id.astype(int),
                                           g.rating.astype(float)))
        pos = ratings[ratings.rating >= config.POS_RATING]
        self._pos_seq = {int(u): list(map(int, g))
                         for u, g in pos.groupby("user_id")["movie_id"]}
        self._events: dict = {}                          # [(mid, action, ts, value)]
        self.db = sqlite3.connect(db_path or config.DB_PATH,
                                  check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id INTEGER, movie_id INTEGER, action TEXT, ts INTEGER, "
            "value REAL)")
        try:                                             # 旧库迁移：补 value 列
            self.db.execute("ALTER TABLE events ADD COLUMN value REAL")
        except sqlite3.OperationalError:
            pass
        self.db.commit()
        self._replay()

    def _replay(self) -> None:
        for user_id, movie_id, action, ts, value in self.db.execute(
                "SELECT user_id, movie_id, action, ts, value "
                "FROM events ORDER BY id"):
            self._events.setdefault(int(user_id), []).append(
                (int(movie_id), action, int(ts), value))

    def _apply(self, st: UserState, movie_id: int, action: str,
               value: float | None) -> None:
        st.seen.add(movie_id)
        if action in POSITIVE_ACTIONS:
            # 正反馈即时成为最新种子（O1：下一次请求即反映）
            st.recent_positives.append(movie_id)
        elif action == "dislike":
            st.suppress.add(movie_id)
        elif action == "rate" and value is not None:
            if value >= config.POS_RATING:
                st.recent_positives.append(movie_id)
            elif value <= RATING_NEGATIVE:
                st.suppress.add(movie_id)

    def get(self, user_id: int) -> UserState:
        st = UserState(user_id=user_id)
        st.seen = set(self._rated.get(user_id, {}))
        st.recent_positives = list(self._pos_seq.get(user_id, []))
        for movie_id, action, _ts, value in self._events.get(user_id, []):
            self._apply(st, movie_id, action, value)
        return st

    def rating_of(self, user_id: int, movie_id: int):
        """历史评分（MovieLens）；未评过返回 None。"""
        return self._rated.get(user_id, {}).get(movie_id)

    def last_mark(self, user_id: int, movie_id: int):
        """演示期该影片最近一次动作，返回 (action, value) 或 None。"""
        for movie_id_, action, _ts, value in reversed(
                self._events.get(user_id, [])):
            if movie_id_ == movie_id:
                return action, value
        return None

    def record(self, user_id: int, movie_id: int, action: str,
               value: float | None = None) -> None:
        ts = int(time.time())
        self._events.setdefault(user_id, []).append((movie_id, action, ts, value))
        self.db.execute(
            "INSERT INTO events (user_id, movie_id, action, ts, value) "
            "VALUES (?,?,?,?,?)",
            (user_id, movie_id, action, ts, value))
        self.db.commit()
