"""用户状态实时层（设计文档 §4.6，O1）：内存态 + 写透 SQLite events 表。

内存态：
- 全量评分 → 已看集合（评过分即看过）；
- 正反馈时间序列 → ItemCF 种子（最近 N 部，新→旧）；
- 演示期点击/负反馈事件 → 即时叠加到上述两态（点击成为最新种子，负反馈进抑制集合）。

写透：每次演示事件同步落 SQLite；进程重启时按 id 顺序回放重建（R7）。
这是"实时化"趋势（Kafka/Flink 流式特征）的单机替身——在线更新输入状态，
模型权重（ItemCF 相似矩阵）保持离线训练（决策 D8）。
"""
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd

from src import config


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
        self._rated = {int(u): set(map(int, g))
                       for u, g in ratings.groupby("user_id")["movie_id"]}
        pos = ratings[ratings.rating >= config.POS_RATING]
        self._pos_seq = {int(u): list(map(int, g))
                         for u, g in pos.groupby("user_id")["movie_id"]}
        self._events: dict = {}                          # [(movie_id, action, ts)]
        self.db = sqlite3.connect(db_path or config.DB_PATH,
                                  check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id INTEGER, movie_id INTEGER, action TEXT, ts INTEGER)")
        self.db.commit()
        self._replay()

    def _replay(self) -> None:
        for user_id, movie_id, action, ts in self.db.execute(
                "SELECT user_id, movie_id, action, ts FROM events ORDER BY id"):
            self._events.setdefault(int(user_id), []).append(
                (int(movie_id), action, int(ts)))

    def get(self, user_id: int) -> UserState:
        st = UserState(user_id=user_id)
        st.seen = set(self._rated.get(user_id, []))
        st.recent_positives = list(self._pos_seq.get(user_id, []))
        for movie_id, action, _ts in self._events.get(user_id, []):
            st.seen.add(movie_id)
            if action == "click":
                # 点击即时成为最新种子（O1：下一次请求即反映）
                st.recent_positives.append(movie_id)
            elif action == "dislike":
                st.suppress.add(movie_id)
        return st

    def record(self, user_id: int, movie_id: int, action: str) -> None:
        ts = int(time.time())
        self._events.setdefault(user_id, []).append((movie_id, action, ts))
        self.db.execute(
            "INSERT INTO events (user_id, movie_id, action, ts) VALUES (?,?,?,?)",
            (user_id, movie_id, action, ts))
        self.db.commit()
