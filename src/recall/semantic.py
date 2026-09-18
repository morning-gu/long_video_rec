"""内容语义召回（M6 新增，文档 02_01 §7 多模态召回的文本侧实现）。

用户内容画像 = 最近 K 部正反馈的内容向量均值（来自用户状态实时层，O1：
点击立即改变画像）；候选按内容向量余义相似度召回。

叠加式新增：不改动既有通道。冷启动友好——任何有画像的电影无需交互即可被召回。
"""
import numpy as np
import pandas as pd

from src import config

HIST_K = 32


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


class SemanticRecall:
    def __init__(self, vectors: np.ndarray, movie_ids: np.ndarray):
        self.vectors = vectors                  # [n_items, D]，已 L2 归一
        self.movie_ids = movie_ids
        self.mid2row = {int(m): r for r, m in enumerate(movie_ids)}

    @classmethod
    def load(cls):
        return cls(np.load(config.ART_DIR / "content_emb.npy"), _movie_ids())

    def register(self, movie_id: int, vec: np.ndarray):
        """注册新片（M7 冷启动）：内容向量进入语义召回空间。"""
        self.vectors = np.vstack([self.vectors, vec[None, :]])
        self.movie_ids = np.append(self.movie_ids, movie_id)
        self.mid2row[int(movie_id)] = len(self.movie_ids) - 1

    def user_profile(self, hist_mids: list):
        rows = [self.mid2row[m] for m in hist_mids[-HIST_K:]
                if m in self.mid2row]
        if not rows:
            return None
        p = self.vectors[rows].mean(0)
        n = np.linalg.norm(p)
        return p / n if n > 1e-9 else None

    def recall(self, hist_mids: list, exclude: set, topn: int):
        p = self.user_profile(hist_mids)
        if p is None:
            return []
        scores = self.vectors @ p
        if exclude:
            rows = [self.mid2row[m] for m in exclude if m in self.mid2row]
            scores[rows] = -np.inf
        order = np.argsort(-scores)[:topn]
        return [(int(self.movie_ids[r]), float(scores[r]))
                for r in order if np.isfinite(scores[r])]
