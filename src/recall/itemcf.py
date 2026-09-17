"""ItemCF 召回：共现相似度（余弦 + 热门惩罚），设计文档 §4.1。

    sim(i,j) = |U_i ∩ U_j| / sqrt(|U_i| × |U_j|)

离线：由训练期正反馈二部图计算物品相似矩阵（ML-1M 约 3900² 规模，稠密可承受）。
在线：以用户最近正反馈为种子（按新近度加权），相似度加权求和召回 Top-N；
     种子来自用户状态实时层（§4.6，O1），点击即时成为最新种子。
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src import config


def build_sim(train_pos: pd.DataFrame, movie_ids: np.ndarray):
    """由训练期正反馈计算物品相似矩阵。返回 (sim[n_items, n_items] float32, pop[n_items])。"""
    mid2idx = {m: i for i, m in enumerate(movie_ids)}
    n_users = int(train_pos.user_id.max())
    n_items = len(movie_ids)
    rows = train_pos.user_id.values - 1
    cols = np.array([mid2idx[m] for m in train_pos.movie_id.values])
    R = csr_matrix((np.ones(len(train_pos), dtype=np.float32), (rows, cols)),
                   shape=(n_users, n_items))
    co = (R.T @ R).toarray().astype(np.float32)      # 共现计数
    pop = np.diag(co).copy()                          # 物品热度（正反馈人数）
    denom = np.sqrt(np.outer(pop, pop))
    denom[denom == 0] = 1.0
    sim = co / denom
    np.fill_diagonal(sim, 0.0)
    return sim, pop


class ItemCFRecall:
    def __init__(self, sim: np.ndarray, movie_ids: np.ndarray):
        self.sim = sim
        self.movie_ids = movie_ids
        self.mid2idx = {m: i for i, m in enumerate(movie_ids)}
        self.n_items = len(movie_ids)

    def neighbors(self, movie_id: int, exclude: set, topn: int):
        """单种子相似邻居（"因为你看过X"行直接可用）。"""
        row = self.sim[self.mid2idx[movie_id]]
        return self._top(row, exclude, topn)

    def recall(self, seeds: list, exclude: set, topn: int):
        """多种子加权召回：seeds 按新近度降序（最新在前，权重最高）。"""
        scores = np.zeros(self.n_items, dtype=np.float32)
        for w, m in zip(config.SEED_WEIGHTS, seeds):
            if m in self.mid2idx:
                scores += w * self.sim[self.mid2idx[m]]
        return self._top(scores, exclude, topn)

    def _top(self, scores: np.ndarray, exclude: set, topn: int):
        if exclude:
            idxs = [self.mid2idx[m] for m in exclude if m in self.mid2idx]
            scores = scores.copy()
            scores[idxs] = -np.inf
        order = np.argsort(-scores)[:topn]
        return [(int(self.movie_ids[i]), float(scores[i]))
                for i in order if np.isfinite(scores[i])]
