"""ItemCF 召回：共现相似度（余弦 + 热门惩罚），设计文档 §4.1。

    sim(i,j) = |U_i ∩ U_j| / sqrt(|U_i| × |U_j|)

离线：由训练期正反馈二部图计算物品相似矩阵——
  - itemcf_topk=0（ml-1m）：稠密精确矩阵（约 3900²，历史行为不变）；
  - itemcf_topk>0（ml-25m 等大规模）：分块计算每物品 top-K 稀疏相似
    （59k² 稠密不可行；峰值内存 ≈ 单块稀疏行，GPU 机器内存充裕）。
在线：以用户最近正反馈为种子（按新近度加权），相似度加权求和召回 Top-N；
     种子来自用户状态实时层（§4.6，O1），点击即时成为最新种子。
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from src import config


def build_sim(train_pos: pd.DataFrame, movie_ids: np.ndarray):
    """返回 (sim, pop)。sim 为 ndarray（稠密）或 csr_matrix（稀疏 top-K）。"""
    if config.P["itemcf_topk"] > 0:
        return _build_sim_sparse(train_pos, movie_ids,
                                 config.P["itemcf_topk"]), None
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


def _build_sim_sparse(train_pos: pd.DataFrame, movie_ids: np.ndarray, K: int,
                      block: int = 2048):
    """分块 top-K 余弦相似：每块 S = R[:,I].T @ R → 行内 top-K → 聚合 CSR。"""
    mid2idx = {m: i for i, m in enumerate(movie_ids)}
    _, uidx = np.unique(train_pos.user_id.values, return_inverse=True)
    n_items = len(movie_ids)
    cols = np.array([mid2idx[m] for m in train_pos.movie_id.values])
    R = csr_matrix((np.ones(len(train_pos), dtype=np.float32), (uidx, cols)),
                   shape=(len(np.unique(uidx)), n_items))
    pop = np.asarray(R.sum(0)).ravel()                # 每物品正反馈人数
    inv = np.where(pop > 0, 1.0 / np.sqrt(pop), 0.0)

    rows_data, rows_idx, rows_ptr = [], [], [0]
    for s in range(0, n_items, block):
        idx = np.arange(s, min(s + block, n_items))
        Sb = (R[:, idx].T @ R).tocsr()                # [B, n_items] 共现
        Sb = sparse.diags(inv[idx]) @ Sb @ sparse.diags(inv)   # 余弦归一
        Sb.setdiag(0.0)
        Sb.eliminate_zeros()
        for r in range(Sb.shape[0]):
            a, b = Sb.indptr[r], Sb.indptr[r + 1]
            if b - a > K:
                keep = np.argpartition(-Sb.data[a:b], K - 1)[:K]
                sel = a + np.sort(keep)
            else:
                sel = np.arange(a, b)
            order = np.argsort(-Sb.data[sel])         # 行内降序
            d, c = Sb.data[sel][order], Sb.indices[sel][order]
            rows_data.append(d)
            rows_idx.append(c)
            rows_ptr.append(rows_ptr[-1] + len(d))
        if (s // block) % 5 == 0:
            print(f"  [itemcf-sparse] {min(s + block, n_items)}/{n_items}")
    return csr_matrix((np.concatenate(rows_data), np.concatenate(rows_idx),
                       np.array(rows_ptr)), shape=(n_items, n_items))


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


class ItemCFRecall:
    """双模式：稠密（ml-1m）/ 稀疏 top-K（大规模）——接口一致。"""

    def __init__(self, sim, movie_ids: np.ndarray):
        self.is_sparse = sparse.issparse(sim)
        self.sim = sim
        self.movie_ids = movie_ids
        self.mid2idx = {m: i for i, m in enumerate(movie_ids)}
        self.n_items = len(movie_ids)

    @classmethod
    def load(cls):
        p_sparse = config.ART_DIR / "itemcf_sim_sparse.npz"
        if p_sparse.exists():
            z = np.load(p_sparse)
            sim = csr_matrix((z["data"], z["indices"], z["indptr"]),
                             shape=tuple(z["shape"]))
            return cls(sim, _movie_ids())
        return cls(np.load(config.ART_DIR / "itemcf_sim.npy"), _movie_ids())

    def neighbors(self, movie_id: int, exclude: set, topn: int):
        """单种子相似邻居（"因为你看过X"行直接可用）。"""
        if self.is_sparse:
            row = self.sim.getrow(self.mid2idx[movie_id])
            out = [(int(self.movie_ids[c]), float(v))
                   for c, v in zip(row.indices, row.data)
                   if int(self.movie_ids[c]) not in exclude]
            out.sort(key=lambda x: -x[1])
            return out[:topn]
        row = self.sim[self.mid2idx[movie_id]]
        return self._top(row, exclude, topn)

    def recall(self, seeds: list, exclude: set, topn: int):
        """多种子加权召回：seeds 按新近度降序（最新在前，权重最高）。"""
        if self.is_sparse:
            scores = {}
            for w, m in zip(config.SEED_WEIGHTS, seeds):
                if m not in self.mid2idx:
                    continue
                row = self.sim.getrow(self.mid2idx[m])
                for c, v in zip(row.indices, row.data):
                    mid = int(self.movie_ids[c])
                    if mid in exclude:
                        continue
                    scores[mid] = scores.get(mid, 0.0) + w * float(v)
            return sorted(scores.items(), key=lambda x: -x[1])[:topn]
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
