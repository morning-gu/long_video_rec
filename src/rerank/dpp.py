"""DPP 多样性重排（M6 新增，文档 02 §4）：贪心 MAP 推断。

    L = Diag(q) · S · Diag(q)，q = 秩归一相关性（+底噪防归零），S = 双塔
    item embedding 余弦相似。贪心 MAP 用增量 Cholesky，20 项列表毫秒级。

与 MMR 并存可切换（?rerank=mmr|dpp），不删除旧实现。
"""

import numpy as np

from src import config


def dpp_order(items: list, item_embs: np.ndarray, mid2row: dict,
              rel_floor: float = 0.2):
    n = len(items)
    if n <= 2:
        return items
    embs = np.stack([item_embs[mid2row[int(c["movie_id"])]]
                     for c in items])                   # [n, d] 已归一
    rel = 1.0 - np.arange(n) / (n - 1)                 # 秩归一（与 MMR 一致）
    q = rel_floor + (1 - rel_floor) * rel
    s = embs @ embs.T
    L = q[:, None] * s * q[None, :]
    np.fill_diagonal(L, q * q)                          # 对角 = q²（自相似=1）

    # 贪心 MAP（增量 Cholesky）
    c = np.zeros((n, n))
    d = np.diag(L).copy()
    chosen = []
    for i in range(n):
        j = int(np.argmax(d))
        if d[j] <= 1e-10:
            break
        chosen.append(j)
        if i == n - 1:
            break
        dj = d[j]                       # 先保存，再屏蔽已选项
        d[j] = -np.inf
        e = L[j].copy()
        if i > 0:
            e -= c[:, :i] @ c[j, :i]
        e /= np.sqrt(dj)
        c[:, i] = e
        d = d - e ** 2
        d[j] = -np.inf
    return [items[j] for j in chosen]
