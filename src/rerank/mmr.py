"""MMR 多样性重排（设计文档 §4.4）。

    MMR = λ·rel(i) - (1-λ)·max_{j∈已选} cos(emb_i, emb_j)

相关性 rel 用列表内秩归一（跨通道/跨阶段分数量纲不一致，与融合层同理）；
物品向量复用双塔 item embedding。
"""

import numpy as np

from src import config


def mmr_order(items: list, item_embs: np.ndarray, mid2row: dict,
              lam: float = config.MMR_LAMBDA) -> list:
    n = len(items)
    if n <= 2:
        return items
    embs = np.stack([item_embs[mid2row[int(c["movie_id"])]]
                     for c in items])                   # [n, dim]，已 L2 归一
    rel = 1.0 - np.arange(n) / (n - 1)                 # 秩归一相关性
    chosen = [0]
    remaining = set(range(1, n))
    while remaining:
        best_j, best_s = None, -np.inf
        for j in remaining:
            sim = float((embs[j] @ embs[chosen].T).max())
            s = lam * rel[j] - (1 - lam) * sim
            if s > best_s:
                best_j, best_s = j, s
        chosen.append(best_j)
        remaining.remove(best_j)
    return [items[j] for j in chosen]
