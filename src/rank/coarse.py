"""粗排（设计文档 §4.2）：两种策略并存，可运行时切换（M6+）。

- "keep"（默认，M3 评测驱动修订）：通道保持式压缩——每通道保留头部配额，
  只过滤不重排，维持融合交错顺序。
- "tt"（M3 初版：双塔点积 + 口碑先验）：双塔 user/item 向量点积 + wr 分位
  先验（α=0.15）对全部融合候选做**全局重排**，取 Top-N（COARSE_TOPN）。
  M3 评测显示该策略使 HR@10 从融合 0.186 降至 0.119——用最弱通道（双塔）的
  相似度重排全部候选，会系统性压制 SASRec 等强通道的头部结果，即业界
  "粗排与召回/精排目标一致性"问题（文档 03 文档 2 §3.2），故被修订为 keep；
  现恢复为可切换选项，用于现场对比演示这一发现。
- "none"：跳过粗排（在 funnel 层处理，融合候选直通精排）。
"""

import numpy as np

from src import config

# keep 策略：每通道保留配额（合计 200 = COARSE_TOPN）
COARSE_QUOTAS = {"itemcf": 45, "twotower": 30, "sasrec": 60, "hot": 12,
                 "lightgcn": 27, "semantic": 26}


class CoarseRank:
    def __init__(self, quotas=None, tt_item_embs=None, hot=None,
                 movie_ids=None):
        self.quotas = quotas or COARSE_QUOTAS
        # tt 策略依赖：双塔 item 向量 + 口碑分位（wr 越高先验越大，
        # 未上榜冷门为 0，不加成不惩罚）
        self.item_embs = tt_item_embs
        self.mid2row = ({int(m): i for i, m in enumerate(movie_ids)}
                        if movie_ids is not None else None)
        self.wr_pct = None
        if hot is not None and self.mid2row is not None:
            self.wr_pct = np.zeros(len(tt_item_embs), dtype=np.float32)
            n = max(len(hot.wr_order), 1)
            for rank, (mid, _wr) in enumerate(hot.wr_order):
                self.wr_pct[self.mid2row[mid]] = 1.0 - rank / n

    def rank(self, candidates: list, strategy: str = "keep",
             user_emb: np.ndarray = None,
             topn: int = None, alpha: float = 0.15) -> list:
        if strategy == "tt":
            return self._rank_tt(candidates, user_emb,
                                 topn or config.COARSE_TOPN, alpha)
        return self._rank_keep(candidates)

    def _rank_keep(self, candidates: list) -> list:
        """通道保持式压缩：按 source 截断到配额，保持融合顺序（只过滤不重排）。"""
        kept, cnt = [], {}
        for c in candidates:
            src = c.get("source", "")
            if cnt.get(src, 0) < self.quotas.get(src, len(candidates)):
                cnt[src] = cnt.get(src, 0) + 1
                kept.append(c)
        return kept

    def _rank_tt(self, candidates: list, user_emb: np.ndarray, topn: int,
                 alpha: float) -> list:
        """双塔点积 + 口碑先验的全局重排（M3 初版）。"""
        rows = np.array([self.mid2row[c["movie_id"]] for c in candidates])
        sims = self.item_embs[rows] @ user_emb
        scores = sims + alpha * self.wr_pct[rows]
        order = np.argsort(-scores)[:topn]
        return [{**candidates[i], "score": float(scores[i])} for i in order]
