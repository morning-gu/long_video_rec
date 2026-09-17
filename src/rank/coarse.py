"""粗排（设计文档 §4.2，M3 评测驱动修订）：通道保持式压缩。

初版按设计稿实现为"双塔点积 + 口碑先验"的跨通道重排，评测显示 HR@10 从
融合的 0.186 降至 0.119——用最弱通道（双塔）的相似度重排全部候选，系统性
压制了 SASRec 等强通道的头部结果，即业界"粗排与召回/精排目标一致性"问题
（文档 03 文档 2 §3.2）。

修订：粗排退化为按通道配额截断的过滤器（保留各通道头部，维持融合顺序），
跨通道排序完全交给精排。诚实定位不变：Demo 规模下粗排的工业必要性本就
不成立，保留它是为了完整演示四段漏斗结构。
"""

from src import config

# 每通道保留配额（合计 200 = COARSE_TOPN）
COARSE_QUOTAS = {"itemcf": 60, "twotower": 40, "sasrec": 80, "hot": 20}


class CoarseRank:
    def __init__(self, quotas: dict = None):
        self.quotas = quotas or COARSE_QUOTAS

    def rank(self, candidates: list) -> list:
        """按 source 通道截断到配额，保持融合交错顺序（只过滤不重排）。"""
        kept, cnt = [], {}
        for c in candidates:
            src = c.get("source", "")
            if cnt.get(src, 0) < self.quotas.get(src, len(candidates)):
                cnt[src] = cnt.get(src, 0) + 1
                kept.append(c)
        return kept
