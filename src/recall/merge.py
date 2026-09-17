"""多路召回配额融合（设计文档 §4.1 融合策略）。

各路分数仅在路内排序，跨路不做分数加权（量纲不一致，且演示时更易讲解）；
采用配额截断 + 轮流交错去重，输出顺序即进入粗排/重排的优先级顺序。
"""


def merge(channels: dict, quotas: dict) -> list:
    """channels: {通道名: [(movie_id, score), ...] 按分降序}；
    quotas: {通道名: 配额}。返回 [{movie_id, score, source}]，交错优先级排序。
    """
    streams = [(name, items[:quotas.get(name, len(items))])
               for name, items in channels.items()]
    seen_ids, out = set(), []
    i = 0
    while any(i < len(items) for _, items in streams):
        for name, items in streams:
            if i < len(items):
                mid, score = items[i]
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    out.append({"movie_id": int(mid), "score": float(score),
                                "source": name})
        i += 1
    return out
