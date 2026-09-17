"""规则重排：已看过滤 / 同类型连续打散 / 单类型上限 / 冷门保量（设计文档 §4.4）。

输入候选已按优先级排序（merge 的交错顺序），贪心选择时施加列表级约束。
"""
import math

from src import config


def rule_rerank(candidates: list, movie_info: dict, state, hot_set: set,
                k: int = config.ROW_K) -> list:
    """candidates: [{movie_id, score, source}]；movie_info: {mid: {title, genres}}；
    state: UserState（已看/负反馈双保险过滤）。返回最终 Top-K 列表。
    """
    seen = state.seen | state.suppress
    pool = [c for c in candidates if c["movie_id"] not in seen]

    def primary_genre(mid):
        genres = movie_info.get(mid, {}).get("genres") or []
        return genres[0] if genres else ""

    selected, deferred = [], []
    genre_count: dict = {}
    last_genre, consec = "", 0

    def try_pick(c):
        nonlocal last_genre, consec
        g = primary_genre(c["movie_id"])
        if g == last_genre and consec >= config.GENRE_CONSEC_MAX:
            return False                    # 同类型连续超限，延迟回填
        if genre_count.get(g, 0) >= config.GENRE_CAP:
            return False                    # 单类型总量上限
        selected.append(c)
        genre_count[g] = genre_count.get(g, 0) + 1
        consec = consec + 1 if g == last_genre else 1
        last_genre = g
        return True

    for c in pool:
        if len(selected) >= k:
            break
        if not try_pick(c):
            deferred.append(c)
    for c in deferred:                      # 打散被延迟的候选回填剩余空位
        if len(selected) >= k:
            break
        selected.append(c)

    # 冷门保量：Top-K 中非热门占比不足时，用未选冷门候选替换低优先级热门
    need_cold = math.ceil(config.COLD_RATIO * len(selected))
    cold = [c for c in selected if c["movie_id"] not in hot_set]
    if len(cold) < need_cold:
        sel_mids = {c["movie_id"] for c in selected}
        rest_cold = [c for c in pool
                     if c["movie_id"] not in hot_set and c["movie_id"] not in sel_mids]
        i = len(selected) - 1               # 从列表末尾（低优先级热门）开始替换
        while len(cold) < need_cold and rest_cold and i >= 0:
            if selected[i]["movie_id"] in hot_set:
                nc = rest_cold.pop(0)
                selected[i] = nc
                cold.append(nc)
            i -= 1
    return selected[:k]
