"""热门召回：贝叶斯平均评分平滑的全局热门（设计文档 §4.1）。

    WR(i) = (C·m + Σr_i) / (C + n_i)    m 为全局均分，C 为先验票数

仅用训练期评分计算（剔除测试交互），与离线评测口径一致，避免测试泄漏。
热门集合（Top-N 热门）同时作为重排层"冷门保量"的分界（§4.4）。
"""
import pandas as pd

from src import config


def build_hot(ratings: pd.DataFrame, test_pos: pd.DataFrame) -> pd.DataFrame:
    """返回 DataFrame[movie_id, wr, pos_count, hot_rank]，按 wr 降序。"""
    test_keys = test_pos[["user_id", "movie_id"]]
    rt = ratings.merge(test_keys, on=["user_id", "movie_id"],
                       how="left", indicator=True)
    rt = rt[rt._merge == "left_only"]                 # 训练期评分
    g = rt.groupby("movie_id")["rating"].agg(["sum", "count"])
    m = float(rt.rating.mean())
    wr = (config.BAYES_C * m + g["sum"]) / (config.BAYES_C + g["count"])
    pos_count = rt[rt.rating >= config.POS_RATING].groupby("movie_id").size()
    hot = pd.DataFrame({"wr": wr, "pos_count": pos_count})
    hot["pos_count"] = hot["pos_count"].fillna(0).astype(int)
    hot = hot.sort_values("wr", ascending=False).reset_index()
    hot["hot_rank"] = range(len(hot))
    return hot


class HotRecall:
    def __init__(self, hot: pd.DataFrame):
        self.wr_order = list(zip(hot.movie_id.astype(int), hot.wr.astype(float)))
        self.wr = dict(self.wr_order)
        self.pos_count = {int(m): int(c) for m, c in
                          zip(hot.movie_id, hot.pos_count)}
        # 热门排序按正反馈人数（流行度），wr 用于质量分
        self.pop_order = [int(m) for m in
                          hot.sort_values("pos_count", ascending=False).movie_id]
        self.pop_rank = {m: i for i, m in enumerate(self.pop_order)}
        self.hot_set = set(self.pop_order[:config.HOT_SET_SIZE])

    def top(self, exclude: set, topn: int):
        """全局热门 Top-N（按流行度排序，wr 作为分值）。"""
        out = []
        for mid in self.pop_order:
            if mid in exclude:
                continue
            out.append((mid, self.wr[mid]))
            if len(out) >= topn:
                break
        return out

    def cold_top(self, exclude: set, topn: int):
        """冷门佳片：非热门集合中口碑（wr）最高者——服务多样性叙事的展示行。"""
        out = []
        for mid, wr in self.wr_order:
            if mid in self.hot_set or mid in exclude:
                continue
            out.append((mid, wr))
            if len(out) >= topn:
                break
        return out
