"""物品侧特征工程与序列构造（供双塔/SASRec 使用，设计文档 §4.3 特征分组）。"""
import re

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
N_YEAR_BUCKETS = 10      # 0=未知，1..9 = 1920s..2000s


def build_item_features(movies: pd.DataFrame):
    """返回 (genre_vocab, genre_multihot[n_items, G], year_bucket[n_items])。

    行序与 movies parquet 一致（即双塔连续索引的行序）。
    """
    genre_vocab = sorted({g for gs in movies.genres.fillna("")
                          for g in str(gs).split("|") if g})
    g2i = {g: i for i, g in enumerate(genre_vocab)}
    multihot = np.zeros((len(movies), len(genre_vocab)), dtype=np.float32)
    years = np.zeros(len(movies), dtype=np.int64)
    for i, (gs, title) in enumerate(zip(movies.genres.fillna(""),
                                        movies.title.astype(str))):
        for g in str(gs).split("|"):
            if g in g2i:
                multihot[i, g2i[g]] = 1.0
        m = YEAR_RE.search(title)
        if m:
            years[i] = int(m.group(1))
    bucket = np.zeros(len(movies), dtype=np.int64)
    known = years > 0
    bucket[known] = np.clip((years[known] - 1920) // 10, 0, 8) + 1
    return genre_vocab, multihot, bucket


def positive_sequences(train_pos: pd.DataFrame) -> dict:
    """返回 {user_id: [movie_id ...]}，训练期正反馈、时间升序。"""
    tp = train_pos.sort_values(["user_id", "timestamp"], kind="stable")
    return {int(u): list(map(int, g))
            for u, g in tp.groupby("user_id")["movie_id"]}
