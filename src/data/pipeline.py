"""ML-1M 数据管线：下载、解析、标签构造、时间切分、落盘。

对应设计文档 §3：
- 评分 >= 4 视为隐式正反馈（A1）；
- 时间切分（每用户留最后一个正样本做测试），避免随机切分的时间泄漏（D7）。
"""
import urllib.request
import zipfile

import pandas as pd

from src import config


def ensure_downloaded() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    if (config.RAW_DIR / "ratings.dat").exists():
        return
    if not config.ML1M_ZIP.exists():
        print(f"downloading {config.ML1M_URL} ...")
        urllib.request.urlretrieve(config.ML1M_URL, config.ML1M_ZIP)
    with zipfile.ZipFile(config.ML1M_ZIP) as z:
        z.extractall(config.DATA_DIR)
    print(f"extracted to {config.RAW_DIR}")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sep = dict(sep="::", engine="python", encoding="latin-1")
    ratings = pd.read_csv(
        config.RAW_DIR / "ratings.dat",
        names=["user_id", "movie_id", "rating", "timestamp"], **sep)
    movies = pd.read_csv(
        config.RAW_DIR / "movies.dat",
        names=["movie_id", "title", "genres"], **sep)
    users = pd.read_csv(
        config.RAW_DIR / "users.dat",
        names=["user_id", "gender", "age", "occupation", "zip"], **sep)
    return ratings, movies, users


def build_split(ratings: pd.DataFrame):
    """返回 (train_pos, test_pos)：正反馈按用户时间排序，各留最后一个做测试。"""
    pos = ratings[ratings.rating >= config.POS_RATING].sort_values(
        ["user_id", "timestamp"], kind="stable")
    is_last = pos.groupby("user_id").cumcount(ascending=False) == 0
    cols = ["user_id", "movie_id", "rating", "timestamp"]
    return (pos[~is_last][cols].reset_index(drop=True),
            pos[is_last][cols].reset_index(drop=True))


def run() -> None:
    ensure_downloaded()
    ratings, movies, users = load_raw()
    train_pos, test_pos = build_split(ratings)
    config.ART_DIR.mkdir(exist_ok=True)
    ratings.to_parquet(config.ART_DIR / "ratings.parquet")
    movies.to_parquet(config.ART_DIR / "movies.parquet")
    users.to_parquet(config.ART_DIR / "users.parquet")
    train_pos.to_parquet(config.ART_DIR / "train_pos.parquet")
    test_pos.to_parquet(config.ART_DIR / "test_pos.parquet")
    print(f"ratings={len(ratings)}  movies={len(movies)}  users={len(users)}")
    print(f"positives(train)={len(train_pos)}  positives(test, 每用户最后1个)={len(test_pos)}")


if __name__ == "__main__":
    run()
