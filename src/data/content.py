"""内容画像向量（M6）：LLM 结构化画像 → 混合内容向量。

编码器可插拔（CONTENT_EMBED_SOURCE）：
- 'tfidf'（默认）：画像文本（desc）字符 2/3-gram TF-IDF → LSA 压缩 128 维，
  拼接标签 one-hot（genres / mood / era，词表从数据派生），整体 L2 归一；
  零依赖本地计算（当前 API key 无 embedding 模型权限）。
- 'api'：dashscope text-embedding（预留，权限开通后切换）。

产出：data/artifacts/content_emb.npy [n_items, D]，行对齐 movies parquet。
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src import config

LSA_DIM = 128
MAX_VOCAB = 8000


def _char_ngrams(text: str):
    t = str(text).lower()
    out = []
    for k in (2, 3):
        out.extend(t[i:i + k] for i in range(len(t) - k + 1))
    return out


def _tfidf_lsa(texts: list, dim: int = LSA_DIM) -> np.ndarray:
    """字符 n-gram TF-IDF → 截断 SVD（LSA）。"""
    docs, df = [], {}
    for t in texts:
        g = set(_char_ngrams(t))
        docs.append(g)
        for x in g:
            df[x] = df.get(x, 0) + 1
    vocab = {g: i for i, (g, c) in enumerate(
        sorted(df.items(), key=lambda kv: -kv[1])[:MAX_VOCAB]) if c >= 2}
    n = len(texts)
    rows, cols, vals = [], [], []
    for r, g in enumerate(docs):
        tf = {}
        for x in g:
            if x in vocab:
                tf[x] = tf.get(x, 0) + 1
        for x, c in tf.items():
            rows.append(r)
            cols.append(vocab[x])
            vals.append((1 + np.log(c)) * np.log(n / df[x]))
    X = csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)),
                   dtype=np.float64)
    if X.shape[1] <= dim:
        return np.asarray(X.todense(), dtype=np.float32)
    k = min(dim, min(X.shape) - 1)
    U, S, _ = svds(X, k=k)
    U = U[:, ::-1] * (S[::-1] ** 0.5)          # 按奇异值降序
    norm = np.linalg.norm(U, axis=1, keepdims=True)
    return (U / np.maximum(norm, 1e-9)).astype(np.float32)


def _onehot_block(values: list):
    """标签列 → one-hot 矩阵（词表从数据派生）。"""
    vocab = sorted({v for v in values if v})
    v2i = {v: i for i, v in enumerate(vocab)}
    m = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for r, v in enumerate(values):
        if v in v2i:
            m[r, v2i[v]] = 1.0
    return m, vocab


def build_content_vectors() -> np.ndarray:
    """读取 profiles.parquet → [n_items, D] 内容向量（L2 归一，行对齐 movies）。"""
    movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
    profiles = pd.read_parquet(config.ART_DIR / "profiles.parquet")
    profiles = profiles.set_index("movie_id")
    n = len(movies)

    text, genre_vals, mood_vals, era_vals = [], [], [], []
    for r in movies.itertuples():
        p = profiles.loc[int(r.movie_id)] if int(r.movie_id) in profiles.index \
            else None
        genres = str(p.genres).split("|") if p is not None else []
        mood = str(p.mood) if p is not None else ""
        era = str(p.era) if p is not None else ""
        desc = str(p.desc) if p is not None else ""
        text.append(f"{desc} {desc} {r.title} {' '.join(genres)} {mood} {era}")
        genre_vals.append("|".join(genres))
        mood_vals.append(mood)
        era_vals.append(era)

    lsa = _tfidf_lsa(text)
    g_m, _ = _onehot_block(genre_vals)         # 多值：手动展开
    # genres 是多值标签，单独处理 one-hot（多热）
    g_vocab = sorted({g for v in genre_vals for g in v.split("|") if g})
    g2i = {g: i for i, g in enumerate(g_vocab)}
    g_m = np.zeros((n, len(g_vocab)), dtype=np.float32)
    for r, v in enumerate(genre_vals):
        for g in v.split("|"):
            if g in g2i:
                g_m[r, g2i[g]] = 1.0
    mood_m, _ = _onehot_block(mood_vals)
    era_m, _ = _onehot_block(era_vals)
    tags = np.hstack([g_m, mood_m, era_m])
    tag_norm = np.maximum(np.linalg.norm(tags, axis=1, keepdims=True), 1e-9)

    # 混合：文本 0.6 + 标签 0.4（两块各自归一后拼接，再整体归一）
    vec = np.hstack([lsa * 0.6, (tags / tag_norm) * 0.4]).astype(np.float32)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    return vec


def mood_onehot() -> tuple:
    """返回 (mood 矩阵 [n_items, n_moods], mood 词表)，行对齐 movies。"""
    movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
    profiles = pd.read_parquet(config.ART_DIR / "profiles.parquet")
    profiles = profiles.set_index("movie_id")
    moods = [str(profiles.loc[int(r.movie_id)].mood)
             if int(r.movie_id) in profiles.index else ""
             for r in movies.itertuples()]
    return _onehot_block(moods)


if __name__ == "__main__":
    v = build_content_vectors()
    np.save(config.ART_DIR / "content_emb.npy", v)
    print(f"content_emb.npy  shape={v.shape}")
