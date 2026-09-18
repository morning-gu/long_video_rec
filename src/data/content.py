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


def _tfidf_lsa(texts: list, dim: int = LSA_DIM):
    """字符 n-gram TF-IDF → 截断 SVD（LSA）。返回 (向量, 变换模型)。

    变换模型持久化后可对**新文档**做同空间投影（M7 新片冷启动依赖）：
    new_vec = (tfidf(new) @ Vt.T / sqrt(S)) 归一化，与训练口径一致
    （训练向量 = U·S^0.5 归一 = X @ Vt.T / sqrt(S) 归一）。
    """
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
        return (np.asarray(X.todense(), dtype=np.float32),
                {"vocab": vocab, "df": df, "n": n, "Vt": None, "S": None})
    k = min(dim, min(X.shape) - 1)
    U, S, Vt = svds(X, k=k)
    order = np.argsort(-S)
    S, Vt = S[order], Vt[order]
    vec = U[:, order] * (S ** 0.5)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return ((vec / np.maximum(norm, 1e-9)).astype(np.float32),
            {"vocab": vocab, "df": df, "n": n, "Vt": Vt, "S": S})


def transform_text(text: str, model: dict, dim: int = LSA_DIM) -> np.ndarray:
    """用已拟合的 TF-IDF+LSA 模型投影新文档（M7 新片冷启动）。"""
    if model.get("Vt") is None:
        raise ValueError("LSA 模型未拟合")
    g = set(_char_ngrams(text))
    tf = {}
    for x in g:
        if x in model["vocab"]:
            tf[x] = tf.get(x, 0) + 1
    row = np.zeros(len(model["vocab"]), dtype=np.float64)
    for x, c in tf.items():
        row[model["vocab"][x]] = (1 + np.log(c)) * np.log(model["n"] /
                                                          model["df"][x])
    v = row @ model["Vt"].T / np.sqrt(model["S"])
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def _profile_text(desc, title, genres, mood, era):
    return f"{desc} {desc} {title} {' '.join(genres)} {mood} {era}"


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
    """读取 profiles.parquet → [n_items, D] 内容向量（L2 归一，行对齐 movies）。

    同时持久化变换模型 content_model.npz（vocab/idf/SVD 投影/标签词表），
    供新片冷启动做同空间投影（M7）。
    """
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
        text.append(_profile_text(desc, str(r.title), genres, mood, era))
        genre_vals.append("|".join(genres))
        mood_vals.append(mood)
        era_vals.append(era)

    lsa, tfidf_model = _tfidf_lsa(text)
    # genres 是多值标签，单独处理 one-hot（多热）
    g_vocab = sorted({g for v in genre_vals for g in v.split("|") if g})
    g2i = {g: i for i, g in enumerate(g_vocab)}
    g_m = np.zeros((n, len(g_vocab)), dtype=np.float32)
    for r, v in enumerate(genre_vals):
        for g in v.split("|"):
            if g in g2i:
                g_m[r, g2i[g]] = 1.0
    mood_m, mood_vocab = _onehot_block(mood_vals)
    era_m, era_vocab = _onehot_block(era_vals)
    tags = np.hstack([g_m, mood_m, era_m])
    tag_norm = np.maximum(np.linalg.norm(tags, axis=1, keepdims=True), 1e-9)

    # 混合：文本 0.6 + 标签 0.4（两块各自归一后拼接，再整体归一）
    vec = np.hstack([lsa * 0.6, (tags / tag_norm) * 0.4]).astype(np.float32)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)

    np.savez(config.ART_DIR / "content_model.npz",
             tfidf_model=tfidf_model,          # dict（pickle）
             genre_vocab=np.array(g_vocab),
             mood_vocab=np.array(mood_vocab),
             era_vocab=np.array(era_vocab))
    return vec


def load_content_model() -> dict:
    d = np.load(config.ART_DIR / "content_model.npz", allow_pickle=True)
    return {"tfidf": d["tfidf_model"].item(),
            "genres": list(d["genre_vocab"]),
            "moods": list(d["mood_vocab"]),
            "eras": list(d["era_vocab"])}


def embed_new_movie(title: str, genres: list, mood: str, era: str,
                    desc: str) -> np.ndarray:
    """新片内容向量：用已拟合模型做同空间投影（M7 冷启动）。

    返回与 content_emb.npy 同维（LSA 128 + 标签）且同口径的向量。
    """
    m = load_content_model()
    lsa = transform_text(_profile_text(desc, title, genres, mood, era),
                         m["tfidf"])
    tag = np.zeros(len(m["genres"]) + len(m["moods"]) + len(m["eras"]),
                   dtype=np.float32)
    for g in genres:
        if g in m["genres"]:
            tag[m["genres"].index(g)] = 1.0
    if mood in m["moods"]:
        tag[len(m["genres"]) + m["moods"].index(mood)] = 1.0
    if era in m["eras"]:
        tag[len(m["genres"]) + len(m["moods"]) + m["eras"].index(era)] = 1.0
    tn = np.linalg.norm(tag)
    tag = tag / tn if tn > 1e-9 else tag
    v = np.concatenate([lsa * 0.6, tag * 0.4])
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


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
