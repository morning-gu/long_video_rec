"""双塔召回（设计文档 §4.1）：user tower × item tower + Faiss ANN。

- item tower：电影 ID embedding + 类型 multi-hot + 年代桶 → MLP → L2 归一化；
- user tower：用户 ID embedding + 历史行为聚合（最近 K 部正反馈经独立历史
  embedding 表的均值）+ 类型偏好直方图 → MLP → L2 归一化；
- 训练：sampled softmax（in-batch 负样本 + 均匀采样负样本，温度 τ）；
- O1（§4.6）：user tower 每次请求在线前向，输入状态（历史/类型偏好）实时生效；
  模型权重离线训练（决策 D8）。item embedding 离线预计算 + Faiss IndexFlatIP
  （25M 规模换 HNSW，决策 D1）。

连续索引约定：movie_id → idx = 行号 + 1（0 保留给 padding/未知）。
性能注：训练样本（滑动历史窗口 + 类型前缀和）一次性预计算为全局数组，
每 epoch 仅做向量化采样索引，避免逐样本 Python 循环（本机 2 核 CPU）。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.device import get_device

HIST_K = 32            # 用户塔历史聚合窗口
TEMP = 0.1             # sampled softmax 温度
UNIFORM_NEG = 256      # 均匀负采样数（另有 in-batch 负样本）
MAX_PER_USER = 20      # 每用户每 epoch 采样训练位置数


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_genres,
                 n_year_buckets=10, dim=64):
        super().__init__()
        self.item_table = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.hist_table = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.user_table = nn.Embedding(n_users + 1, dim)
        self.genre_proj = nn.Linear(n_genres, dim)      # 用户/物品塔共享类型空间投影
        self.year_emb = nn.Embedding(n_year_buckets + 1, dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.user_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def item_forward(self, idx, genre_row, year_row):
        """idx/genre_row/year_row: [B]，均为连续索引（含 0 padding 行）。"""
        e = self.item_table(idx)
        x = torch.cat([e, self.genre_proj(genre_row),
                       self.year_emb(year_row)], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def user_forward(self, user_idx, hist_idx, hist_mask, genre_hist):
        """hist_idx: [B, K]；hist_mask: [B, K]；genre_hist: [B, G]。"""
        u = self.user_table(user_idx)
        h = self.hist_table(hist_idx)                    # [B, K, dim]
        hm = (h * hist_mask.unsqueeze(-1)).sum(1) / \
            hist_mask.sum(1, keepdim=True).clamp(min=1.0)
        x = torch.cat([u, hm, self.genre_proj(genre_hist)], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)


def _precompute_samples(seqs: dict, genre_mh: np.ndarray, mid2idx: dict):
    """一次性预计算全部训练位置的 (user, target, hist[K], mask[K], ghist[G])。

    返回 (arrays 元组, per_user (offsets, counts))，全局位置为拼接索引。
    """
    K = HIST_K
    tgt_l, usr_l, hist_l, mask_l, gh_l = [], [], [], [], []
    offsets, counts = [], []
    cursor = 0
    for u, seq in seqs.items():
        idx = np.array([mid2idx[m] for m in seq], dtype=np.int64)
        n = len(idx)
        offsets.append(cursor)
        counts.append(max(n - 1, 0))
        cursor += max(n - 1, 0)
        if n < 2:
            continue
        pos = np.arange(1, n)                       # 目标位置
        starts = np.maximum(0, pos - K)
        lens = pos - starts
        H = np.zeros((n - 1, K), dtype=np.int64)    # 左对齐历史窗口
        M = np.zeros((n - 1, K), dtype=np.float32)
        for j in range(n - 1):
            s, l = starts[j], lens[j]
            H[j, :l] = idx[s:s + l]
            M[j, :l] = 1.0
        rows = genre_mh[idx - 1]                    # idx 均 >= 1
        cum = np.vstack([np.zeros((1, rows.shape[1])),
                         np.cumsum(rows, axis=0)])
        G = (cum[pos] - cum[starts]) / lens[:, None]
        tgt_l.append(idx[pos])
        usr_l.append(np.full(n - 1, u, dtype=np.int64))
        hist_l.append(H)
        mask_l.append(M)
        gh_l.append(G.astype(np.float32))
    arrays = (np.concatenate(usr_l), np.concatenate(tgt_l),
              np.concatenate(hist_l), np.concatenate(mask_l),
              np.concatenate(gh_l))
    return arrays, (np.array(offsets), np.array(counts))


def _sample_epoch(offsets, counts, rng: np.random.Generator) -> np.ndarray:
    """每用户采样 ≤MAX_PER_USER 个位置，返回全局位置索引（已打乱）。"""
    gs = []
    for off, c in zip(offsets, counts):
        if c == 0:
            continue
        k = min(c, MAX_PER_USER)
        take = rng.choice(c, k, replace=False) if c > k else np.arange(c)
        gs.append(off + take)
    g = np.concatenate(gs)
    rng.shuffle(g)
    return g


def train_twotower(seqs: dict, genre_mh: np.ndarray, year_b: np.ndarray,
                   n_users: int, epochs=None, batch=1024, lr=2e-3, seed=42,
                   dim=None):
    """训练并返回 (model, item_embs[n_items, dim])。"""
    torch.manual_seed(seed)
    device = get_device()
    dim = dim or config.P["tt_dim"]
    epochs = epochs or config.P["tt_epochs"]
    n_items, n_genres = genre_mh.shape
    model = TwoTower(n_users, n_items, n_genres, dim=dim).to(device)
    # Pinned to `device` so GPU index tensors (tgt_t / neg_idx / idx) can
    # gather from these tables without a cross-device indexing error.
    genre_pad = torch.cat([torch.zeros(1, n_genres),
                           torch.tensor(genre_mh)]).to(device)
    year_pad = torch.cat([torch.zeros(1, dtype=torch.long),
                          torch.tensor(year_b.astype(np.int64))]).to(device)
    movie_ids = _movie_ids()
    mid2idx = {int(m): i + 1 for i, m in enumerate(movie_ids)}
    (usr_all, tgt_all, hist_all, mask_all, gh_all), (offsets, counts) = \
        _precompute_samples(seqs, genre_mh, mid2idx)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        g = _sample_epoch(offsets, counts, rng)
        total, n_batch = 0.0, 0
        for s in range(0, len(g), batch):
            gc = g[s:s + batch]
            B = len(gc)
            user_t = torch.from_numpy(usr_all[gc]).to(device)
            tgt_t = torch.from_numpy(tgt_all[gc]).to(device)
            hist_t = torch.from_numpy(hist_all[gc]).to(device)
            mask_t = torch.from_numpy(mask_all[gc]).to(device)
            gh_t = torch.from_numpy(gh_all[gc]).to(device)
            u = model.user_forward(user_t, hist_t, mask_t, gh_t)
            pos = model.item_forward(tgt_t, genre_pad[tgt_t], year_pad[tgt_t])
            neg_idx = torch.randint(1, n_items + 1, (UNIFORM_NEG,),
                                    device=device)
            neg = model.item_forward(neg_idx, genre_pad[neg_idx],
                                     year_pad[neg_idx])
            logits = torch.cat([u @ pos.T, u @ neg.T], dim=1) / TEMP
            loss = F.cross_entropy(logits, torch.arange(B, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batch += 1
        print(f"  [twotower] epoch {epoch + 1}/{epochs}  loss={total / n_batch:.4f}"
              + (f"  device={device}" if epoch == 0 else ""))

    # 离线预计算全量 item embedding（产物统一落 CPU）
    model.eval()
    embs = []
    with torch.no_grad():
        for s in range(0, n_items, 1024):
            idx = torch.arange(s + 1, min(s + 1024, n_items) + 1,
                               device=device)
            embs.append(model.item_forward(idx, genre_pad[idx],
                                           year_pad[idx]).cpu())
    return model, torch.cat(embs).cpu().numpy().astype(np.float32)


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


class TwoTowerRecall:
    """在线召回：user tower 在线前向（O1）+ Faiss ANN 检索。"""

    def __init__(self, model, item_embs, movie_ids, genre_mh, n_users):
        import faiss
        self.device = get_device()
        self.model = model.eval().to(self.device)
        self.movie_ids = movie_ids
        self.item_embs = item_embs                  # 供粗排/MMR 复用
        self.mid2idx = {int(m): i + 1 for i, m in enumerate(movie_ids)}
        self.mid2row = {int(m): i for i, m in enumerate(movie_ids)}
        self.genre_mh = genre_mh
        self.n_genres = genre_mh.shape[1]
        self.n_users = n_users
        if config.P["faiss"] == "hnsw" and item_embs.shape[0] > 20000:
            # 大规模：HNSW 近似检索（兑现 D1 预留）
            self.index = faiss.IndexHNSWFlat(item_embs.shape[1], 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            self.index.add(item_embs)
        else:
            self.index = faiss.IndexFlatIP(item_embs.shape[1])
            self.index.add(item_embs)

    @classmethod
    def load(cls):
        feats = np.load(config.ART_DIR / "item_feats.npz", allow_pickle=True)
        genre_mh = feats["genre_multihot"]
        movie_ids = _movie_ids()
        n_users = int(pd.read_parquet(
            config.ART_DIR / "users.parquet").user_id.max())
        model = TwoTower(n_users, len(movie_ids), genre_mh.shape[1],
                         dim=config.P["tt_dim"])
        model.load_state_dict(torch.load(
            config.ART_DIR / "twotower.pt", map_location=str(get_device()),
            weights_only=True))
        item_embs = np.load(config.ART_DIR / "tt_item_emb.npy")
        return cls(model, item_embs, movie_ids, genre_mh, n_users)

    def _genre_hist(self, hist_mids: list) -> np.ndarray:
        h = [m for m in hist_mids[-HIST_K:] if m in self.mid2row]
        if not h:
            return np.zeros(self.n_genres, dtype=np.float32)
        rows = [self.mid2row[m] for m in h]
        return self.genre_mh[rows].sum(0) / len(rows)

    def user_embed(self, user_id: int, hist_mids: list) -> np.ndarray:
        """O1 在线前向：由当前历史状态计算用户向量（归一化，[dim]）。"""
        h = [m for m in hist_mids[-HIST_K:] if m in self.mid2idx]
        hist = np.zeros((1, HIST_K), dtype=np.int64)
        mask = np.zeros((1, HIST_K), dtype=np.float32)
        hist[0, :len(h)] = [self.mid2idx[m] for m in h]
        mask[0, :len(h)] = 1.0
        uid = user_id if 1 <= user_id <= self.n_users else 0   # 未知用户走 0 号
        with torch.no_grad():
            return self.model.user_forward(
                torch.tensor([uid], device=self.device),
                torch.tensor(hist, device=self.device),
                torch.tensor(mask, device=self.device),
                torch.tensor(self._genre_hist(hist_mids),
                             device=self.device).unsqueeze(0)
            ).cpu().numpy()[0]

    def recall(self, user_id: int, hist_mids: list, exclude: set, topn: int):
        u = self.user_embed(user_id, hist_mids).reshape(1, -1)
        k = min(self.index.ntotal, topn * 5)
        scores, pos = self.index.search(u, k)
        out = []
        for s, p in zip(scores[0], pos[0]):
            if p < 0:
                continue
            mid = int(self.movie_ids[p])
            if mid in exclude:
                continue
            out.append((mid, float(s)))
            if len(out) >= topn:
                break
        return out
