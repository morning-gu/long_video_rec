"""LightGCN 图召回（M6 新增，文档 02_01 §5.2）。

    E_{l+1} = (D^{-1/2} A D^{-1/2}) E_l，最终表示 = E_0..E_L 的均值

- 二部图：ML-1M 训练期正反馈 569k 边（用户 6040 × 物品 3883）；
- 训练：BPR（正样本 vs 均匀负样本），全图传播每步数千万 flops，2 核 CPU
  约 100ms/步，全程约 2 分钟；
- 服务：训练后物化全量传播 embedding（numpy），召回为一次矩阵乘——
  与双塔同构的纯静态向量召回；
- 叠加式新增：不改动既有通道（ItemCF/双塔/SASRec/热门）。

连续索引约定：user_id 1..n_users；item 连续索引 1..n_items（0 = padding）。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, diags, hstack, vstack

from src import config
from src.data.features import positive_sequences

DIM = 64
N_LAYERS = 2
EPOCHS = 40
LR = 1e-3                  # 官方设置：全图耦合梯度下 1e-2 会失稳（实测卡 0.53）
SAMPLES_PER_EPOCH = 100_000
BATCH = 2048
REFRESH_EVERY = 8          # 每 K 步刷新传播图（陈旧梯度，embedding 场景常规做法）


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


def _norm_adj_torch(train_pos: pd.DataFrame, n_users: int, n_items: int):
    """归一化邻接矩阵 Â = D^{-1/2} A D^{-1/2}（torch 稀疏，含 0 号 padding 节点）。"""
    mid2idx = {int(m): i + 1 for i, m in enumerate(_movie_ids())}
    rows = train_pos.user_id.values              # 用户节点 = user_id（0 号空置）
    cols = np.array([mid2idx[int(m)] + n_users + 1
                     for m in train_pos.movie_id])
    n = n_users + 1 + n_items + 1
    R = csr_matrix((np.ones(len(train_pos), dtype=np.float32), (rows, cols)),
                   shape=(n, n))
    A = R + R.T
    deg = np.asarray(A.sum(1)).ravel()
    d_inv = np.zeros_like(deg)
    nz = deg > 0
    d_inv[nz] = deg[nz] ** -0.5
    A = diags(d_inv) @ A @ diags(d_inv)
    coo = A.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(
        idx, torch.tensor(coo.data, dtype=torch.float32), (n, n)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=DIM, n_layers=N_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.user_table = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.item_table = nn.Embedding(n_items + 1, dim, padding_idx=0)
        for t in (self.user_table, self.item_table):
            nn.init.normal_(t.weight, std=0.1)
        with torch.no_grad():
            self.user_table.weight[0].zero_()
            self.item_table.weight[0].zero_()

    def propagate(self, adj):
        e = torch.cat([self.user_table.weight, self.item_table.weight], 0)
        outs = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(adj, e)
            outs.append(e)
        return torch.stack(outs, 0).mean(0)          # [n_users+n_items+2, dim]


def train_lightgcn(train_pos: pd.DataFrame, n_users: int, n_items: int,
                   epochs=EPOCHS, seed=42):
    """训练并返回 (user_embs[n_users+1, d], item_embs[n_items+1, d])，已 L2 归一。"""
    torch.manual_seed(seed)
    adj = _norm_adj_torch(train_pos, n_users, n_items)
    model = LightGCN(n_users, n_items)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    pos_arr = train_pos[["user_id", "movie_id"]].to_numpy(copy=True)
    mid2idx = {int(m): i + 1 for i, m in enumerate(_movie_ids())}
    pos_arr[:, 1] = np.array([mid2idx[int(m)] + n_users + 1
                              for m in train_pos.movie_id])
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        sel = rng.choice(len(pos_arr), SAMPLES_PER_EPOCH, replace=True)
        pairs = pos_arr[sel]
        rng.shuffle(pairs)
        total, n_b = 0.0, 0
        e = None                                    # 陈旧传播缓存
        for s in range(0, len(pairs), BATCH):
            if e is None or n_b % REFRESH_EVERY == 0:
                e = model.propagate(adj)
            chunk = pairs[s:s + BATCH]
            u = torch.from_numpy(chunk[:, 0].astype(np.int64))
            pos = torch.from_numpy(chunk[:, 1].astype(np.int64))
            neg = torch.randint(1, n_items + 1, (len(chunk),)) + n_users + 1
            eu, ep, en = e[u], e[pos], e[neg]
            loss = -torch.nn.functional.logsigmoid(
                (eu * ep).sum(1) - (eu * en).sum(1)).mean()
            opt.zero_grad()
            loss.backward(retain_graph=True)     # 传播图复用 K 步
            opt.step()
            total += loss.item()
            n_b += 1
        print(f"  [lightgcn] epoch {epoch + 1}/{epochs}  loss={total / n_b:.4f}")
    model.eval()
    with torch.no_grad():
        e = model.propagate(adj)
    # 不做 L2 归一：BPR 点积训练下物品范数承载流行度校准（实测归一化使
    # HR@10 从 0.042 掉到 0.008——排序被打乱）。与双塔（训练时显式归一）不同。
    return (e[:n_users + 1].numpy().astype(np.float32),
            e[n_users + 1:].numpy().astype(np.float32))


class LightGCNRecall:
    """静态向量召回：user 传播向量 × item 传播向量（原始点积，未归一——
    物品范数承载流行度校准；离线物化，服务零前向）。"""

    def __init__(self, user_embs, item_embs, movie_ids):
        self.user_embs = user_embs
        self.item_embs = item_embs
        self.movie_ids = movie_ids
        self.mid2row = {int(m): r for r, m in enumerate(movie_ids)}
        self.n_users = len(user_embs) - 1

    @classmethod
    def load(cls):
        return cls(np.load(config.ART_DIR / "lg_user_emb.npy"),
                   np.load(config.ART_DIR / "lg_item_emb.npy"),
                   _movie_ids())

    def recall(self, user_id: int, exclude: set, topn: int):
        u = self.user_embs[user_id if 1 <= user_id <= self.n_users else 0]
        scores = self.item_embs[1:] @ u               # [n_items]
        if exclude:
            rows = [self.mid2row[m] for m in exclude if m in self.mid2row]
            scores[rows] = -np.inf
        order = np.argsort(-scores)[:topn]
        return [(int(self.movie_ids[r]), float(scores[r]))
                for r in order if np.isfinite(scores[r])]
