"""RQ-VAE 语义 ID（M7，文档 01 趋势一：TIGER 的表征基础）。

- 输入：item 表示 = 双塔向量 ⊕ 内容向量（两块均已 L2 归一，等权拼接）；
- 残差量化：L 级 × K 码本，前缀层级语义（共享一级码 ≈ 粗粒度相似，
  二级细化，三级近似唯一）；
- 直通估计（STE）+ commitment loss，CPU 训练分钟级（3883 个样本）；
- 产出：sem_ids.npy [n_items, L]（行对齐 movies）+ rqvae.pt（新片冷启动
  时对内容向量编码分配语义 ID）。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config

L_LEVELS = config.P["rq_levels"]
K_CODES = config.P["rq_k"]
DIM_HIDDEN = 128
EPOCHS = None                       # None = config.P["rq_epochs"]
BETA = 0.25                 # commitment loss 权重


class RQVAE(nn.Module):
    def __init__(self, dim_in, dim_h=DIM_HIDDEN, k=K_CODES, levels=L_LEVELS):
        super().__init__()
        self.levels = levels
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, 256), nn.ReLU(), nn.Linear(256, dim_h))
        self.codebooks = nn.ModuleList(
            [nn.Embedding(k, dim_h) for _ in range(levels)])
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -0.05, 0.05)
        self.decoder = nn.Sequential(
            nn.Linear(dim_h, 256), nn.ReLU(), nn.Linear(256, dim_in))

    def quantize(self, z):
        """残差量化：返回 (codes [n, L], z_q [n, dim_h])。"""
        r = z
        codes, qs = [], []
        for cb in self.codebooks:
            idx = torch.cdist(r, cb.weight).argmin(1)
            e = cb(idx)
            codes.append(idx)
            qs.append(e)
            r = r - e
        return torch.stack(codes, 1), torch.stack(qs, 0).sum(0)

    def forward(self, x):
        z = self.encoder(x)
        codes, z_q = self.quantize(z)
        z_q_ste = z + (z_q - z).detach()           # 直通估计
        x_hat = self.decoder(z_q_ste)
        loss = (torch.nn.functional.mse_loss(x_hat, x)
                + BETA * torch.nn.functional.mse_loss(z, z_q.detach()))
        return loss, codes


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


def train_rqvae(items_mat: np.ndarray, epochs=None, seed=42):
    """训练 RQ-VAE 并返回 (model, codes [n, L] numpy)。"""
    torch.manual_seed(seed)
    epochs = epochs or config.P["rq_epochs"]
    n, dim_in = items_mat.shape
    model = RQVAE(dim_in)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(items_mat.astype(np.float32))
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for s in range(0, n, 512):
            chunk = x[perm[s:s + 512]]
            loss, _ = model(chunk)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f"  [rqvae] epoch {epoch + 1}/{epochs}  loss={total:.4f}")
    model.eval()
    with torch.no_grad():
        _, codes = model(x)
    return model, codes.numpy().astype(np.int32)


def encode_new(model: RQVAE, vec: np.ndarray) -> np.ndarray:
    """对新片向量分配语义 ID（冷启动：encoder + 残差量化）。"""
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(vec.astype(np.float32)).unsqueeze(0))
        codes, _ = model.quantize(z)
    return codes.numpy()[0].astype(np.int32)


def main() -> None:
    tt = np.load(config.ART_DIR / "tt_item_emb.npy")
    content_path = config.ART_DIR / "content_emb.npy"
    if content_path.exists():
        content = np.load(content_path)
        items_mat = np.hstack([tt, content])       # 两块均已 L2 归一
        print(f"items_mat: {items_mat.shape}（双塔{tt.shape[1]} ⊕ 内容{content.shape[1]}）")
    else:
        items_mat = tt                             # 无内容画像时退化为纯协同向量
        print(f"items_mat: {items_mat.shape}（纯双塔；内容画像缺失）")
    model, codes = train_rqvae(items_mat)
    torch.save(model.state_dict(), config.ART_DIR / "rqvae.pt")
    np.save(config.ART_DIR / "sem_ids.npy", codes)
    tuples = [tuple(c) for c in codes]
    uniq = len(set(tuples))
    dup = sum(1 for t in set(tuples) if tuples.count(t) > 1)
    l1 = len(set(codes[:, 0]))
    print(f"语义 ID：{uniq}/{len(codes)} 唯一（{dup} 个码被多片共享），"
          f"一级码使用 {l1}/{K_CODES}")
    print("saved rqvae.pt + sem_ids.npy")


if __name__ == "__main__":
    main()
