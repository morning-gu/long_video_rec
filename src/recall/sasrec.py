"""SASRec-lite 序列召回（设计文档 §4.1）：自回归 next-item 预测。

- 输入：用户最近 MAX_LEN 部正反馈（时间升序，来自用户状态实时层，O1 在线前向）；
- 结构：item embedding + 位置 embedding + 因果自注意力 → 最后位置隐状态打分；
- 输出投影与输入 embedding 权重共享（weight tying）；
- 训练：每序列随机采样 LOSS_POSITIONS 个位置算交叉熵（保留完整上下文，
  将 [B, L, V] 全 softmax 的开销压缩约 4×，适配 2 核 CPU）；
- 选型依据（D6）：自回归目标与线上 next-item 一致，无 BERT4Rec 的训练-推理偏差。

连续索引约定与 twotower 相同：movie_id → idx = 行号 + 1，0 = padding。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.device import get_device

MAX_LEN = 50
LOSS_POSITIONS = 12       # 每序列每 epoch 采样参与 loss 的位置数
WARMUP_STEPS = 200


class SASRec(nn.Module):
    def __init__(self, n_items, dim=64, n_layers=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.item_table = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_LEN, dim)
        # 小方差初始化：weight tying 下避免初始 logits 爆炸（loss 从 40+ 回到 ~ln V）
        nn.init.normal_(self.item_table.weight, std=0.02)
        with torch.no_grad():
            self.item_table.weight[0].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, n_heads, dim * 2, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("causal",
                             torch.triu(torch.ones(MAX_LEN, MAX_LEN,
                                                   dtype=torch.bool), 1))

    def forward(self, seq_idx, attn_mask):
        """seq_idx: [B, L] 右填充 0；attn_mask: [B, L] bool（True=真实 item）。

        返回 [B, L, dim]；因果注意力保证位置 i 只看 ≤i。
        """
        B, L = seq_idx.shape
        pos = torch.arange(L, device=seq_idx.device).unsqueeze(0).expand(B, L)
        h = self.item_table(seq_idx) + self.pos_emb(pos)
        h = self.encoder(h, mask=self.causal[:L, :L],
                         src_key_padding_mask=~attn_mask)
        return self.norm(h)


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


def load_sas_model():
    """加载训练好的 SASRec 权重（eval 态）。返回 (model, movie_ids)。"""
    movie_ids = _movie_ids()
    model = SASRec(len(movie_ids), dim=config.P["sas_dim"],
                   n_layers=config.P["sas_layers"])
    model.load_state_dict(torch.load(
        config.ART_DIR / "sasrec.pt", map_location=str(get_device()),
        weights_only=True))
    return model.eval(), movie_ids


def train_sasrec(seqs: dict, epochs=None, batch=128, lr=1e-3, seed=42):
    """训练并返回 model。seqs: {user: [mid...] 升序}，仅使用长度 ≥2 的序列。"""
    torch.manual_seed(seed)
    device = get_device()
    epochs = epochs or config.P["sas_epochs"]
    movie_ids = _movie_ids()
    n_items = len(movie_ids)
    mid2idx = {int(m): i + 1 for i, m in enumerate(movie_ids)}
    model = SASRec(n_items, dim=config.P["sas_dim"],
                   n_layers=config.P["sas_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP_STEPS))
    data = [seq[-(MAX_LEN + 1):] for seq in seqs.values() if len(seq) >= 2]
    rng = np.random.default_rng(seed)
    step = 0

    for epoch in range(epochs):
        order = rng.permutation(len(data))
        total, n_b, n_pos = 0.0, 0, 0
        for s in range(0, len(order), batch):
            chunk = [data[i] for i in order[s:s + batch]]
            B = len(chunk)
            L = max(len(c) - 1 for c in chunk)
            inp = np.zeros((B, L), dtype=np.int64)
            tgt = np.zeros((B, L), dtype=np.int64)
            mask = np.zeros((B, L), dtype=bool)
            for b, c in enumerate(chunk):
                l = len(c) - 1
                inp[b, :l] = [mid2idx[m] for m in c[:-1]]
                tgt[b, :l] = [mid2idx[m] for m in c[1:]]
                mask[b, :l] = True
            h = model(torch.tensor(inp, device=device),
                      torch.tensor(mask, device=device))     # [B, L, dim]
            tgt_t = torch.tensor(tgt, device=device)
            # 每行随机采样 ≤LOSS_POSITIONS 个真实位置参与 loss
            rand = torch.rand(B, L, device=device)
            rand[tgt_t == 0] = 2.0
            k = min(LOSS_POSITIONS, L)
            cols = rand.topk(k, dim=1, largest=False).indices     # [B, k]
            rows = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            h_sel, t_sel = h[rows, cols], tgt_t[rows, cols]       # [B, k]
            keep = t_sel != 0
            logits = h_sel[keep] @ model.item_table.weight.T      # weight tying
            loss = F.cross_entropy(logits, t_sel[keep])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            total += loss.item()
            n_b += 1
            n_pos += int(keep.sum())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [sasrec] epoch {epoch + 1}/{epochs}  "
                  f"loss={total / n_b:.4f}  (每步平均 {n_pos // n_b} 个位置"
                  + (f"，device={device}" if epoch == 0 else ")"))
    model.eval()
    return model


class SASRecRecall:
    """在线召回：当前序列在线编码（O1）→ 全库打分 Top-N。"""

    def __init__(self, model, movie_ids):
        self.device = get_device()
        self.model = model.eval().to(self.device)
        self.movie_ids = movie_ids
        self.mid2idx = {int(m): i + 1 for i, m in enumerate(movie_ids)}

    @classmethod
    def load(cls):
        model, movie_ids = load_sas_model()
        return cls(model, movie_ids)

    def recall(self, seq_mids_ascending: list, exclude: set, topn: int):
        s = [self.mid2idx[m] for m in seq_mids_ascending[-MAX_LEN:]
             if m in self.mid2idx]
        if not s:
            return []
        L = len(s)
        inp = np.zeros((1, MAX_LEN), dtype=np.int64)
        mask = np.zeros((1, MAX_LEN), dtype=bool)
        inp[0, :L] = s
        mask[0, :L] = True
        with torch.no_grad():
            h = self.model(torch.tensor(inp, device=self.device),
                           torch.tensor(mask, device=self.device))
            last = h[0, L - 1]                          # 最后一个真实位置
            scores = (last @ self.model.item_table.weight.T).cpu().numpy()
        scores[0] = -np.inf                             # padding 槽位
        banned = [self.mid2idx[m] for m in exclude if m in self.mid2idx]
        scores[banned] = -np.inf
        order = np.argsort(-scores)[:topn]
        return [(int(self.movie_ids[i - 1]), float(scores[i]))
                for i in order if np.isfinite(scores[i])]
