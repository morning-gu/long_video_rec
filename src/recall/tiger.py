"""TIGER-lite 生成式召回（M7，文档 01 趋势一：生成式检索）。

- 语义 ID 序列语言模型：用户历史 → 每部影片 L 个码字展平 → 因果
  Transformer 做下一 token 预测；推理时自回归**生成**下一部影片的码字
  元组（判别式检索 → 生成式检索）；
- **约束解码防幻觉**：逐步只允许"合法前缀/完整元组"中出现的码字
  （catalog 约束，与 LLM 层同一设计哲学，文档 03 GenRec）；
- 批量 beam search（beam=16，3 步生成），CPU 单次召回数十毫秒；
- 新片冷启动：新内容向量 → RQ-VAE 编码语义 ID → 立即进入生成词表，
  零交互数据即可被召回（语义 ID 的核心卖点）。

码字 → token 映射：code c ∈ [0, K) → token c+1；0 = padding。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.data.features import positive_sequences

MAX_HIST_ITEMS = 50
L_LEVELS = config.P["rq_levels"]
K_CODES = config.P["rq_k"]
VOCAB = K_CODES + 1                 # 0 = pad
MAX_TOKENS = MAX_HIST_ITEMS * L_LEVELS + L_LEVELS + 8
DIM = 96
BEAM = 32
GEN_TOPK = 12
GEN_ROUNDS = 2                      # 第二轮禁用首轮元组，扩展候选覆盖
                                    # （LM 倾向生成已看内容，单轮排除后候选过少）


class Tiger(nn.Module):
    def __init__(self, dim=DIM, layers=2, heads=2, dropout=0.2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.pos = nn.Embedding(MAX_TOKENS, dim)
        nn.init.normal_(self.tok.weight, std=0.02)
        with torch.no_grad():
            self.tok.weight[0].zero_()
        nn.init.normal_(self.pos.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers,
                                              enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("causal",
                             torch.triu(torch.ones(MAX_TOKENS, MAX_TOKENS,
                                                   dtype=torch.bool), 1))

    def forward(self, tokens, mask):
        """tokens: [B, T] 右填充 0；mask: [B, T] bool（True=真实）。"""
        B, T = tokens.shape
        pos = torch.arange(T).unsqueeze(0).expand(B, T)
        h = self.tok(tokens) + self.pos(pos)
        h = self.encoder(h, mask=self.causal[:T, :T],
                         src_key_padding_mask=~mask)
        return self.norm(h) @ self.tok.weight.T       # weight tying → [B,T,V]


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


def train_tiger(seqs: dict, epochs=15, batch=64, lr=1e-3, seed=42):
    """训练语义 ID 序列 LM。seqs: {user: [mid...] 升序}。"""
    torch.manual_seed(seed)
    sem_ids = np.load(config.ART_DIR / "sem_ids.npy")      # [n_items, L]
    mid2row = {int(m): i for i, m in enumerate(_movie_ids())}
    model = Tiger()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200))
    data = []
    for seq in seqs.values():
        if len(seq) >= 2:
            for m in seq[-(MAX_HIST_ITEMS + 1):]:
                if m not in mid2row:
                    break
            else:
                data.append([m for m in seq[-(MAX_HIST_ITEMS + 1):]
                             if m in mid2row])
    rng = np.random.default_rng(seed)
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(data))
        total, n_b = 0.0, 0
        for s in range(0, len(order), batch):
            chunk = [data[i] for i in order[s:s + batch]]
            toks = [[c + 1 for m in itemlist for c in sem_ids[mid2row[m]]]
                    for itemlist in chunk]
            T = max(len(t) for t in toks)
            inp = np.zeros((len(toks), T), dtype=np.int64)
            msk = np.zeros((len(toks), T), dtype=bool)
            for b, t in enumerate(toks):
                inp[b, :len(t)] = t
                msk[b, :len(t)] = True
            inp_t = torch.tensor(inp)
            logits = model(inp_t, torch.tensor(msk))
            tgt = np.zeros_like(inp)
            tgt[:, :-1] = inp[:, 1:]                       # 下一 token
            loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                                   torch.tensor(tgt).reshape(-1),
                                   ignore_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            total += loss.item()
            n_b += 1
        print(f"  [tiger] epoch {epoch + 1}/{epochs}  loss={total / n_b:.4f}")
    model.eval()
    return model


class TigerRecall:
    """在线生成式召回：当前历史 → 约束 beam 生成码字元组 → 映射物品。"""

    def __init__(self, model, sem_ids, movie_ids):
        self.model = model
        self.sem_ids = sem_ids
        self.movie_ids = movie_ids
        self.mid2row = {int(m): i for i, m in enumerate(movie_ids)}
        # catalog 约束结构：合法前缀与元组→物品（预计算，避免逐请求扫描）
        self.tuple2rows = {}
        for r, codes in enumerate(sem_ids):
            self.tuple2rows.setdefault(tuple(int(c) for c in codes), []).append(r)
        self.prefix1 = {t[0] for t in self.tuple2rows}
        self.c2map = {}                     # c1 → {c2}
        self.prefix2 = {}                   # (c1, c2) → {c3}
        for t in self.tuple2rows:
            self.c2map.setdefault(t[0], set()).add(t[1])
            self.prefix2.setdefault(t[:2], set()).add(t[2])

    @classmethod
    def load(cls):
        model = Tiger()
        model.load_state_dict(torch.load(
            config.ART_DIR / "tiger.pt", map_location="cpu",
            weights_only=True))
        return cls(model.eval(), np.load(config.ART_DIR / "sem_ids.npy"),
                   _movie_ids())

    def register(self, movie_id: int, codes: np.ndarray):
        """注册新片（冷启动）：加入生成词表与元组映射。"""
        row = len(self.movie_ids)
        self.movie_ids = np.append(self.movie_ids, movie_id)
        self.mid2row[int(movie_id)] = row
        t = tuple(int(c) for c in codes)
        self.tuple2rows.setdefault(t, []).append(row)
        self.prefix1.add(t[0])
        self.c2map.setdefault(t[0], set()).add(t[1])
        self.prefix2.setdefault(t[:2], set()).add(t[2])
        self.sem_ids = np.vstack([self.sem_ids, codes.reshape(1, -1)])

    def _hist_tokens(self, hist_mids):
        rows = [self.mid2row[m] for m in hist_mids[-MAX_HIST_ITEMS:]
                if m in self.mid2row]
        if not rows:
            return None
        return [int(c) + 1 for r in rows for c in self.sem_ids[r]]

    def recall(self, hist_mids: list, exclude: set, topn: int,
               beam: int = BEAM):
        tokens = self._hist_tokens(hist_mids)
        if tokens is None:
            return []
        banned = set()
        best = {}
        for _round in range(GEN_ROUNDS):
            with torch.no_grad():
                beams = [(0.0, list(tokens))]
                for step in range(L_LEVELS):
                    inp = np.array([b[1] for b in beams], dtype=np.int64)
                    msk = np.ones_like(inp, dtype=bool)
                    logits = self.model(torch.tensor(inp),
                                        torch.tensor(msk))[:, -1, :].numpy()
                    cands = []
                    for bi, (lp, toks) in enumerate(beams):
                        # 约束解码：只允许合法前缀中出现的码字（catalog 约束）
                        if step == 0:
                            allowed = self.prefix1
                        elif step == 1:
                            allowed = self.c2map.get(toks[-1] - 1, set())
                        else:
                            allowed = self.prefix2.get(
                                (toks[-2] - 1, toks[-1] - 1), set())
                            allowed = {c for c in allowed
                                       if (toks[-2] - 1, toks[-1] - 1, c)
                                       not in banned}
                        mask = np.full(VOCAB, -np.inf)
                        for c in allowed:
                            mask[c + 1] = 0.0
                        s = logits[bi] + mask
                        top = np.argsort(-s)[:GEN_TOPK]
                        for t in top:
                            if not np.isfinite(s[t]):
                                continue
                            cands.append((lp + float(s[t]),
                                          toks + [int(t)]))
                    cands.sort(key=lambda x: -x[0])
                    beams = cands[:beam]
            for lp, toks in beams:
                codes = tuple(t - 1 for t in toks[-L_LEVELS:])
                banned.add(codes)
                for r in self.tuple2rows.get(codes, []):
                    mid = int(self.movie_ids[r])
                    if mid not in best or lp > best[mid]:
                        best[mid] = lp
        res = [(mid, lp) for mid, lp in best.items() if mid not in exclude]
        res.sort(key=lambda x: -x[1])
        return res[:topn]
