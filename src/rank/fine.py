"""DeepFM-lite 精排（设计文档 §4.3）。

- FM 部分：稀疏字段二阶交叉——user / movie / 年代桶 / 时段桶 / 周末 / 18 个类型槽位；
- DNN 部分：全量字段 embedding 拼接 + 稠密特征 → MLP（高阶交叉）；
- 稠密特征：用户活跃度、类型匹配度（用户类型直方图 × 电影类型）、序列相似度
  （DIN 简化版：目标电影双塔 embedding 与用户最近正反馈 embedding 均值的内积）、
  热度（log 正反馈数）、口碑（贝叶斯平均评分 wr）；
- 单目标（正反馈概率），正负样本 1:4 均匀负采样（曝光偏差缓解列为扩展，§4.3）；
- 多目标接口预留：forward 可扩展多头输出（ML-1M 无多行为标签，不伪造，决策 D3）。

连续索引约定：movie_id → idx = 行号 + 1（0 = padding/未知），与召回层一致。
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.device import get_device

HIST_K = 32                 # 序列相似度/类型偏好窗口（与双塔一致）
NEG_RATIO = 4               # 负采样比例
MAX_PER_USER = 20           # 每用户每 epoch 采样正样本位置数
FIELD_DIMS = None           # 由 n_users/n_items 动态确定


def _hour_weekend(ts: np.ndarray):
    """时间上下文（§4.6）：时段桶 1早(5-11) 2午(12-17) 3晚(18-23) 4夜(0-4)；周末 1/2。"""
    h = (ts // 3600) % 24
    hb = np.where(h >= 18, 3, np.where(h >= 12, 2, np.where(h >= 5, 1, 4)))
    day = ts // 86400
    wk = (((day + 4) % 7 >= 5).astype(np.int64)) + 1   # 1970-01-01 为周四
    return hb.astype(np.int64), wk


class DeepFM(nn.Module):
    """23 个稀疏字段（5 基础 + 18 类型槽位）+ 稠密特征。

    n_dense：v1=6（M3）；v2=8（M6，+内容相似度/基调匹配，见 train_fine）。
    """

    N_FIELDS = 5 + 18
    N_DENSE = 6

    def __init__(self, n_users, n_items, n_genres, emb_dim=16,
                 hidden=(128, 64), n_dense=N_DENSE):
        super().__init__()
        self.n_dense = n_dense
        # 字段：user / movie / year / hour / weekend / genre×18（均 0=padding）
        dims = [n_users + 1, n_items + 1, 11, 6, 3] + [n_genres + 1] * n_genres
        self.emb = nn.ModuleList([
            nn.Embedding(d, emb_dim, padding_idx=0) for d in dims])
        self.lin = nn.ModuleList([
            nn.Embedding(d, 1, padding_idx=0) for d in dims])
        self.lin_dense = nn.Linear(n_dense, 1)
        self.mlp = nn.Sequential(
            nn.Linear(self.N_FIELDS * emb_dim + n_dense, hidden[0]),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1))

    def forward(self, x_idx, x_val, x_dense):
        """x_idx/x_val: [B, F]；x_dense: [B, D]。返回 logits。"""
        e = torch.stack([m(x_idx[:, i]) for i, m in enumerate(self.emb)], 1)
        e = e * x_val.unsqueeze(-1)                     # [B, F, E]
        lin = torch.stack([m(x_idx[:, i]) for i, m in enumerate(self.lin)], 1)
        lin = (lin.squeeze(-1) * x_val).sum(1) + self.lin_dense(x_dense).squeeze(-1)
        s = e.sum(1)                                    # FM: 0.5*(||Σe||²-Σ||e||²)
        fm = 0.5 * (s.pow(2).sum(1) - e.pow(2).sum((1, 2)))
        dnn = self.mlp(torch.cat([e.flatten(1), x_dense], 1)).squeeze(-1)
        return lin + fm + dnn


class _ItemStats:
    """物品侧静态特征（行对齐 movies parquet）。"""

    def __init__(self, genre_mh, year_b, tt_embs, hot_df,
                 content=None, mood=None):
        self.genre_mh = genre_mh
        self.year = year_b.astype(np.int64)
        self.tt = tt_embs
        self.content = content      # [n_items, D] 内容向量（M6，可 None）
        self.mood = mood            # [n_items, n_moods] one-hot（M6，可 None）
        n_items, n_genres = genre_mh.shape
        ar = np.arange(1, n_genres + 1)
        self.genre_idx = (genre_mh * ar).astype(np.int64)      # 槽位激活时=类型id
        self.genre_val = genre_mh.astype(np.float32)
        wr = dict(zip(hot_df.movie_id.astype(int), hot_df.wr.astype(float)))
        pc = dict(zip(hot_df.movie_id.astype(int), hot_df.pos_count.astype(int)))
        self.wr = np.array([wr.get(int(m), 3.0) for m in _movie_ids()],
                           dtype=np.float32)
        self.pop_log = (np.log1p([pc.get(int(m), 0) for m in _movie_ids()])
                        / 8.0).astype(np.float32)
        self.pos_count = np.array(
            [pc.get(int(m), 0) for m in _movie_ids()], dtype=np.float32)


def _movie_ids() -> np.ndarray:
    return pd.read_parquet(config.ART_DIR / "movies.parquet").movie_id.values


def _precompute_positions(train_pos: pd.DataFrame, genre_mh, tt_embs, mid2idx,
                          sas_model=None):
    """全部训练位置的 (user, target, ts, hist_genre, hist_tt, sas_prev, sas_score)。

    前缀和向量化，无逐位置 Python 循环。hist 严格取目标之前的行为（无泄漏）：
    cg[p] = 前 p 个物品的前缀和，hist(p) = (cg[p] - cg[max(0,p-K)]) / len。
    sas_prev[p] = SASRec 在前缀 seq[:p] 上的末位隐状态（h_{p-1}，无泄漏），
    sas_score[p] = sigmoid(h_{p-1} · emb(target))——序列模型的 next-item 分数
    作为精排特征（排序模型消费最强信号，工业常规做法）。
    返回全局数组 + 每用户 (offsets, counts)。
    """
    K = HIST_K
    tp = train_pos.sort_values(["user_id", "timestamp"], kind="stable")
    usr_l, tgt_l, ts_l, hg_l, ht_l, hp_l, ss_l = [], [], [], [], [], [], []
    sas_W = (sas_model.item_table.weight.detach().cpu().numpy()
             if sas_model is not None else None)
    offsets, counts = [], []
    cursor = 0
    for u, g in tp.groupby("user_id"):
        idx = np.array([mid2idx[int(m)] for m in g.movie_id], dtype=np.int64)
        ts = g.timestamp.values.astype(np.int64)
        n = len(idx)
        offsets.append(cursor)
        counts.append(n)
        cursor += n
        gr = genre_mh[idx - 1]                          # [n, G]
        tr = tt_embs[idx - 1]                           # [n, 64]
        cg = np.vstack([np.zeros((1, gr.shape[1])), np.cumsum(gr, 0)])
        ct = np.vstack([np.zeros((1, tr.shape[1])), np.cumsum(tr, 0)])
        pos = np.arange(n)
        starts = np.maximum(0, pos - K)
        lens = np.maximum(pos - starts, 1)              # pos=0 时分子为 0
        usr_l.append(np.full(n, int(u), dtype=np.int64))
        tgt_l.append(idx)
        ts_l.append(ts)
        hg_l.append((cg[pos] - cg[starts]) / lens[:, None])
        ht_l.append((ct[pos] - ct[starts]) / lens[:, None])
        if sas_model is not None:
            # 只对最近 MAX_LEN 部做一次前向（SASRec 位置编码上限 50）：
            # 对窗口内目标 p，上下文 = 窗口起点到 p-1（因果、无泄漏）；
            # 窗口外目标特征置 0.5（中性，训练采样已限定在窗口内）。
            from src.recall.sasrec import MAX_LEN
            win = idx[-MAX_LEN:]
            m = len(win)
            with torch.no_grad():
                inp = np.zeros((1, m), dtype=np.int64)
                mask = np.ones((1, m), dtype=bool)
                inp[0] = win
                dev = next(sas_model.parameters()).device
                h = sas_model(torch.tensor(inp, device=dev),
                              torch.tensor(mask, device=dev)
                              )[0].cpu().numpy()                # [m, 64]
            Hp = np.zeros((n, h.shape[1]), dtype=np.float32)
            ss = np.zeros(n, dtype=np.float32)
            for p in range(max(1, n - m + 1), n):
                j = p - 1 - (n - m)          # 窗口内第 j 项 = 位置 p-1
                Hp[p] = h[j]
                # 原始 logit / 10：SASRec logits 达 ±30，sigmoid 会饱和，
                # 顶部候选间差异被压平，特征失去候选集内排序信息
                ss[p] = (h[j] * sas_W[idx[p]]).sum() / 10.0
            hp_l.append(Hp)
            ss_l.append(ss)
    arrays = (np.concatenate(usr_l), np.concatenate(tgt_l), np.concatenate(ts_l),
              np.concatenate(hg_l).astype(np.float32),
              np.concatenate(ht_l).astype(np.float32),
              np.concatenate(hp_l) if hp_l else None,
              np.concatenate(ss_l) if ss_l else None)
    return arrays, (np.array(offsets), np.array(counts))


def train_fine(train_pos, ratings, genre_mh, year_b, tt_embs, hot_df,
               n_users, epochs=6, batch=1024, lr=1e-3, seed=42,
               with_content=False):
    """训练 DeepFM，返回 model。

    with_content=False → v1（6 稠密特征，fine.pt，M3）；
    with_content=True  → v2（+内容相似度/基调匹配 2 特征，fine_v2.pt，M6）。
    内容特征（M6）：用户全量正反馈的内容向量和 S_u / 基调计数 M_u——
    正样本行按"排除目标自身"计算（训练历史含目标，防泄漏）；负/服务行
    无排除（候选未看过），训练-服务口径一致。

    负采样为流行度加权（unigram^0.75）：均匀负采样下"流行"本身即完美正样本
    特征，模型学会流行度捷径（AUC 虚高但真实候选集上失效）；流行度负采样
    迫使个性化特征（类型匹配/序列相似度）承担判别（文档 01 §5.2）。
    """
    torch.manual_seed(seed)
    movie_ids = _movie_ids()
    n_items, n_genres = genre_mh.shape
    mid2idx = {int(m): i + 1 for i, m in enumerate(movie_ids)}
    content = mood_m = None
    S_u = M_u = n_u = None
    if with_content:
        content = np.load(config.ART_DIR / "content_emb.npy")
        from src.data.content import mood_onehot
        mood_m, _ = mood_onehot()
        S_u = np.zeros((n_users + 1, content.shape[1]), dtype=np.float32)
        M_u = np.zeros((n_users + 1, mood_m.shape[1]), dtype=np.float32)
        n_u = np.zeros(n_users + 1, dtype=np.float32)
        for u, m in zip(train_pos.user_id.values, train_pos.movie_id.values):
            r = mid2idx[int(m)] - 1
            S_u[int(u)] += content[r]
            M_u[int(u)] += mood_m[r]
            n_u[int(u)] += 1
    st = _ItemStats(genre_mh, year_b, tt_embs, hot_df,
                    content=content, mood=mood_m)
    from src.recall.sasrec import load_sas_model
    sas_model, _ = load_sas_model()
    sas_W = sas_model.item_table.weight.detach().cpu().numpy()   # [V, 64]
    (usr_all, tgt_all, ts_all, hg_all, ht_all, hp_all, ss_pos_all), \
        (offsets, counts) = _precompute_positions(
            train_pos, genre_mh, tt_embs, mid2idx, sas_model)
    activity = np.zeros(n_users + 1, dtype=np.float32)
    rated_mask = np.zeros((n_users + 1, n_items + 1), dtype=bool)
    for u, g in ratings.groupby("user_id"):
        u = int(u)
        if 1 <= u <= n_users:
            activity[u] = np.log1p(len(g)) / 10.0
            rated_mask[u, [mid2idx[int(m)] for m in g.movie_id]] = True
    item_universe = np.arange(1, n_items + 1)
    pop_w = st.pos_count ** 0.75
    pop_p = pop_w / pop_w.sum()
    device = get_device()
    model = DeepFM(n_users, n_items, n_genres,
                   n_dense=8 if with_content else 6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        gs = []
        for off, c in zip(offsets, counts):
            if c <= 1:
                continue
            # 只采样最近 49 个目标位置：SASRec 特征的有效窗口
            # （也使训练分布偏向近期行为，与线上 next-item 语义一致）
            lo = max(1, c - 49)
            span = c - lo
            k = min(span, MAX_PER_USER)
            take = rng.choice(span, k, replace=False) if span > k \
                else np.arange(span)
            gs.append(off + lo + take)
        g = np.concatenate(gs)
        rng.shuffle(g)
        total, n_b = 0.0, 0
        for s in range(0, len(g), batch):
            gc = g[s:s + batch]
            P = len(gc)
            # 1:4 流行度加权负采样（与正样本共享用户与时间上下文），
            # 向量化剔除与用户已看冲突者
            negs = rng.choice(item_universe, size=(P, NEG_RATIO), p=pop_p)
            keep = ~rated_mask[usr_all[gc][:, None], negs]     # [P, NEG_RATIO]
            g_pos, t_pos = gc, tgt_all[gc]
            g_neg = np.repeat(gc, NEG_RATIO)[keep.ravel()]
            t_neg = negs.ravel()[keep.ravel()]
            rg = np.concatenate([g_pos, g_neg])
            rt = np.concatenate([t_pos, t_neg])
            rl = np.concatenate([np.ones(P, dtype=np.float32),
                                 np.zeros(len(t_neg), dtype=np.float32)])
            # SASRec 分数特征：正样本查表，负样本由 h_{p-1}·emb(neg) 现算
            # （原始 logit / 10，与 _precompute_positions 口径一致）
            ss_neg = (hp_all[g_neg] * sas_W[t_neg]).sum(1) / 10.0
            ss = np.concatenate([ss_pos_all[g_pos], ss_neg]).astype(np.float32)
            extra = None
            if with_content:
                # 正样本行：排除目标自身（其内容向量在 S_u 中，防泄漏）
                up, tp = usr_all[g_pos], t_pos - 1
                S_p, v_p = S_u[up], content[tp]
                dot_p = (S_p * v_p).sum(1)
                norm_p = np.linalg.norm(S_p - v_p, axis=1)
                csim_p = np.where(n_u[up] > 1,
                                  (dot_p - 1) / np.maximum(norm_p, 1e-9), 0.0)
                mm_p = np.where(n_u[up] > 1,
                                ((M_u[up] * mood_m[tp]).sum(1) - 1)
                                / np.maximum(n_u[up] - 1, 1), 0.0)
                # 负样本行：目标不在历史，无排除（与服务口径一致）
                un, tn = usr_all[g_neg], t_neg - 1
                csim_n = ((S_u[un] * content[tn]).sum(1)
                          / np.maximum(np.linalg.norm(S_u[un], axis=1), 1e-9))
                mm_n = (M_u[un] * mood_m[tn]).sum(1) / np.maximum(n_u[un], 1)
                extra = np.stack([np.concatenate([csim_p, csim_n]),
                                  np.concatenate([mm_p, mm_n])],
                                 1).astype(np.float32)
            idx_mat, val_mat, dense = _assemble(
                rg, rt, usr_all, ts_all, hg_all, ht_all, st, activity, ss,
                extra)
            logits = model(idx_mat.to(device), val_mat.to(device),
                           dense.to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.tensor(rl, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_b += 1
        print(f"  [fine] epoch {epoch + 1}/{epochs}  loss={total / n_b:.4f}"
              + (f"  device={device}" if epoch == 0 else ""))
    model.eval()
    return model


def _assemble(g, tgt, usr_all, ts_all, hg_all, ht_all, st, activity, ss,
              extra=None):
    """给定（位置 id × 目标物品）行集合，组装模型输入。正负样本共用。"""
    usr = usr_all[g]
    hb, wk = _hour_weekend(ts_all[g])
    hg, ht = hg_all[g], ht_all[g]
    n = len(g)
    gm = (hg * st.genre_mh[tgt - 1]).sum(1)
    tsim = (ht * st.tt[tgt - 1]).sum(1)
    idx_mat = np.zeros((n, DeepFM.N_FIELDS), dtype=np.int64)
    idx_mat[:, 0] = usr
    idx_mat[:, 1] = tgt
    idx_mat[:, 2] = st.year[tgt - 1]
    idx_mat[:, 3] = hb
    idx_mat[:, 4] = wk
    idx_mat[:, 5:] = st.genre_idx[tgt - 1]
    val_mat = np.ones((n, DeepFM.N_FIELDS), dtype=np.float32)
    val_mat[:, 5:] = st.genre_val[tgt - 1]
    dense = np.stack([activity[usr], gm.astype(np.float32),
                      tsim.astype(np.float32), st.pop_log[tgt - 1],
                      st.wr[tgt - 1] / 5.0, ss], 1)
    if extra is not None:                    # M6 v2：内容相似度 + 基调匹配
        dense = np.hstack([dense, extra])
    return (torch.from_numpy(idx_mat), torch.from_numpy(val_mat),
            torch.from_numpy(dense))


class FineRank:
    """在线精排：组装 (用户 × 候选) 特征 → DeepFM 打分 → Top-N。

    SASRec 分数特征：对用户当前序列（含演示期点击，O1）一次前向得到
    全库分数向量，候选按连续索引查表。
    """

    def __init__(self, model, movie_ids, genre_mh, year_b, tt_embs, hot_df,
                 n_users, sas_model, content=None, mood=None):
        self.device = get_device()
        self.model = model.eval().to(self.device)
        self.movie_ids = movie_ids
        self.mid2idx = {int(m): i + 1 for i, m in enumerate(movie_ids)}
        self.mid2row = {int(m): i for i, m in enumerate(movie_ids)}
        self.st = _ItemStats(genre_mh, year_b, tt_embs, hot_df,
                             content=content, mood=mood)
        self.sas_model = sas_model
        self.n_users = n_users
        self.with_content = content is not None and model.n_dense == 8

    @classmethod
    def load(cls, variant="v1"):
        """variant: 'v1'（fine.pt，M3）| 'v2'（fine_v2.pt，M6 +内容特征）。"""
        feats = np.load(config.ART_DIR / "item_feats.npz", allow_pickle=True)
        genre_mh = feats["genre_multihot"]
        year_b = feats["year_bucket"]
        tt_embs = np.load(config.ART_DIR / "tt_item_emb.npy")
        hot_df = pd.read_parquet(config.ART_DIR / "hot.parquet")
        movie_ids = _movie_ids()
        n_users = int(pd.read_parquet(
            config.ART_DIR / "users.parquet").user_id.max())
        content = mood = None
        path = config.ART_DIR / "fine.pt"
        n_dense = 6
        if variant == "v2":
            path = config.ART_DIR / "fine_v2.pt"
            n_dense = 8
            content = np.load(config.ART_DIR / "content_emb.npy")
            from src.data.content import mood_onehot
            mood, _ = mood_onehot()
        model = DeepFM(n_users, len(movie_ids), genre_mh.shape[1],
                       n_dense=n_dense)
        model.load_state_dict(torch.load(path,
                                         map_location=str(get_device()),
                                         weights_only=True))
        from src.recall.sasrec import load_sas_model, MAX_LEN
        sas_model, _ = load_sas_model()
        cls.MAX_LEN = MAX_LEN
        return cls(model, movie_ids, genre_mh, year_b, tt_embs, hot_df,
                   n_users, sas_model, content=content, mood=mood)

    def _sas_scores(self, hist_mids):
        """当前序列一次前向 → 全库 logit/10 分数向量 [V]；无历史返回 None。"""
        s = [self.mid2idx[m] for m in hist_mids[-self.MAX_LEN:]
             if m in self.mid2idx]
        if not s:
            return None
        L = len(s)
        inp = np.zeros((1, self.MAX_LEN), dtype=np.int64)
        mask = np.zeros((1, self.MAX_LEN), dtype=bool)
        inp[0, :L] = s
        mask[0, :L] = True
        with torch.no_grad():
            h = self.sas_model(torch.tensor(inp, device=self.device),
                               torch.tensor(mask, device=self.device))
            last = h[0, L - 1]
            return ((last @ self.sas_model.item_table.weight.T)
                    / 10.0).cpu().numpy()

    def _user_hist_stats(self, hist_mids):
        h = [self.mid2idx[m] for m in hist_mids[-HIST_K:] if m in self.mid2idx]
        if not h:
            z_g = np.zeros(self.st.genre_mh.shape[1], dtype=np.float32)
            return z_g, np.zeros(self.st.tt.shape[1], dtype=np.float32)
        rows = np.array(h) - 1
        return self.st.genre_mh[rows].mean(0), self.st.tt[rows].mean(0)

    def rank(self, user_id, hist_mids, activity_count, candidates, topn,
             now_ts=None, sas_gamma=None):
        """candidates: [{movie_id, score, source}]；返回按精排分排序的 Top-N。

        精排分 = DeepFM logit + γ·SASRec logit（分数级融合）：
        序列模型分数既作为模型特征，也在分数层显式加回——保障排序不被
        ID embedding 的记忆噪声稀释（链路目标一致性，文档 03 趋势 2）。
        """
        gamma = config.RANK_SAS_GAMMA if sas_gamma is None else sas_gamma
        cands = [c for c in candidates if c["movie_id"] in self.mid2idx]
        if not cands:
            return []
        tgt = np.array([self.mid2idx[c["movie_id"]] for c in cands],
                       dtype=np.int64)
        hg, ht = self._user_hist_stats(hist_mids)
        ts = np.full(len(cands), now_ts or time.time(), dtype=np.int64)
        hb, wk = _hour_weekend(ts)
        n = len(cands)
        usr = np.full(n, user_id if 1 <= user_id <= self.n_users else 0,
                      dtype=np.int64)
        activity = float(np.log1p(activity_count)) / 10.0
        sas_vec = self._sas_scores(hist_mids)
        sas_sig = (sas_vec[tgt] if sas_vec is not None
                   else np.zeros(n, dtype=np.float32))
        idx_mat = np.zeros((n, DeepFM.N_FIELDS), dtype=np.int64)
        idx_mat[:, 0] = usr
        idx_mat[:, 1] = tgt
        idx_mat[:, 2] = self.st.year[tgt - 1]
        idx_mat[:, 3] = hb
        idx_mat[:, 4] = wk
        idx_mat[:, 5:] = self.st.genre_idx[tgt - 1]
        val_mat = np.ones((n, DeepFM.N_FIELDS), dtype=np.float32)
        val_mat[:, 5:] = self.st.genre_val[tgt - 1]
        dense = np.stack([
            np.full(n, activity, dtype=np.float32),
            (hg * self.st.genre_mh[tgt - 1]).sum(1),
            (ht * self.st.tt[tgt - 1]).sum(1),
            self.st.pop_log[tgt - 1],
            self.st.wr[tgt - 1] / 5.0,
            sas_sig.astype(np.float32)], 1)
        if self.with_content:
            # M6 v2：内容相似度 + 基调匹配（候选未看过，无排除，与训练口径一致）
            rows = [self.mid2row[m] for m in hist_mids
                    if m in self.mid2row]
            if rows:
                S = self.st.content[rows].sum(0)
                M = self.st.mood[rows].sum(0)
                csim = (self.st.content[tgt - 1] @ S) / max(
                    float(np.linalg.norm(S)), 1e-9)
                mmatch = (self.st.mood[tgt - 1] @ M) / len(rows)
            else:
                csim = np.zeros(n, dtype=np.float32)
                mmatch = np.zeros(n, dtype=np.float32)
            dense = np.hstack([dense, np.stack([csim, mmatch], 1
                                               ).astype(np.float32)])
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(idx_mat).to(self.device),
                torch.from_numpy(val_mat).to(self.device),
                torch.from_numpy(dense).to(self.device)).cpu().numpy()
        score = logits + gamma * sas_sig           # 分数级融合
        order = np.argsort(-score)[:topn]
        return [{**cands[i], "score": float(1.0 / (1.0 + np.exp(-score[i])))}
                for i in order]
