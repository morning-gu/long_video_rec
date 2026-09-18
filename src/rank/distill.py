"""蒸馏粗排（M7，文档 03 文档 2 §3.2：粗排目标一致性）。

M3 的教训：粗排用单一信号（双塔相似度）重排会压制强通道；M3–M6 的
"通道保持式压缩"是评测驱动的临时方案。蒸馏是工业正解：

- Teacher = 精排融合分（DeepFM logit + γ·SASRec logit）；
- Student = 轻量 MLP(双塔相似度, 口碑, 热度)——只见精排特征的廉价子集；
- 训练分布 = 真实漏斗候选（每用户通道保持压缩后的候选集），即
  "以漏斗曝光分布训练"（M3 报告备注指出的工业要求）。

CPU 训练分钟级；策略名 "distill"，与 keep/tt/none 并存可切换。
"""
import numpy as np
import torch
import torch.nn as nn

from src import config


class DistillStudent(nn.Module):
    """输入 [tt_sim, wr_pct, pop_log] → 精排分估计。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(-1)


def generate_training_set(rec, n_sample=1200, seed=42):
    """对抽样用户跑真实漏斗到粗排候选，用精排打分作 teacher。

    返回 (features [N,3], teacher [N])。rec: 离线构建的 Recommender。
    """
    rng = np.random.default_rng(seed)
    users = [u for u in rec.store._rated
             if rec.store.get(u).seeds][:5000]
    users = rng.choice(users, min(n_sample, len(users)), replace=False)
    feats, teach = [], []
    for u in users:
        u = int(u)
        state = rec.store.get(u)
        if not state.seeds:
            continue
        stages, _ = rec.funnel(u, "full")
        cands = stages["coarse"]
        if not cands:
            continue
        scored = rec.fine.rank(u, state.recent_positives, len(state.seen),
                               cands, len(cands))     # teacher：全量打分
        emb = rec.tt.user_embed(u, state.recent_positives)
        rows = np.array([rec.tt.mid2row[c["movie_id"]] for c in scored])
        sims = rec.tt.item_embs[rows] @ emb
        for c, s in zip(scored, sims):
            feats.append([s, rec.hot.wr.get(c["movie_id"], 3.0) / 5.0,
                          np.log1p(rec.hot.pos_count.get(c["movie_id"], 0)) / 8.0])
            teach.append(c["score"])
    return (np.array(feats, dtype=np.float32),
            np.array(teach, dtype=np.float32))


def train_distill(rec, epochs=8, seed=42):
    """生成训练集并训练 student，返回 (student, 训练集规模)。"""
    torch.manual_seed(seed)
    x, y = generate_training_set(rec, seed=seed)
    print(f"  [distill] 训练集 {len(x)} 行（真实漏斗候选分布）")
    student = DistillStudent()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    n = len(x)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, 4096):
            idx = perm[s:s + 4096]
            loss = nn.functional.mse_loss(student(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  [distill] epoch {epoch + 1}/{epochs}  mse={tot:.5f}")
    student.eval()
    return student, n
