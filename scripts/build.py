"""一键离线构建：数据管线 → ItemCF → 热门 → 双塔 → SASRec → DeepFM 精排。

用法：
  python scripts/build.py          # 全量构建（首次约 30 分钟，2 核 CPU）
  python scripts/build.py fine     # 仅重建精排（复用已有 M2 产物）

评测：
  python scripts/eval.py           # 漏斗逐级 + beyond-accuracy 报告 → docs/评测报告.md
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):     # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch

from src import config
from src.data.features import build_item_features, positive_sequences
from src.data.pipeline import run as run_pipeline
from src.rank.fine import FineRank, train_fine
from src.recall.hot import HotRecall, build_hot
from src.recall.itemcf import ItemCFRecall, build_sim
from src.recall.sasrec import SASRecRecall, train_sasrec
from src.recall.twotower import TwoTowerRecall, train_twotower


def sanity_eval(recall_fns: dict, hot: HotRecall, ratings: pd.DataFrame,
                train_pos: pd.DataFrame, test_pos: pd.DataFrame) -> None:
    """HR@10 快速体检：各召回通道 vs 纯热门基线（漏斗逐级报告见 scripts/eval.py）。"""
    rated = {int(u): set(map(int, g))
             for u, g in ratings.groupby("user_id")["movie_id"]}
    seqs = positive_sequences(train_pos)
    test_map = dict(zip(test_pos.user_id.astype(int), test_pos.movie_id.astype(int)))
    hot_top10 = {m for m, _ in hot.top(set(), 10)}

    hits = {name: 0 for name in recall_fns}
    hits["hot"] = 0
    n = 0
    for u, t in test_map.items():
        if u not in seqs:
            continue
        seen = rated.get(u, set()) - {t}
        for name, fn in recall_fns.items():
            hits[name] += t in {m for m, _ in fn(u, seqs[u], seen)}
        hits["hot"] += t in hot_top10
        n += 1
    print("\n[sanity] HR@10（时间切分，每用户最后 1 个正反馈）")
    for name, h in hits.items():
        print(f"  {name:<10} {h / n:.4f}")
    print(f"  (测试用户 n={n})")


def fine_auc(fine: FineRank, ratings, train_pos, test_pos, movie_ids,
             n_sample=2000, seed=42):
    """精排 AUC：每用户 1 正样本 + 99 均匀负样本的排序。"""
    rng = np.random.default_rng(seed)
    seqs = positive_sequences(train_pos)
    rated = {int(u): set(map(int, g))
             for u, g in ratings.groupby("user_id")["movie_id"]}
    test_map = dict(zip(test_pos.user_id.astype(int), test_pos.movie_id.astype(int)))
    users = [u for u in test_map if u in seqs and seqs[u]]
    users = rng.choice(users, min(n_sample, len(users)), replace=False)
    auc_sum = hr10 = 0
    for u in users:
        u, t = int(u), test_map[int(u)]
        seen = rated.get(u, set())
        cand = [t]
        while len(cand) < 100:
            m = int(rng.choice(movie_ids))
            if m not in seen and m not in cand:
                cand.append(m)
        cands = [{"movie_id": m, "score": 0.0, "source": ""} for m in cand]
        order = [c["movie_id"] for c in
                 fine.rank(u, seqs[u], len(seen), cands, 100)]
        r = order.index(t)                       # 0-based，共 100 个候选
        auc_sum += (99 - r) / 99                 # 排在正样本之后的负样本比例
        hr10 += r < 10
    print(f"[fine] AUC={auc_sum / len(users):.4f}  "
          f"HR@10(1正+99负)={hr10 / len(users):.4f}  (n={len(users)})")


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None    # 可选："fine"

    if only != "fine":
        run_pipeline()
        ratings = pd.read_parquet(config.ART_DIR / "ratings.parquet")
        movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
        train_pos = pd.read_parquet(config.ART_DIR / "train_pos.parquet")
        test_pos = pd.read_parquet(config.ART_DIR / "test_pos.parquet")
        movie_ids = movies.movie_id.values
        n_users = int(pd.read_parquet(
            config.ART_DIR / "users.parquet").user_id.max())

        # ---- M1：ItemCF + 热门 ----
        print("building ItemCF similarity matrix ...")
        sim, _pop = build_sim(train_pos, movie_ids)
        np.save(config.ART_DIR / "itemcf_sim.npy", sim)
        print(f"saved itemcf_sim.npy  shape={sim.shape}")

        print("building hot lists ...")
        hot_df = build_hot(ratings, test_pos)
        hot_df.to_parquet(config.ART_DIR / "hot.parquet")
        print(f"saved hot.parquet  items={len(hot_df)}")

        # ---- M2：物品特征 + 双塔 + SASRec ----
        print("building item features (genre / year) ...")
        genre_vocab, genre_mh, year_b = build_item_features(movies)
        np.savez(config.ART_DIR / "item_feats.npz",
                 genre_multihot=genre_mh, year_bucket=year_b,
                 genre_vocab=np.array(genre_vocab))
        seqs = positive_sequences(train_pos)
        print(f"item features: genres={len(genre_vocab)}  users={n_users}")

        print("training two-tower (sampled softmax) ...")
        tt_model, tt_item_embs = train_twotower(seqs, genre_mh, year_b, n_users)
        torch.save(tt_model.state_dict(), config.ART_DIR / "twotower.pt")
        np.save(config.ART_DIR / "tt_item_emb.npy", tt_item_embs)
        print(f"saved twotower.pt + tt_item_emb.npy  dim={tt_item_embs.shape}")

        print("training SASRec-lite ...")
        sas_model = train_sasrec(seqs)
        torch.save(sas_model.state_dict(), config.ART_DIR / "sasrec.pt")
        print("saved sasrec.pt")

        # ---- M6：LightGCN 图召回（CPU 约 15 分钟）----
        print("training LightGCN ...")
        from src.recall.lightgcn import train_lightgcn
        lg_u, lg_i = train_lightgcn(train_pos, n_users, len(movies))
        np.save(config.ART_DIR / "lg_user_emb.npy", lg_u)
        np.save(config.ART_DIR / "lg_item_emb.npy", lg_i)
        print("saved lg_user_emb.npy + lg_item_emb.npy")

        # ---- M6：内容画像 → 内容向量（依赖 scripts/enrich_profiles.py 先跑）----
        if (config.ART_DIR / "profiles.parquet").exists():
            from src.data.content import build_content_vectors
            vec = build_content_vectors()
            np.save(config.ART_DIR / "content_emb.npy", vec)
            print(f"saved content_emb.npy  shape={vec.shape}")
        else:
            print("profiles.parquet 不存在，跳过内容向量"
                  "（先运行 python scripts/enrich_profiles.py）")
    else:
        # 仅重建精排：加载 M1/M2 产物
        ratings = pd.read_parquet(config.ART_DIR / "ratings.parquet")
        movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
        train_pos = pd.read_parquet(config.ART_DIR / "train_pos.parquet")
        test_pos = pd.read_parquet(config.ART_DIR / "test_pos.parquet")
        movie_ids = movies.movie_id.values
        n_users = int(pd.read_parquet(
            config.ART_DIR / "users.parquet").user_id.max())
        feats = np.load(config.ART_DIR / "item_feats.npz", allow_pickle=True)
        genre_mh, year_b = feats["genre_multihot"], feats["year_bucket"]
        hot_df = pd.read_parquet(config.ART_DIR / "hot.parquet")
        tt_item_embs = np.load(config.ART_DIR / "tt_item_emb.npy")

    # ---- M3/M6：精排 v1（DeepFM）+ v2（+内容特征，需 content_emb.npy）----
    print("training fine rank v1 (DeepFM) ...")
    fine_model = train_fine(train_pos, ratings, genre_mh, year_b, tt_item_embs,
                            hot_df, n_users)
    torch.save(fine_model.state_dict(), config.ART_DIR / "fine.pt")
    print("saved fine.pt")
    fine_auc(FineRank.load("v1"), ratings, train_pos, test_pos, movie_ids)
    if (config.ART_DIR / "content_emb.npy").exists():
        print("training fine rank v2 (DeepFM + 内容特征) ...")
        fine_v2 = train_fine(train_pos, ratings, genre_mh, year_b,
                             tt_item_embs, hot_df, n_users, with_content=True)
        torch.save(fine_v2.state_dict(), config.ART_DIR / "fine_v2.pt")
        print("saved fine_v2.pt")

    if only != "fine":
        # ---- sanity 评测（走与线上相同的加载路径）----
        itemcf = ItemCFRecall(sim, movie_ids)
        tt = TwoTowerRecall.load()
        sas = SASRecRecall.load()
        fns = {
            "itemcf": lambda u, seq, seen: itemcf.recall(
                list(reversed(seq[-config.SEED_TOPK:])), seen, 10),
            "twotower": lambda u, seq, seen: tt.recall(u, seq, seen, 10),
            "sasrec": lambda u, seq, seen: sas.recall(seq, seen, 10),
        }
        if (config.ART_DIR / "lg_user_emb.npy").exists():
            from src.recall.lightgcn import LightGCNRecall
            lg = LightGCNRecall.load()
            fns["lightgcn"] = lambda u, seq, seen: lg.recall(u, seen, 10)
        if (config.ART_DIR / "content_emb.npy").exists():
            from src.recall.semantic import SemanticRecall
            sem = SemanticRecall.load()
            fns["semantic"] = lambda u, seq, seen: sem.recall(seq, seen, 10)
        sanity_eval(fns, HotRecall(hot_df), ratings, train_pos, test_pos)
    print("\nbuild done. 启动服务：python scripts/run.py → http://localhost:8000")
    print("评测报告：python scripts/eval.py → docs/评测报告.md")


if __name__ == "__main__":
    main()
