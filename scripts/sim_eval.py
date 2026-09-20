"""LLM 用户模拟器在线评估（M8，文档 01 §5.3：simulation-based eval）。

动机：离线 HR/NDCG 与在线效果存在错位（曝光偏差、位置偏差）；A/B 不可得时，
用 LLM 扮演用户对推荐列表做模拟反馈，比较不同策略——明确标注为 **simulation**，
不能替代真实在线实验（LLM 用户 ≠ 真实用户，对提示词敏感）。

策略对比：完整链路（full）vs 纯热门（popular）vs 随机（random，下界参照）。
指标：模拟 CTR（点击率）、模拟 NDCG@10、零点击率。
缓存于 llm_cache；输出 docs/模拟评估报告.md。

用法：python scripts/sim_eval.py [n_users=150] [--dataset ml-25m]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_users", nargs="?", type=int, default=150)
    ap.add_argument("--dataset", default=None,
                    help="数据集名（须先完成对应 build），默认 REC_DATASET/ml-1m")
    return ap.parse_args()


_ARGS = _parse_args()
if _ARGS.dataset:
    import os
    os.environ["REC_DATASET"] = _ARGS.dataset

import numpy as np
import pandas as pd

from src import config

print(f"数据集 {config.DATASET}（产物目录 {config.ART_DIR}）")
from src.llm.service import _extract_json


def build_lists(rec, u, rated):
    """返回 {strategy: [movie_id×10]}（评测隔离：测试目标不进已看）。"""
    state = rec.store.get(u)
    exclude = state.seen | state.suppress
    out = {}
    stages, _ = rec.funnel(u, "full")
    out["full"] = [c["movie_id"] for c in stages["final"][:10]]
    stages_p, _ = rec.funnel(u, "popular")
    out["popular"] = [c["movie_id"] for c in stages_p["final"][:10]]
    pool = [m for m in rec.hot.pop_order if m not in exclude]
    rng = np.random.default_rng(u)
    out["random"] = list(rng.choice(pool, 10, replace=False))
    return out


def simulate(llm, u, history_titles, movies_info, mids):
    """LLM 扮演用户：返回点击的 movie_id 列表。"""
    key = llm._key("sim", u, mids)
    cached = llm._cached(key)
    if cached is not None:
        return cached.get("clicks", [])
    hist = "\n".join(f"{i + 1}. {t}"
                     for i, t in enumerate(history_titles[:10]))
    cand = "\n".join(f"{i + 1}. {movies_info[m]['title']}"
                     f"（{'/'.join(movies_info[m]['genres'][:3])}）"
                     for i, m in enumerate(mids))
    prompt = (
        "你在扮演一位影视平台用户。根据你的真实观影历史，判断你会点击观看"
        "推荐列表中的哪几部电影。\n\n"
        f"【你的观影历史】（新→旧）\n{hist}\n\n"
        f"【推荐列表】\n{cand}\n\n"
        "要求：只推荐你真正会点开看的（宁缺毋滥，可以都不选）；"
        '只输出 JSON：{"clicks": [列表序号，1~10]}，不选则 []。')
    try:
        obj = _extract_json(llm._chat(prompt))
        clicks = [int(x) for x in (obj.get("clicks") or [])
                  if str(x).isdigit() and 1 <= int(x) <= 10]
        clicks = sorted(set(clicks))
        llm._put(key, {"clicks": clicks})
        return clicks
    except Exception as e:
        llm._log_fail("sim", f"{type(e).__name__}: {e}")
        return None


def run(n_users=150, seed=42):
    from src.serve.app import Recommender
    test_pos = pd.read_parquet(config.ART_DIR / "test_pos.parquet")
    ratings = pd.read_parquet(config.ART_DIR / "ratings.parquet")
    tk = test_pos[["user_id", "movie_id"]]
    rt = ratings.merge(tk, on=["user_id", "movie_id"], how="left",
                       indicator=True)
    rec = Recommender(state_db=":memory:",
                      ratings=rt[rt._merge == "left_only"].drop(columns="_merge"))
    if not rec.llm.available:
        print("未配置 LLM_API_KEY，无法运行模拟评估")
        return
    from src.data.features import positive_sequences
    seqs = positive_sequences(
        pd.read_parquet(config.ART_DIR / "train_pos.parquet"))
    pool = [u for u in seqs if seqs[u]][:2000]
    users = list(np.random.default_rng(seed).choice(
        pool, min(n_users, len(pool)), replace=False))

    strat_stats = {s: {"ctr": [], "ndcg": [], "zero": []}
                   for s in ("full", "popular", "random")}
    t0 = time.time()
    for i, u in enumerate(users):
        u = int(u)
        lists = build_lists(rec, u, None)
        hist_titles = [rec.movie_info[m]["title"]
                       for m in reversed(seqs[u][-10:]) if m in rec.movie_info]
        for strat, mids in lists.items():
            clicks = simulate(rec.llm, u, hist_titles, rec.movie_info, mids)
            if clicks is None:
                continue
            clicked_mids = [mids[c - 1] for c in clicks]
            strat_stats[strat]["ctr"].append(len(clicks) / 10)
            strat_stats[strat]["zero"].append(1 if not clicks else 0)
            # NDCG@10：点击位置 → 1/log2(rank+1)，按点击数归一
            dcg = sum(1 / np.log2(c + 1) for c in clicks)
            ideal = sum(1 / np.log2(r + 1)
                        for r in range(1, len(clicks) + 1))
            strat_stats[strat]["ndcg"].append(
                dcg / ideal if ideal > 0 else 0.0)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(users)}  ({time.time() - t0:.0f}s)")

    lines = [
        "# LLM 用户模拟器评估（M8 · simulation）",
        "",
        f"- 协议：{len(users)} 个测试用户，每策略 Top-10，LLM 扮演用户多选点击；"
        "缓存复用",
        "- **性质声明**：simulation-based eval（文档 01 §5.3）——LLM 用户 ≠ "
        "真实用户、对提示词敏感，用于策略间相对比较，不能替代 A/B 实验",
        "",
        "| 策略 | 模拟 CTR | 模拟 NDCG@10 | 零点击率 |",
        "|---|---|---|---|",
    ]
    data = {"n_users": len(users), "strategies": {}}
    for strat in ("full", "popular", "random"):
        s = strat_stats[strat]
        if not s["ctr"]:
            continue
        row = {"sim_ctr": round(float(np.mean(s["ctr"])), 4),
               "sim_ndcg10": round(float(np.mean(s["ndcg"])), 4),
               "zero_rate": round(float(np.mean(s["zero"])), 4)}
        data["strategies"][strat] = row
        label = {"full": "完整链路", "popular": "纯热门",
                 "random": "随机（下界）"}[strat]
        lines.append(f"| {label} | {row['sim_ctr']:.4f} | "
                     f"{row['sim_ndcg10']:.4f} | {row['zero_rate']:.4f} |")
    suffix = "" if config.DATASET == "ml-1m" else f"-{config.DATASET}"
    out = config.ROOT / "docs" / f"模拟评估报告{suffix}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (config.ART_DIR / "sim_report.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    run(n_users=_ARGS.n_users)
