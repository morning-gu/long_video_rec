"""离线评测报告（设计文档 §6）：漏斗逐级准确性 + beyond-accuracy + 热门基线。

协议：
- 时间切分 leave-last-out（每用户最后 1 个正反馈为测试目标，构建期已定）；
- 评测状态 = 仅训练期数据（内存态隔离，不含演示事件）；
- 漏斗逐级：纯热门 / 各召回单通道 / 四路融合 / +粗排 / +精排 / +规则 / +MMR；
- 单相关物品下 Recall@K 与 HR@K 等价，NDCG@10 = 1/log2(rank+1)；
- beyond-accuracy：列表内多样性（ILAD）、目录覆盖率、长尾占比、流行度分位。

产出：docs/评测报告.md（人读） + data/artifacts/eval_report.json（展示页）。
"""
import math

import numpy as np
import pandas as pd

from src import config


def _hit(items, target, k):
    ids = [c["movie_id"] if isinstance(c, dict) else c[0]
           for c in items[:k]]
    return target in ids


def _rank(items, target):
    for i, c in enumerate(items):
        mid = c["movie_id"] if isinstance(c, dict) else c[0]
        if mid == target:
            return i + 1
    return 10 ** 9


def _ilad(embs):
    """列表内平均成对距离 1-cos（embs 已归一）。"""
    n = len(embs)
    if n < 2:
        return 0.0
    sim = embs @ embs.T
    return float((1.0 - sim).sum() / (n * (n - 1)))


def run(n_users: int = 1000, seed: int = 42):
    from src.serve.app import Recommender
    test_pos = pd.read_parquet(config.ART_DIR / "test_pos.parquet")
    ratings = pd.read_parquet(config.ART_DIR / "ratings.parquet")
    movies = pd.read_parquet(config.ART_DIR / "movies.parquet")

    test_map = dict(zip(test_pos.user_id.astype(int),
                        test_pos.movie_id.astype(int)))
    # 评测状态 = 剔除测试交互后的训练期评分：leave-last-out 中测试目标是
    # "未来"行为，不得进入已看集合（否则被漏斗的已看过滤排除，命中恒为 0）
    tk = test_pos[["user_id", "movie_id"]]
    rt = ratings.merge(tk, on=["user_id", "movie_id"],
                       how="left", indicator=True)
    train_ratings = rt[rt._merge == "left_only"].drop(columns="_merge")
    rec = Recommender(state_db=":memory:", ratings=train_ratings)

    from src.data.features import positive_sequences
    train_pos = pd.read_parquet(config.ART_DIR / "train_pos.parquet")
    seqs = positive_sequences(train_pos)
    pool = [u for u in test_map if u in seqs and seqs[u]]
    rng = np.random.default_rng(seed)
    users = rng.choice(pool, min(n_users, len(pool)), replace=False)

    # 基线：全局热门 Top-10（对所有用户相同）
    hot_top10 = [m for m, _ in rec.hot.top(set(), 10)]

    # 物品侧统计（beyond-accuracy）
    tt = rec.tt.item_embs
    row_of = rec.tt.mid2row
    pop_order = rec.hot.pop_order
    pop_pct = np.zeros(len(movies), dtype=np.float32)      # 1=最热门
    for r, mid in enumerate(pop_order):
        pop_pct[row_of[mid]] = 1.0 - r / len(pop_order)
    hot_set = rec.hot.hot_set

    stage_hits = {}          # stage -> 命中数
    ndcg10 = 0.0
    ilad_sum, longtail_slots, pop_sum, total_slots = 0.0, 0, 0.0, 0
    covered = set()
    for u in users:
        t = test_map[u]
        stages, _state = rec.funnel(int(u), "full")
        for name, ch in stages["channels"].items():
            k = f"通道:{name}"
            stage_hits[k] = stage_hits.get(k, 0) + _hit(ch, t, 10)
        for st in ("recall", "coarse", "fine", "rules", "final"):
            stage_hits[st] = stage_hits.get(st, 0) + _hit(stages[st], t, 10)
        stage_hits["final@20"] = stage_hits.get("final@20", 0) + \
            _hit(stages["final"], t, 20)
        stage_hits["hot"] = stage_hits.get("hot", 0) + (t in hot_top10)
        r = _rank(stages["final"], t)
        if r <= 10:
            ndcg10 += 1.0 / math.log2(r + 1)
        # beyond-accuracy（final Top-10）
        final10 = stages["final"][:10]
        embs = np.stack([tt[row_of[c["movie_id"]]] for c in final10])
        ilad_sum += _ilad(embs)
        for c in final10:
            covered.add(c["movie_id"])
            total_slots += 1
            if c["movie_id"] not in hot_set:
                longtail_slots += 1
            pop_sum += float(pop_pct[row_of[c["movie_id"]]])

    n = len(users)
    hot10_embs = np.stack([tt[row_of[m]] for m in hot_top10])

    # ---- 结构化数据（展示页 / JSON）----
    ch_label = {"itemcf": "ItemCF 单通道", "twotower": "双塔 单通道",
                "sasrec": "SASRec 单通道", "hot": "热门 通道"}
    ch_note = {"itemcf": "共现相似", "twotower": "向量召回",
               "sasrec": "序列建模", "hot": "流行度"}
    funnel = [{"stage": "纯热门（非个性化基线）", "hr10": stage_hits["hot"] / n,
               "note": "必须打赢的简单强基线（Ferrari Dacrema）", "channel": False}]
    for name in ("itemcf", "twotower", "sasrec", "hot"):
        k = f"通道:{name}"
        if k in stage_hits:
            funnel.append({"stage": ch_label[name],
                           "hr10": stage_hits[k] / n,
                           "note": ch_note.get(name, ""), "channel": True})
    funnel += [
        {"stage": "四路融合（召回后）", "hr10": stage_hits["recall"] / n,
         "note": "配额交错去重", "channel": False},
        {"stage": "+ 粗排", "hr10": stage_hits["coarse"] / n,
         "note": "通道保持式压缩（目标一致性修订）", "channel": False},
        {"stage": "+ 精排", "hr10": stage_hits["fine"] / n,
         "note": "DeepFM + 序列信号分数级融合", "channel": False},
        {"stage": "+ 规则重排", "hr10": stage_hits["rules"] / n,
         "note": "打散/冷门保量（多样性约束的代价）", "channel": False},
        {"stage": "+ MMR（最终 Top-10）", "hr10": stage_hits["final"] / n,
         "note": "完整链路", "channel": False},
        {"stage": "最终 Top-20", "hr10": stage_hits["final@20"] / n,
         "note": "", "channel": False},
    ]
    beyond = [
        {"metric": "列表内多样性 ILAD", "desc": "越高越好",
         "full": ilad_sum / n, "hot": _ilad(hot10_embs)},
        {"metric": "目录覆盖率", "desc": "全量推荐物品 / 片库",
         "full": len(covered) / len(movies),
         "hot": len(set(hot_top10)) / len(movies)},
        {"metric": "长尾占比", "desc": "非 Top-500 热门槽位比例",
         "full": longtail_slots / total_slots, "hot": 0.0},
        {"metric": "平均流行度分位", "desc": "1=最热，越低越普惠",
         "full": pop_sum / total_slots,
         "hot": float(pop_pct[[row_of[m] for m in hot_top10]].mean())},
    ]
    data = {
        "n_users": int(n),
        "protocol": "时间切分 leave-last-out（每用户最后 1 个正反馈），"
                    "推荐时排除全部已看；单相关物品下 HR@K 等价于 Recall@K",
        "funnel": [{"stage": f["stage"], "hr10": round(f["hr10"], 4),
                    "note": f["note"], "channel": f["channel"]} for f in funnel],
        "ndcg10": round(ndcg10 / n, 4),
        "final_vs_hot": round(stage_hits["final"] / n / max(stage_hits["hot"] / n, 1e-9), 1),
        "beyond": [{"metric": b["metric"], "desc": b["desc"],
                    "full": round(b["full"], 4), "hot": round(b["hot"], 4)}
                   for b in beyond],
        "notes": [
            "精排为单目标（正反馈概率）；多行为标签（完播/弃剧/时长）为公开数据已知"
            "空白（文档 02_06 §10.2），多目标结构接口已预留（决策 D3）",
            "精排负采样为流行度加权（unigram^0.75）；曝光偏差缓解列为扩展（文档 01 §5.2）",
            "精排分数 = DeepFM logit + γ·SASRec logit，γ=40 为验证集网格选择；"
            "纯序列排序 HR@10 为 0.258——特征交叉模型在本数据规模相对序列信号无净增量，"
            "系 Ferrari Dacrema 批评（文档 01 §5.2）的本地复现；工业系统中精排需以"
            "漏斗曝光分布训练方能超越（文档 03 趋势 2）",
            "评测与线上共用同一 funnel() 代码路径，无训练-服务偏移",
        ],
    }

    # ---- Markdown ----
    lines = [
        "# 离线评测报告（M3–M5）",
        "",
        f"- 协议：{data['protocol']}；抽样 {n} 个测试用户",
        "- 指标：漏斗各阶段输出按该阶段排序取 Top-10 判命中",
        "",
        "## 准确性：漏斗逐级（HR@10）",
        "",
        "| 阶段 | HR@10 | 说明 |",
        "|---|---|---|",
    ]
    for f in funnel:
        indent = "  " if f["channel"] else ""
        lines.append(f"| {indent}{f['stage']} | {f['hr10']:.4f} | {f['note']} |")
    lines += [
        "",
        f"**NDCG@10（最终链路）= {data['ndcg10']:.4f}**；"
        f"最终链路 = 纯热门基线的 **{data['final_vs_hot']}×**",
        "",
        "## Beyond-accuracy（最终链路 Top-10 vs 纯热门）",
        "",
        "| 指标 | 完整链路 | 纯热门 |",
        "|---|---|---|",
    ]
    for b in beyond:
        lines.append(f"| {b['metric']}（{b['desc']}） | {b['full']:.4f} | "
                     f"{b['hot']:.4f} |")
    lines += ["", "## 备注", ""] + [f"- {x}" for x in data["notes"]]

    (config.ROOT / "docs" / "评测报告.md").write_text(
        "\n".join(lines), encoding="utf-8")
    (config.ART_DIR / "eval_report.json").write_text(
        json_dump(data), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已写入 docs/评测报告.md + data/artifacts/eval_report.json")


def json_dump(data) -> str:
    import json
    return json.dumps(data, ensure_ascii=False, indent=2)
