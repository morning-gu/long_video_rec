"""生成离线评测报告：python scripts/eval.py [n_users] [--dataset ml-25m]

报告输出：docs/评测报告[-<dataset>].md + <ART_DIR>/eval_report.json
（展示页 /eval 读 eval_report.json）。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):     # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_users", nargs="?", type=int, default=1000)
    ap.add_argument("--dataset", default=None,
                    help="数据集名（须先完成对应 build），默认 REC_DATASET/ml-1m")
    return ap.parse_args()


_ARGS = _parse_args()
if _ARGS.dataset:
    import os
    os.environ["REC_DATASET"] = _ARGS.dataset

from src import config

from src.eval.report import run

if __name__ == "__main__":
    print(f"数据集 {config.DATASET}（产物目录 {config.ART_DIR}）")
    run(n_users=_ARGS.n_users)
