"""启动在线服务：python scripts/run.py [--dataset ml-25m]

数据集通过 --dataset 指定（默认环境变量 REC_DATASET 或 ml-1m）；
产物目录按数据集自动隔离（ml-1m → data/artifacts，其他 → data/artifacts-<name>）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dataset = None
port = 8000
if "--dataset" in sys.argv:
    dataset = sys.argv[sys.argv.index("--dataset") + 1]
    os.environ["REC_DATASET"] = dataset
if "--port" in sys.argv:
    port = int(sys.argv[sys.argv.index("--port") + 1])

import uvicorn

from src import config

if __name__ == "__main__":
    print(f"数据集 {config.DATASET}，产物目录 {config.ART_DIR}")
    uvicorn.run("src.serve.app:app", host="127.0.0.1", port=port)
