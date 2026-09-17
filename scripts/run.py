"""启动在线服务：python scripts/run.py → http://localhost:8000"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.serve.app:app", host="127.0.0.1", port=8000)
