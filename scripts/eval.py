"""生成离线评测报告：python scripts/eval.py → docs/评测报告.md"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):     # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.report import run

if __name__ == "__main__":
    run(n_users=int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
