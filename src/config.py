"""全局配置：路径与漏斗参数（对应设计文档 §3–§5）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "ml-1m"
ART_DIR = DATA_DIR / "artifacts"
WEB_DIR = ROOT / "web"


def _load_env() -> None:
    """轻量 .env 加载（KEY=VALUE，不含引号处理），不覆盖已有环境变量。"""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_ZIP = DATA_DIR / "ml-1m.zip"
DB_PATH = ART_DIR / "demo.db"

# ---- LLM 层（设计文档 §4.5，O2 预算无上限）----
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.7-plus")
LLM_TIMEOUT = 25            # 单次调用超时（秒），超时降级（R2）

# ---- TMDB / 海报 ----
# 本网络 api.themoviedb.org 被阻断（DNS 污染 + SNI 阻断），海报改用
# IMDb suggestion API（大陆可达）；TMDB key 保留供网络允许时使用
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")

# ---- 标签构造（设计文档 §3）----
POS_RATING = 4              # 评分 >= 4 视为隐式正反馈

# ---- 召回层（设计文档 §4.1）----
SEED_TOPK = 3               # ItemCF 种子：最近正反馈电影数
SEED_WEIGHTS = [1.0, 0.7, 0.5]   # 种子按新近度加权
ITEMCF_QUOTA = 200          # ItemCF 通道配额
TWOTOWER_QUOTA = 300        # 双塔通道配额（M2）
SASREC_QUOTA = 150          # 序列通道配额（M2）
HOT_QUOTA = 50              # 热门通道配额
LIGHTGCN_QUOTA = 150        # LightGCN 通道配额（M6）
SEMANTIC_QUOTA = 150        # 内容语义通道配额（M6）
TIGER_QUOTA = 40            # TIGER 生成式召回配额（M7，受 beam 宽度限制）
COARSE_TOPN = 200           # 粗排输出（M3）
FINE_TOPN = 50              # 精排输出（M3）
ROW_K = 20                  # 主推荐行数量
SUB_ROW_K = 10              # 次级行数量

# ---- 精排层（设计文档 §4.3，M3）----
# 精排分数 = DeepFM logit + γ·SASRec logit（分数级融合，链路目标一致性）
# γ=40 为 500 用户验证集网格 {0,1,2,3,5,8,12,16,24,40} 的最优点；
# 纯序列排序（γ→∞）为 0.258，特征交叉模型在本数据规模无净增量——
# Ferrari Dacrema 批评的本地复现，如实标注（docs/评测报告.md 备注）
RANK_SAS_GAMMA = 40.0

# ---- 重排层（设计文档 §4.4）----
GENRE_CONSEC_MAX = 2        # 同类型最多连续部数
GENRE_CAP = 6               # 最终列表单类型上限
HOT_SET_SIZE = 500          # 热门集合规模（冷门保量分界）
COLD_RATIO = 0.2            # 冷门保量最低比例
MMR_LAMBDA = 0.4            # MMR 多样性权重 λ
RERANKERS = ("mmr", "dpp")  # 可用重排器（M6 起可切换，旧实现保留）

# ---- 热门平滑（设计文档 §4.1）----
BAYES_C = 200               # 贝叶斯平均先验票数
