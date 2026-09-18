"""全局配置：路径与漏斗参数（对应设计文档 §3–§5）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
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

# ---- 数据集注册表（M8：多数据集抽象，ml-1m 参数与历史完全一致）----
# 通过环境变量 REC_DATASET 选择（scripts 的 --dataset 参数会设置它）。
# "ml-25m-test" 为本地小样 fixture（代码路径验证用，正式训练用 ml-25m）。
DATASET = os.environ.get("REC_DATASET", "ml-1m")

DATASETS = {
    "ml-1m": dict(
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        raw_dir="ml-1m", fmt="dat",
        tt_dim=64, sas_dim=64, sas_layers=2,
        itemcf_topk=0,                 # 0 = 稠密精确相似矩阵
        faiss="flat",                  # n_items 小用精确检索
        rq_k=256, rq_levels=3,
        tt_epochs=30, sas_epochs=80, lg_epochs=40,
        fine_epochs=6, tiger_epochs=15, rq_epochs=400,
    ),
    "ml-25m": dict(
        url="https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        raw_dir="ml-25m", fmt="csv",
        tt_dim=128, sas_dim=128, sas_layers=3,
        itemcf_topk=200,               # >0 = 分块 top-K 稀疏相似（59k² 稠密不可行）
        faiss="hnsw",                  # 近似检索（59k+ 物品）
        rq_k=512, rq_levels=3,
        tt_epochs=25, sas_epochs=30, lg_epochs=60,
        fine_epochs=4, tiger_epochs=10, rq_epochs=600,
    ),
    "ml-25m-test": dict(               # 本地 fixture：同 ml-25m 代码路径
        url=None, raw_dir="ml-25m-test", fmt="csv",
        tt_dim=128, sas_dim=128, sas_layers=3,
        itemcf_topk=200, faiss="hnsw",
        rq_k=512, rq_levels=3,
        tt_epochs=2, sas_epochs=2, lg_epochs=2,
        fine_epochs=2, tiger_epochs=2, rq_epochs=20,
    ),
}
if DATASET not in DATASETS:
    raise RuntimeError(f"未知数据集 {DATASET}；可选：{list(DATASETS)}")
P = DATASETS[DATASET]                  # 当前数据集参数（P = params）

RAW_DIR = DATA_DIR / P["raw_dir"]
# 产物目录按数据集隔离：ml-1m 沿用历史路径（现有产物不迁移），
# 其他数据集用 artifacts-<name>（避免 fixture/规模化训练覆盖 ml-1m 产物）
if DATASET == "ml-1m":
    ART_DIR = DATA_DIR / "artifacts"
else:
    ART_DIR = DATA_DIR / f"artifacts-{DATASET}"
ML1M_URL = P["url"]                    # 兼容旧引用（pipeline 已改用 P["url"]）
ML1M_ZIP = DATA_DIR / f"{P['raw_dir']}.zip"
DB_PATH = ART_DIR / "demo.db"

# ---- LLM 层（设计文档 §4.5，O2 预算无上限）----
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.7-plus")
LLM_TIMEOUT = 25            # 单次调用超时（秒），超时降级（R2）
# M8：LLM rerank 后端——"api"（零样本提示）| "local"（LoRA 微调，
# scripts/train_lora.py 产出 data/lora-adapter/）。加载失败自动回退 api。
LLM_RANKER_BACKEND = os.environ.get("LLM_RANKER_BACKEND", "api")
LORA_BASE_MODEL = os.environ.get(
    "LORA_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

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
