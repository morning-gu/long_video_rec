# M8 GPU 训练指南

> 代码已全部开发完成并在本机验证（数据集抽象 / 稀疏 ItemCF / HNSW / LoRA 数据管线
> 均通过 ml-25m-test fixture 端到端构建 + 服务验证）。本文档是在 GPU 机器上的
> 操作步骤与验收标准。

## 0. 机器要求

| 项 | 最低 | 建议 |
|---|---|---|
| 显存 | 8GB（Qwen2.5-1.5B LoRA + 各模型训练） | 24GB（可换 7B、更大 batch） |
| 内存 | 32GB（25M 评分 pandas + 稀疏 ItemCF 分块） | 64GB |
| 磁盘 | 20GB（数据 1.5GB + 模型产物 + HF 缓存） | 50GB |
| 网络 | 能访问 files.grouplens.org（下载数据）与 HuggingFace（或镜像） | |

## 1. 环境准备

```bash
git clone <repo> && cd long_video_rec
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) 先装 GPU 版 torch（按 CUDA 版本选，示例为 cu121）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) 基础依赖 + M8 依赖
pip install -r requirements.txt -r requirements-m8.txt

# 3) 国内网络访问 HuggingFace 用镜像（Qwen 模型下载）
export HF_ENDPOINT=https://hf-mirror.com            # Windows: set HF_ENDPOINT=...

# 4) 配置密钥（可选：LLM 增强/画像/模拟评估；不配则这些功能自动降级）
cp .env.example .env   # 填入 LLM_API_KEY 等
```

## 2. ML-25M 全量构建（预计 2–5 小时，视显卡）

```bash
python scripts/build.py --dataset ml-25m
```

自动完成：下载 ml-25m.zip（250MB）→ 解析 → 全链路训练。各阶段预计耗时
（以 RTX 4090 / A10 级别估算，CPU-only 会慢 10–20 倍）：

| 阶段 | 说明 | 预计 |
|---|---|---|
| 数据管线 | 25M 评分 → parquet | ~3 分钟 |
| ItemCF 稀疏 | 59k 物品分块 top-200 | ~30–60 分钟 |
| 双塔 | dim 128（config 可调） | ~20 分钟 |
| SASRec | dim 128 × 3 层 | ~30 分钟 |
| LightGCN | 162k×59k 图 | ~40 分钟 |
| RQ-VAE + TIGER | K=512 码本；TIGER LM | ~30 分钟 |
| 精排 v1 | DeepFM | ~30 分钟 |
| sanity + 蒸馏粗排 | 2000 用户抽样 | ~15 分钟 |

**注意**：产物写到 `data/artifacts-ml-25m/`（与 ml-1m 隔离，互不覆盖）。
内容画像（语义通道/精排v2/冷上架依赖）默认跳过——59k 部 LLM 画像约 11 小时 API
调用，可选执行：`python scripts/enrich_profiles.py 4` 后重跑
`build.py --dataset ml-25m`（会增量补 content_emb / fine_v2）。

**验收标准**（sanity 输出，HR@10）：

| 通道 | 参考区间（25M 规模应显著高于 1M） |
|---|---|
| sasrec | 0.25–0.40（1M 上为 0.250） |
| itemcf | 0.12–0.20 |
| twotower | 0.10–0.18 |
| lightgcn | 0.08–0.18（充分训练后应远超 1M 上的 0.045） |
| tiger | 0.10–0.20 |
| hot | ~0.01–0.02 |

任意通道显著低于区间下界 → 把该数字反馈回来，我们调参。

## 3. LoRA 微调（TALLRec 式，预计 1–3 小时）

```bash
# 1) 生成训练数据（纯 CPU，分钟级；默认 5 万对，可 --n 调整）
python scripts/train_lora.py --prepare

# 2) LoRA 微调（Qwen2.5-1.5B-Instruct，r=16；显存不足见下方 FAQ）
python scripts/train_lora.py --train

# 3) 评测：LoRA vs API 零样本（M8 研究问题：微调是否优于提示词）
python scripts/train_lora.py --test --compare-api
```

产物：`data/lora-adapter/`（PEFT adapter）。

**验收标准**：
- LoRA acc 应 ≥ 0.70（1.5B 在 ML-1M 二分类任务上的合理水平；API 零样本
  参考：待测，预计 0.65–0.80）；
- 训练 loss 应从 ~1.0 降到 0.4 以下；
- 两者对比结果无论谁赢都是有价值的结论（微调赢 → 印证 TALLRec；提示词赢 →
  大模型零样本能力已够强，1.5B 微调不足，可试 7B）。

## 4. 服务与完整评测（GPU 机器上）

```bash
python scripts/run.py --dataset ml-25m --port 8000      # 服务
python scripts/eval.py                                    # 漏斗逐级评测报告
python scripts/sim_eval.py 150                            # LLM 用户模拟器（需 .env）
```

启用本地 LoRA ranker（在线对照"微调 vs 提示词"）：`.env` 加
`LLM_RANKER_BACKEND=local`，重启服务——LLM 重排走本地模型，推荐理由仍走 API。

## 5. 需要反馈给我的内容

1. **sanity 输出**（build 末尾的各通道 HR@10 表）；
2. **`data/artifacts-ml-25m/eval_report.json`**（跑完 eval.py 后）；
3. LoRA 训练 loss 曲线末尾几行 + `--test --compare-api` 的两组数字；
4. 各阶段实际耗时与显存峰值（`nvidia-smi -l 5` 观察即可）；
5. 任何 Traceback 全文。

## 6. FAQ

| 问题 | 处理 |
|---|---|
| 显存不足（1.5B LoRA） | `--train` 改小 batch：编辑 scripts/train_lora.py 中 bs=2, accum=16；或换 Qwen2.5-0.5B-Instruct（`--model`） |
| HuggingFace 下载失败 | 确认 HF_ENDPOINT=https://hf-mirror.com；或手动下载模型放到本地路径，`--model /path/to/qwen` |
| ItemCF 稀疏阶段内存爆 | build.py 里 `_build_sim_sparse` 的 block=2048 调小到 1024 |
| torch 装成 CPU 版 | `python -c "import torch; print(torch.cuda.is_available())"` 应为 True，否则重装 |
| 双塔/SASRec 效果差 | GPU 上训练快，可把 config.py 里 ml-25m 的 tt_epochs/sas_epochs 加倍再跑 |
| 25M 评分加载慢/内存紧 | build_hot 的 merge 在 25M 行上较慢属正常；内存不足时先 `ratings = ratings[["user_id","movie_id","rating","timestamp"]]` |

## 7. 代码结构速查（M8 新增/改动）

```
src/config.py            数据集注册表（ml-1m / ml-25m / ml-25m-test），产物目录隔离
src/data/pipeline.py     双格式加载器（dat / csv）
src/recall/itemcf.py     稠密 + 分块 top-K 稀疏双模式
src/recall/twotower.py   dim 可配 + Faiss HNSW（>20k 物品自动切换）
src/recall/sasrec.py     dim/layers 可配
src/data/semantic_id.py  RQ-VAE k/levels 可配；无内容画像时纯双塔退化
scripts/build.py         --dataset 参数；sanity 大规模自动抽样
scripts/run.py           --dataset / --port 参数
scripts/train_lora.py    prepare / train / test / compare-api 四模式
scripts/sim_eval.py      LLM 用户模拟器（full vs popular vs random）
src/llm/local_ranker.py  本地 LoRA ranker（LLM_RANKER_BACKEND=local）
requirements-m8.txt      transformers / peft / accelerate
```
