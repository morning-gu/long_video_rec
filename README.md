# 长视频推荐系统 Demo

基于研究洞察（`research/`）与系统设计（`docs/长视频推荐系统Demo_系统设计草案.md`）构建的
长视频推荐系统演示，目标是完整演示工业界四段式推荐漏斗：

```
多路召回 → 粗排 → 精排 → 重排 → LLM 增强层（异步可降级）
```

## 当前进度

- [x] **M1** 数据管线 + ItemCF/热门召回 + 规则重排 + FastAPI + Web 页面（含 O1 实时回流、O3 消融开关）
- [x] **M2** 双塔召回（在线前向 user tower + Faiss ANN）+ SASRec-lite 序列召回，多路召回成型
- [x] **M3** DeepFM 精排（特征交叉 + 序列信号分数级融合）+ 通道保持式粗排 + MMR 重排 + 评测报告
- [x] **M4** 完整 Web 界面（海报/推荐理由）+ LLM 层（qwen 重排 + 理由生成，异步缓存可降级）
- [x] **M4.5** 前端视觉重设计（零依赖：叠层海报/悬停浮层/骨架屏/来源徽标/暗色滚动条/动效）
- [x] **M5** 自然语言查询、同屏对比视图、评测展示页、演示脚本

## 快速开始（M1–M4）

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows（Git Bash: source .venv/Scripts/activate）
pip install -r requirements.txt
cp .env.example .env            # 或直接创建 .env 填入 LLM/TMDB key（可选）

python scripts/build.py             # 全量离线构建（首次约 30 分钟，2 核 CPU）
python scripts/enrich_posters.py    # 海报增强（约 15 分钟，可断点续跑，可选）
python scripts/run.py               # 启动服务 → http://localhost:8000
python scripts/eval.py              # 生成评测报告 → docs/评测报告.md
```

增量重建：`python scripts/build.py fine`（仅重训精排，复用 M1/M2 产物）。

`.env`（可选，gitignored）：`LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / TMDB_API_KEY`。
LLM 未配置时系统完整可用（LLM 层自动降级，主链路零依赖）。

## 功能说明（M5 全量）

- **完整四段漏斗**：多路召回（ItemCF/双塔/SASRec/热门，配额交错融合）→ 粗排
  （通道保持式压缩）→ 精排（DeepFM 特征交叉 + SASRec 分数级融合）→ 重排
  （已看过滤/类型打散/冷门保量 + MMR 多样性）；
- **LLM 增强层**（异步、缓存、可降级）：
  - LLM 重排：Top-20 + 用户画像 → JSON 重排，**候选集白名单约束防幻觉**
    （对应 GenRec catalog-aware scoring）；
  - 推荐理由：Top-12 每部一句中文理由，引用用户观影历史；
  - **自然语言查询**：「来部轻松的科幻片」→ LLM 解析（类型白名单）→ 类型
    过滤 + 双塔个性化检索；
  - 缓存命中 <50ms；界面开关可关；LLM 不可用时静默降级；
- **同屏对比**（O3）：同一用户同一时刻 完整链路 vs 纯热门，实时展示类型
  覆盖/长尾占比/平均口碑差异；
- **评测展示页**（/eval）：漏斗逐级 HR@10 条形可视化、beyond-accuracy
  对比、诚实备注（数据驱动 `python scripts/eval.py` 重新生成）；
- **海报**：IMDb suggestion API 源（api.themoviedb.org 在本网络被 DNS
  污染 + SNI 阻断，详见 README 功能说明与 .env 注释）；
- **O1 实时回流**：反馈动作收口在**详情页**（点击卡片打开：大图海报/口碑
  热度/ItemCF 相似推荐），支持 👍喜欢 / ✅看完 / 👎不感兴趣 / 1~5 星评分
  （≥4 星正反馈、≤2 星负反馈，与离线标签口径一致）；动作经用户状态实时层
  下一次请求即生效。动作体系为可扩展注册表（`user_state.py`）；
- **演示脚本**：docs/演示脚本.md（6 分钟评审流程 + 话术 + 故障预案）。

## 目录结构

```
src/
├── data/       # 数据管线：下载、解析、标签构造（评分>=4 为正反馈）、时间切分
├── state/      # 用户状态实时层（内存态 + 写透 SQLite，O1）
├── recall/     # 召回：itemcf / hot / merge（配额融合）
├── rank/       # 粗排/精排（M2/M3 交付）
├── rerank/     # 规则重排
├── llm/        # LLM 增强层（M4 交付）
├── serve/      # FastAPI 在线服务
└── eval/       # 离线评测（M3 交付）
```
