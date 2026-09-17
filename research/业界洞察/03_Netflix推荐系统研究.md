# Netflix 推荐系统研究文档

> 研究问题：
> 1. Netflix 推荐、搜索与个性化排序的公开架构与关键算法有哪些？
> 2. multi-task、sequence modeling、bandits / RL、causal inference、contextual recommendation、artwork personalization、长期会员价值、LLM / 生成式 AI 如何组合？
> 3. 从公开证据看，Netflix 2024—2026 年最新方案与技术演进是什么？

## 摘要

Netflix 是四家厂商中 2024—2026 年推荐系统公开叙事最完整的公司之一。

其最新公开技术演进可以概括为：

> **从大量专用推荐模型，走向统一的个性化 foundation model、大规模 sequence modeling、generative recommender、LLM-backed ranker 和长期满意度优化。**

Netflix 的公开技术路线不是简单地把通用大模型接到推荐系统上，而是分成几条相互关联的路线：

1. **个性化 foundation model**
   - 学习会员长期交互历史；
   - 学习内容侧深层表示；
   - 服务多个下游推荐应用。

2. **大规模 sequence modeling**
   - 将用户行为 token 化；
   - 用 transformer 类结构建模长期行为序列；
   - 使用 sparse attention、sliding window sampling、KV caching 处理长历史与低延迟冲突。

3. **层级多任务学习**
   - FM-Intent 先预测用户 session intent；
   - 再用 intent 辅助 next-item prediction。

4. **长期满意度与 reward engineering**
   - 将推荐视为 contextual bandit 问题；
   - 通过 proxy reward 和 delayed feedback prediction 对齐长期会员价值。

5. **生成式推荐**
   - 将推荐建模为 next-event prediction；
   - 扩展到约 O(1B) 参数量级；
   - 探索推荐系统自己的 scaling law。

6. **LLM-backed ranker**
   - GenRec 将内部 foundation LLM 适配为推荐排序器；
   - 引入 context engineering、catalog-aware scoring、reward alignment 和低延迟推理策略。

## 1. 总体架构演进

### 阶段 1：多个专用模型

传统 Netflix 推荐系统由多个专用模型组成，例如：

- Continue Watching；
- Today’s Top Picks for You；
- 个性化榜单；
- 搜索与推荐联动；
- artwork personalization。

这种模式的问题包括：

- 模型维护成本高；
- 创新难以跨场景复用；
- 特征和样本体系分散；
- 不同模型对同一用户的理解不一致。

### 阶段 2：Foundation Model for Personalized Recommendation

2025 年，Netflix 公开其个性化推荐 foundation model。

核心思想是：

- 用一个模型学习会员长期交互历史；
- 学习内容侧表示；
- 通过 embeddings、fine-tuning、shared weights 支持下游任务；
- 将原本分散的 preference learning 集中化。

这可以理解为推荐系统的“中枢表征层”。

### 阶段 3：应用集成

2025—2026 年公开资料显示，Netflix 将 foundation model 集成到多个个性化应用中，包括：

- homepage；
- search；
- recommendation；
- artwork personalization。

公开材料提到过不同的集成方式：

- 直接使用模型输出；
- 使用 user / entity embeddings；
- 复用模型子图；
- 对下游任务 fine-tuning。

2026 年公开资料还显示，foundation model 可能采用周期性预训练与更高频微调结合的方式，例如：

- 每月从大规模数据预训练；
- 之后基于较新数据进行日常或周期性 fine-tuning。

这说明 Netflix 的 foundation model 不是一次性静态模型，而是持续更新的生产系统。

## 2. 关键技术模块

### 2.1 Interaction tokenization：用户行为 token 化

Netflix 将用户交互历史转换为序列 token，需要处理：

- 什么算一次有意义交互；
- 多次交互是否合并；
- 同一 title 的重复行为如何表示；
- 观看时长如何编码；
- 设备、时间、地点、上下文如何进入 token；
- 过长历史如何截断或采样。

这使推荐问题从传统排序问题变成：

> **用户行为序列上的 next-event prediction。**

### 2.2 Request-time features 与 post-action features

Netflix 将特征分为两类：

1. **Request-time features**
   - 请求发生时已经可用；
   - 例如设备、时间、地点、surface、登录上下文。

2. **Post-action features**
   - 用户行为发生后才可用；
   - 例如实际观看的 title、观看时长、互动结果、内容 metadata。

在 next-item prediction 中，Netflix 会将：

- 当前 step 的 request-time features；
- 上一个 step 的 post-action features；

结合起来预测下一步交互。

这个设计的意义是：

> **上下文信息和历史行为信息共同进入同一个序列建模框架。**

### 2.3 Sparse attention、sliding window sampling、KV caching

Netflix 公开资料强调，foundation model 必须同时处理：

- 超长用户历史；
- 毫秒级线上延迟；
- 大规模 serving 请求。

为此，公开方案包括：

1. **Sparse attention**
   - 降低 self-attention 计算成本；
   - 扩展可用上下文窗口。

2. **Sliding window sampling**
   - 训练时从完整历史中采样重叠窗口；
   - 让模型在不同 epoch 看到不同历史片段。

3. **KV caching**
   - 推理时复用历史上下文计算；
   - 降低多步或重复推理成本。

这说明 Netflix 的 foundation model 不是单纯离线研究模型，而是面向线上低延迟推荐场景设计的。

### 2.4 FM-Intent：层级多任务学习

2025 年 Netflix 公开 **FM-Intent: Predicting User Session Intent with Hierarchical Multi-Task Learning**。

核心思想是：

- 用户在一个 session 中有潜在 intent；
- 只预测 next item 可能不够；
- 可以先预测 intent；
- 再让 intent 辅助 next-item prediction。

这形成层级结构：

```text
长期用户历史
   ↓
预测 session intent
   ↓
intent-aware next-item prediction
   ↓
下游推荐应用
```

公开材料提到，FM-Intent 在 Netflix user engagement dataset 上取得了离线提升，但需要注意：

- 这是离线实验结果；
- 数据集是 sampled dataset；
- 不能直接等同于线上业务指标。

### 2.5 长期会员满意度：contextual bandit 与 reward engineering

Netflix 2024 年公开《Recommending for Long-Term Member Satisfaction at Netflix》。

核心观点是：

- 只优化点击、播放、短期 engagement 可能无法代表长期满意度；
- 推荐系统应优化长期会员价值；
- Netflix 将推荐建模为 contextual bandit 问题；
- 通过 reward engineering 设计 proxy reward；
- 处理 delayed feedback 和 missing feedback。

这意味着 Netflix 的目标函数正在从：

```text
短期互动最大化
```

转向：

```text
长期会员满意度 / 长期 utility 最大化
```

其关键挑战包括：

- 长期反馈延迟；
- 反馈稀疏；
- reward 不可直接观测；
- 在线实验周期长；
- 短期指标与长期目标可能冲突。

### 2.6 大规模生成式推荐

Netflix 2026 年公开《Towards Generalizable and Efficient Large-Scale Generative Recommenders》。

公开要点包括：

- 将推荐建模为行为序列上的生成任务；
- 模型从约 O(1M) 扩展到 O(1B) 参数；
- 使用大规模周期性训练数据；
- 关注 efficiency、cold start、multi-token alignment、serving distribution shift 等问题；
- 探索推荐系统自身的 scaling law。

这表明 Netflix 正在尝试建立推荐领域自己的大规模训练体系，而不是简单复用通用 LLM 的训练范式。

### 2.7 GenRec：LLM-backed recommendation ranker

Netflix 2026 年公开 **GenRec: Towards LLM-Native Recommendation at Netflix**。

公开叙事显示，GenRec 的路线包括：

1. **内部 foundation LLM 适配**
   - 不是直接调用通用外部 LLM；
   - 而是在 Netflix 数据和推荐场景上适配模型。

2. **Context engineering**
   - 将用户上下文、设备、surface、locale、时间等信息表达为 LLM 可理解的形式。

3. **Catalog-aware scoring**
   - LLM 不自由生成任意片名；
   - 输出受到 catalog 约束，避免幻觉。

4. **Reward alignment**
   - 使用 reward signal 对齐长期满意度和业务目标。

5. **低延迟推理**
   - 公开讨论过 prefill-only 等推理策略；
   - 目标是在大规模线上服务中控制成本和延迟。

GenRec 的意义在于：

> **Netflix 正在把 LLM 从内容理解工具，推进为推荐排序模型本身。**

但需要注意，公开资料显示的是有限 A/B 或实验进展，不代表整个 Netflix 推荐系统已经完全被 LLM ranker 替代。

### 2.8 Artwork personalization 与 LLM post-training

Netflix 还公开过 artwork personalization 相关的 LLM post-training 研究。

其目标是：

- 根据用户偏好选择不同封面、宣传图；
- 让素材更符合用户兴趣；
- 用 LLM 后训练学习用户与视觉素材之间的关系。

公开结果显示，LLM post-training 在 held-out 实验中优于现有 production model，但这类结果仍需区分：

- offline held-out evaluation；
- online A/B test；
- 长期业务影响。

## 3. Netflix 推荐系统架构图

```text
会员长期行为 + 内容 metadata + 上下文
        ↓
Interaction tokenization
        ↓
Foundation Model / Sequence Encoder
  - sparse attention
  - sliding window sampling
  - KV caching
        ↓
任务适配层
  - embeddings
  - fine-tuning
  - shared weights
  - FM-Intent
        ↓
下游个性化应用
  - homepage
  - search
  - recommendation ranking
  - artwork personalization
        ↓
长期满意度 / reward engineering
  - contextual bandit
  - proxy reward
  - delayed feedback prediction
        ↓
线上实验与反馈回流
```

## 4. 模块级证据表

| 模块 | 公开证据强度 | 主要发现 |
|---|---:|---|
| Foundation model | 高 | 2025 年系统性公开 |
| Sequence modeling | 高 | 用户行为 token 化，transformer 建模长期历史 |
| Sparse attention / KV caching | 高 | 明确用于长历史和低延迟推理 |
| Multi-task learning | 高 | FM-Intent 使用层级多任务学习 |
| Intent prediction | 高 | session intent 与 next-item prediction 联合建模 |
| Contextual bandit | 高 | 2024 年长期满意度文章明确使用 |
| Reward engineering | 高 | proxy reward、delayed feedback prediction |
| Generative recommender | 高 | 2026 年公开 1M 到 1B 参数扩展 |
| LLM ranker | 高 | GenRec 公开披露 |
| Artwork personalization | 中高 | LLM post-training 有 held-out 实验 |
| Search 个性化 | 中 | 已披露被 foundation model 覆盖，但细节不足 |
| Causal inference | 中低 | 公开证据不如 reward engineering 明确 |

## 5. Netflix 的核心差异

Netflix 与其他厂商相比，最大的差异化是：

1. **订阅制业务目标**
   - 不以广告点击为核心；
   - 更关注长期会员满意度、留存和内容发现。

2. **体验级个性化**
   - 不只是推荐一个列表；
   - 还包括 homepage、search、artwork、榜单、session intent。

3. **Foundation model 集中化**
   - 从多个专用模型走向统一个性化基础模型。

4. **LLM ranker 探索较深**
   - GenRec 展示了 LLM 直接参与推荐排序的方向。

5. **长期价值建模证据强**
   - contextual bandit、proxy reward、delayed feedback prediction 的公开链条较完整。

## 6. 主要局限

1. **公开材料以工程博客为主，非完整论文。** 关键文章通常披露设计动机、架构思想和部分技术细节，但不披露完整模型结构、特征列表、训练细节、数据分布和完整 A/B test 指标。
2. **内部指标难以外部比较。** Generative recommender 的 scaling law 建立在 Netflix 内部任务、内部数据和指标上，不能直接外推。
3. **FM-Intent 的提升是离线结果。** 不能直接等同于线上业务指标提升。
4. **GenRec 的公开结果缺少完整量化细节。** 不能量化其相对 production ranker 的收益。
5. **Search 个性化细节披露不足。** 只能作为低证据强度方向，不能展开成完整架构结论。
6. **RL 与 causal inference 不是当前公开主线。** 更准确的判断是：bandit / reward modeling 证据较强；RL 与 causal inference 的生产细节公开不足。

## 7. 结论

Netflix 2024—2026 年的公开技术主线是：

> **长期满意度目标 + 用户行为序列 token 化 + 个性化 foundation model + FM-Intent 层级多任务学习 + generative recommender + LLM-backed ranker。**

更准确的判断是：

> **Netflix 不是抛弃传统推荐系统，而是把传统系统中的用户理解、内容理解、上下文建模和多任务目标，重新组织到一个 foundation model 与 LLM 后训练体系中。**

## 参考文献

1. Foundation Model for Personalized Recommendation  
   https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39

2. Integrating Netflix's Foundation Model into Personalization Applications  
   https://netflixtechblog.medium.com/integrating-netflixs-foundation-model-into-personalization-applications-cf176b5860eb

3. Recommending for Long-Term Member Satisfaction at Netflix  
   https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef

4. FM-Intent: Predicting User Session Intent with Hierarchical Multi-Task Learning  
   https://netflixtechblog.com/fm-intent-predicting-user-session-intent-with-hierarchical-multi-task-learning-94c75e18f4b8

5. Predicting User Session Intent with Hierarchical Multi-Task Learning  
   https://arxiv.org/html/2408.05353v2

6. Towards Generalizable and Efficient Large-Scale Generative Recommenders  
   https://netflixtechblog.com/towards-generalizable-and-efficient-large-scale-generative-recommenders-a7db648aa257

7. GenRec: Towards LLM-Native Recommendation at Netflix  
   https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3

8. Netflix Research, Recommendations  
   https://research.netflix.com/research-area/recommendations
