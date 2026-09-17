# YouTube / Google 推荐系统研究文档

> 研究问题：
> 1. YouTube home / watch-next 推荐的公开架构与关键算法有哪些？
> 2. two-tower retrieval、sequence modeling、ranking、multi-objective、LLM / 生成式推荐、RL / long-term value、负反馈、公平性与评估如何组合？
> 3. 从公开证据看，2024—2026 年其最新方案与技术演进是什么？

## 摘要

YouTube / Google 推荐系统的公开主线是：

> **大规模多阶段推荐漏斗 + embedding retrieval + multi-task ranking + sequence modeling，正在向 Semantic ID、Gemini / LLM、generative retrieval 和语义理解方向演进。**

与 Netflix 相比，YouTube 公开披露的重点更偏：

- 超大规模候选生成；
- 多任务排序；
- 视频语义 token 化；
- Semantic ID；
- Gemini 适配；
- 生成式检索。

与字节跳动相比，YouTube 公开的工程系统细节较少，但其经典论文和 Google 研究生态使得其算法架构证据非常清晰。

## 1. 公开架构主线

YouTube 最经典的公开架构来自 2016 年论文：

**《Deep Neural Networks for YouTube Recommendations》**

其核心架构是：

```text
用户历史行为 / 上下文
   ↓
Candidate Generation
大规模候选生成
   ↓
Ranking
候选排序
   ↓
展示
```

其中：

- Candidate Generation 将海量视频缩小到较小候选集；
- Ranking 对候选视频进行精细排序；
- 用户历史行为被映射为 embedding；
- embedding 相似度用于召回。

2019 年论文：

**《Recommending What Video to Watch Next: A Multitask Ranking System》**

进一步公开了 watch-next 场景中的 multi-objective ranking 设计。

这说明 YouTube 的公开架构长期保持：

> **候选生成 + 精细排序** 的主线。

## 2. 关键技术模块

### 2.1 Candidate Generation / Retrieval

YouTube 2016 年论文明确了 candidate generation 的重要性：

- 视频库规模极大；
- 无法对所有视频做完整精排；
- 需要先用高效方法召回候选。

公开技术路线包括：

- 用户 embedding；
- 视频 embedding；
- 近似最近邻检索；
- 多路候选生成；
- 历史行为序列表示。

这与工业界 two-tower retrieval 思路一致：

```text
user / query tower
        ↓
user embedding
                → ANN retrieval
item tower
        ↓
item embedding
```

two-tower 的优势是：

- item embedding 可以离线构建；
- 线上只计算 user tower；
- 支持大规模低延迟召回。

### 2.2 Multi-objective ranking

2019 年 YouTube 论文公开了大规模 multi-objective ranking 系统。

其核心问题是：

推荐下一视频不能只优化一个指标，而需要同时考虑：

- 是否观看；
- 观看时长；
- 满意度；
- 互动行为；
- 后续行为；
- 平台生态目标。

因此，ranking 层需要：

- 多任务学习；
- 多目标融合；
- 偏差处理；
- 不同任务之间的冲突处理。

这也是现代视频推荐系统的共同趋势：

> **从 CTR 模型走向 multi-objective / multi-task ranking。**

### 2.3 Sequence modeling

YouTube 2016 年论文已经使用用户历史观看行为作为核心特征。

后续工业推荐系统普遍将 sequence modeling 引入：

- 短期兴趣；
- 长期兴趣；
- session 内兴趣转移；
- 跨场景行为；
- 负反馈；
- 时间衰减。

虽然 YouTube 对最新 sequence model 细节公开有限，但其技术主线与 Google 推荐研究生态一致：

```text
用户行为序列
   ↓
embedding / attention / transformer
   ↓
用户兴趣表示
   ↓
召回与排序
```

### 2.4 Semantic ID 与 generative retrieval

Google 的推荐系统研究在 2023 年后明显转向 semantic ID 和 generative retrieval。

代表工作是：

**《Recommender Systems with Generative Retrieval》**

其核心思想是：

- 为 item 学习 semantic ID；
- 将推荐任务转化为生成 semantic ID 的任务；
- 模型不再只是通过向量相似度检索 item；
- 而是可以直接生成候选 item 的 token 序列。

这带来一个重要变化：

```text
传统召回：
user embedding → ANN → item embedding

生成式召回：
user history / context → generate semantic ID → candidate item
```

这类方法的意义在于：

- item 可以被结构化 token 表示；
- 检索过程可以与序列生成统一；
- LLM 或 transformer 可以更自然地建模推荐任务。

### 2.5 Gemini / Large Recommender Model

2025 年公开演讲：

**“Teaching Gemini to Speak YouTube: Adapting LLMs for Video Recommendations to 2B+ DAU”**

披露了 YouTube 将 Gemini 适配为视频推荐模型的方向。

公开叙事中的关键词包括：

- Large Recommender Model / LRM；
- Gemini checkpoint adaptation；
- Semantic ID；
- 视频 token 化；
- generative video retrieval；
- 大规模低延迟 serving。

其技术方向可以概括为：

```text
视频内容
   ↓
Semantic ID / video tokenization
   ↓
Gemini-based Large Recommender Model
   ↓
生成式候选检索 / 排序
   ↓
现有推荐漏斗集成
```

需要注意的是：

- 该证据主要来自公开演讲；
- 不是完整系统论文；
- 不能证明 YouTube 已经完全用 LRM 替换现有推荐系统。

更合理的判断是：

> **YouTube 正在把 Gemini / Semantic ID / generative retrieval 叠加到既有大规模推荐漏斗中，而不是完全推翻传统 retrieval-ranking 架构。**

## 3. YouTube 推荐系统架构图

基于公开证据，可以概括为：

```text
用户历史观看、搜索、上下文、显式反馈
        ↓
多路候选生成 / Retrieval
  - embedding retrieval
  - two-tower / ANN
  - sequence features
  - semantic ID / generative retrieval
        ↓
候选合并、过滤、去重
        ↓
Multi-objective Ranking
  - 观看概率
  - 观看时长
  - 满意度
  - 互动
  - 平台目标
        ↓
页面级重排 / 展示策略
        ↓
Home feed / Watch Next / Shorts
        ↓
A/B 实验与反馈回流
```

## 4. 模块级证据表

| 模块 | 公开证据强度 | 主要发现 |
|---|---:|---|
| Candidate generation | 高 | 2016 年经典论文明确披露 |
| Ranking | 高 | 2016、2019 年论文均披露 |
| Multi-objective ranking | 高 | 2019 年 watch-next 论文重点 |
| Two-tower retrieval | 中高 | Google 推荐系统公开资料中广泛出现 |
| Sequence modeling | 中高 | 用户历史是核心特征，但最新模型细节公开有限 |
| Semantic ID | 中高 | Google generative retrieval 研究明确 |
| Gemini / LRM | 中高 | 2025 年公开演讲披露 |
| Generative retrieval | 中高 | Semantic ID + 生成式检索成为新方向 |
| 重排 | 中 | 可推断存在，但公开细节有限 |
| 负反馈 | 低中 | 产品功能存在，但生产建模细节公开不足 |
| 公平性 / 创作者生态 | 低 | 公开证据不完整 |
| 长期价值 RL | 低 | 不能仅凭通用文献推断 YouTube 生产方案 |

## 5. YouTube 的核心差异

YouTube 与其他厂商相比，差异化在于：

1. **内容形态复杂**
   - 长视频、短视频、直播、搜索、推荐、订阅、频道关系交织。

2. **搜索与推荐高度耦合**
   - 用户主动搜索和历史观看共同影响推荐。

3. **超大规模候选空间**
   - 需要极强的 retrieval 和 ANN 能力。

4. **多目标排序成熟**
   - 2019 年已经公开大规模 multi-objective ranking 系统。

5. **Semantic ID / Gemini 方向明确**
   - 2025 年公开将 Gemini 适配为 Large Recommender Model 的方向。

## 6. 主要局限

1. **公开论文不等于生产系统。** YouTube 公开材料通常只披露局部模块，而非完整线上系统。
2. **LLM 化推荐仍处于公开披露早期。** 2025 年演讲显示了方向，但缺少完整技术论文、消融实验、上线指标与系统延迟细节。
3. **公平性、负反馈、长期价值优化公开证据不足。** 这些方向对理解 YouTube 推荐系统非常重要，但本次公开材料不足以完整还原。
4. **评估体系难以外部还原。** YouTube 内部可能使用在线 A/B、长期满意度、创作者生态指标、观看质量指标等，但公开材料不足以完整还原。

## 7. 结论

YouTube 的推荐系统主线是：

> **大规模多阶段漏斗 + embedding retrieval + multi-objective ranking + sequence modeling，正在引入 Semantic ID、Gemini / LLM 和 generative retrieval。**

最重要的公开变化是：

> **从“专用 embedding + 排序模型”，走向“语义 token + 大模型适配 + 生成式检索与排序”。**

但这不意味着传统 two-tower、multi-task ranking 被完全替代。更合理的判断是：

> **LLM / generative recommendation 正在被叠加进既有系统，成为新的召回、语义理解和排序能力。**

## 参考文献

1. Deep Neural Networks for YouTube Recommendations  
   https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/

2. Recommending What Video to Watch Next: A Multitask Ranking System  
   https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/

3. Recommending What Video to Watch Next: A Multitask Ranking System, OpenReview PDF  
   https://openreview.net/pdf?id=gxlKBGBBwN

4. Recommender Systems with Generative Retrieval  
   https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf

5. Teaching Gemini to Speak YouTube: Adapting LLMs for Video Recommendations to 2B+ DAU  
   https://www.youtube.com/watch?v=LxQsQ3vZDqo

6. Teaching Gemini to Speak YouTube, AI Engineer talk page  
   https://ai.engineer/talks/LxQsQ3vZDqo-teaching-gemini-speak-youtube-adapting-llms-video

7. Google, Recommendation systems overview  
   https://developers.google.com/machine-learning/recommendation/overview/types
