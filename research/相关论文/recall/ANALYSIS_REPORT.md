# Recall Layer Paper-Design Consistency Analysis Report

> Date: 2026-09-15 | Papers analyzed: 14 | Reference doc: 02-recommendation-pipeline.md

---

## 1. Paper Inventory and Download Status

| # | Algorithm | Paper Title | Authors | Venue | arXiv ID | Status |
|---|-----------|-------------|---------|-------|----------|--------|
| 1 | SASRec | Self-Attentive Sequential Recommendation | Kang, McAuley | ICDM 2018 | 1808.09781 | OK |
| 2 | BERT4Rec | BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer | Sun et al. | CIKM 2019 | 1904.06690 | OK |
| 3 | LRU | Resurrecting Recurrent Neural Networks for Long Sequences | Orvieto et al. | 2023 | 2303.06349 | OK |
| 4 | ReLLa | ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation | Lin et al. | WWW 2024 | 2308.11131 | OK |
| 5 | NGCF | Neural Graph Collaborative Filtering | Wang et al. | SIGIR 2019 | 1905.08108 | OK |
| 6 | SR-GNN | Session-based Recommendation with Graph Neural Networks | Wu et al. | AAAI 2019 | 1811.00855 | OK |
| 7 | LightGCN | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation | He et al. | SIGIR 2020 | 2002.02126 | OK |
| 8 | KGAT | KGAT: Knowledge Graph Attention Network for Recommendation | Wang et al. | KDD 2019 | 1905.07854 | OK |
| 9 | CL4SRec | Contrastive Learning for Sequential Recommendation | Xie et al. | ICDE 2022 | 2010.14395 | OK |
| 10 | ICLRec | Intent Contrastive Learning for Sequential Recommendation | Chen et al. | WWW 2022 | 2202.02519 | OK |
| 11 | LLMRec | LLMRec: Large Language Models with Graph Augmentation for Recommendation | Wei et al. | WSDM 2024 | 2311.00423 | OK |
| 12 | DiffKG | DiffKG: Knowledge Graph Diffusion Model for Recommendation | Jiang et al. | WSDM 2024 | 2312.16890 | OK |
| 13 | DimeRec | DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models | Li et al. | WWW 2025 | 2408.12153 | OK |
| 14 | MM-LLM Seq | Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs | Wang et al. | SIGIR 2025 | 2509.02017 | OK |

---

## 2. Per-Paper Comparison

### R1 Popular Recall - No cited paper

Design doc: Regional/time-slot statistics + real-time hot list, direct Redis read.

**Verdict**: No paper cited, method is engineering practice, no consistency issue.

---

### R2 Sequential Recall

#### 2.1 SASRec (Kang 2018)

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | self-attention captures long-range dependencies | self-attention mechanism, at each time step identifies relevant items from history | Consistent |
| Applicability | History length less than 50: use SASRec | Paper says MC works best on sparse data, RNN on dense, SASRec balances both. No sequence length threshold mentioned | Inconsistent - threshold 50 is engineering decision, not paper conclusion |
| Author citation | Kang 2018 | Wang-Cheng Kang, Julian McAuley | Consistent |

#### 2.2 BERT4Rec (Sun 2019)

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Bidirectional Transformer | Bidirectional encoder, uses Cloze task (masked prediction), contrasts with SASRec unidirectional (left-to-right) architecture | Consistent |
| Applicability | History length 50-200: use BERT4Rec | Paper mentions no sequence length threshold, BERT4Rec is a general-purpose sequential model | Inconsistent - threshold 50-200 is engineering decision, not paper conclusion |
| Author citation | Sun 2019 | Fei Sun et al. (Alibaba Group) | Consistent |

#### 2.3 LRU (Orvieto 2023)

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Linear recurrent, inference latency under 2ms | Proposes Linear Recurrent Unit (LRU), linearizes and diagonalizes recurrence, better parameterization/initialization, matches SSM performance on long sequences | Partially consistent - LRU is indeed linear recurrent, but paper is general deep learning architecture research, unrelated to recommendation systems |
| Applicability | Low-latency degradation fallback | Paper focuses on Long Range Arena benchmark, no recommendation systems involved | Inconsistent - paper is not a recommendation paper, using it as a recommendation model is cross-domain application |
| Author citation | Orvieto 2023 | Antonio Orvieto et al. (ETH Zurich and DeepMind) | Consistent |

#### 2.4 ReLLa (Design doc labels "Ren 2024")

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Retrieval-augmented, for ultra-long history (over 200) | ReLLa: Retrieval-enhanced LLMs for Lifelong Sequential Behavior Comprehension. Core is Semantic User Behavior Retrieval (SUBR), using LLM to comprehend lifelong behavior sequences | Partially consistent - retrieval augmentation direction correct, but paper focuses on CTR prediction/few-shot recommendation, not recall |
| Applicability | History length over 200: use ReLLa | Paper addresses lifelong sequential behavior incomprehension problem, no length threshold of 200 set | Inconsistent - threshold 200 is engineering decision |
| Author citation | Ren 2024 | Jianghao Lin, Rong Shan, Chenxu Zhu et al. (SJTU and Huawei) | Inconsistent - first author is Lin, not Ren |

---

### R3 CF Recall

#### 3.1 NGCF (Wang et al. 2019)

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Graph collaborative filtering, GNN message propagation | Models user-item interactions as bipartite graph, GNN embedding propagation, captures high-order connectivity | Consistent |
| Author citation | Wang et al. 2019 | Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua | Consistent |

#### 3.2 Matrix Factorization

Design doc states as dual-path complement to NGCF. Matrix factorization is a classic method with no specific paper cited, no consistency issue.

---

### R4 Graph Recall

#### 4.1 SR-GNN (Wu et al. 2019)

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Session graph neural network, builds session graph, GNN message propagation learns session representation | Models session sequences as graph-structured data, GNN captures complex transitions of items | Consistent |
| Author citation | Wu et al. 2019 | Shu Wu, Yuyuan Tang, Yanqiao Zhu et al. | Consistent |

#### 4.2 LightGCN

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Graph convolution collaborative filtering, no feature transformation or nonlinear activation | Empirically finds that feature transformation and nonlinear activation in GCN contribute little to CF, only keeps neighborhood aggregation | Consistent |
| Author citation | Not labeled | Xiangnan He et al. (SIGIR 2020) | Missing - design doc does not label author |

---

### R5 KG Path Recall

#### 5.1 KGAT (Wang et al. 2019)

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Knowledge graph attention propagation, KGAT attention mechanism assigns weights to paths | KGAT explicitly models high-order connectivity, attention propagation on KG and user-item graph | Consistent |
| Data source | "Neo4j graph traversal + KGAT inference" | Original paper uses knowledge graphs, does not specify Neo4j | Inconsistent - Neo4j is an engineering implementation choice, not paper content |
| Venue | Wang et al. 2019 (venue not specified) | KDD 2019 (not SIGIR) | Missing - venue not labeled |
| Author citation | Wang et al. 2019 | Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, Tat-Seng Chua | Consistent |

---

### R6 Contrastive Learning Recall

#### 6.1 CL4SRec (Design doc labels "Wu 2022")

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Contrastive learning sequence augmentation, crop/reorder/mask | Proposes three data augmentation methods (crop/mask/reorder) to construct self-supervised signals | Consistent |
| Author citation | Wu 2022 | Xu Xie, Fei Sun, Zhaoyang Liu et al. (Peking Univ. and Alibaba) | Inconsistent - first author is Xie, not Wu |
| Venue | Not labeled | ICDE 2022 | Missing |

#### 6.2 ICLRec (Design doc labels "Zhou 2022")

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Intent contrastive learning, extracts user intent prototypes | Uses latent intent variables, learns user intent distribution functions from unlabeled behavior sequences | Consistent |
| Author citation | Zhou 2022 | Yongjun Chen, Zhiwei Liu, Jia Li, Julian McAuley, Caiming Xiong (Salesforce Research) | Inconsistent - first author is Chen, not Zhou |
| Venue | Not labeled | WWW 2022 | Missing |

---

### R7 LLM Semantic Recall

#### 7.1 LLMRec (Design doc labels "Ren et al. 2024")

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | "LLMRec knowledge enhancement + natural language preference query"; flow is "user behavior to natural language description then LLM generates query vector then Milvus retrieval then LLM semantic scoring" | LLMRec proposes three LLM-based graph augmentation strategies: using LLM to augment user-item interaction graphs (adding semantic content edges), addressing data sparsity | Severely inconsistent - design doc flow is completely different from paper actual method |
| Author citation | Ren et al. 2024 | Wei Wei, Xubin Ren, Jiabin Tang et al. (HKU and Baidu) | Inconsistent - first author is Wei, Ren is second author |
| Venue | Not labeled | WSDM 2024 | Missing |

---

### R8 Diffusion Generative Recall

#### 8.1 DiffKG (Jiang 2024)

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Knowledge graph diffusion denoising, forward noising + backward denoising | Integrates generative diffusion model with knowledge graph, removes noisy relations in KG | Consistent - diffusion denoising direction correct |
| Author citation | Jiang 2024 | Yangqin Jiang, Yuhao Yang, Lianghao Xia, Chao Huang (HKU) | Consistent |
| Venue | Not labeled | WSDM 2024 | Missing |

#### 8.2 DimeRec (Design doc labels "An 2025")

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | Unified generative framework | DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models, simultaneously optimizes item representation and diversity | Consistent |
| Author citation | An 2025 | Wuchao Li, Rui Huang, Haijun Zhao et al. (USTC and Kuaishou) | Severely inconsistent - "An" not in author list, first author is Li |
| Venue | Not labeled | WWW 2025 | Missing |

---

### R9 Content Recall

#### 9.1 MM-LLM Seq 2025

| Dimension | Design doc claim | Original paper | Consistency |
|-----------|-----------------|----------------|-------------|
| Core method | "Multimodal embedding similarity retrieval" | "Empowering LLM for Sequential Recommendation via Multimodal Embeddings and Semantic IDs", core is using multimodal embeddings and semantic IDs to empower LLM for sequential recommendation, addressing embedding collapse and catastrophic forgetting | Partially inconsistent - paper is an LLM-based sequential recommendation method, not just "embedding similarity retrieval" |
| Citation | "MM-LLM Seq 2025" (no specific author/title) | Yuhao Wang et al. (CityU and Tencent), SIGIR 2025 | Vague - citation imprecise |

---

## 3. Issue Summary

### P0 - Severe Inconsistency (method description does not match paper)

| # | Issue | Details |
|---|-------|---------|
| 1 | LLMRec method description error | Design doc describes LLMRec as "natural language preference query to LLM to query vector to Milvus retrieval to LLM semantic scoring". Actual paper proposes three LLM-based graph augmentation strategies, using LLM to augment user-item interaction graphs, not semantic query retrieval. This is a fundamental method deviation |
| 2 | LRU is not a recommendation paper | LRU (Orvieto et al. 2023) paper title is "Resurrecting Recurrent Neural Networks for Long Sequences", it is general deep learning architecture research, evaluated on Long Range Arena benchmark, does not involve recommendation systems. Design doc using it as a recommendation model is cross-domain application with no paper support |

### P1 - Author Citation Errors

| # | Algorithm | Design doc label | Paper first author | Deviation |
|---|-----------|-----------------|-------------------|-----------|
| 1 | ReLLa | Ren 2024 | Jianghao Lin | First author completely different |
| 2 | CL4SRec | Wu 2022 | Xu Xie | First author completely different |
| 3 | ICLRec | Zhou 2022 | Yongjun Chen | First author completely different |
| 4 | LLMRec | Ren et al. 2024 | Wei Wei | Ren is second author |
| 5 | DimeRec | An 2025 | Wuchao Li | "An" not in author list |

### P2 - Method Applicability Conditions Without Paper Support

| # | Issue | Details |
|---|-------|---------|
| 1 | Sequence length thresholds | Design doc claims SASRec for under 50, BERT4Rec for 50-200, ReLLa for over 200. All three thresholds have no paper basis. SASRec and BERT4Rec are both general-purpose sequential models, papers do not divide applicability by sequence length |
| 2 | LRU inference latency under 2ms | Paper does not mention inference latency metrics, nor evaluate in recommendation scenarios |
| 3 | ReLLa as recall model | ReLLa paper focuses on CTR prediction and few-shot recommendation, not recall scenario |

### P3 - Engineering Implementation Confused with Paper Method

| # | Issue | Details |
|---|-------|---------|
| 1 | KGAT + Neo4j | Design doc claims "Neo4j graph traversal + KGAT inference". Original KGAT paper uses knowledge graphs but does not specify Neo4j. Neo4j is an engineering implementation choice |
| 2 | SR-GNN + LightGCN dual-path merge | Design doc merges two independent papers methods into dual-path. Original papers are independent, do not propose merge scheme |
| 3 | MM-LLM Seq as content recall | Paper is an LLM-based sequential recommendation method, design doc simplifies to "multimodal embedding similarity retrieval", weakens paper core contribution |

### P4 - Citation Information Missing

| # | Issue | Details |
|---|-------|---------|
| 1 | Venue widely missing | Of 14 papers, only author+year labeled, most do not label venue (e.g., KGAT is actually KDD 2019 not SIGIR, CL4SRec is ICDE 2022, ICLRec is WWW 2022, LLMRec is WSDM 2024, etc.) |
| 2 | LightGCN author missing | Design doc does not label LightGCN author |
| 3 | MM-LLM Seq citation vague | Only writes "MM-LLM Seq 2025", no author, title, or venue, difficult to confirm specific paper |

---

## 4. Per-Recall-Path Consistency Summary

| Recall path | Method description | Author citation | Applicability | Overall |
|------------|-------------------|-----------------|--------------|---------|
| R1 Popular | N/A | N/A | N/A | No issue |
| R2 Sequential | SASRec/BERT4Rec consistent, LRU cross-domain, ReLLa direction correct but scenario deviation | ReLLa author wrong | Length thresholds unfounded | Needs correction |
| R3 CF | NGCF consistent | Consistent | N/A | Consistent |
| R4 Graph | SR-GNN/LightGCN consistent | LightGCN author missing | N/A | Mostly consistent |
| R5 KG | KGAT consistent, Neo4j is engineering choice | Consistent | N/A | Mostly consistent |
| R6 Contrastive | CL4SRec/ICLRec consistent | Both authors wrong | N/A | Needs citation correction |
| R7 LLM | Method description does not match paper | Author wrong | N/A | Needs redesign |
| R8 Diffusion | DiffKG/DimeRec consistent | DimeRec author wrong | N/A | Needs citation correction |
| R9 Content | Method simplified but direction correct | Citation vague | N/A | Needs citation completion |

---

## 5. Correction Suggestions

1. **LLMRec recall path needs redesign**: Current flow description (natural language preference query to query vector to Milvus retrieval) does not match paper actual method (LLM-based graph augmentation). Either modify flow description to match paper, or find a paper that better fits the "natural language preference query" approach

2. **LRU citation needs clarification**: Clearly label that this paper is not a recommendation system paper, LRU architecture is cross-domain applied to recommendation scenarios, "inference latency under 2ms" is engineering measurement not paper conclusion

3. **Fix 5 author citations**: ReLLa (Lin, not Ren), CL4SRec (Xie, not Wu), ICLRec (Chen, not Zhou), LLMRec (Wei, not Ren), DimeRec (Li, not An)

4. **Sequence length thresholds need labeling as engineering decisions**: Clearly state that the boundaries are system engineering choices, not paper experimental conclusions

5. **Add venue information**: KGAT (KDD 2019), CL4SRec (ICDE 2022), ICLRec (WWW 2022), LLMRec (WSDM 2024), DiffKG (WSDM 2024), DimeRec (WWW 2025), ReLLa (WWW 2024), MMLLMSeq (SIGIR 2025)

6. **MM-LLM Seq citation needs precision**: Add full title "Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs" and author (Wang et al., SIGIR 2025)
