"""本地 LoRA ranker（M8）：加载 Qwen + LoRA adapter 做候选重排。

与 API 零样本 LLM rerank（LLMService._rerank）同一任务形态的本地版——
"微调 vs 提示词"的线上对照（文档 03 趋势 7）。

- 打分：P("是") vs P("否") 的首 token logprob 差，候选按差值排序；
- 仅建议 GPU 部署（1.5B × 20 候选：GPU ~200ms，2 核 CPU ~20s+）；
- transformers/peft 延迟导入：API-only 安装不受影响。

启用：.env 配置 LLM_RANKER_BACKEND=local + LORA_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct，
且 data/lora-adapter/ 存在（scripts/train_lora.py train 产出）。
加载失败自动回退 API（主链路零依赖原则不变）。
"""
import numpy as np


class LocalLLMRanker:
    def __init__(self, base_model: str, adapter_dir, history_k=10):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16 if
            torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu")
        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        self.history_k = history_k
        self.yes_id = self.tok.encode("是", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("否", add_special_tokens=False)[0]

    def rerank(self, history_titles: list, movies: list):
        """movies: [{movie_id, title, genres}]；返回按偏好降序的 movie_id 列表。"""
        hist = "\n".join(f"{i + 1}. {t}"
                         for i, t in enumerate(history_titles[:self.history_k]))
        prompts = []
        for m in movies:
            cand = f"{m['title']}（{'/'.join(m['genres'][:3])}）"
            prompts.append(
                f"【用户历史】（新→旧）\n{hist}\n\n【候选电影】\n{cand}\n\n"
                '用户会喜欢这部电影吗？只回答"是"或"否"。')
        texts = [self.tok.apply_chat_template(
            [{"role": "system", "content":
              "你是影视推荐助手。根据用户的历史观影记录，"
              "判断用户是否会喜欢候选电影。"},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        diffs = []
        for s in range(0, len(texts), 8):
            enc = self.tok(texts[s:s + 8], return_tensors="pt", padding=True,
                           padding_side="left", truncation=True,
                           max_length=768).to(self.model.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            lp = self.torch.log_softmax(logits, -1)
            diffs.extend((lp[:, self.yes_id] - lp[:, self.no_id])
                         .cpu().numpy().tolist())
        order = np.argsort(-np.array(diffs))
        return [int(movies[i]["movie_id"]) for i in order]
