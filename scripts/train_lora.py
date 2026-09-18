"""TALLRec 式 LoRA 微调（M8，文档 03 趋势 7：LLM 后训练 vs 零样本提示）。

三步用法（prepare 可在任意机器跑；train/test 需 GPU）：

  python scripts/train_lora.py --prepare            # 生成训练数据（本地可跑）
  python scripts/train_lora.py --train              # LoRA 微调（GPU）
  python scripts/train_lora.py --test               # held-out AUC（GPU）
  python scripts/train_lora.py --test --compare-api # 加测 API 零样本对照

任务形式（TALLRec 二分类偏好）：输入 = 用户历史 + 候选电影，输出 = "是/否"。
对照实验：同一验证集上，本地 LoRA 微调模型 vs qwen3.7-plus 零样本提示——
直接回答"微调是否优于提示词"（M8 研究问题）。

模型：Qwen2.5-1.5B-Instruct（可用 --model 覆盖；7B 需 ≥24GB 显存）。
产物：data/lora-adapter/（PEFT adapter，服务端 LLM_RANKER_BACKEND=local 加载）。
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src import config

OUT_DIR = ROOT / "data" / "lora"
ADAPTER_DIR = ROOT / "data" / "lora-adapter"
HIST_K = 10
SYSTEM = "你是影视推荐助手。根据用户的历史观影记录，判断用户是否会喜欢候选电影。"


def build_prompt(history: list, cand: dict) -> str:
    hist = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(history[:HIST_K]))
    return (f"【用户历史】（新→旧）\n{hist}\n\n"
            f"【候选电影】\n{cand}\n\n"
            '用户会喜欢这部电影吗？只回答"是"或"否"。')


def load_meta():
    movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
    meta = {int(r.movie_id): (str(r.title),
                              str(r.genres).replace("|", "/"))
            for r in movies.itertuples()}
    return movies, meta


def prepare(n_pairs=50000, seed=42):
    """生成训练/验证 jsonl：正样本 = 训练位置目标，负样本 = 流行度采样。"""
    from src.data.features import positive_sequences
    ratings = pd.read_parquet(config.ART_DIR / "ratings.parquet")
    train_pos = pd.read_parquet(config.ART_DIR / "train_pos.parquet")
    hot = pd.read_parquet(config.ART_DIR / "hot.parquet")
    _, meta = load_meta()
    seqs = positive_sequences(train_pos)
    rated = {int(u): set(map(int, g))
             for u, g in ratings.groupby("user_id")["movie_id"]}
    pop = hot.movie_id.astype(int).values
    rng = random.Random(seed)

    samples = []
    for u, seq in seqs.items():
        for p in range(1, len(seq)):
            t = seq[p]
            if t not in meta:
                continue
            hist = [f"{meta[m][0]}（{meta[m][1]}）"
                    for m in reversed(seq[max(0, p - HIST_K):p]) if m in meta]
            if not hist:
                continue
            # 正样本
            samples.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(
                    hist, f"{meta[t][0]}（{meta[t][1]}）")},
                {"role": "assistant", "content": "是"}]})
            # 负样本（流行度采样，排除已看）
            for _ in range(3):
                neg = int(rng.choice(pop))
                if neg in rated.get(u, ()) or neg not in meta:
                    continue
                samples.append({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_prompt(
                        hist, f"{meta[neg][0]}（{meta[neg][1]}）")},
                    {"role": "assistant", "content": "否"}]})
                break
            if len(samples) >= n_pairs:
                break
        if len(samples) >= n_pairs:
            break
    rng.shuffle(samples)
    n_valid = min(len(samples) // 20, 2000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
        for s in samples[:n_valid]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples[n_valid:]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"训练 {len(samples) - n_valid} / 验证 {n_valid} → {OUT_DIR}")


def _load_model(model_name, adapter=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu")
    if adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tok


def _yes_no_logprob(model, tok, prompt: str, batch_size=32):
    """批量计算 P("是") 与 P("否") 的首 token logprob。"""
    import torch
    yes_id = tok.encode("是", add_special_tokens=False)[0]
    no_id = tok.encode("否", add_special_tokens=False)[0]
    scores = []
    for s in range(0, len(prompt), batch_size):
        chunk = prompt[s:s + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left", truncation=True, max_length=768).to(
            model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        scores.append(torch.stack([lp[:, yes_id], lp[:, no_id]], 1).cpu())
    return torch.cat(scores).numpy()          # [N, 2]


def _valid_prompts():
    with open(OUT_DIR / "valid.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def train(model_name="Qwen/Qwen2.5-1.5B-Instruct", epochs=2, bs=4,
          accum=8, lr=1e-4):
    import torch
    from torch.utils.data import Dataset
    from transformers import (Trainer, TrainingArguments,
                              AutoTokenizer)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu")
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    class SFTData(Dataset):
        def __init__(self, path):
            self.rows = [json.loads(x) for x in
                         open(path, encoding="utf-8")]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            m = self.rows[i]["messages"]
            full = tok.apply_chat_template(m, tokenize=False)
            prompt = tok.apply_chat_template(
                m[:2], tokenize=False, add_generation_prompt=True)
            full_ids = tok(full, truncation=True, max_length=768)[
                "input_ids"]
            prompt_len = len(tok(prompt, truncation=True,
                                 max_length=768)["input_ids"])
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            return {"input_ids": full_ids, "labels": labels}

    def collate(batch):
        mx = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor(
                [b["input_ids"] + [pad] * (mx - len(b["input_ids"]))
                 for b in batch]),
            "labels": torch.tensor(
                [b["labels"] + [-100] * (mx - len(b["labels"]))
                 for b in batch]),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (mx - len(b["input_ids"]))
                 for b in batch]),
        }

    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR), num_train_epochs=epochs,
        per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        learning_rate=lr, warmup_ratio=0.03, logging_steps=20,
        save_strategy="no", bf16=torch.cuda.is_available(),
        report_to=[])
    trainer = Trainer(model=model, args=args,
                      train_dataset=SFTData(OUT_DIR / "train.jsonl"),
                      data_collator=collate)
    trainer.train()
    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print(f"adapter 已保存 → {ADAPTER_DIR}")


def test(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    model, tok = _load_model(model_name, adapter=True)
    rows = _valid_prompts()
    prompts = [r["messages"][1]["content"] for r in rows]
    labels = [1 if r["messages"][2]["content"] == "是" else 0 for r in rows]
    scores = _yes_no_logprob(model, tok, prompts)
    pred = scores[:, 0] > scores[:, 1]           # 是 > 否
    acc = float((pred == np.array(labels)).mean())
    # AUC（正样本分数 = lp_yes - lp_no）
    diff = scores[:, 0] - scores[:, 1]
    pos = diff[np.array(labels) == 1]
    neg = diff[np.array(labels) == 0]
    auc = float((pos[:, None] > neg[None, :]).mean())
    print(f"[LoRA 本地模型] acc={acc:.4f}  AUC={auc:.4f}  (n={len(rows)})")
    return acc, auc


def compare_api():
    """同一验证集跑 qwen3.7-plus 零样本（微调 vs 提示词的研究对照）。"""
    from openai import OpenAI
    from src.llm.service import _extract_json
    client = OpenAI(api_key=config.LLM_API_KEY,
                    base_url=config.LLM_BASE_URL, timeout=30)
    rows = _valid_prompts()[:500]
    correct = 0
    for r in rows:
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL, temperature=0,
                extra_body={"enable_thinking": False},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user",
                           "content": r["messages"][1]["content"]}])
            ans = (resp.choices[0].message.content or "").strip()
            pred = "是" if ans.startswith("是") else "否"
            correct += pred == r["messages"][2]["content"]
        except Exception:
            pass
    print(f"[API 零样本 {config.LLM_MODEL}] acc={correct / len(rows):.4f}  "
          f"(n={len(rows)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "train", "test"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--compare-api", action="store_true")
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(n_pairs=a.n)
    elif a.mode == "train":
        train(a.model, epochs=a.epochs)
    elif a.mode == "test":
        test(a.model)
        if a.compare_api:
            compare_api()
