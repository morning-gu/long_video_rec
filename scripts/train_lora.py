"""TALLRec 式 LoRA 微调（M8，文档 03 趋势 7：LLM 后训练 vs 零样本提示）。

三步用法（prepare 可在任意机器跑；train/test 需 GPU；mode 为位置参数）：

  python scripts/train_lora.py prepare --dataset ml-25m   # 生成训练数据
  python scripts/train_lora.py train --dataset ml-25m     # LoRA 微调（GPU）
  python scripts/train_lora.py test --dataset ml-25m      # held-out AUC（GPU）
  python scripts/train_lora.py test --dataset ml-25m --compare-api  # 加测 API 对照

任务形式（TALLRec 二分类偏好）：输入 = 用户历史 + 候选电影，输出 = "是/否"。
对照实验：同一验证集上，本地 LoRA 微调模型 vs qwen3.7-plus 零样本提示——
直接回答"微调是否优于提示词"（M8 研究问题）。

模型：Qwen2.5-1.5B-Instruct（可用 --model 覆盖；7B 需 ≥24GB 显存）。
产物：data/lora-adapter/（PEFT adapter，服务端 LLM_RANKER_BACKEND=local 加载）。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# dataloader 多进程 + tokenizers 并行会触发 fork 死锁告警，提前关闭
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "train", "test"])
    ap.add_argument("--dataset", default=None,
                    help="数据集名（须先完成对应 build），默认 REC_DATASET/ml-1m")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4,
                    help="per_device_train_batch_size (T4 default 4)")
    ap.add_argument("--accum", type=int, default=8,
                    help="gradient_accumulation_steps (effective = bs * accum)")
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (e.g. --device 0; default: all visible GPUs)")
    ap.add_argument("--compare-api", action="store_true")
    return ap.parse_args()


_ARGS = _parse_args()
if _ARGS.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_ARGS.device)
if _ARGS.dataset:
    import os
    os.environ["REC_DATASET"] = _ARGS.dataset

import json
import random

import numpy as np
import pandas as pd

from src import config

print(f"数据集 {config.DATASET}（产物目录 {config.ART_DIR}）")

OUT_DIR = ROOT / "data" / "lora"
ADAPTER_DIR = ROOT / "data" / "lora-adapter"
HIST_K = 10
SYSTEM = "你是影视推荐助手。根据用户的历史观影记录，判断用户是否会喜欢候选电影。"


def _chat_text(tok, messages, add_generation_prompt=False):
    """apply_chat_template 包装：对 Qwen3 系模板关闭 thinking 模式
    （否则 assistant 回复前会插入 <think> 段，破坏"是/否"首 token 打分
    与 SFT 标签构造）；Qwen2.5 模板不使用该参数，自动忽略。"""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False)
    except TypeError:                  # 旧版 transformers 不透传未知参数
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt)


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
    from src.device import get_dtype, get_device
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=get_dtype(),
        device_map=_device_map() if get_device().type == "cuda" else "cpu")
    if adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tok


def _yes_no_ids(tok):
    """是/否 token id（打分逻辑假定单 token；多 token 时告警）。"""
    yes = tok.encode("是", add_special_tokens=False)
    no = tok.encode("否", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        print(f"[warn] 「是/否」非单 token（{yes} / {no}），"
              "首 token 打分近似可能不准")
    return yes[0], no[0]


def _yes_no_logprob(model, tok, prompt: list, batch_size=32):
    """批量计算 P("是") 与 P("否") 的首 token logprob。"""
    import torch
    yes_id, no_id = _yes_no_ids(tok)
    scores = []
    for s in range(0, len(prompt), batch_size):
        chunk = prompt[s:s + batch_size]
        texts = [_chat_text(tok, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": p}],
            add_generation_prompt=True) for p in chunk]
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


def _device_map():
    """Return device_map: DDP local_rank, or auto-split."""
    lr = os.environ.get("LOCAL_RANK")
    return {"": int(lr)} if lr is not None else "auto"


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
    from src.device import get_dtype, get_device, bf16_supported
    device = get_device()
    if device.type != "cuda":
        print("[warn] 未检测到 CUDA——LLM 在 CPU 上训练会极慢（数小时起）。"
              "常见原因：torch 编译的 CUDA 版本比驱动新（nvidia-smi 查驱动，"
              "torch.version.cuda 查编译版本），重装匹配的 cuXXX 构建"
              "（见 docs/M8-GPU训练指南.md FAQ）。"
              "如确认要用 CPU 跑请忽略本警告。")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=get_dtype(),
        device_map=_device_map() if device.type == "cuda" else "cpu")
    model.config.use_cache = False  # required for gradient checkpointing
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()  # required for grad ckpt + LoRA
    model.print_trainable_parameters()

    class SFTData(Dataset):
        def __init__(self, path):
            self.rows = [json.loads(x) for x in
                         open(path, encoding="utf-8")]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            m = self.rows[i]["messages"]
            full = _chat_text(tok, m)
            prompt = _chat_text(tok, m[:2], add_generation_prompt=True)
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

    # T4（Turing）不支持 bf16：自动降级 fp16（Ampere+ 用 bf16）
    use_bf16 = device.type == "cuda" and bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    args_kwargs = dict(
        output_dir=str(ADAPTER_DIR), num_train_epochs=epochs,
        per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        learning_rate=lr, logging_steps=20,
        save_strategy="no", bf16=use_bf16, fp16=use_fp16,
        # T4 提速：按长度分组减少 padding 浪费 + 后台加载
        group_by_length=True, dataloader_num_workers=2,
        report_to=[])
    # warmup_ratio 在部分 transformers 版本（v5 重构 / 老版本）不可用——
    # 逐级降级：ratio → steps → 无 warmup
    import transformers
    print(f"transformers {transformers.__version__}  "
          f"device={device}  dtype={get_dtype()}"
          f"  bf16={use_bf16} fp16={use_fp16}")
    for extra in ({"warmup_ratio": 0.03}, {"warmup_steps": 100}, {}):
        try:
            args = TrainingArguments(**extra, **args_kwargs)
            break
        except TypeError as e:
            print(f"[warn] TrainingArguments 不支持 {extra}（{e}），降级重试")
    trainer = Trainer(model=model, args=args,
                      train_dataset=SFTData(OUT_DIR / "train.jsonl"),
                      data_collator=collate)
    trainer.train()
    model.to("cpu")                     # 产物落盘前回 CPU（设备无关）
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
    a = _ARGS
    if a.mode == "prepare":
        prepare(n_pairs=a.n)
    elif a.mode == "train":
        train(a.model, epochs=a.epochs, bs=a.bs, accum=a.accum)
    elif a.mode == "test":
        test(a.model)
        if a.compare_api:
            compare_api()
