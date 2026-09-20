"""统一设备支持（M8）：训练与推理的 torch device 单点来源。

REC_DEVICE 环境变量（.env 可配）：
  auto（默认）—— CUDA 可用即用，否则 CPU；
  cpu / cuda / cuda:0 —— 强制指定。

约定：
- 训练器：入口处 `device = get_device()`，model/张量上设备，产物
  （state_dict / numpy）落盘前 `.cpu()`；
- 服务端：加载时 map_location=get_device()，输入张量上设备，输出
  `.cpu().numpy()`——CPU 机器行为与原先完全一致，GPU 机器自动加速；
- 纯 numpy/scipy 组件（ItemCF/LightGCNRecall/SemanticRecall/内容向量）
  不涉及 torch，不受影响。
"""
import os

import torch

_CACHE = None


def get_device() -> torch.device:
    global _CACHE
    if _CACHE is None:
        d = os.environ.get("REC_DEVICE", "auto").strip().lower()
        if d in ("", "auto"):
            _CACHE = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        elif d == "cpu":
            _CACHE = torch.device("cpu")
        else:
            _CACHE = torch.device(d)      # cuda / cuda:0 ...
    return _CACHE


def bf16_supported() -> bool:
    """CUDA bf16 支持（Ampere 及以上）。T4（Turing）为 False → 用 fp16。"""
    d = get_device()
    return d.type == "cuda" and torch.cuda.is_bf16_supported()


def get_dtype() -> torch.dtype:
    """训练/推理 dtype 单点：bf16（Ampere+）> fp16（Turing，如 T4）> float32。"""
    d = get_device()
    if d.type != "cuda":
        return torch.float32
    return torch.bfloat16 if bf16_supported() else torch.float16
