from contextlib import nullcontext
from typing import Optional

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def get_device_type() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device() -> torch.device:
    return torch.device(get_device_type())


def get_amp_dtype() -> torch.dtype:
    if get_device_type() == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def autocast(enabled: bool = True, dtype: Optional[torch.dtype] = None):
    device_type = get_device_type()
    if device_type not in {"cuda", "npu"}:
        return nullcontext()
    kwargs = {"device_type": device_type, "enabled": enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return torch.autocast(**kwargs)


def memory_allocated() -> int:
    if get_device_type() == "npu":
        return torch.npu.memory_allocated()
    if get_device_type() == "cuda":
        return torch.cuda.memory_allocated()
    return 0
