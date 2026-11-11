from typing import Any, Optional

import torch

DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters

DUMMY_INT64 = torch.empty(0, dtype=torch.int64)

DUMMY_KEY_PAST = torch.empty((0, 0, 0))


def is_dummy(tensor: torch.Tensor) -> bool:
    return tensor.numel() == 0


def flag_to_bool(value: Any) -> bool:
    if value is None:
        return False
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(value.bool().any().item())
    return bool(value)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def dtype_name(dtype: Optional[torch.dtype]) -> str:
    return "" if dtype is None else str(dtype).replace("torch.", "")


def slice_batch_aligned(
    value: Any,
    mb_start: int,
    mb_end: int,
    full_batch_size: int,
) -> Any:
    if value is None or not torch.is_tensor(value):
        return value
    if is_dummy(value) or value.ndim == 0:
        return value
    if value.shape[0] == full_batch_size:
        return value[mb_start:mb_end].contiguous()
    return value


SPECIAL_DTYPE_SIZES = {torch.bool: 1, torch.qint8: 1, torch.qint32: 4}


def get_size_in_bytes(dtype: torch.dtype) -> int:
    if dtype in SPECIAL_DTYPE_SIZES:
        return SPECIAL_DTYPE_SIZES[dtype]
    get_info = torch.finfo if dtype.is_floating_point else torch.iinfo
    return (get_info(dtype).bits * (1 + dtype.is_complex)) // 8


def docstring_from(source):
    def add_docstring(dest):
        dest.__doc__ = source.__doc__
        return dest

    return add_docstring
