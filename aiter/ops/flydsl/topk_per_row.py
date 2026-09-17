# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL decode TopK interface."""

from functools import lru_cache

import torch

from .kernels.kernels_common import get_warp_size
from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode import (
    build_topk_per_row_decode_module,
    topk_per_row_decode_chunks,
    topk_per_row_decode_workspace_shapes,
)
from .kernels.topk_per_row_decode_persistent import (
    build_topk_per_row_decode_one_workgroup_module,
)
from .kernels import topk_per_row_decode_adaptive as _adaptive

# Measured crossover between the one-workgroup and multi-kernel paths.
_ONE_WORKGROUP_MAX_ROW_WIDTH = 20_000

# The arches whose sweep says the adaptive kernel takes every cell of the chunked
# path it replaces. gfx942 is absent because the sweep that would admit it was
# taken at a CU count this file cannot assume.
_ADAPTIVE_ARCHES = ("gfx950",)

# The k values the sweeps cover. The kernel builds at any k; an unmeasured one
# keeps the chunked path rather than shipping an unmeasured kernel. Which shapes
# reach here at each k is the gate's decision, not this one's.
_ADAPTIVE_KS = (256, 512, 1024, 2048, 4096)


@lru_cache(maxsize=8)
def _adaptive_cu_count(device_index: int) -> int:
    """CU count of the device the row will run on, which one arch name spans
    several of, so the kernel's grid tables cannot take it as a constant."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _adaptive_admits(arch: str, k: int, values: torch.Tensor | None) -> bool:
    """Whether the call is inside the region the adaptive kernel was measured in;
    `values` is the hard one, because this kernel emits indices only."""
    return (
        any(arch.startswith(name) for name in _ADAPTIVE_ARCHES)
        and k in _ADAPTIVE_KS
        and values is None
    )


@lru_cache(maxsize=16)
def _get_cached_adaptive_workspace(
    device: torch.device, stream_id: int, slots: int
) -> torch.Tensor:
    return torch.zeros(slots, device=device, dtype=torch.int32)


def _get_adaptive_workspace(
    device: torch.device, stream_id: int, slots: int
) -> torch.Tensor:
    # Do not let graph-pool allocations escape through the process cache.
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros(slots, device=device, dtype=torch.int32)
    return _get_cached_adaptive_workspace(device, stream_id, slots)


@lru_cache(maxsize=16)
def _get_cached_workspace(
    device: torch.device,
    stream_id: int,
    hist_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep scratch isolated by device, stream, and exact kernel layout."""
    return (
        torch.empty(hist_shape, device=device, dtype=torch.int32),
        torch.empty(state_shape, device=device, dtype=torch.int32),
    )


def _get_topk_workspace(
    device: torch.device,
    stream_id: int,
    hist_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Do not let graph-pool allocations escape through the process cache.
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.empty(hist_shape, device=device, dtype=torch.int32),
            torch.empty(state_shape, device=device, dtype=torch.int32),
        )
    return _get_cached_workspace(
        device,
        stream_id,
        hist_shape,
        state_shape,
    )


def clear_topk_per_row_decode_workspace_cache() -> None:
    _get_cached_workspace.cache_clear()


@lru_cache(maxsize=128)
def _validate_topk_signature(
    logits_shape: torch.Size,
    logits_stride: tuple[int, ...],
    logits_dtype: torch.dtype,
    logits_device: torch.device,
    seq_lens_shape: torch.Size,
    seq_lens_stride: tuple[int, ...],
    seq_lens_dtype: torch.dtype,
    seq_lens_device: torch.device,
    indices_shape: torch.Size,
    indices_stride: tuple[int, ...],
    indices_dtype: torch.dtype,
    indices_device: torch.device,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
) -> None:
    if len(logits_shape) != 2 or logits_dtype != torch.float32:
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if logits_device.type != "cuda":
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if k <= 0 or k > logits_shape[1]:
        raise ValueError("k must be in the range [1, logits.shape[1]]")
    if logits_stride[1] != 1:
        raise ValueError("logits must have inner stride 1")

    rows = logits_shape[0]
    if num_rows != rows:
        raise ValueError("num_rows must equal logits.shape[0]")
    # There used to be a refusal here at 4 GiB. The kernels built one buffer
    # descriptor over the whole logits tensor, and both `num_records` and the
    # load's voffset in it are 32-bit byte quantities, so a row past 4 GiB was
    # unaddressable and read as zero -- silently wrong rather than faulting, at
    # 4096 x 262144 but not 4095 x 262144. They now slice the row before
    # building the descriptor, which puts the row's base in the descriptor's
    # 48-bit base address and leaves only an offset within the row, so the
    # tensor as a whole is no longer bounded. The refusal went with it.
    if (stride0, stride1) != logits_stride:
        raise ValueError("stride0 and stride1 must match logits strides")

    if len(seq_lens_shape) != 1 or seq_lens_dtype != torch.int32:
        raise ValueError("seq_lens must be a 1D int32 tensor")
    if seq_lens_stride != (1,):
        raise ValueError("seq_lens must be contiguous")
    if seq_lens_device != logits_device:
        raise ValueError("seq_lens must be on the same CUDA device as logits")
    if next_n <= 0:
        raise ValueError("next_n must be positive")
    required_seq_lens = (rows + next_n - 1) // next_n
    if seq_lens_shape[0] < required_seq_lens:
        raise ValueError("seq_lens does not have enough entries for logits rows")

    if indices_shape != (rows, k) or indices_dtype != torch.int32:
        raise ValueError("indices must be an int32 tensor with shape [rows, k]")
    if indices_stride != (k, 1):
        raise ValueError("indices must be contiguous")
    if indices_device != logits_device:
        raise ValueError("indices must be on the same CUDA device as logits")


@lru_cache(maxsize=128)
def _validate_values_signature(
    values_shape: torch.Size,
    values_stride: tuple[int, ...],
    values_dtype: torch.dtype,
    values_device: torch.device,
    rows: int,
    k: int,
    logits_device: torch.device,
) -> None:
    if values_dtype != torch.float32:
        raise ValueError("values must be a float32 tensor")
    if values_shape != (rows, k):
        raise ValueError("values must have shape [rows, k]")
    if values_stride != (k, 1):
        raise ValueError("values must be contiguous")
    if values_device != logits_device:
        raise ValueError("values must be on the same CUDA device as logits")


def _validate_flydsl_topk_call(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None,
) -> None:
    _validate_topk_signature(
        logits.shape,
        logits.stride(),
        logits.dtype,
        logits.device,
        seq_lens.shape,
        seq_lens.stride(),
        seq_lens.dtype,
        seq_lens.device,
        indices.shape,
        indices.stride(),
        indices.dtype,
        indices.device,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
    )
    if values is not None:
        _validate_values_signature(
            values.shape,
            values.stride(),
            values.dtype,
            values.device,
            logits.shape[0],
            k,
            logits.device,
        )


_TensorSignature = tuple[
    torch.Size,
    tuple[int, ...],
    torch.dtype,
    torch.device,
]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


@lru_cache(maxsize=128)
def _is_flydsl_topk_call_supported(
    logits_signature: _TensorSignature,
    seq_lens_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values_signature: _TensorSignature | None,
) -> bool:
    try:
        _validate_topk_signature(
            *logits_signature,
            *seq_lens_signature,
            *indices_signature,
            next_n,
            num_rows,
            stride0,
            stride1,
            k,
        )
        if values_signature is not None:
            _validate_values_signature(
                *values_signature,
                logits_signature[0][0],
                k,
                logits_signature[3],
            )
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def is_flydsl_top_k_per_row_decode_supported(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None = None,
) -> bool:
    """Return whether a call satisfies every FlyDSL-only precondition."""
    return _is_flydsl_topk_call_supported(
        _tensor_signature(logits),
        _tensor_signature(seq_lens),
        _tensor_signature(indices),
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        None if values is None else _tensor_signature(values),
    )


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    values: torch.Tensor | None = None,
) -> None:
    """Write per-row TopK indices using each request's effective context length."""

    _validate_flydsl_topk_call(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        k,
        values,
    )

    rows, width = logits.shape
    arch = torch.cuda.get_device_properties(logits.device).gcnArchName
    wave_size = get_warp_size(arch)
    stream = torch.cuda.current_stream(logits.device)
    if width <= _ONE_WORKGROUP_MAX_ROW_WIDTH:
        launcher = build_topk_per_row_decode_one_workgroup_module(
            k,
            wave_size=wave_size,
            write_values=values is not None,
        )
        _run_compiled(
            launcher,
            logits,
            seq_lens,
            indices,
            values if values is not None else logits,
            width,
            next_n,
            stride0,
            rows,
            stream,
        )
        return

    if _adaptive_admits(arch, k, values):
        # The config takes the physical width, not a row's live length: seq_lens is
        # device-resident, so the length is not a host quantity. The kernel folds
        # each row onto as many of the launch's workgroups as its own length needs.
        cfg = _adaptive.decode_adaptive_config(
            rows,
            width,
            k,
            ordered=stable,
            cu_count=_adaptive_cu_count(logits.device.index),
        )
        kw = cfg["kw"]
        launcher = _adaptive.create_topk_per_row_decode_adaptive_kernel(top_k=k, **kw)
        workspace = _get_adaptive_workspace(
            logits.device,
            stream.cuda_stream,
            _adaptive.topk_workspace_slots(
                rows,
                11,
                compact=cfg["compact"],
                compact_cap=kw.get("compact_cap_mult", 16) * k,
            ),
        )
        if _adaptive.needs_workspace_zero(
            width,
            k,
            kw["tiered_short_max"],
            tier_mode=kw.get("tier_mode", "auto"),
            bits_per_pass=11,
        ):
            workspace.zero_()
        _run_compiled(
            launcher,
            logits,
            next_n,
            seq_lens,
            indices,
            workspace,
            rows,
            stride0,
            stride1,
            stream,
        )
        return

    # One row is `chunks` blocks, so the split has to follow the row count: at
    # one row, 16 chunks leaves all but a handful of CUs idle, and at many rows
    # it makes the single-block reduce walk counters nobody needed.
    chunks = topk_per_row_decode_chunks(rows, width, wave_size)
    hist_shape, state_shape = topk_per_row_decode_workspace_shapes(rows, stable, chunks)
    partial_hist, state = _get_topk_workspace(
        logits.device,
        stream.cuda_stream,
        hist_shape,
        state_shape,
    )

    launcher = build_topk_per_row_decode_module(
        k,
        stable,
        wave_size=wave_size,
        write_values=values is not None,
        chunks_per_row=chunks,
    )
    _run_compiled(
        launcher,
        logits,
        seq_lens,
        indices,
        values if values is not None else logits,
        partial_hist,
        state,
        width,
        next_n,
        stride0,
        rows,
        stream,
    )
