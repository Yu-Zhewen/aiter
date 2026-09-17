# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits for DeepSeek-style sparse attention on gfx1250 (OPUS kernel).

Per query row ``r`` over a window ``[s, e)``:
``out[r, s:e] = sum_H( relu(Q[r] . K^T) * weight[r] ) * weight_scale``

PREFILL AND DECODE, through ONE launch over a per-tile schedule table built on device.

ALL FIVE INPUTS TAKE THEIR NATURAL LAYOUT, unlike the gfx950 sibling, which adopted FlyDSL's
preshuffled forms for three of them:

===============  ===========================================  ==================
tensor           shape                                        layout
===============  ===========================================  ==================
``q``            ``[total_q, H, D/2]`` uint8                   natural
``q_scale``      ``[total_q, H, 4]`` uint8                     natural
``kv_cache``     ``[num_blocks, PAGE, D/2]`` uint8             natural
``kv_scale``     ``[num_blocks, PAGE, 4]`` uint8               natural
``weights``      ``[total_q, H]`` bfloat16                     natural
===============  ===========================================  ==================

``q_scale[t, h, b]`` and ``kv_scale[blk, o, b]`` are the plain E8M0 byte for 32-element K block
``b`` of that row. ``q[t, h]`` and ``kv_cache[blk, o]`` are the 64 packed bytes of that row's
128 fp4 elements, low nibble first.

**Passing the gfx950 op's arrays here is SILENT.** Every fp4 scale layout has the same byte
count, so the C++ size checks cannot tell them apart and the result is plausible-looking wrong
logits. Validate against a dequantized reference on RANDOM data -- uniform data passes under
any permutation of K.

ONE CTA SERVES A SLOT: a TILE of up to ``Q_PER_BLOCK`` query rows that share the KV window, plus
a contiguous run of that tile's KV tiles. Sharing pays the HBM->LDS traffic once per tile instead
of once per row; SPLITTING is what lets a long window fill the part, and is worth **1.8x to
19.7x on decode** over one CTA per tile -- a decode forward has few tiles and each has a long
window, so without the split most of the GPU idles.

Three calls, and the split is not arbitrary. The first two depend only on per-FORWARD data while
the kernel runs per CSA LAYER, so a caller builds them once and launches 61 times::

    cu_tiles            = compute_tiles(cu_seq_q, total_q)          # once per forward
    cta_info, num_ctas  = compute_schedule(cu_tiles, local_ends...) # once per forward
    out                 = pa_mqa_logits_mxfp4_gfx1250(...)          # once per layer

All three are device-side, sync with nothing and read nothing back, so the path is
cudagraph-safe: the grid is a caller-held constant and the schedule absorbs the shape variation.

TWO CONDITIONS ON THE INPUT that the kernel cannot check and does not survive -- a CTA whose
waves disagree about the trip count DEADLOCKS on the phase barrier rather than returning a wrong
answer:

1. the window rule is NON-DECREASING in the row index within a tile, which every causal and
   CSA-compressed rule is, and :func:`assert_qshare_windows` checks on demand. The loop bound is
   the tile's UNION window; the store mask is each wave's OWN row, read per row from
   ``local_ends``;
2. the store is bounded by the WINDOW, so a ``local_ends`` entry past ``out.shape[1]`` writes
   past the row.

"A tile is contiguous rows of one batch" used to be a third and is now guaranteed by
:func:`compute_tiles`.

**WHY THE WINDOWS ARE PER ROW AND NOT PER TILE.** One window per tile would let the loop bound
and the store mask collapse, but it forces tiles to be cut on the CSA visibility runs --
``visible_csa(pos) = (pos+1)//4`` is constant on runs of four that sit at ``pos = 4k-1 .. 4k+2``,
one off the natural alignment. Measured, that makes 1.375x to 1.75x more tiles in MTP decode and
costs 24-45% of the runtime, because each extra tile re-reads the same KV for fewer rows.
"""

import torch

from ...jit.core import compile_ops
from ...jit.utils.chip_info import get_gfx_runtime

MD_NAME_MXFP4_GFX1250 = "module_pa_mqa_logits_mxfp4_gfx1250_opus"

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128
DEFAULT_KV_BLOCK_SIZE = 64

# Query rows per CTA == waves per CTA; the group size the builders and the kernel agree on.
Q_PER_BLOCK = 4

# The KV tile in tokens. Not an argument, unlike the gfx950 op's ``block_k``: this target
# compiles one variant. Exported because ``block_tables`` must be sized for it -- a CTA rounds
# its window up to a whole tile and indexes the table there.
BLOCK_K = 128


# ── JIT stubs: signatures must match PA_MQA_LOGITS_MXFP4_GFX1250_PYBIND exactly ───────────────
@compile_ops(MD_NAME_MXFP4_GFX1250, develop=True)
def pa_mqa_logits_mxfp4_gfx1250_fwd_sched(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    cta_info: torch.Tensor,
    out: torch.Tensor,
    num_rows: int,
    num_ctas: int,
    weight_scale: float,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4_GFX1250, develop=True)
def pa_mqa_logits_mxfp4_gfx1250_build_tiles(
    cu_seq_q: torch.Tensor,
    cu_tiles: torch.Tensor,
    total_q: int,
    max_tiles: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4_GFX1250, develop=True)
def pa_mqa_logits_mxfp4_gfx1250_build_sched(
    cu_tiles: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    row_to_batch: torch.Tensor,
    cta_info: torch.Tensor,
    num_tiles: int,
    num_ctas: int,
    cta_resident: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4_GFX1250, develop=True)
def pa_mqa_logits_mxfp4_gfx1250_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    total_q: int,
) -> None: ...


# Mirrors of the C++ constants. Open-coding `(num_ctas + 96) * 8` is one header change away from
# under-allocating, so these are the only supported way to size the buffers.
SCHED_CTA_RESIDENT = 256
SCHED_RECORD_INTS = 8
SCHED_SCRATCH_RECORDS = 96


def sched_slots(num_tiles: int, resident: int = SCHED_CTA_RESIDENT) -> int:
    """CTA slots to launch, i.e. the ``num_ctas`` GRID. For the BUFFER use
    :func:`sched_buffer_ints`.

    The tile count rounded up to a whole number of the part's resident rounds, one round as the
    floor. The floor is what lets a handful of long tiles spread over the GPU; the rounding is
    what leaves the split any room above it, since a split needs MORE slots than tiles.

    A surplus slot is not free -- its CTA reads a 32-byte record at an address no other CTA
    touches -- so this is deliberately tight rather than generous.
    """
    n = max(int(num_tiles), int(resident))
    return -(-n // int(resident)) * int(resident)


def sched_buffer_ints(num_ctas: int) -> int:
    """int32 elements a ``cta_info`` buffer needs for ``num_ctas`` slots.

    Slots plus the builder's own scratch, which sits past them in the same buffer. Sizing it at
    ``num_ctas * SCHED_RECORD_INTS`` instead is under-allocation; the launcher raises on it.
    """
    return (int(num_ctas) + SCHED_SCRATCH_RECORDS) * SCHED_RECORD_INTS


def max_tiles_for(total_q: int, batch: int) -> int:
    """Tiles the cut can produce, from the static shapes alone -- which is what keeps the launch
    cudagraph-safe.

    The real count is ``sum_b ceil(qlen_b / Q_PER_BLOCK)`` and depends on device data. This sums
    the per-batch roundings before the divide, so it is never short and is exact whenever they
    tile. The slack is at most ``batch - 1`` tiles, each of which gets an empty record.
    """
    return (int(total_q) + int(batch) * (Q_PER_BLOCK - 1)) // Q_PER_BLOCK


def compute_tiles(
    cu_seq_q: torch.Tensor,
    total_q: int,
    out: torch.Tensor | None = None,
):
    """Cut the query rows into tiles of at most ``Q_PER_BLOCK``, device-side.

    Returns ``(cu_tiles, num_tiles)``, where tile ``t`` covers rows
    ``[cu_tiles[t], cu_tiles[t + 1])``. ``num_tiles`` is :func:`max_tiles_for`, an upper bound
    rather than the exact count, so it stays a host int and no device read is needed to launch;
    tiles past the real count are written empty.

    This is what guarantees "a tile is contiguous rows of one batch", which the kernel cannot
    check and does not survive.
    """
    cu = cu_seq_q.to(torch.int32).contiguous()
    batch = int(cu.shape[0]) - 1
    num_tiles = max_tiles_for(total_q, batch)
    cu_tiles = (
        torch.empty(num_tiles + 1, dtype=torch.int32, device=cu.device)
        if out is None
        else out
    )
    pa_mqa_logits_mxfp4_gfx1250_build_tiles(cu, cu_tiles, int(total_q), int(num_tiles))
    return cu_tiles, num_tiles


def compute_schedule(
    cu_tiles: torch.Tensor,
    local_ends: torch.Tensor,
    num_tiles: int,
    *,
    local_starts: torch.Tensor | None = None,
    row_to_batch: torch.Tensor | None = None,
    num_ctas: int | None = None,
    cta_resident: int = SCHED_CTA_RESIDENT,
    cta_info: torch.Tensor | None = None,
):
    """Build the per-tile schedule. Device-side, cudagraph-safe, no sync.

    Call this ONCE PER FORWARD, not once per layer: it depends only on the windows, a per-forward
    quantity, while the kernel runs per CSA layer. A caller inside a CUDAGraph capture builds it
    in its metadata builder and hands the same buffer to the capture.

    ``local_starts`` is the per-ROW window start. Leave it ``None`` when every row starts at 0,
    which is what both ATOM paths do.

    ``row_to_batch`` is the per-ROW ``block_tables`` row. Leave it ``None`` when the table is
    indexed by query ROW rather than by sequence -- the per-token convention. Passing a
    per-SEQUENCE map while the launch gets a per-token table, or the reverse, reads the wrong
    pages and produces plausible wrong numbers.

    ``cta_resident`` is the CTAs the part holds and is what the split aims at; 0 turns the split
    off entirely, which is the A/B control for the whole schedule.

    Returns ``(cta_info, num_ctas)``; pass both to the launch. Safe to reuse the buffer across
    forwards -- every slot is written, surplus ones included.
    """
    n = int(num_tiles)
    slots = sched_slots(n) if num_ctas is None else int(num_ctas)
    if cta_info is None:
        cta_info = torch.empty(
            (slots + SCHED_SCRATCH_RECORDS, SCHED_RECORD_INTS),
            dtype=torch.int32,
            device=local_ends.device,
        )
    empty = torch.empty(0, dtype=torch.int32, device=local_ends.device)
    pa_mqa_logits_mxfp4_gfx1250_build_sched(
        cu_tiles,
        local_starts if local_starts is not None else empty,
        local_ends.to(torch.int32).contiguous(),
        row_to_batch if row_to_batch is not None else empty,
        cta_info,
        n,
        slots,
        int(cta_resident),
    )
    return cta_info, slots


def compute_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    total_q: int,
    out: tuple | None = None,
):
    """Build the per-row ``[local_start, local_end)`` window arrays, device-side.

    MTP tail-causal: batch ``b``'s ``n``-th row sees
    ``[0, context_lens[b] - (qlen - 1 - n))``, which reduces to plain causal when
    ``qlen == ctx``. That is the ONLY rule this expresses, and not the one a CSA-compressed
    cache follows -- row ``n`` there sees ``floor((pos + 1) / R)``, and
    ``floor((x - d) / R) != floor(x / R) - d``. Such a caller builds ``local_ends`` itself.
    """
    dev = cu_seq_q.device
    cu = cu_seq_q.to(torch.int32).contiguous()
    ctx = context_lens.to(torch.int32).contiguous()
    if out is None:
        row_to_batch = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_starts = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_ends = torch.empty(total_q, dtype=torch.int32, device=dev)
    else:
        row_to_batch, local_starts, local_ends = out
    pa_mqa_logits_mxfp4_gfx1250_prefill_windows(
        cu, ctx, row_to_batch, local_starts, local_ends, int(total_q)
    )
    return row_to_batch, local_starts, local_ends


def assert_qshare_windows(cu_tiles, num_tiles, local_starts, local_ends):
    """Check the one condition the schedule takes on faith: within a tile the window rule is
    NON-DECREASING, so the union is the first row's start and the LAST row's end.

    HOST-SIDE AND SYNCHRONISING -- a debug/test helper, not something for a hot path. It exists
    because breaking the condition DEADLOCKS the CTA rather than returning a wrong answer: the
    builder would take a union that is not one, and the waves would disagree about the trip
    count.

    "A tile is contiguous rows of one batch" is not checked here, because
    :func:`compute_tiles` is what produces the array and guarantees it.
    """
    ct = cu_tiles[: num_tiles + 1].tolist()
    ls = local_starts.tolist()
    le = local_ends.tolist()
    for t in range(num_tiles):
        lo, hi = ct[t], ct[t + 1]
        if hi == lo:
            continue  # an empty tile; its CTA gets a zero-count record
        if not (0 < hi - lo <= Q_PER_BLOCK):
            raise AssertionError(
                f"qshare: tile {t} spans rows [{lo},{hi}), which is not 1..{Q_PER_BLOCK} rows"
            )
        for r in range(lo + 1, hi):
            if ls[r] < ls[r - 1] or le[r] < le[r - 1]:
                raise AssertionError(
                    f"qshare: tile {t} window is not non-decreasing (row {r - 1}: "
                    f"[{ls[r - 1]},{le[r - 1]}) then row {r}: [{ls[r]},{le[r]}))"
                )


def _require_gfx1250(name):
    gfx = get_gfx_runtime()
    if gfx != "gfx1250":
        raise RuntimeError(f"{name} requires gfx1250, got {gfx}")


def pa_mqa_logits_mxfp4_gfx1250(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    local_ends: torch.Tensor,
    cta_info: torch.Tensor,
    num_ctas: int,
    max_seq_len: int,
    *,
    local_starts: torch.Tensor | None = None,
    weight_scale: float = 1.0,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged MQA logits over the schedule :func:`compute_schedule` produced -- prefill or decode,
    the table says which. The only launch entry point.

    See the module docstring for the input layouts -- in particular that all of them are NATURAL
    and that the gfx950 op's permuted scales are accepted silently -- and for the two conditions
    a caller owes.

    ``cta_info`` / ``num_ctas`` come from :func:`compute_schedule`, once per forward. ``local_ends``
    is still an argument even though the schedule was built from it, because the kernel reads it
    per row for the STORE MASK while the table carries only each tile's union. It must be the
    SAME array the schedule was built from; a shorter one is rejected, a differently-valued one
    is not.

    ``block_tables`` must be sized for ``BLOCK_K`` = 128, not for ``kv_block_size``: a CTA rounds
    its window up to a whole tile and indexes the table there. A reused ``out`` must be
    pre-filled with -inf, since the kernel only writes in-window cells.
    """
    _require_gfx1250("pa_mqa_logits_mxfp4_gfx1250")
    total_rows = int(q_fp4.shape[0])
    if out is None:
        out = torch.full(
            (total_rows, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    empty = torch.empty(0, dtype=torch.int32, device=q_fp4.device)
    pa_mqa_logits_mxfp4_gfx1250_fwd_sched(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        local_starts if local_starts is not None else empty,
        local_ends.to(torch.int32).contiguous(),
        cta_info,
        out,
        total_rows,
        int(num_ctas),
        float(weight_scale),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


__all__ = [
    "BLOCK_K",
    "Q_PER_BLOCK",
    "SCHED_CTA_RESIDENT",
    "assert_qshare_windows",
    "compute_prefill_windows",
    "compute_schedule",
    "compute_tiles",
    "max_tiles_for",
    "pa_mqa_logits_mxfp4_gfx1250",
    "pa_mqa_logits_mxfp4_gfx1250_build_sched",
    "pa_mqa_logits_mxfp4_gfx1250_build_tiles",
    "pa_mqa_logits_mxfp4_gfx1250_fwd_sched",
    "pa_mqa_logits_mxfp4_gfx1250_prefill_windows",
    "sched_buffer_ints",
    "sched_slots",
]
