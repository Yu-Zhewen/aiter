// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel-argument ABI for the MXFP4 paged MQA logits kernels. Byte-compatible with the gfx950
// op field for field, which is what the offsets below assert: the pointers are opaque, so the
// two targets share this struct and differ only in the layouts they read through it.
#pragma once

#include <cstdint>
#include <cstddef>

using bf16_t = __bf16;

// The relu must be the IEEE-754-2019 `maximum`, which PROPAGATES NaN, and not a
// compare-and-select, which returns 0 for it. An E8M0 scale of 0xFF is NaN, the indexer's KV
// scale pool carries such bytes at the first compressed row of essentially every sequence, and
// the engine's radix top-k ranks NaN above every finite value -- so propagating pins that
// column into the top-k and swallowing drops it. Worth ~10pp of DSpark speculative acceptance.
//
// REQUIRES `-fno-finite-math-only`: `-ffast-math` implies `-ffinite-math-only`, which lets the
// compiler assume no operand is NaN and fold the builtin back to the select. The umbrella
// header turns that into a build error rather than a silent regression.
#ifndef OPUS_LOGITS_IEEE_RELU
#define OPUS_LOGITS_IEEE_RELU 1
#endif

#if OPUS_LOGITS_IEEE_RELU
#define OPUS_LOGITS_RELU(x) __builtin_elementwise_maximum((x), 0.0f)
#else
#define OPUS_LOGITS_RELU(x) ((x) > 0.0f ? (x) : 0.0f)
#endif

namespace opus_logits {
// gfx950's two schedule-free modes are names only here; this target compiles `Table` and nothing
// else. Prefill gave every tile one CTA whatever its window length, which is the load imbalance
// the table exists to fix.
enum class mqa_logits_sched {
    Prefill,
    Decode,
    Table,
};
}

// One 32-byte record per CTA slot, and a CTA's whole assignment arrives in one
// `s_load_dwordx8` -- do not shrink it. `chunk_count == 0` marks a surplus slot.
//
// **THIS IS A TILE, WHERE gfx950's RECORD IS A ROW.** A gfx1250 CTA is Q_PER_BLOCK waves sharing
// a KV window, so a slot covers 1..Q_PER_BLOCK contiguous query rows and `group_rows` says how
// many; it takes one of gfx950's two pad dwords and the first six fields keep their meaning.
//
// `chunk_start` is ABSOLUTE -- a tile index into the sequence, not an offset from `local_start`.
// The store's out-of-range proof depends on that; see `do_store`'s bound in the kernel.
//
// `local_start` / `local_end` are the tile's UNION window and set the loop bound, which is the
// only thing they may set: the trip count fixes the barrier count, so anything per-row reaching
// it deadlocks the CTA. The STORE MASK is separate and per row -- each wave reads its own
// `[ptr_local_starts[row], ptr_local_ends[row])`.
struct opus_mqa_cta_record {
    int row_id;       // FIRST packed query row of the tile
    int batch_id;     // block_tables row
    int chunk_start;  // first KV tile (block_k units) this CTA covers, absolute
    int chunk_count;  // KV tiles it covers; 0 = surplus slot
    int local_start;  // the tile's UNION window start
    int local_end;    // the tile's UNION window end
    int group_rows;   // query rows in the tile, 1..Q_PER_BLOCK
    int _pad;
};
static_assert(sizeof(opus_mqa_cta_record) == 32,
              "the record must stay one s_load_dwordx8");
static_assert(alignof(opus_mqa_cta_record) == 4);

struct opus_mqa_logits_kargs {
    const void* __restrict__ ptr_q;         // [total_tokens, H, D/2]                fp4 (E2M1)
    const void* __restrict__ ptr_q_scale;   // [total_tokens, H, 4]                  e8m0
    const void* __restrict__ ptr_kv;        // [num_blocks, PAGE, D/2]               fp4 (E2M1)
    const void* __restrict__ ptr_kv_scale;  // [num_blocks, PAGE, 4]                 e8m0
    const int*  __restrict__ ptr_block_tables; // [batch, max_blocks_per_seq] int32
    const void* __restrict__ ptr_weights;   // [total_tokens, H] bf16
    float* __restrict__ ptr_out;             // [total_tokens, max_seq_len] fp32

    // PER-ROW window arrays, each [total_q] int32. The kernel reads them for its STORE MASK only
    // -- one entry per wave, its own row's -- while the loop bound comes from the record's
    // union. They are also the schedule builder's inputs.
    //
    // `ptr_local_starts` MAY BE NULL, meaning every row starts at 0, which every caller does
    // today. `ptr_local_ends` may not: it is the store's upper bound.
    const int* __restrict__ ptr_row_to_batch;   // builder only; the record carries batch_id
    const int* __restrict__ ptr_local_starts;
    const int* __restrict__ ptr_local_ends;
    // Read by neither the kernel nor the schedule. `ptr_cu_seq_q` is the BATCH prefix sum and is
    // an input to the tile cut, not to the launch.
    const int* __restrict__ ptr_cu_seq_q;
    int   split_kv;            // context splits per row (>= 1); unused here
    int   num_rows;            // total query rows
    int   num_batches;         // real batch count

    int   max_seq_len;
    int   stride_out_row;      // out row stride in elements (== max_seq_len for dense out)
    float weight_scale;
    int   block_k;             // KV tile size along seq_kv (== Traits::KV_TILE_SIZE)
    int   kv_block_size;       // paged block (page) size (== Traits::PAGE_SIZE)
    int   max_blocks_per_seq;  // block_tables row stride

    // APPENDED, not grouped with the pointers above: every field before this keeps the kernarg
    // offset it has on gfx950, which the offsetof block below enforces.
    const opus_mqa_cta_record* __restrict__ ptr_cta_info;  // [num_ctas]
    int   num_ctas;            // grid.x of the launch; slots past the work idle
};

// A field reordered or retyped on either target fails here instead of producing a launcher that
// reads the wrong dword.
static_assert(sizeof(opus_mqa_logits_kargs)  == 144);   // 128 + the appended pair
static_assert(alignof(opus_mqa_logits_kargs) == 8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q             ) ==   0);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q_scale       ) ==   8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv            ) ==  16);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv_scale      ) ==  24);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_block_tables  ) ==  32);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_weights       ) ==  40);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_out           ) ==  48);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_row_to_batch  ) ==  56);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_starts  ) ==  64);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_ends    ) ==  72);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_cu_seq_q      ) ==  80);
static_assert(offsetof(opus_mqa_logits_kargs, split_kv          ) ==  88);
static_assert(offsetof(opus_mqa_logits_kargs, num_rows          ) ==  92);
static_assert(offsetof(opus_mqa_logits_kargs, num_batches       ) ==  96);
static_assert(offsetof(opus_mqa_logits_kargs, max_seq_len       ) == 100);
static_assert(offsetof(opus_mqa_logits_kargs, stride_out_row    ) == 104);
static_assert(offsetof(opus_mqa_logits_kargs, weight_scale      ) == 108);
static_assert(offsetof(opus_mqa_logits_kargs, block_k           ) == 112);
static_assert(offsetof(opus_mqa_logits_kargs, kv_block_size     ) == 116);
static_assert(offsetof(opus_mqa_logits_kargs, max_blocks_per_seq) == 120);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_cta_info      ) == 128);
static_assert(offsetof(opus_mqa_logits_kargs, num_ctas          ) == 136);

__host__ __device__ inline int ceil_div_i(int a, int b) { return (a + b - 1) / b; }
