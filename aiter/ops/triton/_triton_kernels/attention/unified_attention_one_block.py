# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def reduce_segments_one_block(
    output_ptr,  # [num_tokens, num_query_heads, v_head_size]
    segm_output_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS, V_HEAD_SIZE_PADDED]
    segm_max_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS]
    seq_lens_ptr,  # [num_seqs]
    query_start_len_ptr,  # [num_seqs + 1]
    out_scale_ptr,  # float32, or None
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_QUERIES_PER_KV: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    MIN_SEGMENT_TILES: tl.constexpr,  # the producer's floor on tiles per segment
    V_HEAD_SIZE: tl.constexpr,
    V_HEAD_SIZE_PADDED: tl.constexpr,
    ROWS: tl.constexpr,  # (query token, query head) rows per program
    SEGMENT_BLOCK: tl.constexpr,  # segments merged per step, as one [ROWS, block, D] load
    # the producer stored the output of a sequence that fits one segment
    SKIP_SINGLE_SEGMENT: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # out * sigmoid(gate), rounded as reduce_segments' epilogue, or None
    gate_ptr=None,
    gate_stride_0=None,
):
    """reduce_segments for the gfx950 one-block split: one program per (sequence,
    kv head, ROWS of its query rows), the producer's grid, so the sequence comes
    from the program id and its live length decides, from one load, whether
    there is anything to merge.
    """
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    row_block = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = tl.maximum(
        (seq_len + NUM_SEGMENTS * TILE_SIZE - 1) // (NUM_SEGMENTS * TILE_SIZE),
        MIN_SEGMENT_TILES,
    )
    act_num_segments = (seq_len + tiles_per_segment * TILE_SIZE - 1) // (
        tiles_per_segment * TILE_SIZE
    )
    # constexpr gate first: a nested if keeps the load-free launch static
    if SKIP_SINGLE_SEGMENT:  # noqa: SIM102
        if act_num_segments <= 1:
            return
    q_start = tl.load(query_start_len_ptr + seq_idx)
    q_len = tl.load(query_start_len_ptr + seq_idx + 1) - q_start
    if row_block * ROWS >= q_len * NUM_QUERIES_PER_KV:
        return

    offs_r = row_block * ROWS + tl.arange(0, ROWS)
    query_pos = offs_r // NUM_QUERIES_PER_KV
    head_idx = kv_head_idx * NUM_QUERIES_PER_KV + offs_r % NUM_QUERIES_PER_KV
    row_mask = (query_pos < q_len) & (head_idx < NUM_QUERY_HEADS)
    token_idx = (q_start + query_pos).to(tl.int64)
    offs_s = tl.arange(0, NUM_SEGMENTS)
    offs_d = tl.arange(0, V_HEAD_SIZE_PADDED)
    d_mask = offs_d < V_HEAD_SIZE

    ml_base = token_idx * (NUM_QUERY_HEADS * NUM_SEGMENTS) + head_idx * NUM_SEGMENTS
    ml_mask = row_mask[:, None] & (offs_s < act_num_segments)[None, :]
    segm_max = tl.load(
        segm_max_ptr + ml_base[:, None] + offs_s[None, :],
        mask=ml_mask,
        other=float("-inf"),
    )
    overall_max = tl.max(segm_max, axis=1)
    # masked rows have no segment; keep their weights finite
    overall_max = tl.where(row_mask, overall_max, 0.0)
    weight = tl.math.exp2(segm_max - overall_max[:, None])
    segm_expsum = tl.load(
        segm_expsum_ptr + ml_base[:, None] + offs_s[None, :], mask=ml_mask, other=0.0
    )
    overall_expsum = tl.sum(segm_expsum * weight, axis=1)

    acc_base = token_idx * (
        NUM_QUERY_HEADS * NUM_SEGMENTS * V_HEAD_SIZE_PADDED
    ) + head_idx * (NUM_SEGMENTS * V_HEAD_SIZE_PADDED)
    if SEGMENT_BLOCK >= NUM_SEGMENTS:
        segm_output = tl.load(
            segm_output_ptr
            + acc_base[:, None, None]
            + offs_s[None, :, None] * V_HEAD_SIZE_PADDED
            + offs_d[None, None, :],
            mask=ml_mask[:, :, None] & d_mask[None, None, :],
            other=0.0,
        )
        acc_sum = tl.sum(segm_output * weight[:, :, None], axis=1)
    elif SEGMENT_BLOCK == 1:
        acc_sum = tl.zeros([ROWS, V_HEAD_SIZE_PADDED], dtype=tl.float32)
        for s in range(act_num_segments):
            w = tl.math.exp2(
                tl.load(segm_max_ptr + ml_base + s, mask=row_mask, other=float("-inf"))
                - overall_max
            )
            segm_output = tl.load(
                segm_output_ptr
                + acc_base[:, None]
                + s * V_HEAD_SIZE_PADDED
                + offs_d[None, :],
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            acc_sum += segm_output * w[:, None]
    else:
        acc_sum = tl.zeros([ROWS, V_HEAD_SIZE_PADDED], dtype=tl.float32)
        offs_b = tl.arange(0, SEGMENT_BLOCK)
        for s0 in range(0, act_num_segments, SEGMENT_BLOCK):
            b_mask = row_mask[:, None] & ((s0 + offs_b) < act_num_segments)[None, :]
            w = tl.math.exp2(
                tl.load(
                    segm_max_ptr + ml_base[:, None] + s0 + offs_b[None, :],
                    mask=b_mask,
                    other=float("-inf"),
                )
                - overall_max[:, None]
            )
            segm_output = tl.load(
                segm_output_ptr
                + acc_base[:, None, None]
                + (s0 + offs_b)[None, :, None] * V_HEAD_SIZE_PADDED
                + offs_d[None, None, :],
                mask=b_mask[:, :, None] & d_mask[None, None, :],
                other=0.0,
            )
            acc_sum += tl.sum(segm_output * w[:, :, None], axis=1)
    acc = tl.where(
        overall_expsum[:, None] == 0.0, 0.0, acc_sum / overall_expsum[:, None]
    )

    if out_scale_ptr is not None:
        acc = acc * (1 / tl.load(out_scale_ptr))
    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
    result = acc.to(output_ptr.type.element_ty)
    out_mask = row_mask[:, None] & d_mask[None, :]
    if gate_ptr is not None:
        gate = tl.load(
            gate_ptr
            + token_idx[:, None] * gate_stride_0
            + head_idx[:, None] * V_HEAD_SIZE
            + offs_d[None, :],
            mask=out_mask,
            other=0.0,
        )
        gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
        result = (result.to(tl.float32) * gate).to(output_ptr.type.element_ty)
    tl.store(
        output_ptr
        + token_idx[:, None] * output_stride_0
        + head_idx[:, None] * output_stride_1
        + offs_d[None, :],
        result,
        mask=out_mask,
    )
