# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 group GEMM1 (MXFP4 A and W, SiLU, FP4 out): the packed A-scale loader (ascale_packed) and the
8-column FP4 store (fp4_v8) are bit-exact rewrites, so each of the four (ascale_packed, fp4_v8) variants must
write exactly the bytes of the plain kernel, in the output and in the whole output-scale buffer.

One gfx950 GPU:

    pytest -q op_tests/test_mega_moe_gemm1_packed.py
"""

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility import dtypes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="gfx950 required"
)

LOCAL_EXPERTS = 6
# Few CUs, so every workgroup runs several (m, n) tiles back to back through the persistent loop.
NUM_CU = 32
SENTINEL = 0x5A
VARIANTS = {
    "FF": (False, False),
    "TF": (True, False),
    "FT": (False, True),
    "TT": (True, True),
}


def _weights(model_dim, inter_dim, device):
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    generator = torch.Generator(device=device).manual_seed(model_dim + inter_dim)
    w1 = torch.randn(
        (LOCAL_EXPERTS, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = aiter.get_torch_quant(aiter.QuantType.per_1x32)(
        w1, quant_dtype=dtypes.fp4x2
    )
    w1_q = w1_q.view(LOCAL_EXPERTS, 2 * inter_dim, model_dim // 2)
    w = shuffle_weight_a16w4(w1_q, 16, True).contiguous().view(torch.uint8)
    scale_w = (
        shuffle_scale_a16w4(w1_scale, LOCAL_EXPERTS, True)
        .contiguous()
        .view(torch.uint8)
    )
    return w, scale_w


_WEIGHTS = {}


def _cached_weights(model_dim, inter_dim, device):
    key = (model_dim, inter_dim)
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        _WEIGHTS[key] = _weights(model_dim, inter_dim, device)
    return _WEIGHTS[key]


def _routes(model_dim, sort_block_m, seed, device):
    """Expert-sorted MXFP4 A rows, each expert's rows padded to sort_block_m with zero rows (as dispatch
    leaves them): random counts with an empty expert, one row over a tile and an exact tile.
    """
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    generator = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 3 * sort_block_m, (LOCAL_EXPERTS,), generator=generator)
    counts[0], counts[1], counts[2] = 0, sort_block_m + 1, sort_block_m
    padded = [-(-int(c) // sort_block_m) * sort_block_m for c in counts.tolist()]
    num_valid = sum(padded)
    expert_ids = torch.cat(
        [
            torch.full((p // sort_block_m,), e, dtype=torch.int32)
            for e, p in enumerate(padded)
        ]
    )
    tile_row_base = (
        torch.arange(num_valid // sort_block_m, dtype=torch.int32) * sort_block_m
    )
    a = torch.randn(num_valid, model_dim, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    row = 0
    for c, p in zip(counts.tolist(), padded):
        a[row + c : row + p] = 0
        row += p
    aq, a_scale = dynamic_mxfp4_quant(a)
    x = aq.view(torch.uint8).contiguous()
    scale_x = a_scale.view(torch.uint8)[:num_valid, : model_dim // 32].contiguous()
    return x, scale_x, tile_row_base.to(device), expert_ids.to(device), num_valid


def _run(
    variant, x, scale_x, w, scale_w, tile_row_base, expert_ids, num_valid, **geometry
):
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.mega_moe.gemm1 import gemm1_kernel

    inter_dim = geometry["inter_dim"]
    ascale_packed, fp4_v8 = VARIANTS[variant]
    scale_cols = (inter_dim // 32 + 7) // 8 * 8
    out = torch.full(
        (num_valid, inter_dim // 2), SENTINEL, dtype=torch.uint8, device=x.device
    )
    out_scale = torch.full(
        (num_valid * scale_cols + 8192,), SENTINEL, dtype=torch.uint8, device=x.device
    )
    gemm1_kernel(
        out,
        x,
        w,
        scale_x,
        scale_w,
        tile_row_base,
        expert_ids,
        out_scale,
        num_valid,
        fx.Stream(torch.cuda.current_stream().cuda_stream),
        a_dtype="fp4",
        out_dtype="fp4",
        pipe_weights=True,
        mfma_amajor=True,
        async_a_copy=True,
        grid_mult=1,
        num_cu=NUM_CU,
        waves_per_eu_hint=1 if geometry["sort_block_m"] == 128 else 2,
        ascale_packed=ascale_packed,
        fp4_v8=fp4_v8,
        **geometry,
    )
    torch.cuda.synchronize()
    return out, out_scale


@pytest.mark.parametrize("num_waves", (4, 8))
@pytest.mark.parametrize("tile_n", (256, 512))
@pytest.mark.parametrize("sort_block_m", (64, 128))
@pytest.mark.parametrize("inter_dim", (2048, 4096))
@pytest.mark.parametrize("model_dim", (4096, 7168))
def test_gemm1_packed_variants_bit_exact(
    model_dim, inter_dim, sort_block_m, tile_n, num_waves
):
    device = torch.device("cuda", torch.cuda.current_device())
    w, scale_w = _cached_weights(model_dim, inter_dim, device)
    x, scale_x, tile_row_base, expert_ids, num_valid = _routes(
        model_dim,
        sort_block_m,
        seed=model_dim + inter_dim + sort_block_m + tile_n + num_waves,
        device=device,
    )
    geometry = {
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "sort_block_m": sort_block_m,
        "tile_n": tile_n,
        "num_waves": num_waves,
    }
    args = (x, scale_x, w, scale_w, tile_row_base, expert_ids, num_valid)
    ref_out, ref_scale = _run("FF", *args, **geometry)
    assert (ref_out != SENTINEL).any() and (ref_scale != SENTINEL).any()
    for variant in ("TF", "FT", "TT"):
        out, out_scale = _run(variant, *args, **geometry)
        assert torch.equal(
            out, ref_out
        ), f"{variant}: {(out != ref_out).sum().item()} output bytes differ"
        assert torch.equal(
            out_scale, ref_scale
        ), f"{variant}: {(out_scale != ref_scale).sum().item()} scale bytes differ"
