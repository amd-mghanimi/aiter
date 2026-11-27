# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""int8_blockwise_1x32 combine rows: a CPU reference of the Stage2 packing and the combine's decode, and the
config rules of the mode.

A row of k values travels as k offset-int8 bytes (u8 = q + 128, |q| <= 127) and then k / 32 E8M0 bytes, one
per 32-value group, in the fp8_blockwise_1x32 row layout. The reference follows the kernels' f32 arithmetic
step for step (mega_moe_stage2._fp8_scale_e8m0 with 1 / 127, _pack_offset_u8, and the combine's
(u8 - 128) * 2^(e - 127) decode), so it is bit-exact for them and is what a GPU test compares rows against.
"""

import math
from dataclasses import replace

import pytest
import torch

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    BLOCKWISE_P2P_QUANT_MODES,
    P2P_QUANT_FP8_BLOCKWISE,
    P2P_QUANT_I8_BLOCKWISE,
    SUPPORTED_P2P_QUANT_MODES,
    TOKEN_BUCKETS,
    build_mega_moe_bundle_plan,
    select_mega_moe_config,
)

GROUP = 32
# The f32 that Stage2 multiplies the group |max| by (0x3C010204, 1 / 127 rounded down).
_INV_QMAX = torch.tensor([0x3C010204], dtype=torch.int32).view(torch.float32)
# max|v| / scale can exceed 127 by one f32 ulp (fl(max * _INV_QMAX) rounds down to a power of two for
# max = nextafter(127 * 2^k)); that still encodes to q = 127.
_RATIO_BOUND = 127.0 + 2.0**-17
# Above this |max| the scale is 2^122 and the values round to q = 64: 64 * 2^122 decodes to Inf.
_MAX_FINITE_AMAX = 127.0 * 2.0**121


def _f32(bits):
    return bits.to(torch.int32).view(torch.float32)


def e8m0_scales(rows):
    """E8M0 scale per 32-value group of f32 ``rows`` [n, k]: the biased exponent of fl(max|v| * fl(1 / 127)),
    plus one unless that product is a power of two (so scale = 2^(e - 127) >= max|v| / 127); 0 for an all-zero
    group, 255 for a group holding Inf / NaN."""
    amax = rows.reshape(rows.shape[0], -1, GROUP).abs().amax(-1)
    w = (amax * _INV_QMAX).view(torch.int32)
    e = ((w >> 23) & 0xFF) + ((w & 0x7FFFFF) != 0).to(torch.int32)
    e = e.clamp(max=0xFF)
    return torch.where(amax.view(torch.int32) == 0, torch.zeros_like(e), e)


def encode_rows(rows):
    """Stage2's int8_blockwise_1x32 P2P rows [n, k + k / 32] (uint8) of f32 ``rows`` [n, k]."""
    n, k = rows.shape
    e = e8m0_scales(rows)
    # _inv_e8m0_scale: 2^(127 - e). v * 2^(127 - e) is exact, so the kernel's fma(v, inv, 128) rounds once,
    # as this product-then-add does.
    inv = ((254 - e) << 23).view(torch.float32)
    x = rows.reshape(n, -1, GROUP) * inv[..., None] + 128.0
    # x + 2^23 keeps round_half_even(x) in the low byte (x lies in [1, 255]).
    u8 = torch.round(x).to(torch.uint8).reshape(n, k)
    return torch.cat([u8, e.to(torch.uint8)], dim=1)


def decode_rows(p2p_rows, k):
    """The combine's f32 values of int8_blockwise_1x32 rows: (u8 - 128) * 2^(e - 127) (e = 0: 0)."""
    n = p2p_rows.shape[0]
    q = p2p_rows[:, :k].to(torch.int32) - 128
    scale = (p2p_rows[:, k:].to(torch.int32) << 23).view(torch.float32)
    return (q.to(torch.float32).reshape(n, -1, GROUP) * scale[..., None]).reshape(n, k)


def _scale(e):
    return torch.ldexp(torch.ones_like(e, dtype=torch.float64), e.to(torch.int64) - 127)


def _check_codec(rows):
    """Row layout, |q| <= 127, the smallest power-of-two scale and the round-trip bound for f32 ``rows``."""
    n, k = rows.shape
    p2p = encode_rows(rows)
    assert p2p.shape == (n, k + k // GROUP) and p2p.dtype == torch.uint8
    payload, e = p2p[:, :k], p2p[:, k:].to(torch.int64)
    assert int(payload.min()) >= 1 and int(payload.max()) <= 255, "|q| <= 127"
    amax = rows.reshape(n, -1, GROUP).abs().amax(-1).to(torch.float64)
    assert bool((amax <= _MAX_FINITE_AMAX).all())
    scale = _scale(e)
    assert bool((amax / scale <= _RATIO_BOUND).all())
    # Smallest such power of two: half of it would not hold the |max| (e <= 1: f32 underflow / subnormals).
    assert bool(((amax / (scale / 2) > 127) | (e <= 1)).all())
    # E8M0 0 (decoded as scale 0) only for zero groups and |max| < 127 * 2^-150, where fl(max / 127) is 0.
    assert bool((e[amax == 0] == 0).all()) and bool((amax[e == 0] < 2.0**-142).all())
    decoded = decode_rows(p2p, k).to(torch.float64)
    err = (decoded - rows.to(torch.float64)).abs().reshape(n, -1, GROUP)
    # Half a step, plus the f32 rounding of v / scale + 128 (2^-16 of a step) before the integer rounding.
    assert bool((err <= scale[..., None] * (0.5 + 2.0**-16)).all())
    return p2p


def test_codec_random_rows():
    gen = torch.Generator().manual_seed(0)
    k = 4096
    rows = torch.randn(64, k // GROUP, GROUP, generator=gen)
    # Heavy tails, sparse groups, and per-group magnitudes from 2^-100 to 2^100.
    rows[:8] = rows[:8] ** 3
    rows[8:16] *= torch.rand(8, k // GROUP, GROUP, generator=gen) < 0.1
    exponents = torch.randint(-100, 100, (64, k // GROUP, 1), generator=gen)
    rows = torch.ldexp(rows, exponents).reshape(64, k)
    p2p = _check_codec(rows)
    assert p2p.shape[1] == 4224, "a 4096-wide row is 4224 bytes, as fp8_blockwise_1x32"


def test_codec_scale_over_every_bf16_magnitude():
    """Every finite bf16 |max| (and the f32 values one to 64 ulps around 127 * 2^k) gets the smallest
    power-of-two scale that keeps |q| <= 127."""
    bf16_bits = torch.arange(0, 0x7F80, dtype=torch.int64) << 16
    amaxes = [_f32(bf16_bits)]
    for k in range(-126, 122):
        center = (
            torch.tensor([127.0 * 2.0**k], dtype=torch.float32)
            .view(torch.int32)
            .to(torch.int64)
        )
        amaxes.append(_f32(center + torch.arange(-64, 65)))
    amax = torch.cat(amaxes)
    amax = amax[amax <= _MAX_FINITE_AMAX]
    # Group g: its |max| in lane 0, smaller values of both signs in the others.
    groups = amax[:, None] * torch.linspace(1, 0, GROUP)[None, :]
    groups[:, 1::2] *= -1
    pad = (-groups.shape[0]) % 128
    rows = torch.cat([groups, torch.zeros(pad, GROUP)]).reshape(-1, 128 * GROUP)
    _check_codec(rows)


def test_codec_edge_values():
    k = 4 * GROUP
    one_ulp_over = _RATIO_BOUND
    rows = torch.zeros(1, k)
    rows[0, 0] = -0.0  # group 0 all zero
    rows[0, GROUP] = 127.0 * 2.0**-3  # group 1: 127 * 2^k exactly
    rows[0, GROUP + 1] = -1.0
    rows[0, 2 * GROUP] = one_ulp_over * 2.0**20  # group 2: one ulp over 127 * 2^k
    rows[0, 3 * GROUP] = 1000.0  # group 3: an outlier flushes the small values to 0
    rows[0, 3 * GROUP + 1 : 4 * GROUP] = 1e-3
    p2p = _check_codec(rows)
    payload, e = p2p[0, :k], p2p[0, k:]
    assert e.tolist() == [0, 124, 147, 130]
    assert bool((payload[:GROUP] == 128).all()) and bool(
        (decode_rows(p2p, k)[0, :GROUP] == 0).all()
    )
    assert payload[GROUP].item() == 255 and payload[GROUP + 1].item() == 128 - 8
    assert payload[2 * GROUP].item() == 255
    assert payload[3 * GROUP].item() == 128 + 125 and bool(
        (payload[3 * GROUP + 1 :] == 128).all()
    )
    decoded = decode_rows(p2p, k)[0]
    assert (
        decoded[GROUP].item() == 127.0 * 2.0**-3 and decoded[GROUP + 1].item() == -1.0
    )


def test_codec_extremes():
    """Subnormal groups quantize to 0; a |max| above 127 * 2^121 (the top 0.8% of the f32 range, bf16 max
    included) takes scale 2^122, rounds to q = 64 and decodes to Inf; Inf / NaN groups get E8M0 255 and have no
    finite encoding."""
    sub = torch.full((1, GROUP), 1e-40)
    p2p = _check_codec(sub)
    assert p2p[0, GROUP].item() == 1 and bool((decode_rows(p2p, GROUP) == 0).all())

    finite_top = torch.tensor([[_MAX_FINITE_AMAX] + [-1e38] + [0.0] * (GROUP - 2)])
    p2p = _check_codec(finite_top)
    assert p2p[0, GROUP].item() == 248 and p2p[0, 0].item() == 255
    assert decode_rows(p2p, GROUP)[0, 0].item() == _MAX_FINITE_AMAX
    bf16_max = torch.tensor([[torch.finfo(torch.bfloat16).max] + [0.0] * (GROUP - 1)])
    p2p = encode_rows(bf16_max)
    assert p2p[0, GROUP].item() == 249 and p2p[0, 0].item() == 128 + 64
    assert decode_rows(p2p, GROUP)[0, 0].item() == math.inf

    for bad in (math.inf, -math.inf, math.nan):
        group = torch.zeros(1, GROUP)
        group[0, 5] = bad
        assert e8m0_scales(group).item() == 255


def test_codec_groups_are_independent():
    """One E8M0 per 32 values: changing a group moves only its own payload bytes and scale byte."""
    gen = torch.Generator().manual_seed(1)
    k = 1024
    rows = torch.randn(4, k, generator=gen)
    base = encode_rows(rows)
    for g in (0, 7, k // GROUP - 1):
        moved = rows.clone()
        moved[:, g * GROUP : (g + 1) * GROUP] *= 1000.0
        changed = (encode_rows(moved) != base).any(0).nonzero().flatten().tolist()
        allowed = set(range(g * GROUP, (g + 1) * GROUP)) | {k + g}
        assert changed and set(changed) <= allowed, g


def test_codec_matches_direct_rounding_but_for_f32_ties():
    """q is round_half_even(v / scale) except where v / scale + 128 rounds to a tie in f32 first (the kernel
    adds the 128 offset before the integer rounding)."""
    gen = torch.Generator().manual_seed(2)
    rows = torch.randn(256, 4096, generator=gen) * 3
    p2p = encode_rows(rows)
    q = p2p[:, :4096].to(torch.float64) - 128
    scale = _scale(p2p[:, 4096:].to(torch.int64)).repeat_interleave(GROUP, 1)
    ratio = rows.to(torch.float64) / scale
    direct = torch.round(ratio)
    off = q != direct
    near_tie = (ratio - ratio.floor() - 0.5).abs() <= 2.0**-16
    assert bool((~off | near_tie).all())


_LAYOUTS = (
    # world_size, experts_per_rank, model_dim, inter_dim: Muse Spark, R1, V4-Pro, GLM-5.2, Kimi-K3, V4.1 EP4.
    (8, 32, 4096, 4096),
    (8, 32, 7168, 2048),
    (8, 48, 7168, 3072),
    (8, 32, 6144, 2048),
    (8, 112, 3584, 3072),
    (4, 96, 5120, 2304),
)


def _selections():
    for world_size, experts_per_rank, model_dim, inter_dim in _LAYOUTS:
        for a_dtype in ("fp4", "fp8"):
            for mtpr in (2**i for i in range(16)):
                tokens = sorted(
                    {
                        t
                        for b in TOKEN_BUCKETS
                        for t in (b - 1, b, b + 1, 3 * b // 4)
                        if 1 <= t <= mtpr
                    }
                )
                for t in tokens:
                    yield t, mtpr, {
                        "a_dtype": a_dtype,
                        "experts_per_rank": experts_per_rank,
                        "model_dim": model_dim,
                        "inter_dim": inter_dim,
                        "world_size": world_size,
                    }


def test_int8_is_explicit_and_the_fp8_config_on_the_plain_stage2():
    """ "auto" never selects INT8 rows; an explicit int8_blockwise_1x32 is the FP8 config with the
    aligned-pair Stage2 (FP8 rows only) turned off."""
    assert P2P_QUANT_I8_BLOCKWISE in SUPPORTED_P2P_QUANT_MODES
    assert BLOCKWISE_P2P_QUANT_MODES == (
        P2P_QUANT_FP8_BLOCKWISE,
        P2P_QUANT_I8_BLOCKWISE,
    )
    count = 0
    for tokens, mtpr, kw in _selections():
        assert (
            select_mega_moe_config(tokens, mtpr, **kw).p2p_quant
            != P2P_QUANT_I8_BLOCKWISE
        )
        fp8 = select_mega_moe_config(
            tokens, mtpr, p2p_quant=P2P_QUANT_FP8_BLOCKWISE, **kw
        )
        int8 = select_mega_moe_config(
            tokens, mtpr, p2p_quant=P2P_QUANT_I8_BLOCKWISE, **kw
        )
        assert (
            not int8.stage2.aligned_pair
            and int8.stage2.pair_cu == 0
            and not int8.stage2.bf16_lds
        )
        assert int8 == replace(
            fp8,
            p2p_quant=P2P_QUANT_I8_BLOCKWISE,
            stage2=replace(fp8.stage2, aligned_pair=False, pair_cu=0),
        ), (tokens, mtpr, kw)
        count += 1
    assert count > 5000


@pytest.mark.parametrize("mtpr", (1024, 2048, 8192, 32768))
def test_int8_bundle_plan(mtpr):
    plan = build_mega_moe_bundle_plan(
        mtpr,
        a_dtype="fp4",
        p2p_quant=P2P_QUANT_I8_BLOCKWISE,
        experts_per_rank=32,
        model_dim=4096,
        inter_dim=4096,
    )
    assert {entry.config.p2p_quant for entry in plan.entries} == {
        P2P_QUANT_I8_BLOCKWISE
    }
    assert not any(entry.config.stage2.aligned_pair for entry in plan.entries)


def test_int8_row_nbytes():
    """The Stage2 buffers size INT8 rows as FP8 rows (needs the image's flydsl)."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import stage2_p2p_row_nbytes

    for model_dim in (3584, 4096, 7168):
        nbytes = model_dim + model_dim // GROUP
        assert stage2_p2p_row_nbytes(model_dim, P2P_QUANT_I8_BLOCKWISE) == nbytes
        assert stage2_p2p_row_nbytes(model_dim, P2P_QUANT_FP8_BLOCKWISE) == nbytes
        assert encode_rows(torch.zeros(1, model_dim)).shape[1] == nbytes
