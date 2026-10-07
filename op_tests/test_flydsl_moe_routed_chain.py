# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the one-launch routed MoE chain (``routed_chain``).

The reference is the two calls the chain replaces on the a4w4 SiTUv2 decode
path: ``biased_grouped_topk`` then ``fused_moe``. Weights are laid out as
vLLM's AITER_MXFP4_BF16 loader leaves them (``[gate; up]`` rows,
``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4``).

Routing is drawn from a pool of experts per run (``--pool``): decode batches
touch far fewer distinct experts than N(0, 1) logits over every expert do, and
uniform routing overstates the GEMM share. ``--pool 0`` is uniform.

The chain is also replayed from a CUDA graph. It resets its own control words
at exit, so replaying it is a correctness check, not only a perf number.
"""

import argparse
import os

os.environ.setdefault("AITER_SITUV2_A4W4", "1")

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, get_torch_quant
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.moe_routed_chain import FUSED_M_MAX, fused_supported, routed_chain
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import checkAllclose, run_perftest

torch.set_default_device("cuda")

BETA = 4.0
LINEAR_BETA = 25.0


def make_weights(ne, hidden, inter, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    quant = get_torch_quant(QuantType.per_1x32)
    fp4 = torch.float4_e2m1fn_x2

    def make(n, k, gate_up):
        qs, ss = [], []
        for e0 in range(0, ne, 64):
            w = torch.randn((min(64, ne - e0), n, k), generator=g, dtype=torch.bfloat16) * 0.05
            q, s = quant(w, quant_dtype=dtypes.fp4x2)
            qs.append(q.view(w.shape[0], n, k // 2))
            ss.append(s.view(w.shape[0], n, k // 32))
        q = shuffle_weight_a16w4(torch.cat(qs).view(fp4), 16, gate_up)
        s = torch.cat(ss)
        s = shuffle_scale_a16w4(s.view(-1, s.shape[-1]), ne, gate_up)
        q.is_shuffled = True
        return q, s

    w1, w1s = make(2 * inter, hidden, True)
    w2, w2s = make(hidden, inter, False)
    return w1, w2, w1s, w2s


def make_inputs(m, ne, hidden, pool, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(m, ne, generator=g)
    if 0 < pool < ne:
        logits -= 8.0
        logits[:, torch.randperm(ne, generator=g, device="cuda")[:pool]] += 16.0
    x = torch.randn(m, hidden, generator=g, dtype=torch.bfloat16)
    bias = torch.randn(ne, generator=g) * 0.01
    return logits, x, bias


def reference(logits, bias, x, w1, w2, w1s, w2s, topk):
    m = x.shape[0]
    tw = torch.empty(m, topk, dtype=torch.float32)
    ti = torch.empty(m, topk, dtype=torch.int32)
    aiter.biased_grouped_topk(logits, bias, tw, ti, 1, 1, True, 1.0)
    out = fused_moe(
        x, w1, w2, tw, ti,
        quant_type=QuantType.per_1x32, activation=ActivationType.Situv2,
        w1_scale=w1s, w2_scale=w2s, gate_mode=GateMode.SEPARATED.value,
        swiglu_limit=0.0, beta=BETA, linear_beta=LINEAR_BETA,
    )
    return out, tw, ti


def fused(logits, bias, x, w1, w2, w1s, w2s, topk, tw, ti, out):
    return routed_chain(
        logits, bias, x, w1, w2, w1s, w2s, out, topk=topk, topk_weights=tw, topk_ids=ti,
        situ_beta=BETA, situ_linear_beta=LINEAR_BETA,
    )


def graph_runner(fn):
    """Capture ``fn`` into a CUDA graph; warm up first (JIT compile is illegal in capture)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def test_routed_chain(m, ne, topk, hidden, inter, pool, weights):
    w1, w2, w1s, w2s = weights
    logits, x, bias = make_inputs(m, ne, hidden, pool, seed=100 + m)
    ref, ref_w, ref_i = reference(logits, bias, x, w1, w2, w1s, w2s, topk)

    tw = torch.empty(m, topk, dtype=torch.float32)
    ti = torch.empty(m, topk, dtype=torch.int32)
    out = torch.empty(m, hidden, dtype=torch.bfloat16)
    args = (logits, bias, x, w1, w2, w1s, w2s, topk, tw, ti, out)
    fused(*args)
    torch.cuda.synchronize()

    ids_equal = torch.equal(torch.sort(ti, 1)[0], torch.sort(ref_i, 1)[0])
    w_err = (torch.sort(tw, 1)[0] - torch.sort(ref_w, 1)[0]).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=1).min().item()
    err = checkAllclose(ref, out, rtol=5e-2, atol=5e-2, msg=f"routed_chain M={m} pool={pool}")

    replay = graph_runner(lambda: fused(*args))
    out.fill_(7)
    replay()
    torch.cuda.synchronize()
    graph_cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=1).min().item()

    _, us_fused = run_perftest(replay, num_iters=100, num_warmup=10)
    ref_replay = graph_runner(lambda: reference(logits, bias, x, w1, w2, w1s, w2s, topk))
    _, us_ref = run_perftest(ref_replay, num_iters=100, num_warmup=10)
    assert ids_equal and w_err < 1e-6 and cos > 0.9999 and graph_cos > 0.9999, (m, pool)
    return dict(M=m, pool=pool, ids_equal=ids_equal, w_maxabs=w_err, cos_min=cos,
                graph_cos_min=graph_cos, allclose_err=err, us_ref=us_ref, us_fused=us_fused)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    p.add_argument("-m", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--pool", type=int, nargs="*", default=[64, 0])
    p.add_argument("-e", type=int, default=896, help="experts")
    p.add_argument("-k", type=int, default=16, help="top-k")
    p.add_argument("--hidden", type=int, default=3584)
    p.add_argument("--inter", type=int, default=384, help="intermediate size per partition")
    a = p.parse_args()
    if get_gfx() != "gfx950":
        print(f"skip: routed_chain needs gfx950, got {get_gfx()}")
        return
    weights = make_weights(a.e, a.hidden, a.inter)
    rows = []
    for pool in a.pool:
        for m in a.m:
            assert fused_supported(m, a.e, a.k, a.hidden, a.inter, FUSED_M_MAX), m
            rows.append(test_routed_chain(m, a.e, a.k, a.hidden, a.inter, pool, weights))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
