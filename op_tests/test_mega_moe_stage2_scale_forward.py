# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 Stage2 scale forwarding (scale_fwd_rows + run_stage2_scale_forward) writes exactly the peer bytes of
the direct scale stores, for FP8 and INT8 blockwise rows.

Stage2 runs on synthetic expert-sorted rows (each expert's rows padded to SBM, as compact dispatch leaves them)
whose routes scatter to 8 rank-local "peer" combine buffers. The peers start as 0xA5 everywhere, so a scale line
that is never written, or written where nothing should be, shows up; the scale stage starts as garbage and is
never cleared, so a forwarded line Stage2 did not restage in the same call (a stale line of an earlier route set)
shows up too.

One gfx950 GPU:

    pytest -q op_tests/test_mega_moe_stage2_scale_forward.py
    MEGA_MOE_SF_ITERS=1000 pytest -q op_tests/test_mega_moe_stage2_scale_forward.py -k stale
"""

import os

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.utility import dtypes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="gfx950 required"
)

# Muse Spark EP8 Stage2 (the forward pass expects 16 n-blocks of BN 256).
MODEL_DIM = 4096
INTER_DIM = 4096
LOCAL_EXPERTS = 32
TOPK = 8
NPES = 8
MAX_TOK = 2048
RECV_CAP = NPES * MAX_TOK
RANK = 3
SBM = 128
# Sorted-row capacity: every scenario's padded rows fit, and one compile serves them all.
ROW_CAP = 48 * 1024
PEER_FILL = 0xA5
STAGE_FILL = 0x3C
# The tuned Muse bucket's Stage2 (sbm128_deep).
STAGE2 = {
    "BM": 64,
    "SBM": SBM,
    "BN": 256,
    "BK": 256,
    "use_nt": False,
    "g2_bhoist": True,
    "g2_b2stage": True,
    "g2_ascale_pf": True,
    "g2_spart": 402,
    "g2_deep_a_pipeline": True,
    "persist": True,
    "persist_cu": 256,
    "persist_strided": True,
    "skew_cu": 96,
    "g2_bf16_lds": False,
    "scatter_vec": 8,
}
QUANTS = ("fp8_blockwise_1x32", "int8_blockwise_1x32")


def _route_counts(scenario, generator):
    """Routes per local expert."""
    if scenario == "empty":
        return torch.zeros(LOCAL_EXPERTS, dtype=torch.int64)
    if scenario == "one":
        counts = torch.zeros(LOCAL_EXPERTS, dtype=torch.int64)
        counts[LOCAL_EXPERTS - 1] = 1
        return counts
    if scenario == "single_expert":
        counts = torch.zeros(LOCAL_EXPERTS, dtype=torch.int64)
        counts[5] = 3001
        return counts
    if scenario in ("uniform", "ragged"):
        total = 16384 if scenario == "uniform" else 5003
        experts = torch.randint(0, LOCAL_EXPERTS, (total,), generator=generator)
        counts = torch.bincount(experts, minlength=LOCAL_EXPERTS)
        if scenario == "ragged":
            counts[-1] = SBM + 77
        return counts
    if scenario == "worst":
        # Zipf-like skew: the hottest expert holds about a third of 40000 routes.
        weight = 1.0 / torch.arange(1, LOCAL_EXPERTS + 1, dtype=torch.float64)
        experts = torch.multinomial(
            weight, 40000, replacement=True, generator=generator
        )
        counts = torch.bincount(experts, minlength=LOCAL_EXPERTS)
        return counts[torch.randperm(LOCAL_EXPERTS, generator=generator)]
    raise ValueError(scenario)


SCENARIOS = ("uniform", "worst", "single_expert", "one", "empty", "ragged")


class Routes:
    """Stage2 inputs of compact dispatch for one route set: distinct (token, slot) destinations on the 8 peers,
    sorted by local expert. ``ragged``: the last expert's padding rows lie past cumsum0, which is then not a
    multiple of SBM."""

    def __init__(self, scenario, seed, device):
        generator = torch.Generator().manual_seed(seed)
        counts = _route_counts(scenario, generator)
        total = int(counts.sum())
        dest = torch.randperm(RECV_CAP * TOPK, generator=generator)[:total]
        token, slot = dest // TOPK, dest % TOPK
        t = (token // MAX_TOK) << (MAX_TOK.bit_length() - 1) | (token % MAX_TOK)
        packed = slot << 24 | t
        padded = [-(-int(c) // SBM) * SBM for c in counts.tolist()]
        rows = sum(padded)
        assert rows <= ROW_CAP
        stids = torch.full((ROW_CAP,), RECV_CAP, dtype=torch.int64)
        row = off = 0
        for c, p in zip(counts.tolist(), padded):
            stids[row : row + c] = packed[off : off + c]
            row, off = row + p, off + c
        tiles = [p // SBM for p in padded]
        eids = torch.cat(
            [
                torch.full((n,), RANK * LOCAL_EXPERTS + e, dtype=torch.int32)
                for e, n in enumerate(tiles)
            ]
            + [
                torch.full(
                    (ROW_CAP // SBM - rows // SBM,),
                    RANK * LOCAL_EXPERTS,
                    dtype=torch.int32,
                )
            ]
        )
        # ragged: cumsum0 ends at the last expert's last route, inside its final sort block.
        cumsum0 = (
            rows - (padded[-1] - int(counts[-1])) if scenario == "ragged" else rows
        )
        assert scenario != "ragged" or cumsum0 % SBM
        self.scenario, self.routes, self.rows = scenario, total, rows
        self.stids = stids.to(torch.int32).to(device)
        self.eids = eids.to(device)
        self.trb = (torch.arange(ROW_CAP // SBM, dtype=torch.int32) * SBM).to(device)
        self.cumsum = torch.tensor([cumsum0], dtype=torch.int32, device=device)
        self.max_tiles = torch.tensor([max(tiles)], dtype=torch.int32, device=device)
        self.tile_end = (
            torch.tensor(tiles, dtype=torch.int32).cumsum(0).to(torch.int32).to(device)
        )
        self.wts = torch.ones(ROW_CAP, dtype=torch.float32, device=device)


class Harness:
    def __init__(self, device):
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        self.device = device
        generator = torch.Generator(device=device).manual_seed(7)
        w2 = torch.randn(
            (LOCAL_EXPERTS, MODEL_DIM, INTER_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w2.mul_(INTER_DIM**-0.25)
        w2_q, w2_scale = aiter.get_torch_quant(aiter.QuantType.per_1x32)(
            w2, quant_dtype=dtypes.fp4x2
        )
        del w2
        w2_q = w2_q.view(LOCAL_EXPERTS, MODEL_DIM, INTER_DIM // 2)
        self.w2 = shuffle_weight_a16w4(w2_q, 16, False).contiguous().view(torch.uint8)
        self.w2_scale = (
            shuffle_scale_a16w4(w2_scale, LOCAL_EXPERTS, False)
            .contiguous()
            .view(torch.uint8)
        )
        # FP4 A (every byte is a valid E2M1 pair) with E8M0 scales 2^-1 .. 2^0.
        self.aq = torch.randint(
            0,
            256,
            (ROW_CAP, INTER_DIM // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        self.ascale = torch.randint(
            126,
            128,
            (ROW_CAP * (INTER_DIM // 32) + 8192,),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        self.dummy = torch.zeros(4096, dtype=torch.int32, device=device)
        self.peers = None
        self.stage = torch.full(
            ((MODEL_DIM // 256) * ROW_CAP * 8,),
            STAGE_FILL,
            dtype=torch.uint8,
            device=device,
        )
        self.cu_num = get_cu_num()

    def bind(self, p2p_quant):
        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import (
            stage2_p2p_row_nbytes,
        )

        self.p2p_quant = p2p_quant
        self.row_nbytes = stage2_p2p_row_nbytes(MODEL_DIM, p2p_quant)
        self.comb_inp_nbytes = MAX_TOK * TOPK * self.row_nbytes
        self.peers = torch.empty(
            (NPES, self.comb_inp_nbytes), dtype=torch.uint8, device=self.device
        )
        self.table = torch.tensor(
            [int(self.peers[pe].data_ptr()) for pe in range(NPES)],
            dtype=torch.int64,
            device=self.device,
        )

    def run(self, routes, scale_forward, skew_normal_strided=False):
        """Fill the peers with PEER_FILL, run Stage2 (and the forward pass) and return the peer bytes."""
        import flydsl.expr as fx

        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import (
            run_mega_moe_stage2,
            run_stage2_scale_forward,
        )

        D = fx.Int64
        self.peers.fill_(PEER_FILL)
        stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        common = {
            "model_dim": MODEL_DIM,
            "topk": TOPK,
            "npes": NPES,
            "max_tok": MAX_TOK,
            "recv_cap": RECV_CAP,
            "comb_inp_nbytes": self.comb_inp_nbytes,
            "p2p_quant_type": self.p2p_quant,
        }
        run_mega_moe_stage2(
            D(self.aq.data_ptr()),
            D(self.ascale.data_ptr()),
            D(self.w2.data_ptr()),
            D(self.w2_scale.data_ptr()),
            D(routes.eids.data_ptr()),
            D(routes.cumsum.data_ptr()),
            D(routes.max_tiles.data_ptr()),
            D(routes.stids.data_ptr()),
            D(routes.wts.data_ptr()),
            D(routes.trb.data_ptr()),
            D(routes.tile_end.data_ptr()),
            D(self.dummy.data_ptr()),
            D(self.dummy.data_ptr()),
            D(self.dummy.data_ptr()),
            D(self.table.data_ptr()),
            ROW_CAP,
            INTER_DIM,
            MODEL_DIM,
            stream,
            inter_dim=INTER_DIM,
            experts=LOCAL_EXPERTS,
            rank=RANK,
            HIDDEN_MAX=MODEL_DIM,
            INTER_MAX=INTER_DIM,
            cu_num=self.cu_num,
            a_dtype="fp4",
            scale_fwd_rows=ROW_CAP if scale_forward else 0,
            arg_scale_stage=self.stage.data_ptr() if scale_forward else 0,
            skew_normal_strided=skew_normal_strided,
            **STAGE2,
            **common,
        )
        if scale_forward:
            run_stage2_scale_forward(
                routes.cumsum.data_ptr(),
                routes.stids.data_ptr(),
                routes.trb.data_ptr(),
                self.stage.data_ptr(),
                D(self.table.data_ptr()),
                stream,
                BN=STAGE2["BN"],
                SBM=SBM,
                scale_fwd_rows=ROW_CAP,
                **common,
            )
        torch.cuda.synchronize()
        return self.peers


@pytest.fixture(scope="module")
def harness():
    return Harness(torch.device("cuda", torch.cuda.current_device()))


@pytest.fixture(scope="module")
def route_sets():
    device = torch.device("cuda", torch.cuda.current_device())
    return {s: Routes(s, seed=i, device=device) for i, s in enumerate(SCENARIOS)}


def _mismatch(peers, ref, row_nbytes):
    diff = (peers != ref).view(NPES, -1, row_nbytes)
    rows = diff.any(-1).nonzero()
    first = rows[0].tolist() if len(rows) else None
    in_scale_line = bool(diff[..., MODEL_DIM:].any())
    return (
        f"{len(rows)} rows differ (first peer/row {first}; scale line: {in_scale_line})"
    )


@pytest.mark.parametrize("skew_normal_strided", (False, True))
@pytest.mark.parametrize("p2p_quant", QUANTS)
def test_scale_forward_bit_exact(harness, route_sets, p2p_quant, skew_normal_strided):
    harness.bind(p2p_quant)
    for scenario, routes in route_sets.items():
        ref = harness.run(routes, scale_forward=False).clone()
        touched = int(
            (ref != PEER_FILL).view(NPES, -1, harness.row_nbytes).any(-1).sum()
        )
        assert (touched > 0) == (routes.routes > 0), scenario
        peers = harness.run(
            routes, scale_forward=True, skew_normal_strided=skew_normal_strided
        )
        assert torch.equal(peers, ref), f"{scenario}: " + _mismatch(
            peers, ref, harness.row_nbytes
        )


@pytest.mark.parametrize("p2p_quant", QUANTS)
def test_scale_forward_stale_stage(harness, route_sets, p2p_quant):
    """Cycle the route sets without clearing the stage: every call forwards only lines it staged itself."""
    iters = int(os.environ.get("MEGA_MOE_SF_ITERS", "200"))
    harness.bind(p2p_quant)
    order = list(route_sets)
    refs = {s: harness.run(route_sets[s], scale_forward=False).clone() for s in order}
    for i in range(iters):
        scenario = order[i % len(order)]
        peers = harness.run(route_sets[scenario], scale_forward=True)
        assert torch.equal(peers, refs[scenario]), f"iter {i} {scenario}: " + _mismatch(
            peers, refs[scenario], harness.row_nbytes
        )
