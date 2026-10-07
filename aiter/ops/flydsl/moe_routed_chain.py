# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the one-launch routed MoE chain (kernels/moe_routed_chain.py).

``routed_chain`` is a drop-in for

    aiter.biased_grouped_topk(logits, bias, tw, ti, 1, 1, True, 1.0)
    fused_moe(x, w1, w2, tw, ti, activation=Situv2, quant_type=per_1x32, ...)

on the a4w4 SiTUv2 path, for the shapes the fused kernel was built for. Other
shapes run exactly those two calls.
"""

import functools
import os

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA

from .kernels.moe_routed_chain import BM, compile_routed_chain, ctrl_words, max_m_blocks, ws_layout

FUSED_M_MAX = 32
# Above this M the two-kernel path wins (MI355X: M=32 144 vs 162 us); routed_chain still
# fuses up to FUSED_M_MAX when called directly.
FUSED_M_DISPATCH = 16
G1_BN = 128
# Wide gemm2 tiles amortise the per-tile wait and setup; 512 needs hidden % 512 == 0.
G2_BN = 512
# The kernel switches gemm1 to this tile width once routing yields many m-blocks.
G1_BN_WIDE = 256


# Shared expert inside the chain: K splits per gate/up pair, down tile width.
SH_KS = 4
SH_DN_BN = 64


@functools.cache
def _launcher(m_max, ne, topk, hidden, inter, g1_bn, beta, linear_beta, trace=False,
              route_only=False, g2_bn=G2_BN, sh=None):
    """sh: (hidden, inter, beta, linear_beta) of the shared expert, or None."""
    if hidden % g2_bn:
        g2_bn = 256
    g1_wide = G1_BN_WIDE if (2 * inter) % G1_BN_WIDE == 0 and G1_BN_WIDE != g1_bn else 0
    sh_kw = {}
    if sh is not None:
        sh_kw = dict(SH_HIDDEN=sh[0], SH_INTER=sh[1], SH_KS=SH_KS, SH_DN_BN=SH_DN_BN,
                     sh_beta=sh[2], sh_linear_beta=sh[3])
    return compile_routed_chain(
        M_MAX=m_max, NE=ne, TOPK=topk, D_HIDDEN=hidden, D_INTER=inter, G1_BN=g1_bn, G2_BN=g2_bn,
        G1_BN_WIDE=g1_wide,
        situ_beta=beta, situ_linear_beta=linear_beta, TRACE=trace, ROUTE_ONLY=route_only, **sh_kw,
    )


class _Workspace:
    """Per-device buffers sized for the largest fused M.

    The control words must start at zero; the kernel's last workgroup returns
    them to zero, so one zero fill at allocation covers every later launch.
    Sharing is safe while launches are ordered, as on one stream; callers that
    run the chain on two streams at once must pass each its own workspace.
    """

    def __init__(self, device, m_max, topk, inter, sh_inter=0):
        max_sorted = m_max * topk * BM
        offs, total = ws_layout(m_max, topk, inter, sh_inter, SH_KS)
        self.buf = torch.zeros(total, dtype=torch.uint8, device=device)

        def view(name, nbytes, dtype):
            return self.buf[offs[name]:offs[name] + nbytes].view(dtype)

        self.ctrl = view("ctrl", ctrl_words(m_max, topk, sh_inter) * 4, torch.int32)
        self.stids = view("stids", max_sorted * 4, torch.int32)
        self.sw = view("sw", max_sorted * 4, torch.float32)
        self.eids = view("eids", max_m_blocks(m_max, topk) * 4, torch.int32)
        self.cumsum = view("cumsum", 8, torch.int32)
        self.mind = view("mind", max_sorted * 4, torch.int32)
        self.inter = view("inter", max_sorted * (inter // 2), torch.uint8).view(max_sorted, inter // 2)
        self.inter_scale = self.buf[offs["inter_scale"]:]


_WORKSPACES = {}


def _workspace(device, m_max, topk, inter, sh_inter=0):
    key = (device, m_max, topk, inter, sh_inter)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _Workspace(device, m_max, topk, inter, sh_inter)
        _WORKSPACES[key] = ws
    return ws


def fused_supported(m, ne, topk, hidden, inter, m_max=FUSED_M_DISPATCH):
    """The kernel is the a4w4 path: fused_moe must be on it too (AITER_SITUV2_A4W4)."""
    return (
        1 <= m <= m_max
        and ne % 64 == 0
        and topk <= 64
        and inter % 128 == 0
        and os.environ.get("AITER_SITUV2_A4W4", "0") == "1"
        and get_gfx() == "gfx950"
    )


def _reference(logits, bias, x, w1, w2, w1_scale, w2_scale, topk, tw, ti, beta, linear_beta):
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    aiter.biased_grouped_topk(logits, bias, tw, ti, 1, 1, True, 1.0)
    return fused_moe(
        x, w1, w2, tw, ti,
        activation=ActivationType.Situv2, quant_type=QuantType.per_1x32,
        w1_scale=w1_scale, w2_scale=w2_scale, swiglu_limit=0.0,
        beta=beta, linear_beta=linear_beta, gate_mode=GateMode.SEPARATED.value,
    )


def shared_reference(x, w_gu, w_dn, beta, linear_beta, out=None):
    """Kimi-K3 bf16 shared expert as vLLM's KimiMLP runs it: gate_up GEMM to bf16,
    SiTU (no hard clamp; linear_beta None or <= 0 leaves up unclipped), down GEMM."""
    gu = (x @ w_gu.t()).float()
    d = gu.shape[-1] // 2
    gate = beta * torch.tanh(gu[:, :d] / beta) * torch.sigmoid(gu[:, :d])
    up = gu[:, d:]
    if linear_beta is not None and linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    res = (gate * up).to(x.dtype) @ w_dn.t()
    return res if out is None else out.copy_(res)


def shared_supported(m, sh_x, w_gu, w_dn):
    hidden, inter = w_dn.shape
    return (
        m <= BM
        and sh_x.dtype == w_gu.dtype == w_dn.dtype == torch.bfloat16
        and sh_x.is_contiguous() and w_gu.is_contiguous() and w_dn.is_contiguous()
        and tuple(sh_x.shape) == (m, hidden)
        and tuple(w_gu.shape) == (2 * inter, hidden)
        and inter % 32 == 0
        and hidden % (SH_KS * 4 * 32) == 0
        and hidden % SH_DN_BN == 0
    )


def routed_chain(
    logits,
    bias,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    out=None,
    *,
    topk=16,
    topk_weights=None,
    topk_ids=None,
    situ_beta=DEFAULT_SITUV2_BETA,
    situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    shared_x=None,
    shared_w_gu=None,
    shared_w_dn=None,
    shared_out=None,
    shared_beta=1.0,
    shared_linear_beta=None,
    stream=None,
    workspace=None,
    trace=None,
    wgs_per_cu=1,
    grid=None,
    _route_only=False,
):
    """With shared_x / shared_w_gu [2I, H] / shared_w_dn [H, I] the bf16 shared expert
    runs in the same launch into shared_out [M, H] (see shared_reference); returns
    (out, shared_out) then, else out."""
    m, hidden = x.shape
    ne = w1.shape[0]
    inter = w1.shape[1] // 2
    device = x.device
    with_shared = shared_x is not None
    if with_shared and shared_out is None:
        shared_out = torch.empty_like(shared_x)
    if topk_weights is None:
        topk_weights = torch.empty((m, topk), dtype=torch.float32, device=device)
    if topk_ids is None:
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device=device)
    fuse_shared = with_shared and shared_supported(m, shared_x, shared_w_gu, shared_w_dn)
    if with_shared and not fuse_shared:
        shared_reference(shared_x, shared_w_gu, shared_w_dn, shared_beta, shared_linear_beta,
                         shared_out)
    if not fused_supported(m, ne, topk, hidden, inter, FUSED_M_MAX):
        res = _reference(logits, bias, x, w1, w2, w1_scale, w2_scale, topk,
                         topk_weights, topk_ids, situ_beta, situ_linear_beta)
        res = res if out is None else out.copy_(res)
        return (res, shared_out) if with_shared else res

    if out is None:
        out = torch.empty((m, hidden), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream() if stream is None else stream
    m_max = FUSED_M_MAX
    sh = None
    sh_inter = 0
    if fuse_shared:
        sh_inter = shared_w_dn.shape[1]
        lb = shared_linear_beta
        sh = (shared_w_dn.shape[0], sh_inter, float(shared_beta),
              float(lb) if lb is not None and lb > 0 else 0.0)
    launch = _launcher(m_max, ne, topk, hidden, inter, G1_BN, float(situ_beta),
                       float(situ_linear_beta), trace is not None, _route_only, sh=sh)
    ws = workspace if workspace is not None else _workspace(device, m_max, topk, inter, sh_inter)
    meta = launch.chain_meta
    work = m + meta["SH_T"] + max_m_blocks(m, topk) * (meta["NNB1"] + meta["NNB2"])
    grid = min(work, get_cu_num() * wgs_per_cu) if grid is None else grid
    from .kernels.tensor_shim import _run_compiled

    _run_compiled(
        launch,
        logits.data_ptr(), bias.data_ptr(), x.data_ptr(),
        w1.data_ptr(), w1_scale.data_ptr(), w2.data_ptr(), w2_scale.data_ptr(),
        out.data_ptr(), topk_weights.data_ptr(), topk_ids.data_ptr(),
        ws.ctrl.data_ptr(), ws.stids.data_ptr(), ws.sw.data_ptr(), ws.eids.data_ptr(),
        ws.cumsum.data_ptr(), ws.mind.data_ptr(), ws.inter.data_ptr(), ws.inter_scale.data_ptr(),
        shared_x.data_ptr() if fuse_shared else 0,
        shared_w_gu.data_ptr() if fuse_shared else 0,
        shared_w_dn.data_ptr() if fuse_shared else 0,
        shared_out.data_ptr() if fuse_shared else 0,
        0 if trace is None else trace.data_ptr(), m, grid, stream,
    )
    return (out, shared_out) if with_shared else out
