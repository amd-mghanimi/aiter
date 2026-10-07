# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Routed MoE chain in one launch: biased sigmoid top-k, sort, gemm1, gemm2.

Decode at small M runs top-k, sort, gemm1 and gemm2 as four launches whose
device time is a few microseconds each, so most of the chain is launch and
drain. This kernel runs the same work under one persistent grid.

Workgroups loop on a global ticket counter and the ticket picks the work item:

    0 .. M-1                        top-k of token t, zero its output row
    M .. M+G1-1                     gemm1 tile
    M+G1 ..                         gemm2 tile, m-block major

The workgroup that finishes the last top-k also sorts the routes and raises
the route flag. A gemm1 tile waits for that flag; a gemm2 tile also waits for
every gemm1 n-block of its m-block. Each wait names an item of a smaller
ticket, and a workgroup holding a ticket works on it until done, so the grid
cannot deadlock however many workgroups are resident. The last workgroup to
exit resets the control words for the next launch on the same workspace.

The sort gives an expert ceil(routes / BM) m-blocks, so there are at most
M * TOPK m-blocks, the size of the per-m-block counters.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from . import communication_ops_utils as comm_ops
from .mxfp4_gemm1 import (
    _bm_constants,
    _gemm1_body,
    default_epi_splits,
    default_k_stages,
)
from .mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr
from .mxmoe_dispatcher import compile_gemm2_a4w4_port

BM = 16
WAVE = 64
THREADS = 256
N_WAVES = THREADS // WAVE
LOG2E = 1.4426950408889634
SPIN_SLEEP = 8
ROUTE_MARKS = 16

# Control words (i32) at the head of the workspace.
CTRL_TICKET = 0
CTRL_ROUTED = 1
CTRL_DONE = 2
CTRL_TOPK = 3
CTRL_MBLOCK = 16
# CTRL_PAD layout: each control word on its own 128-byte line.
LINE_WORDS = 32


def max_m_blocks(m, topk):
    return m * topk


def ctrl_words(m_max, topk):
    return LINE_WORDS * (4 + max_m_blocks(m_max, topk))


def _lds_i32(base, off_words):
    return lds_typed_ptr(base, T.i32, byte_offset=fx.Int32(off_words * 4))


def _lds_f32(base, off_words):
    return lds_typed_ptr(base, T.f32, byte_offset=fx.Int32(off_words * 4))


def _now():
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _rcp(x):
    return fx.Float32(_llvm.call_intrinsic(T.f32, "llvm.amdgcn.rcp.f32", [fx.Float32(x).ir_value()], [], []))


@comm_ops.traced
def _spin_ge(addr_i64, val, sleep):
    cur = fx.Int32(comm_ops.load_i32_global_agent(addr_i64))
    while cur < fx.Int32(val):
        rocdl.s_sleep(sleep)
        cur = fx.Int32(comm_ops.load_i32_global_agent(addr_i64))
    return cur


@comm_ops.traced
def _mark(tid, arg_trace, t, k):
    if tid == fx.Int32(0):
        trace = global_typed_ptr(arg_trace, T.i64, align=8)
        trace[fx.Int32(ROUTE_MARKS) + t * fx.Int32(4) + fx.Int32(k)] = _now()


@comm_ops.traced
def _route_mark(tid, arg_trace, k):
    if tid == fx.Int32(0):
        trace = global_typed_ptr(arg_trace, T.i64, align=8)
        trace[fx.Int32(k)] = _now()


# Control-word updates branch on the wave-uniform `wave == 0`, never on
# `tid == 0`: LLVM tail-merges identical divergent blocks across the barriers
# that follow them, which deadlocks the workgroup. With one=False all 64 lanes
# issue the atomic (lane 0 adds 1, the rest 0); the memory side serialises
# those, ~2.5 us per update on gfx950. With one=True only lane 0 issues it
# (~0.4 us); each call site then needs its own LDS slot so no two divergent
# blocks are identical.
@comm_ops.traced
def _ctl_add(lane, addr, one):
    """Add one to a control word from wave 0."""
    if const_expr(one):
        if lane == fx.Int32(0):
            comm_ops.atomic_add_agent(addr, fx.Int32(1))
    else:
        comm_ops.atomic_add_agent(addr, (lane == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0)))


@comm_ops.traced
def _ctl_fetch_add(lane, addr, one, slot):
    """_ctl_add that leaves the old value in slot[0]."""
    if const_expr(one):
        if lane == fx.Int32(0):
            slot[0] = fx.Int32(comm_ops.atomic_add_agent(addr, fx.Int32(1)))
    else:
        old = comm_ops.atomic_add_agent(addr, (lane == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0)))
        slot[0] = rocdl.readfirstlane(T.i32, fx.Int32(old))


@comm_ops.traced
def _count(wave, lane, lds_base, addr, slot_word, one=False, release=True):
    """Release (fence, or drain write-through stores), add one; every thread gets the old value."""
    slot = _lds_i32(lds_base, slot_word)
    if const_expr(not release):
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if wave == fx.Int32(0):
        if const_expr(release):
            comm_ops.fence_agent_release()
        _ctl_fetch_add(lane, addr, one, slot)
    gpu.barrier()
    return rocdl.readfirstlane(T.i32, fx.Int32(slot[0]))


@comm_ops.traced
def _grab(wave, lane, lds_base, a_ticket, slot_word, one=False):
    """Next ticket for the whole workgroup."""
    slot = _lds_i32(lds_base, slot_word)
    gpu.barrier()
    if wave == fx.Int32(0):
        _ctl_fetch_add(lane, a_ticket, one, slot)
    gpu.barrier()
    return rocdl.readfirstlane(T.i32, fx.Int32(slot[0]))


@comm_ops.traced
def _bump(wave, lane, addr, one=False, release=True):
    """Release (fence, or drain write-through stores), then add one to a control word."""
    if const_expr(not release):
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if wave == fx.Int32(0):
        if const_expr(release):
            comm_ops.fence_agent_release()
        _ctl_add(lane, addr, one)


def _st_wt(base_i64, index, val, nbytes):
    """Device-scope store: writes through the XCD's L2, so no L2 writeback is needed."""
    _llvm.StoreOp(
        val.ir_value(), comm_ops._ptr_plus(base_i64, index, nbytes), alignment=nbytes,
        ordering=_llvm.AtomicOrdering.monotonic, syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def _store(wt, base_i64, ptr, index, val, nbytes):
    if const_expr(wt):
        _st_wt(base_i64, index, val, nbytes)
    else:
        ptr[index] = val


@comm_ops.traced
def _wait_ge(wave, addr, val, l1_only=False, sleep=SPIN_SLEEP):
    """Spin until a control word reaches val, then acquire.

    l1_only drops only this CU's L1. That is enough when everything the word
    guards was written through (_st_wt) before it was raised and nothing in
    this launch read those lines earlier: the dispatch already invalidated L2.
    """
    if wave == fx.Int32(0):
        _spin_ge(addr, val, sleep)
        if const_expr(l1_only):
            rocdl.s_waitcnt(vmcnt=0)
            _llvm.InlineAsmOp(None, [], "buffer_inv sc0", "", has_side_effects=True)
        else:
            comm_ops.fence_agent_acquire()
    gpu.barrier()


def _order_key(x):
    """i32 whose signed order is the order of the (non-NaN) f32 ``x``."""
    b = fx.Float32(x).bitcast(fx.Int32)
    return (b < fx.Int32(0)).select(b ^ fx.Int32(0x7FFFFFFF), b)


def _key_value(k):
    return ((k < fx.Int32(0)).select(k ^ fx.Int32(0x7FFFFFFF), k)).bitcast(fx.Float32)


def _wave_kth_key(key, k):
    """Largest t with at least k lanes of the wave holding key >= t, by bisection on ballots."""
    def n_ge(t):
        return fx.Int64(fmath.ctpop(fx.Int64(rocdl.ballot(T.i64, key >= t)).ir_value()))

    t = (n_ge(fx.Int32(0)) >= fx.Int64(k)).select(fx.Int32(0), fx.Int32(-(1 << 31)))
    for b in range_constexpr(30, -1, -1):
        c = t | fx.Int32(1 << b)
        t = (n_ge(c) >= fx.Int64(k)).select(c, t)
    return t


@comm_ops.traced
def _block_excl_scan(v, lane, wave, wtot):
    """Exclusive prefix and total of one i32 per thread across the workgroup."""
    incl = v
    for sh in range_constexpr(6):
        o = fx.Int32(gpu.shuffle_up(incl, 1 << sh, WAVE))
        incl = (lane >= fx.Int32(1 << sh)).select(incl + o, incl)
    if lane == fx.Int32(WAVE - 1):
        wtot[wave] = incl
    gpu.barrier()
    prefix = fx.Int32(0)
    total = fx.Int32(0)
    for wv in range_constexpr(N_WAVES):
        t = fx.Int32(wtot[fx.Int32(wv)])
        prefix = prefix + (fx.Int32(wv) < wave).select(t, fx.Int32(0))
        total = total + t
    return prefix + incl - v, total


@flyc.jit
def _topk_token(
    lds_base,
    arg_logits,
    arg_bias,
    arg_tw,
    arg_ti,
    arg_out,
    tok,
    tid,
    lane,
    wave,
    arg_trace,
    *,
    NE,
    TOPK,
    D_HIDDEN,
    TRACE,
    WT,
    PIVOT_BALLOT,
    L_LM,
    L_PIV,
    L_WTOT,
    L_CV,
    L_CI,
    L_CS,
    L_SEL,
):
    """Biased sigmoid top-k of one token over the whole workgroup.

    Matches aiter's topk_reg_kernel: score = sigmoid(logit), select on
    score + bias, ties to the lower expert id, weights = selected scores
    renormalized to sum 1.
    """
    EPW = NE // N_WAVES
    EPL = (EPW + WAVE - 1) // WAVE
    s_lm = _lds_f32(lds_base, L_LM)
    s_piv = _lds_f32(lds_base, L_PIV)
    wtot = _lds_i32(lds_base, L_WTOT)
    s_cv = _lds_f32(lds_base, L_CV)
    s_ci = _lds_i32(lds_base, L_CI)
    s_cs = _lds_f32(lds_base, L_CS)
    s_sel = _lds_f32(lds_base, L_SEL)

    def tmark(k):
        if const_expr(TRACE):
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            _route_mark(tid, arg_trace, k)

    tmark(2)
    out64 = global_typed_ptr(arg_out, T.i64, align=8)
    row64 = tok * fx.Int32(D_HIDDEN // 4)
    for i in range_constexpr((D_HIDDEN // 4 + THREADS - 1) // THREADS):
        if const_expr((i + 1) * THREADS <= D_HIDDEN // 4):
            _store(WT, arg_out, out64, row64 + tid + fx.Int32(i * THREADS), fx.Int64(0), 8)
        else:
            if tid + fx.Int32(i * THREADS) < fx.Int32(D_HIDDEN // 4):
                _store(WT, arg_out, out64, row64 + tid + fx.Int32(i * THREADS), fx.Int64(0), 8)

    logits = global_typed_ptr(arg_logits, T.f32)
    bias = global_typed_ptr(arg_bias, T.f32)
    neg_inf = fx.Float32(float("-inf"))
    row = tok * fx.Int32(NE)
    ids, sig, choice = [], [], []
    for j in range_constexpr(EPL):
        e = wave * fx.Int32(EPW) + fx.Int32(j * WAVE) + lane
        ok = fx.Int32(j * WAVE) + lane < fx.Int32(EPW)
        ec = ok.select(e, fx.Int32(0))
        x = fx.Float32(logits[row + ec])
        s = _rcp(fx.Float32(1.0) + (x * fx.Float32(-LOG2E)).exp2())
        ids.append(e)
        sig.append(s)
        choice.append(ok.select(s + fx.Float32(bias[ec]), neg_inf))
    tmark(3)

    # Pivot: the largest over waves of each wave's TOPK-th largest lane
    # maximum. At least TOPK elements reach it, so every winner does too.
    lmax = choice[0]
    for j in range_constexpr(1, EPL):
        lmax = lmax.maximumf(choice[j])
    if const_expr(PIVOT_BALLOT):
        wpiv = _key_value(_wave_kth_key(_order_key(lmax), TOPK))
        if lane == fx.Int32(0):
            s_piv[wave] = wpiv
    else:
        s_lm[tid] = lmax
        lrank = fx.Int32(0)
        for q in range_constexpr(WAVE):
            o = fx.Float32(s_lm[wave * fx.Int32(WAVE) + fx.Int32(q)])
            beats = (o > lmax) | ((o == lmax) & (fx.Int32(q) < lane))
            lrank = lrank + beats.select(fx.Int32(1), fx.Int32(0))
        if lrank == fx.Int32(TOPK - 1):
            s_piv[wave] = lmax
    gpu.barrier()
    pv = fx.Float32(s_piv[0])
    for wv in range_constexpr(1, N_WAVES):
        pv = pv.maximumf(fx.Float32(s_piv[fx.Int32(wv)]))

    tmark(4)
    hits = [choice[j] >= pv for j in range_constexpr(EPL)]
    c_l = fx.Int32(0)
    for j in range_constexpr(EPL):
        c_l = c_l + hits[j].select(fx.Int32(1), fx.Int32(0))
    pos, n_cand = _block_excl_scan(c_l, lane, wave, wtot)
    for j in range_constexpr(EPL):
        if hits[j]:
            s_cv[pos] = choice[j]
            s_ci[pos] = ids[j]
            s_cs[pos] = sig[j]
        pos = pos + hits[j].select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    tmark(5)

    ti = global_typed_ptr(arg_ti, T.i32)
    tw = global_typed_ptr(arg_tw, T.f32)
    for c in range(tid, n_cand, fx.Int32(THREADS)):
        mv = fx.Float32(s_cv[c])
        mi = fx.Int32(s_ci[c])
        rank = fx.Int32(0)
        for q in range(fx.Int32(0), n_cand, fx.Int32(1)):
            ov = fx.Float32(s_cv[q])
            oi = fx.Int32(s_ci[q])
            beats = (ov > mv) | ((ov == mv) & (oi < mi))
            rank = rank + beats.select(fx.Int32(1), fx.Int32(0))
        if rank < fx.Int32(TOPK):
            _store(WT, arg_ti, ti, tok * fx.Int32(TOPK) + rank, mi, 4)
            s_sel[rank] = fx.Float32(s_cs[c])
    gpu.barrier()
    tmark(6)
    if wave == fx.Int32(0):
        mine = lane < fx.Int32(TOPK)
        sv = mine.select(fx.Float32(s_sel[lane & fx.Int32(TOPK - 1)]), fx.Float32(0.0))
        tot = sv
        for sh in range_constexpr(6):
            tot = tot + tot.shuffle_xor(fx.Int32(1 << sh), fx.Int32(WAVE))
        if mine:
            _store(WT, arg_tw, tw, tok * fx.Int32(TOPK) + lane, sv / tot, 4)
    tmark(7)


@flyc.jit
def _sort_routes(
    lds_base,
    arg_tw,
    arg_ti,
    arg_stids,
    arg_sw,
    arg_eids,
    arg_cumsum,
    arg_mind,
    i32_M,
    tid,
    lane,
    wave,
    arg_trace,
    *,
    NE,
    TOPK,
    M_MAX,
    TRACE,
    WT,
    L_CNT,
    L_BLK,
    L_B2E,
    L_RPOS,
    L_WTOT,
):
    """Expert-sorted layout as moe_sort_quant's one-shot sort writes it.

    Experts ascend; each takes ceil(routes / BM) consecutive m-blocks and pads
    its last block with token M.
    """
    CNT_WORDS = (NE + THREADS - 1) // THREADS * THREADS
    EPT = CNT_WORDS // THREADS
    RPT = (M_MAX * TOPK + THREADS - 1) // THREADS
    cnt = _lds_i32(lds_base, L_CNT)
    blk = _lds_i32(lds_base, L_BLK)
    b2e = _lds_i32(lds_base, L_B2E)
    r_pos = _lds_i32(lds_base, L_RPOS)
    wtot = _lds_i32(lds_base, L_WTOT)

    def tmark(k):
        if const_expr(TRACE):
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            _route_mark(tid, arg_trace, k)

    for i in range_constexpr(EPT):
        cnt[tid + fx.Int32(i * THREADS)] = fx.Int32(0)
    gpu.barrier()

    n_routes = i32_M * fx.Int32(TOPK)
    ti = global_typed_ptr(arg_ti, T.i32)
    tw = global_typed_ptr(arg_tw, T.f32)
    lds_i64 = fx.Int64(lds_base)
    es, ws = [], []
    for k in range_constexpr(RPT):
        r = tid + fx.Int32(k * THREADS)
        has = r < n_routes
        rc = has.select(r, fx.Int32(0))
        e = fx.Int32(ti[rc])
        es.append(e)
        ws.append(fx.Float32(tw[rc]))
        if has:
            r_pos[r] = fx.Int32(
                comm_ops.atomic_add_lds(
                    lds_i64 + fx.Int64(L_CNT * 4) + fx.Int64(e) * fx.Int64(4), fx.Int32(1)
                )
            )
    gpu.barrier()
    tmark(10)

    counts = [fx.Int32(cnt[tid * fx.Int32(EPT) + fx.Int32(i)]) for i in range_constexpr(EPT)]
    nblks = [(c + fx.Int32(BM - 1)) // fx.Int32(BM) for c in counts]
    local = nblks[0]
    for i in range_constexpr(1, EPT):
        local = local + nblks[i]
    run, total = _block_excl_scan(local, lane, wave, wtot)
    eids = global_typed_ptr(arg_eids, T.i32)
    for i in range_constexpr(EPT):
        ex = tid * fx.Int32(EPT) + fx.Int32(i)
        blk[ex] = run
        for q in range(fx.Int32(0), nblks[i], fx.Int32(1)):
            b2e[run + q] = ex
            _store(WT, arg_eids, eids, run + q, ex, 4)
        run = run + nblks[i]
    if tid == fx.Int32(0):
        cs = global_typed_ptr(arg_cumsum, T.i32)
        _store(WT, arg_cumsum, cs, fx.Int32(0), total * fx.Int32(BM), 4)
        _store(WT, arg_cumsum, cs, fx.Int32(1), i32_M, 4)
    gpu.barrier()
    tmark(11)

    stids = global_typed_ptr(arg_stids, T.i32)
    sw = global_typed_ptr(arg_sw, T.f32)
    mind = global_typed_ptr(arg_mind, T.i32)
    for k in range_constexpr(RPT):
        r = tid + fx.Int32(k * THREADS)
        if r < n_routes:
            row = fx.Int32(blk[es[k]]) * fx.Int32(BM) + fx.Int32(r_pos[r])
            tok = r // fx.Int32(TOPK)
            slot = r - tok * fx.Int32(TOPK)
            _store(WT, arg_stids, stids, row, tok | (slot << fx.Int32(24)), 4)
            _store(WT, arg_mind, mind, row, tok, 4)
            _store(WT, arg_sw, sw, row, ws[k], 4)
    for rr in range(tid, total * fx.Int32(BM), fx.Int32(THREADS)):
        b = rr // fx.Int32(BM)
        ex = fx.Int32(b2e[b])
        p = rr - fx.Int32(blk[ex]) * fx.Int32(BM)
        if p >= fx.Int32(cnt[ex]):
            _store(WT, arg_stids, stids, rr, i32_M, 4)
            _store(WT, arg_mind, mind, rr, i32_M, 4)
            _store(WT, arg_sw, sw, rr, fx.Float32(0.0), 4)


def compile_routed_chain(
    *,
    M_MAX=16,
    NE=896,
    TOPK=16,
    D_HIDDEN=3584,
    D_INTER=384,
    G1_BN=256,
    G1_PREFETCH_HIDDEN=True,
    G2_BN=128,
    G2_BK=128,
    G2_USE_NT=False,
    situ_beta=4.0,
    situ_linear_beta=25.0,
    TRACE=False,
    ROUTE_ONLY=False,
    GEMM1_ONLY=False,
    EMPTY=False,
    WT_ROUTE=True,
    ONE_LANE=True,
    ACQ_ROUTE_L1=True,
    PIVOT_BALLOT=True,
    LDS_PAD=0,
    ACQ_MBLOCK_L1=True,
    WT_INTER=True,
    SPIN=SPIN_SLEEP,
    CTRL_PAD=True,
):
    """Compile the fused chain; returns the launcher.

    TRACE: arg_trace receives ROUTE_MARKS i64 for the sort, then four i64 per
    ticket (start, dependencies met, end); the unit is the 100 MHz
    s_memrealtime clock.
    """
    assert NE % N_WAVES == 0 and TOPK <= WAVE
    G1_BK = 256
    N_OUT1 = 2 * D_INTER
    NNB1 = N_OUT1 // G1_BN
    NNB2 = D_HIDDEN // G2_BN
    K_TILES1 = D_HIDDEN // G1_BK
    epi_splits = default_epi_splits(BM, G1_BN)
    k_stages = default_k_stages(BM, G1_BN, G1_BK // 2, K_TILES1, N_OUT1, 1, epi_splits)
    _, _, _, g1_lds_bytes = _bm_constants(BM, G1_BN, G1_BK // 2, K_TILES1, 1, epi_splits, k_stages)

    CNT_WORDS = (NE + THREADS - 1) // THREADS * THREADS
    L_CNT = 0
    L_BLK = L_CNT + CNT_WORDS
    L_B2E = L_BLK + CNT_WORDS
    L_RPOS = L_B2E + M_MAX * TOPK
    L_WTOT = L_RPOS + M_MAX * TOPK
    L_LM = L_WTOT + N_WAVES
    L_PIV = L_LM + THREADS
    L_CV = L_PIV + N_WAVES
    L_CI = L_CV + NE
    L_CS = L_CI + NE
    L_SEL = L_CS + NE
    L_TICKET = L_SEL + TOPK
    route_lds_words = L_TICKET + 8

    topk_kw = dict(
        NE=NE, TOPK=TOPK, D_HIDDEN=D_HIDDEN, TRACE=TRACE, WT=WT_ROUTE, PIVOT_BALLOT=PIVOT_BALLOT, L_LM=L_LM, L_PIV=L_PIV, L_WTOT=L_WTOT,
        L_CV=L_CV, L_CI=L_CI, L_CS=L_CS, L_SEL=L_SEL,
    )
    sort_kw = dict(NE=NE, TOPK=TOPK, M_MAX=M_MAX, TRACE=TRACE, WT=WT_ROUTE, L_CNT=L_CNT, L_BLK=L_BLK, L_B2E=L_B2E, L_RPOS=L_RPOS,
                   L_WTOT=L_WTOT)
    g1_kw = dict(
        BM=BM, BN=G1_BN, BK=G1_BK, inline_quant=True, prefetch_hidden=G1_PREFETCH_HIDDEN,
        a_dtype="fp4", out_dtype="fp4", act="situv2", situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta, swiglu_limit=7.0, enable_bias=False,
        K=D_HIDDEN, N_OUT=N_OUT1, NE=NE, interleave=False, native_scale_layout=True,
        num_waves=4, k_wave=1, epi_splits=epi_splits, k_stages=k_stages, wt_out=WT_INTER,
    )
    name = (
        f"moe_routed_chain_m{M_MAX}_ne{NE}_k{TOPK}_h{D_HIDDEN}_i{D_INTER}"
        f"_g1bn{G1_BN}{'_hpf' if G1_PREFETCH_HIDDEN else ''}_g2bn{G2_BN}{'_nt' if G2_USE_NT else ''}"
        f"{'_trace' if TRACE else ''}{'_routeonly' if ROUTE_ONLY else ''}"
    f"{'_g1only' if GEMM1_ONLY else ''}{'_empty' if EMPTY else ''}{'_wtr' if WT_ROUTE else ''}{'_1l' if ONE_LANE else ''}{'_aql1' if ACQ_ROUTE_L1 else ''}{'_pvb' if PIVOT_BALLOT else ''}{f'_pad{LDS_PAD}' if LDS_PAD else ''}{'_aqm1' if ACQ_MBLOCK_L1 else ''}{'_wti' if WT_INTER else ''}{f'_spin{SPIN}' if SPIN != SPIN_SLEEP else ''}{'_cpad' if CTRL_PAD else ''}"
    )

    # FlyDSL keys its compile cache on sources and scalar closure values; the
    # knob dicts reach the kernel only through calls, so key on them here.
    if CTRL_PAD:
        W_TICKET, W_ROUTED, W_DONE, W_TOPK = 0, LINE_WORDS, 2 * LINE_WORDS, 3 * LINE_WORDS
        W_MBLOCK, MB_STRIDE = 4 * LINE_WORDS, LINE_WORDS
    else:
        W_TICKET, W_ROUTED, W_DONE, W_TOPK, W_MBLOCK, MB_STRIDE = (
            CTRL_TICKET, CTRL_ROUTED, CTRL_DONE, CTRL_TOPK, CTRL_MBLOCK, 1)
    cache_tag = repr((name, sorted(topk_kw.items()), sorted(sort_kw.items()), sorted(g1_kw.items())))

    def compose(*, module_name, emit_gemm2_tile, shared_storage):
        @fx.struct
        class RouteStorage:
            raw: fx.Array[fx.Int32, route_lds_words, 16]

        @fx.struct
        class G1Storage:
            raw: fx.Array[fx.Uint8, g1_lds_bytes, 16]

        @fx.struct
        class PadStorage:
            raw: fx.Array[fx.Uint8, max(LDS_PAD, 16), 16]

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def routed_chain_kernel(
            arg_logits: fx.Int64,
            arg_bias: fx.Int64,
            arg_x: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1s: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2s: fx.Int64,
            arg_out: fx.Int64,
            arg_tw: fx.Int64,
            arg_ti: fx.Int64,
            arg_ctrl: fx.Int64,
            arg_stids: fx.Int64,
            arg_sw: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_mind: fx.Int64,
            arg_inter: fx.Int64,
            arg_inter_scale: fx.Int64,
            arg_trace: fx.Int64,
            i32_M: fx.Int32,
            i32_grid: fx.Int32,
        ):
            _ = cache_tag
            tid = fx.Int32(gpu.thread_id("x"))
            lane = tid % fx.Int32(WAVE)
            wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(WAVE))
            smem = fx.SharedAllocator()
            route_lds = smem.allocate(RouteStorage).peek().raw.ptr
            g1_lds = smem.allocate(G1Storage).peek().raw.ptr
            g2_lds = smem.allocate(shared_storage).peek()
            if const_expr(LDS_PAD):
                smem.allocate(PadStorage)
            lds_base = fx.Int32(ptrtoint(route_lds))

            ctrl = fx.Int64(arg_ctrl)
            a_ticket = ctrl + fx.Int64(W_TICKET * 4)
            a_routed = ctrl + fx.Int64(W_ROUTED * 4)
            a_done = ctrl + fx.Int64(W_DONE * 4)
            a_topk = ctrl + fx.Int64(W_TOPK * 4)
            a_mblock = ctrl + fx.Int64(W_MBLOCK * 4)

            def mark(t, k):
                if const_expr(TRACE):
                    _mark(tid, arg_trace, t, k)

            def rmark(k):
                if const_expr(TRACE):
                    _route_mark(tid, arg_trace, k)

            mb_max = i32_M * fx.Int32(TOPK)
            if const_expr(not EMPTY):
                t = _grab(wave, lane, lds_base, a_ticket, L_TICKET + 3, ONE_LANE)

                while t < i32_M:
                    mark(t, 0)
                    _topk_token(
                        lds_base, arg_logits, arg_bias, arg_tw, arg_ti, arg_out, t, tid, lane,
                        wave, arg_trace, **topk_kw,
                    )
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    rmark(8)
                    n_topk = _count(wave, lane, lds_base, a_topk, L_TICKET + 1, ONE_LANE,
                                    not WT_ROUTE)
                    rmark(9)
                    mark(t, 1)
                    if n_topk == i32_M - fx.Int32(1):
                        _wait_ge(wave, a_topk, i32_M, ACQ_ROUTE_L1 and WT_ROUTE, SPIN)
                        rmark(0)
                        _sort_routes(
                            lds_base, arg_tw, arg_ti, arg_stids, arg_sw, arg_eids, arg_cumsum,
                            arg_mind, i32_M, tid, lane, wave, arg_trace, **sort_kw,
                        )
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        rmark(12)
                        _bump(wave, lane, a_routed, ONE_LANE, not WT_ROUTE)
                        rmark(1)
                    mark(t, 2)
                    t = _grab(wave, lane, lds_base, a_ticket, L_TICKET + 4, ONE_LANE)

                _wait_ge(wave, a_routed, fx.Int32(1), ACQ_ROUTE_L1 and WT_ROUTE, SPIN)
                total_mb = fx.Int32(global_typed_ptr(arg_cumsum, T.i32)[0]) // fx.Int32(BM)
                n_g1 = total_mb * fx.Int32(NNB1)
                n_work = i32_M + n_g1 + total_mb * fx.Int32(NNB2)
                if const_expr(ROUTE_ONLY):
                    n_work = i32_M
                if const_expr(GEMM1_ONLY):
                    n_work = i32_M + n_g1

                if const_expr(not ROUTE_ONLY):
                    while t < n_work:
                        mark(t, 0)
                        wk = t - i32_M
                        if wk < n_g1:
                            mb1 = wk // fx.Int32(NNB1)
                            mark(t, 1)
                            _gemm1_body(
                                g1_lds, arg_x, arg_x, arg_w1, arg_w1s, arg_eids, arg_mind,
                                arg_inter, arg_inter_scale, arg_x, fx.Int64(0), wk, lane, wave,
                                True, i32_M, total_mb, **g1_kw,
                            )
                            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                            _bump(wave, lane, a_mblock + fx.Int64(mb1) * fx.Int64(4 * MB_STRIDE), ONE_LANE, not WT_INTER)
                            mark(t, 2)
                        if wk >= n_g1:
                            u = wk - n_g1
                            mb2 = u // fx.Int32(NNB2)
                            nb2 = u - mb2 * fx.Int32(NNB2)
                            _wait_ge(wave, a_mblock + fx.Int64(mb2) * fx.Int64(4 * MB_STRIDE), fx.Int32(NNB1), ACQ_MBLOCK_L1, SPIN)
                            mark(t, 1)
                            emit_gemm2_tile(
                                arg_inter, arg_inter_scale, arg_w2, arg_w2s, arg_eids, arg_stids,
                                arg_sw, arg_w2, arg_out, mb2, nb2, lane, wave, i32_M, mb_max,
                                fx.Int32(D_INTER), fx.Int32(D_HIDDEN), g2_lds,
                            )
                            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                            mark(t, 2)
                        t = _grab(wave, lane, lds_base, a_ticket, L_TICKET + 5, ONE_LANE)

            # Last workgroup out resets the control words for the next launch.
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            n_done = _count(wave, lane, lds_base, a_done, L_TICKET + 2, ONE_LANE)
            if n_done == i32_grid - fx.Int32(1):
                ctrl32 = global_typed_ptr(arg_ctrl, T.i32)
                for i in range(tid, mb_max, fx.Int32(THREADS)):
                    ctrl32[fx.Int32(W_MBLOCK) + i * fx.Int32(MB_STRIDE)] = fx.Int32(0)
                if tid == fx.Int32(0):
                    ctrl32[W_TICKET] = fx.Int32(0)
                    ctrl32[W_ROUTED] = fx.Int32(0)
                    ctrl32[W_DONE] = fx.Int32(0)
                    ctrl32[W_TOPK] = fx.Int32(0)

        @flyc.jit
        def launch_routed_chain(
            arg_logits: fx.Int64,
            arg_bias: fx.Int64,
            arg_x: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1s: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2s: fx.Int64,
            arg_out: fx.Int64,
            arg_tw: fx.Int64,
            arg_ti: fx.Int64,
            arg_ctrl: fx.Int64,
            arg_stids: fx.Int64,
            arg_sw: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_mind: fx.Int64,
            arg_inter: fx.Int64,
            arg_inter_scale: fx.Int64,
            arg_trace: fx.Int64,
            i32_M: fx.Int32,
            i32_grid: fx.Int32,
            stream: fx.Stream,
        ):
            routed_chain_kernel(
                arg_logits, arg_bias, arg_x, arg_w1, arg_w1s, arg_w2, arg_w2s, arg_out,
                arg_tw, arg_ti, arg_ctrl, arg_stids, arg_sw, arg_eids, arg_cumsum, arg_mind,
                arg_inter, arg_inter_scale, arg_trace, i32_M, i32_grid,
            ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(THREADS, 1, 1), stream=stream)

        return launch_routed_chain

    launch = compile_gemm2_a4w4_port(
        BM=BM, BN=G2_BN, BK=G2_BK, use_nt=G2_USE_NT, HIDDEN_MAX=8192, epilog="atomic",
        INTER_MAX=D_INTER, a_dtype="fp4", b_dtype="fp4", SBM=BM, g2_kstatic=True,
        _composition=compose,
    )
    launch.chain_meta = dict(NNB1=NNB1, NNB2=NNB2, M_MAX=M_MAX, TOPK=TOPK)
    return launch
