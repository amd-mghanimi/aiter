# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
r"""Stress tests of Stage1's acquire_once (one system-scope acquire per GEMM tile claim).

With ``Stage1Config.acquire_once`` the thread that observed a tile's ready flag runs the
claim's only ``fence_system_acquire`` before the CTA barrier; the other 511 threads no
longer acquire after it. That relies on the barrier's workgroup fences and on every wave
of the CTA reading through the CU's vector L1 and the XCD's L2 that the one
``buffer_inv sc0 sc1`` invalidated (no tgsplit).

--suite litmus (2+ ranks for the cross-device mode, gfx950): the claim pattern in a small
FlyDSL kernel over mori symmetric-heap buffers (the memory MegaMoEV2's payload lives in).
Per epoch and consumer CTA (512 threads): every thread reads its part of a buffer (warming
L1/L2 with the previous contents), signals the producer, tid 0 spins on a monotonic load of
the ready counter; the producer CTA (on the partner GPU, or on another XCD of the same GPU)
writes ``f(epoch, i)`` into the buffer, releases and RMWs the counter; then every consumer
thread checks its words against ``f(epoch, i)``. Consumer variants after the spin:

  once  tid 0 acquires, then the barrier             (acquire_once)       must be clean
  all   barrier, then every thread acquires          (the previous code)  must be clean
  none  barrier only                                 negative control
  wg    barrier, then a workgroup-scope acquire      negative control (orders the loads in
        on every thread (no cache invalidate)        the compiler, so a hit is a stale cache line)

``once`` and ``all`` must show no mismatch. A negative control with mismatches shows that the
litmus can see a stale read, so a clean ``once`` means something. A clean negative control
means the litmus cannot tell on this machine and memory type, and the result says so
(--require-negative fails the run then). Two ways it can be clean: a remote XGMI write
that invalidates or bypasses the consumer XCD's L2 line, with a warmed L1 line evicted
before the check; or symmetric-heap memory mapped uncached, so every load goes past L1/L2.
The buffer size list (--litmus-words) has one size that fits the 32 KB L1 and one of 128 KB.

--suite insitu (8 ranks, Muse Spark A4W4 at 2048 tokens a rank): --insitu-rounds full
MegaMoEV2 forwards of the tuned instance (acquire_once on) against an instance whose plans
differ only in acquire_once=False, every output compared bitwise. Routes (those of
test_mega_moe_v2.py --suite dispatch) cycle uniform / rank-local / Zipf / single-hot-expert /
empty-destination on fresh inputs every round, so each payload parity buffer holds new data
on every reuse, and every other round has uneven per-rank tokens. Uniform routes take the
sharded claim order, skewed ones the arrival
order; both carry an acquire_once hunk. A mismatched row that equals the row two rounds
back (the same payload parity) is counted as stale.

An in-situ negative control (a Stage1 build without tid 0's acquire) is not here: it would
need a test-only Stage1 knob, and patching the fence at trace time leaves FlyDSL's
compile-cache key unchanged, so that build could be served to the normal instance later.

    torchrun --nproc-per-node 2 op_tests/multigpu_tests/test_mega_moe_acquire_once.py --suite litmus
    torchrun --nproc-per-node 8 op_tests/multigpu_tests/test_mega_moe_acquire_once.py --suite insitu
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import faulthandler
import functools
from dataclasses import replace

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
import torch.distributed as dist
from flydsl.expr import const_expr, range_constexpr
from mori.shmem import mori_shmem_create_tensor
from test_mega_moe_v2 import _cleanup, _fwd, _quantize_weights, _setup_dist

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.communication_ops_host_utils import build_p2p_table
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_buf_tensor

# ---- litmus -------------------------------------------------------------------

LITMUS_THREADS = 512  # Stage1's CTA size
LITMUS_VARIANTS = ("once", "all", "none", "wg")
LITMUS_CLEAN = ("once", "all")
# f(epoch, i) = epoch * 2^16 + i, so i < 2^16 words a consumer.
LITMUS_MAX_WORDS = 1 << 16


@functools.cache
def _litmus_launcher(variant, words, consumers, same_device):
    """Producer / consumer CTAs of one litmus variant.

    same_device: one launch, block 2c consumes and block 2c + 1 produces (consecutive
    blocks are placed on different XCDs); otherwise each rank launches ``consumers``
    blocks of one role (``role`` 1 = producer) against its partner's buffers.
    """
    assert variant in LITMUS_VARIANTS
    assert words % LITMUS_THREADS == 0 and words <= LITMUS_MAX_WORDS
    per_thread = words // LITMUS_THREADS
    variant_id = LITMUS_VARIANTS.index(variant)
    grid_x = 2 * consumers if same_device else consumers
    name = f"mmoe_acquire_litmus_{variant}_w{words}_c{consumers}_{'xcd' if same_device else 'xdev'}"

    @flyc.kernel(name=name, known_block_size=[LITMUS_THREADS, 1, 1])
    def kernel(
        addr_buf: fx.Int64,
        addr_buf_peer: fx.Int64,
        addr_ready: fx.Int64,
        addr_ready_peer: fx.Int64,
        addr_warmed: fx.Int64,
        addr_warmed_peer: fx.Int64,
        addr_bad: fx.Int64,
        addr_sink: fx.Int64,
        addr_xcc: fx.Int64,
        first_epoch: fx.Int32,
        last_epoch: fx.Int32,
        role: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        if const_expr(same_device):
            producer = (bid & fx.Int32(1)) == fx.Int32(1)
            pair = bid >> fx.Int32(1)
        else:
            producer = role == fx.Int32(1)
            pair = bid
        buf = ptr_buf_tensor(addr_buf, fx.Int32)
        buf_peer = ptr_buf_tensor(addr_buf_peer, fx.Int32)
        sink = ptr_buf_tensor(addr_sink, fx.Int32)
        if tid == fx.Int32(0):
            ptr_buf_tensor(addr_xcc, fx.Int32)[bid] = fx.Int32(comm_ops.hw_xcc_id())
        for epoch in range(first_epoch, last_epoch + fx.Int32(1), 1):
            base = ((epoch & fx.Int32(1)) * fx.Int32(consumers) + pair) * fx.Int32(
                words
            )
            if producer:
                # Write only after the consumer warmed its caches with the old contents.
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_greater_than(
                        addr_warmed + fx.Int64(pair) * 4, epoch - fx.Int32(1)
                    )
                fx.barrier()
                for j in range_constexpr(per_thread):
                    word = tid + fx.Int32(j * LITMUS_THREADS)
                    buf_peer[base + word] = epoch * fx.Int32(1 << 16) + word
                comm_ops.fence_system_release()
                fx.barrier()
                if tid == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.atomic_add_system(
                        addr_ready_peer + fx.Int64(pair) * 4, fx.Int32(1)
                    )
            else:
                warm = fx.Int32(0)
                for j in range_constexpr(per_thread):
                    warm = warm + buf[base + tid + fx.Int32(j * LITMUS_THREADS)]
                sink[pair * fx.Int32(LITMUS_THREADS) + tid] = warm
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(addr_warmed_peer, pair, epoch)
                    comm_ops.wait_i32_until_greater_than(
                        addr_ready + fx.Int64(pair) * 4, epoch - fx.Int32(1)
                    )
                    if const_expr(variant_id == 0):
                        comm_ops.fence_system_acquire()
                fx.barrier()
                if const_expr(variant_id == 1):
                    comm_ops.fence_system_acquire()
                if const_expr(variant_id == 3):
                    comm_ops.fence_acquire(fx.rocdl.SyncScope.WorkgroupOneAs)
                wrong = fx.Int32(0)
                stale = fx.Int32(0)
                for j in range_constexpr(per_thread):
                    word = tid + fx.Int32(j * LITMUS_THREADS)
                    value = buf[base + word]
                    wrong = wrong + (value != epoch * fx.Int32(1 << 16) + word).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                    stale = stale + (
                        value == (epoch - fx.Int32(2)) * fx.Int32(1 << 16) + word
                    ).select(fx.Int32(1), fx.Int32(0))
                # Per consumer: threads x epochs with a wrong word, with a stale word, last bad epoch.
                if wrong != fx.Int32(0):
                    comm_ops.atomic_add_agent(
                        addr_bad + fx.Int64(pair) * 16, fx.Int32(1)
                    )
                    ptr_buf_tensor(addr_bad, fx.Int32)[
                        pair * fx.Int32(4) + fx.Int32(2)
                    ] = epoch
                if stale != fx.Int32(0):
                    comm_ops.atomic_add_agent(
                        addr_bad + fx.Int64(pair) * 16 + 4, fx.Int32(1)
                    )

    @flyc.jit
    def launch(
        addr_buf: fx.Int64,
        addr_buf_peer: fx.Int64,
        addr_ready: fx.Int64,
        addr_ready_peer: fx.Int64,
        addr_warmed: fx.Int64,
        addr_warmed_peer: fx.Int64,
        addr_bad: fx.Int64,
        addr_sink: fx.Int64,
        addr_xcc: fx.Int64,
        first_epoch: fx.Int32,
        last_epoch: fx.Int32,
        role: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            addr_buf,
            addr_buf_peer,
            addr_ready,
            addr_ready_peer,
            addr_warmed,
            addr_warmed_peer,
            addr_bad,
            addr_sink,
            addr_xcc,
            first_epoch,
            last_epoch,
            role,
        ).launch(grid=(grid_x, 1, 1), block=(LITMUS_THREADS, 1, 1), stream=stream)

    return launch


class _LitmusBuffers:
    """Symmetric buffer / ready / warmed words for up to ``consumers`` x ``words``, with peer tables."""

    def __init__(self, words, consumers, rank, world, device):
        self.words, self.consumers = words, consumers
        self.buf = mori_shmem_create_tensor((2 * consumers * words,), torch.int32)
        self.ready = mori_shmem_create_tensor((consumers,), torch.int32)
        self.warmed = mori_shmem_create_tensor((consumers,), torch.int32)
        ms.shmem_barrier_all()
        self.peers = {
            name: build_p2p_table(getattr(self, name), rank, world, device).tolist()
            for name in ("buf", "ready", "warmed")
        }
        self.bad = torch.zeros(consumers * 4, dtype=torch.int32, device=device)
        self.sink = torch.zeros(
            consumers * LITMUS_THREADS, dtype=torch.int32, device=device
        )
        self.xcc = torch.full((2 * consumers,), -1, dtype=torch.int32, device=device)

    def reset(self):
        for t in (self.buf, self.ready, self.warmed, self.bad, self.sink):
            t.zero_()
        self.xcc.fill_(-1)
        torch.cuda.synchronize()
        dist.barrier()
        ms.shmem_barrier_all()


def _run_litmus_case(bufs, mode, variant, words, args, rank, world, device):
    """One (mode, variant, words) run; returns this rank's [wrong, stale, last bad epoch] (consumer ranks)."""
    same_device = mode == "xcd"
    if same_device:
        partner, role, consumer = rank, 0, True
    else:
        direction = (
            0 if mode == "xdev" else 1
        )  # xdev: even ranks produce, xdev_rev: odd ranks
        partner = rank ^ 1
        if partner >= world:
            partner = None
        role = int((rank & 1) == direction)
        consumer = partner is not None and not role
    consumers = args.litmus_consumers
    launch = _litmus_launcher(variant, words, consumers, same_device)
    bufs.reset()
    local = {name: bufs.peers[name][rank] for name in bufs.peers}
    peer = (
        {name: bufs.peers[name][partner] for name in bufs.peers}
        if partner is not None
        else local
    )
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    epochs, chunk = args.litmus_epochs, args.litmus_chunk
    for first in range(1, epochs + 1, chunk):
        last = min(epochs, first + chunk - 1)
        faulthandler.dump_traceback_later(args.timeout, exit=True)
        if partner is not None:
            _run_compiled(
                launch,
                local["buf"],
                peer["buf"],
                local["ready"],
                peer["ready"],
                local["warmed"],
                peer["warmed"],
                bufs.bad.data_ptr(),
                bufs.sink.data_ptr(),
                bufs.xcc.data_ptr(),
                first,
                last,
                role,
                stream,
            )
        torch.cuda.synchronize()
    faulthandler.cancel_dump_traceback_later()
    bad = bufs.bad.view(consumers, 4).cpu()
    xcc = bufs.xcc.cpu().tolist()
    if same_device:
        cross = sum(xcc[2 * c] != xcc[2 * c + 1] for c in range(consumers))
        if cross < consumers:
            print(
                f"[LITMUS] rank={rank} xcd: only {cross}/{consumers} producer/consumer pairs on "
                f"different XCDs ({xcc[:16]}...)",
                flush=True,
            )
    if not consumer:
        return [0, 0, 0]
    return [int(bad[:, 0].sum()), int(bad[:, 1].sum()), int(bad[:, 2].max())]


def _run_litmus(args, rank, world, device):
    if world >= 2 and world % 2:
        raise ValueError(
            "--suite litmus pairs ranks (2k, 2k + 1): use an even rank count"
        )
    modes = ["xcd"] + (["xdev", "xdev_rev"] if world >= 2 else [])
    words_list = [int(w) for w in args.litmus_words.split(",") if w]
    bufs = _LitmusBuffers(max(words_list), args.litmus_consumers, rank, world, device)
    if rank == 0:
        print(
            f"[SETUP] litmus world={world} modes={modes} words={words_list} "
            f"consumers={args.litmus_consumers} epochs={args.litmus_epochs}",
            flush=True,
        )
    failures, notes = [], []
    for mode in modes:
        for words in words_list:
            for variant in LITMUS_VARIANTS:
                result = _run_litmus_case(
                    bufs, mode, variant, words, args, rank, world, device
                )
                counts = torch.tensor(result[:2], dtype=torch.int64, device=device)
                last_bad = torch.tensor(result[2:], dtype=torch.int64, device=device)
                dist.all_reduce(counts)
                dist.all_reduce(last_bad, op=dist.ReduceOp.MAX)
                wrong, stale = int(counts[0]), int(counts[1])
                line = (
                    f"{mode} words={words} {variant}: wrong={wrong} stale={stale} "
                    f"last_bad_epoch={int(last_bad[0])} (thread-epochs)"
                )
                if rank == 0:
                    print(f"[LITMUS] {line}", flush=True)
                if variant in LITMUS_CLEAN and wrong:
                    failures.append(line)
                if variant not in LITMUS_CLEAN and not wrong:
                    notes.append(f"{mode} words={words} {variant}")
    _barrier_all()
    if notes and rank == 0:
        print(
            "[LITMUS] negative control clean (the litmus cannot show a stale read there; see the "
            "module docstring): " + ", ".join(notes),
            flush=True,
        )
    if failures:
        raise AssertionError("FAILED: " + "; ".join(failures))
    if notes and args.require_negative:
        raise AssertionError(
            "negative control clean with --require-negative: " + ", ".join(notes)
        )
    if rank == 0:
        print("[RESULT] litmus: once / all clean in every mode: PASS", flush=True)


def _barrier_all():
    torch.cuda.synchronize()
    dist.barrier()
    ms.shmem_barrier_all()


# ---- in situ ------------------------------------------------------------------

# Muse Spark A4W4 prefill geometry (EP8: 32 experts a rank), the layout _tune_muse_spark tunes.
MUSE_SPARK = {
    "model_dim": 4096,
    "inter_dim": 4096,
    "experts": 256,
    "topk": 8,
    "swiglu_limit": 0.0,
}
ROUTE_PATTERNS = ("uniform", "local", "zipf", "single_hot", "empty_destination")
# Per-rank tokens of the uneven rounds: 512 is below the tuned buckets (1024-2048).
UNEVEN_TOKENS = (2048, 1536, 512, 1023, 2048, 512, 1024, 1900)


def _round_tokens(round_idx, rank, mtpr):
    """Even rounds: every rank at one token count; odd rounds: uneven per-rank counts."""
    if round_idx % 2 == 0:
        return min(mtpr, (2048, 1024, 1536)[(round_idx // 2) % 3]), "even"
    return min(mtpr, UNEVEN_TOKENS[(rank + round_idx) % 8]), "uneven"


def _skewed_route(pattern, tokens, world, rank, round_idx, seed, device):
    """Fresh inputs of one round; the hot experts and the empty destination are the same on every rank."""
    experts, topk = MUSE_SPARK["experts"], MUSE_SPARK["topk"]
    owner = torch.arange(experts) // (experts // world)
    shared = torch.Generator().manual_seed(seed + 31 * round_idx)
    weight = torch.ones(experts, dtype=torch.float64)
    if pattern == "local":
        weight = (owner == rank).double()
    elif pattern == "zipf":
        weight = 1.0 / torch.arange(1, experts + 1, dtype=torch.float64) ** 1.2
        weight = weight[torch.randperm(experts, generator=shared)]
    elif pattern == "single_hot":
        weight[int(torch.randint(0, experts, (1,), generator=shared))] = 1.0e4
    elif pattern == "empty_destination":
        weight[owner == round_idx % world] = 0.0
    generator = torch.Generator(device=device).manual_seed(
        seed + 7919 * round_idx + rank
    )
    # Distinct top-k experts per token, as a router's top-k gives them.
    ids = torch.multinomial(
        weight.to(device).expand(tokens, experts),
        topk,
        replacement=False,
        generator=generator,
    ).to(torch.int32)
    scores = torch.randn(
        (tokens, topk), dtype=torch.float32, device=device, generator=generator
    )
    x = torch.randn(
        (tokens, MUSE_SPARK["model_dim"]), device=device, generator=generator
    )
    return x.to(torch.bfloat16), scores.softmax(dim=-1).contiguous(), ids.contiguous()


@contextlib.contextmanager
def _stage1_plans(**fields):
    """Bundle plans built inside set ``fields`` on every Stage1 config (the reference instance)."""
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config

    original = mega_moe_config.select_mega_moe_config

    def select(*args, **kwargs):
        config = original(*args, **kwargs)
        return replace(config, stage1=replace(config.stage1, **fields))

    mega_moe_config.build_mega_moe_bundle_plan.cache_clear()
    mega_moe_config.select_mega_moe_config = select
    try:
        yield
    finally:
        mega_moe_config.select_mega_moe_config = original
        mega_moe_config.build_mega_moe_bundle_plan.cache_clear()


def _insitu_instances(args, rank, world, device):
    w1, w1_scale, w2, w2_scale = _quantize_weights(
        MUSE_SPARK["model_dim"],
        MUSE_SPARK["inter_dim"],
        MUSE_SPARK["experts"] // world,
        rank,
        args.seed,
        device,
    )[:4]

    def build():
        return MegaMoEV2(
            rank=rank,
            world_size=world,
            quant=args.quant,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=args.max_tok_per_rank,
            stage2_p2p_quant=args.stage2_p2p_quant,
            large_m_enabled=False,
            small_m_mtprs=(),
            **MUSE_SPARK,
        )

    tuned = build()
    with _stage1_plans(acquire_once=False):
        reference = build()
    entries = list(
        zip(tuned._bundle_plan.entries, reference._bundle_plan.entries, strict=True)
    )
    if not any(t.config.stage1.acquire_once for t, _ in entries):
        raise ValueError(
            f"Muse Spark {args.quant} mtpr={args.max_tok_per_rank} selects no acquire_once "
            "bucket (the tune needs a4w4 and mtpr >= 2048)"
        )
    for t, r in entries:
        if (
            r.config.stage1.acquire_once
            or replace(t.config.stage1, acquire_once=False) != r.config.stage1
        ):
            raise AssertionError(
                "the reference plan must differ from the tuned one in acquire_once only"
            )
    return tuned, reference


def _run_insitu(args, rank, world, device):
    tuned, reference = _insitu_instances(args, rank, world, device)
    if rank == 0:
        buckets = [
            e.token_bucket
            for e in tuned._bundle_plan.entries
            if e.config.stage1.acquire_once
        ]
        print(
            f"[SETUP] insitu muse_spark quant={args.quant} mtpr={tuned.mtpr} world={world} "
            f"acquire_once buckets={buckets} rounds={args.insitu_rounds}",
            flush=True,
        )
    history = collections.deque(maxlen=2)  # reference outputs of the last two rounds
    failures = []
    for round_idx in range(args.insitu_rounds):
        faulthandler.dump_traceback_later(args.timeout, exit=True)
        pattern = ROUTE_PATTERNS[round_idx % len(ROUTE_PATTERNS)]
        tokens, kind = _round_tokens(round_idx, rank, tuned.mtpr)
        x, w, ids = _skewed_route(
            pattern, tokens, world, rank, round_idx, args.seed, device
        )
        output = _fwd(tuned, x, w, ids)[:tokens].clone()
        expected = _fwd(reference, x, w, ids)[:tokens].clone()
        torch.cuda.synchronize()
        mismatched = (output.view(torch.int16) != expected.view(torch.int16)).any(
            dim=-1
        )
        stale = 0
        if len(history) == 2 and bool(mismatched.any()):
            previous = history[0]
            rows = mismatched.nonzero().flatten()
            rows = rows[rows < previous.shape[0]]
            stale = int(
                (output[rows].view(torch.int16) == previous[rows].view(torch.int16))
                .all(dim=-1)
                .sum()
            )
        history.append(expected)
        stat = torch.tensor([int(mismatched.sum()), stale], device=device)
        dist.all_reduce(stat)
        if int(stat[0]):
            failures.append(
                f"round={round_idx} {pattern} {kind} mismatched_rows={int(stat[0])} "
                f"rows_equal_to_round_{round_idx - 2}={int(stat[1])}"
            )
            if len(failures) >= 8:
                break
        if rank == 0 and (round_idx + 1) % 500 == 0:
            print(
                f"[INSITU] {round_idx + 1} rounds, {len(failures)} failed", flush=True
            )
    faulthandler.cancel_dump_traceback_later()
    _barrier_all()
    if failures:
        raise AssertionError("FAILED: " + "; ".join(failures))
    if rank == 0:
        print(
            f"[RESULT] insitu rounds={args.insitu_rounds} bitwise vs acquire_once=False: ALL PASS",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=("litmus", "insitu", "all"), default="all")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="seconds a chunk / round may take"
    )
    parser.add_argument("--seed", type=int, default=13)
    litmus = parser.add_argument_group("litmus suite")
    litmus.add_argument("--litmus-epochs", type=int, default=1_000_000)
    litmus.add_argument(
        "--litmus-chunk", type=int, default=50_000, help="epochs a launch"
    )
    litmus.add_argument(
        "--litmus-consumers",
        type=int,
        default=32,
        help="consumer CTAs (same-device: + as many producers)",
    )
    litmus.add_argument(
        "--litmus-words",
        default="4096,32768",
        help="int32 words a consumer: 16 KB and 128 KB",
    )
    litmus.add_argument(
        "--require-negative",
        action="store_true",
        help="fail when a negative control is clean",
    )
    insitu = parser.add_argument_group("insitu suite")
    insitu.add_argument("--insitu-rounds", type=int, default=2000)
    insitu.add_argument("--quant", choices=("a4w4", "a8w4"), default="a4w4")
    insitu.add_argument("--max-tok-per-rank", type=int, default=2048)
    insitu.add_argument(
        "--stage2-p2p-quant",
        choices=("auto", "none", "fp8_blockwise_1x32", "int8_blockwise_1x32"),
        default="auto",
    )
    args = parser.parse_args()
    if any(
        int(w) % LITMUS_THREADS or int(w) > LITMUS_MAX_WORDS
        for w in args.litmus_words.split(",")
        if w
    ):
        raise ValueError(
            f"--litmus-words must be multiples of {LITMUS_THREADS} up to {LITMUS_MAX_WORDS}"
        )
    if 2 * args.litmus_consumers > 256:
        raise ValueError(
            "--litmus-consumers: the same-device grid must fit one CTA a CU"
        )
    if get_gfx() != "gfx950":
        print(
            f"[SKIP] acquire_once is a gfx950 (Muse Spark tune) knob, got {get_gfx()}"
        )
        return
    rank, world, device = _setup_dist()
    try:
        if args.suite in ("litmus", "all"):
            _run_litmus(args, rank, world, device)
        if args.suite in ("insitu", "all"):
            _run_insitu(args, rank, world, device)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
