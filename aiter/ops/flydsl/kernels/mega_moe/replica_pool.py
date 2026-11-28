# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 weight windows with replica slots, readable by every EP rank.

One layer's weights on one rank are ``P`` parts (e.g. w1, w1_scale, w2, w2_scale), each
``EPN`` experts of ``row_bytes[p]`` bytes. The pool lays them out in a single device
buffer as ``[EPN + B]``-expert windows (home rows, then ``B`` replica slots), shares the
buffer with the other ranks of the group through CUDA IPC, and ``fill(placed)`` (a copy
kernel) or ``fill_dma(placed_row, stream)`` (peer DMA, host-known slots) pulls the experts
named in ``placed`` (``ReplicaPlanner.select``) from their owners' home rows into this
rank's slots. Home rows never change after construction, so the pull can run at any
time on a side stream; the caller orders it after the last read of the slots it rewrites
and before their next use. Next to MegaMoEV2's persistent kernels prefer ``fill_dma``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

_ALIGN = 4096


@triton.jit
def _fill_kernel(
    placed_ptr, peers_ptr, dst_ptr, off_ptr, rb_ptr, rank,
    EPN: tl.constexpr, B: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr,
):
    """Program (g, t): slot t's expert, strided over the G programs. Offsets and row
    sizes are in 8-byte units."""
    g = tl.program_id(0)
    G = tl.num_programs(0)
    t = tl.program_id(1)
    e = tl.load(placed_ptr + rank * B + t)
    if e >= 0:
        owner = e // EPN
        src_base = tl.load(peers_ptr + owner).to(tl.pointer_type(tl.int64))
        lane = tl.arange(0, BLOCK)
        for p in tl.static_range(P):
            off = tl.load(off_ptr + p)
            rb = tl.load(rb_ptr + p)
            src = src_base + off + (e - owner * EPN) * rb
            dst = dst_ptr + off + (EPN + t) * rb
            for i in range(g * BLOCK, rb, G * BLOCK):
                o = i + lane
                m = o < rb
                tl.store(dst + o, tl.load(src + o, mask=m), mask=m)


class ReplicaWeightPool:
    """``parts``: this rank's home weights for one layer, each with ``experts_per_rank``
    equal expert blocks along dim 0. ``windows`` are the ``[EPN + B]`` views to pass to
    MegaMoEV2 (same dtype and trailing shape as the parts). Collective over ``group``
    (``torch.distributed``; every rank builds its pools in the same order)."""

    def __init__(self, parts: Sequence[torch.Tensor], *, rank: int, world_size: int,
                 experts_per_rank: int, slots: int, group=None, programs: int = 32):
        import torch.distributed as dist
        from torch.multiprocessing.reductions import reduce_tensor

        self.rank, self.world_size = rank, world_size
        self.epn, self.slots, self.programs = experts_per_rank, slots, programs
        device = parts[0].device
        offs, rbs, total = [], [], 0
        for w in parts:
            nbytes = w.numel() * w.element_size()
            if w.shape[0] % experts_per_rank or nbytes % (16 * experts_per_rank):
                raise ValueError(f"part of shape {tuple(w.shape)} does not split into "
                                 f"{experts_per_rank} 16-byte-aligned experts")
            rb = nbytes // experts_per_rank
            offs.append(total)
            rbs.append(rb)
            total += -(-(experts_per_rank + slots) * rb // _ALIGN) * _ALIGN
        self.buf = torch.zeros(total, dtype=torch.uint8, device=device)
        self.windows = []
        for w, off, rb in zip(parts, offs, rbs):
            lead = w.shape[0] // experts_per_rank * (experts_per_rank + slots)
            seg = self.buf[off:off + (experts_per_rank + slots) * rb]
            seg[: experts_per_rank * rb].copy_(w.contiguous().view(-1).view(torch.uint8))
            self.windows.append(seg.view(w.dtype).view(lead, *w.shape[1:]))
        self.offsets = torch.tensor([o // 8 for o in offs], dtype=torch.int64, device=device)
        self.row_units = torch.tensor([r // 8 for r in rbs], dtype=torch.int64, device=device)
        torch.cuda.synchronize(device)
        handles = [None] * world_size
        dist.all_gather_object(handles, reduce_tensor(self.buf), group=group)
        self._peer_bufs, ptrs = [], []
        for d, (fn, args) in enumerate(handles):
            if d == rank:
                ptrs.append(self.buf.data_ptr())
                continue
            t = fn(*args)
            self._peer_bufs.append(t)
            ptrs.append(t.data_ptr())
        self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self._peer_ptrs, self._offs, self._rbs = ptrs, offs, rbs

    def fill(self, placed: torch.Tensor) -> None:
        """Pull ``placed[rank, t]`` (global expert ids, -1 = keep) into slot ``t``."""
        _fill_kernel[(self.programs, self.slots)](
            placed, self.peers, self.buf.view(torch.int64), self.offsets, self.row_units, self.rank,
            EPN=self.epn, B=self.slots, P=len(self.windows), BLOCK=2048, num_warps=8,
        )

    def fill_dma(self, placed_row: Sequence[int], stream: torch.cuda.Stream) -> None:
        """``fill`` with host-known slot contents (``placed[rank]``) as peer
        ``hipMemcpyAsync`` copies on ``stream``: DMA engines, no CUs, so it can overlap
        persistent kernels (a copy kernel next to MegaMoEV2 slows it down)."""
        hip, base = _hip(), self.buf.data_ptr()
        for t, e in enumerate(placed_row):
            if e < 0:
                continue
            owner, local = divmod(int(e), self.epn)
            for off, rb in zip(self._offs, self._rbs):
                err = hip.hipMemcpyAsync(
                    _vp(base + off + (self.epn + t) * rb),
                    _vp(self._peer_ptrs[owner] + off + local * rb),
                    _sz(rb), 3, _vp(stream.cuda_stream))
                if err:
                    raise RuntimeError(f"hipMemcpyAsync failed: {err}")


_HIP = None


def _hip():
    global _HIP
    if _HIP is None:
        import ctypes

        _HIP = ctypes.CDLL("libamdhip64.so")
        _HIP.hipMemcpyAsync.restype = ctypes.c_int
        _HIP.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.c_int, ctypes.c_void_p]
    return _HIP


def _vp(x: int):
    import ctypes

    return ctypes.c_void_p(x)


def _sz(x: int):
    import ctypes

    return ctypes.c_size_t(x)


__all__ = ["ReplicaWeightPool"]
