# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Per-batch hot-expert replicas in front of an unchanged MegaMoEV2.

Each EP rank owns ``EPN`` experts and ``B`` replica slots: its MegaMoEV2 weight
window is ``[EPN + B]`` experts (home rows, then slots), and the MegaMoEV2
instance is built over ``R * (EPN + B)`` virtual experts. Virtual id
``d * (EPN + B) + k`` is row ``k`` of rank ``d``'s window.

Per forward (``plan``, on the critical path):

1. ``count``: this rank's routes per expert, and each route's index within its
   expert on this rank.
2. exchange: all-gather of the ``[E]`` histograms into ``[R, E]``.
3. ``place`` (one program): every expert with replicas has its routes split
   between its owner and the ranks whose slots hold it by water-filling, the
   hottest expert first, over the loads of the experts without replicas.
   Every rank runs it on the same matrix, so all agree without further traffic.
4. ``rewrite``: route ``i`` of expert ``e`` (ranked source-major) goes to the
   destination whose share ``[alloc_cumsum[e, d-1], alloc_cumsum[e, d])`` holds
   it, as the virtual id of that destination's copy.

The slots themselves are chosen off the critical path (``select``) from the
same gathered histogram, for the next forward of the layer, and filled by the
caller (a P2P pull from the owners' windows) before that forward. The slot set
is therefore one forward stale; the split is live.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_CBLOCK = 128
_TAG = 1 << 16


@triton.jit
def _count_kernel(ids_ptr, n, hist_ptr, li_ptr, base_ptr, E: tl.constexpr, BLOCK: tl.constexpr):
    """hist[e] += this block's routes of e (one atomic per block and expert); li = the
    route's index within its expert on this rank (block base + order within the block)."""
    pid = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    offs = pid * BLOCK + i
    m = offs < n
    e = tl.load(ids_ptr + offs, mask=m, other=-1)
    v = m & (e >= 0)
    ev = tl.where(v, e, -1)
    order = tl.sum(((ev[:, None] == ev[None, :]) & (i[None, :] < i[:, None])).to(tl.int32), axis=1)
    ex = tl.arange(0, E)
    bh = tl.sum((ev[:, None] == ex[None, :]).to(tl.int32), axis=0)
    base = tl.atomic_add(hist_ptr + ex, bh, mask=bh > 0, sem="relaxed")
    tl.store(base_ptr + pid * E + ex, base, mask=bh > 0)
    tl.debug_barrier()
    b = tl.load(base_ptr + pid * E + tl.where(v, e, 0), mask=v, other=0)
    tl.store(li_ptr + offs, b + order, mask=v)


@triton.jit
def _place_kernel(
    hist_ptr, mat_ptr, peers_ptr, epoch_ptr, held_ptr, rank, gate_pct, refresh_pct,
    acs_ptr, e2s_ptr, info_ptr, scratch_ptr, ecount_out_ptr,
    R: tl.constexpr, E: tl.constexpr, B: tl.constexpr, EPN: tl.constexpr, P2P: tl.constexpr,
    ECOUNT_OUT: tl.constexpr,
):
    r = tl.arange(0, R)
    e = tl.arange(0, E)
    s = tl.arange(0, B)
    if P2P:
        # Push this rank's histogram into row `rank` of every peer's [2, R, E] matrix
        # (parity buffers: a rank one launch ahead never overwrites the matrix a slower
        # peer still reads), then raise flag [par, rank] on every peer with a system-scope
        # release. The kernel is one wave, so that release orders all of its stores.
        ep = tl.load(epoch_ptr) + 1
        par = ep & 1
        h = tl.load(hist_ptr + e)
        # Data moves as system-scope atomics, like the flags: plain stores into a peer
        # and loads of the local matrix can meet stale L2 lines.
        for d in tl.static_range(R):
            peer = tl.load(peers_ptr + d).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(peer + par * R * E + rank * E + e, h, sem="relaxed", scope="sys")
        for d in tl.static_range(R):
            peer = tl.load(peers_ptr + d).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(peer + 2 * R * E + par * R + rank, ep, sem="release", scope="sys")
        for src in tl.static_range(R):
            fp = mat_ptr + 2 * R * E + par * R + src
            f = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
            while f < ep:
                f = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
        m = tl.atomic_add(mat_ptr + par * R * E + r[:, None] * E + e[None, :], 0, sem="relaxed",
                          scope="sys")
        tl.store(epoch_ptr, ep)
    else:
        m = tl.load(mat_ptr + r[:, None] * E + e[None, :])
    tl.store(hist_ptr + e, tl.zeros([E], tl.int32))
    ecount = tl.sum(m, axis=0)
    tl.store(info_ptr + e, tl.sum(tl.where(r[:, None] < rank, m, 0), axis=0))
    tl.store(info_ptr + E + e, ecount)
    if ECOUNT_OUT:
        tl.store(ecount_out_ptr + e, ecount)
    tl.store(info_ptr + 2 * E + e, tl.sum(tl.where(r[:, None] == rank, m, 0), axis=0))
    owner = e // EPN
    held = tl.load(held_ptr + r[:, None] * B + s[None, :])
    slot_of = tl.full([R, E], B, tl.int32)
    for t in tl.static_range(B):
        h_t = tl.load(held_ptr + r * B + t)
        slot_of = tl.where((h_t[:, None] == e[None, :]) & (slot_of == B), t, slot_of)
    is_owner = owner[None, :] == r[:, None]
    home_load = tl.sum(tl.where(is_owner, ecount[None, :], 0), axis=1)
    # Gate: a batch whose busiest rank is within gate_pct% of the mean stays at home.
    on = tl.max(home_load, axis=0) * 100 * R > tl.sum(home_load, axis=0) * gate_pct
    rep = (slot_of < B) & (~is_owner) & on
    nrep = tl.sum(rep.to(tl.int32), axis=0)
    single = nrep == 0
    load = tl.sum(tl.where(is_owner & single[None, :], ecount[None, :], 0), axis=1)
    pending = tl.where(single, -1, ecount)
    n_iter = tl.sum((~single).to(tl.int32), axis=0)
    j = tl.arange(0, R) + 1
    for _ in range(n_iter):
        best = tl.max(pending, axis=0)
        ebest = tl.argmax(pending, axis=0)
        # copies of ebest: its owner and the ranks holding it in a slot
        c = (tl.sum((held == ebest).to(tl.int32), axis=1) > 0) | (r == ebest // EPN)
        # water level: the k lowest copies rise to L = (best + their loads) // k, where k is
        # the most copies the routes can lift to the k-th lowest load
        ls = tl.sort(tl.where(c, load, 1 << 22))
        cs = tl.cumsum(ls, axis=0)
        k = tl.sum((((j * ls - cs) <= best) & (ls < (1 << 22))).to(tl.int32), axis=0)
        lvl = (best + tl.sum(tl.where(j <= k, ls, 0), axis=0)) // k
        a = tl.where(c, tl.maximum(lvl - load, 0), 0)
        rem = best - tl.sum(a, axis=0)
        dmin = tl.argmin(tl.where(c, load + a, 1 << 22), axis=0)
        a = a + tl.where(r == dmin, rem, 0)
        load = load + a
        tl.store(scratch_ptr + ebest * R + r, a)
        pending = tl.where(e == ebest, -1, pending)
    tl.debug_barrier()
    ra = tl.load(scratch_ptr + e[None, :] * R + r[:, None], mask=(~single)[None, :], other=0)
    alloc = tl.where(single[None, :], tl.where(is_owner, ecount[None, :], 0), ra)
    tl.store(acs_ptr + e[None, :] * R + r[:, None], tl.cumsum(alloc, axis=0))
    # Slot refresh flag for select: the held slots left this batch's busiest rank
    # above refresh_pct% of the mean.
    lmax = tl.max(tl.sum(alloc, axis=1), axis=0)
    miss = (lmax * 100 * R > tl.sum(ecount, axis=0) * refresh_pct).to(tl.int32)
    tl.store(info_ptr + 3 * E, miss)
    if ECOUNT_OUT:
        tl.store(ecount_out_ptr + E, miss)
    e2s = tl.where(is_owner, e[None, :] - r[:, None] * EPN, tl.where(rep, EPN + slot_of, -1))
    tl.store(e2s_ptr + r[:, None] * E + e[None, :], e2s)


@triton.jit
def _rewrite_kernel(
    ids_ptr, n, li_ptr, info_ptr, acs_ptr, e2s_ptr, out_ptr,
    R: tl.constexpr, E: tl.constexpr, VS: tl.constexpr, BLOCK: tl.constexpr, IL: tl.constexpr,
):
    """IL=0: expert e's routes are ranked source-major, so a source's routes of e go to
    one or two copies. IL=1: each source spreads its routes of e over the copies in
    proportion to their shares (per-destination totals then match alloc to +-R)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    e = tl.load(ids_ptr + offs, mask=m, other=-1)
    v = m & (e >= 0)
    es = tl.where(v, e, 0)
    li = tl.load(li_ptr + offs, mask=v, other=0)
    if IL:
        tot = tl.load(info_ptr + E + es).to(tl.int64)
        mine = tl.maximum(tl.load(info_ptr + 2 * E + es), 1).to(tl.int64)
        rk = (((2 * li + 1).to(tl.int64) * tot) // (2 * mine)).to(tl.int32)
    else:
        rk = tl.load(info_ptr + es) + li
    dest = tl.zeros([BLOCK], tl.int32)
    for d in tl.static_range(R - 1):
        dest += (rk >= tl.load(acs_ptr + es * R + d)).to(tl.int32)
    slot = tl.load(e2s_ptr + dest * E + es)
    tl.store(out_ptr + offs, tl.where(v, dest * VS + slot, e), mask=m)


@triton.jit
def _select_kernel(
    ecount_ptr, held_ptr, placed_ptr, hyst, flag_ptr,
    R: tl.constexpr, E: tl.constexpr, B: tl.constexpr, EPN: tl.constexpr,
):
    """Greedy slot choice on the gathered histogram: R*B times, the busiest rank's
    largest per-copy share gets one more copy on the least-loaded rank with a free
    slot (loads tracked with an equal split between copies). Sticky: a slot whose
    expert stays wanted keeps it; new experts take the other slots in order.
    ``hyst`` (fraction) damps churn: shares of held experts count (1 + hyst) times,
    and a held expert's new copy goes to a rank already holding it when one is free.
    Writes ``held`` (after) and ``placed`` (slots to fill, -1 = keep); both stay
    unchanged / all -1 when ``flag`` (place's refresh flag) is 0."""
    r = tl.arange(0, R)
    e = tl.arange(0, E)
    s = tl.arange(0, B)
    ecount = tl.load(ecount_ptr + e).to(tl.float32)
    held0 = tl.load(held_ptr + r[:, None] * B + s[None, :])
    # holder[r, e]: rank r holds e in a slot now
    holder = tl.zeros([R, E], tl.int32)
    for t in tl.static_range(B):
        h_t = tl.load(held_ptr + r * B + t)
        holder = holder | (h_t[:, None] == e[None, :]).to(tl.int32)
    bonus = tl.where(tl.sum(holder, axis=0) > 0, 1.0 + hyst, 1.0)
    hosts = (e[None, :] // EPN) == r[:, None]
    ncop = tl.full([E], 1.0, tl.float32)
    load = tl.sum(tl.where(hosts, ecount[None, :], 0.0), axis=1)
    nwant = tl.zeros([R], tl.int32)
    want = tl.full([R, B], -1, tl.int32)
    for _ in range(R * B):
        h = tl.argmax(load, axis=0)
        free = nwant < B
        on_h = tl.sum(tl.where((r[:, None] == h) & hosts, 1, 0), axis=0) > 0
        cand = tl.sum(tl.where(free[:, None] & (~hosts), 1, 0), axis=0) > 0
        share = ecount / ncop
        sc = tl.where(on_h & cand, share * bonus, -1.0)
        be = tl.argmax(sc, axis=0)
        active = tl.max(sc, axis=0) > 0.0
        col = e == be
        best = tl.sum(tl.where(col, share, 0.0), axis=0)
        hb = tl.sum(tl.where(col[None, :] & hosts, 1, 0), axis=1) > 0
        ok = free & (~hb)
        had = ok & (tl.sum(tl.where(col[None, :], holder, 0), axis=1) > 0)
        pick = tl.where(tl.sum(had.to(tl.int32), axis=0) > 0, had, ok)
        d = tl.argmin(tl.where(pick, load, float("inf")), axis=0)
        nb = tl.sum(tl.where(col, ecount, 0.0), axis=0) / (tl.sum(tl.where(col, ncop, 0.0), axis=0) + 1.0)
        load = tl.where(active, load + tl.where(hb, nb - best, 0.0) + tl.where(r == d, nb, 0.0), load)
        hosts = hosts | (col[None, :] & (r[:, None] == d) & active)
        ncop = tl.where(col & active, ncop + 1.0, ncop)
        nd = tl.sum(tl.where(r == d, nwant, 0), axis=0)
        want = tl.where((r[:, None] == d) & (s[None, :] == nd) & active, be, want)
        nwant = tl.where((r == d) & active, nwant + 1, nwant)
    held = held0
    # keep[r, t]: slot t's expert is still wanted; new[r, s]: want[r, s] is not held
    keep = tl.sum(((want[:, :, None] == held[:, None, :]) & (held[:, None, :] >= 0)).to(tl.int32), axis=1) > 0
    new = (want >= 0) & (tl.sum((want[:, :, None] == held[:, None, :]).to(tl.int32), axis=2) == 0)
    free_rank = tl.cumsum((~keep).to(tl.int32), axis=1) - 1
    new_rank = tl.cumsum(new.to(tl.int32), axis=1) - 1
    hit = new[:, :, None] & (new_rank[:, :, None] == free_rank[:, None, :]) & (~keep)[:, None, :]
    placed = tl.max(tl.where(hit, want[:, :, None], -1), axis=1)
    go = tl.load(flag_ptr) != 0
    tl.store(held_ptr + r[:, None] * B + s[None, :], tl.where(keep | (~go), held, placed))
    tl.store(placed_ptr + r[:, None] * B + s[None, :], tl.where(go, placed, -1))


class ReplicaPlanner:
    """Plans one EP group's routes onto home experts + ``slots`` replica slots per rank.

    ``plan(topk_ids, held)`` returns virtual ids for a MegaMoEV2 over
    ``world_size * (experts_per_rank + slots)`` experts; ``held`` is the layer's
    int32 ``[world_size, slots]`` table of the global expert each slot holds (-1 empty).
    ``select(held, placed)`` then updates ``held`` for the next forward of that
    layer from this forward's histogram; ``placed`` names the slots to fill.

    ``exchange="p2p"`` pushes the histograms through a mori symmetric buffer inside the
    place kernel (mori shmem must be initialized, every rank constructs the planner in the
    same order); ``"allgather"`` uses ``torch.distributed`` on ``group``.
    ``gate_pct``: batches whose busiest rank is within this percentage of the mean keep
    every route at home. ``interleave``: see ``_rewrite_kernel``. ``select_hyst``: churn
    damping of ``select`` (see ``_select_kernel``; 0 = none). ``refresh_pct``: ``select``
    changes the slots only after a batch whose planned busiest rank (with the held slots)
    exceeded this percentage of the mean (0 = every batch); a miss-driven refresh keeps
    the weight pulls rare once the slots fit the running prompt.
    """

    def __init__(self, *, rank: int, world_size: int, experts: int, slots: int,
                 max_routes: int, device: torch.device, group=None, exchange: str = "p2p",
                 gate_pct: int = 110, interleave: bool = True, select_hyst: float = 0.0,
                 refresh_pct: int = 0):
        if experts % world_size:
            raise ValueError("experts must divide evenly over the ranks")
        if slots not in (1, 2, 4, 8):
            raise ValueError(f"slots must be 1, 2, 4 or 8, got {slots}")
        if world_size & (world_size - 1) or experts & (experts - 1):
            raise ValueError("world_size and experts must be powers of two")
        if exchange not in ("p2p", "allgather", "none"):
            raise ValueError(f"unknown exchange {exchange!r}")
        self.rank, self.world_size, self.experts, self.slots = rank, world_size, experts, slots
        self.epn = experts // world_size
        self.vs = self.epn + slots
        self.virtual_experts = world_size * self.vs
        self.group, self.exchange_mode = group, exchange
        self.gate_pct, self.interleave = int(gate_pct), bool(interleave)
        self.select_hyst = float(select_hyst)
        self.refresh_pct = int(refresh_pct)
        R, E = world_size, experts
        i32 = lambda *n: torch.zeros(*n, dtype=torch.int32, device=device)
        self.hist = i32(E)
        self.local_index = i32(max_routes)
        self.block_base = i32(triton.cdiv(max_routes, _CBLOCK) * E)
        self.info = i32(3 * E + 1)
        self.alloc_cumsum = i32(E * R)
        self.expert_to_slot = i32(R * E)
        self.scratch = i32(E * R)
        self.vids = i32(max_routes)
        self.epoch = i32(1)
        if exchange == "p2p":
            import mori.shmem as ms
            from mori.shmem import mori_shmem_create_tensor

            self.mat = mori_shmem_create_tensor((2 * R * E + 2 * R,), torch.int32)
            self.mat.zero_()
            self.peers = torch.tensor(
                [ms.shmem_ptr_p2p(self.mat.data_ptr(), rank, pe) for pe in range(R)],
                dtype=torch.int64, device=device)
            torch.cuda.synchronize(device)
            ms.shmem_barrier_all()
        else:
            self.mat = i32(R * E)
            self.peers = torch.zeros(R, dtype=torch.int64, device=device)

    def new_held(self) -> torch.Tensor:
        return torch.full((self.world_size, self.slots), -1, dtype=torch.int32,
                          device=self.hist.device)

    def home_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        return topk_ids + (topk_ids.clamp(min=0) // self.epn) * self.slots

    def count(self, topk_ids: torch.Tensor) -> None:
        n = topk_ids.numel()
        _count_kernel[(triton.cdiv(max(n, 1), _CBLOCK),)](
            topk_ids.reshape(-1), n, self.hist, self.local_index, self.block_base,
            E=self.experts, BLOCK=_CBLOCK, num_warps=8)

    def place(self, held: torch.Tensor, ecount_out: torch.Tensor | None = None) -> None:
        """``ecount_out``: optional int32 ``[experts + 1]`` that also receives the expert
        totals and the refresh flag (for a later ``select`` after other layers have
        planned)."""
        p2p = self.exchange_mode == "p2p"
        if self.exchange_mode == "allgather":
            import torch.distributed as dist

            dist.all_gather_into_tensor(self.mat, self.hist, group=self.group)
        _place_kernel[(1,)](
            self.hist, self.mat, self.peers, self.epoch, held, self.rank, self.gate_pct,
            self.refresh_pct,
            self.alloc_cumsum, self.expert_to_slot, self.info, self.scratch,
            self.info if ecount_out is None else ecount_out,
            R=self.world_size, E=self.experts, B=self.slots, EPN=self.epn, P2P=p2p,
            ECOUNT_OUT=ecount_out is not None, num_warps=1,
        )

    def rewrite(self, topk_ids: torch.Tensor) -> torch.Tensor:
        n = topk_ids.numel()
        _rewrite_kernel[(triton.cdiv(max(n, 1), _BLOCK),)](
            topk_ids.reshape(-1), n, self.local_index, self.info, self.alloc_cumsum,
            self.expert_to_slot, self.vids, R=self.world_size, E=self.experts, VS=self.vs,
            BLOCK=_BLOCK, IL=self.interleave,
        )
        return self.vids[:n].view(topk_ids.shape)

    def plan(self, topk_ids: torch.Tensor, held: torch.Tensor,
             ecount_out: torch.Tensor | None = None) -> torch.Tensor:
        self.count(topk_ids)
        self.place(held, ecount_out)
        return self.rewrite(topk_ids)

    def select(self, held: torch.Tensor, placed: torch.Tensor,
               ecount: torch.Tensor | None = None) -> None:
        """Next forward's slots from expert totals + refresh flag ``ecount``
        (``place``'s ``ecount_out``; default: those of the last ``place``)."""
        E = self.experts
        _select_kernel[(1,)](
            self.info[E:] if ecount is None else ecount, held, placed, self.select_hyst,
            self.info[3 * E:] if ecount is None else ecount[E:],
            R=self.world_size, E=self.experts,
            B=self.slots, EPN=self.epn, num_warps=4,
        )


__all__ = ["ReplicaPlanner"]
