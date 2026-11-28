# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ReplicaPlanner (MegaMoE hot-expert replicas) against a CPU reference, on one GPU.

The 8 EP ranks run in turn with the gathered [R, E] histogram set directly (exchange="none"). Per synthetic
route set, slot count B and rewrite mode:

- ``select``: the held / placed tables equal a CPU mirror of _select_kernel (its f32 arithmetic step for step),
  and keep their invariants (no duplicate or home expert in a rank's slots, placed = the slots that changed,
  nothing changes without the refresh flag);
- ``count``: the histogram, and each route's index within its expert is a permutation of that expert's routes;
- ``place``: info, alloc_cumsum and expert_to_slot equal the CPU water-filling (integer, exact), and the gate
  keeps balanced batches at home;
- ``rewrite``: the virtual ids equal the CPU rewrite of the kernel's route indices, every virtual id resolves
  to the route's own expert, and the routes each destination receives match place's allocation (exactly
  without interleave, to +-R per expert and destination with it).

Follow-up (8 GPUs, not here): MegaMoEV2 over the virtual experts with a forced slot refresh, B = 1/2/4/8, output
bitwise equal to the plain MegaMoEV2 on the same routes, plus a negative control (a wrong slot fill must fail).

    pytest -q op_tests/test_mega_moe_replica_plan.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

R, E, TOPK = 8, 256, 8
EPN = E // R
TOKENS_PER_RANK = 2048
_FAR = 1 << 22


def _route_set(name, seed):
    """[R * TOKENS_PER_RANK, TOPK] int32 global expert ids (distinct per token; -1 = masked slot)."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.ones(E, dtype=torch.float64)
    if name in ("zipf", "masked"):
        weight = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** 1.2
        weight = weight[torch.randperm(E, generator=generator)]
    elif name == "hot_rank":
        weight[2 * EPN : 3 * EPN] = 10.0
    elif name == "single_hot":
        weight[77] = 400.0
    ids = torch.multinomial(
        weight.expand(R * TOKENS_PER_RANK, E),
        TOPK,
        replacement=False,
        generator=generator,
    )
    if name == "masked":
        ids[torch.rand(ids.shape, generator=generator) < 0.05] = -1
    return ids.to(torch.int32)


ROUTE_SETS = ("uniform", "zipf", "hot_rank", "single_hot", "masked")


def _histograms(ids):
    mat = torch.zeros(R, E, dtype=torch.int64)
    for r in range(R):
        mine = ids[r * TOKENS_PER_RANK : (r + 1) * TOKENS_PER_RANK].flatten().long()
        mat[r] = torch.bincount(mine[mine >= 0], minlength=E)
    return mat


def ref_select(ecount, held, flag, hyst, slots):
    """_select_kernel on the CPU: f32 like the kernel, leftmost argmax / argmin."""
    f32 = torch.float32
    e = torch.arange(E)
    r = torch.arange(R)
    ec = ecount.to(f32)
    held0 = held.clone()
    holder = (held0[:, :, None] == e[None, None, :]).any(1)
    bonus = torch.where(
        holder.any(0),
        torch.tensor(1.0, dtype=f32) + torch.tensor(hyst, dtype=f32),
        torch.tensor(1.0, dtype=f32),
    )
    hosts = (e[None, :] // EPN) == r[:, None]
    ncop = torch.ones(E, dtype=f32)
    load = torch.where(hosts, ec[None, :], torch.zeros((), dtype=f32)).sum(1)
    nwant = torch.zeros(R, dtype=torch.int64)
    want = torch.full((R, slots), -1, dtype=torch.int64)
    for _ in range(R * slots):
        h = int(torch.argmax(load))
        free = nwant < slots
        on_h = hosts[h]
        cand = (free[:, None] & ~hosts).any(0)
        share = ec / ncop
        sc = torch.where(on_h & cand, share * bonus, torch.tensor(-1.0, dtype=f32))
        be = int(torch.argmax(sc))
        if not float(sc.max()) > 0.0:
            continue
        best = share[be]
        hb = hosts[:, be]
        ok = free & ~hb
        had = ok & holder[:, be]
        pick = had if bool(had.any()) else ok
        d = int(
            torch.argmin(torch.where(pick, load, torch.tensor(float("inf"), dtype=f32)))
        )
        nb = ec[be] / (ncop[be] + torch.tensor(1.0, dtype=f32))
        load = (
            load + torch.where(hb, nb - best, torch.zeros((), dtype=f32))
        ) + torch.where(r == d, nb, torch.zeros((), dtype=f32))
        hosts[d, be] = True
        ncop[be] = ncop[be] + 1.0
        want[d, nwant[d]] = be
        nwant[d] += 1
    keep = ((want[:, :, None] == held0[:, None, :]) & (held0[:, None, :] >= 0)).any(1)
    new = (want >= 0) & ~(want[:, :, None] == held0[:, None, :]).any(2)
    free_rank = torch.cumsum((~keep).long(), 1) - 1
    new_rank = torch.cumsum(new.long(), 1) - 1
    hit = (
        new[:, :, None]
        & (new_rank[:, :, None] == free_rank[:, None, :])
        & (~keep)[:, None, :]
    )
    placed = torch.where(hit, want[:, :, None], torch.tensor(-1)).amax(1)
    if not flag:
        return held0, torch.full_like(held0, -1)
    return torch.where(keep, held0, placed), placed


def ref_place(mat, held, rank, gate_pct, refresh_pct, slots):
    """_place_kernel (exchange="none") on the CPU: info, alloc_cumsum [E, R] and expert_to_slot [R, E]."""
    mat = mat.tolist()
    held = held.tolist()
    ecount = [sum(mat[r][e] for r in range(R)) for e in range(E)]
    info = (
        [sum(mat[r][e] for r in range(rank)) for e in range(E)]
        + ecount
        + list(mat[rank])
        + [0]
    )
    slot_of = [
        [next((t for t in range(slots) if held[r][t] == e), slots) for e in range(E)]
        for r in range(R)
    ]
    home_load = [sum(ecount[r * EPN : (r + 1) * EPN]) for r in range(R)]
    on = max(home_load) * 100 * R > sum(home_load) * gate_pct
    rep = [
        [slot_of[r][e] < slots and e // EPN != r and on for e in range(E)]
        for r in range(R)
    ]
    single = [not any(rep[r][e] for r in range(R)) for e in range(E)]
    load = [
        sum(ecount[e] for e in range(r * EPN, (r + 1) * EPN) if single[e])
        for r in range(R)
    ]
    pending = [-1 if single[e] else ecount[e] for e in range(E)]
    shares = {}
    for _ in range(sum(not s for s in single)):
        best = max(pending)
        ebest = pending.index(best)
        c = [ebest in held[r] or r == ebest // EPN for r in range(R)]
        ls = sorted(load[r] if c[r] else _FAR for r in range(R))
        cs = [sum(ls[: j + 1]) for j in range(R)]
        k = sum(1 for j in range(R) if (j + 1) * ls[j] - cs[j] <= best and ls[j] < _FAR)
        lvl = (best + sum(ls[:k])) // k
        a = [max(lvl - load[r], 0) if c[r] else 0 for r in range(R)]
        rem = best - sum(a)
        after = [load[r] + a[r] if c[r] else _FAR for r in range(R)]
        a[after.index(min(after))] += rem
        load = [load[r] + a[r] for r in range(R)]
        shares[ebest] = a
        pending[ebest] = -1
    alloc = [
        [
            (ecount[e] if e // EPN == r else 0) if single[e] else shares[e][r]
            for e in range(E)
        ]
        for r in range(R)
    ]
    alloc_cumsum = torch.tensor(alloc).cumsum(0).T.contiguous()
    lmax = max(sum(row) for row in alloc)
    info[3 * E] = int(lmax * 100 * R > sum(ecount) * refresh_pct)
    e2s = [
        [
            e - r * EPN if e // EPN == r else (EPN + slot_of[r][e] if rep[r][e] else -1)
            for e in range(E)
        ]
        for r in range(R)
    ]
    return torch.tensor(info), alloc_cumsum, torch.tensor(e2s), torch.tensor(alloc)


def ref_rewrite(ids, li, info, alloc_cumsum, e2s, slots, interleave):
    """_rewrite_kernel on the CPU, given the count kernel's route indices ``li``."""
    ids, li = ids.long(), li.long()
    valid = ids >= 0
    es = ids.clamp(min=0)
    if interleave:
        tot = info[E + es]
        mine = info[2 * E + es].clamp(min=1)
        rk = ((2 * li + 1) * tot) // (2 * mine)
    else:
        rk = info[es] + li
    dest = (rk[:, None] >= alloc_cumsum[es][:, : R - 1]).sum(1)
    return torch.where(valid, dest * (EPN + slots) + e2s[dest, es], ids)


def _check_held(held, placed, before, slots):
    for r in range(R):
        row = [e for e in held[r].tolist() if e >= 0]
        assert len(row) == len(set(row)) and all(e // EPN != r for e in row), (
            r,
            held[r],
        )
        for t in range(slots):
            changed = held[r, t] != before[r, t] and held[r, t] >= 0
            assert placed[r, t] == (held[r, t] if changed else -1), (r, t)


@pytest.fixture(scope="module")
def route_sets():
    return {name: _route_set(name, seed) for seed, name in enumerate(ROUTE_SETS)}


@pytest.mark.parametrize("gate_pct", (0, 110))
@pytest.mark.parametrize("interleave", (False, True))
@pytest.mark.parametrize("slots", (1, 2, 4, 8))
def test_planner_matches_cpu_reference(route_sets, slots, interleave, gate_pct):
    from aiter.ops.flydsl.kernels.mega_moe.replica_plan import ReplicaPlanner

    device = torch.device("cuda", torch.cuda.current_device())
    planners = [
        ReplicaPlanner(
            rank=r,
            world_size=R,
            experts=E,
            slots=slots,
            max_routes=TOKENS_PER_RANK * TOPK,
            device=device,
            exchange="none",
            gate_pct=gate_pct,
            interleave=interleave,
            refresh_pct=110,
        )
        for r in range(R)
    ]
    p0 = planners[0]
    for name, ids in route_sets.items():
        mat = _histograms(ids)
        ecount = mat.sum(0)
        held, placed = p0.new_held(), p0.new_held()
        p0.info[E : 2 * E].copy_(ecount.to(torch.int32))
        p0.info[3 * E] = 1
        p0.select(held, placed)
        ref_held, ref_placed = ref_select(
            ecount, torch.full((R, slots), -1), 1, 0.0, slots
        )
        assert torch.equal(held.cpu().long(), ref_held), name
        assert torch.equal(placed.cpu().long(), ref_placed), name
        _check_held(held.cpu(), placed.cpu(), torch.full((R, slots), -1), slots)

        received = torch.zeros(E, R, dtype=torch.int64)
        for r, planner in enumerate(planners):
            mine = ids[r * TOKENS_PER_RANK : (r + 1) * TOKENS_PER_RANK].contiguous()
            planner.mat.copy_(mat.flatten().to(torch.int32))
            vids = planner.plan(mine.to(device), held).cpu().long().flatten()
            flat = mine.flatten().long()
            valid = flat >= 0
            li = planner.local_index[: flat.numel()].cpu().long()
            assert (
                int(planner.hist.abs().sum()) == 0
            ), "place leaves the histogram cleared"
            for e in torch.unique(flat[valid]).tolist():
                got = li[flat == e].sort().values
                assert torch.equal(got, torch.arange(int(mat[r, e]))), (name, r, e)
            info, alloc_cumsum, e2s, alloc = ref_place(
                mat, held.cpu().long(), r, gate_pct, 110, slots
            )
            assert torch.equal(planner.info.cpu().long(), info), (name, r)
            assert torch.equal(
                planner.alloc_cumsum.cpu().long().view(E, R), alloc_cumsum
            ), (name, r)
            assert torch.equal(planner.expert_to_slot.cpu().long().view(R, E), e2s), (
                name,
                r,
            )
            assert torch.equal(
                vids, ref_rewrite(flat, li, info, alloc_cumsum, e2s, slots, interleave)
            ), (name, r)
            dest, row = vids[valid] // (EPN + slots), vids[valid] % (EPN + slots)
            slot_expert = held.cpu().long()[
                dest, (row - EPN).clamp(min=0, max=slots - 1)
            ]
            resolved = torch.where(row < EPN, dest * EPN + row, slot_expert)
            assert torch.equal(resolved, flat[valid]), (name, r)
            assert torch.equal(vids[~valid], flat[~valid])
            received.index_put_(
                (flat[valid], dest), torch.ones_like(dest), accumulate=True
            )
        if gate_pct == 110 and name == "uniform":
            assert bool(
                (
                    alloc
                    == torch.where(
                        torch.arange(E)[None, :] // EPN == torch.arange(R)[:, None],
                        ecount[None, :],
                        0,
                    )
                ).all()
            ), "a balanced batch stays at home"
        diff = (received - alloc.T).abs()
        assert int(diff.max()) <= (R if interleave else 0), (name, int(diff.max()))


@pytest.mark.parametrize("hyst", (0.0, 0.5, 2.0))
@pytest.mark.parametrize("slots", (2, 4))
def test_select_sticky_slots(route_sets, slots, hyst):
    """Consecutive forwards: select from held slots keeps wanted experts in place, and does nothing without
    place's refresh flag."""
    from aiter.ops.flydsl.kernels.mega_moe.replica_plan import ReplicaPlanner

    device = torch.device("cuda", torch.cuda.current_device())
    planner = ReplicaPlanner(
        rank=0,
        world_size=R,
        experts=E,
        slots=slots,
        max_routes=TOKENS_PER_RANK * TOPK,
        device=device,
        exchange="none",
        select_hyst=hyst,
    )
    held, placed = planner.new_held(), planner.new_held()
    ref_held = torch.full((R, slots), -1)
    for name in ("zipf", "hot_rank", "masked", "single_hot", "zipf", "uniform"):
        ecount = _histograms(route_sets[name]).sum(0)
        for flag in (1, 0):
            ecount_flag = (
                torch.cat([ecount, torch.tensor([flag])]).to(torch.int32).to(device)
            )
            before = held.cpu().clone()
            planner.select(held, placed, ecount_flag)
            ref_held, ref_placed = ref_select(ecount, ref_held, flag, hyst, slots)
            assert torch.equal(held.cpu().long(), ref_held), (name, flag)
            assert torch.equal(placed.cpu().long(), ref_placed), (name, flag)
            _check_held(held.cpu(), placed.cpu(), before, slots)
            if not flag:
                assert torch.equal(held.cpu(), before) and bool((placed == -1).all())
