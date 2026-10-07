# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""All-gather of a column-parallel GEMM shard (latent-MoE up projection).

Each rank holds ``shard`` of shape ``[M, shard_n]`` bf16, the local N-slice
of a skinny GEMM (Kimi-K3: ``shard_n = 7168 / TP``, ``M <= 16``). The kernel
publishes that slice into peer IPC mailboxes as LL flag-in-data packets, then
polls and writes the full hidden ``[M, shard_n * world]``.

The GEMM writes a column shard and this collective all-gathers it. The
row-parallel o_proj path is separate: that GEMM writes the full ``N`` and
QuickReduce sums the partials.

``develop=True`` is required: the JIT module is ``torch_exclude`` and the
push/poll entry takes ``aiter_tensor_t``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..jit.core import compile_ops

MD_NAME = "module_latent_ag_mailbox"
_HIP_IPC_HANDLE_BYTES = 64


@compile_ops(MD_NAME, develop=True)
def latent_ag_init(rank: int, world_size: int, max_m: int, shard_n: int) -> int: ...


@compile_ops(MD_NAME, develop=True)
def latent_ag_destroy(fa: int) -> None: ...


@compile_ops(MD_NAME, develop=True)
def latent_ag_get_handle(fa: int, out_ptr: int) -> None: ...


@compile_ops(MD_NAME, develop=True)
def latent_ag_open_handles(fa: int, handle_ptrs: list[int]) -> None: ...


@compile_ops(MD_NAME, develop=True)
def latent_ag_mailbox_bytes(world_size: int, max_m: int, shard_n: int) -> int: ...


@compile_ops(MD_NAME, develop=True)
def latent_ag_push_poll(fa: int, shard: torch.Tensor, out: torch.Tensor) -> None: ...


class LatentAgMailbox:
    """IPC mailbox + push/poll all-gather for a column-parallel up-proj shard."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        max_m: int = 16,
        shard_n: int = 896,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        if world_size < 2 or world_size > 8:
            raise ValueError("world_size must be in [2, 8]")
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.max_m = max_m
        self.shard_n = shard_n
        self.group = group
        self._fa = int(latent_ag_init(rank, world_size, max_m, shard_n))
        self._closed = False
        self._exchange_handles()

    @property
    def mailbox_bytes(self) -> int:
        return int(latent_ag_mailbox_bytes(self.world_size, self.max_m, self.shard_n))

    def _exchange_handles(self) -> None:
        buf = torch.empty(_HIP_IPC_HANDLE_BYTES, dtype=torch.uint8, device="cpu")
        latent_ag_get_handle(self._fa, int(buf.data_ptr()))
        mine = bytes(buf.tolist())
        handles: list[bytes | None] = [None] * self.world_size
        dist.all_gather_object(handles, mine, group=self.group)
        # Keep the tensors alive: open_handles reads the pointers, it does not copy.
        keep: list[torch.Tensor] = []
        ptrs: list[int] = []
        for h in handles:
            if h is None or len(h) != _HIP_IPC_HANDLE_BYTES:
                got = None if h is None else len(h)
                raise ValueError(f"IPC handle must be {_HIP_IPC_HANDLE_BYTES} bytes, got {got}")
            t = torch.tensor(list(h), dtype=torch.uint8)
            keep.append(t)
            ptrs.append(int(t.data_ptr()))
        self._handle_keep = keep
        latent_ag_open_handles(self._fa, ptrs)

    def allgather(self, shard: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Gather column shards. ``shard`` is ``[M, shard_n]``; result is ``[M, shard_n * world]``."""
        if self._closed:
            raise RuntimeError("LatentAgMailbox is closed")
        if shard.dtype != torch.bfloat16 or shard.dim() != 2:
            raise ValueError("shard must be bf16 [M, shard_n]")
        if shard.device != self.device:
            raise ValueError(f"shard device {shard.device} != mailbox device {self.device}")
        m = int(shard.shape[0])
        if int(shard.shape[1]) != self.shard_n:
            raise ValueError(f"shard_n mismatch: {shard.shape[1]} vs {self.shard_n}")
        if m > self.max_m or m < 1:
            raise ValueError(f"M={m} is outside [1, {self.max_m}]")
        full_n = self.shard_n * self.world_size
        if out is None:
            out = torch.empty(m, full_n, dtype=torch.bfloat16, device=shard.device)
        elif (
            out.dtype != torch.bfloat16
            or out.dim() != 2
            or tuple(out.shape) != (m, full_n)
            or out.device != shard.device
            or not out.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous bf16 [{m}, {full_n}] on {shard.device}")
        latent_ag_push_poll(self._fa, shard if shard.is_contiguous() else shard.contiguous(), out)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        latent_ag_destroy(self._fa)
        self._fa = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
