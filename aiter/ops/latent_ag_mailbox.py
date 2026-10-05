# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Latent-MoE all-gather mailbox (F2 S2).

LL flag-in-data publish of a column shard, then poll-assemble into the full
hidden. Host API for the HIP kernels in ``module_latent_ag_mailbox``.
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
        keep: list[torch.Tensor] = []
        ptrs: list[int] = []
        for h in handles:
            assert h is not None and len(h) == _HIP_IPC_HANDLE_BYTES
            t = torch.tensor(list(h), dtype=torch.uint8)
            keep.append(t)
            ptrs.append(int(t.data_ptr()))
        self._handle_keep = keep
        latent_ag_open_handles(self._fa, ptrs)

    def allgather(self, shard: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if shard.dtype != torch.bfloat16 or shard.dim() != 2:
            raise ValueError("shard must be bf16 [M, shard_n]")
        if shard.shape[1] != self.shard_n:
            raise ValueError(f"shard_n mismatch: {shard.shape[1]} vs {self.shard_n}")
        if shard.shape[0] > self.max_m:
            raise ValueError(f"M={shard.shape[0]} > max_m={self.max_m}")
        if out is None:
            out = torch.empty(
                shard.shape[0],
                self.shard_n * self.world_size,
                dtype=torch.bfloat16,
                device=shard.device,
            )
        latent_ag_push_poll(self._fa, shard.contiguous(), out)
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
