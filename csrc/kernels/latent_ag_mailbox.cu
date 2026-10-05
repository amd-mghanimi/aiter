// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "latent_ag_mailbox.h"
#include "latent_ag_mailbox.cuh"
#include "aiter_stream.h"
#include "aiter_hip_common.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace aiter {
namespace {

using latent_ag::LLPackedMsg;
using latent_ag::kMaxRanks;
using latent_ag::mailbox_bytes;

struct LatentAgState {
    int rank;
    int world;
    int max_m;
    int shard_n;
    size_t mailbox_sz;
    void* local_mailbox = nullptr; // owned
    hipIpcMemHandle_t local_handle{};
    void* peer_ptrs_host[kMaxRanks]{};
    LLPackedMsg** peer_ptrs_dev = nullptr; // device table
    uint32_t* block_flags       = nullptr;
    int n_blocks                = 0;
    bool opened                 = false;

    ~LatentAgState()
    {
        if (peer_ptrs_dev)
            (void)hipFree(peer_ptrs_dev);
        if (block_flags)
            (void)hipFree(block_flags);
        if (local_mailbox) {
            for (int i = 0; i < world; ++i) {
                if (i == rank)
                    continue;
                if (peer_ptrs_host[i])
                    (void)hipIpcCloseMemHandle(peer_ptrs_host[i]);
            }
            (void)hipFree(local_mailbox);
        }
    }
};

} // namespace

fptr_t latent_ag_init(int64_t rank, int64_t world_size, int64_t max_m, int64_t shard_n)
{
    if (world_size < 2 || world_size > kMaxRanks)
        throw std::invalid_argument("latent_ag: world_size must be in [2, 8]");
    if (rank < 0 || rank >= world_size)
        throw std::invalid_argument("latent_ag: invalid rank");
    if (max_m <= 0 || max_m > latent_ag::kMaxM)
        throw std::invalid_argument("latent_ag: max_m must be in [1, 16]");
    if (shard_n <= 0 || (shard_n % 4) != 0)
        throw std::invalid_argument("latent_ag: shard_n must be > 0 and multiple of 4");

    auto* st       = new LatentAgState();
    st->rank       = (int)rank;
    st->world      = (int)world_size;
    st->max_m      = (int)max_m;
    st->shard_n    = (int)shard_n;
    st->mailbox_sz = mailbox_bytes(st->world, st->max_m, st->shard_n);

    HIP_CALL(hipExtMallocWithFlags(&st->local_mailbox, st->mailbox_sz, hipDeviceMallocUncached));
    HIP_CALL(hipMemset(st->local_mailbox, 0, st->mailbox_sz));
    HIP_CALL(hipIpcGetMemHandle(&st->local_handle, st->local_mailbox));
    st->peer_ptrs_host[st->rank] = st->local_mailbox;

    // Enough blocks for the max packet count.
    size_t pk = (size_t)st->max_m * (size_t)st->shard_n * 2u / 8u;
    st->n_blocks = (int)std::min<size_t>((pk + 255) / 256, 256);
    if (st->n_blocks < 1)
        st->n_blocks = 1;
    HIP_CALL(hipMalloc(&st->block_flags, sizeof(uint32_t) * (size_t)st->n_blocks));
    HIP_CALL(hipMemset(st->block_flags, 0, sizeof(uint32_t) * (size_t)st->n_blocks));
    HIP_CALL(hipMalloc(&st->peer_ptrs_dev, sizeof(LLPackedMsg*) * (size_t)st->world));

    return (fptr_t)st;
}

void latent_ag_destroy(fptr_t fa)
{
    delete reinterpret_cast<LatentAgState*>(fa);
}

void latent_ag_get_handle(fptr_t fa, int64_t out_ptr)
{
    auto* st = reinterpret_cast<LatentAgState*>(fa);
    std::memcpy((void*)out_ptr, &st->local_handle, sizeof(hipIpcMemHandle_t));
}

void latent_ag_open_handles(fptr_t fa, const std::vector<int64_t>& handle_ptrs)
{
    auto* st = reinterpret_cast<LatentAgState*>(fa);
    if ((int)handle_ptrs.size() != st->world)
        throw std::invalid_argument("latent_ag: handle count != world");
    for (int i = 0; i < st->world; ++i) {
        if (i == st->rank) {
            st->peer_ptrs_host[i] = st->local_mailbox;
            continue;
        }
        hipIpcMemHandle_t h{};
        std::memcpy(&h, (void*)handle_ptrs[i], sizeof(hipIpcMemHandle_t));
        void* ptr = nullptr;
        HIP_CALL(hipIpcOpenMemHandle(
            &ptr, h, hipIpcMemLazyEnablePeerAccess));
        st->peer_ptrs_host[i] = ptr;
    }
    // Upload device pointer table.
    LLPackedMsg* table[kMaxRanks]{};
    for (int i = 0; i < st->world; ++i)
        table[i] = reinterpret_cast<LLPackedMsg*>(st->peer_ptrs_host[i]);
    HIP_CALL(hipMemcpy(
        st->peer_ptrs_dev,
        table,
        sizeof(LLPackedMsg*) * (size_t)st->world,
        hipMemcpyHostToDevice));
    st->opened = true;
}

int64_t latent_ag_mailbox_bytes(int64_t world_size, int64_t max_m, int64_t shard_n)
{
    return (int64_t)mailbox_bytes((int)world_size, (int)max_m, (int)shard_n);
}

void latent_ag_push_poll(fptr_t fa, const aiter_tensor_t& shard, const aiter_tensor_t& out)
{
    auto* st = reinterpret_cast<LatentAgState*>(fa);
    if (!st->opened)
        throw std::runtime_error("latent_ag: open_handles not called");
    if (shard.dtype() != AITER_DTYPE_bf16 || out.dtype() != AITER_DTYPE_bf16)
        throw std::invalid_argument("latent_ag: bf16 only");
    if (shard.dim() != 2 || out.dim() != 2)
        throw std::invalid_argument("latent_ag: expect 2-D tensors");
    if (!shard.is_contiguous() || !out.is_contiguous())
        throw std::invalid_argument("latent_ag: tensors must be contiguous");
    int M       = (int)shard.size(0);
    int shard_n = (int)shard.size(1);
    if (M > st->max_m || shard_n != st->shard_n)
        throw std::invalid_argument("latent_ag: shape exceeds init caps");
    if ((int)out.size(0) != M || (int)out.size(1) != shard_n * st->world)
        throw std::invalid_argument("latent_ag: out shape mismatch");
    if (((size_t)M * (size_t)shard_n * 2u) % 8u != 0)
        throw std::invalid_argument("latent_ag: shard bytes must be multiple of 8");

    hipStream_t stream = aiter::getCurrentHIPStream();
    size_t pk          = (size_t)M * (size_t)shard_n * 2u / 8u;
    int blocks         = (int)std::min<size_t>((pk + 255) / 256, (size_t)st->n_blocks);
    if (blocks < 1)
        blocks = 1;

    hipLaunchKernelGGL(
        (latent_ag::latent_ag_push_poll_bf16<256>),
        dim3(blocks),
        dim3(256),
        0,
        stream,
        st->peer_ptrs_dev,
        reinterpret_cast<const uint16_t*>(shard.data_ptr()),
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        M,
        shard_n,
        st->world,
        st->rank,
        st->block_flags);
}

} // namespace aiter
