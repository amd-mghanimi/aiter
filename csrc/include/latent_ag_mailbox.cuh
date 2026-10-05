#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Latent-MoE all-gather mailbox (F2 S2): LL flag-in-data publish of a
// column-shard, then poll-assemble into the full hidden row. First cut is
// push+poll only; the skinny GEMM epilogue can call the same publish path.

#include <cstdint>
#include <hip/hip_runtime.h>

#ifndef DINLINE
#define DINLINE __device__ __forceinline__
#endif

namespace aiter {
namespace latent_ag {

constexpr int kMaxRanks = 8;
constexpr int kMaxM     = 16;
constexpr int kDefaultShardN = 896; // 7168 / 8
constexpr int kBanks    = 2;

// 16-byte LL line: two (4B data, 4B flag) pairs = 8B payload.
union LLPackedMsg {
    struct {
        uint32_t data0;
        uint32_t flag0;
        uint32_t data1;
        uint32_t flag1;
    };
    uint4 raw;
};
static_assert(sizeof(LLPackedMsg) == 16, "LLPackedMsg must be 16 bytes");

using llx_v4u = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;

DINLINE void ll_store_b128(uint32_t* dst, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3)
{
    union {
        llx_v4u v;
        uint32_t w[4];
    } u;
    u.w[0] = a0;
    u.w[1] = a1;
    u.w[2] = a2;
    u.w[3] = a3;
    __builtin_nontemporal_store(u.v, reinterpret_cast<llx_v4u*>(dst));
    asm volatile("" ::: "memory");
}

DINLINE void ll_load_b128(
    const uint32_t* src, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3)
{
    asm volatile("" ::: "memory");
    union {
        llx_v4u v;
        uint32_t w[4];
    } u;
    u.v = __builtin_nontemporal_load(reinterpret_cast<llx_v4u*>(const_cast<uint32_t*>(src)));
    o0 = u.w[0];
    o1 = u.w[1];
    o2 = u.w[2];
    o3 = u.w[3];
}

// Per-rank mailbox bytes for max shape (LL-expanded).
inline size_t mailbox_bytes(int world, int max_m, int shard_n)
{
    // packets per shard = (max_m * shard_n * 2 bytes) / 8
    size_t pk_per_shard = (size_t)max_m * (size_t)shard_n * 2u / 8u;
    size_t pk_per_bank  = (size_t)world * pk_per_shard;
    return (size_t)kBanks * pk_per_bank * sizeof(LLPackedMsg);
}

// Publish shard[M, shard_n] bf16 into every peer's mailbox, then poll-assemble
// into out[M, shard_n * world]. Self is copied from shard (no self IPC write).
template <int BLOCK_SIZE = 256>
__global__ void __launch_bounds__(BLOCK_SIZE) latent_ag_push_poll_bf16(
    LLPackedMsg* const* __restrict__ peer_mailbox, // world bases
    const uint16_t* __restrict__ shard,            // [M, shard_n] bf16 bits
    uint16_t* __restrict__ out,                    // [M, shard_n * world]
    int M,
    int shard_n,
    int world,
    int rank,
    uint32_t* __restrict__ block_flags)
{
    const size_t pk_per_shard = (size_t)M * (size_t)shard_n * 2u / 8u; // 8B payload packets
    const size_t slot         = pk_per_shard; // one shard per src rank
    const size_t bank_stride  = (size_t)world * slot;

    __shared__ uint32_t s_flag;
    if (threadIdx.x == 0) {
        uint32_t f = block_flags[blockIdx.x] + 1u;
        if (f == 0u)
            f = 1u;
        s_flag = f;
    }
    __syncthreads();
    const uint32_t flag        = s_flag;
    const size_t bank_off_pkts = (size_t)(flag & 1u) * bank_stride;

    const size_t gtid   = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

    const uint32_t* in = reinterpret_cast<const uint32_t*>(shard);

    // Phase 1: publish my shard into every *other* peer's slot[rank].
    for (size_t pk = gtid; pk < pk_per_shard; pk += stride) {
        const uint32_t d0 = in[2 * pk];
        const uint32_t d1 = in[2 * pk + 1];
#pragma unroll
        for (int r = 1; r < world; ++r) {
            int peer = (rank + r) % world;
            LLPackedMsg* dst = peer_mailbox[peer] + bank_off_pkts + (size_t)rank * slot;
            ll_store_b128(reinterpret_cast<uint32_t*>(&dst[pk]), d0, flag, d1, flag);
        }
    }

    // Phase 2: poll my slots for the other ranks; copy self from shard.
    LLPackedMsg* my_base = peer_mailbox[rank] + bank_off_pkts;
    const size_t full_n  = (size_t)shard_n * (size_t)world;

    for (size_t pk = gtid; pk < pk_per_shard; pk += stride) {
        // Element index within the shard for this 8B packet (4 bf16).
        const size_t elem0 = pk * 4u;
        const size_t local_row = elem0 / (size_t)shard_n;
        const size_t local_col = elem0 % (size_t)shard_n;

        // Self: write 4 bf16 into out at column rank*shard_n + local_col.
        {
            const uint32_t w0 = in[2 * pk];
            const uint32_t w1 = in[2 * pk + 1];
            uint16_t vals[4];
            vals[0] = (uint16_t)(w0 & 0xffffu);
            vals[1] = (uint16_t)(w0 >> 16);
            vals[2] = (uint16_t)(w1 & 0xffffu);
            vals[3] = (uint16_t)(w1 >> 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                size_t c = local_col + (size_t)j;
                if (c < (size_t)shard_n) {
                    out[local_row * full_n + (size_t)rank * (size_t)shard_n + c] = vals[j];
                }
            }
        }

        uint32_t load_reg[kMaxRanks - 1][4];
        volatile LLPackedMsg* src[kMaxRanks - 1];
        int n_peers = world - 1;
#pragma unroll
        for (int i = 1; i < kMaxRanks; ++i) {
            if (i < world) {
                int peer   = (rank + i) % world;
                src[i - 1] = my_base + (size_t)peer * slot;
            }
        }
        bool still_loop;
        do {
            still_loop = false;
#pragma unroll
            for (int i = 0; i < kMaxRanks - 1; ++i) {
                if (i < n_peers) {
                    ll_load_b128(
                        reinterpret_cast<const uint32_t*>(
                            const_cast<LLPackedMsg*>(&src[i][pk])),
                        load_reg[i][0],
                        load_reg[i][1],
                        load_reg[i][2],
                        load_reg[i][3]);
                }
            }
#pragma unroll
            for (int i = 0; i < kMaxRanks - 1; ++i) {
                if (i < n_peers) {
                    still_loop = still_loop || load_reg[i][1] != flag;
                    still_loop = still_loop || load_reg[i][3] != flag;
                }
            }
        } while (still_loop);

#pragma unroll
        for (int i = 1; i < kMaxRanks; ++i) {
            if (i >= world)
                break;
            int peer = (rank + i) % world;
            uint16_t vals[4];
            vals[0] = (uint16_t)(load_reg[i - 1][0] & 0xffffu);
            vals[1] = (uint16_t)(load_reg[i - 1][0] >> 16);
            vals[2] = (uint16_t)(load_reg[i - 1][2] & 0xffffu);
            vals[3] = (uint16_t)(load_reg[i - 1][2] >> 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                size_t c = local_col + (size_t)j;
                if (c < (size_t)shard_n) {
                    out[local_row * full_n + (size_t)peer * (size_t)shard_n + c] = vals[j];
                }
            }
        }
    }

    if (threadIdx.x == 0)
        block_flags[blockIdx.x] = flag;
}

} // namespace latent_ag
} // namespace aiter
