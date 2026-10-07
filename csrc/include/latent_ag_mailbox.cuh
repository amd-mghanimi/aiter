#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// All-gather for a column-parallel GEMM output.
//
// shard[M, shard_n] bf16 is one rank's N-slice. Each rank publishes that slice
// into the other ranks' IPC mailboxes (LL flag-in-data, 8 bytes of payload per
// 16-byte line), then polls its own mailbox and writes out[M, shard_n * world].
// The self slice is copied from the local shard; this rank does not IPC-write
// its own mailbox.
//
// The publish loop is the seam a skinny GEMM epilogue would replace. The poll
// and the mailbox layout stay.

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

// Unpack one 8-byte payload (two uint32) into four contiguous bf16 lanes.
// Valid when shard_n is a multiple of 4, so a packet never crosses a row.
DINLINE void store_bf16_packet(
    uint16_t* out, size_t row, size_t col, size_t full_n, uint32_t w0, uint32_t w1)
{
    const size_t base = row * full_n + col;
    out[base + 0]     = (uint16_t)(w0 & 0xffffu);
    out[base + 1]     = (uint16_t)(w0 >> 16);
    out[base + 2]     = (uint16_t)(w1 & 0xffffu);
    out[base + 3]     = (uint16_t)(w1 >> 16);
}

// Per-rank mailbox bytes for max shape (LL-expanded, two banks).
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

        // Self: four bf16 at column rank * shard_n + local_col.
        store_bf16_packet(
            out,
            local_row,
            (size_t)rank * (size_t)shard_n + local_col,
            full_n,
            in[2 * pk],
            in[2 * pk + 1]);

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
            store_bf16_packet(
                out,
                local_row,
                (size_t)peer * (size_t)shard_n + local_col,
                full_n,
                load_reg[i - 1][0],
                load_reg[i - 1][2]);
        }
    }

    if (threadIdx.x == 0)
        block_flags[blockIdx.x] = flag;
}

} // namespace latent_ag
} // namespace aiter
