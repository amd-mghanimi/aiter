#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"
#include <cstdint>
#include <vector>

using fptr_t = int64_t;

namespace aiter {

// Opaque handle for the column-shard all-gather (IPC mailboxes + epoch flags).
fptr_t latent_ag_init(int64_t rank, int64_t world_size, int64_t max_m, int64_t shard_n);
void latent_ag_destroy(fptr_t fa);
void latent_ag_get_handle(fptr_t fa, int64_t out_ptr);
void latent_ag_open_handles(fptr_t fa, const std::vector<int64_t>& handle_ptrs);
int64_t latent_ag_mailbox_bytes(int64_t world_size, int64_t max_m, int64_t shard_n);

// shard: [M, shard_n] bf16 contiguous. out: [M, shard_n * world] bf16.
void latent_ag_push_poll(fptr_t fa, const aiter_tensor_t& shard, const aiter_tensor_t& out);

} // namespace aiter
