// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "latent_ag_mailbox.h"

#define LATENT_AG_MAILBOX_PYBIND                                                     \
    m.def("latent_ag_init",                                                          \
          &aiter::latent_ag_init,                                                    \
          py::arg("rank"),                                                           \
          py::arg("world_size"),                                                     \
          py::arg("max_m"),                                                          \
          py::arg("shard_n"));                                                       \
    m.def("latent_ag_destroy", &aiter::latent_ag_destroy, py::arg("fa"));            \
    m.def("latent_ag_get_handle",                                                    \
          &aiter::latent_ag_get_handle,                                              \
          py::arg("fa"),                                                             \
          py::arg("out_ptr"));                                                       \
    m.def("latent_ag_open_handles",                                                  \
          &aiter::latent_ag_open_handles,                                            \
          py::arg("fa"),                                                             \
          py::arg("handle_ptrs"));                                                   \
    m.def("latent_ag_mailbox_bytes",                                                 \
          &aiter::latent_ag_mailbox_bytes,                                           \
          py::arg("world_size"),                                                     \
          py::arg("max_m"),                                                          \
          py::arg("shard_n"));                                                       \
    m.def("latent_ag_push_poll",                                                     \
          &aiter::latent_ag_push_poll,                                               \
          py::arg("fa"),                                                             \
          py::arg("shard"),                                                          \
          py::arg("out"));

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    LATENT_AG_MAILBOX_PYBIND;
}
