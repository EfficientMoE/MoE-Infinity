// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include "memory/kv_cache_pool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<archer::KVCachePool>(m, "kv_cache_pool")
      .def(py::init<size_t, int>(), py::arg("block_size_bytes"),
           py::arg("num_blocks"))
      .def("allocate", &archer::KVCachePool::Allocate)
      .def("free", &archer::KVCachePool::Free, py::arg("block_id"))
      .def("async_copy_to_cpu", &archer::KVCachePool::AsyncCopyToCPU,
           py::arg("tensor"), py::arg("block_id"), py::arg("stream_idx"))
      .def("async_copy_to_gpu", &archer::KVCachePool::AsyncCopyToGPU,
           py::arg("block_id"), py::arg("tensor"), py::arg("stream_idx"))
      .def("sync_stream", &archer::KVCachePool::SyncStream,
           py::arg("stream_idx"))
      .def("num_blocks", &archer::KVCachePool::NumBlocks)
      .def("block_size_bytes", &archer::KVCachePool::BlockSizeBytes);
}
