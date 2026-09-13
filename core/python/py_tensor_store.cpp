// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/extension.h>
#include "prefetch/archer_prefetch_handle.h"

// Store-facing bindings, split from the engine bindings in
// py_archer_prefetch.cpp. These methods form the checkpoint/tensor-store
// surface (registration, offload, index snapshot, derivative overlays) that
// relocates to the moe-store repository; the engine surface (prefetch,
// dispatch, cache, topology) stays in MoE-Infinity.
void BindTensorStoreSurface(py::class_<ArcherPrefetchHandle>& cls) {
  cls.def("offload", &ArcherPrefetchHandle::OffloadTensor)
      .def("register", (void(ArcherPrefetchHandle::*)(torch::Tensor&,
                                                      const std::uint32_t)) &
                           ArcherPrefetchHandle::RegisterTensor)
      .def("register", (void(ArcherPrefetchHandle::*)(torch::Tensor*)) &
                           ArcherPrefetchHandle::RegisterTensor)
      .def("update_tensor_map",
           (void(ArcherPrefetchHandle::*)(std::uint64_t, std::uint64_t)) &
               ArcherPrefetchHandle::UpdateTensorMap)
      .def("is_tensor_offloaded", &ArcherPrefetchHandle::IsTensorOffloaded)
      .def("is_tensor_index_initialized",
           &ArcherPrefetchHandle::IsTensorIndexInitialized)
      .def("get_canonical_tensor_index_snapshot",
           &ArcherPrefetchHandle::GetCanonicalTensorIndexSnapshot)
      .def("begin_derivative_overlay",
           &ArcherPrefetchHandle::BeginDerivativeOverlay)
      .def("register_derivative_tensor",
           &ArcherPrefetchHandle::RegisterDerivativeTensor)
      .def("commit_derivative_overlay",
           &ArcherPrefetchHandle::CommitDerivativeOverlay)
      .def("abort_derivative_overlay",
           &ArcherPrefetchHandle::AbortDerivativeOverlay);
}
