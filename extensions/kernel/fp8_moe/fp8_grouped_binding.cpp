// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Standalone binding so the grouped FP8 kernel can be built and timed
// without relinking the whole _store extension. The same entry is also
// registered on _store from py_archer_prefetch.cpp.

#include <torch/extension.h>

#include "fp8_moe_grouped_sm120_pro6000.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_moe_grouped_forward_sm120_pro6000",
        &fp8_moe_grouped_forward_sm120_pro6000, py::arg("x"),
        py::arg("topk_ids"), py::arg("topk_w"), py::arg("table"),
        py::arg("inter"));
}
