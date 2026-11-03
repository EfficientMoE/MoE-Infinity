// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

bool gptoss_grouped_mxfp4_supported(int64_t tokens, int64_t top_k,
                                    int64_t experts, int64_t hidden,
                                    int64_t inter);

torch::Tensor gptoss_grouped_mxfp4(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& gu_blocks,
    const torch::Tensor& gu_scales, const torch::Tensor& gu_bias,
    const torch::Tensor& d_blocks, const torch::Tensor& d_scales,
    const torch::Tensor& d_bias, double alpha, double limit);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("supported", &gptoss_grouped_mxfp4_supported,
        "Whether the grouped MXFP4 decode kernel covers this shape");
  m.def("grouped_mxfp4_moe", &gptoss_grouped_mxfp4,
        "GPT-OSS resident MoE (MXFP4 experts) in two launches, no host sync");
}
