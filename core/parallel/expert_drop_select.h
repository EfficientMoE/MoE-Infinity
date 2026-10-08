// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <vector>

namespace moe {

struct ExpertDropParams {
  int min_k = 1;
  double mass_budget = 0.0;
  double flatness_floor = 0.5;
  double head_budget = 0.0;
  double adaptive_slope = 0.0;
  int adaptive_miss0 = 0;
};

struct ExpertDropCounts {
  std::int64_t tokens_seen = 0;
  std::int64_t tokens_changed = 0;
  std::int64_t experts_dropped = 0;
  std::int64_t flatness_bypasses = 0;
};

struct ExpertDropScratch {
  std::vector<int> candidate_ids;
  std::vector<float> selected_weights;
};

// Host-side port of moe_infinity/runtime/expert_drop.py::select_expert_drops.
// weights and routed_mask are row-major [rows x num_experts]; resident is
// [num_experts]. Mutates weights/routed_mask in place (drop -> mask 0,
// weight 0, survivors renormalized by their float32 sum). Fail-open per row:
// non-finite or negative selected weights, zero max weight, count <= min_k,
// flatness ratio strictly above the floor, or no non-resident candidates
// leave the row untouched.
ExpertDropCounts SelectExpertDrops(int rows, int num_experts, float* weights,
                                   std::uint8_t* routed_mask,
                                   const std::uint8_t* resident,
                                   const ExpertDropParams& params,
                                   ExpertDropScratch* scratch = nullptr);

}  // namespace moe
