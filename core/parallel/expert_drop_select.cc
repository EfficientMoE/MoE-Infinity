// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_drop_select.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace moe {

ExpertDropCounts SelectExpertDrops(int rows, int num_experts, float* weights,
                                   std::uint8_t* routed_mask,
                                   const std::uint8_t* resident,
                                   const ExpertDropParams& params,
                                   ExpertDropScratch* scratch) {
  ExpertDropCounts counts;
  if (params.mass_budget == 0.0) return counts;

  ExpertDropScratch local;
  ExpertDropScratch& s = scratch ? *scratch : local;

  for (int r = 0; r < rows; ++r) {
    ++counts.tokens_seen;
    float* w = weights + static_cast<std::size_t>(r) * num_experts;
    std::uint8_t* m = routed_mask + static_cast<std::size_t>(r) * num_experts;

    int selected_count = 0;
    bool finite_ok = true;
    bool nonneg_ok = true;
    float max_w = -std::numeric_limits<float>::infinity();
    float min_w = std::numeric_limits<float>::infinity();
    for (int e = 0; e < num_experts; ++e) {
      if (!m[e]) continue;
      ++selected_count;
      const float v = w[e];
      if (!std::isfinite(v)) finite_ok = false;
      if (v < 0.0f) nonneg_ok = false;
      max_w = std::max(max_w, v);
      min_w = std::min(min_w, v);
    }
    if (selected_count <= params.min_k) continue;
    if (!finite_ok || !nonneg_ok) continue;
    if (max_w == 0.0f) continue;
    // Match the reference: the ratio is formed from the float32 values
    // widened to double; strict ">" bypasses only strictly flatter rows.
    if (static_cast<double>(min_w) / static_cast<double>(max_w) >
        params.flatness_floor) {
      ++counts.flatness_bypasses;
      continue;
    }

    s.candidate_ids.clear();
    for (int e = num_experts - 1; e >= 0; --e) {
      if (m[e] && !resident[e]) s.candidate_ids.push_back(e);
    }
    if (s.candidate_ids.empty()) continue;
    // Descending-id prefill + stable ascending-weight sort reproduces the
    // reference tie-break: lower weight first, equal weights higher id first.
    std::stable_sort(s.candidate_ids.begin(), s.candidate_ids.end(),
                     [&](int a, int b) { return w[a] < w[b]; });

    double cumulative = 0.0;
    int budget_count = 0;
    for (int id : s.candidate_ids) {
      cumulative += static_cast<double>(w[id]);
      if (cumulative <= params.mass_budget) {
        ++budget_count;
      } else {
        break;
      }
    }
    const int dropped_here =
        std::min(budget_count, selected_count - params.min_k);
    if (dropped_here == 0) continue;

    for (int i = 0; i < dropped_here; ++i) {
      const int id = s.candidate_ids[i];
      m[id] = 0;
      w[id] = 0.0f;
    }
    ++counts.tokens_changed;
    counts.experts_dropped += dropped_here;

    float surviving_sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
      if (m[e]) surviving_sum += w[e];
    }
    if (surviving_sum != 0.0f) {
      for (int e = 0; e < num_experts; ++e) {
        if (m[e]) w[e] /= surviving_sum;
      }
    }
  }
  return counts;
}

}  // namespace moe
