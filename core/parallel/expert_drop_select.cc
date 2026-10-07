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
  if (params.mass_budget == 0.0 && params.adaptive_slope == 0.0 &&
      params.head_budget == 0.0)
    return counts;

  ExpertDropScratch local;
  ExpertDropScratch& s = scratch ? *scratch : local;

  for (int r = 0; r < rows; ++r) {
    ++counts.tokens_seen;
    float* w = weights + static_cast<std::size_t>(r) * num_experts;
    std::uint8_t* m = routed_mask + static_cast<std::size_t>(r) * num_experts;
    bool row_changed = false;

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

    const bool guards_ok = (selected_count > params.min_k) && finite_ok &&
                           nonneg_ok && (max_w > 0.0f);
    bool flat = false;
    if (guards_ok) {
      // Ratio formed from the float32 values widened to double; strict ">"
      // bypasses only strictly flatter rows (reference parity).
      flat = static_cast<double>(min_w) / static_cast<double>(max_w) >
             params.flatness_floor;
      if (flat) ++counts.flatness_bypasses;
    }

    if (guards_ok && !flat) {
      // Policy D: per-token adaptive budget grows with this token's miss
      // count, so fidelity is spent only on the tokens that are already slow.
      double row_budget = params.mass_budget;
      if (params.adaptive_slope > 0.0) {
        int miss = 0;
        for (int e = 0; e < num_experts; ++e) {
          if (m[e] && !resident[e]) ++miss;
        }
        const int over = std::max(miss - params.adaptive_miss0, 0);
        row_budget = params.mass_budget + params.adaptive_slope * over;
        if (row_budget > 1.0) row_budget = 1.0;
      }

      if (row_budget > 0.0) {
        s.candidate_ids.clear();
        for (int e = num_experts - 1; e >= 0; --e) {
          if (m[e] && !resident[e]) s.candidate_ids.push_back(e);
        }
        if (!s.candidate_ids.empty()) {
          // Descending-id prefill + stable ascending-weight sort reproduces
          // the reference tie-break: lower weight first, higher id on ties.
          std::stable_sort(s.candidate_ids.begin(), s.candidate_ids.end(),
                           [&](int a, int b) { return w[a] < w[b]; });
          double cumulative = 0.0;
          int budget_count = 0;
          for (int id : s.candidate_ids) {
            cumulative += static_cast<double>(w[id]);
            if (cumulative <= row_budget) {
              ++budget_count;
            } else {
              break;
            }
          }
          const int dropped_here =
              std::min(budget_count, selected_count - params.min_k);
          if (dropped_here > 0) {
            for (int i = 0; i < dropped_here; ++i) {
              const int id = s.candidate_ids[i];
              m[id] = 0;
              w[id] = 0.0f;
            }
            counts.experts_dropped += dropped_here;
            row_changed = true;
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
        }
      }
    }

    // Policy C: head-budget hybrid -- additionally drop ONE high-mass expert
    // iff it is the sole non-resident kept on the critical path and its
    // post-base renormalized mass is within a small separate head_budget.
    if (params.head_budget > 0.0) {
      int kept_count = 0;
      int nonres_kept = 0;
      int sole = -1;
      for (int e = 0; e < num_experts; ++e) {
        if (!m[e]) continue;
        ++kept_count;
        if (!resident[e]) {
          ++nonres_kept;
          sole = e;
        }
      }
      if (nonres_kept == 1 && kept_count > params.min_k &&
          static_cast<double>(w[sole]) <= params.head_budget) {
        m[sole] = 0;
        w[sole] = 0.0f;
        ++counts.experts_dropped;
        row_changed = true;
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
    }

    if (row_changed) ++counts.tokens_changed;
  }
  return counts;
}

}  // namespace moe
