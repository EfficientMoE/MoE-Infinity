// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// Which grouped MoE kernel (if any) runs the GPU-resident experts of a layer
// (#306 decode, #307 prefill). Header-only and free of CUDA / torch so the
// selection can be unit-tested on the CPU (tests/cpp/unit/parallel).
// moe_infinity/runtime/grouped_moe_pro6000.py mirrors it for Python callers.

#pragma once

#include <algorithm>
#include <cstdint>

namespace moe {

enum class GroupedMoeMode { kNone, kDecode, kPrefill };

// Default token count from which the grouped prefill path is considered. Below
// the grouped decode limit the decode kernels run; this is the first prefill
// size of the kernel audit (#297) where the per-expert path leaves the BF16
// roof (M = tokens * top_k / experts is in the hundreds).
constexpr std::int64_t kGroupedPrefillDefaultMinTokens = 2048;

// Both grouped kernels size a [tokens * top_k, inter] bf16 activation
// workspace; grouped_moe_prefill keeps one per exec stream for the process
// lifetime, so the cap is deliberately the decode path's 1 GiB (about 12K
// tokens of a top-8, inter = 2048 model).
constexpr std::int64_t kGroupedMaxActBytes = std::int64_t(1) << 30;

struct GroupedPrefillPolicy {
  bool enabled = false;  // opt-in until validated on a GPU
  std::int64_t min_tokens = kGroupedPrefillDefaultMinTokens;
};

// decode_max_tokens: largest batch the grouped decode kernels accept.
// Batches in (decode_max_tokens, min_tokens) and everything when the prefill
// path is off keep per-expert execution.
inline GroupedMoeMode SelectGroupedMoeMode(
    std::int64_t tokens, std::int64_t decode_max_tokens,
    const GroupedPrefillPolicy& prefill) {
  if (tokens <= 0) return GroupedMoeMode::kNone;
  if (tokens <= decode_max_tokens) return GroupedMoeMode::kDecode;
  if (!prefill.enabled) return GroupedMoeMode::kNone;
  const std::int64_t floor =
      std::max<std::int64_t>(prefill.min_tokens, decode_max_tokens + 1);
  return tokens >= floor ? GroupedMoeMode::kPrefill : GroupedMoeMode::kNone;
}

// True when the [tokens * kcap, inter] bf16 activation workspace fits.
inline bool GroupedWorkspaceFits(std::int64_t tokens, std::int64_t kcap,
                                 std::int64_t inter) {
  return tokens * kcap * inter * 2 <= kGroupedMaxActBytes;
}

}  // namespace moe
