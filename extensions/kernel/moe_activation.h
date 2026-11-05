// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Gated activation descriptor for routed-expert FFNs (gate/up/down).
// Dependency-free so CUTLASS epilogues, the expert module and other expert
// kernels can share the exact same math.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
  #define MOE_ACT_HOST_DEVICE __host__ __device__ __forceinline__
#else
  #define MOE_ACT_HOST_DEVICE inline
#endif

enum class MoEActivationKind : int {
  // silu(g) * u
  kSiLU = 0,
  // silu(min(g, limit)) * clamp(u, -limit, limit)
  // (GLM-5.3-Flash, DeepSeek-V4 swiglu_limit)
  kClampedSiLU = 1,
  // (clamp(u, -limit, limit) + 1) * g' * sigmoid(alpha * g'),
  // g' = min(g, limit)  (SwiGLU-OAI: MiniMax-M3, GPT-OSS)
  kSwiGLUOAI = 2,
};

struct MoEActivation {
  MoEActivationKind kind = MoEActivationKind::kSiLU;
  float alpha = 1.0f;
  float limit = 0.0f;
};

template <MoEActivationKind Kind>
MOE_ACT_HOST_DEVICE float moe_gated_act(float g, float u, float alpha,
                                        float limit) {
  if constexpr (Kind == MoEActivationKind::kSiLU) {
    return (g * (1.0f / (1.0f + expf(-g)))) * u;
  } else if constexpr (Kind == MoEActivationKind::kClampedSiLU) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    return (g * (1.0f / (1.0f + expf(-g)))) * u;
  } else {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    return (g * (1.0f / (1.0f + expf(-alpha * g)))) * (u + 1.0f);
  }
}

#undef MOE_ACT_HOST_DEVICE
