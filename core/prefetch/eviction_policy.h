// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// Selectable GPU expert-cache eviction policies.
//
// Each policy is a C++ port of the matching class in
// benchmarks/cache_simulator/policies.py and replays that simulator's demand
// paging bit-for-bit through SimulateEvictionPolicy (checked by
// tests/python/unit/test_eviction_policy_parity.py). The engine drives the same
// object through ExpertResidencyManager: OnAccess for every routed expert in
// execution order, SelectVictim when an admission needs room (restricted to
// victims the manager can evict right now), then OnEvict / OnAdmit as the
// manager moves experts. This file has no CUDA or torch dependency.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// (layer << 32) | expert, matching the simulator's (layer, expert) ordering.
using ExpertCacheKey = std::uint64_t;

inline ExpertCacheKey MakeExpertCacheKey(int layer, int expert) {
  return (static_cast<ExpertCacheKey>(static_cast<std::uint32_t>(layer))
          << 32) |
         static_cast<std::uint32_t>(expert);
}

inline int ExpertCacheKeyLayer(ExpertCacheKey key) {
  return static_cast<int>(key >> 32);
}

struct EvictionPolicyOptions {
  // Number of MoE layers in one decode step; 0 = largest layer id seen + 1.
  int num_layers = 0;
  // LCNU: EMA rate, EMA weight, and prior routing probability (top_k /
  // experts).
  double lcnu_alpha = 0.3;
  double lcnu_beta = 0.5;
  double lcnu_prior = 1.0 / 16;
};

class EvictionPolicy {
 public:
  using EligibleFn = std::function<bool(ExpertCacheKey)>;

  virtual ~EvictionPolicy() = default;

  virtual const char* Name() const = 0;

  // Cache size in experts; sizes the internal queues of the classic policies.
  virtual void SetCapacity(std::size_t slots) { capacity_ = slots; }
  std::size_t Capacity() const { return capacity_; }

  // A routed (demand) access in execution order. `resident` tells whether the
  // expert is cached right now; a miss is followed by OnAdmit once it is.
  virtual void OnAccess(ExpertCacheKey key, bool resident) = 0;

  // Whether admitting `incoming` should evict even though the cache has room
  // (per-layer quotas, S3-FIFO's second eviction).
  virtual bool WantsVictim(ExpertCacheKey /*incoming*/) const { return false; }

  // Pick a resident to evict for admitting `incoming`, among keys for which
  // `eligible` is true. `need_room` is false when the cache has room and the
  // call only serves WantsVictim(). Returns nullopt when nothing should or
  // can be evicted.
  virtual std::optional<ExpertCacheKey> SelectVictim(
      ExpertCacheKey incoming, bool need_room, const EligibleFn& eligible) = 0;

  // `key` left the cache.
  virtual void OnEvict(ExpertCacheKey key) = 0;

  // `key` became resident; `prefetch` = admitted speculatively, before any
  // demand access.
  virtual void OnAdmit(ExpertCacheKey key, bool prefetch) = 0;

 protected:
  std::size_t capacity_ = 0;
};

// Names accepted by CreateEvictionPolicy, in documentation order.
const std::vector<std::string>& EvictionPolicyNames();

// nullptr for unknown names ("legacy" is not a policy object).
std::unique_ptr<EvictionPolicy> CreateEvictionPolicy(
    const std::string& name, const EvictionPolicyOptions& options);

// Replays `accesses` through a fresh policy as a demand-paging cache of
// `capacity` experts (the simulator's `simulate()` semantics) and returns the
// number of hits.
std::size_t SimulateEvictionPolicy(const std::string& name,
                                   const std::vector<ExpertCacheKey>& accesses,
                                   std::size_t capacity,
                                   const EvictionPolicyOptions& options);
