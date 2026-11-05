// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// Replays access traces through the engine's eviction policies for
// tests/python/unit/test_eviction_policy_parity.py, which compares the hit
// counts with benchmarks/cache_simulator/policies.py.
//
// stdin:  <n> then n "<layer> <expert>" pairs, then <q> and q lines of
//         "<policy> <capacity> <num_layers> <alpha> <beta> <prior>"
// stdout: one hit count per query line ("-1" for an unknown policy)
// With --names: print EvictionPolicyNames(), one per line.

#include <iostream>
#include <string>
#include <vector>

#include "prefetch/eviction_policy.h"

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--names") {
    for (const auto& name : EvictionPolicyNames()) std::cout << name << "\n";
    return 0;
  }
  std::size_t n = 0;
  std::cin >> n;
  std::vector<ExpertCacheKey> accesses;
  accesses.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    int layer = 0;
    int expert = 0;
    std::cin >> layer >> expert;
    accesses.push_back(MakeExpertCacheKey(layer, expert));
  }
  std::size_t queries = 0;
  std::cin >> queries;
  for (std::size_t q = 0; q < queries; ++q) {
    std::string name;
    std::size_t capacity = 0;
    EvictionPolicyOptions options;
    std::cin >> name >> capacity >> options.num_layers >> options.lcnu_alpha >>
        options.lcnu_beta >> options.lcnu_prior;
    if (!CreateEvictionPolicy(name, options)) {
      std::cout << -1 << "\n";
      continue;
    }
    std::cout << SimulateEvictionPolicy(name, accesses, capacity, options)
              << "\n";
  }
  return 0;
}
