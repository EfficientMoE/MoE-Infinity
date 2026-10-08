// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// Parity + perf harness for core/parallel/expert_drop_select.cc against the
// Python-reference fixture written by gen_expert_drop_fixture.py.
// Standalone (no gtest): exits non-zero on any mismatch.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../../core/parallel/expert_drop_select.h"

namespace {

// libstdc++ operator>> rejects "nan"/"inf"; route floats through strtof.
float ReadFloat(std::ifstream& in) {
  std::string token;
  in >> token;
  return std::strtof(token.c_str(), nullptr);
}

bool CloseEnough(float got, float want) {
  if (std::isnan(want)) return std::isnan(got);
  const double diff = std::fabs(static_cast<double>(got) - want);
  return diff <= 1e-7 + 1e-5 * std::fabs(static_cast<double>(want));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <fixture>\n", argv[0]);
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }

  int num_cases = 0;
  in >> num_cases;
  int failures = 0;
  for (int c = 0; c < num_cases; ++c) {
    int rows, experts;
    moe::ExpertDropParams params;
    in >> rows >> experts >> params.min_k >> params.mass_budget >>
        params.flatness_floor;
    const std::size_t n = static_cast<std::size_t>(rows) * experts;
    std::vector<float> weights(n);
    std::vector<std::uint8_t> mask(n);
    std::vector<std::uint8_t> resident(experts);
    std::vector<float> exp_weights(n);
    std::vector<std::uint8_t> exp_mask(n);
    for (auto& v : weights) v = ReadFloat(in);
    for (auto& v : mask) {
      int b;
      in >> b;
      v = static_cast<std::uint8_t>(b);
    }
    for (auto& v : resident) {
      int b;
      in >> b;
      v = static_cast<std::uint8_t>(b);
    }
    for (auto& v : exp_weights) v = ReadFloat(in);
    for (auto& v : exp_mask) {
      int b;
      in >> b;
      v = static_cast<std::uint8_t>(b);
    }
    moe::ExpertDropCounts want;
    in >> want.tokens_seen >> want.tokens_changed >> want.experts_dropped >>
        want.flatness_bypasses;

    auto got = moe::SelectExpertDrops(rows, experts, weights.data(),
                                      mask.data(), resident.data(), params);
    bool ok = got.tokens_seen == want.tokens_seen &&
              got.tokens_changed == want.tokens_changed &&
              got.experts_dropped == want.experts_dropped &&
              got.flatness_bypasses == want.flatness_bypasses;
    for (std::size_t i = 0; ok && i < n; ++i) {
      ok = mask[i] == exp_mask[i] && CloseEnough(weights[i], exp_weights[i]);
    }
    if (!ok) {
      ++failures;
      std::fprintf(
          stderr,
          "case %d FAIL (rows=%d experts=%d): counts got "
          "%lld/%lld/%lld/%lld want %lld/%lld/%lld/%lld\n",
          c, rows, experts, (long long)got.tokens_seen,
          (long long)got.tokens_changed, (long long)got.experts_dropped,
          (long long)got.flatness_bypasses, (long long)want.tokens_seen,
          (long long)want.tokens_changed, (long long)want.experts_dropped,
          (long long)want.flatness_bypasses);
    }
  }
  std::printf("parity: %d/%d cases passed\n", num_cases - failures, num_cases);

  const int rows = 1, experts = 64, iters = 10000;
  std::vector<float> w(experts, 0.0f);
  std::vector<std::uint8_t> m(experts, 0);
  std::vector<std::uint8_t> res(experts, 0);
  for (int e = 0; e < 8; ++e) {
    m[e * 7] = 1;
    w[e * 7] = 0.125f;
  }
  moe::ExpertDropParams params{1, 0.05, 0.5};
  moe::ExpertDropScratch scratch;
  std::vector<float> w0 = w;
  std::vector<std::uint8_t> m0 = m;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    w = w0;
    m = m0;
    moe::SelectExpertDrops(rows, experts, w.data(), m.data(), res.data(),
                           params, &scratch);
  }
  const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count() /
                    iters;
  std::printf("perf: %.3f us/call at rows=1 experts=64 (bound 5 us)\n", us);
  if (us > 5.0) {
    std::fprintf(stderr, "perf bound exceeded\n");
    return 1;
  }
  return failures == 0 ? 0 : 1;
}
