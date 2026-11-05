// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// Parity + perf harness for core/parallel/expert_drop_select.cc against the
// Python-reference fixture written by gen_expert_drop_fixture.py.
// Standalone (no gtest): exits non-zero on any mismatch.
//
// Fixture validation is strict: an empty, truncated, or otherwise
// malformed fixture is rejected with exit code 2 ("parity: 0/0 cases
// passed" can never be a passing result).
//
// The perf section times two separate rows with independent bounds:
//   - drop-producing: a skewed row whose selector call must drop experts
//     every iteration (exercises candidate collection, sort, budget walk,
//     write-back, and renormalization);
//   - flatness-bypass: a uniform row that must take the early flatness
//     bypass every iteration (exercises only the guard scan).
// Each row asserts its expected per-call counts so the timing cannot
// silently measure the wrong path.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../../core/parallel/expert_drop_select.h"

namespace {

bool g_read_ok = true;

// libstdc++ operator>> rejects "nan"/"inf"; route floats through strtof.
// A failed extraction (truncated fixture) poisons g_read_ok instead of
// silently yielding 0.
float ReadFloat(std::ifstream& in) {
  std::string token;
  if (!(in >> token)) {
    g_read_ok = false;
    return 0.0f;
  }
  char* end = nullptr;
  const float v = std::strtof(token.c_str(), &end);
  if (end == token.c_str()) g_read_ok = false;
  return v;
}

int ReadInt(std::ifstream& in) {
  long long v = 0;
  if (!(in >> v)) g_read_ok = false;
  return static_cast<int>(v);
}

long long ReadInt64(std::ifstream& in) {
  long long v = 0;
  if (!(in >> v)) g_read_ok = false;
  return v;
}

double ReadDouble(std::ifstream& in) {
  double v = 0.0;
  if (!(in >> v)) g_read_ok = false;
  return v;
}

bool CloseEnough(float got, float want) {
  if (std::isnan(want)) return std::isnan(got);
  const double diff = std::fabs(static_cast<double>(got) - want);
  return diff <= 1e-7 + 1e-5 * std::fabs(static_cast<double>(want));
}

struct PerfRow {
  const char* name;
  std::vector<float> weights;
  std::vector<std::uint8_t> mask;
  std::vector<std::uint8_t> resident;
  moe::ExpertDropParams params;
  long long want_dropped_per_call;
  long long want_bypasses_per_call;
};

// Times one selector configuration and verifies its per-call counts, so a
// "perf pass" is always a timing of the intended code path.
bool RunPerfRow(const PerfRow& row, double bound_us) {
  const int rows = 1;
  const int experts = static_cast<int>(row.resident.size());
  const int iters = 10000;
  moe::ExpertDropScratch scratch;
  std::vector<float> w = row.weights;
  std::vector<std::uint8_t> m = row.mask;

  // Count check on a single call before timing.
  auto counts =
      moe::SelectExpertDrops(rows, experts, w.data(), m.data(),
                             row.resident.data(), row.params, &scratch);
  if (counts.experts_dropped != row.want_dropped_per_call ||
      counts.flatness_bypasses != row.want_bypasses_per_call) {
    std::fprintf(stderr,
                 "perf[%s]: wrong path — dropped %lld (want %lld), "
                 "bypasses %lld (want %lld)\n",
                 row.name, (long long)counts.experts_dropped,
                 row.want_dropped_per_call, (long long)counts.flatness_bypasses,
                 row.want_bypasses_per_call);
    return false;
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    w = row.weights;
    m = row.mask;
    moe::SelectExpertDrops(rows, experts, w.data(), m.data(),
                           row.resident.data(), row.params, &scratch);
  }
  const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count() /
                    iters;
  std::printf(
      "perf[%s]: %.3f us/call at rows=1 experts=%d (bound %.1f us, "
      "%lld drops + %lld bypasses per call)\n",
      row.name, us, experts, bound_us, row.want_dropped_per_call,
      row.want_bypasses_per_call);
  if (us > bound_us) {
    std::fprintf(stderr, "perf[%s]: bound exceeded\n", row.name);
    return false;
  }
  return true;
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

  const int num_cases = ReadInt(in);
  if (!g_read_ok || num_cases <= 0) {
    std::fprintf(stderr,
                 "fixture rejected: empty or missing positive case count "
                 "(regenerate with gen_expert_drop_fixture.py)\n");
    return 2;
  }
  int failures = 0;
  for (int c = 0; c < num_cases; ++c) {
    const int rows = ReadInt(in);
    const int experts = ReadInt(in);
    moe::ExpertDropParams params;
    params.min_k = ReadInt(in);
    params.mass_budget = ReadDouble(in);
    params.flatness_floor = ReadDouble(in);
    if (!g_read_ok || rows <= 0 || experts <= 0) {
      std::fprintf(stderr, "fixture rejected: truncated header at case %d\n",
                   c);
      return 2;
    }
    const std::size_t n = static_cast<std::size_t>(rows) * experts;
    std::vector<float> weights(n);
    std::vector<std::uint8_t> mask(n);
    std::vector<std::uint8_t> resident(experts);
    std::vector<float> exp_weights(n);
    std::vector<std::uint8_t> exp_mask(n);
    for (auto& v : weights) v = ReadFloat(in);
    for (auto& v : mask) v = static_cast<std::uint8_t>(ReadInt(in));
    for (auto& v : resident) v = static_cast<std::uint8_t>(ReadInt(in));
    for (auto& v : exp_weights) v = ReadFloat(in);
    for (auto& v : exp_mask) v = static_cast<std::uint8_t>(ReadInt(in));
    moe::ExpertDropCounts want;
    want.tokens_seen = ReadInt64(in);
    want.tokens_changed = ReadInt64(in);
    want.experts_dropped = ReadInt64(in);
    want.flatness_bypasses = ReadInt64(in);
    if (!g_read_ok) {
      std::fprintf(stderr, "fixture rejected: truncated at case %d\n", c);
      return 2;
    }

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
  in >> std::ws;
  if (!in.eof()) {
    std::fprintf(stderr, "fixture rejected: trailing content after %d cases\n",
                 num_cases);
    return 2;
  }
  std::printf("parity: %d/%d cases passed\n", num_cases - failures, num_cases);

  const int experts = 64;
  // Drop-producing row: 8 selected experts with skewed weights; the 4
  // low-mass ones are non-resident and fall inside the 0.20 mass budget
  // (the shipped arm), so every call drops exactly 4 and renormalizes.
  PerfRow drop_row;
  drop_row.name = "drop-producing";
  drop_row.weights.assign(experts, 0.0f);
  drop_row.mask.assign(experts, 0);
  drop_row.resident.assign(experts, 1);
  {
    const float skewed[8] = {0.30f, 0.25f, 0.15f, 0.10f,
                             0.08f, 0.06f, 0.04f, 0.02f};
    for (int e = 0; e < 8; ++e) {
      const int id = e * 7;
      drop_row.mask[id] = 1;
      drop_row.weights[id] = skewed[e];
      // Low-mass half non-resident -> drop candidates.
      drop_row.resident[id] = e < 4 ? 1 : 0;
    }
  }
  drop_row.params = moe::ExpertDropParams{1, 0.20, 0.5};
  drop_row.want_dropped_per_call = 4;
  drop_row.want_bypasses_per_call = 0;

  // Flatness-bypass row: uniform weights take the early bypass, timing
  // only the guard scan. Kept separate so the bypass cost can never stand
  // in for drop-producing selection cost.
  PerfRow flat_row;
  flat_row.name = "flatness-bypass";
  flat_row.weights.assign(experts, 0.0f);
  flat_row.mask.assign(experts, 0);
  flat_row.resident.assign(experts, 0);
  for (int e = 0; e < 8; ++e) {
    flat_row.mask[e * 7] = 1;
    flat_row.weights[e * 7] = 0.125f;
  }
  flat_row.params = moe::ExpertDropParams{1, 0.05, 0.5};
  flat_row.want_dropped_per_call = 0;
  flat_row.want_bypasses_per_call = 1;

  bool perf_ok = RunPerfRow(drop_row, 5.0);
  perf_ok = RunPerfRow(flat_row, 5.0) && perf_ok;
  if (!perf_ok) return 1;
  return failures == 0 ? 0 : 1;
}
