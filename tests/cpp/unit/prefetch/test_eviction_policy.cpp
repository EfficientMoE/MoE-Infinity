// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// ExpertResidencyManager with a selectable eviction policy. The replay tests
// drive the manager the way the dispatcher's demand path does (RecordRouted,
// BeginAdmission, EvictReserved, CommitAdmission) and require the hit count of
// the policy's simulator replay (SimulateEvictionPolicy), which
// tests/python/unit/test_eviction_policy_parity.py checks against
// benchmarks/cache_simulator/policies.py.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "expert_residency_test_fixture.h"
#include "prefetch/eviction_policy.h"

namespace {

constexpr std::int64_t kExpertBytes = 16;
using Trace = std::vector<std::pair<int, int>>;

Trace DecodeTrace(std::mt19937& rng, int steps, int layers, int experts,
                  int top_k) {
  std::vector<double> weights;
  for (int rank = 0; rank < experts; ++rank)
    weights.push_back(1.0 / (rank + 1));
  std::discrete_distribution<int> pick(weights.begin(), weights.end());
  std::bernoulli_distribution keep(0.4);
  std::vector<std::vector<int>> current(layers);
  Trace trace;
  for (int step = 0; step < steps; ++step) {
    for (int layer = 0; layer < layers; ++layer) {
      std::vector<int> routed;
      for (int expert : current[layer]) {
        if (keep(rng)) routed.push_back(expert);
      }
      while (static_cast<int>(routed.size()) < top_k) {
        const int expert = pick(rng);
        if (std::find(routed.begin(), routed.end(), expert) == routed.end()) {
          routed.push_back(expert);
        }
      }
      current[layer] = routed;
      for (int expert : routed) trace.emplace_back(layer, expert);
    }
  }
  return trace;
}

Trace RandomTrace(std::mt19937& rng, int length, int layers, int experts) {
  std::uniform_int_distribution<int> layer(0, layers - 1);
  std::uniform_int_distribution<int> expert(0, experts - 1);
  Trace trace;
  for (int i = 0; i < length; ++i) trace.emplace_back(layer(rng), expert(rng));
  return trace;
}

class EvictionPolicyManagerTest : public ExpertResidencyManagerTest {
 protected:
  // One canonical node per (layer, expert), registered as the dispatcher's
  // RegisterExpert does at load time.
  void RegisterExperts(int layers, int experts) {
    for (int layer = 0; layer < layers; ++layer) {
      for (int expert = 0; expert < experts; ++expert) {
        auto node = MakeNode(nodes_.size() + 1, kExpertBytes);
        manager->RegisterExpertKey(node->id, layer, expert);
        nodes_.emplace(MakeExpertCacheKey(layer, expert), node);
      }
    }
  }

  NodePtr Expert(int layer, int expert) {
    return nodes_.at(MakeExpertCacheKey(layer, expert));
  }

  std::size_t Replay(const Trace& trace, std::size_t slots) {
    EXPECT_TRUE(manager->ConfigureCapacity(
        0, static_cast<std::int64_t>(slots) * kExpertBytes));
    std::size_t hits = 0;
    for (const auto& access : trace) {
      auto node = Expert(access.first, access.second);
      manager->RecordRouted(node, 0);
      if (manager->ActiveGeneration(0, node->id).has_value()) {
        ++hits;
        continue;
      }
      auto ticket = manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                            AdmissionMode::CACHE,
                                            AdmissionSource::DEMAND);
      EXPECT_TRUE(ticket.valid);
      if (!ticket.valid) return hits;
      if (ticket.reserved_victim != nullptr) {
        EXPECT_TRUE(manager->EvictReserved(ticket));
      }
      node->device = node->default_device;
      EXPECT_TRUE(manager->CommitAdmission(ticket));
      EXPECT_LE(manager->ResidentCount(0), slots);
    }
    return hits;
  }

  std::unordered_map<ExpertCacheKey, NodePtr> nodes_;
};

std::vector<ExpertCacheKey> Keys(const Trace& trace) {
  std::vector<ExpertCacheKey> keys;
  for (const auto& access : trace) {
    keys.push_back(MakeExpertCacheKey(access.first, access.second));
  }
  return keys;
}

struct ReplayCase {
  std::string policy;
  bool derive_layers;
};

class EvictionPolicyReplayTest
    : public EvictionPolicyManagerTest,
      public ::testing::WithParamInterface<ReplayCase> {};

TEST_P(EvictionPolicyReplayTest, ManagerReplayMatchesSimulator) {
  const auto& param = GetParam();
  std::mt19937 rng(7);
  const int layers = 6;
  const int experts = 16;
  const int top_k = 2;
  const Trace trace = DecodeTrace(rng, 40, layers, experts, top_k);
  EvictionPolicyOptions options;
  options.num_layers = layers;
  options.lcnu_prior = static_cast<double>(top_k) / experts;
  EvictionPolicyOptions configured = options;
  if (param.derive_layers) configured.num_layers = 0;

  for (std::size_t slots : {3, 10, 13, 24, 48}) {
    SetUp();
    nodes_.clear();
    RegisterExperts(layers, experts);
    ASSERT_TRUE(manager->ConfigureEvictionPolicy(param.policy, configured));
    const std::size_t hits = Replay(trace, slots);
    if (param.policy == "s3fifo") {
      // The second eviction of an S3-FIFO promotion pass waits for the next
      // admission (one victim per admission), so only the invariants hold.
      EXPECT_LE(hits, trace.size());
      continue;
    }
    EXPECT_EQ(hits,
              SimulateEvictionPolicy(param.policy, Keys(trace), slots, options))
        << param.policy << " slots=" << slots;
  }
}

std::vector<ReplayCase> ReplayCases() {
  std::vector<ReplayCase> cases;
  for (const auto& name : EvictionPolicyNames()) {
    cases.push_back({name, false});
  }
  cases.push_back({"lcnu", true});
  cases.push_back({"layer_lru", true});
  return cases;
}

INSTANTIATE_TEST_SUITE_P(AllPolicies, EvictionPolicyReplayTest,
                         ::testing::ValuesIn(ReplayCases()),
                         [](const ::testing::TestParamInfo<ReplayCase>& info) {
                           std::string name = info.param.policy;
                           if (name == "2q") name = "two_q";
                           return name + (info.param.derive_layers
                                              ? "_derived_layers"
                                              : "");
                         });

TEST_F(EvictionPolicyManagerTest, RandomTraceReplayMatchesSimulator) {
  std::mt19937 rng(11);
  const Trace trace = RandomTrace(rng, 400, 3, 8);
  EvictionPolicyOptions options;
  options.num_layers = 3;
  for (const auto& name : EvictionPolicyNames()) {
    if (name == "s3fifo") continue;
    for (std::size_t slots : {1, 4, 9}) {
      SetUp();
      nodes_.clear();
      RegisterExperts(3, 8);
      ASSERT_TRUE(manager->ConfigureEvictionPolicy(name, options));
      EXPECT_EQ(Replay(trace, slots),
                SimulateEvictionPolicy(name, Keys(trace), slots, options))
          << name << " slots=" << slots;
    }
  }
}

// Full-scan LCNU scorer, written from the policy's definition, to check the
// cached per-layer victim search when only some residents are eligible.
class LcnuReference {
 public:
  LcnuReference(int layers, double alpha, double beta, double prior)
      : layers_(layers), alpha_(alpha), beta_(beta), prior_(prior) {}

  void Access(ExpertCacheKey key) {
    const int layer = ExpertCacheKeyLayer(key);
    if (cur_ != layer) {
      cur_ = layer;
      ++visits_[layer];
      used_now_.clear();
    }
    const std::int64_t visit = visits_[layer];
    auto it = stats_.find(key);
    if (it == stats_.end()) {
      stats_[key] = {(1.0 - alpha_) * prior_ + alpha_, 1, visit - 1, visit};
    } else {
      if (visit > it->second.last) {
        it->second.ema =
            it->second.ema * std::pow(1.0 - alpha_, visit - it->second.last) +
            alpha_;
        it->second.count += 1;
      }
      it->second.last = visit;
    }
    used_now_.insert(key);
  }

  std::optional<ExpertCacheKey> Victim(
      const std::unordered_set<ExpertCacheKey>& resident,
      const std::unordered_set<ExpertCacheKey>& eligible) const {
    std::optional<ExpertCacheKey> best;
    double best_score = 0.0;
    for (ExpertCacheKey key : resident) {
      if (!eligible.count(key)) continue;
      const int layer = ExpertCacheKeyLayer(key);
      const double d = layer == cur_
                           ? (used_now_.count(key) ? layers_ : 0)
                           : ((layer - cur_) % layers_ + layers_) % layers_;
      const auto& s = stats_.at(key);
      const std::int64_t visits = visits_.at(layer);
      const double p =
          beta_ * s.ema * std::pow(1.0 - alpha_, visits - s.last) +
          (1.0 - beta_) * (s.count + 4.0 * prior_) / (visits - s.first + 4.0);
      const double score = d + layers_ * (1.0 / p - 1.0);
      // Scores are compared with a tolerance: pow and exp may round apart.
      if (!best || score > best_score + 1e-9 ||
          (std::abs(score - best_score) <= 1e-9 && key < *best)) {
        best = key;
        best_score = score;
      }
    }
    return best;
  }

 private:
  struct Stats {
    double ema;
    std::int64_t count, first, last;
  };
  int layers_;
  double alpha_, beta_, prior_;
  int cur_ = -1;
  std::unordered_map<int, std::int64_t> visits_;
  std::unordered_map<ExpertCacheKey, Stats> stats_;
  std::unordered_set<ExpertCacheKey> used_now_;
};

TEST(LcnuPolicy, CachedSearchHonoursEligibility) {
  std::mt19937 rng(3);
  const int layers = 5;
  const int experts = 12;
  const int top_k = 3;
  EvictionPolicyOptions options;
  options.num_layers = layers;
  options.lcnu_prior = static_cast<double>(top_k) / experts;
  for (std::size_t slots : {4, 9, 16}) {
    auto policy = CreateEvictionPolicy("lcnu", options);
    policy->SetCapacity(slots);
    LcnuReference reference(layers, options.lcnu_alpha, options.lcnu_beta,
                            options.lcnu_prior);
    std::unordered_set<ExpertCacheKey> resident;
    std::bernoulli_distribution coin(0.6);
    std::size_t checked = 0;
    for (const auto& access : DecodeTrace(rng, 60, layers, experts, top_k)) {
      const auto key = MakeExpertCacheKey(access.first, access.second);
      const bool hit = resident.count(key) > 0;
      policy->OnAccess(key, hit);
      reference.Access(key);
      if (hit) continue;
      if (resident.size() >= slots) {
        std::unordered_set<ExpertCacheKey> eligible;
        for (ExpertCacheKey k : resident) {
          if (coin(rng)) eligible.insert(k);
        }
        const auto chosen = policy->SelectVictim(
            key, true, [&](ExpertCacheKey k) { return eligible.count(k) > 0; });
        ASSERT_EQ(chosen, reference.Victim(resident, eligible));
        ++checked;
        if (!chosen) continue;  // nothing eligible: the miss is not cached
        resident.erase(*chosen);
        policy->OnEvict(*chosen);
      }
      resident.insert(key);
      policy->OnAdmit(key, false);
    }
    EXPECT_GT(checked, 50u);
  }
}

TEST_F(EvictionPolicyManagerTest, LegacyIsDefaultAndUnmanaged) {
  EXPECT_EQ(manager->EvictionPolicyName(), "legacy");
  EXPECT_FALSE(manager->ManagedResidency());
  EXPECT_FALSE(manager->ConfigureEvictionPolicy("no_such_policy", {}));
  EXPECT_EQ(manager->EvictionPolicyName(), "legacy");

  ASSERT_TRUE(manager->ConfigureEvictionPolicy("lcnu", {}));
  EXPECT_EQ(manager->EvictionPolicyName(), "lcnu");
  EXPECT_TRUE(manager->ManagedResidency());
  EXPECT_FALSE(manager->PolicyEnabled());
  EXPECT_EQ(manager->Snapshot().at("eviction_policy_active"), 1);

  ASSERT_TRUE(manager->ConfigureEvictionPolicy("legacy", {}));
  EXPECT_FALSE(manager->ManagedResidency());
  EXPECT_EQ(manager->Snapshot().at("eviction_policy_active"), 0);
}

TEST_F(EvictionPolicyManagerTest, ReconfigureWaitsForPendingAdmissions) {
  RegisterExperts(1, 2);
  ASSERT_TRUE(manager->ConfigureCapacity(0, 4 * kExpertBytes));
  auto ticket =
      manager->BeginAdmission(Expert(0, 0), 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  ASSERT_TRUE(ticket.valid);
  EXPECT_FALSE(manager->ConfigureEvictionPolicy("lru", {}));
  ASSERT_TRUE(manager->AbortAdmission(ticket));
  EXPECT_TRUE(manager->ConfigureEvictionPolicy("lru", {}));
}

TEST_F(EvictionPolicyManagerTest, PolicyChoosesVictimAndSkipsLeased) {
  RegisterExperts(1, 3);
  ASSERT_TRUE(manager->ConfigureEvictionPolicy("lfu", {}));
  // Expert 0 is used twice, expert 1 once and more recently: LFU keeps 0 where
  // the built-in least-recently-accessed choice would evict it.
  Expert(0, 0)->last_access_time = 1;
  Expert(0, 1)->last_access_time = 2;
  Replay({{0, 0}, {0, 0}, {0, 1}}, 2);
  manager->RecordRouted(Expert(0, 2), 0);
  auto ticket =
      manager->BeginAdmission(Expert(0, 2), 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.reserved_victim, Expert(0, 1));
  ASSERT_TRUE(manager->AbortAdmission(ticket));

  const auto lease = manager->AcquireLease(Expert(0, 1), LeaseKind::DEMAND);
  ticket =
      manager->BeginAdmission(Expert(0, 2), 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.reserved_victim, Expert(0, 0));
  ASSERT_TRUE(manager->AbortAdmission(ticket));
  ASSERT_TRUE(manager->ReleaseLease(lease));

  const auto stats = manager->Snapshot();
  EXPECT_EQ(stats.at("policy_routed_hits"), 1);
  EXPECT_EQ(stats.at("policy_routed_misses"), 3);
  EXPECT_EQ(stats.at("eviction_policy_slots"), 2);
}

TEST_F(EvictionPolicyManagerTest, PolicyIsSizedByRegisteredExperts) {
  RegisterExperts(1, 2);
  ASSERT_TRUE(manager->ConfigureEvictionPolicy("lru", {}));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100 * kExpertBytes));
  // A dense node admitted through the same manager before any expert.
  auto dense = MakeNode(1000, 10 * kExpertBytes);
  auto ticket =
      manager->BeginAdmission(dense, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::PREFETCH);
  ASSERT_TRUE(ticket.valid);
  dense->device = dense->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  manager->RecordRouted(Expert(0, 0), 0);
  EXPECT_EQ(manager->Snapshot().at("eviction_policy_slots"), 100);
}

TEST_F(EvictionPolicyManagerTest, UnregisteredExpertsUseBuiltInVictim) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy("lru", {}));
  ASSERT_TRUE(manager->ConfigureCapacity(0, kExpertBytes));
  auto first = MakeNode(1, kExpertBytes);
  auto ticket =
      manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  ASSERT_TRUE(ticket.valid);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  auto second = MakeNode(2, kExpertBytes);
  ticket =
      manager->BeginAdmission(second, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.reserved_victim, first);
  ASSERT_TRUE(manager->AbortAdmission(ticket));
}

}  // namespace
