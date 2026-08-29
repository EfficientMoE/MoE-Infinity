// tests/cpp/unit/prefetch/test_expert_policy.cpp
#include <gtest/gtest.h>
#include "prefetch/expert_policy.h"

TEST(ExpertPolicy, DecodeWeightsDecodeReuse) {
  PhasePolicyConfig cfg{true,
                        AdmissionMode::TRANSIENT_ON_PRESSURE,
                        AdmissionMode::CACHE,
                        1.0,
                        4.0,
                        8};
  ExpertPolicyMetadata prefill_hot{10, 0, 10, 0};
  ExpertPolicyMetadata decode_hot{0, 3, 0, 9};
  EXPECT_LT(VictimUtility(prefill_hot, ExpertPhase::DECODE, cfg),
            VictimUtility(decode_hot, ExpertPhase::DECODE, cfg));
}

TEST(ExpertPolicy, StableVictimTieBreakUsesLayerThenExpert) {
  VictimCandidate a{2, 7, 1.0, 4};
  VictimCandidate b{3, 0, 1.0, 4};
  EXPECT_TRUE(VictimLess(a, b));
}

TEST(ExpertPolicy, BypassPromotionNeverPassesDemand) {
  EXPECT_EQ(ServiceClass(0, 99, 8), 0);
  EXPECT_EQ(ServiceClass(2, 8, 8), 1);
  EXPECT_EQ(ServiceClass(1, 0, 8), 1);
}
