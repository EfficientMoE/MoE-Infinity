// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Tests for utils/cache.h (LFUCache<KeyType, ValueType>)
//
// Known bugs found during review:
//
//  BUG-1 (use-after-erase in get()):
//    `get()` stores an iterator `node = keyNodeMap[key]`, calls
//    `freqListMap[freq].erase(node)` which invalidates `node`, then reads
//    `node->freq`, `node->value` and pushes `*node` into another list.
//    Dereferencing an erased list iterator is undefined behaviour.
//    The tests below document the *intended* correct behaviour; they may
//    crash or produce wrong results with AddressSanitizer enabled.
//
//  BUG-2 (concurrent modification in reset()):
//    `reset()` iterates over `freqListMap` while simultaneously calling
//    `freqListMap[1].push_back(node)` which can insert a new key=1 entry and
//    invalidate the range-for iterator.

#include "utils/cache.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Basic correctness
// ---------------------------------------------------------------------------

TEST(LFUCacheTest, BasicPutGet) {
  LFUCache<int, int> cache(3);
  cache.put(1, 100);
  EXPECT_EQ(cache.get(1), 100);
}

TEST(LFUCacheTest, PutUpdatesExistingValue) {
  LFUCache<int, int> cache(3);
  cache.put(1, 100);
  cache.put(1, 200);
  EXPECT_EQ(cache.get(1), 200);
}

TEST(LFUCacheTest, GetNonExistentThrows) {
  LFUCache<int, int> cache(3);
  EXPECT_THROW(cache.get(42), std::range_error);
}

TEST(LFUCacheTest, CapacityZeroDropsAllPuts) {
  LFUCache<int, int> cache(0);
  cache.put(1, 100);  // should not crash
  EXPECT_THROW(cache.get(1), std::range_error);
}

TEST(LFUCacheTest, StringKeyAndValue) {
  LFUCache<std::string, std::string> cache(2);
  cache.put("hello", "world");
  EXPECT_EQ(cache.get("hello"), "world");
}

// ---------------------------------------------------------------------------
// Capacity / eviction
// ---------------------------------------------------------------------------

// When capacity is full, the least-frequently-used element is evicted.
// After one access of key=1 and two accesses of key=2:
//   freq(1) = 1 (initial put), freq(2) = 2 (put + get)
// Inserting key=3 should evict key=1 (LFU).
TEST(LFUCacheTest, EvictLFUOnOverflow) {
  LFUCache<int, int> cache(2);
  cache.put(1, 10);  // freq[1]=1
  cache.put(2, 20);  // freq[2]=1
  cache.get(2);      // freq[2]=2 (BUG-1 affects this call)
  cache.put(3, 30);  // capacity full → evict key=1 (lowest freq=1)

  EXPECT_EQ(cache.get(2), 20);
  EXPECT_EQ(cache.get(3), 30);
  // key=1 must have been evicted.
  EXPECT_THROW(cache.get(1), std::range_error);
}

// When multiple keys share the minimum frequency, the LRU among them is
// evicted (tail of the frequency list).
TEST(LFUCacheTest, EvictLRUAmongTiedFrequency) {
  LFUCache<int, int> cache(3);
  cache.put(1, 10);  // freq[1]=1, inserted first
  cache.put(2, 20);  // freq[2]=1
  cache.put(3, 30);  // freq[3]=1
  // All at freq=1; key=1 was least recently used (tail of freq-1 list).
  cache.put(4, 40);  // should evict key=1

  EXPECT_THROW(cache.get(1), std::range_error);
  EXPECT_EQ(cache.get(2), 20);
  EXPECT_EQ(cache.get(3), 30);
  EXPECT_EQ(cache.get(4), 40);
}

// A key that is accessed frequently survives multiple evictions.
TEST(LFUCacheTest, FrequentlyAccessedKeyNotEvicted) {
  LFUCache<int, int> cache(2);
  cache.put(1, 10);
  cache.put(2, 20);
  // Bump key=1 to freq=4.
  cache.get(1);  // freq[1]=2
  cache.get(1);  // freq[1]=3
  cache.get(1);  // freq[1]=4

  // Now add key=3; key=2 (freq=1) should be evicted, not key=1.
  cache.put(3, 30);
  EXPECT_EQ(cache.get(1), 10);
  EXPECT_EQ(cache.get(3), 30);
  EXPECT_THROW(cache.get(2), std::range_error);
}

// Capacity-1 cache: every new put evicts the previous entry.
TEST(LFUCacheTest, CapacityOne) {
  LFUCache<int, int> cache(1);
  cache.put(1, 100);
  EXPECT_EQ(cache.get(1), 100);

  cache.put(2, 200);  // evicts key=1
  EXPECT_EQ(cache.get(2), 200);
  EXPECT_THROW(cache.get(1), std::range_error);
}

// Fill cache to exact capacity, verify all entries retrievable.
TEST(LFUCacheTest, FillToCapacityNoEviction) {
  const int cap = 10;
  LFUCache<int, int> cache(cap);
  for (int i = 0; i < cap; i++) cache.put(i, i * 10);
  for (int i = 0; i < cap; i++) EXPECT_EQ(cache.get(i), i * 10);
}

// After eviction, inserting the evicted key again works correctly.
TEST(LFUCacheTest, ReinsertEvictedKey) {
  LFUCache<int, int> cache(2);
  cache.put(1, 10);
  cache.put(2, 20);
  cache.put(3, 30);  // evicts key=1 (LFU / LRU tie-break)

  // Re-insert key=1.
  cache.put(1, 99);
  EXPECT_EQ(cache.get(1), 99);
}

// ---------------------------------------------------------------------------
// minFreq tracking
// ---------------------------------------------------------------------------

// After eviction and new insertion, minFreq must reset to 1.
TEST(LFUCacheTest, MinFreqResetAfterEviction) {
  LFUCache<int, int> cache(2);
  cache.put(1, 10);
  cache.put(2, 20);
  cache.get(1);  // freq[1]=2
  cache.get(2);  // freq[2]=2
  // Now both at freq=2; put(3,...) evicts one and minFreq should become 1.
  cache.put(3, 30);

  // At least two of the three keys must be accessible.
  int accessible = 0;
  for (int k : {1, 2, 3}) {
    try {
      cache.get(k);
      accessible++;
    } catch (const std::range_error&) {
    }
  }
  EXPECT_EQ(accessible, 2);
}

// ---------------------------------------------------------------------------
// reset()
// ---------------------------------------------------------------------------

// After reset(), all frequencies drop to 1 so all entries are equally
// likely to be evicted.  We verify that existing keys are still readable.
// NOTE: BUG-2 may cause a crash here; this test documents intended behaviour.
TEST(LFUCacheTest, ResetPreservesKeys) {
  LFUCache<int, int> cache(3);
  cache.put(1, 10);
  cache.put(2, 20);
  cache.put(3, 30);
  cache.get(1);
  cache.get(1);

  cache.reset();

  // After reset, all keys must still be readable.
  EXPECT_EQ(cache.get(1), 10);
  EXPECT_EQ(cache.get(2), 20);
  EXPECT_EQ(cache.get(3), 30);
}

// After reset(), all keys have freq=1, so the next eviction follows LRU
// among equal frequencies.
TEST(LFUCacheTest, ResetMakesEvictionLRU) {
  LFUCache<int, int> cache(2);
  cache.put(1, 10);
  cache.put(2, 20);
  cache.get(1);  // freq[1]=2

  cache.reset();  // both back to freq=1

  // key=1 was accessed more recently, so key=2 should be LRU-evicted.
  cache.put(3, 30);  // evicts key=2 (LRU among freq=1 entries)
  EXPECT_EQ(cache.get(1), 10);
  EXPECT_EQ(cache.get(3), 30);
  EXPECT_THROW(cache.get(2), std::range_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
