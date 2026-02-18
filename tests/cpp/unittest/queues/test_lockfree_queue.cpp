// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Tests for utils/lockfree_queue.h
//
// Known issues found during review:
//  - LockFreeRecyclingQueue::Pop returns void but base Pop returns bool, and
//    it uses `override` on a non-virtual base method -> compile error.
//  - LockFreeRecyclingQueue::TryPop similarly broken.
//  - LockFreeQueue::Pop: after a successful CAS that advances `head_`, the
//    read of `old_head->next.load()->data` can race with another thread that
//    already popped (and freed) the node that old_head->next points to.
//    This is a latent ABA / use-after-free hazard in high concurrency.

#include "utils/lockfree_queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Existing tests (kept intact)
// ---------------------------------------------------------------------------

TEST(LockFreeQueueTest, SingleThreadedPushPop) {
  LockFreeQueue<int> queue;
  int value;

  int a = 1;
  queue.Push(a);
  ASSERT_TRUE(queue.Pop(value));
  ASSERT_EQ(value, 1);
}

TEST(LockFreeQueueTest, SequentialPushParallelPop) {
  LockFreeQueue<int> queue;

  for (int i = 0; i < 10; i++) {
    queue.Push(i);
  }

  std::vector<std::thread> threads;
  std::vector<int> results(10);
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&queue, &results, i]() {
      int val;
      while (!queue.Pop(val)) {
        // Busy wait until our slot is filled
      }
      results[i] = val;
    });
  }
  for (auto& t : threads) t.join();

  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; i++) ASSERT_EQ(results[i], i);
}

TEST(LockFreeQueueTest, ParallelPushSequentialPop) {
  LockFreeQueue<int> queue;

  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&queue, i]() {
      int val = i;
      queue.Push(val);
    });
  }
  for (auto& t : threads) t.join();

  std::vector<int> results(10);
  for (int i = 0; i < 10; i++) {
    int value;
    ASSERT_TRUE(queue.Pop(value));
    results[i] = value;
  }
  int value;
  ASSERT_FALSE(queue.Pop(value));  // Queue should be empty

  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; i++) ASSERT_EQ(results[i], i);
}

TEST(LockFreeQueueTest, ParallelPushParallelPop) {
  LockFreeQueue<int> queue;

  std::vector<std::thread> push_threads;
  for (int i = 0; i < 10; i++) {
    push_threads.emplace_back([&queue, i]() {
      int val = i;
      queue.Push(val);
    });
  }

  std::vector<std::thread> pop_threads;
  std::vector<int> results(10);
  for (int i = 0; i < 10; i++) {
    pop_threads.emplace_back([&queue, &results, i]() {
      int val;
      while (!queue.Pop(val)) {
        // Busy wait
      }
      results[i] = val;
    });
  }

  for (auto& t : push_threads) t.join();
  for (auto& t : pop_threads) t.join();

  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; i++) ASSERT_EQ(results[i], i);
}

// ---------------------------------------------------------------------------
// New edge-case and correctness tests
// ---------------------------------------------------------------------------

// Pop on freshly-constructed (empty) queue must return false immediately.
TEST(LockFreeQueueTest, PopOnEmptyReturnsFalse) {
  LockFreeQueue<int> queue;
  int value = -1;
  EXPECT_FALSE(queue.Pop(value));
  // value must not have been modified
  EXPECT_EQ(value, -1);
}

// Empty() is true before any push and false afterwards.
TEST(LockFreeQueueTest, EmptyMethod) {
  LockFreeQueue<int> queue;
  EXPECT_TRUE(queue.Empty());

  int v = 42;
  queue.Push(v);
  EXPECT_FALSE(queue.Empty());

  queue.Pop(v);
  EXPECT_TRUE(queue.Empty());
}

// Full() must always return false (unbounded queue).
TEST(LockFreeQueueTest, FullAlwaysFalse) {
  LockFreeQueue<int> queue;
  EXPECT_FALSE(queue.Full());
  for (int i = 0; i < 1000; i++) {
    queue.Push(i);
    EXPECT_FALSE(queue.Full());
  }
}

// Push then pop a single item and verify the returned value is correct.
TEST(LockFreeQueueTest, SingleItemRoundTrip) {
  LockFreeQueue<int> queue;
  int in = 123, out = 0;
  queue.Push(in);
  ASSERT_TRUE(queue.Pop(out));
  EXPECT_EQ(out, 123);
  EXPECT_TRUE(queue.Empty());
}

// Verify FIFO ordering in single-threaded use.
TEST(LockFreeQueueTest, FifoOrderSingleThread) {
  LockFreeQueue<int> queue;
  for (int i = 0; i < 10; i++) queue.Push(i);

  for (int i = 0; i < 10; i++) {
    int v;
    ASSERT_TRUE(queue.Pop(v));
    EXPECT_EQ(v, i);
  }
}

// Works correctly with large objects (heap-allocated string).
TEST(LockFreeQueueTest, LargeObjectRoundTrip) {
  LockFreeQueue<std::string> queue;
  std::string in(10000, 'x');
  queue.Push(in);

  std::string out;
  ASSERT_TRUE(queue.Pop(out));
  EXPECT_EQ(out, in);
}

// Stress test: many producers and consumers, all items accounted for.
// This also stresses the ABA / memory-reclamation path in Pop().
TEST(LockFreeQueueTest, HighConcurrencyStress) {
  constexpr int kProducers = 8;
  constexpr int kConsumers = 8;
  constexpr int kItemsPerProducer = 1000;
  constexpr int kTotal = kProducers * kItemsPerProducer;

  LockFreeQueue<int> queue;
  std::atomic<int> produced{0};
  std::atomic<int> consumed{0};

  std::vector<std::thread> producers;
  for (int t = 0; t < kProducers; t++) {
    producers.emplace_back([&queue, &produced, t]() {
      for (int i = 0; i < kItemsPerProducer; i++) {
        int v = t * kItemsPerProducer + i;
        queue.Push(v);
        produced.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  std::vector<int> seen(kTotal, 0);
  std::mutex seen_mutex;
  std::vector<std::thread> consumers;
  for (int t = 0; t < kConsumers; t++) {
    consumers.emplace_back([&queue, &consumed, &seen, &seen_mutex]() {
      while (true) {
        int v;
        if (queue.Pop(v)) {
          {
            std::lock_guard<std::mutex> lk(seen_mutex);
            seen[v]++;
          }
          if (consumed.fetch_add(1, std::memory_order_relaxed) + 1 == kTotal) {
            return;
          }
        }
        if (consumed.load(std::memory_order_relaxed) >= kTotal) return;
      }
    });
  }

  for (auto& t : producers) t.join();
  for (auto& t : consumers) t.join();

  EXPECT_EQ(produced.load(), kTotal);
  EXPECT_EQ(consumed.load(), kTotal);
  // Every value seen exactly once
  for (int i = 0; i < kTotal; i++) EXPECT_EQ(seen[i], 1) << "item " << i;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
