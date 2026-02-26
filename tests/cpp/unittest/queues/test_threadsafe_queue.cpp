// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Tests for utils/threadsafe_queue.h
//
// Known issues found during review:
//  - ThreadSafeQueue::Pop blocks indefinitely when the queue is empty.
//    The return value is always true; callers that spin on `while (!Pop(v))`
//    will deadlock if the queue is permanently empty.
//  - ThreadSafeRecyclingQueue::Pop and TryPop use `override` on non-virtual
//    base methods, which is a compile error.  Those classes are untested here.

#include "utils/threadsafe_queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Existing tests (kept intact, using TryPop where the original had Pop on
// empty which would deadlock)
// ---------------------------------------------------------------------------

TEST(ThreadSafeQueueTest, SingleThreadedPushPop) {
  ThreadSafeQueue<int> queue;
  int value;

  int a = 1;
  queue.Push(a);
  ASSERT_TRUE(queue.Pop(value));
  ASSERT_EQ(value, 1);
}

TEST(ThreadSafeQueueTest, SequentialPushParallelPop) {
  ThreadSafeQueue<int> queue;

  for (int i = 0; i < 10; i++) queue.Push(i);

  std::vector<std::thread> threads;
  std::vector<int> results(10);
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&queue, &results, i]() {
      int val;
      // Pop blocks until an item is available — no busy-wait needed.
      queue.Pop(val);
      results[i] = val;
    });
  }
  for (auto& t : threads) t.join();

  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; i++) ASSERT_EQ(results[i], i);
}

TEST(ThreadSafeQueueTest, ParallelPushSequentialPop) {
  ThreadSafeQueue<int> queue;

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
    ASSERT_TRUE(queue.Pop(results[i]));
  }
  // Now empty — TryPop (non-blocking) must return false.
  int value;
  ASSERT_FALSE(queue.TryPop(value));

  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; i++) ASSERT_EQ(results[i], i);
}

TEST(ThreadSafeQueueTest, ParallelPushParallelPop) {
  ThreadSafeQueue<int> queue;

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
      queue.Pop(val);  // blocks until item available
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

// TryPop on an empty queue returns false without blocking.
TEST(ThreadSafeQueueTest, TryPopOnEmptyReturnsFalse) {
  ThreadSafeQueue<int> queue;
  int value = -1;
  EXPECT_FALSE(queue.TryPop(value));
  EXPECT_EQ(value, -1);  // must not be modified
}

// TryPop after Push returns true and the correct value.
TEST(ThreadSafeQueueTest, TryPopAfterPush) {
  ThreadSafeQueue<int> queue;
  int v = 99;
  queue.Push(v);

  int out = -1;
  EXPECT_TRUE(queue.TryPop(out));
  EXPECT_EQ(out, 99);

  // Second TryPop on now-empty queue returns false.
  EXPECT_FALSE(queue.TryPop(out));
}

// Empty() reflects queue state correctly.
TEST(ThreadSafeQueueTest, EmptyMethod) {
  ThreadSafeQueue<int> queue;
  EXPECT_TRUE(queue.Empty());

  int v = 42;
  queue.Push(v);
  EXPECT_FALSE(queue.Empty());

  queue.TryPop(v);
  EXPECT_TRUE(queue.Empty());
}

// FIFO ordering guaranteed in single-threaded use.
TEST(ThreadSafeQueueTest, FifoOrderSingleThread) {
  ThreadSafeQueue<int> queue;
  for (int i = 0; i < 10; i++) queue.Push(i);

  for (int i = 0; i < 10; i++) {
    int v;
    ASSERT_TRUE(queue.Pop(v));
    EXPECT_EQ(v, i);
  }
}

// Pop blocks until a producer delivers an item.
TEST(ThreadSafeQueueTest, PopBlocksUntilProducer) {
  ThreadSafeQueue<int> queue;
  std::atomic<bool> consumer_started{false};
  std::atomic<bool> consumer_done{false};
  int received = -1;

  std::thread consumer([&]() {
    consumer_started.store(true);
    queue.Pop(received);
    consumer_done.store(true);
  });

  // Wait until the consumer thread has started.
  while (!consumer_started.load()) std::this_thread::yield();

  // Consumer must not have finished (queue is empty → blocked).
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(consumer_done.load());

  int v = 7;
  queue.Push(v);
  consumer.join();

  EXPECT_TRUE(consumer_done.load());
  EXPECT_EQ(received, 7);
}

// Multiple consumers all receive exactly one item each.
TEST(ThreadSafeQueueTest, MultipleConsumersEachGetOne) {
  constexpr int N = 8;
  ThreadSafeQueue<int> queue;

  std::atomic<int> total_received{0};
  std::vector<int> got(N, -1);

  std::vector<std::thread> consumers;
  for (int i = 0; i < N; i++) {
    consumers.emplace_back([&queue, &got, &total_received, i]() {
      queue.Pop(got[i]);
      total_received.fetch_add(1);
    });
  }

  // Produce N items after consumers are likely blocked.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (int i = 0; i < N; i++) queue.Push(i);

  for (auto& t : consumers) t.join();

  EXPECT_EQ(total_received.load(), N);
  // All slots filled with valid values.
  for (int i = 0; i < N; i++) EXPECT_GE(got[i], 0);

  // Verify the full set of values [0, N) was distributed.
  std::vector<int> sorted = got;
  std::sort(sorted.begin(), sorted.end());
  for (int i = 0; i < N; i++) EXPECT_EQ(sorted[i], i);
}

// High-concurrency stress: no item is lost or duplicated.
TEST(ThreadSafeQueueTest, HighConcurrencyStress) {
  constexpr int kProducers = 8;
  constexpr int kConsumers = 8;
  constexpr int kItemsPerProducer = 500;
  constexpr int kTotal = kProducers * kItemsPerProducer;

  ThreadSafeQueue<int> queue;
  std::atomic<int> consumed{0};
  std::vector<int> seen(kTotal, 0);
  std::mutex seen_mutex;

  std::vector<std::thread> producers;
  for (int t = 0; t < kProducers; t++) {
    producers.emplace_back([&queue, t]() {
      for (int i = 0; i < kItemsPerProducer; i++) {
        int v = t * kItemsPerProducer + i;
        queue.Push(v);
      }
    });
  }

  std::vector<std::thread> consumers;
  for (int t = 0; t < kConsumers; t++) {
    consumers.emplace_back([&queue, &consumed, &seen, &seen_mutex, kTotal]() {
      while (true) {
        int remaining = kTotal - consumed.load(std::memory_order_acquire);
        if (remaining <= 0) return;

        int v;
        if (queue.TryPop(v)) {
          {
            std::lock_guard<std::mutex> lk(seen_mutex);
            seen[v]++;
          }
          consumed.fetch_add(1, std::memory_order_release);
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  for (auto& t : producers) t.join();
  // After all producers finish, drain remaining items.
  {
    int v;
    while (queue.TryPop(v)) {
      std::lock_guard<std::mutex> lk(seen_mutex);
      seen[v]++;
      consumed.fetch_add(1, std::memory_order_relaxed);
    }
  }
  // Signal consumers to stop if they haven't already.
  for (auto& t : consumers) t.join();

  EXPECT_EQ(consumed.load(), kTotal);
  for (int i = 0; i < kTotal; i++) EXPECT_EQ(seen[i], 1) << "item " << i;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
