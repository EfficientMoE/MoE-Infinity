// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Tests for utils/simple_object_pool.h (SimpleObjectPool<T>)

#include "utils/simple_object_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <set>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

struct Widget {
  int value;
  Widget() : value(0) {}
  explicit Widget(int v) : value(v) {}
};

// ---------------------------------------------------------------------------
// Basic functional tests
// ---------------------------------------------------------------------------

TEST(SimpleObjectPoolTest, GetDefaultCreatesObject) {
  SimpleObjectPool<Widget> pool;
  auto ptr = pool.getDefault();
  ASSERT_NE(ptr.get(), nullptr);
}

TEST(SimpleObjectPoolTest, GetWithFactory) {
  SimpleObjectPool<Widget> pool;
  auto ptr = pool.get([] { return new Widget(42); });
  ASSERT_NE(ptr.get(), nullptr);
  EXPECT_EQ(ptr->value, 42);
}

// When the unique_ptr is destroyed, the object must be returned to the pool
// and reused on the next get().
TEST(SimpleObjectPoolTest, ObjectReturnedToPoolOnDestroy) {
  SimpleObjectPool<Widget> pool;

  Widget* raw = nullptr;
  {
    auto ptr = pool.getDefault();
    ptr->value = 99;
    raw = ptr.get();
  }  // ptr destroyed here → object returned to pool

  // Next get() must hand back the same object (stack reuse).
  auto ptr2 = pool.getDefault();
  EXPECT_EQ(ptr2.get(), raw);
  // Value survives because we just recycle the allocation.
  EXPECT_EQ(ptr2->value, 99);
}

// Pool with multiple returned objects acts as a LIFO stack of free objects.
TEST(SimpleObjectPoolTest, LIFOReuse) {
  SimpleObjectPool<Widget> pool;

  auto a = pool.getDefault();
  auto b = pool.getDefault();
  Widget* raw_a = a.get();
  Widget* raw_b = b.get();

  b.reset();  // return b first → top of stack
  a.reset();  // return a second → new top of stack

  // Next two get()s should return a, then b (LIFO).
  auto first = pool.getDefault();
  auto second = pool.getDefault();
  EXPECT_EQ(first.get(), raw_a);
  EXPECT_EQ(second.get(), raw_b);
}

TEST(SimpleObjectPoolTest, FactoryNotCalledWhenPoolHasObject) {
  SimpleObjectPool<Widget> pool;
  std::atomic<int> factory_calls{0};

  auto ptr = pool.get([&factory_calls] {
    factory_calls.fetch_add(1);
    return new Widget;
  });
  ptr.reset();  // return to pool

  auto ptr2 = pool.get([&factory_calls] {
    factory_calls.fetch_add(1);
    return new Widget;
  });
  (void)ptr2;

  // Factory was called exactly once (for the first get, not the second).
  EXPECT_EQ(factory_calls.load(), 1);
}

TEST(SimpleObjectPoolTest, FactoryCalledWhenPoolIsEmpty) {
  SimpleObjectPool<Widget> pool;
  std::atomic<int> factory_calls{0};

  // No pre-existing objects; factory must be called.
  auto ptr = pool.get([&factory_calls] {
    factory_calls.fetch_add(1);
    return new Widget;
  });
  EXPECT_EQ(factory_calls.load(), 1);
  (void)ptr;
}

// ---------------------------------------------------------------------------
// getMany()
// ---------------------------------------------------------------------------

TEST(SimpleObjectPoolTest, GetManyReturnsCorrectCount) {
  SimpleObjectPool<Widget> pool;
  auto ptrs = pool.getDefaultMany(5);
  EXPECT_EQ(ptrs.size(), 5u);
  for (auto& p : ptrs) EXPECT_NE(p.get(), nullptr);
}

TEST(SimpleObjectPoolTest, GetManyWithPartialPool) {
  SimpleObjectPool<Widget> pool;
  std::atomic<int> factory_calls{0};

  // Pre-populate pool with 2 objects.
  {
    auto a = pool.getDefault();
    auto b = pool.getDefault();
  }  // both returned to pool

  // Ask for 5 → 2 from pool + 3 from factory.
  auto ptrs = pool.getMany(5, [&factory_calls] {
    factory_calls.fetch_add(1);
    return new Widget;
  });

  EXPECT_EQ(ptrs.size(), 5u);
  EXPECT_EQ(factory_calls.load(), 3);
}

TEST(SimpleObjectPoolTest, GetManyAllFromFactory) {
  SimpleObjectPool<Widget> pool;
  std::atomic<int> factory_calls{0};
  auto ptrs = pool.getMany(4, [&factory_calls] {
    factory_calls.fetch_add(1);
    return new Widget;
  });
  EXPECT_EQ(ptrs.size(), 4u);
  EXPECT_EQ(factory_calls.load(), 4);
}

TEST(SimpleObjectPoolTest, GetManyAllDistinct) {
  SimpleObjectPool<Widget> pool;
  auto ptrs = pool.getDefaultMany(8);
  std::set<Widget*> addresses;
  for (auto& p : ptrs) addresses.insert(p.get());
  EXPECT_EQ(addresses.size(), 8u);
}

TEST(SimpleObjectPoolTest, GetManyReturnedObjectsReused) {
  SimpleObjectPool<Widget> pool;

  auto ptrs = pool.getDefaultMany(3);
  std::set<Widget*> addrs;
  for (auto& p : ptrs) addrs.insert(p.get());

  // Return all.
  ptrs.clear();

  // Retrieve them again — same addresses expected.
  auto ptrs2 = pool.getDefaultMany(3);
  std::set<Widget*> addrs2;
  for (auto& p : ptrs2) addrs2.insert(p.get());

  EXPECT_EQ(addrs, addrs2);
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------

TEST(SimpleObjectPoolTest, ConcurrentGetReturnDoesNotCorrupt) {
  constexpr int kThreads = 8;
  constexpr int kIterations = 200;

  SimpleObjectPool<Widget> pool;
  std::atomic<int> errors{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([&pool, &errors, t]() {
      for (int i = 0; i < kIterations; i++) {
        auto ptr = pool.get([t, i] { return new Widget(t * 1000 + i); });
        if (ptr.get() == nullptr) {
          errors.fetch_add(1);
        }
        // Object returned to pool when ptr goes out of scope.
      }
    });
  }
  for (auto& th : threads) th.join();

  EXPECT_EQ(errors.load(), 0);
}

TEST(SimpleObjectPoolTest, ConcurrentGetManyDoesNotCorrupt) {
  SimpleObjectPool<Widget> pool;
  std::atomic<int> errors{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < 4; t++) {
    threads.emplace_back([&pool, &errors]() {
      for (int i = 0; i < 50; i++) {
        auto ptrs = pool.getDefaultMany(3);
        if (ptrs.size() != 3) {
          errors.fetch_add(1);
        }
        // All returned when ptrs goes out of scope.
      }
    });
  }
  for (auto& th : threads) th.join();

  EXPECT_EQ(errors.load(), 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
