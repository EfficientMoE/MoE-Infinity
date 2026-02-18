// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// C++ pybind11 test module for I/O integration tests.
// Exposes test functions callable from Python, each returning
// py::dict with {"passed": bool, "detail": str}.

#include <torch/extension.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <set>
#include <sstream>
#include <thread>

#include "aio/archer_aio_thread.h"
#include "aio/archer_aio_threadpool.h"
#include "aio/archer_aio_utils.h"
#include "aio/archer_prio_aio_handle.h"
#include "aio/archer_tensor_handle.h"
#include "aio/archer_tensor_index.h"
#include "memory/pinned_memory_pool.h"

namespace py = pybind11;

static py::dict make_result(bool passed, const std::string& detail) {
  py::dict d;
  d["passed"] = passed;
  d["detail"] = detail;
  return d;
}

// =========================================================================
// Tier 1 tests (no CUDA required)
// =========================================================================

// Test A: ArcherAioThread basic operation (Phase 0: CV waits, atomics)
static py::dict test_aio_thread_basic() {
  try {
    ArcherAioThread thread(0);
    thread.Start();

    std::atomic<int> counter{0};
    for (int i = 0; i < 10; i++) {
      AioCallback cb = [&counter]() -> int {
        counter.fetch_add(1);
        return 0;
      };
      thread.Enqueue(cb);
    }
    thread.Wait();
    thread.Stop();

    int count = counter.load();
    if (count != 10) {
      return make_result(false,
                         "Expected counter=10, got " + std::to_string(count));
    }
    return make_result(true, "All 10 callbacks executed");
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test B: ArcherAioThreadPool round-robin distribution (Phase 0C + Phase 1)
static py::dict test_aio_threadpool_round_robin(int num_threads) {
  try {
    ArcherAioThreadPool pool(num_threads);
    pool.Start();

    std::mutex ids_mutex;
    std::set<std::thread::id> thread_ids;
    int total = num_threads * 3;

    for (int i = 0; i < total; i++) {
      AioCallback cb = [&ids_mutex, &thread_ids]() -> int {
        {
          std::lock_guard<std::mutex> lock(ids_mutex);
          thread_ids.insert(std::this_thread::get_id());
        }
        // Sleep briefly so callbacks are distributed across threads
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return 0;
      };
      pool.Enqueue(cb);
    }
    pool.Wait();
    pool.Stop();

    int min_expected = std::min(num_threads, 2);
    bool passed = static_cast<int>(thread_ids.size()) >= min_expected;
    std::ostringstream oss;
    oss << "Used " << thread_ids.size() << " distinct threads"
        << " (min expected: " << min_expected << ")";
    return make_result(passed, oss.str());
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test C: Direct file I/O with O_DIRECT alignment
static py::dict test_file_io_direct(std::string tmpdir) {
  try {
    std::string filepath = tmpdir + "/test_direct_io";
    int fd = ArcherOpenFile(filepath.c_str());
    if (fd < 0) {
      return make_result(false, "Failed to open file: " + filepath);
    }

    const size_t buf_size = 4096;
    void* write_buf = nullptr;
    if (posix_memalign(&write_buf, 4096, buf_size) != 0) {
      ArcherCloseFile(fd);
      return make_result(false, "posix_memalign failed for write buffer");
    }
    // Fill with deterministic pattern
    memset(write_buf, 0xAB, buf_size);

    int ret = ArcherWriteFile(fd, write_buf, buf_size, 0);
    if (ret != 0) {
      free(write_buf);
      ArcherCloseFile(fd);
      return make_result(false, "ArcherWriteFile failed");
    }

    void* read_buf = nullptr;
    if (posix_memalign(&read_buf, 4096, buf_size) != 0) {
      free(write_buf);
      ArcherCloseFile(fd);
      return make_result(false, "posix_memalign failed for read buffer");
    }
    memset(read_buf, 0, buf_size);

    ret = ArcherReadFile(fd, read_buf, buf_size, 0);
    if (ret != 0) {
      free(write_buf);
      free(read_buf);
      ArcherCloseFile(fd);
      return make_result(false, "ArcherReadFile failed");
    }

    bool match = (memcmp(write_buf, read_buf, buf_size) == 0);
    free(write_buf);
    free(read_buf);
    ArcherCloseFile(fd);

    return make_result(match,
                       match ? "Write/read data matches" : "Data mismatch");
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test D: Thin wrapper for GetDefaultNumIoThreads (Phase 1)
static int get_default_num_io_threads() {
  return ArcherPrioAioContext::GetDefaultNumIoThreads();
}

// =========================================================================
// Tier 2 tests (CUDA required)
// =========================================================================

// Test E: PinnedMemoryPool acquire/release cycle (Phase 2)
static py::dict test_pinned_memory_pool(size_t chunk_size, int num_chunks) {
  try {
    auto pool = std::make_unique<PinnedMemoryPool>(chunk_size, num_chunks);

    // Acquire all chunks
    std::vector<void*> chunks;
    for (int i = 0; i < num_chunks; i++) {
      void* ptr = pool->Acquire();
      if (ptr == nullptr) {
        return make_result(
            false, "Acquire returned null for chunk " + std::to_string(i));
      }
      chunks.push_back(ptr);
    }

    // Verify all pointers are distinct
    std::set<void*> unique_ptrs(chunks.begin(), chunks.end());
    if (unique_ptrs.size() != static_cast<size_t>(num_chunks)) {
      return make_result(false, "Duplicate pointers detected");
    }

    // Release all
    for (auto* ptr : chunks) {
      pool->Release(ptr);
    }

    // Re-acquire and verify same set is reused (FIFO order)
    std::vector<void*> chunks2;
    for (int i = 0; i < num_chunks; i++) {
      chunks2.push_back(pool->Acquire());
    }

    std::set<void*> unique_ptrs2(chunks2.begin(), chunks2.end());
    bool same_set = (unique_ptrs2 == unique_ptrs);
    bool fifo = (chunks == chunks2);

    // Release all
    for (auto* ptr : chunks2) {
      pool->Release(ptr);
    }

    std::ostringstream oss;
    oss << "same_set=" << same_set << ", fifo_order=" << fifo;
    return make_result(same_set, oss.str());
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test F: PinnedMemoryPool contention - blocking when exhausted (Phase 2)
static py::dict test_pinned_pool_contention(size_t chunk_size) {
  try {
    auto pool = std::make_shared<PinnedMemoryPool>(chunk_size, 1);

    // Take the only chunk
    void* chunk = pool->Acquire();

    std::atomic<bool> bg_started{false};
    std::atomic<bool> bg_acquired{false};

    // Background thread tries to acquire (should block)
    std::thread bg([pool, &bg_started, &bg_acquired]() {
      bg_started.store(true);
      void* ptr = pool->Acquire();
      bg_acquired.store(true);
      pool->Release(ptr);
    });

    // Wait for background thread to start
    while (!bg_started.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Give background thread time to block on Acquire
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    bool was_blocked = !bg_acquired.load();

    // Release our chunk to unblock the background thread
    pool->Release(chunk);
    bg.join();

    bool bg_got_it = bg_acquired.load();
    bool passed = was_blocked && bg_got_it;

    std::ostringstream oss;
    oss << "was_blocked=" << was_blocked << ", bg_acquired=" << bg_got_it;
    return make_result(passed, oss.str());
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test G: Full tensor store/read roundtrip (Phase 1+2+4)
static py::dict test_tensor_store_read_roundtrip(std::string tmpdir) {
  // Reset global tensor index
  kTensorIndex = std::make_unique<ArcherTensorIndex>();

  bool match = false;
  std::string detail;
  try {
    ArcherTensorHandle handle(tmpdir, 4);

    auto tensor = torch::randn({256, 128});  // 131072 bytes (float32)
    handle.StoreTensor(42, tensor);

    auto aligned_size = handle.GetTensorSizeAligned(42);
    void* read_buf = nullptr;
    if (posix_memalign(&read_buf, 4096, aligned_size) != 0) {
      kTensorIndex = std::make_unique<ArcherTensorIndex>();
      return make_result(false, "posix_memalign failed");
    }

    handle.ReadTensor(42, read_buf, false);

    match = (memcmp(read_buf, tensor.data_ptr(), tensor.nbytes()) == 0);
    free(read_buf);

    detail = match ? "Roundtrip data matches (131072 bytes)"
                   : "Data mismatch after store/read";
  } catch (const std::exception& e) {
    detail = std::string("Exception: ") + e.what();
  }
  // handle destroyed here (joins threads, closes files)
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  return make_result(match, detail);
}

// Test H: Verify partition offset tracking (Phase 4)
static py::dict test_tensor_offset_tracking(std::string tmpdir) {
  // Reset global tensor index
  kTensorIndex = std::make_unique<ArcherTensorIndex>();

  bool passed = false;
  std::string detail;
  try {
    ArcherTensorHandle handle(tmpdir, 4);

    // Create tensors of known sizes (float32)
    auto t0 = torch::randn({256});  // 1024 bytes
    auto t1 = torch::randn({512});  // 2048 bytes
    auto t2 = torch::randn({128});  // 512 bytes

    handle.StoreTensor(0, t0);
    handle.StoreTensor(1, t1);
    handle.StoreTensor(2, t2);

    // Verify index entries
    auto it0 = kTensorIndex->find(0);
    auto it1 = kTensorIndex->find(1);
    auto it2 = kTensorIndex->find(2);

    if (it0 == kTensorIndex->end() || it1 == kTensorIndex->end() ||
        it2 == kTensorIndex->end()) {
      detail = "Missing tensor in index";
    } else {
      // All should have file_id == 0 (well under 10GB partition)
      bool all_file0 = (it0->second.file_id == 0) &&
                       (it1->second.file_id == 0) && (it2->second.file_id == 0);

      // Verify offsets: aligned(1024)=4096, aligned(2048)=4096
      int64_t expected_off0 = 0;
      int64_t expected_off1 = 4096;         // 0 + aligned(1024)
      int64_t expected_off2 = 4096 + 4096;  // 4096 + aligned(2048)

      bool offsets_ok = (it0->second.offset == expected_off0) &&
                        (it1->second.offset == expected_off1) &&
                        (it2->second.offset == expected_off2);

      passed = all_file0 && offsets_ok;

      std::ostringstream oss;
      oss << "file_ids=[" << it0->second.file_id << "," << it1->second.file_id
          << "," << it2->second.file_id << "] " << "offsets=["
          << it0->second.offset << "," << it1->second.offset << ","
          << it2->second.offset << "] " << "expected=[" << expected_off0 << ","
          << expected_off1 << "," << expected_off2 << "]";
      detail = oss.str();
    }
  } catch (const std::exception& e) {
    detail = std::string("Exception: ") + e.what();
  }
  // handle destroyed here
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  return make_result(passed, detail);
}

// =========================================================================
// New Tier-1 tests (no CUDA required)
// =========================================================================

// Test I: AioThread stops cleanly without pending callbacks.
static py::dict test_aio_thread_stop_empty() {
  try {
    ArcherAioThread thread(0);
    thread.Start();
    thread.Stop();  // must not hang
    return make_result(true, "Stop on idle thread returned cleanly");
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test J: AioThread handles error-returning callbacks without crashing.
static py::dict test_aio_thread_error_callback() {
  try {
    ArcherAioThread thread(0);
    thread.Start();

    std::atomic<int> executed{0};
    for (int i = 0; i < 5; i++) {
      AioCallback cb = [&executed, i]() -> int {
        executed.fetch_add(1);
        return (i % 2 == 0) ? -1 : 0;  // alternating errors
      };
      thread.Enqueue(cb);
    }
    thread.Wait();
    thread.Stop();

    bool ok = executed.load() == 5;
    return make_result(ok, "All 5 callbacks executed despite errors; count=" +
                               std::to_string(executed.load()));
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test K: AioThread correctly executes high-volume callbacks (throughput).
static py::dict test_aio_thread_throughput(int num_callbacks) {
  try {
    ArcherAioThread thread(0);
    thread.Start();

    std::atomic<int> counter{0};
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < num_callbacks; i++) {
      AioCallback cb = [&counter]() -> int {
        counter.fetch_add(1, std::memory_order_relaxed);
        return 0;
      };
      thread.Enqueue(cb);
    }
    thread.Wait();
    thread.Stop();

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = counter.load() == num_callbacks;
    std::ostringstream oss;
    oss << "Executed " << counter.load() << "/" << num_callbacks
        << " callbacks in " << ms << " ms";
    return make_result(ok, oss.str());
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test L: AioThreadPool targeted dispatch (specific thread_id).
static py::dict test_aio_threadpool_targeted(int num_threads) {
  try {
    ArcherAioThreadPool pool(num_threads);
    pool.Start();

    // Record which OS thread ID executed each callback per pool-thread slot.
    std::vector<std::atomic<int>> per_slot_count(num_threads);
    for (auto& c : per_slot_count) c.store(0);

    // Enqueue 3 callbacks to each slot specifically.
    int per_slot = 3;
    for (int t = 0; t < num_threads; t++) {
      for (int i = 0; i < per_slot; i++) {
        AioCallback cb = [&per_slot_count, t]() -> int {
          per_slot_count[t].fetch_add(1);
          return 0;
        };
        pool.Enqueue(cb, t);
      }
    }
    pool.Wait();
    pool.Stop();

    bool all_ok = true;
    std::ostringstream oss;
    for (int t = 0; t < num_threads; t++) {
      int count = per_slot_count[t].load();
      oss << "slot[" << t << "]=" << count << " ";
      if (count != per_slot) all_ok = false;
    }
    return make_result(all_ok, oss.str());
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test M: AioThreadPool concurrent enqueue from multiple producers.
static py::dict test_aio_threadpool_concurrent_producers(int num_threads,
                                                         int num_producers) {
  try {
    ArcherAioThreadPool pool(num_threads);
    pool.Start();

    constexpr int kCallbacksPerProducer = 50;
    std::atomic<int> total{0};

    std::vector<std::thread> producers;
    for (int p = 0; p < num_producers; p++) {
      producers.emplace_back([&pool, &total]() {
        for (int i = 0; i < kCallbacksPerProducer; i++) {
          AioCallback cb = [&total]() -> int {
            total.fetch_add(1, std::memory_order_relaxed);
            return 0;
          };
          pool.Enqueue(cb);
        }
      });
    }
    for (auto& t : producers) t.join();
    pool.Wait();
    pool.Stop();

    int expected = num_producers * kCallbacksPerProducer;
    bool ok = total.load() == expected;
    std::ostringstream oss;
    oss << "total=" << total.load() << " expected=" << expected;
    return make_result(ok, oss.str());
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test N: File I/O — multiple sizes including non-multiple of 4096.
// Data is zero-padded to the next 4096-byte boundary before write.
static py::dict test_file_io_various_sizes(std::string tmpdir) {
  try {
    // Sizes to test: exactly 4096, multiple of 4096, and a small buffer
    // that must be padded.
    const std::vector<size_t> sizes = {4096, 8192, 16384, 512};

    for (size_t sz : sizes) {
      std::string path = tmpdir + "/test_sz_" + std::to_string(sz);
      int fd = ArcherOpenFile(path.c_str());
      if (fd < 0)
        return make_result(false, "open failed for size " + std::to_string(sz));

      // Pad write to next 4096-byte boundary.
      size_t aligned = ((sz + 4095) / 4096) * 4096;

      void* wbuf = nullptr;
      if (posix_memalign(&wbuf, 4096, aligned) != 0) {
        ArcherCloseFile(fd);
        return make_result(false, "memalign failed");
      }
      memset(wbuf, 0xCD, aligned);  // deterministic pattern

      if (ArcherWriteFile(fd, wbuf, aligned, 0) != 0) {
        free(wbuf);
        ArcherCloseFile(fd);
        return make_result(false,
                           "write failed for size " + std::to_string(sz));
      }

      void* rbuf = nullptr;
      if (posix_memalign(&rbuf, 4096, aligned) != 0) {
        free(wbuf);
        ArcherCloseFile(fd);
        return make_result(false, "memalign (read) failed");
      }
      memset(rbuf, 0, aligned);

      if (ArcherReadFile(fd, rbuf, aligned, 0) != 0) {
        free(wbuf);
        free(rbuf);
        ArcherCloseFile(fd);
        return make_result(false, "read failed for size " + std::to_string(sz));
      }

      bool match = (memcmp(wbuf, rbuf, aligned) == 0);
      free(wbuf);
      free(rbuf);
      ArcherCloseFile(fd);

      if (!match)
        return make_result(false,
                           "data mismatch for size " + std::to_string(sz));
    }

    return make_result(true, "All sizes matched: 4096, 8192, 16384, 512");
  } catch (const std::exception& e) {
    return make_result(false, std::string("Exception: ") + e.what());
  }
}

// Test O: Multiple tensor roundtrips — different dtypes (float32, float16).
static py::dict test_multi_tensor_roundtrip(std::string tmpdir) {
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  bool passed = false;
  std::string detail;
  try {
    ArcherTensorHandle handle(tmpdir, 4);

    auto t_f32 = torch::randn({64, 64});                   // float32
    auto t_f16 = torch::randn({64, 64}).to(torch::kHalf);  // float16

    handle.StoreTensor(10, t_f32);
    handle.StoreTensor(11, t_f16);

    // Read float32
    size_t sz32 = handle.GetTensorSizeAligned(10);
    void* buf32 = nullptr;
    if (posix_memalign(&buf32, 4096, sz32) != 0) {
      kTensorIndex = std::make_unique<ArcherTensorIndex>();
      return make_result(false, "memalign failed (f32)");
    }
    handle.ReadTensor(10, buf32, false);
    bool match32 = (memcmp(buf32, t_f32.data_ptr(), t_f32.nbytes()) == 0);
    free(buf32);

    // Read float16
    size_t sz16 = handle.GetTensorSizeAligned(11);
    void* buf16 = nullptr;
    if (posix_memalign(&buf16, 4096, sz16) != 0) {
      kTensorIndex = std::make_unique<ArcherTensorIndex>();
      return make_result(false, "memalign failed (f16)");
    }
    handle.ReadTensor(11, buf16, false);
    bool match16 = (memcmp(buf16, t_f16.data_ptr(), t_f16.nbytes()) == 0);
    free(buf16);

    passed = match32 && match16;
    std::ostringstream oss;
    oss << "float32_match=" << match32 << " float16_match=" << match16;
    detail = oss.str();
  } catch (const std::exception& e) {
    detail = std::string("Exception: ") + e.what();
  }
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  return make_result(passed, detail);
}

// Test P: Tensor index serialize / deserialize roundtrip.
static py::dict test_tensor_index_serialization(std::string tmpdir) {
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  bool passed = false;
  std::string detail;
  try {
    {
      ArcherTensorHandle handle(tmpdir, 2);
      auto t0 = torch::randn({32});
      auto t1 = torch::randn({64});
      handle.StoreTensor(0, t0);
      handle.StoreTensor(1, t1);

      // Serialize index.
      std::string idx_path = tmpdir + "/index.bin";
      kTensorIndex->Serialize(idx_path.c_str());
    }

    // Reset and deserialize.
    kTensorIndex = std::make_unique<ArcherTensorIndex>();
    std::string idx_path = tmpdir + "/index.bin";
    kTensorIndex->Deserialize(idx_path.c_str());

    bool has0 = kTensorIndex->find(0) != kTensorIndex->end();
    bool has1 = kTensorIndex->find(1) != kTensorIndex->end();

    if (!has0 || !has1) {
      detail = "Missing entries after deserialize";
    } else {
      auto& m0 = (*kTensorIndex)[0];
      auto& m1 = (*kTensorIndex)[1];
      // Both in file 0.
      bool file_ok = (m0.file_id == 0) && (m1.file_id == 0);
      // t0 is 32 floats = 128 bytes → aligned to 4096
      // t1 starts at 4096
      bool offset_ok = (m0.offset == 0) && (m1.offset == 4096);
      passed = file_ok && offset_ok;
      std::ostringstream oss;
      oss << "file_ids=[" << m0.file_id << "," << m1.file_id << "] "
          << "offsets=[" << m0.offset << "," << m1.offset << "]";
      detail = oss.str();
    }
  } catch (const std::exception& e) {
    detail = std::string("Exception: ") + e.what();
  }
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  return make_result(passed, detail);
}

// =========================================================================
// Module definition
// =========================================================================

PYBIND11_MODULE(test_io_module, m) {
  m.doc() = "C++ test harnesses for MoE-Infinity I/O layer";

  // Tier 1 (no CUDA)
  m.def("test_aio_thread_basic", &test_aio_thread_basic,
        "Test ArcherAioThread basic callback execution");
  m.def("test_aio_thread_stop_empty", &test_aio_thread_stop_empty,
        "Test ArcherAioThread clean stop with no pending callbacks");
  m.def("test_aio_thread_error_callback", &test_aio_thread_error_callback,
        "Test ArcherAioThread with error-returning callbacks");
  m.def("test_aio_thread_throughput", &test_aio_thread_throughput,
        "Measure ArcherAioThread throughput", py::arg("num_callbacks") = 5000);
  m.def("test_aio_threadpool_round_robin", &test_aio_threadpool_round_robin,
        "Test ArcherAioThreadPool round-robin distribution",
        py::arg("num_threads") = 4);
  m.def("test_aio_threadpool_targeted", &test_aio_threadpool_targeted,
        "Test ArcherAioThreadPool targeted per-slot enqueue",
        py::arg("num_threads") = 4);
  m.def("test_aio_threadpool_concurrent_producers",
        &test_aio_threadpool_concurrent_producers,
        "Test ArcherAioThreadPool with concurrent producer threads",
        py::arg("num_threads") = 4, py::arg("num_producers") = 8);
  m.def("test_file_io_direct", &test_file_io_direct,
        "Test O_DIRECT file I/O write/read roundtrip", py::arg("tmpdir"));
  m.def("test_file_io_various_sizes", &test_file_io_various_sizes,
        "Test O_DIRECT I/O with 4096, 8192, 16384, and 512-byte payloads",
        py::arg("tmpdir"));
  m.def("get_default_num_io_threads", &get_default_num_io_threads,
        "Return ArcherPrioAioContext::GetDefaultNumIoThreads()");

  // Tier 2 (CUDA required)
  m.def("test_pinned_memory_pool", &test_pinned_memory_pool,
        "Test PinnedMemoryPool acquire/release cycle",
        py::arg("chunk_size") = 1048576, py::arg("num_chunks") = 4);
  m.def("test_pinned_pool_contention", &test_pinned_pool_contention,
        "Test PinnedMemoryPool blocking when exhausted",
        py::arg("chunk_size") = 1048576);
  m.def("test_tensor_store_read_roundtrip", &test_tensor_store_read_roundtrip,
        "Test full tensor store/read roundtrip via ArcherTensorHandle",
        py::arg("tmpdir"));
  m.def("test_tensor_offset_tracking", &test_tensor_offset_tracking,
        "Test partition offset tracking in ArcherTensorIndex",
        py::arg("tmpdir"));
  m.def("test_multi_tensor_roundtrip", &test_multi_tensor_roundtrip,
        "Test store/read roundtrip for float32 and float16 tensors",
        py::arg("tmpdir"));
  m.def("test_tensor_index_serialization", &test_tensor_index_serialization,
        "Test TensorIndex serialize/deserialize roundtrip", py::arg("tmpdir"));
}
