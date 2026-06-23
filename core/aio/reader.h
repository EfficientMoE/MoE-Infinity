// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "base/noncopyable.h"
#include "utils/logger.h"

namespace base {

// High-performance multi-threaded file reader with direct I/O
class DirectFileReader : public noncopyable {
 public:
  static constexpr size_t kOptimalChunkSize = 4 * 1024 * 1024;  // 4MB chunks

  struct ReadRequest {
    void* buffer;                   // Aligned buffer for direct I/O
    size_t offset;                  // File offset to read from
    size_t size;                    // Number of bytes to read
    std::promise<ssize_t> promise;  // Promise for async result

    ReadRequest(void* buf, size_t off, size_t sz)
        : buffer(buf), offset(off), size(sz) {}
  };

  explicit DirectFileReader(
      const std::string& filename,
      size_t num_threads = std::thread::hardware_concurrency(),
      size_t buffer_alignment = 4096)
      : filename_(filename),
        num_threads_(num_threads),
        buffer_alignment_(buffer_alignment),
        fd_(-1),
        file_size_(0),
        stop_flag_(false) {
    Open();
    StartWorkerThreads();
  }

  ~DirectFileReader() {
    Stop();
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  // Synchronous read - blocks until completion with 4MB chunking
  ssize_t Read(void* buffer, size_t offset, size_t size) {
    if (!IsAligned(buffer) || !IsAligned(size) || !IsAligned(offset)) {
      DLOG_ERROR("Buffer, size, and offset must be aligned to",
                 buffer_alignment_, "bytes");
      return -1;
    }

    // For small reads, use direct pread
    if (size <= kOptimalChunkSize) {
      return pread(fd_, buffer, size, offset);
    }

    // For large reads, break into 4MB chunks and parallelize
    const size_t num_chunks =
        (size + kOptimalChunkSize - 1) / kOptimalChunkSize;
    std::vector<std::future<ssize_t>> futures;
    futures.reserve(num_chunks);

    char* buf_ptr = static_cast<char*>(buffer);
    size_t remaining = size;
    size_t current_offset = offset;

    // Submit chunks as async requests
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t chunk_size = std::min(remaining, kOptimalChunkSize);

      futures.push_back(ReadAsync(buf_ptr + (i * kOptimalChunkSize),
                                  current_offset, chunk_size));

      current_offset += chunk_size;
      remaining -= chunk_size;
    }

    // Collect results
    ssize_t total_read = 0;
    for (auto& future : futures) {
      ssize_t chunk_result = future.get();
      if (chunk_result < 0) {
        return chunk_result;  // Return error
      }
      total_read += chunk_result;

      // If we got less than expected, we've hit EOF
      if (chunk_result < static_cast<ssize_t>(kOptimalChunkSize)) {
        break;
      }
    }

    return total_read;
  }

  // Asynchronous read - returns future for result
  std::future<ssize_t> ReadAsync(void* buffer, size_t offset, size_t size) {
    if (!IsAligned(buffer) || !IsAligned(size) || !IsAligned(offset)) {
      DLOG_ERROR("Buffer, size, and offset must be aligned to",
                 buffer_alignment_, "bytes");
      std::promise<ssize_t> promise;
      promise.set_value(-1);
      return promise.get_future();
    }

    auto request = std::make_unique<ReadRequest>(buffer, offset, size);
    auto future = request->promise.get_future();

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      request_queue_.push(std::move(request));
    }
    queue_cv_.notify_one();

    return future;
  }

  // Chunked asynchronous read - automatically breaks large reads into 4MB
  // chunks
  std::future<ssize_t> ReadAsyncChunked(void* buffer, size_t offset,
                                        size_t size) {
    if (!IsAligned(buffer) || !IsAligned(size) || !IsAligned(offset)) {
      DLOG_ERROR("Buffer, size, and offset must be aligned to",
                 buffer_alignment_, "bytes");
      std::promise<ssize_t> promise;
      promise.set_value(-1);
      return promise.get_future();
    }

    // For small reads, use regular async
    if (size <= kOptimalChunkSize) {
      return ReadAsync(buffer, offset, size);
    }

    // For large reads, create a coordinating promise/future
    auto promise = std::make_shared<std::promise<ssize_t>>();
    auto future = promise->get_future();

    // Launch coordination in a separate thread
    std::thread([this, buffer, offset, size, promise]() {
      const size_t num_chunks =
          (size + kOptimalChunkSize - 1) / kOptimalChunkSize;
      std::vector<std::future<ssize_t>> chunk_futures;
      chunk_futures.reserve(num_chunks);

      char* buf_ptr = static_cast<char*>(buffer);
      size_t remaining = size;
      size_t current_offset = offset;

      // Submit all chunks
      for (size_t i = 0; i < num_chunks; ++i) {
        size_t chunk_size = std::min(remaining, kOptimalChunkSize);

        chunk_futures.push_back(ReadAsync(buf_ptr + (i * kOptimalChunkSize),
                                          current_offset, chunk_size));

        current_offset += chunk_size;
        remaining -= chunk_size;
      }

      // Collect results
      ssize_t total_read = 0;
      for (auto& chunk_future : chunk_futures) {
        ssize_t chunk_result = chunk_future.get();
        if (chunk_result < 0) {
          promise->set_value(chunk_result);
          return;
        }
        total_read += chunk_result;

        // If we got less than expected, we've hit EOF
        if (chunk_result < static_cast<ssize_t>(std::min(
                               remaining + chunk_result, kOptimalChunkSize))) {
          break;
        }
      }

      promise->set_value(total_read);
    }).detach();

    return future;
  }

  // Allocate aligned buffer for direct I/O
  void* AllocateAlignedBuffer(size_t size) {
    size_t aligned_size = AlignUp(size, buffer_alignment_);
    void* buffer = nullptr;

    if (posix_memalign(&buffer, buffer_alignment_, aligned_size) != 0) {
      DLOG_ERROR("Failed to allocate aligned buffer:", strerror(errno));
      return nullptr;
    }

    PrefaultPages(buffer, aligned_size);
    DLOG_DEBUG("Allocated aligned buffer of size", aligned_size, "at", buffer);

    cudaHostRegister(buffer, aligned_size, cudaHostRegisterDefault);

    return buffer;
  }

  // Free aligned buffer
  void FreeAlignedBuffer(void* buffer) {
    if (buffer) {
      cudaHostUnregister(buffer);
      free(buffer);
    }
  }

  void PrefaultPages(void* buffer, size_t size) {
    for (size_t offset = 0; offset < size; offset += buffer_alignment_) {
      // Touch each page to ensure it's loaded into memory
      volatile char* ptr = static_cast<volatile char*>(buffer) + offset;
      *ptr = 0;  // Prevent compiler optimization
    }
    DLOG_DEBUG("Prefaulted", size, "bytes at", buffer);
  }

  // Get file size
  size_t GetFileSize() const { return file_size_; }

  // Get buffer alignment requirement
  size_t GetBufferAlignment() const { return buffer_alignment_; }

  // Check if value is aligned
  bool IsAligned(const void* ptr) const {
    return (reinterpret_cast<uintptr_t>(ptr) % buffer_alignment_) == 0;
  }

  bool IsAligned(size_t value) const {
    return (value % buffer_alignment_) == 0;
  }

  // Align value up to alignment boundary
  size_t AlignUp(size_t value, size_t alignment) const {
    return (value + alignment - 1) & ~(alignment - 1);
  }

  // Prefetch data to page cache (optional optimization)
  void Prefetch(size_t offset, size_t size) {
    if (posix_fadvise(fd_, offset, size, POSIX_FADV_WILLNEED) != 0) {
      DLOG_WARN("Prefetch failed:", strerror(errno));
    }
  }

  // Get number of worker threads
  size_t GetNumThreads() const { return num_threads_; }

  // Get optimal chunk size for I/O operations
  static constexpr size_t GetOptimalChunkSize() { return kOptimalChunkSize; }

  // Get pending request count
  size_t GetPendingRequests() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return request_queue_.size();
  }

 private:
  void Open() {
    // Open with direct I/O flags for best performance
    fd_ = open(filename_.c_str(), O_RDONLY | O_DIRECT | O_LARGEFILE);
    if (fd_ < 0) {
      // Fallback without O_DIRECT if not supported
      DLOG_WARN("Direct I/O not supported, falling back to buffered I/O");
      fd_ = open(filename_.c_str(), O_RDONLY | O_LARGEFILE);
      if (fd_ < 0) {
        DLOG_FATAL("Failed to open file", filename_, ":", strerror(errno));
        throw std::runtime_error("Failed to open file: " + filename_);
      }
    }

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) != 0) {
      DLOG_ERROR("Failed to get file size:", strerror(errno));
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to get file size");
    }
    file_size_ = st.st_size;

    DLOG_INFO("Opened file", filename_, "size:", file_size_, "bytes");
  }

  void StartWorkerThreads() {
    workers_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
      workers_.emplace_back(&DirectFileReader::WorkerLoop, this, i);
    }
    DLOG_INFO("Started", num_threads_, "worker threads");
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_flag_ = true;
    }
    queue_cv_.notify_all();

    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers_.clear();
  }

  void WorkerLoop(size_t worker_id) {
    DLOG_DEBUG("Worker", worker_id, "started");

    while (true) {
      std::unique_ptr<ReadRequest> request;

      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(
            lock, [this] { return stop_flag_ || !request_queue_.empty(); });

        if (stop_flag_ && request_queue_.empty()) {
          break;
        }

        if (!request_queue_.empty()) {
          request = std::move(request_queue_.front());
          request_queue_.pop();
        }
      }

      if (request) {
        // Perform the actual read operation
        ssize_t result =
            pread(fd_, request->buffer, request->size, request->offset);

        if (result < 0) {
          DLOG_ERROR("Worker", worker_id, "read failed:", strerror(errno));
        } else {
          DLOG_DEBUG("Worker", worker_id, "read", result, "bytes at offset",
                     request->offset);
        }

        // Set the result
        request->promise.set_value(result);
      }
    }

    DLOG_DEBUG("Worker", worker_id, "stopped");
  }

 private:
  const std::string filename_;
  const size_t num_threads_;
  const size_t buffer_alignment_;

  int fd_;
  size_t file_size_;

  std::vector<std::thread> workers_;
  std::queue<std::unique_ptr<ReadRequest>> request_queue_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<bool> stop_flag_;

  void* buffer_;
};

// RAII wrapper for aligned buffer
class AlignedBuffer : public noncopyable {
 public:
  AlignedBuffer(DirectFileReader& reader, size_t size)
      : reader_(reader), size_(size) {
    buffer_ = reader_.AllocateAlignedBuffer(size);
    if (!buffer_) {
      throw std::bad_alloc();
    }
  }

  ~AlignedBuffer() { reader_.FreeAlignedBuffer(buffer_); }

  void* get() { return buffer_; }
  const void* get() const { return buffer_; }
  size_t size() const { return size_; }

  template <typename T>
  T* as() {
    return static_cast<T*>(buffer_);
  }

  template <typename T>
  const T* as() const {
    return static_cast<const T*>(buffer_);
  }

 private:
  DirectFileReader& reader_;
  void* buffer_;
  size_t size_;
};

}  // namespace base
