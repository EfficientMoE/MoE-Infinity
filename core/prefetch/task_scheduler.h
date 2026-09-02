// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/noncopyable.h"
#include "common/pytorch.h"
#include "model/model_topology.h"

#define SKIP_TO_NEXT_ITERATION                                \
  std::this_thread::sleep_for(std::chrono::microseconds(10)); \
  continue;

#define NUM_PRIORITY 20UL

// Priority bands, lowest value serviced first (GPUThreadFunc scans queue 0 up).
// Band 0 is also the on-demand queue: dedup sweeps skip index 0, so route-ahead
// work parked at 0 keeps on-demand's non-preemptible semantics.
constexpr std::uint32_t kOnDemandPriority = 0;
constexpr std::uint32_t kRouteAheadPriority = 1;
constexpr std::uint32_t kBackgroundPrefetchPriority = 2;

struct Task {
  bool on_demand = false;
  NodePtr node;
  std::vector<NodePtr> remove_nodes;
  std::uint32_t priority;
  std::uint64_t request_id;
  torch::Device src_device = DISK_DEVICE;
  torch::Device dst_device = DISK_DEVICE;
  cudaStream_t stream = nullptr;

  bool remove_layer = false;

  std::string DebugString() {
    std::stringstream ss;
    ss << "Task: node: " << node->str() << ", on_demand: " << on_demand
       << ", priority: " << priority << "[" << src_device.str() << "->"
       << dst_device.str() << "]";
    return ss.str();
  }
};
typedef std::shared_ptr<Task> TaskPtr;

struct SparseVictimReservation {
  std::uint64_t id;
  int device_id;
  std::int64_t target_bytes;
  std::int64_t resident_bytes;
  bool ready;
  std::string reason;
  std::vector<NodePtr> victims;
};

struct SparseCacheResizeResult {
  ResizeOutcome outcome;
  int device_id;
  std::int64_t target_bytes;
  std::int64_t resident_bytes;
  std::string reason;
};

class ArcherTaskPool : public base::noncopyable {
 public:
  void StartExec(const std::uint64_t& request_id, const NodePtr& node);
  void FetchExec(const std::uint64_t& request_id, const NodePtr& node);
  void StopExec(const std::uint64_t& request_id, const NodePtr& node);
  void EnqueueTask(const TaskPtr& task);

  void ClearQueue() {
    std::lock_guard<std::mutex> lock(unified_mutex_);
    for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
      unified_queue_[priority].clear();
    }
  }

  bool RemoveCachedSparseNode(const NodePtr& node, int device_id = -1);
  bool RemoveCachedDenseNode(const NodePtr& node);
  // void RemoveCachedNode(const NodePtr& node);

  void ReplaceCacheCandidates(const NodePtrList& candidates) {
    std::lock_guard<std::mutex> lock(unified_mutex_);
    {
      std::lock_guard<std::mutex> lock(this->candidates_mutex_);
      candidates_.clear();
      for (auto& node : candidates) {
        candidates_.insert(node);
      }
    }

    for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
      unified_queue_[priority].clear();
    }
  }

  SparseVictimReservation ReserveSparseCacheVictims(int device_id,
                                                    std::int64_t target_bytes);
  void CancelSparseCacheReservation(std::uint64_t id);
  SparseCacheResizeResult CommitSparseCacheReservation(std::uint64_t id);

#ifdef MOE_BUILD_TESTS
  void SetAfterExecSnapshotHookForTest(std::function<void()> hook);
  std::vector<NodePtr> SnapshotResizeExclusionsForTest(int device_id);
#endif

  DELETE_COPY_AND_ASSIGN(ArcherTaskPool);
  STATIC_GET_INSTANCE(ArcherTaskPool);

  ArcherTaskPool();
  ~ArcherTaskPool() {
    main_thread_stop_flag_.store(true, std::memory_order_release);
    for (auto& thread_list : exec_threads_) {
      for (auto& thread : thread_list) {
        if (thread.joinable()) thread.detach();
      }
    }
  }

 private:
  void GPUThreadFunc(int gpu_id, int thread_id);

  void SetNodeDevice(const TaskPtr& task);

  std::string DebugString(const std::vector<std::deque<TaskPtr>>& queue);

  // Snapshots the exec-queue membership and protected candidates for a device
  // WITHOUT ever holding exec_mutex_ and candidates_mutex_ at the same time.
  // Returns nodes that must be excluded from resize victim selection.
  std::vector<NodePtr> SnapshotResizeExclusions(int device_id);

 private:
  std::vector<std::deque<TaskPtr>> unified_queue_;  // For ordered prefetch
  std::vector<std::vector<std::uint32_t>> gpu_min_priority_;
  std::unordered_map<std::uint64_t, TaskPtr> exec_queue_;
  // Source lock rule: unified_mutex_ may precede candidates_mutex_ (see
  // ReplaceCacheCandidates). exec_mutex_ and candidates_mutex_ are snapshot
  // locks and are NEVER nested in either order.
  std::mutex exec_mutex_;
  std::mutex unified_mutex_;
  std::mutex candidates_mutex_;

  std::mutex reservation_mutex_;
  std::uint64_t next_reservation_id_ = 1;
  std::unordered_map<std::uint64_t, SparseVictimReservation> reservations_;
#ifdef MOE_BUILD_TESTS
  std::function<void()> after_exec_snapshot_hook_for_test_;
#endif

  std::vector<std::list<std::thread>> exec_threads_;

  std::unordered_set<NodePtr> candidates_;

  std::atomic<bool> main_thread_stop_flag_;
};

extern std::unique_ptr<ArcherTaskPool> kTaskPool;
