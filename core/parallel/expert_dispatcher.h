// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef NVTX_DISABLE
  #include <nvtx3/nvtx3.hpp>
#endif

#include "common/sync.h"
#include "base/noncopyable.h"
#include "base/thread.h"
#include "utils/threadsafe_queue.h"
#include "expert_module.h"
#include "prefetch/expert_residency.h"

enum MUTEX_TYPE {
  INPUT_MUTEX = 0,
  OUTPUT_MUTEX = 1,
  EXEC_MUTEX = 2,
  PENDING_MUTEX = 3
};

struct CUevent_st;
using cudaEvent_t = CUevent_st*;

class ExpertDispatcher : public base::noncopyable {
 public:
  typedef struct {
    int layer_idx = -1;
    int expert_idx = -1;
    int gpu_id = -1;
    bool remote = false;
    bool wait_for_prefetch = false;
  } CallArgs;
  typedef struct {
    torch::Tensor hidden_states =
        torch::empty({0});  // shallow copy, real tensor in python code
    ExpertNodePtr expert_node = nullptr;
    int out_gpu_id = -1;
    torch::ScalarType out_dtype = torch::kFloat32;
    bool evict = false;
    bool hit = false;
    cudaEvent_t transfer_event = nullptr;
    std::uint64_t execution_lease_id = 0;
    ResidencyVariantKey execution_key;
  } ExecArgs;
  typedef std::tuple<torch::Tensor, int, int, int> CallResult;

 public:
  explicit ExpertDispatcher(int num_experts, int num_layers, int dtype,
                            int expert_type, int num_threads = 1);
  ~ExpertDispatcher() {
    main_thread_stop_flag_.store(true, std::memory_order_release);
    for (auto& expert_list : experts_) {
      for (auto& expert_node : expert_list) {
        if (expert_node && expert_node->node) {
          expert_node->node->exec_state.store(NodeExecState::IDLE,
                                              std::memory_order_release);
        }
      }
    }
    for (int i = 0; i < static_cast<int>(input_queue_.size()); ++i) {
      input_queue_[i].Close();
    }
    for (int i = 0; i < static_cast<int>(exec_queue_.size()); ++i) {
      exec_queue_[i].Close();
    }
    for (int i = 0; i < static_cast<int>(cache_cv_.size()); ++i) {
      cache_cv_[i].notify_all();
    }
    for (auto& t : threads_) {
      if (t) {
        t->join();
      }
    }
    for (auto& stream : exec_streams_) {
      cudaStreamDestroy(stream);
    }
    for (auto& stream : fetch_streams_) {
      cudaStreamDestroy(stream);
    }
    for (auto* m : modules_) {
      delete m;
    }
  }

  void SetInputs(const torch::Tensor& hidden_states,
                 const torch::Tensor& router_mask,
                 const torch::Tensor& router_weight);

  void EnqueueExpert(int layer_idx, int expert_idx, int gpu_id = -1,
                     bool remote = false);
  void NotifyFetchStart();

  void RegisterExpert(int layer_idx, int expert_idx,
                      const std::vector<std::uint32_t>& tensor_ids,
                      std::string jit_path);
  void ClearExpertCacheCounts();
  // Read-only observability accessors; neither alters routing/dispatch.
  std::int64_t GetCacheOccupancyBytes();
  double GetCacheHitRate() const;
  void SetExpectedQueue(int expected_pending = 0) {
    pending_.store(expected_pending);
  }

  void SetScales(const std::map<std::string, torch::Tensor>& scales);
  bool RegisterExpertVariant(
      int layer_idx, int expert_idx, const std::string& format,
      std::uint64_t generation, const std::string& execution,
      const std::vector<std::uint32_t>& tensor_ids,
      const std::vector<std::string>& tensor_roles, std::int64_t payload_bytes,
      std::int64_t aligned_bytes, std::int64_t workspace_bytes);
  bool SetPrecisionTargets(
      const std::vector<std::tuple<int, int, std::string, std::uint64_t>>&
          targets,
      std::uint64_t epoch);
  bool SetAdaptiveHbmBudgetBytes(std::int64_t bytes);
  ExpertPolicyStats GetPrecisionMetrics() const;
  std::vector<std::tuple<std::uint64_t, std::uint8_t, std::uint64_t,
                         std::int64_t, std::int64_t, std::uint8_t>>
  GetResidentGenerationEntries() const;
  std::uintptr_t GetResidencyManagerId() const;
  void ConfigureResidencyManager(bool manager_enabled,
                                 bool phase_policy_enabled);
  std::string GetActiveFormat() const;
#ifdef MOE_INFINITY_TESTING
  void InjectTransitionFailureOnceForTest(int layer_idx, int expert_idx,
                                          const std::string& format);
#endif

  std::vector<CallResult> WaitExpert() { return Wait(); }
  torch::Tensor WaitHiddenStates();

 private:
  void Enqueue(CallArgs& args);
  std::vector<CallResult> Wait();
  void Start() { start_ = true; }

  void GPUFetchFunc(int gpu_id);
  void GPUExecFunc(int gpu_id, int thread_idx);
  bool ApplyPrecisionTarget(const ExpertNodePtr& expert_node, int gpu_id);

  // void GPUThreadFunc(int gpu_id);

  void OutputFunc(ExecArgs args, torch::Tensor output, torch::Tensor token_mask,
                  int gpu_id);

  ExpertNodePtr FindExpertEvict(int gpu_id);

 private:
  std::vector<std::unique_ptr<base::Thread>> threads_;
  mutable std::mutex mutex_;
  // std::vector<std::deque<CallArgs>> input_queue_;
  std::vector<ThreadSafeQueue<CallArgs>> input_queue_;
  // std::vector<std::deque<ExecArgs>> exec_queue_;
  std::vector<ThreadSafeQueue<ExecArgs>> exec_queue_;
  std::vector<CallResult> output_queue_;
  std::vector<std::vector<ExpertNodePtr>> experts_;
  std::atomic<size_t> num_enqueued_;
  bool start_;
  int expert_type_;
  int dtype_;
  int num_experts_;
  std::atomic<bool> main_thread_stop_flag_;

  std::atomic<size_t> pending_;

  // Passive counters for GetCacheHitRate(); reset by ClearExpertCacheCounts().
  std::atomic<std::uint64_t> cache_hit_count_{0};
  std::atomic<std::uint64_t> cache_access_count_{0};

  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;

  // std::vector<std::mutex> input_mutex_;
  // std::vector<std::mutex> exec_mutex_;
  // std::vector<std::condition_variable> input_cv_;
  // std::vector<std::condition_variable> exec_cv_;

  std::vector<std::mutex> cache_mutex_;
  std::vector<std::condition_variable> cache_cv_;

  std::mutex output_mutex_;
  std::mutex accum_mutex_;

  std::vector<cudaStream_t> exec_streams_;
  std::vector<cudaStream_t> fetch_streams_;

  std::unique_ptr<std::atomic<bool>[]> gpu_overload_;

  torch::Tensor hidden_states_;
  torch::Tensor final_hidden_states_;
  torch::Tensor router_mask_;
  torch::Tensor router_weight_;

  std::vector<int64_t> cache_sizes_;
  std::vector<std::unordered_set<uint64_t>> cached_experts_;

  int cache_capacity_ = 0;
  int num_threads_ = 1;

  std::vector<MoEMLP*> modules_;

  bool fp8_in_store_ = false;
  std::vector<std::vector<std::vector<torch::Tensor>>> fp8_scales_;
  std::map<ResidencyVariantKey, ResidencyVariant> registered_variants_;
  std::map<ResidencyVariantKey, ExpertExecutionDescriptor>
      execution_descriptors_;
  std::unordered_map<std::uint64_t, ResidencyVariantKey> precision_targets_;
  std::uint64_t precision_epoch_ = 0;
  std::uint64_t published_generation_ = 0;
  std::int64_t adaptive_hbm_budget_bytes_ = 0;
  std::int64_t transition_failed_count_ = 0;
  std::int64_t h2d_payload_bytes_ = 0;
  std::int64_t h2d_transfers_ = 0;
  std::int64_t promotions_ = 0;
  std::int64_t demotions_ = 0;
  std::int64_t representation_hits_ = 0;
  std::int64_t representation_misses_ = 0;
  bool manager_enabled_ = false;
  bool phase_policy_enabled_ = false;
  std::string active_format_ = "bf16";
#ifdef MOE_INFINITY_TESTING
  std::optional<std::pair<std::uint64_t, ExpertFormat>> fail_transition_once_;
#endif
};

#define SET_TENSORS_AND_MODULE_FROM_BLOB(cls, module, node, device, \
                                         jit_module)                \
  do {                                                              \
    reinterpret_cast<cls*>(module)->SetTensorsFromBlob(             \
        node->device_memory_ptr, node->tensor_ids, device);         \
    reinterpret_cast<cls*>(module)->SetModuleFromBlob(jit_module);  \
  } while (0)
