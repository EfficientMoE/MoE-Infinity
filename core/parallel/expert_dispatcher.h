// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
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
#include "memory/event_pool.h"
#include "expert_drop_select.h"
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

struct ExpertComputeSample {
  std::uint64_t invocation_id = 0;
  int layer_id = -1;
  int expert_id = -1;
  int gpu_id = -1;
  std::int64_t kernel_start_offset_ns = 0;
  std::int64_t kernel_end_offset_ns = 0;
  std::int64_t kernel_duration_ns = 0;
  std::int64_t forward_return_host_ns = 0;
  std::int64_t output_complete_host_ns = 0;
  std::int64_t output_delay_ns = 0;
};

struct GroupedExecBatch;

class ExpertDispatcher : public base::noncopyable {
 public:
  typedef struct {
    int layer_idx = -1;
    int expert_idx = -1;
    int gpu_id = -1;
    bool remote = false;
    bool wait_for_prefetch = false;
    ExpertPhase phase = ExpertPhase::MIXED;
    std::uint64_t invocation_id = 0;
    // Stamp the creating generation so late workers cannot underflow a failed
    // generation's pending count.
    std::uint64_t generation = 0;
    bool cache_slot_reserved = false;
    bool cache_key_inserted = false;
  } CallArgs;
  typedef struct {
    torch::Tensor hidden_states =
        torch::empty({0});  // shallow copy, real tensor in python code
    ExpertNodePtr expert_node = nullptr;
    int out_gpu_id = -1;
    torch::ScalarType out_dtype = torch::kFloat32;
    bool evict = false;
    bool hit = false;
    bool managed_transient = false;
    std::uint64_t residency_lease = 0;
    cudaEvent_t transfer_event = nullptr;
    std::uint64_t invocation_id = 0;
    std::uint64_t generation = 0;
    bool cache_slot_reserved = false;
    bool cache_key_inserted = false;
    bool pending_counted_down = false;
    std::uint64_t execution_lease_id = 0;
    ResidencyVariantKey execution_key;
    // Set on the single exec item that runs a whole group of resident experts
    // with the grouped decode kernels (expert_node is then nullptr).
    std::shared_ptr<GroupedExecBatch> group;
  } ExecArgs;
  typedef std::tuple<torch::Tensor, int, int, int> CallResult;

  struct ResizeToken {
    std::uint64_t id = 0;
    int device_id = -1;
    bool ready = false;
    std::string reason;
  };

  typedef struct {
    int layer_idx = -1;
    torch::Tensor active_flags_host;
    // Staged only when the fused expert-drop policy is enabled: pinned host
    // copies of router_mask_ (bool [tokens, E]) and router_weight_ (float32
    // [tokens, E]) landed by the same stream as active_flags_host.
    torch::Tensor router_mask_host;
    torch::Tensor router_weight_host;
    std::uint64_t generation = 0;
  } RouteArgs;

  typedef struct {
    ExpertDispatcher* dispatcher = nullptr;
    RouteArgs route;
  } RouteCallbackArgs;

  enum class DispatchFaultPoint : int {
    NONE = 0,
    ROUTE_CALLBACK = 1,
    ROUTE_WORKER = 2,
    FETCH_WORKER = 3,
    EXEC_WORKER = 4,
    OUTPUT = 5,
    COMPLETION_EVENT_RECORD = 6,
    RETIREMENT_CALLBACK_LAUNCH = 7,
    SUBMISSION = 8,
  };

  typedef struct {
    ExpertNodePtr expert_node = nullptr;
    int gpu_id = -1;
    bool cache_slot_reserved = false;
    bool cache_key_inserted = false;
    bool overload_owned = false;
  } FailureContext;

  typedef struct {
    ExpertDispatcher* dispatcher = nullptr;
    ExpertNodePtr expert_node = nullptr;
    std::uint64_t generation = 0;
    int gpu_id = -1;
    bool evict = false;
    bool managed_transient = false;
    std::uint64_t residency_lease = 0;
    std::uint64_t execution_lease_id = 0;
    ResidencyVariantKey execution_key;
  } ExpertRetireArgs;

  struct GroupRetireArgs {
    ExpertDispatcher* dispatcher = nullptr;
    std::uint64_t generation = 0;
    std::vector<ExpertRetireArgs> items;
  };

  struct CompletionEventRecord {
    cudaEvent_t event = nullptr;
    std::uint64_t generation = 0;
    int device = 0;
  };
  struct CompletionRetireBatch {
    ExpertDispatcher* dispatcher = nullptr;
    std::vector<CompletionEventRecord> records;
  };
  struct CompletionRetireItem {
    CompletionEventRecord record;
    bool caller_wait_consumed = false;
  };

 public:
  explicit ExpertDispatcher(int num_experts, int num_layers, int dtype,
                            int expert_type, int num_threads = 1);
  ~ExpertDispatcher() {
    // Let any in-flight dispatch finish routing and expert execution before
    // draining callbacks: OutputFunc launches retirement/completion callbacks,
    // so waiting for the pipeline to quiesce prevents a callback from being
    // launched against a half-destroyed dispatcher.
    {
      std::unique_lock<std::mutex> lock(pending_mutex_);
      pending_cv_.wait(lock, [&] {
        return !route_pending_.load(std::memory_order_acquire) &&
               pending_.load(std::memory_order_acquire) == 0;
      });
    }
    {
      std::unique_lock<std::mutex> lock(route_callback_mutex_);
      route_callback_cv_.wait(lock, [&] {
        return pending_route_callbacks_.load(std::memory_order_acquire) == 0;
      });
    }
    {
      std::unique_lock<std::mutex> lock(completion_retirement_callback_mutex_);
      completion_retirement_callback_cv_.wait(lock, [&] {
        return pending_completion_retirement_callbacks_.load(
                   std::memory_order_acquire) == 0;
      });
    }
    {
      std::unique_lock<std::mutex> lock(retirement_callback_mutex_);
      retirement_callback_cv_.wait(lock, [&] {
        return pending_retirement_callbacks_.load(std::memory_order_acquire) ==
               0;
      });
    }
    FlushExpertDropTrace();
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
    route_queue_.Close();
    completion_retirement_queue_.Close();
    retirement_queue_.Close();
    for (int i = 0; i < static_cast<int>(cache_cv_.size()); ++i) {
      NotifyCacheWaiters(i);
    }
    pending_cv_.notify_all();
    for (auto& t : threads_) {
      if (t) {
        t->join();
      }
    }
    // Destructor-only synchronization: drain every device so no in-flight
    // kernel, stream callback, or completion event still references the
    // modules, streams, or events destroyed below.
    for (int device = 0; device < kNumDevices(); ++device) {
      cudaSetDevice(device);
      cudaDeviceSynchronize();
    }
    for (auto& record : completion_events_) {
      if (record.event != nullptr) {
        cudaEventSynchronize(record.event);
        cudaEventDestroy(record.event);
        completion_events_retired_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    for (auto& record : destructor_fallback_events_) {
      if (record.event != nullptr) {
        cudaEventSynchronize(record.event);
        cudaEventDestroy(record.event);
        completion_events_retired_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    for (auto& completion_pool : completion_event_pool_) {
      for (auto& event : completion_pool.second) {
        cudaEventDestroy(event);
      }
    }
    for (auto& stream : exec_streams_) {
      cudaStreamDestroy(stream);
    }
    for (auto& stream : fetch_streams_) {
      cudaStreamDestroy(stream);
    }
    for (auto& event : exec_done_events_) {
      cudaEventDestroy(event);
    }
    if (input_ready_event_ != nullptr) {
      cudaEventDestroy(input_ready_event_);
    }
    for (auto* m : modules_) {
      delete m;
    }
  }

  void SetInputs(const torch::Tensor& hidden_states,
                 const torch::Tensor& router_mask,
                 const torch::Tensor& router_weight);

  std::uint64_t SetInputsWithInvocation(const torch::Tensor& hidden_states,
                                        const torch::Tensor& router_mask,
                                        const torch::Tensor& router_weight);
  void SetOverlapComputeTimingEnabled(bool enabled);
  std::vector<ExpertComputeSample> DrainComputeSamples();

  void EnqueueExpert(int layer_idx, int expert_idx, int gpu_id = -1,
                     bool remote = false,
                     int phase = static_cast<int>(ExpertPhase::MIXED));
  void NotifyFetchStart();

  void RegisterExpert(int layer_idx, int expert_idx,
                      const std::vector<std::uint32_t>& tensor_ids,
                      std::string jit_path);
  void ClearExpertCacheCounts();
  // Read-only observability accessors; neither alters routing/dispatch.
  std::int64_t GetCacheOccupancyBytes();
  std::int64_t GetCacheOccupancyBytes(int device_id);
  double GetCacheHitRate() const;
  // Per-layer GPU residency snapshot: one byte per expert (1 == on GPU, 0 ==
  // off GPU). Returns an empty vector when residency is unknown (layer_idx < 0,
  // no experts registered, or any expert lacks a node at that layer).
  std::vector<std::uint8_t> ResidentOnGpu(int layer_idx) const;

  ResizeToken BeginMemoryResize(int device_id, int timeout_ms);
  void EndMemoryResize(const ResizeToken& token);
  void SetCacheLimit(int device_id, std::int64_t target_bytes,
                     const ResizeToken& token);
  void SetExpectedQueue(int expected_pending = 0) {
    pending_.store(expected_pending);
  }

  void SetScales(const std::map<std::string, torch::Tensor>& scales);
  // Allows/denies the sm120 + RTX PRO 6000 FP8 expert kernels on every exec
  // module (decided once at setup by moe_infinity/kernel/fp8_experts.py; the
  // modules still check the hardware gate themselves).
  void SetFp8Sm120Pro6000Experts(bool allowed);
  // Gated activation of the routed experts (kind = MoEActivationKind). Must be
  // called before the first dispatch; exec workers read it without locking.
  void SetExpertActivation(int kind, float alpha, float limit);
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

  // active_flags_host: optional pinned bool [num_experts] written by the
  // routing kernel on the current stream (moe_infinity._router). When given,
  // the mask reduction + device-to-host copy are skipped.
  void DispatchExperts(
      int layer_idx,
      c10::optional<torch::Tensor> active_flags_host = c10::nullopt);
  std::vector<int> TakeLastActiveExperts();
  std::vector<int> TakeLastRoutedExperts();
  // GPT-OSS offloaded experts: allow the RTX PRO 6000 MXFP4 kernels (set by
  // Python from hw_profile.supports(RTX_PRO_6000); each module re-checks its
  // device). Off: the previous de-quantise + matmul path.
  void SetGptOssMxfp4Pro6000(bool enabled) { gptoss_mxfp4_pro6000_ = enabled; }
  void SetExpertDropPolicy(bool enabled, int min_k, double mass_budget,
                           double flatness_floor, double head_budget = 0.0,
                           double adaptive_slope = 0.0, int adaptive_miss0 = 0);
  // Expert-drop trace tap (enabled by MOE_EXPERT_DROP_TRACE). Writes the
  // buffered pre-drop routing/residency records to `path` (or the env path
  // when empty). No-op when tracing was never enabled.
  void DumpExpertDropTrace(const std::string& path = std::string());
  std::map<std::string, std::int64_t> GetFusedDropStats() const;
  std::map<int, std::int64_t> GetFusedDropStatsByLayer() const;
  std::map<std::string, std::int64_t> GetRoutingStats() const;
  // Grouped decode path (#306): resident experts of a layer run as one group
  // (two kernel launches) when the batch has <= kGroupedMoePro6000MaxTokens
  // tokens. RTX PRO 6000 only (Gate::kRtxPro6000); other devices keep
  // per-expert execution. MOE_GROUPED_DECODE=0 disables it at construction.
  // top_k bounds the routed experts per token and sizes the activation
  // workspace (0 = num_experts). activation_kind follows MoEActivationKind
  // (0 = SiLU, 1 = clamped SiLU, 2 = SwiGLU-OAI, < 0 = unknown); the grouped
  // kernels implement SiLU only, so any other kind keeps per-expert execution.
  void SetGroupedDecode(bool enabled, int top_k, int activation_kind = 0);
  // Grouped prefill path (#307): batches of >= min_tokens tokens (and above the
  // decode limit) run their GPU-resident experts through grouped_moe_prefill
  // (routing + grouped gate/up + grouped down) instead of one CUTLASS FFN per
  // expert. Same SiLU-only / BF16 / RTX PRO 6000 restrictions as the decode
  // path; opt-in (default off). MOE_GROUPED_PREFILL=1 and
  // MOE_GROUPED_PREFILL_MIN_TOKENS=<n> set it at construction. min_tokens <= 0
  // restores the default.
  void SetGroupedPrefill(bool enabled, std::int64_t min_tokens = 0);
  void SetDispatchFaultForTest(const std::string& stage);
  void FailDispatchForTest(std::uint64_t generation,
                           const std::string& message);

 private:
  void Enqueue(CallArgs& args);
  // Blocks while the target device's resize gate is closed, then counts one
  // in-flight item for that device. Pairs 1:1 with the exec-worker decrement.
  void AdmitAndCount(int device_id);
  std::vector<CallResult> Wait();
  void Start() { start_ = true; }

  void GPUFetchFunc(int gpu_id);
  void GPUExecFunc(int gpu_id, int thread_idx);
  bool ApplyPrecisionTarget(const ExpertNodePtr& expert_node, int gpu_id);

  // void GPUThreadFunc(int gpu_id);

  bool OutputFunc(ExecArgs args, torch::Tensor output, torch::Tensor token_mask,
                  int gpu_id, cudaStream_t exec_stream) noexcept;

  ExpertNodePtr FindExpertEvict(int gpu_id);

  void RouteFunc() noexcept;
  // RouteFunc-thread only. Runs the fused drop selector on the staged pinned
  // buffers, synchronously writes corrected weights back into router_weight_
  // (Task 1 ordering note), and shrinks dispatch_experts to the post-drop
  // survivors. Any failure leaves dispatch_experts at the PRE-drop set.
  void ApplyFusedExpertDrop(const RouteArgs& args,
                            std::vector<int>& dispatch_experts) noexcept;
  static void CUDART_CB RouteReadyCallback(void* opaque);
  void FailDispatch(const std::uint64_t failing_generation,
                    std::exception_ptr error,
                    const std::vector<FailureContext>& contexts = {}) noexcept;
  void CompleteOne(std::uint64_t generation) noexcept;
  void NotifyPendingWaiters() noexcept;
  void NotifyCacheWaiters(int gpu_id) noexcept;

  static void CUDART_CB CompletionWaitsConsumedCallback(void* opaque);
  void CompletionRetirementFunc();
  void QueueUnwaitedEventsForQuery(
      std::vector<CompletionEventRecord> records) noexcept;

  static void CUDART_CB ExpertReadyToRetireCallback(void* opaque);
  static void CUDART_CB GroupReadyToRetireCallback(void* opaque);
  void RetirementFunc();

  // Grouped decode path. GroupedEligible is evaluated on the cache-hit
  // enqueue path; eligible experts are parked per device until
  // FlushGroupedExperts (NotifyFetchStart / Wait*) hands each device's group
  // to an exec worker as one item, which GroupedExec runs.
  bool GroupedEligible(const ExpertNodePtr& expert_node, int gpu_id,
                       bool remote);
  void FlushGroupedExperts();
  void GroupedExec(ExecArgs& args, int gpu_id, cudaStream_t stream);

  class DispatchSubmissionGuard {
   public:
    DispatchSubmissionGuard(ExpertDispatcher* dispatcher,
                            const std::uint64_t generation)
        : dispatcher_(dispatcher), generation_(generation) {}
    ~DispatchSubmissionGuard() {
      if (armed_) {
        dispatcher_->FailDispatch(
            generation_, error_ ? error_
                                : std::make_exception_ptr(std::runtime_error(
                                      "dispatch submission failed")));
      }
    }
    void Capture(std::exception_ptr error) noexcept {
      error_ = std::move(error);
    }
    void Release() noexcept { armed_ = false; }

   private:
    ExpertDispatcher* dispatcher_;
    const std::uint64_t generation_;
    std::exception_ptr error_;
    bool armed_ = true;
  };

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

  // Per-device admission gate and drain counters for memory resize. Enqueue()
  // blocks on resize_cv_ while a device's gate is closed so no new fetch/exec
  // work is dropped during a maintenance window.
  std::vector<std::uint8_t> admissions_paused_;
  std::vector<std::int64_t> pending_by_device_;
  std::vector<std::int64_t> active_fetch_workers_;
  std::vector<std::int64_t> active_exec_workers_;
  std::vector<std::int64_t> cache_limit_bytes_;
  std::mutex resize_mutex_;
  std::condition_variable resize_cv_;
  std::uint64_t next_resize_token_id_ = 1;

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

  void SyncExecStreamsWithCurrent();

  std::vector<cudaStream_t> exec_streams_;
  std::vector<cudaEvent_t> exec_done_events_;
  cudaEvent_t input_ready_event_ = nullptr;
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
  bool gptoss_mxfp4_pro6000_ = false;
  std::vector<std::vector<std::vector<torch::Tensor>>> fp8_scales_;
  // FP32 copies of fp8_scales_ cached on every CUDA device at SetScales time
  // (~70-90 MB per GPU for a 290 GB FP8 model) so exec workers never pay a
  // blocking pageable H2D upload per expert execution. Indexed
  // [gpu_id][layer][expert][weight].
  std::vector<std::vector<std::vector<std::vector<torch::Tensor>>>>
      fp8_scales_gpu_;
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

  std::mutex grouped_mutex_;
  std::vector<std::vector<ExecArgs>> grouped_pending_;  // per device
  std::atomic<bool> grouped_enabled_{true};
  // Per device: RTX PRO 6000 gate of the grouped kernels, evaluated once at
  // construction. Other devices keep per-expert execution.
  std::vector<bool> grouped_pro6000_ok_;
  // Grouped prefill (#307): default off, see SetGroupedPrefill. Per device
  // RTX PRO 6000 gate of grouped_moe_prefill.
  std::atomic<bool> grouped_prefill_enabled_{false};
  std::atomic<std::int64_t> grouped_prefill_min_tokens_{2048};
  std::vector<bool> grouped_prefill_ok_;
  std::atomic<std::int64_t> grouped_prefill_batches_{0};
  std::atomic<int> grouped_top_k_{0};
  std::atomic<int> grouped_activation_kind_{0};
  std::atomic<std::int64_t> grouped_batches_{0};
  std::atomic<std::int64_t> grouped_experts_{0};
  // Per-device [1] int64 zero index: the single-token accumulate uses an
  // atomic index_add_ so it can run next to the grouped kernel's atomics.
  std::unordered_map<int, torch::Tensor> zero_index_;

  std::atomic<bool> overlap_timing_enabled_{false};
  std::atomic<std::uint64_t> invocation_counter_{0};
  std::uint64_t current_invocation_id_{0};
  std::mutex compute_samples_mutex_;
  std::vector<ExpertComputeSample> compute_samples_;
  ThreadSafeQueue<RouteArgs> route_queue_;
  std::atomic<bool> route_pending_{false};
  std::atomic<std::uint64_t> dispatch_generation_{0};
  std::atomic<std::int64_t> pending_route_callbacks_{0};
  std::mutex route_callback_mutex_;
  std::condition_variable route_callback_cv_;
  mutable std::mutex route_state_mutex_;
  std::exception_ptr route_error_;
  std::atomic<std::uint64_t> current_generation_{0};
  std::atomic<std::uint64_t> failed_generation_{0};
  std::atomic<int> dispatch_fault_for_test_{0};
  std::vector<int> last_active_experts_;
  std::vector<int> last_routed_experts_;
  std::atomic<bool> expert_drop_enabled_{false};
  moe::ExpertDropParams expert_drop_params_;
  moe::ExpertDropScratch expert_drop_scratch_;
  std::vector<std::uint8_t> expert_drop_resident_;
  std::atomic<std::int64_t> drop_tokens_seen_{0};
  std::atomic<std::int64_t> drop_tokens_changed_{0};
  std::atomic<std::int64_t> drop_experts_dropped_{0};
  std::atomic<std::int64_t> drop_flatness_bypasses_{0};
  std::atomic<std::int64_t> drop_residency_unknown_{0};
  std::atomic<std::int64_t> drop_failures_{0};
  std::map<int, std::int64_t> drop_by_layer_;
  // Expert-drop trace tap state (enabled by MOE_EXPERT_DROP_TRACE). The
  // record callback runs on the off-critical-path route worker and only
  // appends to an in-memory buffer, so it never issues file I/O or CUDA
  // calls on the dispatch path. The buffer is serialized to disk by
  // DumpExpertDropTrace (explicitly from Python, or best-effort at teardown).
  std::atomic<bool> expert_drop_trace_enabled_{false};
  std::string expert_drop_trace_path_;
  mutable std::mutex expert_drop_trace_mutex_;
  std::string expert_drop_trace_buffer_;
  std::int64_t expert_drop_trace_seq_{0};
  bool expert_drop_trace_dumped_{false};
  void RecordExpertDropTrace(const RouteArgs& args);
  void FlushExpertDropTrace() noexcept;
  std::vector<CompletionEventRecord> completion_events_;
  std::vector<CompletionEventRecord> destructor_fallback_events_;
  ThreadSafeQueue<std::vector<CompletionRetireItem>>
      completion_retirement_queue_;
  std::atomic<std::int64_t> pending_completion_retirement_callbacks_{0};
  std::mutex completion_retirement_callback_mutex_;
  std::condition_variable completion_retirement_callback_cv_;

  ThreadSafeQueue<ExpertRetireArgs> retirement_queue_;
  std::atomic<std::int64_t> pending_retirement_callbacks_{0};
  std::mutex retirement_callback_mutex_;
  std::condition_variable retirement_callback_cv_;

  cudaEvent_t AcquireCompletionEvent(int device);
  void ReleaseCompletionEvent(cudaEvent_t event, int device) noexcept;
  std::mutex completion_event_pool_mutex_;
  std::unordered_map<int, std::vector<cudaEvent_t>> completion_event_pool_;

  std::atomic<std::int64_t> route_batches_{0};
  std::atomic<std::int64_t> route_failures_{0};
  std::atomic<std::int64_t> last_active_experts_count_{0};
  std::atomic<std::int64_t> last_route_handoff_us_{0};
  std::atomic<std::int64_t> completion_events_retired_{0};
  std::atomic<std::int64_t> completion_events_outstanding_{0};
  std::atomic<std::int64_t> stale_failures_quarantined_{0};
};

struct GroupedExecBatch {
  std::vector<ExpertDispatcher::ExecArgs> members;
};

#define SET_TENSORS_AND_MODULE_FROM_BLOB(cls, module, node, device, \
                                         jit_module)                \
  do {                                                              \
    reinterpret_cast<cls*>(module)->SetTensorsFromBlob(             \
        node->device_memory_ptr, node->tensor_ids, device);         \
    reinterpret_cast<cls*>(module)->SetModuleFromBlob(jit_module);  \
  } while (0)
