// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "base/noncopyable.h"
#include "base/thread.h"
#include "expert_module.h"

enum MUTEX_TYPE {
  INPUT_MUTEX = 0,
  OUTPUT_MUTEX = 1,
  EXEC_MUTEX = 2,
  PENDING_MUTEX = 3
};

class ExpertDispatcher : public base::noncopyable {
 public:
  typedef struct {
    int layer_idx = -1;
    int expert_idx = -1;
    int gpu_id = -1;
    bool remote = false;
  } CallArgs;
  typedef struct {
    torch::Tensor hidden_states =
        torch::empty({0});  // shallow copy, real tensor in python code
    ExpertNodePtr expert_node = nullptr;
    int out_gpu_id = -1;
    torch::ScalarType out_dtype = torch::kFloat32;
    bool evict = false;
    bool hit = false;
  } ExecArgs;
  typedef std::tuple<torch::Tensor, int, int, int> CallResult;

 public:
  explicit ExpertDispatcher(int num_experts, int num_layers, int dtype,
                            int expert_type, int num_threads = 1);
  ~ExpertDispatcher() {
    main_thread_stop_flag_.store(true);
    for (auto& thread : threads_) {
      thread->join();
    }

    // for (auto& stream : fetch_streams_) {
    //   cudaStreamDestroy(stream);
    // }
    for (auto& stream : exec_streams_) {
      cudaStreamDestroy(stream);
    }
    // for (auto& stream : out_streams_) {
    //   cudaStreamDestroy(stream);
    // }
  }

  void SetInputs(const torch::Tensor& hidden_states,
                 const torch::Tensor& router_mask,
                 const torch::Tensor& router_weight);

  void EnqueueExpert(int layer_idx, int expert_idx, int gpu_id = -1,
                     bool remote = false);
  void NofityFetchStart();

  void RegisterExpert(int layer_idx, int expert_idx,
                      const std::vector<std::uint32_t>& tensor_ids,
                      std::string jit_path);
  void ClearExpertCacheCounts();
  void SetExpectedQueue(int expected_pending = 0) {
    pending_.store(expected_pending);
  }

  std::vector<CallResult> WaitExpert() { return Wait(); }
  torch::Tensor WaitHiddenStates();

 private:
  void Enqueue(CallArgs& args);
  std::vector<CallResult> Wait();
  void Start() { start_ = true; }

  void GPUFetchFunc(int gpu_id);
  void GPUExecFunc(int gpu_id);

  // void GPUThreadFunc(int gpu_id);

  void OutputFunc(ExecArgs args, torch::Tensor output, torch::Tensor token_mask,
                  int gpu_id);

 private:
  std::vector<std::unique_ptr<base::Thread>> threads_;
  std::mutex mutex_;
  std::vector<std::deque<CallArgs>> input_queue_;
  std::vector<std::deque<ExecArgs>> exec_queue_;
  std::vector<CallResult> output_queue_;
  std::vector<std::vector<ExpertNodePtr>> experts_;
  std::atomic<size_t> num_enqueued_;
  bool start_;
  int expert_type_;
  int dtype_;
  int num_experts_;
  std::atomic<bool> main_thread_stop_flag_;

  std::atomic<size_t> pending_;

  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;

  std::vector<std::mutex> input_mutex_;
  std::vector<std::mutex> exec_mutex_;
  std::vector<std::condition_variable> input_cv_;
  std::vector<std::condition_variable> exec_cv_;

  std::mutex output_mutex_;
  // std::mutex exec_mutex_;
  std::mutex gpu_overload_mutex_;

  std::vector<cudaStream_t> exec_streams_;

  std::vector<bool> gpu_overload_;

  torch::Tensor hidden_states_;
  torch::Tensor final_hidden_states_;
  torch::Tensor router_mask_;
  torch::Tensor router_weight_;

  std::vector<int64_t> cache_sizes_;

  int cache_capacity_ = 0;

  std::vector<MoEMLP*> modules_;
};

#define SET_TENSORS_AND_MODULE_FROM_BLOB(cls, module, node, device, \
                                         jit_module)                \
  do {                                                              \
    reinterpret_cast<cls*>(module)->SetTensorsFromBlob(             \
        node->device_memory_ptr, node->tensor_ids, device);         \
    reinterpret_cast<cls*>(module)->SetModuleFromBlob(jit_module);  \
  } while (0)
