// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "model/model_topology.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <future>

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type, int num_threads)
    : pending_(0),
      num_enqueued_(0),
      start_(false),
      expert_type_(expert_type),
      input_mutex_(kNumDevices),
      input_cv_(kNumDevices),
      exec_mutex_(kNumDevices),
      exec_cv_(kNumDevices),
      input_queue_(kNumDevices),
      exec_queue_(kNumDevices),
      gpu_overload_(kNumDevices, false),
      modules_(kNumDevices, nullptr) {
  main_thread_stop_flag_.store(false);

  // module_ = new MoEMLP(dtype, expert_type);

  for (int i = 0; i < kNumDevices; ++i) {
    auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
    std::string thread_name = "GPUFetchFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);

    modules_[i] = new MoEMLP(dtype, expert_type);
  }

  for (int i = 0; i < kNumDevices * num_threads; ++i) {
    cudaSetDevice(i % kNumDevices);
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.emplace_back(exec_stream);
    // cudaDeviceSynchronize();

    auto thread_func =
        std::bind(&ExpertDispatcher::GPUExecFunc, this, i % kNumDevices);
    std::string thread_name = "GPUExecFunc" + std::to_string(i % kNumDevices);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());
  }

  at::InferenceMode infer_guard(0);

  for (int i = 0; i < num_experts; ++i) {
    experts_.emplace_back();
    for (int j = 0; j < num_layers; ++j) {
      experts_[i].emplace_back();
      experts_[i][j] = std::make_shared<ExpertNode>();
      experts_[i][j]->expert_type = expert_type;
      int expert_type = expert_type_;
      switch (expert_type) {
        case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
          experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
          break;
        case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
          experts_[i][j]->module =
              new SwitchTransformersDenseGatedActDense(dtype);
          break;
        case NLLB_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new NllbMoeDenseActDense(dtype);
          break;
        case FSGPT_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new FSGPTMoEDenseActDense(dtype);
          break;
        case MIXTRAL_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new MixtralMoEDenseActDense(dtype);
          break;
        case DEEPSEEK_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new DeepSeekMoEDenseActDense(dtype);
          break;
        default:
          DLOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                     expert_type);
      }
      experts_[i][j]->module->eval();
      experts_[i][j]->layer_idx = j;
      experts_[i][j]->expert_idx = i;
    }
  }
}

void ExpertDispatcher::EnqueueExpert(int layer_idx, int expert_idx, int gpu_id,
                                     bool remote) {
  ExpertDispatcher::CallArgs args;
  args.layer_idx = layer_idx;
  args.expert_idx = expert_idx;
  args.gpu_id = gpu_id;
  args.remote = remote;
  Enqueue(args);
}

void ExpertDispatcher::Enqueue(CallArgs& args) {
  // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
  int layer_idx = args.layer_idx;
  int expert_idx = args.expert_idx;
  auto expert_node = experts_[expert_idx][layer_idx];

  if (!expert_node->node->mutex.try_lock()) {
    // NOTE: try lock must success, if there is no prefetching
    DLOG_FATAL("ExpertDispatcher::Enqueue: mutex try_lock failed (expert_idx ",
               expert_idx, " layer_idx ", layer_idx, "node ",
               expert_node->node->str(), ")");
  }
  expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

  if (expert_node->node->device.is_cuda()) {
    args.gpu_id = expert_node->node->device.index();

    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();

    ExecArgs exec_args;
    // exec_args.hidden_states = std::move(input);
    exec_args.expert_node = expert_node;
    expert_node->SetTensorsFromBlob(expert_node->node->device);
    exec_args.out_gpu_id = original_device.index();
    exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
    exec_args.evict = false;
    exec_args.hit = true;

    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    std::unique_lock<std::mutex> lock(exec_mutex_[args.gpu_id]);
    exec_queue_[args.gpu_id].push_back(std::move(exec_args));
  } else {
    std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
    input_queue_[args.gpu_id].push_back(std::move(args));
  }
  // input_cv_[args.gpu_id].notify_all();
  exec_cv_[args.gpu_id].notify_all();
  // input_queue_.push_back(std::move(args));
  num_enqueued_.fetch_add(1);

  // auto& a = input_queue_.back();
  // if (expert_node->node->device.is_cuda()) {
  //   a.gpu_id = expert_node->node->device.index();
  // }
  // DLOG_TRACE("ExpertDispatcher::Enqueue: num_enqueued_ ",
  // num_enqueued_.load(),
  //            "input_queue_ ", input_queue_.size(), "gpu_id ", a.gpu_id,
  //            "layer_idx ", a.layer_idx, "expert_idx ", a.expert_idx, "remote
  //            ", a.remote);
  // lock.unlock();
  // cvs_[MUTEX_TYPE::INPUT_MUTEX].notify_all();
}

void ExpertDispatcher::RegisterExpert(
    int layer_idx, int expert_idx, const std::vector<std::uint32_t>& tensor_ids,
    std::string jit_path) {
  NodePtr cached_node = nullptr;
  for (auto tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (cached_node == nullptr) {
      cached_node = node;
      experts_[expert_idx][layer_idx]->node = node;
      experts_[expert_idx][layer_idx]->jit_module =
          new torch::jit::script::Module(torch::jit::load(jit_path));
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
  }
}

void ExpertDispatcher::NofityFetchStart() {
  for (int i = 0; i < kNumDevices; ++i) {
    // std::unique_lock<std::mutex> lock(input_mutex_[i]);
    input_cv_[i].notify_all();
  }
}

void ExpertDispatcher::ClearExpertCacheCounts() {
  for (auto& expert : experts_) {
    for (auto& expert_node : expert) {
      if (expert_node->node == nullptr) {
        continue;
      }
      expert_node->node->incache_visit_count = 0;
    }
  }
}

// void ExpertDispatcher::GPUThreadFunc(int gpu_id) {
//   while (!main_thread_stop_flag_.load()) {
//   }
// }

void ExpertDispatcher::GPUFetchFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  while (!main_thread_stop_flag_.load()) {
    // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
    // if (cache_ == nullptr) {
    //   auto cache_limit =
    //   kDeviceMemoryPool->GetSparseCacheLimit(torch::Device(torch::kCUDA,
    //   gpu_id));
    //   // get any one expert size
    //   auto num_layers = experts_[0].size();
    //   auto num_experts = experts_.size();
    //   auto expert_node = experts_[num_layers-1][num_experts-1];

    //   int cache_capacity = cache_limit / expert_node->node->byte_size;
    //   cache_capacity_ = cache_capacity;
    // }
    std::unique_lock<std::mutex> lock(input_mutex_[gpu_id]);
    input_cv_[gpu_id].wait(lock, [&] { return !input_queue_[gpu_id].empty(); });

    CallArgs args = std::move(input_queue_[gpu_id].front());
    input_queue_[gpu_id].pop_front();

    lock.unlock();

    auto device = CUDA_DEVICE(gpu_id);
    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();
    int layer_idx = args.layer_idx;
    int expert_idx = args.expert_idx;

    auto expert_node = experts_[expert_idx][layer_idx];
    bool cache_hit = expert_node->node->device.is_cuda();

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: gpu_id " << gpu_id
    //           << " layer_idx " << layer_idx << " expert_idx " << expert_idx
    //           << " cache_hit " << cache_hit << " node "
    //           << expert_node->node->device.str() << std::endl;

    if (!expert_node->node->device.is_cuda() &&
        cache_sizes_[gpu_id] < expert_node->node->byte_size) {
      // find the expert in gpu and min incache_visit_count
      NodePtr evict_node = nullptr;
      auto num_layers = experts_[0].size();
      auto num_experts = experts_.size();
      int min_visit_count = INT_MAX;
      for (size_t i = 0; i < num_experts; ++i) {
        for (size_t j = 0; j < num_layers; ++j) {
          auto node = experts_[i][j]->node;
          if (node == nullptr) {
            // std::cerr << "ExpertDispatcher::GPUFetchFunc: node is nullptr"
            //           << " layer_idx " << j << " expert_idx " << i <<
            //           std::endl;
            continue;
          }
          if (node->device.is_cuda() &&
              node->incache_visit_count < min_visit_count &&
              node->mutex.try_lock()) {
            evict_node = node;
            min_visit_count = node->incache_visit_count;
            node->mutex.unlock();
            // std::cerr << "ExpertDispatcher::GPUFetchFunc: evict node "
            //           << evict_node->device.str() << " incache_visit_count "
            //           << min_visit_count << std::endl;
          }
        }
      }
      assert(evict_node != nullptr);
      evict_node->SetDevice(evict_node->default_host);
      cache_sizes_[gpu_id] += evict_node->byte_size;
    }

    bool success = true;

    expert_node->node->SetDevice(device, true, stream);
    expert_node->node->incache_visit_count += 1;
    expert_node->SetTensorsFromBlob(device);
    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);
    cache_sizes_[gpu_id] -= expert_node->node->byte_size;
    // std::cerr << "ExpertDispatcher::GPUFetchFunc: move to device gpu_id "
    //           << gpu_id << " layer_idx " << layer_idx << " expert_idx "
    //           << expert_idx << " node "
    //           << expert_node->node->device.str() << std::endl;

    // int expert_type = expert_type_;
    // torch::Tensor input;
    // auto token_indices =
    //     router_mask_.index({"...", expert_idx}).to(torch::kBool);
    // switch (expert_type) {
    //   case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
    //   case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
    //   case NLLB_MOE_DENSE_ACT_DENSE:
    //   case FSGPT_MOE_DENSE_ACT_DENSE:
    //   case MIXTRAL_MOE_DENSE_ACT_DENSE:
    //   case DEEPSEEK_MOE_DENSE_ACT_DENSE:
    //     input =
    //         hidden_states_.index({token_indices}).to(expert_node->node->device);
    //     break;
    //   default:
    //     DLOG_FATAL("ExpertDispatcher::expert_type: unknown expert type ",
    //                expert_type);
    // }

    // DLOG_TRACE("ExpertDispatcher::GPUFetchFunc gpu_id ", gpu_id, "layer_idx
    // ",
    //            layer_idx, "expert_idx ", expert_idx, "input ",
    //            input.device().str(), "node ",
    //            expert_node->node->device.str());
    {
      ExecArgs exec_args;
      // exec_args.hidden_states = std::move(input);
      exec_args.expert_node = expert_node;
      exec_args.out_gpu_id = original_device.index();
      exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
      exec_args.evict = !success;
      exec_args.hit = cache_hit;
      std::lock_guard<std::mutex> lock(exec_mutex_[gpu_id]);
      exec_queue_[gpu_id].emplace_back(std::move(exec_args));
    }
    exec_cv_[gpu_id].notify_all();
  }

  cudaStreamDestroy(stream);
}

void ExpertDispatcher::GPUExecFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  while (!main_thread_stop_flag_.load()) {
    std::unique_lock<std::mutex> lock(exec_mutex_[gpu_id]);
    exec_cv_[gpu_id].wait(lock, [&] { return !exec_queue_[gpu_id].empty(); });

    ExecArgs args = std::move(exec_queue_[gpu_id].front());
    exec_queue_[gpu_id].pop_front();

    lock.unlock();

    if (args.expert_node == nullptr) {
      continue;
    }

    auto expert_idx = args.expert_node->expert_idx;
    auto device = CUDA_DEVICE(gpu_id);
    auto token_mask = router_mask_.index({"...", expert_idx}).to(torch::kBool);
    auto input = hidden_states_.index({token_mask}).to(device);
    // args.hidden_states = std::move(input);
    // assert(args.hidden_states.sum().to(torch::kCPU).item<float>() != 0);
    // at::InferenceMode infer_guard(true);

    // // prepare jit input vector
    // std::vector<torch::jit::IValue> jit_inputs;
    // jit_inputs.push_back(input);

    // cudaDeviceSynchronize();

    modules_[gpu_id]->SetTensorsFromIds(args.expert_node->node->tensor_ids);

    // random int [0,8)
    // int rnd = std::rand() % kNumDevices;
    c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromExternal(stream, gpu_id);
    c10::cuda::CUDAStreamGuard guard(torch_stream);
    // auto start = TIME_NOW;
    // c10::cuda::CUDAStreamGuard guard(stream);

    // auto* expert_module = args.expert_node->module;
    // int expert_type = expert_type_;
    // cudaStreamSynchronize(stream);  // make sure the input is ready

    auto output = modules_[gpu_id]->forward(input, stream);
    // try {
    //   switch (expert_type) {
    //     case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
    //       output = reinterpret_cast<SwitchTransformersDenseActDense*>(
    //                    expert_module)
    //                    ->forward(input);
    //       break;
    //     case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
    //       output =
    //       reinterpret_cast<SwitchTransformersDenseGatedActDense*>(
    //                    expert_module)
    //                    ->forward(input);
    //       break;
    //     case NLLB_MOE_DENSE_ACT_DENSE:
    //       output = reinterpret_cast<NllbMoeDenseActDense*>(expert_module)
    //                    ->forward(input);
    //       break;
    //     case FSGPT_MOE_DENSE_ACT_DENSE:
    //       output =
    //       reinterpret_cast<FSGPTMoEDenseActDense*>(expert_module)
    //                    ->forward(input);
    //       break;
    //     case MIXTRAL_MOE_DENSE_ACT_DENSE:
    //       output =
    //       reinterpret_cast<MixtralMoEDenseActDense*>(expert_module)
    //                    ->forward(input);
    //       break;
    //     case DEEPSEEK_MOE_DENSE_ACT_DENSE:
    //       output =
    //       reinterpret_cast<DeepSeekMoEDenseActDense*>(expert_module)
    //                    ->forward(input, stream.stream());
    //       // output =
    //       //
    //       args.expert_node->jit_module->forward(jit_inputs).toTensor();
    //       break;
    //     default:
    //       DLOG_FATAL("ExpertDispatcher::GPUExecFunc: unknown expert
    //       type",
    //                  expert_type);
    //   }

    // } catch (const std::exception& e) {
    //   std::stringstream ss;
    //   ss << "DenseActDense tensor_ids: [";
    //   for (auto& id : args.expert_node->node->tensor_ids) {
    //     ss << id << " ";
    //   }
    //   ss << "]";
    //   DLOG_FATAL("ExpertDispatcher::GPUExecFunc", ss.str(), "expert_type",
    //              expert_type, e.what());
    // }

    // stream.synchronize();
    // auto end = TIME_NOW;
    // DLOG_INFO("ExpertDispatcher::GPUExecFunc: forward time ",
    //                  std::chrono::duration_cast<MCIROSECONDS>(end -
    //                  start).count(), "us");

    // (void)std::async(std::launch::async, &ExpertDispatcher::OutputFunc, this,
    //                  std::move(args), std::move(output), gpu_id);
    // DLOG_DEBUG("ExpertDispatcher::GPUExecFunc: output ",
    // output.sizes().vec(),
    //            "(", output.device().str(), ")");
    // output cannot be all zeros using torch.all
    // assert(torch::all(output == 0).item<bool>() == false);
    OutputFunc(args, output, gpu_id);
  }

  cudaStreamDestroy(stream);
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  int gpu_id) {
  auto output_device =
      (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
  torch::Tensor output_tensor = output.to(output_device).to(torch::kFloat32);

  DLOG_DEBUG("ExpertDispatcher::OutputFunc: output_tensor ",
             output_tensor.sizes().vec(), "(", output_tensor.device().str(),
             ")");

  if (args.evict) {
    args.expert_node->node->SetDevice(args.expert_node->node->default_host,
                                      true, nullptr);
    {
      std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
      gpu_overload_[gpu_id] = false;
    }
  }

  args.expert_node->node->mutex.unlock();
  int64_t expert_idx = args.expert_node->expert_idx;
  int64_t layer_idx = args.expert_node->layer_idx;

  auto token_mask = router_mask_.index({"...", expert_idx}).to(torch::kBool);
  auto token_indices = torch::nonzero(token_mask).squeeze(1);
  auto weights = router_weight_.index({token_mask, expert_idx}).unsqueeze(1);
  auto weighted_output = output_tensor * weights;
  final_hidden_states_.index_add_(0, token_indices, weighted_output);

  // {
  //   std::lock_guard<std::mutex> lock(output_mutex_);
  //   output_queue_.emplace_back(std::move(output_tensor),
  //                              args.expert_node->layer_idx,
  //                              args.expert_node->expert_idx, args.hit);
  //   DLOG_TRACE("ExpertDispatcher::OutputFunc: output_queue_",
  //              output_queue_.size(), "output",
  //              std::get<0>(output_queue_.back()).device().str(), "evict",
  //              args.evict, "(", args.expert_node->layer_idx,
  //              args.expert_node->expert_idx, gpu_id, args.hit, ")");
  // }

  // stream.synchronize();
  pending_.fetch_sub(1);
  if (pending_.load() == 0) {
    pending_cv_.notify_all();
  }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait() {
  // int wait_count = 0;

  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] { return pending_.load() == 0; });

  num_enqueued_.store(0);
  std::vector<CallResult> output_queue;
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue.swap(output_queue_);
  }

  return output_queue;
}

torch::Tensor ExpertDispatcher::WaitHiddenStates() {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] { return pending_.load() == 0; });
  num_enqueued_.store(0);
  return final_hidden_states_;
}

void ExpertDispatcher::SetInputs(const torch::Tensor& hidden_states,
                                 const torch::Tensor& router_mask,
                                 const torch::Tensor& router_weight) {
  int device = at::cuda::current_device();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(CUDA_DEVICE(device));
  hidden_states_ = hidden_states.clone();
  router_mask_ = router_mask.clone();
  final_hidden_states_ = torch::zeros_like(hidden_states_, options);
  router_weight_ = router_weight.clone();
}
