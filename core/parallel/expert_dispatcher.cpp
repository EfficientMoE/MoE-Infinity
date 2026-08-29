// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "memory/event_pool.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "model/model_topology.h"
#include "model/moe.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <future>
#include <optional>
#include <regex>
#include <sstream>

namespace {

std::optional<ExpertFormat> ParseExpertFormat(const std::string& value) {
  if (value == "bf16") return ExpertFormat::BF16;
  if (value == "fp8_e4m3_block128") return ExpertFormat::FP8_E4M3_BLOCK128;
  if (value == "marlin_int4_group128")
    return ExpertFormat::MARLIN_INT4_GROUP128;
  if (value == "gpt_oss_mxfp4") return ExpertFormat::GPT_OSS_MXFP4;
  if (value == "glm_fp8_block128") return ExpertFormat::GLM_FP8_BLOCK128;
  if (value == "deepseek_v4_fp4") return ExpertFormat::DEEPSEEK_V4_FP4;
  if (value == "gptq") return ExpertFormat::GPTQ;
  if (value == "awq") return ExpertFormat::AWQ;
  return std::nullopt;
}

std::optional<std::uint8_t> ParseExecutionKind(const std::string& value) {
  if (value == "bf16_gemm") return 0;
  if (value == "fp8_dequant_bf16_gemm") return 1;
  if (value == "marlin_w4a16") return 2;
  if (value == "gpt_oss_mxfp4") return 3;
  if (value == "deepseek_v4_fp4") return 4;
  if (value == "legacy_quantized") return 5;
  return std::nullopt;
}

std::string FormatName(ExpertFormat format) {
  switch (format) {
    case ExpertFormat::BF16:
      return "bf16";
    case ExpertFormat::FP8_E4M3_BLOCK128:
      return "fp8_e4m3_block128";
    case ExpertFormat::MARLIN_INT4_GROUP128:
      return "marlin_int4_group128";
    case ExpertFormat::GPT_OSS_MXFP4:
      return "gpt_oss_mxfp4";
    case ExpertFormat::GLM_FP8_BLOCK128:
      return "glm_fp8_block128";
    case ExpertFormat::DEEPSEEK_V4_FP4:
      return "deepseek_v4_fp4";
    case ExpertFormat::GPTQ:
      return "gptq";
    case ExpertFormat::AWQ:
      return "awq";
  }
  return "unknown";
}

std::uint64_t LogicalExpertKey(int layer_idx, int expert_idx) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(layer_idx))
          << 32) |
         static_cast<std::uint32_t>(expert_idx);
}

bool GenericFormatExecutionPair(ExpertFormat format, std::uint8_t execution) {
  return (format == ExpertFormat::BF16 && execution == 0) ||
         (format == ExpertFormat::FP8_E4M3_BLOCK128 && execution == 1) ||
         (format == ExpertFormat::MARLIN_INT4_GROUP128 && execution == 2);
}

}  // namespace

extern void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                       void* out, int N, int K,
                                       cudaStream_t stream);

static torch::Tensor _fp8_dequant_on_device(const torch::Tensor& w_fp8,
                                            const torch::Tensor& scale,
                                            cudaStream_t stream) {
  int N = w_fp8.size(0);
  int K = w_fp8.size(1);
  auto out = torch::empty(
      {N, K},
      torch::TensorOptions().dtype(torch::kBFloat16).device(w_fp8.device()));
  auto w_u8 = w_fp8.view(torch::kUInt8).contiguous();
  auto s_f32 = scale.to(torch::kFloat32).contiguous();
  fp8_dequant_blockwise_cuda(w_u8.data_ptr(), s_f32.data_ptr(), out.data_ptr(),
                             N, K, stream);
  return out;
}

void ExpertDispatcher::SetScales(
    const std::map<std::string, torch::Tensor>& scales) {
  if (scales.empty()) return;

  static const std::regex kLayerRe(R"(layers\.(\d+)\.)");
  static const std::regex kExpertRe(R"(experts\.(\d+)\.)");
  static const std::vector<std::string> kWeightNames = {
      "gate_proj.weight", "up_proj.weight", "down_proj.weight"};

  int max_layer = 0;
  int max_expert = 0;
  for (auto& kv : scales) {
    std::smatch m;
    std::string k = kv.first;
    if (std::regex_search(k, m, kLayerRe)) {
      max_layer = std::max(max_layer, std::stoi(m[1].str()));
    }
    if (std::regex_search(k, m, kExpertRe)) {
      max_expert = std::max(max_expert, std::stoi(m[1].str()));
    }
  }

  fp8_scales_.assign(max_layer + 1,
                     std::vector<std::vector<torch::Tensor>>(
                         max_expert + 1, std::vector<torch::Tensor>(3)));

  for (auto& kv : scales) {
    std::smatch m;
    std::string k = kv.first;
    int layer_idx = -1, expert_idx = -1, weight_idx = -1;
    if (std::regex_search(k, m, kLayerRe)) layer_idx = std::stoi(m[1].str());
    if (std::regex_search(k, m, kExpertRe)) expert_idx = std::stoi(m[1].str());
    for (int wi = 0; wi < 3; ++wi) {
      if (k.find(kWeightNames[wi]) != std::string::npos) {
        weight_idx = wi;
        break;
      }
    }
    if (layer_idx >= 0 && expert_idx >= 0 && weight_idx >= 0) {
      fp8_scales_[layer_idx][expert_idx][weight_idx] = kv.second.cpu();
    }
  }

  fp8_in_store_ = true;
}

bool ExpertDispatcher::RegisterExpertVariant(
    int layer_idx, int expert_idx, const std::string& format,
    std::uint64_t generation, const std::string& execution,
    const std::vector<std::uint32_t>& tensor_ids,
    const std::vector<std::string>& tensor_roles, std::int64_t payload_bytes,
    std::int64_t aligned_bytes, std::int64_t workspace_bytes) {
  const auto parsed_format = ParseExpertFormat(format);
  const auto parsed_execution = ParseExecutionKind(execution);
  if (!parsed_format.has_value() || !parsed_execution.has_value() ||
      !GenericFormatExecutionPair(*parsed_format, *parsed_execution) ||
      layer_idx < 0 || expert_idx < 0 || tensor_ids.empty() ||
      tensor_ids.size() != tensor_roles.size() || payload_bytes <= 0 ||
      aligned_bytes < payload_bytes || workspace_bytes < 0 ||
      kTopologyHandle == nullptr || kExpertResidencyManager == nullptr) {
    return false;
  }

  NodePtr node;
  try {
    const int gpu_id = expert_idx % std::max(kNumDevices(), 1);
    node = kTopologyHandle->CreateDetachedNode(tensor_ids, gpu_id);
  } catch (const std::exception&) {
    return false;
  }
  if (!node || node->byte_size != aligned_bytes) return false;

  ResidencyVariantKey key{LogicalExpertKey(layer_idx, expert_idx),
                          *parsed_format, generation};
  ResidencyVariant variant{key, node, payload_bytes, aligned_bytes,
                           workspace_bytes};
  ExpertExecutionDescriptor descriptor{key, *parsed_execution, tensor_ids,
                                       tensor_roles};
  std::lock_guard<std::mutex> lock(mutex_);
  if (registered_variants_.count(key) != 0) return false;
  registered_variants_.emplace(key, variant);
  execution_descriptors_.emplace(key, std::move(descriptor));
  kExpertResidencyManager->RegisterVariant(variant);
  return true;
}

bool ExpertDispatcher::SetPrecisionTargets(
    const std::vector<std::tuple<int, int, std::string, std::uint64_t>>&
        targets,
    std::uint64_t epoch) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (epoch <= precision_epoch_) return false;
  std::unordered_map<std::uint64_t, ResidencyVariantKey> validated;
  for (const auto& target : targets) {
    const int layer_idx = std::get<0>(target);
    const int expert_idx = std::get<1>(target);
    const auto format = ParseExpertFormat(std::get<2>(target));
    if (!format.has_value()) return false;
    ResidencyVariantKey key{LogicalExpertKey(layer_idx, expert_idx), *format,
                            std::get<3>(target)};
    if (registered_variants_.count(key) == 0 ||
        validated.count(key.logical_expert_key) != 0) {
      return false;
    }
#ifdef MOE_INFINITY_TESTING
    if (fail_transition_once_.has_value() &&
        fail_transition_once_->first == key.logical_expert_key &&
        fail_transition_once_->second == key.format) {
      ++transition_failed_count_;
      fail_transition_once_.reset();
      continue;
    }
#endif
    validated.emplace(key.logical_expert_key, key);
  }
  precision_targets_ = std::move(validated);
  precision_epoch_ = epoch;
  return true;
}

bool ExpertDispatcher::SetAdaptiveHbmBudgetBytes(std::int64_t bytes) {
  if (bytes <= 0 || kExpertResidencyManager == nullptr) return false;
  for (int gpu_id = 0; gpu_id < std::max(kNumDevices(), 1); ++gpu_id) {
    if (!kExpertResidencyManager->ConfigureCapacity(gpu_id, bytes))
      return false;
  }
  adaptive_hbm_budget_bytes_ = bytes;
  return true;
}

ExpertPolicyStats ExpertDispatcher::GetPrecisionMetrics() const {
  ExpertPolicyStats metrics = kExpertResidencyManager == nullptr
                                  ? ExpertPolicyStats{}
                                  : kExpertResidencyManager->Snapshot();
  metrics["budget_bytes"] = adaptive_hbm_budget_bytes_;
  metrics["published_generation"] =
      static_cast<std::int64_t>(published_generation_);
  metrics["manager_instance_id"] =
      static_cast<std::int64_t>(GetResidencyManagerId());
  metrics["manager_enabled"] = manager_enabled_ ? 1 : 0;
  metrics["phase_policy_enabled"] = phase_policy_enabled_ ? 1 : 0;
  metrics["transition_failed"] = transition_failed_count_;
  metrics["h2d_payload_bytes"] = h2d_payload_bytes_;
  metrics["h2d_transfers"] = h2d_transfers_;
  metrics["promotions"] = promotions_;
  metrics["demotions"] = demotions_;
  metrics["representation_hits"] = representation_hits_;
  metrics["representation_misses"] = representation_misses_;
  metrics["policy_epochs"] = static_cast<std::int64_t>(precision_epoch_);
  for (const char* name :
       {"conversion_input_bytes", "conversion_output_bytes",
        "conversion_seconds", "external_shared_resident_bytes"}) {
    metrics.emplace(name, 0);
  }
  return metrics;
}

std::vector<std::tuple<std::uint64_t, std::uint8_t, std::uint64_t, std::int64_t,
                       std::int64_t, std::uint8_t>>
ExpertDispatcher::GetResidentGenerationEntries() const {
  std::vector<std::tuple<std::uint64_t, std::uint8_t, std::uint64_t,
                         std::int64_t, std::int64_t, std::uint8_t>>
      rows;
  if (kExpertResidencyManager == nullptr) return rows;
  for (int gpu_id = 0; gpu_id < std::max(kNumDevices(), 1); ++gpu_id) {
    for (const auto& entry :
         kExpertResidencyManager->ResidentGenerations(gpu_id)) {
      rows.emplace_back(entry.key.logical_expert_key,
                        static_cast<std::uint8_t>(entry.key.format),
                        entry.key.generation, entry.payload_bytes, entry.bytes,
                        static_cast<std::uint8_t>(entry.state));
    }
  }
  return rows;
}

std::uintptr_t ExpertDispatcher::GetResidencyManagerId() const {
  return reinterpret_cast<std::uintptr_t>(kExpertResidencyManager.get());
}

void ExpertDispatcher::ConfigureResidencyManager(bool manager_enabled,
                                                 bool phase_policy_enabled) {
  manager_enabled_ = manager_enabled;
  phase_policy_enabled_ = phase_policy_enabled;
}

std::string ExpertDispatcher::GetActiveFormat() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_format_;
}

#ifdef MOE_INFINITY_TESTING
void ExpertDispatcher::InjectTransitionFailureOnceForTest(
    int layer_idx, int expert_idx, const std::string& format) {
  const auto parsed_format = ParseExpertFormat(format);
  TORCH_CHECK(parsed_format.has_value(), "unknown expert format");
  std::lock_guard<std::mutex> lock(mutex_);
  fail_transition_once_ =
      std::make_pair(LogicalExpertKey(layer_idx, expert_idx), *parsed_format);
}
#endif

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type, int num_threads)
    : pending_(0),
      num_enqueued_(0),
      start_(false),
      expert_type_(expert_type),
      dtype_(dtype),
      num_experts_(num_experts),
      cache_mutex_(kNumDevices()),
      cache_cv_(kNumDevices()),
      input_queue_(kNumDevices()),
      exec_queue_(kNumDevices()),
      cached_experts_(kNumDevices()),
      num_threads_(num_threads),
      modules_(kNumDevices() * num_threads, nullptr) {
  main_thread_stop_flag_.store(false);

  gpu_overload_ = std::make_unique<std::atomic<bool>[]>(kNumDevices());
  for (int i = 0; i < kNumDevices(); ++i) {
    gpu_overload_[i].store(false);
  }

  for (int i = 0; i < kNumDevices(); ++i) {
    cudaSetDevice(i);
    cudaStream_t fetch_stream;
    cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
    fetch_streams_.emplace_back(fetch_stream);

    auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
    std::string thread_name = "GPUFetchFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);
  }

  for (int i = 0; i < kNumDevices() * num_threads; ++i) {
    cudaSetDevice(i % kNumDevices());
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.emplace_back(exec_stream);

    modules_[i] = new MoEMLP(dtype, expert_type);

    auto thread_func =
        std::bind(&ExpertDispatcher::GPUExecFunc, this, i % kNumDevices(), i);
    std::string thread_name = "GPUExecFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
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
        case GPT_OSS_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new GptOssMoeDenseActDense();
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
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("expert_enqueue");
#endif
  // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
  int layer_idx = args.layer_idx;
  int expert_idx = args.expert_idx;
  auto expert_node = experts_[expert_idx][layer_idx];
  if (manager_enabled_ && !ApplyPrecisionTarget(expert_node, args.gpu_id)) {
    DLOG_WARN("adaptive precision target failed; using current generation");
  }

  {
    auto expected = NodeExecState::IDLE;
    if (!expert_node->node->exec_state.compare_exchange_strong(
            expected, NodeExecState::FETCHING, std::memory_order_acq_rel)) {
      if (expert_node->node->is_prefetching.load(std::memory_order_acquire)) {
        expert_node->node->pending_dispatches.fetch_add(
            1, std::memory_order_acq_rel);
        args.wait_for_prefetch = true;
        args.gpu_id = expert_node->node->default_device.index();
        input_queue_[args.gpu_id].Push(args);
        return;
      }
      DLOG_FATAL(
          "ExpertDispatcher::Enqueue: exec_state CAS failed (expert_idx ",
          expert_idx, " layer_idx ", layer_idx, " node ",
          expert_node->node->str(), " state=", static_cast<int>(expected), ")");
    }
  }
  expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

  if (expert_node->node->device.is_cuda()) {
    // Cache-hit path: expert already on GPU, skip fetch queue entirely.
    // transfer_event stays nullptr — GPUExecFunc will skip cudaStreamWaitEvent.
    args.gpu_id = expert_node->node->device.index();
    DLOG_DEBUG("cache_hit_immediate: expert_idx ", args.expert_idx,
               " layer_idx ", args.layer_idx, " gpu_id ", args.gpu_id);

    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();

    ExecArgs exec_args;
    // exec_args.hidden_states = std::move(input);
    exec_args.expert_node = expert_node;
    expert_node->SetTensorsFromBlob(expert_node->node->device);
    exec_args.out_gpu_id = original_device.index();
    exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
    exec_args.evict = false;
    exec_args.hit = true;
    if (manager_enabled_ && kExpertResidencyManager != nullptr) {
      auto active = kExpertResidencyManager->ActiveGeneration(
          args.gpu_id, LogicalExpertKey(layer_idx, expert_idx));
      if (active.has_value()) {
        exec_args.execution_key = *active;
        exec_args.execution_lease_id =
            kDemandResidencyClient->AcquireLease(*active, LeaseKind::EXECUTION);
      }
    }
    cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
    cache_access_count_.fetch_add(1, std::memory_order_relaxed);
    // transfer_event = nullptr: expert is already on GPU, no H2D wait needed

    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    // std::unique_lock<std::mutex> lock(exec_mutex_[args.gpu_id]);
    // exec_queue_[args.gpu_id].push_back(std::move(exec_args));
    exec_queue_[args.gpu_id].Push(exec_args);
  } else {
    // std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
    // input_queue_[args.gpu_id].push_back(std::move(args));
    input_queue_[args.gpu_id].Push(args);
  }
  // input_cv_[args.gpu_id].notify_all();
  // exec_cv_[args.gpu_id].notify_all();
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

bool ExpertDispatcher::ApplyPrecisionTarget(const ExpertNodePtr& expert_node,
                                            int gpu_id) {
  if (!expert_node || gpu_id < 0 || kExpertResidencyManager == nullptr ||
      kDemandResidencyClient == nullptr) {
    return true;
  }
  const auto logical_key =
      LogicalExpertKey(expert_node->layer_idx, expert_node->expert_idx);
  ResidencyVariant destination;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto target = precision_targets_.find(logical_key);
    if (target == precision_targets_.end()) {
      destination = ResidencyVariant{
          ResidencyVariantKey{logical_key, ExpertFormat::BF16, 0},
          expert_node->node, expert_node->node->byte_size,
          expert_node->node->byte_size, 0};
    } else {
      const auto variant = registered_variants_.find(target->second);
      if (variant == registered_variants_.end()) return false;
      destination = variant->second;
    }
  }
  kExpertResidencyManager->RegisterVariant(destination);
  const auto active =
      kExpertResidencyManager->ActiveGeneration(gpu_id, logical_key);
  if (active.has_value() && *active == destination.key) {
    ++representation_hits_;
    return true;
  }
  ++representation_misses_;

  ResidencyTicket transaction =
      active.has_value()
          ? kDemandResidencyClient->BeginTransition(*active, destination,
                                                    gpu_id, ExpertPhase::MIXED)
          : kDemandResidencyClient->BeginAdmission(
                destination, gpu_id, ExpertPhase::MIXED, AdmissionMode::CACHE);
  if (!transaction.valid) return false;
  if (transaction.reserved_victim_key.has_value() &&
      !kExpertResidencyManager->EvictReserved(transaction)) {
    kExpertResidencyManager->AbortTransaction(transaction);
    return false;
  }

  try {
    cudaEvent_t transfer_done = nullptr;
    destination.node->SetDevice(CUDA_DEVICE(gpu_id), true,
                                fetch_streams_[gpu_id], &transfer_done);
    if (transfer_done != nullptr) {
      cudaEventSynchronize(transfer_done);
      kCudaEventPool->Release(transfer_done);
    }
  } catch (const std::exception&) {
    kExpertResidencyManager->AbortTransaction(transaction);
    ++transition_failed_count_;
    return false;
  }
  if (!kExpertResidencyManager->CommitTransaction(transaction)) return false;
  h2d_payload_bytes_ += destination.payload_bytes;
  ++h2d_transfers_;
  if (active.has_value()) {
    if (static_cast<int>(destination.key.format) <
        static_cast<int>(active->format)) {
      ++promotions_;
    } else if (static_cast<int>(destination.key.format) >
               static_cast<int>(active->format)) {
      ++demotions_;
    }
  }
  expert_node->node = destination.node;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_format_ = FormatName(destination.key.format);
    ++published_generation_;
  }
  return true;
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
      // experts_[expert_idx][layer_idx]->jit_module =
      //     new torch::jit::script::Module(torch::jit::load(jit_path));
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
  }
}

void ExpertDispatcher::NotifyFetchStart() {
  for (int i = 0; i < kNumDevices(); ++i) {
    // std::unique_lock<std::mutex> lock(input_mutex_[i]);
    input_queue_[i].NotifyAll();
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
  cache_hit_count_.store(0, std::memory_order_relaxed);
  cache_access_count_.store(0, std::memory_order_relaxed);
}

std::int64_t ExpertDispatcher::GetCacheOccupancyBytes() {
  std::int64_t total = 0;
  for (auto& gpu_cache : cached_experts_) {
    for (auto key : gpu_cache) {
      int layer_idx = static_cast<int>(key >> 32);
      int expert_idx = static_cast<int>(key & 0xFFFFFFFF);
      if (expert_idx < 0 || expert_idx >= static_cast<int>(experts_.size())) {
        continue;
      }
      if (layer_idx < 0 ||
          layer_idx >= static_cast<int>(experts_[expert_idx].size())) {
        continue;
      }
      auto& expert_node = experts_[expert_idx][layer_idx];
      if (expert_node && expert_node->node) {
        total += expert_node->node->byte_size;
      }
    }
  }
  return total;
}

double ExpertDispatcher::GetCacheHitRate() const {
  std::uint64_t access = cache_access_count_.load(std::memory_order_relaxed);
  if (access == 0) {
    return 0.0;
  }
  std::uint64_t hits = cache_hit_count_.load(std::memory_order_relaxed);
  return static_cast<double>(hits) / static_cast<double>(access);
}

// void ExpertDispatcher::GPUThreadFunc(int gpu_id) {
//   while (!main_thread_stop_flag_.load()) {
//   }
// }

ExpertNodePtr ExpertDispatcher::FindExpertEvict(int gpu_id) {
  uint64_t min_visit_count = INT_MAX;
  ExpertNodePtr evict_expert_node = nullptr;

  for (auto& key : cached_experts_[gpu_id]) {
    auto layer_idx = key >> 32;
    auto expert_idx = key & 0xFFFFFFFF;
    auto node = experts_[expert_idx][layer_idx]->node;
    if (node == nullptr) continue;
    if (node->device.is_cuda() && node->incache_visit_count < min_visit_count &&
        node->pending_dispatches.load(std::memory_order_acquire) == 0 &&
        node->exec_state.load(std::memory_order_acquire) ==
            NodeExecState::IDLE) {
      evict_expert_node = experts_[expert_idx][layer_idx];
      min_visit_count = node->incache_visit_count;
    }
  }
  return evict_expert_node;
}

void ExpertDispatcher::GPUFetchFunc(int gpu_id) {
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("gpu_fetch");
#endif
  cudaSetDevice(gpu_id);
  cudaStream_t stream = fetch_streams_[gpu_id];

  while (!main_thread_stop_flag_.load(std::memory_order_acquire)) {
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
    // std::unique_lock<std::mutex> lock(input_mutex_[gpu_id]);
    // input_cv_[gpu_id].wait(lock, [&] { return !input_queue_[gpu_id].empty();
    // });

    // CallArgs args = std::move(input_queue_[gpu_id].front());
    // input_queue_[gpu_id].pop_front();

    // lock.unlock();
    CallArgs args;
    if (!input_queue_[gpu_id].Pop(args)) {
      break;
    }

    auto device = CUDA_DEVICE(gpu_id);
    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();
    int64_t layer_idx = args.layer_idx;
    int64_t expert_idx = args.expert_idx;
    int64_t batch_size = hidden_states_.size(0);

    auto expert_node = experts_[expert_idx][layer_idx];

    if (args.wait_for_prefetch) {
      while (expert_node->node->exec_state.load(std::memory_order_acquire) !=
                 NodeExecState::IDLE &&
             !main_thread_stop_flag_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
      if (main_thread_stop_flag_.load(std::memory_order_acquire)) break;
      {
        auto expected = NodeExecState::IDLE;
        expert_node->node->exec_state.compare_exchange_strong(
            expected, NodeExecState::FETCHING, std::memory_order_acq_rel);
      }
      expert_node->node->pending_dispatches.fetch_sub(
          1, std::memory_order_acq_rel);
      if (expert_node->node->device.is_cuda()) {
        expert_node->node->incache_visit_count += 1;
        expert_node->SetTensorsFromBlob(expert_node->node->device);
        ExecArgs exec_args;
        exec_args.expert_node = expert_node;
        exec_args.out_gpu_id = original_device.index();
        exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
        exec_args.evict = false;
        exec_args.hit = true;
        cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
        cache_access_count_.fetch_add(1, std::memory_order_relaxed);
        exec_args.transfer_event = nullptr;
        if (manager_enabled_ && kExpertResidencyManager != nullptr) {
          auto active = kExpertResidencyManager->ActiveGeneration(
              gpu_id, LogicalExpertKey(layer_idx, expert_idx));
          if (active.has_value()) {
            exec_args.execution_key = *active;
            exec_args.execution_lease_id = kDemandResidencyClient->AcquireLease(
                *active, LeaseKind::EXECUTION);
          }
        }
        exec_queue_[gpu_id].Push(exec_args);
        continue;
      }
    }

    bool cache_hit = expert_node->node->device.is_cuda();

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: gpu_id " << gpu_id
    //           << " layer_idx " << layer_idx << " expert_idx " << expert_idx
    //           << " cache_hit " << cache_hit << " node "
    //           << expert_node->node->device.str() << std::endl;
    DLOG_DEBUG("ExpertDispatcher::GPUFetchFunc: gpu_id ", gpu_id, " layer_idx ",
               layer_idx, " expert_idx ", expert_idx, "cache_hit ", cache_hit,
               "cache_size ", cache_sizes_[gpu_id], " incache count ",
               cached_experts_[gpu_id].size());

    if (!cache_hit && cache_sizes_[gpu_id] < expert_node->node->byte_size) {
      if (batch_size > 1) {
        // force fetch to GPU regardless of cache size, only for prefill
        // only one extra cache slot for prefill
        DLOG_DEBUG("overloading expert cache: gpu_id ", gpu_id, " cache size ",
                   cache_sizes_[gpu_id], " incache count ",
                   cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                   " expert_idx ", expert_idx);
        while (gpu_overload_[gpu_id].load(std::memory_order_acquire) &&
               !main_thread_stop_flag_.load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (main_thread_stop_flag_.load(std::memory_order_acquire)) {
          break;
        }
        gpu_overload_[gpu_id].store(true, std::memory_order_release);
      } else {
        // find the expert in gpu and min incache_visit_count
        ExpertNodePtr evict_expert_node = FindExpertEvict(gpu_id);
        if (evict_expert_node == nullptr) {
          // wait for notification that cache is available
          DLOG_WARN(
              "All cached expert locked, waiting for cache to be available. "
              "gpu_id ",
              gpu_id, " cache size ", cache_sizes_[gpu_id], " incache count ",
              cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
              " expert_idx ", expert_idx);
          {
            std::unique_lock<std::mutex> lock(cache_mutex_[gpu_id]);
            cache_cv_[gpu_id].wait(lock, [&] {
              return main_thread_stop_flag_.load(std::memory_order_acquire) ||
                     FindExpertEvict(gpu_id) != nullptr;
            });
          }
          if (main_thread_stop_flag_.load(std::memory_order_acquire)) break;
          evict_expert_node = FindExpertEvict(gpu_id);
        }
        // auto num_layers = experts_[0].size();
        // auto num_experts = experts_.size();

        // for (size_t i = 0; i < num_experts; ++i) {
        //   for (size_t j = 0; j < num_layers; ++j) {
        // auto node = experts_[i][j]->node;
        // if (node == nullptr) {
        //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: node is nullptr"
        //   //           << " layer_idx " << j << " expert_idx " << i <<
        //   //           std::endl;
        //   continue;
        // }
        // if (node->device.is_cuda() &&
        //     node->incache_visit_count < min_visit_count &&
        //     node->mutex.try_lock()) {
        //   evict_node = node;
        //   min_visit_count = node->incache_visit_count;
        //   node->mutex.unlock();
        //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: evict node "
        //   //           << evict_node->device.str() << " incache_visit_count "
        //   //           << min_visit_count << std::endl;
        // }
        //   }
        // }
        if (evict_expert_node == nullptr) {
          for (int retry = 0;
               retry < 100 && evict_expert_node == nullptr &&
               !main_thread_stop_flag_.load(std::memory_order_acquire);
               ++retry) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            evict_expert_node = FindExpertEvict(gpu_id);
          }
          if (main_thread_stop_flag_.load(std::memory_order_acquire)) break;
          DLOG_FATAL_IF(
              evict_expert_node == nullptr,
              "ExpertDispatcher::GPUFetchFunc: evict_node is nullptr after "
              "retries, gpu_id",
              gpu_id, "cache size", cache_sizes_[gpu_id], "in cache count",
              cached_experts_[gpu_id].size());
        }

        DLOG_DEBUG("evicting expert: gpu_id ", gpu_id, " cache size ",
                   cache_sizes_[gpu_id], " incache count ",
                   cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                   " expert_idx ", expert_idx);

        auto evict_node = evict_expert_node->node;
        evict_node->SetDevice(evict_node->default_host);
        cache_sizes_[gpu_id] += evict_node->byte_size;
        int64_t evict_layer_idx = evict_expert_node->layer_idx;
        int64_t evict_expert_idx = evict_expert_node->expert_idx;

        // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
        uint64_t evict_key = (evict_layer_idx << 32) + evict_expert_idx;
        auto it = cached_experts_[gpu_id].find(evict_key);
        if (it != cached_experts_[gpu_id].end()) {
          cached_experts_[gpu_id].erase(it);
        } else {
          DLOG_FATAL(
              "ExpertDispatcher::GPUFetchFunc: evict_key not found. layer_idx ",
              evict_layer_idx, " expert_idx ", evict_expert_idx);
        }
      }
    }

    if (!gpu_overload_[gpu_id].load(std::memory_order_acquire)) {
      cache_sizes_[gpu_id] -= expert_node->node->byte_size;
      uint64_t key = (layer_idx << 32) + expert_idx;
      cached_experts_[gpu_id].insert(key);
    }

    cudaEvent_t transfer_done = nullptr;
    expert_node->node->SetDevice(device, true, stream, &transfer_done);
    expert_node->node->incache_visit_count += 1;
    expert_node->SetTensorsFromBlob(device);
    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: move to device gpu_id "
    //           << gpu_id << " layer_idx " << layer_idx << " expert_idx "
    //           << expert_idx << " node "
    //           << expert_node->node->device.str() << std::endl;

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
      exec_args.evict = gpu_overload_[gpu_id].load(std::memory_order_acquire);
      exec_args.hit = cache_hit;
      cache_access_count_.fetch_add(1, std::memory_order_relaxed);
      if (cache_hit) cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
      exec_args.transfer_event = transfer_done;
      if (manager_enabled_ && kExpertResidencyManager != nullptr) {
        auto active = kExpertResidencyManager->ActiveGeneration(
            gpu_id, LogicalExpertKey(layer_idx, expert_idx));
        if (active.has_value()) {
          exec_args.execution_key = *active;
          exec_args.execution_lease_id = kDemandResidencyClient->AcquireLease(
              *active, LeaseKind::EXECUTION);
        }
      }
      // std::lock_guard<std::mutex> lock(exec_mutex_[gpu_id]);
      // exec_queue_[gpu_id].emplace_back(std::move(exec_args));
      exec_queue_[gpu_id].Push(exec_args);
    }
    // exec_cv_[gpu_id].notify_all();
  }
}

void ExpertDispatcher::GPUExecFunc(int gpu_id, int thread_idx) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream = exec_streams_[thread_idx];

  while (!main_thread_stop_flag_.load(std::memory_order_acquire)) {
    ExecArgs args;
    if (!exec_queue_[gpu_id].Pop(args)) {
      break;
    }

    if (args.expert_node == nullptr) {
      continue;
    }

    if (args.transfer_event != nullptr) {
      cudaStreamWaitEvent(stream, args.transfer_event, 0);
      kCudaEventPool->Release(args.transfer_event);
      args.transfer_event = nullptr;
    }

    try {
      if (args.transfer_event != nullptr) {
        cudaStreamWaitEvent(stream, args.transfer_event, 0);
        cudaEventDestroy(args.transfer_event);
        args.transfer_event = nullptr;
      }

      int64_t batch_size = hidden_states_.size(0);
      auto device = CUDA_DEVICE(gpu_id);
      auto expert_idx = args.expert_node->expert_idx;

      auto token_mask = router_mask_.index({"...", expert_idx});
      torch::Tensor input = (batch_size == 1)
                                ? hidden_states_.to(device)
                                : hidden_states_.index({token_mask}).to(device);

      std::optional<ExpertExecutionDescriptor> descriptor;
      if (manager_enabled_ && kExpertResidencyManager != nullptr) {
        auto active = kExpertResidencyManager->ActiveGeneration(
            gpu_id, LogicalExpertKey(args.expert_node->layer_idx,
                                     args.expert_node->expert_idx));
        if (active.has_value()) {
          std::lock_guard<std::mutex> lock(mutex_);
          const auto found = execution_descriptors_.find(*active);
          if (found != execution_descriptors_.end()) descriptor = found->second;
        }
      }
      if (descriptor.has_value()) {
        modules_[thread_idx]->SetRepresentation(*descriptor);
        modules_[thread_idx]->PrepareRepresentation(stream);
      } else {
        modules_[thread_idx]->SetTensorsFromIds(
            args.expert_node->node->tensor_ids);
      }

      if (fp8_in_store_) {
        int64_t layer_idx = args.expert_node->layer_idx;
        int64_t expert_idx_val = args.expert_node->expert_idx;
        if (layer_idx < (int64_t)fp8_scales_.size() &&
            expert_idx_val < (int64_t)fp8_scales_[layer_idx].size()) {
          modules_[thread_idx]->SetFp8Scales(
              fp8_scales_[layer_idx][expert_idx_val]);
          modules_[thread_idx]->DequantFp8Params(stream);
        }
      }

      c10::cuda::CUDAStream torch_stream =
          c10::cuda::getStreamFromExternal(stream, gpu_id);
      c10::cuda::CUDAStreamGuard guard(torch_stream);

      if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
        modules_[thread_idx]->DequantMxfp4Params(stream);
      }

      torch::Tensor output;
      {
#ifndef NVTX_DISABLE
        nvtx3::scoped_range r("expert_compute");
#endif
        output = modules_[thread_idx]->forward(input, stream);
      }
      OutputFunc(args, output, token_mask, gpu_id);
    } catch (const std::exception& e) {
      if (args.transfer_event != nullptr) {
        cudaEventDestroy(args.transfer_event);
        args.transfer_event = nullptr;
      }
      DLOG_WARN("GPUExecFunc: expert forward failed: ", e.what(),
                " (expert_idx=", args.expert_node->expert_idx,
                " layer_idx=", args.expert_node->layer_idx, ")");
      args.expert_node->node->exec_state.store(NodeExecState::IDLE,
                                               std::memory_order_release);
      if (args.execution_lease_id != 0) {
        kDemandResidencyClient->ReleaseLease(args.execution_lease_id);
      }
      pending_.fetch_sub(1);
      if (pending_.load() == 0) {
        pending_cv_.notify_all();
      }
    }
  }
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  torch::Tensor token_mask, int gpu_id) {
  auto output_device =
      (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
  torch::Tensor output_tensor = output.to(output_device).to(torch::kFloat32);

  DLOG_TRACE("ExpertDispatcher::OutputFunc: output_tensor ",
             output_tensor.sizes().vec(), "(", output_tensor.device().str(),
             ")");

  // args.expert_node->node->mutex.unlock();
  int64_t expert_idx = args.expert_node->expert_idx;
  int64_t layer_idx = args.expert_node->layer_idx;
  int64_t batch_size = hidden_states_.size(0);

  args.expert_node->node->exec_state.store(NodeExecState::IDLE,
                                           std::memory_order_release);
  if (args.execution_lease_id != 0 && kDemandResidencyClient != nullptr) {
    kExpertResidencyManager->RecordLastUse(args.execution_key, nullptr);
    kDemandResidencyClient->ReleaseLease(args.execution_lease_id);
    kDemandResidencyClient->ReapWorkspace(gpu_id);
    kDemandResidencyClient->ReapRetired(gpu_id);
  }
  if (args.evict) {
    args.expert_node->node->SetDevice(args.expert_node->node->default_host,
                                      true, nullptr);
    DLOG_DEBUG("pop out overloaded expert cache_sizes_[gpu_id] ",
               cache_sizes_[gpu_id], "gpu_id ", gpu_id, "layer_idx ", layer_idx,
               "expert_idx ", expert_idx);
    gpu_overload_[gpu_id].store(false, std::memory_order_release);
  }
  cache_cv_[gpu_id].notify_all();

  {
    std::lock_guard<std::mutex> lock(accum_mutex_);
    if (batch_size == 1) {
      final_hidden_states_.add_(
          output_tensor *
          router_weight_.index({torch::indexing::Slice(), expert_idx}));
    } else {
      auto token_indices = torch::nonzero(token_mask).squeeze(1);
      auto weights =
          router_weight_.index({token_mask, expert_idx}).unsqueeze(1);
      auto weighted_output = output_tensor * weights;
      final_hidden_states_.index_add_(0, token_indices, weighted_output);
    }
  }
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
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("expert_wait_barrier");
#endif
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
  hidden_states_ = hidden_states;
  router_mask_ = router_mask;
  router_weight_ = router_weight;  // this can be float32
  final_hidden_states_ = torch::zeros_like(hidden_states, options);
}
