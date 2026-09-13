// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "store/tensor_store.h"

#include <mutex>
#include <utility>

namespace {

class V1TensorStore final : public TensorStore {
 public:
  V1TensorStore(const std::string& prefix, int num_io_threads)
      : index_(std::make_unique<ArcherTensorIndex>()),
        handle_(std::make_unique<ArcherTensorHandle>(prefix, num_io_threads,
                                                     index_.get())) {}

  ArcherTensorIndex& index() override { return *index_; }
  const ArcherTensorIndex& index() const override { return *index_; }

  void StoreTensor(std::uint32_t tensor_id, torch::Tensor& buffer) override {
    handle_->StoreTensor(tensor_id, buffer);
  }
  void RegisterTensor(std::uint32_t tensor_id, torch::Tensor& buffer) override {
    handle_->RegisterTensor(tensor_id, buffer);
  }
  void SerializeIndex(const std::string& path) override {
    index_->Serialize(path.c_str());
  }

  void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer) override {
    handle_->SetTensor(tensor_id, buffer);
  }
  void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer,
                 const torch::Device& device) override {
    handle_->SetTensor(tensor_id, buffer, device);
  }
  void ReadTensor(std::uint32_t tensor_id, void* dst, bool on_demand) override {
    handle_->ReadTensor(tensor_id, dst, on_demand);
  }
  void ReadBulk(const std::string& filename, void* dst, bool on_demand,
                std::int64_t num_bytes, std::int64_t offset) override {
    handle_->ReadBulk(filename, dst, on_demand, num_bytes, offset);
  }

  std::uint32_t GetTensorId(void* data_ptr) const override {
    return handle_->GetTensorId(data_ptr);
  }
  void UpdateTensorMap(void* old_data_ptr, void* new_data_ptr) override {
    handle_->UpdateTensorMap(old_data_ptr, new_data_ptr);
  }
  bool IsTensorIndexInitialized() const override {
    return handle_->IsTensorIndexInitialized();
  }
  std::int64_t GetTensorSizeAligned(std::uint32_t tensor_id) const override {
    return handle_->GetTensorSizeAligned(tensor_id);
  }
  torch::TensorOptions GetTensorOptions(
      std::uint32_t tensor_id) const override {
    return handle_->GetTensorOptions(tensor_id);
  }
  std::string GetIndexFileName(std::uint32_t file_id) const override {
    return handle_->GetIndexFileName(file_id);
  }

  std::vector<std::unordered_map<std::string, pybind11::object>>
  GetCanonicalTensorIndexSnapshot() const override {
    return handle_->GetCanonicalTensorIndexSnapshot();
  }
  void BeginDerivativeOverlay(const std::string& generation,
                              std::int64_t canonical_max_tensor_id,
                              std::int64_t canonical_max_file_id) override {
    handle_->BeginDerivativeOverlay(generation, canonical_max_tensor_id,
                                    canonical_max_file_id);
  }
  void RegisterDerivativeTensor(const std::string& generation,
                                std::int64_t tensor_id, std::int64_t file_id,
                                std::int64_t offset, std::int64_t size,
                                const std::vector<std::int64_t>& shape,
                                const std::string& dtype) override {
    handle_->RegisterDerivativeTensor(generation, tensor_id, file_id, offset,
                                      size, shape, dtype);
  }
  void CommitDerivativeOverlay(const std::string& generation) override {
    handle_->CommitDerivativeOverlay(generation);
  }
  void AbortDerivativeOverlay(const std::string& generation) override {
    handle_->AbortDerivativeOverlay(generation);
  }

 private:
  std::unique_ptr<ArcherTensorIndex> index_;
  std::unique_ptr<ArcherTensorHandle> handle_;
};

std::shared_ptr<TensorStore> g_current_store;
std::mutex g_current_store_mutex;

}  // namespace

std::shared_ptr<TensorStore> CreateV1TensorStore(const std::string& prefix,
                                                 int num_io_threads) {
  return std::make_shared<V1TensorStore>(prefix, num_io_threads);
}

TensorStore* GetTensorStore() {
  std::lock_guard<std::mutex> lock(g_current_store_mutex);
  return g_current_store.get();
}

std::shared_ptr<TensorStore> GetTensorStoreShared() {
  std::lock_guard<std::mutex> lock(g_current_store_mutex);
  return g_current_store;
}

void SetCurrentTensorStore(std::shared_ptr<TensorStore> store) {
  std::lock_guard<std::mutex> lock(g_current_store_mutex);
  g_current_store = std::move(store);
}
