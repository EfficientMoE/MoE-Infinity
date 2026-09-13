// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "aio/archer_tensor_handle.h"
#include "aio/archer_tensor_index.h"

// TensorStore is the seam between the offloading engine (prefetch, memory,
// dispatch, topology) and the parameter/tensor store (checkpoint-backed
// index + async disk I/O).
//
// PR1 (in-tree): a single V1 implementation wraps the legacy aio
// ArcherTensorIndex + ArcherTensorHandle pair, replacing the former
// kTensorIndex / kArcherTensorHandle process globals.
//
// The index() accessor is transitional: it exposes the v1 per-tensor index
// so existing call sites keep byte-identical behavior. It is removed when
// the store moves to the moe-store repository and group-based reads
// (ReadGroup) replace per-tensor index walks.
class TensorStore {
 public:
  virtual ~TensorStore() = default;

  // Transitional v1 index access (removed with the moe-store v2 format).
  virtual ArcherTensorIndex& index() = 0;
  virtual const ArcherTensorIndex& index() const = 0;

  // Write / registration surface (checkpoint conversion + load time).
  virtual void StoreTensor(std::uint32_t tensor_id, torch::Tensor& buffer) = 0;
  virtual void RegisterTensor(std::uint32_t tensor_id,
                              torch::Tensor& buffer) = 0;
  virtual void SerializeIndex(const std::string& path) = 0;

  // Read surface (engine fetch paths).
  virtual void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer) = 0;
  virtual void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer,
                         const torch::Device& device) = 0;
  virtual void ReadTensor(std::uint32_t tensor_id, void* dst,
                          bool on_demand = false) = 0;
  virtual void ReadBulk(const std::string& filename, void* dst, bool on_demand,
                        std::int64_t num_bytes, std::int64_t offset) = 0;

  // Identity / metadata.
  virtual std::uint32_t GetTensorId(void* data_ptr) const = 0;
  virtual void UpdateTensorMap(void* old_data_ptr, void* new_data_ptr) = 0;
  virtual bool IsTensorIndexInitialized() const = 0;
  virtual std::int64_t GetTensorSizeAligned(std::uint32_t tensor_id) const = 0;
  virtual torch::TensorOptions GetTensorOptions(
      std::uint32_t tensor_id) const = 0;
  virtual std::string GetIndexFileName(std::uint32_t file_id) const = 0;

  // Canonical snapshot + derivative overlay surface (dflash).
  virtual std::vector<std::unordered_map<std::string, pybind11::object>>
  GetCanonicalTensorIndexSnapshot() const = 0;
  virtual void BeginDerivativeOverlay(const std::string& generation,
                                      std::int64_t canonical_max_tensor_id,
                                      std::int64_t canonical_max_file_id) = 0;
  virtual void RegisterDerivativeTensor(const std::string& generation,
                                        std::int64_t tensor_id,
                                        std::int64_t file_id,
                                        std::int64_t offset, std::int64_t size,
                                        const std::vector<std::int64_t>& shape,
                                        const std::string& dtype) = 0;
  virtual void CommitDerivativeOverlay(const std::string& generation) = 0;
  virtual void AbortDerivativeOverlay(const std::string& generation) = 0;
};

// Creates the v1 store backed by the legacy aio index/handle pair.
std::shared_ptr<TensorStore> CreateV1TensorStore(const std::string& prefix,
                                                 int num_io_threads);

// Transitional process-wide accessor for legacy call sites that predate
// dependency injection and whose construction graph does not reach the
// composition root (Node::SetDevice, SetModuleMemoryFromDisk & friends,
// Expert<T>/MoEMLP tensor binding). Set/cleared by ArcherPrefetchHandle,
// which enforces one model per process. Removed in the moe-store split
// when group reads replace these paths.
TensorStore* GetTensorStore();
std::shared_ptr<TensorStore> GetTensorStoreShared();
void SetCurrentTensorStore(std::shared_ptr<TensorStore> store);
