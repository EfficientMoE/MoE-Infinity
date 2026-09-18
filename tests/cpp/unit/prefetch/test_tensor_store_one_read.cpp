// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "aio/archer_tensor_index.h"
#include "model/model_topology.h"
#include "store/tensor_store.h"

namespace {

class CountingTensorStore final : public TensorStore {
 public:
  explicit CountingTensorStore(std::string filename)
      : filename_(std::move(filename)) {}

  ArcherTensorIndex& index() override { return index_; }
  const ArcherTensorIndex& index() const override { return index_; }

  void StoreTensor(std::uint32_t, torch::Tensor&) override {}
  void RegisterTensor(std::uint32_t, torch::Tensor&) override {}
  void SerializeIndex(const std::string&) override {}
  void SetTensor(std::uint32_t, torch::Tensor&) override {}
  void SetTensor(std::uint32_t, torch::Tensor&, const torch::Device&) override {
  }

  void ReadTensor(std::uint32_t, void*, bool) override { ++tensor_reads; }

  void ReadBulk(const std::string& filename, void* dst, bool,
                std::int64_t num_bytes, std::int64_t offset) override {
    ++pread_calls;
    last_size = num_bytes;
    last_offset = offset;
    const int fd = open(filename.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(pread(fd, dst, static_cast<std::size_t>(num_bytes), offset),
              num_bytes);
    close(fd);
  }

  std::uint32_t GetTensorId(void*) const override { return 0; }
  void UpdateTensorMap(void*, void*) override {}
  bool IsTensorIndexInitialized() const override { return true; }
  std::int64_t GetTensorSizeAligned(std::uint32_t tensor_id) const override {
    return static_cast<std::int64_t>(index_.at(tensor_id).size);
  }
  torch::TensorOptions GetTensorOptions(
      std::uint32_t tensor_id) const override {
    return index_.at(tensor_id).options;
  }
  std::string GetIndexFileName(std::uint32_t) const override {
    return filename_;
  }

  std::vector<std::unordered_map<std::string, pybind11::object>>
  GetCanonicalTensorIndexSnapshot() const override {
    return {};
  }
  void BeginDerivativeOverlay(const std::string&, std::int64_t,
                              std::int64_t) override {}
  void RegisterDerivativeTensor(const std::string&, std::int64_t, std::int64_t,
                                std::int64_t, std::int64_t,
                                const std::vector<std::int64_t>&,
                                const std::string&) override {}
  void CommitDerivativeOverlay(const std::string&) override {}
  void AbortDerivativeOverlay(const std::string&) override {}

  int pread_calls = 0;
  int tensor_reads = 0;
  std::int64_t last_size = 0;
  std::int64_t last_offset = -1;

 private:
  std::string filename_;
  ArcherTensorIndex index_;
};

TEST(TensorStoreOneRead, ExpertGroupFetchIssuesOnePread) {
  char path[] = "/tmp/moe-infinity-one-read-XXXXXX";
  const int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  constexpr std::int64_t kMemberSize = 4096;
  constexpr std::int64_t kGroupSize = 3 * kMemberSize;
  std::array<char, kGroupSize> source{};
  ASSERT_EQ(write(fd, source.data(), source.size()), kGroupSize);
  close(fd);

  auto store = std::make_shared<CountingTensorStore>(path);
  const auto options = torch::TensorOptions().dtype(torch::kUInt8);
  std::vector<TensorID> tensor_ids{10, 11, 12};
  for (std::size_t slot = 0; slot < tensor_ids.size(); ++slot) {
    TensorStorageMeta meta{
        0,           static_cast<std::int64_t>(slot) * kMemberSize,
        kMemberSize, {kMemberSize},
        options,     tensor_ids[slot]};
    store->index().emplace(tensor_ids[slot], std::move(meta));
  }

  SetCurrentTensorStore(store);
  std::array<char, kGroupSize> destination{};
  SetModuleMemoryFromDisk(tensor_ids, destination.data(), true);

  EXPECT_EQ(store->pread_calls, 1);
  EXPECT_EQ(store->tensor_reads, 0);
  EXPECT_EQ(store->last_offset, 0);
  EXPECT_EQ(store->last_size, kGroupSize);

  SetCurrentTensorStore(nullptr);
  unlink(path);
}

}  // namespace
