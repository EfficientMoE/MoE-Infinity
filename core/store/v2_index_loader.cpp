// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "store/v2_index_loader.h"

#include <sys/stat.h>

#include <stdexcept>

#include "aio/archer_tensor_index.h"
#include "index_v2.h"
#include "utils/logger.h"

namespace {

torch::ScalarType DtypeTokenToScalarType(const std::string& token) {
  if (token == "float32") return torch::kFloat;
  if (token == "float16") return torch::kHalf;
  if (token == "bfloat16") return torch::kBFloat16;
  if (token == "float8_e4m3fn") return torch::kFloat8_e4m3fn;
  if (token == "uint8") return torch::kByte;
  if (token == "int8") return torch::kChar;
  if (token == "int32") return torch::kInt;
  if (token == "int64") return torch::kLong;
  throw std::runtime_error("moe-store index: unknown dtype token " + token);
}

}  // namespace

bool HasV2Index(const std::string& store_dir) {
  std::string prefix = store_dir;
  if (!prefix.empty() && prefix.back() != '/') prefix += '/';
  struct stat st;
  return stat((prefix + moe_store::kIndexFileName).c_str(), &st) == 0;
}

bool LoadV2IndexInto(const std::string& store_dir, ArcherTensorIndex* index) {
  if (!HasV2Index(store_dir)) {
    return false;
  }
  std::string prefix = store_dir;
  if (!prefix.empty() && prefix.back() == '/') prefix.pop_back();

  auto v2 = moe_store::ReadIndex(prefix);
  for (const auto& group : v2.groups) {
    for (const auto& member : group.members) {
      TensorStorageMeta meta;
      meta.file_id = group.file_id;
      meta.offset = group.offset + member.rel_offset;
      meta.size = static_cast<std::size_t>(member.size);
      meta.shape = member.shape;
      meta.options =
          torch::TensorOptions().dtype(DtypeTokenToScalarType(member.dtype));
      meta.id = member.tensor_id;
      index->emplace(member.tensor_id, std::move(meta));
    }
  }
  DLOG_INFO("Loaded moe-store v2 index: ", v2.groups.size(), " groups, ",
            index->size(), " tensors");
  return true;
}
