// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <string>

class ArcherTensorIndex;

// Populates the engine's tensor index from a moe-store v2 store
// (store_index + store_data_<N>); returns false when the directory holds
// no v2 index. Member entries become per-tensor metadata with absolute
// file offsets (group offset + member rel_offset), so every existing
// engine read path works unchanged and group members satisfy the
// contiguous bulk-read fast path by construction (4KiB member alignment).
bool LoadV2IndexInto(const std::string& store_dir, ArcherTensorIndex* index);

bool HasV2Index(const std::string& store_dir);
