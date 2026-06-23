// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "event_pool.h"

std::unique_ptr<CudaEventPool> kCudaEventPool =
    std::make_unique<CudaEventPool>();
