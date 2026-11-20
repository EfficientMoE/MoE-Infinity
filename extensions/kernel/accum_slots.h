// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Kernels of the dispatcher's order-deterministic accumulation ("slot mode",
// see ExpertDispatcher::SetDeterministicAccum). One launch each:
//
//   accum_slots_prepare  per dispatch (caller stream, before the input-ready
//                        event): the rank table, the overflow counter and the
//                        zero fill of the slot buffer, replacing a zero fill, a
//                        cumsum chain and a counter update of about ten small
//                        torch ops.
//   accum_slot_store     per expert (exec stream): weighted output of the
//                        expert -> its (token, rank) slots (or the dump row).
//   accum_slots_reduce   per dispatch (caller stream, after the expert
//                        completion events): the fixed-order sum of the slots
//                        into the accumulator (overwrites it).
//
// Layout: slots is float32 [T, rows, H]. rows == R, or R + 1 when ranks >= R
// are possible (a top_k hint smaller than num_experts): row R is the dump row
// that collects those contributions with atomics (correct, not deterministic).
// slot_index is int32 [T, E] = min(rank of expert e among token t's active
// experts in ascending expert index, R). Slots nobody wrote are zero (the
// prepare launch zero-fills the buffer), and adding a zero changes no value, so
// the reduction simply sums all R rows.

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

// router_mask [T, E] bool/uint8 (any strides). Writes slot_index,
// zero-fills the whole slot buffer, and adds the number of tokens with more
// than R active experts to *overflow_counter when it is defined (int64 [1]).
void accum_slots_prepare(const torch::Tensor& router_mask, torch::Tensor& slots,
                         torch::Tensor& slot_index, int64_t slot_count,
                         const torch::Tensor& overflow_counter,
                         cudaStream_t stream);

// output [n, H] float32/bfloat16/float16 contiguous (rows of the tokens in
// token_indices, int64 [n]); router_weight [T, E] float32/bfloat16/float16
// (any strides). Stores float(output) * float(weight) into
// slots[t, slot_index[t, expert], :] (plain store) or, for slot_index ==
// slot_count, adds it atomically to the dump row slots[t, slot_count, :].
void accum_slot_store(const torch::Tensor& output,
                      const torch::Tensor& token_indices, int64_t expert,
                      const torch::Tensor& router_weight,
                      const torch::Tensor& slot_index, torch::Tensor& slots,
                      int64_t slot_count, cudaStream_t stream);

// out[t, :] = dump[t, :] (when has_dump_row) + slots[t, 0, :] + slots[t, 1, :]
// + ... + slots[t, slot_count - 1, :], added in that order in float32, for
// out float32 [T, H] with stride(1) == 1 (it is overwritten, not accumulated).
void accum_slots_reduce(const torch::Tensor& slots, int64_t slot_count,
                        bool has_dump_row, torch::Tensor& out,
                        cudaStream_t stream);
