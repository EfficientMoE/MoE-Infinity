#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

// using namespace nvcuda;
namespace cg = cooperative_groups;

// Fused kernel: count + collect indices + extract tokens in one pass
__global__ void fused_extract_expert_tokens_bf16(
    const __nv_bfloat16* __restrict__ hidden_states,
    const bool* __restrict__ router_mask, __nv_bfloat16* __restrict__ output,
    int* __restrict__ output_count, const int num_tokens, const int hidden_dim,
    const int expert_idx, const int num_experts) {
  // Use cooperative groups for better warp-level primitives
  auto g = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(g);

  // Shared memory for warp-level scan
  extern __shared__ int shared_data[];
  int* warp_counts = shared_data;
  int* block_offset = &shared_data[32];

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;

  // Initialize shared memory
  if (threadIdx.x == 0) *block_offset = 0;
  if (lane_id == 0) warp_counts[warp_id] = 0;
  __syncthreads();

  // Process tokens in chunks for better memory access
  const int tokens_per_thread =
      (num_tokens + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
  const int start_token =
      (blockIdx.x * blockDim.x + threadIdx.x) * tokens_per_thread;
  const int end_token = min(start_token + tokens_per_thread, num_tokens);

  // Phase 1: Count and mark tokens
  int local_count = 0;
#pragma unroll 4
  for (int token_idx = start_token; token_idx < end_token; token_idx++) {
    if (router_mask[token_idx * num_experts + expert_idx]) {
      local_count++;
    }
  }

// Warp-level reduction
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_count += __shfl_down_sync(0xffffffff, local_count, offset);
  }

  // Store warp count
  if (lane_id == 0) {
    warp_counts[warp_id] = local_count;
  }
  __syncthreads();

  // Block-level scan to get offsets
  if (threadIdx.x < warps_per_block) {
    int val = warp_counts[threadIdx.x];
#pragma unroll
    for (int i = 1; i < warps_per_block; i <<= 1) {
      int n = __shfl_up_sync(0xffffffff, val, i);
      if (threadIdx.x >= i) val += n;
    }
    warp_counts[threadIdx.x] = val;
  }
  __syncthreads();

  // Get block's starting position
  int block_start = 0;
  if (threadIdx.x == warps_per_block - 1) {
    block_start = atomicAdd(output_count, warp_counts[warps_per_block - 1]);
    *block_offset = block_start;
  }
  __syncthreads();

  // Calculate this thread's output offset
  int thread_offset = *block_offset;
  if (warp_id > 0) {
    thread_offset += warp_counts[warp_id - 1];
  }

  // Warp-level scan for lane offsets
  int lane_offset = 0;
  int lane_count = 0;
#pragma unroll 4
  for (int token_idx = start_token; token_idx < end_token; token_idx++) {
    if (router_mask[token_idx * num_experts + expert_idx]) {
      lane_count++;
    }
  }

#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    int n = __shfl_up_sync(0xffffffff, lane_count, i);
    if (lane_id >= i) lane_offset += n;
  }

  thread_offset += lane_offset;

  // Phase 2: Extract tokens with coalesced writes
  // Use vector loads/stores for better bandwidth utilization
  const int vec_size = 8;  // Process 8 bf16 elements at once
  using vec_t = float4;    // 8 bf16s = 4 floats = 128 bits

#pragma unroll 4
  for (int token_idx = start_token; token_idx < end_token; token_idx++) {
    if (router_mask[token_idx * num_experts + expert_idx]) {
      const int src_offset = token_idx * hidden_dim;
      const int dst_offset = thread_offset * hidden_dim;

      // Vectorized copy
      const vec_t* src_vec =
          reinterpret_cast<const vec_t*>(&hidden_states[src_offset]);
      vec_t* dst_vec = reinterpret_cast<vec_t*>(&output[dst_offset]);

// Copy in chunks of 8 bf16 elements
#pragma unroll 4
      for (int i = 0; i < hidden_dim / vec_size; i++) {
        dst_vec[i] = src_vec[i];
      }

      // Handle remainder
      const int remainder = hidden_dim % vec_size;
      if (remainder > 0) {
        const int base_idx = (hidden_dim / vec_size) * vec_size;
#pragma unroll
        for (int i = 0; i < remainder; i++) {
          output[dst_offset + base_idx + i] =
              hidden_states[src_offset + base_idx + i];
        }
      }

      thread_offset++;
    }
  }
}

// // Optimized kernel for batch_size > 1 case
// void extract_expert_tokens_fused_cuda(torch::Tensor hidden_states,
//                                       torch::Tensor router_mask,
//                                       torch::Tensor output,
//                                       torch::Tensor output_count,
//                                       int expert_idx, int batch_size) {
//   // Skip if batch_size == 1
//   if (batch_size == 1) {
//     output.copy_(hidden_states);
//     return;
//   }

//   const int num_tokens = hidden_states.size(0);
//   const int hidden_dim = hidden_states.size(1);
//   const int num_experts = router_mask.size(1);

//   // Reset output count
//   cudaMemset(output_count.data_ptr<int>(), 0, sizeof(int));

//   // Configure kernel launch
//   const int threads = 256;
//   const int warps_per_block = threads / 32;
//   const int blocks = min(65535, (num_tokens + threads - 1) / threads);
//   const int smem_size = sizeof(int) * (warps_per_block + 1);

//   // Launch fused kernel
//   fused_extract_expert_tokens_bf16<<<blocks, threads, smem_size>>>(
//       reinterpret_cast<const __nv_bfloat16*>(
//           hidden_states.data_ptr<at::BFloat16>()),
//       router_mask.data_ptr<bool>(),
//       reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
//       output_count.data_ptr<int>(), num_tokens, hidden_dim, expert_idx,
//       num_experts);
// }

// Alternative simpler version without cooperative groups
__global__ void fused_extract_expert_tokens_bf16_simple(
    const __nv_bfloat16* __restrict__ hidden_states,
    const bool* __restrict__ router_mask, __nv_bfloat16* __restrict__ output,
    int* __restrict__ output_count, const int num_tokens, const int hidden_dim,
    const int expert_idx, const int num_experts) {
  // Shared memory for block-level reduction
  extern __shared__ char shared_mem[];
  __shared__ int num_idices = 0;
  int* block_idx = reinterpret_cast<int*>(shared_mem);

  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  if (global_tid < num_tokens) {
    if (router_mask[global_tid * num_experts + expert_idx]) {
      // Increment count for this token
      int count = atomicAdd(&num_idices, 1);
      block_idx[count] = global_tid;  // Store index of selected token
    }
  }
  __syncthreads();

  // Initialize shared memory
  if (tid == 0) {
    block_count[0] = 0;
  }
  __syncthreads();

  // Phase 1: Count selected tokens for this thread
  int local_count = 0;
  for (int token_idx = global_tid; token_idx < num_tokens;
       token_idx += grid_size) {
    if (router_mask[token_idx * num_experts + expert_idx]) {
      local_count++;
    }
  }

  // Phase 2: Get block offset using atomic
  __shared__ int block_offset;
  if (local_count > 0) {
    atomicAdd(&block_count[0], local_count);
  }
  __syncthreads();

  if (tid == 0 && block_count[0] > 0) {
    block_offset = atomicAdd(output_count, block_count[0]);
  }
  __syncthreads();

  // Phase 3: Each thread writes its tokens independently
  if (local_count > 0) {
    // Get thread's offset within block
    int thread_offset = atomicAdd(&block_count[0], local_count) - local_count;
    int write_idx = block_offset + thread_offset;

    // Write tokens
    for (int token_idx = global_tid; token_idx < num_tokens;
         token_idx += grid_size) {
      if (router_mask[token_idx * num_experts + expert_idx]) {
        // Copy token data
        for (int i = 0; i < hidden_dim; i++) {
          output[write_idx * hidden_dim + i] =
              hidden_states[token_idx * hidden_dim + i];
        }
        write_idx++;
      }
    }
  }
}

// Host wrapper function
void extract_expert_tokens_fused_cuda(torch::Tensor hidden_states,
                                      torch::Tensor router_mask,
                                      torch::Tensor output,
                                      torch::Tensor output_count,
                                      int expert_idx, int batch_size) {
  // Skip if batch_size == 1
  if (batch_size == 1) {
    output.copy_(hidden_states);
    return;
  }

  const int num_tokens = hidden_states.size(0);
  const int hidden_dim = hidden_states.size(1);
  const int num_experts = router_mask.size(1);

  // Reset output count
  cudaMemset(output_count.data_ptr<int>(), 0, sizeof(int));

  // Configure kernel launch
  const int threads = 256;
  const int blocks = min(65535, (num_tokens + threads - 1) / threads);
  const int smem_size = sizeof(int) * threads;

  // Launch the simpler kernel (more robust)
  fused_extract_expert_tokens_bf16_simple<<<blocks, threads, smem_size>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          hidden_states.data_ptr<at::BFloat16>()),
      router_mask.data_ptr<bool>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      output_count.data_ptr<int>(), num_tokens, hidden_dim, expert_idx,
      num_experts);
}

// #include <cutlass/cutlass.h>
// #include <cutlass/tensor_ref.h>
// #include <cutlass/layout/matrix.h>
// #include <cutlass/fast_math.h>
// #include <cutlass/gemm/warp/mma.h>
// #include <cutlass/transform/threadblock/predicated_tile_iterator.h>
// #include <cutlass/transform/threadblock/regular_tile_iterator.h>
// #include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
// #include <cutlass/util/device_memory.h>
// #include <cub/cub.cuh>
// #include <torch/extension.h>

// using namespace cutlass;

// // Simplified kernel using CUTLASS iterators and CUB for scan operations
// template <int kThreads, int kElementsPerThread>
// __global__ void extract_expert_tokens_cutlass(
//     TensorRef<bfloat16_t, layout::RowMajor> hidden_states,
//     TensorRef<bool, layout::RowMajor> router_mask,
//     TensorRef<bfloat16_t, layout::RowMajor> output, int* output_count,
//     int expert_idx) {
//   using BlockScan = cub::BlockScan<int, kThreads>;
//   __shared__ typename BlockScan::TempStorage temp_storage;
//   __shared__ int block_offset;

//   const int num_tokens = hidden_states.extent(0);
//   const int hidden_dim = hidden_states.extent(1);

//   // Phase 1: Efficient counting with CUB
//   int thread_data[kElementsPerThread];
//   int thread_count = 0;

// #pragma unroll
//   for (int i = 0; i < kElementsPerThread; ++i) {
//     int token_idx = blockIdx.x * kThreads * kElementsPerThread +
//                     threadIdx.x * kElementsPerThread + i;

//     bool selected = false;
//     if (token_idx < num_tokens) {
//       selected = router_mask.at({token_idx, expert_idx});
//     }
//     thread_data[i] = selected ? 1 : 0;
//     thread_count += thread_data[i];
//   }

//   // Block-wide exclusive scan
//   int thread_offset;
//   int block_total;
//   BlockScan(temp_storage)
//       .ExclusiveSum(thread_count, thread_offset, block_total);

//   // Get global offset
//   if (threadIdx.x == 0) {
//     block_offset = atomicAdd(output_count, block_total);
//   }
//   __syncthreads();

//   thread_offset += block_offset;

//   // Phase 2: Extract using CUTLASS iterators for optimal memory access
//   using ThreadMap =
//       layout::PitchLinearThreadMap layout::PitchLinearShape<hidden_dim, 1>,
//         kThreads, layout::PitchLinearShape<8, 1>,  // 8 elements per access
//       1 > ;

//   using Iterator = transform::threadblock::PredicatedTileIterator
//       layout::PitchLinearShape<hidden_dim, 1>,
//         bfloat16_t, layout::RowMajor, 1, ThreadMap > ;

//   // Process selected tokens
//   int local_offset = 0;
// #pragma unroll
//   for (int i = 0; i < kElementsPerThread; ++i) {
//     if (thread_data[i]) {
//       int token_idx = blockIdx.x * kThreads * kElementsPerThread +
//                       threadIdx.x * kElementsPerThread + i;

//       // Use CUTLASS iterator for coalesced copy
//       Iterator src_iterator(hidden_states.data() + token_idx * hidden_dim,
//                             {hidden_dim, 1}, threadIdx.x);

//       Iterator dst_iterator(
//           output.data() + (thread_offset + local_offset) * hidden_dim,
//           {hidden_dim, 1}, threadIdx.x);

//       // Vectorized copy using CUTLASS fragments
//       CUTLASS_PRAGMA_UNROLL
//       for (int j = 0; j < Iterator::kIterations; ++j) {
//         typename Iterator::Fragment fragment;
//         src_iterator.load(fragment);
//         dst_iterator.store(fragment);
//         ++src_iterator;
//         ++dst_iterator;
//       }

//       local_offset++;
//     }
//   }
// }

// // Even simpler version using CUTLASS's DeviceSelect
// void extract_expert_tokens_cutlass_v2(torch::Tensor hidden_states,
//                                       torch::Tensor router_mask,
//                                       torch::Tensor output,
//                                       torch::Tensor output_count,
//                                       int expert_idx, int batch_size) {
//   if (batch_size == 1) {
//     output.copy_(hidden_states);
//     return;
//   }

//   const int num_tokens = hidden_states.size(0);
//   const int hidden_dim = hidden_states.size(1);

//   // Create CUTLASS tensor refs
//   TensorRef<bfloat16_t, layout::RowMajor> hidden_ref(
//       reinterpret_cast<bfloat16_t*>(hidden_states.data_ptr<at::BFloat16>()),
//       layout::RowMajor(hidden_dim));

//   TensorRef<bool, layout::RowMajor> mask_ref(
//       router_mask.data_ptr<bool>(), layout::RowMajor(router_mask.size(1)));

//   TensorRef<bfloat16_t, layout::RowMajor> output_ref(
//       reinterpret_cast<bfloat16_t*>(output.data_ptr<at::BFloat16>()),
//       layout::RowMajor(hidden_dim));

//   // Reset count
//   cudaMemset(output_count.data_ptr<int>(), 0, sizeof(int));

//   // Launch optimized kernel
//   const int kThreads = 128;
//   const int kElementsPerThread = 4;
//   const int blocks = (num_tokens + kThreads * kElementsPerThread - 1) /
//                      (kThreads * kElementsPerThread);

//   extract_expert_tokens_cutlass<kThreads, kElementsPerThread>
//       <<<blocks, kThreads>>>(hidden_ref, mask_ref, output_ref,
//                              output_count.data_ptr<int>(), expert_idx);
// }

// // Alternative: Use CUB DeviceSelect directly for maximum simplicity
// void extract_expert_tokens_cub(torch::Tensor hidden_states,
//                                torch::Tensor router_mask, torch::Tensor
//                                output, torch::Tensor output_count, int
//                                expert_idx, int batch_size) {
//   if (batch_size == 1) {
//     output.copy_(hidden_states);
//     return;
//   }

//   const int num_tokens = hidden_states.size(0);
//   const int hidden_dim = hidden_states.size(1);

//   // Create index array
//   auto indices = torch::arange(
//       num_tokens,
//       torch::dtype(torch::kInt32).device(hidden_states.device()));

//   // Get mask column for this expert
//   auto expert_mask = router_mask.index({"...", expert_idx});

//   // Allocate temporary storage for CUB
//   size_t temp_storage_bytes = 0;
//   cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes,
//                              indices.data_ptr<int>(),
//                              expert_mask.data_ptr<bool>(),
//                              indices.data_ptr<int>(),  // reuse for output
//                              output_count.data_ptr<int>(), num_tokens);

//   auto temp_storage =
//       torch::empty(temp_storage_bytes,
//                    torch::dtype(torch::kUInt8).device(hidden_states.device()));

//   // Select indices
//   cub::DeviceSelect::Flagged(
//       temp_storage.data_ptr(), temp_storage_bytes, indices.data_ptr<int>(),
//       expert_mask.data_ptr<bool>(), indices.data_ptr<int>(),
//       output_count.data_ptr<int>(), num_tokens);

//   // Get selected count
//   int num_selected;
//   cudaMemcpy(&num_selected, output_count.data_ptr<int>(), sizeof(int),
//              cudaMemcpyDeviceToHost);

//   // Copy selected tokens using CUTLASS batched copy
//   if (num_selected > 0) {
//     // Simple kernel to copy using selected indices
//     auto copy_kernel = [=] __device__(int idx) {
//       if (idx < num_selected) {
//         int src_idx = indices.data_ptr<int>()[idx];

//         // Use CUTLASS aligned memory copy
//         using CopyOp =
//             cutlass::AlignedCopy sizeof(float4),  // 128-bit alignment
//             layout::RowMajor > ;

//         CopyOp copy_op;
//         for (int i = 0; i < hidden_dim; i += 8) {
//           copy_op(output.data_ptr<at::BFloat16>() + idx * hidden_dim + i,
//                   hidden_states.data_ptr<at::BFloat16>() +
//                       src_idx * hidden_dim + i);
//         }
//       }
//     };

//     // Launch copy kernel
//     int threads = 256;
//     int blocks = (num_selected + threads - 1) / threads;
//     copy_kernel<<<blocks, threads>>>(num_selected);
//   }
// }
