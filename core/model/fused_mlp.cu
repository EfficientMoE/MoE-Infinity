#include "fused_mlp.h"
#include "parallel/expert_module.h"

torch::Tensor launch_fused_moe_ffn(torch::Tensor hidden,  // [M, K]
                                   torch::Tensor w1,      // [N, K]
                                   torch::Tensor w2,      // [N, K]
                                   torch::Tensor w3,      // [K, N]
                                   cudaStream_t stream)   // CUDA stream
{
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "BF16 only kernel");

  int device_id = at::cuda::current_device();

  int M = hidden.size(0);
  int K = hidden.size(1);
  int N = w1.size(0);

  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks((K + NUM_THREADS - 1) / NUM_THREADS,
              (M + NUM_THREADS - 1) / NUM_THREADS);

  auto options = hidden.options().dtype(at::kBFloat16).device(hidden.device());

  // // get torch cuda allocator
  // auto allocator = c10::cuda::CUDACachingAllocator::get();
  // TORCH_CHECK(allocator != nullptr, "CUDACachingAllocator is not
  // initialized");
  // // allocate memory using torch cuda allocator
  // void* output_ptr = allocator->raw_alloc_with_stream(
  //     M * K * sizeof(__nv_bfloat16), stream);
  void* output_ptr = c10::DeviceCachingAllocator::get(device_id)->allocate(
      M * K * sizeof(__nv_bfloat16));
  // cudaMalloc(&output_ptr, M * K * sizeof(__nv_bfloat16));

  // cudamalloc and create output tensor
  // torch::Tensor output = torch::empty({M, K}, options);
  // std::cout << "Output tensor sum: " << output.sum().item<float>() <<
  // std::endl; TORCH_CHECK(output.is_contiguous(), "Output tensor must be
  // contiguous"); TORCH_CHECK(w1.is_contiguous() && w2.is_contiguous() &&
  // w3.is_contiguous(), "Weight tensors must be contiguous");
  // TORCH_CHECK(hidden.is_contiguous(), "Hidden tensor must be contiguous");

  int shared_mem_bytes = TILE_DIM * TILE_DIM * sizeof(__nv_bfloat16);

  auto start = std::chrono::high_resolution_clock::now();
  fused_moe_ffn_kernel<<<blocks, threads, shared_mem_bytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(hidden.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(w1.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(w2.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(w3.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output_ptr), M, K, N);

  cudaStreamSynchronize(stream);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << "Kernel execution time: " << duration.count() << "
  // microseconds" << std::endl; cudaDeviceSynchronize();

  // torch::Tensor output = torch::from_blob(output_ptr, {M, K},
  // torch_deleter{}, options);

  // torch::Tensor output = torch::from_blob(output_ptr, {M, K}, options);
  // std::cout << "Output tensor sum: " << output.sum().item<float>() <<
  // std::endl;
  return torch::from_blob(output_ptr, {M, K}, torch_deleter{device_id},
                          options);
}
