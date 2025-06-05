#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

__device__ __forceinline__ int load_lu(const int* ptr) {
  int val;
  asm volatile("ld.global.lu.s32 %0, [%1];" : "=r"(val) : "l"(ptr));
  return val;
}

__device__ __forceinline__ void store_wb(int* ptr, int val) {
  asm volatile("st.global.wb.s32 [%0], %1;" ::"l"(ptr), "r"(val));
}

__global__ __launch_bounds__(512) void readEvery64B_Uncached(
    const int* __restrict__ data, int N) {
  const int stride = 4096 / sizeof(int);  // 64-byte stride
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;

  unsigned int local = 0;
  if (idx < N) {
    local = load_lu(&data[idx]);
  }

  if (local == 0xDEADBEEF) {
    printf("unreachable: %u\n", local);
  }
}

__global__ __launch_bounds__(512) void copyAllUVM_Uncached(
    const int* __restrict__ src, int* __restrict__ dst, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    dst[idx] = load_lu(&src[idx]);
  }
}

__device__ __forceinline__ void copy_64B_uncached(const void* src, void* dst) {
  int4 reg0, reg1, reg2, reg3;

  // Load 64 bytes (4 x int4 = 4 x 16B) from uncached global memory
  asm volatile("ld.global.lu.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(reg0.x), "=r"(reg0.y), "=r"(reg0.z), "=r"(reg0.w)
               : "l"(src));
  asm volatile("ld.global.lu.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(reg1.x), "=r"(reg1.y), "=r"(reg1.z), "=r"(reg1.w)
               : "l"((const char*)src + 16));
  asm volatile("ld.global.lu.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(reg2.x), "=r"(reg2.y), "=r"(reg2.z), "=r"(reg2.w)
               : "l"((const char*)src + 32));
  asm volatile("ld.global.lu.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(reg3.x), "=r"(reg3.y), "=r"(reg3.z), "=r"(reg3.w)
               : "l"((const char*)src + 48));

  // Store 64 bytes with write-through to global memory (bypass L1)
  asm volatile("st.global.wb.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(dst), "r"(reg0.x), "r"(reg0.y), "r"(reg0.z), "r"(reg0.w));
  asm volatile("st.global.wb.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"((char*)dst + 16), "r"(reg1.x), "r"(reg1.y), "r"(reg1.z),
                 "r"(reg1.w));
  asm volatile("st.global.wb.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"((char*)dst + 32), "r"(reg2.x), "r"(reg2.y), "r"(reg2.z),
                 "r"(reg2.w));
  asm volatile("st.global.wb.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"((char*)dst + 48), "r"(reg3.x), "r"(reg3.y), "r"(reg3.z),
                 "r"(reg3.w));
}

__global__ __launch_bounds__(512) void copyEvery4KB_Uncached(
    const int* __restrict__ src, int* __restrict__ dst, int N) {
  const int stride = 4096 / sizeof(int);  // 4KB = 1024 int32
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;

  if (idx < N) {
    dst[idx] = load_lu(&src[idx]);  // uncached load
                                    // for (int i = 0; i < stride; ++i) {
                                    //     if (idx + i < N) {
                                    //         int val = load_lu(&src[idx+i]);
                                    //         store_wb(&dst[idx+i], val);
                                    //     }
                                    // }
  }
}

int main() {
  const int N = 1 << 26;                      // 64M ints = 256MB
  const int strideInts = 4096 / sizeof(int);  // 4KB stride

  cudaSetDevice(0);

  // ✅ Allocate Unified Memory
  int* uvm_src = nullptr;
  int* dev_dst = nullptr;
  cudaMallocManaged(&uvm_src, N * sizeof(int));
  cudaMalloc(&dev_dst, N * sizeof(int));  // plain device memory

  // ✅ Touch source on host to keep pages resident in CPU memory initially
  for (int i = 0; i < N; ++i) uvm_src[i] = i;

  cudaDeviceSynchronize();  // ensure all CPU writes are visible

  // ⛔ Do NOT prefetch — we want fault-triggered migration

  // Launch read kernel (will trigger page migrations)
  auto t_start = std::chrono::high_resolution_clock::now();

  int threads = 512;
  int blocks = (N + threads - 1) / threads;
  copyAllUVM_Uncached<<<blocks, threads>>>(uvm_src, dev_dst, N);
  cudaDeviceSynchronize();

  auto t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_sec = t_end - t_start;

  // ✅ Validate correctness (check values at stride points)
  int* host_check = (int*)malloc(N * sizeof(int));
  cudaMemcpy(host_check, dev_dst, N * sizeof(int), cudaMemcpyDeviceToHost);

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (host_check[i] != i) {
      if (++errors < 10) {
        std::cerr << "Mismatch at index " << i << ": expected " << i << ", got "
                  << host_check[i] << "\n";
      }
    }
  }

  if (errors == 0)
    std::cout << "✅ UVM migration correctness passed.\n";
  else
    std::cout << "❌ UVM migration correctness failed. Errors: " << errors
              << "\n";

  // ✅ Print timing
  double gb = N * sizeof(int) / double(1 << 30);
  std::cout << "Time: " << elapsed_sec.count() * 1000.0 << " ms\n";
  std::cout << "Throughput: " << gb / elapsed_sec.count() << " GB/s\n";

  // Cleanup
  free(host_check);
  cudaFree(uvm_src);
  cudaFree(dev_dst);

  return 0;
}

// int main() {
//     const int N = 1 << 26; // 64M ints = 256MB
//     const int chunkSize = 1 << 24; // 16M ints = 64MB
//     const int numChunks = N / chunkSize;

//     // Allocate host memory with malloc and register as pinned
//     int* host_data = (int*)aligned_alloc(4096, N * sizeof(int));
//     if (!host_data) {
//         std::cerr << "Host malloc failed.\n";
//         return -1;
//     }

//     // Fill with data
//     for (int i = 0; i < N; ++i) {
//         host_data[i] = i;
//     }

//     // Register memory as pinned and mapped
//     cudaHostRegister(host_data, N * sizeof(int), cudaHostRegisterMapped);

//     // Get device-accessible pointer
//     int* device_ptr = nullptr;
//     cudaHostGetDevicePointer(&device_ptr, host_data, 0);

//     // Create compute stream
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     // Timing setup
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     for (int i = 0; i < numChunks; ++i) {
//         int* chunk_ptr = device_ptr + i * chunkSize;

//         int stride = 4096 / sizeof(int);
//         int effectiveElems = chunkSize / stride;
//         int threads = 512;
//         int blocks = (effectiveElems + threads - 1) / threads;

//         readEvery64B_Uncached<<<blocks, threads, 0, stream>>>(chunk_ptr,
//         chunkSize);
//     }

//     cudaEventRecord(stop, stream);
//     cudaEventSynchronize(stop);

//     float ms = 0;
//     cudaEventElapsedTime(&ms, start, stop);
//     double gb = N * sizeof(int) / double(1 << 30);
//     std::cout << "Pinned Host Mem Read Throughput: " << gb / (ms / 1000.0) <<
//     " GB/s\n";

//     // Cleanup
//     cudaHostUnregister(host_data);
//     free(host_data);
//     cudaStreamDestroy(stream);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     return 0;
// }
