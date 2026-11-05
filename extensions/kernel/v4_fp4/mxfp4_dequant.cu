#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __constant__ float kMxfp4Values[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// BF16 bit patterns of the E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6};
// bit 15 carries the sign. Every value times a power of two is exact in BF16.
__device__ __forceinline__ uint32_t e2m1_bf16_bits(uint32_t code) {
  constexpr uint32_t kHi = 0x40404040u;      // high bytes of codes 4..7
  constexpr uint32_t kLo = 0x3F3F3F00u;      // high bytes of codes 0..3
  constexpr uint32_t kMantLo = 0xC0800000u;  // low bytes of codes 0..3
  constexpr uint32_t kMantHi = 0xC0804000u;  // low bytes of codes 4..7
  uint32_t mag = code & 7u;
  uint32_t hi = __byte_perm(kLo, kHi, mag) & 0xFFu;
  uint32_t lo = __byte_perm(kMantLo, kMantHi, mag) & 0xFFu;
  return (hi << 8) | lo | ((code & 8u) << 12);
}

// One thread per 32-value MX block: one 16-B load of packed codes, one scale
// byte, four 16-B stores. block_size == 32 is the MXFP4 layout.
__global__ void mxfp4_dequant_block32_kernel(const uint4* __restrict__ packed,
                                             const uint8_t* __restrict__ scales,
                                             uint4* __restrict__ output,
                                             long n_blocks) {
  for (long b = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       b < n_blocks; b += static_cast<long>(gridDim.x) * blockDim.x) {
    uint4 q = __ldg(packed + b);
    int e = static_cast<int>(__ldg(scales + b));
    e = max(1, min(254, e));  // == clamp(e - 127, -126, 127) + 127
    __nv_bfloat162 s2;
    s2.x = __ushort_as_bfloat16(static_cast<unsigned short>(e << 7));
    s2.y = s2.x;
    uint32_t words[4] = {q.x, q.y, q.z, q.w};
    uint4 out[4];
    uint32_t* o = reinterpret_cast<uint32_t*>(out);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        uint32_t byte = (words[w] >> (8 * p)) & 0xFFu;
        uint32_t v =
            e2m1_bf16_bits(byte & 0xFu) | (e2m1_bf16_bits(byte >> 4) << 16);
        __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&v);
        x = __hmul2(x, s2);
        o[w * 4 + p] = *reinterpret_cast<uint32_t*>(&x);
      }
    }
    uint4* dst = output + b * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = out[i];
  }
}

__global__ void mxfp4_dequant_kernel(const uint8_t* packed,
                                     const uint8_t* scales,
                                     __nv_bfloat16* output, int rows,
                                     int packed_cols, int scale_cols,
                                     int block_size) {
  long output_col = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  long output_cols = static_cast<long>(packed_cols) * 2;
  long total = static_cast<long>(rows) * output_cols;
  if (output_col >= total) return;

  int row = static_cast<int>(output_col / output_cols);
  int col = static_cast<int>(output_col % output_cols);
  uint8_t byte = packed[static_cast<long>(row) * packed_cols + col / 2];
  uint8_t code = (col & 1) == 0 ? (byte & 0x0F) : (byte >> 4);
  int exponent =
      static_cast<int>(
          scales[static_cast<long>(row) * scale_cols + col / block_size]) -
      127;
  exponent = max(-126, min(127, exponent));
  output[output_col] = __float2bfloat16(ldexpf(kMxfp4Values[code], exponent));
}

}  // namespace

void mxfp4_dequant_cuda(const void* packed, const void* scales, void* output,
                        int rows, int packed_cols, int scale_cols,
                        int block_size, cudaStream_t stream) {
  constexpr int threads = 256;
  bool aligned = (reinterpret_cast<uintptr_t>(packed) % 16 == 0) &&
                 (reinterpret_cast<uintptr_t>(output) % 16 == 0);
  if (block_size == 32 && scale_cols * 16 == packed_cols && aligned) {
    // Rows are contiguous MX blocks, so the tensor is one flat block array.
    long n_blocks = static_cast<long>(rows) * scale_cols;
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    long want = (n_blocks + threads - 1) / threads;
    int blocks = static_cast<int>(want < 8L * sms ? want : 8L * sms);
    mxfp4_dequant_block32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint4*>(packed), static_cast<const uint8_t*>(scales),
        static_cast<uint4*>(output), n_blocks);
    return;
  }
  long total = static_cast<long>(rows) * packed_cols * 2;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  mxfp4_dequant_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(packed), static_cast<const uint8_t*>(scales),
      static_cast<__nv_bfloat16*>(output), rows, packed_cols, scale_cols,
      block_size);
}
