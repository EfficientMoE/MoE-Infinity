#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __constant__ float kMxfp4Values[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

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
  long total = static_cast<long>(rows) * packed_cols * 2;
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  mxfp4_dequant_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(packed), static_cast<const uint8_t*>(scales),
      static_cast<__nv_bfloat16*>(output), rows, packed_cols, scale_cols,
      block_size);
}
