/**
 * green_ctx.cpp  —  CUDA green-context setup helper
 *
 * Compiled as a shared library (green_ctx.so) and loaded from Python via
 * ctypes.  All CUDA driver API calls are made here against the real
 * <cuda.h> headers, eliminating the need for hand-maintained ctypes struct
 * layouts in Python.
 *
 * Exported API (extern "C"):
 *   int gc_setup_sm    (int device_id,
 *                       uint64_t *stream0_out, uint64_t *stream1_out,
 *                       int *sm0_out, int *sm1_out, int *total_out);
 *   int gc_setup_sm_wq (int device_id,
 *                       uint64_t *stream0_out, uint64_t *stream1_out,
 *                       int *sm0_out, int *sm1_out, int *total_out);
 *
 * Return value: 0 on success, non-zero CUDA error code on failure.
 * stream*_out receive CUstream handles cast to uint64_t; wrap with
 * torch.cuda.ExternalStream() on the Python side.
 *
 * Green context handles are kept in module-level statics so they outlive
 * the function call and the streams remain valid.
 */

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Module-level state — keeps green contexts alive for the lifetime of the SO.
// ---------------------------------------------------------------------------
static CUgreenCtx g_gctx[2] = {nullptr, nullptr};
static CUstream g_streams[2] = {nullptr, nullptr};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
#define CU_CHECK(expr)                                                \
  do {                                                                \
    CUresult _r = (expr);                                             \
    if (_r != CUDA_SUCCESS) {                                         \
      fprintf(stderr, "[green_ctx] %s failed: %d\n", #expr, (int)_r); \
      return (int)_r;                                                 \
    }                                                                 \
  } while (0)

static int init_device(int device_id, CUdevice* dev_out) {
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(dev_out, device_id));
  return 0;
}

// ---------------------------------------------------------------------------
// gc_setup_sm: SM-partitioned green contexts (CUDA >= 12.4)
// ---------------------------------------------------------------------------
extern "C" int gc_setup_sm(int device_id, uint64_t* stream0_out,
                           uint64_t* stream1_out, int* sm0_out, int* sm1_out,
                           int* total_out) {
  CUdevice dev;
  int rc = init_device(device_id, &dev);
  if (rc != 0) return rc;

  // ----- Get total SM resource -----
  CUdevResource sm_res;
  CU_CHECK(cuDeviceGetDevResource(dev, &sm_res, CU_DEV_RESOURCE_TYPE_SM));

  int total_sms = (int)sm_res.sm.smCount;
  if (total_out) *total_out = total_sms;

  // ----- Split into 2 equal partitions -----
  // minCount = total/2 ensures each group gets half the SMs.
  // minCount=1 would produce only smCoscheduledAlignment SMs/group (2 on
  // Ampere), wasting the rest.
  CUdevResource result_arr[2];
  unsigned int nb = 2;
  unsigned int min_count = (unsigned int)(total_sms / 2);

  CU_CHECK(cuDevSmResourceSplitByCount(result_arr, &nb, &sm_res, nullptr, 0,
                                       min_count));

  if (sm0_out) *sm0_out = (int)result_arr[0].sm.smCount;
  if (sm1_out) *sm1_out = (int)result_arr[1].sm.smCount;

  // ----- Generate descriptors -----
  CUdevResourceDesc desc0 = nullptr;
  CUdevResourceDesc desc1 = nullptr;
  CU_CHECK(cuDevResourceGenerateDesc(&desc0, &result_arr[0], 1));
  CU_CHECK(cuDevResourceGenerateDesc(&desc1, &result_arr[1], 1));

  // ----- Create green contexts -----
  CU_CHECK(
      cuGreenCtxCreate(&g_gctx[0], desc0, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CU_CHECK(
      cuGreenCtxCreate(&g_gctx[1], desc1, dev, CU_GREEN_CTX_DEFAULT_STREAM));

  // ----- Create streams (CU_STREAM_NON_BLOCKING is required) -----
  CU_CHECK(cuGreenCtxStreamCreate(&g_streams[0], g_gctx[0],
                                  CU_STREAM_NON_BLOCKING, 0));
  CU_CHECK(cuGreenCtxStreamCreate(&g_streams[1], g_gctx[1],
                                  CU_STREAM_NON_BLOCKING, 0));

  *stream0_out = (uint64_t)(uintptr_t)g_streams[0];
  *stream1_out = (uint64_t)(uintptr_t)g_streams[1];
  return 0;
}

// ---------------------------------------------------------------------------
// gc_setup_sm_wq: SM + work-queue partition (CUDA >= 13.1 / driver >= 575)
// ---------------------------------------------------------------------------
extern "C" int gc_setup_sm_wq(int device_id, uint64_t* stream0_out,
                              uint64_t* stream1_out, int* sm0_out, int* sm1_out,
                              int* total_out) {
  CUdevice dev;
  int rc = init_device(device_id, &dev);
  if (rc != 0) return rc;

  // ----- SM resource: split into 2 equal halves -----
  CUdevResource sm_res;
  CU_CHECK(cuDeviceGetDevResource(dev, &sm_res, CU_DEV_RESOURCE_TYPE_SM));

  int total_sms = (int)sm_res.sm.smCount;
  if (total_out) *total_out = total_sms;

  CUdevResource sm_parts[2];
  unsigned int nb = 2;
  unsigned int min_count = (unsigned int)(total_sms / 2);
  CU_CHECK(cuDevSmResourceSplitByCount(sm_parts, &nb, &sm_res, nullptr, 0,
                                       min_count));

  if (sm0_out) *sm0_out = (int)sm_parts[0].sm.smCount;
  if (sm1_out) *sm1_out = (int)sm_parts[1].sm.smCount;

  // ----- Work-queue configuration resource
  // (CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG) This is a configuration/scope hint,
  // not a splittable resource. Setting sharingScope =
  // CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED requests non-overlapping WQ
  // scheduling between the two green contexts.
  CUdevResource wq_res;
  CU_CHECK(cuDeviceGetDevResource(dev, &wq_res,
                                  CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));
  wq_res.wqConfig.sharingScope = CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED;

  // ----- Build [SM_partition, WQ_config] pairs per green context -----
  CUdevResource pair0[2] = {sm_parts[0], wq_res};
  CUdevResource pair1[2] = {sm_parts[1], wq_res};

  // ----- Generate descriptors -----
  CUdevResourceDesc desc0 = nullptr;
  CUdevResourceDesc desc1 = nullptr;
  CU_CHECK(cuDevResourceGenerateDesc(&desc0, pair0, 2));
  CU_CHECK(cuDevResourceGenerateDesc(&desc1, pair1, 2));

  // ----- Create green contexts -----
  CU_CHECK(
      cuGreenCtxCreate(&g_gctx[0], desc0, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CU_CHECK(
      cuGreenCtxCreate(&g_gctx[1], desc1, dev, CU_GREEN_CTX_DEFAULT_STREAM));

  // ----- Create streams -----
  CU_CHECK(cuGreenCtxStreamCreate(&g_streams[0], g_gctx[0],
                                  CU_STREAM_NON_BLOCKING, 0));
  CU_CHECK(cuGreenCtxStreamCreate(&g_streams[1], g_gctx[1],
                                  CU_STREAM_NON_BLOCKING, 0));

  *stream0_out = (uint64_t)(uintptr_t)g_streams[0];
  *stream1_out = (uint64_t)(uintptr_t)g_streams[1];
  return 0;
}
