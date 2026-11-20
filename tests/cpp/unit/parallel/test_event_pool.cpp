// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// CudaEventPool hands out events that belong to the requested device: an event
// is created with that device current and is recorded on streams of that
// device only (issue #362).

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include "memory/event_pool.h"

namespace {

int DeviceCount() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

int CurrentDevice() {
  int d = -1;
  cudaGetDevice(&d);
  return d;
}

cudaStream_t MakeStream(int device) {
  int previous = CurrentDevice();
  cudaSetDevice(device);
  cudaStream_t stream = nullptr;
  EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  cudaSetDevice(previous);
  return stream;
}

}  // namespace

TEST(CudaEventPool, AcquireRecordsOnStreamOfRequestedDevice) {
  if (DeviceCount() < 1) GTEST_SKIP() << "needs a CUDA device";
  CudaEventPool pool;
  cudaStream_t stream = MakeStream(0);
  cudaEvent_t ev = pool.Acquire(0);
  ASSERT_NE(ev, nullptr);
  EXPECT_EQ(pool.DeviceOf(ev), 0);
  EXPECT_EQ(cudaEventRecord(ev, stream), cudaSuccess);
  EXPECT_EQ(cudaEventSynchronize(ev), cudaSuccess);
  pool.Release(ev);
  cudaStreamDestroy(stream);
}

TEST(CudaEventPool, EventIsCreatedOnRequestedDeviceNotCurrentDevice) {
  if (DeviceCount() < 2) GTEST_SKIP() << "needs >= 2 CUDA devices";
  CudaEventPool pool;
  cudaStream_t stream1 = MakeStream(1);

  // The calling thread is on device 0; the event must still belong to 1.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaEvent_t ev = pool.Acquire(1);
  ASSERT_NE(ev, nullptr);
  EXPECT_EQ(CurrentDevice(), 0) << "Acquire must not leak a device switch";
  EXPECT_EQ(pool.DeviceOf(ev), 1);

  // Recording on a stream of device 1 works; with the old process-wide pool
  // (event created on the caller's device 0) this failed with
  // cudaErrorInvalidResourceHandle and a later wait did not wait.
  EXPECT_EQ(cudaEventRecord(ev, stream1), cudaSuccess);
  EXPECT_EQ(cudaEventSynchronize(ev), cudaSuccess);

  // Waiting on it from a stream of the other device is allowed.
  cudaStream_t stream0 = MakeStream(0);
  EXPECT_EQ(cudaStreamWaitEvent(stream0, ev, 0), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream0), cudaSuccess);

  pool.Release(ev);
  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
}

TEST(CudaEventPool, ReleaseReturnsEventToItsOwnDevicePool) {
  if (DeviceCount() < 2) GTEST_SKIP() << "needs >= 2 CUDA devices";
  CudaEventPool pool;
  cudaEvent_t on1 = pool.Acquire(1);
  pool.Release(on1);

  // Device 0 never receives device 1's event, whatever the call order.
  cudaEvent_t on0 = pool.Acquire(0);
  EXPECT_NE(on0, on1);
  EXPECT_EQ(pool.DeviceOf(on0), 0);

  // Device 1 gets its own recycled event back.
  cudaEvent_t again = pool.Acquire(1);
  EXPECT_EQ(again, on1);
  EXPECT_EQ(pool.DeviceOf(again), 1);

  pool.Release(on0);
  pool.Release(again);
}
