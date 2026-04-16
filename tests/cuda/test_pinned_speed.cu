#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>

const int chunkCount = 4;
const size_t chunkSize = (1 << 20) * 4;  // 4 MB

std::queue<float*> bufferQueue;
std::mutex queueMutex;
std::condition_variable bufferReady;
bool doneAllocating = false;

std::vector<float*> pinnedBuffers;
std::mutex pinnedBuffersMutex;

// Thread A: malloc + enqueue
void allocatorThread() {
  for (int i = 0; i < chunkCount; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    float* ptr = (float*)malloc(chunkSize);
    if (!ptr) {
      std::cerr << "malloc failed on chunk " << i << std::endl;
      continue;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "[Allocator] Chunk " << i
              << " malloc time: " << duration.count() << " ms\n";

    {
      std::lock_guard<std::mutex> lock(queueMutex);
      bufferQueue.push(ptr);
    }
    bufferReady.notify_one();
  }

  {
    std::lock_guard<std::mutex> lock(queueMutex);
    doneAllocating = true;
  }
  bufferReady.notify_one();
}

// Thread B: pinning registered memory
void pinningThread() {
  int chunk = 0;

  while (true) {
    float* ptr = nullptr;

    {
      std::unique_lock<std::mutex> lock(queueMutex);
      bufferReady.wait(lock,
                       [] { return !bufferQueue.empty() || doneAllocating; });

      if (!bufferQueue.empty()) {
        ptr = bufferQueue.front();
        bufferQueue.pop();
      } else if (doneAllocating) {
        break;
      }
    }

    if (ptr) {
      auto start = std::chrono::high_resolution_clock::now();

      cudaError_t err =
          cudaHostRegister(ptr, chunkSize, cudaHostRegisterDefault);
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> duration = end - start;

      if (err != cudaSuccess) {
        std::cerr << "[Pinner] cudaHostRegister failed: "
                  << cudaGetErrorString(err) << std::endl;
        free(ptr);
      } else {
        std::cout << "[Pinner] Chunk " << chunk
                  << " pin time: " << duration.count() << " ms\n";
        std::lock_guard<std::mutex> lock(pinnedBuffersMutex);
        pinnedBuffers.push_back(ptr);
      }

      chunk++;
    }
  }
}

int main() {
  auto totalStart = std::chrono::high_resolution_clock::now();

  std::thread t1(allocatorThread);
  std::thread t2(pinningThread);

  t1.join();
  t2.join();

  auto totalEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> totalDuration =
      totalEnd - totalStart;

  std::cout << "\nAll buffers allocated and pinned.\n";
  std::cout << "Total time: " << totalDuration.count() << " ms\n";

  // Cleanup
  for (float* ptr : pinnedBuffers) {
    cudaHostUnregister(ptr);
    free(ptr);
  }

  return 0;
}
