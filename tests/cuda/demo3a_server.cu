#include <iostream>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cuda_runtime_api.h>
#include <chrono>

constexpr const char* SHM_NAME = "/my_shared_mem";
constexpr size_t SIZE = 20ULL * 1024 * 1024 * 1024;  // 40 GB
constexpr const char* MESSAGE = "Writer says hello :)";

int main() {
  size_t page_size = 4096;  // 4MB
  size_t aligned_bytes = ((SIZE + page_size - 1) / page_size) * page_size;

  void* ptr = aligned_alloc(page_size, aligned_bytes);
  if (ptr == nullptr) {
    std::cerr << "Failed to allocate aligned memory\n";
    return 1;
  }

  // Create and size shared memory object
  int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0640);
  if (shm_fd < 0) {
    perror("shm_open");
    free(ptr);
    return 1;
  }

  if (ftruncate(shm_fd, aligned_bytes) == -1) {
    perror("ftruncate");
    close(shm_fd);
    shm_unlink(SHM_NAME);
    free(ptr);
    return 1;
  }

  // Register with CUDA
  auto start = std::chrono::high_resolution_clock::now();
  cudaError_t cuda_err =
      cudaHostRegister(ptr, aligned_bytes, cudaHostRegisterDefault);
  if (cuda_err != cudaSuccess) {
    std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(cuda_err)
              << "\n";
    close(shm_fd);
    shm_unlink(SHM_NAME);
    return 1;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "cudaHostRegister took " << duration.count() * 1000 << " ms\n";

  // Map shared memory
  void* h_data = mmap(ptr, aligned_bytes, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_FIXED, shm_fd, 0);
  if (h_data == MAP_FAILED) {
    perror("mmap");
    close(shm_fd);
    shm_unlink(SHM_NAME);
    return 1;
  }

  std::cout << "ptr = " << ptr << std::endl;
  std::cout << "Mapped h_data = " << h_data << " (size: " << aligned_bytes
            << ")\n";

  // Verify registration
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, h_data);
  std::cout << "Memory type: " << attr.type << " (should be 1 for host)\n";

  // Warm up the memory (touch all pages)
  std::cout << "Warming up memory...\n";
  char* touch = static_cast<char*>(h_data);
  for (size_t i = 0; i < SIZE; i += page_size) {
    touch[i] = 0;
  }

  // Write to shared memory
  strcpy(static_cast<char*>(h_data), MESSAGE);
  std::cout << "Message written: " << MESSAGE << "\n";

  std::cout << "Press Enter to cleanup and exit...";
  std::cin.get();

  // Cleanup
  cudaHostUnregister(ptr);
  munmap(h_data, aligned_bytes);
  close(shm_fd);
  shm_unlink(SHM_NAME);

  return 0;
}
