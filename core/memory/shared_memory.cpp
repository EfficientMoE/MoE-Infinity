#include "shared_memory.h"

void* OpenSharedMemory(const char* name, size_t size) {
  int shm_fd = shm_open(name, O_RDWR, 0666);
  LOG_FATAL_IF(shm_fd == -1,
               "shm_open failed. name: {}, size: {}; errno: {}, message: {}",
               name, size, errno, strerror(errno));

  void* ptr =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  LOG_FATAL_IF(ptr == MAP_FAILED,
               "mmap failed. name: {}, size: {}; errno: {}, message: {}", name,
               size, errno, strerror(errno));
  return ptr;
}

void CloseSharedMemory(void* ptr, size_t size) {
  int ret = munmap(ptr, size);
  LOG_FATAL_IF(ret == -1,
               "munmap failed. ptr: {0:x}, size: {}; errno: {}, message: {}",
               ptr, size, errno, strerror(errno));
}

std::tuple<void*, int> AttachSharedMemory(const char* name, size_t size) {
  int shm_fd = shm_open(name, O_RDWR, 0666);
  LOG_FATAL_IF(shm_fd == -1,
               "shm_open failed. name: {}, size: {}; errno: {}, message: {}",
               name, size, errno, strerror(errno));

  void* ptr =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  LOG_FATAL_IF(ptr == MAP_FAILED,
               "mmap failed. name: {}, size: {}; errno: {}, message: {}", name,
               size, errno, strerror(errno));
  return {ptr, shm_fd};
}

void DetachSharedMemory(void* ptr, int fd, size_t size) {
  int ret = munmap(ptr, size);
  LOG_FATAL_IF(ret == -1,
               "munmap failed. ptr: {0:x}, size: {}; errno: {}, message: {}",
               ptr, size, errno, strerror(errno));

  ret = close(fd);
  LOG_FATAL_IF(ret == -1, "close failed. fd: {}, errno: {}, message: {}", fd,
               errno, strerror(errno));
}
