#pragma once

#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>  // for memset
#include <memory>
#include <string>

#include "common/types.h"
#include "utils/logger.h"

struct ShmMeta {
  int id;
  void* ptr;
  size_t size;
  std::string name;
  bool is_remote = false;
};

void* OpenSharedMemory(const char* name, size_t size);
void CloseSharedMemory(void* ptr, size_t size);
std::tuple<void*, int> AttachSharedMemory(const char* name, size_t size);
void DetachSharedMemory(void* ptr, int fd, size_t size);

struct ShmDeleter {
  void operator()(void* ptr) const {
    LOG_DEBUG << "ShmDeleter: ptr: " << ptr;
    DetachSharedMemory(ptr, fd, size);
  }
  size_t size;
  int fd;
};

template <typename T>
std::shared_ptr<T> AttachSharedMemoryPtr(const char* name, size_t size) {
  auto [ptr, shm_fd] = AttachSharedMemory(name, size);
  return std::shared_ptr<T>(static_cast<T*>(ptr), ShmDeleter{size, shm_fd});
}
