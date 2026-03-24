# C++ Interface Requirements: KV Cache Transport for MoE-Infinity

> Status: SPECIFICATION (not yet implemented)
> Authors: MoE-Infinity Team
> Related: `moe_infinity/memory/kv_cache_manager.py`, `moe_infinity/runtime/attention_backend.py`

## Overview

MoE-Infinity's Python layer currently manages KV cache through HuggingFace `DynamicCache` on GPU. To enable KV cache offloading to a CPU memory tier, the system needs a C++ extension that provides all of the following:

1. Preallocated pinned memory pools for KV cache blocks
2. Asynchronous H2D and D2H transfer using CUDA streams
3. Python bindings for block level allocation and swap operations that match the existing `KVCacheManager` and `BlockPool` abstractions

This document specifies the required C++ interface. The Python `KVCacheManager` and `BlockPool` classes in `moe_infinity/memory/kv_cache_manager.py` will call these methods once implemented.

At a high level, the design goal is:

- Keep all scheduling and block bookkeeping in Python, inside `KVCacheManager` and `AttentionBackend`
- Keep all bulk data movement and pinned memory management in C++ and CUDA
- Reuse the patterns and constraints already established by `PinnedMemoryPool`, `StreamPool`, and `ArcherPrefetchHandle`

## Current C++ Architecture

The existing `moe_infinity._store` extension, built from `core/python/py_archer_prefetch.cpp`, provides `ArcherPrefetchHandle` for expert weight management:

```text
Python: archer_engine.offload(tensor) / archer_engine.begin(id, tensor)
           ↓ pybind11
C++: ArcherPrefetchHandle::OffloadTensor / AcquireTensor
           ↓
Memory: PinnedMemoryPool (core/memory/pinned_memory_pool.cpp)
           ↓ cudaMemcpyAsync
Storage: SSD (via libaio) or CPU DRAM
```

Key constraint: `PinnedMemoryPool` uses `posix_memalign(..., 4096, chunk_size)` with `cudaHostRegister` for pinning. Block size is fixed at pool construction time. The same pool design is suitable for KV cache blocks.

Relevant components:

- `core/memory/pinned_memory_pool.cpp` and `.h` implement a fixed size, chunk based pinned pool with `Acquire` and `Release`
- `core/memory/stream_pool.cpp` and `.h` manage CUDA streams via a global `TorchStreamPool` instance
- `core/prefetch` and `core/python/py_archer_prefetch.cpp` show the established pybind11 style and error handling patterns

## Proposed C++ Interface

### Extension Module: `moe_infinity._kv_cache` (new module)

Implemented in: `core/python/py_kv_cache.cpp` (new file)

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<KVCachePool>(m, "kv_cache_pool")
        .def(py::init<std::size_t, int>(),       // (block_size_bytes, num_blocks)
             py::arg("block_size"), py::arg("num_blocks"))
        .def("allocate", &KVCachePool::Allocate,  // () -> int block_id
             "Allocate one block, return block ID. Blocks if pool empty.")
        .def("free", &KVCachePool::Free,          // (int block_id) -> void
             "Return block to pool.")
        .def("async_copy_to_cpu",                  // (tensor, block_id, stream_id) -> void
             &KVCachePool::AsyncCopyToCPU,
             py::arg("tensor"), py::arg("block_id"), py::arg("stream_id") = 0)
        .def("async_copy_to_gpu",                  // (block_id, tensor, stream_id) -> void
             &KVCachePool::AsyncCopyToGPU,
             py::arg("block_id"), py::arg("tensor"), py::arg("stream_id") = 0)
        .def("sync_stream",                        // (stream_id) -> void
             &KVCachePool::SyncStream,
             py::arg("stream_id") = 0,
             "Wait for all async copies on stream to complete.");
}
```

The `_kv_cache` module is separate from `_store` so that KV cache swapping can evolve independently from expert offloading. It should still link against the same `archer_core` static library and reuse the same CUDA stream and logging utilities.

### Class: `KVCachePool`

Header: `core/memory/kv_cache_pool.h`

```cpp
class KVCachePool {
public:
    // Allocate pinned memory blocks at construction time.
    // block_size: bytes per block (should be multiple of 4096 for alignment)
    // num_blocks: total blocks in pool
    explicit KVCachePool(std::size_t block_size, int num_blocks);
    ~KVCachePool();

    // Allocate a block. Returns block_id. Blocks until one is available.
    int Allocate();

    // Return block to free pool.
    void Free(int block_id);

    // Async GPU→CPU: copy tensor data into pinned block.
    // tensor: source GPU tensor (must be contiguous, size <= block_size)
    // block_id: target block in pinned pool
    // stream_id: CUDA stream index (from StreamPool)
    void AsyncCopyToCPU(const torch::Tensor& tensor, int block_id, int stream_id = 0);

    // Async CPU→GPU: copy pinned block into tensor.
    // block_id: source block in pinned pool
    // tensor: target GPU tensor (pre-allocated, must match size)
    // stream_id: CUDA stream index
    void AsyncCopyToGPU(int block_id, torch::Tensor& tensor, int stream_id = 0);

    // Wait for all async operations on stream to complete.
    void SyncStream(int stream_id = 0);

private:
    std::size_t block_size_;
    std::vector<void*> blocks_;           // pinned memory pointers
    std::vector<bool> pinned_registered_; // cudaHostRegister success
    std::queue<int> free_list_;
    std::mutex mutex_;
    std::condition_variable cv_;
};
```

`KVCachePool` intentionally mirrors the behavior of `PinnedMemoryPool`, but it exposes block identifiers instead of raw pointers so that the Python layer can safely store and pass block handles.

### Implementation Notes

1. Reuse the `PinnedMemoryPool` pattern. The `KVCachePool` implementation should follow `core/memory/pinned_memory_pool.cpp` closely. It should use `posix_memalign` with 4096 byte alignment and `cudaHostRegister` for pinning, with a graceful fallback if registration fails.

2. Reuse `StreamPool` for CUDA streams. Use `StreamPool` from `core/memory/stream_pool.cpp` for managing CUDA streams. Do not create new streams per transfer. The `stream_id` argument passed from Python should index into the existing global stream pool that is already used for expert offloading.

3. Block size recommendation. For typical LLM KV cache, a natural block size in bytes is

   `block_size_tokens × num_kv_heads × head_dim × sizeof(dtype) × 2`

   The factor 2 accounts for key and value. For example:

   - 16 tokens
   - 8 KV heads
   - 128 head dimension
   - fp16 (2 bytes per value)

   Then the block size in bytes is 16 × 8 × 128 × 2 × 2 which equals 65536 bytes, that is, 64 KB. Always round up to a multiple of 4096 bytes for page alignment.

4. Thread safety. `Allocate` and `Free` must be thread safe. Use the same mutex and condition variable pattern as `PinnedMemoryPool`. Async copy operations can use separate CUDA streams per caller; the `KVCachePool` does not need additional locking beyond access to the internal block free list.

5. Error handling. All CUDA calls should use the same logging and error macros that the rest of the codebase uses. In particular, failed `cudaMemcpyAsync` or invalid tensor size checks should raise Python exceptions through pybind11 so that `KVCacheManager` can surface clear errors.

## Python Integration Points

Once the extension is implemented, `moe_infinity/memory/kv_cache_manager.py` will route swap operations through it instead of the current `NotImplementedError` stubs.

The expected integration pattern is sketched below, using a dedicated `_kv_cache` module. Exact naming can be adjusted during implementation, but the responsibilities should match this structure.

```python
# In KVCacheManager.__init__():
import moe_infinity._kv_cache as _kv_cache

self._cpp_pool = _kv_cache.kv_cache_pool(block_size_bytes, num_blocks)


# In KVCacheManager.swap_out(block_ids):
for bid in block_ids:
    cpu_id = self.cpu_pool.allocate(1)[0]
    self._cpp_pool.async_copy_to_cpu(self._gpu_tensors[bid], cpu_id, stream_id=0)
self._cpp_pool.sync_stream(0)


# In KVCacheManager.swap_in(block_ids):
for bid in block_ids:
    gpu_id = self.gpu_pool.allocate(1)[0]
    self._cpp_pool.async_copy_to_gpu(bid, self._gpu_tensors[gpu_id], stream_id=0)
self._cpp_pool.sync_stream(0)
```

Notes on alignment with existing Python code:

- `BlockPool` already models a pool of integer block identifiers on a given device string. `KVCachePool` should use the same block index space for CPU pinned blocks so that `KVCacheManager` can pass block ids through without extra mapping.
- `KVCacheManager.get_block_table` returns a dense `torch.int32` tensor of block ids for a given sequence. C++ code must treat these ids as opaque handles and must not assume any particular layout.
- `AttentionBackend.get_kv_cache_shape` defines the logical shape of the KV cache tensor for a given block layout. The C++ layer only moves raw bytes and does not depend on the exact tensor shape as long as the storage size matches the configured block size.

## Memory Budget Guidance

From `MemoryBudget` in `moe_infinity/memory/kv_cache_manager.py`:

- `expert_cache_ratio` (default 0.75) is the fraction of GPU memory reserved for the expert weight LRU cache
- `kv_cache_ratio` (default 0.0, disabled) is the fraction reserved for the KV cache block pool
- Constraint: `expert_cache_ratio + kv_cache_ratio` must be less than or equal to 1.0

Recommended ranges, based on PiKV 2025 and ProMoE style designs:

- For large MoE models, such as Mixtral and DeepSeek V2, keep `expert_cache_ratio` at or above 0.60 to maintain expert hit rate
- Use `kv_cache_ratio` at or below 0.25 by default

PCIe bandwidth is the shared bottleneck. When expert weights and KV cache blocks are both swapping at high rates, throughput collapses. A static split between expert cache and KV cache capacity in GPU memory, driven by `MemoryBudget`, helps keep the combined traffic under control.

## Implementation Checklist

For the C++ implementer:

- [ ] Create `core/memory/kv_cache_pool.h` and `core/memory/kv_cache_pool.cpp`
- [ ] Create `core/python/py_kv_cache.cpp` with pybind11 bindings
- [ ] Add `kv_cache_pool.cpp` to `ARCHER_CORE_CXX_SOURCES` in `core/CMakeLists.txt`
- [ ] Add the new Python extension in `setup.py` alongside `_store` and `_engine`
- [ ] Update `moe_infinity/memory/kv_cache_manager.py` to remove `NotImplementedError` stubs and to call the new `_kv_cache` bindings
- [ ] Add an integration test in `tests/python/integration` that exercises swap in, swap out, and attention computation against a small toy model

## Files Modified or Created

| File | Action | Notes |
|------|--------|-------|
| `core/memory/kv_cache_pool.h` | CREATE | New header for KV cache pool |
| `core/memory/kv_cache_pool.cpp` | CREATE | Implementation of pinned block pool and async copies |
| `core/python/py_kv_cache.cpp` | CREATE | Python bindings for `KVCachePool` |
| `core/CMakeLists.txt` | MODIFY | Add new source file to `archer_core` build |
| `setup.py` | MODIFY | Register `_kv_cache` extension module |
| `moe_infinity/memory/kv_cache_manager.py` | MODIFY | Replace `NotImplementedError` swap stubs with real integration |
