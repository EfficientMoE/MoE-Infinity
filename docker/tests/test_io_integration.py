"""
I/O integration tests for the refactored MoE-Infinity AIO layer.

Tier 1 (no CUDA): threading, file I/O, thread count logic
Tier 2 (CUDA required): pinned memory, tensor roundtrip, prefetch_handle
"""

import glob
import importlib.util
import os
import sys
import threading

import pytest
import torch

# ---------------------------------------------------------------------------
# Import the C++ test module (built by build_test_module.py)
# ---------------------------------------------------------------------------
_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _test_dir)
import test_io_module  # noqa: E402

# ---------------------------------------------------------------------------
# Import the main prefetch_op extension (built by build_prefetch.py)
# ---------------------------------------------------------------------------
_prefetch_dir = os.path.join(
    os.path.dirname(_test_dir), "..", "moe_infinity", "ops", "prefetch"
)
_prefetch_dir = os.path.normpath(_prefetch_dir)
_so_files = glob.glob(os.path.join(_prefetch_dir, "prefetch_op*.so"))
prefetch_op = None
if _so_files:
    spec = importlib.util.spec_from_file_location("prefetch_op", _so_files[0])
    prefetch_op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefetch_op)


# =========================================================================
# Tier 1 tests (no CUDA required)
# =========================================================================


class TestTier1AioThread:
    def test_cpp_aio_thread_basic(self):
        """Phase 0: CV waits, atomics — enqueue 10 callbacks."""
        result = test_io_module.test_aio_thread_basic()
        assert result["passed"], result["detail"]

    def test_cpp_aio_thread_stop_empty(self):
        """Stop on idle thread must not hang."""
        result = test_io_module.test_aio_thread_stop_empty()
        assert result["passed"], result["detail"]

    def test_cpp_aio_thread_error_callback(self):
        """All callbacks executed even if some return non-zero."""
        result = test_io_module.test_aio_thread_error_callback()
        assert result["passed"], result["detail"]

    def test_cpp_aio_thread_throughput(self):
        """5000 callbacks executed without loss."""
        result = test_io_module.test_aio_thread_throughput(5000)
        assert result["passed"], result["detail"]


class TestTier1ThreadPool:
    def test_cpp_threadpool_round_robin(self):
        """Phase 0C + Phase 1: round-robin dispatch with 2 and 4 threads."""
        for n in (2, 4):
            result = test_io_module.test_aio_threadpool_round_robin(n)
            assert result["passed"], f"num_threads={n}: {result['detail']}"

    def test_cpp_threadpool_targeted(self):
        """Targeted enqueue: callbacks reach the correct thread slot."""
        for n in (2, 4):
            result = test_io_module.test_aio_threadpool_targeted(n)
            assert result["passed"], f"num_threads={n}: {result['detail']}"

    def test_cpp_threadpool_concurrent_producers(self):
        """Multiple producers enqueue concurrently without loss."""
        result = test_io_module.test_aio_threadpool_concurrent_producers(
            num_threads=4, num_producers=8
        )
        assert result["passed"], result["detail"]


class TestTier1FileIO:
    def test_cpp_file_io_direct(self, workspace_tmpdir):
        """O_DIRECT write/read roundtrip with 4096-aligned buffers."""
        result = test_io_module.test_file_io_direct(workspace_tmpdir)
        assert result["passed"], result["detail"]

    def test_cpp_file_io_various_sizes(self, workspace_tmpdir):
        """O_DIRECT roundtrip for 4096, 8192, 16384 and 512-byte payloads."""
        result = test_io_module.test_file_io_various_sizes(workspace_tmpdir)
        assert result["passed"], result["detail"]


class TestTier1DefaultThreads:
    def test_default_io_threads(self):
        """Phase 1: GetDefaultNumIoThreads() >= 4."""
        n = test_io_module.get_default_num_io_threads()
        assert n >= 4, f"Expected >= 4 I/O threads, got {n}"


# =========================================================================
# Tier 2 tests (CUDA required)
# =========================================================================


@pytest.mark.cuda
class TestTier2PinnedPool:
    def test_cpp_pinned_pool(self):
        """Phase 2: acquire/release cycle with reuse verification."""
        result = test_io_module.test_pinned_memory_pool()
        assert result["passed"], result["detail"]

    def test_cpp_pinned_pool_contention(self):
        """Phase 2: blocking when pool is exhausted."""
        result = test_io_module.test_pinned_pool_contention()
        assert result["passed"], result["detail"]


@pytest.mark.cuda
class TestTier2TensorHandle:
    def test_cpp_tensor_roundtrip(self, workspace_tmpdir):
        """Phase 1+2+4: store tensor, read back, memcmp."""
        result = test_io_module.test_tensor_store_read_roundtrip(
            workspace_tmpdir
        )
        assert result["passed"], result["detail"]

    def test_cpp_tensor_offsets(self, workspace_tmpdir):
        """Phase 4: verify partition offset tracking for 3 tensors."""
        result = test_io_module.test_tensor_offset_tracking(workspace_tmpdir)
        assert result["passed"], result["detail"]

    def test_cpp_multi_tensor_roundtrip(self, workspace_tmpdir):
        """Store/read roundtrip for float32 and float16 tensors."""
        result = test_io_module.test_multi_tensor_roundtrip(workspace_tmpdir)
        assert result["passed"], result["detail"]

    def test_cpp_tensor_index_serialization(self, workspace_tmpdir):
        """TensorIndex serialize/deserialize preserves offsets."""
        result = test_io_module.test_tensor_index_serialization(
            workspace_tmpdir
        )
        assert result["passed"], result["detail"]


@pytest.mark.cuda
class TestTier2PrefetchHandle:
    @pytest.fixture(autouse=True)
    def _skip_if_no_prefetch(self):
        if prefetch_op is None:
            pytest.skip("prefetch_op not available")

    def test_prefetch_offload_verify(self, workspace_tmpdir):
        """Offload tensor via prefetch_handle, verify on-disk data."""
        handle = prefetch_op.prefetch_handle(workspace_tmpdir, 0.75)

        tensor = torch.randn(256, 128)
        tensor_id = 42
        handle.offload(tensor, tensor_id)

        # Verify is_tensor_offloaded
        assert handle.is_tensor_offloaded(
            tensor_id
        ), "is_tensor_offloaded() returned False"

        # Verify parameter file exists
        param_file = os.path.join(workspace_tmpdir, "archer_param_0")
        assert os.path.exists(
            param_file
        ), f"Parameter file not found: {param_file}"

        # Verify raw bytes match tensor data
        nbytes = tensor.numel() * tensor.element_size()
        with open(param_file, "rb") as f:
            raw_bytes = f.read(nbytes)
        expected_bytes = tensor.numpy().tobytes()
        assert raw_bytes == expected_bytes, "File bytes don't match tensor data"

        del handle

    def test_prefetch_index_persistence(self, workspace_tmpdir):
        """Offload with handle1, destroy, create handle2, verify index."""
        tensor_id = 99

        # Offload with first handle
        handle1 = prefetch_op.prefetch_handle(workspace_tmpdir, 0.75)
        tensor = torch.randn(64, 32)
        handle1.offload(tensor, tensor_id)
        del handle1

        # Create second handle on same directory
        handle2 = prefetch_op.prefetch_handle(workspace_tmpdir, 0.75)
        assert (
            handle2.is_tensor_index_initialized()
        ), "Index not loaded from disk"
        assert handle2.is_tensor_offloaded(
            tensor_id
        ), f"Tensor {tensor_id} not found after index reload"
        del handle2

    def test_prefetch_concurrent_offload(self, workspace_tmpdir):
        """3 threads each offload 50 tensors concurrently."""
        handle = prefetch_op.prefetch_handle(workspace_tmpdir, 0.75)

        errors = []
        n_per_thread = 50
        n_threads = 3

        def offload_worker(thread_idx):
            try:
                base_id = thread_idx * n_per_thread
                for i in range(n_per_thread):
                    t = torch.randn(32, 16)
                    handle.offload(t, base_id + i)
            except Exception as e:
                errors.append(f"Thread {thread_idx}: {e}")

        threads = []
        for idx in range(n_threads):
            t = threading.Thread(target=offload_worker, args=(idx,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Errors in worker threads: {errors}"

        # Verify all 150 tensors are offloaded
        total = n_threads * n_per_thread
        for tid in range(total):
            assert handle.is_tensor_offloaded(
                tid
            ), f"Tensor {tid} not offloaded"

        del handle
