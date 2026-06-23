from __future__ import annotations

import time
from typing import Callable

from benchmarks.expert_io_microbench.stats import TimingCollector

try:
    import torch as _torch
except Exception:
    _torch = None

torch = _torch


class BenchmarkHarness:
    def run(
        self,
        fn: Callable[[], object],
        warmup: int = 10,
        iters: int = 100,
        use_cuda_events: bool = False,
    ) -> TimingCollector:
        if warmup < 0 or iters < 0:
            raise ValueError("warmup and iters must be non-negative")

        collector = TimingCollector()
        if torch is not None and torch.cuda.is_available():
            cuda_available = True
            cuda = torch.cuda
        else:
            cuda_available = False
            cuda = None

        for _ in range(warmup):
            _ = fn()
            if cuda_available and cuda is not None:
                cuda.synchronize()

        for _ in range(iters):
            if cuda_available and cuda is not None:
                cuda.synchronize()

            if use_cuda_events and cuda_available and cuda is not None:
                current_stream = cuda.current_stream()
                start_event = cuda.Event(enable_timing=True)
                end_event = cuda.Event(enable_timing=True)
                start_event.record(stream=current_stream)
                _ = fn()
                end_event.record(stream=current_stream)
                cuda.synchronize()
                duration_ns = int(
                    start_event.elapsed_time(end_event) * 1_000_000
                )
            else:
                start = time.perf_counter_ns()
                _ = fn()
                if cuda_available and cuda is not None:
                    cuda.synchronize()
                duration_ns = time.perf_counter_ns() - start

            collector.record("benchmark", duration_ns)

        return collector
