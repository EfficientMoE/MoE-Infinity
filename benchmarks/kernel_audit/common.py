"""Shared timing / L2-flush / GPU-telemetry helpers for the sm_120 kernel audit."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Callable

import torch

RESULTS_DIR = Path(
    os.environ.get(
        "KAUDIT_RESULTS",
        str(Path(__file__).resolve().parent / "results"),
    )
)

_L2_FLUSH: torch.Tensor | None = None


def l2_flush() -> None:
    """Evict the 128 MB L2 by writing a 512 MB buffer."""
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(
            512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda"
        )
    _L2_FLUSH.zero_()


def warm_gpu(ms: float = 200.0) -> None:
    """Bring an idle GPU up to boost clocks before the first timed case.

    The first case of a fresh process otherwise runs during the clock ramp
    (~0.7 us slower on a 20 us op). 200 ms of copies is too short to heat the
    card into its sustained power-capped state.
    """
    src = torch.empty(16 * 1024 * 1024, dtype=torch.int32, device="cuda")
    dst = torch.empty_like(src)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    while True:
        for _ in range(20):
            dst.copy_(src)
        end.record()
        end.synchronize()
        if start.elapsed_time(end) >= ms:
            break


@dataclass
class Timing:
    median_us: float
    min_us: float
    p90_us: float
    n: int
    # CUDA events tick at ~1.024 us on sm_120, so medians of short kernels snap
    # to tick multiples; the mean over many calls dithers below one tick.
    mean_us: float = float("nan")


def time_cuda(
    fn: Callable[[], object],
    *,
    iters: int = 20,
    warmup: int = 3,
    flush_l2: bool = True,
    max_seconds: float = 20.0,
) -> Timing:
    """Per-call device time with CUDA events; L2 flushed before each call.

    The flush kernel runs before the start event, so it is excluded from the
    measured interval.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    t_end = time.time() + max_seconds
    for _ in range(iters):
        if flush_l2:
            l2_flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1e3)
        if time.time() > t_end and len(samples) >= 3:
            break
    samples.sort()
    return Timing(
        median_us=median(samples),
        min_us=samples[0],
        p90_us=samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        n=len(samples),
        mean_us=fmean(samples),
    )


def time_graph(
    fn: Callable[[], object], *, reps: int = 50, rounds: int = 5
) -> Timing:
    """Back-to-back device time of `fn` captured in a CUDA graph (warm L2).

    Used for host-overhead-free timing of short kernels / multi-kernel layers.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        samples.append(st.elapsed_time(en) * 1e3 / reps)
    samples.sort()
    return Timing(
        median_us=median(samples),
        min_us=samples[0],
        p90_us=samples[-1],
        n=len(samples),
        mean_us=fmean(samples),
    )


class GpuMonitor:
    """Samples SM clock / power / util of the visible GPU via nvidia-smi."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, float, float, float]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.index = idx

    def __enter__(self):
        self._proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                self.index,
                "--query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"-lms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

        def _reader():
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                try:
                    parts = [float(x) for x in line.strip().split(",")]
                    self.samples.append(tuple(parts))  # type: ignore[arg-type]
                except ValueError:
                    pass

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        busy = [s for s in self.samples if s[3] > 50]
        use = busy or self.samples
        if not use:
            return {}
        return {
            "sm_clock_mhz_median": median(s[0] for s in use),
            "mem_clock_mhz_median": median(s[1] for s in use),
            "power_w_median": median(s[2] for s in use),
            "power_w_max": max(s[2] for s in use),
            "n_samples": len(use),
        }


def gpu_identity() -> dict:
    p = torch.cuda.get_device_properties(0)
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    q = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            idx,
            "--query-gpu=name,power.limit,clocks.max.sm,clocks.max.mem,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "torch_name": p.name,
        "cc": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "l2_mb": p.L2_cache_size / 2**20,
        "mem_gb": p.total_memory / 2**30,
        "nvidia_smi": q,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def other_gpu_procs() -> list[str]:
    """Compute processes on the visible GPU that are not this process."""
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            idx,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    me = str(os.getpid())
    return [
        line
        for line in out.splitlines()
        if line.strip() and line.split(",")[0].strip() != me
    ]


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
