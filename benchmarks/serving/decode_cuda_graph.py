from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired decode CUDA graph benchmark"
    )
    parser.add_argument(
        "--mode", choices=("fixture", "resident", "model"), default="fixture"
    )
    parser.add_argument(
        "--resident-config",
        choices=("tiny", "qwen3-30b-a3b"),
        default="tiny",
        help="resident mode: random-weight Qwen3-MoE shape",
    )
    parser.add_argument(
        "--experts-impl",
        choices=("batched_mm", "grouped_mm"),
        default="batched_mm",
        help="resident mode: HF experts implementation (capture-safe ones)",
    )
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--model")
    parser.add_argument("--offload-dir")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--context-sizes", nargs="+", type=int, default=[128, 512]
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--max-graph-memory-bytes", type=int, default=0)
    parser.add_argument("--profile-launches", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "model" and (not args.model or not args.offload_dir):
        raise ValueError("model mode requires --model and --offload-dir")
    for name in ("batch_sizes", "context_sizes"):
        values = getattr(args, name)
        if not values or any(value <= 0 for value in values):
            raise ValueError(
                f"--{name.replace('_', '-')} values must be positive"
            )
    if args.warmup_iters < 1 or args.measure_iters < 1:
        raise ValueError("warmup and measurement iterations must be positive")
    if args.max_graph_memory_bytes < 0:
        raise ValueError("max graph memory must be non-negative")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": float(statistics.median(values)),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
    }


def build_result(
    *,
    config: dict[str, object],
    eager_us: list[float],
    replay_us: list[float],
    graph_stats: dict[str, object],
    environment: dict[str, object],
    correctness: dict[str, object] | None = None,
    host: dict[str, object] | None = None,
) -> dict[str, object]:
    measurements: dict[str, object] = {
        "eager_us": eager_us,
        "replay_us": replay_us,
    }
    if eager_us:
        measurements["eager_summary_us"] = _summary(eager_us)
    if replay_us:
        measurements["replay_summary_us"] = _summary(replay_us)
    if host:
        measurements.update(host)
    if eager_us and replay_us:
        replay_p50 = statistics.median(replay_us)
        measurements["observed_ratio"] = (
            statistics.median(eager_us) / replay_p50 if replay_p50 else None
        )
    return {
        "schema_version": 1,
        "config": config,
        "measurements": measurements,
        "graph_stats": graph_stats,
        "environment": environment,
        "correctness": correctness or {},
    }


def _time_cuda(
    operation: Callable[[], torch.Tensor | None],
) -> tuple[float, float, torch.Tensor | None]:
    """Return (GPU microseconds, host microseconds, output) for one call.

    The host time is the wall time of the call itself, measured before the
    stream is synchronised: the CPU cost of issuing the step.
    """
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    host_start = time.perf_counter()
    output = operation()
    host_us = (time.perf_counter() - host_start) * 1e6
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0), host_us, output


_LAUNCH_APIS = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaGraphLaunch",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
)


def _profile_cuda_operation(
    operation: Callable[[], torch.Tensor | None],
) -> dict[str, float | int]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        _ = operation()
        torch.cuda.synchronize()
    host_launches = 0
    graph_launches = 0
    device_activities = 0
    device_time_us = 0.0
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            device_activities += 1
            device_time_us += float(event.device_time)
        elif event.name in _LAUNCH_APIS:
            host_launches += 1
            graph_launches += int(event.name == "cudaGraphLaunch")
    return {
        "host_launch_calls": host_launches,
        "graph_launches": graph_launches,
        "device_activities": device_activities,
        "device_time_us": device_time_us,
    }


def _environment() -> dict[str, object]:
    return {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def _build_fixture(args: argparse.Namespace, batch_sizes, context_sizes):
    from benchmarks.serving.decode_cuda_graph_fixture import (
        build_fixture,
        build_resident_fixture,
    )

    if args.mode == "resident":
        return build_resident_fixture(
            batch_sizes=batch_sizes,
            context_sizes=context_sizes,
            warmup_iters=args.warmup_iters,
            config_name=args.resident_config,
            experts_impl=args.experts_impl,
            num_layers=args.num_layers,
            max_graph_memory_bytes=args.max_graph_memory_bytes,
        )
    return build_fixture(
        batch_sizes=batch_sizes,
        context_sizes=context_sizes,
        warmup_iters=args.warmup_iters,
        max_graph_memory_bytes=args.max_graph_memory_bytes,
    )


def _run_fixture(args: argparse.Namespace) -> dict[str, object]:
    batch_sizes = tuple(sorted(set(args.batch_sizes)))
    context_sizes = tuple(sorted(set(args.context_sizes)))
    fixture = _build_fixture(args, batch_sizes, context_sizes)
    runner = fixture.model_runner
    points: list[dict[str, object]] = []
    try:
        for batch_size in batch_sizes:
            for context_size in context_sizes:
                batch = fixture.make_batch(batch_size, context_size)

                def eager_step() -> torch.Tensor:
                    return runner.execute(batch)

                def graph_step() -> torch.Tensor | None:
                    return fixture.graph_runner.try_execute(batch)

                for _ in range(args.warmup_iters):
                    _ = eager_step()
                graph_output = graph_step()
                if graph_output is None:
                    raise RuntimeError(
                        f"fixture graph fallback: {fixture.graph_runner.stats()}"
                    )
                for _ in range(args.warmup_iters):
                    _ = graph_step()
                torch.cuda.synchronize()

                eager_us: list[float] = []
                replay_us: list[float] = []
                eager_host_us: list[float] = []
                replay_host_us: list[float] = []
                eager_output = eager_step()
                for iteration in range(args.measure_iters):
                    operations = ("eager", "graph")
                    if iteration % 2:
                        operations = tuple(reversed(operations))
                    for operation in operations:
                        if operation == "eager":
                            elapsed, host, eager_output = _time_cuda(eager_step)
                            eager_us.append(elapsed)
                            eager_host_us.append(host)
                        else:
                            elapsed, host, replayed = _time_cuda(graph_step)
                            if replayed is None:
                                raise RuntimeError(
                                    "fixture graph unexpectedly fell back"
                                )
                            graph_output = replayed
                            replay_us.append(elapsed)
                            replay_host_us.append(host)

                assert graph_output is not None and eager_output is not None
                torch.testing.assert_close(
                    graph_output.float(),
                    eager_output[:batch_size].float(),
                    rtol=args.rtol,
                    atol=args.atol,
                )
                argmax_equal = bool(
                    torch.equal(
                        graph_output.argmax(-1),
                        eager_output[:batch_size].argmax(-1),
                    )
                )
                checksums = [
                    float(fixture.storage.value_cache[layer].float().sum())
                    for layer in range(fixture.storage.spec.num_layers)
                ]
                stats = fixture.graph_runner.stats()
                scratch_kv_bytes = (
                    fixture.storage.num_graph_scratch_blocks
                    * fixture.storage.spec.block_size
                    * fixture.storage.spec.num_layers
                    * 2
                    * fixture.storage.spec.num_kv_heads
                    * fixture.storage.spec.head_dim
                    * torch.empty(
                        (), dtype=fixture.storage.spec.dtype
                    ).element_size()
                )
                registered = len(runner.paged_attention_registry.bindings)
                stats.update(
                    {
                        "capability_reason": "eligible",
                        "registered_paged_layers": registered,
                        "proved_write_layers": registered,
                        "kv_storage_owner_id": fixture.storage.owner_id,
                        "per_layer_kv_checksums": checksums[:4],
                        "native_attention_kernel": True,
                        "scratch_kv_bytes": scratch_kv_bytes,
                        "replay_coverage": 1.0,
                    }
                )
                launch_profile: dict[str, object] = {}
                if args.profile_launches:
                    launch_profile = {
                        "eager": _profile_cuda_operation(eager_step),
                        "replay": _profile_cuda_operation(graph_step),
                    }
                replay_gpu_p50 = statistics.median(replay_us)
                replay_host_p50 = statistics.median(replay_host_us)
                points.append(
                    build_result(
                        config={
                            "mode": args.mode,
                            "resident_config": (
                                args.resident_config
                                if args.mode == "resident"
                                else None
                            ),
                            "experts_impl": (
                                args.experts_impl
                                if args.mode == "resident"
                                else None
                            ),
                            "num_layers": fixture.storage.spec.num_layers,
                            "batch_size": batch_size,
                            "context_size": context_size,
                            "batch_buckets": batch_sizes,
                            "context_buckets": context_sizes,
                            "dtype": str(fixture.storage.dtype),
                        },
                        eager_us=eager_us,
                        replay_us=replay_us,
                        graph_stats=stats,
                        environment=_environment(),
                        correctness={
                            "eager_replay_close": True,
                            "eager_replay_argmax_equal": argmax_equal,
                            "launch_profile": launch_profile,
                        },
                        host={
                            "eager_host_us": _summary(eager_host_us),
                            "replay_host_us": _summary(replay_host_us),
                            "replay_host_over_gpu": (
                                replay_host_p50 / replay_gpu_p50
                                if replay_gpu_p50
                                else None
                            ),
                        },
                    )
                )
    finally:
        fixture.close()
    return {"schema_version": 1, "mode": args.mode, "results": points}


def _run_model(args: argparse.Namespace) -> dict[str, object]:
    from moe_infinity import MoE

    runtime = MoE(
        args.model,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": 0.75,
        },
    )
    capability_fn = getattr(runtime, "decode_graph_capability", None)
    capability = capability_fn() if callable(capability_fn) else None
    reason = getattr(capability, "reason", "missing_capability")
    points: list[dict[str, object]] = []
    for batch_size in sorted(set(args.batch_sizes)):
        for context_size in sorted(set(args.context_sizes)):
            input_ids = torch.ones(
                (batch_size, context_size),
                dtype=torch.long,
                device="cuda",
            )

            def eager_once() -> torch.Tensor:
                return runtime.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                )

            for _ in range(args.warmup_iters):
                _ = eager_once()
            torch.cuda.synchronize()
            eager_us = [
                _time_cuda(eager_once)[0] for _ in range(args.measure_iters)
            ]
            points.append(
                build_result(
                    config={
                        "mode": "model",
                        "model": args.model,
                        "batch_size": batch_size,
                        "context_size": context_size,
                    },
                    eager_us=eager_us,
                    replay_us=[],
                    graph_stats={
                        "captures": 0,
                        "replays": 0,
                        "capability_reason": reason,
                    },
                    environment=_environment(),
                    correctness={"comparison": "eager_only"},
                )
            )
    return {
        "schema_version": 1,
        "mode": "model",
        "model": args.model,
        "capability_reason": reason,
        "results": points,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("decode CUDA graph benchmark requires CUDA")
    result = _run_model(args) if args.mode == "model" else _run_fixture(args)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
