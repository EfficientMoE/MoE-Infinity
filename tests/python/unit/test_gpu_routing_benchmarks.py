from benchmarks.expert_io_microbench.bench_bubble import summarize_iterations
from benchmarks.expert_io_microbench.bench_routing import summarize_gpu_routing


def test_gpu_routing_summary_exposes_submit_fallback_and_native_stats():
    events = [
        {"stage": "gpu_route_submit", "dur_ns": 100},
        {"stage": "gpu_route_submit", "dur_ns": 300},
        {"stage": "gpu_route_fallback", "dur_ns": 900},
    ]
    summary = summarize_gpu_routing(
        events,
        {
            "route_batches": 2,
            "route_failures": 0,
            "last_active_experts": 4,
            "last_route_handoff_us": 7,
            "completion_events_retired": 8,
        },
    )
    assert summary["submit_p50_ns"] == 200.0
    assert summary["fallback_count"] == 1
    assert summary["native"]["route_failures"] == 0


def test_bubble_summary_splits_route_and_completion_handoff():
    summary = summarize_iterations(
        [
            {
                "step_total_ns": 1000,
                "expert_wait_ns": 300,
                "route_submit_ns": 40,
                "completion_handoff_ns": 60,
                "bubble_ratio": 0.3,
                "layer_wait_ns": {},
                "sync_event_count": 1,
                "io_profiler_event_count": 3,
            }
        ]
    )
    assert summary["step_decomposition_mean"]["route_submit_ns"] == 40.0
    assert summary["step_decomposition_mean"]["completion_handoff_ns"] == 60.0


def test_nsys_summary_reports_activity_bitmap_d2h(monkeypatch):
    from benchmarks.expert_io_microbench import nsys_parser

    monkeypatch.setattr(
        nsys_parser,
        "parse_nsys_report",
        lambda _path: {
            "ranges": {
                "gpu_route_submit": {
                    "total_ns": 100,
                    "count": 1,
                    "mean_ns": 100,
                    "p50_ns": 100,
                }
            },
            "memcpy": {
                "h2d_bytes": 0,
                "h2d_count": 0,
                "d2h_bytes": 8,
                "d2h_count": 1,
                "d2d_bytes": 0,
                "d2d_count": 0,
            },
            "gpu_memcpy_ns": {"h2d": 0, "d2h": 10, "d2d": 0},
            "cuda_api": {
                "stream_synchronize_count": 0,
                "device_synchronize_count": 0,
            },
            "duration_ns": 100,
        },
    )
    summary = nsys_parser.summarise(
        "unused.nsys-rep", 1, {"link_width": 16, "link_gen": 4}
    )
    assert summary["routing_sync"]["device_to_host_memcpy_count"] == 1
    assert summary["routing_sync"]["device_to_host_memcpy_bytes"] == 8
    assert summary["routing_sync"]["stream_synchronize_count"] == 0
