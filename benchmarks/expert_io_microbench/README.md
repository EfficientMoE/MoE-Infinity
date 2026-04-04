# Expert I/O Microbench Integration Runner

`run_all.py` executes expert I/O microbench scenarios and writes a unified JSON report.

## Scenarios

- `routing` → `bench_routing.py`
- `transfer` → `bench_transfer.py`
- `compute` → `bench_compute_evict.py`
- `bubble` → `bench_bubble.py`

Use `--scenario all` (default) to run every scenario, or run one scenario at a time.

## CLI

```bash
python benchmarks/expert_io_microbench/run_all.py \
  --model <hf-model-or-local-path> \
  --offload-dir <offload-dir> \
  --device-memory-ratio 0.5 \
  --warmup 10 \
  --iters 100 \
  --scenario all \
  --output-json benchmarks/expert_io_microbench/results/all.json
```

Optional:

- `--theoretical-pcie-gbps <float>`: provide theoretical PCIe throughput explicitly. If omitted, runner attempts to read it from CUDA device properties.

## Output sections

Unified output JSON includes:

- `scenarios`: raw per-scenario benchmark JSON (or `{ "error": ... }` on failure)
- `bandwidth_analysis`:
  - `theoretical_bandwidth_gbps`
  - per-link `actual_bandwidth_gbps`
  - per-link `utilization_pct`
- `executive_summary`:
  - top 3 bottlenecks ranked by `% of step time`

The runner always continues if one scenario fails and records the failure under that scenario key.

## Help

```bash
python benchmarks/expert_io_microbench/run_all.py --help
```
