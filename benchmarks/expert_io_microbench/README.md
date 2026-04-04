# Expert I/O Microbench Integration Runner

`run_all.py` executes expert I/O microbench scenarios and writes a unified JSON report.

## Docker Setup

### Shared Memory (Required for Large Models)
By default Docker limits `/dev/shm` to 64MB, which causes `Bus error` with large models.
Always run with sufficient shared memory:

```bash
docker run --gpus all --shm-size=32g --ipc=host moe-infinity-bench \
  python benchmarks/expert_io_microbench/run_all.py ...
```

### Host-Only Mode (Isolate PCIe Overhead)
Use `--host-only` to copy expert weights to tmpfs (RAM), eliminating disk I/O:

```bash
python benchmarks/expert_io_microbench/run_all.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /path/to/offload \
  --host-only \
  --output-json host_only_results.json
```

This reveals the **lower bound bubble ratio** — how much stall remains even with perfect prefetching
(all experts in CPU RAM). Compare against disk mode to quantify disk I/O contribution.

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
- `--host-only`: copy offload dir to `/dev/shm` once in `run_all.py` and run all scenarios from tmpfs.

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
