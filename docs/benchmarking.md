# Benchmarking MoE-Infinity

This guide explains how to correctly measure throughput and latency when running MoE-Infinity, and how to make fair comparisons with other inference frameworks.

## Key Terminology

| Term | Definition |
|------|-----------|
| **TTFT** (Time To First Token) | Time from submitting a prompt until the first generated token is produced. This covers prompt encoding (prefill). |
| **ITL** (Inter-Token Latency) | Average time between consecutive generated tokens during the decode phase. |
| **Decode Throughput** | Tokens generated per second during the decode phase only (excludes prefill). |
| **End-to-End Throughput** | Total tokens generated divided by total wall-clock time (includes prefill). |
| **Prefill** | The initial phase where the model processes all input tokens to build the KV cache. This is compute-bound and typically much faster per-token than decode. |
| **Decode** | The autoregressive phase where the model generates one token at a time. This is memory-bandwidth-bound in MoE offloading scenarios. |

## Common Measurement Pitfalls

### Pitfall 1: Including Prefill Time in Decode Throughput

The most common mistake is dividing total generated tokens by total wall-clock time (which includes prefill). This conflates two fundamentally different phases and produces misleadingly low throughput numbers.

```python
# WRONG: This includes prefill time in the throughput calculation
start_time = time.time()
output_ids = model.generate(input_ids, max_new_tokens=256)
elapsed = time.time() - start_time
throughput = len(output_ids[0]) / elapsed  # Includes prefill -- NOT decode throughput
```

### Pitfall 2: Not Warming Up the Expert Cache

The first inference run loads experts from disk into CPU memory and then transfers them to GPU. Subsequent runs benefit from cached experts. Always run at least one warmup request before measuring.

### Pitfall 3: Comparing Different Metrics Across Frameworks

When comparing with llama.cpp, vLLM, or other frameworks, ensure you are comparing the same metric. For example, llama.cpp reports separate `prompt eval time` and `eval time` -- use `eval time` for decode throughput comparison.

## Using the StopWatch Utility

MoE-Infinity provides a `StopWatch` class in [`examples/interface_example.py`](../examples/interface_example.py) that correctly separates prefill from decode timing by hooking into HuggingFace's `TextStreamer` callback.

### How StopWatch Works

```
generate() called
  |
  v
put() called (1st time) --> start_prefilling = now
  |
  v
put() called (2nd time) --> prefilling_time = now - start_prefilling
                            start_decoding = now
                            clear expert cache counts
  |
  v
put() called (3rd+ time) --> decoding_iterations++
  |
  v
end() called --> decoding_time = now - start_decoding
```

The key insight: the `TextStreamer.put()` callback is invoked once per generated token. The first call marks the beginning of prefill, the second marks the first decoded token (end of prefill / start of decode), and all subsequent calls are decode iterations.

### Standalone Measurement Example

```python
import time
import torch
from transformers import AutoTokenizer, TextStreamer
from moe_infinity import MoE


class StopWatch(TextStreamer):
    """Separates prefill (TTFT) from decode latency."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.start_decoding = time.time()
        self.decoding_iterations += 1
        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
        return super().end()


# --- Setup ---
model_path = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

config = {
    "offload_path": "/path/to/offload/dir",
    "device_memory_ratio": 0.75,
}
model = MoE(model_path, config)

# --- Warmup (important!) ---
warmup_ids = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda:0")
with torch.no_grad():
    model.generate(warmup_ids, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id)

# --- Measurement ---
prompt = "Explain the theory of relativity in simple terms."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

streamer = StopWatch(tokenizer)
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# --- Results ---
print(f"TTFT (prefill):              {streamer.prefilling_time:.3f} s")
print(f"Decode time:                 {streamer.decoding_time:.3f} s")
print(f"Decode iterations:           {streamer.decoding_iterations}")
print(f"Per-token latency (decode):  {streamer.decoding_time / streamer.decoding_iterations:.4f} s")
print(f"Decode throughput:           {streamer.decoding_iterations / streamer.decoding_time:.2f} tokens/s")
```

## Benchmark Scripts

MoE-Infinity includes ready-to-use benchmark scripts in [`benchmarks/serving/`](../benchmarks/serving/).

### Baseline Performance (Single Request)

Measures single-request TTFT, per-token latency, and peak GPU memory across multiple prompt lengths.

```bash
python benchmarks/serving/baseline_performance.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --num-requests 10 \
    --output-json baseline_results.json
```

**Output fields:**
- `ttft_ms` -- Time to first token (milliseconds)
- `per_token_latency_ms` -- Average per-token decode latency (milliseconds)
- `peak_gpu_memory_mb` -- Peak GPU memory usage (MB)

### Throughput Sweep

Measures tokens/s across different batch sizes to find the throughput-optimal batch size.

```bash
python benchmarks/serving/throughput.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --num-requests 50 \
    --batch-sizes 1 2 4 8 16 32 \
    --prompt-length 128 \
    --max-new-tokens 16 \
    --output-json throughput_results.json
```

### Latency Under Concurrency

Measures TTFT and ITL at p50/p90/p99 percentiles across different concurrency levels.

```bash
python benchmarks/serving/latency.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --concurrency 1 2 4 8 \
    --num-rounds 5 \
    --prompt-length 128 \
    --max-new-tokens 16 \
    --output-json latency_results.json
```

**Output fields per concurrency level:**
- `ttft_p50_ms`, `ttft_p90_ms`, `ttft_p99_ms` -- TTFT percentiles
- `itl_p50_ms`, `itl_p90_ms`, `itl_p99_ms` -- Inter-token latency percentiles

### Kernel Microbenchmark

Compares gating kernel (top-k softmax) performance across implementations.

```bash
python benchmarks/bench_p0_topk_softmax.py --num-iters 200 --warmup 50
```

## Comparing with Other Frameworks

### llama.cpp

llama.cpp reports timing in its output after `Ctrl+C`:

```
llama_perf_context_print: prompt eval time = 2251.78 ms /  39 tokens (57.74 ms per token, 17.32 tokens per second)
llama_perf_context_print:        eval time = 122985.89 ms / 491 runs  (250.48 ms per token, 3.99 tokens per second)
```

- **`prompt eval time`** = Prefill. Compare with MoE-Infinity's TTFT.
- **`eval time`** = Decode. Compare with MoE-Infinity's decode time.
- **`eval` tokens per second** = Decode throughput. Compare with `decoding_iterations / decoding_time`.

### vLLM / SGLang

These frameworks report TTFT and ITL directly in their benchmark outputs. Use p50 values for comparison with MoE-Infinity's `StopWatch` measurements (which report averages).

### Fair Comparison Checklist

- [ ] Same model weights (not quantized vs full-precision)
- [ ] Same GPU and same `device_memory_ratio` / memory allocation
- [ ] Same number of GPU layers offloaded (e.g., llama.cpp's `-ngl` flag)
- [ ] Prefill excluded from decode throughput in both frameworks
- [ ] At least one warmup run before measurement
- [ ] Same `max_new_tokens` / generation length
- [ ] Same sampling strategy (`do_sample=False` / greedy for deterministic comparison)

## Tuning `device_memory_ratio`

The `device_memory_ratio` parameter controls what fraction of GPU memory is allocated for expert caching. The remainder is used by PyTorch for activations, KV cache, and other tensors.

| Value | Effect |
|-------|--------|
| **Higher** (e.g., 0.85) | More experts cached on GPU = fewer cache misses = faster decode. Risk: OOM if model activations are large. |
| **Lower** (e.g., 0.50) | Fewer experts cached = more cache misses = slower decode. Benefit: more headroom for large prompts / batches. |
| **Default** (0.75) | Good starting point for most single-GPU setups. |

If you encounter CUDA OOM errors, lower this value. If decode throughput is poor, try raising it (assuming no OOM).
