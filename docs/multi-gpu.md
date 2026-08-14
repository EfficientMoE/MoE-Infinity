# Single-server multi-GPU operation

MoE-Infinity supports one Python process on one host with multiple visible CUDA devices.
The generic expert-offload path assigns experts by visible GPU index and keeps a cache per GPU.
It does not need `torchrun` for the standard `MoE` path.

## Supported Topology

- One server, one host, multiple visible GPUs.
- One process for the generic `MoE` path.
- Logical GPU numbering comes from `CUDA_VISIBLE_DEVICES`, so the first visible device becomes `cuda:0`.
- The runtime assumes the visible set is fixed while the model is loaded.
- Multi-node inference across separate machines is not supported here.

### Exact commands

Single visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python script.py
```

Two visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

Any visible order works, but it changes which physical GPU becomes logical `cuda:0` and therefore changes expert ownership.

## Device Selection

- `torch.cuda.device_count()` reports the visible GPUs.
- `get_default_device()` returns `cuda:0` when CUDA is available.
- `ArcherConfig.device_per_node` is set from `torch.cuda.device_count()`.
- The expert dispatcher uses the visible device count, not PCIe slot order or UUIDs.
- If you reorder `CUDA_VISIBLE_DEVICES`, you reorder logical ownership.

## Expert Ownership and Caching

- Expert placement is round robin by expert id, `gpu_id = expert_id % num_visible_gpus`.
- `ExpertOffloadCoordinator` and the Python tests use the same rule.
- `ExpertPrefetcher` warms the cache with `replace_cache_candidates(...)` and `enqueue_prefetch(...)`.
- The C++ `ExpertDispatcher` keeps `cached_experts_`, `cache_sizes_`, and per-GPU fetch and exec queues.
- `num_threads` is per GPU. `ExpertDispatcher` creates `num_threads` execution workers for each visible GPU, plus one fetch worker per GPU.
- Eviction is per GPU and picks a cached expert that is idle and least used, so caching is local to the GPU that owns the expert.

## Peer Transfers and Host Memory

- The safe assumption is host staging.
- `Node::SetDevice` stages data through pinned host memory when it loads from disk, then copies host to GPU with `cudaMemcpyHostToDevice`.
- The KV offload path also uses CPU staging with `async_d2h` and `async_h2d`.
- Do not assume universal GPU to GPU peer transfers. Whether a direct P2P path exists depends on the runtime, driver, and topology.
- If peer access is absent or expensive, the runtime still works, but the transfer path will lean on host memory and the cache budget matters more.
- For DeepSeek-V4-Flash, the official host store keeps routed experts in pinned host RAM and streams the needed experts into each rank's GPU cache.

## Tensor Parallelism and Model-Specific Constraints

- Expert distribution is not tensor parallelism.
- The generic `MoE` path uses one process and round robin expert placement across visible GPUs.
- Tensor-parallel loaders are model specific.
- DeepSeek-V4-Flash uses the official `torchrun --nproc-per-node 4 ...` path, one shard per rank, with `model{rank}-mp{world_size}.safetensors`.
- Its sparse-attention kernel is validated in repo with `--model-parallel 4`; mp1 is not covered by repo tests.
- `patch_moe_with_offload(..., world_size > 1)` all-reduces the expert output across ranks in the official DeepSeek-V4 path.
- Prefix-cache scaffolding is serving-side KV bookkeeping, but the current OpenAI request path has no active prefix-reuse integration. It does not change expert ownership or make multi-node inference supported; see [Serving](serving.md#prefix-caching).

## Validation

Generic single-process path:

```bash
pytest tests/python/unit/test_multi_gpu.py -q
```

Live multi-GPU smoke, one host:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tests/python/integration/test_multi_gpu_live.py
```

Single-process example with multiple visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/readme_example.py --checkpoint deepseek-ai/DeepSeek-V2-Lite-Chat --offload_dir /ssd/moe --max_new_tokens 64
```

Official DeepSeek-V4 tensor-parallel path:

```bash
torchrun --nproc-per-node 4 tests/python/v4/e2e_mp4_offload.py
```

## Unsupported Multi-Node

- This guide covers one host only.
- The public `MoE` path does not support running experts across separate machines.
- If you need multi-node expert parallel, you need a separate distributed design.
- The supported multi-GPU path is still single-server, even when multiple ranks are used for a model-specific loader.

## Related guides

- [Configuration](configuration.md)
- [Serving](serving.md)
- [Environment variables](environment-variables.md)
- [Troubleshooting](troubleshooting.md)
- [DeepSeek-V4-Flash](../moe_infinity/models/deepseek_v4/README.md)
