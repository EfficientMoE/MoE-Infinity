from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

from .official_offload_adapter import (
    OfficialExpertHostStore,
    patch_moe_with_offload,
)

_NATIVE_TP_SHARD_DIMS = {
    "attn_sink": 0,
    "embed": 0,
    "head": 0,
    "weights_proj": 0,
    "wo": 1,
    "wo_a": 0,
    "wo_b": 1,
    "wq": 0,
    "wq_b": 0,
    "wkv_b": 0,
}


def _make_null_routed_moe_init(official_module, orig_moe_init):
    def _null_routed_moe_init(self, layer_id, args):
        import torch.nn as nn

        orig_expert = official_module.Expert

        class _NullExpert(nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, x, weights=None):
                return x

        official_module.Expert = _NullExpert
        try:
            orig_moe_init(self, layer_id, args)
        finally:
            official_module.Expert = orig_expert
        for i in range(len(self.experts)):
            self.experts[i] = None
        self.shared_experts = orig_expert(
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    return _null_routed_moe_init


def _tensor_name(key: str) -> str:
    parts = key.split(".")
    if parts[-1] in {"weight", "scale", "bias"} and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def _shard_dim_for_key(key: str) -> int | None:
    return _NATIVE_TP_SHARD_DIMS.get(_tensor_name(key))


def _shard_tensor(
    tensor: torch.Tensor,
    key: str,
    shard_dim: int | None,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    if shard_dim is None or world_size == 1:
        return tensor
    if tensor.size(shard_dim) % world_size != 0:
        raise ValueError(
            f"tensor {key} dim {shard_dim} with size {tensor.size(shard_dim)} "
            f"is not divisible by world_size={world_size}"
        )
    shard_size = tensor.size(shard_dim) // world_size
    start = rank * shard_size
    return tensor.narrow(shard_dim, start, shard_size).contiguous()


def _load_non_expert_weights_sharded(
    model,
    single_ckpt_file: str,
    device: torch.device,
    world_size: int,
    rank: int,
):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    device = torch.device(device)

    with safe_open(single_ckpt_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if ".ffn.experts." in key:
                continue
            target = params.get(key)
            if target is None:
                target = buffers.get(key)
            if target is None:
                continue
            full_tensor = f.get_tensor(key)
            shard_dim = _shard_dim_for_key(key)
            tensor = _shard_tensor(
                full_tensor, key, shard_dim, world_size, rank
            )
            if tuple(tensor.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"shape mismatch for {key}: full={tuple(full_tensor.shape)} "
                    f"shard={tuple(tensor.shape)} target={tuple(target.shape)} "
                    f"shard_dim={shard_dim} world_size={world_size} rank={rank}"
                )
            target.copy_(tensor.to(device))
    return model


def _iter_moe_modules(model):
    for block in model.layers:
        if hasattr(block, "ffn") and hasattr(block.ffn, "experts_start_idx"):
            yield block.ffn
    for block in getattr(model, "mtp", []):
        if hasattr(block, "ffn") and hasattr(block.ffn, "experts_start_idx"):
            yield block.ffn


def _verify_local_expert_ranges(model, world_size: int, rank: int):
    for moe in _iter_moe_modules(model):
        if moe.n_local_experts * world_size != moe.n_routed_experts:
            raise RuntimeError(
                f"expert sharding mismatch: local={moe.n_local_experts} "
                f"world_size={world_size} total={moe.n_routed_experts}"
            )
        expected_start = rank * moe.n_local_experts
        expected_end = expected_start + moe.n_local_experts
        if (
            moe.experts_start_idx != expected_start
            or moe.experts_end_idx != expected_end
        ):
            raise RuntimeError(
                f"expert range mismatch: expected=({expected_start}, {expected_end}) "
                f"got=({moe.experts_start_idx}, {moe.experts_end_idx})"
            )


def _patch_sparse_attn_head_chunks(official_module):
    original_sparse_attn = official_module.sparse_attn
    if getattr(official_module, "_moe_infinity_sparse_attn_chunked", False):
        return

    def sparse_attn_head_chunks(
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        h = q.size(2)
        if h <= 16:
            return original_sparse_attn(
                q, kv, attn_sink, topk_idxs, softmax_scale
            )
        outputs = []
        for start in range(0, h, 16):
            width = min(16, h - start)
            outputs.append(
                original_sparse_attn(
                    q.narrow(2, start, width).contiguous(),
                    kv,
                    attn_sink.narrow(0, start, width).contiguous(),
                    topk_idxs,
                    softmax_scale,
                )
            )
        return torch.cat(outputs, dim=2)

    official_module.sparse_attn = sparse_attn_head_chunks
    official_module._moe_infinity_sparse_attn_chunked = True


def load_sharded_v4_flash(
    official_module,
    single_ckpt_file: str,
    config_path: str,
    device: torch.device,
    world_size: int,
    rank: int,
    max_resident_experts: int = 8,
    use_native=None,
    max_seq_len: int | None = None,
):
    world_size = int(world_size)
    rank = int(rank)
    device = torch.device(device)
    if world_size < 1:
        raise ValueError(f"invalid world_size={world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"invalid rank={rank} for world_size={world_size}")

    dist = official_module.dist
    if dist.is_initialized():
        actual_world_size = dist.get_world_size()
        actual_rank = dist.get_rank()
        if actual_world_size != world_size or actual_rank != rank:
            raise RuntimeError(
                f"torch.distributed rank/world_size mismatch: "
                f"expected ({rank}, {world_size}) got ({actual_rank}, {actual_world_size})"
            )
    elif world_size != 1 or rank != 0:
        raise RuntimeError(
            "torch.distributed must be initialized before loading TP-sharded V4-Flash"
        )

    official_module.world_size = world_size
    official_module.rank = rank
    _patch_sparse_attn_head_chunks(official_module)

    with open(config_path) as f:
        margs = official_module.ModelArgs(**json.load(f))
    margs.max_batch_size = 1
    if max_seq_len is not None:
        margs.max_seq_len = int(max_seq_len)

    orig_moe_init = official_module.MoE.__init__
    official_module.MoE.__init__ = _make_null_routed_moe_init(
        official_module, orig_moe_init
    )
    try:
        with torch.device(device):
            model = official_module.Transformer(margs)
    finally:
        official_module.MoE.__init__ = orig_moe_init
    torch.set_grad_enabled(False)

    _verify_local_expert_ranges(model, world_size, rank)

    ckpt_dir = os.path.dirname(single_ckpt_file)
    store = OfficialExpertHostStore(
        ckpt_dir,
        device,
        max_resident_experts=max_resident_experts,
        shard_file=single_ckpt_file,
    )
    patch_moe_with_offload(model, store, official_module, use_native=use_native)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _load_non_expert_weights_sharded(
        model,
        single_ckpt_file,
        device,
        world_size,
        rank,
    )
    return model, store
