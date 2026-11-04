"""Real shapes of the latest model of every MoE family MoE-Infinity supports.

Values are copied from each checkpoint's config.json (HF hub, Nov 2026).
Only fields that determine kernel shapes are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Attn:
    kind: str  # "gqa" | "mla_dsa" | "dsv4" | "minimax_sparse"
    heads: int
    kv_heads: int
    head_dim: int
    full_layers: int  # layers using full (global) softmax attention
    swa_layers: int = 0  # sliding-window layers
    window: int = 0
    linear_layers: int = 0  # linear-attention (GDN / KDA) layers
    v_head_dim: int | None = None
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    rope_dim: int | None = None
    index_topk: int | None = None
    sinks: bool = False


@dataclass(frozen=True)
class MoEModel:
    key: str
    hf_id: str
    arch: str
    expert_type: int  # moe_store registry type (5 = gate/up/down, 6 = GPT-OSS)
    hidden: int
    layers: int
    moe_layers: int
    experts: int
    top_k: int
    inter: int  # routed expert intermediate size
    shared: int = 0  # number of shared experts
    shared_inter: int = 0
    expert_dtype: str = "bf16"  # checkpoint format of routed experts
    max_ctx: int = 0
    attn: Attn = field(default_factory=lambda: Attn("gqa", 0, 0, 0, 0))


MODELS: dict[str, MoEModel] = {
    m.key: m
    for m in [
        MoEModel(
            "qwen3.5-35b-a3b",
            "Qwen/Qwen3.5-35B-A3B",
            "Qwen3_5MoeForConditionalGeneration",
            5,
            2048,
            40,
            40,
            256,
            8,
            512,
            shared=1,
            shared_inter=512,
            max_ctx=262144,
            attn=Attn("gqa", 16, 2, 256, full_layers=10, linear_layers=30),
        ),
        MoEModel(
            "qwen3.5-122b-a10b",
            "Qwen/Qwen3.5-122B-A10B",
            "Qwen3_5MoeForConditionalGeneration",
            5,
            3072,
            48,
            48,
            256,
            8,
            1024,
            shared=1,
            shared_inter=1024,
            max_ctx=262144,
            attn=Attn("gqa", 32, 2, 256, full_layers=12, linear_layers=36),
        ),
        MoEModel(
            "qwen3.5-397b-a17b",
            "Qwen/Qwen3.5-397B-A17B",
            "Qwen3_5MoeForConditionalGeneration",
            5,
            4096,
            60,
            60,
            512,
            10,
            1024,
            shared=1,
            shared_inter=1024,
            max_ctx=262144,
            attn=Attn("gqa", 32, 2, 256, full_layers=15, linear_layers=45),
        ),
        MoEModel(
            "qwen3-30b-a3b",
            "Qwen/Qwen3-30B-A3B",
            "Qwen3MoeForCausalLM",
            5,
            2048,
            48,
            48,
            128,
            8,
            768,
            max_ctx=40960,
            attn=Attn("gqa", 32, 4, 128, full_layers=48),
        ),
        MoEModel(
            "qwen3-235b-a22b",
            "Qwen/Qwen3-235B-A22B",
            "Qwen3MoeForCausalLM",
            5,
            4096,
            94,
            94,
            128,
            8,
            1536,
            max_ctx=40960,
            attn=Attn("gqa", 64, 4, 128, full_layers=94),
        ),
        MoEModel(
            "gpt-oss-20b",
            "openai/gpt-oss-20b",
            "GptOssForCausalLM",
            6,
            2880,
            24,
            24,
            32,
            4,
            2880,
            expert_dtype="mxfp4",
            max_ctx=131072,
            attn=Attn(
                "gqa",
                64,
                8,
                64,
                full_layers=12,
                swa_layers=12,
                window=128,
                sinks=True,
            ),
        ),
        MoEModel(
            "gpt-oss-120b",
            "openai/gpt-oss-120b",
            "GptOssForCausalLM",
            6,
            2880,
            36,
            36,
            128,
            4,
            2880,
            expert_dtype="mxfp4",
            max_ctx=131072,
            attn=Attn(
                "gqa",
                64,
                8,
                64,
                full_layers=18,
                swa_layers=18,
                window=128,
                sinks=True,
            ),
        ),
        MoEModel(
            "glm-5.3",
            "zai-org/GLM-5.3 (=GLM-5.2-FP8 shapes)",
            "GlmMoeDsaForCausalLM",
            5,
            6144,
            78,
            75,
            256,
            8,
            2048,
            shared=1,
            shared_inter=2048,
            expert_dtype="fp8_block128",
            max_ctx=1048576,
            attn=Attn(
                "mla_dsa",
                64,
                64,
                192,
                full_layers=78,
                v_head_dim=256,
                kv_lora_rank=512,
                q_lora_rank=2048,
                rope_dim=64,
                index_topk=2048,
            ),
        ),
        MoEModel(
            "glm-5.3-flash",
            "zai-org/GLM-5.3-Flash",
            "Glm5NextForConditionalGeneration",
            5,
            4096,
            45,
            42,
            288,
            8,
            2048,
            shared=1,
            shared_inter=2048,
            expert_dtype="fp8_block128",
            max_ctx=1048576,
            attn=Attn(
                "mla_dsa",
                64,
                64,
                256,
                full_layers=11,
                linear_layers=34,
                v_head_dim=256,
                kv_lora_rank=512,
                q_lora_rank=1536,
                index_topk=2048,
            ),
        ),
        MoEModel(
            "deepseek-v4-flash",
            "deepseek-ai/DeepSeek-V4-Flash",
            "DeepseekV4ForCausalLM",
            5,
            4096,
            43,
            43,
            256,
            6,
            2048,
            shared=1,
            shared_inter=2048,
            expert_dtype="fp4_e2m1_ue8m0",
            max_ctx=1048576,
            attn=Attn(
                "dsv4",
                64,
                1,
                512,
                full_layers=43,
                window=128,
                q_lora_rank=1024,
                rope_dim=64,
                index_topk=512,
            ),
        ),
        MoEModel(
            "deepseek-v4.1-flash",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "DeepseekV41ForCausalLM",
            5,
            5120,
            40,
            40,
            384,
            6,
            2304,
            shared=1,
            shared_inter=2304,
            expert_dtype="fp4_e2m1_ue8m0",
            max_ctx=1048576,
            attn=Attn(
                "dsv4",
                64,
                1,
                512,
                full_layers=40,
                window=128,
                q_lora_rank=1280,
                rope_dim=64,
                index_topk=512,
            ),
        ),
        MoEModel(
            "minimax-m3",
            "MiniMaxAI/MiniMax-M3",
            "MiniMaxM3SparseForConditionalGeneration",
            5,
            6144,
            60,
            60,
            128,
            4,
            3072,
            shared=1,
            shared_inter=3072,
            max_ctx=1048576,
            attn=Attn("minimax_sparse", 64, 4, 128, full_layers=60),
        ),
        MoEModel(
            "mixtral-8x7b",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "MixtralForCausalLM",
            4,
            4096,
            32,
            32,
            8,
            2,
            14336,
            max_ctx=32768,
            attn=Attn("gqa", 32, 8, 128, full_layers=32),
        ),
    ]
}

PREFILL_TOKENS = {
    "1K": 1024,
    "2K": 2048,
    "9K": 9216,
    "12K": 12288,
    "50K": 51200,
    "200K": 204800,
}
DECODE_BATCHES = (1, 4, 16, 64, 256)
DECODE_CONTEXTS = PREFILL_TOKENS


def active_experts(m: MoEModel, tokens: int) -> float:
    """Expected number of distinct routed experts hit by `tokens` tokens."""
    return m.experts * (1.0 - (1.0 - m.top_k / m.experts) ** tokens)


def tokens_per_active_expert(m: MoEModel, tokens: int) -> float:
    return tokens * m.top_k / active_experts(m, tokens)
