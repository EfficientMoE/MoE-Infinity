from .attention_backend import (
    AttentionBackend,
    AttentionMetadata,
    PlaceholderAttentionBackend,
)

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "PlaceholderAttentionBackend",
    "OffloadEngine",
]


def __getattr__(name: str):
    if name == "OffloadEngine":
        import importlib

        module = importlib.import_module("moe_infinity.runtime.model_offload")
        return getattr(module, "OffloadEngine")
    raise AttributeError(
        f"module 'moe_infinity.runtime' has no attribute {name!r}"
    )
