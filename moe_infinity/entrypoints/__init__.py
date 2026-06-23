__all__ = ["MoE"]


def __getattr__(name: str):
    import importlib

    if name == "MoE":
        module = importlib.import_module(
            "moe_infinity.entrypoints.big_modeling"
        )
        return getattr(module, "MoE")
    raise AttributeError(
        f"module 'moe_infinity.entrypoints' has no attribute {name!r}"
    )
