import importlib


def test_contextpilot_import_and_instantiation_with_moe_infinity():
    contextpilot = importlib.import_module("contextpilot")
    cp = contextpilot.ContextPilot(use_gpu=False)

    moe_infinity = importlib.import_module("moe_infinity")

    assert contextpilot is not None
    assert cp is not None
    assert moe_infinity is not None
