import inspect

import pytest

contextpilot = pytest.importorskip("contextpilot")


def test_installed_contextpilot_is_0_5():
    assert contextpilot.__version__.startswith("0.5.")


def test_serving_methods_match_0_5_contract():
    cp = contextpilot.ContextPilot(use_gpu=False)
    assert "conversation_id" in inspect.signature(cp.reorder).parameters
    assert "conversation_id" in inspect.signature(cp.deduplicate).parameters
    assert callable(cp.get_all_request_ids)
    assert callable(cp.remove_requests)
    assert not isinstance(getattr(cp, "live_index", None), dict)
    assert not hasattr(cp, "predict_prefix_reuse")


def test_optimize_replaces_chat_with_two_messages():
    cp = contextpilot.ContextPilot(use_gpu=False)
    messages = cp.optimize(
        ["alpha doc", "beta doc"],
        "what changed?",
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "what changed?"
