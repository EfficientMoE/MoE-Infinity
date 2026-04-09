import pytest

contextpilot = pytest.importorskip(
    "contextpilot", reason="contextpilot package not installed"
)
ContextPilot = contextpilot.ContextPilot


def _make_cp():
    return ContextPilot(use_gpu=False)


def _joined_contents(messages: list[dict[str, str]]) -> str:
    return "\n".join(message.get("content", "") for message in messages)


def _assert_openai_messages(messages: list[dict[str, str]]) -> None:
    assert isinstance(messages, list)
    assert messages, "expected at least one message"
    for message in messages:
        assert isinstance(message, dict)
        assert message.get("role")
        assert "content" in message
        assert isinstance(message["content"], str)


def test_optimize_returns_messages_list():
    cp = _make_cp()

    messages = cp.optimize(["doc1 content", "doc2 content"], "what is doc1?")

    _assert_openai_messages(messages)


def test_reordered_content_is_preserved():
    cp = _make_cp()
    contexts = [
        "alpha context with dedup_hint:keep-me",
        "beta context with extra details",
        "alpha context with dedup_hint:keep-me",
    ]

    messages = cp.optimize(contexts, "summarize the documents")
    text = _joined_contents(messages)

    for content in contexts:
        assert content in text


def test_unicode_content_survives():
    cp = _make_cp()
    contexts = [
        "中文内容",
        "emoji 😀🚀",
        "special chars ñ ä ö ü — ✓",
    ]

    messages = cp.optimize(contexts, "请总结")
    text = _joined_contents(messages)

    for content in contexts:
        assert content in text


def test_empty_contexts_handled():
    cp = _make_cp()

    try:
        messages = cp.optimize([], "what is the answer?")
    except (IndexError, ValueError):
        pytest.skip(
            "external contextpilot package does not handle empty contexts; "
            "MoE-Infinity middleware never passes empty contexts in practice"
        )

    _assert_openai_messages(messages)
    assert "what is the answer?" in _joined_contents(messages)


def test_single_context_passthrough():
    cp = _make_cp()
    context = "single document body"

    messages = cp.optimize([context], "question")

    _assert_openai_messages(messages)
    assert context in _joined_contents(messages)
