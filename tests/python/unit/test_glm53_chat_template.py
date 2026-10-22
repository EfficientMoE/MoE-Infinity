"""GLM-5.3 chat-template contract: reasoning_effort and clear_thinking.

GLM-5.3's only user-facing delta over GLM-5.2 (same GlmMoeDsa architecture) is
its chat template: `reasoning_effort` accepts `low`/`high` and falls back to
`max` for anything else (including unset), rendered as a
`<|system|>Reasoning Effort: <Level>` prefix; `clear_thinking` defaults to
false (replay prior-turn `reasoning_content`); `clear_thinking=True` strips
reasoning from turns at or before the last user message, leaving an empty
`<think></think>`. The fixture is the real `chat_template.jinja` from
zai-org/GLM-5.3.
"""

import os

import pytest

pytest.importorskip("transformers")
from transformers.utils.chat_template_utils import (  # noqa: E402
    render_jinja_template,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "fixtures",
    "glm_5_3",
    "chat_template.jinja",
)


def _render(messages, **kwargs):
    with open(FIXTURE) as f:
        template = f.read()
    rendered = render_jinja_template(
        conversations=[messages],
        chat_template=template,
        add_generation_prompt=True,
        **kwargs,
    )
    out = rendered[0] if isinstance(rendered, tuple) else rendered
    return out[0]


_USER = [{"role": "user", "content": "hello"}]


def test_default_reasoning_effort_is_max():
    assert "<|system|>Reasoning Effort: Max" in _render(_USER)


@pytest.mark.parametrize(
    ("effort", "label"), [("low", "Low"), ("high", "High")]
)
def test_explicit_reasoning_effort(effort, label):
    out = _render(_USER, reasoning_effort=effort)
    assert f"<|system|>Reasoning Effort: {label}" in out


def test_invalid_reasoning_effort_falls_back_to_max():
    out = _render(_USER, reasoning_effort="medium")
    assert "<|system|>Reasoning Effort: Max" in out


def test_clear_thinking_strips_prior_reasoning():
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "a1",
            "reasoning_content": "PRIOR_THOUGHTS",
        },
        {"role": "user", "content": "q2"},
    ]
    assert "PRIOR_THOUGHTS" in _render(messages)
    cleared = _render(messages, clear_thinking=True)
    assert "PRIOR_THOUGHTS" not in cleared
    assert "<think></think>" in cleared
