# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import pytest

from moe_infinity.serving.validation import (
    ContextLengthExceededError,
    validate_context_length,
)


def test_validate_context_length_raises_when_over_limit() -> None:
    with pytest.raises(ContextLengthExceededError):
        validate_context_length(
            prompt_tokens=4000, max_tokens=200, max_seq_length=4096
        )


def test_validate_context_length_passes_when_within_limit() -> None:
    validate_context_length(
        prompt_tokens=100, max_tokens=50, max_seq_length=4096
    )


def test_validate_context_length_allows_zero_generation_at_boundary() -> None:
    validate_context_length(
        prompt_tokens=4096, max_tokens=0, max_seq_length=4096
    )


def test_validate_context_length_raises_when_exceeds_by_one() -> None:
    with pytest.raises(ContextLengthExceededError):
        validate_context_length(
            prompt_tokens=4096, max_tokens=1, max_seq_length=4096
        )


def test_validate_context_length_error_message_contains_actual_values() -> None:
    with pytest.raises(ContextLengthExceededError) as exc_info:
        validate_context_length(
            prompt_tokens=4000, max_tokens=200, max_seq_length=4096
        )

    message = str(exc_info.value)
    assert "4000" in message
    assert "200" in message
    assert "4096" in message


def test_validate_context_length_allows_empty_prompt_with_valid_generation() -> (
    None
):
    validate_context_length(
        prompt_tokens=0, max_tokens=100, max_seq_length=4096
    )
