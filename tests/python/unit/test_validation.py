# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import pytest

from moe_infinity.serving.validation import (
    ContextLengthExceededError,
    InvalidRequestError,
    validate_context_length,
    validate_required_params,
    validate_sampling_params,
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


def test_validate_sampling_params_raises_on_temperature_above_upper_bound() -> (
    None
):
    with pytest.raises(InvalidRequestError) as exc_info:
        validate_sampling_params(temperature=3.0)

    assert exc_info.value.param == "temperature"


def test_validate_sampling_params_accepts_temperature_within_bounds() -> None:
    validate_sampling_params(temperature=0.7)


def test_validate_required_params_raises_when_max_tokens_missing() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        validate_required_params(None)

    assert exc_info.value.param == "max_tokens"


def test_validate_required_params_accepts_positive_max_tokens() -> None:
    validate_required_params(10)


def test_validate_sampling_params_raises_on_top_p_exclusive_lower_bound() -> (
    None
):
    with pytest.raises(InvalidRequestError) as exc_info:
        validate_sampling_params(top_p=0.0)

    assert exc_info.value.param == "top_p"


def test_validate_sampling_params_accepts_top_p_inclusive_upper_bound() -> None:
    validate_sampling_params(top_p=1.0)
