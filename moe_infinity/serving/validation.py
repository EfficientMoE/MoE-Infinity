from typing import Optional


class ContextLengthExceededError(ValueError):
    prompt_tokens: int
    max_tokens: int
    max_seq_length: int

    def __init__(
        self, prompt_tokens: int, max_tokens: int, max_seq_length: int
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_seq_length = max_seq_length
        total = prompt_tokens + max_tokens
        super().__init__(
            f"prompt ({prompt_tokens} tokens) + max_tokens ({max_tokens}) = {total} exceeds model max ({max_seq_length})"
        )


def validate_context_length(
    prompt_tokens: int,
    max_tokens: int,
    max_seq_length: int,
) -> None:
    if prompt_tokens + max_tokens > max_seq_length:
        raise ContextLengthExceededError(
            prompt_tokens, max_tokens, max_seq_length
        )


class InvalidRequestError(ValueError):
    param: Optional[str]

    def __init__(self, message: str, param: Optional[str] = None) -> None:
        self.param = param
        super().__init__(message)


def validate_required_params(max_tokens: Optional[int]) -> None:
    if max_tokens is None:
        raise InvalidRequestError(
            "max_tokens is required. Please provide a positive integer.",
            param="max_tokens",
        )


def validate_sampling_params(
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> None:
    if temperature is not None and not (0 <= temperature <= 2):
        raise InvalidRequestError(
            f"temperature must be between 0 and 2, got {temperature}",
            param="temperature",
        )
    if top_p is not None and not (0 < top_p <= 1):
        raise InvalidRequestError(
            f"top_p must be between 0 (exclusive) and 1 (inclusive), got {top_p}",
            param="top_p",
        )
    if top_k is not None and top_k <= 0:
        raise InvalidRequestError(
            f"top_k must be > 0, got {top_k}",
            param="top_k",
        )
    if max_tokens is not None and max_tokens <= 0:
        raise InvalidRequestError(
            f"max_tokens must be > 0, got {max_tokens}",
            param="max_tokens",
        )
