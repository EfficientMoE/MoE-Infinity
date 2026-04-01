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
