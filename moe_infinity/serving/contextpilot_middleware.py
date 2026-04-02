from __future__ import annotations

import logging
import threading
from typing import Optional, cast

from contextpilot import ContextPilot

logger = logging.getLogger(__name__)


class ContextPilotMiddleware:
    _enabled: bool
    _reorder_enabled: bool
    _dedup_enabled: bool
    _cp: ContextPilot
    _lock: threading.Lock
    token_savings_total: int
    _requests_processed: int
    _savings_pct_total: float

    def __init__(
        self,
        use_gpu: bool = False,
        enabled: bool = True,
        dedup_enabled: bool = True,
        reorder_enabled: bool = True,
    ):
        self._enabled = bool(enabled)
        self._reorder_enabled = bool(reorder_enabled)
        self._dedup_enabled = bool(dedup_enabled)
        self._cp = ContextPilot(use_gpu=use_gpu)
        self._lock = threading.Lock()
        self.token_savings_total = 0
        self._requests_processed = 0
        self._savings_pct_total = 0.0

    def process_chat_request(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if not self._enabled:
            return messages
        if not messages:
            return []

        try:
            if self._reorder_enabled:
                optimized_messages = self._reorder_messages(messages)
            else:
                optimized_messages = [dict(message) for message in messages]

            request_tokens_saved = 0
            request_savings_pct = 0.0
            if self._dedup_enabled:
                (
                    optimized_messages,
                    request_tokens_saved,
                    request_savings_pct,
                ) = self._deduplicate_messages(optimized_messages)
                logger.info(
                    "CP dedup: removed ~%d duplicate tokens (%.1f%%)",
                    request_tokens_saved,
                    request_savings_pct,
                )

            self.token_savings_total += request_tokens_saved
            self._requests_processed += 1
            self._savings_pct_total += request_savings_pct
            return optimized_messages
        except Exception as exc:
            logger.warning("ContextPilot optimize failed: %s", exc)
        return messages

    def process_completion_request(self, prompt: str) -> str:
        if not self._enabled:
            return prompt

        try:
            with self._lock:
                optimized = self._cp.optimize(contexts=[], query=str(prompt))
            if not optimized:
                return prompt

            for message in reversed(optimized):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            return prompt
        except Exception as exc:
            logger.warning("ContextPilot completion optimize failed: %s", exc)
            return prompt

    def on_request_complete(self, request_id: str) -> None:
        if not self._enabled:
            return

        with self._lock:
            try:
                self._call_if_present("on_request_complete", request_id)
                self._call_if_present("evict", request_id)
                self._call_if_present("remove_request", request_id)
                self._call_if_present("remove", request_id)
                self._call_if_present("delete", request_id)
                live_index_obj = getattr(self._cp, "live_index", None)
                if isinstance(live_index_obj, dict):
                    live_index = cast(dict[str, object], live_index_obj)
                    _ = live_index.pop(request_id, None)
            except Exception as exc:
                logger.warning(
                    "ContextPilot request cleanup failed for %s: %s",
                    request_id,
                    exc,
                )

    def is_enabled(self) -> bool:
        return self._enabled

    def get_token_savings(self) -> dict[str, object]:
        avg_savings_pct = 0.0
        if self._requests_processed > 0:
            avg_savings_pct = self._savings_pct_total / self._requests_processed
        return {
            "total_tokens_saved": int(self.token_savings_total),
            "avg_savings_pct": float(avg_savings_pct),
            "requests_processed": int(self._requests_processed),
        }

    @staticmethod
    def _extract_query(
        messages: list[dict[str, str]],
    ) -> tuple[str, Optional[int]]:
        last_user_index: Optional[int] = None
        query = ""

        for index, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if content is None:
                continue
            last_user_index = index
            query = str(content)

        return query, last_user_index

    def _reorder_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        query, query_index = self._extract_query(messages)
        contexts: list[str] = []

        for index, message in enumerate(messages):
            content = message.get("content")
            if content is None:
                continue

            role = message.get("role")
            if (
                query_index is not None
                and index == query_index
                and role == "user"
            ):
                continue
            contexts.append(str(content))

        with self._lock:
            optimized = self._cp.optimize(contexts=contexts, query=query)
        return [dict(message) for message in optimized]

    def _deduplicate_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], int, float]:
        with self._lock:
            deduplicate_fn = getattr(self._cp, "deduplicate", None)
            if callable(deduplicate_fn):
                deduped = deduplicate_fn(messages)
                if isinstance(deduped, list):
                    normalized: list[dict[str, str]] = []
                    deduped_list = cast(list[object], deduped)
                    for candidate in deduped_list:
                        if not isinstance(candidate, dict):
                            continue
                        message_dict = cast(dict[object, object], candidate)
                        role_obj = message_dict.get("role")
                        content_obj = message_dict.get("content")
                        normalized.append(
                            {
                                "role": (
                                    "" if role_obj is None else str(role_obj)
                                ),
                                "content": (
                                    ""
                                    if content_obj is None
                                    else str(content_obj)
                                ),
                            }
                        )
                    tokens_saved, pct = self._estimate_tokens_saved(
                        messages, normalized
                    )
                    return normalized, tokens_saved, pct

        return self._fallback_deduplicate(messages)

    @staticmethod
    def _fallback_deduplicate(
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], int, float]:
        seen_content_to_index: dict[str, int] = {}
        deduped_messages: list[dict[str, str]] = []
        saved_tokens = 0
        original_token_estimate = 0

        for index, message in enumerate(messages):
            new_message = dict(message)
            content = new_message.get("content")
            if not isinstance(content, str):
                deduped_messages.append(new_message)
                continue

            estimated_tokens = len(content) // 4
            original_token_estimate += estimated_tokens
            first_seen_index = seen_content_to_index.get(content)
            if first_seen_index is None:
                seen_content_to_index[content] = index
                deduped_messages.append(new_message)
                continue

            saved_tokens += estimated_tokens
            new_message["content"] = (
                f"[Deduplicated content; same as message #{first_seen_index}]"
            )
            deduped_messages.append(new_message)

        savings_pct = 0.0
        if original_token_estimate > 0:
            savings_pct = (saved_tokens / original_token_estimate) * 100.0
        return deduped_messages, saved_tokens, savings_pct

    @staticmethod
    def _estimate_tokens_saved(
        before: list[dict[str, str]],
        after: list[dict[str, str]],
    ) -> tuple[int, float]:
        before_tokens = 0
        after_tokens = 0

        for message in before:
            content = message.get("content")
            if isinstance(content, str):
                before_tokens += len(content) // 4

        for message in after:
            content = message.get("content")
            if isinstance(content, str):
                after_tokens += len(content) // 4

        saved_tokens = max(0, before_tokens - after_tokens)
        savings_pct = 0.0
        if before_tokens > 0:
            savings_pct = (saved_tokens / before_tokens) * 100.0
        return saved_tokens, savings_pct

    def _call_if_present(self, method_name: str, request_id: str) -> None:
        method = getattr(self._cp, method_name, None)
        if callable(method):
            _ = method(request_id)


__all__ = ["ContextPilotMiddleware"]
