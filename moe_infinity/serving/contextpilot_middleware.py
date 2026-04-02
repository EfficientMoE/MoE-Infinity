from __future__ import annotations

import logging
import threading
from typing import Optional, cast

from contextpilot import ContextPilot

logger = logging.getLogger(__name__)


class ContextPilotMiddleware:
    _enabled: bool
    _cp: ContextPilot
    _lock: threading.Lock

    def __init__(self, use_gpu: bool = False, enabled: bool = True):
        self._enabled = bool(enabled)
        self._cp = ContextPilot(use_gpu=use_gpu)
        self._lock = threading.Lock()

    def process_chat_request(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if not self._enabled:
            return messages
        if not messages:
            return []

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

        try:
            with self._lock:
                optimized = self._cp.optimize(contexts=contexts, query=query)
            return [dict(message) for message in optimized]
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

    def _call_if_present(self, method_name: str, request_id: str) -> None:
        method = getattr(self._cp, method_name, None)
        if callable(method):
            _ = method(request_id)


__all__ = ["ContextPilotMiddleware"]
