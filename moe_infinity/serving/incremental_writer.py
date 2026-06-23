"""Append-only JSONL writer for crash recovery."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class IncrementalWriter:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._file = self._path.open("a", encoding="utf-8")

    def save(
        self, seq_id: str, output_tokens: list[int], metadata: dict
    ) -> None:
        record = {
            "seq_id": seq_id,
            "output_tokens": output_tokens,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._file.write(line)
            self._file.write("\n")
            self._file.flush()
            os.fsync(self._file.fileno())

    def load_completed(self) -> set[str]:
        with self._lock:
            try:
                completed: set[str] = set()
                with self._path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        raw = raw.strip()
                        if not raw:
                            continue
                        obj = json.loads(raw)
                        seq_id = obj.get("seq_id")
                        if isinstance(seq_id, str):
                            completed.add(seq_id)
                return completed
            except FileNotFoundError:
                return set()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
