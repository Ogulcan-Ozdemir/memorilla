"""Helpers shared by the Turkish-evaluation scripts: an append-only JSONL cache and JSON parsing of model replies."""

import hashlib
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any

LOGGER = logging.getLogger(__name__)

CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def stable_key(*parts: Any) -> str:
    """Hash JSON-serialisable parts into a cache key.

    Args:
        *parts: The parts, e.g. a task name, a model and a source text.

    Returns:
        The SHA-256 hex digest of their JSON encoding.
    """
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model reply that should be a single JSON object, tolerating a Markdown code fence around it.

    Args:
        text: The reply.

    Returns:
        The object.

    Raises:
        ValueError: If the reply is not a JSON object.
    """
    text = text.strip()
    fenced = CODE_FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Reply is not valid JSON: {error}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Reply is a JSON {type(value).__name__}, not an object.")
    return value


class JsonlCache:
    """A thread-safe, append-only key-value cache stored as JSON lines.

    Every ``put`` appends one line and flushes it, so an interrupted run keeps everything finished before the
    interruption. A truncated last line (from a crash mid-write) is skipped when the cache is read back.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the cache, reading any entries already on disk.

        Args:
            path: The JSONL file; created with its parent directories on the first ``put``.
        """
        self.path = Path(path)
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {}
        self._needs_newline = False
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as handle:
                handle.seek(-1, 2)
                self._needs_newline = handle.read(1) != b"\n"
        if self.path.exists():
            with open(self.path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    try:
                        entry = json.loads(line)
                        self._values[entry["key"]] = entry["value"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        LOGGER.warning("Skipping unreadable line %d of %s", number, self.path)

    def __contains__(self, key: str) -> bool:
        """Whether the cache holds a key.

        Args:
            key: The key.

        Returns:
            True when the key has a value.
        """
        return key in self._values

    def __len__(self) -> int:
        """Number of cached entries.

        Returns:
            The count.
        """
        return len(self._values)

    def get(self, key: str) -> Any:
        """Look up a value.

        Args:
            key: The key.

        Returns:
            The value, or None when the key is missing.
        """
        return self._values.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store a value and append it to the file.

        Args:
            key: The key.
            value: A JSON-serialisable value.
        """
        line = json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(("\n" if self._needs_newline else "") + line)
                handle.flush()
            self._needs_newline = False
            self._values[key] = value
