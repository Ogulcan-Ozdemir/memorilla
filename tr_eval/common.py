"""Helpers shared by the Turkish-evaluation scripts: the LLM client, an append-only JSONL cache and JSON parsing.

The client talks to OpenRouter when ``OPENROUTER_API_KEY`` is set (Chat Completions, with OpenRouter's ``reasoning``
parameter and its per-request cost added to a running total), and otherwise to the OpenAI Responses API through
``personalizationv4.generate``.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from personalizationv4 import generate

if TYPE_CHECKING:
    from openai import OpenAI

LOGGER = logging.getLogger(__name__)

CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HOST = "openrouter.ai"


@dataclass
class UsageMeter:
    """Thread-safe running totals of the tokens and the cost (as reported by OpenRouter) of the requests so far."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Any) -> None:
        """Add one response's usage.

        Args:
            usage: The ``usage`` object of a Chat Completions response, or None.
        """
        with self._lock:
            self.requests += 1
            if usage is None:
                return
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.cost += float(getattr(usage, "cost", 0) or 0)

    def summary(self) -> str:
        """Describe the totals.

        Returns:
            A one-line summary.
        """
        return (
            f"{self.requests} requests, {self.prompt_tokens} prompt and {self.completion_tokens} completion tokens, "
            f"${self.cost:.4f}"
        )


USAGE = UsageMeter()


def create_client() -> "OpenAI":
    """Create the LLM client: OpenRouter when ``OPENROUTER_API_KEY`` is set, otherwise OpenAI (``OPENAI_API_KEY``).

    Returns:
        An ``openai.OpenAI`` client.

    Raises:
        ImportError: If the ``openai`` package is not installed.
    """
    key = os.environ.get(OPENROUTER_KEY_ENV)
    if not key:
        return generate.create_client()
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError("The Turkish evaluation needs the openai package: pip install -e '.[tr]'") from error
    LOGGER.info("Using OpenRouter (%s).", OPENROUTER_BASE_URL)
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def is_openrouter(client: Any) -> bool:
    """Whether a client points at OpenRouter.

    Args:
        client: An ``openai.OpenAI`` client (or a test double without ``base_url``).

    Returns:
        True for an OpenRouter client.
    """
    return OPENROUTER_HOST in str(getattr(client, "base_url", ""))


def complete(client: "OpenAI", prompt: str, model: str, reasoning_effort: str, max_retries: int) -> str:
    """Send one user prompt and return the reply text, through OpenRouter or the OpenAI Responses API.

    Args:
        client: The client from :func:`create_client`.
        prompt: User message.
        model: Model name (an OpenRouter id such as ``anthropic/claude-haiku-5.5`` for OpenRouter).
        reasoning_effort: Reasoning effort (``none`` disables reasoning).
        max_retries: Attempts before giving up; failed or empty responses are retried with exponential backoff.

    Returns:
        The non-empty reply text.

    Raises:
        RuntimeError: If every attempt fails or returns an empty response.
    """
    if not is_openrouter(client):
        return generate.complete(client, prompt, model, reasoning_effort, max_retries)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"reasoning": {"effort": reasoning_effort}},
            )
            USAGE.add(response.usage)
            text = response.choices[0].message.content if response.choices else None
            if text and text.strip():
                return text
            LOGGER.warning("Empty response from %s (attempt %d/%d)", model, attempt + 1, max_retries)
        except Exception as error:
            LOGGER.warning("Request to %s failed (attempt %d/%d): %s", model, attempt + 1, max_retries, error)
        if attempt + 1 < max_retries:
            time.sleep(generate.RETRY_DELAY_SECONDS * 2**attempt)
    raise RuntimeError(f"No response from {model} after {max_retries} attempts")


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
