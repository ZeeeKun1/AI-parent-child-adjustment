"""Qwen text chat provider for constrained intervention message generation.

This provider calls the Qwen Plus / Turbo text chat API (OpenAI-compatible
endpoint at dashscope.aliyuncs.com) to rephrase a strategy card's approved
template into a context-adaptive variant.

The provider is synchronous and blocking.  It is designed to be called
from :class:`StrategySelector` when an LLM is available.  All network
errors, timeouts, and validation failures are caught by the caller and
result in a fallback to the approved template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class TextChatResult:
    """Result of a text chat completion call."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_latency_ms: int
    finish_reason: str | None


class QwenTextChatProvider:
    """Call the OpenAI-compatible Qwen text chat endpoint.

    The provider uses the synchronous ``requests`` library because the
    intervention message generation is a one-shot blocking call within
    the strategy selector's synchronous flow.
    """

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.3,
        max_tokens: int = 128,
        timeout_seconds: float = 10,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> TextChatResult:
        """Send a single-turn chat completion request and return the text.

        Parameters
        ----------
        prompt:
            The full prompt string (system + user combined).

        Returns
        -------
        TextChatResult
            The generated text and metadata.

        Raises
        ------
        ConnectionError
            If the HTTP request fails or times out.
        ValueError
            If the response is malformed or empty.
        """
        import time

        started = time.monotonic()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = requests.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"Qwen text chat request failed: {exc}") from exc

        if response.status_code != 200:
            raise ConnectionError(
                f"Qwen text chat returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ConnectionError(f"Qwen text chat returned invalid JSON: {exc}") from exc

        choices = body.get("choices")
        if not choices or not isinstance(choices, list):
            raise ValueError("Qwen text chat response missing choices")

        first = choices[0]
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("Qwen text chat response missing message")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen text chat returned empty content")

        finish_reason = first.get("finish_reason")
        usage = body.get("usage", {})
        latency_ms = round((time.monotonic() - started) * 1000)

        return TextChatResult(
            text=content.strip(),
            model=body.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            completion_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
            total_latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
