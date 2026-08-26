"""Qwen text chat provider for text-based LLM calls.

This provider calls the Qwen Plus / Turbo text chat API (OpenAI-compatible
endpoint at dashscope.aliyuncs.com). It supports two usage patterns:

1. **Intervention message generation** — single-turn, no system message,
   used by :class:`StrategySelector` to rephrase strategy card templates.
2. **Stage-2 judgment** — system + user messages with JSON response format,
   used by the two-stage recognition pipeline to classify coregulation
   states from structured perception reports.

The provider is synchronous and blocking. When called from async code,
wrap it with ``asyncio.to_thread``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


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

    The provider uses the synchronous ``httpx`` client because the
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
            # Qwen 3.7 enables thinking by default. Realtime intervention
            # messages do not need a hidden reasoning pass; disabling it keeps
            # latency bounded and sends the approved wording directly.
            "enable_thinking": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = httpx.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
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

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> TextChatResult:
        """Send a system+user chat completion with optional JSON mode.

        Used by the stage-2 judgment model to classify coregulation states
        from structured perception reports.

        Parameters
        ----------
        system_prompt:
            The system message containing the classification framework and
            codebook.
        user_prompt:
            The user message containing the assessment-specific data
            (perception report, acoustic features, history).
        max_tokens:
            Override the instance default. The judgment JSON can be large,
            so callers typically pass a higher value (e.g. 2048).
        temperature:
            Override the instance default.
        json_mode:
            If True, request ``response_format: {"type": "json_object"}``.

        Returns
        -------
        TextChatResult
            The generated text (expected to be valid JSON) and metadata.

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
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Structured state judgment needs a concise JSON answer, not a
            # long reasoning trace. Qwen 3.7 otherwise enables thinking by
            # default and can exceed the synchronous request timeout.
            "enable_thinking": False,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
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
