from __future__ import annotations

from typing import Any

import httpx

from coregulation_poc.providers.qwen_text_chat import QwenTextChatProvider


def _response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "model": "qwen3.7-plus",
            "choices": [
                {
                    "message": {"content": '{"state":"normal"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_plain_generation_disables_thinking(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs["json"])
        request = httpx.Request("POST", url)
        return _response(request)

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = QwenTextChatProvider(api_key="test-key", model="qwen3.7-plus")

    provider.generate("hello")

    assert captured["enable_thinking"] is False


def test_structured_generation_disables_thinking(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs["json"])
        request = httpx.Request("POST", url)
        return _response(request)

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = QwenTextChatProvider(api_key="test-key", model="qwen3.7-plus")

    provider.generate_structured(system_prompt="rules", user_prompt="observations")

    assert captured["enable_thinking"] is False
    assert captured["response_format"] == {"type": "json_object"}
