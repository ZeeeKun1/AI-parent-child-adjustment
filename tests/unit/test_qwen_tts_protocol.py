from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
import websocket

from coregulation_poc.providers.qwen_tts_realtime import QwenRealtimeTTSProvider, TTSServiceError


def _provider(model: str = "qwen3-tts-flash-realtime") -> QwenRealtimeTTSProvider:
    return QwenRealtimeTTSProvider(
        model=model,
        voice="Maia",
        api_key="test-key",
        workspace_id="test-workspace",
        base_url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        language_type="chinese",
        response_format="pcm",
        sample_rate_hz=24000,
        mode="commit",
        instructions="冷静、中性、支持性",
        optimize_instructions=False,
    )


def test_endpoint_uses_the_selected_stable_flash_model() -> None:
    parsed = urlparse(_provider().endpoint)

    assert parsed.hostname == "dashscope.aliyuncs.com"
    assert parse_qs(parsed.query) == {"model": ["qwen3-tts-flash-realtime"]}


def test_flash_preserves_maia_without_sending_unsupported_style_instructions() -> None:
    event = _provider()._session_update_event()

    assert event["type"] == "session.update"
    assert event["session"] == {
        "voice": "Maia",
        "mode": "commit",
        "language_type": "chinese",
        "response_format": "pcm",
        "sample_rate": 24000,
    }
    assert _provider().supports_instructions is False


def test_instruct_models_remain_compatible_with_style_instructions() -> None:
    event = _provider("qwen3-tts-instruct-flash-realtime")._session_update_event()
    assert event["session"]["instructions"] == "冷静、中性、支持性"
    assert event["session"]["optimize_instructions"] is False
    assert event["session"]["voice"] == "Maia"


def test_text_buffer_protocol_preserves_the_approved_message() -> None:
    provider = _provider()
    message = "我们先停一下，一起看看现在卡在哪里。"

    append = provider._input_text_append_event(message)
    commit = provider._input_text_commit_event()

    assert append["type"] == "input_text_buffer.append"
    assert append["text"] == message
    assert commit["type"] == "input_text_buffer.commit"


def test_service_rejection_without_type_preserves_actual_error() -> None:
    class Connection:
        def recv_data_frame(self, **kwargs):
            return websocket.ABNF.OPCODE_TEXT, SimpleNamespace(
                data=json.dumps({"code": "InvalidParameter", "message": "Model not exist."})
            )

    with pytest.raises(TTSServiceError, match="Model not exist") as error:
        _provider()._receive(Connection())
    assert error.value.code == "InvalidParameter"
