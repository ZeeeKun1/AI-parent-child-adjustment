from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from coregulation_poc.providers.qwen_tts_realtime import QwenRealtimeTTSProvider


def _provider() -> QwenRealtimeTTSProvider:
    return QwenRealtimeTTSProvider(
        model="qwen3-tts-instruct-flash-realtime-2026-01-22",
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


def test_endpoint_uses_the_fixed_snapshot_model() -> None:
    parsed = urlparse(_provider().endpoint)

    assert parsed.hostname == "dashscope.aliyuncs.com"
    assert parse_qs(parsed.query) == {
        "model": ["qwen3-tts-instruct-flash-realtime-2026-01-22"]
    }


def test_session_update_fixes_voice_and_experiment_instruction() -> None:
    event = _provider()._session_update_event()

    assert event["type"] == "session.update"
    assert event["session"] == {
        "voice": "Maia",
        "mode": "commit",
        "language_type": "chinese",
        "response_format": "pcm",
        "sample_rate": 24000,
        "instructions": "冷静、中性、支持性",
        "optimize_instructions": False,
    }


def test_text_buffer_protocol_preserves_the_approved_message() -> None:
    provider = _provider()
    message = "我们先停一下，一起看看现在卡在哪里。"

    append = provider._input_text_append_event(message)
    commit = provider._input_text_commit_event()

    assert append["type"] == "input_text_buffer.append"
    assert append["text"] == message
    assert commit["type"] == "input_text_buffer.commit"
