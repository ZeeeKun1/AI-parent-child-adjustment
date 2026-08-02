from coregulation_poc.providers.qwen_omni_realtime import QwenOmniRealtimeProvider


def _provider() -> QwenOmniRealtimeProvider:
    return QwenOmniRealtimeProvider(
        model="qwen3.5-omni-flash-realtime",
        api_key="sk-test",
        workspace_id="ws-test",
        base_url="wss://ws-test.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        instructions="test instruction",
    )


def test_session_update_contains_only_documented_fields() -> None:
    event = _provider()._session_update_event()

    assert event["type"] == "session.update"
    assert event["session"] == {
        "modalities": ["text"],
        "input_audio_format": "pcm",
        "output_audio_format": "pcm",
        "turn_detection": None,
        "instructions": "test instruction",
    }


def test_response_create_has_no_unsupported_response_object() -> None:
    event = _provider()._event("response.create")

    assert set(event) == {"event_id", "type"}
