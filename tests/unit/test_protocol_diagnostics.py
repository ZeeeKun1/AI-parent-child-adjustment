import struct

from coregulation_poc.protocol_diagnostics import (
    build_session_update_probes,
    parse_close_frame,
    probe_session_update_variants,
    sanitize_for_log,
)


def test_session_update_probes_add_one_configuration_layer_at_a_time() -> None:
    probes = build_session_update_probes("diagnostic instruction")

    assert [probe["event"]["session"] for probe in probes] == [
        {},
        {"modalities": ["text"]},
        {
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
        },
        {
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": None,
        },
        {
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": None,
            "instructions": "diagnostic instruction",
        },
    ]
    assert {probe["event"]["type"] for probe in probes} == {"session.update"}


def test_close_frame_preserves_status_and_utf8_reason() -> None:
    payload = struct.pack("!H", 1008) + "参数无效".encode()

    assert parse_close_frame(payload) == {
        "status_code": 1008,
        "reason": "参数无效",
        "payload_valid": True,
    }


def test_sanitizer_redacts_credentials_recursively() -> None:
    value = {"headers": {"Authorization": "Bearer secret"}, "type": "session.update"}

    assert sanitize_for_log(value) == {
        "headers": {"Authorization": "[REDACTED]"},
        "type": "session.update",
    }


def test_missing_prerequisites_mark_every_probe_skipped() -> None:
    result = probe_session_update_variants(
        endpoint=None,
        api_key="",
        workspace_id=None,
        instructions="diagnostic instruction",
    )

    assert [probe["status"] for probe in result["probes"]] == ["SKIPPED"] * 5
    assert all("api_key" not in probe["sent_event"] for probe in result["probes"])
