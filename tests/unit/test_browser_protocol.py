from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from coregulation_poc.capture.imaging import encode_timestamped_jpeg
from coregulation_poc.capture.media import MediaChunk, MediaFormat, MediaKind
from coregulation_poc.web.protocol import (
    BrowserCaptureRecorder,
    BrowserProtocolError,
    decode_binary_packet,
    encode_binary_packet,
)


def test_binary_packet_round_trip() -> None:
    chunk = MediaChunk(MediaKind.AUDIO, 125, b"\x01\x02" * 1600)

    kind, timestamp_ms, payload = decode_binary_packet(
        encode_binary_packet(chunk),
        max_payload_bytes=3200,
    )

    assert kind is MediaKind.AUDIO
    assert timestamp_ms == 125
    assert payload == chunk.payload


def test_binary_packet_rejects_unknown_type() -> None:
    with pytest.raises(BrowserProtocolError, match="unknown media packet type"):
        decode_binary_packet(b"\x09" + b"\x00" * 8 + b"payload", max_payload_bytes=100)


def test_browser_recorder_validates_both_modalities_without_saving_payloads(
    tmp_path: Path,
) -> None:
    media_format = MediaFormat()
    recorder = BrowserCaptureRecorder(
        output_dir=tmp_path / "output",
        session_id="browser_test",
        media_format=media_format,
        max_image_bytes=750_000,
        client_capabilities={
            "secure_context": True,
            "page_version": "test",
            "device_label": "must-not-be-saved",
        },
    )
    recorder.start()
    audio = MediaChunk(
        MediaKind.AUDIO,
        0,
        b"\x00" * media_format.audio_chunk_bytes,
    )
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    jpeg = encode_timestamped_jpeg(image, timestamp_ms=0, max_bytes=750_000)

    recorder.accept_packet(encode_binary_packet(audio))
    recorder.accept_packet(encode_binary_packet(MediaChunk(MediaKind.IMAGE, 0, jpeg)))
    summary = recorder.finish(status="completed")

    manifest = json.loads((summary.run_dir / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((summary.run_dir / "result.json").read_text(encoding="utf-8"))
    events = (summary.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert summary.valid is True
    assert summary.normalized_timestamp_count == 1
    assert manifest["privacy"]["raw_media_saved"] is False
    assert "must-not-be-saved" not in json.dumps(manifest)
    assert result["api_called"] is False
    assert '"payload_saved": false' in events
    assert not list(summary.run_dir.glob("*.pcm"))
    assert not list(summary.run_dir.glob("*.jpg"))
    assert not list(summary.run_dir.glob("*.webm"))


def test_browser_recorder_rejects_wrong_audio_chunk_size(tmp_path: Path) -> None:
    recorder = BrowserCaptureRecorder(
        output_dir=tmp_path / "output",
        session_id="bad_audio",
        media_format=MediaFormat(),
        max_image_bytes=750_000,
    )
    recorder.start()

    with pytest.raises(BrowserProtocolError, match="audio packet size"):
        recorder.accept_packet(
            encode_binary_packet(MediaChunk(MediaKind.AUDIO, 0, b"\x00" * 100))
        )

    recorder.finish(status="failed", error="invalid audio")
