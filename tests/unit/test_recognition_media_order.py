from __future__ import annotations

import pytest

from coregulation_poc.capture.media import MediaChunk, MediaKind
from coregulation_poc.runtime.recognition import (
    _qwen_input_chunks,
    _qwen_upload_chunks,
)
from coregulation_poc.runtime.window import MediaWindow


def _window(*chunks: MediaChunk) -> MediaWindow:
    return MediaWindow(
        chunks=chunks,
        start_ms=chunks[0].timestamp_ms,
        end_ms=chunks[-1].timestamp_ms,
    )


def test_qwen_input_starts_with_audio_when_window_starts_with_image() -> None:
    image_before_audio = MediaChunk(MediaKind.IMAGE, 1_000, b"image-before")
    first_audio = MediaChunk(MediaKind.AUDIO, 1_100, b"first-audio")
    image_after_audio = MediaChunk(MediaKind.IMAGE, 2_000, b"image-after")
    second_audio = MediaChunk(MediaKind.AUDIO, 2_100, b"second-audio")
    chunks = (image_before_audio, first_audio, image_after_audio, second_audio)

    ordered = _qwen_input_chunks(_window(*chunks))

    assert ordered[0] is first_audio
    assert len(ordered) == len(chunks)
    assert {id(chunk) for chunk in ordered} == {id(chunk) for chunk in chunks}
    assert ordered[1:] == (image_before_audio, image_after_audio, second_audio)


def test_qwen_input_preserves_window_when_audio_is_already_first() -> None:
    chunks = (
        MediaChunk(MediaKind.AUDIO, 1_000, b"audio"),
        MediaChunk(MediaKind.IMAGE, 1_001, b"image"),
    )

    assert _qwen_input_chunks(_window(*chunks)) == chunks


def test_qwen_input_rejects_image_only_window_before_api_call() -> None:
    window = _window(MediaChunk(MediaKind.IMAGE, 1_000, b"image"))

    with pytest.raises(ValueError, match="requires audio"):
        _qwen_input_chunks(window)


def test_qwen_upload_batches_one_second_of_audio_without_changing_bytes() -> None:
    audio_chunks = tuple(
        MediaChunk(MediaKind.AUDIO, index * 100, bytes([index]) * 3_200)
        for index in range(10)
    )
    window = _window(*audio_chunks)

    uploaded = _qwen_upload_chunks(window)

    assert len(uploaded) == 1
    assert uploaded[0].kind is MediaKind.AUDIO
    assert uploaded[0].timestamp_ms == 0
    assert uploaded[0].payload == b"".join(chunk.payload for chunk in audio_chunks)
    assert window.audio_chunks == audio_chunks


def test_qwen_upload_preserves_image_boundaries_and_all_payloads() -> None:
    chunks = (
        MediaChunk(MediaKind.AUDIO, 0, b"audio-a"),
        MediaChunk(MediaKind.AUDIO, 100, b"audio-b"),
        MediaChunk(MediaKind.IMAGE, 150, b"image"),
        MediaChunk(MediaKind.AUDIO, 200, b"audio-c"),
        MediaChunk(MediaKind.AUDIO, 300, b"audio-d"),
    )

    uploaded = _qwen_upload_chunks(_window(*chunks))

    assert [chunk.kind for chunk in uploaded] == [
        MediaKind.AUDIO,
        MediaKind.IMAGE,
        MediaKind.AUDIO,
    ]
    assert uploaded[0].payload == b"audio-aaudio-b"
    assert uploaded[1] is chunks[2]
    assert uploaded[2].payload == b"audio-caudio-d"


def test_qwen_upload_splits_audio_batches_at_one_second_limit() -> None:
    audio_chunks = tuple(
        MediaChunk(MediaKind.AUDIO, index * 100, bytes([index]) * 3_200)
        for index in range(11)
    )

    uploaded = _qwen_upload_chunks(_window(*audio_chunks))

    assert [len(chunk.payload) for chunk in uploaded] == [32_000, 3_200]
    assert b"".join(chunk.payload for chunk in uploaded) == b"".join(
        chunk.payload for chunk in audio_chunks
    )
