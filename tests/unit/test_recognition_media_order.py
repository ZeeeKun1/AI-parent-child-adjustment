from __future__ import annotations

import pytest

from coregulation_poc.capture.media import MediaChunk, MediaKind
from coregulation_poc.runtime.recognition import _qwen_input_chunks
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
