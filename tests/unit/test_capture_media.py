from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from coregulation_poc.capture.buffer import BoundedMediaBuffer
from coregulation_poc.capture.media import (
    MediaChunk,
    MediaFormat,
    MediaKind,
    MediaSequenceError,
    MediaSourceError,
    StrictTimestampNormalizer,
)
from coregulation_poc.capture.session import MediaCaptureSession
from coregulation_poc.capture.video_replay import (
    ReplayMedia,
    ReplayMediaSource,
    VideoMetadata,
)


def test_timestamp_normalizer_produces_one_strict_timeline() -> None:
    timeline = StrictTimestampNormalizer()

    assert [timeline.normalize(value) for value in (0, 0, 100, 50)] == [0, 1, 100, 101]


def test_media_format_rejects_zero_length_audio_chunks() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        MediaFormat(audio_chunk_ms=0)


def test_bounded_buffer_drops_old_images_but_not_audio() -> None:
    buffer = BoundedMediaBuffer(max_audio_chunks=2, max_image_chunks=1)
    buffer.put(MediaChunk(MediaKind.IMAGE, 0, b"old"))
    buffer.put(MediaChunk(MediaKind.IMAGE, 1, b"new"))
    buffer.put(MediaChunk(MediaKind.AUDIO, 2, b"audio"))

    assert buffer.get(timeout=0.01) == MediaChunk(MediaKind.IMAGE, 1, b"new")
    assert buffer.get(timeout=0.01) == MediaChunk(MediaKind.AUDIO, 2, b"audio")
    metrics = buffer.metrics()
    assert metrics.dropped_image_chunks == 1
    assert metrics.received_audio_chunks == 1


def test_bounded_buffer_stops_instead_of_silently_dropping_audio() -> None:
    buffer = BoundedMediaBuffer(max_audio_chunks=1, max_image_chunks=1)
    buffer.put(MediaChunk(MediaKind.AUDIO, 0, b"first"))

    with pytest.raises(MediaSourceError, match="Audio queue remained full"):
        buffer.put(MediaChunk(MediaKind.AUDIO, 1, b"second"), timeout=0.001)


def test_bounded_buffer_rejects_non_monotonic_timestamps() -> None:
    buffer = BoundedMediaBuffer(max_audio_chunks=2, max_image_chunks=2)
    buffer.put(MediaChunk(MediaKind.AUDIO, 5, b"first"))

    with pytest.raises(MediaSequenceError, match="strictly increasing"):
        buffer.put(MediaChunk(MediaKind.IMAGE, 5, b"duplicate"))


def test_local_video_implements_common_media_source_contract() -> None:
    chunk = MediaChunk(MediaKind.AUDIO, 0, b"pcm")
    source = ReplayMediaSource(
        ReplayMedia(
            metadata=VideoMetadata(
                path=Path(__file__),
                duration_ms=100,
                width=160,
                height=120,
                frame_rate=10,
                video_codec="mpeg4",
                audio_codec="pcm_s16le",
                audio_sample_rate=16_000,
            ),
            chunks=(chunk,),
            audio_chunk_count=1,
            image_count=0,
            audio_sample_rate=16_000,
            audio_chunk_ms=100,
            image_interval_ms=1_000,
            image_timestamp_labels=True,
        )
    )

    chunks = list(source.iter_chunks(Event()))

    assert chunks
    assert source.description.source_type == "local_video_replay"
    assert source.description.media_format.audio_sample_rate == 16_000


def test_capture_session_surfaces_device_disconnect_and_releases_source() -> None:
    class DisconnectingSource:
        closed = False

        @property
        def description(self):  # type: ignore[no-untyped-def]
            return None

        def iter_chunks(self, _stop_event: Event):  # type: ignore[no-untyped-def]
            yield MediaChunk(MediaKind.AUDIO, 0, b"pcm")
            raise MediaSourceError("device disconnected")

        def close(self) -> None:
            self.closed = True

    source = DisconnectingSource()
    session = MediaCaptureSession(
        source=source,
        buffer=BoundedMediaBuffer(max_audio_chunks=2, max_image_chunks=1),
    )
    session.start()

    assert session.read(timeout=0.5) == MediaChunk(MediaKind.AUDIO, 0, b"pcm")
    with pytest.raises(MediaSourceError, match="device disconnected"):
        session.read(timeout=0.5)
    session.stop()

    assert source.closed is True
    assert session.worker_alive is False
