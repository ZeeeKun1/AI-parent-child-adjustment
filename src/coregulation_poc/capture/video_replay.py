from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import av

from coregulation_poc.capture.imaging import encode_timestamped_jpeg
from coregulation_poc.capture.media import (
    MediaChunk,
    MediaFormat,
    MediaKind,
    MediaSourceDescription,
)


class VideoValidationError(ValueError):
    """Raised when a clip cannot support the intended audio-video test."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    duration_ms: int
    width: int
    height: int
    frame_rate: float
    video_codec: str
    audio_codec: str
    audio_sample_rate: int


@dataclass(frozen=True, slots=True)
class ReplayMedia:
    metadata: VideoMetadata
    chunks: tuple[MediaChunk, ...]
    audio_chunk_count: int
    image_count: int
    audio_sample_rate: int
    audio_chunk_ms: int
    image_interval_ms: int
    image_timestamp_labels: bool


@dataclass(frozen=True, slots=True)
class ReplayMediaSource:
    """Expose decoded local-video chunks through the common media-source contract."""

    media: ReplayMedia

    @property
    def description(self) -> MediaSourceDescription:
        return MediaSourceDescription(
            source_type="local_video_replay",
            source_id=self.media.metadata.path.name,
            media_format=MediaFormat(
                audio_sample_rate=self.media.audio_sample_rate,
                audio_chunk_ms=self.media.audio_chunk_ms,
                image_interval_ms=self.media.image_interval_ms,
                image_timestamp_labels=self.media.image_timestamp_labels,
            ),
        )

    def iter_chunks(self, stop_event: Event) -> Iterator[MediaChunk]:
        for chunk in self.media.chunks:
            if stop_event.is_set():
                return
            yield chunk

    def close(self) -> None:
        return None


def _frame_timestamp_ms(frame: av.AudioFrame | av.VideoFrame) -> int:
    if frame.pts is None or frame.time_base is None:
        return 0
    return max(0, round(float(frame.pts * frame.time_base) * 1000))


def _rate_as_float(rate: object | None) -> float:
    return float(rate) if rate is not None else 0.0


def inspect_video(path: Path) -> VideoMetadata:
    resolved = path.expanduser().resolve()
    if not resolved.is_absolute() or not resolved.is_file():
        raise VideoValidationError(f"Video file does not exist: {resolved}")

    try:
        with av.open(str(resolved)) as container:
            if not container.streams.video:
                raise VideoValidationError("The clip has no video track.")
            if not container.streams.audio:
                raise VideoValidationError("The clip has no audio track.")
            video = container.streams.video[0]
            audio = container.streams.audio[0]
            duration_ms = round(float(container.duration or 0) / av.time_base * 1000)
            return VideoMetadata(
                path=resolved,
                duration_ms=duration_ms,
                width=video.codec_context.width,
                height=video.codec_context.height,
                frame_rate=_rate_as_float(video.average_rate),
                video_codec=video.codec_context.name,
                audio_codec=audio.codec_context.name,
                audio_sample_rate=audio.codec_context.sample_rate,
            )
    except (av.FFmpegError, OSError) as exc:
        raise VideoValidationError(f"Cannot open video: {exc}") from exc


def decode_video_for_replay(
    path: Path,
    *,
    image_interval_ms: int = 1000,
    audio_chunk_ms: int = 100,
    audio_sample_rate: int = 16_000,
    max_image_bytes: int = 190_000,
) -> ReplayMedia:
    """Decode one clip into chronological PCM audio and JPEG image chunks."""
    if image_interval_ms < 250:
        raise ValueError("image_interval_ms must be at least 250")
    if audio_chunk_ms < 20:
        raise ValueError("audio_chunk_ms must be at least 20")

    metadata = inspect_video(path)
    chunks: list[MediaChunk] = []
    audio_bytes = bytearray()
    audio_samples_emitted = 0
    bytes_per_audio_chunk = audio_sample_rate * 2 * audio_chunk_ms // 1000
    next_image_ms = 0

    try:
        with av.open(str(metadata.path)) as container:
            audio_stream = container.streams.audio[0]
            video_stream = container.streams.video[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=audio_sample_rate)

            for packet in container.demux(audio_stream, video_stream):
                for frame in packet.decode():
                    if isinstance(frame, av.AudioFrame):
                        for converted in resampler.resample(frame):
                            audio_bytes.extend(converted.to_ndarray().tobytes())
                            while len(audio_bytes) >= bytes_per_audio_chunk:
                                timestamp_ms = audio_samples_emitted * 1000 // audio_sample_rate
                                payload = bytes(audio_bytes[:bytes_per_audio_chunk])
                                del audio_bytes[:bytes_per_audio_chunk]
                                chunks.append(MediaChunk(MediaKind.AUDIO, timestamp_ms, payload))
                                audio_samples_emitted += bytes_per_audio_chunk // 2
                    elif isinstance(frame, av.VideoFrame):
                        timestamp_ms = _frame_timestamp_ms(frame)
                        if timestamp_ms >= next_image_ms:
                            jpeg = encode_timestamped_jpeg(
                                frame.to_ndarray(format="bgr24"),
                                max_bytes=max_image_bytes,
                                timestamp_ms=timestamp_ms,
                            )
                            chunks.append(MediaChunk(MediaKind.IMAGE, timestamp_ms, jpeg))
                            while next_image_ms <= timestamp_ms:
                                next_image_ms += image_interval_ms

            for converted in resampler.resample(None):
                audio_bytes.extend(converted.to_ndarray().tobytes())
            if audio_bytes:
                padding = (-len(audio_bytes)) % bytes_per_audio_chunk
                audio_bytes.extend(b"\x00" * padding)
                while audio_bytes:
                    timestamp_ms = audio_samples_emitted * 1000 // audio_sample_rate
                    payload = bytes(audio_bytes[:bytes_per_audio_chunk])
                    del audio_bytes[:bytes_per_audio_chunk]
                    chunks.append(MediaChunk(MediaKind.AUDIO, timestamp_ms, payload))
                    audio_samples_emitted += bytes_per_audio_chunk // 2
    except (av.FFmpegError, OSError) as exc:
        raise VideoValidationError(f"Failed while decoding video: {exc}") from exc

    audio_count = sum(chunk.kind is MediaKind.AUDIO for chunk in chunks)
    image_count = sum(chunk.kind is MediaKind.IMAGE for chunk in chunks)
    if audio_count == 0 or image_count == 0:
        raise VideoValidationError("The clip did not yield both audio and image chunks.")

    chunks.sort(key=lambda chunk: (chunk.timestamp_ms, 0 if chunk.kind is MediaKind.AUDIO else 1))
    return ReplayMedia(
        metadata=metadata,
        chunks=tuple(chunks),
        audio_chunk_count=audio_count,
        image_count=image_count,
        audio_sample_rate=audio_sample_rate,
        audio_chunk_ms=audio_chunk_ms,
        image_interval_ms=image_interval_ms,
        image_timestamp_labels=True,
    )
