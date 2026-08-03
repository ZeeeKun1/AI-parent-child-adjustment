from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from threading import Event
from typing import Protocol


class MediaSourceError(OSError):
    """Raised when a media source cannot start or stops unexpectedly."""


class MediaSequenceError(ValueError):
    """Raised when a source emits invalid or non-monotonic media chunks."""


class MediaKind(StrEnum):
    AUDIO = "audio"
    IMAGE = "image"


class SpeakerRole(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MediaChunk:
    """One API-ready media payload on the shared capture timeline."""

    kind: MediaKind
    timestamp_ms: int
    payload: bytes

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not self.payload:
            raise ValueError("media payload must not be empty")


@dataclass(frozen=True, slots=True)
class MediaFormat:
    audio_sample_rate: int = 16_000
    audio_channels: int = 1
    audio_sample_width_bytes: int = 2
    audio_chunk_ms: int = 100
    image_interval_ms: int = 1_000
    image_max_width: int = 1_280
    image_max_height: int = 720
    image_encoding: str = "jpeg"
    image_timestamp_labels: bool = True

    def __post_init__(self) -> None:
        if self.audio_sample_rate < 1:
            raise ValueError("audio_sample_rate must be positive")
        if self.audio_channels < 1 or self.audio_sample_width_bytes < 1:
            raise ValueError("audio channel count and sample width must be positive")
        if self.audio_chunk_ms < 20:
            raise ValueError("audio_chunk_ms must be at least 20")
        if self.image_interval_ms < 250:
            raise ValueError("image_interval_ms must be at least 250")
        if self.image_max_width < 1 or self.image_max_height < 41:
            raise ValueError("image dimensions are too small for timestamped frames")
        if self.image_encoding != "jpeg":
            raise ValueError("the realtime provider currently requires JPEG images")

    @property
    def audio_chunk_bytes(self) -> int:
        return (
            self.audio_sample_rate
            * self.audio_channels
            * self.audio_sample_width_bytes
            * self.audio_chunk_ms
            // 1_000
        )


@dataclass(frozen=True, slots=True)
class MediaSourceDescription:
    source_type: str
    source_id: str
    media_format: MediaFormat


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """Reserved output contract for a later validated speaker-binding component."""

    start_ms: int
    end_ms: int
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("speaker segment timestamps are invalid")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("speaker segment confidence must be between zero and one")


class MediaSource(Protocol):
    """Synchronous source contract shared by replay and live capture."""

    @property
    def description(self) -> MediaSourceDescription: ...

    def iter_chunks(self, stop_event: Event) -> Iterator[MediaChunk]: ...

    def close(self) -> None: ...


class StrictTimestampNormalizer:
    """Map candidate timestamps onto one strictly increasing millisecond timeline."""

    def __init__(self) -> None:
        self._last_timestamp_ms = -1

    @property
    def last_timestamp_ms(self) -> int | None:
        return None if self._last_timestamp_ms < 0 else self._last_timestamp_ms

    def normalize(self, candidate_ms: int) -> int:
        if candidate_ms < 0:
            raise ValueError("candidate timestamp must be non-negative")
        timestamp_ms = max(candidate_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms
