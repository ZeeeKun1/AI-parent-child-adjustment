from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from coregulation_poc.capture.media import MediaChunk, MediaKind


@dataclass(frozen=True, slots=True)
class MediaWindow:
    """Immutable chronological media view used by one model assessment."""

    chunks: tuple[MediaChunk, ...]
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def audio_chunks(self) -> tuple[MediaChunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.kind is MediaKind.AUDIO)

    @property
    def image_chunks(self) -> tuple[MediaChunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.kind is MediaKind.IMAGE)

    @property
    def has_both_modalities(self) -> bool:
        return bool(self.audio_chunks and self.image_chunks)


class RollingMediaWindow:
    """Keep a bounded in-memory media window without persisting raw payloads."""

    def __init__(self, *, duration_ms: int, max_chunks: int = 2_000) -> None:
        if duration_ms < 1_000:
            raise ValueError("media window duration must be at least 1000 ms")
        if max_chunks < 2:
            raise ValueError("media window max_chunks must be at least two")
        self.duration_ms = duration_ms
        self.max_chunks = max_chunks
        self._chunks: deque[MediaChunk] = deque()

    def append(self, chunk: MediaChunk) -> None:
        if self._chunks and chunk.timestamp_ms <= self._chunks[-1].timestamp_ms:
            raise ValueError("rolling media chunks must use strictly increasing timestamps")
        self._chunks.append(chunk)
        cutoff_ms = max(0, chunk.timestamp_ms - self.duration_ms)
        while self._chunks and self._chunks[0].timestamp_ms < cutoff_ms:
            self._chunks.popleft()
        while len(self._chunks) > self.max_chunks:
            self._chunks.popleft()

    def snapshot(self) -> MediaWindow | None:
        if not self._chunks:
            return None
        chunks = tuple(self._chunks)
        return MediaWindow(
            chunks=chunks,
            start_ms=chunks[0].timestamp_ms,
            end_ms=chunks[-1].timestamp_ms,
        )

    def clear(self) -> None:
        self._chunks.clear()
