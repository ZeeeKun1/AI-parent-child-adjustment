"""Audio-energy-based natural turn boundary detection.

The detector consumes 100 ms / 16 kHz / s16 / mono PCM audio chunks
(the same format produced by ``capture.video_replay.decode_video_for_replay``)
and maintains a sliding window of RMS energy values.  When the most recent
chunks are all below an energy threshold, the detector reports a natural
turn boundary, meaning nobody is currently speaking and it is safe to
insert an intervention without cutting off active communication.

Research basis (formative study, Section 3.4.1):
  - "I would prefer not to interrupt the natural rhythm and atmosphere of
    the parent-child interaction..." (E2)
  - "Because the child is in a learning state, interrupting will affect
    concentration." (E2)
  - The detector implements the simplest faithful operationalisation:
    sustained audio silence implies nobody is speaking, which is a safe
    insertion point.
"""

from __future__ import annotations

import array
import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TurnBoundaryConfig:
    """Configuration for the silence-based turn boundary detector.

    Attributes:
        chunk_ms: Duration of each audio chunk in milliseconds
          (must match the decoder output, default 100).
        silence_window_ms: Total window length in milliseconds.  When all
          chunks within this window are below the threshold, a boundary
          is reported.  Default 500 (5 chunks at 100 ms each).
        rms_threshold: RMS amplitude threshold below which a chunk is
          considered silent.  PCM samples are signed 16-bit (range
          -32768..32767).  300 corresponds to a very quiet signal,
          roughly -44 dBFS, which is above typical microphone noise floors
          but well below normal speech.
    """

    chunk_ms: int = 100
    silence_window_ms: int = 500
    rms_threshold: float = 300.0

    def __post_init__(self) -> None:
        if self.chunk_ms < 20:
            raise ValueError("chunk_ms must be at least 20")
        if self.silence_window_ms < self.chunk_ms:
            raise ValueError("silence_window_ms must be at least one chunk")
        if self.silence_window_ms % self.chunk_ms != 0:
            raise ValueError("silence_window_ms must be a multiple of chunk_ms")
        if self.rms_threshold <= 0:
            raise ValueError("rms_threshold must be positive")

    @property
    def window_chunks(self) -> int:
        return self.silence_window_ms // self.chunk_ms


class TurnBoundaryDetector:
    """Detect natural turn boundaries from audio chunk RMS energy.

    Feed each 100 ms PCM audio chunk via :meth:`ingest_chunk`.  The
    detector maintains a sliding window and reports whether the recent
    audio is consistently silent, indicating a natural gap in speech.

    Usage::

        detector = TurnBoundaryDetector()
        for chunk in audio_chunks:
            detector.ingest_chunk(chunk)
        if detector.is_at_boundary():
            # safe to intervene
    """

    def __init__(self, config: TurnBoundaryConfig | None = None) -> None:
        self._config = config or TurnBoundaryConfig()
        self._window: deque[float] = deque(maxlen=self._config.window_chunks)
        self._chunk_count: int = 0
        self._last_boundary: bool = False

    @property
    def config(self) -> TurnBoundaryConfig:
        return self._config

    @property
    def chunk_count(self) -> int:
        """Total number of chunks ingested so far."""
        return self._chunk_count

    @property
    def window_ready(self) -> bool:
        """Whether enough chunks have been collected to fill the window."""
        return len(self._window) >= self._config.window_chunks

    @property
    def last_rms(self) -> float | None:
        """RMS of the most recently ingested chunk, or None if empty."""
        return self._window[-1] if self._window else None

    def ingest_chunk(self, pcm_chunk: bytes) -> bool:
        """Ingest one PCM audio chunk and return the current boundary state.

        Args:
            pcm_chunk: Raw PCM bytes (s16, mono, 16 kHz).  The caller must
              ensure the chunk matches the decoder's output format.

        Returns:
            True if the current window indicates a natural turn boundary.
        """
        rms = _compute_rms(pcm_chunk)
        self._window.append(rms)
        self._chunk_count += 1
        self._last_boundary = (
            self.window_ready
            and all(value < self._config.rms_threshold for value in self._window)
        )
        return self._last_boundary

    def is_at_boundary(self) -> bool:
        """Return whether the detector currently reports a turn boundary.

        This reflects the state after the most recent :meth:`ingest_chunk`
        call.  If no chunks have been ingested yet, returns False.
        """
        return self._last_boundary

    def reset(self) -> None:
        """Clear the sliding window and reset internal state."""
        self._window.clear()
        self._chunk_count = 0
        self._last_boundary = False


def _compute_rms(pcm_bytes: bytes) -> float:
    """Compute the RMS amplitude of a signed-16-bit PCM buffer.

    An empty buffer returns 0.0.  Only complete 16-bit samples are
    considered; a trailing odd byte is ignored.
    """
    if len(pcm_bytes) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm_bytes[: len(pcm_bytes) // 2 * 2])
    if not samples:
        return 0.0
    sum_sq = sum(sample * sample for sample in samples)
    return math.sqrt(sum_sq / len(samples))
