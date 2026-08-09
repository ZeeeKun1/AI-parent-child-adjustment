from __future__ import annotations

import array
import math

import pytest

from coregulation_poc.capture.turn_boundary import (
    TurnBoundaryConfig,
    TurnBoundaryDetector,
)


# 100 ms @ 16 kHz, s16, mono = 1600 samples = 3200 bytes
_SAMPLES_PER_CHUNK = 1600


def _silent_chunk() -> bytes:
    return b"\x00" * (_SAMPLES_PER_CHUNK * 2)


def _loud_chunk(amplitude: int = 5000) -> bytes:
    samples = array.array("h", [amplitude, -amplitude] * (_SAMPLES_PER_CHUNK // 2))
    return samples.tobytes()


def _quiet_chunk(amplitude: int = 100) -> bytes:
    samples = array.array("h", [amplitude, -amplitude] * (_SAMPLES_PER_CHUNK // 2))
    return samples.tobytes()


class TestTurnBoundaryConfig:
    def test_default_config(self) -> None:
        config = TurnBoundaryConfig()
        assert config.chunk_ms == 100
        assert config.silence_window_ms == 500
        assert config.rms_threshold == 300.0
        assert config.window_chunks == 5

    def test_custom_config(self) -> None:
        config = TurnBoundaryConfig(chunk_ms=50, silence_window_ms=200, rms_threshold=200.0)
        assert config.window_chunks == 4

    def test_chunk_ms_too_small(self) -> None:
        with pytest.raises(ValueError, match="chunk_ms must be at least 20"):
            TurnBoundaryConfig(chunk_ms=10)

    def test_silence_window_smaller_than_chunk(self) -> None:
        with pytest.raises(ValueError, match="silence_window_ms must be at least one chunk"):
            TurnBoundaryConfig(chunk_ms=100, silence_window_ms=50)

    def test_silence_window_not_multiple(self) -> None:
        with pytest.raises(ValueError, match="silence_window_ms must be a multiple"):
            TurnBoundaryConfig(chunk_ms=100, silence_window_ms=150)

    def test_rms_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="rms_threshold must be positive"):
            TurnBoundaryConfig(rms_threshold=0)


class TestTurnBoundaryDetector:
    def test_no_chunks_reports_no_boundary(self) -> None:
        detector = TurnBoundaryDetector()
        assert detector.is_at_boundary() is False
        assert detector.chunk_count == 0
        assert detector.window_ready is False
        assert detector.last_rms is None

    def test_silence_triggers_boundary(self) -> None:
        detector = TurnBoundaryDetector()
        for _ in range(5):
            result = detector.ingest_chunk(_silent_chunk())
        assert result is True
        assert detector.is_at_boundary() is True
        assert detector.window_ready is True
        assert detector.chunk_count == 5
        assert detector.last_rms == 0.0

    def test_partial_window_no_boundary(self) -> None:
        detector = TurnBoundaryDetector()
        for i in range(4):
            detector.ingest_chunk(_silent_chunk())
        assert detector.is_at_boundary() is False
        assert detector.window_ready is False

    def test_loud_audio_prevents_boundary(self) -> None:
        detector = TurnBoundaryDetector()
        for _ in range(4):
            detector.ingest_chunk(_silent_chunk())
        detector.ingest_chunk(_loud_chunk())
        assert detector.is_at_boundary() is False
        assert detector.last_rms is not None
        assert detector.last_rms > 300.0

    def test_boundary_lost_when_speech_resumes(self) -> None:
        detector = TurnBoundaryDetector()
        for _ in range(5):
            detector.ingest_chunk(_silent_chunk())
        assert detector.is_at_boundary() is True
        detector.ingest_chunk(_loud_chunk())
        assert detector.is_at_boundary() is False

    def test_boundary_regained_after_silence_returns(self) -> None:
        detector = TurnBoundaryDetector()
        for _ in range(5):
            detector.ingest_chunk(_silent_chunk())
        assert detector.is_at_boundary() is True
        detector.ingest_chunk(_loud_chunk())
        assert detector.is_at_boundary() is False
        for _ in range(5):
            detector.ingest_chunk(_silent_chunk())
        assert detector.is_at_boundary() is True

    def test_quiet_audio_below_threshold_triggers_boundary(self) -> None:
        detector = TurnBoundaryDetector()
        for _ in range(5):
            detector.ingest_chunk(_quiet_chunk(amplitude=100))
        assert detector.is_at_boundary() is True
        assert detector.last_rms == 100.0

    def test_quiet_audio_at_threshold_does_not_trigger(self) -> None:
        detector = TurnBoundaryDetector()
        for _ in range(5):
            detector.ingest_chunk(_quiet_chunk(amplitude=300))
        assert detector.is_at_boundary() is False

    def test_reset_clears_state(self) -> None:
        detector = TurnBoundaryDetector()
        for _ in range(5):
            detector.ingest_chunk(_silent_chunk())
        assert detector.is_at_boundary() is True
        detector.reset()
        assert detector.is_at_boundary() is False
        assert detector.chunk_count == 0
        assert detector.window_ready is False
        assert detector.last_rms is None

    def test_sliding_window_evicts_old_chunks(self) -> None:
        detector = TurnBoundaryConfig(chunk_ms=100, silence_window_ms=300)
        det = TurnBoundaryDetector(detector)
        assert det.config.window_chunks == 3
        det.ingest_chunk(_loud_chunk())
        det.ingest_chunk(_loud_chunk())
        det.ingest_chunk(_loud_chunk())
        assert det.is_at_boundary() is False
        det.ingest_chunk(_silent_chunk())
        det.ingest_chunk(_silent_chunk())
        assert det.is_at_boundary() is False
        det.ingest_chunk(_silent_chunk())
        assert det.is_at_boundary() is True

    def test_empty_chunk_returns_zero_rms(self) -> None:
        detector = TurnBoundaryDetector()
        detector.ingest_chunk(b"")
        assert detector.last_rms == 0.0
        assert detector.is_at_boundary() is False

    def test_single_byte_chunk_handled_gracefully(self) -> None:
        detector = TurnBoundaryDetector()
        detector.ingest_chunk(b"\x00")
        assert detector.last_rms == 0.0

    def test_ingest_chunk_returns_boundary_state(self) -> None:
        detector = TurnBoundaryDetector()
        for i in range(4):
            assert detector.ingest_chunk(_silent_chunk()) is False
        assert detector.ingest_chunk(_silent_chunk()) is True

    def test_custom_config_with_smaller_window(self) -> None:
        config = TurnBoundaryConfig(chunk_ms=100, silence_window_ms=200, rms_threshold=500.0)
        detector = TurnBoundaryDetector(config)
        assert detector.config.window_chunks == 2
        detector.ingest_chunk(_quiet_chunk(amplitude=400))
        assert detector.is_at_boundary() is False
        detector.ingest_chunk(_quiet_chunk(amplitude=400))
        assert detector.is_at_boundary() is True
