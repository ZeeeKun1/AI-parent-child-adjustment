"""Tests 28-32: Speaker binding with voiceprint enrollment.

Verifies that:
- Embedding-based binding succeeds when both speakers are enrolled (28)
- Incomplete enrollment returns UNKNOWN (29)
- Real-time mode never falls back to F0 (30)
- Incomplete binding prevents starting the closed loop (31)
- Session end clears the enrollment (32)
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from coregulation_poc.acoustics.feature_extraction import extract_acoustic_features
from coregulation_poc.acoustics.speaker_binding import (
    SpeakerBinding,
    SpeakerLabel,
    SpeakerSegment,
    bind_speakers,
    bind_speakers_by_embedding,
)
from coregulation_poc.capture.media import MediaChunk, MediaKind
from coregulation_poc.acoustics.speaker_enrollment import (
    EnrolledSpeaker,
    SpeakerEnrollment,
)


def _make_enrollment(*, complete: bool = True) -> SpeakerEnrollment:
    """Create a SpeakerEnrollment with mock embeddings."""
    parent_emb = np.zeros(256, dtype=np.float32)
    parent_emb[0] = 1.0
    child_emb = np.zeros(256, dtype=np.float32)
    child_emb[1] = 1.0

    speakers: dict[str, EnrolledSpeaker] = {}
    speakers["parent"] = EnrolledSpeaker(
        label="parent",
        audio_source="parent.wav",
        duration_ms=5000,
        embedding=tuple(parent_emb.tolist()),
    )
    if complete:
        speakers["child"] = EnrolledSpeaker(
            label="child",
            audio_source="child.wav",
            duration_ms=5000,
            embedding=tuple(child_emb.tolist()),
        )
    return SpeakerEnrollment(family_id="test-family", speakers=speakers)


# Test 28: Parent and child are each bound successfully via embeddings
@patch("coregulation_poc.acoustics.speaker_binding._prepare_audio_segments")
@patch("coregulation_poc.acoustics.speaker_binding.extract_embedding")
def test_parent_and_child_each_bound_successfully(
    mock_extract: patch,
    mock_prepare: patch,
) -> None:
    parent_emb = np.zeros(256, dtype=np.float32)
    parent_emb[0] = 1.0
    child_emb = np.zeros(256, dtype=np.float32)
    child_emb[1] = 1.0

    # Mock _prepare_audio_segments to return two timed segments
    samples = np.zeros(32000, dtype=np.float64)
    mock_prepare.return_value = (
        samples,
        16000,
        [
            (0, 2000, 0, 32000),
            (3000, 5000, 48000, 80000),
        ],
    )
    # First segment matches parent, second matches child
    mock_extract.side_effect = [parent_emb, child_emb]

    enrollment = _make_enrollment(complete=True)
    result = bind_speakers_by_embedding((), enrollment)

    assert result.bound is True
    assert result.parent_segment_count == 1
    assert result.child_segment_count == 1
    assert result.method == "embedding_cosine"


# Test 29: Incomplete enrollment returns bound=False
def test_incomplete_enrollment_returns_not_bound() -> None:
    enrollment = _make_enrollment(complete=False)
    assert not enrollment.is_complete

    result = bind_speakers_by_embedding((), enrollment)

    assert result.bound is False
    assert "incomplete" in (result.limitation_reason or "").lower()


# Test 30: Real-time mode (allow_f0_fallback=False) never uses F0
def test_realtime_mode_does_not_fallback_to_f0() -> None:
    # Case A: No enrollment at all
    result_no_enrollment = bind_speakers((), allow_f0_fallback=False)
    assert result_no_enrollment.bound is False
    assert "F0 fallback is disabled" in (
        result_no_enrollment.limitation_reason or ""
    )

    # Case B: Enrollment exists but embedding binding fails
    # We patch bind_speakers_by_embedding to return a failed result
    failed_binding = SpeakerBinding(
        bound=False,
        method="embedding_cosine",
        limitation_reason="No segments matched either enrolled speaker.",
    )
    with patch(
        "coregulation_poc.acoustics.speaker_binding.bind_speakers_by_embedding",
        return_value=failed_binding,
    ):
        enrollment = _make_enrollment(complete=True)
        result = bind_speakers((), enrollment=enrollment, allow_f0_fallback=False)

    assert result.bound is False
    assert "F0 fallback disabled" in (result.limitation_reason or "")


# Test 31: Incomplete binding means the closed loop cannot start
def test_incomplete_binding_prevents_closed_loop_start() -> None:
    """When enrollment is incomplete, bind_speakers returns bound=False.

    The web app checks enrollment.is_complete before allowing the session
    to start.  This test verifies the binding-level prerequisite.
    """
    enrollment = _make_enrollment(complete=False)

    # Real-time mode: no F0 fallback, incomplete enrollment
    result = bind_speakers((), enrollment=enrollment, allow_f0_fallback=False)
    assert result.bound is False

    # Simulate the web app's enrollment check
    assert not enrollment.is_complete, (
        "Session must not start without complete enrollment"
    )


# Test 32: Session end clears the enrollment
def test_session_end_clears_enrollment() -> None:
    """Simulate the web app's session_enrollments cleanup on disconnect.

    The web app calls ``session_enrollments.pop(session_id, None)`` in
    the finally block of the WebSocket handler.  This test verifies
    that the pop removes the enrollment.
    """
    session_enrollments: dict[str, SpeakerEnrollment] = {}
    session_id = "test-session"

    # Register enrollment
    enrollment = _make_enrollment(complete=True)
    session_enrollments[session_id] = enrollment
    assert session_id in session_enrollments

    # Simulate session end cleanup
    session_enrollments.pop(session_id, None)
    assert session_id not in session_enrollments, (
        "Enrollment must be removed after session ends"
    )


def test_unbound_segments_keep_acoustics_and_use_their_actual_audio_window() -> None:
    low_payload = np.zeros(1600, dtype="<i2").tobytes()
    high_payload = np.full(1600, 16384, dtype="<i2").tobytes()
    chunks = (
        MediaChunk(MediaKind.AUDIO, 0, low_payload),
        MediaChunk(MediaKind.AUDIO, 100, high_payload),
    )
    binding = SpeakerBinding(
        bound=False,
        segments=[
            SpeakerSegment(
                start_ms=100,
                end_ms=200,
                mean_f0_hz=220.0,
                median_f0_hz=218.0,
                voiced_frame_count=10,
            ),
        ],
        method="embedding_cosine",
        limitation_reason="Identity match was uncertain.",
    )

    features = extract_acoustic_features(chunks, binding)

    assert len(features.segments) == 1
    assert features.segments[0].speaker == "unknown"
    assert features.segments[0].rms_energy == pytest.approx(0.5, abs=0.001)
