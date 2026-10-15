from coregulation_poc.acoustics.feature_extraction import extract_acoustic_features
from coregulation_poc.acoustics.speaker_binding import SpeakerBinding, SpeakerSegment
from coregulation_poc.delivery import OutputModality
from coregulation_poc.runtime.session import RealtimeSession


def test_voice_failure_before_playback_does_not_require_a_fictitious_start_time():
    channel = RealtimeSession._channel_execution(
        {"status": "failed", "error": "TTS disconnected before first audio"},
        modality=OutputModality.SPOKEN_VOICE, default_provider="test-tts",
    )
    assert channel.started_at_ms is None
    assert channel.status.value == "failed"


def test_invalid_acoustic_segment_does_not_discard_other_window_evidence():
    binding = SpeakerBinding(bound=True, segments=[
        SpeakerSegment(500, 400, 150, 150, 2),
        SpeakerSegment(600, 800, 150, 150, 2),
    ])
    features = extract_acoustic_features((), binding)
    assert len(features.segments) == 1
    assert features.segments[0].start_ms == 600
    assert features.total_speech_ms == 200
