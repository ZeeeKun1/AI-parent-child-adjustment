"""Objective acoustic measurements used as supporting module-one evidence."""

from coregulation_poc.acoustics.prosody import (
    AcousticAnalysis,
    AcousticAnalysisConfig,
    analyze_replay_audio,
    load_acoustic_analysis_config,
)
from coregulation_poc.acoustics.speaker_binding import (
    SpeakerBinding,
    SpeakerLabel,
    SpeakerSegment,
    bind_speakers,
    bind_speakers_by_embedding,
    bind_speakers_by_f0,
)
from coregulation_poc.acoustics.speaker_enrollment import (
    EnrolledSpeaker,
    SpeakerEnrollment,
    enroll_from_file,
    extract_embedding,
    load_enrollment,
)

__all__ = [
    "AcousticAnalysis",
    "AcousticAnalysisConfig",
    "analyze_replay_audio",
    "load_acoustic_analysis_config",
    "SpeakerBinding",
    "SpeakerLabel",
    "SpeakerSegment",
    "bind_speakers",
    "bind_speakers_by_embedding",
    "bind_speakers_by_f0",
    "EnrolledSpeaker",
    "SpeakerEnrollment",
    "enroll_from_file",
    "extract_embedding",
    "load_enrollment",
]
