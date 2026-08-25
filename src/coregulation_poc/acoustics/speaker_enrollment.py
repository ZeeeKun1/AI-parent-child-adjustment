"""Speaker enrollment: one-time voiceprint registration for parent/child.

This module provides the enrollment workflow:

1.  **Registration** (once per family, before experiments):
        enroll_from_file("parent_reading.wav", "parent", "P01")
        enroll_from_file("child_reading.wav",  "child",  "P01")

    The system extracts a 256-dimensional speaker embedding from each
    audio clip using Resemblyzer's pretrained VoiceEncoder and persists
    them to ``data/enrollments/{family_id}.json``.

2.  **Loading** (at the start of each experiment session):
        enrollment = load_enrollment("P01")

3.  **Runtime identification** is handled by
    :func:`coregulation_poc.acoustics.speaker_binding.bind_speakers_by_embedding`,
    which compares each voiced segment against the enrolled embeddings.

The embedding approach does not depend on F0 separation, so it remains
reliable even when adult and child pitch ranges overlap.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --- Lazy imports ----------------------------------------------------------
# Resemblyzer / librosa are imported lazily so that the rest of the
# acoustic module (F0 fallback) remains importable without them.
#
# Resemblyzer depends on webrtcvad for internal silence trimming.  On
# platforms where the C extension is unavailable (e.g. Python 3.13 on
# Windows without Visual C++ Build Tools), we inject a pure-Python
# no-op shim that treats every frame as speech.  This is safe because
# our own _energy_based_vad already handles voice activity detection
# before embedding extraction.

_ENCODER: object | None = None  # cached VoiceEncoder singleton
_ENCODER_LOCK = threading.RLock()
FAMILY_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _ensure_webrtcvad_shim() -> None:
    """Inject a no-op webrtcvad shim if the real package is unavailable."""
    if "webrtcvad" in sys.modules:
        return
    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        import types

        class _NoopVad:
            """Drop-in replacement for webrtcvad.Vad that marks all frames as speech."""

            def __init__(self, aggressiveness: int = 3) -> None:
                pass

            def is_speech(self, frame: bytes, sample_rate: int) -> bool:
                return True

        shim = types.ModuleType("webrtcvad")
        shim.Vad = _NoopVad
        sys.modules["webrtcvad"] = shim


def _get_encoder() -> object:
    """Return a cached ``VoiceEncoder`` instance (creates one on first call)."""
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            _ensure_webrtcvad_shim()
            from resemblyzer import VoiceEncoder

            _ENCODER = VoiceEncoder()
    return _ENCODER


def _load_audio_mono_16k(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float32 at 16 kHz.

    Returns ``(samples, sample_rate)``.
    """
    import librosa

    samples, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    return np.asarray(samples, dtype=np.float32), int(sr)


# --- Data structures -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnrolledSpeaker:
    """One enrolled speaker's voiceprint."""

    label: str  # "parent" or "child"
    audio_source: str  # filename of enrollment audio
    duration_ms: int
    embedding: tuple[float, ...]  # 256-dim, L2-normalised


@dataclass(frozen=True, slots=True)
class SpeakerEnrollment:
    """Collection of enrolled speakers for one family."""

    family_id: str
    speakers: dict[str, EnrolledSpeaker] = field(default_factory=dict)
    model_name: str = "resemblyzer-ve"

    @property
    def is_complete(self) -> bool:
        """True when both parent and child are enrolled."""
        return "parent" in self.speakers and "child" in self.speakers

    @property
    def parent_embedding(self) -> np.ndarray | None:
        s = self.speakers.get("parent")
        return np.array(s.embedding, dtype=np.float32) if s else None

    @property
    def child_embedding(self) -> np.ndarray | None:
        s = self.speakers.get("child")
        return np.array(s.embedding, dtype=np.float32) if s else None

    def to_json(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "family_id": self.family_id,
            "model_name": self.model_name,
            "speakers": {
                label: asdict(speaker) for label, speaker in self.speakers.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> SpeakerEnrollment:
        """Deserialise from a JSON-compatible dict."""
        speakers: dict[str, EnrolledSpeaker] = {}
        for label, spk in data.get("speakers", {}).items():
            speakers[label] = EnrolledSpeaker(
                label=spk["label"],
                audio_source=spk["audio_source"],
                duration_ms=spk["duration_ms"],
                embedding=tuple(spk["embedding"]),
            )
        return cls(
            family_id=data["family_id"],
            speakers=speakers,
            model_name=data.get("model_name", "resemblyzer-ve"),
        )


# --- Core operations -------------------------------------------------------


def extract_embedding(samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Extract a 256-dim L2-normalised speaker embedding.

    Parameters
    ----------
    samples
        Mono audio samples, float32 in [-1, 1].
    sample_rate
        Sample rate (must be 16 kHz for Resemblyzer).

    Returns
    -------
    np.ndarray
        256-dimensional embedding vector.
    """
    if sample_rate != 16_000:
        import librosa

        samples = librosa.resample(
            samples.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=16_000,
        )
        sample_rate = 16_000

    # Resemblyzer needs at least ~1.6 s of audio; pad if shorter.
    min_samples = int(1.6 * sample_rate)
    if samples.size < min_samples:
        samples = np.pad(samples, (0, min_samples - samples.size))

    with _ENCODER_LOCK:
        encoder = _get_encoder()
        embedding = encoder.embed_utterance(samples)
    return np.asarray(embedding, dtype=np.float32)



def _validate_family_id(family_id: str) -> None:
    if FAMILY_ID.fullmatch(family_id) is None:
        raise ValueError(
            "family_id must contain 1-32 letters, numbers, underscores or hyphens"
        )


def enrolled_speaker_from_pcm16(
    pcm_audio: bytes,
    speaker_label: str,
    *,
    sample_rate: int = 16_000,
) -> EnrolledSpeaker:
    """Create an in-memory speaker binding without saving raw audio or embeddings."""

    if speaker_label not in ("parent", "child"):
        raise ValueError(
            f"speaker_label must be 'parent' or 'child', got {speaker_label!r}"
        )
    if sample_rate != 16_000:
        raise ValueError("browser enrollment audio must use a 16000 Hz sample rate")
    if not pcm_audio or len(pcm_audio) % 2:
        raise ValueError("browser enrollment audio must be non-empty PCM16 data")

    samples_i16 = np.frombuffer(pcm_audio, dtype="<i2")
    duration_seconds = samples_i16.size / sample_rate
    if duration_seconds < 3:
        raise ValueError("请至少录制 3 秒清晰语音")
    if duration_seconds > 15:
        raise ValueError("单次声纹录制不能超过 15 秒")
    rms = float(np.sqrt(np.mean(samples_i16.astype(np.float64) ** 2)))
    if rms < 120:
        raise ValueError("没有检测到清晰语音，请靠近设备后重新录制")

    samples = samples_i16.astype(np.float32) / 32768.0
    embedding = extract_embedding(samples, sample_rate)
    return EnrolledSpeaker(
        label=speaker_label,
        audio_source="browser_recording_not_saved",
        duration_ms=round(duration_seconds * 1000),
        embedding=tuple(float(x) for x in embedding),
    )


def enroll_from_pcm16(
    pcm_audio: bytes,
    speaker_label: str,
    family_id: str,
    enrollment_dir: Path,
    *,
    sample_rate: int = 16_000,
) -> SpeakerEnrollment:
    """Enroll browser-recorded PCM16 speech without saving the raw audio."""

    if speaker_label not in ("parent", "child"):
        raise ValueError(
            f"speaker_label must be 'parent' or 'child', got {speaker_label!r}"
        )
    _validate_family_id(family_id)
    if sample_rate != 16_000:
        raise ValueError("browser enrollment audio must use a 16000 Hz sample rate")
    if not pcm_audio or len(pcm_audio) % 2:
        raise ValueError("browser enrollment audio must be non-empty PCM16 data")

    samples_i16 = np.frombuffer(pcm_audio, dtype="<i2")
    duration_seconds = samples_i16.size / sample_rate
    if duration_seconds < 3:
        raise ValueError("请至少录制 3 秒清晰语音")
    if duration_seconds > 15:
        raise ValueError("单次声纹录制不能超过 15 秒")
    rms = float(np.sqrt(np.mean(samples_i16.astype(np.float64) ** 2)))
    if rms < 120:
        raise ValueError("没有检测到清晰语音，请靠近设备后重新录制")

    samples = samples_i16.astype(np.float32) / 32768.0
    embedding = extract_embedding(samples, sample_rate)
    enrollment_dir.mkdir(parents=True, exist_ok=True)
    json_path = enrollment_dir / f"{family_id}.json"
    if json_path.exists():
        enrollment = SpeakerEnrollment.from_json(
            json.loads(json_path.read_text(encoding="utf-8"))
        )
    else:
        enrollment = SpeakerEnrollment(family_id=family_id)

    enrolled = EnrolledSpeaker(
        label=speaker_label,
        audio_source="browser_recording_not_saved",
        duration_ms=round(duration_seconds * 1000),
        embedding=tuple(float(x) for x in embedding),
    )
    enrollment = SpeakerEnrollment(
        family_id=family_id,
        speakers={**enrollment.speakers, speaker_label: enrolled},
        model_name="resemblyzer-ve",
    )
    json_path.write_text(
        json.dumps(enrollment.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return enrollment

def enroll_from_file(
    audio_path: Path,
    speaker_label: str,
    family_id: str,
    enrollment_dir: Path,
) -> SpeakerEnrollment:
    """Enroll one speaker from an audio file and persist the result.

    Parameters
    ----------
    audio_path
        Path to the enrollment audio (wav, mp3, etc.).
    speaker_label
        ``"parent"`` or ``"child"``.
    family_id
        Family identifier used to group enrollments.
    enrollment_dir
        Directory where enrollment JSON files are stored.

    Returns
    -------
    SpeakerEnrollment
        The updated enrollment for the family (including the newly
        registered speaker).
    """
    _validate_family_id(family_id)
    if speaker_label not in ("parent", "child"):
        raise ValueError(f"speaker_label must be 'parent' or 'child', got {speaker_label!r}")

    resolved = audio_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")

    # Load existing enrollment or create new
    enrollment_dir.mkdir(parents=True, exist_ok=True)
    json_path = enrollment_dir / f"{family_id}.json"
    if json_path.exists():
        enrollment = SpeakerEnrollment.from_json(
            json.loads(json_path.read_text(encoding="utf-8"))
        )
    else:
        enrollment = SpeakerEnrollment(family_id=family_id)

    # Extract embedding
    samples, sr = _load_audio_mono_16k(resolved)
    embedding = extract_embedding(samples, sr)
    duration_ms = int(len(samples) / sr * 1000)

    enrolled = EnrolledSpeaker(
        label=speaker_label,
        audio_source=resolved.name,
        duration_ms=duration_ms,
        embedding=tuple(float(x) for x in embedding),
    )

    # Update enrollment (frozen dataclass → replace)
    new_speakers = {**enrollment.speakers, speaker_label: enrolled}
    enrollment = SpeakerEnrollment(
        family_id=family_id,
        speakers=new_speakers,
        model_name="resemblyzer-ve",
    )

    # Persist
    json_path.write_text(
        json.dumps(enrollment.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return enrollment


def load_enrollment(family_id: str, enrollment_dir: Path) -> SpeakerEnrollment | None:
    """Load a previously saved enrollment for a family.

    Returns ``None`` if no enrollment file exists.
    """
    _validate_family_id(family_id)
    json_path = enrollment_dir / f"{family_id}.json"
    if not json_path.exists():
        return None
    return SpeakerEnrollment.from_json(
        json.loads(json_path.read_text(encoding="utf-8"))
    )
