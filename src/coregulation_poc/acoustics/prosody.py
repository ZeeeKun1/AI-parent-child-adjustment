from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import parselmouth
import yaml
from pydantic import BaseModel, ConfigDict, Field

from coregulation_poc.capture.video_replay import MediaKind, ReplayMedia
from coregulation_poc.models import Actor
from coregulation_poc.paths import ACOUSTIC_ANALYSIS_PATH, resolve_project_path


class MeasurementQuality(StrEnum):
    SUFFICIENT = "sufficient"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


class AcousticExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["praat_parselmouth"]
    pitch_floor_hz: float = Field(gt=0)
    pitch_ceiling_hz: float = Field(gt=0)
    analysis_time_step_seconds: float = Field(gt=0)
    intensity_minimum_pitch_hz: float = Field(gt=0)


class AcousticInterpretationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["supporting_measurement_only"]
    single_feature_must_not_trigger_state: bool
    raw_pitch_must_not_be_interpreted_as_emotion: bool
    raw_intensity_must_not_be_interpreted_as_emotion: bool
    within_speaker_change_required_for_behavioral_interpretation: bool
    preserve_measurement_when_interpretation_is_limited: bool


class AcousticAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    source: str = Field(min_length=1)
    extractor: AcousticExtractorConfig
    interpretation_policy: AcousticInterpretationPolicy
    current_runtime_limits: list[str] = Field(min_length=1)


class PitchMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    analysis_frame_count: int = Field(ge=0)
    voiced_frame_count: int = Field(ge=0)
    voiced_frame_fraction: float = Field(ge=0, le=1)
    mean_hz: float | None = None
    median_hz: float | None = None
    standard_deviation_hz: float | None = None
    percentile_10_hz: float | None = None
    percentile_90_hz: float | None = None


class IntensityMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    analysis_frame_count: int = Field(ge=0)
    measured_frame_count: int = Field(ge=0)
    mean_db: float | None = None
    standard_deviation_db: float | None = None
    percentile_10_db: float | None = None
    percentile_90_db: float | None = None
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None


class SpeechRateMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    value: float | None = None
    unit: Literal["characters_per_second"] = "characters_per_second"
    limitation_reason: str


class AcousticAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_version: int = Field(ge=1)
    source: str
    extractor: Literal["praat_parselmouth"]
    scope: Literal["full_mixed_channel", "evidence_interval"]
    actor: Actor
    speaker_roles_bound: bool
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    sample_rate_hz: int = Field(gt=0)
    quality: MeasurementQuality
    pitch: PitchMeasurement
    intensity: IntensityMeasurement
    speech_rate: SpeechRateMeasurement
    interpretation_role: Literal["supporting_measurement_only"]
    limitations: list[str]


def load_acoustic_analysis_config(
    path: str | Path = ACOUSTIC_ANALYSIS_PATH,
) -> AcousticAnalysisConfig:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return AcousticAnalysisConfig.model_validate(payload)


def _round_or_none(value: float | np.floating[float] | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 3)


def _summary(values: np.ndarray) -> tuple[float | None, ...]:
    if values.size == 0:
        return (None, None, None, None, None)
    return tuple(
        _round_or_none(value)
        for value in (
            np.mean(values),
            np.median(values),
            np.std(values),
            np.percentile(values, 10),
            np.percentile(values, 90),
        )
    )


def _dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return _round_or_none(20 * math.log10(value))


def _pcm_from_replay(media: ReplayMedia) -> np.ndarray:
    payload = b"".join(
        chunk.payload for chunk in media.chunks if chunk.kind is MediaKind.AUDIO
    )
    if not payload:
        return np.array([], dtype=np.float64)
    return np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0


def analyze_replay_audio(
    media: ReplayMedia,
    *,
    config: AcousticAnalysisConfig | None = None,
    start_ms: int = 0,
    end_ms: int | None = None,
    actor: Actor = Actor.UNKNOWN,
    speaker_roles_bound: bool = False,
) -> AcousticAnalysis:
    """Measure a replay interval without inferring emotion or co-regulation state."""
    settings = config or load_acoustic_analysis_config()
    samples = _pcm_from_replay(media)
    total_duration_ms = round(samples.size * 1000 / media.audio_sample_rate)
    effective_end_ms = total_duration_ms if end_ms is None else min(end_ms, total_duration_ms)
    if effective_end_ms < start_ms:
        raise ValueError("end_ms must be greater than or equal to start_ms")

    start_sample = round(start_ms * media.audio_sample_rate / 1000)
    end_sample = round(effective_end_ms * media.audio_sample_rate / 1000)
    interval = samples[start_sample:end_sample]
    duration_ms = round(interval.size * 1000 / media.audio_sample_rate)

    pitch_values = np.array([], dtype=np.float64)
    pitch_frame_count = 0
    intensity_values = np.array([], dtype=np.float64)
    intensity_frame_count = 0
    if interval.size:
        sound = parselmouth.Sound(interval, sampling_frequency=media.audio_sample_rate)
        try:
            pitch = sound.to_pitch_ac(
                time_step=settings.extractor.analysis_time_step_seconds,
                pitch_floor=settings.extractor.pitch_floor_hz,
                pitch_ceiling=settings.extractor.pitch_ceiling_hz,
            )
            raw_pitch = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
            pitch_frame_count = int(raw_pitch.size)
            pitch_values = raw_pitch[np.isfinite(raw_pitch) & (raw_pitch > 0)]
        except parselmouth.PraatError:
            pass
        try:
            intensity = sound.to_intensity(
                minimum_pitch=settings.extractor.intensity_minimum_pitch_hz,
                time_step=settings.extractor.analysis_time_step_seconds,
            )
            raw_intensity = np.asarray(intensity.values, dtype=np.float64).reshape(-1)
            intensity_frame_count = int(raw_intensity.size)
            intensity_values = raw_intensity[
                np.isfinite(raw_intensity) & (raw_intensity > -299.0)
            ]
        except parselmouth.PraatError:
            pass

    pitch_mean, pitch_median, pitch_std, pitch_p10, pitch_p90 = _summary(pitch_values)
    intensity_mean, _, intensity_std, intensity_p10, intensity_p90 = _summary(
        intensity_values
    )
    rms = float(np.sqrt(np.mean(np.square(interval)))) if interval.size else 0.0
    peak = float(np.max(np.abs(interval))) if interval.size else 0.0

    limitations = list(settings.current_runtime_limits)
    if not speaker_roles_bound:
        quality = MeasurementQuality.LIMITED
    else:
        quality = MeasurementQuality.SUFFICIENT
    if interval.size == 0 or (pitch_values.size == 0 and rms == 0):
        quality = MeasurementQuality.INSUFFICIENT
        limitations.append("No measurable voiced or non-zero acoustic signal was present.")

    return AcousticAnalysis(
        analysis_version=settings.version,
        source=settings.source,
        extractor=settings.extractor.name,
        scope=(
            "full_mixed_channel"
            if start_ms == 0 and effective_end_ms == total_duration_ms
            else "evidence_interval"
        ),
        actor=actor,
        speaker_roles_bound=speaker_roles_bound,
        start_ms=start_ms,
        end_ms=effective_end_ms,
        duration_ms=duration_ms,
        sample_rate_hz=media.audio_sample_rate,
        quality=quality,
        pitch=PitchMeasurement(
            available=bool(pitch_values.size),
            analysis_frame_count=pitch_frame_count,
            voiced_frame_count=int(pitch_values.size),
            voiced_frame_fraction=(
                0.0 if pitch_frame_count == 0 else round(pitch_values.size / pitch_frame_count, 6)
            ),
            mean_hz=pitch_mean,
            median_hz=pitch_median,
            standard_deviation_hz=pitch_std,
            percentile_10_hz=pitch_p10,
            percentile_90_hz=pitch_p90,
        ),
        intensity=IntensityMeasurement(
            available=bool(intensity_values.size and rms > 0),
            analysis_frame_count=intensity_frame_count,
            measured_frame_count=int(intensity_values.size),
            mean_db=intensity_mean if rms > 0 else None,
            standard_deviation_db=intensity_std if rms > 0 else None,
            percentile_10_db=intensity_p10 if rms > 0 else None,
            percentile_90_db=intensity_p90 if rms > 0 else None,
            rms_dbfs=_dbfs(rms),
            peak_dbfs=_dbfs(peak),
        ),
        speech_rate=SpeechRateMeasurement(
            available=False,
            value=None,
            limitation_reason=(
                "Time-aligned text and speaker segments are not available for this mixed track."
            ),
        ),
        interpretation_role=settings.interpretation_policy.role,
        limitations=list(dict.fromkeys(limitations)),
    )
