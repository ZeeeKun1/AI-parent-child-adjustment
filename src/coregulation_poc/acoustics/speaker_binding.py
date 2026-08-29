"""F0-based parent/child speaker binding for mixed-channel replay audio.

This module provides a lightweight, offline speaker classification that
segments the mixed audio track into voiced utterances, extracts the
fundamental frequency (F0) of each utterance, and clusters utterances
into two groups.  The group with the lower mean F0 is labelled *parent*
and the group with the higher mean F0 is labelled *child*, following the
well-established finding that children's vocal-fold vibration rate is
substantially higher than adults'.

The binding is intentionally conservative: it only reports success when
at least two voiced segments with reliable F0 exist and the two cluster
centroids are separated by a meaningful gap.  When binding fails, callers
fall back to the existing ``actor=unknown`` behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import parselmouth

from coregulation_poc.acoustics.speaker_enrollment import SpeakerEnrollment, extract_embedding


class SpeakerLabel(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One voiced utterance with its F0 statistics and speaker label."""

    start_ms: int
    end_ms: int
    mean_f0_hz: float
    median_f0_hz: float
    voiced_frame_count: int
    speaker: SpeakerLabel = SpeakerLabel.UNKNOWN
    parent_cosine: float | None = None
    child_cosine: float | None = None
    provider_score: float | None = None
    parent_provider_score: float | None = None
    child_provider_score: float | None = None
    confidence: str | None = None
    forced_assignment: bool = False


@dataclass(frozen=True, slots=True)
class SpeakerBinding:
    """Result of parent/child speaker binding.

    The binding may use either F0 K-means clustering (when no voiceprint
    enrollment is available) or embedding cosine similarity (when
    pre-registered voiceprints exist).  The ``method`` field indicates
    which approach was used.
    """

    bound: bool
    segments: list[SpeakerSegment] = field(default_factory=list)
    parent_mean_f0_hz: float | None = None
    child_mean_f0_hz: float | None = None
    parent_median_f0_hz: float | None = None
    child_median_f0_hz: float | None = None
    parent_segment_count: int = 0
    child_segment_count: int = 0
    method: str = "f0_kmeans_clustering"
    separation_hz: float | None = None
    limitation_reason: str | None = None
    parent_mean_cosine: float | None = None
    child_mean_cosine: float | None = None
    provider_request_count: int = 0
    provider_failure_count: int = 0
    low_confidence_segment_count: int = 0

    def to_prompt_description(self) -> str:
        """Render a compact natural-language description for the model prompt."""
        if not self.bound:
            return (
                "Speaker binding is not available. "
                "Use actor=unknown unless the role is directly and unambiguously observable."
            )
        is_tencent = self.method.startswith("tencent_voiceprint")
        is_embedding = "embedding" in self.method
        if is_tencent:
            lines = [
                (
                    "Speaker roles are anchored by pairwise Tencent Cloud 1:1 "
                    "voiceprint matching against both enrolled roles; explicitly "
                    "marked forced assignments are continuity "
                    "inferences, not direct provider matches:"
                ),
                f"  Parent: {self.parent_segment_count} segments.",
                f"  Child: {self.child_segment_count} segments.",
                (
                    "  Low-confidence forced assignments: "
                    f"{self.low_confidence_segment_count}."
                ),
                f"  Provider request failures: {self.provider_failure_count}.",
                (
                    "Voiced segments (start_ms-end_ms, speaker, parent score, "
                    "child score, confidence):"
                ),
            ]
            for seg in self.segments:
                parent_score = (
                    "N/A"
                    if seg.parent_provider_score is None
                    else f"{seg.parent_provider_score:.1f}"
                )
                child_score = (
                    "N/A"
                    if seg.child_provider_score is None
                    else f"{seg.child_provider_score:.1f}"
                )
                confidence = seg.confidence or "unavailable"
                suffix = ", forced assignment" if seg.forced_assignment else ""
                lines.append(
                    f"  {seg.start_ms}-{seg.end_ms} ms: {seg.speaker.value} "
                    f"(parent {parent_score}, child {child_score}, "
                    f"{confidence}{suffix})"
                )
        elif is_embedding:
            parent_cos_str = (
                f"mean cosine = {self.parent_mean_cosine:.3f}"
                if self.parent_mean_cosine is not None
                else "cosine N/A"
            )
            child_cos_str = (
                f"mean cosine = {self.child_mean_cosine:.3f}"
                if self.child_mean_cosine is not None
                else "cosine N/A"
            )
            lines = [
                "Speaker roles are externally bound by voiceprint embedding matching:",
                f"  Parent: {self.parent_segment_count} segments, {parent_cos_str}.",
                f"  Child: {self.child_segment_count} segments, {child_cos_str}.",
                "Voiced segments (start_ms-end_ms, speaker, cosine_sim):",
            ]
            for seg in self.segments:
                if seg.parent_cosine is not None and seg.child_cosine is not None:
                    lines.append(
                        f"  {seg.start_ms}-{seg.end_ms} ms: "
                        f"{seg.speaker.value} "
                        f"(parent {seg.parent_cosine:.3f}, child {seg.child_cosine:.3f})"
                    )
                else:
                    lines.append(
                        f"  {seg.start_ms}-{seg.end_ms} ms: {seg.speaker.value}"
                    )
        else:
            lines = [
                "Speaker roles are externally bound by F0 clustering:",
                f"  Parent: mean F0 = {self.parent_mean_f0_hz:.0f} Hz, "
                f"median F0 = {self.parent_median_f0_hz:.0f} Hz, "
                f"{self.parent_segment_count} voiced segments.",
                f"  Child: mean F0 = {self.child_mean_f0_hz:.0f} Hz, "
                f"median F0 = {self.child_median_f0_hz:.0f} Hz, "
                f"{self.child_segment_count} voiced segments.",
                f"  Cluster separation: {self.separation_hz:.0f} Hz.",
                "Voiced segments (start_ms-end_ms, speaker, mean_F0_Hz):",
            ]
            for seg in self.segments:
                lines.append(
                    f"  {seg.start_ms}-{seg.end_ms} ms: "
                    f"{seg.speaker.value} (mean F0 {seg.mean_f0_hz:.0f} Hz)"
                )
        lines.append(
            "Use this binding to assign actor=parent or actor=child in evidence. "
            "The binding is probabilistic; if visual or content evidence clearly "
            "contradicts it for a specific utterance, note the discrepancy in the "
            "observation but still use the binding as the default."
        )
        return "\n".join(lines)


def _energy_based_vad(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = 20,
    min_segment_ms: int = 200,
    merge_gap_ms: int = 150,
) -> list[tuple[int, int]]:
    """Split audio into voiced segments using adaptive energy thresholding.

    Returns a list of *(start_sample, end_sample)* pairs.
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    if frame_size < 2 or len(samples) < frame_size:
        return []

    num_frames = len(samples) // frame_size
    if num_frames < 2:
        return []

    energies = np.empty(num_frames, dtype=np.float64)
    for i in range(num_frames):
        block = samples[i * frame_size : (i + 1) * frame_size]
        energies[i] = float(np.mean(np.square(block)))

    # Adaptive threshold: use a percentile of the energy distribution.
    # The 35th percentile gives a reasonable voice/silence cutoff for
    # homework interactions where both speakers are near the microphone.
    threshold = max(np.percentile(energies, 35) * 0.4, 1e-6)
    voiced = energies > threshold

    min_frames = max(1, int(min_segment_ms / frame_ms))
    merge_frames = max(1, int(merge_gap_ms / frame_ms))

    segments: list[tuple[int, int]] = []
    i = 0
    while i < num_frames:
        if not voiced[i]:
            i += 1
            continue
        start = i
        # Extend through consecutive voiced frames, bridging short gaps.
        while i < num_frames:
            if voiced[i]:
                i += 1
                continue
            # Look ahead to see if we can bridge a gap.
            gap_start = i
            while i < num_frames and not voiced[i] and (i - gap_start) < merge_frames:
                i += 1
            if i < num_frames and voiced[i]:
                continue  # gap bridged, keep extending
            break  # gap too long, segment ends here
        end = i
        if end - start >= min_frames:
            segments.append(
                (start * frame_size, min(end * frame_size, len(samples)))
            )

    return segments


def _extract_segment_f0(
    samples: np.ndarray,
    sample_rate: int,
    *,
    pitch_floor: float = 50.0,
    pitch_ceiling: float = 800.0,
) -> tuple[np.ndarray, int]:
    """Extract voiced F0 values from a short audio segment.

    Returns *(voiced_f0_array, total_pitch_frame_count)*.
    """
    if samples.size < int(sample_rate * 0.01):
        return np.array([], dtype=np.float64), 0
    sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    try:
        pitch = sound.to_pitch_ac(
            time_step=0.01,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )
    except parselmouth.PraatError:
        return np.array([], dtype=np.float64), 0
    raw = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    total = int(raw.size)
    voiced = raw[np.isfinite(raw) & (raw > 0)]
    return voiced, total


def bind_speakers_by_f0(
    audio_chunks: tuple,
    *,
    sample_rate: int = 16_000,
    pitch_floor: float = 50.0,
    pitch_ceiling: float = 800.0,
    min_voiced_frames_per_segment: int = 3,
    min_segments: int = 2,
    min_centroid_separation_hz: float = 25.0,
) -> SpeakerBinding:
    """Classify voiced segments as parent or child using F0 K-means clustering.

    Parameters
    ----------
    audio_chunks
        Chronologically ordered audio chunks from a ``MediaWindow``.
    sample_rate
        PCM sample rate (default 16 kHz).
    pitch_floor, pitch_ceiling
        Praat pitch tracking range.
    min_voiced_frames_per_segment
        Minimum voiced frames (10 ms each) for a segment to be retained.
    min_segments
        Minimum number of retained voiced segments for clustering.
    min_centroid_separation_hz
        Minimum gap between the two cluster centroids for binding to succeed.
    """
    # --- Concatenate PCM samples and track chunk timestamps ----------------
    chunk_boundaries: list[tuple[int, int, int]] = []  # (start_sample, end_sample, timestamp_ms)
    pcm_parts: list[bytes] = []
    offset = 0
    first_ts: int | None = None
    for chunk in audio_chunks:
        payload = chunk.payload
        if not payload:
            continue
        num_samples = len(payload) // 2
        if num_samples == 0:
            continue
        ts = chunk.timestamp_ms
        if first_ts is None:
            first_ts = ts
        chunk_boundaries.append((offset, offset + num_samples, ts))
        pcm_parts.append(payload)
        offset += num_samples

    if not pcm_parts or first_ts is None:
        return SpeakerBinding(
            bound=False,
            limitation_reason="No audio data available for speaker binding.",
        )

    pcm_data = b"".join(pcm_parts)
    samples = np.frombuffer(pcm_data, dtype="<i2").astype(np.float64) / 32768.0

    # --- Voice activity detection ------------------------------------------
    raw_segments = _energy_based_vad(samples, sample_rate)
    if len(raw_segments) < min_segments:
        return SpeakerBinding(
            bound=False,
            limitation_reason="Insufficient voiced segments for speaker separation.",
        )

    # --- F0 extraction per segment -----------------------------------------
    segment_data: list[tuple[int, int, np.ndarray]] = []  # (start_ms, end_ms, f0_values)

    def _sample_to_ms(sample_offset: int) -> int:
        """Map a sample offset to a session timestamp in milliseconds."""
        # Binary search through chunk boundaries
        lo, hi = 0, len(chunk_boundaries) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if chunk_boundaries[mid][1] <= sample_offset:
                lo = mid + 1
            else:
                hi = mid
        start_s, _, ts = chunk_boundaries[lo]
        delta_ms = round((sample_offset - start_s) * 1000 / sample_rate)
        return ts + delta_ms

    for start_sample, end_sample in raw_segments:
        segment_audio = samples[start_sample:end_sample]
        voiced_f0, total_frames = _extract_segment_f0(
            segment_audio,
            sample_rate,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )
        if voiced_f0.size < min_voiced_frames_per_segment:
            continue
        start_ms = _sample_to_ms(start_sample)
        end_ms = _sample_to_ms(end_sample)
        segment_data.append((start_ms, end_ms, voiced_f0))

    if len(segment_data) < min_segments:
        return SpeakerBinding(
            bound=False,
            limitation_reason="Insufficient segments with reliable F0 for clustering.",
        )

    # --- K-means clustering (k=2) on segment mean F0 -----------------------
    f0_means = np.array([float(np.mean(f0)) for _, _, f0 in segment_data])

    # Initialise centroids from sorted split
    order = np.argsort(f0_means)
    mid = len(order) // 2
    centroid_low = float(np.mean(f0_means[order[:mid]])) if mid > 0 else float(f0_means[order[0]])
    centroid_high = (
        float(np.mean(f0_means[order[mid:]]))
        if mid < len(order)
        else float(f0_means[order[-1]])
    )

    if abs(centroid_high - centroid_low) < min_centroid_separation_hz:
        return SpeakerBinding(
            bound=False,
            limitation_reason=(
                f"F0 cluster separation ({abs(centroid_high - centroid_low):.1f} Hz) "
                f"is below the {min_centroid_separation_hz:.0f} Hz minimum for reliable binding."
            ),
        )

    labels = np.zeros(len(f0_means), dtype=int)
    for _ in range(20):
        dist_low = np.abs(f0_means - centroid_low)
        dist_high = np.abs(f0_means - centroid_high)
        new_labels = (dist_high < dist_low).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        low_mask = labels == 0
        high_mask = labels == 1
        if np.any(low_mask):
            centroid_low = float(np.mean(f0_means[low_mask]))
        if np.any(high_mask):
            centroid_high = float(np.mean(f0_means[high_mask]))

    # Ensure centroid_low < centroid_high (lower F0 = parent)
    if centroid_low > centroid_high:
        centroid_low, centroid_high = centroid_high, centroid_low
        labels = 1 - labels

    separation = round(abs(centroid_high - centroid_low), 1)

    # --- Build labelled segments -------------------------------------------
    parent_f0_all: list[float] = []
    child_f0_all: list[float] = []
    speaker_segments: list[SpeakerSegment] = []

    for (start_ms, end_ms, f0_values), label in zip(segment_data, labels, strict=True):
        mean_f0 = round(float(np.mean(f0_values)), 1)
        median_f0 = round(float(np.median(f0_values)), 1)
        if label == 0:  # lower F0 -> parent
            speaker = SpeakerLabel.PARENT
            parent_f0_all.extend(f0_values.tolist())
        else:  # higher F0 -> child
            speaker = SpeakerLabel.CHILD
            child_f0_all.extend(f0_values.tolist())
        speaker_segments.append(
            SpeakerSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                mean_f0_hz=mean_f0,
                median_f0_hz=median_f0,
                voiced_frame_count=int(f0_values.size),
                speaker=speaker,
            )
        )

    parent_arr = (
        np.array(parent_f0_all, dtype=np.float64)
        if parent_f0_all
        else np.array([], dtype=np.float64)
    )
    child_arr = (
        np.array(child_f0_all, dtype=np.float64)
        if child_f0_all
        else np.array([], dtype=np.float64)
    )

    # Both clusters must have at least one segment
    if not parent_f0_all or not child_f0_all:
        return SpeakerBinding(
            bound=False,
            limitation_reason=(
                "All voiced segments fell into one cluster; cannot separate speakers."
            ),
        )

    return SpeakerBinding(
        bound=True,
        segments=speaker_segments,
        parent_mean_f0_hz=round(float(np.mean(parent_arr)), 1) if parent_arr.size else None,
        child_mean_f0_hz=round(float(np.mean(child_arr)), 1) if child_arr.size else None,
        parent_median_f0_hz=round(float(np.median(parent_arr)), 1) if parent_arr.size else None,
        child_median_f0_hz=round(float(np.median(child_arr)), 1) if child_arr.size else None,
        parent_segment_count=int(np.sum(labels == 0)),
        child_segment_count=int(np.sum(labels == 1)),
        separation_hz=separation,
    )


# ---------------------------------------------------------------------------
# Embedding-based speaker binding (requires pre-registered voiceprints)
# ---------------------------------------------------------------------------


def _prepare_audio_segments(
    audio_chunks: tuple,
    sample_rate: int = 16_000,
    *,
    min_segment_ms: int = 200,
    merge_gap_ms: int = 150,
) -> tuple[np.ndarray, int, list[tuple[int, int, int, int]]]:
    """Concatenate audio chunks, run VAD, and return timed segments.

    Returns ``(samples, sample_rate, segments)`` where each segment is
    ``(start_ms, end_ms, start_sample, end_sample)``.
    """
    chunk_boundaries: list[tuple[int, int, int]] = []
    pcm_parts: list[bytes] = []
    offset = 0
    first_ts: int | None = None
    for chunk in audio_chunks:
        payload = chunk.payload
        if not payload:
            continue
        num_samples = len(payload) // 2
        if num_samples == 0:
            continue
        ts = chunk.timestamp_ms
        if first_ts is None:
            first_ts = ts
        chunk_boundaries.append((offset, offset + num_samples, ts))
        pcm_parts.append(payload)
        offset += num_samples

    if not pcm_parts or first_ts is None:
        return np.array([], dtype=np.float64), sample_rate, []

    pcm_data = b"".join(pcm_parts)
    samples = np.frombuffer(pcm_data, dtype="<i2").astype(np.float64) / 32768.0

    raw_segments = _energy_based_vad(
        samples, sample_rate, min_segment_ms=min_segment_ms, merge_gap_ms=merge_gap_ms
    )

    def _sample_to_ms(sample_offset: int) -> int:
        lo, hi = 0, len(chunk_boundaries) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if chunk_boundaries[mid][1] <= sample_offset:
                lo = mid + 1
            else:
                hi = mid
        start_s, _, ts = chunk_boundaries[lo]
        delta_ms = round((sample_offset - start_s) * 1000 / sample_rate)
        return ts + delta_ms

    timed: list[tuple[int, int, int, int]] = []
    for start_sample, end_sample in raw_segments:
        start_ms = _sample_to_ms(start_sample)
        end_ms = _sample_to_ms(end_sample)
        timed.append((start_ms, end_ms, start_sample, end_sample))

    return samples, sample_rate, timed


def bind_speakers_by_embedding(
    audio_chunks: tuple,
    enrollment: SpeakerEnrollment,
    *,
    sample_rate: int = 16_000,
    min_segment_ms: int = 1_600,
    min_segments: int = 2,
    cosine_threshold: float = 0.25,
) -> SpeakerBinding:
    """Classify voiced segments as parent or child using embedding cosine similarity.

    This method compares each voiced segment's speaker embedding against
    pre-registered parent and child voiceprints.  It does not depend on F0
    separation, so it remains reliable even when adult and child pitch
    ranges overlap.

    Parameters
    ----------
    audio_chunks
        Chronologically ordered audio chunks from a ``MediaWindow``.
    enrollment
        Completed :class:`SpeakerEnrollment` with both parent and child
        voiceprints registered.
    sample_rate
        PCM sample rate (default 16 kHz).
    min_segment_ms
        Minimum segment duration for embedding extraction.  Resemblyzer's
        VoiceEncoder requires at least ~1.6 s of audio.
    min_segments
        Minimum number of retained segments for binding.
    cosine_threshold
        Minimum cosine similarity to assign a segment to a speaker.
        Segments below this threshold for both speakers remain
        ``unknown``.
    """
    if not enrollment.is_complete:
        return SpeakerBinding(
            bound=False,
            method="embedding_cosine",
            limitation_reason=(
                "Enrollment is incomplete; both parent and child must be "
                "registered before embedding-based binding can be used."
            ),
        )

    parent_emb = enrollment.parent_embedding
    child_emb = enrollment.child_embedding

    # --- Concatenate PCM, VAD, and build timed segments -------------------
    samples, sr, timed_segments = _prepare_audio_segments(
        audio_chunks, sample_rate, min_segment_ms=min_segment_ms
    )

    if not timed_segments:
        return SpeakerBinding(
            bound=False,
            method="embedding_cosine",
            limitation_reason="No audio data available for speaker binding.",
        )

    if len(timed_segments) < min_segments:
        unidentified_segments: list[SpeakerSegment] = []
        for start_ms, end_ms, start_sample, end_sample in timed_segments:
            segment_audio = samples[start_sample:end_sample].astype(np.float32)
            voiced_f0, _ = _extract_segment_f0(segment_audio, sr)
            unidentified_segments.append(
                SpeakerSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    mean_f0_hz=(
                        round(float(np.mean(voiced_f0)), 1)
                        if voiced_f0.size
                        else 0.0
                    ),
                    median_f0_hz=(
                        round(float(np.median(voiced_f0)), 1)
                        if voiced_f0.size
                        else 0.0
                    ),
                    voiced_frame_count=int(voiced_f0.size),
                    speaker=SpeakerLabel.UNKNOWN,
                )
            )
        return SpeakerBinding(
            bound=False,
            segments=unidentified_segments,
            method="embedding_cosine",
            limitation_reason=(
                f"Only {len(timed_segments)} voiced segment(s) >= {min_segment_ms} ms; "
                f"need at least {min_segments}."
            ),
        )

    # --- Extract embedding per segment and compare -------------------------
    speaker_segments: list[SpeakerSegment] = []
    parent_sims: list[float] = []
    child_sims: list[float] = []
    parent_count = 0
    child_count = 0

    for start_ms, end_ms, start_sample, end_sample in timed_segments:
        segment_audio = samples[start_sample:end_sample].astype(np.float32)
        try:
            emb = extract_embedding(segment_audio, sr)
        except Exception:
            continue

        parent_cos = float(np.dot(emb, parent_emb))
        child_cos = float(np.dot(emb, child_emb))

        if parent_cos >= cosine_threshold and parent_cos >= child_cos:
            speaker = SpeakerLabel.PARENT
            parent_sims.append(parent_cos)
            parent_count += 1
        elif child_cos >= cosine_threshold and child_cos > parent_cos:
            speaker = SpeakerLabel.CHILD
            child_sims.append(child_cos)
            child_count += 1
        else:
            speaker = SpeakerLabel.UNKNOWN

        # Extract F0 for informational purposes (not used for classification)
        voiced_f0, _ = _extract_segment_f0(segment_audio, sr)
        mean_f0 = round(float(np.mean(voiced_f0)), 1) if voiced_f0.size else 0.0
        median_f0 = round(float(np.median(voiced_f0)), 1) if voiced_f0.size else 0.0

        speaker_segments.append(
            SpeakerSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                mean_f0_hz=mean_f0,
                median_f0_hz=median_f0,
                voiced_frame_count=int(voiced_f0.size),
                speaker=speaker,
                parent_cosine=round(parent_cos, 3),
                child_cosine=round(child_cos, 3),
            )
        )

    if parent_count == 0 and child_count == 0:
        return SpeakerBinding(
            bound=False,
            segments=speaker_segments,
            method="embedding_cosine",
            limitation_reason=(
                "No segments matched either enrolled speaker above the "
                f"cosine threshold ({cosine_threshold})."
            ),
        )

    return SpeakerBinding(
        bound=True,
        segments=speaker_segments,
        parent_mean_cosine=round(float(np.mean(parent_sims)), 3) if parent_sims else None,
        child_mean_cosine=round(float(np.mean(child_sims)), 3) if child_sims else None,
        parent_segment_count=parent_count,
        child_segment_count=child_count,
        method="embedding_cosine",
    )


def bind_speakers(
    audio_chunks: tuple,
    *,
    enrollment: SpeakerEnrollment | None = None,
    sample_rate: int = 16_000,
    allow_f0_fallback: bool = False,
) -> SpeakerBinding:
    """Bind speakers using the best available method.

    When a completed :class:`SpeakerEnrollment` is provided, embedding-based
    binding is attempted first.  If embedding binding fails, the function
    falls back to F0-based clustering **only** when ``allow_f0_fallback`` is
    ``True``.

    In formal real-time experiments, ``allow_f0_fallback`` must be ``False``
    so that F0 is never used to guess family roles.  Offline replay tools
    may set it to ``True`` for diagnostic purposes.

    Parameters
    ----------
    audio_chunks
        Chronologically ordered audio chunks from a ``MediaWindow``.
    enrollment
        Optional pre-registered voiceprints.  When ``None`` or incomplete,
        the behaviour depends on ``allow_f0_fallback``.
    sample_rate
        PCM sample rate (default 16 kHz).
    allow_f0_fallback
        When ``False`` (default), F0 clustering is never used for identity
        assignment.  When ``True``, F0 clustering is used as a fallback
        when embedding binding fails or no enrollment is available.
    """
    if enrollment is not None and enrollment.is_complete:
        embedding_result = bind_speakers_by_embedding(
            audio_chunks, enrollment, sample_rate=sample_rate
        )
        if embedding_result.bound:
            return embedding_result

        if not allow_f0_fallback:
            return SpeakerBinding(
                bound=False,
                segments=embedding_result.segments,
                method="embedding_cosine",
                limitation_reason=(
                    f"Embedding binding failed ({embedding_result.limitation_reason}); "
                    f"F0 fallback disabled for real-time mode."
                ),
            )

        # Embedding binding failed; try F0 as fallback (offline only)
        f0_result = bind_speakers_by_f0(audio_chunks, sample_rate=sample_rate)
        if f0_result.bound:
            return SpeakerBinding(
                bound=True,
                segments=f0_result.segments,
                parent_mean_f0_hz=f0_result.parent_mean_f0_hz,
                child_mean_f0_hz=f0_result.child_mean_f0_hz,
                parent_median_f0_hz=f0_result.parent_median_f0_hz,
                child_median_f0_hz=f0_result.child_median_f0_hz,
                parent_segment_count=f0_result.parent_segment_count,
                child_segment_count=f0_result.child_segment_count,
                method="f0_kmeans_clustering (embedding_fallback)",
                separation_hz=f0_result.separation_hz,
                limitation_reason=(
                    f"Embedding binding failed ({embedding_result.limitation_reason}); "
                    f"fell back to F0 clustering."
                ),
            )

        # Both methods failed
        return SpeakerBinding(
            bound=False,
            method="embedding_cosine+f0_kmeans_clustering",
            limitation_reason=(
                f"Embedding: {embedding_result.limitation_reason}; "
                f"F0: {f0_result.limitation_reason}"
            ),
        )

    # No enrollment available
    if not allow_f0_fallback:
        return SpeakerBinding(
            bound=False,
            method="none",
            limitation_reason=(
                "No voiceprint enrollment available and F0 fallback is "
                "disabled for real-time mode."
            ),
        )
    return bind_speakers_by_f0(audio_chunks, sample_rate=sample_rate)
