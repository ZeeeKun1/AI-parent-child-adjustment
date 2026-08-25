"""Local acoustic feature extraction for the two-stage recognition pipeline.

This module computes RMS energy and silence gaps from the audio window,
complementing the F0 and speaker-label information already available from
:mod:`coregulation_poc.acoustics.speaker_binding`.

The output :class:`AcousticFeatures` is consumed by the stage-2 judgment
model alongside the :class:`PerceptionReport` from stage 1.
"""

from __future__ import annotations

import numpy as np

from coregulation_poc.acoustics.speaker_binding import SpeakerBinding
from coregulation_poc.models import AcousticFeatures, AcousticSegment, SilenceGap


def _compute_segment_rms(
    audio_chunks: tuple,
    start_ms: int,
    end_ms: int,
    sample_rate: int = 16_000,
) -> float:
    """Compute normalised RMS energy for a time-limited audio region.

    Returns a float in [0, 1] where 1 is the maximum possible PCM16 amplitude.
    """
    chunk_boundaries: list[tuple[int, int, int]] = []
    pcm_parts: list[bytes] = []
    offset = 0
    for chunk in audio_chunks:
        payload = chunk.payload
        if not payload:
            continue
        num_samples = len(payload) // 2
        if num_samples == 0:
            continue
        chunk_boundaries.append((offset, offset + num_samples, chunk.timestamp_ms))
        pcm_parts.append(payload)
        offset += num_samples

    if not pcm_parts:
        return 0.0

    pcm_data = b"".join(pcm_parts)
    all_samples = np.frombuffer(pcm_data, dtype="<i2").astype(np.float64) / 32768.0

    def _ms_to_sample(ms: int) -> int:
        """Map a session timestamp to the matching audio chunk's sample offset."""
        if ms <= chunk_boundaries[0][2]:
            return chunk_boundaries[0][0]

        lo, hi = 0, len(chunk_boundaries) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if chunk_boundaries[mid][2] <= ms:
                lo = mid
            else:
                hi = mid - 1

        start_s, end_s, timestamp_ms = chunk_boundaries[lo]
        delta_samples = round((ms - timestamp_ms) * sample_rate / 1000)
        return max(start_s, min(start_s + delta_samples, end_s))

    start_sample = _ms_to_sample(start_ms)
    end_sample = _ms_to_sample(end_ms)
    if end_sample <= start_sample:
        return 0.0

    segment = all_samples[start_sample:end_sample]
    if segment.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(segment))))
    return min(rms, 1.0)


def extract_acoustic_features(
    audio_chunks: tuple,
    binding: SpeakerBinding,
    *,
    sample_rate: int = 16_000,
    min_silence_ms: int = 500,
) -> AcousticFeatures:
    """Build :class:`AcousticFeatures` from audio chunks and speaker binding.

    Parameters
    ----------
    audio_chunks
        Chronologically ordered audio chunks from a ``MediaWindow``.
    binding
        Result of :func:`bind_speakers` — provides voiced segments with
        speaker labels and F0 statistics.
    sample_rate
        PCM sample rate (default 16 kHz).
    min_silence_ms
        Minimum gap duration to record as a silence gap.
    """
    if not binding.segments:
        return AcousticFeatures()

    segments: list[AcousticSegment] = []
    for seg in binding.segments:
        rms = _compute_segment_rms(audio_chunks, seg.start_ms, seg.end_ms, sample_rate)
        segments.append(
            AcousticSegment(
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                speaker=seg.speaker.value,
                mean_f0_hz=seg.mean_f0_hz if seg.mean_f0_hz > 0 else None,
                median_f0_hz=seg.median_f0_hz if seg.median_f0_hz > 0 else None,
                rms_energy=round(rms, 4),
            )
        )

    # Sort segments by start time for silence gap computation
    segments.sort(key=lambda s: s.start_ms)

    # Compute silence gaps between consecutive segments
    silence_gaps: list[SilenceGap] = []
    for i in range(1, len(segments)):
        prev_end = segments[i - 1].end_ms
        curr_start = segments[i].start_ms
        gap = curr_start - prev_end
        if gap >= min_silence_ms:
            silence_gaps.append(
                SilenceGap(
                    start_ms=prev_end,
                    end_ms=curr_start,
                    duration_ms=gap,
                )
            )

    total_speech = sum(s.end_ms - s.start_ms for s in segments)
    total_silence = sum(g.duration_ms for g in silence_gaps)

    return AcousticFeatures(
        segments=segments,
        silence_gaps=silence_gaps,
        total_speech_ms=total_speech,
        total_silence_ms=total_silence,
    )


def merge_perception_into_acoustic(
    features: AcousticFeatures,
    speech_turns: list,
) -> AcousticFeatures:
    """Attach transcribed text from perception to matching acoustic segments.

    Matches by temporal overlap: a speech turn's [start_ms, end_ms] interval
    must overlap with an acoustic segment's interval for the text to be
    attached.
    """
    updated_segments: list[AcousticSegment] = []
    for seg in features.segments:
        best_text: str | None = None
        best_overlap = 0
        for turn in speech_turns:
            overlap_start = max(seg.start_ms, turn.start_ms)
            overlap_end = min(seg.end_ms, turn.end_ms)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_text = turn.text
        updated_segments.append(
            seg.model_copy(update={"text": best_text})
        )
    return features.model_copy(update={"segments": updated_segments})
