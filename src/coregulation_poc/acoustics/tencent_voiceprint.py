"""Tencent Cloud voiceprint enrollment and parent/child identification.

The browser sends short PCM16 enrollment recordings to the application server.
This module registers those recordings with Tencent Cloud, keeps only opaque
voiceprint identifiers in memory for the active experiment, performs 1:N group
verification on voiced interaction segments, and deletes the remote records at
the end of the experiment.
"""

from __future__ import annotations

import base64
import secrets
import string
import threading
from dataclasses import dataclass, field

import numpy as np
from tencentcloud.asr.v20190614 import asr_client, models
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from coregulation_poc.acoustics.speaker_binding import (
    SpeakerBinding,
    SpeakerLabel,
    SpeakerSegment,
    _extract_segment_f0,
    _prepare_audio_segments,
)
from coregulation_poc.acoustics.speaker_enrollment import SpeakerEnrollment


class TencentVoiceprintError(RuntimeError):
    """Safe application-level error for Tencent voiceprint failures."""


@dataclass(frozen=True, slots=True)
class TencentEnrolledSpeaker:
    """One cloud-enrolled speaker; the identifier must never reach the browser."""

    label: str
    duration_ms: int
    voiceprint_id: str = field(repr=False)
    audio_source: str = "browser_recording_not_saved_locally"


@dataclass(frozen=True, slots=True)
class TencentSpeakerEnrollment:
    """Temporary Tencent voiceprint group for one experiment session."""

    family_id: str
    group_id: str = field(repr=False)
    speakers: dict[str, TencentEnrolledSpeaker] = field(default_factory=dict)
    model_name: str = "tencent_voiceprint_1n"

    @property
    def is_complete(self) -> bool:
        return "parent" in self.speakers and "child" in self.speakers


def _validate_pcm16_enrollment(
    pcm_audio: bytes,
    speaker_label: str,
    *,
    sample_rate: int,
) -> int:
    if speaker_label not in {"parent", "child"}:
        raise ValueError("speaker_label must be 'parent' or 'child'")
    if sample_rate != 16_000:
        raise ValueError("browser enrollment audio must use a 16000 Hz sample rate")
    if not pcm_audio or len(pcm_audio) % 2:
        raise ValueError("browser enrollment audio must be non-empty PCM16 data")

    samples = np.frombuffer(pcm_audio, dtype="<i2")
    duration_seconds = samples.size / sample_rate
    if duration_seconds < 3:
        raise ValueError("请至少录制 3 秒清晰语音")
    if duration_seconds > 15:
        raise ValueError("单次声纹录制不能超过 15 秒")
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms < 120:
        raise ValueError("没有检测到清晰语音，请靠近设备后重新录制")
    return round(duration_seconds * 1000)


def _new_group_id() -> str:
    # VoicePrintEnroll currently accepts letters and underscores for GroupId.
    random_letters = "".join(secrets.choice(string.ascii_letters) for _ in range(24))
    return f"coreg_{random_letters}"


class TencentVoiceprintService:
    """Synchronous official-SDK adapter; call from ``asyncio.to_thread``."""

    provider_name = "tencent_cloud_voiceprint"

    def __init__(
        self,
        *,
        secret_id: str,
        secret_key: str,
        region: str = "ap-guangzhou",
        minimum_score: float = 70.0,
        timeout_seconds: int = 15,
        client: object | None = None,
    ) -> None:
        if not secret_id.strip() or not secret_key.strip():
            raise ValueError("Tencent SecretId and SecretKey are both required")
        if not 0 <= minimum_score <= 100:
            raise ValueError("Tencent voiceprint minimum score must be between 0 and 100")
        self.minimum_score = float(minimum_score)
        self.region = region
        self._lock = threading.RLock()
        self._call_count = 0

        if client is None:
            cred = credential.Credential(secret_id, secret_key)
            http_profile = HttpProfile(
                endpoint="asr.tencentcloudapi.com",
                reqTimeout=timeout_seconds,
                keepAlive=True,
            )
            profile = ClientProfile(httpProfile=http_profile)
            client = asr_client.AsrClient(cred, region, profile)
        self._client = client

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def _call(self, method_name: str, request: object) -> object:
        try:
            with self._lock:
                self._call_count += 1
                method = getattr(self._client, method_name)
                return method(request)
        except TencentCloudSDKException as exc:
            error_code = exc.get_code() or ""
            if method_name == "VoicePrintDelete" and "NotExistent" in error_code:
                return object()
            raise TencentVoiceprintError(
                f"Tencent voiceprint request {method_name} failed: {exc}"
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise TencentVoiceprintError(
                f"Tencent voiceprint request {method_name} failed: {exc}"
            ) from exc
        except Exception as exc:
            raise TencentVoiceprintError(
                f"Tencent voiceprint request {method_name} returned an invalid response"
            ) from exc

    def enroll_speaker(
        self,
        pcm_audio: bytes,
        speaker_label: str,
        family_id: str,
        current: TencentSpeakerEnrollment | None = None,
        *,
        sample_rate: int = 16_000,
    ) -> TencentSpeakerEnrollment:
        """Register or replace one role without retaining the raw audio locally."""

        duration_ms = _validate_pcm16_enrollment(
            pcm_audio,
            speaker_label,
            sample_rate=sample_rate,
        )
        encoded = base64.b64encode(pcm_audio).decode("ascii")
        group_id = current.group_id if current is not None else _new_group_id()
        existing = None if current is None else current.speakers.get(speaker_label)

        if existing is None:
            request = models.VoicePrintEnrollRequest()
            request.VoiceFormat = 0
            request.SampleRate = sample_rate
            request.Data = encoded
            request.SpeakerNick = speaker_label
            request.GroupId = group_id
            response = self._call("VoicePrintEnroll", request)
        else:
            request = models.VoicePrintUpdateRequest()
            request.VoiceFormat = 0
            request.SampleRate = sample_rate
            request.Data = encoded
            request.SpeakerNick = speaker_label
            request.VoicePrintId = existing.voiceprint_id
            response = self._call("VoicePrintUpdate", request)

        response_data = getattr(response, "Data", None)
        voiceprint_id = getattr(response_data, "VoicePrintId", None)
        if not isinstance(voiceprint_id, str) or not voiceprint_id:
            raise TencentVoiceprintError("Tencent voiceprint registration returned no identifier")

        speakers = {} if current is None else dict(current.speakers)
        speakers[speaker_label] = TencentEnrolledSpeaker(
            label=speaker_label,
            duration_ms=duration_ms,
            voiceprint_id=voiceprint_id,
        )
        return TencentSpeakerEnrollment(
            family_id=family_id,
            group_id=group_id,
            speakers=speakers,
        )

    def identify_speakers(
        self,
        audio_chunks: tuple,
        enrollment: TencentSpeakerEnrollment,
        *,
        sample_rate: int = 16_000,
        min_segment_ms: int = 800,
    ) -> SpeakerBinding:
        """Assign every accepted voiced segment to the best parent/child match."""

        if not enrollment.is_complete:
            raise TencentVoiceprintError(
                "Tencent voiceprint enrollment is incomplete for this session"
            )
        samples, sample_rate, timed_segments = _prepare_audio_segments(
            audio_chunks,
            sample_rate,
            min_segment_ms=200,
            merge_gap_ms=250,
        )
        if not timed_segments:
            return SpeakerBinding(
                bound=False,
                method="tencent_voiceprint_1n",
                limitation_reason="No sufficiently long voiced segment was available.",
            )

        verified_candidates = [
            item for item in timed_segments if item[1] - item[0] >= min_segment_ms
        ]
        if not verified_candidates:
            raise TencentVoiceprintError(
                "Voiced audio was present, but no utterance was long enough "
                "for reliable Tencent voiceprint verification"
            )

        id_to_label = {
            speaker.voiceprint_id: label
            for label, speaker in enrollment.speakers.items()
        }
        segments: list[SpeakerSegment] = []
        parent_count = 0
        child_count = 0
        low_confidence_count = 0
        request_count = 0

        for start_ms, end_ms, start_sample, end_sample in verified_candidates:
            segment_audio = samples[start_sample:end_sample]
            pcm_audio = np.clip(
                np.rint(segment_audio * 32768.0), -32768, 32767
            ).astype("<i2").tobytes()

            request = models.VoicePrintGroupVerifyRequest()
            request.VoiceFormat = 0
            request.SampleRate = sample_rate
            request.Data = base64.b64encode(pcm_audio).decode("ascii")
            request.GroupId = enrollment.group_id
            request.TopN = 2
            response = self._call("VoicePrintGroupVerify", request)
            request_count += 1

            verify_tops = getattr(getattr(response, "Data", None), "VerifyTops", None)
            known_matches: list[tuple[float, str]] = []
            for match in verify_tops or []:
                voiceprint_id = getattr(match, "VoicePrintId", None)
                label = id_to_label.get(voiceprint_id)
                if label is None:
                    continue
                try:
                    score = float(match.Score)
                except (TypeError, ValueError):
                    continue
                known_matches.append((score, label))
            if not known_matches:
                raise TencentVoiceprintError(
                    "Tencent 1:N verification returned no registered family speaker"
                )

            score, label = max(known_matches, key=lambda item: item[0])
            forced_assignment = score < self.minimum_score
            if forced_assignment:
                low_confidence_count += 1
            confidence = "low" if forced_assignment else "high"
            speaker = (
                SpeakerLabel.PARENT if label == "parent" else SpeakerLabel.CHILD
            )
            if speaker is SpeakerLabel.PARENT:
                parent_count += 1
            else:
                child_count += 1

            voiced_f0, _ = _extract_segment_f0(segment_audio, sample_rate)
            segments.append(
                SpeakerSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    mean_f0_hz=(
                        round(float(np.mean(voiced_f0)), 1) if voiced_f0.size else 0.0
                    ),
                    median_f0_hz=(
                        round(float(np.median(voiced_f0)), 1) if voiced_f0.size else 0.0
                    ),
                    voiced_frame_count=int(voiced_f0.size),
                    speaker=speaker,
                    provider_score=round(score, 1),
                    confidence=confidence,
                    forced_assignment=forced_assignment,
                )
            )

        # Very short responses cannot be submitted reliably to the provider.
        # Keep the no-UNKNOWN experiment contract by assigning them to the
        # nearest provider-verified turn and marking that continuity inference
        # explicitly as low confidence and forced.
        short_segments = [
            item for item in timed_segments if item[1] - item[0] < min_segment_ms
        ]
        for start_ms, end_ms, start_sample, end_sample in short_segments:
            midpoint = (start_ms + end_ms) / 2
            nearest = min(
                segments,
                key=lambda item: abs(((item.start_ms + item.end_ms) / 2) - midpoint),
            )
            segment_audio = samples[start_sample:end_sample]
            voiced_f0, _ = _extract_segment_f0(segment_audio, sample_rate)
            if nearest.speaker is SpeakerLabel.PARENT:
                parent_count += 1
            else:
                child_count += 1
            low_confidence_count += 1
            segments.append(
                SpeakerSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    mean_f0_hz=(
                        round(float(np.mean(voiced_f0)), 1) if voiced_f0.size else 0.0
                    ),
                    median_f0_hz=(
                        round(float(np.median(voiced_f0)), 1) if voiced_f0.size else 0.0
                    ),
                    voiced_frame_count=int(voiced_f0.size),
                    speaker=nearest.speaker,
                    provider_score=None,
                    confidence="low",
                    forced_assignment=True,
                )
            )

        segments.sort(key=lambda item: (item.start_ms, item.end_ms))

        return SpeakerBinding(
            bound=bool(segments),
            segments=segments,
            parent_segment_count=parent_count,
            child_segment_count=child_count,
            method="tencent_voiceprint_1n",
            provider_request_count=request_count,
            low_confidence_segment_count=low_confidence_count,
        )

    def delete_enrollment(self, enrollment: TencentSpeakerEnrollment) -> None:
        """Delete remote speaker records, then remove the now-empty group metadata."""

        errors: list[Exception] = []
        for speaker in enrollment.speakers.values():
            request = models.VoicePrintDeleteRequest()
            request.VoicePrintId = speaker.voiceprint_id
            request.DelMod = 0
            try:
                self._call("VoicePrintDelete", request)
            except TencentVoiceprintError as exc:
                errors.append(exc)

        group_request = models.VoicePrintDeleteRequest()
        group_request.GroupId = enrollment.group_id
        group_request.DelMod = 2
        try:
            self._call("VoicePrintDelete", group_request)
        except TencentVoiceprintError as exc:
            errors.append(exc)

        if errors:
            raise TencentVoiceprintError(
                f"Tencent voiceprint cleanup failed for {len(errors)} operation(s)"
            ) from errors[0]


SpeakerEnrollmentRecord = SpeakerEnrollment | TencentSpeakerEnrollment
