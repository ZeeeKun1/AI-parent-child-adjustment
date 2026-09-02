from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from coregulation_poc.acoustics.speaker_binding import (
    SpeakerBinding,
    SpeakerLabel,
    SpeakerSegment,
)
from coregulation_poc.acoustics.tencent_voiceprint import (
    TencentEnrolledSpeaker,
    TencentSpeakerEnrollment,
    TencentVoiceprintService,
)


def _pcm16(seconds: int = 5) -> bytes:
    timeline = np.arange(seconds * 16_000, dtype=np.float64) / 16_000
    samples = np.rint(np.sin(2 * np.pi * 180 * timeline) * 4_000).astype("<i2")
    return samples.tobytes()


class FakeTencentClient:
    def __init__(self) -> None:
        self.enroll_requests: list[object] = []
        self.update_requests: list[object] = []
        self.verify_requests: list[object] = []
        self.delete_requests: list[object] = []
        self.verify_responses: list[object] = []

    def VoicePrintEnroll(self, request: object) -> object:
        self.enroll_requests.append(request)
        label = request.SpeakerNick
        return SimpleNamespace(
            Data=SimpleNamespace(VoicePrintId=f"voiceprint-{label}")
        )

    def VoicePrintUpdate(self, request: object) -> object:
        self.update_requests.append(request)
        return SimpleNamespace(
            Data=SimpleNamespace(VoicePrintId=request.VoicePrintId)
        )

    def VoicePrintVerify(self, request: object) -> object:
        self.verify_requests.append(request)
        response = self.verify_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def VoicePrintDelete(self, request: object) -> object:
        self.delete_requests.append(request)
        return SimpleNamespace(Data=SimpleNamespace())


def _service(client: FakeTencentClient, *, minimum_score: float = 70) -> TencentVoiceprintService:
    return TencentVoiceprintService(
        secret_id="test-id",
        secret_key="test-key",
        minimum_score=minimum_score,
        client=client,
    )


def _verify_response(score: float, decision: int = 1) -> object:
    return SimpleNamespace(Data=SimpleNamespace(Score=str(score), Decision=decision))


def _complete_enrollment() -> TencentSpeakerEnrollment:
    return TencentSpeakerEnrollment(
        family_id="family",
        group_id="coreg_TestGroup",
        speakers={
            "parent": TencentEnrolledSpeaker(
                label="parent",
                duration_ms=5_000,
                voiceprint_id="vp-parent",
            ),
            "child": TencentEnrolledSpeaker(
                label="child",
                duration_ms=5_000,
                voiceprint_id="vp-child",
            ),
        },
    )


def test_enrollment_registers_both_roles_in_one_random_group() -> None:
    client = FakeTencentClient()
    service = _service(client)

    parent = service.enroll_speaker(_pcm16(), "parent", "family-visible-id")
    complete = service.enroll_speaker(
        _pcm16(),
        "child",
        "family-visible-id",
        parent,
    )

    assert complete.is_complete
    assert complete.group_id == parent.group_id
    assert complete.group_id.startswith("coreg_")
    assert "family-visible-id" not in complete.group_id
    assert [request.SpeakerNick for request in client.enroll_requests] == [
        "parent",
        "child",
    ]
    assert all(request.SampleRate == 16_000 for request in client.enroll_requests)
    assert all(request.VoiceFormat == 0 for request in client.enroll_requests)


def test_re_recording_a_role_updates_instead_of_creating_an_orphan() -> None:
    client = FakeTencentClient()
    service = _service(client)
    enrollment = service.enroll_speaker(_pcm16(), "parent", "family")

    updated = service.enroll_speaker(_pcm16(), "parent", "family", enrollment)

    assert len(client.enroll_requests) == 1
    assert len(client.update_requests) == 1
    assert client.update_requests[0].VoicePrintId == "voiceprint-parent"
    assert updated.speakers["parent"].voiceprint_id == "voiceprint-parent"


@patch("coregulation_poc.acoustics.tencent_voiceprint._extract_segment_f0")
@patch("coregulation_poc.acoustics.tencent_voiceprint._prepare_audio_segments")
def test_pairwise_verification_uses_relative_margin_when_absolute_score_is_low(
    prepare_segments: object,
    extract_f0: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client, minimum_score=70)
    enrollment = _complete_enrollment()
    samples = np.zeros(32_000, dtype=np.float64)
    prepare_segments.return_value = (
        samples,
        16_000,
        [(0, 1_000, 0, 16_000), (1_000, 2_000, 16_000, 32_000)],
    )
    extract_f0.return_value = (np.array([], dtype=np.float64), 0)
    client.verify_responses = [
        _verify_response(91.5),
        _verify_response(45.0, decision=0),
        _verify_response(50.0, decision=0),
        _verify_response(64.0, decision=0),
    ]

    binding = service.identify_speakers((), enrollment)

    assert binding.bound
    assert [segment.speaker.value for segment in binding.segments] == [
        "parent",
        "child",
    ]
    assert binding.segments[0].forced_assignment is False
    assert binding.segments[1].forced_assignment is False
    assert binding.segments[1].confidence == "medium"
    assert binding.low_confidence_segment_count == 0
    assert binding.provider_request_count == 4
    assert binding.provider_failure_count == 0
    assert [request.VoicePrintId for request in client.verify_requests] == [
        "vp-parent",
        "vp-child",
        "vp-parent",
        "vp-child",
    ]
    assert binding.segments[0].parent_provider_score == 91.5
    assert binding.segments[0].child_provider_score == 45.0
    assert binding.segments[1].parent_provider_score == 50.0
    assert binding.segments[1].child_provider_score == 64.0


@patch("coregulation_poc.acoustics.tencent_voiceprint._extract_segment_f0")
@patch("coregulation_poc.acoustics.tencent_voiceprint._prepare_audio_segments")
def test_close_pairwise_scores_are_assigned_but_never_claim_high_confidence(
    prepare_segments: object,
    extract_f0: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client)
    prepare_segments.return_value = (
        np.zeros(16_000, dtype=np.float64),
        16_000,
        [(0, 1_000, 0, 16_000)],
    )
    extract_f0.return_value = (np.array([180.0]), 1)
    client.verify_responses = [_verify_response(84.0), _verify_response(81.0)]

    binding = service.identify_speakers((), _complete_enrollment())

    segment = binding.segments[0]
    assert segment.speaker.value == "parent"
    assert segment.confidence == "low"
    assert segment.forced_assignment is True


def test_cleanup_deletes_voiceprints_before_group_metadata() -> None:
    client = FakeTencentClient()
    service = _service(client)
    enrollment = TencentSpeakerEnrollment(
        family_id="family",
        group_id="coreg_TestGroup",
        speakers={
            "parent": TencentEnrolledSpeaker(
                label="parent",
                duration_ms=5_000,
                voiceprint_id="vp-parent",
            ),
            "child": TencentEnrolledSpeaker(
                label="child",
                duration_ms=5_000,
                voiceprint_id="vp-child",
            ),
        },
    )

    service.delete_enrollment(enrollment)

    assert [request.DelMod for request in client.delete_requests] == [0, 0, 2]
    assert [request.VoicePrintId for request in client.delete_requests[:2]] == [
        "vp-parent",
        "vp-child",
    ]
    assert client.delete_requests[2].GroupId == "coreg_TestGroup"


@patch("coregulation_poc.acoustics.tencent_voiceprint._extract_segment_f0")
@patch("coregulation_poc.acoustics.tencent_voiceprint._prepare_audio_segments")
def test_short_utterance_is_padded_and_forced_to_a_known_role(
    prepare_segments: object,
    extract_f0: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client, minimum_score=70)
    enrollment = _complete_enrollment()
    prepare_segments.return_value = (
        np.zeros(6_400, dtype=np.float64),
        16_000,
        [(0, 400, 0, 6_400)],
    )
    extract_f0.return_value = (np.array([180.0]), 1)
    client.verify_responses = [
        _verify_response(92.0),
        _verify_response(35.0, decision=0),
    ]

    binding = service.identify_speakers((), enrollment)

    assert binding.bound is True
    assert binding.segments[0].speaker.value == "parent"
    assert binding.segments[0].forced_assignment is True
    assert binding.segments[0].confidence == "low"
    assert len(base64.b64decode(client.verify_requests[0].Data)) == 25_600
    assert len(base64.b64decode(client.verify_requests[1].Data)) == 25_600


@patch("coregulation_poc.acoustics.tencent_voiceprint.time.sleep")
@patch("coregulation_poc.acoustics.tencent_voiceprint._extract_segment_f0")
@patch("coregulation_poc.acoustics.tencent_voiceprint._prepare_audio_segments")
def test_pairwise_verification_retries_one_transient_role_failure(
    prepare_segments: object,
    extract_f0: object,
    sleep: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client)
    prepare_segments.return_value = (
        np.zeros(16_000, dtype=np.float64),
        16_000,
        [(0, 1_000, 0, 16_000)],
    )
    extract_f0.return_value = (np.array([180.0]), 1)
    client.verify_responses = [
        TimeoutError("temporary timeout"),
        _verify_response(91.0),
        _verify_response(30.0, decision=0),
    ]

    binding = service.identify_speakers((), _complete_enrollment())

    assert binding.segments[0].speaker.value == "parent"
    assert binding.segments[0].confidence == "high"
    assert binding.provider_request_count == 3
    assert binding.provider_failure_count == 1
    sleep.assert_called_once_with(0.4)


@patch("coregulation_poc.acoustics.tencent_voiceprint.time.sleep")
@patch("coregulation_poc.acoustics.tencent_voiceprint._extract_segment_f0")
@patch("coregulation_poc.acoustics.tencent_voiceprint._prepare_audio_segments")
def test_one_unavailable_role_keeps_a_verified_assignment_but_marks_it_low(
    prepare_segments: object,
    extract_f0: object,
    _sleep: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client)
    prepare_segments.return_value = (
        np.zeros(16_000, dtype=np.float64),
        16_000,
        [(0, 1_000, 0, 16_000)],
    )
    extract_f0.return_value = (np.array([180.0]), 1)
    client.verify_responses = [
        _verify_response(91.0),
        TimeoutError("child timeout"),
        TimeoutError("child timeout"),
    ]

    binding = service.identify_speakers((), _complete_enrollment())

    segment = binding.segments[0]
    assert segment.speaker.value == "parent"
    assert segment.parent_provider_score == 91.0
    assert segment.child_provider_score is None
    assert segment.confidence == "low"
    assert segment.forced_assignment is True
    assert binding.provider_request_count == 3
    assert binding.provider_failure_count == 2


@patch("coregulation_poc.acoustics.tencent_voiceprint.time.sleep")
@patch("coregulation_poc.acoustics.tencent_voiceprint._extract_segment_f0")
@patch("coregulation_poc.acoustics.tencent_voiceprint._prepare_audio_segments")
def test_both_roles_unavailable_fall_back_to_f0_without_losing_the_window(
    prepare_segments: object,
    extract_f0: object,
    sleep: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client)
    prepare_segments.return_value = (
        np.zeros(16_000, dtype=np.float64),
        16_000,
        [(0, 1_000, 0, 16_000)],
    )
    extract_f0.return_value = (np.array([310.0, 315.0]), 2)
    client.verify_responses = [
        TimeoutError("parent timeout"),
        TimeoutError("parent timeout"),
        TimeoutError("child timeout"),
        TimeoutError("child timeout"),
    ]

    binding = service.identify_speakers((), _complete_enrollment())

    segment = binding.segments[0]
    assert segment.speaker.value == "child"
    assert segment.provider_score is None
    assert segment.confidence == "low"
    assert segment.forced_assignment is True
    assert binding.provider_request_count == 4
    assert binding.provider_failure_count == 4
    assert sleep.call_count == 2


@patch("coregulation_poc.acoustics.tencent_voiceprint._extract_segment_f0")
@patch("coregulation_poc.acoustics.tencent_voiceprint._prepare_audio_segments")
def test_ambiguous_overlap_uses_stable_role_from_previous_window(
    prepare_segments: object,
    extract_f0: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client)
    prepare_segments.return_value = (
        np.zeros(16_000, dtype=np.float64),
        16_000,
        [(1_000, 2_000, 0, 16_000)],
    )
    extract_f0.return_value = (np.array([190.0]), 1)
    client.verify_responses = [_verify_response(60.0), _verify_response(58.0)]
    previous = SpeakerBinding(
        bound=True,
        segments=[
            SpeakerSegment(
                start_ms=900,
                end_ms=2_100,
                mean_f0_hz=195.0,
                median_f0_hz=195.0,
                voiced_frame_count=10,
                speaker=SpeakerLabel.CHILD,
                confidence="medium",
            )
        ],
        child_segment_count=1,
    )

    binding = service.identify_speakers(
        (),
        _complete_enrollment(),
        previous_binding=previous,
    )

    segment = binding.segments[0]
    assert segment.speaker is SpeakerLabel.CHILD
    assert segment.confidence == "medium"
    assert segment.forced_assignment is True
    assert binding.continuity_assisted_segment_count == 1
    assert binding.low_confidence_segment_count == 0
