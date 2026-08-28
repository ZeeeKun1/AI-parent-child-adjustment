from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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

    def VoicePrintGroupVerify(self, request: object) -> object:
        self.verify_requests.append(request)
        return self.verify_responses.pop(0)

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
def test_group_verification_always_assigns_parent_or_child_and_marks_low_score(
    prepare_segments: object,
    extract_f0: object,
) -> None:
    client = FakeTencentClient()
    service = _service(client, minimum_score=70)
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
    samples = np.zeros(32_000, dtype=np.float64)
    prepare_segments.return_value = (
        samples,
        16_000,
        [(0, 1_000, 0, 16_000), (1_000, 2_000, 16_000, 32_000)],
    )
    extract_f0.return_value = (np.array([], dtype=np.float64), 0)
    client.verify_responses = [
        SimpleNamespace(
            Data=SimpleNamespace(
                VerifyTops=[
                    SimpleNamespace(VoicePrintId="vp-parent", Score="91.5"),
                    SimpleNamespace(VoicePrintId="vp-child", Score="45.0"),
                ]
            )
        ),
        SimpleNamespace(
            Data=SimpleNamespace(
                VerifyTops=[
                    SimpleNamespace(VoicePrintId="vp-child", Score="64.0"),
                    SimpleNamespace(VoicePrintId="vp-parent", Score="50.0"),
                ]
            )
        ),
    ]

    binding = service.identify_speakers((), enrollment)

    assert binding.bound
    assert [segment.speaker.value for segment in binding.segments] == [
        "parent",
        "child",
    ]
    assert binding.segments[0].forced_assignment is False
    assert binding.segments[1].forced_assignment is True
    assert binding.low_confidence_segment_count == 1
    assert binding.provider_request_count == 2
    assert all(request.TopN == 2 for request in client.verify_requests)


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
    enrollment = TencentSpeakerEnrollment(
        family_id="family",
        group_id="coreg_TestGroup",
        speakers={
            "parent": TencentEnrolledSpeaker(
                label="parent", duration_ms=5_000, voiceprint_id="vp-parent"
            ),
            "child": TencentEnrolledSpeaker(
                label="child", duration_ms=5_000, voiceprint_id="vp-child"
            ),
        },
    )
    prepare_segments.return_value = (
        np.zeros(6_400, dtype=np.float64),
        16_000,
        [(0, 400, 0, 6_400)],
    )
    extract_f0.return_value = (np.array([180.0]), 1)
    client.verify_responses = [
        SimpleNamespace(
            Data=SimpleNamespace(
                VerifyTops=[SimpleNamespace(VoicePrintId="vp-parent", Score="92.0")]
            )
        )
    ]

    binding = service.identify_speakers((), enrollment)

    assert binding.bound is True
    assert binding.segments[0].speaker.value == "parent"
    assert binding.segments[0].forced_assignment is True
    assert binding.segments[0].confidence == "low"
    assert len(base64.b64decode(client.verify_requests[0].Data)) == 25_600
