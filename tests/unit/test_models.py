import pytest
from pydantic import ValidationError

from coregulation_poc.models import EvidenceSufficiency, StateAssessment, TaskContext


def _insufficient_modality(reason: str) -> dict[str, object]:
    return {"sufficiency": "insufficient", "items": [], "limitation_reason": reason}


def test_insufficient_assessment_has_no_forced_state() -> None:
    assessment = StateAssessment(
        session_id="demo",
        assessed_at_ms=1000,
        state=None,
        previous_state=None,
        trajectory="unclear",
        evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
        confidence="low",
        ambiguity_reason="Neither modality supports a state assessment.",
        interaction_performance=[],
        task_process=None,
        support_need=None,
        support_target="unknown",
        interruptibility="unclear",
        modality_evidence={
            "audio": _insufficient_modality("Audio is not intelligible."),
            "video": _insufficient_modality("Relevant behavior is not visible."),
        },
        reason="Not enough observable evidence.",
    )
    assert assessment.state is None


def test_insufficient_assessment_rejects_forced_state() -> None:
    with pytest.raises(ValidationError):
        StateAssessment(
            session_id="demo",
            assessed_at_ms=1000,
            state="normal",
            previous_state=None,
            trajectory="unclear",
            evidence_sufficiency="insufficient",
            confidence="low",
            ambiguity_reason="Evidence is insufficient.",
            interaction_performance=[],
            task_process=None,
            support_need=None,
            support_target="unknown",
            interruptibility="unclear",
            modality_evidence={
                "audio": _insufficient_modality("Audio is not intelligible."),
                "video": _insufficient_modality("Relevant behavior is not visible."),
            },
            reason="Invalid forced state.",
        )


def test_audio_evidence_requires_verbatim_quote() -> None:
    with pytest.raises(ValidationError, match="verbatim quote"):
        StateAssessment(
            session_id="demo",
            assessed_at_ms=1000,
            state="normal",
            previous_state=None,
            trajectory="stable",
            evidence_sufficiency="sufficient",
            confidence="high",
            interaction_performance=["normal task progression"],
            task_process="smooth_progress",
            support_need="none",
            support_target="unknown",
            interruptibility="natural_pause",
            modality_evidence={
                "audio": {
                    "sufficiency": "sufficient",
                    "items": [
                        {
                            "modality": "audio",
                            "actor": "unknown",
                            "start_ms": 100,
                            "end_ms": 300,
                            "code": "task_talk",
                            "observation": "A participant answers the task question.",
                        }
                    ],
                },
                "video": _insufficient_modality("No diagnostic visual behavior."),
            },
            reason="The task proceeds normally.",
        )


def test_one_sufficient_modality_can_support_assessment() -> None:
    assessment = StateAssessment(
        session_id="demo",
        assessed_at_ms=1000,
        state="normal",
        previous_state=None,
        trajectory="stable",
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=["normal task progression"],
        task_process="smooth_progress",
        support_need="none",
        support_target="unknown",
        interruptibility="natural_pause",
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": "unknown",
                        "start_ms": 100,
                        "end_ms": 300,
                        "code": "task_talk",
                        "observation": "A participant answers the task question.",
                        "quote": "four",
                    }
                ],
            },
            "video": _insufficient_modality("Relevant behavior is outside the frame."),
        },
        reason="The task proceeds normally.",
    )

    assert assessment.modality_evidence.video.items == []


def _sufficient_audio_evidence(actor: str = "unknown") -> dict[str, object]:
    return {
        "sufficiency": "sufficient",
        "items": [
            {
                "modality": "audio",
                "actor": actor,
                "start_ms": 100,
                "end_ms": 300,
                "code": "task_talk",
                "observation": "A participant answers the task question.",
                "quote": "four",
            }
        ],
    }


def test_invalid_task_process_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StateAssessment(
            session_id="demo",
            assessed_at_ms=1000,
            state="normal",
            previous_state=None,
            trajectory="stable",
            evidence_sufficiency="sufficient",
            confidence="high",
            interaction_performance=["normal task progression"],
            task_process="not_a_valid_process",
            support_need="none",
            support_target="unknown",
            interruptibility="natural_pause",
            modality_evidence={
                "audio": _sufficient_audio_evidence(),
                "video": _insufficient_modality("No diagnostic visual behavior."),
            },
            reason="The task proceeds normally.",
        )


def test_invalid_support_need_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StateAssessment(
            session_id="demo",
            assessed_at_ms=1000,
            state="normal",
            previous_state=None,
            trajectory="stable",
            evidence_sufficiency="sufficient",
            confidence="high",
            interaction_performance=["normal task progression"],
            task_process="smooth_progress",
            support_need="not_a_valid_need",
            support_target="unknown",
            interruptibility="natural_pause",
            modality_evidence={
                "audio": _sufficient_audio_evidence(),
                "video": _insufficient_modality("No diagnostic visual behavior."),
            },
            reason="The task proceeds normally.",
        )


def test_invalid_interruptibility_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StateAssessment(
            session_id="demo",
            assessed_at_ms=1000,
            state="normal",
            previous_state=None,
            trajectory="stable",
            evidence_sufficiency="sufficient",
            confidence="high",
            interaction_performance=["normal task progression"],
            task_process="smooth_progress",
            support_need="none",
            support_target="unknown",
            interruptibility="not_a_valid_interruptibility",
            modality_evidence={
                "audio": _sufficient_audio_evidence(),
                "video": _insufficient_modality("No diagnostic visual behavior."),
            },
            reason="The task proceeds normally.",
        )


def test_task_context_validates_raw_dict() -> None:
    with pytest.raises(ValidationError):
        TaskContext(
            task_name="Test Task",
            task_type="not_a_valid_type",
            task_difficulty="easy",
            child_grade="3",
        )


def test_insufficient_assessment_rejects_known_support_target() -> None:
    with pytest.raises(ValidationError, match="support_target must be unknown"):
        StateAssessment(
            session_id="demo",
            assessed_at_ms=1000,
            state=None,
            previous_state=None,
            trajectory="unclear",
            evidence_sufficiency="insufficient",
            confidence="low",
            ambiguity_reason="Evidence is insufficient.",
            interaction_performance=[],
            task_process=None,
            support_need=None,
            support_target="parent",
            interruptibility="unclear",
            modality_evidence={
                "audio": _insufficient_modality("Audio is not intelligible."),
                "video": _insufficient_modality("Relevant behavior is not visible."),
            },
            reason="Not enough observable evidence.",
        )
