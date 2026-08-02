import pytest
from pydantic import ValidationError

from coregulation_poc.models import EvidenceSufficiency, StateAssessment


def _insufficient_modality(reason: str) -> dict[str, object]:
    return {"sufficiency": "insufficient", "items": [], "limitation_reason": reason}


def test_insufficient_assessment_has_no_forced_state() -> None:
    assessment = StateAssessment(
        session_id="demo",
        assessed_at_ms=1000,
        state=None,
        evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
        confidence="low",
        ambiguity_reason="Neither modality supports a state assessment.",
        interaction_performance=[],
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
            evidence_sufficiency="insufficient",
            confidence="low",
            ambiguity_reason="Evidence is insufficient.",
            interaction_performance=[],
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
            evidence_sufficiency="sufficient",
            confidence="high",
            interaction_performance=["normal task progression"],
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
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=["normal task progression"],
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
