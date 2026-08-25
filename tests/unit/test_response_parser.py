import pytest

from coregulation_poc.codebook import load_state_codebook
from coregulation_poc.fusion.response_parser import (
    RealtimeResponseAccumulator,
    validate_assessment_context,
)
from coregulation_poc.models import CoregulationState


def test_accumulator_parses_valid_assessment() -> None:
    accumulator = RealtimeResponseAccumulator()
    accumulator.add(
        {
            "type": "response.text.delta",
            "delta": (
                '{"session_id":"S01","assessed_at_ms":12000,'
                '"state":"fluctuation","previous_state":null,'
                '"trajectory":"stable",'
                '"evidence_sufficiency":"sufficient",'
                '"confidence":"medium","alternative_state":"normal",'
                '"ambiguity_reason":"The boundary with normal support is uncertain",'
                '"interaction_performance":["brief task stall"],'
                '"task_process":"brief_stall","support_need":"none",'
                '"support_target":"unknown","interruptibility":"natural_pause",'
                '"modality_evidence":{"audio":{"sufficiency":"sufficient","items":'
                '[{"modality":"audio","actor":"unknown","start_ms":5000,'
                '"end_ms":7000,"code":"brief task stall","observation":'
                '"A wrong answer is followed by correction","quote":"three"}]},'
                '"video":{"sufficiency":"insufficient","items":[],"limitation_reason":'
                '"Relevant behavior is not visible"}},'
                '"reason":"短暂波动，但仍可恢复"}'
            ),
        }
    )

    assessment = accumulator.parse_assessment()

    assert assessment.state is CoregulationState.FLUCTUATION
    assert assessment.modality_evidence.audio.items[0].start_ms == 5000

    validate_assessment_context(
        assessment,
        expected_session_id="S01",
        duration_ms=12_000,
        codebook=load_state_codebook(),
    )


def test_accumulator_rejects_unstructured_response() -> None:
    accumulator = RealtimeResponseAccumulator(text_deltas=["not json"])

    with pytest.raises(ValueError, match="did not contain"):
        accumulator.parse_assessment()


def test_accumulator_surfaces_transcription_failure_for_audit() -> None:
    accumulator = RealtimeResponseAccumulator()
    accumulator.add(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"code": "UNEXPECTED_ASR_ERROR"},
        }
    )

    assert accumulator.transcription_status == "failed"
    assert accumulator.audit_warnings == [
        "input_audio_transcription_failed:UNEXPECTED_ASR_ERROR"
    ]


def test_accumulator_preserves_qwen_emotion_as_supporting_observation() -> None:
    accumulator = RealtimeResponseAccumulator()
    accumulator.add(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-1",
            "text": "再看一下",
            "stash": "这道题",
            "emotion": "angry",
        }
    )

    assert accumulator.input_emotion_observations == [
        {
            "item_id": "item-1",
            "emotion": "angry",
            "text": "再看一下这道题",
            "source_event_type": "conversation.item.input_audio_transcription.delta",
            "interpretation_role": "supporting_observation_only",
        }
    ]


def test_context_validation_rejects_evidence_outside_clip() -> None:
    accumulator = RealtimeResponseAccumulator()
    accumulator.add(
        {
            "type": "response.text.delta",
            "delta": (
                '{"session_id":"S01","assessed_at_ms":12000,"state":"normal",'
                '"previous_state":null,"trajectory":"stable",'
                '"evidence_sufficiency":"sufficient","confidence":"high",'
                '"interaction_performance":["normal task progression"],'
                '"task_process":"smooth_progress","support_need":"none",'
                '"support_target":"unknown","interruptibility":"natural_pause",'
                '"modality_evidence":{"audio":{"sufficiency":"sufficient","items":'
                '[{"modality":"audio","actor":"unknown","start_ms":11000,'
                '"end_ms":13000,"code":"task_talk","observation":"Task talk",'
                '"quote":"four"}]},"video":{"sufficiency":"insufficient","items":[],'
                '"limitation_reason":"Not visible"}},"reason":"Normal progression"}'
            ),
        }
    )
    assessment = accumulator.parse_assessment()

    with pytest.raises(ValueError, match="outside the source clip"):
        validate_assessment_context(
            assessment,
            expected_session_id="S01",
            duration_ms=12_000,
            codebook=load_state_codebook(),
        )
