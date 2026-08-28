"""Tests 22-27: Constrained contextual message generation.

Verifies that the LLM-generated observation clause is validated against
strategy-card constraints. Invalid drafts are never delivered and repeated
generation failure holds the intervention instead of silently substituting a
generic template.
"""

from __future__ import annotations

import json
from typing import Any

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.intervention import StrategySelector, load_strategy_library
from coregulation_poc.intervention.models import StrategyHoldReason
from coregulation_poc.models import (
    ControlObservation,
    StateAssessment,
)


def _insufficient_video() -> dict[str, object]:
    return {
        "sufficiency": "insufficient",
        "items": [],
        "limitation_reason": "Relevant behavior is outside the frame.",
    }


def _assessment(
    *,
    assessed_at_ms: int = 1000,
    session_id: str = "spec-message",
) -> StateAssessment:
    return StateAssessment(
        session_id=session_id,
        assessed_at_ms=assessed_at_ms,
        state="dysregulation",
        previous_state=None,
        trajectory="stable",
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=["pace conflict"],
        task_process="pace_mismatch",
        support_need="task_pacing",
        support_target="parent",
        interruptibility="natural_pause",
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": "parent",
                        "start_ms": max(0, assessed_at_ms - 500),
                        "end_ms": assessed_at_ms,
                        "code": "pace conflict",
                        "observation": "The parent is rushing through the steps.",
                        "quote": "快点写完这一步",
                    }
                ],
            },
            "video": _insufficient_video(),
        },
        reason="The parent is pressing the child to move faster.",
    )


def _observation() -> ControlObservation:
    return ControlObservation(
        assessment=_assessment(),
        natural_turn_boundary=True,
        post_intervention_response_observed=False,
        interaction_history_available=False,
    )


class _MockLLMProvider:
    """Mock provider that returns a pre-configured JSON response."""

    def __init__(self, response_json: str | None = None, raise_error: bool = False) -> None:
        self._json = response_json
        self._raise = raise_error

    def generate(self, prompt: str) -> Any:
        if self._raise:
            raise ConnectionError("LLM service unavailable")
        result = type("_Result", (), {"text": self._json})()
        return result


def _make_llm_json(
    *,
    strategy_id: str = "PARENT_PACE_RESET",
    target_actor: str = "parent",
    evidence_ids: list[str] | None = None,
    message: str = "家长，孩子刚才还在思考，可以先放慢一点，只问一个问题。",
) -> str:
    return json.dumps(
        {
            "strategy_id": strategy_id,
            "target_actor": target_actor,
            "evidence_ids": evidence_ids if evidence_ids is not None else ["evidence_0"],
            "message": message,
        },
        ensure_ascii=False,
    )


def _selector_with_mock(mock_provider: _MockLLMProvider) -> StrategySelector:
    library = load_strategy_library()
    from coregulation_poc.intervention.selector import MessageGenerator

    generator = MessageGenerator(
        provider=mock_provider,
        max_characters=library.principles.maximum_message_characters,
        max_sentences=library.principles.maximum_message_sentences,
        banned_phrases=list(library.banned_phrases),
    )
    return StrategySelector(library, message_generator=generator)


def _run_selection(selector: StrategySelector):
    observation = _observation()
    decision = StateTrajectoryController(load_intervention_policy()).ingest(observation)
    return selector.select(
        assessment=observation.assessment,
        decision=decision,
    )


# Test 22: LLM output contains an answer indicator -> hold after rewrite failure
def test_llm_output_with_answer_is_not_delivered() -> None:
    mock = _MockLLMProvider(_make_llm_json(message="答案是三所以不用再想了"))
    result = _run_selection(_selector_with_mock(mock))

    assert result.plan is None
    assert result.hold_reason is StrategyHoldReason.MESSAGE_GENERATION_FAILED


# Test 23: LLM output contains blame/command indicator -> do not deliver
def test_llm_output_with_blame_is_not_delivered() -> None:
    mock = _MockLLMProvider(_make_llm_json(message="你必须停下来听孩子说"))
    result = _run_selection(_selector_with_mock(mock))

    assert result.plan is None
    assert result.hold_reason is StrategyHoldReason.MESSAGE_GENERATION_FAILED


# Test 24: LLM returns target_actor that does not match the card -> do not deliver
def test_llm_actor_mismatch_is_not_delivered() -> None:
    mock = _MockLLMProvider(_make_llm_json(target_actor="child"))
    result = _run_selection(_selector_with_mock(mock))

    assert result.plan is None
    assert result.hold_reason is StrategyHoldReason.MESSAGE_GENERATION_FAILED


# Test 25: LLM returns strategy_id that does not match the card -> do not deliver
def test_llm_strategy_id_mismatch_is_not_delivered() -> None:
    mock = _MockLLMProvider(_make_llm_json(strategy_id="CHILD_PACE_RESET"))
    result = _run_selection(_selector_with_mock(mock))

    assert result.plan is None
    assert result.hold_reason is StrategyHoldReason.MESSAGE_GENERATION_FAILED


# Test 26: LLM references non-existent evidence -> do not deliver
def test_llm_invalid_evidence_id_is_not_delivered() -> None:
    mock = _MockLLMProvider(_make_llm_json(evidence_ids=["evidence_99"]))
    result = _run_selection(_selector_with_mock(mock))

    assert result.plan is None
    assert result.hold_reason is StrategyHoldReason.MESSAGE_GENERATION_FAILED


# Test 27: LLM call fails entirely -> hold instead of generic template
def test_llm_failure_holds_intervention() -> None:
    mock = _MockLLMProvider(raise_error=True)
    result = _run_selection(_selector_with_mock(mock))

    assert result.plan is None
    assert result.hold_reason is StrategyHoldReason.MESSAGE_GENERATION_FAILED
