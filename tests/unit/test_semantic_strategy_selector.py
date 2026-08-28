from __future__ import annotations

import json

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.intervention import (
    StrategyChoiceGenerator,
    StrategySelector,
    load_strategy_library,
)
from coregulation_poc.intervention.models import StrategySelectionSource
from coregulation_poc.models import ControlObservation, StateAssessment


class FakeProvider:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload
        self.call_count = 0
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> object:
        self.call_count += 1
        self.last_prompt = prompt
        if self.payload is None:
            raise ConnectionError("unavailable")

        class Result:
            text = json.dumps(self.payload, ensure_ascii=False)

        return Result()


def _assessment(*, support_target: str = "both") -> StateAssessment:
    return StateAssessment(
        session_id="semantic-strategy",
        assessed_at_ms=10_000,
        state="dysregulation",
        previous_state="fluctuation",
        trajectory="worsening",
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=["sustained task stall"],
        task_process="sustained_stall",
        support_need="mutual_understanding",
        support_target=support_target,
        interruptibility="natural_pause",
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": "parent",
                        "start_ms": 8_000,
                        "end_ms": 8_800,
                        "code": "parental pressure",
                        "observation": "家长要求重新开始。",
                        "quote": "重新读。",
                    },
                    {
                        "modality": "audio",
                        "actor": "child",
                        "start_ms": 9_000,
                        "end_ms": 9_800,
                        "code": "child refusal",
                        "observation": "儿童拒绝继续。",
                        "quote": "我不想读了。",
                    },
                ],
            },
            "video": {
                "sufficiency": "insufficient",
                "items": [],
                "limitation_reason": "Relevant behavior is outside the frame.",
            },
        },
        reason="持续停滞伴随催促与拒绝，双方需要恢复理解。",
    )


def _decision(assessment: StateAssessment):
    controller = StateTrajectoryController(load_intervention_policy())
    return controller.ingest(
        ControlObservation(
            assessment=assessment,
            natural_turn_boundary=True,
            post_intervention_response_observed=False,
            interaction_history_available=True,
        )
    )


def _choice(strategy_id: str | None, confidence: str = "high") -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "confidence": confidence,
        "reason": "该策略直接帮助双方停止责备并表达具体困难。",
        "matched_dimensions": ["support_need", "support_target"],
        "relaxed_dimensions": ["interaction_performance", "task_process"],
    }


def test_bounded_llm_resolves_soft_field_mismatch() -> None:
    assessment = _assessment(support_target="parent")
    provider = FakeProvider(_choice("PARENT_INTENT_TRANSLATION"))
    selector = StrategySelector(
        load_strategy_library(),
        strategy_choice_generator=StrategyChoiceGenerator(provider),
    )

    result = selector.select(assessment=assessment, decision=_decision(assessment))

    assert result.plan is not None
    assert result.plan.strategy_id == "PARENT_INTENT_TRANSLATION"
    assert result.plan.strategy_selection_source is StrategySelectionSource.BOUNDED_LLM
    assert result.plan.semantic_selection_confidence == "high"
    assert result.plan.semantic_relaxed_dimensions == [
        "interaction_performance",
        "task_process",
    ]
    assert provider.call_count == 1
    assert provider.last_prompt is not None
    assert "PARENT_INTENT_TRANSLATION" in provider.last_prompt


def test_bounded_llm_cannot_select_unapproved_candidate() -> None:
    assessment = _assessment(support_target="parent")
    provider = FakeProvider(_choice("DYAD_RELATIONSHIP_RESET"))
    selector = StrategySelector(
        load_strategy_library(),
        strategy_choice_generator=StrategyChoiceGenerator(provider),
    )

    result = selector.select(assessment=assessment, decision=_decision(assessment))

    assert result.plan is not None
    assert result.plan.strategy_selection_source is StrategySelectionSource.DETERMINISTIC_FALLBACK


def test_low_confidence_semantic_choice_is_held() -> None:
    assessment = _assessment(support_target="parent")
    provider = FakeProvider(_choice("PARENT_INTENT_TRANSLATION", confidence="low"))
    selector = StrategySelector(
        load_strategy_library(),
        strategy_choice_generator=StrategyChoiceGenerator(provider),
    )

    result = selector.select(assessment=assessment, decision=_decision(assessment))

    assert result.plan is not None
    assert result.plan.strategy_selection_source is StrategySelectionSource.DETERMINISTIC_FALLBACK


def test_semantic_selector_failure_holds_output() -> None:
    assessment = _assessment(support_target="parent")
    selector = StrategySelector(
        load_strategy_library(),
        strategy_choice_generator=StrategyChoiceGenerator(FakeProvider()),
    )

    result = selector.select(assessment=assessment, decision=_decision(assessment))

    assert result.plan is not None
    assert result.plan.strategy_selection_source is StrategySelectionSource.DETERMINISTIC_FALLBACK


def test_without_semantic_selector_uses_deterministic_closest_card() -> None:
    assessment = _assessment()
    result = StrategySelector(load_strategy_library()).select(
        assessment=assessment,
        decision=_decision(assessment),
    )

    assert result.plan is not None
    assert result.plan.strategy_selection_source is StrategySelectionSource.EXACT_RULE
