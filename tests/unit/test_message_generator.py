from __future__ import annotations

import json

import pytest

from coregulation_poc.intervention import load_strategy_library
from coregulation_poc.intervention.message_prompt import build_message_prompt
from coregulation_poc.intervention.models import MessageSource
from coregulation_poc.intervention.selector import MessageGenerator, StrategySelector
from coregulation_poc.models import (
    Actor,
    ControlObservation,
    CoregulationState,
    EvidenceModality,
    EvidenceReference,
    EvidenceSufficiency,
    InterventionDecision,
    InterventionAction,
    InterventionDecisionReason,
    RecoveryStatus,
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
    state: str = "dysregulation",
    performance: str = "pace conflict",
    actor: str = "parent",
    assessed_at_ms: int = 1000,
    support_need: str | None = "emotional_support",
    task_process: str | None = "pace_mismatch",
) -> StateAssessment:
    return StateAssessment(
        session_id="msg-gen-demo",
        assessed_at_ms=assessed_at_ms,
        state=state,
        previous_state=None,
        trajectory="stable",
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=[performance],
        task_process=task_process,
        support_need=support_need,
        support_target=actor,
        interruptibility="natural_pause",
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": actor,
                        "start_ms": max(0, assessed_at_ms - 500),
                        "end_ms": assessed_at_ms,
                        "code": performance,
                        "observation": "A directly observed interaction pattern.",
                        "quote": "快点写",
                    }
                ],
            },
            "video": _insufficient_video(),
        },
        reason="The observed sequence supports this state and performance.",
    )


def _observation(
    *,
    state: str = "dysregulation",
    performance: str = "pace conflict",
    actor: str = "parent",
    assessed_at_ms: int = 1000,
    history_available: bool = False,
) -> ControlObservation:
    return ControlObservation(
        assessment=_assessment(
            state=state, performance=performance, actor=actor, assessed_at_ms=assessed_at_ms
        ),
        natural_turn_boundary=True,
        post_intervention_response_observed=False,
        interaction_history_available=history_available,
    )


def _decision(
    *,
    action: InterventionAction = InterventionAction.INTERVENE,
    state: CoregulationState = CoregulationState.DYSREGULATION,
    assessed_at_ms: int = 1000,
    evidence_actors: list[Actor] | None = None,
) -> InterventionDecision:
    return InterventionDecision(
        session_id="msg-gen-demo",
        sequence=1,
        decided_at_ms=assessed_at_ms,
        previous_state=None,
        current_state=state,
        action=action,
        reason_code=InterventionDecisionReason.DYAD_CANNOT_SELF_RECOVER,
        reason="The dyad cannot self-recover.",
        natural_turn_boundary=True,
        intervention_permitted=True,
        strategy_selection_required=True,
        recovery_status=RecoveryStatus.NOT_APPLICABLE,
        evidence_actors=evidence_actors or [Actor.PARENT],
        interaction_performance=["pace conflict"],
        research_basis=["timing_by_state"],
    )


class FakeTextChatProvider:
    """Fake provider for testing without real API calls."""

    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> object:
        self.call_count += 1
        self.last_prompt = prompt

        class _Result:
            text = self.response_text

        return _Result


def _llm_json(
    *,
    strategy_id: str = "PARENT_TONE_AND_PACE",
    target_actor: str = "parent",
    evidence_ids: list[str] | None = None,
    observation_clause: str = "刚才语速比较快",
) -> str:
    """Build a valid JSON LLM response for tests."""
    return json.dumps({
        "strategy_id": strategy_id,
        "target_actor": target_actor,
        "evidence_ids": evidence_ids or ["evidence_0"],
        "observation_clause": observation_clause,
    })


class FailingTextChatProvider:
    """Provider that always raises an exception."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or ConnectionError("API unavailable")

    def generate(self, prompt: str) -> object:
        raise self.exc


def _selector(generator: MessageGenerator | None = None) -> StrategySelector:
    return StrategySelector(load_strategy_library(), message_generator=generator)


def _generator(provider: object) -> MessageGenerator:
    return MessageGenerator(
        provider=provider,
        max_characters=90,
        max_sentences=2,
        banned_phrases=["孩子就是", "答案是", "你错了"],
    )


# ---------------------------------------------------------------------------
# MessageGenerator + prompt tests
# ---------------------------------------------------------------------------


class TestMessagePrompt:
    def test_prompt_contains_strategy_intent(self) -> None:
        library = load_strategy_library()
        card = library.cards[2]  # PARENT_TONE_AND_PACE
        prompt = build_message_prompt(
            card=card,
            evidence=[],
            max_characters=90,
            max_sentences=2,
            banned_phrases=["答案是"],
        )
        assert "语气" in prompt or "语速" in prompt
        assert "90" in prompt
        assert "2" in prompt
        assert "答案是" in prompt

    def test_prompt_contains_evidence_quote(self) -> None:
        library = load_strategy_library()
        card = library.cards[2]
        evidence = [
            EvidenceReference(
                modality=EvidenceModality.AUDIO,
                actor=Actor.PARENT,
                start_ms=500,
                end_ms=1000,
                code="pace conflict",
                observation="Parent is rushing.",
                quote="快点写",
            )
        ]
        prompt = build_message_prompt(
            card=card,
            evidence=evidence,
            max_characters=90,
            max_sentences=2,
            banned_phrases=[],
        )
        assert "快点写" in prompt


# ---------------------------------------------------------------------------
# Selector + LLM tests
# ---------------------------------------------------------------------------


class TestSelectorWithoutGenerator:
    def test_no_generator_uses_approved_template(self) -> None:
        selector = _selector()
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.APPROVED_TEMPLATE
        assert all(result.plan.validation_checks.values())

    def test_no_generator_does_not_use_llm_source(self) -> None:
        selector = _selector()
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is not MessageSource.CONSTRAINED_LLM


class TestSelectorWithGenerator:
    def test_valid_llm_response_uses_constrained_llm(self) -> None:
        provider = FakeTextChatProvider(_llm_json())
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.CONSTRAINED_LLM
        assert "刚才语速比较快" in result.plan.message
        assert "可以先放慢语速" in result.plan.message
        assert all(result.plan.validation_checks.values())
        assert provider.call_count == 1

    def test_llm_too_long_falls_back(self) -> None:
        long_clause = "这是一个非常非常非常非常非常非常非常非常非常长的观察描述超过三十个字"
        provider = FakeTextChatProvider(_llm_json(observation_clause=long_clause))
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.APPROVED_TEMPLATE_FALLBACK
        assert all(result.plan.validation_checks.values())

    def test_llm_with_banned_phrase_falls_back(self) -> None:
        provider = FakeTextChatProvider(_llm_json(observation_clause="答案是三"))
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.APPROVED_TEMPLATE_FALLBACK

    def test_llm_too_many_sentences_falls_back(self) -> None:
        clause = "刚才语速很快。孩子跟不上了。需要调整。"
        provider = FakeTextChatProvider(_llm_json(observation_clause=clause))
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.APPROVED_TEMPLATE_FALLBACK

    def test_llm_empty_response_falls_back(self) -> None:
        provider = FakeTextChatProvider("")
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.APPROVED_TEMPLATE_FALLBACK

    def test_provider_connection_error_falls_back(self) -> None:
        provider = FailingTextChatProvider()
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.APPROVED_TEMPLATE_FALLBACK
        assert all(result.plan.validation_checks.values())

    def test_provider_value_error_falls_back(self) -> None:
        provider = FailingTextChatProvider(ValueError("malformed response"))
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        assert result.plan.message_source is MessageSource.APPROVED_TEMPLATE_FALLBACK

    def test_llm_receives_prompt_with_card_and_evidence(self) -> None:
        provider = FakeTextChatProvider(_llm_json())
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert provider.last_prompt is not None
        assert "语速" in provider.last_prompt or "节奏" in provider.last_prompt
        assert "快点写" in provider.last_prompt

    def test_validation_checks_include_all_fields(self) -> None:
        provider = FakeTextChatProvider(_llm_json())
        selector = _selector(_generator(provider))
        observation = _observation()
        decision = _decision()

        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
        )

        assert result.plan is not None
        checks = result.plan.validation_checks
        assert "strategy_id_matches" in checks
        assert "target_actor_matches" in checks
        assert "evidence_ids_valid" in checks
        assert "observation_within_limit" in checks
        assert "within_character_limit" in checks
        assert "within_sentence_limit" in checks
        assert "contains_no_banned_phrase" in checks
        assert "contains_no_answer" in checks
        assert "contains_no_blame_command" in checks
        assert "action_clause_unchanged" in checks
        assert "target_actor_explicit" in checks
