from __future__ import annotations

import logging
import re

from coregulation_poc.intervention.message_prompt import build_message_prompt
from coregulation_poc.intervention.models import (
    InterventionPlan,
    MessageSource,
    StrategyCard,
    StrategyHoldReason,
    StrategyLibraryConfig,
    StrategySelectionResult,
    StrategySelectionStatus,
)
from coregulation_poc.intervention.strategy_library import cards_by_id
from coregulation_poc.models import (
    Actor,
    CoregulationState,
    EvidenceReference,
    InterventionAction,
    InterventionDecision,
    StateAssessment,
)

logger = logging.getLogger(__name__)


class StrategySelector:
    """Select one research card after module two has authorized intervention.

    When an optional :class:`MessageGenerator` is supplied, the selector
    asks the LLM to rephrase the approved template into a context-adaptive
    variant.  If the LLM output fails validation or the generator is
    unavailable, the selector falls back to the approved template.
    """

    def __init__(
        self,
        library: StrategyLibraryConfig,
        message_generator: MessageGenerator | None = None,
    ) -> None:
        self.library = library
        self.cards = cards_by_id(library)
        self.message_generator = message_generator

    def select(
        self,
        *,
        assessment: StateAssessment,
        decision: InterventionDecision,
        previous_plan: InterventionPlan | None = None,
    ) -> StrategySelectionResult:
        mismatch = self._context_mismatch(assessment, decision)
        if mismatch:
            return self._held(StrategyHoldReason.ASSESSMENT_DECISION_MISMATCH)
        if not decision.strategy_selection_required or not decision.intervention_permitted:
            return self._held(StrategyHoldReason.MODULE_TWO_DID_NOT_AUTHORIZE)
        supported_state_action = {
            CoregulationState.NORMAL: InterventionAction.REINFORCE,
            CoregulationState.DYSREGULATION: InterventionAction.INTERVENE,
            CoregulationState.HIGH_RISK: InterventionAction.PROGRESSIVE_SUPPORT,
        }
        if supported_state_action.get(assessment.state) is not decision.action:
            return self._held(StrategyHoldReason.STATE_NOT_SUPPORTED)

        routed = self._route(assessment, decision)
        if routed is None:
            return self._held(StrategyHoldReason.NO_ROUTED_STRATEGY)
        performance, candidate_ids, selection_reason = routed
        card = self._first_supported_card(
            candidate_ids,
            state=assessment.state,
            evidence_actors=set(decision.evidence_actors),
            previous_plan=previous_plan,
        )
        if card is None:
            return self._held(StrategyHoldReason.TARGET_ACTOR_EVIDENCE_INSUFFICIENT)

        evidence = assessment.modality_evidence.all_items
        message, message_source, checks = self._resolve_message(card, evidence)
        plan = InterventionPlan(
            session_id=assessment.session_id,
            sequence=decision.sequence,
            planned_at_ms=decision.decided_at_ms,
            state=assessment.state,
            decision_action=decision.action,
            strategy_library_version=self.library.version,
            strategy_id=card.strategy_id,
            strategy_name=card.name,
            target_actor=card.target_actor,
            repair_target=card.repair_target,
            message=message,
            message_source=message_source,
            selection_reason=selection_reason,
            selected_from_interaction_performance=performance,
            evidence_references=evidence,
            expected_recovery=card.expected_recovery,
            outcome_interpretation=self.library.outcome_mapping,
            validation_checks=checks,
            previous_strategy_id=(
                None if previous_plan is None else previous_plan.strategy_id
            ),
            progressive_support=decision.action is InterventionAction.PROGRESSIVE_SUPPORT,
            research_codes=card.research_codes,
            research_sources=self.library.source,
        )
        return StrategySelectionResult(status=StrategySelectionStatus.READY, plan=plan)

    @staticmethod
    def _context_mismatch(
        assessment: StateAssessment,
        decision: InterventionDecision,
    ) -> bool:
        return any(
            (
                assessment.session_id != decision.session_id,
                assessment.assessed_at_ms != decision.decided_at_ms,
                assessment.state is not decision.current_state,
                assessment.interaction_performance != decision.interaction_performance,
            )
        )

    def _route(
        self,
        assessment: StateAssessment,
        decision: InterventionDecision,
    ) -> tuple[str, list[str], str] | None:
        observed = set(assessment.interaction_performance)
        for rule in self.library.routing_rules:
            if rule.state is assessment.state and rule.interaction_performance in observed:
                return (
                    rule.interaction_performance,
                    list(rule.strategy_ids),
                    "依据版本化规则，将模块一观察到的互动表现映射到主题分析中的修复策略。",
                )
        return None

    def _first_supported_card(
        self,
        candidate_ids: list[str],
        *,
        state: CoregulationState,
        evidence_actors: set[Actor],
        previous_plan: InterventionPlan | None,
    ) -> StrategyCard | None:
        for strategy_id in dict.fromkeys(candidate_ids):
            if previous_plan is not None and strategy_id == previous_plan.strategy_id:
                continue
            card = self.cards.get(strategy_id)
            if card is None or state not in card.states:
                continue
            if self._actor_supported(card.target_actor, evidence_actors):
                return card
        return None

    @staticmethod
    def _actor_supported(target: Actor, observed: set[Actor]) -> bool:
        if target is Actor.BOTH:
            return Actor.BOTH in observed or {
                Actor.PARENT,
                Actor.CHILD,
            }.issubset(observed)
        if Actor.BOTH in observed:
            return False
        return target in observed

    def _resolve_message(
        self,
        card: StrategyCard,
        evidence: list[EvidenceReference],
    ) -> tuple[str, MessageSource, dict[str, bool]]:
        """Resolve the intervention message, preferring LLM rephrasing.

        When no generator is configured, returns the approved template
        with source ``APPROVED_TEMPLATE``.  When the LLM succeeds and
        passes all validation checks, returns the LLM message with source
        ``CONSTRAINED_LLM``.  When the LLM fails or validation fails,
        falls back to the approved template with source
        ``APPROVED_TEMPLATE_FALLBACK``.
        """
        if self.message_generator is None:
            message, checks = self._validated_template(card)
            return message, MessageSource.APPROVED_TEMPLATE, checks

        try:
            llm_text = self.message_generator.generate(card, evidence)
            checks = self._validate_message(llm_text, card)
            if all(checks.values()):
                return llm_text, MessageSource.CONSTRAINED_LLM, checks
            logger.warning(
                "LLM message for %s failed validation, falling back: %s",
                card.strategy_id,
                checks,
            )
        except Exception:
            logger.warning(
                "LLM message generation failed for %s, falling back to approved template",
                card.strategy_id,
                exc_info=True,
            )

        message, checks = self._validated_template(card)
        return message, MessageSource.APPROVED_TEMPLATE_FALLBACK, checks

    def _validate_message(self, text: str, card: StrategyCard) -> dict[str, bool]:
        """Validate an LLM-generated message against all constraints."""
        message = re.sub(r"\s+", " ", text).strip()
        sentence_count = len(
            [part for part in re.split(r"(?<=[。！？!?])", message) if part.strip()]
        )
        return {
            "non_empty": bool(message),
            "within_character_limit": (
                len(message) <= self.library.principles.maximum_message_characters
            ),
            "within_sentence_limit": (
                sentence_count <= self.library.principles.maximum_message_sentences
            ),
            "contains_no_banned_phrase": not any(
                phrase in message for phrase in self.library.banned_phrases
            ),
            "target_actor_explicit": card.target_actor is not Actor.UNKNOWN,
        }

    def _validated_template(self, card: StrategyCard) -> tuple[str, dict[str, bool]]:
        message = re.sub(r"\s+", " ", card.approved_template).strip()
        sentence_count = len(
            [part for part in re.split(r"(?<=[。！？!?])", message) if part.strip()]
        )
        checks = {
            "non_empty": bool(message),
            "within_character_limit": (
                len(message) <= self.library.principles.maximum_message_characters
            ),
            "within_sentence_limit": (
                sentence_count <= self.library.principles.maximum_message_sentences
            ),
            "contains_no_banned_phrase": not any(
                phrase in message for phrase in self.library.banned_phrases
            ),
            "target_actor_explicit": card.target_actor is not Actor.UNKNOWN,
        }
        if not all(checks.values()):
            raise ValueError(f"approved template validation failed: {card.strategy_id}")
        return message, checks

    @staticmethod
    def _held(reason: StrategyHoldReason) -> StrategySelectionResult:
        return StrategySelectionResult(
            status=StrategySelectionStatus.HELD,
            hold_reason=reason,
        )


class MessageGenerator:
    """Wrap the LLM provider and prompt builder for message rephrasing.

    This abstraction keeps the selector testable: in tests, a mock or
    ``None`` can be supplied, while in production a real
    :class:`QwenTextChatProvider` is used.
    """

    def __init__(
        self,
        provider: object,
        *,
        max_characters: int,
        max_sentences: int,
        banned_phrases: list[str],
    ) -> None:
        self.provider = provider
        self.max_characters = max_characters
        self.max_sentences = max_sentences
        self.banned_phrases = banned_phrases

    def generate(self, card: StrategyCard, evidence: list[EvidenceReference]) -> str:
        """Generate a rephrased message for the given card and evidence.

        Raises:
            ConnectionError, ValueError, or other exceptions from the
            underlying provider.  The caller is responsible for catching
            these and falling back to the approved template.
        """
        prompt = build_message_prompt(
            card=card,
            evidence=evidence,
            max_characters=self.max_characters,
            max_sentences=self.max_sentences,
            banned_phrases=self.banned_phrases,
        )
        result = self.provider.generate(prompt)
        return result.text
