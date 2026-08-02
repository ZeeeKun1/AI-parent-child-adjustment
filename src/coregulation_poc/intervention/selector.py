from __future__ import annotations

import re

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
    InterventionAction,
    InterventionDecision,
    RecoveryStatus,
    StateAssessment,
)


class StrategySelector:
    """Select one research card after module two has authorized intervention."""

    def __init__(self, library: StrategyLibraryConfig) -> None:
        self.library = library
        self.cards = cards_by_id(library)

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
        if assessment.state not in {
            CoregulationState.DYSREGULATION,
            CoregulationState.HIGH_RISK,
        }:
            return self._held(StrategyHoldReason.STATE_NOT_SUPPORTED)

        routed = self._route(assessment, decision, previous_plan)
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

        message, checks = self._validated_template(card)
        evidence = assessment.modality_evidence.all_items
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
            message_source=MessageSource.APPROVED_TEMPLATE,
            selection_reason=selection_reason,
            selected_from_interaction_performance=performance,
            evidence_references=evidence,
            expected_recovery=card.expected_recovery,
            outcome_interpretation=self.library.outcome_mapping,
            validation_checks=checks,
            previous_strategy_id=(
                None if previous_plan is None else previous_plan.strategy_id
            ),
            next_strategy_if_not_recovered=card.next_strategy_if_not_recovered,
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
        previous_plan: InterventionPlan | None,
    ) -> tuple[str, list[str], str] | None:
        if previous_plan is not None and decision.recovery_status is RecoveryStatus.DETERIORATED:
            return (
                "post-intervention deterioration",
                ["DYAD_AFFECT_BRAKE"],
                "模块二观察到干预后状态恶化，因此优先使用经过审查的双方中性制动策略。",
            )
        if (
            previous_plan is not None
            and decision.recovery_status is RecoveryStatus.NOT_RECOVERED
            and previous_plan.next_strategy_if_not_recovered is not None
        ):
            return (
                "previous strategy did not produce observable recovery",
                [previous_plan.next_strategy_if_not_recovered],
                "模块二已经观察到回应但未出现恢复，因此沿用策略卡记录的下一步，"
                "不重复相同话术。",
            )

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
        expanded = list(candidate_ids)
        if previous_plan is not None and previous_plan.strategy_id in expanded:
            previous_card = self.cards[previous_plan.strategy_id]
            next_id = previous_card.next_strategy_if_not_recovered
            expanded = [item for item in expanded if item != previous_plan.strategy_id]
            if next_id is not None:
                expanded.insert(0, next_id)

        for strategy_id in dict.fromkeys(expanded):
            card = self.cards.get(strategy_id)
            if card is None or state not in card.states:
                continue
            if self._actor_supported(card.target_actor, evidence_actors):
                return card
        return None

    @staticmethod
    def _actor_supported(target: Actor, observed: set[Actor]) -> bool:
        if target is Actor.BOTH:
            return True
        return target in observed

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
