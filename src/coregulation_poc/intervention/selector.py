from __future__ import annotations

import logging
import re
from typing import Any

from coregulation_poc.intervention.message_prompt import (
    LLMMessageResult,
    build_message_prompt,
    parse_llm_response,
)
from coregulation_poc.intervention.models import (
    InterventionPlan,
    MessageSource,
    StrategyCard,
    StrategyHoldReason,
    StrategyLibraryConfig,
    StrategySelectionResult,
    StrategySelectionSource,
    StrategySelectionStatus,
)
from coregulation_poc.intervention.strategy_choice_prompt import (
    LLMStrategyChoice,
    build_strategy_choice_prompt,
    parse_strategy_choice,
)
from coregulation_poc.intervention.strategy_library import cards_by_id
from coregulation_poc.models import (
    Actor,
    ConfidenceLevel,
    CoregulationState,
    EvidenceReference,
    InterventionAction,
    InterventionDecision,
    StateAssessment,
    TaskProcess,
)

logger = logging.getLogger(__name__)

_ADDRESS_WORD: dict[str, str] = {
    "parent": "家长",
    "child": "小朋友",
    "both": "你们",
}

_ANSWER_INDICATORS: list[str] = [
    "答案是", "答案为", "正确答案", "答案：", "答案:",
]

_BLAME_COMMAND_INDICATORS: list[str] = [
    "你必须", "你应该", "你需要", "你错了", "家长错了",
    "你应该反思", "你必须道歉", "赶紧", "马上", "立刻",
    "诊断为", "多动症", "自闭症",
]

_OBSERVATION_MAX_LENGTH = 30

# Strategy families that directly address task progress and therefore
# require a matching task_process.  When task_process is None or unclear,
# candidates from these families are removed to prevent selecting a
# task-specific strategy without task evidence.
_TASK_DEPENDENT_FAMILIES: frozenset[str] = frozenset({
    "task_pacing",
    "learning_support",
    "autonomy_support",
})


class StrategySelector:
    """Select one approved card after module two authorizes intervention.

    Exact rules remain the first path. If their soft fields do not produce a
    card, an optional LLM may choose among a bounded set of cards that already
    satisfy hard state, target, evidence and non-repetition constraints.

    1. Collect all routing rules matching state + interaction_performance.
    2. Merge all candidate strategy IDs.
    3. Filter by ``support_need``.
    4. Filter by ``task_process`` (skip when None or unclear).
    5. Filter by ``support_target`` → ``target_actor`` mapping.
    6. Verify evidence exists for the support target.
    7. Exclude the previously used strategy.
    8. Select by ``priority`` (lower value = higher priority).
    """

    def __init__(
        self,
        library: StrategyLibraryConfig,
        message_generator: MessageGenerator | None = None,
        strategy_choice_generator: StrategyChoiceGenerator | None = None,
    ) -> None:
        self.library = library
        self.cards = cards_by_id(library)
        self.message_generator = message_generator
        self.strategy_choice_generator = strategy_choice_generator

    def select(
        self,
        *,
        assessment: StateAssessment,
        decision: InterventionDecision,
        previous_plan: InterventionPlan | None = None,
        task_context: dict[str, Any] | None = None,
        difficulty_feedback_boost: bool = False,
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
        evidence = assessment.modality_evidence.all_items
        if not self._support_target_has_evidence(assessment.support_target, evidence):
            return self._held(StrategyHoldReason.TARGET_ACTOR_EVIDENCE_INSUFFICIENT)

        selection_source = StrategySelectionSource.EXACT_RULE
        semantic_confidence: ConfidenceLevel | None = None
        semantic_relaxed_dimensions: list[str] = []
        selection_checks: dict[str, bool] = {}
        card = self._select_best_card(
            candidate_ids,
            assessment=assessment,
            previous_plan=previous_plan,
            difficulty_feedback_boost=difficulty_feedback_boost,
        )
        if card is None:
            if self.strategy_choice_generator is None:
                return self._held(StrategyHoldReason.TARGET_ACTOR_EVIDENCE_INSUFFICIENT)
            semantic_candidates = self._semantic_candidates(
                routed_candidate_ids=candidate_ids,
                assessment=assessment,
                previous_plan=previous_plan,
            )
            if not semantic_candidates:
                return self._held(StrategyHoldReason.SEMANTIC_SELECTOR_NO_MATCH)
            try:
                choice = self.strategy_choice_generator.generate(
                    assessment=assessment,
                    candidates=semantic_candidates,
                    task_context=task_context,
                )
            except Exception:
                logger.warning("bounded strategy selection failed", exc_info=True)
                return self._held(StrategyHoldReason.SEMANTIC_SELECTOR_UNAVAILABLE)
            if choice.strategy_id is None:
                return self._held(StrategyHoldReason.SEMANTIC_SELECTOR_NO_MATCH)
            candidate_map = {item.strategy_id: item for item in semantic_candidates}
            card = candidate_map.get(choice.strategy_id)
            if card is None or choice.confidence is ConfidenceLevel.LOW:
                return self._held(StrategyHoldReason.SEMANTIC_SELECTOR_REJECTED)

            semantic_relaxed_dimensions = self._relaxed_dimensions(
                card=card,
                routed_candidate_ids=candidate_ids,
                assessment=assessment,
            )
            selection_source = StrategySelectionSource.BOUNDED_LLM
            semantic_confidence = choice.confidence
            selection_reason = (
                "规则已完成干预授权及状态、对象、证据硬约束校验；"
                f"受限语义选择从批准卡片中选择 {card.strategy_id}：{choice.reason}"
            )
            selection_checks = self._semantic_selection_checks(
                choice=choice,
                card=card,
                candidates=semantic_candidates,
                assessment=assessment,
                evidence=evidence,
            )

        message, message_source, message_checks = self._resolve_message(
            card, assessment, evidence,
            task_context=task_context,
            previous_state=(decision.previous_state.value if decision.previous_state else None),
            recovery_status=decision.recovery_status.value,
        )
        checks = {**selection_checks, **message_checks}
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
            strategy_selection_source=selection_source,
            semantic_selection_confidence=semantic_confidence,
            semantic_relaxed_dimensions=semantic_relaxed_dimensions,
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

    def _semantic_candidates(
        self,
        *,
        routed_candidate_ids: list[str],
        assessment: StateAssessment,
        previous_plan: InterventionPlan | None,
    ) -> list[StrategyCard]:
        """Return approved cards satisfying every hard selection constraint."""

        expected_actor = self._support_target_to_target_actor(assessment.support_target)
        routed_ids = set(routed_candidate_ids)
        need_value = assessment.support_need.value if assessment.support_need else None
        process_value = assessment.task_process.value if assessment.task_process else None
        candidates: list[StrategyCard] = []
        for card in self.library.cards:
            if card.inactive or assessment.state not in card.states:
                continue
            if card.target_actor is not expected_actor:
                continue
            if previous_plan is not None and card.strategy_id == previous_plan.strategy_id:
                continue
            semantic_anchor = any(
                (
                    card.strategy_id in routed_ids,
                    need_value is not None and need_value in card.support_needs,
                    process_value is not None and process_value in card.task_processes,
                )
            )
            if not semantic_anchor:
                continue
            candidates.append(card)
        return sorted(candidates, key=lambda item: (item.priority, item.strategy_id))

    @staticmethod
    def _relaxed_dimensions(
        *,
        card: StrategyCard,
        routed_candidate_ids: list[str],
        assessment: StateAssessment,
    ) -> list[str]:
        """Compute soft dimensions relaxed by code rather than trusting the LLM."""

        relaxed: list[str] = []
        if card.strategy_id not in set(routed_candidate_ids):
            relaxed.append("interaction_performance")
        if (
            assessment.support_need is not None
            and assessment.support_need.value not in card.support_needs
        ):
            relaxed.append("support_need")
        if (
            assessment.task_process is not None
            and assessment.task_process is not TaskProcess.UNCLEAR
            and assessment.task_process.value not in card.task_processes
        ):
            relaxed.append("task_process")
        return relaxed

    def _semantic_selection_checks(
        self,
        *,
        choice: LLMStrategyChoice,
        card: StrategyCard,
        candidates: list[StrategyCard],
        assessment: StateAssessment,
        evidence: list[EvidenceReference],
    ) -> dict[str, bool]:
        """Revalidate an LLM choice against program-controlled hard facts."""

        expected_actor = self._support_target_to_target_actor(assessment.support_target)
        return {
            "semantic_strategy_from_approved_candidates": card.strategy_id
            in {item.strategy_id for item in candidates},
            "semantic_state_hard_constraint": assessment.state in card.states,
            "semantic_target_hard_constraint": card.target_actor is expected_actor,
            "semantic_evidence_hard_constraint": self._support_target_has_evidence(
                assessment.support_target, evidence
            ),
            "semantic_confidence_accepted": choice.confidence
            in {ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM},
        }

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
        """Collect all candidates from every matching routing rule.

        Returns ``None`` when no rule matches.  Otherwise returns a tuple
        of (joined performances, merged candidate IDs, selection reason).
        """
        observed = set(assessment.interaction_performance)
        matched_performances: list[str] = []
        candidate_ids: list[str] = []
        for rule in self.library.routing_rules:
            if rule.state is assessment.state and rule.interaction_performance in observed:
                matched_performances.append(rule.interaction_performance)
                candidate_ids.extend(rule.strategy_ids)
        if not candidate_ids:
            return None
        performance = "; ".join(matched_performances)
        return (
            performance,
            candidate_ids,
            "依据版本化规则，将模块一观察到的互动表现映射到主题分析中的修复策略。",
        )

    def _select_best_card(
        self,
        candidate_ids: list[str],
        *,
        assessment: StateAssessment,
        previous_plan: InterventionPlan | None,
        difficulty_feedback_boost: bool = False,
    ) -> StrategyCard | None:
        """Apply the 8-step filtering pipeline to select the best card."""

        # Steps 1-2: deduplicate and collect valid candidates
        candidates: list[StrategyCard] = []
        for strategy_id in dict.fromkeys(candidate_ids):
            card = self.cards.get(strategy_id)
            if card is None or card.inactive:
                continue
            if assessment.state is not None and assessment.state not in card.states:
                continue
            candidates.append(card)

        if not candidates:
            return None

        # Step 3: filter by support_need
        if assessment.support_need is not None:
            need_value = assessment.support_need.value
            candidates = [
                c for c in candidates if need_value in c.support_needs
            ]

        # Step 4: filter by task_process
        if (
            assessment.task_process is not None
            and assessment.task_process is not TaskProcess.UNCLEAR
        ):
            process_value = assessment.task_process.value
            candidates = [
                c for c in candidates if process_value in c.task_processes
            ]
        else:
            # task_process is None or unclear: exclude task-dependent
            # strategy families (task_pacing, learning_support,
            # autonomy_support) that require specific task evidence.
            candidates = [
                c for c in candidates
                if c.strategy_family not in _TASK_DEPENDENT_FAMILIES
            ]

        # Step 5: filter by support_target → target_actor
        expected_actor = self._support_target_to_target_actor(assessment.support_target)
        if expected_actor is not None:
            candidates = [c for c in candidates if c.target_actor is expected_actor]

        # If no cards match the target actor, hold (do not switch actors)
        if not candidates:
            return None

        # Step 6: verify evidence exists for the support target
        evidence = assessment.modality_evidence.all_items
        if not self._support_target_has_evidence(assessment.support_target, evidence):
            return None

        # Step 7: exclude previously used strategy
        if previous_plan is not None:
            candidates = [
                c for c in candidates if c.strategy_id != previous_plan.strategy_id
            ]

        if not candidates:
            return None

        # Step 8: select by priority (lower = higher priority).
        # When difficulty_feedback_boost is active, give task_pacing and
        # learning_support families a -1 priority bonus so they are
        # preferred over equally-priority alternatives.
        _BOOST_FAMILIES = frozenset({"task_pacing", "learning_support"})
        if difficulty_feedback_boost:
            candidates.sort(
                key=lambda c: (
                    c.priority - 1
                    if c.strategy_family in _BOOST_FAMILIES
                    else c.priority
                )
            )
        else:
            candidates.sort(key=lambda c: c.priority)
        return candidates[0]

    @staticmethod
    def _support_target_to_target_actor(support_target: Actor) -> Actor | None:
        """Map support_target to the expected target_actor for filtering.

        - parent  → parent
        - child   → child
        - both    → both
        - unknown → both
        """
        if support_target is Actor.PARENT:
            return Actor.PARENT
        if support_target is Actor.CHILD:
            return Actor.CHILD
        # both and unknown both map to both
        return Actor.BOTH

    @staticmethod
    def _support_target_has_evidence(
        support_target: Actor,
        evidence: list[EvidenceReference],
    ) -> bool:
        """Check that modality evidence supports the declared support target."""
        if support_target is Actor.PARENT:
            return any(item.actor is Actor.PARENT for item in evidence)
        if support_target is Actor.CHILD:
            return any(item.actor is Actor.CHILD for item in evidence)
        if support_target is Actor.BOTH:
            has_parent = any(item.actor is Actor.PARENT for item in evidence)
            has_child = any(item.actor is Actor.CHILD for item in evidence)
            has_both = any(item.actor is Actor.BOTH for item in evidence)
            return has_both or (has_parent and has_child)
        # unknown: already filtered to both-targeted cards; sufficient
        # evidence is guaranteed by the assessment validation
        return True

    # ------------------------------------------------------------------
    # Message resolution
    # ------------------------------------------------------------------

    def _resolve_message(
        self,
        card: StrategyCard,
        assessment: StateAssessment,
        evidence: list[EvidenceReference],
        *,
        task_context: dict[str, Any] | None = None,
        previous_state: str | None = None,
        recovery_status: str | None = None,
    ) -> tuple[str, MessageSource, dict[str, bool]]:
        """Resolve the intervention message.

        When no generator is configured, returns the approved template.
        When the LLM succeeds and passes all validation checks, returns
        the assembled message (address + observation + action clause)
        with source ``CONSTRAINED_LLM``.  When the LLM fails or
        validation fails, falls back to the approved template as the
        complete message with source ``APPROVED_TEMPLATE_FALLBACK``.
        """
        if self.message_generator is None:
            message, checks = self._validated_template(card)
            return message, MessageSource.APPROVED_TEMPLATE, checks

        try:
            result = self.message_generator.generate(
                card, evidence,
                task_context=task_context,
                previous_state=previous_state,
                recovery_status=recovery_status,
            )
            checks = self._validate_llm_result(result, card, assessment, evidence)
            if all(checks.values()):
                message = self._assemble_message(result, card)
                return message, MessageSource.CONSTRAINED_LLM, checks
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

    def _validate_llm_result(
        self,
        result: LLMMessageResult,
        card: StrategyCard,
        assessment: StateAssessment,
        evidence: list[EvidenceReference],
    ) -> dict[str, bool]:
        """Validate an LLM-generated structured result."""
        assembled = self._assemble_message(result, card)
        sentence_count = len(
            [part for part in re.split(r"(?<=[。！？!?])", assembled) if part.strip()]
        )

        # Valid evidence IDs
        valid_evidence_ids = {f"evidence_{i}" for i in range(len(evidence))}

        # Expected target actor from support_target
        expected_actor = self._support_target_to_target_actor(assessment.support_target)
        expected_actor_value = expected_actor.value if expected_actor else ""

        return {
            "strategy_id_matches": result.strategy_id == card.strategy_id,
            "target_actor_matches": result.target_actor == expected_actor_value,
            "evidence_ids_valid": bool(result.evidence_ids) and all(
                eid in valid_evidence_ids for eid in result.evidence_ids
            ),
            "observation_within_limit": (
                len(result.observation_clause) <= _OBSERVATION_MAX_LENGTH
            ),
            "within_character_limit": (
                len(assembled) <= self.library.principles.maximum_message_characters
            ),
            "within_sentence_limit": (
                sentence_count <= self.library.principles.maximum_message_sentences
            ),
            "contains_no_banned_phrase": not any(
                phrase in assembled for phrase in self.library.banned_phrases
            ),
            "contains_no_answer": not any(
                phrase in assembled for phrase in _ANSWER_INDICATORS
            ),
            "contains_no_blame_command": not any(
                phrase in assembled for phrase in _BLAME_COMMAND_INDICATORS
            ),
            "action_clause_unchanged": card.approved_action_clause in assembled,
            "target_actor_explicit": card.target_actor is not Actor.UNKNOWN,
        }

    @staticmethod
    def _assemble_message(result: LLMMessageResult, card: StrategyCard) -> str:
        """Assemble the final message from LLM observation and card action."""
        address_word = _ADDRESS_WORD.get(result.target_actor, result.target_actor)
        observation = result.observation_clause.strip().rstrip("。！？!?，,;；")
        return f"{address_word}，{observation}，{card.approved_action_clause}"

    def _validate_message(self, text: str, card: StrategyCard) -> dict[str, bool]:
        """Validate a plain-text LLM message (legacy compatibility)."""
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
            "contains_no_answer": not any(
                phrase in message for phrase in _ANSWER_INDICATORS
            ),
            "contains_no_blame_command": not any(
                phrase in message for phrase in _BLAME_COMMAND_INDICATORS
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
            "contains_no_answer": not any(
                phrase in message for phrase in _ANSWER_INDICATORS
            ),
            "contains_no_blame_command": not any(
                phrase in message for phrase in _BLAME_COMMAND_INDICATORS
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
    """Wrap the LLM provider and prompt builder for observation generation.

    The generator asks the LLM to produce only an ``observation_clause``.
    The selector assembles the final message by combining the observation
    with the strategy card's fixed ``approved_action_clause``.

    In tests, a mock or ``None`` can be supplied, while in production a
    real :class:`QwenTextChatProvider` is used.
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
        self.call_count = 0

    def generate(
        self,
        card: StrategyCard,
        evidence: list[EvidenceReference],
        *,
        task_context: dict[str, Any] | None = None,
        previous_state: str | None = None,
        recovery_status: str | None = None,
    ) -> LLMMessageResult:
        """Generate a structured observation result for the given card.

        Raises
        ------
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
            task_context=task_context,
            previous_state=previous_state,
            recovery_status=recovery_status,
        )
        self.call_count += 1
        result = self.provider.generate(prompt)
        return parse_llm_response(result.text)


class StrategyChoiceGenerator:
    """Use an LLM only to rank a program-bounded set of approved cards."""

    def __init__(self, provider: object) -> None:
        self.provider = provider
        self.call_count = 0

    def generate(
        self,
        *,
        assessment: StateAssessment,
        candidates: list[StrategyCard],
        task_context: dict[str, Any] | None = None,
    ) -> LLMStrategyChoice:
        prompt = build_strategy_choice_prompt(
            assessment=assessment,
            candidates=candidates,
            task_context=task_context,
        )
        self.call_count += 1
        result = self.provider.generate(prompt)
        return parse_strategy_choice(result.text)
