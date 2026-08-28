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


class ContextualMessageGenerationError(RuntimeError):
    """Raised when two contextual generation attempts cannot produce a safe message."""

_ANSWER_INDICATORS: list[str] = [
    "答案是",
    "答案为",
    "正确答案",
    "答案：",
    "答案:",
]

_BLAME_COMMAND_INDICATORS: list[str] = [
    "你必须",
    "你应该",
    "你需要",
    "你错了",
    "家长错了",
    "你应该反思",
    "你必须道歉",
    "赶紧",
    "马上",
    "立刻",
    "诊断为",
    "多动症",
    "自闭症",
]


class StrategySelector:
    """Select one approved card after module two authorizes intervention.

    State authorization and an explicit target are hard constraints.  The
    observed performance, support need, task process and recent strategy are
    soft ranking dimensions.  Therefore an actionable state always receives
    the closest approved card even when one auxiliary label is absent.
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
            performance = "; ".join(assessment.interaction_performance) or assessment.state.value
            candidate_ids: list[str] = []
        else:
            performance, candidate_ids, _ = routed
        evidence = assessment.modality_evidence.all_items

        semantic_candidates = self._semantic_candidates(
            routed_candidate_ids=candidate_ids,
            assessment=assessment,
            previous_plan=previous_plan,
        )
        if not semantic_candidates:
            return self._held(StrategyHoldReason.SEMANTIC_SELECTOR_NO_MATCH)

        card = self._select_best_card(
            [item.strategy_id for item in semantic_candidates],
            assessment=assessment,
            previous_plan=previous_plan,
            difficulty_feedback_boost=difficulty_feedback_boost,
            routed_candidate_ids=candidate_ids,
        )
        if card is None:
            return self._held(StrategyHoldReason.SEMANTIC_SELECTOR_NO_MATCH)

        selection_source = StrategySelectionSource.EXACT_RULE
        semantic_confidence: ConfidenceLevel | None = None
        semantic_relaxed_dimensions = self._relaxed_dimensions(
            card=card,
            routed_candidate_ids=candidate_ids,
            assessment=assessment,
        )
        selection_reason = (
            "状态授权后，在同状态、同对象的批准策略卡中，依据互动表现、"
            "支持需要与任务过程进行相似度排序。"
        )
        selection_checks: dict[str, bool] = {
            "strategy_from_approved_library": True,
            "strategy_state_authorized": assessment.state in card.states,
            "strategy_target_explicit": card.target_actor is not Actor.UNKNOWN,
        }
        if self.strategy_choice_generator is not None and len(semantic_candidates) > 1:
            try:
                choice = self.strategy_choice_generator.generate(
                    assessment=assessment,
                    candidates=semantic_candidates,
                    task_context=task_context,
                )
            except Exception:
                logger.warning("bounded strategy selection failed", exc_info=True)
                selection_source = StrategySelectionSource.DETERMINISTIC_FALLBACK
            else:
                candidate_map = {item.strategy_id: item for item in semantic_candidates}
                llm_card = candidate_map.get(choice.strategy_id)
                if llm_card is not None and choice.confidence is not ConfidenceLevel.LOW:
                    card = llm_card
                    semantic_relaxed_dimensions = self._relaxed_dimensions(
                        card=card,
                        routed_candidate_ids=candidate_ids,
                        assessment=assessment,
                    )
                    selection_source = StrategySelectionSource.BOUNDED_LLM
                    semantic_confidence = choice.confidence
                    selection_reason = (
                        "状态和对象由规则限定；大模型从批准策略卡中选择与当前情境"
                        f"最接近的 {card.strategy_id}：{choice.reason}"
                    )
                    selection_checks = self._semantic_selection_checks(
                        choice=choice,
                        card=card,
                        candidates=semantic_candidates,
                        assessment=assessment,
                        evidence=evidence,
                    )
                else:
                    selection_source = StrategySelectionSource.DETERMINISTIC_FALLBACK
                    selection_reason += " 语义选择不确定，采用程序计算的最接近卡片。"

        try:
            message, message_source, message_checks = self._resolve_message(
                card,
                assessment,
                evidence,
                task_context=task_context,
                previous_state=(
                    decision.previous_state.value if decision.previous_state else None
                ),
                recovery_status=decision.recovery_status.value,
            )
        except ContextualMessageGenerationError:
            logger.warning(
                "contextual message generation failed twice for %s",
                card.strategy_id,
                exc_info=True,
            )
            return self._held(StrategyHoldReason.MESSAGE_GENERATION_FAILED)
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
            previous_strategy_id=(None if previous_plan is None else previous_plan.strategy_id),
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
        """Return approved same-state cards for the requested target.

        Performance, support need and task process intentionally remain soft
        dimensions.  Repetition is avoided when alternatives exist, but it is
        not allowed to erase the only safe option.
        """

        expected_actor = self._support_target_to_target_actor(assessment.support_target)
        candidates: list[StrategyCard] = []
        for card in self.library.cards:
            if card.inactive or assessment.state not in card.states:
                continue
            if card.target_actor is not expected_actor:
                continue
            candidates.append(card)
        if previous_plan is not None and len(candidates) > 1:
            without_previous = [
                card for card in candidates if card.strategy_id != previous_plan.strategy_id
            ]
            if without_previous:
                candidates = without_previous
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

        return {
            "semantic_strategy_from_approved_candidates": card.strategy_id
            in {item.strategy_id for item in candidates},
            "semantic_state_hard_constraint": assessment.state in card.states,
            "semantic_target_hard_constraint": card.target_actor
            is self._support_target_to_target_actor(assessment.support_target),
            "semantic_evidence_traceable": bool(evidence),
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
        routed_candidate_ids: list[str] | None = None,
    ) -> StrategyCard | None:
        """Rank approved cards without turning auxiliary fields into vetoes."""

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

        expected_actor = self._support_target_to_target_actor(assessment.support_target)
        candidates = [c for c in candidates if c.target_actor is expected_actor]
        if not candidates:
            return None

        routed_ids = set(routed_candidate_ids or [])
        need_value = assessment.support_need.value if assessment.support_need else None
        process_value = assessment.task_process.value if assessment.task_process else None
        boost_families = {"task_pacing", "learning_support"}

        def ranking(card: StrategyCard) -> tuple[int, int, str]:
            similarity = 0
            if card.strategy_id in routed_ids:
                similarity += 5
            if need_value not in {None, "none", "unclear"} and need_value in card.support_needs:
                similarity += 4
            if (
                process_value not in {None, TaskProcess.UNCLEAR.value}
                and process_value in card.task_processes
            ):
                similarity += 3
            if difficulty_feedback_boost and card.strategy_family in boost_families:
                similarity += 2
            if previous_plan is not None and card.strategy_id == previous_plan.strategy_id:
                similarity -= 2
            return (-similarity, card.priority, card.strategy_id)

        candidates.sort(key=ranking)
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

        The LLM writes the complete contextual message while the selected card
        fixes its target and strategy intent. A failed first draft is repaired
        once with explicit validation feedback; production does not silently
        replace it with a generic approved template.
        """
        if self.message_generator is None:
            message, checks = self._validated_template(card)
            return message, MessageSource.APPROVED_TEMPLATE, checks

        previous_result: LLMMessageResult | None = None
        failed_checks: list[str] = []
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                result = self.message_generator.generate(
                    card,
                    evidence,
                    task_context=task_context,
                    previous_state=previous_state,
                    recovery_status=recovery_status,
                    previous_result=previous_result,
                    failed_checks=failed_checks,
                )
                checks = self._validate_llm_result(result, card, assessment, evidence)
                if all(checks.values()):
                    message = self._assemble_message(result)
                    if attempt:
                        checks["contextual_rewrite_completed"] = True
                    return message, MessageSource.CONSTRAINED_LLM, checks
                previous_result = result
                failed_checks = [name for name, passed in checks.items() if not passed]
                logger.warning(
                    "LLM message for %s failed validation; requesting rewrite: %s",
                    card.strategy_id,
                    failed_checks,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM message generation attempt %s failed for %s",
                    attempt + 1,
                    card.strategy_id,
                    exc_info=True,
                )
        raise ContextualMessageGenerationError(
            f"unable to generate a valid contextual message for {card.strategy_id}"
        ) from last_error

    def _validate_llm_result(
        self,
        result: LLMMessageResult,
        card: StrategyCard,
        assessment: StateAssessment,
        evidence: list[EvidenceReference],
    ) -> dict[str, bool]:
        """Validate an LLM-generated structured result."""
        assembled = self._assemble_message(result)
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
            "evidence_ids_valid": bool(result.evidence_ids)
            and all(eid in valid_evidence_ids for eid in result.evidence_ids),
            "message_non_empty": bool(assembled),
            "within_character_limit": (
                len(assembled) <= self.library.principles.maximum_message_characters
            ),
            "within_sentence_limit": (
                sentence_count <= self.library.principles.maximum_message_sentences
            ),
            "contains_no_banned_phrase": not any(
                phrase in assembled for phrase in self.library.banned_phrases
            ),
            "contains_no_answer": not any(phrase in assembled for phrase in _ANSWER_INDICATORS),
            "contains_no_blame_command": not any(
                phrase in assembled for phrase in _BLAME_COMMAND_INDICATORS
            ),
            "target_actor_explicit": card.target_actor is not Actor.UNKNOWN,
        }

    @staticmethod
    def _assemble_message(result: LLMMessageResult) -> str:
        """Normalise the complete contextual message returned by the LLM."""
        return re.sub(r"\s+", " ", result.message).strip()

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
            "contains_no_answer": not any(phrase in message for phrase in _ANSWER_INDICATORS),
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
            "contains_no_answer": not any(phrase in message for phrase in _ANSWER_INDICATORS),
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
    """Wrap the LLM provider and prompt builder for complete message generation.

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
        previous_result: LLMMessageResult | None = None,
        failed_checks: list[str] | None = None,
    ) -> LLMMessageResult:
        """Generate a complete contextual message for the given card.

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
            previous_draft=(
                None if previous_result is None else self._normalise_draft(previous_result)
            ),
            failed_checks=failed_checks,
        )
        self.call_count += 1
        result = self.provider.generate(prompt)
        return parse_llm_response(result.text)

    @staticmethod
    def _normalise_draft(result: LLMMessageResult) -> str:
        return re.sub(r"\s+", " ", result.message).strip()


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
