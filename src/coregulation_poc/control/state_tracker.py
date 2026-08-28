from __future__ import annotations

from coregulation_poc.control.intervention_policy import (
    DecisionRule,
    InterventionPolicy,
    StateActionRule,
)
from coregulation_poc.models import (
    Actor,
    ConfidenceLevel,
    ControlObservation,
    CoregulationState,
    EvidenceSufficiency,
    InteractionTrajectory,
    Interruptibility,
    InterventionAction,
    InterventionDecision,
    InterventionDecisionReason,
    RecoveryStatus,
    StateAssessment,
    StateTrajectoryPoint,
    StateTrajectorySnapshot,
    SupportNeed,
)

STATE_RANK = {
    CoregulationState.NORMAL: 0,
    CoregulationState.FLUCTUATION: 1,
    CoregulationState.DYSREGULATION: 2,
    CoregulationState.HIGH_RISK: 3,
}


class StateTrajectoryController:
    """Convert module-one state assessments into research-grounded timing decisions."""

    def __init__(self, policy: InterventionPolicy) -> None:
        self.policy = policy
        self._session_id: str | None = None
        self._points: list[StateTrajectoryPoint] = []
        self._decisions: list[InterventionDecision] = []
        self._pending_intervention_state: CoregulationState | None = None
        self._post_intervention_wait_count: int = 0
        self._reinforced_positive_performances: set[str] = set()
        self._episode_active = False
        self._episode_interventions: list[
            tuple[int, CoregulationState, SupportNeed | None, Actor, frozenset[str]]
        ] = []

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def awaiting_post_intervention_response(self) -> bool:
        return self._pending_intervention_state is not None

    def mark_intervention_not_delivered(self) -> None:
        """Release the response guard when every delivery channel failed."""

        if self._pending_intervention_state is None:
            raise ValueError("there is no pending intervention delivery to release")
        self._pending_intervention_state = None
        self._post_intervention_wait_count = 0
        if self._episode_interventions:
            self._episode_interventions.pop()
        if not self._episode_interventions:
            self._episode_active = False

    def mark_intervention_delivered(
        self,
        state: CoregulationState,
        *,
        delivered_at_ms: int | None = None,
    ) -> None:
        """Confirm delivery and start episode timing from the actual delivery time."""

        if state not in {
            CoregulationState.NORMAL,
            CoregulationState.DYSREGULATION,
            CoregulationState.HIGH_RISK,
        }:
            raise ValueError("delivered intervention state is not actionable")
        if self._pending_intervention_state is None:
            self._pending_intervention_state = state
        elif self._pending_intervention_state is not state:
            raise ValueError("delivered intervention state does not match the pending state")
        self._post_intervention_wait_count = 0
        if delivered_at_ms is None:
            return
        if isinstance(delivered_at_ms, bool) or delivered_at_ms < 0:
            raise ValueError("delivered_at_ms must be a non-negative integer")
        if state not in {
            CoregulationState.DYSREGULATION,
            CoregulationState.HIGH_RISK,
        }:
            return
        if not self._episode_interventions:
            raise ValueError("delivered intervention has no episode record")

        assessed_at_ms, recorded_state, need, target, performances = (
            self._episode_interventions[-1]
        )
        if recorded_state is not state:
            raise ValueError("delivered intervention does not match the latest episode record")
        self._episode_interventions[-1] = (
            max(assessed_at_ms, delivered_at_ms),
            recorded_state,
            need,
            target,
            performances,
        )

    def ingest(
        self,
        observation: ControlObservation,
        *,
        defer_delivery_timing: bool = False,
    ) -> InterventionDecision:
        """Record a state and decide whether an intervention is authorized.

        ``defer_delivery_timing`` is used by the realtime runtime, which can retain
        an authorized plan and deliver it at the next live audio boundary.  The
        default behavior remains conservative for offline callers that cannot
        queue a plan.
        """
        assessment = observation.assessment
        self._validate_observation(observation)
        previous_state = self._points[-1].state if self._points else None
        sequence = len(self._points) + 1
        point = StateTrajectoryPoint(
            sequence=sequence,
            assessed_at_ms=assessment.assessed_at_ms,
            state=assessment.state,
            confidence=assessment.confidence,
            evidence_sufficiency=assessment.evidence_sufficiency,
            interaction_performance=assessment.interaction_performance,
        )
        self._points.append(point)

        # BoundaryStateTracker emits NORMAL only after confirmed stable recovery.
        # A provisional recovery is FLUCTUATION and intentionally keeps the same
        # episode open.
        if assessment.state is CoregulationState.NORMAL:
            self._reset_episode()

        recovery_status = self._recovery_status(observation)
        if self.awaiting_post_intervention_response:
            if not observation.post_intervention_response_observed:
                self._post_intervention_wait_count += 1
                if (
                    self._post_intervention_wait_count
                    >= self.policy.principles.post_intervention_max_wait_count
                ):
                    self._pending_intervention_state = None
                    self._post_intervention_wait_count = 0
                    recovery_status = RecoveryStatus.TIMEOUT
                else:
                    return self._record_guard_decision(
                        observation=observation,
                        previous_state=previous_state,
                        sequence=sequence,
                        reason=InterventionDecisionReason.WAITING_FOR_POST_INTERVENTION_RESPONSE,
                        recovery_status=RecoveryStatus.PENDING,
                    )
            else:
                self._pending_intervention_state = None
                self._post_intervention_wait_count = 0

        if (
            assessment.evidence_sufficiency is EvidenceSufficiency.INSUFFICIENT
            or assessment.state is None
        ):
            return self._record_guard_decision(
                observation=observation,
                previous_state=previous_state,
                sequence=sequence,
                reason=InterventionDecisionReason.INSUFFICIENT_EVIDENCE,
                recovery_status=recovery_status,
            )
        if assessment.confidence is ConfidenceLevel.LOW:
            return self._record_guard_decision(
                observation=observation,
                previous_state=previous_state,
                sequence=sequence,
                reason=InterventionDecisionReason.LOW_CONFIDENCE,
                recovery_status=recovery_status,
            )

        positive_performances = set()
        if self.policy.principles.positive_maintenance_enabled:
            positive_performances = self._positive_maintenance_performances(
                observation=observation,
                previous_state=previous_state,
                recovery_status=recovery_status,
            )
        if positive_performances:
            rule = self.policy.positive_maintenance
            if assessment.interruptibility is not Interruptibility.NATURAL_PAUSE:
                return self._record_guard_decision(
                    observation=observation,
                    previous_state=previous_state,
                    sequence=sequence,
                    reason=InterventionDecisionReason.WAITING_FOR_NATURAL_TURN_BOUNDARY,
                    recovery_status=recovery_status,
                )
            if rule.requires_natural_turn_boundary and not observation.natural_turn_boundary:
                return self._record_guard_decision(
                    observation=observation,
                    previous_state=previous_state,
                    sequence=sequence,
                    reason=InterventionDecisionReason.WAITING_FOR_NATURAL_TURN_BOUNDARY,
                    recovery_status=recovery_status,
                )
            decision = self._record_state_decision(
                observation=observation,
                previous_state=previous_state,
                sequence=sequence,
                rule=rule,
                recovery_status=recovery_status,
            )
            self._reinforced_positive_performances.update(positive_performances)
            return decision

        rule = self.policy.state_actions[assessment.state]

        if rule.history_required and not observation.interaction_history_available:
            return self._record_guard_decision(
                observation=observation,
                previous_state=previous_state,
                sequence=sequence,
                reason=InterventionDecisionReason.HISTORY_REQUIRED,
                recovery_status=recovery_status,
            )
        if assessment.state in {
            CoregulationState.DYSREGULATION,
            CoregulationState.HIGH_RISK,
        }:
            episode_guard = self._same_episode_guard(assessment)
            if episode_guard is not None:
                return self._record_guard_decision(
                    observation=observation,
                    previous_state=previous_state,
                    sequence=sequence,
                    reason=episode_guard,
                    recovery_status=recovery_status,
                )
        return self._record_state_decision(
            observation=observation,
            previous_state=previous_state,
            sequence=sequence,
            rule=rule,
            recovery_status=recovery_status,
        )

    def _positive_maintenance_performances(
        self,
        *,
        observation: ControlObservation,
        previous_state: CoregulationState | None,
        recovery_status: RecoveryStatus,
    ) -> set[str]:
        assessment = observation.assessment
        rule = self.policy.positive_maintenance
        observed = set(assessment.interaction_performance)
        explicit = observed.intersection(rule.explicit_trigger_performances)
        self._reinforced_positive_performances.intersection_update(explicit)

        candidates = set()

        # Case 1: explicit positive reinforcement need in an allowed state.
        if (
            assessment.state in rule.allowed_states
            and assessment.support_need is SupportNeed.POSITIVE_REINFORCEMENT
        ):
            candidates |= explicit

        # Case 2: recovery from fluctuation confirmed by trajectory.
        if (
            assessment.state in rule.allowed_states
            and previous_state in rule.recovery_transition_states
            and assessment.trajectory is InteractionTrajectory.RECOVERING
            and recovery_status is RecoveryStatus.NOT_APPLICABLE
        ):
            candidates |= explicit

        return candidates - self._reinforced_positive_performances

    def _same_episode_guard(
        self,
        assessment: StateAssessment,
    ) -> InterventionDecisionReason | None:
        """Keep only a delivery-anchored interval between repeated prompts."""

        if not self._episode_active or not self._episode_interventions:
            return None
        last_time = self._episode_interventions[-1][0]
        elapsed_ms = max(0, assessment.assessed_at_ms - last_time)
        if elapsed_ms < self.policy.principles.same_episode_observation_ms:
            return InterventionDecisionReason.SAME_EPISODE_OBSERVATION_PERIOD
        return None

    def _reset_episode(self) -> None:
        self._episode_active = False
        self._episode_interventions.clear()

    def snapshot(self) -> StateTrajectorySnapshot:
        if self._session_id is None:
            raise ValueError("cannot create a trajectory snapshot before the first observation")
        return StateTrajectorySnapshot(
            session_id=self._session_id,
            policy_version=self.policy.version,
            points=list(self._points),
            decisions=list(self._decisions),
            awaiting_post_intervention_response=self.awaiting_post_intervention_response,
        )

    def _validate_observation(self, observation: ControlObservation) -> None:
        assessment = observation.assessment
        if self._session_id is None:
            self._session_id = assessment.session_id
        elif assessment.session_id != self._session_id:
            raise ValueError("all trajectory observations must use the same session_id")
        if self._points and assessment.assessed_at_ms < self._points[-1].assessed_at_ms:
            raise ValueError("trajectory assessment timestamps must be non-decreasing")
        if assessment.previous_state is not None:
            if not observation.interaction_history_available:
                raise ValueError("previous_state requires available interaction history")
            if self._points and assessment.previous_state is not self._points[-1].state:
                raise ValueError("assessment previous_state does not match the trajectory")

    def _recovery_status(self, observation: ControlObservation) -> RecoveryStatus:
        if self._pending_intervention_state is None:
            return RecoveryStatus.NOT_APPLICABLE
        if not observation.post_intervention_response_observed:
            return RecoveryStatus.PENDING
        current_state = observation.assessment.state
        if current_state is None:
            return RecoveryStatus.INDETERMINATE
        previous_rank = STATE_RANK[self._pending_intervention_state]
        current_rank = STATE_RANK[current_state]
        if current_state is CoregulationState.NORMAL:
            return RecoveryStatus.RECOVERED
        if current_rank < previous_rank:
            return RecoveryStatus.PARTIAL_RECOVERY
        if current_rank == previous_rank:
            return RecoveryStatus.NOT_RECOVERED
        return RecoveryStatus.DETERIORATED

    def _record_guard_decision(
        self,
        *,
        observation: ControlObservation,
        previous_state: CoregulationState | None,
        sequence: int,
        reason: InterventionDecisionReason,
        recovery_status: RecoveryStatus,
    ) -> InterventionDecision:
        rule = self.policy.guard_actions[reason]
        return self._record_decision(
            observation=observation,
            previous_state=previous_state,
            sequence=sequence,
            rule=rule,
            recovery_status=recovery_status,
        )

    def _record_state_decision(
        self,
        *,
        observation: ControlObservation,
        previous_state: CoregulationState | None,
        sequence: int,
        rule: StateActionRule,
        recovery_status: RecoveryStatus,
    ) -> InterventionDecision:
        decision = self._record_decision(
            observation=observation,
            previous_state=previous_state,
            sequence=sequence,
            rule=rule,
            recovery_status=recovery_status,
        )
        if decision.action in {
            InterventionAction.REINFORCE,
            InterventionAction.INTERVENE,
            InterventionAction.PROGRESSIVE_SUPPORT,
        }:
            self._pending_intervention_state = decision.current_state
            self._post_intervention_wait_count = 0
            if decision.current_state in {
                CoregulationState.DYSREGULATION,
                CoregulationState.HIGH_RISK,
            }:
                assessment = observation.assessment
                self._episode_active = True
                self._episode_interventions.append(
                    (
                        assessment.assessed_at_ms,
                        decision.current_state,
                        assessment.support_need,
                        assessment.support_target,
                        frozenset(assessment.interaction_performance),
                    )
                )
        return decision

    def _record_decision(
        self,
        *,
        observation: ControlObservation,
        previous_state: CoregulationState | None,
        sequence: int,
        rule: DecisionRule,
        recovery_status: RecoveryStatus,
    ) -> InterventionDecision:
        assessment = observation.assessment
        action_requires_strategy = rule.action in {
            InterventionAction.REINFORCE,
            InterventionAction.INTERVENE,
            InterventionAction.PROGRESSIVE_SUPPORT,
        }
        actors = self._evidence_actors(observation)
        decision = InterventionDecision(
            session_id=assessment.session_id,
            sequence=sequence,
            decided_at_ms=assessment.assessed_at_ms,
            previous_state=previous_state,
            current_state=assessment.state,
            action=rule.action,
            reason_code=rule.reason_code,
            reason=rule.rationale,
            natural_turn_boundary=observation.natural_turn_boundary,
            intervention_permitted=action_requires_strategy,
            strategy_selection_required=action_requires_strategy,
            recovery_status=recovery_status,
            evidence_actors=actors,
            interaction_performance=assessment.interaction_performance,
            research_basis=rule.research_basis,
        )
        self._decisions.append(decision)
        return decision

    @staticmethod
    def _evidence_actors(observation: ControlObservation) -> list[Actor]:
        actors = [item.actor for item in observation.assessment.modality_evidence.all_items]
        return list(dict.fromkeys(actors)) or [Actor.UNKNOWN]
