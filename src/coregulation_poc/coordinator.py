"""Session coordinator connecting all four co-regulation modules.

The coordinator receives a Module 1 state assessment, determines whether a
post-intervention response has been observed by comparing the current
assessment against the pending intervention state, then feeds the
observation through Module 2 (timing control), Module 3 (strategy
selection), and Module 4 (delivery).  State between cycles is maintained
so that Module 3 can avoid repeating a strategy before observing the
dyad's response.

Natural turn boundaries are detected internally from audio energy: the
coordinator holds a :class:`TurnBoundaryDetector` and audio chunks are
fed via :meth:`ingest_audio_chunk`.  When :meth:`process` is called, the
coordinator queries the detector to determine whether it is safe to
intervene without interrupting active speech.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from coregulation_poc.capture.turn_boundary import TurnBoundaryDetector
from coregulation_poc.control import (
    InterventionPolicy,
    StateTrajectoryController,
    load_intervention_policy,
)
from coregulation_poc.delivery import (
    DeliveryCoordinator,
    DeliveryPolicy,
    DeliveryPreparationResult,
    DeliveryPreparationStatus,
    DeliveryRuntimeContext,
    load_delivery_policy,
)
from coregulation_poc.intervention import (
    InterventionPlan,
    StrategyLibraryConfig,
    StrategySelectionResult,
    StrategySelector,
    load_strategy_library,
)
from coregulation_poc.models import (
    ControlObservation,
    EvidenceSufficiency,
    InterventionDecision,
    StateAssessment,
    StateTrajectorySnapshot,
)


class CoordinatorCycleOutcome(StrEnum):
    """What happened in one coordinator cycle."""

    NO_INTERVENTION = "no_intervention"
    INTERVENTION_DELIVERED = "intervention_delivered"
    INTERVENTION_HELD = "intervention_held"
    STRATEGY_HELD = "strategy_held"


class CoordinatorCycleResult(BaseModel):
    """One complete coordinator cycle: Module 1 -> Module 2 -> Module 3 -> Module 4."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    assessment: StateAssessment
    decision: InterventionDecision
    post_intervention_response_observed: bool
    natural_turn_boundary: bool
    selection_result: StrategySelectionResult | None = None
    delivery_result: DeliveryPreparationResult | None = None
    outcome: CoordinatorCycleOutcome
    intervention_plan: InterventionPlan | None = None


class SessionCoordinator:
    """Connect all four modules into one continuous intervention pipeline.

    The coordinator wraps a StateTrajectoryController (Module 2), a
    StrategySelector (Module 3), a DeliveryCoordinator (Module 4), and a
    TurnBoundaryDetector that determines natural turn boundaries from
    audio energy.  Audio chunks are fed via :meth:`ingest_audio_chunk`;
    each call to :meth:`process` queries the detector to decide whether
    it is safe to intervene.
    """

    def __init__(
        self,
        intervention_policy: InterventionPolicy,
        strategy_library: StrategyLibraryConfig,
        delivery_policy: DeliveryPolicy,
    ) -> None:
        self._controller = StateTrajectoryController(intervention_policy)
        self._selector = StrategySelector(strategy_library)
        self._delivery = DeliveryCoordinator(delivery_policy)
        self._boundary_detector = TurnBoundaryDetector()
        self._previous_plan: InterventionPlan | None = None
        self._cycle_count: int = 0

    @property
    def session_id(self) -> str | None:
        return self._controller.session_id

    @property
    def awaiting_post_intervention_response(self) -> bool:
        return self._controller.awaiting_post_intervention_response

    @property
    def previous_plan(self) -> InterventionPlan | None:
        return self._previous_plan

    @property
    def at_turn_boundary(self) -> bool:
        """Whether the audio detector currently reports a natural turn boundary."""
        return self._boundary_detector.is_at_boundary()

    def ingest_audio_chunk(self, pcm_chunk: bytes) -> bool:
        """Feed one PCM audio chunk to the turn boundary detector.

        Args:
            pcm_chunk: Raw PCM bytes (s16, mono, 16 kHz, 100 ms).

        Returns:
            True if the detector now reports a natural turn boundary.
        """
        return self._boundary_detector.ingest_chunk(pcm_chunk)

    def snapshot(self) -> StateTrajectorySnapshot:
        """Return the current state trajectory snapshot from Module 2."""
        return self._controller.snapshot()

    def process(
        self,
        *,
        assessment: StateAssessment,
        interaction_history_available: bool,
        delivery_runtime: DeliveryRuntimeContext,
    ) -> CoordinatorCycleResult:
        """Run one complete coordinator cycle.

        Steps:
        1. Query the :class:`TurnBoundaryDetector` for the current
           natural turn boundary state.
        2. Determine ``post_intervention_response_observed`` by checking
           whether the controller is awaiting a response and Module 1
           has supplied a sufficient assessment with a valid state.
        3. Build a :class:`ControlObservation` and feed it to Module 2.
        4. If Module 2 authorizes intervention, call Module 3 with the
           assessment, decision, and the previous plan (for do-not-repeat).
        5. If Module 3 returns a ready plan, call Module 4 to prepare
           delivery and update ``previous_plan``.
        """
        self._cycle_count += 1
        sequence = self._cycle_count

        natural_turn_boundary = self._boundary_detector.is_at_boundary()
        post_response = self._should_observe_response(assessment)

        observation = ControlObservation(
            assessment=assessment,
            natural_turn_boundary=natural_turn_boundary,
            post_intervention_response_observed=post_response,
            interaction_history_available=interaction_history_available,
        )

        decision = self._controller.ingest(observation)

        if not decision.strategy_selection_required or not decision.intervention_permitted:
            return CoordinatorCycleResult(
                session_id=assessment.session_id,
                sequence=sequence,
                assessment=assessment,
                decision=decision,
                post_intervention_response_observed=post_response,
                natural_turn_boundary=natural_turn_boundary,
                outcome=CoordinatorCycleOutcome.NO_INTERVENTION,
            )

        selection_result = self._selector.select(
            assessment=assessment,
            decision=decision,
            previous_plan=self._previous_plan,
        )

        if selection_result.plan is None:
            return CoordinatorCycleResult(
                session_id=assessment.session_id,
                sequence=sequence,
                assessment=assessment,
                decision=decision,
                post_intervention_response_observed=post_response,
                natural_turn_boundary=natural_turn_boundary,
                selection_result=selection_result,
                outcome=CoordinatorCycleOutcome.STRATEGY_HELD,
            )

        plan = selection_result.plan
        delivery_result = self._delivery.prepare(
            plan=plan,
            runtime=delivery_runtime,
        )

        delivered = delivery_result.status in {
            DeliveryPreparationStatus.READY,
            DeliveryPreparationStatus.DEGRADED,
        }

        if delivered:
            self._previous_plan = plan

        return CoordinatorCycleResult(
            session_id=assessment.session_id,
            sequence=sequence,
            assessment=assessment,
            decision=decision,
            post_intervention_response_observed=post_response,
            natural_turn_boundary=natural_turn_boundary,
            selection_result=selection_result,
            delivery_result=delivery_result,
            outcome=(
                CoordinatorCycleOutcome.INTERVENTION_DELIVERED
                if delivered
                else CoordinatorCycleOutcome.INTERVENTION_HELD
            ),
            intervention_plan=plan,
        )

    def _should_observe_response(self, assessment: StateAssessment) -> bool:
        """Determine whether a post-intervention response has been observed.

        When the controller is awaiting a post-intervention response and
        Module 1 provides a sufficient assessment with a valid state, the
        response is considered observed.  This lets the controller compare
        the current state against the intervention-baseline state and
        compute a :class:`RecoveryStatus`.  Insufficient assessments do
        not clear the wait; the controller's timeout mechanism handles
        prolonged insufficient evidence.
        """
        if not self._controller.awaiting_post_intervention_response:
            return False
        if assessment.evidence_sufficiency is EvidenceSufficiency.INSUFFICIENT:
            return False
        if assessment.state is None:
            return False
        return True


def create_coordinator() -> SessionCoordinator:
    """Create a coordinator from default configuration files."""
    return SessionCoordinator(
        intervention_policy=load_intervention_policy(),
        strategy_library=load_strategy_library(),
        delivery_policy=load_delivery_policy(),
    )
