from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.intervention import StrategySelector, load_strategy_library
from coregulation_poc.intervention.models import (
    InterventionPlan,
    StrategySelectionStatus,
)
from coregulation_poc.models import TrajectoryReplayRequest
from coregulation_poc.paths import INTERVENTION_POLICY_PATH, STRATEGY_CARDS_PATH
from coregulation_poc.settings import Settings
from coregulation_poc.storage.run_artifacts import RunArtifactStore, sha256_file


def _read_request(input_path: Path) -> TrajectoryReplayRequest:
    try:
        payload: Any = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"strategy input is not valid JSON: {exc}") from exc
    return TrajectoryReplayRequest.model_validate(payload)


def run_strategy_test(
    *,
    input_path: Path,
    settings: Settings,
) -> tuple[Path, bool]:
    """Replay module-one observations through modules two and three without an API call."""
    resolved_input = input_path.expanduser().resolve()
    request = _read_request(resolved_input)
    policy = load_intervention_policy()
    library = load_strategy_library()
    controller = StateTrajectoryController(policy)
    selector = StrategySelector(library)
    store = RunArtifactStore(settings.output_dir, request.session_id)

    store.write_json(
        "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "session_id": request.session_id,
            "mode": "strategy_replay",
            "source": {
                "filename": resolved_input.name,
                "sha256": sha256_file(resolved_input),
            },
            "research_basis": {
                "intervention_policy_version": policy.version,
                "intervention_policy_sha256": sha256_file(INTERVENTION_POLICY_PATH),
                "strategy_library_version": library.version,
                "strategy_library_source": library.source,
                "strategy_library_sha256": sha256_file(STRATEGY_CARDS_PATH),
            },
        },
    )
    store.write_json("intervention_policy.json", policy.model_dump(mode="json"))
    store.write_json("strategy_library.json", library.model_dump(mode="json"))
    store.write_json("observations.json", request.model_dump(mode="json"))

    decisions = []
    selection_results = []
    plans: list[InterventionPlan] = []
    previous_plan: InterventionPlan | None = None
    for observation in request.observations:
        decision = controller.ingest(observation)
        result = selector.select(
            assessment=observation.assessment,
            decision=decision,
            previous_plan=previous_plan,
        )
        decisions.append(decision.model_dump(mode="json"))
        selection_results.append(result.model_dump(mode="json"))
        store.append_event(
            {
                "direction": "controller",
                "type": "intervention.decision",
                "sequence": decision.sequence,
                "state": decision.current_state,
                "action": decision.action,
                "recovery_status": decision.recovery_status,
            }
        )
        if result.status is StrategySelectionStatus.READY and result.plan is not None:
            previous_plan = result.plan
            plans.append(result.plan)
            store.append_event(
                {
                    "direction": "strategy_selector",
                    "type": "intervention.plan",
                    "sequence": result.plan.sequence,
                    "strategy_id": result.plan.strategy_id,
                    "target_actor": result.plan.target_actor,
                    "message_source": result.plan.message_source,
                }
            )
        else:
            store.append_event(
                {
                    "direction": "strategy_selector",
                    "type": "intervention.plan_held",
                    "sequence": decision.sequence,
                    "reason": result.hold_reason,
                }
            )

    snapshot = controller.snapshot()
    store.write_json("decisions.json", decisions)
    store.write_json("strategy_selections.json", selection_results)
    store.write_json(
        "intervention_plans.json",
        [plan.model_dump(mode="json") for plan in plans],
    )
    store.write_json("state_trajectory.json", snapshot.model_dump(mode="json"))
    store.write_json(
        "result.json",
        {
            "valid": True,
            "observation_count": len(request.observations),
            "decision_count": len(decisions),
            "intervention_plan_count": len(plans),
            "final_state": snapshot.points[-1].state,
            "final_action": snapshot.decisions[-1].action,
            "awaiting_post_intervention_response": (
                snapshot.awaiting_post_intervention_response
            ),
        },
    )
    return store.run_dir, True
