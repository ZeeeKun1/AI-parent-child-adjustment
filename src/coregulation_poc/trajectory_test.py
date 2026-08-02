from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.models import TrajectoryReplayRequest
from coregulation_poc.paths import INTERVENTION_POLICY_PATH
from coregulation_poc.settings import Settings
from coregulation_poc.storage.run_artifacts import RunArtifactStore, sha256_file


def _read_request(input_path: Path) -> TrajectoryReplayRequest:
    try:
        payload: Any = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"trajectory input is not valid JSON: {exc}") from exc
    return TrajectoryReplayRequest.model_validate(payload)


def run_trajectory_test(
    *,
    input_path: Path,
    settings: Settings,
) -> tuple[Path, bool]:
    resolved_input = input_path.expanduser().resolve()
    request = _read_request(resolved_input)
    policy = load_intervention_policy()
    controller = StateTrajectoryController(policy)
    store = RunArtifactStore(settings.output_dir, request.session_id)

    store.write_json(
        "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "session_id": request.session_id,
            "mode": "trajectory_replay",
            "source": {
                "filename": resolved_input.name,
                "sha256": sha256_file(resolved_input),
            },
            "research_basis": {
                "intervention_policy_version": policy.version,
                "intervention_policy_source": policy.source,
                "intervention_policy_sha256": sha256_file(INTERVENTION_POLICY_PATH),
            },
        },
    )
    store.write_json("intervention_policy.json", policy.model_dump(mode="json"))
    store.write_json("observations.json", request.model_dump(mode="json"))

    decisions = []
    for observation in request.observations:
        decision = controller.ingest(observation)
        decisions.append(decision.model_dump(mode="json"))
        store.append_event(
            {
                "direction": "controller",
                "type": "intervention.decision",
                "sequence": decision.sequence,
                "assessment_time_ms": decision.decided_at_ms,
                "state": decision.current_state,
                "action": decision.action,
                "reason_code": decision.reason_code,
                "recovery_status": decision.recovery_status,
            }
        )

    snapshot = controller.snapshot()
    store.write_json("decisions.json", decisions)
    store.write_json("state_trajectory.json", snapshot.model_dump(mode="json"))
    store.write_json(
        "result.json",
        {
            "valid": True,
            "observation_count": len(request.observations),
            "decision_count": len(decisions),
            "final_state": snapshot.points[-1].state,
            "final_action": snapshot.decisions[-1].action,
            "awaiting_post_intervention_response": (
                snapshot.awaiting_post_intervention_response
            ),
        },
    )
    return store.run_dir, True
