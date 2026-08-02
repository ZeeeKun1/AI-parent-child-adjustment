from __future__ import annotations

import json

from coregulation_poc.settings import Settings
from coregulation_poc.trajectory_test import run_trajectory_test


def test_trajectory_test_saves_auditable_outputs(tmp_path) -> None:
    input_path = tmp_path / "trajectory.json"
    output_dir = tmp_path / "output"
    input_path.write_text(
        json.dumps(
            {
                "session_id": "demo",
                "observations": [
                    {
                        "assessment": {
                            "session_id": "demo",
                            "assessed_at_ms": 1000,
                            "state": "fluctuation",
                            "evidence_sufficiency": "sufficient",
                            "confidence": "high",
                            "alternative_state": None,
                            "ambiguity_reason": None,
                            "interaction_performance": ["brief task stall"],
                            "modality_evidence": {
                                "audio": {
                                    "sufficiency": "sufficient",
                                    "items": [
                                        {
                                            "modality": "audio",
                                            "actor": "child",
                                            "start_ms": 500,
                                            "end_ms": 1000,
                                            "code": "difficulty_expression",
                                            "observation": "The child expresses a task difficulty.",
                                            "quote": "我不会",
                                            "frame_timestamp_ms": None,
                                        }
                                    ],
                                    "limitation_reason": None,
                                },
                                "video": {
                                    "sufficiency": "insufficient",
                                    "items": [],
                                    "limitation_reason": "Relevant behavior is not visible.",
                                },
                            },
                            "previous_state": None,
                            "reason": "The difficulty is temporary and recovery remains possible.",
                        },
                        "natural_turn_boundary": True,
                        "post_intervention_response_observed": False,
                        "interaction_history_available": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    run_dir, valid = run_trajectory_test(
        input_path=input_path,
        settings=Settings(output_dir=output_dir),
    )

    assert valid is True
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "intervention_policy.json").exists()
    assert (run_dir / "decisions.json").exists()
    assert (run_dir / "state_trajectory.json").exists()
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["final_state"] == "fluctuation"
    assert result["final_action"] == "observe"
