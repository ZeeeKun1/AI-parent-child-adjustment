from __future__ import annotations

import json

from coregulation_poc.fusion.judgment import parse_judgment_result


def test_audio_null_observation_reuses_verbatim_quote() -> None:
    payload = {
        "session_id": "S01",
        "assessed_at_ms": 10000,
        "state": "normal",
        "previous_state": None,
        "trajectory": "unclear",
        "evidence_sufficiency": "sufficient",
        "confidence": "high",
        "alternative_state": None,
        "ambiguity_reason": None,
        "interaction_performance": ["normal task progression"],
        "task_process": "smooth_progress",
        "support_need": "none",
        "support_target": "unknown",
        "interruptibility": "natural_pause",
        "boundary_signals": {
            "task_stall_observed": False,
            "parental_prompt_count": None,
            "conflict_action_observed": False,
            "child_disengaged_observed": False,
            "regulation_balance": "unclear",
        },
        "modality_evidence": {
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": "unknown",
                        "start_ms": 1000,
                        "end_ms": 2000,
                        "code": "task talk",
                        "observation": None,
                        "quote": "我们继续做这道题。",
                        "frame_timestamp_ms": None,
                    }
                ],
                "limitation_reason": None,
            },
            "video": {
                "sufficiency": "insufficient",
                "items": [],
                "limitation_reason": "画面证据不足",
            },
        },
        "reason": "任务正常推进。",
        "limitations": [],
    }

    assessment = parse_judgment_result(json.dumps(payload, ensure_ascii=False))

    assert assessment.modality_evidence.audio.items[0].observation == "我们继续做这道题。"
