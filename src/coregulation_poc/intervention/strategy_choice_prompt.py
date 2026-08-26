"""Bounded LLM prompt for choosing among approved intervention cards."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from coregulation_poc.intervention.models import StrategyCard
from coregulation_poc.models import ConfidenceLevel, StateAssessment

RelaxedDimension = Literal["interaction_performance", "support_need", "task_process"]


class LLMStrategyChoice(BaseModel):
    """Auditable strategy choice returned by the bounded selector."""

    model_config = ConfigDict(extra="forbid")

    strategy_id: str | None = None
    confidence: ConfidenceLevel
    reason: str = Field(min_length=1, max_length=500)
    matched_dimensions: list[str] = Field(default_factory=list)
    relaxed_dimensions: list[RelaxedDimension] = Field(default_factory=list)


def build_strategy_choice_prompt(
    *,
    assessment: StateAssessment,
    candidates: list[StrategyCard],
    task_context: dict[str, Any] | None = None,
) -> str:
    """Ask the model to select, or reject, only the supplied approved cards."""

    evidence_lines: list[str] = []
    for index, item in enumerate(assessment.modality_evidence.all_items[:8]):
        parts = [
            f"evidence_{index}",
            f"modality={item.modality.value}",
            f"actor={item.actor.value}",
            f"observation={item.observation}",
        ]
        if item.quote:
            parts.append(f"quote={item.quote}")
        evidence_lines.append(" | ".join(parts))

    candidate_payload = [
        {
            "strategy_id": card.strategy_id,
            "name": card.name,
            "target_actor": card.target_actor.value,
            "repair_target": card.repair_target.value,
            "support_needs": card.support_needs,
            "task_processes": card.task_processes,
            "use_when": card.use_when,
            "avoid_when": card.avoid_when,
            "approved_action": card.approved_action_clause,
        }
        for card in candidates
    ]
    context = task_context or {}

    return "\n".join(
        [
            "You select one intervention strategy from an approved research library.",
            (
                "The program has already authorized intervention and enforced "
                "state, target, and evidence safety constraints."
            ),
            (
                "Select by semantic fit with the current interaction; support_need "
                "and task_process are soft descriptors, not exact lookup keys."
            ),
            (
                "Do not invent, combine, rewrite, or broaden a strategy. Respect "
                "every avoid_when condition."
            ),
            "If none directly addresses the observed repair need, return strategy_id=null.",
            "Return only concise JSON. Do not provide hidden reasoning or commentary.",
            "",
            "CURRENT ASSESSMENT:",
            json.dumps(
                {
                    "state": assessment.state.value if assessment.state else None,
                    "previous_state": (
                        assessment.previous_state.value if assessment.previous_state else None
                    ),
                    "trajectory": assessment.trajectory.value,
                    "interaction_performance": assessment.interaction_performance,
                    "task_process": (
                        assessment.task_process.value if assessment.task_process else None
                    ),
                    "support_need": (
                        assessment.support_need.value if assessment.support_need else None
                    ),
                    "support_target": assessment.support_target.value,
                    "reason": assessment.reason,
                    "task_context": context,
                },
                ensure_ascii=False,
            ),
            "",
            "OBSERVED EVIDENCE:",
            "\n".join(evidence_lines) if evidence_lines else "(none)",
            "",
            "APPROVED CANDIDATES:",
            json.dumps(candidate_payload, ensure_ascii=False),
            "",
            "OUTPUT SCHEMA:",
            '{"strategy_id":"<candidate ID or null>","confidence":"high|medium|low",',
            (
                '"reason":"<observable-fit explanation, <=80 Chinese characters '
                'or <=50 English words>",'
            ),
            '"matched_dimensions":["<dimension>"],'
            '"relaxed_dimensions":["interaction_performance|support_need|task_process"]}',
            (
                "Use high or medium only when the approved action directly "
                "addresses the observed need."
            ),
        ]
    )


def parse_strategy_choice(text: str) -> LLMStrategyChoice:
    """Parse one strict JSON strategy choice."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"strategy choice is not valid JSON: {exc}") from exc
    try:
        return LLMStrategyChoice.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"strategy choice does not match the schema: {exc}") from exc
