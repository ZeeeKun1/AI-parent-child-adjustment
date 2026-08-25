"""Prompt construction for LLM-based intervention message generation.

The LLM generates only an ``observation_clause`` -- a short situational
description of what was observed in the current evidence.  The code then
assembles the final message by combining:

    [address_word], [observation_clause], [approved_action_clause]

The ``approved_action_clause`` comes directly from the strategy card and
cannot be modified by the LLM.  This keeps the strategy intent fixed while
allowing the observation to adapt to the current context.

The LLM must output JSON:

    {
      "strategy_id": "<selected strategy ID>",
      "target_actor": "<parent/child/both>",
      "evidence_ids": ["<evidence ID>"],
      "observation_clause": "<<= 30 char situational description>"
    }
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from coregulation_poc.intervention.models import StrategyCard
from coregulation_poc.models import EvidenceReference


class LLMMessageResult(BaseModel):
    """Structured result returned by :class:`MessageGenerator`."""

    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(min_length=1)
    target_actor: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    observation_clause: str = Field(min_length=1)


def _actor_label(target_actor: str) -> str:
    """Map target_actor enum value to a human-readable label for the prompt."""
    return {
        "parent": "家长",
        "child": "小朋友",
        "both": "你们",
    }.get(target_actor, target_actor)


def build_message_prompt(
    *,
    card: StrategyCard,
    evidence: list[EvidenceReference],
    max_characters: int,
    max_sentences: int,
    banned_phrases: list[str],
    task_context: dict[str, Any] | None = None,
    previous_state: str | None = None,
    recovery_status: str | None = None,
) -> str:
    """Build the prompt that asks the LLM to generate an observation clause.

    The LLM receives the strategy card's identity, target actor, and fixed
    action clause so it knows what to describe.  It must NOT change the
    action clause -- it only generates the observation.

    Parameters
    ----------
    card:
        The selected strategy card.
    evidence:
        Evidence references from the current assessment.
    max_characters:
        Hard character limit for the *assembled* message.
    max_sentences:
        Hard sentence limit for the *assembled* message.
    banned_phrases:
        Phrases that must not appear in the output.
    task_context:
        Optional dict with task metadata.
    previous_state:
        Optional previous coregulation state value.
    recovery_status:
        Optional recovery status value.

    Returns
    -------
    str
        The full prompt string ready to send to the LLM.
    """
    actor_label = _actor_label(card.target_actor.value)

    evidence_lines: list[str] = []
    for idx, item in enumerate(evidence[:6]):
        parts = [f"[evidence_{idx}]", f"[{item.modality.value}]"]
        if item.actor.value != "unknown":
            parts.append(f"对象={item.actor.value}")
        parts.append(f"观察={item.observation}")
        if item.quote:
            parts.append(f'原话="{item.quote}"')
        evidence_lines.append(" ".join(parts))
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "（无具体证据片段）"

    context_lines: list[str] = []
    if task_context:
        context_lines.append(f"任务名称：{task_context.get('task_name', '未知')}")
        context_lines.append(f"任务难度：{task_context.get('task_difficulty', '未知')}")
        context_lines.append(f"儿童年级：{task_context.get('child_grade', '未知')}")
    if previous_state:
        context_lines.append(f"前一个状态：{previous_state}")
    if recovery_status:
        context_lines.append(f"恢复状态：{recovery_status}")
    context_block = "\n".join(context_lines) if context_lines else "（无额外上下文）"

    banned_str = "、".join(banned_phrases)

    return "\n".join([
        "你是一个亲子作业辅导干预系统的话术生成模块。",
        "你的任务是根据当前观察到的证据，生成一句简短的情境观察描述（observation_clause）。",
        "你不需要生成建议动作，建议动作由系统固定提供。",
        "",
        "当前策略信息：",
        f"- 策略ID：{card.strategy_id}",
        f"- 干预对象：{card.target_actor.value}（{actor_label}）",
        f"- 固定建议动作：{card.approved_action_clause}",
        "",
        "当前观察到的互动证据：",
        evidence_block,
        "",
        "会话上下文：",
        context_block,
        "",
        "请输出JSON格式（不要输出JSON以外的任何内容）：",
        "{",
        '  "strategy_id": "<策略ID>",',
        '  "target_actor": "<parent/child/both>",',
        '  "evidence_ids": ["<证据ID>"],',
        '  "observation_clause": "<不超过30字的情境描述>"',
        "}",
        "",
        "硬性约束（违反则话术作废）：",
        f"1. observation_clause 不超过30个字",
        f"2. 最终话术（地址词+观察描述+建议动作）不超过{max_characters}个字符",
        f"3. 最终话术不超过{max_sentences}句话",
        f"4. 不得包含以下短语：{banned_str}",
        '5. 不得直接给出答案（如"答案是"等）',
        "6. 不得使用责备、命令或诊断性语气",
        "7. observation_clause 只描述观察到的情境，不要包含建议动作",
        "8. evidence_ids 必须来自上方列出的证据ID",
        "9. strategy_id 必须与当前策略ID一致",
        "10. target_actor 必须与当前干预对象一致",
    ])


def parse_llm_response(text: str) -> LLMMessageResult:
    """Parse the LLM's JSON response into a :class:`LLMMessageResult`.

    Strips markdown code fences if present, then validates the JSON
    structure against the expected schema.

    Raises
    ------
    ValueError
        If the text is not valid JSON or does not match the schema.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
    try:
        return LLMMessageResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM response does not match expected schema: {exc}") from exc
