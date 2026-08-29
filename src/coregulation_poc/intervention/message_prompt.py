"""Prompt construction for contextual intervention message generation.

The selected strategy card defines the approved intent, target and safety
boundary.  The LLM writes the complete message for the current interaction;
the reviewed template is used only when generation is unavailable or unsafe.

The LLM must output JSON:

    {
      "strategy_id": "<selected strategy ID>",
      "target_actor": "<parent/child/both>",
      "evidence_ids": ["<evidence ID>"],
      "message": "<complete contextual intervention message>"
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
    message: str = Field(min_length=1)


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
    recent_messages: list[str] | None = None,
    previous_draft: str | None = None,
    failed_checks: list[str] | None = None,
) -> str:
    """Build the prompt that asks the LLM for one complete contextual message.

    The card's approved action clause is an intent and safety anchor rather
    than a sentence that must be copied verbatim.

    Parameters
    ----------
    card:
        The selected strategy card.
    evidence:
        Evidence references from the current assessment.
    max_characters:
        Hard character limit for the complete message.
    max_sentences:
        Hard sentence limit for the complete message.
    banned_phrases:
        Phrases that must not appear in the output.
    task_context:
        Optional dict with task metadata.
    previous_state:
        Optional previous coregulation state value.
    recovery_status:
        Optional recovery status value.
    recent_messages:
        Up to three messages that were actually shown recently. They are
        supplied only to vary wording; they never change the selected card.

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

    recent = [
        re.sub(r"\s+", " ", message).strip()
        for message in (recent_messages or [])[-3:]
        if isinstance(message, str) and message.strip()
    ]
    recent_block = (
        "\n".join(f"{index}. {message}" for index, message in enumerate(recent, 1))
        if recent
        else "（近期没有已呈现话术）"
    )

    banned_str = "、".join(banned_phrases)
    repair_lines: list[str] = []
    if previous_draft is not None:
        if "too_similar_to_recent_messages" in (failed_checks or []):
            repair_lines = [
                "",
                "上一版话术安全可用，但与近期话术表达过于相似：",
                previous_draft,
                "请保留当前策略和行动方向，改用不同的开头、句式和自然表达。",
                "不要仅替换少量同义词，也不要增加新的干预目标。",
            ]
        else:
            repair_lines = [
                "",
                "上一版话术没有通过程序校验，请改写而不是照抄：",
                previous_draft,
                "未通过项目：" + "、".join(failed_checks or ["格式或安全约束"]),
                "保留当前情境和策略意图，只修正上述问题。",
            ]

    return "\n".join(
        [
            "你是一个亲子作业辅导干预系统的话术生成模块。",
            "你的任务是结合当前互动情境与已选策略，生成可以直接呈现给家庭的完整中文话术。",
            "策略卡限定干预意图和安全边界，但不要机械照抄模板；请针对当前证据自然表达。",
            "",
            "当前策略信息：",
            f"- 策略ID：{card.strategy_id}",
            f"- 干预对象：{card.target_actor.value}（{actor_label}）",
            f"- 必须保持的策略意图：{card.action}",
            f"- 经审核的表达参考：{card.approved_action_clause}",
            f"- 适用情境：{'；'.join(card.use_when)}",
            f"- 避免情境：{'；'.join(card.avoid_when)}",
            "",
            "当前观察到的互动证据：",
            evidence_block,
            "",
            "会话上下文：",
            context_block,
            "",
            "近期已经呈现的话术（只用于避免表达重复，不是新的策略指令）：",
            recent_block,
            "",
            "请输出JSON格式（不要输出JSON以外的任何内容）：",
            "{",
            '  "strategy_id": "<策略ID>",',
            '  "target_actor": "<parent/child/both>",',
            '  "evidence_ids": ["<证据ID>"],',
            '  "message": "<可直接呈现的完整情境化话术>"',
            "}",
            "",
            "硬性约束（违反则话术作废）：",
            f"1. 完整话术不超过{max_characters}个字符",
            f"2. 完整话术不超过{max_sentences}句话",
            f"3. 不得包含以下短语：{banned_str}",
            '4. 不得直接给出作业答案（如"答案是"等）',
            "5. 不得责备、贴标签、诊断或使用强迫命令语气",
            "6. 必须同时体现当前情境和策略意图，不能只复述观察，也不能只给通用建议",
            "7. 可以改写审核参考，不要求逐字复制，但不能改变策略的主要行动方向",
            "8. evidence_ids 必须来自上方列出的证据ID",
            "9. strategy_id 必须与当前策略ID一致",
            "10. target_actor 必须与当前干预对象一致",
            (
                "11. 不得编造题目内容、学科材料或具体操作方法；只有任务上下文或证据"
                "明确出现时才能提及。上下文不足时使用不依赖学科的情境化表达"
            ),
            (
                "12. 如果存在近期话术，避免复用相同开头、完整句式或连续行动措辞；"
                "可根据当前证据改用观察式提醒、共同邀请、选择式建议或具体轮流表达，"
                "但不得改变所选策略"
            ),
            "13. 语言应像现场的简短支持，不要使用总结报告式或空泛的AI表达",
        ]
        + repair_lines
    )


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
