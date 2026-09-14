"""Prompt construction for contextual intervention message generation.

The selected strategy card defines the non-negotiable action, target and
safety boundary.  The LLM writes the complete message for the current interaction;
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

    The card's action is a non-negotiable instruction.  Its approved action
    clause anchors tone and safety, but does not need to be copied verbatim.

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
                "必须完整保留所选策略的对象、核心动作和边界，只改表达方式。",
                "像现场专家重新说一遍：换切入点、语序和说话节奏，不要只替换同义词。",
                "不要加入第二个行动，也不要为了求变化而说得书面、夸张或绕口。",
            ]
        else:
            repair_lines = [
                "",
                "上一版话术没有通过程序校验，请改写而不是照抄：",
                previous_draft,
                "未通过项目：" + "、".join(failed_checks or ["格式或安全约束"]),
                "所选策略的核心动作不可替换、删减或弱化；只修正上述问题。",
                "修正后再默读一遍，确保听起来像现场专家说的话，而不是系统通知。",
            ]

    return "\n".join(
        [
            "你是一名长期在家庭作业现场工作的亲子沟通专家。",
            "你的话会立刻通过屏幕和语音说给家长和孩子听。",
            "像真人专家在旁边自然开口：语气柔和，表达顺口，也把具体建议说清楚。",
            "专业性体现在行动恰当，不体现在专业术语、正式措辞或完整分析。",
            "你只负责表达，不负责重新判断问题，也不负责更换、补充或组合策略。",
            "所选策略的核心动作是本轮唯一行动指令；语言可以自然改写，行动不能改变。",
            "",
            "当前策略信息：",
            f"- 策略ID：{card.strategy_id}",
            f"- 干预对象：{card.target_actor.value}（{actor_label}）",
            f"- 本轮唯一核心动作（必须完整执行）：{card.action}",
            (
                "- 经审核的表达参考（只参考安全边界，不照抄措辞和句式）："
                f"{card.approved_action_clause}"
            ),
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
            "人类专家的说话方式：",
            "- 先弄清核心动作中的“谁、做什么、做到什么程度”，再把它换成口语。",
            "- 直接对当事人说话，不描述或评价他们。家长用“家长”，孩子用“你”，",
            "  双方用“你们”或“咱们”；不要使用“儿童”“目标对象”“亲子双方”。",
            "- 把证据自然带进话里，不要说“我注意到”“从刚才可以看出”或复述分析过程。",
            "- 围绕本轮核心动作给出一个当下能做的具体建议，别只安慰或泛泛鼓励。",
            "- 建议可以有少量解释和商量的语气，但不列清单，也不混入另一种策略。",
            "- 根据策略选择最自然的说法：一句提醒、一个简单问题、一句邀请，或肯定后接动作。",
            "- 不要每次都使用“先……再……”“我们来……”或相同的对称句式。",
            "- 能短就短，通常20至50个汉字已经够用，不要为了凑长度补充解释。",
            "- 可以少量使用“吧”“一下”“就好”等口语词，但不要故意装可爱或连续安慰。",
            "- 只说能从证据中看到或听到的情况，不猜测情绪、性格、意图或家庭关系。",
            "- 冲突明显时语气保持温和、平稳，把局面接住；状态积极时肯定具体行为。",
            "",
            "口语风格示例（只模仿语气和节奏；示例行动不属于本轮策略，不得照搬）：",
            "- 生硬：根据当前互动情况，建议家长适当调整沟通节奏。",
            "  专家会说：家长稍微慢一点，听孩子把这句说完。",
            "- 生硬：建议儿童明确表达当前需求，以便家长提供支持。",
            "  专家会说：你卡在哪儿了？直接告诉家长就好。",
            "- 生硬：双方目前合作状态良好，请继续保持。",
            "  专家会说：刚才配合得挺好，就照这个节奏来。",
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
            "6. 必须结合当前证据，并准确落实本轮唯一核心动作，不能只复述观察或给通用建议",
            (
                "7. 不得替换、删减、弱化核心动作，也不得增加第二个行动或混入其他策略；"
                "审核参考可以改写，但不能改变谁做什么以及动作边界"
            ),
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
            (
                "13. 不要出现报告式或AI式措辞，例如“检测到”“根据当前状态”"
                "“系统建议”“建议您”“可以考虑”“需要进行”“这有助于”"
                "“促进”“当前互动”“共调节”“干预策略”“给予支持”“保持良好状态”"
            ),
            (
                "14. 不要堆叠“请”“建议”“可以”“需要”等劝告词，不要解释这样做的好处，"
                "也不要以“我理解”“我注意到”开头表演共情"
            ),
            (
                "15. 输出前简单默读一遍，尽量让话术像现场专家说的：自然、柔和、具体；"
                "不要输出检查过程"
            ),
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
