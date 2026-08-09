"""Prompt construction for LLM-based intervention message generation.

The prompt asks the LLM to rephrase a strategy card's approved template
into a context-adaptive variant while preserving the strategy intent.
The LLM receives:

- The strategy card's intent (action, target_actor, repair_target)
- The approved template as a reference wording
- The current interaction evidence (quotes and observations)
- Hard constraints (character limit, sentence limit, banned phrases)

The LLM must output only the rephrased message, with no explanation.
"""

from __future__ import annotations

from coregulation_poc.intervention.models import StrategyCard
from coregulation_poc.models import EvidenceReference


def build_message_prompt(
    *,
    card: StrategyCard,
    evidence: list[EvidenceReference],
    max_characters: int,
    max_sentences: int,
    banned_phrases: list[str],
) -> str:
    """Build the system+user prompt for LLM message rephrasing.

    Parameters
    ----------
    card:
        The selected strategy card providing intent and approved template.
    evidence:
        Evidence references from the current assessment, giving context.
    max_characters:
        Hard character limit for the output message.
    max_sentences:
        Hard sentence limit for the output message.
    banned_phrases:
        Phrases that must not appear in the output.

    Returns
    -------
    str
        The full prompt string ready to send to the LLM.
    """
    actor_label = {
        "parent": "家长",
        "child": "儿童",
        "both": "亲子双方",
    }.get(card.target_actor.value, card.target_actor.value)

    evidence_lines: list[str] = []
    for item in evidence[:4]:
        parts = [f"[{item.modality.value}]"]
        if item.actor.value != "unknown":
            parts.append(f"对象={item.actor.value}")
        parts.append(f"观察={item.observation}")
        if item.quote:
            parts.append(f'原话="{item.quote}"')
        evidence_lines.append(" ".join(parts))
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "（无具体证据片段）"

    banned_str = "、".join(banned_phrases)

    return "\n".join([
        "你是一个亲子作业辅导干预系统的话术生成模块。",
        f"当前策略意图：{card.action}",
        f"干预对象：{actor_label}",
        f"修复目标：{card.repair_target.value}",
        f"参考话术（请据此改写，不要照搬）：{card.approved_template}",
        "",
        "当前观察到的互动证据：",
        evidence_block,
        "",
        "硬性约束（违反则话术作废）：",
        f"1. 不超过{max_characters}个字符",
        f"2. 不超过{max_sentences}句话",
        f"3. 不得包含以下短语：{banned_str}",
        "4. 不得直接给出答案",
        "5. 不得使用责备、命令或评判性语气",
        "6. 保持与参考话术相同的策略意图，只调整措辞以适应当前情境",
        "",
        "请直接输出改写后的话术，不要包含任何解释、标号或引号。",
    ])
