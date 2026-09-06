from __future__ import annotations

import asyncio
from types import SimpleNamespace

from coregulation_poc.acoustics.speaker_binding import SpeakerBinding
from coregulation_poc.capture.media import MediaChunk, MediaKind
from coregulation_poc.intervention.baseline import (
    BASELINE_SYSTEM_PROMPT,
    BaselineAction,
    BaselineDecisionGenerator,
)
from coregulation_poc.models import AcousticFeatures, PerceptionReport, SpeechTurn
from coregulation_poc.runtime.recognition import WindowObservation
from coregulation_poc.runtime.session import RealtimeLoopConfig, RealtimeSession
from coregulation_poc.runtime.window import MediaWindow


class _Provider:
    model = "baseline-test-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def generate_structured(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            text=self.responses.pop(0),
            model=self.model,
            prompt_tokens=20,
            completion_tokens=10,
            total_latency_ms=12,
        )


def _observation(end_ms: int = 10_000) -> WindowObservation:
    window = MediaWindow(
        chunks=(
            MediaChunk(MediaKind.AUDIO, 0, b"\x00\x00" * 1600),
            MediaChunk(MediaKind.IMAGE, end_ms, b"jpeg"),
        ),
        start_ms=0,
        end_ms=end_ms,
    )
    return WindowObservation(
        session_id="baseline-test",
        window=window,
        perception_report=PerceptionReport(
            speech_turns=[
                SpeechTurn(
                    speaker="parent",
                    start_ms=100,
                    end_ms=800,
                    text="我们先停一下，看看哪里不会。",
                )
            ]
        ),
        acoustic_features=AcousticFeatures(total_speech_ms=700),
        speaker_binding=SpeakerBinding(bound=False),
    )


def test_baseline_prompt_has_no_research_state_or_strategy_framework() -> None:
    lowered = BASELINE_SYSTEM_PROMPT.lower()
    for forbidden in ("high_risk", "dysregulation", "fluctuation", "codebook", "strategy"):
        assert forbidden not in lowered


def test_baseline_generator_returns_direct_intervention() -> None:
    provider = _Provider(
        [
            '{"action":"intervene","target_actor":"both",'
            '"message":"先停一下，孩子说说卡在哪里，家长听完再回应。",'
            '"reason":"交流暂时受阻","confidence":"high"}'
        ]
    )
    decision = BaselineDecisionGenerator(provider).generate(
        observation=_observation(),
        sequence=1,
        task_context={"task_name": "数学练习"},
        recent_messages=[],
    )

    assert decision.action is BaselineAction.INTERVENE
    assert decision.target_actor.value == "both"
    assert decision.message


class _ObservationOnlyRecognizer:
    def __init__(self) -> None:
        self.api_call_count = 0
        self.judge_called = False

    async def observe(self, *, session_id: str, window: MediaWindow) -> WindowObservation:
        self.api_call_count += 1
        item = _observation(window.end_ms)
        return WindowObservation(
            session_id=session_id,
            window=window,
            perception_report=item.perception_report,
            acoustic_features=item.acoustic_features,
            speaker_binding=item.speaker_binding,
        )

    async def judge(self, **_: object) -> object:
        self.judge_called = True
        raise AssertionError("baseline must not call state judgment")

    async def assess(self, **_: object) -> object:
        raise AssertionError("baseline must use observation only")


def test_baseline_runtime_skips_state_and_strategy_modules() -> None:
    asyncio.run(_exercise_baseline_runtime())


async def _exercise_baseline_runtime() -> None:
    events: list[dict[str, object]] = []
    recognizer = _ObservationOnlyRecognizer()
    provider = _Provider(
        [
            '{"action":"intervene","target_actor":"both",'
            '"message":"先轮流说一句，再一起继续。",'
            '"reason":"双方同时说话","confidence":"medium"}'
        ]
    )

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="baseline-runtime",
        recognizer=recognizer,
        send_event=send_event,
        config=RealtimeLoopConfig(assessment_interval_ms=100_000),
        text_chat_provider=provider,
    )
    session.set_experiment_condition("baseline")
    await session.start()
    await session.accept_chunk(MediaChunk(MediaKind.AUDIO, 0, b"\x00\x00" * 1600))
    await session.accept_chunk(MediaChunk(MediaKind.IMAGE, 10_000, b"jpeg"))
    await session.analyze_now()
    await session.stop("completed")

    event_types = [event["type"] for event in events]
    assert "baseline_decision" in event_types
    assert "intervention" in event_types
    assert "state_update" not in event_types
    assert recognizer.judge_called is False
    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["strategy_id"] is None
    assert intervention["repair_target"] is None
