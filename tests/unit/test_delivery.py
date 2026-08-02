from __future__ import annotations

import json
from pathlib import Path

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.delivery import (
    ChannelExecution,
    DeliveryCoordinator,
    DeliveryPreparationStatus,
    DeliveryRuntimeContext,
    OutputExecutionStatus,
    OutputModality,
    load_delivery_policy,
    not_attempted_voice_execution,
    render_delivery_preview,
)
from coregulation_poc.delivery_test import run_delivery_test
from coregulation_poc.intervention import StrategySelector, load_strategy_library
from coregulation_poc.intervention.models import InterventionPlan
from coregulation_poc.models import TrajectoryReplayRequest
from coregulation_poc.paths import PROJECT_ROOT
from coregulation_poc.providers.qwen_tts_realtime import SpeechSynthesisResult
from coregulation_poc.settings import Settings


def _authorized_plan() -> InterventionPlan:
    request = TrajectoryReplayRequest.model_validate_json(
        (PROJECT_ROOT / "examples" / "strategy_replay.json").read_text(encoding="utf-8")
    )
    controller = StateTrajectoryController(load_intervention_policy())
    selector = StrategySelector(load_strategy_library())
    plan = None
    for observation in request.observations:
        decision = controller.ingest(observation)
        selection = selector.select(
            assessment=observation.assessment,
            decision=decision,
            previous_plan=plan,
        )
        if selection.plan is not None:
            plan = selection.plan
    assert plan is not None
    return plan


def test_policy_requires_prominent_text_and_autoplay_voice() -> None:
    policy = load_delivery_policy()

    assert policy.visual.prominence == "high"
    assert policy.visual.blocks_primary_task is False
    assert policy.voice.autoplay is True
    assert policy.voice.model == "qwen3-tts-instruct-flash-realtime-2026-01-22"
    assert policy.voice.voice == "Maia"
    assert policy.voice.optimize_instructions is False


def test_authorized_plan_becomes_identical_dual_channel_package() -> None:
    plan = _authorized_plan()
    coordinator = DeliveryCoordinator(load_delivery_policy())

    result = coordinator.prepare(
        plan=plan,
        runtime=DeliveryRuntimeContext(prepared_at_ms=plan.planned_at_ms),
    )

    assert result.status is DeliveryPreparationStatus.READY
    assert result.package is not None
    assert result.package.target_actor == plan.target_actor
    assert result.package.visual_prompt.message == plan.message
    assert result.package.voice_prompt.message == plan.message
    assert result.package.visual_prompt.prominence == "high"
    assert result.package.voice_prompt.enabled is True
    assert result.package.voice_prompt.autoplay is True
    assert result.package.core_content_identical is True


def test_intervention_pause_holds_both_outputs() -> None:
    plan = _authorized_plan()

    result = DeliveryCoordinator(load_delivery_policy()).prepare(
        plan=plan,
        runtime=DeliveryRuntimeContext(
            prepared_at_ms=plan.planned_at_ms,
            interventions_paused=True,
        ),
    )

    assert result.status is DeliveryPreparationStatus.HELD
    assert result.hold_reason == "interventions_paused"
    assert result.package is None


def test_voice_failure_preserves_visual_fallback() -> None:
    plan = _authorized_plan()

    result = DeliveryCoordinator(load_delivery_policy()).prepare(
        plan=plan,
        runtime=DeliveryRuntimeContext(
            prepared_at_ms=plan.planned_at_ms,
            voice_available=False,
        ),
    )

    assert result.status is DeliveryPreparationStatus.DEGRADED
    assert result.package is not None
    assert result.package.visual_prompt.message == plan.message
    assert result.package.voice_prompt.enabled is False
    assert result.package.fallback_reason == "voice_output_unavailable"


def test_execution_report_keeps_channel_outcomes_and_interpretation_separate() -> None:
    plan = _authorized_plan()
    coordinator = DeliveryCoordinator(load_delivery_policy())
    preparation = coordinator.prepare(
        plan=plan,
        runtime=DeliveryRuntimeContext(prepared_at_ms=plan.planned_at_ms),
    )
    assert preparation.package is not None

    report = coordinator.record_execution(
        package=preparation.package,
        recorded_at_ms=2200,
        visual=ChannelExecution(
            modality=OutputModality.VISUAL_TEXT,
            status=OutputExecutionStatus.DELIVERED,
            started_at_ms=2000,
            provider="frontend_overlay",
        ),
        voice=ChannelExecution(
            modality=OutputModality.SPOKEN_VOICE,
            status=OutputExecutionStatus.FAILED,
            started_at_ms=2020,
            provider="alibaba_qwen_realtime_tts",
            error="voice unavailable",
        ),
    )

    assert report.overall_status == "partial"
    assert report.start_offset_ms == 20
    assert report.recipient_response_observed is None
    assert "不能据此推断" in report.interpretation_limit


def test_degraded_package_can_record_voice_not_attempted() -> None:
    voice = not_attempted_voice_execution()

    assert voice.modality is OutputModality.SPOKEN_VOICE
    assert voice.status is OutputExecutionStatus.NOT_ATTEMPTED


def test_browser_preview_uses_saved_maia_audio_not_system_speech() -> None:
    plan = _authorized_plan()
    preparation = DeliveryCoordinator(load_delivery_policy()).prepare(
        plan=plan,
        runtime=DeliveryRuntimeContext(prepared_at_ms=plan.planned_at_ms),
    )
    assert preparation.package is not None

    preview = render_delivery_preview(
        preparation.package,
        audio_filename="delivery_audio_001.wav",
    )

    assert plan.message in preview
    assert "new Audio(payload.audioUrl)" in preview
    assert "SpeechSynthesisUtterance" not in preview
    assert '\"voice\": \"Maia\"' in preview
    assert '\"audioUrl\": \"delivery_audio_001.wav\"' in preview
    assert "再次播报" in preview


def test_delivery_replay_writes_packages_and_preview(tmp_path: Path) -> None:
    input_path = PROJECT_ROOT / "examples" / "strategy_replay.json"

    run_dir, valid = run_delivery_test(
        input_path=input_path,
        settings=Settings(output_dir=tmp_path / "output"),
    )

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    packages = json.loads(
        (run_dir / "delivery_packages.json").read_text(encoding="utf-8")
    )
    assert valid is True
    assert result["delivery_package_count"] == 1
    assert result["degraded_delivery_count"] == 0
    assert packages[0]["visual_prompt"]["message"] == packages[0]["voice_prompt"]["message"]
    assert (run_dir / result["preview_files"][0]).exists()
    assert result["requested_voice_synthesis"] is False
    assert result["audio_files"] == []


class _FakeSynthesizer:
    def synthesize(self, text: str) -> SpeechSynthesisResult:
        return SpeechSynthesisResult(
            pcm_audio=b"\x00\x00" * 2400,
            model="qwen3-tts-instruct-flash-realtime-2026-01-22",
            voice="Maia",
            sample_rate_hz=24000,
            text=text,
            session_id="session-test",
            response_id="response-test",
            usage_characters=len(text),
            first_audio_latency_ms=120,
            total_latency_ms=240,
            events=[{"type": "response.done"}],
        )


def test_delivery_replay_saves_traceable_maia_wav(tmp_path: Path) -> None:
    run_dir, valid = run_delivery_test(
        input_path=PROJECT_ROOT / "examples" / "strategy_replay.json",
        settings=Settings(output_dir=tmp_path / "output"),
        synthesize_voice=True,
        synthesizer=_FakeSynthesizer(),
    )

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    synthesis = json.loads(
        (run_dir / "voice_synthesis_results.json").read_text(encoding="utf-8")
    )[0]
    audio_path = run_dir / result["audio_files"][0]
    preview = (run_dir / result["preview_files"][0]).read_text(encoding="utf-8")

    assert valid is True
    assert result["voice_synthesis_count"] == 1
    assert synthesis["model"] == "qwen3-tts-instruct-flash-realtime-2026-01-22"
    assert synthesis["voice"] == "Maia"
    assert synthesis["source_message_sha256"]
    assert synthesis["audio_sha256"]
    assert audio_path.read_bytes().startswith(b"RIFF")
    assert audio_path.name in preview
