from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.delivery import (
    DeliveryCoordinator,
    DeliveryPreparationStatus,
    DeliveryRuntimeContext,
    load_delivery_policy,
    render_delivery_preview,
)
from coregulation_poc.intervention import StrategySelector, load_strategy_library
from coregulation_poc.intervention.models import InterventionPlan, StrategySelectionStatus
from coregulation_poc.models import TrajectoryReplayRequest
from coregulation_poc.paths import (
    DELIVERY_POLICY_PATH,
    INTERVENTION_POLICY_PATH,
    STRATEGY_CARDS_PATH,
)
from coregulation_poc.providers.qwen_tts_realtime import (
    QwenRealtimeTTSProvider,
    SpeechSynthesisResult,
    write_pcm_wav,
)
from coregulation_poc.settings import Settings
from coregulation_poc.storage.run_artifacts import RunArtifactStore, sha256_file


def _read_request(input_path: Path) -> TrajectoryReplayRequest:
    try:
        payload: Any = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"delivery input is not valid JSON: {exc}") from exc
    return TrajectoryReplayRequest.model_validate(payload)


def run_delivery_test(
    *,
    input_path: Path,
    settings: Settings,
    voice_enabled: bool = True,
    voice_available: bool = True,
    synthesize_voice: bool = False,
    synthesizer: Any | None = None,
) -> tuple[Path, bool]:
    """Replay modules one to four and optionally generate traceable Maia audio."""
    if synthesize_voice and not voice_enabled:
        raise ValueError("cannot synthesize voice when voice output is disabled")
    resolved_input = input_path.expanduser().resolve()
    request = _read_request(resolved_input)
    timing_policy = load_intervention_policy()
    strategy_library = load_strategy_library()
    delivery_policy = load_delivery_policy()
    controller = StateTrajectoryController(timing_policy)
    selector = StrategySelector(strategy_library)
    coordinator = DeliveryCoordinator(delivery_policy)
    store = RunArtifactStore(settings.output_dir, request.session_id)

    store.write_json(
        "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "session_id": request.session_id,
            "mode": "delivery_replay",
            "source": {
                "filename": resolved_input.name,
                "sha256": sha256_file(resolved_input),
            },
            "research_basis": {
                "intervention_policy_version": timing_policy.version,
                "intervention_policy_sha256": sha256_file(INTERVENTION_POLICY_PATH),
                "strategy_library_version": strategy_library.version,
                "strategy_library_sha256": sha256_file(STRATEGY_CARDS_PATH),
                "delivery_policy_version": delivery_policy.version,
                "delivery_policy_source": delivery_policy.source,
                "delivery_policy_sha256": sha256_file(DELIVERY_POLICY_PATH),
            },
            "runtime": {
                "voice_enabled": voice_enabled,
                "voice_available": voice_available,
                "synthesize_voice": synthesize_voice,
                "tts_model": delivery_policy.voice.model,
                "tts_voice": delivery_policy.voice.voice,
            },
        },
    )
    store.write_json("intervention_policy.json", timing_policy.model_dump(mode="json"))
    store.write_json("strategy_library.json", strategy_library.model_dump(mode="json"))
    store.write_json("delivery_policy.json", delivery_policy.model_dump(mode="json"))
    store.write_json("observations.json", request.model_dump(mode="json"))

    decisions = []
    selections = []
    plans: list[InterventionPlan] = []
    preparations = []
    packages = []
    preview_files = []
    audio_files = []
    synthesis_results = []
    synthesis_failed = False
    previous_plan: InterventionPlan | None = None
    for observation in request.observations:
        decision = controller.ingest(observation)
        selection = selector.select(
            assessment=observation.assessment,
            decision=decision,
            previous_plan=previous_plan,
        )
        decisions.append(decision.model_dump(mode="json"))
        selections.append(selection.model_dump(mode="json"))
        store.append_event(
            {
                "direction": "controller",
                "type": "intervention.decision",
                "sequence": decision.sequence,
                "state": decision.current_state,
                "action": decision.action,
            }
        )
        if selection.status is not StrategySelectionStatus.READY or selection.plan is None:
            continue

        previous_plan = selection.plan
        plans.append(selection.plan)
        preparation = coordinator.prepare(
            plan=selection.plan,
            runtime=DeliveryRuntimeContext(
                prepared_at_ms=observation.assessment.assessed_at_ms,
                voice_enabled=voice_enabled,
                voice_available=voice_available,
            ),
        )
        preparations.append(preparation.model_dump(mode="json"))
        if preparation.package is None:
            store.append_event(
                {
                    "direction": "delivery_coordinator",
                    "type": "intervention.delivery_held",
                    "sequence": decision.sequence,
                    "reason": preparation.hold_reason,
                }
            )
            continue

        package = preparation.package
        packages.append(package.model_dump(mode="json"))
        package_number = len(packages)
        audio_name: str | None = None
        if synthesize_voice and package.voice_prompt.enabled:
            active_synthesizer = synthesizer
            if active_synthesizer is None:
                if settings.dashscope_api_key is None:
                    raise ValueError("DASHSCOPE_API_KEY is required for voice synthesis")
                active_synthesizer = QwenRealtimeTTSProvider(
                    model=package.voice_prompt.model,
                    voice=package.voice_prompt.voice,
                    api_key=settings.dashscope_api_key.get_secret_value(),
                    workspace_id=settings.aliyun_workspace_id,
                    base_url=settings.resolved_tts_base_url,
                    language_type=package.voice_prompt.language_type,
                    response_format=package.voice_prompt.response_format,
                    sample_rate_hz=package.voice_prompt.sample_rate_hz,
                    mode=package.voice_prompt.mode,
                    instructions=package.voice_prompt.instructions,
                    optimize_instructions=package.voice_prompt.optimize_instructions,
                    connection_timeout_seconds=settings.connection_timeout_seconds,
                    response_timeout_seconds=settings.response_timeout_seconds,
                )
            try:
                synthesis: SpeechSynthesisResult = active_synthesizer.synthesize(
                    package.voice_prompt.message
                )
                if synthesis.model != package.voice_prompt.model:
                    raise ValueError("TTS result model does not match the approved voice prompt")
                if synthesis.voice != package.voice_prompt.voice:
                    raise ValueError("TTS result voice does not match the approved voice prompt")
                if synthesis.text != package.voice_prompt.message:
                    raise ValueError("TTS result text does not match the approved intervention")
                audio_name = f"delivery_audio_{package_number:03d}.wav"
                audio_path = write_pcm_wav(
                    store.run_dir / audio_name,
                    synthesis.pcm_audio,
                    synthesis.sample_rate_hz,
                )
                event_name = f"tts_events_{package_number:03d}.json"
                store.write_json(event_name, synthesis.events)
                audio_duration_ms = round(
                    len(synthesis.pcm_audio) / (2 * synthesis.sample_rate_hz) * 1000
                )
                synthesis_record = {
                    "sequence": package.sequence,
                    "delivery_id": package.delivery_id,
                    "status": "synthesized",
                    "provider": package.voice_prompt.provider,
                    "model": synthesis.model,
                    "voice": synthesis.voice,
                    "instructions": package.voice_prompt.instructions,
                    "optimize_instructions": package.voice_prompt.optimize_instructions,
                    "source_message": package.voice_prompt.message,
                    "source_message_sha256": hashlib.sha256(
                        package.voice_prompt.message.encode("utf-8")
                    ).hexdigest(),
                    "audio_file": audio_name,
                    "audio_sha256": sha256_file(audio_path),
                    "audio_bytes": audio_path.stat().st_size,
                    "audio_duration_ms": audio_duration_ms,
                    "sample_rate_hz": synthesis.sample_rate_hz,
                    "session_id": synthesis.session_id,
                    "response_id": synthesis.response_id,
                    "usage_characters": synthesis.usage_characters,
                    "first_audio_latency_ms": synthesis.first_audio_latency_ms,
                    "total_latency_ms": synthesis.total_latency_ms,
                    "events_file": event_name,
                    "interpretation_limit": (
                        "音频文件生成成功不代表参与者已经听到、理解或采纳。"
                    ),
                }
                synthesis_results.append(synthesis_record)
                audio_files.append(audio_name)
                store.append_event(
                    {
                        "direction": "tts_provider",
                        "type": "intervention.voice_synthesized",
                        "sequence": package.sequence,
                        "delivery_id": package.delivery_id,
                        "model": synthesis.model,
                        "voice": synthesis.voice,
                        "audio_file": audio_name,
                        "audio_sha256": synthesis_record["audio_sha256"],
                    }
                )
            except (ConnectionError, TimeoutError, ValueError, OSError) as exc:
                synthesis_failed = True
                synthesis_results.append(
                    {
                        "sequence": package.sequence,
                        "delivery_id": package.delivery_id,
                        "status": "failed",
                        "provider": package.voice_prompt.provider,
                        "model": package.voice_prompt.model,
                        "voice": package.voice_prompt.voice,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                store.append_event(
                    {
                        "direction": "tts_provider",
                        "type": "intervention.voice_synthesis_failed",
                        "sequence": package.sequence,
                        "delivery_id": package.delivery_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        preview_name = f"delivery_preview_{len(packages):03d}.html"
        store.write_text(
            preview_name,
            render_delivery_preview(package, audio_filename=audio_name),
        )
        preview_files.append(preview_name)
        store.append_event(
            {
                "direction": "delivery_coordinator",
                "type": "intervention.delivery_prepared",
                "sequence": package.sequence,
                "delivery_id": package.delivery_id,
                "status": package.status,
                "modalities": [
                    package.visual_prompt.modality,
                    package.voice_prompt.modality,
                ],
                "target_actor": package.target_actor,
                "core_content_identical": package.core_content_identical,
            }
        )

    snapshot = controller.snapshot()
    store.write_json("decisions.json", decisions)
    store.write_json("strategy_selections.json", selections)
    store.write_json(
        "intervention_plans.json",
        [plan.model_dump(mode="json") for plan in plans],
    )
    store.write_json("delivery_preparations.json", preparations)
    store.write_json("delivery_packages.json", packages)
    store.write_json("voice_synthesis_results.json", synthesis_results)
    store.write_json("delivery_execution_reports.json", [])
    store.write_json("state_trajectory.json", snapshot.model_dump(mode="json"))
    degraded_count = sum(
        item["status"] == DeliveryPreparationStatus.DEGRADED.value for item in packages
    )
    store.write_json(
        "result.json",
        {
            "valid": not synthesis_failed,
            "observation_count": len(request.observations),
            "decision_count": len(decisions),
            "intervention_plan_count": len(plans),
            "delivery_package_count": len(packages),
            "degraded_delivery_count": degraded_count,
            "preview_files": preview_files,
            "requested_voice_synthesis": synthesize_voice,
            "voice_synthesis_count": sum(
                item["status"] == "synthesized" for item in synthesis_results
            ),
            "voice_synthesis_failure_count": sum(
                item["status"] == "failed" for item in synthesis_results
            ),
            "audio_files": audio_files,
            "execution_reports_pending_frontend": len(packages),
            "final_state": snapshot.points[-1].state,
            "final_action": snapshot.decisions[-1].action,
        },
    )
    return store.run_dir, not synthesis_failed
