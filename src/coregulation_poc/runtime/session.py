from __future__ import annotations

import asyncio
import base64
import io
import wave
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from coregulation_poc.capture.media import MediaChunk
from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.delivery import (
    ChannelExecution,
    DeliveryCoordinator,
    DeliveryPackage,
    DeliveryRuntimeContext,
    OutputExecutionStatus,
    OutputModality,
    load_delivery_policy,
    not_attempted_voice_execution,
)
from coregulation_poc.intervention import StrategySelector, load_strategy_library
from coregulation_poc.intervention.models import InterventionPlan
from coregulation_poc.models import ControlObservation, CoregulationState, StateAssessment
from coregulation_poc.runtime.recognition import StateRecognizer
from coregulation_poc.runtime.window import MediaWindow, RollingMediaWindow

ServerEventSender = Callable[[dict[str, Any]], Awaitable[None]]
TurnBoundaryDetector = Callable[[MediaWindow], bool]


@dataclass(frozen=True, slots=True)
class RealtimeLoopConfig:
    """Engineering cadence controls; none of these values classify dyadic state."""

    window_duration_ms: int = 12_000
    assessment_interval_ms: int = 12_000
    post_intervention_observation_ms: int = 4_000
    max_assessments_per_session: int = 30
    history_assessments: int = 4
    voice_enabled: bool = False

    def __post_init__(self) -> None:
        if self.window_duration_ms < 3_000:
            raise ValueError("window_duration_ms must be at least 3000")
        if self.assessment_interval_ms < 1_000:
            raise ValueError("assessment_interval_ms must be at least 1000")
        if self.post_intervention_observation_ms < 0:
            raise ValueError("post_intervention_observation_ms cannot be negative")
        if self.max_assessments_per_session < 1:
            raise ValueError("max_assessments_per_session must be positive")
        if not 1 <= self.history_assessments <= 20:
            raise ValueError("history_assessments must be between 1 and 20")


@dataclass(frozen=True, slots=True)
class VoiceAudio:
    pcm_audio: bytes
    sample_rate_hz: int
    provider: str
    output_identifier: str | None = None


class VoiceSynthesizer(Protocol):
    async def synthesize(self, text: str) -> VoiceAudio: ...


class RealtimeBrowserSession(Protocol):
    @property
    def api_call_count(self) -> int: ...

    @property
    def runtime_metrics(self) -> dict[str, Any]: ...

    async def start(self) -> None: ...

    async def accept_chunk(self, chunk: MediaChunk) -> None: ...

    async def handle_control(self, control: dict[str, Any]) -> bool: ...

    async def stop(self, status: str) -> None: ...


RealtimeSessionFactory = Callable[[str, ServerEventSender], RealtimeBrowserSession]


def pcm16_wav_bytes(pcm_audio: bytes, sample_rate_hz: int) -> bytes:
    if not pcm_audio or len(pcm_audio) % 2:
        raise ValueError("voice audio must contain non-empty 16-bit PCM samples")
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm_audio)
    return target.getvalue()


def detect_trailing_pcm_silence(window: MediaWindow) -> bool:
    """Conservative technical turn-boundary hint, kept separate from state recognition."""

    chunks = window.audio_chunks[-4:]
    if len(chunks) < 4:
        return False
    root_mean_squares = []
    for chunk in chunks:
        samples = np.frombuffer(chunk.payload, dtype="<i2").astype(np.float64)
        if not samples.size:
            return False
        root_mean_squares.append(float(np.sqrt(np.mean(np.square(samples)))))
    return max(root_mean_squares) <= 450


class RealtimeSession:
    """One browser session coordinating all four modules as a feedback loop."""

    def __init__(
        self,
        *,
        session_id: str,
        recognizer: StateRecognizer,
        send_event: ServerEventSender,
        config: RealtimeLoopConfig | None = None,
        voice_synthesizer: VoiceSynthesizer | None = None,
        turn_boundary_detector: TurnBoundaryDetector = detect_trailing_pcm_silence,
    ) -> None:
        self.session_id = session_id
        self.recognizer = recognizer
        self.send_event = send_event
        self.config = config or RealtimeLoopConfig()
        self.voice_synthesizer = voice_synthesizer
        self.turn_boundary_detector = turn_boundary_detector
        self.window = RollingMediaWindow(duration_ms=self.config.window_duration_ms)
        self.controller = StateTrajectoryController(load_intervention_policy())
        self.selector = StrategySelector(load_strategy_library())
        self.delivery = DeliveryCoordinator(load_delivery_policy())
        self.previous_state: CoregulationState | None = None
        self.previous_plan: InterventionPlan | None = None
        self.assessment_history: list[StateAssessment] = []
        self.pending_delivery: DeliveryPackage | None = None
        self._pending_plan: InterventionPlan | None = None
        self.delivery_reports: list[dict[str, Any]] = []
        self.intervention_outcomes: list[dict[str, Any]] = []
        self.assessment_count = 0
        self._last_scheduled_at_ms: int | None = None
        self._post_response_not_before_ms: int | None = None
        self._analysis_task: asyncio.Task[None] | None = None
        self._analysis_error_count = 0
        self._interventions_paused = False
        self._started = False
        self._stopped = False

    @property
    def api_call_count(self) -> int:
        return int(getattr(self.recognizer, "api_call_count", self.assessment_count))

    @property
    def runtime_metrics(self) -> dict[str, Any]:
        return {
            "assessment_count": self.assessment_count,
            "api_call_count": self.api_call_count,
            "delivery_report_count": len(self.delivery_reports),
            "intervention_outcome_count": len(self.intervention_outcomes),
            "analysis_error_count": self._analysis_error_count,
            "awaiting_post_intervention_response": (
                self.controller.awaiting_post_intervention_response
            ),
            "interventions_paused": self._interventions_paused,
            "voice_enabled": self.config.voice_enabled,
        }

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await self.send_event(
            {
                "type": "loop_started",
                "window_duration_ms": self.config.window_duration_ms,
                "assessment_interval_ms": self.config.assessment_interval_ms,
                "voice_enabled": self.config.voice_enabled,
            }
        )

    async def accept_chunk(self, chunk: MediaChunk) -> None:
        if not self._started or self._stopped:
            return
        self.window.append(chunk)
        snapshot = self.window.snapshot()
        if snapshot is None or not snapshot.has_both_modalities:
            return
        if self.assessment_count >= self.config.max_assessments_per_session:
            return
        if self._analysis_task is not None and not self._analysis_task.done():
            return
        if self._last_scheduled_at_ms is None:
            due = snapshot.end_ms >= self.config.assessment_interval_ms
        else:
            due = (
                snapshot.end_ms - self._last_scheduled_at_ms
                >= self.config.assessment_interval_ms
            )
        if not due:
            return
        self._last_scheduled_at_ms = snapshot.end_ms
        self._analysis_task = asyncio.create_task(self._analyze(snapshot))

    async def analyze_now(self) -> None:
        """Run one deterministic analysis cycle; primarily useful for local verification."""

        snapshot = self.window.snapshot()
        if snapshot is None or not snapshot.has_both_modalities:
            raise ValueError("both audio and image media are required before analysis")
        if self._analysis_task is not None and not self._analysis_task.done():
            await self._analysis_task
            return
        self._last_scheduled_at_ms = snapshot.end_ms
        await self._analyze(snapshot)

    async def _analyze(self, snapshot: MediaWindow) -> None:
        history = tuple(self.assessment_history[-self.config.history_assessments :])
        history_available = bool(history)
        post_response_observed = (
            self._post_response_not_before_ms is not None
            and snapshot.end_ms >= self._post_response_not_before_ms
        )
        try:
            await self.send_event(
                {
                    "type": "analysis_started",
                    "window_start_ms": snapshot.start_ms,
                    "window_end_ms": snapshot.end_ms,
                }
            )
            assessment = await self.recognizer.assess(
                session_id=self.session_id,
                window=snapshot,
                previous_state=self.previous_state,
                history=history,
                history_available=history_available,
            )
            observation = ControlObservation(
                assessment=assessment,
                natural_turn_boundary=self.turn_boundary_detector(snapshot),
                post_intervention_response_observed=post_response_observed,
                interaction_history_available=history_available,
            )
            decision = self.controller.ingest(observation)
            self.assessment_count += 1
            self.assessment_history.append(assessment)
            self.previous_state = assessment.state
            if post_response_observed:
                outcome = self._build_intervention_outcome(
                    assessment=assessment,
                    recovery_status=decision.recovery_status.value,
                )
                if outcome is not None:
                    self.intervention_outcomes.append(outcome)
                    await self.send_event({"type": "intervention_outcome", **outcome})
                self._post_response_not_before_ms = None
                self.pending_delivery = None
                self._pending_plan = None
            await self.send_event(
                {
                    "type": "state_update",
                    "sequence": decision.sequence,
                    "assessed_at_ms": assessment.assessed_at_ms,
                    "state": None if assessment.state is None else assessment.state.value,
                    "confidence": assessment.confidence.value,
                    "evidence_sufficiency": assessment.evidence_sufficiency.value,
                    "action": decision.action.value,
                    "reason_code": decision.reason_code.value,
                    "recovery_status": decision.recovery_status.value,
                    "post_intervention_response_observed": post_response_observed,
                }
            )
            selection = self.selector.select(
                assessment=assessment,
                decision=decision,
                previous_plan=self.previous_plan,
            )
            if selection.plan is None:
                if self.controller.awaiting_post_intervention_response:
                    self.controller.mark_intervention_not_delivered()
                    await self.send_event(
                        {
                            "type": "intervention_held",
                            "sequence": decision.sequence,
                            "reason": selection.hold_reason.value,
                        }
                    )
                return
            self.previous_plan = selection.plan
            preparation = self.delivery.prepare(
                plan=selection.plan,
                runtime=DeliveryRuntimeContext(
                    prepared_at_ms=assessment.assessed_at_ms,
                    interventions_paused=self._interventions_paused,
                    voice_enabled=self.config.voice_enabled,
                    voice_available=self.voice_synthesizer is not None,
                ),
            )
            if preparation.package is None:
                self.controller.mark_intervention_not_delivered()
                await self.send_event(
                    {
                        "type": "intervention_held",
                        "sequence": decision.sequence,
                        "reason": preparation.hold_reason.value,
                    }
                )
                return
            self._pending_plan = selection.plan
            await self._deliver(preparation.package)
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            self._analysis_error_count += 1
            with suppress(ConnectionError, RuntimeError):
                await self.send_event(
                    {
                        "type": "loop_error",
                        "stage": "analysis",
                        "message": str(exc),
                        "retryable": True,
                    }
                )

    async def _deliver(self, package: DeliveryPackage) -> None:
        audio_base64: str | None = None
        audio_mime_type: str | None = None
        voice_error: str | None = None
        voice_output_identifier: str | None = None
        if package.voice_prompt.enabled and self.voice_synthesizer is not None:
            try:
                voice = await self.voice_synthesizer.synthesize(package.voice_prompt.message)
                wav = pcm16_wav_bytes(voice.pcm_audio, voice.sample_rate_hz)
                audio_base64 = base64.b64encode(wav).decode("ascii")
                audio_mime_type = "audio/wav"
                voice_output_identifier = voice.output_identifier
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                voice_error = str(exc)
        self.pending_delivery = package
        await self.send_event(
            {
                "type": "intervention",
                "delivery_id": package.delivery_id,
                "sequence": package.sequence,
                "prepared_at_ms": package.prepared_at_ms,
                "target_actor": package.target_actor.value,
                "strategy_id": package.strategy_id,
                "heading": package.visual_prompt.heading,
                "message": package.visual_prompt.message,
                "dismissible": package.visual_prompt.dismissible,
                "voice_expected": package.voice_prompt.enabled,
                "voice_provider": package.voice_prompt.provider,
                "voice_output_identifier": voice_output_identifier,
                "audio_mime_type": audio_mime_type,
                "audio_base64": audio_base64,
                "voice_error": voice_error,
            }
        )

    async def handle_control(self, control: dict[str, Any]) -> bool:
        control_type = control.get("type")
        if control_type in {"pause_interventions", "resume_interventions"}:
            self._interventions_paused = control_type == "pause_interventions"
            await self.send_event(
                {
                    "type": (
                        "interventions_paused"
                        if self._interventions_paused
                        else "interventions_resumed"
                    ),
                    "recorded_at_ms": self._optional_non_negative_int(
                        control.get("recorded_at_ms"), "recorded_at_ms"
                    ),
                }
            )
            return True
        if control_type != "delivery_execution":
            return False
        package = self.pending_delivery
        if package is None or control.get("delivery_id") != package.delivery_id:
            raise ValueError("delivery_execution does not match the pending intervention")
        recorded_at_ms = self._non_negative_int(control.get("recorded_at_ms"), "recorded_at_ms")
        visual = self._channel_execution(
            control.get("visual"),
            modality=OutputModality.VISUAL_TEXT,
            default_provider="browser_overlay",
        )
        if package.voice_prompt.enabled:
            voice = self._channel_execution(
                control.get("voice"),
                modality=OutputModality.SPOKEN_VOICE,
                default_provider=package.voice_prompt.provider,
            )
        else:
            voice = not_attempted_voice_execution()
        report = self.delivery.record_execution(
            package=package,
            recorded_at_ms=recorded_at_ms,
            visual=visual,
            voice=voice,
            user_acknowledged=(
                bool(control["user_acknowledged"])
                if isinstance(control.get("user_acknowledged"), bool)
                else None
            ),
        )
        self.delivery_reports.append(report.model_dump(mode="json"))
        delivered = any(
            channel.status is OutputExecutionStatus.DELIVERED for channel in (visual, voice)
        )
        if delivered:
            self._post_response_not_before_ms = (
                recorded_at_ms + self.config.post_intervention_observation_ms
            )
            # The next model window must contain only behavior observed after delivery.
            self.window.clear()
            self._last_scheduled_at_ms = recorded_at_ms
        else:
            self.controller.mark_intervention_not_delivered()
            self.pending_delivery = None
            self._pending_plan = None
        await self.send_event(
            {
                "type": "delivery_execution_received",
                "delivery_id": package.delivery_id,
                "overall_status": report.overall_status.value,
                "recorded_at_ms": report.recorded_at_ms,
                "visual_status": report.visual.status.value,
                "voice_status": report.voice.status.value,
                "post_intervention_observation_armed": delivered,
            }
        )
        return True

    def _build_intervention_outcome(
        self,
        *,
        assessment: StateAssessment,
        recovery_status: str,
    ) -> dict[str, Any] | None:
        package = self.pending_delivery
        plan = self._pending_plan
        if package is None or plan is None:
            return None
        effect_category = {
            "recovered": "positive",
            "partial_recovery": "limited",
            "not_recovered": "no_change",
            "deteriorated": "negative",
        }.get(recovery_status, "indeterminate")
        return {
            "delivery_id": package.delivery_id,
            "strategy_id": plan.strategy_id,
            "target_actor": plan.target_actor.value,
            "repair_target": plan.repair_target.value,
            "observed_at_ms": assessment.assessed_at_ms,
            "recovery_status": recovery_status,
            "effect_category": effect_category,
            "expected_recovery": list(plan.expected_recovery),
            "observed_state": (
                None if assessment.state is None else assessment.state.value
            ),
            "observed_interaction_performance": list(
                assessment.interaction_performance
            ),
            "observed_evidence": [
                {
                    "modality": item.modality.value,
                    "actor": item.actor.value,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "code": item.code,
                }
                for item in assessment.modality_evidence.all_items
            ],
            "interpretation_limit": (
                "The record links the strategy to the next observable window; "
                "it does not by itself establish that the intervention caused the change."
            ),
        }

    @staticmethod
    def _channel_execution(
        raw: object,
        *,
        modality: OutputModality,
        default_provider: str,
    ) -> ChannelExecution:
        if not isinstance(raw, dict):
            raise ValueError(f"{modality.value} execution must be an object")
        status = OutputExecutionStatus(str(raw.get("status")))
        if status is OutputExecutionStatus.NOT_ATTEMPTED:
            return ChannelExecution(modality=modality, status=status)
        started_at_ms = RealtimeSession._non_negative_int(
            raw.get("started_at_ms"), f"{modality.value}.started_at_ms"
        )
        completed_raw = raw.get("completed_at_ms")
        completed_at_ms = (
            None
            if completed_raw is None
            else RealtimeSession._non_negative_int(
                completed_raw, f"{modality.value}.completed_at_ms"
            )
        )
        error = raw.get("error")
        return ChannelExecution(
            modality=modality,
            status=status,
            started_at_ms=started_at_ms,
            completed_at_ms=completed_at_ms,
            provider=(
                str(raw.get("provider"))
                if isinstance(raw.get("provider"), str)
                else default_provider
            ),
            output_identifier=(
                str(raw.get("output_identifier"))
                if isinstance(raw.get("output_identifier"), str)
                else None
            ),
            error=str(error) if isinstance(error, str) else None,
        )

    @staticmethod
    def _non_negative_int(value: object, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value

    @staticmethod
    def _optional_non_negative_int(value: object, field: str) -> int | None:
        if value is None:
            return None
        return RealtimeSession._non_negative_int(value, field)

    async def stop(self, status: str) -> None:
        del status
        if self._stopped:
            return
        self._stopped = True
        if self._analysis_task is not None:
            if not self._analysis_task.done():
                self._analysis_task.cancel()
            with suppress(asyncio.CancelledError, ConnectionError, RuntimeError):
                await self._analysis_task
        self.window.clear()
