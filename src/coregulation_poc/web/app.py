from __future__ import annotations

import asyncio
import hmac
import json
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from coregulation_poc.acoustics.speaker_enrollment import (
    SpeakerEnrollment,
    enrolled_speaker_from_pcm16,
)
from coregulation_poc.acoustics.tencent_voiceprint import (
    SpeakerEnrollmentRecord,
    TencentSpeakerEnrollment,
    TencentVoiceprintError,
    TencentVoiceprintService,
)
from coregulation_poc.capture.media import MediaChunk, MediaFormat
from coregulation_poc.models import TaskContext
from coregulation_poc.paths import DEFAULT_OUTPUT_DIR, PACKAGE_DIR, resolve_project_path
from coregulation_poc.runtime.session import RealtimeBrowserSession, RealtimeSessionFactory
from coregulation_poc.web.protocol import (
    PROTOCOL_VERSION,
    BrowserCaptureRecorder,
    BrowserProtocolError,
)
from coregulation_poc.web.research import ResearchSessionRegistry

SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
SPEAKER_BINDING_TIMEOUT_SECONDS = 25
SESSION_ADMISSION_TTL_SECONDS = 30 * 60
MAX_PENDING_ADMISSIONS = 500
MAX_ACTIVE_SESSIONS = 4
MAX_JSON_BODY_BYTES = 64_000
MAX_SPEAKER_AUDIO_BYTES = 1_000_000
REALTIME_ONLY_CONTROLS = {
    "pause_interventions",
    "resume_interventions",
    "family_response",
    "delivery_execution",
}
PARTICIPANT_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
SESSION_ROUND = re.compile(r"^[A-Za-z0-9_-]{1,20}$")
EXPERIMENT_LABEL = re.compile(r"^[\w\u4e00-\u9fff-]{1,40}$")
STATIC_DIR = (PACKAGE_DIR / "web" / "static").resolve()
ChunkHandler = Callable[[MediaChunk], Awaitable[None]]
AUDITED_RUNTIME_EVENTS = {
    "loop_started",
    "analysis_started",
    "state_update",
    "speaker_binding",
    "voiceprint_cleanup",
    "intervention",
    "intervention_held",
    "intervention_outcome",
    "interventions_paused",
    "interventions_resumed",
    "delivery_execution_received",
    "family_response_received",
    "expert_takeover_started",
    "expert_takeover_ended",
    "expert_intervention_recorded",
    "loop_error",
    "control_unavailable",
}


@dataclass(frozen=True, slots=True)
class SessionAdmission:
    token: str = field(repr=False)
    expires_at_monotonic: float


@dataclass(frozen=True, slots=True)
class BrowserServerConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    media_format: MediaFormat = field(default_factory=MediaFormat)
    max_image_bytes: int = 750_000
    max_session_seconds: int = 2 * 60 * 60
    hello_timeout_seconds: int = 10
    admission_ttl_seconds: int = SESSION_ADMISSION_TTL_SECONDS
    max_pending_admissions: int = MAX_PENDING_ADMISSIONS
    max_active_sessions: int = MAX_ACTIVE_SESSIONS
    max_json_body_bytes: int = MAX_JSON_BODY_BYTES
    max_speaker_audio_bytes: int = MAX_SPEAKER_AUDIO_BYTES
    access_token: str | None = field(default=None, repr=False)
    research_access_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", resolve_project_path(self.output_dir))
        if self.max_image_bytes < 10_000:
            raise ValueError("max_image_bytes is too small for browser JPEG frames")
        if self.max_session_seconds < 1:
            raise ValueError("max_session_seconds must be positive")
        if self.hello_timeout_seconds < 1:
            raise ValueError("hello_timeout_seconds must be positive")
        if self.access_token is not None and len(self.access_token) < 12:
            raise ValueError("browser capture access token must contain at least 12 characters")
        if self.research_access_token is not None and len(self.research_access_token) < 12:
            raise ValueError("research console access token must contain at least 12 characters")

        if self.admission_ttl_seconds < 60:
            raise ValueError("admission_ttl_seconds must be at least 60")
        if self.max_pending_admissions < 1:
            raise ValueError("max_pending_admissions must be positive")
        if self.max_active_sessions < 1:
            raise ValueError("max_active_sessions must be positive")
        if self.max_json_body_bytes < 1 or self.max_speaker_audio_bytes < 1:
            raise ValueError("request body limits must be positive")

def _parse_control(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BrowserProtocolError("control message must be valid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise BrowserProtocolError("control message must contain a string type")
    return value


def _parse_hello(
    message: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if message.get("type") != "hello":
        raise BrowserProtocolError("the first message must be hello")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise BrowserProtocolError("browser protocol version does not match the server")
    session_id = message.get("session_id")
    if not isinstance(session_id, str) or SESSION_ID.fullmatch(session_id) is None:
        raise BrowserProtocolError(
            "session_id must contain 1-80 letters, numbers, underscores or hyphens"
        )
    capabilities = message.get("capabilities")
    study_context = message.get("study_context")
    if not isinstance(study_context, dict):
        raise BrowserProtocolError("开始实验前必须填写基本信息")
    participant_id = study_context.get("participant_id")
    experiment_label = study_context.get("experiment_label")
    session_round = study_context.get("session_round")
    if not isinstance(participant_id, str) or PARTICIPANT_ID.fullmatch(participant_id) is None:
        raise BrowserProtocolError("参与者编号只能包含字母、数字、下划线或连字符")
    if (
        not isinstance(experiment_label, str)
        or EXPERIMENT_LABEL.fullmatch(experiment_label) is None
    ):
        raise BrowserProtocolError("实验类型只能包含文字、数字、下划线或连字符")
    if not isinstance(session_round, str) or SESSION_ROUND.fullmatch(session_round) is None:
        raise BrowserProtocolError("实验轮次只能包含字母、数字、下划线或连字符")
    basic_info = study_context.get("basic_info")
    if not isinstance(basic_info, dict):
        raise BrowserProtocolError("开始实验前必须填写基本信息")
    parent_age = basic_info.get("parent_age")
    child_age = basic_info.get("child_age")
    child_grade = basic_info.get("child_grade")
    if (
        not isinstance(parent_age, int)
        or isinstance(parent_age, bool)
        or not 18 <= parent_age <= 80
    ):
        raise BrowserProtocolError("家长年龄必须在 18 至 80 岁之间")
    if (
        not isinstance(child_age, int)
        or isinstance(child_age, bool)
        or not 5 <= child_age <= 18
    ):
        raise BrowserProtocolError("儿童年龄必须在 5 至 18 岁之间")
    if not isinstance(child_grade, str) or not child_grade.strip():
        raise BrowserProtocolError("必须填写儿童年级")
    family_roles = study_context.get("family_roles")
    task_context = study_context.get("task_context")
    if not isinstance(family_roles, dict) or not isinstance(task_context, dict):
        raise BrowserProtocolError("角色和作业信息不完整")
    return (
        session_id,
        capabilities if isinstance(capabilities, dict) else {},
        {
            "participant_id": participant_id,
            "experiment_label": experiment_label,
            "session_round": session_round,
            "basic_info": {
                "parent_age": parent_age,
                "child_age": child_age,
                "child_grade": child_grade.strip(),
            },
            "family_roles": {
                "parent": family_roles.get("parent", ""),
                "child": family_roles.get("child", ""),
            },
            "task_context": {
                "task_name": task_context.get("task_name", ""),
                "task_type": task_context.get("task_type", ""),
                "task_difficulty": task_context.get("task_difficulty", ""),
                "child_grade": child_grade.strip(),
            },
        },
    )


def _origin_matches_host(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")
    if not origin or not host:
        return True
    return urlsplit(origin).netloc.casefold() == host.casefold()


def _valid_access_token(message: dict[str, Any], expected_token: str | None) -> bool:
    if expected_token is None:
        return True
    supplied_token = message.get("access_token")
    return isinstance(supplied_token, str) and hmac.compare_digest(
        supplied_token,
        expected_token,
    )


def _valid_session_access_token(
    supplied_token: Any,
    configured_token: str | None,
    session_token: str | None,
) -> bool:
    if configured_token is None and session_token is None:
        return True
    if not isinstance(supplied_token, str):
        return False
    return any(
        hmac.compare_digest(supplied_token, expected)
        for expected in (configured_token, session_token)
        if expected is not None
    )


def _request_origin_matches_host(request: Request) -> bool:
    origin = request.headers.get("origin")
    host = request.headers.get("host")
    if not origin or not host:
        return True
    return urlsplit(origin).netloc.casefold() == host.casefold()


async def _read_request_body(request: Request, *, max_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="请求长度格式不正确") from exc
        if declared_bytes > max_bytes:
            raise HTTPException(status_code=413, detail="请求数据过大")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail="请求数据过大")
        body.extend(chunk)
    return bytes(body)


def create_browser_capture_app(
    config: BrowserServerConfig | None = None,
    *,
    chunk_handler: ChunkHandler | None = None,
    session_factory: RealtimeSessionFactory | None = None,
    research_registry: ResearchSessionRegistry | None = None,
    voiceprint_service: TencentVoiceprintService | None = None,
) -> FastAPI:
    server_config = config or BrowserServerConfig()
    registry = research_registry or ResearchSessionRegistry()
    app = FastAPI(
        title="Parent-child co-regulation browser capture",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.browser_server_config = server_config
    app.state.research_registry = registry
    app.state.voiceprint_service = voiceprint_service
    session_enrollments: dict[str, SpeakerEnrollmentRecord] = {}
    app.state.session_enrollments = session_enrollments
    enrollment_locks: dict[str, asyncio.Lock] = {}
    cleanup_tasks: set[asyncio.Task[None]] = set()
    app.state.voiceprint_cleanup_tasks = cleanup_tasks
    failed_cleanup_enrollments: list[TencentSpeakerEnrollment] = []
    session_admissions: dict[str, SessionAdmission] = {}
    app.state.session_admissions = session_admissions
    active_session_ids: set[str] = set()
    app.state.active_session_ids = active_session_ids
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    async def cleanup_remote_enrollment(
        enrollment: SpeakerEnrollmentRecord | None,
        *,
        retain_on_failure: bool = True,
    ) -> bool:
        if (
            voiceprint_service is None
            or not isinstance(enrollment, TencentSpeakerEnrollment)
        ):
            return True
        try:
            await asyncio.wait_for(
                asyncio.to_thread(voiceprint_service.delete_enrollment, enrollment),
                timeout=SPEAKER_BINDING_TIMEOUT_SECONDS * 3,
            )
            return True
        except (TimeoutError, TencentVoiceprintError, OSError, RuntimeError):
            app.state.voiceprint_cleanup_failed = (
                getattr(app.state, "voiceprint_cleanup_failed", 0) + 1
            )
            if retain_on_failure and all(
                item.group_id != enrollment.group_id
                for item in failed_cleanup_enrollments
            ):
                failed_cleanup_enrollments.append(enrollment)
            return False

    def schedule_remote_cleanup(enrollment: SpeakerEnrollmentRecord | None) -> None:
        if (
            voiceprint_service is None
            or not isinstance(enrollment, TencentSpeakerEnrollment)
        ):
            return
        task = asyncio.create_task(cleanup_remote_enrollment(enrollment))
        cleanup_tasks.add(task)
        task.add_done_callback(cleanup_tasks.discard)

    async def shutdown_voiceprint_cleanup() -> None:
        if cleanup_tasks:
            await asyncio.gather(*tuple(cleanup_tasks), return_exceptions=True)
        remaining: list[TencentSpeakerEnrollment] = list(failed_cleanup_enrollments)
        remaining.extend(
            enrollment
            for enrollment in session_enrollments.values()
            if isinstance(enrollment, TencentSpeakerEnrollment)
            and all(item.group_id != enrollment.group_id for item in remaining)
        )
        for enrollment in remaining:
            await cleanup_remote_enrollment(enrollment, retain_on_failure=False)

    app.router.add_event_handler("shutdown", shutdown_voiceprint_cleanup)

    def cleanup_expired_admissions() -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, admission in session_admissions.items()
            if admission.expires_at_monotonic <= now
            and session_id not in active_session_ids
        ]
        for expired_session_id in expired:
            session_admissions.pop(expired_session_id, None)
            enrollment = session_enrollments.pop(expired_session_id, None)
            enrollment_locks.pop(expired_session_id, None)
            schedule_remote_cleanup(enrollment)

    def session_admission_token(session_id: str) -> str | None:
        cleanup_expired_admissions()
        admission = session_admissions.get(session_id)
        return None if admission is None else admission.token

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable) -> Any:
        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; "
            "connect-src 'self' ws: wss:; worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(self), microphone=(self), geolocation=()",
        )
        if request.url.path in {"/", "/research"}:
            response.headers["Cache-Control"] = "no-store"
        return response


    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/research", include_in_schema=False)
    async def research_console() -> FileResponse:
        return FileResponse(STATIC_DIR / "research.html", media_type="text/html")

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, Any]:
        cleanup_expired_admissions()
        return {
            "status": "ok",
            "capture_mode": "browser_camera_microphone",
            "protocol_version": PROTOCOL_VERSION,
            "runtime_mode": (
                "closed_loop" if session_factory is not None else "capture_preview"
            ),
            "active_session_count": len(active_session_ids),
            "pending_admission_count": len(session_admissions),
            "raw_media_saved": False,
            "api_called": False,
            "api_calls_enabled": session_factory is not None,
            "downstream_chunk_handler_configured": chunk_handler is not None,
            "realtime_loop_configured": session_factory is not None,
            "voiceprint_provider": (
                voiceprint_service.provider_name
                if voiceprint_service is not None
                else "local_development"
            ),
            "voiceprint_cleanup_pending": len(cleanup_tasks),
            "voiceprint_cleanup_failed": getattr(
                app.state, "voiceprint_cleanup_failed", 0
            ),
            "voiceprint_cleanup_retry_count": len(failed_cleanup_enrollments),
            "access_control_required": server_config.access_token is not None,
            "research_console_enabled": True,
            "research_access_control_required": (
                server_config.research_access_token is not None
            ),
        }

    @app.post("/api/session-admission")
    async def create_session_admission(request: Request) -> dict[str, Any]:
        if not _request_origin_matches_host(request):
            raise HTTPException(status_code=403, detail="请求来源不受信任")
        try:
            raw_body = await _read_request_body(
                request,
                max_bytes=server_config.max_json_body_bytes,
            )
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="基本信息格式不正确") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="基本信息格式不正确")

        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or SESSION_ID.fullmatch(session_id) is None:
            raise HTTPException(status_code=400, detail="本次实验编号无效")
        basic_info = payload.get("basic_info")
        if not isinstance(basic_info, dict):
            raise HTTPException(status_code=400, detail="请先填写基本信息")
        parent_age = basic_info.get("parent_age")
        child_age = basic_info.get("child_age")
        child_grade = basic_info.get("child_grade")
        if (
            not isinstance(parent_age, int)
            or isinstance(parent_age, bool)
            or not 18 <= parent_age <= 80
        ):
            raise HTTPException(status_code=400, detail="请填写正确的家长年龄")
        if (
            not isinstance(child_age, int)
            or isinstance(child_age, bool)
            or not 5 <= child_age <= 18
        ):
            raise HTTPException(status_code=400, detail="请填写正确的儿童年龄")
        if (
            not isinstance(child_grade, str)
            or not child_grade.strip()
            or len(child_grade.strip()) > 50
        ):
            raise HTTPException(status_code=400, detail="请填写儿童年级")

        cleanup_expired_admissions()
        if session_id in active_session_ids:
            raise HTTPException(status_code=409, detail="本次会话已经开始")
        if (
            session_id not in session_admissions
            and len(session_admissions) >= server_config.max_pending_admissions
        ):
            raise HTTPException(status_code=429, detail="当前等待会话较多，请稍后重试")

        old_enrollment = session_enrollments.pop(session_id, None)
        enrollment_locks.pop(session_id, None)
        schedule_remote_cleanup(old_enrollment)

        session_token = secrets.token_urlsafe(32)
        session_admissions[session_id] = SessionAdmission(
            token=session_token,
            expires_at_monotonic=(
                time.monotonic() + server_config.admission_ttl_seconds
            ),
        )
        return {
            "session_id": session_id,
            "session_token": session_token,
            "expires_in_seconds": server_config.admission_ttl_seconds,
        }

    @app.post("/api/speaker-binding/{session_id}/{speaker_label}")
    async def bind_session_speaker(
        session_id: str,
        speaker_label: str,
        request: Request,
    ) -> dict[str, Any]:
        if not _request_origin_matches_host(request):
            raise HTTPException(status_code=403, detail="请求来源不受信任")
        if SESSION_ID.fullmatch(session_id) is None:
            raise HTTPException(status_code=400, detail="本次实验编号无效")
        if speaker_label not in {"parent", "child"}:
            raise HTTPException(status_code=404, detail="只能绑定家长或儿童")
        supplied_token = request.headers.get("x-study-access-token")
        if not _valid_session_access_token(
            supplied_token,
            server_config.access_token,
            session_admission_token(session_id),
        ):
            raise HTTPException(status_code=403, detail="本次会话已失效，请刷新页面后重试")

        if request.headers.get("content-type", "").split(";", 1)[0] != "application/octet-stream":
            raise HTTPException(status_code=415, detail="录音格式不正确")
        pcm_audio = await _read_request_body(
            request,
            max_bytes=server_config.max_speaker_audio_bytes,
        )
        lock = enrollment_locks.setdefault(session_id, asyncio.Lock())
        try:
            async with lock:
                current = session_enrollments.get(session_id)
                if voiceprint_service is not None:
                    cloud_current = (
                        current if isinstance(current, TencentSpeakerEnrollment) else None
                    )
                    enrollment = await asyncio.wait_for(
                        asyncio.to_thread(
                            voiceprint_service.enroll_speaker,
                            pcm_audio,
                            speaker_label,
                            session_id,
                            cloud_current,
                        ),
                        timeout=SPEAKER_BINDING_TIMEOUT_SECONDS,
                    )
                    speaker = enrollment.speakers[speaker_label]
                else:
                    speaker = await asyncio.wait_for(
                        asyncio.to_thread(
                            enrolled_speaker_from_pcm16,
                            pcm_audio,
                            speaker_label,
                        ),
                        timeout=SPEAKER_BINDING_TIMEOUT_SECONDS,
                    )
                    speakers = {} if current is None else dict(current.speakers)
                    speakers[speaker_label] = speaker
                    enrollment = SpeakerEnrollment(
                        family_id=session_id,
                        speakers=speakers,
                    )
                session_enrollments[session_id] = enrollment
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="声音确认时间过长，请重新录制",
            ) from exc
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="服务器尚未安装声纹识别组件，请联系研究人员",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (TencentVoiceprintError, OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail="服务器暂时无法确认声音，请稍后重新录制",
            ) from exc
        return {
            "session_id": session_id,
            "speaker": speaker_label,
            "complete": enrollment.is_complete,
            "bound_speakers": sorted(enrollment.speakers),
            "provider": enrollment.model_name,
            "raw_audio_saved_locally": False,
            "voiceprint_embedding_saved_locally": False,
            "audio_transmitted_to_voiceprint_provider": (
                voiceprint_service is not None
            ),
            "cloud_voiceprint_registered": voiceprint_service is not None,
            "duration_ms": speaker.duration_ms,
        }

    @app.websocket("/ws/research")
    async def research_supervision(websocket: WebSocket) -> None:
        if not _origin_matches_host(websocket):
            await websocket.close(code=1008, reason="WebSocket origin is not allowed")
            return
        await websocket.accept()
        send_lock = asyncio.Lock()

        async def send_research(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        subscribed = False
        try:
            try:
                hello_event = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=server_config.hello_timeout_seconds,
                )
            except TimeoutError as exc:
                raise BrowserProtocolError("研究端未在规定时间内完成验证") from exc
            if hello_event.get("text") is None:
                raise BrowserProtocolError("研究端的首条消息必须是文本验证消息")
            hello = _parse_control(hello_event["text"])
            if hello.get("type") != "research_hello":
                raise BrowserProtocolError("研究端的首条消息必须是 research_hello")
            if not _valid_access_token(hello, server_config.research_access_token):
                raise BrowserProtocolError("研究端访问码无效")
            await registry.subscribe(send_research)
            subscribed = True
            await send_research(await registry.snapshot())
            while True:
                event = await websocket.receive()
                if event.get("text") is None:
                    raise WebSocketDisconnect(code=1000)
                control = _parse_control(event["text"])
                control_type = control["type"]
                if control_type == "ping":
                    await send_research({"type": "pong"})
                elif control_type == "refresh":
                    await send_research(await registry.snapshot())
                elif control_type in {
                    "expert_takeover",
                    "expert_release",
                    "expert_intervention",
                }:
                    try:
                        handled = await registry.handle_control(control)
                    except ValueError as exc:
                        await send_research({"type": "error", "message": str(exc)})
                    else:
                        await send_research(
                            {
                                "type": "research_control_ack",
                                "control_type": control_type,
                                "session_id": control.get("session_id"),
                                "handled": handled,
                            }
                        )
                else:
                    raise BrowserProtocolError(f"unsupported research control: {control_type}")
        except WebSocketDisconnect:
            pass
        except (BrowserProtocolError, ValueError) as exc:
            try:
                await send_research({"type": "error", "message": str(exc)})
                await websocket.close(code=1003)
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if subscribed:
                await registry.unsubscribe(send_research)

    @app.websocket("/ws/live")
    async def live_capture(websocket: WebSocket) -> None:
        if not _origin_matches_host(websocket):
            await websocket.close(code=1008, reason="WebSocket origin is not allowed")
            return
        await websocket.accept()
        recorder: BrowserCaptureRecorder | None = None
        runtime_session: RealtimeBrowserSession | None = None
        runtime_stopped = False
        completion_status = "disconnected"
        session_id: str | None = None
        session_authorized = False
        send_lock = asyncio.Lock()

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)
                if recorder is not None and payload.get("type") in AUDITED_RUNTIME_EVENTS:
                    audited = dict(payload)
                    audio_attached = isinstance(audited.pop("audio_base64", None), str)
                    recorder.store.append_event(
                        {
                            "type": "realtime_loop_event",
                            "event": audited,
                            "audio_attached_to_browser_message": audio_attached,
                            "payload_saved": False,
                        }
                    )
                if session_id is not None:
                    await registry.observe(session_id, payload)

        async def stop_runtime(status: str) -> None:
            nonlocal runtime_stopped
            if runtime_session is None or runtime_stopped:
                return
            runtime_stopped = True
            await runtime_session.stop(status)

        def runtime_metrics() -> dict[str, Any]:
            if runtime_session is None:
                return {}
            metrics = getattr(runtime_session, "runtime_metrics", None)
            if isinstance(metrics, dict):
                return metrics
            return {"api_call_count": runtime_session.api_call_count}

        try:
            try:
                hello_event = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=server_config.hello_timeout_seconds,
                )
            except TimeoutError as exc:
                raise BrowserProtocolError("browser did not send hello before the timeout") from exc
            if hello_event.get("text") is None:
                raise BrowserProtocolError("the first WebSocket message must be a text hello")
            hello_message = _parse_control(hello_event["text"])
            session_id, capabilities, study_context = _parse_hello(hello_message)
            if not _valid_session_access_token(
                hello_message.get("access_token"),
                server_config.access_token,
                session_admission_token(session_id),
            ):
                raise BrowserProtocolError("本次会话已失效，请刷新页面后重试")
            enrollment = session_enrollments.get(session_id)
            if session_factory is not None and (
                enrollment is None or not enrollment.is_complete
            ):
                raise BrowserProtocolError("请先完成家长和儿童的声音绑定")

            if session_id in active_session_ids:
                raise BrowserProtocolError("本次会话已在另一页面中运行")
            if len(active_session_ids) >= server_config.max_active_sessions:
                raise BrowserProtocolError("当前进行中的会话较多，请稍后重试")
            active_session_ids.add(session_id)

            session_authorized = True
            recorder = BrowserCaptureRecorder(
                output_dir=server_config.output_dir,
                session_id=session_id,
                media_format=server_config.media_format,
                max_image_bytes=server_config.max_image_bytes,
                client_capabilities=capabilities,
                study_context=study_context,
            )
            enrollment_event: dict[str, Any] | None = None
            if enrollment is not None and enrollment.is_complete:
                enrollment_event = {
                    "type": "speaker_enrollment",
                    "method": enrollment.model_name,
                    "complete": True,
                    "bound_speakers": sorted(enrollment.speakers),
                    "durations_ms": {
                        label: speaker.duration_ms
                        for label, speaker in enrollment.speakers.items()
                    },
                    "raw_audio_saved_locally": False,
                    "voiceprint_embedding_saved_locally": False,
                    "audio_transmitted_to_voiceprint_provider": isinstance(
                        enrollment, TencentSpeakerEnrollment
                    ),
                    "cloud_voiceprint_registered": isinstance(
                        enrollment, TencentSpeakerEnrollment
                    ),
                }
                recorder.store.append_event(enrollment_event)
            if session_factory is not None:
                runtime_session = session_factory(session_id, send_json, enrollment)
                task_context_raw = hello_message.get("study_context", {}).get("task_context")
                if isinstance(task_context_raw, dict):
                    try:
                        validated = TaskContext.model_validate(task_context_raw)
                        if hasattr(runtime_session, "set_task_context"):
                            runtime_session.set_task_context(validated.model_dump())
                    except ValueError as exc:
                        raise BrowserProtocolError(
                            f"任务上下文验证失败: {exc}"
                        ) from exc
            await registry.register(session_id, runtime_session)
            if enrollment_event is not None:
                await registry.observe(session_id, enrollment_event)
            await send_json(
                {
                    "type": "ready",
                    "protocol_version": PROTOCOL_VERSION,
                    "media_format": {
                        **asdict(server_config.media_format),
                        "audio_chunk_bytes": server_config.media_format.audio_chunk_bytes,
                    },
                    "max_image_bytes": server_config.max_image_bytes,
                    "max_session_seconds": server_config.max_session_seconds,
                    "raw_media_saved": False,
                    "realtime_loop_enabled": runtime_session is not None,
                    "speaker_enrollment_complete": enrollment_event is not None,
                    "runtime_controls_enabled": runtime_session is not None,
                }
            )
            session_deadline = time.monotonic() + server_config.max_session_seconds
            while True:
                remaining_seconds = session_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise BrowserProtocolError("capture session exceeded the server time limit")
                try:
                    event = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=remaining_seconds,
                    )
                except TimeoutError as exc:
                    raise BrowserProtocolError(
                        "capture session exceeded the server time limit"
                    ) from exc
                if event.get("bytes") is not None:
                    chunk = recorder.accept_packet(event["bytes"])
                    if chunk_handler is not None:
                        await chunk_handler(chunk)
                    if runtime_session is not None:
                        await runtime_session.accept_chunk(chunk)
                    continue
                raw_text = event.get("text")
                if raw_text is None:
                    raise WebSocketDisconnect(code=1000)
                control = _parse_control(raw_text)
                if control["type"] == "start":
                    recorder.start()
                    await registry.mark_status(session_id, "active")
                    await send_json({"type": "started"})
                    if runtime_session is not None:
                        await runtime_session.start()
                elif control["type"] == "stop":
                    completion_status = "completed"
                    await stop_runtime(completion_status)
                    client_metrics = control.get("client_metrics")
                    summary = recorder.finish(
                        status=completion_status,
                        client_metrics=(
                            client_metrics if isinstance(client_metrics, dict) else None
                        ),
                        runtime_metrics=runtime_metrics(),
                    )
                    await send_json({"type": "summary", **summary.as_public_dict()})
                    await websocket.close(code=1000)
                    return
                elif control["type"] == "abort":
                    completion_status = "failed"
                    await stop_runtime(completion_status)
                    client_metrics = control.get("client_metrics")
                    summary = recorder.finish(
                        status=completion_status,
                        error="browser reported a capture failure",
                        client_metrics=(
                            client_metrics if isinstance(client_metrics, dict) else None
                        ),
                        runtime_metrics=runtime_metrics(),
                    )
                    await send_json({"type": "summary", **summary.as_public_dict()})
                    await websocket.close(code=1011)
                    return
                elif control["type"] == "ping":
                    await send_json({"type": "pong"})
                elif control["type"] == "device_health":
                    device = control.get("device")
                    status = control.get("status")
                    reason = control.get("reason")
                    recorded_at_ms = control.get("recorded_at_ms")
                    if device not in {"camera", "microphone"}:
                        raise BrowserProtocolError("device_health contains an invalid device")
                    if status not in {"normal", "abnormal"}:
                        raise BrowserProtocolError("device_health contains an invalid status")
                    if (
                        not isinstance(recorded_at_ms, int)
                        or isinstance(recorded_at_ms, bool)
                        or recorded_at_ms < 0
                    ):
                        raise BrowserProtocolError(
                            "device_health contains an invalid recorded_at_ms"
                        )
                    if reason is not None and (
                        not isinstance(reason, str) or len(reason) > 100
                    ):
                        raise BrowserProtocolError("device_health contains an invalid reason")
                    device_event = {
                        "type": "device_health",
                        "device": device,
                        "status": status,
                        "reason": reason,
                        "recorded_at_ms": recorded_at_ms,
                    }
                    recorder.store.append_event(device_event)
                    await registry.observe(session_id, device_event)
                elif runtime_session is not None and await runtime_session.handle_control(control):
                    continue
                elif (
                    runtime_session is None
                    and control["type"] in REALTIME_ONLY_CONTROLS
                ):
                    await send_json(
                        {
                            "type": "control_unavailable",
                            "control_type": control["type"],
                            "reason": "realtime_loop_disabled",
                        }
                    )
                else:
                    raise BrowserProtocolError(
                        f"unsupported control message: {control['type']}"
                    )
        except WebSocketDisconnect:
            completion_status = "disconnected"
        except (BrowserProtocolError, ValueError, OSError) as exc:
            completion_status = "failed"
            if recorder is not None:
                recorder.finish(
                    status="failed",
                    error=str(exc),
                    runtime_metrics=runtime_metrics(),
                )
            try:
                await send_json({"type": "error", "message": str(exc)})
                await websocket.close(code=1003)
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            await stop_runtime(completion_status)
            if session_id is not None and session_authorized:
                enrollment = session_enrollments.pop(session_id, None)
                enrollment_locks.pop(session_id, None)
                cleanup_succeeded = await cleanup_remote_enrollment(enrollment)
                if recorder is not None and isinstance(
                    enrollment, TencentSpeakerEnrollment
                ):
                    cleanup_event = {
                        "type": "voiceprint_cleanup",
                        "provider": enrollment.model_name,
                        "remote_records_deleted": cleanup_succeeded,
                        "retry_queued": not cleanup_succeeded,
                    }
                    recorder.store.append_event(cleanup_event)
                    await registry.observe(session_id, cleanup_event)
                active_session_ids.discard(session_id)
                session_admissions.pop(session_id, None)
                await registry.mark_status(session_id, completion_status)
            if recorder is not None and not recorder.finished:
                recorder.finish(
                    status=completion_status,
                    error=(
                        "browser disconnected before a normal stop"
                        if completion_status == "disconnected"
                        else None
                    ),
                    runtime_metrics=runtime_metrics(),
                )

    return app


def run_browser_capture_server(
    *,
    host: str,
    port: int,
    output_dir: Path,
    access_token: str | None,
    research_access_token: str | None = None,
    log_level: str = "info",
    session_factory: RealtimeSessionFactory | None = None,
    voiceprint_service: TencentVoiceprintService | None = None,
) -> None:
    import uvicorn

    config = BrowserServerConfig(
        output_dir=output_dir,
        access_token=access_token,
        research_access_token=research_access_token,
    )
    uvicorn.run(
        create_browser_capture_app(
            config,
            session_factory=session_factory,
            voiceprint_service=voiceprint_service,
        ),
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
