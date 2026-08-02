from __future__ import annotations

import asyncio
import json
import re
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from coregulation_poc.capture.video_replay import MediaKind, decode_video_for_replay
from coregulation_poc.connection_check import check_realtime_connection
from coregulation_poc.paths import resolve_project_path
from coregulation_poc.protocol_diagnostics import probe_session_update_variants
from coregulation_poc.providers.qwen_omni_realtime import QwenOmniRealtimeProvider
from coregulation_poc.settings import Settings

ProgressReporter = Callable[[str], None]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "video"


def _write_report(settings: Settings, video_path: Path, report: dict[str, Any]) -> Path:
    output_dir = (resolve_project_path(settings.output_dir) / "diagnostics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = (output_dir / f"{timestamp}_{_safe_name(video_path.stem)}.json").resolve()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def run_all_diagnostics(
    *,
    video_path: Path,
    settings: Settings,
    progress: ProgressReporter = print,
) -> tuple[Path, bool]:
    """Run dependent checks in order and preserve a machine-readable report."""
    resolved_video = video_path.expanduser().resolve()
    checks: list[dict[str, Any]] = []
    total = 7

    def record(index: int, name: str, ok: bool | None, detail: Any) -> None:
        status = "SKIPPED" if ok is None else "PASS" if ok else "FAIL"
        progress(f"[{index}/{total}] {name}: {status}")
        if detail:
            progress(f"    {detail}")
        checks.append(
            {"index": index, "name": name, "status": status, "ok": ok is True, "detail": detail}
        )

    key = settings.dashscope_api_key.get_secret_value() if settings.dashscope_api_key else ""
    config_ok = (
        key.startswith("sk-")
        and bool(settings.aliyun_workspace_id)
        and settings.aliyun_region in {"cn-beijing", "ap-southeast-1"}
        and bool(settings.realtime_endpoint)
    )
    record(
        1,
        "configuration",
        config_ok,
        {
            "api_key_format": "valid" if key.startswith("sk-") else "invalid",
            "workspace_configured": bool(settings.aliyun_workspace_id),
            "region": settings.aliyun_region,
            "model": settings.omni_model,
        },
    )

    media = None
    try:
        media = await asyncio.to_thread(decode_video_for_replay, resolved_video)
        record(
            2,
            "video_and_audio_decode",
            True,
            {
                "duration_ms": media.metadata.duration_ms,
                "resolution": f"{media.metadata.width}x{media.metadata.height}",
                "video_codec": media.metadata.video_codec,
                "audio_codec": media.metadata.audio_codec,
                "audio_chunks": media.audio_chunk_count,
                "image_frames": media.image_count,
            },
        )
    except (ValueError, OSError) as exc:
        record(2, "video_and_audio_decode", False, str(exc))

    host = urlparse(settings.realtime_endpoint or "").hostname
    addresses: list[str] = []
    try:
        if not host:
            raise ValueError("Realtime endpoint host is missing.")
        address_info = await asyncio.to_thread(socket.getaddrinfo, host, 443)
        addresses = sorted({item[4][0] for item in address_info})
        record(3, "dns_resolution", bool(addresses), {"host": host, "addresses": addresses})
    except (OSError, ValueError) as exc:
        record(3, "dns_resolution", False, str(exc))

    tcp_ok = False
    try:
        if not host:
            raise ValueError("Realtime endpoint host is missing.")
        tcp_socket = await asyncio.to_thread(socket.create_connection, (host, 443), 10)
        tcp_socket.close()
        tcp_ok = True
        record(4, "tcp_443", True, {"host": host, "port": 443})
    except (OSError, ValueError) as exc:
        record(4, "tcp_443", False, str(exc))

    native_result: dict[str, Any] = {"ok": False, "error": "prerequisite_failed"}
    if config_ok and tcp_ok:
        native_result = await asyncio.to_thread(check_realtime_connection, settings)
    record(5, "websocket_authentication", bool(native_result.get("ok")), native_result)

    session_ok = False
    provider: QwenOmniRealtimeProvider | None = None
    if config_ok and native_result.get("ok"):
        protocol_result = await asyncio.to_thread(
            probe_session_update_variants,
            endpoint=settings.realtime_endpoint,
            api_key=key,
            workspace_id=settings.aliyun_workspace_id,
            instructions="Connection diagnostic only. Do not generate a response.",
            timeout_seconds=settings.connection_timeout_seconds,
        )
        for probe in protocol_result["probes"]:
            progress(
                f"    Probe {probe['index']}/5 {probe['name']}: {probe['status']}"
            )
            progress(
                "        sent: "
                + json.dumps(probe["sent_event"], ensure_ascii=False, separators=(",", ":"))
            )
            if probe.get("close"):
                progress(f"        close: {probe['close']}")
            elif probe.get("error"):
                progress(f"        error: {probe['error']}")

        provider = QwenOmniRealtimeProvider(
            model=settings.omni_model,
            api_key=key,
            workspace_id=settings.aliyun_workspace_id or "",
            base_url=settings.realtime_base_url or "",
            instructions="Connection diagnostic only. Do not generate a response.",
            connection_timeout_seconds=settings.connection_timeout_seconds,
        )
        try:
            await provider.connect()
            session_ok = True
            record(
                6,
                "session_update",
                True,
                {
                    "active_provider": "session.updated received",
                    "protocol_probes": protocol_result,
                },
            )
        except (TimeoutError, OSError, ConnectionError) as exc:
            record(
                6,
                "session_update",
                False,
                {"active_provider_error": str(exc), "protocol_probes": protocol_result},
            )
    else:
        protocol_result = probe_session_update_variants(
            endpoint=None,
            api_key="",
            workspace_id=None,
            instructions="Connection diagnostic only. Do not generate a response.",
        )
        for probe in protocol_result["probes"]:
            progress(
                f"    Probe {probe['index']}/5 {probe['name']}: {probe['status']}"
            )
            progress(
                "        not sent: "
                + json.dumps(probe["sent_event"], ensure_ascii=False, separators=(",", ":"))
            )
        record(
            6,
            "session_update",
            None,
            {
                "reason": "WebSocket authentication prerequisite failed.",
                "protocol_probes": protocol_result,
            },
        )

    payload_ok = False
    if session_ok and provider is not None and media is not None:
        try:
            audio = next(chunk for chunk in media.chunks if chunk.kind is MediaKind.AUDIO)
            image = next(chunk for chunk in media.chunks if chunk.kind is MediaKind.IMAGE)
            await provider.send_audio(audio.payload, audio.timestamp_ms)
            await provider.send_frame(image.payload, image.timestamp_ms)
            await provider.commit_input()
            payload_ok = True
            record(7, "audio_image_buffer", True, "input_audio_buffer.committed received")
        except (TimeoutError, OSError, ConnectionError, StopIteration) as exc:
            record(7, "audio_image_buffer", False, str(exc))
    else:
        record(7, "audio_image_buffer", None, "Skipped because a prerequisite failed.")

    if provider is not None:
        await provider.close()

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "video_filename": resolved_video.name,
        "all_passed": all(check["ok"] for check in checks),
        "checks": checks,
    }
    report_path = _write_report(settings, resolved_video, report)
    progress(f"Diagnostic report: {report_path}")
    return report_path, bool(report["all_passed"] and payload_ok)
