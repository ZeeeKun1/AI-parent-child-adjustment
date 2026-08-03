from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from coregulation_poc.capture.devices import list_windows_media_devices
from coregulation_poc.capture.media import MediaFormat, MediaSourceError
from coregulation_poc.codebook import load_state_codebook
from coregulation_poc.connection_check import check_realtime_connection
from coregulation_poc.control import load_intervention_policy
from coregulation_poc.delivery import load_delivery_policy
from coregulation_poc.delivery_test import run_delivery_test
from coregulation_poc.diagnostics import run_all_diagnostics
from coregulation_poc.intervention import load_strategy_library
from coregulation_poc.live_test import run_live_test
from coregulation_poc.paths import (
    DELIVERY_POLICY_PATH,
    INTERVENTION_POLICY_PATH,
    PROJECT_ROOT,
    STATE_CODEBOOK_PATH,
    STRATEGY_CARDS_PATH,
)
from coregulation_poc.settings import Settings
from coregulation_poc.strategy_test import run_strategy_test
from coregulation_poc.trajectory_test import run_trajectory_test
from coregulation_poc.video_test import run_video_test
from coregulation_poc.web.app import run_browser_capture_server


def doctor() -> int:
    settings = Settings()
    codebook = load_state_codebook()
    intervention_policy = load_intervention_policy()
    strategy_library = load_strategy_library()
    delivery_policy = load_delivery_policy()
    report = {
        "project_root": str(PROJECT_ROOT),
        "state_codebook": str(STATE_CODEBOOK_PATH),
        "state_labels": list(codebook["states"]),
        "intervention_policy": str(INTERVENTION_POLICY_PATH),
        "intervention_policy_version": intervention_policy.version,
        "strategy_library": str(STRATEGY_CARDS_PATH),
        "strategy_library_version": strategy_library.version,
        "strategy_card_count": len(strategy_library.cards),
        "strategy_target_actors": sorted(
            {card.target_actor.value for card in strategy_library.cards}
        ),
        "delivery_policy": str(DELIVERY_POLICY_PATH),
        "delivery_policy_version": delivery_policy.version,
        "delivery_modalities": ["visual_text", "spoken_voice"],
        "tts_provider": delivery_policy.voice.provider,
        "tts_model": delivery_policy.voice.model,
        "tts_voice": delivery_policy.voice.voice,
        "tts_realtime_base_url": settings.tts_realtime_base_url,
        "input_dir": str(settings.input_dir),
        "output_dir": str(settings.output_dir),
        "cache_dir": str(settings.cache_dir),
        "log_dir": str(settings.log_dir),
        "api_key_configured": settings.dashscope_api_key is not None,
        "workspace_configured": settings.aliyun_workspace_id is not None,
        "realtime_endpoint_configured": settings.realtime_endpoint is not None,
        "live_capture_backend": "windows_directshow",
        "live_capture_command": "live-test --dry-run",
        "live_capture_api_called": False,
        "live_capture_raw_media_saved": False,
        "browser_capture_command": "web-live --host 127.0.0.1 --port 8000",
        "browser_capture_raw_media_saved": False,
        "browser_capture_api_called": False,
        "browser_capture_access_token_configured": (
            settings.browser_capture_access_token is not None
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-regulation realtime PoC utilities")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="Check local configuration")
    subparsers.add_parser("connection-test", help="Test WebSocket authentication only")
    diagnose_parser = subparsers.add_parser("diagnose", help="Run all preflight checks")
    diagnose_parser.add_argument("--video", type=Path, required=True)
    video_parser = subparsers.add_parser("video-test", help="Replay one clip to Qwen Omni")
    video_parser.add_argument("--video", type=Path, required=True)
    video_parser.add_argument("--session-id", required=True)
    video_parser.add_argument("--dry-run", action="store_true")
    live_parser = subparsers.add_parser(
        "live-test",
        help="Capture a short camera and microphone session without calling an API",
    )
    live_parser.add_argument("--list-devices", action="store_true")
    camera_selector = live_parser.add_mutually_exclusive_group()
    camera_selector.add_argument("--camera-index", type=int)
    camera_selector.add_argument("--camera-name")
    microphone_selector = live_parser.add_mutually_exclusive_group()
    microphone_selector.add_argument("--microphone-index", type=int)
    microphone_selector.add_argument("--microphone-name")
    live_parser.add_argument("--session-id", default="live_dry_run")
    live_parser.add_argument("--duration-seconds", type=float, default=10.0)
    live_parser.add_argument("--dry-run", action="store_true")
    live_parser.add_argument("--audio-chunk-ms", type=int, default=100)
    live_parser.add_argument("--image-interval-ms", type=int, default=1000)
    live_parser.add_argument("--max-audio-queue-chunks", type=int, default=100)
    live_parser.add_argument("--max-image-queue-chunks", type=int, default=10)
    live_parser.add_argument("--camera-width", type=int)
    live_parser.add_argument("--camera-height", type=int)
    live_parser.add_argument("--camera-fps", type=int)
    web_live_parser = subparsers.add_parser(
        "web-live",
        help="Serve a browser camera and microphone capture page",
    )
    web_live_parser.add_argument("--host", default="127.0.0.1")
    web_live_parser.add_argument("--port", type=int, default=8000)
    web_live_parser.add_argument("--output-dir", type=Path)
    web_live_parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
    )
    trajectory_parser = subparsers.add_parser(
        "trajectory-test",
        help="Replay module-one assessments through the intervention controller",
    )
    trajectory_parser.add_argument("--input", type=Path, required=True)
    strategy_parser = subparsers.add_parser(
        "strategy-test",
        help="Replay assessments through timing control and target-aware strategy selection",
    )
    strategy_parser.add_argument("--input", type=Path, required=True)
    delivery_parser = subparsers.add_parser(
        "delivery-test",
        help="Replay assessments through timing, strategy and dual-channel delivery",
    )
    delivery_parser.add_argument("--input", type=Path, required=True)
    delivery_parser.add_argument(
        "--disable-voice",
        action="store_true",
        help="Exercise the visual-only fallback path",
    )
    delivery_parser.add_argument(
        "--synthesize-voice",
        action="store_true",
        help="Generate and save the approved message with Qwen realtime TTS and Maia",
    )
    args = parser.parse_args()
    if args.command in {None, "doctor"}:
        raise SystemExit(doctor())
    if args.command == "connection-test":
        result = check_realtime_connection(Settings())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ok"] else 2)
    if args.command == "diagnose":
        if not args.video.is_absolute():
            parser.error("--video must be an absolute path")
        report_path, valid = asyncio.run(
            run_all_diagnostics(video_path=args.video, settings=Settings())
        )
        print(
            json.dumps(
                {"report": str(report_path), "valid": valid}, ensure_ascii=False, indent=2
            )
        )
        raise SystemExit(0 if valid else 2)
    if args.command == "web-live":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        settings = Settings()
        output_dir = args.output_dir or settings.output_dir
        if not output_dir.is_absolute():
            parser.error("--output-dir must be an absolute path")
        access_token = (
            settings.browser_capture_access_token.get_secret_value()
            if settings.browser_capture_access_token is not None
            else None
        )
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if args.host not in local_hosts and access_token is None:
            parser.error(
                "BROWSER_CAPTURE_ACCESS_TOKEN is required when --host is not localhost"
            )
        if access_token is not None and len(access_token) < 12:
            parser.error("BROWSER_CAPTURE_ACCESS_TOKEN must contain at least 12 characters")
        print(
            json.dumps(
                {
                    "status": "starting",
                    "open": f"http://{args.host}:{args.port}",
                    "https_required_outside_localhost": True,
                    "raw_media_saved": False,
                    "api_called": False,
                    "access_control_required": access_token is not None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        try:
            run_browser_capture_server(
                host=args.host,
                port=args.port,
                output_dir=output_dir,
                access_token=access_token,
                log_level=args.log_level,
            )
        except OSError as exc:
            parser.exit(2, f"Browser capture server failed: {exc}\n")
        raise SystemExit(0)
    if args.command == "video-test":
        if not args.video.is_absolute():
            parser.error("--video must be an absolute path")
        try:
            run_dir, valid = asyncio.run(
                run_video_test(
                    video_path=args.video,
                    session_id=args.session_id,
                    settings=Settings(),
                    dry_run=args.dry_run,
                )
            )
        except (ValueError, OSError) as exc:
            parser.exit(2, f"Video test failed: {exc}\n")
        print(json.dumps({"run_dir": str(run_dir), "valid": valid}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if valid else 2)
    if args.command == "live-test":
        try:
            if args.list_devices:
                inventory = list_windows_media_devices()
                print(json.dumps(inventory.as_public_dict(), ensure_ascii=False, indent=2))
                raise SystemExit(0)
            if not args.dry_run:
                parser.error("live-test currently requires --dry-run; it does not call a paid API")
            if args.camera_index is None and args.camera_name is None:
                parser.error("select a camera with --camera-index or --camera-name")
            if args.microphone_index is None and args.microphone_name is None:
                parser.error("select a microphone with --microphone-index or --microphone-name")
            if (args.camera_width is None) != (args.camera_height is None):
                parser.error("--camera-width and --camera-height must be provided together")
            run_dir, valid = run_live_test(
                session_id=args.session_id,
                settings=Settings(),
                duration_seconds=args.duration_seconds,
                camera_index=args.camera_index,
                camera_name=args.camera_name,
                microphone_index=args.microphone_index,
                microphone_name=args.microphone_name,
                media_format=MediaFormat(
                    audio_chunk_ms=args.audio_chunk_ms,
                    image_interval_ms=args.image_interval_ms,
                ),
                max_audio_queue_chunks=args.max_audio_queue_chunks,
                max_image_queue_chunks=args.max_image_queue_chunks,
                requested_camera_width=args.camera_width,
                requested_camera_height=args.camera_height,
                requested_camera_fps=args.camera_fps,
            )
        except (MediaSourceError, ValueError, OSError) as exc:
            parser.exit(2, f"Live test failed: {exc}\n")
        print(
            json.dumps(
                {"run_dir": str(run_dir), "valid": valid},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(0 if valid else 2)
    if args.command == "trajectory-test":
        if not args.input.is_absolute():
            parser.error("--input must be an absolute path")
        try:
            run_dir, valid = run_trajectory_test(
                input_path=args.input,
                settings=Settings(),
            )
        except (ValueError, OSError) as exc:
            parser.exit(2, f"Trajectory test failed: {exc}\n")
        print(
            json.dumps(
                {"run_dir": str(run_dir), "valid": valid},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(0 if valid else 2)
    if args.command == "strategy-test":
        if not args.input.is_absolute():
            parser.error("--input must be an absolute path")
        try:
            run_dir, valid = run_strategy_test(
                input_path=args.input,
                settings=Settings(),
            )
        except (ValueError, OSError) as exc:
            parser.exit(2, f"Strategy test failed: {exc}\n")
        print(
            json.dumps(
                {"run_dir": str(run_dir), "valid": valid},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(0 if valid else 2)
    if args.command == "delivery-test":
        if not args.input.is_absolute():
            parser.error("--input must be an absolute path")
        if args.disable_voice and args.synthesize_voice:
            parser.error("--disable-voice and --synthesize-voice cannot be used together")
        try:
            run_dir, valid = run_delivery_test(
                input_path=args.input,
                settings=Settings(),
                voice_enabled=not args.disable_voice,
                synthesize_voice=args.synthesize_voice,
            )
        except (ValueError, OSError) as exc:
            parser.exit(2, f"Delivery test failed: {exc}\n")
        print(
            json.dumps(
                {"run_dir": str(run_dir), "valid": valid},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(0 if valid else 2)
