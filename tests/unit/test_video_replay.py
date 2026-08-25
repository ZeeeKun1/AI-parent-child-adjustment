import asyncio
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from coregulation_poc.capture.video_replay import MediaKind, decode_video_for_replay
from coregulation_poc.settings import Settings
from coregulation_poc.video_test import run_video_test


def _write_synthetic_clip(path: Path, *, frequency_hz: float | None = None) -> None:
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("mpeg4", rate=10)
        video.width = 160
        video.height = 120
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("pcm_s16le", rate=16_000)
        audio.layout = "mono"

        for index in range(10):
            image = np.full((120, 160, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)

        if frequency_hz is None:
            samples = np.zeros((1, 16_000), dtype=np.int16)
        else:
            time = np.arange(16_000, dtype=np.float64) / 16_000
            waveform = 0.25 * np.sin(2 * np.pi * frequency_hz * time)
            samples = np.round(waveform * 32767).astype(np.int16).reshape(1, -1)
        audio_frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        audio_frame.sample_rate = 16_000
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, 16_000)
        for packet in audio.encode(audio_frame):
            container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)


def test_decode_video_produces_synchronized_api_chunks(tmp_path: Path) -> None:
    clip = tmp_path / "synthetic.mkv"
    _write_synthetic_clip(clip)

    media = decode_video_for_replay(clip)

    assert media.audio_chunk_count == 10
    assert media.image_count == 1
    assert media.chunks[0].kind is MediaKind.AUDIO
    assert all(chunk.timestamp_ms >= 0 for chunk in media.chunks)
    image_chunk = next(chunk for chunk in media.chunks if chunk.kind is MediaKind.IMAGE)
    image = cv2.imdecode(np.frombuffer(image_chunk.payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert media.image_timestamp_labels is True
    assert image.shape[:2] == (160, 160)
    assert image[:40].max() > 200


def test_dry_run_writes_traceable_artifacts_without_api(tmp_path: Path) -> None:
    clip = tmp_path / "synthetic.mkv"
    _write_synthetic_clip(clip)
    settings = Settings(output_dir=tmp_path / "output")

    run_dir, valid = asyncio.run(
        run_video_test(
            video_path=clip,
            session_id="S01",
            settings=settings,
            dry_run=True,
        )
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    acoustics = json.loads((run_dir / "acoustic_summary.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert valid is True
    assert manifest["source"]["filename"] == "synthetic.mkv"
    assert manifest["research_basis"]["codebook_version"] == 3
    assert manifest["research_basis"]["acoustic_analysis_version"] == 1
    assert manifest["media"]["image_timestamp_labels"] is True
    assert acoustics["quality"] == "insufficient"
    assert acoustics["interpretation_role"] == "supporting_measurement_only"
    assert result == {"status": "ready", "api_called": False}


def test_local_acoustic_analysis_measures_pitch_without_inferring_state(tmp_path: Path) -> None:
    from coregulation_poc.acoustics import analyze_replay_audio

    clip = tmp_path / "tone.mkv"
    _write_synthetic_clip(clip, frequency_hz=220.0)
    media = decode_video_for_replay(clip)

    analysis = analyze_replay_audio(media)

    assert analysis.quality == "limited"
    assert analysis.actor == "unknown"
    assert analysis.pitch.available is True
    assert analysis.pitch.median_hz == pytest.approx(220.0, abs=2.0)
    assert analysis.intensity.rms_dbfs is not None
    assert analysis.speech_rate.available is False
