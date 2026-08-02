from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from coregulation_poc.paths import resolve_project_path

SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunArtifactStore:
    """Write one immutable, ignored-by-Git audit folder per API test."""

    def __init__(self, output_dir: Path, session_id: str) -> None:
        safe_session = SAFE_ID.sub("_", session_id).strip("_") or "session"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{safe_session}_{uuid4().hex[:8]}"
        self.run_dir = (resolve_project_path(output_dir) / "runs" / run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = (self.run_dir / "events.jsonl").resolve()

    def write_json(self, filename: str, value: Any) -> Path:
        path = (self.run_dir / filename).resolve()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def write_text(self, filename: str, value: str) -> Path:
        path = (self.run_dir / filename).resolve()
        path.write_text(value, encoding="utf-8")
        return path

    def append_event(self, value: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
