from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from coregulation_poc.paths import STATE_CODEBOOK_PATH, resolve_project_path


def load_state_codebook(path: str | Path = STATE_CODEBOOK_PATH) -> dict[str, Any]:
    absolute_path = resolve_project_path(path)
    with absolute_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or "states" not in payload:
        raise ValueError(f"Invalid state codebook: {absolute_path}")
    return payload

