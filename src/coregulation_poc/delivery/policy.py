from __future__ import annotations

from pathlib import Path

import yaml

from coregulation_poc.delivery.models import DeliveryPolicy
from coregulation_poc.paths import DELIVERY_POLICY_PATH, resolve_project_path


def load_delivery_policy(path: str | Path = DELIVERY_POLICY_PATH) -> DeliveryPolicy:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return DeliveryPolicy.model_validate(payload)
