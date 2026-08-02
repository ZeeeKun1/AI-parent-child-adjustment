from __future__ import annotations

from pathlib import Path

import yaml

from coregulation_poc.intervention.models import StrategyCard, StrategyLibraryConfig
from coregulation_poc.paths import STRATEGY_CARDS_PATH, resolve_project_path


def load_strategy_library(
    path: str | Path = STRATEGY_CARDS_PATH,
) -> StrategyLibraryConfig:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return StrategyLibraryConfig.model_validate(payload)


def cards_by_id(library: StrategyLibraryConfig) -> dict[str, StrategyCard]:
    return {card.strategy_id: card for card in library.cards}
