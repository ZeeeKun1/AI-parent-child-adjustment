from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent.resolve()
PROJECT_ROOT = SRC_DIR.parent.resolve()
CONFIG_DIR = (PROJECT_ROOT / "config").resolve()
PROMPT_DIR = (CONFIG_DIR / "prompts").resolve()
DATA_DIR = (PROJECT_ROOT / "data").resolve()
DEFAULT_INPUT_DIR = (DATA_DIR / "input").resolve()
DEFAULT_OUTPUT_DIR = (DATA_DIR / "output").resolve()
DEFAULT_CACHE_DIR = (DATA_DIR / "cache").resolve()
DEFAULT_LOG_DIR = (DATA_DIR / "logs").resolve()
STATE_CODEBOOK_PATH = (CONFIG_DIR / "state_codebook.yaml").resolve()
INTERVENTION_POLICY_PATH = (CONFIG_DIR / "intervention_policy.yaml").resolve()
ACOUSTIC_ANALYSIS_PATH = (CONFIG_DIR / "acoustic_analysis.yaml").resolve()
STRATEGY_CARDS_PATH = (CONFIG_DIR / "strategy_cards.yaml").resolve()
DELIVERY_POLICY_PATH = (CONFIG_DIR / "delivery_policy.yaml").resolve()
ENV_FILE = (PROJECT_ROOT / ".env").resolve()


def resolve_project_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    """Return an absolute normalized path while keeping the project movable."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def ensure_runtime_directories(*paths: Path) -> tuple[Path, ...]:
    """Create explicitly supplied runtime directories and return absolute paths."""
    resolved = tuple(resolve_project_path(path) for path in paths)
    for path in resolved:
        path.mkdir(parents=True, exist_ok=True)
    return resolved
