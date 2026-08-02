from coregulation_poc.paths import (
    INTERVENTION_POLICY_PATH,
    PROJECT_ROOT,
    STATE_CODEBOOK_PATH,
    resolve_project_path,
)


def test_project_paths_are_absolute() -> None:
    assert PROJECT_ROOT.is_absolute()
    assert STATE_CODEBOOK_PATH.is_absolute()
    assert INTERVENTION_POLICY_PATH.is_absolute()
    assert resolve_project_path("data/input").is_absolute()


def test_relative_paths_follow_project_root() -> None:
    assert resolve_project_path("data/input") == (PROJECT_ROOT / "data" / "input").resolve()
