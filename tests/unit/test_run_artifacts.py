from pathlib import Path

from coregulation_poc.storage.run_artifacts import RunArtifactStore, sha256_file


def test_artifact_store_uses_absolute_isolated_directory(tmp_path: Path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"research-test")

    store = RunArtifactStore(tmp_path, "P01/E01")
    output = store.write_json("result.json", {"ok": True})

    assert store.run_dir.is_absolute()
    assert output.is_file()
    assert sha256_file(source) == "40cae87d602e74aac82edcdaac42e02ce9a85ed205ac53aaeace205019f4cb2e"


def test_artifact_store_uses_readable_study_name_and_avoids_collision(
    tmp_path: Path,
) -> None:
    run_name = "P012_20260810_192110_正式实验_T1"

    first = RunArtifactStore(tmp_path, "internal-session", run_name=run_name)
    second = RunArtifactStore(tmp_path, "internal-session", run_name=run_name)

    assert first.run_dir.name == run_name
    assert second.run_dir.name == f"{run_name}_02"
