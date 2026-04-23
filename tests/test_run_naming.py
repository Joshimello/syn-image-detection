from pathlib import Path

from me26sid.config import load_settings


def test_load_settings_appends_timestamp_to_run_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train:\n  run_name: baseline\n", encoding="utf-8")

    settings = load_settings(config_path)

    assert settings.train.run_name.startswith("baseline_")
    assert len(settings.train.run_name) > len("baseline_")


def test_load_settings_respects_run_name_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train:\n  run_name: baseline\n", encoding="utf-8")

    settings = load_settings(config_path, run_name_override="manual_run")

    assert settings.train.run_name == "manual_run"
