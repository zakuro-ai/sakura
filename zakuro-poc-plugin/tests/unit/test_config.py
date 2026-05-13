import json

import pytest
from pydantic import ValidationError

from zakuro_poc.config import ZakuroPocConfig, load_config


def test_built_in_defaults_load(monkeypatch, tmp_path):
    # Ensure no config files are found
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_config()
    assert config.artifact_root == "./zakuro-artifacts"
    assert config.max_timeout_seconds == 3600


def test_explicit_config_path_loads(tmp_path):
    config_path = tmp_path / "custom.json"
    config_path.write_text(
        json.dumps({"artifact_root": "/custom/path", "max_timeout_seconds": 120})
    )

    config = load_config(str(config_path))
    assert config.artifact_root == "/custom/path"
    assert config.max_timeout_seconds == 120


def test_invalid_json_fails_clearly(tmp_path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{ invalid json }")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_config(str(config_path))


def test_invalid_docker_network_mode_rejected():
    with pytest.raises(ValidationError):
        ZakuroPocConfig(docker={"network_mode": "host"})


def test_missing_config_falls_back_safely(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    config = load_config(str(missing_path))
    assert config.artifact_root == "./zakuro-artifacts"
