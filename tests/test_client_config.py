"""Unit tests for `ClientConfig`.

No network/model mocking needed — these test pure config-file resolution.
"""

import json

import pytest

from client_config import ClientConfig


@pytest.fixture
def config_path(tmp_path):
    def _write(data):
        path = tmp_path / "llm_config.json"
        path.write_text(json.dumps(data))
        return str(path)

    return _write


def test_loads_default_profile_from_file(config_path):
    path = config_path(
        {
            "default": {"model": "default-model", "temperature": 0.7},
            "agents": {},
        }
    )
    config = ClientConfig(config_path=path)

    assert config.resolve("unconfigured_agent") == {
        "model": "default-model",
        "temperature": 0.7,
    }


def test_explicit_agent_override_replaces_matching_keys(config_path):
    path = config_path(
        {
            "default": {"model": "default-model", "temperature": 0.7},
            "agents": {
                "gm": {"model": "gm-model", "temperature": 0.1},
            },
        }
    )
    config = ClientConfig(config_path=path)

    assert config.resolve("gm") == {"model": "gm-model", "temperature": 0.1}


def test_partial_agent_override_falls_back_to_default_for_other_keys(config_path):
    path = config_path(
        {
            "default": {"model": "default-model", "temperature": 0.7, "num_predict": 512},
            "agents": {
                "gm": {"temperature": 0.1},
            },
        }
    )
    config = ClientConfig(config_path=path)

    assert config.resolve("gm") == {
        "model": "default-model",
        "temperature": 0.1,
        "num_predict": 512,
    }


def test_agent_without_override_falls_back_to_default(config_path):
    path = config_path(
        {
            "default": {"model": "default-model", "temperature": 0.7},
            "agents": {
                "gm": {"temperature": 0.1},
            },
        }
    )
    config = ClientConfig(config_path=path)

    assert config.resolve("some_player") == {
        "model": "default-model",
        "temperature": 0.7,
    }
