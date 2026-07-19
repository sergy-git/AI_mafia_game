"""Config loader for per-agent LLM profiles.

Resolves a per-agent generation profile (model tag, temperature, and other
generation parameters) from a JSON config file (`llm_config.json` by default).
Resolution is purely by `agent_id`: there is no separate role tier — the GM
is just another `agent_id` key. An explicit entry in `agents[agent_id]`
overrides `default` on a key-by-key basis; agents without an override fall
back entirely to `default`.

This module does no network/model I/O and can be tested in isolation.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG_PATH = Path(__file__).parent / "llm_config.json"


class ClientConfig:
    """Loads and resolves per-agent LLM generation profiles."""

    def __init__(self, config_path: Optional[str] = None):
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        with open(path, "r") as f:
            data = json.load(f)

        self._default: Dict[str, Any] = dict(data.get("default", {}))
        self._agents: Dict[str, Dict[str, Any]] = data.get("agents", {})

    def resolve(self, agent_id: str) -> Dict[str, Any]:
        """Return the resolved generation profile for the given `agent_id`.

        Keys present in `agents[agent_id]` override the same key in
        `default`; any key not overridden is taken from `default`.
        """
        override = self._agents.get(agent_id, {})
        return {**self._default, **override}
