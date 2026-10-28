"""Read the project settings from config.yaml."""

from pathlib import Path
from typing import Any

import yaml

from spamjam.constants import CONFIG_PATH


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the settings in config.yaml as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
