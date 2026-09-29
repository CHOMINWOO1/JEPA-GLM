from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from jepa_glm.utils.validation import validate_config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config
