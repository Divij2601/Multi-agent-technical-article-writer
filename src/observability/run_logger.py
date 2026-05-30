from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logging_utils import dump_json


def write_run_log(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / "run_log.json"
    dump_json(path, payload)
    return path
