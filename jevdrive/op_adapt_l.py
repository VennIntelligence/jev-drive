"""op-adapt L: log-imitation adaptation of openpilot Cinque (todos/2026-10-01-op-adapt-L-prereg.md)."""
from __future__ import annotations

import os
from pathlib import Path

from .common import data_dir


def lroot(*p) -> Path:
    """Lane root $DATA_DIR/runs/op_adapt_L (OP_L_ROOT overrides it: tests)."""
    d = Path(os.environ.get("OP_L_ROOT") or data_dir() / "runs" / "op_adapt_L") / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d
