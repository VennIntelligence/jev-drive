#!/usr/bin/env python3
"""zeros_mech's zm_read.py run unedited, with jevdrive.run's default root moved to $DATA_DIR/runs/onpol_long/zm so that its Run directories
(collect / rows) land under this topic and never shadow the zeros_mech lane's latest run (onpol_long Amendment 14). Arguments go to zm_read.py.

  $DATA_DIR/third_party/alpasim/.venv/bin/python experiments/onpol_long/scripts/ol_zm.py collect --runs <label>=<run dir> ...
"""
import os
import runpy
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R))
import jevdrive.run as JR  # noqa: E402

JR._default_root = lambda: Path(os.environ["DATA_DIR"]) / "runs/onpol_long/zm"
ZM = str(R / "experiments/zeros_mech/scripts/zm_read.py")
sys.argv = [ZM] + sys.argv[1:]
runpy.run_path(ZM, run_name="__main__")
