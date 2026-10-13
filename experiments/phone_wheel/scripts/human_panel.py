#!/usr/bin/env python
"""Desktop control panel for the Bench2Drive human exam: lib/exam_panel.py with the B2D backend (lib/b2d_exam.py).

    /usr/bin/python3 experiments/phone_wheel/scripts/human_panel.py      # system Python (envs/carla has no Tcl); or the desktop icon
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "lib"), str(HERE.parent / "lib")]
from exam_panel import panel  # noqa: E402
from b2d_exam import B2DExam  # noqa: E402

if __name__ == "__main__":
    panel(B2DExam())
