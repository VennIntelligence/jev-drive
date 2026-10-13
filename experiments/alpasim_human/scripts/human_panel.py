#!/usr/bin/env python3
"""Desktop control panel for the AlpaSim human exam: lib/exam_panel.py with the AlpaSim backend (lib/alpasim_exam.py).

    /usr/bin/python3 experiments/alpasim_human/scripts/human_panel.py      # system Python (Tk); or the desktop icon (human_panel.sh)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "lib"), str(HERE.parent / "lib")]
from exam_panel import panel  # noqa: E402
from alpasim_exam import AlpaSimExam  # noqa: E402

if __name__ == "__main__":
    panel(AlpaSimExam())
