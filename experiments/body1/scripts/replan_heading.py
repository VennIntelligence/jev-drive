#!/usr/bin/env python3
"""BODY1 arm 4.2, checklist item 7 (Mac side): heading error of the driven ego against the log per decision on decision 205's set (the
log-straight scenes of the 400 diagnosis scenes, start speed > 2 m/s) among the scenes both runs hold. experiments/alpasim/scripts/ot3_heading.py
unchanged, with its outputs redirected to this lane. Needs tmp/c1 (logged paths, labels only) and a decision table made on the box by
experiments/alpasim/scripts/m1_decisions.py --out <npz> base=<baseline run dir> rp=<switch-on run dir>.

  .venv/bin/python experiments/body1/scripts/replan_heading.py <name> <dec npz> [<dec npz> ...]
-> experiments/body1/results/replan/heading_<name>.{md,json}, experiments/body1/figs/replan/heading_growth_<name>.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/alpasim/scripts"))
import ot3_heading as H  # noqa: E402

H.RES, H.FIG = ROOT / "experiments/body1/results/replan", ROOT / "experiments/body1/figs/replan"
name, dec = sys.argv[1], sys.argv[2:]
sys.argv = ["ot3_heading.py", "--name", name, "--recipe", "base=base", "--recipe", "rp=rp", "--base", "base", "--dec", *dec]
H.main()
