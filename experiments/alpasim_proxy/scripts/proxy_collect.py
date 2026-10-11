#!/usr/bin/env python3
"""PROXY: per-scene and per-decision tables of the anchor runs (LTF / DiffusionDrive samples, garage 0014), in the format of
aud1_drift_collect.py, whose readers are reused unchanged; only the run-dir lookup differs.

  proxy_collect.py --out DIR LABEL=GLOB [LABEL=GLOB ...]      (box, AlpaSim venv)
  e.g. LTF-sample='$DATA_DIR/runs/alpasim_proxy/anchors/ltf*/runs/LTF-*/2*'

Every run dir under a label is read in sorted order; the first run dir that holds a scene wins (aud1_collect.py's rule).
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "alpasim/scripts"))
import aud1_drift_collect as C  # noqa: E402


def main():
    specs = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    D = {lab: sorted(glob.glob(g)) for lab, g in (s.split("=", 1) for s in specs)}
    for lab, ds in D.items():
        print(lab, len(ds), "run dirs", flush=True)
        assert ds, f"no run dir for {lab}"
    C.dirs = lambda: D
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in specs]
    C.main()


if __name__ == "__main__":
    main()
