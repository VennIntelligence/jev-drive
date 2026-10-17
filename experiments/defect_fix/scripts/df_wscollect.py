#!/usr/bin/env python3
"""Waiting-set outcome tables for finished reads (AlpaSim venv; box). experiments/alpasim_proxy wait_set.py `collect` run unedited, on a copy of
the set file whose variant-b windows without a logged human launch (h_launch_s null: 3 windows of waitset@v2) carry NaN instead, so the
collect step does not stop on them (their launch delay is NaN; defect_fix df.py ws_read leaves NaN delays out of the median).

  df_wscollect.py TAG [TAG ..]   -> $DATA_DIR/runs/alpasim_proxy/wait_set/reads/<TAG>/table/<TAG>.csv (only reads whose run finished)
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/alpasim_proxy/wait_set/reads"
SET = R / "experiments/alpasim_proxy/results/wait_set/set.json"

spec = importlib.util.spec_from_file_location("wait_set", R / "experiments/alpasim_proxy/scripts/wait_set.py")
WSM = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:1] + sys.argv[1:]
spec.loader.exec_module(WSM)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args()
    S = json.loads(SET.read_text())
    for w in S["windows"]:
        if w["variant"] == "b" and w.get("h_launch_s") is None:
            w["h_launch_s"] = float("nan")
    pat = DATA / "runs/defect_fix/ws_set_patched.json"
    pat.write_text(json.dumps(S))
    for t in a.tags:
        run = W / t / "run"
        if not (run / "aggregate/results-summary.json").exists():
            print(t, "no finished run", flush=True)
            continue
        WSM.cmd_collect(argparse.Namespace(set=str(pat), run=str(run), label=t, out=str(W / t / "table")))
