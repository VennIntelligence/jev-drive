#!/usr/bin/env python3
"""PAI-PARTIALS: where the rear-hit and slow partials of NTV0-L18-s0 + ACC + GAP (stored PAI 281 read A0) lose their distance, serve
side or model plan (plans/2026-10-20-pai-partials-sweep-prereg.md, part 1). Box, repo .venv; stored reads are read, never written.

  pai_partials.py decomp --out $DATA_DIR/runs/alpasim_proxy/pai_partials/decomp.csv

Per partial scene (d381's classes, route_follow/pai_gap/scenes.csv) and per inference decision of A0 from the handoff (1.6 s) to the rear
hit (rear-hit) or the rollout's end (slow):
  v_h    the human's speed 1.5 s after the decision, time-matched (the logged follower is log replay, so what lets it hit is the time lag)
  v15    the served speed at 1.5 s (fix.acc.v15), v_model the model plan's own speed at 1.5 s (fix.acc.v_model)
  deficit = max(v_h - v15, 0); serve part = clip(min(v_model, v_h) - v15, 0, deficit) (speed the plan had and the serve removed), labelled
         by the binding term at 1.5 s (fix.acc.near: lead real / lead phantom (pai_gap.cut_class) / curve / set / ramp); model part = the rest
  serve add = max(min(v15, v_h) - v_model, 0) (ACC serving faster than the plan, up to the human)
Integrated over the window x 0.1 s -> metres. Lag = the human's log station minus ours at the window's end (time-matched).
Scene cause: serve if serve metres >= model metres (sub-cause = the term with the most serve metres), else model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/defect_fix/scripts"), str(REPO / "experiments/wait_cap/scripts")]
import pai_gap as PG  # noqa: E402
import rf_tax as TX  # noqa: E402

D = os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs") + "/runs"
A0, GAP = PG.A0, f"{D}/route_follow/pai_gap"
DT = 0.1
TERMS = ("lead-real", "lead-phantom", "curve", "set", "ramp")


def scene_job(job):
    s, o, recs, cls, rear_t = job
    R = TX.Roll(o)
    th = R.t[0] + PG.HAND
    te = R.t0 + 1e6 * rear_t if np.isfinite(rear_t) else R.t[-1]
    vh = lambda t: float(np.interp(t, R.LT, R.Lv))  # noqa: E731
    acc_m = defaultdict(float)
    n = 0
    for r in recs:
        if not (th <= r["t0"] <= te):
            continue
        a = (r.get("fix") or {}).get("acc") or {}
        if a.get("v15") is None:
            continue
        n += 1
        v15, vm, h = float(a["v15"]), float(a.get("v_model") or 0.0), vh(r["t0"] + 1.5e6)
        d = max(h - v15, 0.0)
        sv = float(np.clip(min(vm, h) - v15, 0.0, d))
        acc_m["deficit"] += d * DT
        acc_m["model"] += (d - sv) * DT
        acc_m["add"] += max(min(v15, h) - vm, 0.0) * DT
        if sv > 0:
            k = a.get("near") or "set"
            if k == "lead":
                k = "lead-real" if PG.cut_class(R, r).get("cls") == "real" else "lead-phantom"
            elif k == "model":                                    # served = min(ACC, model) under the guard: the ramp is what is below
                k = "ramp"
            acc_m[k] += sv * DT
    sh = lambda t: float(np.interp(t, R.LT, R.LS[:len(R.LT)]))  # noqa: E731
    lag = sh(te) - float(R.log_station(R.pose(te)[None, :2])[0])
    serve = sum(acc_m[k] for k in TERMS)
    sub = max(TERMS, key=lambda k: acc_m[k]) if serve > 0 else ""
    return dict(scene=s, cls=cls, n_dec=n, lag_m=lag, deficit_m=acc_m["deficit"], model_m=acc_m["model"], serve_m=serve, add_m=acc_m["add"],
                **{f"{k}_m": acc_m[k] for k in TERMS}, cause=(f"serve:{sub}" if serve >= acc_m["model"] and serve > 0 else "model"))


def cmd_decomp(a):
    T = pd.read_csv(f"{GAP}/scenes.csv", index_col=0)
    P = T[T.cls.isin(["partial rear-hit", "partial slow"])]
    A = PG.rolls(f"{PG.PIL}/tax/A0.pkl")
    DR = PG.drive(A0)
    jobs = [(s, A[s], DR.get(s, []), x.cls, x.rear_t_s if x.cls == "partial rear-hit" else np.nan) for s, x in P.iterrows()]
    with ProcessPoolExecutor(min(len(jobs), 32)) as ex:
        out = pd.DataFrame(list(ex.map(scene_job, jobs)))
    keep = ["score", "stratum", "primary", "v_hand", "N0", "NTV0-s0-hl", "NTV0-s1", "best", "H"]
    out = out.join(P[keep], on="scene")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["scene"]).round(2).to_string())
    print(json.dumps(out.groupby(["cls", "cause"]).agg(n=("score", "size"), points=("score", lambda x: float((1 - x).sum()))).reset_index().to_dict("records")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("decomp"); c.add_argument("--out", required=True)
    a = p.parse_args()
    dict(decomp=cmd_decomp)[a.cmd](a)


if __name__ == "__main__":
    main()
