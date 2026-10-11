#!/usr/bin/env python3
"""ONPOL-LONG Amendment 3, descriptive mechanism read: on held rollouts, how often the corridor hinge K (lib/ol_kc.py: mean over the 8 served
poses of relu(distance to the logged rear-axle path - 1.5 m)) fires on the served plan, and the offroad zeros split by whether K fired on a
served plan in the 3 decisions up to the first at-fault step. Box, AlpaSim's venv (pandas).

  ol_mech.py --runs <label>=<run dir> ... --k-table <report dir>/per_scene_k.parquet --scenes <report dir>/per_scene.csv --out <md>

Frame: the rollout's local frame is the nuPlan map frame translated (no rotation; on_rows.py docstring); the translation is taken from decision
0, where the ego is on the log (log replay): O = log pose at the decision's log time - anchor. The decision's log time = T0 + t0 with T0 = the
token's log time - 1.5 s (1020 of 1021 train rollouts at 18 decisions; on_rows.py gate R2). Approximate by construction: not a gate.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "experiments/onpol_nuplan/scripts")]
import on_rows as OR  # noqa: E402

DEAD = 1.5


def one(job):
    label, run, scene, recs = job
    log, tok = scene.rsplit("-", 1)
    L = OR._log(log)
    i = L["tok"].index(tok)
    T0 = int(L["T"][i]) - 1_500_000
    logp = lambda t: np.array([np.interp(t, L["T"], L["P"][:, q]) for q in range(3)])  # noqa: E731
    G = OR.Path(L["P"][max(i - 13, 0):], L["T"][max(i - 13, 0):])
    r0 = recs[min(recs)]
    O = logp(T0 + int(r0["t0"]))[:2] - np.array(r0["anchor"][:2])
    out = []
    for k in sorted(recs):
        r = recs[k]
        a = np.array(r["anchor"], float)
        P = np.array(r["poses"], float)[:, :2]
        c, s = np.cos(a[2]), np.sin(a[2])
        Pm = a[:2] + O + P @ np.array([[c, s], [-s, c]])
        sl = float(G.s_log(T0 + int(r["t0"])))
        d = np.array([abs(G.project(p, sl - 60, sl + 120)[1]) for p in Pm])
        e0 = abs(G.project(a[:2] + O, sl - 60, sl + 80)[1])
        out.append(dict(label=label, scene=scene, k=int(k), kcorr=float(np.maximum(d - DEAD, 0).mean()), e0=float(e0)))
    return out


def jobs_of(label, run):
    S, J = {}, []
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "start":
            S[r["session"]] = (r["scene"], r["n"], {})
        elif r["kind"] == "drive" and r["session"] in S:
            S[r["session"]][2][r["k"]] = r
    best = {}
    for sc, n, recs in S.values():                                    # the last session of a scene (a retried rollout)
        if recs and (sc not in best or n > best[sc][0]):
            best[sc] = (n, recs)
    return [(label, run, sc, recs) for sc, (n, recs) in sorted(best.items())]


def main(a):
    import pandas as pd
    J = [j for spec in a.runs for j in jobs_of(*spec.split("=", 1))]
    with ProcessPoolExecutor(a.jobs) as ex:
        D = pd.DataFrame([r for rr in ex.map(one, J, chunksize=4) for r in rr])
    D.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    K = pd.read_parquet(a.k_table)
    S = pd.read_csv(a.scenes)
    labs = [s.split("=", 1)[0] for s in a.runs]
    md = ["# Corridor hinge K on the served plans of held rollouts, and offroad zeros (Amendment 3, descriptive)", "",
          "| label | decisions | K fires (share of decisions) | K fires, decisions 0-9 / 10-17 | median off-path at the decision (m) | offroad zeros | of them K fired within 3 decisions up to the first at-fault step |",
          "|:--|--:|--:|:--|--:|--:|--:|"]
    for lab in labs:
        d = D[D.label == lab]
        f = d.kcorr > 0
        z = S[(S.label == lab) & (S.score == 0) & (S.offroad > 0)].scene
        kk = K[(K.label == lab) & K.scene.isin(z) & K.atf].groupby("scene").k.min()
        hit = sum(bool((d[(d.scene == sc) & d.k.between(k0 - 2, k0)].kcorr > 0).any()) for sc, k0 in kk.items())
        md.append(f"| {lab} | {len(d)} | {f.mean():.3f} | {f[d.k < 10].mean():.3f} / {f[d.k >= 10].mean():.3f} | {d.e0.median():.2f} | {len(z)} | {hit} of {len(kk)} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True), ap.add_argument("--k-table", required=True), ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True), ap.add_argument("--jobs", type=int, default=32)
    main(ap.parse_args())
