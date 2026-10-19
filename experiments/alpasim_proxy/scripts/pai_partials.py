#!/usr/bin/env python3
"""PAI-PARTIALS: where the rear-hit and slow partials of NTV0-L18-s0 + ACC + GAP (stored PAI 281 read A0) lose their distance, serve
side or model plan (plans/2026-10-20-pai-partials-sweep-prereg.md, part 1). Box, repo .venv; stored reads are read, never written.

  pai_partials.py decomp --out $DATA_DIR/runs/alpasim_proxy/pai_partials/decomp.csv
  pai_partials.py sweep --rows rows.csv.gz --fits <dir with fits/, boot_*.csv> --out results/pai_partials/sweep.md   (Mac)
  pai_partials.py value --rows tmp/pai_pcs/rows.csv.gz --fits tmp/pai_pcs/out/fits --decomp results/pai_partials/decomp.csv --out results/pai_partials/value.md

Per partial scene (d381's classes, route_follow/pai_gap/scenes.csv) and per inference decision of A0 from the handoff (1.6 s) to the rear
hit (rear-hit) or the rollout's end (slow):
  v_h    the human's speed 1.5 s after the decision, time-matched (the logged follower is log replay, so what lets it hit is the time lag)
  v15    the served speed at 1.5 s (fix.acc.v15), v_model the model plan's own speed at 1.5 s (fix.acc.v_model)
  deficit = max(v_h - v15, 0); serve part = clip(min(v_model, v_h) - v15, 0, deficit) (speed the plan had and the serve removed), labelled
         by the binding term at 1.5 s (fix.acc.near: lead real / lead phantom (pai_gap.cut_class) / curve / set / ramp); model part = the rest
  serve add = max(min(v15, v_h) - v_model, 0) (ACC serving faster than the plan, up to the human)
Integrated over the window x 0.1 s -> metres. Lag = the human's log station minus ours at the window's end (time-matched).
Model part split: during the ACC stop guard (the plan's 2 s arc < 2 m) / brake guard (the plan
  brakes >= 1 m/s^2), launch (handoff speed < 1 m/s, ego < 3 m/s), cruise (rest);
setlow = model metres with the ACC set speed below the human's speed.
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
        g = f"guard{a['guard']}" if a.get("guard") else "launch" if float(r["fix"]["v0"]) < 3.0 and R.speed(th) < 1.0 else "cruise"
        acc_m[f"model-{g}"] += (d - sv) * DT                      # model part: the plan brakes (ACC stop / brake guard), launch, cruise
        acc_m["setlow"] += (d - sv) * DT * (float(a.get("v_set") or 0.0) < h)
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
                **{f"{k}_m": acc_m[k] for k in TERMS + ("model-guardstop", "model-guardbrake", "model-launch", "model-cruise", "setlow")}, cause=(f"serve:{sub}" if serve >= acc_m["model"] and serve > 0 else "model"))


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


def cmd_value(a):
    """Mac (tmp/pcs_local/.venv): PCS worth of each cause group on the d392 ruler (paimix fit, NTV0-s0 + ACC + GAP's row, items fixed, MAP)."""
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    import pai_pcs_local as PL
    sc, _, _ = PL.table(a.rows)
    z = PL.Ruler(dict(np.load(Path(a.fits) / "paimix.npz", allow_pickle=True)), sc)
    it = np.array(z.items)
    y0 = sc.loc["NTV0-s0+AG", it].to_numpy(float)
    base = PL.pcs_of(z, y0)
    d = pd.read_csv(a.decomp).set_index("scene")
    d["loss"] = 1 - d.score
    mg = d[["model-guardstop_m", "model-guardbrake_m", "model-launch_m"]].sum(axis=1)
    d["mcause"] = np.where(mg >= d["model-cruise_m"], "plan brakes / stops / launches late where the human drives", "plan cruises below the human")

    def worth(new):                                          # new: {scene: score}
        y = y0.copy()
        for s, v in new.items():
            y[it == s] = v
        return PL.pcs_of(z, y) - base
    L = ["| group | scenes | points | deficit m | serve-removed m | PCS if -> 1 | PCS of the serve-removed share |", "|---|--:|--:|--:|--:|--:|--:|"]
    groups = [(f"{c}", x) for c, x in d.groupby("cls")] + [(f"{c}: {m}", x) for (c, m), x in d.groupby(["cls", "mcause"])] + [("all partials", d)]
    for lab, x in groups:
        x = x[x.index.isin(it)]
        sh = {s: r.score + r.loss * r.serve_m / max(r.deficit_m, 1e-9) for s, r in x.iterrows()}
        L.append(f"| {lab} | {len(x)} | {x.loss.sum():.1f} | {x.deficit_m.sum():.0f} | {x.serve_m.sum():.0f} | {worth({s: 1.0 for s in x.index}):+.0f} | {worth(sh):+.1f} |")
    t = ["lead-real", "lead-phantom", "curve", "set", "ramp"]
    L += ["", "Deficit metres by term (rear-hit / slow):", "", "| term | rear-hit m | slow m |", "|---|--:|--:|"]
    for k in t + ["model-guardstop", "model-guardbrake", "model-launch", "model-cruise", "setlow", "add"]:
        g = d.groupby("cls")[f"{k}_m"].sum()
        L.append(f"| {k} | {g.get('partial rear-hit', 0):.0f} | {g.get('partial slow', 0):.0f} |")
    d.to_csv(a.decomp)
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


ARMS = {"NTV0-s0+AG": "A0 (ref)", "PP-R0": "R0 (re-read)", "NTV0-s0+AG+hl": "HL (stored)", "PP-V1": "V1 HL+vset1.1",
        "PP-V1b": "V1b HL+vset1.2", "PP-V2": "V2 HL+rel2,2", "PP-V3": "V3 HL+gap0.5,2+alat3", "PP-J": "J joint"}


def cmd_sweep(a):
    """Mac: the sweep table on the d392 ruler (paimix + pool variants; boot_*.csv = the 100-draw scene bootstrap rerun with the arms)."""
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    import pai_pcs_local as PL
    sc, ty, _ = PL.table(a.rows)
    F = Path(a.fits)
    Z = {v: PL.Ruler(dict(np.load(F / f"fits/{v}.npz", allow_pickle=True)), sc) for v in PL.VARIANTS if (F / f"fits/{v}.npz").exists()}
    z = Z["paimix"]
    bf = sorted(F.glob("boot_*.csv"))
    B = pd.concat([pd.read_csv(f) for f in bf], ignore_index=True) if bf else None
    ref = "NTV0-s0+AG"
    rn = abs(z.pcs["PP-R0"] - z.pcs[ref]) if "PP-R0" in z.pcs else np.nan
    L = [f"Ruler: d392 paimix ({len(z.items)} items), arms out of pool (MAP). Read noise N = |PCS(R0) - PCS(A0)| = {rn:.0f}. "
         f"Bootstrap draws: {0 if B is None else len(B)}.", "",
         "| arm | mean | zeros | AF-front | AF-side | corridor | offroad | rear-hit partials | slow partials | PCS paimix | 95 % CI | pool variants | PCS - A0 (paired, 95 % CI) | PCS - HL |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|---|--:|"]
    J = {}
    for s, lab in ARMS.items():
        if s not in z.pcs:
            continue
        t = ty.loc[s].value_counts()
        y = sc.loc[s]
        vv = [Zv.pcs[s] for Zv in Z.values() if s in Zv.pcs]
        ci = d = ""
        if B is not None and s in B:
            ci = f"[{B[s].quantile(.025):.0f}, {B[s].quantile(.975):.0f}]"
            x = B[s] - B[ref]
            d = f"{z.pcs[s] - z.pcs[ref]:+.0f} [{x.quantile(.025):+.0f}, {x.quantile(.975):+.0f}]"
        hl = z.pcs[s] - z.pcs["NTV0-s0+AG+hl"]
        L.append(f"| {lab} | {y.mean():.4f} | {int((y <= 1e-4).sum())} | {t.get('AF-front', 0)} | {t.get('AF-side', 0)} | {t.get('corridor', 0)} | {t.get('offroad', 0)} | "
                 f"{t.get('rear-hit', 0)} | {t.get('slow', 0)} | **{z.pcs[s]:.0f}** | {ci} | {min(vv):.0f}-{max(vv):.0f} | {d} | {hl:+.0f} |")
        J[s] = dict(label=lab, mean=float(y.mean()), pcs=z.pcs[s], variants={v: Zv.pcs.get(s) for v, Zv in Z.items()},
                    af=int(t.get("AF-front", 0) + t.get("AF-side", 0)), types=t.to_dict())
    L += ["", "Rank per pool variant (arms only):", "", "| variant | order (best first) |", "|---|---|"]
    for v, Zv in Z.items():
        L.append(f"| {v} | " + " > ".join(f"{ARMS[s].split(' ')[0]} {Zv.pcs[s]:.0f}" for s in sorted([s for s in ARMS if s in Zv.pcs], key=lambda s: -Zv.pcs[s])) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("decomp"); c.add_argument("--out", required=True)
    v = s.add_parser("value"); v.add_argument("--rows", required=True); v.add_argument("--fits", required=True)
    v.add_argument("--decomp", required=True); v.add_argument("--out", required=True)
    w = s.add_parser("sweep"); w.add_argument("--rows", required=True); w.add_argument("--fits", required=True, help="dir with fits/ and boot_*.csv")
    w.add_argument("--out", required=True)
    a = p.parse_args()
    dict(decomp=cmd_decomp, value=cmd_value, sweep=cmd_sweep)[a.cmd](a)


if __name__ == "__main__":
    main()
