#!/usr/bin/env python3
"""HT14 inventory read (no GPU): strict PAI turn completion + d293 shape gate of RSG-L18-s0 and RBSG2-L18-s0 on the strict turns of
hard_train (14 with a defined completion target) and trainable (3), from the stored r281 reads (Tokyo r281/rsg, r281/rbsg2; path 0).

  rf_ht14.py --pkl <rf_pai_follow r281.pkl> --out results/hard_train14      (Mac, .venv)
completed = rf_pai_follow's strict rule (results/pai_follow/scenes.csv, outcome == completed); shape = rf_shape.track_shape on the
label's own controller track, window [entry - 40 m, exit], cut at its first zero event; gate (d293, rf_p4.gate_of without the entry-speed
term): kmax <= class p95, dkappa <= class p95 (all-turn p95 for classes with < 5 human turns), a_lat <= all-turn human p95 (4.42 m/s^2).
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rf_pai_follow as F  # noqa: E402
import rf_shape as S  # noqa: E402

R = HERE.parent / "results"
EXAM = HERE.parents[1] / "alpasim" / "results" / "pai_exam"
LABELS = ("rsg", "rbsg2")
V_E = {"straight_exit": 4.6, "curve_single": 3.9, "straight_road": 5.3, "no_straight_exit": 4.6}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    strict = json.load(open(R / "turn_gallery" / "plan_strict36.json"))
    J = json.load(open(R / "p3_branch" / "shape.json"))
    n_topo = J["counts"]["topo"]
    hard = set(open(EXAM / "hard_train.txt").read().split()) & set(open(R / "pai281" / "strict36.txt").read().split())
    train = set(open(EXAM / "trainable.txt").read().split()) & set(open(R / "pai281" / "strict36.txt").read().split())
    bad = {s for s in hard if s not in strict["scenes"]}
    sets = {s: "hard_train" for s in hard - bad} | {s: "trainable" for s in train}
    pf = pd.read_csv(R / "pai_follow" / "scenes.csv")
    pf = pf[pf.label.isin(LABELS) & (pf.turn == True)].set_index(["label", "scene"])  # noqa: E712
    r281 = pickle.load(open(a.pkl, "rb"))
    rows = []
    for sc in r281["scenes"]:
        if sc["scene"] not in sets:
            continue
        P = strict["scenes"][sc["scene"]]
        topo = P["topo"]
        b = J["band_topo"][topo] if n_topo.get(topo, 0) >= 5 else J["band_all"]
        thr = dict(kmax=b["kmax"][1], dk=b["dk"][1], alat=J["band_all"]["alat"][1], v0=V_E.get(topo, 4.6) + 1.0)
        for lab in LABELS:
            o, f = sc["rolls"][lab], pf.loc[(lab, sc["scene"])]
            assert (f.entry, f.exit) == (P["s_entry"], P["s_exit"]), (lab, sc["scene"], f.entry, f.exit, P)
            Rt = F.dedup(o["route"]); St = F.stations(Rt)
            C = o["ctrl"]; XY = np.c_[C["x"], C["y"]]
            st, _ = F.track(Rt, St, XY)
            tz = o["tz"] if o["tz"] is not None and not o["score"] else np.inf
            sh = S.track_shape(C["t"], XY, C["vx"], st, f.entry - S.WIN_PRE, f.exit, tz)
            v0 = S.speeds(C["t"], st, C["vx"], f.entry, f.exit, tz)["v_0"]
            done = f.outcome == "completed"
            g = dict(k=sh["kmax"] <= thr["kmax"], dk=sh["dk"] <= thr["dk"], a=sh["alat"] <= thr["alat"])
            rows.append(dict(scene=sc["scene"], short=sc["scene"][7:15], set=sets[sc["scene"]], side=P["side"], deg=round(P["turn_deg"]), topo=topo,
                             label=lab, score=float(o["score"] or 0), outcome=f.outcome, kind=f.kind if isinstance(f.kind, str) else "",
                             done=done, kmax=sh["kmax"], dk=sh["dk"], alat=sh["alat"], v0=v0,
                             thr_kmax=thr["kmax"], thr_dk=thr["dk"], thr_alat=thr["alat"], thr_v0=thr["v0"],
                             shape_ok=bool(all(g.values())), shape_fail="".join(k for k, v in g.items() if not v),
                             gated=bool(done and all(g.values())), v0_ok=bool(np.isfinite(v0) and v0 <= thr["v0"])))
    D = pd.DataFrame(rows).sort_values(["set", "short", "label"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "hard_train14.csv", index=False, float_format="%.4f")
    summ = {}
    for nm, m in (("hard_train14", D.set == "hard_train"), ("trainable3", D.set == "trainable"), ("all17", D.set.notna())):
        for lab in LABELS:
            x = D[m & (D.label == lab)]
            n, k1, k2 = len(x), int(x.done.sum()), int(x.gated.sum())
            summ[f"{nm}:{lab}"] = dict(n=n, completed=k1, completed_ci=wilson(k1, n), gated=k2, gated_ci=wilson(k2, n),
                                       gated_and_v0=int((x.gated & x.v0_ok).sum()))
    (out / "summary.json").write_text(json.dumps(dict(excluded=sorted(bad), summary=summ), indent=1) + "\n")
    print(json.dumps(summ, indent=1)); print("excluded", bad)


if __name__ == "__main__":
    main()
