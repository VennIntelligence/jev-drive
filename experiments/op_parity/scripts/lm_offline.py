"""Lead standstill margin, step 1 (offline, CPU): the shared rule (jevdrive/openpilot/lead_margin.py) replayed on the stored HUGSIM traces
of P2H (P2H10-F-s0 / s1, spec_plan_smooth, r0 / rr1 / rr2). Pre-registration: experiments/op_parity/plans/2026-10-07-lead-margin-prereg.md
(the readings below are its "offline" section with the declared clarifications, written there before this script was run).

  hugsim  (box or Mac, .venv)  -> results/lead_margin/offline_{d3b_runs,d3b_scenarios,complete}.csv, offline.json

Inputs: $DATA_DIR/runs/op_parity/four_dirs/hugsim_traces.json.gz (fd_hugsim.py extract: per state k the ego box X / Y / YAW / V, the plan
sent at k as world points at 0.5 .. 3.0 s, the model's lead head (prob, x, v in model units), the actor boxes [x, y, w, l, yaw]) and
results/four_dirs/hugsim_events.csv (labels, fd_hugsim's bucket rule).

D3b runs: smooth runs that end in fg_collision with a lead_stopped / lead_moving contact, inside the (arm, scenario) cells fd_hugsim labels
D3b (modal label D3_fg, modal fg type a lead). Hit actor = the box intersecting (else nearest to) the final ego box (fd_hugsim's rule).
Per step k of the last 3 s before the contact (k = n - 12 .. n - 1): the rule is applied to the stored plan with the stored lead head
(lead_v / 1.25: model -> simulator m/s); the plan "keeps clear" when the ego box (3.0 x 1.6 m, heading of the plan path) at the plan
points 0.5 / 1.0 / 1.5 s does not intersect the hit actor's box at the recorded state k + 2, k + 4, k + 6 (past the run's end the actor
continues at its last recorded velocity).
  run converted (primary)     every step of the window keeps clear under the rule
  run converted (secondary)   at least half of the steps keep clear
  scenario converted          at least half of its D3b runs converted; gate: >= 7 of 10 scenarios (primary)
Complete runs: smooth runs ending `complete` in cells labelled complete; over all their steps: trigger rate (lead_prob > 0.5), changed
steps (a plan point pulled back > 0.05 m; gate < 5 %), stops made (plan end < 1 m after the rule, >= 1 m before).
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
from jevdrive.openpilot import lead_margin as LM  # noqa: E402

FD = REPO / "experiments/op_parity/results/four_dirs"
OUT = REPO / "experiments/op_parity/results/lead_margin"
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
DT, EGO_L, EGO_W, DIL = 0.25, 3.0, 1.6, 1.25
T_PLAN = np.r_[0.0, 0.5 * np.arange(1, 7)]
WIN, CHANGED_M = 12, 0.05


def box(x, y, w, l, yaw):
    from shapely.geometry import Polygon
    c, s = np.cos(yaw), np.sin(yaw)
    return Polygon([(x + a * c - b * s, y + a * s + b * c) for a, b in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))])


def labels(E):
    """fd_hugsim report's cell labels: modal label per (arm, scenario) over the 3 repeats, D3 split by the modal fg type."""
    import pandas as pd
    E = E.copy()
    E.loc[(E.fg_type == "crossing") & (E.fg_dh.abs() >= 150), "fg_type"] = "oncoming"

    def lab(r):
        if r.end == "fg_collision":
            return "D3_fg"
        if r.end in ("bg_collision", "off_route"):
            return {"sharp": "D1_sharp", "wide": "D2_wide", "straight": "S_straight"}[r.rclass]
        return "stuck" if r.end == "max_steps" else ("complete" if r.end == "complete" else "other")

    E["label"] = E.apply(lab, axis=1)
    sm = E[E.arm.isin(["P2H10-F-s0", "P2H10-F-s1"]) & (E.preset == "spec_plan_smooth")]
    cells = {}
    for (a, sc), g in sm.groupby(["arm", "scenario"]):
        c = Counter(g.label)
        top = max(c.values())
        tied = [k for k, v in c.items() if v == top]
        L = tied[0] if len(tied) == 1 else g[g.rep == "r0"].label.iloc[0]
        if L == "D3_fg":
            ft = Counter(g[g.end == "fg_collision"].fg_type).most_common(1)[0][0]
            L = "D3b" if ft in ("lead_stopped", "lead_moving") else "D3_other"
        cells[(a, sc)] = L
    sm = sm.assign(cell=[cells[(a, s)] for a, s in zip(sm.arm, sm.scenario)])
    return sm


def path_heading(xy, q, yaw0):
    """Heading of the polyline xy (from its first point) at arc positions q: the direction of the segment holding q (yaw0 at a stand)."""
    s = LM.arclen(xy)
    seg = np.diff(xy, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    hd = np.where(ln > 0.05, np.arctan2(seg[:, 1], seg[:, 0]), np.nan)
    out = []
    for v in q:
        i = int(np.clip(np.searchsorted(s, v, side="right") - 1, 0, len(seg) - 1))
        if v <= 1e-6:
            i = next((j for j in range(len(seg)) if np.isfinite(hd[j])), None)
            out.append(yaw0 if i is None else hd[i])
            continue
        out.append(hd[i] if np.isfinite(hd[i]) else yaw0)
    return np.array(out)


def actor_at(objs, i, k, n):
    """Hit actor box [x, y, w, l, yaw] at state k (recorded up to n, then constant velocity from its last step)."""
    if k <= n and len(objs[k]) > i:
        return np.asarray(objs[k][i], float)
    a, b = np.asarray(objs[n][i], float), np.asarray(objs[n - 1][i], float) if len(objs[n - 1]) > i else np.asarray(objs[n][i], float)
    o = a.copy()
    o[:2] = a[:2] + (a[:2] - b[:2]) * (k - n)
    return o


def clear(xy, t, yaw0, act, objs, i, k, n):
    """Ego box at the plan points 0.5 / 1.0 / 1.5 s (xy includes the origin) vs the actor at k + 2 / 4 / 6."""
    q = LM.arclen(xy)[1:4]
    hd = path_heading(xy, q, yaw0)
    for j in range(3):
        o = actor_at(objs, i, k + 2 * (j + 1), n)
        if box(xy[j + 1, 0], xy[j + 1, 1], EGO_W, EGO_L, hd[j]).intersects(box(*o)):
            return False
    return True


def hit_actor(tr):
    n = len(tr["plans"])
    X, Y, YAW = tr["X"][n], tr["Y"][n], tr["YAW"][n]
    ob = tr["objs"][n]
    if not ob:
        return None
    ep = box(X, Y, EGO_W, EGO_L, YAW)
    inter = [ep.intersection(box(*o)).area for o in ob]
    return int(np.argmax(inter)) if max(inter) > 0 else int(np.argmin([ep.distance(box(*o)) for o in ob]))


def rule_plan(tr, k, rule=True):
    """(xy with the origin, info) of the plan at state k, after the rule (rule False: the stored plan)."""
    w = np.asarray(tr["plans"][k], float)
    xy = np.r_[[[tr["X"][k], tr["Y"][k]]], w]
    lp, lx, lv = tr["lead"][k] if k < len(tr.get("lead", [])) else (None, None, None)
    if not rule:
        return xy, None
    nxy, _, info = LM.apply(xy, T_PLAN, lp, lx, None if lv is None else lv / DIL)
    return nxy, info


def cmd_hugsim(a):
    import pandas as pd
    E = pd.read_csv(FD / "hugsim_events.csv")
    sm = labels(E)
    with gzip.open(a.traces, "rt") as f:
        T = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)
    key = lambda r: f"{r.arm}|{r.preset}|{r.rep}|{r.scenario}"  # noqa: E731
    # D3b runs
    d3 = sm[(sm.cell == "D3b") & (sm.end == "fg_collision") & sm.fg_type.isin(["lead_stopped", "lead_moving"])]
    rows = []
    for r in d3.itertuples():
        tr = T[key(r)]
        n = len(tr["plans"])
        i = hit_actor(tr)
        st = dict(base=[], rule=[], trig=[], ds=[])
        for k in range(max(0, n - WIN), n):
            if tr["plans"][k] is None or len(tr["objs"][k]) <= i:
                continue
            for nm, rl in (("base", False), ("rule", True)):
                xy, info = rule_plan(tr, k, rl)
                st[nm].append(clear(xy, T_PLAN, tr["YAW"][k], None, tr["objs"], i, k, n))
                if rl:
                    st["trig"].append(info["trig"])
                    st["ds"].append(info["ds_max"])
        nb, nr = np.mean(st["base"]) if st["base"] else np.nan, np.mean(st["rule"]) if st["rule"] else np.nan
        lp_last = tr["lead"][n - 1][0] if tr.get("lead") else None
        rows.append(dict(arm=r.arm, rep=r.rep, scenario=r.scenario, fg_type=r.fg_type, steps=len(st["rule"]),
                         clear_base=round(nb, 3), clear_rule=round(nr, 3), trig_frac=round(np.mean(st["trig"]), 3),
                         ds_max=round(max(st["ds"]), 2), conv_primary=bool(st["rule"] and all(st["rule"])),
                         conv_secondary=bool(st["rule"] and nr >= 0.5), base_all_clear=bool(st["base"] and all(st["base"])),
                         lead_prob_m1=lp_last))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "offline_d3b_runs.csv", index=False)
    S = R.groupby("scenario").agg(runs=("arm", "size"), conv_primary=("conv_primary", "mean"), conv_secondary=("conv_secondary", "mean"),
                                  base_all_clear=("base_all_clear", "mean"), clear_base=("clear_base", "mean"), clear_rule=("clear_rule", "mean"),
                                  trig_frac=("trig_frac", "mean")).reset_index()
    S["scen_primary"], S["scen_secondary"] = S.conv_primary >= 0.5, S.conv_secondary >= 0.5
    S.to_csv(OUT / "offline_d3b_scenarios.csv", index=False)
    # complete runs: how often the rule touches a drive that went well
    cm = sm[(sm.cell == "complete") & (sm.end == "complete")]
    crow = []
    for r in cm.itertuples():
        tr = T[key(r)]
        for k, p in enumerate(tr["plans"]):
            if p is None:
                continue
            xy, info = rule_plan(tr, k)
            s0 = LM.arclen(np.r_[[[tr["X"][k], tr["Y"][k]]], np.asarray(p, float)])[-1]
            crow.append(dict(arm=r.arm, rep=r.rep, scenario=r.scenario, k=k, v=tr["V"][k], trig=info["trig"], ds=info["ds_max"],
                             changed=info["ds_max"] > CHANGED_M, made_stop=bool(s0 >= 1.0 and info.get("s_end_new", s0) < 1.0)))
    C = pd.DataFrame(crow)
    C.to_csv(OUT / "offline_complete_steps.csv.gz", index=False)
    by = C.groupby("scenario")[["trig", "changed", "made_stop"]].mean()
    summ = dict(d3b_runs=len(R), d3b_scenarios=int(S.scenario.nunique()),
                scen_converted_primary=int(S.scen_primary.sum()), scen_converted_secondary=int(S.scen_secondary.sum()),
                runs_converted_primary=int(R.conv_primary.sum()), runs_converted_secondary=int(R.conv_secondary.sum()),
                runs_base_all_clear=int(R.base_all_clear.sum()),
                step_clear_base=round(float(np.average(R.clear_base, weights=R.steps)), 3),
                step_clear_rule=round(float(np.average(R.clear_rule, weights=R.steps)), 3),
                complete_runs=int(len(cm)), complete_scenarios=int(cm.scenario.nunique()), complete_steps=int(len(C)),
                complete_trigger=round(float(C.trig.mean()), 4), complete_changed=round(float(C.changed.mean()), 4),
                complete_made_stop=round(float(C.made_stop.mean()), 4),
                complete_changed_scen_max=round(float(by.changed.max()), 3), complete_changed_scen_median=round(float(by.changed.median()), 3),
                gate_d3b=bool(S.scen_primary.sum() >= 7), gate_complete=bool(C.changed.mean() < 0.05))
    by.round(4).to_csv(OUT / "offline_complete.csv")
    (OUT / "offline.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    print(S.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    h = sp.add_parser("hugsim")
    h.add_argument("--traces", default=str(D / "runs/op_parity/four_dirs/hugsim_traces.json.gz"))
    a = ap.parse_args()
    {"hugsim": cmd_hugsim}[a.cmd](a)


if __name__ == "__main__":
    main()
