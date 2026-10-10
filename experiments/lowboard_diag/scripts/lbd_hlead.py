#!/usr/bin/env python
"""Lane HLEAD (plans/2026-10-10-hlead-prereg.md): openpilot's lead path in the HUGSIM closed loop (agent option `op_lead`,
jevdrive/openpilot/lead_long.py) against the stored run of decision 237, paired per scenario.

  extract  (box, .venv)  per scenario and arm: end, HD, the limiter's activity from zs_steps.jsonl, the standstill at the end, the struck
                         actor and contact part, the lead head against the simulator's boxes (labels only) -> results/hlead/units.csv
  report   (box or Mac)  outcome of the units decision 237 classified, paired HD differences with CIs, worse / better scenarios
                         -> results/hlead/*.csv, tables.md

Arms: A0 = the stored SH30-F-s0 spec_plan_smooth run (decision 237), A0r = the same configuration re-run after the code change with the
switch off (repeat hlead-off), A1 = --opts '{"op_lead": {}}'. All three are jevdrive.bench run dirs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts")]

OUT = Path(os.environ.get("LBD_OUT", REPO / "experiments/lowboard_diag/results/hlead"))      # LOWDIAG2: per-seed output dir, arms and the decision-237 table (plumbing only)
FAILED = Path(os.environ.get("LBD_FAILED", REPO / "experiments/lowboard_diag/results/hugsim/failed_units.csv"))
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
BASE = "SH30-F-s0_spec_plan_smooth"
ARMS = json.loads(os.environ["LBD_ARMS"]) if "LBD_ARMS" in os.environ else {"A0": BASE, "A0r": BASE + "-rhlead-off", "A1": BASE + "-ca9065d178c07"}
FAIL = ("fg_collision", "bg_collision", "off_route")
WORSE, V_STAND, T_STALL, CAM_FRONT = 0.05, 0.1, 10.0, 1.5      # registered: worse = HD difference <= -0.05; stall = standing >= 10 s
CUT = 0.5                                                      # m at the plan end: a step counts as limited (any change at all is `ol_frac`)


def gap_ahead(FD, R, k):
    """Label only: bumper gap (m) to the nearest actor box ahead of the ego whose lateral extent overlaps the ego's, at state k."""
    ob = np.asarray(R["objs"][k], float)
    if not len(ob):
        return np.nan
    c, s = np.cos(R["YAW"][k]), np.sin(R["YAW"][k])
    q = ob[:, :2] - [R["X"][k], R["Y"][k]]
    lx, ly = q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c
    m = (lx > 0) & (np.abs(ly) < (FD.EGO_W + ob[:, 3]) / 2)
    return float((lx[m] - FD.EGO_L / 2 - ob[m, 4] / 2).min()) if m.any() else np.nan


def one(FD, LB, arm, row):
    d = Path(row["run_dir"])
    R = FD.load_run(d)
    n, V, st = R["n"], R["V"], R["steps"]
    ev = dict(scenario=row["scenario"], arm=arm, end=row["end"], hd=float(row["hdscore"]), rc=float(row["rc"]), n_steps=n, v_end=float(V[n - 1]),
              v_max=float(V.max()))
    stand = 0
    while stand < n and V[n - 1 - stand] < V_STAND:
        stand += 1
    ev["stand_end_s"] = stand * FD.DT
    ol = [st[k].get("ol") for k in range(n) if k in st]
    if any(o is not None for o in ol):
        ch = np.array([bool(o and o["changed"]) for o in ol])
        cut = np.array([o["cut"] if o else 0.0 for o in ol])
        ev.update(ol_steps=int(ch.sum()), ol_frac=float(ch.mean()), ol_cut_max=float(cut.max()),
                  cut05_frac=float((cut >= CUT).mean()), cut05_steps=int((cut >= CUT).sum()),      # plan end pulled back >= 0.5 m
                  lead_frac=float(np.mean([bool(o and o["d"] is not None) for o in ol])),          # radard: a lead is present
                  cut05_lead_frac=float(np.mean([bool(o and o["d"] is not None and o["cut"] >= CUT) for o in ol])),
                  ol_first_s=float(np.argmax(ch) * FD.DT) if ch.any() else np.nan,
                  ol_stand_end=bool(stand and ch[-min(stand, len(ch)):].all()),          # the limit holds the whole final standstill
                  ol_last3=float(ch[-12:].mean()), p_last3_min=float(min(o["p"][0] for o in ol[-12:] if o)))
    lead_on = [k for k in range(n) if k in st and (st[k].get("lead_prob") or 0) >= 0.5]
    ev["lead_on_frac"] = len(lead_on) / max(n, 1)
    if row["end"] == "fg_collision":
        s = LB.struck(R)
        if s is not None:
            part = "front" if s["lx"] > FD.EGO_L / 2 else ("rear" if s["lx"] < -FD.EGO_L / 2 else ("left" if s["ly"] > 0 else "right"))
            o = np.asarray(R["objs"][n], float)[s["i"]]
            ev.update(obj_i=s["i"], obj_lx=s["lx"], obj_ly=s["ly"], obj_dh=s["dh"], obj_v=s["vo"], part=part, obj_x=float(o[0]), obj_y=float(o[1]))
    # the lead head against the boxes over the last 3 s before the end (label): reported distance - 1.5 m against the true bumper gap
    ks = [k for k in range(max(0, n - 12), n) if k in st and st[k].get("lead_x") is not None]
    g = np.array([gap_ahead(FD, R, k) for k in ks])
    rd = np.array([st[k]["lead_x"] - CAM_FRONT for k in ks])
    pr = np.array([st[k]["lead_prob"] for k in ks])
    ok = np.isfinite(g) & (pr >= 0.5)
    ev.update(gap_end=gap_ahead(FD, R, n), gap_min3=float(np.nanmin(g)) if np.isfinite(g).any() else np.nan,
              lead_bias3=float(np.median(rd[ok] - g[ok])) if ok.any() else np.nan, lead_p_min3=float(pr.min()) if len(pr) else np.nan)
    return ev, R


def extract(a):
    import pandas as pd
    import fd_hugsim as FD
    sys.path.insert(0, str(Path(__file__).parent))
    import lbd_hugsim as LB
    rows, runs = [], {}
    for arm, key in ARMS.items():
        u = D / "runs/bench/hugsim" / key / "units.csv"
        if not u.exists():
            print("missing", u)
            continue
        for _, r in pd.read_csv(u).iterrows():
            ev, R = one(FD, LB, arm, r)
            runs[arm, r["scenario"]] = R
            rows.append(ev)
    # A0r against A0: the same decisions (every step's sent plan and ego position), i.e. the switch-off path is unchanged
    for ev in rows:
        if ev["arm"] == "A0r" and ("A0", ev["scenario"]) in runs:
            x, y = runs["A0", ev["scenario"]], runs["A0r", ev["scenario"]]
            same = x["n"] == y["n"] and all(x["steps"][k].get("plan") == y["steps"][k].get("plan") and x["steps"][k]["pos"] == y["steps"][k]["pos"]
                                            for k in x["steps"])
            first = next((k for k in sorted(x["steps"]) if k not in y["steps"] or x["steps"][k].get("plan") != y["steps"][k].get("plan")), -1)
            dmax = np.nan
            if first >= 0 and first in y["steps"] and len(x["steps"][first]["plan"]) == len(y["steps"][first]["plan"]):
                dmax = float(np.abs(np.asarray(x["steps"][first]["plan"]) - np.asarray(y["steps"][first]["plan"])).max())
            ev.update(same_as_A0=bool(same), first_diff_step=first, first_diff_m=dmax)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "units.csv", index=False, float_format="%.4f")
    print("wrote", OUT / "units.csv", len(rows))


def outcome(e0, e1):
    if e1 in ("complete", "max_steps"):
        return "fixed (complete)" if e1 == "complete" else "fixed (standing, max_steps)"
    return "unchanged" if e1 == e0 else "new failure"


def detail(r):
    """Finer than the registered label: what an fg_collision of A1 is against."""
    if r.end_A1 != "fg_collision":
        return ""
    who = "same actor" if r.same_actor else ("other actor" if r.end_A0 == "fg_collision" else "actor")
    return f"{who}, {r.part}, dh {r.obj_dh_A1:.0f} deg, {r.obj_v:.1f} m/s"


def report(a):
    import pandas as pd
    from jevdrive import stats
    U = pd.read_csv(OUT / "units.csv")
    F = pd.read_csv(FAILED).set_index("scenario")
    W = {arm: g.set_index("scenario") for arm, g in U.groupby("arm")}
    sc = sorted(set(W["A1"].index) & set(W["A0"].index))
    has_r = "A0r" in W
    T = pd.DataFrame(index=sc)
    T["cls237"] = [F.cls.get(s, "complete") for s in sc]
    T["sub237"] = [F["sub"].get(s, "-") for s in sc]
    for arm in W:
        T[f"end_{arm}"], T[f"hd_{arm}"] = W[arm].end.reindex(sc), W[arm].hd.reindex(sc)
    T["d_hd"] = T.hd_A1 - T.hd_A0
    T["noise"] = (T.hd_A0r - T.hd_A0) if has_r else np.nan
    for c in ("cut05_frac", "lead_frac", "cut05_lead_frac", "ol_frac", "ol_cut_max", "ol_first_s", "ol_stand_end", "stand_end_s", "v_end", "part", "obj_i", "obj_v", "gap_end", "gap_min3", "lead_bias3",
              "lead_p_min3", "ol_last3", "n_steps", "rc"):
        if c in W["A1"]:
            T[c] = W["A1"][c].reindex(sc)
    # a front-vehicle collision of A0 is "the same" in A1 when A1 also ends on an actor and that actor is the one A0 struck: same index
    # in the simulator's actor list, or (standing actors) the same place within 2 m
    for c in ("obj_i", "obj_x", "obj_y", "obj_dh"):
        T[c + "_A0"] = W["A0"][c].reindex(sc)
        T[c + "_A1"] = W["A1"][c].reindex(sc) if c in W["A1"] else np.nan
    T["same_actor"] = (T.end_A0 == "fg_collision") & (T.end_A1 == "fg_collision") & \
        ((T.obj_i_A0 == T.obj_i_A1) | (np.hypot(T.obj_x_A0 - T.obj_x_A1, T.obj_y_A0 - T.obj_y_A1) < 2.0))
    T["outcome"] = [outcome(r.end_A0, r.end_A1) if r.end_A0 in FAIL else ("still complete" if r.end_A1 == "complete" else "lost: " + r.end_A1)
                    for r in T.itertuples()]

    def cause(r):
        if r.end_A1 == "max_steps" or (r.stand_end_s >= T_STALL and bool(r.ol_stand_end)):
            return "stall behind the limit"
        if r.end_A1 == "fg_collision":
            return {"rear": "rear-ended", "front": "front contact"}.get(r.part, "side contact (%s)" % r.part)
        if r.end_A1 == "complete":
            return "complete, lower sub-scores"
        return r.end_A1
    quiet = (T.noise.abs() < WORSE) if has_r else True
    T["fg_A1"] = [detail(r) for r in T.itertuples()]
    T["worse"] = (T.d_hd <= -WORSE) & quiet
    T["better"] = (T.d_hd >= WORSE) & quiet
    T["cause"] = [cause(r) if r.worse else "" for r in T.itertuples()]
    T.index.name = "scenario"
    T.to_csv(OUT / "paired_units.csv", float_format="%.4f")

    L = []
    P = L.append
    n = len(sc)
    P(f"# HLEAD tables ({n} scenarios with both A0 and A1; generated by scripts/lbd_hlead.py report)\n")
    rows = []
    for lab, m in [("all", T.index == T.index)] + [(f"237 class {c}", T.cls237 == c) for c in ("longitudinal", "clearance", "other", "route", "complete")] + \
                  [("237 sub L1", T.sub237 == "L1"), ("237 sub L2", T.sub237 == "L2")]:
        g = T[m]
        if not len(g):
            continue
        r = dict(subset=lab, n=len(g), HD_A0=g.hd_A0.mean(), HD_A1=g.hd_A1.mean(), **{"A1 - A0 [95% CI]": stats.fmt(stats.paired(g.hd_A1, g.hd_A0), "+.3f")})
        if has_r:
            gr = g[g.hd_A0r.notna()]
            r.update(HD_A0r=gr.hd_A0r.mean(), **{"n A0r": len(gr), "A0r - A0": stats.fmt(stats.paired(gr.hd_A0r, gr.hd_A0), "+.3f"),
                                                "A1 - A0r": stats.fmt(stats.paired(gr.hd_A1, gr.hd_A0r), "+.3f")})
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "paired_hd.csv", index=False, float_format="%.4f")
    P("## T1 HD, paired per scenario (percentile bootstrap over scenarios, B 10000, seed 0)\n")
    P(S.to_markdown(index=False, floatfmt=".3f") + "\n")
    ends = pd.DataFrame({arm: W[arm].end.reindex(sc).value_counts() for arm in W}).fillna(0).astype(int)
    ends.to_csv(OUT / "ends.csv")
    P("## T2 ends\n")
    P(ends.to_markdown() + "\n")
    cols = ["sub237", "end_A0", "end_A1", "hd_A0", "hd_A1", "d_hd"] + (["noise"] if has_r else []) + \
           ["outcome", "fg_A1", "cut05_frac", "lead_frac", "stand_end_s", "gap_end", "gap_min3", "lead_bias3", "lead_p_min3"]
    cols = [c for c in cols if c in T]
    for sub in ("L1", "L2"):
        g = T[T.sub237 == sub]
        if len(g):
            P(f"## T3 {sub} units of decision 237 ({len(g)})\n")
            P(g[cols].to_markdown(floatfmt=".3f") + "\n")
            P("outcome counts: " + ", ".join(f"{k} {v}" for k, v in g.outcome.value_counts().items()) + "\n")
    P("## T4 outcome by decision-237 class\n")
    P(pd.crosstab(T.cls237 + " / " + T.sub237, T.outcome).to_markdown() + "\n")
    for lab, m in (("worse (A1 - A0 <= -0.05, A0r within 0.05 of A0)", T.worse), ("better (A1 - A0 >= +0.05, A0r within 0.05 of A0)", T.better)):
        g = T[m].sort_values("d_hd")
        P(f"## T5 {lab}: {len(g)}\n")
        if len(g):
            P(g[[c for c in ["cls237", "sub237"] + cols[1:] + ["cause", "v_end", "rc"] if c in g]].to_markdown(floatfmt=".3f") + "\n")
    if has_r:
        r = U[U.arm == "A0r"]
        P("## T6 the switch-off re-run against the stored run\n")
        P(f"scenarios {len(r)}; every step's plan and position identical in {int(r.same_as_A0.sum())} (HD equal in "
          f"{int((T.noise.abs() < 1e-12).sum())}); same end in {int((W['A0r'].end == W['A0'].end.reindex(W['A0r'].index)).sum())}; "
          f"|HD difference| max {T.noise.abs().max():.4f}, median {T.noise.abs().median():.4f}, scenarios with |difference| >= 0.05: "
          f"{int((T.noise.abs() >= WORSE).sum())}; where the plans differ, the first differing plan differs by a median "
          f"{r.first_diff_m.median():.3f} m (max {r.first_diff_m.max():.3f} m) at a median step {r.first_diff_step[r.first_diff_step >= 0].median():.0f}\n")
    a1 = W["A1"].reindex(sc)
    P("## T7 limiter activity in A1\n")
    P(f"any change of the plan: {int((a1.ol_steps > 0).sum())} / {n} scenarios, {a1.ol_steps.sum() / a1.n_steps.sum():.3f} of all steps (the lead MPC's "
      f"no-lead solution also caps acceleration at 2.0 m/s^2 in the model's units). Plan end pulled back >= {CUT} m: "
      f"{int((a1.cut05_steps > 0).sum())} / {n} scenarios, {a1.cut05_steps.sum() / a1.n_steps.sum():.3f} of all steps; with a lead present "
      f"(filtered probability > 0.5): {(a1.cut05_lead_frac * a1.n_steps).sum() / a1.n_steps.sum():.3f}; a lead present at all: "
      f"{(a1.lead_frac * a1.n_steps).sum() / a1.n_steps.sum():.3f} of all steps\n")
    act = pd.DataFrame([dict(subset=k, n=len(g), cut05_frac=g.cut05_frac.mean(), lead_frac=g.lead_frac.mean(), d_hd=g.d_hd.mean())
                        for k, g in T.groupby("cls237")])
    P(act.to_markdown(index=False, floatfmt=".3f") + "\n")
    (OUT / "tables.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    sub.add_parser("report")
    a = ap.parse_args()
    {"extract": extract, "report": report}[a.cmd](a)
