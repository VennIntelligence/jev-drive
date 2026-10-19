#!/usr/bin/env python3
"""PAI-INPUT-LOOP read-out (plans/2026-10-19-pai-input-loop-prereg.md). Box, repo .venv; stored reads are read, never written.

  pai_input_loop.py --o $DATA_DIR/runs/route_follow/pai_input_loop --a0 <stored NTV0 + ACC + GAP read> --out experiments/route_follow/results/pai_input_loop

Inputs: <o>/tax/{A0,N0,KA,KN}.pkl (rf_tax.py extract of each arm), each arm's driver/drive.jsonl, <o>/cf.pkl (pil_cf.py). Writes
<out>/tables.md, <out>/numbers.json, <out>/ratio_time.csv, <out>/pil_ratio_time.png.

Item 1 strata (on the reference arm's rollout): static = the log's distance < 1 m; launch = the speed at the handoff (rollout time 1.7 s) < 1
m/s and the log moves; turn = moving and the logged path's heading changes > 30 deg; straight = the other moving scenes.
Rollout time tau = (t - first controller sample) + 0.1 s (the first drive call is 0.1 s before the first sample); the handoff is tau = 1.7 s.
Item 2a: on the 0.5 s grid tau = 1.7 .. 20 s, ego speed / log speed at the ego's station (Roll alignment), steps with the log >= 2 m/s.
Item 2b: per inference decision at t0 (plan frame) with a decision 0.5 s later in the same rollout (both log >= 2 m/s, ego >= 1 m/s, vpm > 0.3):
  plan_model = log(vpm / v) - log(vh1 / vh), fix = log(vps / vpm), track = log(vpt / vps), exec = log(v1 / vpt); vp* = |p(1.0 s)| / 1 s of
  poses_model / poses / poses_track (the controller's reference; = poses when the record has none). The four sum to the change of
  log(ego / log at the ego's station) over the step.
Item 2c: r = plan 1-2 s speed / v_ref (the log's speed 1.5 s after it passed the ego's station, sl_cf_read's rule); medians; recovery =
  (r_X - r_F) / (1 - r_F), decisions tau >= 3 s, v_ref >= 2, ego >= 1, r_F < 0.95.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
from jevdrive import stats  # noqa: E402
import rf_tax as TX  # noqa: E402

ARMS = {"A0": "ACC+GAP, 10 Hz frames (stored)", "KA": "ACC+GAP, keywarp", "N0": "VCONT+LEAD, 10 Hz frames", "KN": "VCONT+LEAD, keywarp"}
PAIRS = [("KA", "A0"), ("KN", "N0"), ("A0", "N0"), ("KA", "KN")]
HAND = 1.7


def tau(R, t):
    return (np.asarray(t, float) - R.t[0]) * 1e-6 + 0.1


def scene_rows(R):
    """-> per-scene facts: score, stratum, progress."""
    m = R.o["metrics"] or {}
    gt = float(m.get("gt_dist_traveled_m") or 0.0)
    th = np.unwrap(np.asarray(R.o["log"]["Y"], float))
    turn = float(np.degrees(th.max() - th.min())) if len(th) else 0.0
    v_h = R.speed(R.t[0] + (HAND - 0.1) * 1e6)
    st = "static" if gt < 1.0 else "launch" if v_h < 1.0 else "turn" if turn > 30 else "straight"
    return dict(score=float(R.o["score"] or 0.0), stratum=st, turn_deg=turn, prog=float(m.get("progress_rel", np.nan)),
                rear=bool(m.get("collision_rear")), err=False)


def ratio_grid(R):
    out = []
    for tq in np.arange(HAND, 20.01, 0.5):
        t = R.t[0] + (tq - 0.1) * 1e6
        if t > R.t[-1]:
            break
        s = R.log_station(R.pose(t)[None, :2])[0]
        vh = R.vlog_at(s)
        if vh >= 2.0:
            out.append((tq, R.speed(t) / vh))
    return out


def vp(P):
    return float(np.hypot(*np.asarray(P, float)[1, :2])) / 1.0 if P is not None else np.nan


def split_rows(R, recs):
    """2b terms for one rollout; recs = its inference records sorted by t0."""
    t0s = np.array([r["t0"] for r in recs], np.int64)
    rows = []
    for i, r in enumerate(recs):
        j = np.searchsorted(t0s, r["t0"] + 500_000)
        if j >= len(recs) or abs(t0s[j] - r["t0"] - 500_000) > 20_000:
            continue
        t, t1 = r["t0"], int(t0s[j])
        vh, vh1 = (R.vlog_at(R.log_station(R.pose(x)[None, :2])[0]) for x in (t, t1))
        v, v1 = R.speed(t), R.speed(t1)
        vpm, vps = vp(r.get("poses_model") or r["poses"]), vp(r["poses"])
        vpt = vp(r.get("poses_track") or r["poses"])
        if min(vh, vh1) < 2 or v < 1 or min(vpm, vps, vpt, v1) < 0.3:
            continue
        rows.append(dict(tau=float(tau(R, t)), plan_model=np.log(vpm / v) - np.log(vh1 / vh), fix=np.log(vps / vpm), track=np.log(vpt / vps),
                         exec=np.log(v1 / vpt), total=np.log(v1 / vh1) - np.log(v / vh), dv_plan=(vpm - v) - (vh1 - vh), dv_fix=vps - vpm,
                         dv_track=vpt - vps, dv_exec=v1 - vpt))
    return rows


def load_arm(o, arm, a0):
    rolls = {x["scene"]: TX.Roll(x) for x in pickle.load(open(f"{o}/tax/{arm}.pkl", "rb"))["rolls"] if "ctrl" in x}   # no trace: infrastructure
    root = a0 if arm == "A0" else f"{o}/{arm}"
    recs, bad = {}, set()
    for f in glob.glob(f"{root}/runs/*/sim/aggregate/results-summary.json"):      # driver RPC errors: infrastructure, not behaviour (d327)
        bad |= {r["clipgt_id"] for r in json.load(open(f))["rollouts"] if "RpcError" in str(r.get("failure_reason") or "")}
    for f in glob.glob(f"{root}/runs/*/driver/drive.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "drive" and r.get("infer"):
                recs.setdefault(r["session"], []).append(r)
    return rolls, recs, bad


def phase(t):
    return "handoff-5 s" if t < HAND + 5 else "5-10 s" if t < HAND + 10 else "10-18 s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o", required=True), ap.add_argument("--a0", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    NUM, L = {}, ["# PAI-INPUT-LOOP tables (generated by scripts/pai_input_loop.py)", ""]
    SC, RT, SP = {}, [], []
    for arm in [k for k in ARMS if Path(f"{a.o}/tax/{k}.pkl").exists()]:
        rolls, recs, bad = load_arm(a.o, arm, a.a0)
        SC[arm] = {s: dict(scene_rows(R), err=s in bad) for s, R in rolls.items()}
        for s, R in rolls.items():
            RT += [dict(arm=arm, scene=s, tau=tq, ratio=x) for tq, x in ratio_grid(R)]
            rr = sorted(recs.get(R.o["rid"], []), key=lambda r: r["t0"])
            SP += [dict(arm=arm, scene=s, **x) for x in split_rows(R, rr)]
    RT, SP = pd.DataFrame(RT), pd.DataFrame(SP)
    RT.to_csv(out / "ratio_time.csv", index=False)

    # item 1: paired by stratum
    L += ["## Item 1: paired scene score by stratum (strata from the reference arm; log-clustered bootstrap 10 000, seed 0)", "",
          "| contrast | stratum | scenes | J mean | ref mean | J - ref [95 % CI] | contribution | J zeros | ref zeros | J progress | ref progress |",
          "|---|---|--:|--:|--:|---|--:|--:|--:|--:|--:|"]
    for j, r in [x for x in PAIRS if x[0] in SC and x[1] in SC]:
        S = sorted(s for s in set(SC[j]) & set(SC[r]) if not SC[j][s]["err"] and not SC[r][s]["err"])
        for st in ("all", "static", "launch", "straight", "turn"):
            ss = [s for s in S if st == "all" or SC[r][s]["stratum"] == st]
            if not ss:
                continue
            x, y = (np.array([SC[k][s]["score"] for s in ss]) for k in (j, r))
            p = stats.paired(x, y, groups=np.array(ss))
            c = (x - y).sum() / len(S)
            pj, pr = (np.nanmean([SC[k][s]["prog"] for s in ss]) for k in (j, r))
            L.append(f"| {j} - {r} | {st} | {len(ss)} | {x.mean():.4f} | {y.mean():.4f} | {stats.fmt(p, '+.4f')} | {c:+.4f} | "
                     f"{int((x == 0).sum())} | {int((y == 0).sum())} | {pj:.3f} | {pr:.3f} |")
            NUM[f"1/{j}-{r}/{st}"] = dict(n=len(ss), j=x.mean(), r=y.mean(), d=p["mean"], lo=p["lo"], hi=p["hi"], zj=int((x == 0).sum()),
                                         zr=int((y == 0).sum()), contrib=c, rear_j=sum(SC[j][s]["rear"] for s in ss), rear_r=sum(SC[r][s]["rear"] for s in ss))
    L += [""]

    # item 2a: ratio vs time
    RT["bin"] = np.floor(RT.tau - HAND).astype(int)
    L += ["## Item 2a: ego / log speed at the same station vs time since the handoff (median over 0.5 s steps, log >= 2 m/s)", "",
          "| s after handoff | " + " | ".join(ARMS) + " | " + " | ".join(f"n {k}" for k in ARMS) + " | share < 0.8 " + " / ".join(ARMS) + " |",
          "|--:|" + "--:|" * (2 * len(ARMS) + 1)]
    for b in range(0, 19):
        g = RT[RT.bin == b]
        if not len(g):
            continue
        med = [g[g.arm == k].ratio.median() for k in ARMS]
        n = [int((g.arm == k).sum()) for k in ARMS]
        lo = [(g[g.arm == k].ratio < 0.8).mean() for k in ARMS]
        L.append(f"| {b}-{b + 1} | " + " | ".join(f"{m:.3f}" for m in med) + " | " + " | ".join(map(str, n)) + " | " + " / ".join(f"{x:.2f}" for x in lo) + " |")
        NUM[f"2a/{b}"] = dict(zip(ARMS, med))
    # paired per-bin differences (scene-level mean ratio per bin, then scene bootstrap)
    L += ["", "Paired per scene (mean ratio of the scene's steps in the bin, scenes with steps in both arms), J - ref:", "",
          "| contrast | 0-3 s | 3-8 s | 8-13 s | 13-18 s |", "|---|---|---|---|---|"]
    W = [(0, 3), (3, 8), (8, 13), (13, 19)]
    for j, r in PAIRS:
        cells = []
        for lo_, hi_ in W:
            g = RT[(RT.bin >= lo_) & (RT.bin < hi_)].groupby(["arm", "scene"]).ratio.mean()
            ss = sorted(set(g[j].index) & set(g[r].index)) if j in g.index.get_level_values(0) and r in g.index.get_level_values(0) else []
            if len(ss) < 5:
                cells.append("-")
                continue
            p = stats.paired(g[j][ss].to_numpy(), g[r][ss].to_numpy(), groups=np.array(ss))
            cells.append(f"{stats.fmt(p, '+.3f')} (n {len(ss)})")
            NUM[f"2a/{j}-{r}/{lo_}-{hi_}"] = dict(d=p["mean"], lo=p["lo"], hi=p["hi"], n=len(ss))
        L.append(f"| {j} - {r} | " + " | ".join(cells) + " |")

    # item 2b: split
    SP["phase"] = SP.tau.map(phase)
    L += ["", "## Item 2b: per 0.5 s step change of log(ego / log) split into plan (model), serve fix, track reference, execution (pooled means)", "",
          "| arm | phase | steps | total | plan_model | fix | track | exec | plan share (model + fix + track) / total | model m/s | fix m/s | track m/s | exec m/s | losing steps: plan share |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm in ARMS:
        for ph in ("all", "handoff-5 s", "5-10 s", "10-18 s"):
            g = SP[(SP.arm == arm) & ((SP.phase == ph) if ph != "all" else True)]
            if not len(g):
                continue
            t, pm, fx, tr, ex = (g[c].mean() for c in ("total", "plan_model", "fix", "track", "exec"))
            gl = g[g.total < 0]
            shl = (gl.plan_model + gl.fix + gl.track).sum() / gl.total.sum() if len(gl) else np.nan
            L.append(f"| {arm} | {ph} | {len(g)} | {t:+.4f} | {pm:+.4f} | {fx:+.4f} | {tr:+.4f} | {ex:+.4f} | {(pm + fx + tr) / t if t else np.nan:.2f} | "
                     f"{g.dv_plan.mean():+.3f} | {g.dv_fix.mean():+.3f} | {g.dv_track.mean():+.3f} | {g.dv_exec.mean():+.3f} | {shl:.2f} |")
            NUM[f"2b/{arm}/{ph}"] = dict(n=len(g), total=t, plan_model=pm, fix=fx, track=tr, exec=ex, losing_plan_share=shl)

    # item 2c: counterfactual
    cfs = sorted(glob.glob(f"{a.o}/cf/*.pkl"))
    if cfs:
        X = {k: v for f in cfs for k, v in pickle.load(open(f, "rb")).items()}
        rolls = {x["scene"]: TX.Roll(x) for x in pickle.load(open(f"{a.o}/tax/N0.pkl", "rb"))["rolls"]}
        v12 = lambda P: float(np.hypot(*(P[2, :2] - P[1, :2])) + np.hypot(*(P[3, :2] - P[2, :2])))  # noqa: E731
        rows, g3, g10, gn = [], 0, 0, 0
        for s, recs in X.items():
            R = rolls[s]
            for r in recs:
                if r["stored"] is not None:
                    gn += 1
                    g3 += float(np.abs(r["F"][:, :2] - r["stored"][:, :2]).max()) <= 0.03
                    g10 += float(np.hypot(*(r["F"][7, :2] - r["stored"][7, :2]))) <= 0.10
                tq = float(tau(R, r["t0"]))
                s0 = R.log_station(R.pose(r["t0"])[None, :2])[0]
                tl = float(np.interp(s0, R.LS, R.LT))
                v_ref = float(np.interp(tl + 1.5e6, R.LT, R.Lv))
                rows.append(dict(scene=s, tau=tq, v_ref=v_ref, v_ego=float(r["v_ego_keys"][-1]), v_trace=float(r["v_trace_t0"]),
                                 vlog_now=float(r["v_log_keys"][-1]), vF=v12(r["F"]), vA=v12(r["A"]), vB=v12(r["B"])))
        D = pd.DataFrame(rows)
        D.to_csv(out / "cf_rows.csv.gz", index=False)
        D["rF"], D["rA"], D["rB"] = D.vF / D.v_ref, D.vA / D.v_ref, D.vB / D.v_ref
        Q = D[(D.tau >= 3) & (D.v_ref >= 2) & (D.v_ego >= 1) & (D.rF < 0.95)]
        L += ["", "## Item 2c: speed-history counterfactual on N0's stored messages (open loop)", "",
              f"Gate F = stored model plan: max |dxy| <= 0.03 m on {g3} / {gn} ({100 * g3 / max(gn, 1):.2f} %); 4 s point within 0.10 m on {g10} / {gn}. "
              f"Fed ego speed against the trace at t0: median |diff| {np.median(np.abs(D.v_ego - D.v_trace)):.3f} m/s.", "",
              "| subset | decisions | scenes | ego / log now | r_F | r_A | r_B | recovery A | recovery B |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for lab, g in (("tau >= 3 s, r_F < 0.95", Q), ("... and ego < 0.8 x log now", Q[Q.v_ego < 0.8 * Q.vlog_now]),
                       ("... tau 3-8 s", Q[Q.tau < 8]), ("... tau >= 8 s", Q[Q.tau >= 8]), ("all moving (no r_F filter)", D[(D.tau >= 3) & (D.v_ref >= 2) & (D.v_ego >= 1)])):
            if not len(g):
                continue
            rF, rA, rB = g.rF.median(), g.rA.median(), g.rB.median()
            re = (g.v_ego / g.vlog_now.where(g.vlog_now > 0.5)).median()
            L.append(f"| {lab} | {len(g)} | {g.scene.nunique()} | {re:.3f} | {rF:.3f} | {rA:.3f} | {rB:.3f} | {(rA - rF) / (1 - rF):+.2f} | {(rB - rF) / (1 - rF):+.2f} |")
            NUM[f"2c/{lab}"] = dict(n=len(g), rF=rF, rA=rA, rB=rB, recA=(rA - rF) / (1 - rF), recB=(rB - rF) / (1 - rF))
        rec = NUM["2c/tau >= 3 s, r_F < 0.95"]["recB"]
        L += ["", f"Registered line on B: recovery {rec:+.2f} -> " + ("through the history row" if rec >= 0.5 else "through the frames (not the history)" if rec <= 0.2 else "partial")]

    (out / "tables.md").write_text("\n".join(L) + "\n")
    (out / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    fig(RT, out / "pil_ratio_time.png")
    print("\n".join(L))


def fig(RT, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"A0": "#1f5fa8", "KA": "#7fb0e6", "N0": "#b8461b", "KN": "#f0a37a"}
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    for arm, lab in ARMS.items():
        g = RT[RT.arm == arm].groupby("bin").ratio
        m, n = g.median(), g.size()
        m = m[n >= 30]
        ax.plot(m.index + 0.5, m.values, "-o", ms=3, color=C[arm], label=f"{arm}: {lab}", lw=1.8 if arm in ("A0", "N0") else 1.2,
                ls="-" if arm in ("A0", "N0") else "--")
    ax.axhline(1, color="0.6", lw=0.8)
    ax.set_xlabel("time since the simulator handoff (s)")
    ax.set_ylabel("ego / human speed at the same station (median)")
    ax.set_title("NTV0-L18-s0 on PAI 281: speed ratio over the rollout", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f)


if __name__ == "__main__":
    main()
