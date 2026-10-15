#!/usr/bin/env python3
"""LC-HYSTERESIS reading (plans/2026-10-15-lc-hysteresis-prereg.md): does the 3-way route command flip L/R -> S in the middle of a required
lane change, and does the plan snap back when it does? CPU only, stored reads, never written.

  extract (Tokyo, python3 + numpy)  rf_lchyst.py extract --runs LABEL=<pai_eval root> ... --scenes <file> --out <pkl>
          per scene: every inference (k, t0, cmd, plan poses 0.5 .. 2 s, anchor in the local frame) and every route message (ts, pose, the
          first waypoint >= 5 m from the rig in the message frame = the command rule's input)
  report  (Mac, .venv)  rf_lchyst.py report --set NAME=<rf_tax pkl>,<extract pkl>:<label> ... --lcmap <rf_laneoff lc_map.csv>
          [--map-set NAME=<lc_map set>] --out <dir> --fig <png>
          -> <dir>/reading.md, decisions.csv, sites.csv and the timeline figure (validated sites, every run)

Definitions (the prereg's): window = decisions with the ego (decision anchor) at route station [s - 40, s + 60]; tangent frame at the site
station s, sg = +1 for a left site; move = sg x lateral of the ego minus its value at the first window decision; d_target = sg x lateral of
the route point at s + 40 minus the ego's; plan_y2 = sg x the plan's y at 2 s (ego frame); cmd_y = the latest message's first waypoint >= 5 m,
y (message frame). Flip = first S after a decision with cmd = the site's side. C1: d_target(k_f) >= 1.5, plan_y2(k_f - 1) >= 0.3 and
min plan_y2 over k_f .. k_f + 2 <= 0.5 plan_y2(k_f - 1). C2: d_target(k_f) in [1.5, 2.5].
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]
CMD = {0: "L", 1: "S", 2: "R", 3: "U"}


def extract(a):
    want = {l.split()[0] for l in open(a.scenes) if l.strip()}
    out = {}
    for spec in a.runs:
        lab, root = spec.split("=", 1)
        D = {}
        for f in glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")):
            for line in open(f):
                if '"scene": "clipgt' not in line:
                    continue
                r = json.loads(line)
                if r.get("scene") not in want:
                    continue
                d = D.setdefault(r["session"], dict(scene=r["scene"], dec=[], msg=[]))
                if r.get("kind") == "route":
                    wp = np.array(r["wp"], float).reshape(-1, 2)
                    wp = wp[np.isfinite(wp).all(1)]
                    far = wp[np.hypot(*wp.T) >= 5.0] if len(wp) else wp
                    d["msg"].append((int(r["ts"]), r["pose"], far[0].tolist() if len(far) else None, wp[:12].tolist()))
                elif r.get("kind") == "drive" and r.get("infer"):
                    d["dec"].append((int(r["k"]), int(r["t0"]), int(r.get("cmd", -1)), np.array(r["poses"])[:4].tolist(), r["anchor"]))
        out[lab] = {}
        for se, d in D.items():                              # one session per scene (a failed start has no inference records)
            if d["dec"]:
                d["dec"].sort(key=lambda x: x[0])
                d["msg"].sort(key=lambda x: x[0])
                out[lab][d["scene"]] = d
        print(lab, len(out[lab]), "scenes", flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)


def site_rows(name, o, d, site):
    import rf_progress as RP
    PF, TX = RP._imports()
    e = RP.Ego(o, None, PF, TX)
    s, sg = float(site["s"]), (1.0 if site["side"] == "L" else -1.0)
    p0, q = e.route_xy(np.array([s, s + 40.0]))
    psi = e.route_psi(s)
    n = np.array([-np.sin(psi), np.cos(psi)])
    rows, y0 = [], None
    ts = np.array([m[0] for m in d["msg"]], np.int64)
    for k, t0, cmd, P, an in d["dec"]:
        p = np.array(an[:2], float)
        st, _ = PF.project(e.R, e.S, p)
        if not s - 40.0 <= st <= s + 60.0:
            continue
        ye = sg * float((p - p0) @ n)
        y0 = ye if y0 is None else y0
        j = int(np.searchsorted(ts, t0, side="right") - 1)
        m = d["msg"][j] if j >= 0 else None
        rows.append(dict(set=name, scene=o["scene"], k=k, t0=t0, st_rel=st - s, cmd=CMD.get(cmd, "?"), move=ye - y0,
                         d_target=sg * float((q - p) @ n), plan_y2=sg * float(P[3][1]), cmd_y=sg * m[2][1] if m and m[2] else np.nan,
                         cmd_x=m[2][0] if m and m[2] else np.nan))
    return rows


def judge(R, side):
    out = dict(n_dec=len(R), had_side=False, flip_k=None, c1=False, c2=False)
    if not R:
        return out
    seen = False
    for i, r in enumerate(R):
        if r["cmd"] == side:
            seen = True
        elif seen and r["cmd"] == "S":
            out.update(had_side=True, flip_k=r["k"], flip_i=i, d_target=r["d_target"], move=r["move"], cmd_y=r["cmd_y"], st_rel=r["st_rel"])
            if i >= 1:
                pre = R[i - 1]["plan_y2"]
                post = min(x["plan_y2"] for x in R[i:i + 3])
                out.update(plan_pre=pre, plan_post=post)
                out["c1"] = bool(r["d_target"] >= 1.5 and pre >= 0.3 and post <= 0.5 * pre)
                out["c2"] = bool(out["c1"] and 1.5 <= r["d_target"] <= 2.5)
            break
    out["had_side"] = out["had_side"] or seen
    return out


def report(a):
    import pandas as pd
    lc = pd.read_csv(a.lcmap)
    lc = lc[lc.site.astype(bool) & (lc.gap <= 6.0)].copy()
    lc["validated"] = (lc.log_shift >= 2.0) & (lc.log_shift <= 6.0)
    mapset = dict(x.split("=", 1) for x in a.map_set)
    rows, sites = [], []
    for spec in a.set:
        name, rest = spec.split("=", 1)
        ftax, fx = rest.split(",")
        fx, lab = fx.rsplit(":", 1)
        rolls = {}
        for o in pickle.load(open(ftax, "rb"))["rolls"]:
            if "ctrl" in o:
                rolls[o["scene"]] = o
        X = pickle.load(open(fx, "rb"))[lab]
        L = lc[lc.set == mapset.get(name, name)]
        for _, st in L.iterrows():
            sc = st.scene
            if sc not in rolls or sc not in X:
                continue
            R = site_rows(name, rolls[sc], X[sc], st)
            rows += R
            j = judge(R, st.side)
            sites.append(dict(set=name, scene=sc, short=sc[7:15], side=st.side, gap=st.gap, validated=bool(st.validated), reached=bool(st.reached),
                              shift=st["shift"], in_lane_after=st.in_lane_after, log_shift=st.log_shift, **j))
    D, S = pd.DataFrame(rows), pd.DataFrame(sites)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "decisions.csv", index=False, float_format="%.3f")
    S.to_csv(out / "sites.csv", index=False, float_format="%.3f")
    f = lambda x, p="{:+.2f}": "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else p.format(x)  # noqa: E731
    L = ["# LC-HYSTERESIS reading (plans/2026-10-15-lc-hysteresis-prereg.md)", "",
         "Map lane-change sites with gap <= 6 m (rf_laneoff lc_map, d299 Amendment 3); validated = the human moved 2-6 m to that side. Per site: "
         "had the side command in the window, the first L/R -> S flip (decision k, station rel. to the site, ego move so far, d_target, "
         "the command input y at the flip), the plan's 2 s lateral toward the side at k_f - 1 and its min over k_f .. k_f + 2, C1 / C2.", ""]
    for name in S.set.unique():
        s = S[S.set == name]
        r = s[s.reached]
        L += [f"## {name}: {len(s)} sites ({int(s.validated.sum())} validated), {len(r)} reached; with the side command {int(r.had_side.sum())}; "
              f"flipped to S {int(r.flip_k.notna().sum())}; **C1 {int(r.c1.sum())}** ({', '.join(r[r.c1].short)}); C2 {int(r.c2.sum())}", "",
              "| scene | val | side | gap | reached | side cmd | flip k (st rel) | move at flip | d_target at flip | cmd y at flip | plan y2 pre / post | C1 | C2 | shift / in lane after |",
              "|:--|:--|:--|--:|:--|:--|:--|--:|--:|--:|:--|:--|:--|:--|"]
        for _, x in s.sort_values(["validated", "short"], ascending=[False, True]).iterrows():
            fk = "" if pd.isna(x.flip_k) else f"{int(x.flip_k)} ({x.st_rel:+.0f} m)"
            L.append(f"| {x.short} | {'yes' if x.validated else ''} | {x.side} | {x.gap:.1f} | {x.reached} | {x.had_side} | {fk} | {f(x.get('move'))} | "
                     f"{f(x.get('d_target'))} | {f(x.get('cmd_y'))} | {f(x.get('plan_pre'))} / {f(x.get('plan_post'))} | {x.c1} | {x.c2} | "
                     f"{f(x['shift'], '{:+.1f}')} / {x.in_lane_after} |")
        L.append("")
    V = S[S.validated]
    L += ["## Validated sites: per-decision timeline (cmd letters from s - 40 m; k of the first window decision)", ""]
    for _, x in V.iterrows():
        d = D[(D.set == x.set) & (D.scene == x.scene)]
        if len(d):
            L.append(f"- {x.set} {x.short} ({x.side}, k {int(d.k.iloc[0])}..{int(d.k.iloc[-1])}): `{''.join(d.cmd)}`")
    (out / "reading.md").write_text("\n".join(L) + "\n")
    if a.fig and len(V):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc = sorted(V.short.unique())
        sets = list(S.set.unique())
        fig, ax = plt.subplots(len(sc), len(sets), figsize=(4.2 * len(sets), 2.3 * len(sc)), squeeze=False, sharex=True)
        for i, short in enumerate(sc):
            for j, name in enumerate(sets):
                g = ax[i, j]
                d = D[(D.set == name) & (D.scene.str[7:15] == short)]
                if not len(d):
                    g.set_axis_off()
                    continue
                g.plot(d.st_rel, d.d_target, color="#1f77b4", lw=1.5, label="d_target (m)")
                g.plot(d.st_rel, d.plan_y2, color="#d62728", lw=1.2, label="plan y at 2 s toward side (m)")
                for c, col in (("L", "#2ca02c"), ("R", "#9467bd"), ("S", "#bbbbbb")):
                    m = d.cmd == c
                    g.scatter(d.st_rel[m], np.full(m.sum(), -0.8), s=6, color=col, label=f"cmd {c}")
                g.axhline(1.5, color="k", lw=0.5, ls=":")
                g.axvline(0, color="k", lw=0.5)
                g.set_title(f"{name} {short}", fontsize=8)
                if i == len(sc) - 1:
                    g.set_xlabel("route station rel. to the site (m)", fontsize=7)
                g.tick_params(labelsize=6)
        ax[0, 0].legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=130)
    print("\n".join(L[:40]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("extract")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--scenes", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("report")
    s.add_argument("--set", nargs="+", required=True)
    s.add_argument("--lcmap", required=True)
    s.add_argument("--map-set", nargs="*", default=[])
    s.add_argument("--out", required=True)
    s.add_argument("--fig", default="")
    a = ap.parse_args()
    {"extract": extract, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
