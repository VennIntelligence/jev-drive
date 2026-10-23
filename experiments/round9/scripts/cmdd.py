#!/usr/bin/env python3
"""ROUND9 CMD-DESIGN (plans/2026-10-23-command-design.md): what the ego row's 3-valued command should mean for NTV1, CPU only, stored data.

  rows    (box, repo .venv)  candidate command rules on the stored training rows: navtrain (12 shards, NAVSIM's own driving command) and the
                             AlpaSim-route pool on1_JTPHD (PAI human ph / ph05 / ph15, DAgger pd, nuPlan S rb0 / rtb1 / y / ys; stored command =
                             the shipped rule as served) -> $O/rows.parquet (per row: part, speed, label 4 s heading / lateral, route geometry, every rule)
  reads   (box, repo .venv)  every inference of the stored PAI 281 ACC + GAP reads (R6DAG / R8K / R7J / R9DL, OFF and ON) -> $O/dec.parquet
                             (round9 rtc.route_scalars + the route-geometry scalars below)
  serve   (box, repo .venv)  the candidate rules on the fed memory of every inference of stored reads (default R6DAG OFF + ON) -> $O/serve.parquet,
                             $O/serve.md (start-gap and compound-turn differences against h20_40)
  report  (box or Mac)       tables -> experiments/round9/results/cmd_design/ (tables.md + csv)

Rules (all a function of the fed old-layout route memory, 40 slots -6 .. +72 m, stored float16, so rows and serve compute the same numbers):
  nav       navtrain's stored NAVSIM driving command (ego cols 1:4; unknown when no column is set)
  stored    the pool rows' stored command (= the shipped rule, served)
  y42       the shipped rule on the memory: lateral y of the first valid slot at >= 42 m (the message's first waypoint); > 2 m L, < -2 m R
  hm20_40   h20_40's trigger, the sign of the largest persistent change in the window (not the first)
  t20       hm20_40 with the window extended to the first valid slot ahead + 4 m when nothing within 40 m is valid (start gap): the proposed command
  hD_L      route heading change against the +0 m slot's heading (rmem.cmd_heading's rule): the first slot in 0 .. L m whose change >= D deg and
            persists 4 m on; h20_40 == rmem.cmd_heading (asserted). hD_v: L = clip(4 s x speed, 20, 60) m
Route geometry (rule-free): dmax60 = max |heading change| over valid slots 0 .. 60 m; dnet = heading change at the last valid slot <= 60 m;
  y0 = the route's lateral at the ego's projection (a parallel lane: |y0| >= 2.5 m with dmax60 < 10 deg); dturn = rmem.turn_dist (30 deg).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("JEV_REPO", str(HERE.parents[2])))       # the script may run from a scratch copy on the box (uncommitted)
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/route_mem/lib"), str(REPO / "experiments/round6/scripts"),
                str(REPO / "experiments/round9/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/cmd_design"
RES = Path(os.environ.get("CMDD_RES", str(REPO / "experiments/round9/results/cmd_design")))
CR, FEAT = DATA / "runs/op_parity/cache", DATA / "runs/route_mem/feat"
POOL = "on1_JTPHD"
DEGS, LOOKS = (15, 20, 30, 45), (20, 30, 40, 60)
READS = {**{f"R6DAG-s{s}-ag": ("R6DAG-L18-s%d-281ag" % s, r) for s, r in ((0, "round6/dag/pai281/R6DAG-L18-s0-ag"), (1, "round6/dag/pai281/R6DAG-L18-s1-ag"),
                                                                       (2, "round8/s2/pai281/R6DAG-L18-s2-ag"))},
         **{f"R6DAG-s{s}-agH": ("R6DAG-L18-s%d-281agH" % s, r) for s, r in ((0, "round7/harmcl/R6DAG-L18-s0-agH"), (1, "round7/harmcl/R6DAG-L18-s1-agH"),
                                                                         (2, "round8/s2/pai281/R6DAG-L18-s2-agH"))},
         **{f"R8K-s{s}-{m}": (f"R8K-L18-s{s}-281{m}", f"round8/{'s2' if s == 2 else 'r8k'}/pai281/R8K-L18-s{s}-{m}") for s in (0, 1, 2) for m in ("ag", "agH")},
         **{f"R7J-s{s}-{m}": (f"R7J-L18-s{s}-281{m}", f"round7/joint/pai281/R7J-L18-s{s}-{m}") for s in (0, 1) for m in ("ag", "agH")},
         **{f"R9DL-s{s}-{m}": (f"R9DL-L18-s{s}-281{m}", f"round9/dag_lane/pai281/R9DL-L18-s{s}-{m}") for s in (0, 1) for m in ("ag", "agH")}}
HEAD = ("R7J", "R9DL")                                   # served and trained with the heading command (JEV_CMD=heading, --cmd-rule heading)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def geom(f):
    """(n, 200) float16 memory -> dict of (n,) route-geometry arrays and the pieces the rules need."""
    import rmem as RM
    P = np.asarray(f, np.float32).reshape(-1, RM.N_PT, RM.NF)
    v = P[..., 0] > 0.5
    h = np.arctan2(P[..., 4], P[..., 3])
    ref = np.where(v[:, RM.I0], h[:, RM.I0], 0.0)
    d = np.degrees(wrap(h - ref[:, None]))
    w60 = v & (RM.OFFS >= 0) & (RM.OFFS <= 60)
    dmax = np.where(w60.any(1), np.abs(np.where(w60, d, 0)).max(1), np.nan)
    last = np.where(w60.any(1), RM.N_PT - 1 - np.argmax(w60[:, ::-1], 1), 0)
    dnet = np.where(w60.any(1), d[np.arange(len(d)), last], np.nan)
    y = P[..., 2] * 20.0
    y0 = np.where(v[:, RM.I0], y[:, RM.I0], np.nan)
    g42 = v & (RM.OFFS >= 42)
    j = np.argmax(g42, 1)
    y42 = np.where(g42.any(1), y[np.arange(len(y)), j], np.nan)
    return dict(v=v, d=d, dmax=dmax, dnet=dnet, y0=y0, y42=y42, dturn=RM.turn_dist(f).astype(np.float32), nval=v.sum(1))


def rule_h(G, deg, look, dom=False, gap=False):
    """rmem.cmd_heading generalised: deg (scalar), look (scalar or (n,) m) -> (n,) 0 L / 1 S / 2 R. dom: the sign of the largest persistent
    change within the window instead of the first one (an S-shaped approach, left jog then right turn, reads R, not L). gap: the window ends
    at max(look, s0 + 4 m), s0 = the first valid slot ahead, so a memory that still starts ~42 m ahead (the episode start, a car waiting at
    its first stop line) is read on its first two slots instead of being taken as STRAIGHT; no effect when a slot within the window is valid."""
    import rmem as RM
    v, d = G["v"], G["d"]
    c = v & (np.abs(d) >= deg)
    c[:, :-2] &= c[:, 2:] | ~v[:, 2:]
    L = np.broadcast_to(np.asarray(look, np.float64), (len(d),))
    ah = v & (RM.OFFS[None] >= 0)
    s0 = np.where(ah.any(1), RM.OFFS[np.argmax(ah, 1)], 0.0) if gap else np.zeros(len(d))
    c &= (RM.OFFS[None] >= 0) & (RM.OFFS[None] <= np.maximum(L, s0 + 4.0)[:, None])
    j = np.argmax(np.where(c, np.abs(d), -1.0), 1) if dom else np.argmax(c, 1)
    dj = d[np.arange(len(d)), j]
    return np.where(c.any(1), np.where(dj > 0, 0, 2), 1).astype(np.int8)


def rules(f, spd):
    import rmem as RM
    G = geom(f)
    R = {"y42": np.where(np.isnan(G["y42"]), 3, np.where(G["y42"] > 2, 0, np.where(G["y42"] < -2, 2, 1))).astype(np.int8)}
    for D in DEGS:
        for L in LOOKS:
            R[f"h{D}_{L}"] = rule_h(G, D, L)
        R[f"h{D}_v"] = rule_h(G, D, np.clip(4.0 * np.asarray(spd, np.float64), 20, 60))
    R["hm20_40"] = rule_h(G, 20, 40, dom=True)
    R["t20"] = rule_h(G, 20, 40, dom=True, gap=True)                 # the proposed definition (plans/2026-10-23-command-design.md)
    k = np.random.default_rng(0).choice(len(f), min(len(f), 20000), replace=False)
    assert np.array_equal(R["h20_40"][k], RM.cmd_heading(f[k]).argmax(1)), "rule_h(20, 40) != rmem.cmd_heading"
    return {("y42m" if k_ == "y42" else k_): G[k_] for k_ in ("dmax", "dnet", "y0", "y42", "dturn", "nval")}, R


def lab(c):
    return np.where(c.max(1) > 0.5, c.argmax(1), 3).astype(np.int8)


# ---------------------------------------------------------------- rows
def cmd_rows(a):
    import pandas as pd
    out = []
    for s in range(12):
        n = f"navtrain_full.s{s}of12"
        T, F = np.load(CR / n / "tab.npz"), np.load(FEAT / f"{n}.npz")
        assert np.array_equal(T["names"], F["names"]), n
        G, R = rules(F["f"], T["speed"])
        df = pd.DataFrame(dict(part="navtrain", spd=T["speed"], th4=np.degrees(wrap(T["fut"][:, 7, 2].astype(np.float64))), y4=T["fut"][:, 7, 1],
                               stored=lab(T["ego"][:, 1:4]), ok=F["ok"], jdist=F["jdist"], dev=F["dev"], **G, **R))
        out.append(df)
        print(n, len(df), flush=True)
    T, F = np.load(CR / POOL / "tab.npz"), np.load(FEAT / f"{POOL}.npz")
    assert np.array_equal(T["names"], F["names"])
    G, R = rules(F["f"], T["speed"])
    out.append(pd.DataFrame(dict(part=T["pool"].astype(str), spd=T["speed"], th4=np.degrees(wrap(T["fut"][:, 7, 2].astype(np.float64))), y4=T["fut"][:, 7, 1],
                                 stored=lab(T["ego"][:, 1:4]), ok=F["ok"], jdist=np.nan, dev=False, scene=T["log"].astype(str), **G, **R)))
    D = pd.concat(out, ignore_index=True)
    O.mkdir(parents=True, exist_ok=True)
    D.to_parquet(O / "rows.parquet")
    print("rows", D.groupby("part").size().to_dict(), flush=True)


# ---------------------------------------------------------------- reads
def _one(job):
    import rtc
    f = job
    rows, _ = rtc._one_log((f, set()))
    if not rows:
        return []
    return rows


def cmd_reads(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    import rtc
    jobs = [(k, f) for k, (_, root) in READS.items() for f in sorted(glob.glob(f"{DATA}/runs/{root}/runs/*/driver/drive.jsonl"))]
    res = par.pmap(_one, [f for _, f in jobs], workers=min(len(jobs), a.workers or n_cpus()), desc="logs")
    res.raise_if_failed()
    allr = []
    for k in READS:
        mine = [x for i, (kk, _) in enumerate(jobs) if kk == k for x in res.values[i]]
        best = {}                                            # a scene read twice (retry): keep the log with most decisions
        for x in mine:
            best.setdefault(x["scene"], {}).setdefault(x["log"], 0)
            best[x["scene"]][x["log"]] += 1
        keep = {(s, max(c, key=c.get)) for s, c in best.items()}
        allr += [dict(read=k, **x) for x in mine if (x["scene"], x["log"]) in keep]
        print(k, len(best), "scenes", flush=True)
    D = pd.DataFrame(allr).drop(columns="log")
    O.mkdir(parents=True, exist_ok=True)
    D.to_parquet(O / "dec.parquet")
    print("decisions", len(D), flush=True)


# ---------------------------------------------------------------- serve: every candidate rule on the fed memory of stored reads
def _serve_one(f):
    import json as js
    out = []
    for line in open(f):
        if '"infer": true' not in line or '"kind": "drive"' not in line:
            continue
        r = js.loads(line)
        if "rmem" not in r:
            continue
        out.append((r["scene"], int(r["t0"]), int(r["cmd"]), float(np.asarray(r["ego"], np.float32)[4] * 10), np.asarray(r["rmem"], np.float16)))
    if not out:
        return None
    sc, t0, cmd, v, F = zip(*out)
    F = np.stack(F)
    G, R = rules(F, np.asarray(v))
    return dict(log=[f] * len(sc), scene=list(sc), t0=list(t0), cmd=list(cmd), v0=list(v), **{k: G[k] for k in ("dmax", "y0", "nval")},
                **{k: R[k] for k in ("y42", "h20_40", "hm20_40", "t20")})


def cmd_serve(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    keys = [k for k in READS if any(k.startswith(p) for p in a.reads)]
    jobs = [(k, f) for k in keys for f in sorted(glob.glob(f"{DATA}/runs/{READS[k][1]}/runs/*/driver/drive.jsonl"))]
    res = par.pmap(_serve_one, [f for _, f in jobs], workers=min(len(jobs), a.workers or n_cpus()), desc="logs")
    res.raise_if_failed()
    D = pd.concat([pd.DataFrame(dict(read=k, **v)) for (k, _), v in zip(jobs, res.values) if v is not None], ignore_index=True)
    D = D.drop(columns="log")
    D.to_parquet(O / "serve.parquet")
    lr = lambda c: np.isin(c, [0, 2])  # noqa: E731
    L = ["", "## 7. Candidate rules on the fed memory of stored reads (served = shipped rule)", "",
         f"Reads: {', '.join(keys)}; decisions {len(D)}.", "", "| rule | L/R share | differs from h20_40 | differs from served | L/R with no slot at the ego, stopped |",
         "|:--|--:|--:|--:|--:|"]
    for r in ("y42", "h20_40", "hm20_40", "t20"):
        g = D[(D.v0 < 0.5) & ~np.isfinite(D.y0)]
        L.append(f"| {r} | {100 * lr(D[r]).mean():.1f} % | {100 * (D[r] != D.h20_40).mean():.2f} % | {100 * (lr(D[r]) != lr(D.cmd)).mean():.1f} % | "
                 f"{100 * lr(g[r]).mean():.1f} % (n {len(g)}) |")
    sc = D.groupby("scene").apply(lambda X: (X.t20 != X.h20_40).mean()).sort_values(ascending=False)
    L += ["", "Scenes where t20 and h20_40 differ most (share of decisions): " + ", ".join(f"{s[7:15]} {100 * x:.0f} %" for s, x in sc.head(10).items())]
    (O / "serve.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- report
def pct(x):
    return f"{100 * float(x):.1f} %" if x == x else "-"


def boot(diff_fn, scenes, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scenes)
    b = [diff_fn(s[rng.integers(0, len(s), len(s))]) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def cmd_report(a):
    import pandas as pd
    RES.mkdir(parents=True, exist_ok=True)
    L = ["# CMD-DESIGN tables (cmdd.py report; generated)", ""]
    D = pd.read_parquet(Path(a.o) / "rows.parquet")
    D["turn"] = pd.cut(D.th4.abs(), [-1, 10, 30, 60, 400], labels=["0-10", "10-30", "30-60", ">=60"])
    D["geo"] = np.select([D.dmax >= 45, D.dmax >= 15, D.dmax >= 5, D.dmax >= 0], ["turn >=45", "bend 15-45", "gentle 5-15", "straight <5"], "no route")
    D["par"] = (D.y0.abs() >= 2.5) & (D.dmax < 10)
    mv = D.spd >= 4
    parts = ["navtrain", "ph", "ph05", "ph15", "pd", "rb0", "rtb1", "y", "ys"]
    rl = ["stored", "y42"] + [f"h{d}_{l}" for d in DEGS for l in LOOKS] + [f"h{d}_v" for d in DEGS] + ["hm20_40", "t20"]
    isLR = lambda c: np.isin(c, [0, 2])  # noqa: E731

    # 1. L/R share by the label's 4 s heading, moving rows
    L += ["## 1. L/R share by the label's 4 s heading change (rows >= 4 m/s)", "",
          "Stored = NAVSIM's command for navtrain, the shipped rule (as served) for the pool parts. Sign = L/R agrees with the label's turn sign on "
          "rows >= 30 deg.", "", "| part | rule | n | 0-10 | 10-30 | 30-60 | >= 60 | sign |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    T1 = []
    for p in parts:
        X = D[(D.part == p) & mv]
        for r in ["stored", "y42", "h20_40", "hm20_40", "t20", "h30_40", "h30_30", "h45_40", "h30_v", "h45_v"]:
            g = X.groupby("turn", observed=False)[r].apply(lambda c: isLR(c).mean())
            big = X[X.th4.abs() >= 30]
            sg = ((big[r] == 0) & (big.th4 > 0) | (big[r] == 2) & (big.th4 < 0))[isLR(big[r])].mean()
            T1.append(dict(part=p, rule=r, n=len(X), **{k: g[k] for k in g.index}, sign=sg))
            L.append(f"| {p} | {r} | {len(X)} | " + " | ".join(pct(g[k]) for k in g.index) + f" | {pct(sg)} |")
    pd.DataFrame(T1).to_csv(RES / "lr_by_turn.csv", index=False)

    # 2. navtrain: agreement with NAVSIM's command, every rule, all rows and by geometry
    N = D[D.part == "navtrain"]
    L += ["", "## 2. navtrain: agreement of each rule with NAVSIM's driving command (all rows; route memory as rebuilt for the row)", "",
          f"NAVSIM command shares: " + ", ".join(f"{k} {pct(v)}" for k, v in zip("LSRU", [(N.stored == i).mean() for i in range(4)])), "",
          "| rule | agree all | kappa | agree moving | NAVSIM L/R recalled | rule L/R that NAVSIM calls S | turn >=45 geo agree | straight <5 geo: rule L/R |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    T2 = []
    for r in rl[1:]:
        ag = (N[r] == N.stored).mean()
        po = ag
        pe = sum((N[r] == i).mean() * (N.stored == i).mean() for i in range(4))
        kap = (po - pe) / (1 - pe)
        rec = (N[r] == N.stored)[isLR(N.stored)].mean()
        fp = (N.stored == 1)[isLR(N[r])].mean()
        tg = N[N.geo == "turn >=45"]
        sg = N[N.geo == "straight <5"]
        row = dict(rule=r, agree=ag, kappa=kap, agree_mv=(N[r] == N.stored)[N.spd >= 4].mean(), recall=rec, false_lr=fp,
                   turn_agree=(tg[r] == tg.stored).mean(), straight_lr=isLR(sg[r]).mean())
        T2.append(row)
        L.append(f"| {r} | {pct(ag)} | {kap:.3f} | {pct(row['agree_mv'])} | {pct(rec)} | {pct(fp)} | {pct(row['turn_agree'])} | {pct(row['straight_lr'])} |")
    pd.DataFrame(T2).to_csv(RES / "nav_agree.csv", index=False)

    # 3. when does the command light: L/R share by the distance to the route's turn (turn_dist, 30 deg), speed bins; navtrain NAVSIM vs rules
    L += ["", "## 3. Lighting distance: L/R share by the route's distance to the next >= 30 deg heading change (rmem.turn_dist), by speed", "",
          "Rows whose memory holds a turn (dmax60 >= 45 deg). navtrain NAVSIM = the shipped weights' meaning.", ""]
    D["dtb"] = pd.cut(D.dturn, [-1, 0.5, 10, 20, 30, 40, 50, 60.5], labels=["at/inside", "0-10", "10-20", "20-30", "30-40", "40-50", "50-60"])
    D["vb"] = pd.cut(D.spd, [-1, 0.5, 4, 8, 12, 50], labels=["stop", "0.5-4", "4-8", "8-12", ">=12"])
    T3 = []
    for p, rr in (("navtrain", ["stored", "y42", "h20_40", "t20", "h30_40", "h30_v", "h45_v"]), ("ph", ["stored", "h20_40", "t20", "h30_40", "h30_v"])):
        X = D[(D.part == p) & (D.dmax >= 45)]
        for r in rr:
            g = X.groupby(["vb", "dtb"], observed=False)[r].apply(lambda c: isLR(c).mean() if len(c) else np.nan).unstack()
            n = X.groupby(["vb", "dtb"], observed=False).size().unstack()
            L += [f"**{p} / {r}** (cell n in brackets)", "", "| speed | " + " | ".join(g.columns) + " |", "|:--|" + "--:|" * len(g.columns)]
            for vb in g.index:
                L.append(f"| {vb} | " + " | ".join(f"{pct(g.loc[vb, c])} ({n.loc[vb, c]})" for c in g.columns) + " |")
                T3 += [dict(part=p, rule=r, vb=vb, dtb=c, lr=g.loc[vb, c], n=n.loc[vb, c]) for c in g.columns]
            L.append("")
    pd.DataFrame(T3).to_csv(RES / "lighting.csv", index=False)

    # 4. per part: L/R on non-turn geometry, parallel-lane rows, changed rows against the stored command
    L += ["## 4. Per part: L/R share by route geometry, the parallel-lane artefact, rows changed against the stored command (all speeds)", "",
          "| part | rule | n | straight <5 | gentle 5-15 | bend 15-45 | turn >=45 | parallel lane (|y0| >= 2.5, dmax < 10): n / L/R | changed vs stored |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    T4 = []
    for p in parts:
        X = D[D.part == p]
        for r in ["stored", "y42", "h20_40", "hm20_40", "t20", "h30_40", "h30_v", "h45_v"]:
            g = X.groupby("geo")[r].apply(lambda c: isLR(c).mean())
            pr = X[X.par]
            ch = (X[r] != X.stored).mean()
            T4.append(dict(part=p, rule=r, n=len(X), **{k: g.get(k, np.nan) for k in ["straight <5", "gentle 5-15", "bend 15-45", "turn >=45"]},
                           par_n=len(pr), par_lr=isLR(pr[r]).mean(), changed=ch))
            L.append(f"| {p} | {r} | {len(X)} | " + " | ".join(pct(g.get(k, np.nan)) for k in ["straight <5", "gentle 5-15", "bend 15-45", "turn >=45"])
                     + f" | {len(pr)} / {pct(isLR(pr[r]).mean())} | {pct(ch)} |")
    pd.DataFrame(T4).to_csv(RES / "by_geo.csv", index=False)

    # 5. the human's label on the parallel-lane rows and the PAI non-turn L/R rows
    L += ["", "## 5. What the label does where the shipped rule fires L/R on non-turn rows (moving, |label 4 s heading| < 10 deg)", "",
          "| part | n | stored L/R | of these: route geometry straight/gentle | parallel lane | label 4 s lateral toward the command > 1.5 m |",
          "|:--|--:|--:|--:|--:|--:|"]
    for p in parts:
        X = D[(D.part == p) & mv & (D.th4.abs() < 10)]
        Y = X[isLR(X.stored)]
        tow = np.where(Y.stored == 0, Y.y4, -Y.y4) > 1.5
        L.append(f"| {p} | {len(X)} | {pct(isLR(X.stored).mean())} | {pct(Y.geo.isin(['straight <5', 'gentle 5-15']).mean())} | {pct(Y.par.mean())} | "
                 f"{pct(tow.mean())} |")

    # 6. stored reads: heading-command arms vs the R6DAG / R8K reads
    if (Path(a.o) / "dec.parquet").exists():
        L += reads_report(a, pd)
    (RES / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def reads_report(a, pd):
    L = ["", "## 6. Stored PAI 281 reads: heading-command arms (R7J, R9DL) vs R6DAG / R8K, OFF and ON", ""]
    Fct = {k: pd.read_csv(DATA / f"runs/pai_fix/facts/{t}.csv") for k, (t, _) in READS.items()}
    common = sorted(set.intersection(*[set(F[~F.rpc.astype(bool)].scene) for F in Fct.values()]))
    off = pd.DataFrame({k: F.set_index("scene").loc[common].fclass.eq("offroad").astype(float) for k, F in Fct.items()})
    sc = pd.DataFrame({k: F.set_index("scene").loc[common].score for k, F in Fct.items()})
    L += [f"Common scenes {len(common)}. Offroad zeros per read: " + ", ".join(f"{k} {int(off[k].sum())}" for k in off), ""]
    L += ["| harmonizer | heading reads | others | offroad / read heading | others | diff [95 % scene bootstrap] | mean heading | others | diff |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for m in ("ag", "agH"):
        H = [k for k in off if k.endswith("-" + m) and k.split("-")[0] in HEAD]
        Nn = [k for k in off if k.endswith("-" + m) and k.split("-")[0] not in HEAD]
        f = lambda s, X=off: X.loc[s, H].mean(1).sum() - X.loc[s, Nn].mean(1).sum()  # noqa: E731
        ci = boot(lambda s: f(s), common)
        g = lambda s: sc.loc[s, H].mean(1).mean() - sc.loc[s, Nn].mean(1).mean()  # noqa: E731
        cg = boot(g, common)
        L.append(f"| {m} | {len(H)} | {len(Nn)} | {off[H].sum().mean():.2f} | {off[Nn].sum().mean():.2f} | {f(common):+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                 f"{sc[H].mean().mean():.4f} | {sc[Nn].mean().mean():.4f} | {g(common):+.4f} [{cg[0]:+.4f}, {cg[1]:+.4f}] |")
    # per-scene offroad rate difference: where does the excess sit
    ex = (off[[k for k in off if k.split("-")[0] in HEAD]].mean(1) - off[[k for k in off if k.split("-")[0] not in HEAD]].mean(1)).sort_values()
    L += ["", "Scenes with the largest offroad-rate excess of the heading reads (rate heading - rate others; + = heading worse):", "",
          "| scene | excess | heading rate | others rate | primaries (heading reads) |", "|:--|--:|--:|--:|:--|"]
    for s in list(ex.index[-12:][::-1]) + list(ex.index[:4]):
        hp = [Fct[k].set_index("scene").loc[s].primary for k in off if k.split("-")[0] in HEAD and off.loc[s, k] > 0]
        L.append(f"| {s[7:15]} | {ex[s]:+.2f} | {off.loc[s, [k for k in off if k.split('-')[0] in HEAD]].mean():.2f} | "
                 f"{off.loc[s, [k for k in off if k.split('-')[0] not in HEAD]].mean():.2f} | {', '.join(sorted(set(hp)))} |")
    pd.DataFrame(dict(excess=ex)).to_csv(RES / "offroad_excess.csv")

    # decisions: command disagreement before offroad events; bend behaviour
    Dd = pd.read_parquet(Path(a.o) / "dec.parquet")
    Dd["arm"] = Dd.read.str.split("-").str[0]
    Dd["hd"] = Dd.arm.isin(HEAD)
    Dd["harm"] = Dd.read.str.endswith("agH")
    Dd["old"] = np.where(Dd.y_rt42.isna(), 1, np.where(Dd.y_rt42 > 2, 0, np.where(Dd.y_rt42 < -2, 2, 1)))
    srv_old = Dd[~Dd.hd]
    agree_proxy = (srv_old.old == srv_old.cmd).mean()
    c0 = {}                                                  # event clock = the controller's first timestamp + t_ev_s (round9 rtc's rule)
    for k, (t, _) in READS.items():
        tax = pickle.load(open(DATA / f"runs/pai_fix/tax/{t}.pkl", "rb"))["rolls"]
        c0.update({(k, o["scene"]): float(o["ctrl"]["t"][0]) for o in tax if "ctrl" in o})
    Dd["c0"] = [c0.get((r, s), np.nan) for r, s in zip(Dd.read, Dd.scene)]
    Dd["t_s"] = (Dd.t0 - Dd.c0) * 1e-6
    L += ["", f"Old-rule proxy (y at the 42 m slot, +-2 m) against the served command on the non-heading reads: {pct(agree_proxy)} agree.", ""]
    ev = []
    for k, F in Fct.items():
        z = F[(F.fclass == "offroad") & F.t_ev_s.notna()]
        ev += [dict(read=k, scene=s, t_ev=t, primary=p) for s, t, p in zip(z.scene, z.t_ev_s, z.primary)]
    E = pd.DataFrame(ev)
    W = Dd.merge(E, on=["read", "scene"])
    W = W[(W.t_s >= W.t_ev - 3.0) & (W.t_s <= W.t_ev)]
    mvd = Dd[Dd.v0 >= 2]
    L += ["Decisions in the 3 s before each offroad event against all moving decisions (v0 >= 2 m/s): command types (old = shipped rule on the fed "
          "memory, h = heading rule 20 deg / 40 m).", "",
          "| set | decisions | events | old L/R & h S (curve cue removed) | old S & h L/R | old L/R & h L/R | both S | median |dth40| | median v0 |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, X, e in (("heading reads, offroad lead-up", W[W.hd], E[E.read.str.split("-").str[0].isin(HEAD)]),
                       ("others, offroad lead-up", W[~W.hd], E[~E.read.str.split("-").str[0].isin(HEAD)]),
                       ("heading reads, all moving", mvd[mvd.hd], None), ("others, all moving", mvd[~mvd.hd], None)):
        oL, hL = np.isin(X.old, [0, 2]), np.isin(X.cmd_h, [0, 2])
        L.append(f"| {name} | {len(X)} | {'-' if e is None else len(e)} | {pct((oL & ~hL).mean())} | {pct((~oL & hL).mean())} | {pct((oL & hL).mean())} | "
                 f"{pct((~oL & ~hL).mean())} | {X.dth40.abs().median():.1f} | {X.v0.median():.1f} |")
    # bends: plan lateral vs the route at the plan's 4 s arc, signed toward the inside of the bend
    B = Dd[(Dd.v0 >= 5) & Dd.dth40.abs().between(10, 45) & Dd.lat4.notna()].copy()
    B["lin"] = B.lat4 * np.sign(B.dth40)
    B["cue"] = np.where(np.isin(B.old, [0, 2]), "old L/R", "old S")
    L += ["", "Bends (10-45 deg route heading change within 40 m, v0 >= 5 m/s): the plan's 4 s point against the route, signed toward the bend's inside "
          "(+ = cuts inside).", "", "| group | harmonizer | old-rule cue | decisions | served L/R | median | p10 | p90 | > 1 m inside | > 1 m outside |",
          "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for (h, hm, cue), X in B.groupby(["hd", "harm", "cue"]):
        L.append(f"| {'heading' if h else 'others'} | {'ON' if hm else 'OFF'} | {cue} | {len(X)} | {pct(np.isin(X.cmd, [0, 2]).mean())} | {X.lin.median():+.2f} | "
                 f"{X.lin.quantile(.1):+.2f} | {X.lin.quantile(.9):+.2f} | {pct((X.lin > 1).mean())} | {pct((X.lin < -1).mean())} |")
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows")
    p = sub.add_parser("reads"); p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("serve"); p.add_argument("--reads", nargs="+", default=["R6DAG"]); p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("report"); p.add_argument("--o", default=str(O))
    a = ap.parse_args()
    dict(rows=cmd_rows, reads=cmd_reads, serve=cmd_serve, report=cmd_report)[a.cmd](a)
