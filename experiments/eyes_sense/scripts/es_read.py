#!/usr/bin/env python3
"""EYES-vs-SENSE read-out (plans/2026-10-11-eyes-sense-prereg.md): the 2 x 2 on navtest by flips and decision 248's no-regression table, navhard,
the test-time options, the registered side-view verdicts, the headline rows, and the 700-scene nuPlan closed loop of (a) / (c).

  box   (box, envs/navsim2, one pool CPU job): vis_train's instrumented replay of every failing navtest token of this phase's reads (cached per model
        under $DATA_DIR/runs/eyes_sense/reads/tok/), then the tables -> $DATA_DIR/runs/eyes_sense/reads/out/{contrast_tables.md, cells.csv, flips.csv,
        stats.json}
  sync  (Mac): `box` as one pool CPU job, wait, copy out/* to experiments/eyes_sense/results/
Every token-level definition (turn buckets, off-road, cut-inside, under-turn, NC classes, the by-log cluster bootstrap) is experiments/vis_train/
scripts/vt_read.py's, imported; the closed-loop helpers are experiments/alpasim/scripts/joint1_report.py's and c0b_report.py's.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_REPO = _H.parents[3]
_sys.path[:0] = [str(_REPO / "experiments/vis_train/scripts"), str(_REPO / "experiments/alpasim/scripts"), str(_REPO / "experiments/body1/scripts"), str(_REPO)]
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import vt_read as V  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/eyes_sense/reads"
RES = _REPO / "experiments/eyes_sense/results"
OPTS = ("noside", "mshuf", "sideoff")
SIDE = [("b", "a"), ("d", "c")]                  # registered side-view comparisons
HEAD = [("c", "a"), ("d", "b")]                  # headline rows (descriptive)
REF = [("d", "a")]                               # both against none (descriptive)
FLC = [("cut_in", ">45"), ("offroad", ">45"), ("under", ">45"), ("offroad", ">20"), ("cut_in", ">20"), ("offroad", "all"), ("nc_fail", "all"), ("nc_A", "all")]


def arms():
    """arm letter -> tag stem; (a) / (c) follow the identity rule (results/identity.json)."""
    f = RES / "identity.json"
    idt = json.loads(f.read_text()) if f.exists() else {"a": "P2H10-F", "c": "P2H10S-F"}
    return {"a": idt["a"], "b": "ESb", "c": idt["c"], "d": "ESd"}, idt


def specs():
    """vt_read.replay's list: (arm, seed, step, bench spec) for every read of this phase."""
    A, _ = arms()
    out = [(k, s, 10, f"{v}-s{s}") for k, v in A.items() for s in (0, 1)]
    out += [(f"{k}:{o}", s, 10, f"{A[k]}-s{s}:{o}") for k in ("b", "d") for o in OPTS for s in (0, 1)]
    return out


def build(run):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.bench import tables as BT
    A, idt = arms()
    tab = np.load(V.TAB)
    names, fut = tab["names"].astype(str), tab["fut"].astype(np.float64)
    sgn = np.sign(fut[:, -1, 2])
    M, _ = V.masks(names, BT.navtest_strata())
    X, have, logs = {}, [], None
    for arm, s, _, spec in specs():
        if V.tokf(spec).exists():
            df = pd.read_parquet(V.tokf(spec)).set_index("token").reindex(names)
            X[(arm, s)] = V.feats(df, sgn)
            have.append(spec)
            logs = df["log"].astype(str).to_numpy()
    assert logs is not None, "no navtest token table yet"
    cb = V.CB(logs)
    col = {k: j for j, k in enumerate(V.METRICS)}
    rows, flips = [], []

    def side(arm, seed):
        ks = [(arm, s) for s in ((0, 1) if seed == "mean" else (seed,))]
        return np.mean([X[k] for k in ks], 0) if all(k in X for k in ks) else None

    comps = [(x, y, "side") for x, y in SIDE] + [(x, y, "head") for x, y in HEAD] + [(x, y, "ref") for x, y in REF] + \
            [(f"{k}:{o}", k, "option") for k in ("b", "d") for o in OPTS]
    for x, y, kind in comps:
        for seed in ("mean", 0, 1):
            xa, xc = side(x, seed), side(y, seed)
            if xa is None or xc is None:
                continue
            for b, m in M.items():
                mu, lo, hi, n = cb.mean(xa - xc, m)
                va, vc = cb.mean(xa, m)[0], cb.mean(xc, m)[0]
                for j, k in enumerate(V.METRICS):
                    rows.append(dict(kind=kind, arm=x, control=y, seed=str(seed), board="navtest", bucket=b, metric=k, n=int(n[j]), value=va[j], ctrl_value=vc[j],
                                     diff=mu[j], ci_lo=lo[j], ci_hi=hi[j]))
        if kind == "option":
            continue
        for k, b in FLC:                                              # flips: per seed pair, then summed
            fx = br = na = nc = 0
            for s in (0, 1):
                if (x, s) not in X or (y, s) not in X:
                    continue
                fa = (X[(x, s)][:, col[k]] > 50)[M[b]]
                fc = (X[(y, s)][:, col[k]] > 50)[M[b]]
                fx, br, na, nc = fx + int((fc & ~fa).sum()), br + int((~fc & fa).sum()), na + int(fa.sum()), nc + int(fc.sum())
            flips.append(dict(arm=x, control=y, kind=kind, cls=k, bucket=b, fixed=fx, broken=br, net=fx - br, ctrl_fail=nc, arm_fail=na))
    # navhard: per seed and seed mean, paired by group, CI by the stage-1 log
    U = {}
    for k, v in A.items():
        for s in (0, 1):
            u, _ = BT.load("navhard", f"{v}-s{s}@gimm")
            if u is not None:
                U[(k, s)] = u
    for x, y, kind in comps:
        if kind == "option":
            continue
        for seed in ("mean", 0, 1):
            ss = (0, 1) if seed == "mean" else (seed,)
            if not all((x, s) in U and (y, s) in U for s in ss):
                continue
            ix = U[(x, ss[0])].index
            for s in ss:
                ix = ix.intersection(U[(x, s)].index).intersection(U[(y, s)].index)
            lg = U[(x, ss[0])].loc[ix, "log"].astype(str).to_numpy()
            for c in ("combined", "stage1", "stage2"):
                a_ = np.mean([U[(x, s)].loc[ix, c].to_numpy(float) for s in ss], 0)
                c_ = np.mean([U[(y, s)].loc[ix, c].to_numpy(float) for s in ss], 0)
                r = stats.paired(a_, c_, lg)
                rows.append(dict(kind=kind, arm=x, control=y, seed=str(seed), board="navhard", bucket="all", metric=c, n=r["n"], value=r["mean_a"],
                                 ctrl_value=r["mean_b"], diff=r["mean"], ci_lo=r["lo"], ci_hi=r["hi"]))
    R = pd.DataFrame(rows)
    FL = pd.DataFrame(flips)
    q = lambda x, y, b, k, seed="mean", board="navtest": R[(R.arm == x) & (R.control == y) & (R.bucket == b) & (R.metric == k) & (R.seed == str(seed)) & (R.board == board)]  # noqa: E731

    def cell(x, y, b, k, seed="mean", board="navtest"):
        r = q(x, y, b, k, seed, board)
        return None if not len(r) else r.iloc[0]

    def ci(r, f="+.2f"):
        return "n/a" if r is None else f"{r['diff']:{f}} [{r['ci_lo']:{f}}, {r['ci_hi']:{f}}]"
    # ---- registered side-view verdicts
    st = dict(arms=A, identity=idt, have=have, verdicts={}, channel={})
    for k in ("b", "d"):
        r = cell(f"{k}:mshuf", k, "all", "EPDMS")
        st["channel"][k] = None if r is None else dict(arm_minus_mshuf=-float(r["diff"]), read=bool(-r["diff"] >= 0.2),
                                                       noside=None if cell(f"{k}:noside", k, "all", "EPDMS") is None else -float(cell(f"{k}:noside", k, "all", "EPDMS")["diff"]),
                                                       sideoff=None if cell(f"{k}:sideoff", k, "all", "EPDMS") is None else -float(cell(f"{k}:sideoff", k, "all", "EPDMS")["diff"]))
    for x, y in SIDE:
        ci_ = cell(x, y, ">45", "cut_in")
        off = cell(x, y, ">45", "offroad")
        g = {"EPDMS < 5": cell(x, y, "<5", "EPDMS"), "NC": cell(x, y, "all", "NC"), "EC": cell(x, y, "all", "EC")}
        if ci_ is None or off is None or any(v is None for v in g.values()):
            st["verdicts"][f"{x}-{y}"] = dict(verdict="pending (reads missing)")
            continue
        down = [k for k, v in g.items() if v["ci_hi"] < 0]
        read = (st["channel"].get(x) or {}).get("read")
        ok = ci_["ci_hi"] < 0 and off["ci_hi"] < 0 and not down
        verdict = "not measured" if read is False else ("REOPEN" if ok else "stays closed") if read else "pending (channel read missing)"
        st["verdicts"][f"{x}-{y}"] = dict(cut_in_45=[float(ci_[c]) for c in ("diff", "ci_lo", "ci_hi")], offroad_45=[float(off[c]) for c in ("diff", "ci_lo", "ci_hi")],
                                          guard={k: [float(v[c]) for c in ("diff", "ci_lo", "ci_hi")] for k, v in g.items()}, guard_down=down, channel_read=read, verdict=verdict)
    # ---- report
    L = ["# EYES-vs-SENSE tables (generated by scripts/es_read.py; do not edit)", "",
         f"Generated {time.strftime('%F %T %Z')}. Arms: (a) `{A['a']}`, (b) `{A['b']}`, (c) `{A['c']}`, (d) `{A['d']}`, seeds 0 / 1 (identity rule: {idt}). "
         "navtest: scores in points, failure rates in percentage points of the bucket's tokens; differences = arm minus control, seed mean per token "
         "(or one seed), by-log cluster bootstrap B 10 000 (vt_read.CB). Definitions: header of experiments/vis_train/scripts/vt_read.py.", "",
         f"Reads present: {len(have)} of {len(specs())} ({', '.join(sorted(set(specs_ for _, _, _, specs_ in specs()) - set(have))) or 'all'} missing).", "",
         "## Registered side-view verdicts", "",
         "| comparison | > 45 cut-inside (pp) | > 45 off-road (pp) | EPDMS < 5 | NC | EC | channel: arm - arm:mshuf | verdict |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for x, y in SIDE:
        v = st["verdicts"][f"{x}-{y}"]
        if "cut_in_45" not in v:
            L.append(f"| ({x}) - ({y}) | | | | | | | {v['verdict']} |")
            continue
        f3 = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"  # noqa: E731
        ch = st["channel"].get(x) or {}
        L.append(f"| ({x}) - ({y}) | {f3(v['cut_in_45'])} | {f3(v['offroad_45'])} | {f3(v['guard']['EPDMS < 5'])} | {f3(v['guard']['NC'])} | {f3(v['guard']['EC'])} | "
                 f"{ch.get('arm_minus_mshuf', float('nan')):+.2f} | **{v['verdict']}** |")
    L += ["", "## Test-time options (arm minus option, navtest EPDMS all tokens, seed mean; positive = the option costs points)", "",
          "| arm | :noside | :mshuf | :sideoff |", "|:--|--:|--:|--:|"]
    for k in ("b", "d"):
        rs = [cell(f"{k}:{o}", k, "all", "EPDMS") for o in OPTS]
        L.append(f"| ({k}) | " + " | ".join("n/a" if r is None else f"{-r['diff']:+.2f} [{-r['ci_hi']:+.2f}, {-r['ci_lo']:+.2f}]" for r in rs) + " |")
    for title, cs in (("Side views", SIDE), ("Headline rows (body lesson)", HEAD), ("Both against none", REF)):
        L += ["", f"## {title}", ""]
        for x, y in cs:
            L += [f"### ({x}) - ({y})", "", "| quantity | seed mean | seed 0 | seed 1 |", "|:--|:--|:--|:--|"]
            for b, k in (("all", "EPDMS"), ("<5", "EPDMS"), ("5-20", "EPDMS"), ("20-45", "EPDMS"), (">45", "EPDMS"), ("all", "NC"), ("all", "DAC"), ("all", "EP"),
                         ("all", "TTC"), ("all", "EC"), (">45", "cut_in"), (">45", "offroad"), (">45", "under"), (">20", "offroad"), ("all", "nc_fail")):
                L.append(f"| {k} {b} | {ci(cell(x, y, b, k))} | {ci(cell(x, y, b, k, 0))} | {ci(cell(x, y, b, k, 1))} |")
            for c in ("combined", "stage1", "stage2"):
                L.append(f"| navhard {c} | {ci(cell(x, y, 'all', c, 'mean', 'navhard'))} | {ci(cell(x, y, 'all', c, 0, 'navhard'))} | {ci(cell(x, y, 'all', c, 1, 'navhard'))} |")
            f_ = FL[(FL.arm == x) & (FL.control == y)] if len(FL) else FL
            L += ["", "Flips (tokens, seeds summed; fixed = control fails and arm passes): " +
                  "; ".join(f"{r.cls} {r.bucket} saved {r.fixed} / hurt {r.broken} ({r.ctrl_fail} -> {r.arm_fail})" for r in f_.itertuples()), ""]
            nav = R[(R.arm == x) & (R.control == y) & (R.seed == "mean") & (R.board == "navtest") & (R.metric.isin(V.SC))]
            up = nav[nav.ci_lo > 0]
            dn = nav[nav.ci_hi < 0]
            L += [f"No-regression table (decision 248, amendment 2; {len(nav)} navtest score cells, sub-score x bucket, seed mean): up {len(up)}, down {len(dn)}, "
                  f"did not move {len(nav) - len(up) - len(dn)}.", "",
                  "Down: " + ("; ".join(f"{r.metric} {r.bucket} {r.diff:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]" for r in dn.itertuples()) or "none") + ".", "",
                  "Up: " + ("; ".join(f"{r.metric} {r.bucket} {r.diff:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]" for r in up.itertuples()) or "none") + ".", ""]
            st.setdefault("noreg", {})[f"{x}-{y}"] = dict(up=len(up), down=len(dn), cells=len(nav), down_cells=[f"{r.metric} {r.bucket}" for r in dn.itertuples()])
    # ---- closed loop (a) / (c), 700 scenes
    try:
        L += closed_loop(A, idt, st)
    except Exception as e:                                           # noqa: BLE001
        L += ["", f"## nuPlan closed loop\n\nnot available: {e!r}"]
    od = OUT / "out"
    od.mkdir(parents=True, exist_ok=True)
    R.to_csv(od / "cells.csv", index=False)
    FL.to_csv(od / "flips.csv", index=False)
    (od / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    (od / "contrast_tables.md").write_text("\n".join(L) + "\n")
    run.summary.update(rows=len(R), have=len(have))


def closed_loop(A, idt, st):
    """(a) / (c) on the 700 scenes of c0b chunk0-2: phase 1's runs of the stored checkpoints, or ESa / ESc's own runs."""
    import c0b_report as CR
    import joint1_report as J
    RA = D / "runs/alpasim"
    sc7 = sorted(set().union(*(set((RA / "c0b/lists" / f"chunk{i}.txt").read_text().split()) for i in range(3))))
    _, fam_ok = J.checks()
    src = J.sources(fam_ok)
    es = json.loads((D / "runs/eyes_sense/cl/manifest.json").read_text()) if (D / "runs/eyes_sense/cl/manifest.json").exists() else {}
    runs = {}
    for k in ("a", "c"):
        for s in (0, 1):
            t = f"{A[k]}-s{s}"
            runs[(k, s)] = es.get(t, []) if A[k].startswith("ES") else src.get(t, [])
    if not all(runs.values()):
        return ["", "## nuPlan closed loop (700 scenes)", "", "not all runs present yet"]
    Ar = {key: J.Arm(*(lambda d: (d[0], d[3]))(CR.load_driver(v)), sc7) for key, v in runs.items()}
    logs = np.array([CR.log_of(s) for s in sc7])
    T = J.SR.turns()
    g45 = np.abs(np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in sc7])) > 45
    m = {k: np.mean([Ar[(k, s)].sc for s in (0, 1)], 0) for k in ("a", "c")}
    zc = {k: {c: np.mean([Ar[(k, s)].z((c,)) for s in (0, 1)], 0) for c in J.CLS} for k in ("a", "c")}
    L = ["", "## nuPlan closed loop, 700 scenes (c0b chunk0-2; (a) and (c) only; two-seed means)", "",
         "| arm | mean [CI logs] | zeros: collision / offroad / corridor | > 45 deg zeros (collision / offroad / corridor) |", "|:--|:--|:--|:--|"]
    for k in ("a", "c"):
        L.append(f"| ({k}) {A[k]} | {CR.fmt(CR.ci(m[k], logs))} | " + " / ".join(f"{zc[k][c].sum():g}" for c in J.CLS) + " | " +
                 " / ".join(f"{zc[k][c][g45].sum():g}" for c in J.CLS) + " |")
    d = CR.ci(m["c"] - m["a"], logs)
    co = J.sboot(zc["c"]["collision_at_fault"] + zc["c"]["offroad"] - zc["a"]["collision_at_fault"] - zc["a"]["offroad"])
    cr = J.sboot(zc["c"]["left_corridor_laterally"] - zc["a"]["left_corridor_laterally"])
    L += ["", f"(c) - (a): mean {CR.fmt(d, 4, True)}; at-fault collision + offroad zeros {J.fz(co)}; corridor zeros {J.fz(cr)} (scene bootstrap)."]
    st["closed_loop"] = dict(mean_diff=d, co=co, corridor=cr, zeros={k: {J.SHORT[c]: float(zc[k][c].sum()) for c in J.CLS} for k in ("a", "c")})
    return L


def cmd_box(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    V.OUT = OUT                                                      # the token caches of this phase (vt_read.tokf reads the module global)
    V.specs = specs
    with Run("eyes_sense", "reads", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        if not a.no_replay:
            V.replay(run)
        build(run)


def cmd_sync(a):
    ssh = ["ssh", "-o", "ControlPath=none", "autodl"]
    ld = f"{OUT}/pool/{time.strftime('%Y%m%d-%H%M%S')}"
    sub = (f"cd ~/data/jev-drive && .venv/bin/python -m jevdrive.cl submit --owner eyes_sense --name es-read --vram 0.5 --cpu {a.cpu} --ram 32 --log-dir {ld} -- "
           f"{D}/envs/navsim2/bin/python experiments/eyes_sense/scripts/es_read.py box" + (" --no-replay" if a.no_replay else ""))
    print(subprocess.run([*ssh, sub], check=True, capture_output=True, text=True).stdout.strip(), ld)
    subprocess.run([*ssh, f"until [ -f {ld}/DONE ] || [ -f {ld}/ERROR ]; do sleep 15; done; [ -f {ld}/DONE ] || {{ tail -30 {ld}/log.txt; exit 1; }}"], check=True)
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-q", "-o", "ControlPath=none", *(f"autodl:{OUT}/out/{f}" for f in ("contrast_tables.md", "cells.csv", "flips.csv", "stats.json")),
                    str(RES / "tables")], check=True)
    print(f"copied to {RES / 'tables'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("box")
    p.add_argument("--no-replay", action="store_true")
    p = sp.add_parser("sync")
    p.add_argument("--cpu", type=int, default=24)
    p.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()
    {"box": cmd_box, "sync": cmd_sync}[a.cmd](a)
