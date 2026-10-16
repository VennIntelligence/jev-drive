#!/usr/bin/env python3
"""RSG-ACC (experiments/route_follow/plans/2026-10-17-rsg-acc-prereg.md): pool checks, merge, draw weights and the registered gates. Box;
envs/op-train (ol_confirm runs in AlpaSim's venv as a subprocess). Chain: rf_rsgacc.sh.

  r1 --pool N --tag T --control C   PAI pool R1: the checkpoint's plans recomputed on the pool's inputs (rf_recov fwd, train-rec + dev-rec) against
        the rollout cache's `poses_model`, 8 poses (<= 4 s), tolerance max(0.03 m, 4 fp16 ulp); Amendment 1: median row max-diff <= 0.02 m and
        share over <= 0.10, wrong model (the other RB0 base) share over >= 0.9
        -> $O/gates/r1_<N>.json (exit 2 on fail)
  merge --tags T..   new pools whose gates passed (nuPlan: rows gates.json + poolgate; PAI: rows G-in >= 0.99 + R1) get the tab tag `dev`
        (rf_recov.is_dev(log): '1' / '0'), then rf_pait merge on1_L18RB0 (prefix s, first) + every new pool (prefix n<x> / p<x>, mode self)
        -> on1_RSGACC; the pool table (rows, scenes, unique scene-decisions, dev / train) -> results/rsgacc/pools.{md,json}
  weights --cell a|b|c|p0k13|p0k26   the cell's --on-weight on on1_RSGACC (a, c: dev rows 0; b: + old pool to 50 % of the drawn train rows;
        p0*: every new pool 0, so the draw equals the old pool's bit for bit)
  gate --stage pilot|full|held|pub|pai   the prereg's lines -> results/rsgacc/<stage>.{md,json}; exit 0 = they hold
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R_), str(HERE), str(R_ / "lib"), str(R_ / "experiments/onpol_nuplan/lib"), str(R_ / "experiments/onpol_nuplan/scripts"),
                str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/body1/lib"), str(R_ / "experiments/route_mem/lib"),
                str(R_ / "experiments/op_parity/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, RF = D / "runs/op_parity/cache", D / "runs/route_follow"
O = RF / "rsgacc"
RES = R_ / "experiments/route_follow/results/rsgacc"
PYA = D / "third_party/alpasim/.venv/bin/python"
POOL, BASE = "RSGACC", "RSG-L18-s0"
SHORT = {"RB0-F-s0": "a0", "RB0-F-s1": "a1", "RSG-L18-s0": "g0", "RSG-L18-s1": "g1", "RSGlr03n5000-L18-s0": "l0", "RSGk51w1-L18-s0": "k0"}
CELLS = dict(a=("p0k13", 13), b=("p0k13", 13), c=("p0k26", 26))
DEV_ADE, TURN = 0.538, -0.3
EXAM = R_ / "experiments/alpasim/results/pai_exam/hard_exam.txt"


def jload(p):
    return json.loads(Path(p).read_text())


def out(name, md, js):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.md").write_text("\n".join(md) + "\n")
    (RES / f"{name}.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md), flush=True)


def is_dev(log) -> bool:                                   # = rf_recov.is_dev (DEV_PCT 15)
    return zlib.crc32(str(log).encode()) % 100 < 15


# ---------------------------------------------------------------- pools
def cmd_r1(a):
    tb = np.load(CR / f"on1_{a.pool}" / "tab.npz")
    C = np.load(O / "cache" / a.pool / "tab.npz")
    key = {(s, int(k)): i for i, (s, k) in enumerate(zip(C["scenes"][C["sid"]].tolist(), C["k"].tolist()))}
    ref = C["poses_model"][np.array([key[(n.split(":")[2], int(n.split(":")[3]))] for n in tb["names"].tolist()])][:, :8, :2].astype(np.float64)
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(ref), 2.0 ** -14))) - 10)
    tol = np.maximum(4 * ulp, 0.03)
    res = {}
    for t in (a.tag, a.control):
        P = np.full(ref.shape, np.nan)
        for rs in ("train-rec", "dev-rec"):
            z = np.load(O / "fwd" / f"{a.pool}-{rs}" / f"plans_{t}.npz")
            P[z["rows"]] = z["poses"][:, :8, :2]
        d = np.abs(P - ref)
        res[t] = dict(share_over=float((d > tol).reshape(len(d), -1).any(1).mean()), median_m=float(np.median(d.reshape(len(d), -1).max(1))),
                      p99_m=float(np.percentile(d.reshape(len(d), -1).max(1), 99)), rows=int(len(d)))
    r = dict(pool=a.pool, tag=a.tag, control=a.control, model=res[a.tag], wrong_seed=res[a.control],
             pass_=bool(res[a.tag]["median_m"] <= 0.02 and res[a.tag]["share_over"] <= 0.10 and res[a.control]["share_over"] >= 0.9))   # Amendment 1
    (O / "gates").mkdir(parents=True, exist_ok=True)
    (O / "gates" / f"r1_{a.pool}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["pass_"] else 2)


def cmd_poolgate(a):
    """Amendment 2: poolgate.py's lines with G-route over the rows that have a served route0 (route0_d finite; NaN = nothing to compare,
    e.g. past the route's end, which ACC reaches more often); the NaN share is reported."""
    f = np.load(D / "runs/route_mem/feat" / f"on1_{a.pool}.npz")
    r = np.load(CR / f"on1_{a.pool}" / "route.npz")
    ok, d = f["ok"], f["route0_d"]
    fin = ok & np.isfinite(d)
    res = dict(pool=a.pool, rows=int(len(ok)), feat_ok=float(ok.mean()), route_ok=float(r["ok"].mean()), cov20=float(r["cov20"].mean()),
               route0_nan_share=float((ok & ~np.isfinite(d)).mean()), route0_share_005_finite=float((d[fin] <= 0.05).mean()))
    res["pass"] = bool(res["rows"] >= 10000 and res["route_ok"] >= 0.95 and res["route0_share_005_finite"] >= 0.99)
    (O / "gates").mkdir(parents=True, exist_ok=True)
    (O / "gates" / f"poolgate_{a.pool}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    raise SystemExit(0 if res["pass"] else 2)


def pool_ok(x):
    """-> (nuPlan ok, PAI ok, facts) for one checkpoint's two pools."""
    n, p = f"L18A{x}", f"PAIA{x}"
    f = {}
    try:
        g = jload(D / "runs/onpol_long/rows" / n / "gates.json")
        pg = jload(O / "gates" / f"poolgate_{n}.json")
        import ast
        r1, r2 = g["R1"], ast.literal_eval(g["R2"]) if isinstance(g["R2"], str) else g["R2"]
        ok1 = all(v >= 0.99 for v in r1["a_one_ulp"].values()) and all(v < 0.2 for v in r1["c_wrong_seed"].values())
        ok2 = r2["ego_max_m"] <= 0.3 and r2["act_share_within"] >= 0.9999
        f[n] = dict(R1=ok1, R2=ok2, K0_not_applied=g.get("pass"), poolgate=pg)
        okn = ok1 and ok2 and bool(pg.get("pass"))
    except FileNotFoundError as e:
        f[n], okn = dict(missing=str(e)), False
    try:
        r1 = jload(O / "gates" / f"r1_{p}.json")
        md = (O / "rows/pai" / f"report_{p}.log").read_text() if (O / "rows/pai" / f"report_{p}.log").exists() else ""
        b = jload(CR / f"on1_{p}" / "build.json")
        gin = gin_share(p)
        f[p] = dict(r1=r1, gin=gin, build=b)
        okp = bool(r1["pass_"]) and gin >= 0.99 and b["rows"] > 0
        f[p]["report_tail"] = md[-400:]
    except FileNotFoundError as e:
        f[p], okp = dict(missing=str(e)), False
    return okn, okp, f


def gin_share(p):
    eq = n = 0
    for z in (O / "rows/pai" / p).glob("clipgt-*.npz"):
        m = json.loads(str(np.load(z)["meta"]))
        eq, n = eq + m["gin_dump_eq"], n + m["gin_dump_n"]
    return eq / max(n, 1)


def cmd_merge(a):
    parts, facts, tab, uniq = [("on1_L18RB0", "s")], {}, [], {}
    for T in a.tags:
        x = SHORT[T]
        okn, okp, f = pool_ok(x)
        facts[T] = f
        for ok, nm, pre in ((okn, f"L18A{x}", f"n{x}"), (okp, f"PAIA{x}", f"p{x}")):
            if not ok:
                print(f"{nm}: gates do not hold, not merged", flush=True)
                continue
            tp = CR / f"on1_{nm}" / "tab.npz"
            t = dict(np.load(tp))
            t["dev"] = np.array(["1" if is_dev(g) else "0" for g in t["log"].tolist()]).astype("U2")
            np.savez(tp, **t)
            parts.append((f"on1_{nm}", pre))
    spec = [f"{p}:{pre}:self" for p, pre in parts]
    subprocess.run([sys.executable, str(HERE / "rf_pait.py"), "merge", "--out", POOL, "--pools", *spec], check=True)
    for p, pre in parts:
        t = np.load(CR / p / "tab.npz")
        lg = t["log"].astype(str)
        dv = np.array([x == "1" for x in t["dev"].tolist()]) if "dev" in t.files else np.zeros(len(lg), bool)
        src = next((T for T in a.tags if p.endswith(SHORT[T])), "RB0-F-s0 (RSG's pool, default serve)")
        tab.append(dict(pool=p, prefix=pre, policy=src, track="PAI" if "PAI" in p else "nuPlan", rows=int(len(lg)), scenes=int(len(set(lg))),
                        train_rows=int((~dv).sum()), dev_rows=int(dv.sum())))
        if pre != "s":
            uniq.setdefault(tab[-1]["track"], set()).update(t["names"].tolist())   # the same (scene, decision) has the same name under every policy
    md = ["# RSG-ACC pools (on1_RSGACC)", "", "| pool | prefix | rolled by | track | rows | train rows | dev rows | scenes |", "|:--|:--|:--|:--|--:|--:|--:|--:|"]
    md += [f"| {r['pool']} | {r['prefix']} | {r['policy']} | {r['track']} | {r['rows']} | {r['train_rows']} | {r['dev_rows']} | {r['scenes']} |" for r in tab]
    new = [r for r in tab if r["prefix"] != "s"]
    md += ["", f"new rows {sum(r['rows'] for r in new)} (train {sum(r['train_rows'] for r in new)}, dev {sum(r['dev_rows'] for r in new)}; "
           f"nuPlan {sum(r['rows'] for r in new if r['track'] == 'nuPlan')}, PAI {sum(r['rows'] for r in new if r['track'] == 'PAI')}); old pool {tab[0]['rows']}",
           "unique (scene, decision) states among the new rows: " + ", ".join(f"{k} {len(v)}" for k, v in uniq.items())]
    out("pools", md, dict(parts=tab, facts=facts, unique={k: len(v) for k, v in uniq.items()}))


def cmd_weights(a):
    t = np.load(CR / f"on1_{POOL}" / "tab.npz")
    pre = t["pool"].astype(str)
    new = sorted(set(pre) - {"s"})
    if a.cell.startswith("p0"):
        print(",".join(f"pool={p}:0" for p in new))
        return
    w = "dev=1:0"
    if a.cell == "b":
        dv = t["dev"].astype(str) == "1"
        n_new, n_old = int(((pre != "s") & ~dv).sum()), int((pre == "s").sum())
        w += f",pool=s:{n_new / n_old:.6f}"
    print(w)


# ---------------------------------------------------------------- gates
def contact(tag, ctl):
    """Served-plan contact (agent or road, 0.1 - 4 s) on the dev rows of the new pools: tag vs control, log-clustered; nuPlan decides, PAI reported."""
    import rf_dagger as DG
    from jevdrive import stats
    z = {t: np.load(O / "fwd" / f"{POOL}-dev-rec" / f"plans_{t}.npz") for t in (tag, ctl)}
    rr = z[tag]["rows"]
    assert np.array_equal(rr, z[ctl]["rows"])
    T = np.load(CR / f"on1_{POOL}" / "tab.npz")
    n = len(T["names"])
    full = lambda P: (lambda X: (X.__setitem__(rr, P), X)[1])(np.full((n, 8, 3), np.nan, np.float32))  # noqa: E731
    o, _, _ = DG.contacts(f"on1_{POOL}", {"t": full(z[tag]["poses"]), "c": full(z[ctl]["poses"])}, rr)
    lg, pre = T["log"][rr].astype(str), T["pool"][rr].astype(str)
    res = {}
    for trk, sel in (("nuplan", np.char.startswith(pre, "n")), ("pai", np.char.startswith(pre, "p"))):
        keep = sel & ~(o["t"]["t0"] | o["c"]["t0"])
        if not keep.any():
            res[trk] = None
            continue
        ct, cc = (o["t"]["agent"] | o["t"]["road"])[keep], (o["c"]["agent"] | o["c"]["road"])[keep]
        r = stats.paired(ct.astype(float), cc.astype(float), groups=lg[keep])
        res[trk] = dict(rows=int(keep.sum()), logs=int(len(set(lg[keep]))), rate=float(ct.mean()), rate_ctl=float(cc.mean()),
                        drop=float(1 - ct.mean() / max(cc.mean(), 1e-9)), diff=[r["mean"], r["lo"], r["hi"]],
                        agent=[float(o["t"]["agent"][keep].mean()), float(o["c"]["agent"][keep].mean())],
                        road=[float(o["t"]["road"][keep].mean()), float(o["c"]["road"][keep].mean())])
    return res


def offline(tag, ctl, turn=True):
    import rf_recov as RC
    tg, _ = RC.trainer_gates(tag)
    rl = tg["rely"] or {}
    g = dict(dev_ade=tg["dev_ade"], rely=dict(read=rl.get("read"), d_junction=rl.get("d_junction"), diff_lo=rl.get("diff_lo")))
    L = dict(G1=g["dev_ade"] is not None and g["dev_ade"] <= DEV_ADE, G2=bool(rl.get("read")))
    if turn:
        t, _ = RC.gate_turn(tag, BASE)
        g["turn"] = t["equal_arc"]
        L["G3"] = t["equal_arc"]["dh_mean"] >= TURN
    g["contact"] = contact(tag, ctl)
    c = g["contact"]["nuplan"]
    L["G4"] = c is not None and c["diff"][2] < 0
    return g, L


def f3(v, p=3):
    return "-" if v is None else f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"


def gate_offline(stage):
    pairs = [(f"RSGACCp{c}-L18-s0", f"RSGACCp{CELLS[c][0]}-L18-s0", c) for c in "abc"] if stage == "pilot" else [("RSGACC-L18-s0", "RSGACC0-L18-s0", "full")]
    turn = stage == "pilot"
    rows, md = {}, [f"# RSG-ACC {stage} offline gates", "", "Lines: G1 dev ADE <= 0.538; G2 rely read; G3 equal-arc turn vs RSG-L18-s0 >= -0.3 deg (pilot); "
                    "G4 contact rate on the new nuPlan dev rows vs the same-k control, CI hi < 0 (PAI dev rows reported).", "",
                    "| tag | control | dev ADE | rely (d_junction) | turn equal-arc deg | contact nuPlan dev (tag / ctl, d [CI]) | contact PAI dev | lines |",
                    "|:--|:--|--:|:--|:--|:--|:--|:--|"]
    import rf_recov as RC
    cg, _ = RC.trainer_gates(pairs[0][1])
    for t, c, cell in pairs:
        g, L = offline(t, c, turn)
        rows[cell] = dict(tag=t, control=c, gates=g, lines=L, pass_=all(L.values()))
        cn, cp = g["contact"]["nuplan"], g["contact"]["pai"]
        md.append(f"| {t} | {c} | {g['dev_ade']:.4f} | {g['rely']['read']} ({(g['rely']['d_junction'] or 0):+.3f}) | "
                  f"{g['turn']['dh_mean']:+.2f} [{g['turn']['ci'][0]:+.2f}, {g['turn']['ci'][1]:+.2f}] | " if turn else
                  f"| {t} | {c} | {g['dev_ade']:.4f} | {g['rely']['read']} ({(g['rely']['d_junction'] or 0):+.3f}) | - | ")
        md[-1] += (f"{cn['rate']:.4f} / {cn['rate_ctl']:.4f}, {f3(cn['diff'], 4)} (n {cn['rows']}, {cn['logs']} logs) | " if cn else "- | ")
        md[-1] += (f"{cp['rate']:.4f} / {cp['rate_ctl']:.4f}, {f3(cp['diff'], 4)} | " if cp else "- | ")
        md[-1] += " ".join(f"{k} {'ok' if v else 'NO'}" for k, v in L.items()) + " |"
    md.append(f"\ncontrol dev ADE {cg['dev_ade']}")
    if stage == "pilot":
        ok = [c for c in "abc" if rows[c]["pass_"]]
        best = None
        if ok:
            top = max(rows[c]["gates"]["contact"]["nuplan"]["drop"] for c in ok)
            best = next(c for c in ok if top - rows[c]["gates"]["contact"]["nuplan"]["drop"] <= 0.02)
            (O / "cell.txt").write_text(best + "\n")
        md += ["", f"Rule: passing cells {ok or 'none'}; selected **{best or 'none: stop, no full run'}**"]
        out("pilot", md, dict(rows=rows, passing=ok, cell=best))
        return 0 if best else 3
    full = rows["full"]
    md += ["", f"Full gates G1 / G2 decide (G4 recorded): **{'PASS' if full['lines']['G1'] and full['lines']['G2'] else 'FAIL'}**"]
    out("full", md, dict(rows=rows))
    return 0 if full["lines"]["G1"] and full["lines"]["G2"] else 3


def runs_of(t, root):
    return ",".join(sorted(str(p) for p in Path(root).glob(f"{t}/c*/run")))


def gate_held():
    M = D / "runs/route_mem/read"
    R = O / "report/held18"
    R.mkdir(parents=True, exist_ok=True)
    tags = ["RSGACC-L18-s0", "RSGACC0-L18-s0", BASE]
    runs = [f"{t}@held18={runs_of(t, M)}" for t in tags]
    subprocess.run([str(PYA), str(R_ / "experiments/onpol_long/scripts/ol_confirm.py"), "--runs", *runs, "--base", BASE,
                    "--pairs", "RSGACC-L18-s0:RSGACC0-L18-s0", "--out", str(R)], check=True)
    J = json.load(open(R / "confirm.json"))["held18"]
    a, c = J["RSGACC-L18-s0"], J["RSGACC0-L18-s0"]
    L = {"zeros <= RSG-L18-s0 + 3": a["zero"][0] <= 3, "GT point <= +0.03 m": a["gt"][0] <= 0.03}
    md = ["# RSG-ACC held 452 @18 (nuPlan default serve), paired vs stored RSG-L18-s0", "", "| tag - RSG-L18-s0 | zeros | at-fault | score | GT (m) |", "|:--|:--|:--|:--|:--|"]
    for k, v in (("RSGACC-L18-s0 (arm)", a), ("RSGACC0-L18-s0 (control)", c), ("arm - control", J.get("RSGACC-L18-s0 - RSGACC0-L18-s0"))):
        if v:
            md.append(f"| {k} | {f3(v['zero'], 1)} | {f3(v['z_atf'], 1)} | {f3(v['score'], 4)} | {f3(v['gt'], 3)} |")
    md += ["", "| line | holds |", "|:--|:--|"] + [f"| {k} | {'yes' if v else 'NO'} |" for k, v in L.items()]
    md += ["", f"verdict: {'PASS' if all(L.values()) else 'FAIL'}; full table {R}/confirm.md"]
    out("held", md, dict(lines=L, arm=a, control=c, pair=J.get("RSGACC-L18-s0 - RSGACC0-L18-s0")))
    return 0 if all(L.values()) else 3


def gate_pub():
    G = RF / "guard/read10"
    A = D / "runs/alpasim"
    R = O / "report/all10"
    R.mkdir(parents=True, exist_ok=True)
    a_, b_ = jload(A / "ot3/a/manifest.json"), jload(A / "cf1/a/manifest.json")
    runs = ["YR10m10-F-s0@all10=" + ",".join(a_["YR10m10-F-s0"] + a_["YR10m10-F-s0@new"] + b_["YR10m10-F-s0@fresh"])]
    for t in ("RSGACC-L18-s0", "RSGACC0-L18-s0", BASE):
        runs.append(f"{t}@all10=" + ",".join(sorted(str(p) for p in (G / t).glob("b*"))))
    subprocess.run([str(PYA), str(R_ / "experiments/onpol_long/scripts/ol_confirm.py"), "--runs", *runs, "--base", "YR10m10-F-s0",
                    "--pairs", f"RSGACC-L18-s0:{BASE}", f"RSGACC0-L18-s0:{BASE}", "RSGACC-L18-s0:RSGACC0-L18-s0", "--out", str(R)], check=True)
    J = json.load(open(R / "confirm.json"))["all10"]
    a = J[f"RSGACC-L18-s0 - {BASE}"]
    L = {"score CI lo > -0.01": a["score"][1] > -0.01, "zeros point <= +5": a["zero"][0] <= 5}
    md = ["# RSG-ACC public 1491 @10 guard, paired vs stored RSG-L18-s0 guard read", "", "| pair | zeros | at-fault | score | GT (m) |", "|:--|:--|:--|:--|:--|"]
    for k in (f"RSGACC-L18-s0 - {BASE}", f"RSGACC0-L18-s0 - {BASE}", "RSGACC-L18-s0 - RSGACC0-L18-s0"):
        v = J[k]
        md.append(f"| {k} | {f3(v['zero'], 1)} | {f3(v['z_atf'], 1)} | {f3(v['score'], 4)} | {f3(v['gt'], 3)} |")
    md += ["", "| line | holds |", "|:--|:--|"] + [f"| {k} | {'yes' if v else 'NO'} |" for k, v in L.items()]
    md += ["", f"verdict: {'PASS' if all(L.values()) else 'FAIL'}; full table {R}/confirm.md"]
    out("pub", md, dict(lines=L, pairs={k: J[k] for k in J if " - " in k}))
    return 0 if all(L.values()) else 3


def boot(x, B=10000):
    bs = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def gate_pai():
    import rf_pai281 as RP
    ref = RP.read(str(RF / "acc_gap/read/s2"))
    Dd = {t: RP.read(str(O / "pai" / t)) for t in ("RSGACC-L18-s0", "RSGACC0-L18-s0")}
    A, C = Dd["RSGACC-L18-s0"], Dd["RSGACC0-L18-s0"]
    sc = sorted(set(A) & set(ref))
    exam = set(EXAM.read_text().split())
    ex = [s for s in sc if s in exam]
    z = lambda X: sum(X[s]["score"] == 0 for s in sc)  # noqa: E731
    sp = lambda X: sum(X[s]["spin"] for s in sc)  # noqa: E731
    d = boot(np.array([A[s]["score"] - ref[s]["score"] for s in sc]))
    xe = float(np.mean([A[s]["score"] - ref[s]["score"] for s in ex]))
    L = {"mean CI lo > -0.01 and point > 0": d[1] > -0.01 and d[0] > 0, "exam point >= -0.01": xe >= -0.01,
         "zeros <= ref + 2": z(A) <= z(ref) + 2, "spins <= ref + 1": sp(A) <= sp(ref) + 1}
    md = [f"# RSG-ACC PAI 281 under ACC+GAP (g10), paired vs RSG-L18-s0 (d330 stage 2)", "", f"{len(sc)} scenes scored by both (exam {len(ex)})", "",
          "| label | mean | zeros | at-fault collision | offroad | corridor | rear-ended | spins | vs ref (95 % CI) | exam vs ref |", "|:--|--:|--:|--:|--:|--:|--:|--:|:--|--:|"]
    for k, X in (("RSGACC-L18-s0 (arm)", A), ("RSGACC0-L18-s0 (control)", C), ("RSG-L18-s0 (ref)", ref)):
        s2 = [s for s in sc if s in X]
        dd = boot(np.array([X[s]["score"] - ref[s]["score"] for s in s2]))
        de = np.mean([X[s]["score"] - ref[s]["score"] for s in ex if s in X])
        md.append(f"| {k} | {np.mean([X[s]['score'] for s in s2]):.4f} | {sum(X[s]['score'] == 0 for s in s2)} | {sum(X[s]['kind'] == 'collision' for s in s2)} | "
                  f"{sum(X[s]['kind'] == 'offroad' for s in s2)} | {sum(X[s]['kind'] == 'corridor' for s in s2)} | {sum(X[s]['rear'] for s in s2)} | "
                  f"{sum(X[s]['spin'] for s in s2)} | {dd[0]:+.4f} [{dd[1]:+.4f}, {dd[2]:+.4f}] | {de:+.4f} |")
    s3 = [s for s in sc if s in C]
    dc = boot(np.array([A[s]["score"] - C[s]["score"] for s in s3]))
    md += ["", f"arm - control: {dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}] ({len(s3)} scenes)", "", "| line | holds |", "|:--|:--|"]
    md += [f"| {k} | {'yes' if v else 'NO'} |" for k, v in L.items()]
    for nm, X, Y in (("arm vs ref", A, ref), ("arm vs control", A, C)):
        ss = [s for s in sc if s in Y]
        new = [s for s in ss if X[s]["score"] == 0 < Y[s]["score"]]
        fix = [s for s in ss if Y[s]["score"] == 0 < X[s]["score"]]
        by = lambda S, Z: ", ".join(f"{k} {sum(Z[s]['kind'] == k for s in S)}" for k in ("collision", "offroad", "corridor", "other"))  # noqa: E731
        md += ["", f"{nm}: new zeros {len(new)} ({by(new, X)}): " + ", ".join(f"{s[7:15]} {X[s]['kind']}{' exam' if s in exam else ''}" for s in new),
               f"{nm}: fixed zeros {len(fix)} ({by(fix, Y)}): " + ", ".join(f"{s[7:15]} {Y[s]['kind']}{' exam' if s in exam else ''}" for s in fix)]
    md += ["", f"verdict: {'PASS' if all(L.values()) else 'FAIL'}"]
    out("pai", md, dict(n=len(sc), diff=d, exam=xe, lines=L, counts={k: dict(zeros=z(X), spins=sp(X)) for k, X in (("arm", A), ("control", C), ("ref", ref))},
                        arm_minus_control=dc))
    return 0 if all(L.values()) else 3


def cmd_gate(a):
    from jevdrive.run import Run
    with Run("route_follow", f"rsgacc-gate-{a.stage}", config=vars(a)) as run:
        rc = gate_offline(a.stage) if a.stage in ("pilot", "full") else dict(held=gate_held, pub=gate_pub, pai=gate_pai)[a.stage]()
        run.summary["rc"] = rc
    raise SystemExit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("r1")
    p.add_argument("--pool", required=True), p.add_argument("--tag", required=True), p.add_argument("--control", required=True)
    p = sp.add_parser("poolgate")
    p.add_argument("--pool", required=True)
    p = sp.add_parser("merge")
    p.add_argument("--tags", nargs="+", required=True)
    p = sp.add_parser("weights")
    p.add_argument("--cell", required=True, choices=("a", "b", "c", "p0k13", "p0k26"))
    p = sp.add_parser("gate")
    p.add_argument("--stage", required=True, choices=("pilot", "full", "held", "pub", "pai"))
    a = ap.parse_args()
    dict(r1=cmd_r1, poolgate=cmd_poolgate, merge=cmd_merge, weights=cmd_weights, gate=cmd_gate)[a.cmd](a)
