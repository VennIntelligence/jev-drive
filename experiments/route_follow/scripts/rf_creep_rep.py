#!/usr/bin/env python3
"""CREEP-REP (plans/2026-10-15-creep-replication-prereg.md): confirmatory replication of decision 308's plan creep on held 460 checkpoints
d308 did not use. CPU only; reads RECEDING-style held extracts (rf_creep_rep.sh), never writes next to them. The measure is rf_switching's
`turn_switching` / `creep` unchanged; this script only flips the sign (+ = outward), adds the 0.5 s / 1-2 s heading split (R4) and pools.

  check (Mac, .venv)  rf_creep_rep.py check --held tmp/receding/held.pkl --tax NAME=<csv> ...    G-code: reproduce switching.md held_all6
  read  (Mac, .venv)  rf_creep_rep.py read --held <pkl> [<pkl> ...] --tax NAME=<csv> ... --out DIR --fig PNG
        -> DIR/{turns.csv, decisions.csv.gz, creep_rep.json, creep_rep.md}; the figure.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import rf_switching as SW  # noqa: E402

N_BOOT = SW.N_BOOT
D308_SETS = {"RSG-L18-s0", "RSG-L18-s1", "RV2a-L18-s0", "RV2a-L18-s1", "RV2b-L18-s0", "RV2b-L18-s1"}
G_CODE = dict(jpm=(0.003, -0.006), x2=(-1.6, 0.4))          # switching.md held_all6, printed digits
FAMILIES = ("lr", "RV2c", "APP0C", "APP2W05", "FMH", "RGH")
CLASSES = ("late_wide", "straight", "over_turn", "wrong_exit", "other", "no_zero")
OUTCOMES = ("event", "off", "short")
REFUTE_C, REFUTE_X, MIN_GROUP = 0.010, -0.5, 10


def family(tag):
    return "lr" if tag.startswith("RSGlr") else next((f for f in FAMILIES if tag.startswith(f)), "other")


def med(v):
    return SW.med(v)


# ---------------------------------------------------------------- per turn
def turn(rec, tax):
    tr, rows = SW.turn_switching(rec, tax)
    if tr is None:
        return None, []
    C, Pl = rec["ctrl"], rec["plans"]
    sg = 1.0 if rec["row"]["side"] == "L" else -1.0
    yaw_c = np.unwrap(C["yaw"])
    t_end = min(rec["tz"], C["t"][-1])
    for o in rows:                                     # R4: plan k alone drives the car over [t0, t0 + 0.5 s]
        psi = sg * np.degrees(np.unwrap(np.r_[0.0, Pl["poses"][o["i"]][:, 2]]))
        te = o["t0"] + 500_000
        o["hp05"] = float(psi[1]) if te < t_end else np.nan
        o["hx05"] = float(sg * np.degrees(np.interp(te, C["t"], yaw_c) - o["yaw_ego"])) if te < t_end else np.nan
        o["x05"], o["x1"], o["x2"] = o["hx05"] - o["hp05"], o["hx1"] - o["hp1"], o["hx2"] - o["hp2"]
        o["hp12"], o["hx12"] = o["hp2"] - o["hp1"], o["hx2"] - o["hx1"]
        o["x12"] = o["hx12"] - o["hp12"]
        o["jh_pm"] = o["j_h"] / o["ds"] if o.get("pair") and o["ds"] > 0.05 else np.nan
    g = lambda k: med([o[k] for o in rows])  # noqa: E731
    tr.update(c=-tr["jpm"], C=-tr["jsum"], c_h=-g("jh_pm"), x05=g("x05"), x12=g("x12"), hp05_med=g("hp05"), hx05_med=g("hx05"),
              hp12_med=g("hp12"), hx12_med=g("hx12"), outcome=rec["row"]["outcome"], tag=rec["set"], family=family(rec["set"]))
    for o in rows:
        o.update(key=rec["set"], scene=rec["scene"])
    return tr, rows


def load(pkls, taxes):
    import pandas as pd
    TR, DR = [], []
    for f in pkls:
        for r in pickle.load(open(f, "rb")):
            tr, rows = turn(r, taxes.get(r["set"], taxes.get("held_" + r["set"])))
            if tr is not None:
                TR.append(tr)
                DR += rows
    T = pd.DataFrame(TR)
    T["failed"] = ~T.completed.astype(bool)
    return T, pd.DataFrame(DR)


def read_taxes(specs):
    import pandas as pd
    out = {}
    for spec in specs or []:
        n, f = spec.split("=", 1)
        x = pd.read_csv(f)
        out[n] = dict(zip(x.scene, x.primary))
    return out


# ---------------------------------------------------------------- statistics
def med_boot(v, cl, n_boot=N_BOOT, seed=0):
    """Median of v with a cluster-bootstrap percentile CI."""
    v, cl = np.asarray(v, float), np.asarray(cl)
    m = np.isfinite(v)
    v, cl = v[m], cl[m]
    if len(v) < 3:
        return dict(n=int(len(v)), m=np.nan, lo=np.nan, hi=np.nan)
    u = np.unique(cl)
    idx = {k: np.flatnonzero(cl == k) for k in u}
    rng = np.random.default_rng(seed)
    out = [np.median(v[np.concatenate([idx[k] for k in rng.choice(u, len(u))])]) for _ in range(n_boot)]
    return dict(n=int(len(v)), m=float(np.median(v)), lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)))


def ratio_ci(v, failed, cl, n_boot=N_BOOT, seed=0):
    """Descriptive: median(failed) / median(completed), cluster bootstrap (replicates with a completed median <= 0 are counted, not kept)."""
    v, failed, cl = np.asarray(v, float), np.asarray(failed, bool), np.asarray(cl)
    m = np.isfinite(v)
    v, failed, cl = v[m], failed[m], cl[m]
    u = np.unique(cl)
    idx = {k: np.flatnonzero(cl == k) for k in u}
    rng = np.random.default_rng(seed)
    out, nonpos = [], 0
    for _ in range(n_boot):
        r = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        a, b = v[r][failed[r]], v[r][~failed[r]]
        if not len(a) or not len(b):
            continue
        mb = np.median(b)
        if mb <= 0:
            nonpos += 1
            continue
        out.append(np.median(a) / mb)
    q = np.percentile(out, [2.5, 97.5]) if out else [np.nan, np.nan]
    return dict(lo=float(q[0]), hi=float(q[1]), nonpos=nonpos)


def r1_verdict(b, n_f, n_c):
    if n_f < MIN_GROUP or n_c < MIN_GROUP:
        return "inconclusive (group < 10)"
    ratio_ok = b["f"] > 0 and (b["c"] <= 0 or b["f"] / b["c"] >= 2)
    if b["lo"] > 0 and ratio_ok:
        return "supported"
    if b["d"] <= 0 or b["hi"] < REFUTE_C:
        return "refuted"
    if b["lo"] > 0:
        return "inconclusive (difference holds, ratio < 2)"
    return "inconclusive"


def r2_verdict(b, n_f, n_c):
    if n_f < MIN_GROUP or n_c < MIN_GROUP:
        return "inconclusive (group < 10)"
    if b["hi"] < 0:
        return "supported"
    if b["d"] >= 0 or b["lo"] > REFUTE_X:
        return "refuted"
    return "inconclusive"


def block(x, cl_col="cluster"):
    cl = x[cl_col].to_numpy()
    out = {k: SW.diff_boot(x[k], x.failed, cl) for k in ("c", "C", "x2", "x1", "x05", "x12")}
    out["n_f"], out["n_c"] = int(x.failed.sum()), int((~x.failed).sum())
    out["n_c_def"] = {g: int(x[x.failed == (g == "f")].c.notna().sum()) for g in ("f", "c")}
    out["n_x_def"] = {g: int(x[x.failed == (g == "f")].x2.notna().sum()) for g in ("f", "c")}
    return out


def check(a):
    T, _ = load([a.held], read_taxes(a.tax))
    T = T[T.src == "held"]
    got = dict(jpm=(T[T.failed].jpm.median(), T[~T.failed].jpm.median()), x2=(T[T.failed].x2.median(), T[~T.failed].x2.median()))
    ok = all(round(got["jpm"][i], 3) == G_CODE["jpm"][i] and round(got["x2"][i], 1) == G_CODE["x2"][i] for i in (0, 1))
    print(f"G-code held_all6 ({len(T)} turns): j per m f / c {got['jpm'][0]:+.3f} / {got['jpm'][1]:+.3f}, x2 {got['x2'][0]:+.1f} / {got['x2'][1]:+.1f}"
          f" vs switching.md {G_CODE}: {'PASS' if ok else 'FAIL'}")
    return ok, got, len(T)


def read(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok_code, got_code, n_code = check(argparse.Namespace(held=a.code_held, tax=a.code_tax)) if a.code_held else (None, None, 0)
    T, D = load(a.held, read_taxes(a.tax))
    T = T[T.src == "held"]
    leak = sorted(set(T.tag) & D308_SETS)
    assert not leak, f"d308 sets in the replication data: {leak}"
    T.to_csv(out / "turns.csv", index=False)
    D.to_csv(out / "decisions.csv.gz", index=False, float_format="%.4g")
    res = dict(n=dict(turns=int(len(T)), tags=int(T.tag.nunique()), failed=int(T.failed.sum()), completed=int((~T.failed).sum()),
                      clusters=int(T.cluster.nunique()), scenes=int(T.scene.nunique()), per_tag={k: int(v) for k, v in T.tag.value_counts().items()}))
    fr = T.groupby("tag").agg(xy=("frame_xy", "median"), yaw=("frame_yaw", "median"))
    hb = med_boot(T.c_h, T.cluster)
    res["gates"] = dict(code=dict(passed=ok_code, got=got_code, n=n_code), frame=dict(passed=bool(((fr.xy <= 0.3) & (fr.yaw <= 1.0)).all()),
                        max_xy=float(fr.xy.max()), max_yaw=float(fr.yaw.max())),
                        human=dict(c_h=hb, abs_med=float(abs(hb["m"])), passed=bool(abs(hb["m"]) <= REFUTE_C),
                                   by_group={g: float(T[T.failed == (g == "failed")].c_h.median()) for g in ("failed", "completed")}),
                        n=dict(passed=bool(len(T) >= 380), turns=int(len(T))))
    P = block(T)
    P["ratio"] = P["c"]["f"] / P["c"]["c"] if P["c"]["c"] > 0 else np.inf
    P["ratio_ci"] = ratio_ci(T.c, T.failed, T.cluster)
    P["R1"] = r1_verdict(P["c"], *P["n_c_def"].values()) if res["gates"]["human"]["passed"] else "inconclusive (G-human failed)"
    P["R2"] = r2_verdict(P["x2"], *P["n_x_def"].values())
    if not (res["gates"]["frame"]["passed"] and res["gates"]["n"]["passed"] and ok_code is not False):
        P["R1"] = P["R2"] = "not read (gate failed)"
    res["pooled"] = P
    # R3: classes / outcomes vs pooled completed; families
    comp = T[~T.failed]
    res["classes"] = {}
    for col, vals in (("cls", CLASSES), ("outcome", OUTCOMES)):
        for v in vals:
            y = pd.concat([comp, T[T.failed & (T[col] == v)]])
            if (y.failed).sum():
                res["classes"][f"{col}:{v}"] = block(y)
    res["families"] = {}
    for fam in FAMILIES:
        x = T[T.family == fam]
        z = dict(n_f=int(x.failed.sum()), n_c=int((~x.failed).sum()), tags=sorted(x.tag.unique()))
        if z["n_f"] >= 3 and z["n_c"] >= 3:
            z["own"] = block(x)
        y = pd.concat([T[~T.failed & (T.family != fam)], x[x.failed]])
        z["vs_pooled_completed"] = block(y)
        z["failed_med"] = {k: float(x[x.failed][k].median()) for k in ("c", "C", "x2", "x1", "x05", "x12")}
        z["completed_med"] = {k: float(x[~x.failed][k].median()) if z["n_c"] else np.nan for k in ("c", "C", "x2")}
        z["cls"] = {k: int(v) for k, v in x[x.failed].cls.value_counts().items()}
        res["families"][fam] = z
    # sensitivity
    sens = {"zero only (event / off)": block(pd.concat([comp, T[T.failed & T.outcome.isin(["event", "off"])]])),
            "scene clusters": block(T, "scene")}
    for fam in FAMILIES:
        sens[f"without {fam}"] = block(T[T.family != fam])
    res["sensitivity"] = sens
    res["per_tag"] = {}
    for tag, x in T.groupby("tag"):
        res["per_tag"][tag] = dict(n_f=int(x.failed.sum()), n_c=int((~x.failed).sum()), c=SW.diff_boot(x.c, x.failed, x.cluster.to_numpy()),
                                   x2=SW.diff_boot(x.x2, x.failed, x.cluster.to_numpy()))
    # R4: group medians with CIs
    res["R4"] = {}
    groups = [("completed", ~T.failed), ("failed", T.failed)] + [(f"failed {c}", T.failed & (T.cls == c)) for c in CLASSES]
    for g, m in groups:
        x = T[m]
        if len(x) < 3:
            continue
        res["R4"][g] = {k: med_boot(x[k], x.cluster) for k in ("x05", "x1", "x12", "x2", "hp05_med", "hx05_med", "hp1_med", "hx1_med", "hp12_med",
                                                               "hx12_med", "hp2_med", "hx2_med")}
    f4 = res["R4"]["failed"]
    if f4["x1"]["hi"] < 0 and f4["x1"]["m"] <= 0.5 * f4["x2"]["m"]:
        rd = "near segment under-tracked"
    elif f4["x1"]["lo"] <= 0 <= f4["x1"]["hi"] and f4["x12"]["hi"] < 0:
        rd = "appears in 1-2 s"
    else:
        rd = "both"
    res["R4_reading"] = rd
    res["A1"] = amend1(T, D)
    (out / "creep_rep.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    write_md(out, res)
    if a.fig:
        fig(T, res, a.fig)


def amend1(T, D):
    """Amendment 1 (post hoc, descriptive): (a) without FMH; (b) x12 = replan + track via plan k+2 (served 1.0 s after plan k)."""
    import pandas as pd
    nf = T[T.family != "FMH"]
    comp = nf[~nf.failed]
    a = {"zero only (event / off)": nf.failed & nf.outcome.isin(["event", "off"]), "late_wide": nf.failed & (nf.cls == "late_wide"),
         "no_zero": nf.failed & (nf.cls == "no_zero"), "over_turn": nf.failed & (nf.cls == "over_turn"), "all failed": nf.failed}
    A = {k: block(pd.concat([comp, nf[m]])) for k, m in a.items()}
    D = D.sort_values(["key", "scene", "t0"])
    parts = []
    for (k, s), d in D.groupby(["key", "scene"]):
        t0 = d.t0.to_numpy(np.int64)
        j = np.searchsorted(t0, t0 + 1_000_000)
        j = np.clip(j, 0, len(t0) - 1)
        ok = np.abs(t0[j] - (t0 + 1_000_000)) <= SW.TOL_US
        hp1n = np.where(ok, d.hp1.to_numpy(float)[j], np.nan)
        parts.append(pd.DataFrame(dict(tag=k, scene=s, replan=hp1n - d.hp12.to_numpy(float), track=d.hx12.to_numpy(float) - hp1n, x12=d.x12.to_numpy(float),
                                       hp12=d.hp12.to_numpy(float), hp1n=hp1n, hx12=d.hx12.to_numpy(float))))
    Q = pd.concat(parts)
    Q = Q[np.isfinite(Q.replan) & np.isfinite(Q.track)]
    per = Q.groupby(["tag", "scene"]).agg(n=("x12", "size"), **{c: (c, "median") for c in ("replan", "track", "x12", "hp12", "hp1n", "hx12")}).reset_index()
    per = per[per.n >= SW.MIN_PAIRS].merge(T[["tag", "scene", "failed", "cls", "family", "cluster"]], on=["tag", "scene"])
    groups = [("completed", ~per.failed), ("failed", per.failed), ("failed without FMH", per.failed & (per.family != "FMH"))] + \
             [(f"failed {c}", per.failed & (per.cls == c)) for c in ("late_wide", "straight", "over_turn", "no_zero")] + \
             [("failed late_wide without FMH", per.failed & (per.cls == "late_wide") & (per.family != "FMH"))]
    B = {g: {c: med_boot(per[m][c], per[m].cluster) for c in ("replan", "track", "x12", "hp12", "hp1n", "hx12")} for g, m in groups if m.sum() >= 3}
    return dict(without_fmh=A, split=B, n_dec=int(len(Q)))


# ---------------------------------------------------------------- report
def write_md(out, res):
    f = lambda v, p=3: "-" if v is None or not np.isfinite(v) else f"{v:+.{p}f}"  # noqa: E731
    dci = lambda b, p=3: f"{f(b['f'], p)} / {f(b['c'], p)} | {f(b['d'], p)} [{f(b['lo'], p)}, {f(b['hi'], p)}]"  # noqa: E731
    mci = lambda b, p=1: f"{f(b['m'], p)} [{f(b['lo'], p)}, {f(b['hi'], p)}]"  # noqa: E731
    g, P, n = res["gates"], res["pooled"], res["n"]
    md = ["# PLAN CREEP replication (rf_creep_rep.py)", "",
          "Prereg `plans/2026-10-15-creep-replication-prereg.md`. Held 460 @ 18 decisions (0.5 s), the 24 strict held turns of 17 checkpoints "
          "decision 308 did not use. Measure = rf_switching Amendment 1 unchanged, sign flipped: **c = creep per metre, + = each new plan's 2 s "
          "point lies outside the previous plan's path** (per-turn median of j / ds over the window entry - 20 .. entry + 20 m before the first "
          "zero); C = summed creep (m); x2 = car - plan heading toward the turn over 2 s (deg, per-turn median; - = the car turns less). "
          "Failed = every non-completed turn. Medians failed / completed, difference [95 % CI], bootstrap B 10 000 over nuPlan-log clusters "
          "(every turn of a scene, across checkpoints, in one cluster).", "",
          f"Turns {n['turns']} ({n['tags']} checkpoints; failed {n['failed']}, completed {n['completed']}; {n['scenes']} scenes in {n['clusters']} "
          "log clusters).", "",
          f"Gates: G-code {'pass' if g['code']['passed'] else ('FAIL' if g['code']['passed'] is False else 'not run')} (d308 held_all6 recomputed: "
          + (f"j per m {g['code']['got']['jpm'][0]:+.3f} / {g['code']['got']['jpm'][1]:+.3f}, x2 {g['code']['got']['x2'][0]:+.1f} / "
             f"{g['code']['got']['x2'][1]:+.1f}, {g['code']['n']} turns; switching.md +0.003 / -0.006, -1.6 / +0.4" if g['code']['got'] else "-")
          + f"); G-frame {'pass' if g['frame']['passed'] else 'FAIL'} (worst checkpoint median |anchor - controller| {g['frame']['max_xy']:.3f} m / "
          f"{g['frame']['max_yaw']:.2f} deg); G-human {'pass' if g['human']['passed'] else 'FAIL'} (human log as a plan: c_h {mci(g['human']['c_h'], 4)} m / m, "
          f"failed {g['human']['by_group']['failed']:+.4f} / completed {g['human']['by_group']['completed']:+.4f}); G-n {'pass' if g['n']['passed'] else 'FAIL'} "
          f"({g['n']['turns']} / 408).", "",
          "## R1 / R2 (pooled)", "",
          "| claim | quantity | n f / c | median f / c | diff [CI] | ratio f / c [CI] | verdict |", "|:--|:--|:--|:--|:--|:--|:--|",
          f"| R1 | creep per m (m / m) | {P['n_c_def']['f']} / {P['n_c_def']['c']} | {dci(P['c'])} | {P['ratio']:.2f} [{P['ratio_ci']['lo']:.2f}, "
          f"{P['ratio_ci']['hi']:.2f}] ({P['ratio_ci']['nonpos']} replicates with completed <= 0 left out) | **{P['R1']}** |",
          f"| - | summed creep (m) | {P['C']['n_f']} / {P['C']['n_c']} | {dci(P['C'], 2)} | - | descriptive |",
          f"| R2 | 2 s heading shortfall (deg) | {P['n_x_def']['f']} / {P['n_x_def']['c']} | {dci(P['x2'], 1)} | - | **{P['R2']}** |", "",
          "Reference (d308, PAI RSG-L18-s0, signed j): -0.063 / -0.015 m / m, diff -0.048 [-0.104, -0.016]; x2 -2.5 / +1.6, diff -4.1 [-6.6, -2.3]; "
          "held late-wide -0.104.", "",
          "## R3: classes and outcomes (failed of the class vs all completed turns; descriptive)", "",
          "| group | n | c f / c | diff [CI] | C f / c (m) | diff [CI] | x2 f / c | diff [CI] |", "|:--|--:|:--|:--|:--|:--|:--|:--|"]
    for k, b in res["classes"].items():
        md.append(f"| {k} | {b['n_f']} | {dci(b['c'])} | {dci(b['C'], 2)} | {dci(b['x2'], 1)} |")
    md += ["", "Families: failed vs completed of the same family (own, where both have >= 3 turns) and failed vs the completed turns of the other "
           "families (vs pooled completed).", "",
           "| family | tags | n f / c | failed classes | own: c diff [CI] | own: x2 diff [CI] | c failed / own completed | vs pooled completed: c f / c, diff [CI] | x2 diff [CI] |",
           "|:--|--:|:--|:--|:--|:--|:--|:--|:--|"]
    for fam, z in res["families"].items():
        o = z.get("own")
        vp = z["vs_pooled_completed"]
        md.append(f"| {fam} | {len(z['tags'])} | {z['n_f']} / {z['n_c']} | {', '.join(f'{a} {b}' for a, b in z['cls'].items())} | "
                  f"{(f(o['c']['d']) + ' [' + f(o['c']['lo']) + ', ' + f(o['c']['hi']) + ']') if o else '-'} | "
                  f"{(f(o['x2']['d'], 1) + ' [' + f(o['x2']['lo'], 1) + ', ' + f(o['x2']['hi'], 1) + ']') if o else '-'} | "
                  f"{f(z['failed_med']['c'])} / {f(z['completed_med']['c'])} | {dci(vp['c'])} | {f(vp['x2']['d'], 1)} [{f(vp['x2']['lo'], 1)}, {f(vp['x2']['hi'], 1)}] |")
    md += ["", "Sensitivity (registered, descriptive): R1 / R2 quantities on subsets.", "",
           "| subset | n f / c | c f / c | diff [CI] | x2 f / c | diff [CI] |", "|:--|:--|:--|:--|:--|:--|"]
    for k, b in res["sensitivity"].items():
        md.append(f"| {k} | {b['n_f']} / {b['n_c']} | {dci(b['c'])} | {dci(b['x2'], 1)} |")
    md += ["", "Each checkpoint alone (24 turns; descriptive):", "", "| checkpoint | n f / c | c f / c | diff [CI] | x2 f / c | diff [CI] |",
           "|:--|:--|:--|:--|:--|:--|"]
    for k, b in res["per_tag"].items():
        md.append(f"| {k} | {b['n_f']} / {b['n_c']} | {dci(b['c'])} | {dci(b['x2'], 1)} |")
    md += ["", "## R4: where the heading shortfall starts", "",
           "Per-turn medians over the window decisions, then the group median [cluster-bootstrap CI], deg toward the turn. x05 = car - plan over "
           "0-0.5 s (plan k alone drives the car), x1 over 0-1 s, x12 over the 1-2 s segment (car's yaw(2 s) - yaw(1 s) vs the plan's psi(2 s) - "
           "psi(1 s)), x2 over 0-2 s. Plan / car columns: the plan's own heading change and the car's over 0-1 s and 1-2 s.", "",
           "| group | turns | x05 | x1 | x12 | x2 | plan 0-1 s / car | plan 1-2 s / car |", "|:--|--:|:--|:--|:--|:--|:--|:--|"]
    for k, z in res["R4"].items():
        md.append(f"| {k} | {z['x2']['n']} | {mci(z['x05'])} | {mci(z['x1'])} | {mci(z['x12'])} | {mci(z['x2'])} | {f(z['hp1_med']['m'], 1)} / "
                  f"{f(z['hx1_med']['m'], 1)} | {f(z['hp12_med']['m'], 1)} / {f(z['hx12_med']['m'], 1)} |")
    md += ["", f"Registered reading on the failed turns: **{res['R4_reading']}**."]
    A = res["A1"]
    md += ["", "## Amendment 1 (post hoc, descriptive)", "", "(a) Without the flow head FMH (15 checkpoints): failed subsets vs the completed turns of the same 15.", "",
           "| subset | n f / c | c f / c | diff [CI] | C f / c (m) | diff [CI] | x2 f / c | diff [CI] |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, b in A["without_fmh"].items():
        md.append(f"| {k} | {b['n_f']} / {b['n_c']} | {dci(b['c'])} | {dci(b['C'], 2)} | {dci(b['x2'], 1)} |")
    md += ["", f"(b) The 1-2 s shortfall x12 split through plan k+2 (served 1.0 s after plan k; {A['n_dec']} decisions): replan = plan k+2's own "
           "0-1 s heading - plan k's 1-2 s segment (- = the newer plan, drawn from where the car is, asks for less turn over the same second); "
           "track = car 1-2 s - plan k+2's 0-1 s (- = the car under-executes the plan driving it). Group medians of per-turn medians [CI], deg.", "",
           "| group | turns | x12 | replan | track | plan k 1-2 s | plan k+2 0-1 s | car 1-2 s |", "|:--|--:|:--|:--|:--|:--|:--|:--|"]
    for g, z in A["split"].items():
        md.append(f"| {g} | {z['x12']['n']} | {mci(z['x12'])} | {mci(z['replan'])} | {mci(z['track'])} | {f(z['hp12']['m'], 1)} | {f(z['hp1n']['m'], 1)} | {f(z['hx12']['m'], 1)} |")
    (out / "creep_rep.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:20]))


def fig(T, res, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, aqua, yellow, magenta, grey = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8c8c8c"
    fg, ax = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.0]))
    rng = np.random.default_rng(0)
    # (a) creep per metre by outcome / class
    cats = [("completed", ~T.failed, grey)] + [(c, T.failed & (T.cls == c), col) for c, col in
                                               (("late_wide", orange), ("straight", blue), ("over_turn", magenta), ("other", yellow), ("no_zero", aqua))]
    cats = [c for c in cats if c[1].sum()]
    for i, (name, m, col) in enumerate(cats):
        y = T[m]
        y = y[np.isfinite(y.c)]
        v, fm = y.c.to_numpy(float), (y.family == "FMH").to_numpy()
        xj = i + rng.uniform(-0.28, 0.28, len(v))
        ax[0].scatter(xj[~fm], np.clip(v[~fm], -0.3, 0.4), s=12, color=col, alpha=0.55, lw=0)
        ax[0].scatter(xj[fm], np.clip(v[fm], -0.3, 0.4), s=16, facecolor="none", edgecolor=col, alpha=0.8, lw=1)
        if len(v):
            ax[0].plot([i - 0.35, i + 0.35], [np.median(v)] * 2, color="#222", lw=2)
            ax[0].text(i, 0.415, f"n {len(v)}\n{np.median(v):+.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax[0].set_xticks(range(len(cats)), [c[0].replace("_", "-") for c in cats], fontsize=9)
    ax[0].set_ylim(-0.3, 0.5)
    ax[0].set_ylabel("creep per metre of travel (m / m; clipped at 0.4)\n+ = new plan's 2 s point outside the previous plan")
    ax[0].scatter([], [], s=16, facecolor="none", edgecolor="#555", label="flow head FMH")
    ax[0].scatter([], [], s=12, color="#555", label="other 15 checkpoints")
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    P = res["pooled"]
    ax[0].set_title(f"(a) creep per turn by class, {T.tag.nunique()} held checkpoints, {len(T)} turns\nR1 failed - completed {P['c']['d']:+.3f} [{P['c']['lo']:+.3f}, {P['c']['hi']:+.3f}]: "
                    f"{P['R1']}; without FMH {res['A1']['without_fmh']['all failed']['c']['d']:+.3f} "
                    f"[{res['A1']['without_fmh']['all failed']['c']['lo']:+.3f}, {res['A1']['without_fmh']['all failed']['c']['hi']:+.3f}]", fontsize=10)
    # (b) family: failed vs completed medians
    fams = [f for f in FAMILIES if f in res["families"]]
    for i, fam in enumerate(fams):
        x = T[T.family == fam]
        for dx, m, col in ((-0.17, ~x.failed, grey), (0.17, x.failed, orange)):
            v = x[m].c.to_numpy(float)
            v = v[np.isfinite(v)]
            ax[1].scatter(i + dx + rng.uniform(-0.1, 0.1, len(v)), np.clip(v, -0.3, 0.4), s=10, color=col, alpha=0.5, lw=0)
            if len(v):
                ax[1].plot([i + dx - 0.13, i + dx + 0.13], [np.median(v)] * 2, color="#222", lw=2)
    ax[1].scatter([], [], color=grey, label="completed"), ax[1].scatter([], [], color=orange, label="failed")
    ax[1].legend(fontsize=8, frameon=False, loc="upper right")
    ax[1].set_xticks(range(len(fams)), [f"{f}\n{res['families'][f]['n_f']} / {res['families'][f]['n_c']}" for f in fams], fontsize=9)
    ax[1].set_ylim(-0.3, 0.5)
    ax[1].set_title("(b) creep per turn by checkpoint family\n(failed / completed turns under each name)", fontsize=10)
    # (c) R4 profile
    R4 = res["R4"]
    ks, xs = ("x05", "x1", "x2"), (0.5, 1.0, 2.0)
    for g, col in (("completed", grey), ("failed", "#333333"), ("failed late_wide", orange), ("failed straight", blue), ("failed no_zero", aqua)):
        if g not in R4:
            continue
        mm = [R4[g][k]["m"] for k in ks]
        lo = [R4[g][k]["lo"] for k in ks]
        hi = [R4[g][k]["hi"] for k in ks]
        ax[2].plot(xs, mm, "-o", color=col, lw=2, ms=6, label=f"{g} ({R4[g]['x2']['n']})")
        if g in ("completed", "failed"):
            ax[2].fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
    ax[2].set_xticks(xs, ["0-0.5 s\n(plan k alone)", "0-1 s", "0-2 s"])
    ax[2].set_ylabel("car - plan heading toward the turn (deg)\n- = the car turns less than the plan")
    ax[2].legend(fontsize=8, frameon=False, loc="lower left")
    ax[2].set_title(f"(c) where the shortfall starts (R4: {res['R4_reading']})", fontsize=10)
    for a_ in ax:
        a_.axhline(0, color="#888", lw=0.8)
        a_.grid(alpha=0.25, axis="y")
        a_.spines[["top", "right"]].set_visible(False)
    fg.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check")
    c.add_argument("--held", required=True), c.add_argument("--tax", nargs="*")
    r = sp.add_parser("read")
    r.add_argument("--held", nargs="+", required=True), r.add_argument("--tax", nargs="*"), r.add_argument("--out", required=True)
    r.add_argument("--fig", default=""), r.add_argument("--code-held", default=""), r.add_argument("--code-tax", nargs="*")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(0 if check(a)[0] else 1)
    read(a)
