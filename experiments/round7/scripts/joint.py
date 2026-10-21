#!/usr/bin/env python3
"""ROUND7 joint arm (plans/2026-10-22-joint-arm.md): R6DAG's recipe with three changes, one arm, 2 seeds. Lane script: joint.sh.
  change 1  longitudinal hygiene: F2 on the PAI human rows (fix.py's rule) + the DAgger iteration-1 states relabelled with teacher b5 = b3's
            lateral path at arc lengths max(student served, human from the matched station) where the human keeps going and nothing leads
            (one-sided: never slower than b3, never faster than the human's log where the student is slower), F2 on the b5 labels
  change 2  S-MIX (research/s-rows-and-paper.md E1): S draws re-weighted by R6DAG-s1's own per-row terms (E0), navtrain 51 -> 25, S 32 -> 58 a batch,
            ho_w 3 -> 1.3 (S pressure 2.41x kept)
  change 3  the command from the route's heading change (rmem.cmd_heading) on the AlpaSim-route pool rows (rm_train --cmd-rule heading) and at serve
            (pai_driver JEV_CMD=heading)

  joint.py cmdcheck                   change 3's offline gate on the stored rows -> results/joint/cmd_rule.{md,json}
  joint.py teach --workers N          (AlpaSim venv) b3 path + b5 candidate labels for every iteration-1 decision -> $O/teach/<scene>.npz
  joint.py label                      lead on the student's state, final b5 labels, T4, G1 checks; pools on1_JTPHDK (b5) / on1_JTPHDI (identity,
                                      floor off: tab content equal to on1_JTPHD) -> $O/k/pd.npz, results/joint/rows.{md,json}
  joint.py weights --out W [--f2 1] [--smix 1] [--pool P]   draw file -> $O/w/W.npz + results/joint/w_W.json (all off: R6DAG.npz asserted equal)
  joint.py smoke --tag X --ctrl Y --wfile W   the 300-step smoke's registered checks (X joint, Y the R6DAG control) -> results/joint/smoke_X.json
  joint.py serve --ident D --on D --ref D     serve gates: switch-off re-read equals the stored read; switch-on cmd = rule(fed memory)
  joint.py verdict --json J --md M --read K=F ..   the registered adoption rule on r2.py r281's json + the report lists -> appended to M
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round6/scripts"), str(REPO / "experiments/route_mem/lib"),
                str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/round3/scripts"), str(REPO / "experiments/route_follow/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round7/joint"
O1 = DATA / "runs/round6/dag"                                     # iteration-1 rows / teacher / R6DAG draw file
CR = DATA / "runs/op_parity/cache"
FEAT = DATA / "runs/route_mem/feat"
RES = REPO / "experiments/round7/results/joint"
BASE, POOL_K, POOL_I, SRC_D = "on1_JTPHD", "on1_JTPHDK", "on1_JTPHDI", "on1_PAID"
T_J = 0.5 * np.arange(1, 9)
# change 1 (registered)
H_WIN, H_VMIN, LAG_MAX, D_ARC = 6.0, 1.0, 3.0, 0.05   # human keeps >= 1 m/s over t_h .. t_h + 6 s; |student - human clock| <= 3 s; floor binds > 5 cm
A_ACC = 2.0                                           # amendment 1: the floor is reached from the fed speed at <= 2 m/s^2 (v0 t + A t^2 / 2)
A_LAT = 4.0                                           # T4 line (position estimator); floored labels over it revert to b3
F2_V0, F2_VSTOP, F2_FRAC, F2_H = 4.0, 0.5, 0.6, 20.0   # fix.py F2
# change 2 (registered)
S_PARTS = ("rb0", "rtb1", "y", "ys")
TERMS = ["agent", "road", "lat_side", "lat_onc", "lat_soft"]
M_ACT, M_OUT = 4.0, 2.0
PER_BATCH = dict(ph=16, ph05=8, ph15=8, pd=8)
S_BATCH = dict(off=32, on=58)


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def q3(x):
    x = np.asarray(x, np.float64)
    return [round(float(np.percentile(x, k)), 2) for k in (10, 50, 90)] if len(x) else None


# ---------------------------------------------------------------- change 3: command rule, offline gate
def cmd_cmdcheck(a):
    import rmem as RM
    T = np.load(CR / BASE / "tab.npz")
    F = np.load(FEAT / f"{BASE}.npz")["f"]
    pool = T["pool"].astype(str)
    bins = [(0, 10), (10, 30), (30, 60), (60, 400)]
    rec, L = {}, ["# Change 3: command from the route's heading change (rmem.cmd_heading, look 40 m, 20 deg), offline (joint.py cmdcheck)", "",
                  "Rows fed >= 4 m/s with a finite label; label turn = |4 s label heading|; L/R share of the old command (the shipped rule on the latest "
                  "route message, as stored) and the new rule on the same rows' stored float16 memory. navtrain: one shard (s0of12), its own NAVSIM command "
                  "as `old` (the target profile).", "",
                  "| rows | rule | n | label turn 0-10 deg | 10-30 | 30-60 | >= 60 | sign agrees (>= 30 deg) |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]

    def prof(name, ego, fut, f, spd):
        m = (spd >= 4) & np.isfinite(fut.reshape(len(fut), -1)).all(1)
        t = np.degrees(np.abs(wrap(fut[m, -1, 2].astype(np.float64))))
        sg = np.sign(wrap(fut[m, -1, 2].astype(np.float64)))
        out = {}
        for lbl, c in (("old", np.where(ego[m, 1:4].max(1) > 0.5, ego[m, 1:4].argmax(1), 3)), ("new", RM.cmd_heading(f[m]).argmax(1))):
            lr = [float(np.isin(c[(t >= lo) & (t < hi)], [0, 2]).mean()) for lo, hi in bins]
            big = (t >= 30) & np.isin(c, [0, 2])
            ag = float(((c == 0) & (sg > 0) | (c == 2) & (sg < 0))[big].mean())
            out[lbl] = dict(n=int(m.sum()), lr=lr, agree=ag)
            L.append(f"| {name} | {lbl} | {m.sum()} | " + " | ".join(f"{100 * x:.1f} %" for x in lr) + f" | {100 * ag:.1f} % |")
        return out
    for p in ("ph", "ph05", "ph15", "pd") + S_PARTS:
        i = np.flatnonzero(pool == p)
        rec[p] = prof(p, T["ego"][i], T["fut"][i], F[i], T["speed"][i])
    n = np.load(CR / "navtrain_full.s0of12/tab.npz")
    rec["navtrain"] = prof("navtrain s0of12", n["ego"], n["fut"], np.load(FEAT / "navtrain_full.s0of12.npz")["f"], n["speed"])
    pai = [rec[p]["new"] for p in ("ph", "ph05", "ph15", "pd")]
    g = dict(nonturn=all(x["lr"][0] <= 0.10 for x in pai), turn30=all(x["lr"][2] >= 0.75 for x in pai), turn60=all(x["lr"][3] >= 0.90 for x in pai),
             sign=all(x["agree"] >= 0.97 for x in pai))
    rec["gate"] = dict(lines="PAI parts (ph, ph05, ph15, pd), new rule: L/R <= 10 % at 0-10 deg, >= 75 % at 30-60, >= 90 % at >= 60, sign >= 97 %", **g,
                       passed=bool(all(g.values())))
    L += ["", f"**G2 (offline): {'PASS' if rec['gate']['passed'] else 'FAIL'}** ({rec['gate']['lines']}; {json.dumps(g)})."]
    dump(RES / "cmd_rule.json", rec)
    (RES / "cmd_rule.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if rec["gate"]["passed"] else 1)


# ---------------------------------------------------------------- change 1: teacher b5 (AlpaSim venv)
def human_t(ref, s0):
    """The moment (s, log clock from the first log pose) the human first reached station s0 of ref, nan outside its recorded stations."""
    ht, hs = getattr(ref, "ht", None), getattr(ref, "hs", None)
    if ht is None or not (hs[0] <= s0 <= hs[-1]):
        return float("nan")
    i = int(np.searchsorted(hs, s0, "left"))
    return float(ht[0] if i == 0 else ht[i - 1] + (s0 - hs[i - 1]) / max(hs[i] - hs[i - 1], 1e-9) * (ht[i] - ht[i - 1]))


def human_minrate(ref, th):
    """min station rate (m/s, 0.5 s steps) of the human over th .. th + H_WIN; nan when the log ends before th + H_WIN + 0.5."""
    ht, hs = ref.ht, ref.hs
    if ht is None or not np.isfinite(th) or th + H_WIN + 0.5 > ht[-1]:
        return float("nan")
    tq = th + np.arange(0.0, H_WIN + 1e-9, 0.5)
    return float(np.min((np.interp(tq + 0.5, ht, hs) - np.interp(tq, ht, hs)) / 0.5))


def _teach_one(job):
    import dag as D
    scene, cfg, rowf, out = job
    with np.load(rowf) as z:
        Z = {k: z[k] for k in ("k", "anchor", "poses", "v0", "after_event", "after_end", "t0_us")}
    ds = D.load_ds(scene, cfg)
    (_, ref), _ = D.route_ref(ds)
    ts = D.traj_of(ds)[2]
    ts0 = float(ts[0]) if ts is not None else float("nan")
    n = len(Z["k"])
    fut3, fut5 = np.full((n, 8, 3), np.nan, np.float32), np.full((n, 8, 3), np.nan, np.float32)
    a_s, a_h, a_f = (np.full((n, 8), np.nan, np.float32) for _ in range(3))
    th, lag, hmin = (np.full(n, np.nan, np.float32) for _ in range(3))
    ok, cond = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        A = Z["anchor"][i].astype(np.float64)
        a = D.arcs_of(Z["poses"][i])
        pr = ref.project(A)
        ah = D.human_arcs(ref, pr[0]) if pr is not None else np.full(8, np.inf)
        amax = max(a[-1], float(ah[-1]) if np.isfinite(ah[-1]) else 0.0)
        path, info = D.teacher_path(ref, A, float(Z["v0"][i]), amax)
        if path is None:
            continue
        s0 = info["s0"]
        ah = D.human_arcs(ref, s0)
        th[i] = human_t(ref, s0)
        lag[i] = (Z["t0_us"][i] - ts0) * 1e-6 - th[i]
        hmin[i] = human_minrate(ref, float(th[i]))
        fl = np.minimum(ah, float(Z["v0"][i]) * T_J + 0.5 * A_ACC * T_J ** 2)   # amendment 1: the human's pace, reached at <= A_ACC
        c = bool(np.isfinite(ah).all() and hmin[i] >= H_VMIN and abs(lag[i]) <= LAG_MAX and fl[-1] > a[-1] + D_ARC)
        af = np.maximum(a, fl) if c else a
        fut3[i] = D.place(path, a, A)[0]
        fut5[i] = D.place(path, af, A)[0]
        a_s[i], a_h[i], a_f[i] = a, np.where(np.isfinite(ah), ah, np.nan), af
        ok[i], cond[i] = np.isfinite(fut3[i]).all() and np.isfinite(fut5[i]).all(), c
    tmp = Path(out) / f"{scene}.tmp.npz"
    np.savez(tmp, k=Z["k"], fut3=fut3, fut5=fut5, a_s=a_s, a_h=a_h, a_f=a_f, th=th, lag=lag, hmin=hmin, ok=ok, cond=cond, v0=Z["v0"],
             served=Z["poses"], version="b5")
    os.replace(tmp, Path(out) / f"{scene}.npz")
    return dict(scene=scene, n=n, ok=int(ok.sum()), cond=int(cond.sum()))


def cmd_teach(a):
    import dag as D
    from jevdrive import par
    out = O / "teach"
    out.mkdir(parents=True, exist_ok=True)
    cfg = D.scene_cfgs(str(O1 / "roll/runs/*/sim/generated-user-config-0.yaml"))
    files = sorted((O1 / "rows/PAID").glob("clipgt-*.npz"))
    want = set(Path(a.chunk_list).read_text().split()) if a.chunk_list else None
    jobs = [(f.stem, cfg[f.stem], str(f), str(out)) for f in files if f.stem in cfg and (want is None or f.stem in want)]
    res = par.pmap(_teach_one, jobs, workers=a.workers or None, desc="teach b5", skip=lambda j: (out / f"{j[0]}.npz").exists())
    res.raise_if_failed()
    v = [x for x in res.values if x]
    print(json.dumps(dict(scenes=len(v), rows=sum(x["n"] for x in v), ok=sum(x["ok"] for x in v), cond=sum(x["cond"] for x in v),
                          no_cfg=[f.stem for f in files if f.stem not in cfg])))


# ---------------------------------------------------------------- change 1: labels, pools, G1
def link_pool(dst, tab):
    """dst = on1_JTPHD with tab.npz := tab (written, never linked); every other file hard-linked (same disk, zero bytes), feat linked."""
    S, T_, Sw, Tw = CR / BASE, CR / dst, CR / f"{BASE}@warp", CR / f"{dst}@warp"
    assert not (T_.exists() or Tw.exists() or (FEAT / f"{dst}.npz").exists()), f"{dst} exists: never overwritten"
    T_.mkdir(), Tw.mkdir()
    for f in S.iterdir():
        if f.name != "tab.npz":
            os.link(f, T_ / f.name)
    for f in Sw.iterdir():
        os.link(f, Tw / f.name)
    os.link(FEAT / f"{BASE}.npz", FEAT / f"{dst}.npz")
    np.savez(T_ / "tab.tmp.npz", **tab)
    os.replace(T_ / "tab.tmp.npz", T_ / "tab.npz")


def cmd_label(a):
    import r5
    from dag import alat_pos
    from offroad_mv import label_feats
    tb = dict(np.load(CR / BASE / "tab.npz"))
    pool = tb["pool"].astype(str)
    j0 = np.flatnonzero(pool == "pd")
    assert np.array_equal(j0, np.arange(j0[0], len(pool))), "pd rows are not the tail of on1_JTPHD"
    P = np.load(CR / SRC_D / "tab.npz")
    X = np.load(CR / SRC_D / "dagger.npz")
    nm = P["names"].astype(str)
    assert all(x == "on:pd:" + y[3:] for x, y in zip(tb["names"][j0].astype(str), nm)), "on1_JTPHD pd rows != on1_PAID rows"
    assert np.array_equal(tb["fut"][j0], P["fut"]), "on1_JTPHD pd labels != on1_PAID's"
    n = len(nm)
    sc = np.array([x.split(":")[2] for x in nm])
    kk = np.array([int(x.split(":")[3]) for x in nm])
    Z = {k: np.full((n, 8, 3), np.nan, np.float32) for k in ("fut3", "fut5")} | {k: np.full((n, 8), np.nan, np.float32) for k in ("a_s", "a_h", "a_f")}
    Z |= {k: np.full(n, np.nan, np.float32) for k in ("lag", "hmin", "v0t")} | {k: np.zeros(n, bool) for k in ("ok", "cond", "found")}
    for s in np.unique(sc):
        j = np.flatnonzero(sc == s)
        f = O / "teach" / f"{s}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            assert str(z["version"]) == "b5"
            pos = {int(k): i for i, k in enumerate(z["k"])}
            hit = np.array([int(k) in pos for k in kk[j]])
            i = np.array([pos[int(k)] for k in kk[j][hit]], int)
            for key in ("fut3", "fut5", "a_s", "a_h", "a_f", "lag", "hmin", "ok", "cond"):
                Z[key][j[hit]] = z[key][i]
            Z["v0t"][j[hit]] = z["v0"][i]
            Z["found"][j[hit]] = True
    draw = X["draw"].astype(bool)
    v0 = P["speed"].astype(np.float64)
    fut_b3 = P["fut"].astype(np.float64)
    # G1a: the b5 code with the floor off reproduces the stored b3 labels on the drawable rows
    m = draw & Z["found"] & Z["ok"]
    g1a = float(np.abs(Z["fut3"][m, :, :2] - fut_b3[m, :, :2]).max()) if m.any() else float("inf")
    g1a_cov = float(m.sum() / max(draw.sum(), 1))
    # lead on the student's state (r5 LEAD rule on the candidate label path, t0 boxes of on1_PAID/agent.npz)
    A = np.load(CR / SRC_D / "agent.npz")
    assert np.array_equal(A["tokens"].astype(str), nm), "agent.npz rows != on1_PAID"
    box0, val0 = A["box"][:, 0], A["valid"][:, 0] & A["ok"][:, None]
    cand = draw & Z["found"] & Z["ok"] & Z["cond"]
    lead = np.zeros(n, bool)
    ci = np.flatnonzero(cand)
    for i in range(0, len(ci), 20000):
        r = ci[i:i + 20000]
        vr = np.maximum(v0[r], np.nan_to_num(Z["a_f"][r, -1]) / 4.0)
        g, _ = r5.lead_gap(Z["fut5"][r], box0[r], val0[r], vr)
        lead[r] = np.isfinite(g) | np.isnan(g)
    fl = cand & ~lead
    fut5 = np.where(fl[:, None, None], Z["fut5"], fut_b3)
    al5 = alat_pos(np.nan_to_num(fut5))
    al3 = alat_pos(fut_b3)
    rev = fl & (al5 > A_LAT)                                          # floored labels over the T4 line revert to b3
    fl &= ~rev
    fut5 = np.where(fl[:, None, None], Z["fut5"], fut_b3).astype(np.float32)
    al5 = alat_pos(fut5.astype(np.float64))
    t4 = float(np.percentile(al5[draw], 99))
    # G1b: one-sided
    arc = lambda f: np.cumsum(np.hypot(*np.diff(np.concatenate([np.zeros((len(f), 1, 2)), np.asarray(f, np.float64)[:, :, :2]], 1), axis=1).transpose(2, 0, 1)), 1)  # noqa: E731
    d_lo = float((arc(fut5[draw]) - arc(fut_b3[draw])).min())
    # amendment 2: the bound on the change against b3 (label arc - b3 label arc <= floor - student arc, and floor <= the human's), not on the
    # absolute chord length (b3's own labels start up to ~6 m ahead of the origin where the anchor lies before the route's start, k = 0 / 1)
    d_hi = float(((arc(fut5[fl]) - arc(fut_b3[fl])) - (Z["a_f"][fl] - Z["a_s"][fl])).max()) if fl.any() else 0.0
    d_h = float((Z["a_f"][fl] - np.maximum(Z["a_s"][fl], Z["a_h"][fl])).max()) if fl.any() else 0.0
    g1b = bool(d_lo >= -5e-3 and d_hi <= 0.05 and d_h <= 1e-3 and not (fl & ~cand).any())
    a3 = arc(fut_b3[draw])
    start_off = int((a3[:, 0] - Z["a_s"][draw, 0] > 1.0).sum())
    # G1c: the clocks (student decision time vs the human's at the matched station) agree at k = 0
    k0 = (kk == 0) & np.isfinite(Z["lag"])
    lag0 = float(np.median(np.abs(Z["lag"][k0]))) if k0.any() else float("inf")
    # F2 on the b5 labels (pd rows), the lead of (iv) where computed; rows never tested for lead count as lead-free only when not candidates
    Fz = label_feats(v0, fut5.astype(np.float64))
    f2 = draw & (v0 >= F2_V0) & (Fz["vend"] >= F2_VSTOP) & (Fz["vend"] < F2_FRAC * v0) & (Fz["hmax"] < F2_H) & ~lead
    cls = X["cls"].astype(str)
    inc = (arc(fut5[fl])[:, -1] - arc(fut_b3[fl])[:, -1]) if fl.any() else np.zeros(0)
    reasons = dict(drawable=int(draw.sum()), teach_found=int((draw & Z["found"]).sum()), no_window=int((draw & Z["found"] & ~np.isfinite(Z["a_h"]).all(1)).sum()),
                   human_slows=int((draw & Z["found"] & (Z["hmin"] < H_VMIN)).sum()), lag=int((draw & Z["found"] & (np.abs(Z["lag"]) > LAG_MAX)).sum()),
                   not_slower=int((draw & Z["found"] & np.isfinite(Z["a_h"]).all(1) & (np.minimum(Z["a_h"][:, -1], Z["v0t"] * 4 + 8 * A_ACC) <= Z["a_s"][:, -1] + D_ARC)).sum()),
                   candidates=int(cand.sum()), lead=int((cand & lead).sum()), t4_revert=int(rev.sum()), floored=int(fl.sum()))
    spd = np.digitize(v0, [5, 10, 15, 20, 25])
    sb = ["2-5", "5-10", "10-15", "15-20", "20-25", ">=25"]
    by = {c: [int((fl & (cls == c)).sum()), int((draw & (cls == c)).sum())] for c in sorted(set(cls[draw]))}
    bys = {sb[b]: [int((fl & (spd == b)).sum()), int((draw & (spd == b)).sum())] for b in range(6)}
    straight = draw & (np.degrees(np.abs(wrap(fut_b3[:, -1, 2]))) < 5)
    g = dict(G1a=bool(g1a <= 1e-3 and g1a_cov >= 0.99), G1b=g1b, G1c=bool(lag0 <= 0.5), T4=bool(t4 <= A_LAT))
    rec = dict(rule=dict(floor=f"arc_j = max(student served, min(human from the matched station, v0 t + {A_ACC} t^2 / 2)) iff the human's 8 arcs exist, the human keeps >= {H_VMIN} m/s over "
                         f"t_h .. t_h + {H_WIN} s, |student clock - t_h| <= {LAG_MAX} s, that floor's 4 s arc exceeds the student's by > {D_ARC} m and no object "
                         "leads on the floored label path (r5 LEAD rule, t0 boxes); floored labels over the T4 line (4 m/s^2) revert to b3",
                         f2="F2 on the final labels: v0 >= 4, 0.5 <= v_end < 0.6 v0, max |heading| < 20 deg, no lead -> weight 0"),
               reasons=reasons, share_floored=float(fl.sum() / max(draw.sum(), 1)), arc4_increase_q=q3(inc), by_class=by, by_speed=bys,
               straight=dict(floored=int((fl & straight).sum()), drawable=int(straight.sum()), inc_q=q3((arc(fut5[fl & straight])[:, -1] - arc(fut_b3[fl & straight])[:, -1])) if (fl & straight).any() else None),
               f2_pd=int(f2.sum()), f2_pd_b3=int((draw & (v0 >= F2_V0) & (label_feats(v0, fut_b3)["vend"] >= F2_VSTOP) & (label_feats(v0, fut_b3)["vend"] < F2_FRAC * v0)
                                                  & (label_feats(v0, fut_b3)["hmax"] < F2_H)).sum()),
               G1a=dict(max_abs_m=g1a, coverage=g1a_cov), G1b=dict(min_arc_change=d_lo, max_change_over_floor=d_hi, max_floor_over_human=d_h, b3_start_offset_rows=start_off), G1c=dict(k0_rows=int(k0.sum()), median_abs_lag_s=lag0,
               lag_q_all=q3(Z["lag"][draw & np.isfinite(Z["lag"])])), T4=dict(p99_b5=t4, p99_b3=float(np.percentile(al3[draw], 99)), over=int((al5[draw] > A_LAT).sum())),
               gates=g, passed=bool(all(g.values())))
    (O / "k").mkdir(parents=True, exist_ok=True)
    np.savez(O / "k/pd.npz", names=nm, draw=draw, floored=fl, cand=cand, lead=lead, t4_revert=rev, f2=f2, lag=Z["lag"], hmin=Z["hmin"], a_s=Z["a_s"],
             a_h=Z["a_h"], fut5=fut5, alat5=al5)
    dump(RES / "rows.json", rec)
    L = ["# Change 1 (DAgger rows): teacher b5 on the iteration-1 states (joint.py label)", "",
         f"Rule: {rec['rule']['floor']}. {rec['rule']['f2']}.", "",
         f"Drawable rows (b3, R6DAG): {draw.sum()}; floored {fl.sum()} ({rec['share_floored']:.1%}); reasons {json.dumps(reasons)}.",
         f"4 s arc increase on floored rows p10 / p50 / p90 {rec['arc4_increase_q']} m; straight rows (|4 s heading| < 5 deg) floored "
         f"{rec['straight']['floored']} / {rec['straight']['drawable']}, increase {rec['straight']['inc_q']} m.",
         f"By class (floored / drawable): {json.dumps(by)}; by speed (m/s): {json.dumps(bys)}.",
         f"F2 on the b5 labels: {f2.sum()} pd rows to weight 0 (on b3 labels the same rule, lead ignored, would take {rec['f2_pd_b3']}).", "",
         f"G1a b3 reproduced: max |diff| {g1a:.2e} m over {g1a_cov:.1%} of the drawable rows -> {'PASS' if g['G1a'] else 'FAIL'}; "
         f"G1b one-sided: min arc change vs b3 {d_lo:+.4f} m, change minus (floor - student) max {d_hi:+.4f} m, floor minus max(student, human) max {d_h:+.4f} m "
         f"(b3 labels starting > 1 m ahead of the student's first arc: {start_off} rows) -> {'PASS' if g1b else 'FAIL'}; "
         f"G1c clocks at k = 0: median |lag| {lag0:.3f} s ({k0.sum()} rows; all drawable lag p10 / p50 / p90 {rec['G1c']['lag_q_all']}) -> {'PASS' if g['G1c'] else 'FAIL'}; "
         f"T4 p99 {t4:.2f} (b3 {rec['T4']['p99_b3']:.2f}) -> {'PASS' if g['T4'] else 'FAIL'}.", "",
         f"**G1: {'PASS' if rec['passed'] else 'FAIL'}.**"]
    (RES / "rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not rec["passed"]:
        sys.exit(1)
    tk = dict(tb)
    tk["fut"] = tb["fut"].copy()
    tk["fut"][j0] = fut5
    link_pool(POOL_K, tk)
    link_pool(POOL_I, tb)
    chk = np.load(CR / POOL_I / "tab.npz")
    assert all(np.array_equal(chk[k], tb[k], equal_nan=chk[k].dtype.kind == "f") for k in tb), "identity pool tab != on1_JTPHD's"
    print(f"pools {POOL_K} (pd fut := b5 on {fl.sum()} rows) and {POOL_I} (identical tab) built")


# ---------------------------------------------------------------- weights (changes 1 and 2)
def f2_human(names):
    """fix.py's F2 on the three PAI human parts -> bool mask over the merged pool's rows."""
    import r3
    from offroad_mv import label_feats
    idx = {x: i for i, x in enumerate(names)}
    LR = np.load(DATA / "runs/round5/r5/N/lead_rows.npz")
    out, cnt = np.zeros(len(names), bool), {}
    for p in r3.PARTS:
        t = r3.load_part(p)
        nm = t["names"].astype(str)
        ix = np.array([idx[f"on:{p}:{x[3:]}"] for x in nm])
        v0, fut = t["speed"].astype(np.float64), t["fut"].astype(np.float64)
        F = label_feats(v0, fut)
        lp = {x: i for i, x in enumerate(LR[f"{p}_names"].astype(str))}
        lead = LR[f"{p}_lead"][np.array([lp[f"on:{p}:{x[3:]}"] for x in nm])].astype(bool)
        m = (v0 >= F2_V0) & (F["vend"] >= F2_VSTOP) & (F["vend"] < F2_FRAC * v0) & (F["hmax"] < F2_H) & ~lead
        out[ix] = m
        cnt[p] = int(m.sum())
    return out, cnt


def smix_mult(names):
    """S-MIX multipliers over the merged pool's rows (1 outside S): 0 unlearnable (t0 off the raster, or active on agent / road while the
    human-consistent path `fut` also violates), 4 active (any S term > 0 under R6DAG-s1) and learnable, 2 from a zero / partial source scene,
    1 otherwise. -> (mult, record)."""
    import pandas as pd
    t = pd.read_parquet(DATA / "runs/practice_nuplan/lat_diag/trainee_R6DAG-L18-s1.parquet")
    n0 = pd.read_parquet(DATA / "runs/practice_nuplan/lat_diag/trainee_NTV0-L18-s0.parquet")
    st = pd.read_parquet(DATA / "runs/s_rows_audit/rows_audit.parquet")[["name", "off_t0", "fut_agent", "fut_road", "fut_ok", "outcome"]]
    t = t[t.pool.isin(S_PARTS)].merge(st, on="name", how="left", validate="1:1")
    assert t.off_t0.notna().all(), "audit rows missing for some S rows"
    act = (t[TERMS] > 0).any(axis=1).to_numpy()
    ag = ((t.agent > 0) | (t.road > 0)).to_numpy()
    u = (ag & ((t.fut_agent > 0) | (t.fut_road > 0)).to_numpy() & t.fut_ok.astype(bool).to_numpy()) | t.off_t0.astype(bool).to_numpy()
    out = t.outcome.isin(["zero", "partial"]).to_numpy()
    m = np.where(u, 0.0, np.where(act, M_ACT, np.where(out, M_OUT, 1.0)))
    idx = {x: i for i, x in enumerate(names)}
    mult = np.ones(len(names))
    ri = np.array([idx[x] for x in t.name])
    mult[ri] = m
    n0 = n0[n0.pool.isin(S_PARTS)].set_index("name").reindex(t.name)
    a0 = (n0[TERMS] > 0).any(axis=1).to_numpy()
    rec = dict(rows=len(t), active_r6dag=float(act.mean()), active_ntv0=float(a0.mean()), e0_diff_pp=float(100 * (act.mean() - a0.mean())),
               unlearnable=float(u.mean()), active_learnable=float((act & ~u).mean()), mult_counts={str(k): int(v) for k, v in zip(*np.unique(m, return_counts=True))})
    return mult, ri, (act & ~u), rec


def cmd_weights(a):
    B = np.load(O1 / "w/R6DAG.npz")
    names, w0, grp = B["names"].astype(str), B["w"].astype(np.float64), B["grp"].astype(str)
    tb = np.load(CR / a.pool / "tab.npz")
    assert np.array_equal(tb["names"].astype(str), names), f"{a.pool} rows != R6DAG's"
    part = tb["pool"].astype(str)
    k_old = 72
    S_b = S_BATCH["on" if a.smix else "off"]
    k_on = sum(PER_BATCH.values()) + S_b
    w = w0.copy()
    rec = dict(out=a.out, pool=a.pool, f2=bool(a.f2), smix=bool(a.smix), k_on=k_on)
    keep = np.ones(len(w), bool)
    if a.f2:
        fh, cnt = f2_human(names)
        K = np.load(O / "k/pd.npz") if a.pool == POOL_K else None
        fpd = np.zeros(len(w), bool)
        if K is not None:
            j0 = np.flatnonzero(part == "pd")
            assert np.array_equal(K["names"].astype(str), np.array(["on:" + x.split(":", 2)[2] for x in names[j0]])), "pd.npz rows"
            fpd[j0] = K["f2"]
        keep = ~(fh | fpd)
        rec |= dict(f2_rows=cnt, f2_human_drawn_removed=int((fh & (w0 > 0)).sum()), f2_pd_removed=int((fpd & (w0 > 0)).sum()))
    mult = np.ones(len(w))
    if a.smix:
        mult, ri, al, srec = smix_mult(names)
        rec["smix"] = srec
    for g in list(PER_BATCH) + ["S"]:
        m = np.isin(part, S_PARTS) if g == "S" else part == g
        tgt = (S_b if g == "S" else PER_BATCH[g]) / k_on
        x = w0[m] * keep[m] * mult[m]
        if np.array_equal(x, w0[m]) and k_on == k_old:                # nothing changed in this group: R6DAG's weights untouched (bit-equal)
            continue
        w[m] = x * tgt / x.sum()
    rest = ~np.isin(part, list(PER_BATCH) + list(S_PARTS))
    assert (w[rest] == 0).all() and (w0[rest] == 0).all(), "weight outside the drawn parts"
    assert abs(w.sum() - 1) < 1e-9, w.sum()
    if not (a.f2 or a.smix):
        assert np.array_equal(w, w0), "all-off draw file != R6DAG.npz"
    sh = {g: float(k_on * w[np.isin(part, S_PARTS) if g == "S" else part == g].sum()) for g in list(PER_BATCH) + ["S"]}
    rec |= dict(per_batch=sh, drawable={g: int((w[np.isin(part, S_PARTS) if g == "S" else part == g] > 0).sum()) for g in sh})
    if a.smix:
        sw = w[ri]
        rec["smix"]["draw_share_active_learnable"] = float(sw[al].sum() / sw.sum())
        r6 = w0[ri]
        rec["smix"]["draw_share_active_learnable_r6dag"] = float(r6[al].sum() / r6.sum())
        rec["smix"]["gate_30pct"] = bool(rec["smix"]["draw_share_active_learnable"] >= 0.30)
    gs = defaultdict(float)
    for g_, x in zip(grp, w):
        gs[g_] += x
    rec["group_share"] = {k: float(v) for k, v in sorted(gs.items()) if v > 0}
    (O / "w").mkdir(parents=True, exist_ok=True)
    np.savez(O / "w" / f"{a.out}.npz", names=B["names"], w=w, grp=B["grp"])
    dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))
    if a.smix and not rec["smix"]["gate_30pct"]:
        sys.exit(1)


# ---------------------------------------------------------------- smoke, serve gates
def events(tag):
    import r3
    d = r3.run_dir(tag)
    S = defaultdict(dict)
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S[e["tag"]][e["step"]] = e["value"]
    return d, S


def cmd_smoke(a):
    dj, J = events(a.tag)
    dc, C = events(a.ctrl)
    ratio = lambda S: sum(S["loss/grad/pool_hinge"].values()) / sum(S["loss/grad/imit"].values())  # noqa: E731
    rj, rc = ratio(J), ratio(C)
    log = (dj / "log.txt").read_text()
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", log)
    done = json.loads((dj / "DONE").read_text()) if (dj / "DONE").exists() else {}
    gs = (done.get("draw_shares") or {}).get("group_share", {})
    ex = json.loads((RES / f"w_{a.wfile}.json").read_text())
    ade = J["dev/ade"]
    last = max(ade)
    cmdr = re.search(r"--cmd-rule \(round7 joint arm\): (\{.*\})", log)
    ck = dict(done=bool(done), finite=all(np.isfinite(list(v.values())).all() for v in J.values()),
              batch=(b.groups() if b else None, (str(a.nav), "13", str(ex["k_on"]))),
              shares=all(abs(gs.get(k, -1) - v) < 1e-6 for k, v in ex["group_share"].items()),
              grad_ratio=dict(joint=rj, control=rc, rel=rj / rc, band=[0.7, 1.3]),
              dev_ade=dict(step=last, joint=ade[last], line=0.5362 + 0.03),
              cmd_rule=json.loads(cmdr.group(1)) if cmdr else None)
    ok = dict(done=ck["done"], finite=ck["finite"], batch=bool(b) and b.groups() == ck["batch"][1], shares=ck["shares"],
              grad=0.7 <= rj / rc <= 1.3, dev=ade[last] <= 0.5362 + 0.03, cmd=(ck["cmd_rule"] is not None) == bool(a.cmd))
    res = dict(tag=a.tag, ctrl=a.ctrl, checks=ck, ok=ok, pass_=bool(all(ok.values())))
    dump(RES / f"smoke_{a.tag}.json", res)
    print(json.dumps(res, indent=1, default=float))
    sys.exit(0 if res["pass_"] else 2)


def drive_recs(run_dir):
    """drive.jsonl inference records of a pai_box read dir -> {(scene, k): rec}."""
    out = {}
    for f in glob.glob(f"{run_dir}/runs/*/sim/driver-logs/drive*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "drive" and r.get("infer"):
                out[(r["scene"], r["k"])] = r
    return out


def scores(run_dir):
    out = {}
    for f in glob.glob(f"{run_dir}/runs/*/sim/aggregate/results-summary.json"):
        for r in json.loads(Path(f).read_text())["rollouts"]:
            out[r["clipgt_id"]] = r.get("score", r.get("pcs"))
    return out


def cmd_serve(a):
    import rmem as RM
    ref, idn, on = drive_recs(a.ref), drive_recs(a.ident), drive_recs(a.on)
    common = sorted(set(ref) & set(idn))
    dp = max((float(np.abs(np.array(ref[k]["poses"]) - np.array(idn[k]["poses"])).max()) for k in common), default=float("inf"))
    sr, si, so = scores(a.ref), scores(a.ident), scores(a.on)
    same_score = all(sr.get(s) == si.get(s) for s in si)
    agree = [int(np.argmax(RM.cmd_heading(np.asarray(r["rmem"], np.float16)))) == r["cmd"] for r in on.values() if "rmem" in r]
    ch = [r["cmd"] != r.get("cmd_msg", r["cmd"]) for r in on.values()]
    res = dict(ident=dict(inferences=len(idn), matched=len(common), poses_max_abs=dp, scores=si, ref_scores={s: sr.get(s) for s in si}, same_scores=same_score),
               on=dict(inferences=len(on), rule_agrees=float(np.mean(agree)) if agree else 0.0, changed_vs_message=float(np.mean(ch)) if ch else 0.0, scores=so))
    res["pass_"] = bool(len(common) >= 0.99 * max(len(idn), 1) and dp <= 1e-3 and same_score and res["on"]["rule_agrees"] >= 0.999 and len(on) > 0)
    dump(RES / "serve_gate.json", res)
    print(json.dumps(res, indent=1, default=float))
    sys.exit(0 if res["pass_"] else 2)


# ---------------------------------------------------------------- verdict
SIX = ("bbc53aa8", "d90fe123", "a1b166cc", "94c1d722", "2a0efada", "37d660b1")
CUTIN = ("596af0c7", "ea468790")
PCS_R6 = {"s0": -213.0, "s1": -155.0}                     # decision 434: R6DAG ruler - NTV0 (replaced by the same report's DAG-s* rows)
ON_R6S1 = "R6DAG-s1-agH"


def pcs_txt(pk, ref):
    return "-" if pk is None else "%+.0f [%+.0f, %+.0f] vs %+.0f" % (pk["mean"], pk["lo"], pk["hi"], ref)


def cmd_verdict(a):
    import pandas as pd
    J = json.loads(Path(a.json).read_text())
    R, P = J["reads"], J["pairs"]
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    lat = lambda k: R[k]["z_offroad"] + R[k]["z_zero-other"]  # noqa: E731
    Q = json.loads(Path(a.pcs).read_text()) if a.pcs and Path(a.pcs).exists() else {}
    f3 = lambda d: f"{d.get('mean', float('nan')):+.3f} [{d.get('lo', float('nan')):+.3f}, {d.get('hi', float('nan')):+.3f}]"  # noqa: E731
    rows, ok_all, ver = [], True, {}
    for sd in ("s0", "s1"):
        k, c, h = f"R7J-{sd}-ag", f"R6DAG-{sd}-ag", f"R7J-{sd}-agH"
        if k not in R:
            continue
        dr, dn = P[f"{k}:{c}"]["mean"], P[f"{k}:NTV0-ag"]["mean"]
        hon = P.get(f"{h}:{ON_R6S1}", {}).get("mean", {})
        qq, q6 = Q.get(f"R7J-{sd}+AG"), Q.get(f"DAG-{sd}+AG")              # pai_pcs_local r5: diff_point / diff_ci vs NTV0-s0+AG, same draws
        pk = dict(mean=qq["diff_point"], lo=qq["diff_ci"][0], hi=qq["diff_ci"][1]) if qq and "diff_point" in qq else None
        PCS_R6[sd] = q6["diff_point"] if q6 and "diff_point" in q6 else PCS_R6[sd]
        ok = dict(mean_vs_r6=dr["mean"] >= 0, ntv0_ci=dn["lo"] > 0, lat=lat(k) <= lat(c), launch=R[k].get("launch", 0) >= 34,
                  ruler=pk is not None and pk["mean"] >= PCS_R6[sd], harm_on=bool(hon) and hon["mean"] >= 0)
        ver[sd] = ok
        ok_all &= all(ok.values())
        rows.append(f"| {sd} | {R[k]['mean']:.3f} | {f3(dr)} | {f3(dn)} | {lat(k)} vs {lat(c)} | {R[k].get('launch', '')} | "
                    f"{pcs_txt(pk, PCS_R6[sd])} | "
                    f"{R.get(h, {}).get('mean', float('nan')):.3f}, {f3(hon)} | {' '.join(q for q, v in ok.items() if not v) or 'all pass'} |")
    com = sorted(set.intersection(*[set(X.index[X.score.notna()]) for X in F.values()]))
    solved = F["NTV0-ag"].score.reindex(com) >= 1 - 1e-4
    L = ["", "## Registered adoption rule (plan section 6)", "",
         "Both seeds: OFF 281 mean - same-seed R6DAG >= 0; - NTV0 95 % CI > 0; offroad + zero-other <= same-seed R6DAG's; launches >= 34 / 37; "
         "local PCS ruler - NTV0 >= same-seed R6DAG's (d434: -213 / -155); Harmonizer-ON 281 - R6DAG-s1 ON (0.780) >= 0 (paired).", "",
         "| seed | OFF mean | - R6DAG same seed | - NTV0 | offroad + zero-other | launched | ruler - NTV0 | ON mean, - R6DAG-s1 ON | failing |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|:--|", *rows, "", f"**Adopt: {'yes (both seeds pass)' if ok_all and len(ver) == 2 else 'no'}.**", "",
         f"### New zeros / partials on the {int(solved.sum())} scenes NTV0 scores 1.0", "", "| read | new zeros | new partials | classes |", "|:--|--:|--:|:--|"]
    for k, X in F.items():
        if k == "NTV0-ag":
            continue
        y, c = X.score.reindex(com)[solved.to_numpy()], X.fclass.reindex(com)[solved.to_numpy()]
        L.append(f"| {k} | {int((y <= 1e-4).sum())} | {int(((y > 1e-4) & (y < 1 - 1e-4)).sum())} | "
                 + ", ".join(f"{q} {n}" for q, n in c[y < 1 - 1e-4].value_counts().items()) + " |")
    pick = lambda X, h: next(((float(X.score[i]), str(X.fclass[i])) for i in X.index if h in i), (float("nan"), "?"))  # noqa: E731
    for title, hs in (("The six ruler-gap partial scenes (d436)", SIX), ("The two stuck curve cut-ins", CUTIN)):
        L += ["", f"### {title}: score / class", "", "| scene | " + " | ".join(F) + " |", "|:--|" + "--:|" * len(F)]
        L += [f"| {h} | " + " | ".join(f"{pick(X, h)[0]:.2f} {pick(X, h)[1]}" for X in F.values()) + " |" for h in hs]
    L += ["", "### Straight-road slow / rear-hit partials (score < 1, no zero) per read", "", "| read | partial scenes by class |", "|:--|:--|"]
    for k, X in F.items():
        y = X.score.reindex(com)
        m = (y > 1e-4) & (y < 1 - 1e-4)
        cl = X.fclass.reindex(com)[m]
        L.append(f"| {k} | " + ", ".join(f"{q} {n}" for q, n in cl.value_counts().items()) + " |")
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J["verdict"] = dict(adopt=bool(ok_all and len(ver) == 2), seeds=ver)
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("cmdcheck")
    p = sp.add_parser("teach")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk-list", default="")
    sp.add_parser("label")
    p = sp.add_parser("weights")
    p.add_argument("--out", required=True)
    p.add_argument("--pool", default=POOL_K)
    p.add_argument("--f2", type=int, default=0)
    p.add_argument("--smix", type=int, default=0)
    p = sp.add_parser("smoke")
    p.add_argument("--tag", required=True)
    p.add_argument("--ctrl", required=True)
    p.add_argument("--wfile", required=True)
    p.add_argument("--nav", type=int, default=25)
    p.add_argument("--cmd", type=int, default=1)
    p = sp.add_parser("serve")
    for k in ("--ref", "--ident", "--on"):
        p.add_argument(k, required=True)
    p = sp.add_parser("verdict")
    p.add_argument("--json", required=True)
    p.add_argument("--md", required=True)
    p.add_argument("--pcs", default="")
    p.add_argument("--read", nargs="+", required=True)
    a = ap.parse_args()
    dict(cmdcheck=cmd_cmdcheck, teach=cmd_teach, label=cmd_label, weights=cmd_weights, smoke=cmd_smoke, serve=cmd_serve, verdict=cmd_verdict)[a.cmd](a)


if __name__ == "__main__":
    main()
