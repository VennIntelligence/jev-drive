#!/usr/bin/env python3
"""ROUND9 DAG-LANE (plans/2026-10-22-dag-lane.md): R6DAG's recipe with (a) the heading command (rm_train --cmd-rule heading, serve JEV_CMD=heading)
and (b) the DAgger teacher's human-lane rule (round6 dag.py teacher b3L), one arm R9DL, 2 seeds. Lane script: dl.sh.

  dl.py teach [--scenes F] --workers N   (AlpaSim venv, DAG_TV=b3L) per iteration-1 DAgger scene (round6/dag/rows/PAID): the b3 and b3L labels of
                                         every decision at the student's served arc lengths, their 2 / 4 s points' lateral distance to the human's
                                         recorded path, the lane rule's weight at the student's station -> $O/teach/<scene>.npz
  dl.py tcheck --scenes F                stage-1 checklist on the listed scenes (files, finite labels, b3 identity, b3L = b3 without a qualifying pose)
  dl.py label                            G-L0 / G-L1 / G-L4 on on1_PAID's rows; pool on1_JTPHDL (on1_JTPHD with the pd labels := b3L, hard links)
                                         -> results rows.{md,json}; exit 1 = a gate failed
  dl.py tcmp --b3 D --b3l D              G-L2 (T1 on human states) and G-L3 (T2, in canvas) from dag.py t1 / t2 run with DAG_TV=b3 and b3L
  dl.py smoke --tag X --ctrl Y           G-S on the two 300-step runs
  dl.py cmdagree --read D ..             G-C after the reads: served command = rmem.cmd_heading(float16 fed memory) on every inference
  dl.py verdict --read K=F ..            per-class losses, target scenes, the registered adoption rule -> results verdict.{md,json}
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
                str(REPO / "experiments/round7/scripts"), str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/round3/scripts"),
                str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/op_probe/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/dag_lane"
O1 = DATA / "runs/round6/dag"
CR = DATA / "runs/op_parity/cache"
RES = Path(os.environ.get("DL_RES", str(O / "results")))          # box results outside the repo tree; copied to experiments/round9/results/dag_lane
BASE, POOL, SRC_D = "on1_JTPHD", "on1_JTPHDL", "on1_PAID"
A_LAT, LANE_HALF, MOVE = 4.0, 1.75, (0.3, 1.0)
SPD = [2, 10, 15, 20, 25]
SPD_LAB = ["2-10", "10-15", "15-20", "20-25", ">=25"]
TARGET = ("26004ca8", "4cb05d65", "7dc59d0d", "0e05cb0c")          # route_check A: route in the lane next to the human's
LANEKEEP = ("cor_drift", "cor_lane_change", "off_straight", "off_junc_straight")
TURN_IN = ("off_junc_inside", "off_curve_inside")
V_HI = 15.0


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))


def seg_dist(H, X):
    """Distance (m,) of points X (m, 2) to the polyline H (n, 2); inf beyond its ends (projection parameter outside [0, 1] at the end segments)."""
    a, b = H[:-1], H[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.full(len(X), np.inf)
    for i, x in enumerate(X):
        t = ((x - a) * ab).sum(1) / L2
        tc = np.clip(t, 0, 1)
        d = np.hypot(*(a + tc[:, None] * ab - x).T)
        j = int(np.argmin(d))
        if not ((j == 0 and t[0] < 0) or (j == len(t) - 1 and t[-1] > 1)):
            out[i] = d[j]
    return out


# ---------------------------------------------------------------- teach (AlpaSim venv, DAG_TV=b3L)
def _teach_one(job):
    import dag as D
    assert D.TEACHER_V == "b3L", "dl.py teach: DAG_TV=b3L"
    scene, cfg, rowf, out = job
    with np.load(rowf) as z:
        Z = {k: z[k] for k in ("k", "anchor", "poses", "v0")}
    ds = D.load_ds(scene, cfg)
    (ref, refL), rinfo = D.route_ref(ds)
    P, yaw, ts = D.traj_of(ds)
    ref3, _ = D.human_ref(ref, P, yaw, ts)                                      # b3: the same reference without the lane rule
    H = P[:, :2]
    n = len(Z["k"])
    f3, fL = np.full((n, 8, 3), np.nan, np.float32), np.full((n, 8, 3), np.nan, np.float32)
    h3, hL = np.full((n, 2), np.nan, np.float32), np.full((n, 2), np.nan, np.float32)
    w0, ds0 = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    for i in range(n):
        A = Z["anchor"][i].astype(np.float64)
        a = D.arcs_of(Z["poses"][i])
        v0 = float(Z["v0"][i])
        for R_, F_, Hd in ((ref3, f3, h3), (refL, fL, hL)):
            path, info = D.teacher_path(R_, A, v0, a[-1])
            if path is None:
                continue
            fut, pts, _ = D.place(path, a, A)
            F_[i] = fut
            Hd[i] = seg_dist(H, pts[[3, 7]])
            if R_ is refL and refL.w is not None:
                w0[i] = float(np.interp(info["s0"], ref.s, refL.w))          # w lives on the lane-centre reference's station grid
        ds0[i] = seg_dist(H, A[None, :2])[0]
    tmp = Path(out) / f"{scene}.tmp.npz"
    np.savez(tmp, k=Z["k"], fut3=f3, futL=fL, dh3=h3, dhL=hL, w=w0, dh0=ds0, v0=Z["v0"], served=Z["poses"], route=json.dumps(rinfo),
             q_poses=int(rinfo.get("lane_q_poses", 0)), version="b3L")
    os.replace(tmp, Path(out) / f"{scene}.npz")
    return dict(scene=scene, n=n, q_poses=int(rinfo.get("lane_q_poses", 0)), lane_found=rinfo.get("lane_found"))


def cmd_teach(a):
    import dag as D
    from jevdrive import par
    out = O / "teach"
    out.mkdir(parents=True, exist_ok=True)
    cfg = D.scene_cfgs(str(O1 / "roll/runs/*/sim/generated-user-config-0.yaml"))
    files = sorted((O1 / "rows/PAID").glob("clipgt-*.npz"))
    want = set(Path(a.scenes).read_text().split()) if a.scenes else None
    jobs = [(f.stem, cfg[f.stem], str(f), str(out)) for f in files if f.stem in cfg and (want is None or f.stem in want)]
    res = par.pmap(_teach_one, jobs, workers=a.workers or None, desc="teach b3 / b3L", skip=lambda j: (out / f"{j[0]}.npz").exists())
    res.raise_if_failed()
    v = [x for x in res.values if x]
    rec = dict(scenes=len(jobs), done=len(v), rows=sum(x["n"] for x in v), scenes_qualifying=sum(x["q_poses"] > 0 for x in v),
               no_cfg=[f.stem for f in files if f.stem not in cfg])
    print(json.dumps(rec))


def load_teach(names):
    """on1_PAID row names -> dict of per-row arrays from $O/teach (found False where the scene / decision is missing)."""
    sc = np.array([x.split(":")[2] for x in names])
    kk = np.array([int(x.split(":")[3]) for x in names])
    n = len(names)
    Z = {k: np.full((n, 8, 3), np.nan, np.float32) for k in ("fut3", "futL")} | {k: np.full((n, 2), np.nan, np.float32) for k in ("dh3", "dhL")}
    Z |= {k: np.full(n, np.nan, np.float32) for k in ("w", "dh0", "v0t")} | dict(found=np.zeros(n, bool), q_scene=np.zeros(n, bool))
    for s in np.unique(sc):
        j = np.flatnonzero(sc == s)
        f = O / "teach" / f"{s}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            assert str(z["version"]) == "b3L"
            pos = {int(k): i for i, k in enumerate(z["k"])}
            hit = np.array([int(k) in pos for k in kk[j]])
            i = np.array([pos[int(k)] for k in kk[j][hit]], int)
            for key in ("fut3", "futL", "dh3", "dhL", "w", "dh0"):
                Z[key][j[hit]] = z[key][i]
            Z["v0t"][j[hit]] = z["v0"][i]
            Z["found"][j[hit]] = True
            Z["q_scene"][j] = int(z["q_poses"]) > 0
    return Z, sc


def cmd_tcheck(a):
    want = Path(a.scenes).read_text().split()
    P = np.load(CR / SRC_D / "tab.npz")
    nm = P["names"].astype(str)
    sc = np.array([x.split(":")[2] for x in nm])
    m = np.isin(sc, want)
    Z, _ = load_teach(nm[m])
    X = np.load(CR / SRC_D / "dagger.npz")
    draw = X["draw"][m].astype(bool)
    ok = draw & Z["found"] & np.isfinite(Z["fut3"]).reshape(m.sum(), -1).all(1)
    d3 = float(np.abs(Z["fut3"][ok, :, :2] - P["fut"][m][ok, :, :2]).max()) if ok.any() else float("inf")
    nq = draw & Z["found"] & ~Z["q_scene"]
    idn = bool(np.array_equal(Z["fut3"][nq], Z["futL"][nq], equal_nan=True))
    files = sum((O / "teach" / f"{s}.npz").exists() for s in want)
    ck = dict(scenes=len(want), files=files, drawable=int(draw.sum()), found=int((draw & Z["found"]).sum()), b3_max_abs=d3,
              finite_L=float(np.isfinite(Z["futL"][draw & Z["found"]]).all(axis=(1, 2)).mean()) if (draw & Z["found"]).any() else 0.0,
              b3L_eq_b3_no_qualifying=idn, qualifying_scenes=int(len(np.unique(sc[m][Z["q_scene"]]))))
    ok_ = files == len(want) and d3 <= 1e-3 and idn and ck["finite_L"] >= 0.99
    dump(RES / "teach_stage1.json", ck | dict(passed=bool(ok_)))
    print(json.dumps(ck | dict(passed=bool(ok_))))
    sys.exit(0 if ok_ else 1)


# ---------------------------------------------------------------- label, gates G-L0 / G-L1 / G-L4, pool
def cmd_label(a):
    from dag import alat_pos
    from joint import link_pool
    import joint as J
    J.BASE = BASE
    tb = dict(np.load(CR / BASE / "tab.npz"))
    pool = tb["pool"].astype(str)
    j0 = np.flatnonzero(pool == "pd")
    assert np.array_equal(j0, np.arange(j0[0], len(pool))), "pd rows are not the tail of on1_JTPHD"
    P = np.load(CR / SRC_D / "tab.npz")
    X = np.load(CR / SRC_D / "dagger.npz")
    nm = P["names"].astype(str)
    assert all(x == "on:pd:" + y[3:] for x, y in zip(tb["names"][j0].astype(str), nm)), "on1_JTPHD pd rows != on1_PAID rows"
    assert np.array_equal(tb["fut"][j0], P["fut"]), "on1_JTPHD pd labels != on1_PAID's"
    Z, sc = load_teach(nm)
    n = len(nm)
    draw = X["draw"].astype(bool)
    v0 = P["speed"].astype(np.float64)
    fb3 = P["fut"].astype(np.float64)
    # G-L0: b3 reproduced on the drawable rows; b3L = b3 bit for bit on the scenes without a qualifying pose
    m = draw & Z["found"] & np.isfinite(Z["fut3"]).reshape(n, -1).all(1)
    g0 = float(np.abs(Z["fut3"][m, :, :2] - fb3[m, :, :2]).max()) if m.any() else float("inf")
    cov = float(m.sum() / max(draw.sum(), 1))
    nq = Z["found"] & ~Z["q_scene"]
    idn = bool(np.array_equal(Z["fut3"][nq], Z["futL"][nq], equal_nan=True))
    # G-L1: rows the b3 teacher labels into a lane the human never used, and what b3L changes
    okL = m & np.isfinite(Z["futL"]).reshape(n, -1).all(1)
    mv = np.where(okL, np.hypot(*(Z["futL"][:, 7, :2] - Z["fut3"][:, 7, :2]).T), 0.0)
    off3 = m & (Z["dh3"][:, 1] > LANE_HALF) & np.isfinite(Z["dh3"][:, 1])
    offL = okL & (Z["dhL"][:, 1] > LANE_HALF) & np.isfinite(Z["dhL"][:, 1])
    bins = np.digitize(v0, SPD) - 1
    tab = []
    for b, lab in enumerate(SPD_LAB):
        s_ = m & (bins == b)
        tab.append(dict(speed=lab, drawable=int(s_.sum()), b3_off_human_4s=int((s_ & off3).sum()), b3_off_human_2s=int((s_ & (Z["dh3"][:, 0] > LANE_HALF)).sum()),
                        b3L_off_human_4s=int((s_ & offL).sum()), moved_03=int((s_ & (mv > MOVE[0])).sum()), moved_1=int((s_ & (mv > MOVE[1])).sum()),
                        target=int((s_ & off3 & (mv > MOVE[1])).sum()), lane_rule_w05=int((s_ & (np.nan_to_num(Z["w"]) > 0.5)).sum())))
    hi = m & (v0 >= 15)
    tgt = hi & off3 & (mv > MOVE[1])
    fixed = tgt & okL & (Z["dhL"][:, 1] <= LANE_HALF)
    g1 = dict(target_rows=int(tgt.sum()), line_rows=300, fixed_share=float(fixed.sum() / max(tgt.sum(), 1)), line_share=0.9)
    # G-L4: T4 on the final labels; changed rows over the line revert to b3 (the drawable set is unchanged)
    futL = np.where(okL[:, None, None], Z["futL"], fb3)
    alL = alat_pos(np.nan_to_num(futL))
    al3 = alat_pos(fb3)
    rev = draw & (mv > 1e-6) & (alL > A_LAT)
    fin = np.where((draw & okL & ~rev)[:, None, None], futL, fb3).astype(np.float32)
    alF = alat_pos(fin.astype(np.float64))
    t4 = float(np.percentile(alF[draw], 99))
    over = draw & (alF > A_LAT)
    changed = draw & (np.abs(fin - P["fut"]).reshape(n, -1).max(1) > 0)
    gates = dict(G_L0=bool(g0 <= 1e-3 and cov >= 0.99 and idn), G_L1=bool(g1["target_rows"] >= g1["line_rows"] and g1["fixed_share"] >= g1["line_share"]),
                 G_L4=bool(t4 <= A_LAT or over.sum() <= 0.03 * draw.sum()))
    sq = np.unique(sc[Z["q_scene"]])
    rec = dict(rows=n, drawable=int(draw.sum()), G_L0=dict(max_abs_m=g0, coverage=cov, b3L_eq_b3_no_qualifying=idn), scenes_qualifying=int(len(sq)),
               scenes=int(len(np.unique(sc))), by_speed=tab, G_L1=g1, changed_rows=int(changed.sum()), changed_hi=int((changed & (v0 >= 15)).sum()),
               G_L4=dict(p99_final=t4, p99_b3=float(np.percentile(al3[draw], 99)), over=int(over.sum()), over_share=float(over.sum() / max(draw.sum(), 1)),
                         reverted=int(rev.sum())), student_off_human_t0=int((m & (Z["dh0"] > LANE_HALF)).sum()), gates=gates, passed=bool(all(gates.values())))
    dump(RES / "rows.json", rec)
    L = ["# DAG-LANE rows: teacher b3L on the iteration-1 DAgger states (dl.py label)", "",
         f"on1_PAID: {n} rows, {draw.sum()} drawable (R6DAG's set, unchanged); teacher files for {cov:.1%} of the drawable rows; "
         f"{len(sq)} of {rec['scenes']} scenes have a qualifying stretch (human in the route lane's parallel neighbour).", "",
         "Off-human = the label's 4 s (2 s) point lies more than 1.75 m from the human's recorded path. Moved = |b3L - b3| at 4 s. Target = off-human "
         "under b3 and moved > 1 m.", "",
         "| speed (m/s) | drawable | b3 off-human 4 s | 2 s | b3L off-human 4 s | moved > 0.3 m | > 1 m | target | lane-rule weight > 0.5 |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in tab:
        L.append(f"| {r['speed']} | {r['drawable']} | {r['b3_off_human_4s']} | {r['b3_off_human_2s']} | {r['b3L_off_human_4s']} | {r['moved_03']} | "
                 f"{r['moved_1']} | {r['target']} | {r['lane_rule_w05']} |")
    L += ["", f"Students already more than 1.75 m off the human's path at t0: {rec['student_off_human_t0']} drawable rows.", "",
          f"G-L0: b3 reproduced max |dxy| {g0:.2e} m over {cov:.1%} of the drawable rows; b3L = b3 bit for bit on the scenes without a qualifying pose: "
          f"{idn} -> **{'PASS' if gates['G_L0'] else 'FAIL'}**.",
          f"G-L1: target rows at >= 15 m/s {g1['target_rows']} (line >= 300); b3L brings {g1['fixed_share']:.1%} of them within 1.75 m of the human "
          f"(line >= 90 %) -> **{'PASS' if gates['G_L1'] else 'FAIL'}**.",
          f"G-L4: position-estimator p99 {t4:.2f} m/s^2 (b3 {rec['G_L4']['p99_b3']:.2f}), over the line {over.sum()} ({rec['G_L4']['over_share']:.2%}); "
          f"changed rows over the line reverted to b3: {rev.sum()} -> **{'PASS' if gates['G_L4'] else 'FAIL'}**.",
          f"Labels changed against R6DAG's: {changed.sum()} drawable rows ({rec['changed_hi']} at >= 15 m/s).", "",
          f"**{'PASS' if rec['passed'] else 'FAIL'}.**"]
    (RES / "rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    np.savez(O / "pd_b3L.npz", names=nm, fut=fin, changed=changed, reverted=rev, target=tgt, moved=mv.astype(np.float32))
    if not rec["passed"]:
        sys.exit(1)
    tk = dict(tb)
    tk["fut"] = tb["fut"].copy()
    tk["fut"][j0] = fin
    link_pool(POOL, tk)
    print(f"pool {POOL}: on1_JTPHD with {changed.sum()} pd labels := b3L")


# ---------------------------------------------------------------- T1 / T2 comparison (G-L2, G-L3)
def cmd_tcmp(a):
    import pandas as pd
    A3, AL = pd.read_csv(Path(a.b3) / "t1_states.csv.gz"), pd.read_csv(Path(a.b3l) / "t1_states.csv.gz")
    k = ["scene", "t"]
    M = A3.merge(AL, on=k, suffixes=("_3", "_L"))
    M = M[M.e2h_3.notna() & M.e2h_L.notna()]
    e = lambda c: float(np.median(np.abs(M[c])))  # noqa: E731
    dis = M[np.abs(M.e4h_3) > LANE_HALF]
    t1 = dict(states=len(M), b3=dict(e2=e("e2h_3"), e4=e("e4h_3")), b3L=dict(e2=e("e2h_L"), e4=e("e4h_L")),
              disagree=dict(states=len(dis), b3_e4=float(np.median(np.abs(dis.e4h_3))) if len(dis) else None,
                            b3L_e4=float(np.median(np.abs(dis.e4h_L))) if len(dis) else None))
    g2 = bool(t1["b3L"]["e2"] <= t1["b3"]["e2"] + 0.01 and t1["b3L"]["e4"] <= t1["b3"]["e4"] + 0.01 and (not len(dis) or t1["disagree"]["b3L_e4"] <= 0.5))
    T3, TL = json.loads((Path(a.b3) / "t2.json").read_text())["in_canvas"], json.loads((Path(a.b3l) / "t2.json").read_text())["in_canvas"]
    g3 = bool(TL["share"] >= 0.9 and TL["teacher_inside"] >= T3["teacher_inside"] - 1)
    rec = dict(T1=t1, T2=dict(b3=T3, b3L=TL), gates=dict(G_L2=g2, G_L3=g3), passed=bool(g2 and g3))
    dump(RES / "teacher_gates.json", rec)
    L = ["# DAG-LANE teacher gates on human states and offroad events (dl.py tcmp; dag.py t1 / t2 with DAG_TV=b3 and b3L)", "",
         f"T1 human states (both teachers labelled): {len(M)}. Median |error| at 2 / 4 s: b3 {t1['b3']['e2']:.3f} / {t1['b3']['e4']:.3f} m, "
         f"b3L {t1['b3L']['e2']:.3f} / {t1['b3L']['e4']:.3f} m. States where b3 misses the human by > 1.75 m at 4 s: {len(dis)} "
         f"(b3 median {t1['disagree']['b3_e4']}, b3L {t1['disagree']['b3L_e4']} m).",
         f"G-L2 (b3L no worse than b3 + 0.01 m; on those states b3L <= 0.5 m) -> **{'PASS' if g2 else 'FAIL'}**.", "",
         f"T2 in canvas: b3 teacher inside {T3['teacher_inside']} / {T3['events']} ({T3['share']:.0%}); b3L {TL['teacher_inside']} / {TL['events']} "
         f"({TL['share']:.0%}). G-L3 (b3L >= 90 % and >= b3 - 1) -> **{'PASS' if g3 else 'FAIL'}**."]
    (RES / "teacher_gates.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if rec["passed"] else 1)


# ---------------------------------------------------------------- smoke (G-S)
def cmd_smoke(a):
    from joint import events
    dj, J_ = events(a.tag)
    dc, C = events(a.ctrl)
    ratio = lambda S: sum(S["loss/grad/pool_hinge"].values()) / sum(S["loss/grad/imit"].values())  # noqa: E731
    rj, rc = ratio(J_), ratio(C)
    log = (dj / "log.txt").read_text()
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", log)
    gj = (json.loads((dj / "DONE").read_text()).get("draw_shares") or {}).get("group_share", {}) if (dj / "DONE").exists() else {}
    gc = (json.loads((dc / "DONE").read_text()).get("draw_shares") or {}).get("group_share", {}) if (dc / "DONE").exists() else {}
    ade = J_["dev/ade"]
    last = max(ade)
    cmdr = re.search(r"--cmd-rule \(round7 joint arm\): (\{.*\})", log)
    ok = dict(done=bool(gj) and bool(gc), finite=all(np.isfinite(list(v.values())).all() for v in J_.values()),
              batch=bool(b) and b.groups() == ("51", "13", "72"), shares=bool(gj) and gj == gc, grad=0.7 <= rj / rc <= 1.3,
              dev=ade[last] <= 0.5362 + 0.03, cmd=cmdr is not None)
    res = dict(tag=a.tag, ctrl=a.ctrl, grad_ratio=dict(arm=rj, control=rc, rel=rj / rc), dev_ade=ade[last], cmd_rule=json.loads(cmdr.group(1)) if cmdr else None,
               ok=ok, passed=bool(all(ok.values())))
    dump(RES / "smoke.json", res)
    print(json.dumps(res, indent=1, default=float))
    sys.exit(0 if res["passed"] else 2)


# ---------------------------------------------------------------- serve agreement after the reads (G-C)
def cmd_cmdagree(a):
    import rmem as RM
    out, ok_all = {}, True
    for d in a.read:
        n = ag = 0
        for f in glob.glob(f"{d}/runs/*/sim/driver-logs/drive*.jsonl"):
            for line in open(f):
                if '"rmem"' not in line:
                    continue
                r = json.loads(line)
                if r.get("kind") != "drive" or "rmem" not in r:
                    continue
                n += 1
                ag += int(np.argmax(RM.cmd_heading(np.asarray(r["rmem"], np.float16)))) == r["cmd"]
        out[Path(d).name] = dict(inferences=n, agree=ag / max(n, 1))
        ok_all &= n > 0 and ag / max(n, 1) >= 0.999
    dump(RES / f"cmdagree_{a.label}.json", dict(reads=out, passed=bool(ok_all)))
    print(json.dumps(out))
    sys.exit(0 if ok_all else 1)


# ---------------------------------------------------------------- verdict
def classes(F):
    """facts (pandas) -> per-class zeros / loss, target scenes."""
    z = F.score == 0
    pr = F.primary.astype(str)
    hi = F.v_hand >= V_HI
    lk, ti, co = pr.isin(LANEKEEP), pr.isin(TURN_IN), pr.str.startswith("coll_")
    out = dict(n=len(F), mean=float(F.score.mean()), zeros=int(z.sum()), lanekeep_zeros=int((z & lk).sum()), lanekeep_hi_zeros=int((z & lk & hi).sum()),
               other_zeros=int((z & ~lk).sum()), turn_inside_zeros=int((z & ti).sum()), coll_zeros=int((z & co).sum()),
               straight_at_turn_zeros=int((z & (pr == "cor_straight_at_turn")).sum()),
               loss=dict(lanekeep=float(F.loss[lk].sum()), lanekeep_hi=float(F.loss[lk & hi].sum()), turn_inside=float(F.loss[ti].sum()),
                         coll=float(F.loss[co].sum()), **{c: float(F.loss[pr == c].sum()) for c in LANEKEEP}),
               target={t: float(F.score[F.scene.str.contains(t)].mean()) if F.scene.str.contains(t).any() else None for t in TARGET})
    return out


def cmd_verdict(a):
    import pandas as pd
    R = {k: pd.read_csv(f) for k, f in (x.split("=", 1) for x in a.read)}
    common = set.intersection(*[set(F.scene) for F in R.values()])
    C = {k: classes(F[F.scene.isin(common)]) for k, F in R.items()}
    ref = [f"R6DAG-s{s}" for s in (0, 1, 2)]
    avg = lambda arm, key: (C[f"{arm}-ag"][key] + C[f"{arm}-agH"][key]) / 2  # noqa: E731
    r6 = {k: float(np.mean([avg(x, k) for x in ref])) for k in ("mean", "lanekeep_hi_zeros", "other_zeros", "lanekeep_zeros", "zeros")}
    rule = {}
    for s in (0, 1):
        x = f"R9DL-s{s}"
        r1 = avg(x, "mean") >= r6["mean"]
        r2 = avg(x, "lanekeep_hi_zeros") < r6["lanekeep_hi_zeros"]
        r3 = avg(x, "other_zeros") <= r6["other_zeros"] + 2
        rule[x] = dict(mean_offon=avg(x, "mean"), lanekeep_hi=avg(x, "lanekeep_hi_zeros"), other=avg(x, "other_zeros"), r1=bool(r1), r2=bool(r2), r3=bool(r3),
                       passed=bool(r1 and r2 and r3))
    adopt = all(v["passed"] for v in rule.values())
    rec = dict(common_scenes=len(common), classes=C, r6dag_3seed=r6, rule=rule, adopt=bool(adopt))
    dump(RES / "verdict.json", rec)
    keys = sorted(C)
    L = ["# DAG-LANE verdict (dl.py verdict)", "", f"Common scenes: {len(common)}. Lane-keeping = {', '.join(LANEKEEP)}; high speed = the human's start speed "
         f">= {V_HI:.0f} m/s; turn inside = {', '.join(TURN_IN)}; other zeros = every zero outside lane-keeping.", "",
         "| read | mean | zeros | lane-keep zeros | high-speed | other zeros | turn inside | collisions | straight at turn | lane-keep loss | turn-inside loss | "
         + " | ".join(t for t in TARGET) + " |", "|:--|" + "--:|" * (11 + len(TARGET))]
    for k in keys:
        c = C[k]
        L.append(f"| {k} | {c['mean']:.3f} | {c['zeros']} | {c['lanekeep_zeros']} | {c['lanekeep_hi_zeros']} | {c['other_zeros']} | {c['turn_inside_zeros']} | "
                 f"{c['coll_zeros']} | {c['straight_at_turn_zeros']} | {c['loss']['lanekeep']:.1f} | {c['loss']['turn_inside']:.1f} | "
                 + " | ".join("-" if c["target"][t] is None else f"{c['target'][t]:.2f}" for t in TARGET) + " |")
    L += ["", f"R6DAG 3-seed means of (OFF + ON) / 2: mean {r6['mean']:.4f}, high-speed lane-keeping zeros {r6['lanekeep_hi_zeros']:.2f}, other zeros "
          f"{r6['other_zeros']:.2f}.", ""]
    for x, v in rule.items():
        L.append(f"- {x}: (OFF + ON) / 2 {v['mean_offon']:.4f} ({'pass' if v['r1'] else 'fail'}); high-speed lane-keeping zeros {v['lanekeep_hi']:.1f} "
                 f"({'pass' if v['r2'] else 'fail'}); other zeros {v['other']:.1f} ({'pass' if v['r3'] else 'fail'}).")
    L += ["", f"**Adoption rule (plan section 6): {'ADOPT' if adopt else 'NOT ADOPTED'}.**"]
    (RES / "verdict.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("teach"); p.add_argument("--scenes", default=""); p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("tcheck"); p.add_argument("--scenes", required=True)
    sub.add_parser("label")
    p = sub.add_parser("tcmp"); p.add_argument("--b3", required=True); p.add_argument("--b3l", required=True)
    p = sub.add_parser("smoke"); p.add_argument("--tag", required=True); p.add_argument("--ctrl", required=True)
    p = sub.add_parser("cmdagree"); p.add_argument("--read", nargs="+", required=True); p.add_argument("--label", required=True)
    p = sub.add_parser("verdict"); p.add_argument("--read", nargs="+", required=True)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    dict(teach=cmd_teach, tcheck=cmd_tcheck, label=cmd_label, tcmp=cmd_tcmp, smoke=cmd_smoke, cmdagree=cmd_cmdagree, verdict=cmd_verdict)[a.cmd](a)
