#!/usr/bin/env python3
"""NUPLAN-DAG Part 2 (plans/2026-10-22-nuplan-dag.md): the PAI DAgger map teacher (round6 dag.py, teacher b3 = d434's) on nuPlan rows of the
student B2 (STANDFMXH-L18-s1) rolled natively on alpasim_nuplan/onpol-train; gates T1-T4, the appended pool, draw files, smoke. nd.sh runs it.

Teacher (dag.py's functions unchanged, imported): the route = AlpaSim's MAP route generator (route_start_offset 40 m, as the rollouts' config)
rebuilt on the scene's logged rig trajectory over trajdata's nuPlan VectorMap (nuPlan lanes and lane connectors): for every logged pose the
lanes within 4 m and pi / 8 of its heading, previous lane kept / unique successor / most upcoming poses -> the lane / lane connector sequence
the human drove; a non-successor lane switch blended over clip(3 s x v, 15, 60) m; past the log the generator's own 80 m extension.
Reference resampled 0.25 m, Gaussian sigma 3 m; + the human's own lateral offset from it at the matched station (sigma 10 m, saturated
+-0.5 m). Path from the student pose: Frenet quintic back to the reference over clip(3 s x v0, 6, 80) m. No label when |d0| > 5 m,
|psi0| > 60 deg. Label = the 8 path points at the arc lengths of the student's own served plan (b3 / d434: longitudinal = the student).

  (AlpaSim venv)  nd_teach.py lists --out O                    probe c000 (6 scenes by class) + chunks c001.. of the 1031 train scenes
  (AlpaSim venv)  nd_teach.py t1 --cfg-glob G --out-dir R      T1 on HUMAN states of the scenes the configs list (dag._t1_one)
  (AlpaSim venv)  nd_teach.py teach --name N --out F           teacher labels for every row of runs/onpol_nuplan/rows/<N> (+ frame check)
  ($TR)           nd_teach.py label --name N --teach F --res R [--write]   T2 / T4 on the pool's drivable raster; --write: tab.fut and dagger.npz
  ($TR)           nd_teach.py plots --name N --teach F --out D  BEV panels of the probe rows
  ($TR)           nd_teach.py chk1 --res R --root O           the probe gates -> exit 1 = stop
  ($TR)           nd_teach.py append --base on1_JTX --add on1_NDAG --out on1_JTXD   zero-copy append (base rows bit-equal, add rows mode both)
  ($TR)           nd_teach.py weights --pool on1_JTXD --base JTX00 --k-on 70 --k-dag 8 --out NDAG   (k-dag 0: the identity file)
  ($TR)           nd_teach.py smoke --tag X --wfile NDAG --k-dag 8 --res R
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("DAG_SIG", "3"), os.environ.setdefault("DAG_TM", "3"), os.environ.setdefault("DAG_HSIG", "10"), os.environ.setdefault("DAG_HCLIP", "0.5")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round6/scripts")]
import dag as DG  # noqa: E402  (teacher functions; constants checked below)

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = DATA / "runs/op_parity/cache"
ROWS = DATA / "runs/onpol_nuplan/rows"
FEAT = DATA / "runs/route_mem/feat"
WD = DATA / "runs/nuplan_dag/w"
SPLIT = REPO / "experiments/onpol_nuplan/results/split"
V_ROW, K_MIN = 2.0, 2                     # drawable: fed speed >= 2 m/s, policy decisions only (k >= 2; k = 0, 1 are log-replay states)
A_LAT = 4.0                               # T4: label lateral acceleration (position estimator) above this -> weight 0
EGO_L, EGO_W, EGO_RA = 5.176, 2.297, 1.461   # nuPlan ego box (Pacifica): length, width, rear axle -> centre
X0, Y0, RES, NH, NW = -8.0, -24.0, 0.5, 128, 96   # the pool rasters (opb_labels): x in [-8, 56), y in [-24, 24)
HOLD_PCT = 15
assert (DG.SIG, DG.T_MERGE, DG.H_SIG, DG.H_CLIP) == (3.0, 3.0, 10.0, 0.5), "teacher constants != d434's b3"
TEACHER = "nd-b3"


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


def scene_cfgs(patterns):
    out = {}
    for f in [f for p in patterns.split(",") for f in sorted(glob.glob(p))]:
        for s in re.findall(r"scene_id: (\S+)", open(f).read()):
            out.setdefault(s, f)
    return out


# ---------------------------------------------------------------- lists
def cmd_lists(a):
    """c000 = 6 probe scenes (right sharp turn x 2, left turn, stop, launch, other: from the stored L18RB0 rows' logged futures), the rest of
    the 1031 train scenes in --n chunks (interleaved, written once)."""
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    tr = [s for s in (SPLIT / "train.txt").read_text().split()]
    held = set((SPLIT / "held.txt").read_text().split())
    assert len(tr) == 1031 and not held & set(tr), "train split"
    if (O / "c000.txt").exists():
        print("lists exist"); return
    z = np.load(ROWS / "L18RB0/meta.npz")
    sc, lf, vl = z["scene"].astype(str), z["logfut"].astype(np.float64), z["v_log"]
    by = defaultdict(lambda: dict(turn=0.0, vmin=99.0, vmax=0.0))
    for s, f, v in zip(sc, lf, vl):
        d = float(np.degrees(DG.wrap(f[-1, 2] - f[0, 2])))
        b = by[s]
        b["turn"] = d if abs(d) > abs(b["turn"]) else b["turn"]
        b["vmin"], b["vmax"] = min(b["vmin"], v), max(b["vmax"], v)
    rng = np.random.default_rng(22)
    pick = []
    for cond in (lambda b: b["turn"] <= -45, lambda b: b["turn"] <= -45, lambda b: b["turn"] >= 45,
                 lambda b: b["vmax"] > 5 and b["vmin"] < 0.3 and abs(b["turn"]) < 20, lambda b: b["vmax"] < 0.3 or (b["vmin"] < 0.3 and b["vmax"] > 3),
                 lambda b: abs(b["turn"]) < 10 and b["vmin"] > 5):
        c = sorted(s for s in tr if s in by and cond(by[s]) and s not in pick)
        pick.append(str(rng.choice(c)))
    rest = [s for s in tr if s not in pick]
    (O / "c000.txt").write_text("\n".join(pick) + "\n")
    for i in range(a.n):
        (O / f"c{i + 1:03d}.txt").write_text("\n".join(rest[i::a.n]) + "\n")
    dump(O / "lists.json", dict(probe={s: by[s] for s in pick}, chunks=a.n, rest=len(rest)))
    print(json.dumps(dict(probe=pick, rest=len(rest), chunks=a.n)))


# ---------------------------------------------------------------- T1 (human states)
def cmd_t1(a):
    from jevdrive import par
    import pandas as pd
    cfg = scene_cfgs(a.cfg_glob)
    tr = set((SPLIT / "train.txt").read_text().split())
    jobs = sorted((s, c) for s, c in cfg.items() if s in tr)[:a.limit or None]
    res = par.pmap(DG._t1_one, jobs, workers=a.workers or None, desc="t1 scenes")
    st = [x for v in res.values if v for x in v["states"]]
    ri = [v["route"] for v in res.values if v]
    D = pd.DataFrame(st)
    R = Path(a.out_dir)
    R.mkdir(parents=True, exist_ok=True)
    D.to_csv(R / "t1_states.csv.gz", index=False)
    tab = {}
    for c in ("all", "all but lane-change", "turn", "other", "lane-change"):
        m = D[(D.cls == c) if c not in ("all", "all but lane-change") else (D.cls != "lane-change") if c != "all" else np.ones(len(D), bool)]
        m = m[m.e2h.notna()] if "e2h" in m else m.iloc[:0]
        if len(m):
            tab[c] = dict(states=len(m), e2h=float(m.e2h.abs().median()), e4h=float(m.e4h.abs().median()), e2h_p90=float(m.e2h.abs().quantile(.9)),
                          e4h_p90=float(m.e4h.abs().quantile(.9)), e2=float(m.e2.abs().median()), e4=float(m.e4.abs().median()),
                          d0=float(m.d0.abs().median()), d0_signed=float(m.d0.median()))
    t = tab.get("all but lane-change", {})
    passed = bool(t and t["e2h"] <= 0.30 and t["e4h"] <= 0.50)
    turn_inspect = bool(tab.get("turn", {}).get("e2h", 0) > 0.6 or tab.get("turn", {}).get("e4h", 0) > 1.0)
    rec = dict(scenes=len(jobs), scenes_ok=len(ri), failed={jobs[i][0]: e.strip().splitlines()[-1][:200] for i, e in res.errors.items()},
               states=len(D), why=dict(Counter(D.why.fillna("").replace("", "ok"))), table=tab, passed=passed, turn_inspect=turn_inspect,
               switch_scenes=sum(r["switches"] > 0 for r in ri), h_clip_share_mean=float(np.mean([r["h_clip_share"] for r in ri])) if ri else None,
               h_offset_med=float(np.nanmedian([r["h_offset_med"] for r in ri])) if ri else None)
    dump(R / "t1.json", rec)
    L = ["# T1: the teacher on HUMAN states (nd_teach.py t1; dag._t1_one unchanged)", "",
         f"{rec['scenes_ok']} / {rec['scenes']} train scenes; logged states every 0.5 s with v >= 2 m/s and a full 4 s future; teacher from the human "
         "pose, points at the human's own arc lengths; error = the human's point minus the teacher's along the teacher's left normal.", "",
         "| states | n | teacher (lane centre + human offset): median abs at 2 s | 4 s | p90 2 s | p90 4 s | lane centre only: 2 s | 4 s | median abs d0 | signed d0 |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for c, v in tab.items():
        L.append(f"| {c} | {v['states']} | {v['e2h']:.3f} | {v['e4h']:.3f} | {v['e2h_p90']:.2f} | {v['e4h_p90']:.2f} | {v['e2']:.3f} | {v['e4']:.3f} | "
                 f"{v['d0']:.3f} | {v['d0_signed']:+.3f} |")
    L += ["", f"Line (registered): all but lane-change, teacher median <= 0.30 m at 2 s and <= 0.50 m at 4 s -> **{'PASS' if passed else 'FAIL'}**; "
          f"turns over 2x the line (inspect): {turn_inspect}. No label: {rec['why']}. Lateral lane switches in {rec['switch_scenes']} scenes; "
          f"stations at the +-0.5 m offset bound (mean share) {rec['h_clip_share_mean']}; per-scene human offset median {rec['h_offset_med']} m (+ left). "
          f"Failed scenes: {len(rec['failed'])}."]
    (R / "t1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- teacher on the rows
def _teach_scene(job):
    scene, cfg, idx, anchor, poses, speed, t_us, k = job
    ds = DG.load_ds(scene, cfg)
    (_, ref), rinfo = DG.route_ref(ds)
    P, yaw, ts = DG.traj_of(ds)
    tt = (ts - ts[0]).astype(np.float64) if ts is not None else np.arange(len(P)) * 1e5
    n = len(idx)
    fut, ok = np.full((n, 8, 3), np.nan, np.float32), np.zeros(n, bool)
    why, d0, psi0 = np.full(n, "", "U8"), np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    hum = np.full((n, 41, 2), np.nan, np.float32)
    ferr = np.full(n, np.nan, np.float32)
    for i in range(n):
        A = anchor[i].astype(np.float64)
        a = DG.arcs_of(poses[i])
        path, info = DG.teacher_path(ref, A, float(speed[i]), a[-1])
        why[i], d0[i], psi0[i] = info["why"], info.get("d0", np.nan), info.get("psi0", np.nan)
        if path is not None:
            fut[i] = DG.place(path, a, A)[0]
            ok[i] = np.isfinite(fut[i]).all()
        tq = t_us[i] + np.arange(-10, 31) * 1e5                                  # the logged path, t0 - 1 .. + 3 s, row frame (plots)
        hx, hy = np.interp(tq, tt, P[:, 0]), np.interp(tq, tt, P[:, 1])
        c, s = np.cos(A[2]), np.sin(A[2])
        hum[i] = np.c_[c * (hx - A[0]) + s * (hy - A[1]), -s * (hx - A[0]) + c * (hy - A[1])]
        if k[i] == 0:                                                           # frame check: k = 0 is the log-replay state
            ferr[i] = float(np.hypot(np.interp(t_us[i], tt, P[:, 0]) - A[0], np.interp(t_us[i], tt, P[:, 1]) - A[1]))
    return dict(idx=idx, fut=fut, ok=ok, why=why, d0=d0, psi0=psi0, hum=hum, ferr=ferr, route=rinfo, scene=scene)


def cmd_teach(a):
    from jevdrive import par
    z = np.load(ROWS / a.name / "meta.npz")
    names, sc, run = z["names"].astype(str), z["scene"].astype(str), z["run"].astype(str)
    jobs = []
    for s in np.unique(sc):
        i = np.flatnonzero(sc == s)
        cfg = f"{run[i[0]]}/generated-user-config-0.yaml"
        jobs.append((s, cfg, i, z["anchor"][i], z["poses"][i], z["speed"][i], z["t_us"][i], z["k"][i]))
    res = par.pmap(_teach_scene, jobs, workers=a.workers or None, desc="teach scenes")
    n = len(names)
    X = dict(names=names, fut=np.full((n, 8, 3), np.nan, np.float32), ok=np.zeros(n, bool), why=np.full(n, "fail", "U8"),
             d0=np.full(n, np.nan, np.float32), psi0=np.full(n, np.nan, np.float32), hum=np.full((n, 41, 2), np.nan, np.float32),
             ferr=np.full(n, np.nan, np.float32))
    routes = {}
    for v in res.values:
        if v:
            for key in ("fut", "ok", "why", "d0", "psi0", "hum", "ferr"):
                X[key][v["idx"]] = v[key]
            routes[v["scene"]] = v["route"]
    fail = {jobs[i][0]: e.strip().splitlines()[-1][:300] for i, e in res.errors.items()}
    np.savez(a.out, **X, version=TEACHER, routes=json.dumps(routes), failed=json.dumps(fail))
    fe = X["ferr"][np.isfinite(X["ferr"])]
    print(json.dumps(dict(rows=n, ok=int(X["ok"].sum()), why=dict(Counter(X["why"].tolist())), failed=len(fail),
                          frame_err_k0=dict(n=len(fe), med=float(np.median(fe)) if len(fe) else None, max=float(fe.max()) if len(fe) else None))))
    if fail:
        print("failed:", json.dumps(fail)[:2000])


# ---------------------------------------------------------------- T2 / T4 on the pool's drivable raster
def foot_sdf(fut, sdf):
    """fut (n, 8, 3) rear-axle poses in the row frame, sdf (n, 128, 96) -> (n, 8) min SDF over the ego box outline (12 points; nan where a
    point leaves the canvas), bilinear on the 0.5 m cell centres."""
    fut = np.asarray(fut, np.float64)
    u = np.r_[-1, 0, 1, 1, 1, 0, -1, -1, -.5, .5, .5, -.5]
    w = np.r_[-1, -1, -1, 0, 1, 1, 1, 0, -1, -1, 1, 1]
    bx, by = EGO_RA + u * EGO_L / 2, w * EGO_W / 2
    c, s = np.cos(fut[..., 2:3]), np.sin(fut[..., 2:3])
    X = fut[..., 0:1] + c * bx - s * by
    Y = fut[..., 1:2] + s * bx + c * by
    fi, fj = (X - X0) / RES - 0.5, (Y - Y0) / RES - 0.5
    inside = (fi >= 0) & (fi <= NH - 1) & (fj >= 0) & (fj <= NW - 1)
    i0, j0 = np.clip(np.floor(fi), 0, NH - 2).astype(int), np.clip(np.floor(fj), 0, NW - 2).astype(int)
    di, dj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
    r = np.arange(len(fut))[:, None, None]
    S = sdf.astype(np.float32)
    v = (S[r, i0, j0] * (1 - di) * (1 - dj) + S[r, i0 + 1, j0] * di * (1 - dj) + S[r, i0, j0 + 1] * (1 - di) * dj + S[r, i0 + 1, j0 + 1] * di * dj)
    v = np.where(inside, v, np.nan)
    return np.nanmin(np.where(np.isnan(v), np.inf, v), -1).astype(np.float32) * np.where(inside.any(-1), 1, np.nan)


def _sdf_rows(name, futs, chunk=4096):
    """min SDF of each fut set over the 8 poses (nan = a pose fully off canvas is ignored; all off -> nan), and the share of poses on canvas."""
    d = np.load(CR / f"on1_{name}" / "drivable.npz", mmap_mode="r")
    sd = d["sdf"]
    out = [np.full(len(sd), np.nan, np.float32) for _ in futs]
    cov = [np.zeros(len(sd), np.float32) for _ in futs]
    for lo in range(0, len(sd), chunk):
        S = np.asarray(sd[lo:lo + chunk])
        for j, f in enumerate(futs):
            m = foot_sdf(np.nan_to_num(f[lo:lo + chunk]), S)
            cov[j][lo:lo + chunk] = np.isfinite(m).mean(1)
            out[j][lo:lo + chunk] = np.where(np.isfinite(m).any(1), np.nanmin(np.where(np.isfinite(m), m, np.inf), 1), np.nan)
    return out, np.asarray(d["tokens"]).astype(str), cov


def cmd_label(a):
    tb = dict(np.load(CR / f"on1_{a.name}" / "tab.npz"))
    T = np.load(a.teach)
    assert str(T["version"]) == TEACHER
    names = tb["names"].astype(str)
    assert np.array_equal(names, T["names"].astype(str)), "teacher rows != the pool's"
    z = np.load(ROWS / a.name / "meta.npz")
    assert np.array_equal(z["names"].astype(str), names)
    n, k, v0 = len(names), z["k"], tb["speed"].astype(np.float64)
    assert np.abs(z["speed"] - v0).max() < 1e-4 and np.abs(z["poses"] - tb["served"]).max() < 1e-3, "rows meta != the pool's speed / served"
    fut_t, ok = T["fut"], T["ok"]
    now = np.zeros((n, 8, 3), np.float32)                                      # the student's own footprint at t0 (8 copies of the origin)
    (s_t, s_sv, s_now), tok, (c_t, c_sv, _) = _sdf_rows(a.name, [fut_t, tb["served"], now])
    assert np.array_equal(tok, names), "drivable rows != the pool's"
    al_t = np.where(ok, DG.alat_pos(np.nan_to_num(fut_t)), np.nan)
    al_sv = DG.alat_pos(tb["served"])
    cand = ok & (v0 >= V_ROW) & (k >= K_MIN)
    logs = tb["log"].astype(str)
    heldl = {g for g in set(logs[tb["mode"].astype(str) == "self"]) if zlib.crc32(g.encode()) % 100 < HOLD_PCT} if "mode" in tb else \
        {g for g in set(logs) if zlib.crc32(g.encode()) % 100 < HOLD_PCT}
    inside_now = s_now >= 0
    canvas = c_t >= 0.75                                                       # >= 6 of the 8 label poses on the 56 m canvas
    leave = inside_now & canvas & (s_t < 0)
    hi = al_t > A_LAT
    drop = cand & (hi | leave)
    draw = cand & ~drop
    # T2: drawable candidates whose own state is inside the scorer's drivable area, label on canvas
    m2 = cand & inside_now & canvas
    t2 = float((s_t[m2] >= 0).mean()) if m2.any() else float("nan")
    t2_sv = float((s_sv[m2] >= 0).mean()) if m2.any() else float("nan")
    out_now = cand & ~inside_now & canvas                                       # states already off the drivable area: does the teacher come back?
    back = float((s_t[out_now] > s_sv[out_now]).mean()) if out_now.any() else float("nan")
    q = lambda x: [round(float(np.nanpercentile(x, p)), 3) for p in (50, 90, 99)] if np.isfinite(x).any() else None  # noqa: E731
    d0 = T["d0"]
    rec = dict(pool=f"on1_{a.name}", rows=n, scenes=len(set(z["scene"].astype(str))), k_ge2=int((k >= K_MIN).sum()), fed_ge2=int(((k >= K_MIN) & (v0 >= V_ROW)).sum()),
               teacher_ok=int(ok.sum()), candidates=int(cand.sum()), no_label=dict(Counter(T["why"][(k >= K_MIN) & (v0 >= V_ROW) & ~ok].tolist())),
               frame_err_k0=q(T["ferr"][np.isfinite(T["ferr"])]),
               T2=dict(rows=int(m2.sum()), teacher_inside=t2, served_inside=t2_sv, line=0.90, passed=bool(t2 >= 0.90),
                       off_now_rows=int(out_now.sum()), off_now_teacher_better=back),
               T4=dict(line=A_LAT, alat_teacher=q(al_t[cand]), alat_served=q(al_sv[cand]), over=int((cand & hi).sum()), leave=int((cand & leave).sum()),
                       dropped=int(drop.sum()), dropped_share=float(drop.sum() / max(cand.sum(), 1)), line_share=0.10,
                       passed=bool(drop.sum() <= 0.10 * max(cand.sum(), 1))),
               drawable=int(draw.sum()), drawable_held_logs=int((draw & np.isin(logs, sorted(heldl))).sum()),
               d0=dict(abs=q(np.abs(d0[draw])), signed_med=float(np.median(d0[draw])) if draw.any() else None, left_gt_05=int((draw & (d0 > .5)).sum()),
                       right_gt_05=int((draw & (d0 < -.5)).sum()), left_gt_1=int((draw & (d0 > 1)).sum()), right_gt_1=int((draw & (d0 < -1)).sum())),
               lat4=q(np.abs(fut_t[draw, 7, 1] - tb["served"][draw, 7, 1])))
    R = Path(a.res)
    dump(R / f"label_{a.name}.json", rec)
    np.savez(CR / f"on1_{a.name}" / "dagger.npz", names=names, draw=draw, drop=drop, cand=cand, s_t=s_t, s_sv=s_sv, s_now=s_now, al_t=al_t,
             d0=d0, k=k)
    L = [f"# DAgger rows of on1_{a.name}: teacher labels, T2, T4 (nd_teach.py label)", "",
         f"Rows {n} ({rec['scenes']} scenes); policy decisions (k >= 2) {rec['k_ge2']}, fed >= 2 m/s {rec['fed_ge2']}, teacher label "
         f"{rec['candidates']} (no label {rec['no_label']}). Frame check (k = 0 anchor vs the scene loader's logged pose, m, p50 / p90 / p99): {rec['frame_err_k0']}.", "",
         f"**T2** (scorer's drivable raster, ego box outline; rows whose own state is inside and >= 6 / 8 label poses on the 56 m canvas, n = "
         f"{rec['T2']['rows']}): teacher label inside {t2:.3f} (served plan {t2_sv:.3f}); line >= 0.90 -> **{'PASS' if rec['T2']['passed'] else 'FAIL'}**. "
         f"States already off the drivable area: {rec['T2']['off_now_rows']}, teacher deeper inside than the served plan in {back:.2f} of them.", "",
         f"**T4** (drop infeasible labels): lateral acceleration (position estimator, p50 / p90 / p99) teacher {rec['T4']['alat_teacher']} vs served "
         f"{rec['T4']['alat_served']}; > {A_LAT} m/s^2 {rec['T4']['over']}, label leaves the drivable area from an inside state {rec['T4']['leave']}; "
         f"dropped {rec['T4']['dropped']} ({rec['T4']['dropped_share']:.1%}; line <= 10 %) -> **{'PASS' if rec['T4']['passed'] else 'FAIL'}**.", "",
         f"Drawable rows {rec['drawable']} ({rec['drawable_held_logs']} in the recipe's held-out pool logs: weight 0). |d0| p50 / p90 / p99 "
         f"{rec['d0']['abs']}, signed median {rec['d0']['signed_med']}; |d0| > 0.5 m left / right {rec['d0']['left_gt_05']} / {rec['d0']['right_gt_05']}, "
         f"> 1 m {rec['d0']['left_gt_1']} / {rec['d0']['right_gt_1']}. |label - served| lateral at 4 s p50 / p90 / p99 {rec['lat4']} m."]
    (R / f"label_{a.name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.write:
        fut = np.where(draw[:, None, None], fut_t, tb["served"]).astype(np.float32)
        np.savez(CR / f"on1_{a.name}" / "tab.npz", **(tb | dict(fut=fut, src=np.full(n, "dagger").astype("U16"))))
        if not (rec["T2"]["passed"] and rec["T4"]["passed"]):
            sys.exit(1)


# ---------------------------------------------------------------- plots (probe)
def cmd_plots(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tb = np.load(CR / f"on1_{a.name}" / "tab.npz")
    T = np.load(a.teach)
    z = np.load(ROWS / a.name / "meta.npz")
    d = np.load(CR / f"on1_{a.name}" / "drivable.npz", mmap_mode="r")
    sc, k = z["scene"].astype(str), z["k"]
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    xs, ys = X0 + RES * (np.arange(NH) + .5), Y0 + RES * (np.arange(NW) + .5)
    for s in np.unique(sc):
        idx = [i for i in np.flatnonzero(sc == s) if k[i] >= K_MIN][::3][:6]
        if not idx:
            continue
        fig, ax = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 4.2), squeeze=False)
        for x, i in zip(ax[0], idx):
            S = np.asarray(d["sdf"][i], np.float32)
            x.contourf(ys, xs, S, levels=[-50, 0, 50], colors=["#dddddd", "white"])
            x.contour(ys, xs, S, levels=[0], colors="k", linewidths=0.6)
            h = T["hum"][i]
            x.plot(h[:, 1], h[:, 0], color="#888888", lw=1.5, label="human (log)")
            sv = tb["served"][i]
            x.plot(np.r_[0, sv[:, 1]], np.r_[0, sv[:, 0]], "o-", color="#d62728", ms=2.5, lw=1, label="student served")
            if T["ok"][i]:
                f = T["fut"][i]
                x.plot(np.r_[0, f[:, 1]], np.r_[0, f[:, 0]], "o-", color="#1f77b4", ms=2.5, lw=1, label="teacher")
            x.set_xlim(12, -12), x.set_ylim(-6, max(12.0, float(np.nanmax(np.r_[sv[:, 0], 8])) + 4))
            x.set_aspect("equal")
            x.set_title(f"k {k[i]}  v {tb['speed'][i]:.1f}  d0 {T['d0'][i]:+.2f}", fontsize=8)
        ax[0][0].legend(fontsize=6, loc="lower left")
        fig.suptitle(f"{s} (row frame, x forward / y left; grey = off the drivable area)", fontsize=8)
        fig.tight_layout()
        fig.savefig(O / f"{s[-16:]}.png", dpi=110)
        plt.close(fig)
    print("plots ->", O)


# ---------------------------------------------------------------- probe gates
def cmd_chk1(a):
    R = Path(a.res)
    lab = json.loads((R / "label_NDP.json").read_text())
    t1 = json.loads((R / "t1_probe" / "t1.json").read_text())
    rs = json.loads((Path(a.root) / "roll/c000/run/aggregate/results-summary.json").read_text())["rollouts"]
    scored = sum(1 for r in rs if r.get("score") is not None and not r.get("failure_reason"))
    fe = lab["frame_err_k0"]
    ck = dict(scored=(scored, ">= 5", scored >= 5), rows=(lab["rows"], ">= 60", lab["rows"] >= 60),
              teacher_share=(lab["candidates"] / max(lab["fed_ge2"], 1), ">= 0.8", lab["candidates"] >= 0.8 * lab["fed_ge2"]),
              frame=(fe, "p99 <= 0.1 m", fe is not None and fe[2] <= 0.1), T1=(t1["table"].get("all but lane-change"), "registered line", t1["passed"]),
              T2=(lab["T2"]["teacher_inside"], ">= 0.90", lab["T2"]["passed"]), T4=(lab["T4"]["dropped_share"], "<= 0.10", lab["T4"]["passed"]))
    ok = all(v[2] for v in ck.values())
    dump(R / "chk1.json", dict(passed=ok, checks=ck))
    print(json.dumps(dict(passed=ok, checks=ck), indent=1, default=str))
    if not ok:
        (Path(a.root) / "VERDICT").write_text(f"STOP probe gates: {[k_ for k_, v in ck.items() if not v[2]]}\n")
        sys.exit(1)


# ---------------------------------------------------------------- pool append, weights, smoke
def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")


def _cat(x, y, n_y):
    """x (base rows), y (added rows or None) -> concatenation; y missing: NaN (float), '' (str), 0 / False."""
    if y is None:
        y = np.full((n_y,) + x.shape[1:], np.nan if x.dtype.kind == "f" else "" if x.dtype.kind == "U" else 0, x.dtype)
    elif x.dtype.kind == "U" or y.dtype.kind == "U":
        w = max(x.dtype.itemsize, y.dtype.itemsize) // 4
        x, y = x.astype(f"U{w}"), y.astype(f"U{w}")
    else:
        y = y.astype(x.dtype)
    return np.concatenate([x, y])


def cmd_append(a):
    """on1_<out> = on1_<base> (every file bit-equal on its rows, same order and names) + on1_<add> (renamed on:<prefix>:<token>, mode both,
    pool <prefix>); tokens zero-copy; route features (feat/.npz and .a.npz) appended alike. Files of the base the recipe does not read
    (stand_gap, stand_launch, tight_in, turn_cap) are not carried."""
    from jevdrive.tokens import link_front, open_front
    B, Ad, O = CR / a.base, CR / a.add, CR / a.out
    assert not O.exists() and not (CR / f"{a.out}@warp").exists(), f"{O} exists: never overwritten"
    O.mkdir(), (CR / f"{a.out}@warp").mkdir()
    tb, ta = dict(np.load(B / "tab.npz")), dict(np.load(Ad / "tab.npz"))
    nb, na = len(tb["names"]), len(ta["names"])
    ren = np.array([f"on:{a.prefix}:{t[3:]}" for t in ta["names"].astype(str)])
    ta |= dict(names=ren, pool=np.full(na, a.prefix), mode=np.full(na, "both"))
    extra = sorted(set(ta) - set(tb))
    M = {k_: _cat(tb[k_], ta.get(k_), na) for k_ in tb}
    np.savez(O / "tab.npz", **M)
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        zb, za = np.load(B / f), np.load(Ad / f)
        A = {}
        for k_ in zb.files:
            if k_ == "tokens":
                A[k_] = _cat(zb[k_], np.asarray([f"on:{a.prefix}:{t[3:]}" for t in za[k_].astype(str)]) if za[k_].astype(str)[0].startswith("on:") else za[k_], na)
            elif zb[k_].ndim and len(zb[k_]) == nb:
                assert k_ in za.files and len(za[k_]) == na, f"{f}.{k_}: added rows"
                A[k_] = _cat(zb[k_], za[k_], na)
            else:
                A[k_] = zb[k_]
        np.savez(O / f, **A)
    for sfx in ("", ".a"):
        zb, za = np.load(FEAT / f"{a.base}{sfx}.npz"), np.load(FEAT / f"{a.add}{sfx}.npz")
        np.savez(FEAT / f"{a.out}{sfx}.npz", **{k_: _cat(zb[k_], ren if k_ == "names" else za[k_], na) if zb[k_].ndim and len(zb[k_]) == nb else zb[k_]
                                                for k_ in zb.files})
    link_front(CR / f"{a.out}@warp", [CR / f"{a.base}@warp/front.npy", CR / f"{a.add}@warp/front.npy"])
    Tb, Ta = np.load(CR / f"{a.base}@warp/teacher.npz"), np.load(CR / f"{a.add}@warp/teacher.npz")
    assert same(Tb["di"], Ta["di"]) and same(Tb["pi"], Ta["pi"])
    np.savez(CR / f"{a.out}@warp/teacher.npz", out=np.concatenate([Tb["out"], Ta["out"]]), plan=np.concatenate([Tb["plan"], Ta["plan"]]), di=Tb["di"], pi=Tb["pi"])
    # check: every base file bit-equal on the base rows
    bad = []
    to = np.load(O / "tab.npz")
    bad += [f"tab.{k_}" for k_ in tb if not same(to[k_][:nb], tb[k_])]
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        zo, zb = np.load(O / f, mmap_mode="r"), np.load(B / f, mmap_mode="r")
        bad += [f"{f}.{k_}" for k_ in zb.files if zb[k_].ndim and len(zb[k_]) == nb and not same(zo[k_][:nb], zb[k_])]
    for sfx in ("", ".a"):
        zo, zb = np.load(FEAT / f"{a.out}{sfx}.npz"), np.load(FEAT / f"{a.base}{sfx}.npz")
        bad += [f"feat{sfx}.{k_}" for k_ in zb.files if not same(zo[k_][:nb] if zb[k_].ndim and len(zb[k_]) == nb else zo[k_], zb[k_])]
    zo = np.load(CR / f"{a.out}@warp/teacher.npz")
    bad += [f"teacher.{k_}" for k_ in Tb.files if not same(zo[k_][:nb] if Tb[k_].ndim and len(Tb[k_]) == nb else zo[k_], Tb[k_])]
    i = np.sort(np.random.default_rng(0).choice(nb, 4000, replace=False))
    if not same(open_front(CR / f"{a.out}@warp/front.npy")[i], open_front(CR / f"{a.base}@warp/front.npy")[i]):
        bad.append("front sample (base)")
    j = np.sort(np.random.default_rng(1).choice(na, min(na, 1000), replace=False))
    if not same(open_front(CR / f"{a.out}@warp/front.npy")[nb + j], open_front(CR / f"{a.add}@warp/front.npy")[j]):
        bad.append("front sample (added)")
    rec = dict(out=a.out, base_rows=nb, added_rows=na, extra_keys_dropped=extra, mismatch=bad, modes=dict(Counter(to["mode"].astype(str).tolist())))
    dump(Path(a.res) / f"append_{a.out}.json", rec)
    print(json.dumps(rec))
    if bad:
        sys.exit(2)


def cmd_weights(a):
    """the base draw (JTX00) x k_on / (k_on + k_dag) on the base rows; the drawable DAgger rows (dagger.npz draw, not in the recipe's
    held-out pool logs) k_dag / (k_on + k_dag) uniform. k_dag 0: the base weights with the added rows at 0 (T3 identity file)."""
    Bw = np.load(DATA / "runs/defect_fix/w" / f"{a.base}.npz")
    tb = np.load(CR / a.pool / "tab.npz")
    names, n0 = tb["names"].astype(str), len(Bw["names"])
    assert np.array_equal(names[:n0], Bw["names"].astype(str)), "prefix != the base draw's rows"
    X = np.load(CR / a.src / "dagger.npz")
    assert len(names) == n0 + len(X["names"]) and all(x.split(":", 2)[2] == y.split(":", 1)[1] for x, y in zip(names[n0:], X["names"].astype(str)))
    logs, mode = tb["log"].astype(str), tb["mode"].astype(str)
    heldl = sorted(g for g in set(logs[mode == "self"]) if zlib.crc32(g.encode()) % 100 < HOLD_PCT)       # rm_train --on-hold-pct 15's held logs
    dr = np.flatnonzero(X["draw"] & ~np.isin(logs[n0:], heldl))
    k_all = a.k_on + a.k_dag
    w = np.zeros(len(names))
    w[:n0] = Bw["w"].astype(np.float64) * (a.k_on / k_all if a.k_dag else 1.0)
    if a.k_dag:
        w[n0 + dr] = a.k_dag / k_all / len(dr)
    grp = np.concatenate([Bw["grp"].astype(str), np.full(len(names) - n0, "nd")]).astype("U12")
    assert abs(w.sum() - 1) < 1e-9
    if not a.k_dag:
        assert np.array_equal(w[:n0], Bw["w"])
    gs = defaultdict(float)
    for g_, x in zip(grp, w):
        gs[str(g_)] += x
    rec = dict(base=a.base, pool=a.pool, k_on=a.k_on, k_dag=a.k_dag, dag_rows=int(len(dr)), held_logs=len(heldl),
               dag_rows_held_out=int((X["draw"] & np.isin(logs[n0:], heldl)).sum()), group_share={k_: float(v) for k_, v in sorted(gs.items()) if v > 0},
               dag_draws_per_row_9500=float(a.k_dag * 9500 / max(len(dr), 1)), out=a.out)
    WD.mkdir(parents=True, exist_ok=True)
    np.savez(WD / f"{a.out}.npz", names=names, w=w, grp=grp)
    dump(Path(a.res) / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_smoke(a):
    import r3
    d = r3.run_dir(a.tag)
    S = defaultdict(list)
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S[e["tag"]].append(e["value"])
    done = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else {}
    ds = done.get("draw_shares") or {}
    ex = json.loads((Path(a.res) / f"w_{a.wfile}.json").read_text())
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", (d / "log.txt").read_text())
    want = ("45", "13", str(70 + a.k_dag))
    gs = ds.get("group_share", {})
    ck = dict(done=(bool(done), True, bool(done)), finite=(None, None, all(np.isfinite(v).all() for v in S.values())),
              wfile=(ds.get("on_wfile"), a.wfile, f"/{a.wfile}.npz" in str(ds.get("on_wfile"))),
              batch_rows=(b.groups() if b else None, want, bool(b) and b.groups() == want))
    ck |= {f"share_{p}": (gs.get(p), v, abs(gs.get(p, -1) - v) < 1e-6) for p, v in ex["group_share"].items()}
    ok = bool(all(v[2] for v in ck.values()))
    dump(Path(a.res) / "smoke.json", dict(tag=a.tag, pass_=ok, checks=ck, steps=len(S.get("loss/total", [])), draw_shares=ds))
    print(json.dumps(dict(pass_=ok, checks=ck), indent=1, default=float))
    if not ok:
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("lists"); q.add_argument("--out", required=True); q.add_argument("--n", type=int, default=24)
    q = sp.add_parser("t1"); q.add_argument("--cfg-glob", required=True); q.add_argument("--out-dir", required=True)
    q.add_argument("--workers", type=int, default=0); q.add_argument("--limit", type=int, default=0)
    q = sp.add_parser("teach"); q.add_argument("--name", required=True); q.add_argument("--out", required=True); q.add_argument("--workers", type=int, default=0)
    q = sp.add_parser("label"); q.add_argument("--name", required=True); q.add_argument("--teach", required=True); q.add_argument("--res", required=True)
    q.add_argument("--write", action="store_true")
    q = sp.add_parser("plots"); q.add_argument("--name", required=True); q.add_argument("--teach", required=True); q.add_argument("--out", required=True)
    q = sp.add_parser("chk1"); q.add_argument("--res", required=True); q.add_argument("--root", required=True)
    q = sp.add_parser("append"); q.add_argument("--base", required=True); q.add_argument("--add", required=True); q.add_argument("--out", required=True)
    q.add_argument("--prefix", default="nd"); q.add_argument("--res", required=True)
    q = sp.add_parser("weights")
    for k_ in ("pool", "base", "src", "out", "res"):
        q.add_argument(f"--{k_}", required=True)
    q.add_argument("--k-on", type=int, default=70); q.add_argument("--k-dag", type=int, default=8)
    q = sp.add_parser("smoke"); q.add_argument("--tag", required=True); q.add_argument("--wfile", required=True); q.add_argument("--res", required=True)
    q.add_argument("--k-dag", type=int, default=8)
    a = p.parse_args()
    dict(lists=cmd_lists, t1=cmd_t1, teach=cmd_teach, label=cmd_label, plots=cmd_plots, chk1=cmd_chk1, append=cmd_append, weights=cmd_weights,
         smoke=cmd_smoke)[a.cmd](a)


if __name__ == "__main__":
    main()
