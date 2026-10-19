#!/usr/bin/env python3
"""STAND-B (NR1 prereg supplement 19): why arm B of the JTX STAND pilot (STANDFMX1K-L18-s0 = STAND + FREE-M at W 0.002) has more held 460
zeros and front collisions than its control JTX1K-L18-s0, and what a stricter FREE-M free check must block. CPU, stored data only. Box, repo root.

  $AP stand_b.py diag --pair ARM:CTL [--pair ..] --out DIR    held 460 @18: the windows the arm newly zeros / newly front-collides (and the
        reverse); per decision of those windows, both tags: the lead along the served plan's path (gap, closing speed, TTC), the struck object
        before a front collision, and the FREE-M term (supplement 16) scored on the closed-loop state, with the current free check and the
        candidate stricter ones -> DIR/dec.parquet, DIR/win.csv, DIR/diag.json
  $AP stand_b.py pool --out DIR      on1_JTX's FREE-M rows (S rows, moving), the plan = the human's `fut` path cut at RB0-F-s0's own 4 s arc
        (lat_diag trainee): which rows fire, the lead state on the firing rows, how much each candidate check removes -> DIR/pool.json
  $AP stand_b.py ratchet --clear CMIN,CTH --out JSON   free_m.py's ratchet set (JTN-L18-s1 native arena held) with the stricter check:
        the share of the short moving decisions that still fire (supplement 16 line >= 50 %) -> JSON; exit 0 pass, 2 fail

Frames: rollout local frame (the scene loader's), ego = rear axle, x forward, y left. Decision k at t_k = 17 000 + 500 000 k us. The served
plan `poses` (drive.jsonl) is in the anchor frame (the driver's rear-axle pose at the decision, <= 5 cm from the controller's, cap_audit).
Lead: the nearest logged object (vehicle / bicycle / pedestrian / generic) whose box comes within HALF_W + 0.3 m of the plan's path (the 8
poses, extended 80 m straight along the last heading) with its nearest corner ahead of the front bumper; gap = that corner's path station minus
the bumper; closing = ego speed minus the object's velocity along the path tangent (0.5 s backward difference); TTC = gap / closing (closing
> 0.1 m/s, else inf). FREE-M on a closed-loop state: as free_gate.py, the human = the scene's logged rig trajectory at the same scene time
(raw 4 s arc and its version reachable from the controller speed, the smaller per 0.5 s step), the 32 nearest objects followed by id to
+0.5 .. 4 s in the anchor frame, nr1_free.terms(parts=True) with the supplement 16 config (0.95, move).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/alpasim_proxy/scripts"),
                str(REPO / "experiments/zeros_mech/scripts"), str(REPO / "experiments/wait_cap/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
T0, DT, NK, K = 17_000, 500_000, 18, 32
FRONT, REAR, HALF_W = 4.049, 1.127, 1.1485
CLEARS = {"cur": (0.0, 0.0), "d20t3": (20.0, 3.0), "d15t2": (15.0, 2.0), "d30t4": (30.0, 4.0)}   # candidate stricter checks (CMIN m, CTH s)
TTC_MIN = 4.0


def held_dirs(tag):
    return sorted(glob.glob(str(DATA / f"runs/route_mem/read/{tag}/c*/run")))


def rollouts(tag):
    """scene -> (run dir, rollout row) over the held 460 dirs (first per scene, nx_regress.rollouts' rule)."""
    R = {}
    for d in held_dirs(tag):
        f = Path(d) / "aggregate/results-summary.json"
        if f.exists():
            for r in json.loads(f.read_text())["rollouts"]:
                R.setdefault(r["clipgt_id"], (d, r))
    return R


def first_event(run, sc, rid, name):
    import polars as pl
    f = glob.glob(f"{run}/rollouts/{sc}/{rid}/metrics.parquet")
    if not f:
        return None
    M = pl.read_parquet(f[0], columns=["name", "timestamps_us", "values"]).filter((pl.col("name") == name) & (pl.col("values") > 0))
    return int(M["timestamps_us"].cast(pl.Int64).min()) if len(M) else None


def ctrl(run, rid):
    f = Path(run) / f"controller/alpasim_controller_{rid}.csv"
    if not f.exists():
        return None
    c = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 2, 6, 7, 8, 9), ndmin=2)
    return dict(t=c[:, 0], x=c[:, 1], y=c[:, 2], yaw=np.unwrap(2 * np.arctan2(c[:, 3], c[:, 4])), v=np.hypot(c[:, 5], c[:, 6]))


def at(C, t):
    return np.array([np.interp(t, C["t"], C[k]) for k in ("x", "y", "yaw")]), float(np.interp(t, C["t"], C["v"]))


def path_line(poses):
    import shapely
    P = np.vstack([[0.0, 0.0], np.asarray(poses, float)[:, :2]])
    yaw = float(np.asarray(poses, float)[-1, 2])
    P = np.vstack([P, P[-1] + 80.0 * np.array([np.cos(yaw), np.sin(yaw)])])
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-3]
    P = P[keep]
    if len(P) < 2:
        P = np.array([[0.0, 0.0], [80.0, 0.0]])
    return shapely.LineString(P)


def tangent(line, s):
    p0, p1 = np.asarray(line.interpolate(max(s - 0.5, 0)).coords[0]), np.asarray(line.interpolate(min(s + 0.5, line.length)).coords[0])
    d = p1 - p0
    return d / max(np.hypot(*d), 1e-6)


def objects_in(ZM, st, a):
    """obj_state -> centres / corners / headings in the frame of pose a."""
    E = ZM.in_ego(a, np.c_[st["x"], st["y"]])
    dpsi = ZM.wrap(st["yaw"] - a[2])
    C = ZM.corners(E[:, 0], E[:, 1], dpsi, st["L"], st["W"])
    return E, dpsi, C


def lead_of(ZM, line, st, st_prev, a, v_ego, only=None):
    """-> dict(id, gap, closing, ttc, v_obj_long, lat, cls) of the path lead (nearest in-path object ahead), or of object `only`."""
    import shapely
    if not len(st["id"]):
        return None
    E, dpsi, C = objects_in(ZM, st, a)
    prev = {i: (x, y) for i, x, y in zip(st_prev["id"], st_prev["x"], st_prev["y"])} if st_prev is not None else {}
    best = None
    for i in range(len(st["id"])):
        if only is not None and st["id"][i] != only:
            continue
        poly = shapely.Polygon(C[i])
        dist = float(poly.distance(line))
        s = min(float(line.project(shapely.Point(*c))) for c in C[i])
        gap = s - FRONT
        if only is None and (dist > HALF_W + 0.3 or gap < -1.0):
            continue
        q = prev.get(st["id"][i])
        vel = (np.array([st["x"][i], st["y"][i]]) - np.array(q)) / 0.5 if q is not None else np.zeros(2)
        c, sn = np.cos(a[2]), np.sin(a[2])
        vel_e = np.array([c * vel[0] + sn * vel[1], -sn * vel[0] + c * vel[1]])
        vl = float(vel_e @ tangent(line, max(s, 0.0)))
        clo = v_ego - vl
        r = dict(id=str(st["id"][i]), gap=gap, path_dist=dist, closing=clo, ttc=gap / clo if clo > 0.1 and gap > 0 else (0.0 if gap <= 0 else np.inf),
                 v_obj_long=vl, x=float(E[i, 0]), y=float(E[i, 1]), static=bool(st["static"][i]))
        if best is None or r["gap"] < best["gap"]:
            best = r
    return best


def human(Tg, Sg, vg, t, ve):
    """the human's raw 4 s arc from scene time t (0.5 s steps) and its version reachable from ego speed ve (free_gate.state_row)."""
    raw = np.interp(t + 0.5e6 * np.arange(1, 9), Tg, Sg) - np.interp(t, Tg, Sg)
    tau = np.arange(80) / 20
    vb = ve + (np.interp(t + tau * 1e6, Tg, vg) - ve) * np.minimum(tau / 2, 1)
    reach = np.cumsum(vb / 20)[9::10]
    return raw, reach


def boxes(ZM, objs, a, t):
    """the K nearest objects at t followed by id to +0.5 .. 4 s in pose a's frame -> box (9, K, 5), val (9, K)."""
    st0 = ZM.obj_state(objs, t)
    box, val = np.zeros((9, K, 5), np.float32), np.zeros((9, K), bool)
    if not len(st0["id"]):
        return box, val
    near = np.argsort(np.hypot(st0["x"] - a[0], st0["y"] - a[1]))[:K]
    ids = [st0["id"][j] for j in near]
    for j in range(9):
        st = st0 if j == 0 else ZM.obj_state(objs, t + j * 0.5e6)
        pos = {x: r for r, x in enumerate(st["id"])}
        E, dpsi, _ = objects_in(ZM, st, a)
        for k, x in enumerate(ids):
            r = pos.get(x)
            if r is not None:
                box[j, k] = (E[r, 0], E[r, 1], dpsi[r], st["L"][r], st["W"][r])
                val[j, k] = True
    return box, val


def scene_job(job):
    """(scene, cfg, {tag: (run, rollout row)}) -> per-decision rows for each tag."""
    scene, cfg, labs = job
    import torch
    import zm_read as ZM
    from ao_e1 import drive_records, loader
    import nr1_free as NF
    torch.set_num_threads(1)
    try:
        ds = loader(cfg).get_data_source(scene)
        tr = ds.rig.trajectory
        Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2]
        Sg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Pg, axis=0).T))]
        vg = np.gradient(Sg, Tg * 1e-6)
        objs = ZM.objects(ds)
        out = []
        for tag, (run, r) in labs.items():
            C = ctrl(run, r["rollout_id"])
            recs = drive_records(run).get(scene, [])
            if C is None or not recs:
                continue
            tcf = first_event(run, scene, r["rollout_id"], "collision_front")
            struck = None
            if tcf is not None:
                a, _ = at(C, tcf)
                st = ZM.obj_state(objs, tcf)
                if len(st["id"]):
                    hull = ZM.ego_poly(a, -REAR, FRONT, HALF_W)
                    d = [hull.distance(ZM.obj_poly(st, i)) for i in range(len(st["id"]))]
                    struck = str(st["id"][int(np.argmin(d))])
            rows, P, Sr, B, V, mv = [], [], [], [], [], []
            for rec in recs:
                t = float(rec["now"])
                an = np.asarray(rec["anchor"], float)
                _, ve = at(C, t)
                line = path_line(rec["poses"])
                st, sp = ZM.obj_state(objs, t), ZM.obj_state(objs, t - 0.5e6)
                ld = lead_of(ZM, line, st, sp, an, ve)
                sk = lead_of(ZM, line, st, sp, an, ve, only=struck) if struck else None
                raw, reach = human(Tg, Sg, vg, t, ve)                  # clamped at the log's end (flat: the term cannot fire there)
                vh = np.diff(raw, prepend=0.0) / 0.5
                bx, bv = boxes(ZM, objs, an, t)
                P.append(np.asarray(rec["poses"], np.float32))
                Sr.append(np.minimum(raw, reach).astype(np.float32))
                B.append(bx)
                V.append(bv)
                mv.append(bool(NF.moving(np.array([ve]), vh[None])[0]))
                rows.append(dict(tag=tag, scene=scene, k=int(rec["k"]), t=t * 1e-6, v_ego=ve, vh0=float(vh[0]), vh_min=float(vh.min()),
                                 log_short=bool(t + 4e6 > Tg[-1]), t_front=tcf * 1e-6 if tcf is not None else np.nan, struck=struck or "",
                                 **{f"lead_{q}": (ld or {}).get(q, np.nan) for q in ("gap", "closing", "ttc", "v_obj_long", "y")},
                                 lead_id=(ld or {}).get("id", ""),
                                 **{f"struck_{q}": (sk or {}).get(q, np.nan) for q in ("gap", "closing", "ttc", "path_dist", "x", "y")}))
            if not rows:
                continue
            Pt, St, Bt, Vt = (torch.as_tensor(np.stack(x)) for x in (P, Sr, B, V))
            okm = torch.as_tensor(np.array(mv))
            for name, cl in CLEARS.items():
                NF.configure(0.95, "move", cl)
                _, pr = NF.terms(Pt, St, okm, Bt, Vt, parts=True)
                for i, d in enumerate(rows):
                    d[f"fire_{name}"] = bool(pr["fire"][i])
                    if name == "cur":
                        d.update(moving=bool(mv[i]), short=bool(pr["slow"][i]), free_cur=bool(pr["free"][i]), plan4=float(pr["s_plan"][i]),
                                 sref4=float(St[i, -1]))
            NF.configure(0.95, "move")
            out += rows
        return out
    except Exception as e:  # noqa: BLE001
        return [dict(scene=scene, error=repr(e))]


def cmd_diag(a):
    import pandas as pd
    import nx_regress as NR
    from multiprocessing import Pool
    win, jobs, cfg = [], {}, None
    for pr in a.pair:
        A, Cn = pr.split(":")
        RA, RC = rollouts(A), rollouts(Cn)
        cfg = cfg or str(Path(next(iter(RA.values()))[0]) / "generated-user-config-0.yaml")
        for sc in sorted(set(RA) & set(RC)):
            ra, rc = RA[sc][1], RC[sc][1]
            za, zc = float(ra.get("score") or 0) == 0, float(rc.get("score") or 0) == 0
            fa, fc = (((r.get("metrics") or {}).get("collision_front") or 0) > 0 for r in (ra, rc))
            kind = [n for n, c in (("zero_new", za and not zc), ("zero_gone", zc and not za), ("front_new", fa and not fc),
                                   ("front_gone", fc and not fa), ("front_both", fa and fc)) if c]
            if not kind:
                continue
            win.append(dict(pair=pr, scene=sc, kind="+".join(kind), score_arm=float(ra.get("score") or 0), score_ctl=float(rc.get("score") or 0),
                            type_arm=NR.failure_type(ra), type_ctl=NR.failure_type(rc), front_arm=fa, front_ctl=fc,
                            atfault_arm=((ra.get("metrics") or {}).get("collision_at_fault") or 0) > 0,
                            atfault_ctl=((rc.get("metrics") or {}).get("collision_at_fault") or 0) > 0,
                            prog_arm=(ra.get("metrics") or {}).get("progress_rel"), prog_ctl=(rc.get("metrics") or {}).get("progress_rel")))
            jobs.setdefault(sc, {}).update({A: RA[sc], Cn: RC[sc]})
    W = pd.DataFrame(win)
    print(W.groupby(["pair", "kind"]).size().to_string(), flush=True)
    with Pool(a.workers) as p:
        res = p.map(scene_job, [(sc, cfg, labs) for sc, labs in sorted(jobs.items())], chunksize=1)
    D = pd.DataFrame([d for r in res for d in r])
    err = D[D.get("error", pd.Series(dtype=str)).notna()] if "error" in D else D.iloc[:0]
    D = D[D["error"].isna()].drop(columns="error") if "error" in D else D
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D.to_parquet(out / "dec.parquet")
    W.to_csv(out / "win.csv", index=False)
    S = summarise(W, D)
    S["errors"] = err.to_dict("records")
    (out / "diag.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


def summarise(W, D):
    """per pair: window counts by kind and type; for the arm's front-collision windows the struck object's state in the 3 s before the
    collision and the FREE-M fires there under each check; the arm's FREE-M fire share by lead state over all decisions of these windows."""
    S = {}
    for pr, w in W.groupby("pair"):
        A, Cn = pr.split(":")
        s = dict(kinds=w.kind.value_counts().to_dict(), types_new_zero=w[w.kind.str.contains("zero_new")].type_arm.value_counts().to_dict(),
                 types_gone_zero=w[w.kind.str.contains("zero_gone")].type_ctl.value_counts().to_dict())
        fr = w[w.front_arm].scene.tolist()
        x = D[(D.tag == A) & D.scene.isin(fr)]
        pre = x[(x.t <= x.t_front) & (x.t >= x.t_front - 3.0)]
        s["front_arm_windows"] = len(fr)
        s["pre_front"] = dict(decisions=len(pre), windows=int(pre.scene.nunique()),
                              struck_in_path=int((pre.struck_path_dist <= HALF_W + 0.3).sum()),
                              v_ego_med=float(pre.v_ego.median()) if len(pre) else None,
                              struck_gap_med=float(pre.struck_gap.median()) if len(pre) else None,
                              struck_closing_med=float(pre.struck_closing.median()) if len(pre) else None,
                              struck_ttc_lt4=int((pre.struck_ttc < 4).sum()), lead_is_struck=int((pre.lead_id == pre.struck).sum()),
                              moving=int(pre.moving.sum()), short=int((pre.moving & pre.short).sum()),
                              **{f"fire_{n}": int(pre[f"fire_{n}"].sum()) for n in CLEARS},
                              windows_fired_cur=int(pre[pre.fire_cur].scene.nunique()))
        per = []
        for sc, g in x.groupby("scene"):
            g = g.sort_values("k")
            ww = w[w.scene == sc].iloc[0]
            g3 = g[(g.t <= g.t_front) & (g.t >= g.t_front - 3.0)]
            c = D[(D.tag == Cn) & (D.scene == sc)]
            per.append(dict(scene=sc, kind=ww.kind, type_arm=ww.type_arm, type_ctl=ww.type_ctl, score_ctl=ww.score_ctl, t_front=float(g.t_front.iloc[0]),
                            v_at=float(g3.v_ego.iloc[-1]) if len(g3) else None, v_ctl_same_t=float(np.interp(g3.t.iloc[-1], c.t, c.v_ego)) if len(g3) and len(c) else None,
                            gap_first=float(g3.struck_gap.iloc[0]) if len(g3) else None, closing_first=float(g3.struck_closing.iloc[0]) if len(g3) else None,
                            ttc_min=float(g3.struck_ttc.min()) if len(g3) else None, path_dist_min=float(g3.struck_path_dist.min()) if len(g3) else None,
                            fires_cur=int(g3.fire_cur.sum()), fires_cur_window=int(g.fire_cur.sum()), plan_over_ref_med=float((g3.plan4 / g3.sref4.clip(lower=1)).median()) if len(g3) else None,
                            ctl_plan_over_ref_med=float((c.plan4 / c.sref4.clip(lower=1)).median()) if len(c) else None,
                            v_mean_arm=float(g.v_ego.mean()), v_mean_ctl=float(c.v_ego.mean()) if len(c) else None))
        s["front_windows"] = per
        # lead state of the decisions where FREE-M fires (the arm, every decision of these windows)
        y = D[(D.tag == A) & D.fire_cur]
        s["arm_fires_by_lead"] = dict(fires=len(y), no_lead=int(y.lead_gap.isna().sum()),
                                      lead_within_d20t3=int((y.lead_gap < np.maximum(20, 3 * y.v_ego)).sum()), lead_ttc_lt4=int((y.lead_ttc < 4).sum()))
        S[pr] = s
    return S


def cmd_pool(a):
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    import nr1_free as NF
    torch.set_num_threads(16)
    C = DATA / "runs/op_parity/cache/on1_JTX"
    zz = np.load(C / "tab.npz")
    z = {k: zz[k] for k in ("names", "mode", "speed", "fut", "fut_reg")}
    ag = np.load(C / "agent.npz")
    okA = ag["ok"]
    names = z["names"].astype(str)
    s_ref, fin = NF.ref_arc(z["fut"], z["fut_reg"])
    S = (z["mode"].astype(str) == "self") & okA & fin
    mv = NF.moving(z["speed"], NF.seg_speed(z["fut_reg"]))
    m = np.flatnonzero(S & mv)
    T = pq.read_table(DATA / "runs/practice_nuplan/lat_diag/trainee_RB0-F-s0.parquet", columns=["name", "arc4"]).to_pandas().set_index("name")
    arc4 = T.arc4.reindex(names[m]).to_numpy()
    keep = np.isfinite(arc4)
    m, arc4 = m[keep], arc4[keep]
    box, val = ag["box"][m], ag["valid"][m]
    fut = z["fut"][m].astype(np.float32)
    fa = NF.cum_arc(fut)
    # the plan: the human's fut path, cut at RB0's own 4 s arc (each step's arc scaled by arc4 / fut's 4 s arc)
    scale = (arc4 / np.maximum(fa[:, -1], 1e-3)).astype(np.float32)
    P = torch.as_tensor(fut if fut.shape[-1] == 3 else np.concatenate([fut, np.zeros_like(fut[..., :1])], -1))
    sq = torch.as_tensor(fa.astype(np.float32) * scale[:, None])
    x, y, yaw = NF._place(P, sq)
    plan = torch.stack([x, y, yaw], -1)
    out = dict(rows=len(m), s_rows=int(S.sum()), moving_rows=int((S & mv).sum()))
    fires = {}
    for n, cl in CLEARS.items():
        NF.configure(0.95, "move", cl)
        f = []
        for i in range(0, len(m), 8192):
            sl = slice(i, i + 8192)
            _, pr = NF.terms(plan[sl], torch.as_tensor(s_ref[m][sl]), torch.ones(len(plan[sl]), dtype=torch.bool), torch.as_tensor(box[sl]),
                             torch.as_tensor(val[sl]), parts=True)
            f.append(pr["fire"].numpy())
            if n == "cur":
                out.setdefault("_short", []).append(pr["slow"].numpy())
        fires[n] = np.concatenate(f)
    NF.configure()
    short = np.concatenate(out.pop("_short"))
    # lead along the human path at t0 (box at 0 s): the nearest object whose centre is within HALF_W + 0.3 m + half its width of the fut
    # polyline (extended 80 m), its rear ahead of the bumper; closing = ego t0 speed - its 0 -> 0.5 s displacement along x / 0.5
    v0 = z["speed"][m]
    b0, ok0, b1 = box[:, 0], val[:, 0], box[:, 1]
    Pf = np.concatenate([np.zeros((len(m), 1, 2)), fut[..., :2]], 1)
    e = Pf[:, -1] - Pf[:, -2]
    Pf = np.concatenate([Pf, Pf[:, -1:] + 80.0 * (e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-6))[:, None]], 1)
    A0, A1 = Pf[:, :-1], Pf[:, 1:]                                                     # (n, 9, 2) segments
    sl_ = np.linalg.norm(A1 - A0, axis=-1)
    cs = np.concatenate([np.zeros((len(m), 1)), np.cumsum(sl_, 1)[:, :-1]], 1)
    gap, clo = np.full(len(m), np.inf), np.zeros(len(m))
    for k in range(box.shape[2]):
        okk = ok0[:, k]
        if not okk.any():
            continue
        q = b0[:, k, :2][:, None]
        u = np.clip(((q - A0) * (A1 - A0)).sum(-1) / np.maximum(sl_ ** 2, 1e-9), 0, 1)
        d = np.linalg.norm(A0 + u[..., None] * (A1 - A0) - q, axis=-1)
        j = d.argmin(1)
        r = np.arange(len(m))
        sta = cs[r, j] + u[r, j] * sl_[r, j] - b0[:, k, 3] / 2 - FRONT
        g = np.where(okk & (d[r, j] <= HALF_W + 0.3 + b0[:, k, 4] / 2) & (sta > -1.0), sta, np.inf)
        better = g < gap
        gap = np.where(better, g, gap)
        clo = np.where(better, v0 - (b1[:, k, 0] - b0[:, k, 0]) / 0.5, clo)
    ttc = np.where((clo > 0.1) & np.isfinite(gap), gap / np.maximum(clo, 1e-3), np.inf)
    near = gap < np.maximum(20.0, 3.0 * v0)
    out["short"] = int(short.sum())
    for n in CLEARS:
        f = fires[n]
        out[f"fire_{n}"] = dict(fires=int(f.sum()), share_of_moving=float(f.mean()), of_short=float(f[short].mean()) if short.any() else None)
    fc = fires["cur"]
    out["cur_fires_by_lead"] = dict(no_lead=int((fc & ~np.isfinite(gap)).sum()), lead_within_d20t3=int((fc & near).sum()),
                                    lead_ttc_lt4=int((fc & (ttc < 4)).sum()), lead_gap_lt30=int((fc & (gap < 30)).sum()),
                                    gap_quartiles_of_fires_with_lead=np.nanpercentile(gap[fc & np.isfinite(gap)], [10, 25, 50, 75, 90]).tolist()
                                    if (fc & np.isfinite(gap)).any() else None,
                                    closing_median_of_fires_with_lead=float(np.median(clo[fc & np.isfinite(gap)])) if (fc & np.isfinite(gap)).any() else None)
    out["rows_with_near_lead_share"] = float(near.mean())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(name=names[m], v0=v0, gap=gap, closing=clo, ttc=ttc, short=short, **{f"fire_{n}": fires[n] for n in CLEARS})).to_parquet(
        Path(a.out) / "pool.parquet")
    (Path(a.out) / "pool.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def cmd_ratchet(a):
    import pandas as pd
    from multiprocessing import Pool
    import free_m as FM
    cl = tuple(float(x) for x in a.clear.split(","))

    def nf():
        import nr1_free as NF
        NF.configure(FM.FRAC, FM.MODE, cl)
        return NF
    FM._nf = nf
    Sx = pd.read_csv(FM.AR / "set/set.csv.gz").set_index("id")
    held = Sx[Sx.split == "held"]
    jobs = []
    for run in sorted(glob.glob(str(FM.O / "jt/arena" / FM.RATCHET / "c*/run"))):
        if (Path(run) / "aggregate/results-summary.json").exists():
            want = {i: (w.log, int(w.start_idx), w.token, 0.0, 1e9, dict(stratum=w.stratum)) for i, w in held.iterrows()}
            jobs.append(("ratchet", FM.RATCHET, run, want))
    with Pool(a.workers) as p:
        out = p.map(FM._run, jobs, chunksize=1)
    D = pd.DataFrame([d for o in out for d in o])
    for c in ("scored", "moving", "short", "free", "fire"):
        D[c] = D[c].fillna(False).astype(bool)
    R = D[(D.stratum != "static") & (D.k >= 2) & (D.v_ego >= 3.0) & D.scored]
    rs = R[R.short]
    St = D[D.stratum == "static"]
    res = dict(clear=cl, moving_decisions=len(R), short=len(rs), fires=int(rs.fire.sum()), rate=float(rs.fire.mean()) if len(rs) else 0.0,
               fire_share_of_moving=float(R.fire.mean()) if len(R) else 0.0, blocked_not_free=int((rs.moving & ~rs.free).sum()),
               blocked_human=int((~rs.moving).sum()), windows_with_fire=int(rs[rs.fire].scene.nunique()), static_fires=int(St.fire.sum()))
    res["pass_"] = res["rate"] >= 0.5
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    raise SystemExit(0 if res["pass_"] else 2)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("diag")
    q.add_argument("--pair", action="append", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=16)
    q = sp.add_parser("pool")
    q.add_argument("--out", required=True)
    q = sp.add_parser("ratchet")
    q.add_argument("--clear", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    dict(diag=cmd_diag, pool=cmd_pool, ratchet=cmd_ratchet)[a.cmd](a)


if __name__ == "__main__":
    main()
