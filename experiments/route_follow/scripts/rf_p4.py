#!/usr/bin/env python3
"""Turning school P4: demonstrations on the policy's OWN junction states, searched by branching in the AlpaSim PAI simulator
(prereg experiments/route_follow/plans/2026-10-14-p4-self-branch-prereg.md; runner rf_p4.sh; driver experiments/onpol_pai/lib/onp_log_driver.py
ONP_CELLS with experiments/onpol_pai/lib/onp_p4.py). RSG-L18-s0 drives each scene from its start; at the handover h (route station within
D m of the strict turn entry, D in 40 / 30 / 20 / 10 / 0) the served plan is pushed (lateral gain g for 3 s) and / or held to a
curvature-limited speed profile to v_e or 0.8 v_e until exit + 10 m; the rollout runs to its end.

  plan   (AlpaSim image / venv)  rf_p4.py plan --scenes <ids> --roots <run roots with these scenes> --out <plan.json> [--v-e 5.0]
         Per scene, from the logged rig path: AlpaSim's MAP route generated along the log (RouteGenerator.create(rig path, map, MAP, 40 m),
         generate_route at the log's 0.5 s poses: the route a log-following ego gets), its strict turn (rf_pai_follow.find_turn: >= 60 deg
         within 30 m, entry / exit at 10 / 90 % of the change, exit + 10 m inside the log), the route on a 0.5 m grid with its curvature
         (heading over a +-2 m chord, smoothed over 2 m), the human's speed at the entry (rf_pai_follow.speed_at), the log decision of
         entry - D (fallback handovers: k = round(10 t_log)), and P2's turn points on the log (rf_branch.turn_points: the completion target).
  cells  (plain python3)  rf_p4.py cells --plan <plan.json> --scene <id> --phase A|B [--reach <reach.json>] [--d 40,30,20,10,0]
         [--only <g1.25-raw,...>] --out <p4cells.json>   A = the identity cell only; B = from reach.json: per distinct handover the 8 non-identity
         self cells, per unreached distance the 9 log cells (switch at the log decision of entry - D). Distances whose handover decision equals
         a larger distance's are run once (`dup` in reach.json).
  reach  (AlpaSim image / venv)  rf_p4.py reach --plan <plan.json> --run <phase A run dir> --out <reach.json>
         From the identity rollout: its zero event (rf_branch's rule) and per D the first decision k >= 17 with route station within D m of
         the entry, ego within 10 m of the route and t0 before the zero event: reached (h = k) or unreached (fallback to the log).
  merge  (plain python3)  rf_p4.py merge --root <p4 root> --out <read.json>   the per-scene read.json files of a run into one
  read   (AlpaSim image / venv)  rf_p4.py read --plan <plan.json> --root <p4 root> [--scenes <ids>] --out <read.json>
         Per rollout of every <root>/<scene>/{A,B} run: cell, h, D, src, score, zero kind / time, rear-ended, P2's completion (rf_branch:
         station at or past the target within 2.5 m of the route before the zero event), ego-to-route distance and fed speed at the end of the
         lateral window (h + 30), realized speed at the strict entry, min speed inside [entry, exit], crawl (min speed between h and the entry
         < 0.5 v_e), where the rollout ended (station relative to the entry, reason).
  table  (plain python3)  rf_p4.py table --read <read.json> --plan <plan.json> [--md <md>] [--json <summary.json>]
         Scene x D x cell outcome table, per-D / per-profile / per-gain completion, coverage and the registered line.
  s1     (plain python3)  rf_p4.py s1 --run <run dir> --ref <reference read root> [--scenes ...]  served plans and tracks of every identity
         session against the stored reference rollout of its scene (bit-identical gate)
  s2     (plain python3)  rf_p4.py s2 --plan <plan.json> --run <phase B run dir>   per P4 session: h recomputed from the logged stations,
         lateral window length, the profile's served poses recomputed from the logged raw plan / station / speed (tolerance 0.05 m), the
         release at exit + 10 m, realized vs profile speed at the entry.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[1] / "onpol_pai" / "lib"
sys.path[:0] = [str(HERE), str(LIB)]
import onp_p4 as Q  # noqa: E402

DS = (40, 30, 20, 10, 0)
GAINS, LONS = (1.0, 1.25, 1.5), ("raw", "ve", "ve08")
FGT_K, GRID, CHORD = 17, 0.5, 2.0
# v_e per topo class = the human entry-speed median on the 36 strict PAI turns (results/p3_branch/shape.md, shape.json entry_targets)
V_E = {"straight_exit": 4.6, "curve_single": 3.9, "straight_road": 5.3, "no_straight_exit": 4.6}
V_E_OTHER, V_GATE_MARGIN, MIN_CLASS_N = 4.6, 1.0, 5
SHAPE = HERE.parents[0] / "results" / "p3_branch" / "shape.json"


def gate_of(topo, J):
    """Shape gate thresholds of a topo class (rf_shape's rule: the class p95 when the class has >= 5 human turns, else all turns' p95;
    lateral acceleration: all turns' p95; entry speed: class v_e + 1 m/s)."""
    n = J["counts"]["topo"].get(topo, 0)
    b = J["band_topo"][topo] if n >= MIN_CLASS_N else J["band_all"]
    v_e = V_E.get(topo, V_E_OTHER)
    return dict(kmax=b["kmax"][1], dk=b["dk"][1], alat=J["band_all"]["alat"][1], v_entry=v_e + V_GATE_MARGIN, src="class" if n >= MIN_CLASS_N else "all")


def tag_of(D, src, g, lon):
    return "id" if D is None else f"D{D}-{src}-g{g:g}-{lon}"


def branch_poses_fn():
    """onp_log_driver.branch_poses, taken from the instrument's source (the module itself needs the driver image)."""
    src = (LIB / "onp_log_driver.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "branch_poses")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "onp_log_driver.py", "exec"), ns)
    return ns["branch_poses"]


# ---------------------------------------------------------------- plan (AlpaSim image)
def cfg_of(roots):
    out = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "sim", "aggregate", "results-summary.json"), recursive=True):
            for r in json.load(open(f))["rollouts"]:
                out.setdefault(r["clipgt_id"], f.replace("aggregate/results-summary.json", "generated-user-config-0.yaml"))
    return out


def route_grid(R):
    """Route polyline (m, 2) -> 0.5 m grid u, x, y and curvature (heading over a +-CHORD m chord, smoothed over 2 m)."""
    import rf_pai_follow as F
    S = F.stations(R)
    u = np.arange(0.0, S[-1], GRID)
    at = lambda q: np.c_[np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])]  # noqa: E731
    d = at(np.minimum(u + CHORD, S[-1])) - at(np.maximum(u - CHORD, 0.0))
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = np.gradient(hd, GRID)
    n = int(round(2.0 / GRID))
    k = np.convolve(np.r_[[k[0]] * n, k, [k[-1]] * n], np.ones(2 * n + 1) / (2 * n + 1), "valid")
    xy = at(u)
    return u, xy, k


def plan(a):
    import logging
    import rf_branch as B
    import rf_pai_follow as F
    import rf_tax as X
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    logging.disable(logging.WARNING)
    scenes = [x.split()[0] for x in open(a.scenes) if x.strip() and not x.startswith("#")]
    cfg, out, bad = cfg_of(a.roots), {}, {}
    J = json.load(open(a.shape))
    for sc in scenes:
        if sc not in cfg:
            bad[sc] = "no run with its config"
            continue
        ds = X.loader(cfg[sc]).get_data_source(sc)
        tr = ds.rig.trajectory
        T, P, Y = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.asarray(tr.yaws, float)
        g = RouteGenerator.create(P, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
        for i in range(0, len(T), max(1, int(round(0.5e6 / max(np.median(np.diff(T)), 1.0))))):
            g.generate_route(int(T[i]), Pose(P[i], np.array([0.0, 0.0, np.sin(Y[i] / 2), np.cos(Y[i] / 2)])))
        R = F.dedup(np.asarray(g.route_polyline_in_local.points, float)[:, :2])
        S = F.stations(R)
        lt, _ = F.track(R, S, P[:, :2])
        s0, s_log = float(lt[0]), float(lt.max())
        s, h = F.heading(R, S)
        tt = F.find_turn(s, h, s0, s_log)
        if tt is None or tt[1] + F.PAST > s_log:
            bad[sc] = "no strict turn inside the log" if tt is None else f"exit + 10 m beyond the log ({tt[1]:.0f} + 10 > {s_log:.0f})"
            continue
        ent, ext, th = tt
        vlog = np.r_[0.0, np.hypot(*np.diff(P[:, :2], axis=0).T) / np.maximum(np.diff(T) / 1e6, 1e-3)]
        vlog = np.convolve(np.r_[vlog[:2], vlog, vlog[-2:]], np.ones(5) / 5, "valid")
        # topo class with the human as the ego (rf_shape.human_and_rsg's call of rf_pai_follow.topo)
        lanes = {str(l.id): dict(c=np.asarray(l.center.points, float)[:, [0, 1, 3]].astype(np.float32), nx=sorted(map(str, l.next_lanes)),
                                 pv=sorted(map(str, l.prev_lanes)), al=sorted(map(str, l.adj_lanes_left)), ar=sorted(map(str, l.adj_lanes_right)))
                 for l in ds.map.lanes}
        lt2, llat = F.track(R, S, P[:, :2])
        i_ent = int(np.flatnonzero(lt2 >= ent)[0])
        lyaw = np.unwrap(np.arctan2(*np.gradient(P[:, :2], axis=0)[:, ::-1].T))
        fake = dict(ctrl=dict(t=T, x=P[:, 0], y=P[:, 1], yaw=lyaw, vx=vlog))
        z = F.topo(dict(scene=sc, lanes=lanes, log=dict(T=T, P=P[:, :2])), fake, R, S, s, h, P[:, :2], T, lt2, llat, lt2, vlog, float(T[i_ent]), s_log)
        topo = z.get("topo") if abs(z.get("z_entry", -1e9) - ent) < 1e-6 else "mismatch"
        u, xy, k = route_grid(R)
        keep = u <= min(u[-1], ext + F.PAST + 100.0)
        t_at = lambda st: float((T[np.flatnonzero(lt >= st)[0]] - T[0]) * 1e-6) if (lt >= st).any() else None  # noqa: E731
        kl = {str(D): max(int(round(10 * t_at(max(ent - D, s0)))), 0) for D in DS}
        s2, y2, t2 = B.arc_profile(P[:, :2], Y, T)
        p2 = B.turn_points(s2, y2, t2)
        if p2 is None or p2["t_target"] is None:
            bad[sc] = "P2's completion target (log turn exit + 10 m) outside the log"
            continue
        out[sc] = dict(side="L" if th > 0 else "R", turn_deg=round(float(np.degrees(th)), 1), s0=round(s0, 2), s_log=round(s_log, 2),
                       s_entry=round(ent, 2), s_exit=round(ext, 2), s_release=round(ext + F.PAST, 2), topo=topo, v_e=V_E.get(topo, V_E_OTHER),
                       gate=gate_of(topo, J),
                       v_log_entry=round(F.speed_at(T, lt, vlog, ent), 3), v_log={str(D): round(F.speed_at(T, lt, vlog, ent - D), 3) for D in (40, 20)},
                       k_log=kl, p2=p2, u=np.round(u[keep], 2).tolist(), x=np.round(xy[keep, 0], 3).tolist(), y=np.round(xy[keep, 1], 3).tolist(),
                       kappa=np.round(k[keep], 5).tolist(), kappa_max_inside=round(float(np.abs(k[(u >= ent) & (u <= ext + F.PAST)]).max()), 4))
        o = out[sc]
        print(sc[7:15], o["side"], o["turn_deg"], o["topo"], "v_e", o["v_e"], "entry", o["s_entry"], "exit", o["s_exit"], "s0", o["s0"], "v_log_entry", o["v_log_entry"],
              "k_log", kl, "p2", None if p2 is None else (p2["ent"], p2["target"]), flush=True)
    json.dump(dict(v_e=V_E, v_e_other=V_E_OTHER, shape=str(a.shape), ds=DS, scenes=out, bad=bad), open(a.out, "w"), separators=(",", ":"))
    print(len(out), "scenes,", len(bad), "without a usable strict turn:", bad)


# ---------------------------------------------------------------- cells / reach
def route_fields(p):
    return {k: p[k] for k in ("u", "x", "y", "kappa", "s_entry", "s_exit", "s_release", "v_e")}


def cells(a):
    PL = json.load(open(a.plan))["scenes"]
    sc, p = a.scene, PL[a.scene]
    if a.phase == "A":
        cl = [dict(tag="id", D=None, src="self", g=1.0, lon="raw", switch=0)]
    else:
        rc = json.load(open(a.reach))
        cl = []
        for D in [int(x) for x in a.d.split(",")]:
            r = rc["d"][str(D)]
            if r.get("dup") is not None:
                continue
            src = "self" if r["reached"] else "log"
            for g in GAINS:
                for lon in LONS:
                    if src == "self" and g == 1.0 and lon == "raw":
                        continue                                 # = the identity rollout (phase A)
                    cl.append(dict(tag=tag_of(D, src, g, lon), D=D, src=src, g=g, lon=lon, switch=0 if src == "self" else max(int(p["k_log"][str(D)]), FGT_K)))
        if a.only:                                           # cell suffixes g<gain>-<lon>, whatever the source
            keep = a.only.split(",")
            cl = [c for c in cl if any(c["tag"].endswith("-" + o) for o in keep)]
    json.dump(dict(scenes={sc: route_fields(p)}, cells={sc: cl}), open(a.out, "w"), separators=(",", ":"))
    print(sc[7:15], a.phase, len(cl), "cells:", " ".join(c["tag"] for c in cl))


def drive_records(run, scenes=None):
    """{session: [drive records sorted by k]} of a run dir (driver/drive.jsonl), optionally only these scenes."""
    R = {}
    for f in glob.glob(os.path.join(run, "driver", "drive.jsonl")):
        for line in open(f):
            if '"kind": "drive"' in line and (scenes is None or line[line.find('"scene": "') + 10:][:43] in scenes):
                r = json.loads(line)
                R.setdefault(r["session"], []).append(r)
    return {k: sorted(v, key=lambda r: r["k"]) for k, v in R.items()}


def rollouts(run):
    """[(sim dir, results row)] of a run dir."""
    f = os.path.join(run, "sim", "aggregate", "results-summary.json")
    return [(os.path.join(run, "sim"), r) for r in json.load(open(f))["rollouts"]] if os.path.exists(f) else []


def zero_time(sim, sc, r):
    """rf_branch's zero rule: the earliest at-fault collision / offroad / corridor flag that zeroed the score -> (kind, t_zero us)."""
    import rf_tax as X
    m, score = r.get("metrics") or {}, float(r.get("score") or 0)
    ser, _ = X.series(sim, sc, r["rollout_id"])
    if ser is None:
        return None, None, None
    tf = {"collision": np.nanmin([X.first_pos(ser, "collision_front"), X.first_pos(ser, "collision_lateral")]),
          "offroad": X.first_pos(ser, "offroad"), "corridor": X.first_pos(ser, "left_corridor_laterally")}
    flag = {"collision": m.get("collision_at_fault"), "offroad": m.get("offroad"), "corridor": m.get("left_corridor_laterally")}
    c = {k: v for k, v in tf.items() if float(flag[k] or 0) > 0 and np.isfinite(v)} if score == 0 else {}
    zt = (min(c, key=c.get) if c else "other") if score == 0 else ""
    return zt, (c[zt] if c else np.inf), ser


def reach(a):
    PL = json.load(open(a.plan))["scenes"]
    recs, rows = drive_records(a.run), rollouts(a.run)
    assert len(rows) == 1, f"{a.run}: {len(rows)} rollouts, phase A has one"
    sim, r = rows[0]
    sc, p = r["clipgt_id"], PL[r["clipgt_id"]]
    zt, t_zero, _ = zero_time(sim, sc, r)
    rr = [x for x in recs[r["rollout_id"]] if "p4" in x]
    out = dict(scene=sc, rollout=r["rollout_id"], score=float(r.get("score") or 0), zero=zt, t_zero=None if not np.isfinite(t_zero or np.inf) else t_zero,
               n_dec=len(rr), sigma_last=rr[-1]["p4"]["sigma"] if rr else None, d={})
    seen = {}
    for D in DS:
        h = next((x["k"] for x in rr if x["k"] >= FGT_K and p["s_entry"] - x["p4"]["sigma"] <= D and x["p4"]["d_route"] <= Q.FAR
                  and x["t0"] < (t_zero if t_zero is not None else np.inf)), None)
        key = ("self", h) if h is not None else ("log", max(int(p["k_log"][str(D)]), FGT_K))
        out["d"][str(D)] = dict(reached=h is not None, h=h, k_eff=key[1], dup=seen.get(key))
        seen.setdefault(key, D)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(sc[7:15], "score", out["score"], out["zero"] or "-", {D: v["h"] if v["reached"] else f"log@{v['k_eff']}" for D, v in out["d"].items()},
          "dups", {D: v["dup"] for D, v in out["d"].items() if v["dup"] is not None})


# ---------------------------------------------------------------- smoke gates
def s1(a):
    """Identity sessions of --run against the reference run(s) of their scene: served plan at every common k, controller track, score."""
    import csv
    def track(sim, rid):
        f = os.path.join(sim, "controller", f"alpasim_controller_{rid}.csv")
        return np.array([[float(x["x"]), float(x["y"])] for x in csv.DictReader(open(f))]) if os.path.exists(f) else np.zeros((0, 2))
    ref, refrows = {}, {}
    mine = {r["clipgt_id"] for _, r in rollouts(a.run)}
    for run in glob.glob(os.path.join(a.ref, "runs", "*")) + [a.ref]:
        if not any(r["clipgt_id"] in mine for _, r in rollouts(run)):
            continue
        for sess, rr in drive_records(run, mine).items():
            if rr:
                ref[rr[0]["scene"]] = rr
        for sim, r in rollouts(run):
            refrows[r["clipgt_id"]] = (sim, r)
    srv = lambda r: np.array(r.get("poses_branch", r["poses"]))  # noqa: E731
    rows = {r["rollout_id"]: (sim, r) for sim, r in rollouts(a.run)}
    n_ok = n = 0
    for sess, rr in sorted(drive_records(a.run).items(), key=lambda x: x[1][0]["scene"]):
        sc = rr[0]["scene"]
        tagset = {x.get("p4", {}).get("tag") for x in rr}
        B = {x["k"]: x for x in ref.get(sc, []) if "poses" in x}
        ks = [x["k"] for x in rr if "poses" in x and x["k"] in B]
        d = np.array([np.abs(srv(x) - srv(B[x["k"]])).max() for x in rr if x["k"] in B and "poses" in x]) if ks else np.array([np.inf])
        (sa, ra), (sb, rb) = rows.get(sess, (None, None)), refrows.get(sc, (None, None))
        ta, tb = (track(sa, ra["rollout_id"]) if ra else np.zeros((0, 2))), (track(sb, rb["rollout_id"]) if rb else np.zeros((0, 2)))
        m = min(len(ta), len(tb))
        e = float(np.hypot(*(ta[:m] - tb[:m]).T).max()) if m else float("nan")
        ok = bool(len(ks) and d.max() == 0.0 and len(ta) == len(tb) and e == 0.0 and ra and rb and ra.get("score") == rb.get("score"))
        n += 1
        n_ok += ok
        print(f"{sc[7:15]} session {sess[:8]} cell {','.join(sorted(map(str, tagset)))}: decisions {len(ks)} (ref {len(B)}), plan max|diff| {d.max():.6f}; "
              f"track rows {len(ta)}/{len(tb)} max pos diff {e:.6f}; score {ra and ra.get('score')} / {rb and rb.get('score')}: {'IDENTICAL' if ok else 'DIFFERENT'}")
    print(f"S1: {n_ok} / {n} identity sessions bit-identical to the reference")


def s2(a):
    """Per P4 session of --run: the handover, the lateral window, the profile's served poses recomputed, release, entry speed."""
    bp = branch_poses_fn()
    PL = json.load(open(a.plan))["scenes"]
    rows = {r["rollout_id"]: (sim, r) for sim, r in rollouts(a.run)}
    import csv
    for sess, rr in sorted(drive_records(a.run).items(), key=lambda x: x[1][0]["p4"]["tag"] if "p4" in x[1][0] else ""):
        rr = [x for x in rr if "p4" in x and "poses" in x]
        if not rr:
            continue
        sc, c = rr[0]["scene"], rr[0]["p4"]
        p = PL[sc]
        R = Q.Route(route_fields(p))
        # station recomputed from the logged anchors (rounded to 0.1 mm) with the same rule
        hint, sg = None, []
        for x in rr:
            hint, dd = R.station(x["anchor"][:2], hint)
            sg.append((hint, dd))
        dsig = max(abs(s_ - x["p4"]["sigma"]) for (s_, _), x in zip(sg, rr))
        h_log = next((x["p4"]["h"] for x in rr if x["p4"]["h"] is not None), None)
        if c["src"] == "self":
            h_re = next((x["k"] for x, (s_, dd) in zip(rr, sg) if x["k"] >= FGT_K and c["D"] is not None and p["s_entry"] - s_ <= c["D"] and dd <= Q.FAR), None)
        else:
            h_re = max(rr[0]["switch"], FGT_K) if c["D"] is not None else None
        on = [x["k"] for x in rr if x["p4"]["lat"] or x["p4"]["lon"]]
        lat = [x["k"] for x in rr if x["p4"]["lat"]]
        lon = [x["k"] for x in rr if x["p4"]["lon"]]
        g = {"D": c["D"]}
        tag = c["tag"]
        gg = float(tag.split("-g")[1].split("-")[0]) if "-g" in tag else 1.0
        lonk = tag.rsplit("-", 1)[1] if tag != "id" else "raw"
        err = 0.0
        sk_err = 0.0
        for x in rr:
            q = x["p4"]
            if not (q["lat"] or q["lon"]):
                if "poses_branch" in x:
                    err = max(err, 99.0)
                continue
            sk = 1.0
            if q["lon"]:
                pr = Q.profile(R, q["sigma"], q["v_meas"], p["v_e"] * (0.8 if lonk == "ve08" else 1.0), 0.8 * p["v_e"])
                sk = Q.scale(pr["arc"], Q.plan_arc(x["poses"]))
                sk_err = max(sk_err, abs(sk - q["s_k"]))
            srv = bp(np.array(x["poses"]), gg if q["lat"] else 1.0, sk)
            err = max(err, float(np.abs(srv[:, :2] - np.array(x["poses_branch"])[:, :2]).max()))
        rel = next((x["k"] for x in rr if x["p4"]["released"]), None)
        after_rel = [x for x in rr if rel is not None and x["k"] >= rel]
        rel_raw = all("poses_branch" not in x for x in after_rel if not x["p4"]["lat"])
        # realized speed at the entry (controller trace projected on the route) vs the profile speed at the entry planned at the last
        # decision before the ego reaches the entry
        sim, r = rows.get(sess, (None, None))
        v_ent = v_prof = None
        if r is not None:
            f = os.path.join(sim, "controller", f"alpasim_controller_{sess}.csv")
            C = list(csv.DictReader(open(f))) if os.path.exists(f) else []
            hint2, st, vx, tt = None, [], [], []
            for z in C:
                hint2, _ = R.station((float(z["x"]), float(z["y"])), hint2)
                st.append(hint2); vx.append(float(z["vx"])); tt.append(int(float(z["timestamp_us"])))
            st, vx = np.array(st), np.array(vx)
            i = np.flatnonzero(st >= p["s_entry"])
            if len(i):
                v_ent = float(vx[i[0]])
                before = [x for x in rr if x["p4"]["lon"] and x["p4"]["sigma"] < p["s_entry"] and x["p4"].get("v_ent_prof") is not None]
                if before:
                    v_prof = before[-1]["p4"]["v_ent_prof"]
                    g["s_k_last_before_entry"] = before[-1]["p4"]["s_k"]
        g.update(src=c["src"], h_logged=h_log, h_recomputed=h_re, transform_on=f"{on[0]}..{on[-1]} ({len(on)})" if on else "never",
                 first_on_equals_h=bool(on and on[0] == h_log == h_re), lateral_decisions=len(lat), lateral_contiguous=bool(lat and lat[-1] - lat[0] + 1 == len(lat)),
                 lon_decisions=len(lon), released_at_k=rel, sigma_at_release=next((x["p4"]["sigma"] for x in rr if x["p4"]["released"]), None),
                 s_release=p["s_release"], raw_after_release=rel_raw, max_station_recompute_diff_m=round(dsig, 5),
                 max_served_pose_diff_m=round(err, 5), max_s_k_diff=round(sk_err, 6), v_entry_realized=v_ent, v_entry_profile=v_prof,
                 score=r and r.get("score"))
        print(f"{sc[7:15]} {tag}: {json.dumps(g)}")


# ---------------------------------------------------------------- read (AlpaSim image)
def read(a):
    import rf_branch as B
    import rf_pai_follow as F
    import rf_shape as SH
    import rf_tax as X
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    PL = json.load(open(a.plan))["scenes"]
    want = set(x.split()[0] for x in open(a.scenes) if x.strip()) if a.scenes else None
    rows = []
    for sdir in sorted(glob.glob(os.path.join(a.root, "clipgt-*"))):
        sc = os.path.basename(sdir)
        if sc not in PL or (want and sc not in want):
            continue
        p = PL[sc]
        Rg, Sg = np.c_[p["x"], p["y"]], np.asarray(p["u"], float)
        rc = json.load(open(os.path.join(sdir, "reach.json"))) if os.path.exists(os.path.join(sdir, "reach.json")) else None
        ident = {}                                           # phase A's identity plans by k: a self branch must serve them before h
        for run in sorted(glob.glob(os.path.join(sdir, "*-a")) + glob.glob(os.path.join(sdir, "*-b"))):   # lower case: compose project names
            ph = run[-1].upper()
            recs = drive_records(run)
            if ph == "A":
                ident = {x["k"]: x["poses"] for v in recs.values() for x in v if "poses" in x and x.get("p4", {}).get("tag") == "id"}
            for sim, r in rollouts(run):
                rid = r["rollout_id"]
                rr = [x for x in recs.get(rid, []) if "p4" in x]
                if not rr:
                    continue
                q0 = rr[0]["p4"]
                h = next((x["p4"]["h"] for x in rr if x["p4"]["h"] is not None), None)
                row = dict(scene=sc, side=p["side"], phase=ph, rid=rid, tag=q0["tag"], D=q0["D"], src=q0["src"], h=h,
                           score=float(r.get("score") or 0), n_dec=len(rr), wait_s=q0.get("wait_s"))
                if ph == "B" and q0["src"] == "self" and ident:   # determinism: plans before h equal phase A's at the same k
                    pre = [x for x in rr if h is not None and x["k"] < h and x["k"] in ident]
                    row.update(pre_h_n=len(pre), pre_h_diff=max((float(np.abs(np.array(x["poses"]) - np.array(ident[x["k"]])).max()) for x in pre), default=None))
                C = X.ctrl(sim, rid)
                zt, t_zero, ser = zero_time(sim, sc, r)
                if C is None or ser is None:
                    rows.append(dict(row, done=None, note="no controller trace / metrics"))
                    continue
                m = r.get("metrics") or {}
                ds = X.loader(os.path.join(sim, "generated-user-config-0.yaml")).get_data_source(sc)
                tr = ds.rig.trajectory
                Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
                L = X.dedupe(Pg[:, :2])
                xy = np.c_[C["x"], C["y"]]
                st, dl = B.station(L, xy)
                gr = RouteGenerator.create(Pg, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
                try:
                    for i in range(min(4, len(C["t"]) - 1), len(C["t"]), 5):
                        z = float(np.interp(C["t"][i], Tg, Pg[:, 2]))
                        gr.generate_route(int(C["t"][i]), Pose(np.array([C["x"][i], C["y"][i], z]),
                                                               np.array([0.0, 0.0, np.sin(C["yaw"][i] / 2), np.cos(C["yaw"][i] / 2)])))
                    R, rerr = X.dedupe(np.asarray(gr.route_polyline_in_local.points, float)[:, :2]), 0
                except Exception:  # noqa: BLE001
                    R, rerr = L, 1
                dr = X.poly_proj(R, xy)[0]
                t = np.asarray(C["t"], float)
                ok = t < t_zero
                p2 = p["p2"]
                hit = np.where((st >= p2["target"]) & ok & (dr <= B.ROUTE_TOL))[0]
                # P4 measures on the plan route (strict entry / exit), with rf_shape's operators (results/p3_branch/shape.md)
                sp, _ = F.track(Rg, Sg, xy)
                shp = SH.track_shape(t, xy, C["vx"], sp, p["s_entry"] - SH.WIN_PRE, p["s_exit"], t_zero)
                spd = SH.speeds(t, sp, C["vx"], p["s_entry"], p["s_exit"], t_zero)
                v = SH.box5(C["vx"])
                th = next((x["t0"] for x in rr if x["k"] == h), None) if h is not None else None
                pre = ok & (sp < p["s_entry"]) & (t >= (th if th is not None else np.inf))
                gt = p["gate"]
                gk = dict(kmax=bool(shp["kmax"] <= gt["kmax"]), dk=bool(shp["dk"] <= gt["dk"]), alat=bool(shp["alat"] <= gt["alat"]),
                          v_entry=bool(np.isfinite(spd["v_0"]) and spd["v_0"] <= gt["v_entry"]))
                wk = next((x for x in rr if h is not None and x["k"] == h + Q.LAT_N), None)
                iw = int(np.argmin(np.abs(t - wk["t0"]))) if wk else None
                rear = float(m.get("collision_rear") or 0) > 0
                last = int(np.flatnonzero(ok)[-1]) if ok.any() else 0
                row.update(zero=zt, t_zero=(t_zero - t[0]) * 1e-6 if np.isfinite(t_zero) else None, route_err=rerr, rear=rear,
                           done=bool(len(hit)), t_tgt=float((t[hit[0]] - t[0]) * 1e-6) if len(hit) else None,
                           s_short=float(p2["target"] - st[ok].max()) if ok.any() else None,
                           end=("zero" if np.isfinite(t_zero) else "rear" if rear else "end"),
                           end_rel_entry=round(float(sp[last] - p["s_entry"]), 2), end_t=round(float((t[last] - t[0]) * 1e-6), 2),
                           v_entry=None if not np.isfinite(spd["v_0"]) else round(float(spd["v_0"]), 3),
                           v_min_inside=None if not np.isfinite(spd["v_min"]) else round(float(spd["v_min"]), 3),
                           v_min_pre=round(float(v[pre].min()), 3) if pre.any() else None,
                           **{f"shape_{k}": (None if not np.isfinite(shp[k]) else round(float(shp[k]), 4)) for k in ("kmax", "alat", "dk", "dec")},
                           gate_parts=gk, gate=bool(len(hit)) and all(gk.values()),
                           crawl=bool(pre.any() and v[pre].min() < 0.5 * p["v_e"]),
                           crawl_inside=bool(np.isfinite(spd["v_min"]) and spd["v_min"] < 0.5 * p["v_e"]),
                           w_droute=round(float(dr[iw]), 2) if iw is not None else None, w_v=round(10.0 * math.hypot(*wk["ego"][4:6]), 2) if wk else None,
                           w_zeroed=bool(t[iw] >= t_zero) if iw is not None else None,
                           s_k_min=min((x["p4"]["s_k"] for x in rr if x["p4"].get("s_k") is not None), default=None),
                           reach=None if rc is None else rc["d"].get(str(q0["D"])) if q0["D"] is not None else None)
                rows.append(row)
                print(sc[7:15], row["tag"], row["h"], row["score"], row["zero"] or row["end"], "done" if row["done"] else "-", row["v_entry"], flush=True)
    json.dump(dict(plan=a.plan, root=a.root, rows=rows), open(a.out, "w"), indent=1, default=float)
    print(len(rows), "rollouts ->", a.out)


# ---------------------------------------------------------------- merge / table (plain python3)
def merge(a):
    rows, miss = [], []
    for sd in sorted(glob.glob(os.path.join(a.root, "clipgt-*"))):
        f = os.path.join(sd, "read.json")
        if os.path.exists(f):
            rows += json.load(open(f))["rows"]
        else:
            miss.append(os.path.basename(sd))
    json.dump(dict(root=a.root, rows=rows, missing=miss), open(a.out, "w"), indent=1, default=float)
    print(len(rows), "rollouts,", len(miss), "scene dirs without read.json:", miss)


def table(a):
    D = json.load(open(a.read))
    PL = json.load(open(a.plan))["scenes"]
    rows = [r for r in D["rows"] if r.get("done") is not None]
    by = {}
    for r in rows:
        by.setdefault(r["scene"], []).append(r)
    ab = {"collision": "coll", "offroad": "offr", "corridor": "corr", "other": "oth", "": ""}
    f = lambda r: "**done**" if r["done"] else (ab.get(r.get("zero") or "", r.get("zero")) or ("rear" if r["end"] == "rear" else r["end"]))  # noqa: E731
    cellnames = [f"g{g:g}-{lon}" for g in GAINS for lon in LONS]
    L = [f"# P4 self-branch read ({len(rows)} rollouts, {len(by)} scenes)", "",
         "`done` = turn completed (P2's rule). Otherwise the zero type (corr / offr / coll / oth) or rear (rear-ended before the target) / end.",
         "Cells g<gain>-<profile>: raw = the plan's own speed, ve = profile to v_e, ve08 = to 0.8 v_e. `id` = the policy alone (phase A).",
         "(log) = unreached distance, handover from the log (src=log). `=D<n>` = same handover decision as D<n> (run once).", ""]
    L += ["| scene | side | id | D | src | h | " + " | ".join(cellnames) + " |", "|:--|:--|:--|--:|:--|--:|" + "--:|" * len(cellnames)]
    cov = {}
    vthr = {}
    for sc in sorted(by, key=lambda s: (PL[s]["side"], s)):
        rr = by[sc]
        p = PL[sc]
        vthr[sc] = p["gate"]["v_entry"]                         # class v_e (human entry median) + 1 m/s
        idr = next((r for r in rr if r["tag"] == "id"), None)
        reach = json.load(open(os.path.join(D["root"], sc, "reach.json"))) if os.path.exists(os.path.join(D["root"], sc, "reach.json")) else {"d": {}}
        cov[sc] = {}
        for Dd in DS:
            rd = reach["d"].get(str(Dd), {})
            dup = rd.get("dup")
            src = "self" if rd.get("reached") else "log"
            Dr = dup if dup is not None else Dd
            cc = {r["tag"].split(f"-{src}-")[1]: r for r in rr if r["D"] == Dr and r["src"] == src}
            if src == "self" and idr is not None:
                cc["g1-raw"] = idr
            hh = rd.get("h") if rd.get("reached") else rd.get("k_eff")
            L.append(f"| {sc[7:15]} | {p['side']} | {f(idr) if idr else '-'} | {Dd} | {src}{'' if dup is None else f' =D{dup}'} | {hh} | "
                     + " | ".join((f(cc[c]) if c in cc else "-") for c in cellnames) + " |")
            if dup is None:
                done = [c for c, r in cc.items() if r["done"] and r["tag"] != "id"]
                cov[sc][Dd] = dict(src=src, h=hh, done=done, done_slow=[c for c in done if (cc[c].get("v_entry") or 99) <= vthr[sc]],
                                   done_gate=[c for c in done if cc[c].get("gate")])
    # summaries
    n = len(cov)
    any_d = {Dd: sum(bool(c.get(Dd, {}).get("done")) for c in cov.values()) for Dd in DS}
    line_n = sum(sum(bool(v["done_slow"]) for v in c.values()) >= 2 for c in cov.values())
    line_self = sum(sum(bool(v["done_slow"]) for v in c.values() if v["src"] == "self") >= 2 for c in cov.values())
    gate2 = sum(sum(bool(v["done_gate"]) for v in c.values()) >= 2 for c in cov.values())
    gate1 = sum(any(v["done_gate"] for v in c.values()) for c in cov.values())
    any2 = sum(sum(bool(v["done"]) for v in c.values()) >= 2 for c in cov.values())
    any1 = sum(any(v["done"] for v in c.values()) for c in cov.values())
    idd = sum(bool(r["done"]) for r in rows if r["tag"] == "id")
    verdict = "PASS" if line_n >= 0.6 * n else "FAIL" if line_n <= 0.35 * n else "INCONCLUSIVE"
    L += ["", "## Coverage", "",
          f"- turns: {n}; identity (the policy alone) completes {idd} / {sum(r['tag'] == 'id' for r in rows)}",
          f"- turns with >= 1 completing branch cell at some distance: {any1} / {n}; at >= 2 distinct handovers: {any2} / {n}",
          f"- determinism: self branches whose plans before h differ from phase A's identity at the same k: "
          f"{sum((r.get('pre_h_diff') or 0) > 0 for r in rows)} of {sum(r.get('pre_h_diff') is not None for r in rows)}",
          f"- **line: turns with completing cells at >= 2 distinct handovers with realized entry speed <= v_e + 1 m/s: {line_n} / {n} = "
          f"{100 * line_n / max(n, 1):.0f} % -> {verdict}** (pass >= 60 %, fail <= 35 %); self handovers only: {line_self} / {n}",
          f"- row yield (completing cells that pass the full shape gate: class p95 kappa / dkappa, a_lat <= human p95, entry speed <= v_e + 1): "
          f"turns with >= 1: {gate1} / {n}; at >= 2 distinct handovers: {gate2} / {n}",
          "- per distance (turns with >= 1 completing cell): " + ", ".join(f"D{Dd} {any_d[Dd]}" for Dd in DS), ""]
    br = [r for r in rows if r["tag"] != "id"]
    def agg(sel, name):
        rr = [r for r in br if sel(r)]
        if not rr:
            return
        ve = [r["v_entry"] for r in rr if r["done"] and r.get("v_entry") is not None]
        L.append(f"| {name} | {len(rr)} | {sum(r['done'] for r in rr)} | {sum(bool(r.get('gate')) for r in rr)} | {sum(r['score'] == 0 for r in rr)} | {sum(r['rear'] for r in rr)} | "
                 f"{sum(r['crawl'] for r in rr)} | {sum(bool(r.get('crawl_inside')) for r in rr)} | {np.median(ve) if ve else float('nan'):.2f} | {np.mean([r['score'] for r in rr]):.3f} |")
    L += ["| cells | rollouts | done | done + gate | zeros | rear-ended | crawl (approach) | crawl inside | median entry speed (done) | mean score |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for lon in LONS:
        agg(lambda r, lon=lon: r["tag"].endswith(f"-{lon}"), f"profile {lon}")
    for g in GAINS:
        agg(lambda r, g=g: f"-g{g:g}-" in r["tag"], f"gain {g:g}")
    for Dd in DS:
        agg(lambda r, Dd=Dd: r["D"] == Dd, f"D{Dd}")
    agg(lambda r: r["src"] == "self", "src self")
    agg(lambda r: r["src"] == "log", "src log (fallback)")
    hv = [PL[sc]["v_log_entry"] for sc in cov]
    L += ["", f"- human entry speed (log, strict entry) median {np.median(hv):.2f} m/s over these turns; "
          f"fallback (src=log) distances: {sum(v['src'] == 'log' for c in cov.values() for v in c.values())} of {sum(len(c) for c in cov.values())}"]
    s = "\n".join(L)
    print(s)
    if a.md:
        Path(a.md).write_text(s + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(n=n, line=line_n, line_self=line_self, verdict=verdict, any1=any1, any2=any2, gate1=gate1, gate2=gate2, per_d=any_d, cov=cov), indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--scenes", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shape", default=str(SHAPE))
    p = sp.add_parser("cells")
    p.add_argument("--plan", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--phase", choices=("A", "B"), required=True)
    p.add_argument("--reach")
    p.add_argument("--d", default=",".join(map(str, DS)))
    p.add_argument("--only")
    p.add_argument("--out", required=True)
    p = sp.add_parser("reach")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("s1")
    p.add_argument("--run", required=True)
    p.add_argument("--ref", required=True)
    p = sp.add_parser("s2")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p = sp.add_parser("read")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--scenes")
    p.add_argument("--out", required=True)
    p = sp.add_parser("merge")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("table")
    p.add_argument("--read", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--md")
    p.add_argument("--json")
    a = ap.parse_args()
    {"plan": plan, "cells": cells, "reach": reach, "s1": s1, "s2": s2, "read": read, "merge": merge, "table": table}[a.cmd](a)
