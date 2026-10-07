"""Four directions, HUGSIM 64: failure buckets of P2H (P2H10-F-s0/s1) under `spec_plan_smooth`, sized by replacement oracles.

  extract  (box, .venv)  per-run event geometry and per-step traces from the stored runs -> $OUT/hugsim_events.csv,
                         hugsim_hd.csv (HD of every stored arm, for the oracles), hugsim_steps.csv.gz, hugsim_cases.json.gz
  report   (box or Mac)  buckets, oracles, mechanism shares, figures -> four_dirs/hugsim_*.csv, hugsim_tables.md, figs/hugsim_*.png

Runs: P2H10-F-s{0,1} spec_plan_smooth r0 (runs/op_parity/hugsim) + rr1 / rr2 (runs/bench/hugsim); P2H10-F-s{0,1} spec and exam r0
(same-scenario contrast); WA-JEPA exam (runs/hugsim-wajepa). HD for oracle (b): P2-F-s{0,1}, P2H10-F-s{0,1} x exam / spec / spec_plan /
spec_plan_smooth / spec_plan_mpc (r0) and the rr1 / rr2 repeats of spec and spec_plan_smooth, averaged over repeats per (arm, preset).

Geometry is HUGSIM's own: ego box and actor boxes per frame (data.pkl / infos.pkl, imu frame x fwd, y left), the recorded route
(runs/op_parity/hugsim/routes.json, camera xz -> imu (z, -x)), ground.ply (drivable ground points; DAC uses them) and scene.ply
(background points; bg_collision = > 100 points inside the ego box, which spans camera-y [y_ego, y_ego + 1.5], length 3.0, width 1.6).
Off route = the env's own rule: > 10 m from the nearest recorded camera pose.

BUCKET RULE (fixed before any score was read; unit = run, label per (arm, scenario) = modal label over its 3 repeats, ties -> r0):
  event      end in {bg_collision, off_route, fg_collision}; else 'stuck' (max_steps), 'complete' (no event).
  onset      for bg / off_route: the last step before the end with lateral distance to the route < 1.5 m (route departure
             onset); the end step if the distance never exceeded 1.5 m. fg: the end step.
  geometry   route heading psi(s) on the densified route; route curvature k_r(s) = (psi(s + 4 m) - psi(s - 4 m)) / 8 m;
             window W = [s_onset - 10 m, s_onset + 20 m]; R = 1 / max_W |k_r|, turn sign = sign of k_r at that max (+ left).
             straight: R >= 50 m; sharp: R < 15 m; wide: 15 <= R < 50 m.
  buckets    D1 sharp-turn edge = (bg | off_route) and sharp; D2 wide-turn edge = (bg | off_route) and wide;
             S  straight edge   = (bg | off_route) and straight (not a user direction, reported for completeness);
             D3 vehicle contact = fg (any geometry, sub-typed below).
  side       bg: centroid of the scene points inside the final ego box (lateral, ego frame); off_route: sign of the final signed
             lateral offset from the route. inside = same sign as the turn, outside = opposite.
  fg type    the actor box intersecting (else nearest to) the final ego box; dh = actor yaw - ego yaw, (lx, ly) actor centre in the
             ego frame, v_o actor speed from its box motion over the last step, lat3 = actor lateral motion towards the ego path
             over the last 3 s (route frame).
             rear_ended: lx < 0 and |dh| < 45 deg;  crossing: |dh| >= 60 deg (oncoming / crossing);
             cut_in: |dh| < 60, lx >= 0 and the actor moved >= 1.0 m laterally towards the ego over 3 s;
             lead_stopped: |dh| < 30, lx >= 0, |ly| < 1.8, v_o < 0.5;  lead_moving: same with v_o >= 0.5;
             side (ego drifts into a same-direction actor): the rest.  turning flag: route class sharp / wide at the event.
"""
import argparse
import gzip
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/op_parity/results/four_dirs"
DT, EGO_L, EGO_W, EGO_H = 0.25, 3.0, 1.6, 1.5
P2H = ["P2H10-F-s0", "P2H10-F-s1"]
ALL_ARMS = ["P2-F-s0", "P2-F-s1", "P2H10-F-s0", "P2H10-F-s1"]
PRESETS = ["exam", "spec", "spec_plan", "spec_plan_smooth", "spec_plan_mpc"]
REP_PRESETS = ["spec", "spec_plan_smooth"]
R_SHARP, R_STRAIGHT, DEV_ONSET = 15.0, 50.0, 1.5


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def ply(p):
    with open(p, "rb") as f:
        n = 0
        while True:
            ln = f.readline()
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[-1])
            if ln.startswith(b"end_header"):
                break
        return np.frombuffer(f.read(n * 24), "<f8").reshape(n, 3)


class Route:
    """Densified recorded route in the imu frame (x fwd, y left), arc length s, heading psi, curvature k(s)."""

    def __init__(self, xz):
        xz = np.asarray(xz, float)
        p = np.stack([xz[:, 1], -xz[:, 0]], 1)
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-3]
        self.p = p[keep]
        seg = np.diff(self.p, axis=0)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
        psi = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        self.psi = np.r_[psi, psi[-1]]
        self.L = self.s[-1]

    def heading(self, s):
        return np.interp(s, self.s, self.psi)

    def kappa(self, s, h=4.0):
        return (self.heading(np.asarray(s) + h) - self.heading(np.asarray(s) - h)) / (2 * h)

    def project(self, xy):
        """arc length and signed lateral offset (+ left) of points xy (n, 2)."""
        xy = np.atleast_2d(xy)
        d = ((self.p[None] - xy[:, None]) ** 2).sum(-1)
        i = d.argmin(1)
        j = np.clip(i, 0, len(self.p) - 2)
        t = self.p[j + 1] - self.p[j]
        tn = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        r = xy - self.p[j]
        along = (r * tn).sum(1)
        lat = tn[:, 0] * r[:, 1] - tn[:, 1] * r[:, 0]
        return self.s[j] + along, lat


class Scene:
    """Ground (drivable) and background point sets of one scenario, KD-trees on the imu xy plane."""

    def __init__(self, d):
        from scipy.spatial import cKDTree
        g = ply(d / "ground.ply")
        s = ply(d / "scene.ply")
        g, s = g[np.isfinite(g).all(1)], s[np.isfinite(s).all(1)]
        self.gxy = np.stack([g[:, 2], -g[:, 0]], 1)
        self.sxy, self.sy = np.stack([s[:, 2], -s[:, 0]], 1), s[:, 1]   # camera y (down)
        self.gt, self.st = cKDTree(self.gxy), cKDTree(self.sxy)

    def local(self, tree, xy, x, y, yaw, r):
        idx = tree.query_ball_point([x, y], r)
        q = xy[idx] - [x, y]
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array(idx, int), q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c

    def coverage(self, x, y, yaw):
        """HUGSIM DAC cell rule: share of the 2 x 2 footprint cells that hold a ground point."""
        _, lx, ly = self.local(self.gt, self.gxy, x, y, yaw, 1.8)
        if not len(lx):
            return 0.0
        cx = np.clip(np.floor((lx + EGO_L / 2) / (EGO_L / 2)), -1, 2)
        cy = np.clip(np.floor((ly + EGO_W / 2) / (EGO_W / 2)), -1, 2)
        ok = (np.abs(lx) < EGO_L / 2) & (np.abs(ly) < EGO_W / 2)
        return len(set(zip(cx[ok].astype(int), cy[ok].astype(int)))) / 4.0

    def contact(self, x, y, yaw, cam_y, margin=0.0):
        """background points inside the ego box (bg_collision_det geometry): count, centroid (lx, ly)."""
        idx, lx, ly = self.local(self.st, self.sxy, x, y, yaw, 1.9 + margin)
        yy = self.sy[idx] if len(idx) else np.zeros(0)
        m = (np.abs(lx) < EGO_L / 2 + margin) & (np.abs(ly) < EGO_W / 2 + margin) & (yy > cam_y) & (yy < cam_y + EGO_H)
        if not m.any():
            return 0, np.nan, np.nan
        return int(m.sum()), float(lx[m].mean()), float(ly[m].mean())


def plan_world(plan, x, y, yaw):
    """HUGSIM plan (x right, y fwd, ego frame, 0.5..3.0 s) -> imu world points (n, 2) and segment yaws."""
    p = np.asarray(plan, float)
    fwd, left = p[:, 1], -p[:, 0]
    c, s = np.cos(yaw), np.sin(yaw)
    w = np.stack([x + fwd * c - left * s, y + fwd * s + left * c], 1)
    q = np.r_[[[x, y]], w]
    d = np.diff(q, axis=0)
    yw = np.where(np.linalg.norm(d, axis=1) > 0.05, np.arctan2(d[:, 1], d[:, 0]), yaw)
    return w, yw


def plan_curv(w, yw, x, y, yaw):
    """plan curvature over 0.5-1.5 s from positions (heading change / arc length) and circle-through-origin at 1.5 s."""
    q = np.r_[[[x, y]], w]
    psi = np.unwrap(np.r_[yaw, yw])
    h05, h15 = (psi[1] + psi[2]) / 2, (psi[3] + psi[4]) / 2       # headings around the 0.5 s and 1.5 s points
    arc = np.linalg.norm(np.diff(q[1:4], axis=0), axis=1).sum()
    k_seg = (h15 - h05) / arc if arc > 1.0 else np.nan
    r = w[2] - [x, y]
    c, s = np.cos(yaw), np.sin(yaw)
    f, lft = r[0] * c + r[1] * s, -r[0] * s + r[1] * c
    d2 = f * f + lft * lft
    k_circ = 2 * lft / d2 if d2 > 1.0 else np.nan
    return k_seg, k_circ


def load_run(d, wajepa=False):
    infos = pickle.load(open(d / "infos.pkl", "rb"))
    fr = pickle.load(open(d / "data.pkl", "rb"))[0]["frames"]
    box = np.array([i["ego_box"] for i in infos], float)
    fbox = np.array([f["ego_box"] for f in fr], float)
    camy = np.array([i["ego_pos"][1] for i in infos], float)
    v = np.array([float(np.ravel(i["ego_velo"])[0]) for i in infos])
    n = len(infos)
    # state k = 0..n: infos[k] for k < n, the last frame (after the final action) for k = n
    X = np.r_[box[:, 0], fbox[-1, 0]]
    Y = np.r_[box[:, 1], fbox[-1, 1]]
    YAW = np.unwrap(np.r_[box[:, 6], fbox[-1, 6]])
    V = np.r_[v, v[-1]]
    CY = np.r_[camy, camy[-1]]
    objs = [i.get("obj_boxes", []) for i in infos] + [fr[-1]["obj_boxes"]]
    steps, plans = {}, [None] * n
    zf = d / "zs_steps.jsonl"
    if zf.exists() and not wajepa:
        recs = [json.loads(x) for x in open(zf)][1:]
        for r in recs:
            steps[r["step"]] = r
        for k in range(n):
            if k in steps and "plan" in steps[k]:
                plans[k] = plan_world(steps[k]["plan"], X[k], Y[k], YAW[k])
    else:   # WA-JEPA: HUGSIM's stored plan of frame k-1 starts from state k
        for k in range(1, n):
            t = np.array([p[:2] for p in fr[k - 1]["planned_traj"]["traj"]], float)
            q = np.r_[[[X[k], Y[k]]], t]
            dd = np.diff(q, axis=0)
            plans[k] = (t, np.where(np.linalg.norm(dd, axis=1) > 0.05, np.arctan2(dd[:, 1], dd[:, 0]), YAW[k]))
    return dict(X=X, Y=Y, YAW=YAW, V=V, CY=CY, objs=objs, steps=steps, plans=plans, n=n)


def box_poly(b):
    from shapely.geometry import Polygon
    x, y, _, w, l, _, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    pts = [(x + a * c - bb * s, y + a * s + bb * c) for a, bb in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))]
    return Polygon(pts)


def analyse(row, d, route, scene, wajepa=False):
    R = load_run(d, wajepa)
    X, Y, YAW, V, n = R["X"], R["Y"], R["YAW"], R["V"], R["n"]
    xy = np.stack([X, Y], 1)
    s, lat = route.project(xy)
    end = row["end"]
    # per-step series (k = 0..n-1: the action taken at state k)
    kexe = np.r_[wrap(np.diff(YAW)) / (np.maximum(V[:-1], 0.3) * DT)]
    # requested curvature: zs_steps kappa is + right (HUGSIM theta convention; corr with executed -0.8..-0.9) -> + left
    kreq = -np.array([R["steps"].get(k, {}).get("kappa", np.nan) if R["steps"] else np.nan for k in range(n)], float)
    kdes = np.full(n, np.nan)                     # after clip_curvature (sim.log op_ctrl `des`), same sign flip as kappa
    if (d / "sim.log").exists() and R["steps"]:
        oc = [json.loads(ln.split("op_ctrl ", 1)[1]) for ln in open(d / "sim.log", errors="ignore") if ln.startswith("op_ctrl {")]
        m = min(n, len(oc))
        kdes[:m] = [-float(o["des"]) for o in oc[:m]]
    kneed = np.full(n, np.nan)
    kseg = np.full(n, np.nan)
    kcirc = np.full(n, np.nan)
    cov = np.array([scene.coverage(X[k], Y[k], YAW[k]) for k in range(n + 1)])
    plan_cov_min = np.full(n, np.nan)
    plan_bg = np.zeros(n, int)
    plan_v13 = np.full(n, np.nan)
    for k in range(n):
        a = max(V[k], 1.0)
        kneed[k] = (route.heading(s[k] + 1.5 * a) - route.heading(s[k] + 0.5 * a)) / a
        if R["plans"][k] is not None:
            w, yw = R["plans"][k]
            kseg[k], kcirc[k] = plan_curv(w, yw, X[k], Y[k], YAW[k])
            plan_cov_min[k] = min(scene.coverage(w[j, 0], w[j, 1], yw[j]) for j in range(len(w)))
            plan_bg[k] = max(scene.contact(w[j, 0], w[j, 1], yw[j], R["CY"][k])[0] for j in range(len(w)))
            if len(w) >= 5:
                i1, i3 = 1, min(5, len(w) - 1)
                plan_v13[k] = np.linalg.norm(w[i3] - w[i1]) / (0.5 * (i3 - i1))
    ev = dict(row)
    ev.update(n_steps=n, v_end=float(V[-2]), lat_end=float(lat[-1]), s_end=float(s[-1]), route_len=float(route.L))
    # onset and route class
    if end in ("bg_collision", "off_route"):
        far = np.abs(lat) >= DEV_ONSET
        if far[-1]:
            ok = np.where(~far)[0]
            on = int(ok[-1]) if len(ok) else 0
        else:
            on = n
    else:
        on = n
    ev["onset"] = on
    ss = np.linspace(s[on] - 10, s[on] + 20, 61)
    kr = route.kappa(ss)
    j = int(np.abs(kr).argmax())
    Rr = 1 / max(abs(kr[j]), 1e-6)
    ev.update(s_onset=float(s[on]), R_route=float(Rr), turn_sign=int(np.sign(kr[j])), k_route_peak=float(kr[j]),
              dpsi_W=float(np.degrees(route.heading(ss[-1]) - route.heading(ss[0]))),
              rclass="sharp" if Rr < R_SHARP else ("wide" if Rr < R_STRAIGHT else "straight"), v_onset=float(V[min(on, n - 1)]))
    ev["a_lat_need"] = float(V[min(on, n - 1)] ** 2 / Rr)
    # side
    side_lat = np.nan
    if end == "bg_collision":
        cnt, clx, cly = scene.contact(X[-1], Y[-1], YAW[-1], R["CY"][-1])
        if cnt == 0:
            cnt, clx, cly = scene.contact(X[-1], Y[-1], YAW[-1], R["CY"][-1], margin=0.3)
        ev.update(bg_pts=cnt, bg_lx=clx, bg_ly=cly)
        side_lat = cly
    elif end == "off_route":
        side_lat = lat[-1]
    if np.isfinite(side_lat) and ev["rclass"] != "straight":
        ev["side"] = "inside" if np.sign(side_lat) == ev["turn_sign"] else "outside"
    elif np.isfinite(side_lat):
        ev["side"] = "left" if side_lat > 0 else "right"
    # approach window: 12 steps (3 s) before the onset up to the end
    a0, a1 = max(0, min(on, n) - 12), n
    A = slice(a0, a1)
    sg = ev["turn_sign"] or 1
    mv = V[:n] > 0.5
    pick = lambda arr: arr[A][mv[A]] if mv[A].any() else arr[A]   # noqa: E731
    ev.update(k_need=float(np.nanmean(sg * pick(kneed))) if n else np.nan, k_req=float(np.nanmean(sg * pick(kreq))) if R["steps"] else np.nan,
              k_des=float(np.nanmean(sg * pick(kdes))) if R["steps"] else np.nan,
              clip_frac=float(np.mean(np.abs(kdes[A]) < 0.9 * np.abs(kreq[A]) - 1e-4)) if R["steps"] else np.nan,
              k_exe=float(np.nanmean(sg * pick(kexe))), k_plan=float(np.nanmean(sg * pick(kseg))), k_plan_circ=float(np.nanmean(sg * pick(kcirc))),
              v_app=float(np.mean(V[A])), herr_end=float(np.degrees(wrap(YAW[-1] - route.heading(s[-1])))))
    # lag: first step in the approach where plan / requested / executed curvature exceeds half the peak route curvature (signed)
    thr = 0.5 * abs(ev["k_route_peak"])
    for nm, arr in (("plan", sg * kseg), ("req", sg * kreq), ("exe", sg * kexe), ("need", sg * kneed)):
        hit = np.where(np.nan_to_num(arr[A], nan=-9) > thr)[0]
        ev[f"t_in_{nm}"] = float((a0 + hit[0]) * DT) if len(hit) else np.nan
    # plan / execution on the road (HUGSIM DAC rule: coverage < 0.5 = off)
    for lagk, nm in ((8, "m2"), (4, "m1")):
        k = max(0, min(on, n - 1) - lagk)
        ev[f"plan_cov_{nm}"] = float(plan_cov_min[k])
        ev[f"plan_bg_{nm}"] = int(plan_bg[k])
    win = slice(max(0, min(on, n - 1) - 8), max(1, min(on, n - 1) - 1))
    ev["plan_off_any"] = bool(np.any((plan_cov_min[win] < 0.5) | (plan_bg[win] > 100))) if np.isfinite(plan_cov_min[win]).any() else np.nan
    offk = np.where(cov[: n + 1] < 0.5)[0]
    ev["t_exec_offroad"] = float(offk[0] * DT) if len(offk) else np.nan
    ev["frac_offroad"] = float((cov[:n] < 0.5).mean())
    # fg geometry
    if end == "fg_collision" and len(R["objs"][-1]):
        eb = [X[-1], Y[-1], 0, EGO_W, EGO_L, 0, YAW[-1]]
        ep = box_poly(eb)
        ob = np.array(R["objs"][-1], float)
        inter = [ep.intersection(box_poly(o)).area for o in ob]
        dist = [ep.distance(box_poly(o)) for o in ob]
        i = int(np.argmax(inter)) if max(inter) > 0 else int(np.argmin(dist))
        o = ob[i]
        c, sn = np.cos(YAW[-1]), np.sin(YAW[-1])
        q = o[:2] - [X[-1], Y[-1]]
        lx, ly = q[0] * c + q[1] * sn, -q[0] * sn + q[1] * c
        dh = float(np.degrees(wrap(o[6] - YAW[-1])))
        prev = R["objs"][-2] if len(R["objs"]) > 1 else []
        vo = float(np.linalg.norm(o[:2] - np.asarray(prev[i][:2])) / DT) if len(prev) > i else np.nan
        vo_fwd = float(((o[:2] - np.asarray(prev[i][:2])) @ [c, sn]) / DT) if len(prev) > i else np.nan
        k3 = max(0, n - 12)
        lat3 = np.nan
        if len(R["objs"][k3]) > i:
            _, l_now = route.project(o[None, :2])
            _, l_then = route.project(np.asarray(R["objs"][k3][i], float)[None, :2])
            _, le_now = route.project(xy[-1:])
            lat3 = float(abs(l_then[0] - le_now[0]) - abs(l_now[0] - le_now[0]))   # + = moved towards the ego
        if lx < 0 and abs(dh) < 45:
            ft = "rear_ended"
        elif abs(dh) >= 60:
            ft = "crossing"
        elif np.isfinite(lat3) and lat3 >= 1.0:
            ft = "cut_in"
        elif abs(dh) < 30 and abs(ly) < 1.8:
            ft = "lead_stopped" if (np.isfinite(vo) and vo < 0.5) else "lead_moving"
        else:
            ft = "side"
        part = "front" if lx > EGO_L / 2 else ("rear" if lx < -EGO_L / 2 else ("left" if ly > 0 else "right"))
        ev.update(fg_type=ft, fg_part=part, fg_lx=float(lx), fg_ly=float(ly), fg_dh=dh, fg_vo=vo, fg_vo_fwd=vo_fwd, fg_lat3=lat3,
                  fg_inter=float(inter[i]), fg_close=float(V[-2] - vo_fwd) if np.isfinite(vo_fwd) else np.nan,
                  fg_yawrate=float(np.degrees(np.mean(wrap(np.diff(YAW[-9:]))) / DT)))
        # plan speed before contact, executed speed / decel
        w3 = slice(max(0, n - 12), max(1, n - 2))
        ev.update(plan_v13_min=float(np.nanmin(plan_v13[w3])) if np.isfinite(plan_v13[w3]).any() else np.nan,
                  plan_v13_m1=float(plan_v13[max(0, n - 4)]), v_m1=float(V[max(0, n - 4)]), v_m3=float(V[max(0, n - 12)]),
                  dec_exe=float((V[max(0, n - 12)] - V[n - 1]) / (DT * min(12, n - 1))) if n > 1 else np.nan)
        # did the plan at -1.5 s .. -0.5 s keep clear of the actor (actor held at constant velocity)?
        clear = []
        for k in range(max(1, n - 6), max(1, n - 1)):
            if R["plans"][k] is None or len(R["objs"][k]) <= i or len(R["objs"][k - 1]) <= i:
                continue
            w, yw = R["plans"][k]
            ok_, okp = np.asarray(R["objs"][k][i], float), np.asarray(R["objs"][k - 1][i], float)
            vel = (ok_[:2] - okp[:2]) / DT
            hit = False
            for jj in range(min(len(w), 4)):
                ob_t = ok_.copy()
                ob_t[:2] = ok_[:2] + vel * 0.5 * (jj + 1)
                if box_poly([w[jj, 0], w[jj, 1], 0, EGO_W, EGO_L, 0, yw[jj]]).intersects(box_poly(ob_t)):
                    hit = True
                    break
            clear.append(not hit)
        ev["plan_clear_frac"] = float(np.mean(clear)) if clear else np.nan
        k1 = max(0, n - 4)
        st = R["steps"].get(k1, {}) if R["steps"] else {}
        ok1 = R["objs"][k1][i] if len(R["objs"][k1]) > i else None
        if ok1 is not None:
            q1 = np.asarray(ok1[:2], float) - [X[k1], Y[k1]]
            c1, s1 = np.cos(YAW[k1]), np.sin(YAW[k1])
            ev["gap_m1"] = float(q1[0] * c1 + q1[1] * s1 - EGO_L / 2 - float(ok1[4]) / 2)
            ev["lat_m1"] = float(-q1[0] * s1 + q1[1] * c1)
        ev.update(lead_prob_m1=st.get("lead_prob"), lead_x_m1=st.get("lead_x"), lead_v_m1=st.get("lead_v"))
    trace = dict(X=np.round(X, 2).tolist(), Y=np.round(Y, 2).tolist(), YAW=np.round(YAW, 4).tolist(), V=np.round(V, 2).tolist(),
                 s=np.round(s, 2).tolist(), lat=np.round(lat, 2).tolist(), kexe=np.round(kexe, 4).tolist(), kreq=np.round(kreq, 4).tolist(), kdes=np.round(kdes, 4).tolist(),
                 kneed=np.round(kneed, 4).tolist(), kplan=np.round(kseg, 4).tolist(), cov=cov.tolist(), plan_cov=np.round(plan_cov_min, 2).tolist(),
                 v13=np.round(plan_v13, 2).tolist())
    if R["steps"]:
        trace["lead"] = [[R["steps"].get(k, {}).get(x) for x in ("lead_prob", "lead_x", "lead_v")] for k in range(n)]
    trace["plans"] = [None if p is None else np.round(p[0], 2).tolist() for p in R["plans"]]
    trace["objs"] = [np.round(np.asarray(o, float)[:, [0, 1, 3, 4, 6]], 2).tolist() if len(o) else [] for o in R["objs"]]
    return ev, trace


def run_list():
    """(arm, preset, rep, units-row) for the analysed runs, and the HD table of every stored arm."""
    from jevdrive.bench import tables as T
    runs, hd = [], []
    for a in ALL_ARMS:
        for p in PRESETS:
            for rep, key in [("r0", None), ("r1", f"run:{a}_{p}-rr1"), ("r2", f"run:{a}_{p}-rr2")]:
                if rep != "r0" and p not in REP_PRESETS:
                    continue
                u, _ = T.load("hugsim", a, p) if key is None else T.load("hugsim", key)
                if u is None:
                    print("missing", a, p, rep)
                    continue
                for sc, r in u.iterrows():
                    hd.append(dict(arm=a, preset=p, rep=rep, scenario=sc, hd=float(r.hdscore), end=r.end, cls=r.cls))
                    if a in P2H and (p == "spec_plan_smooth" or rep == "r0" and p in ("spec", "exam")):
                        runs.append((a, p, rep, sc, r))
    u, _ = T.load("hugsim", "WA-JEPA", "exam")
    for sc, r in u.iterrows():
        hd.append(dict(arm="WA-JEPA", preset="exam", rep="r0", scenario=sc, hd=float(r.hdscore), end=r.end, cls=r.cls))
        runs.append(("WA-JEPA", "exam", "r0", sc, r))
    return runs, hd


def _work(job):
    sc, items, routes = job
    scenes, evs, trs = {}, [], {}
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for a, p, rep, r in items:
        d = Path(r["run_dir"])
        if not (d / "infos.pkl").exists():
            print("no run dir", d, flush=True)
            continue
        if not scenes:
            scenes[0] = Scene(d)                   # ground / scene point sets are per scenario, identical across runs
        row = dict(arm=a, preset=p, rep=rep, scenario=sc, scene=r["scene"], dataset=r["dataset"], difficulty=r["difficulty"],
                   hd=float(r["hdscore"]), rc=float(r["rc"]), dac=float(r["dac"]), end=r["end"], cls=r["cls"], spin=bool(r["spin"]))
        try:
            ev, tr = analyse(row, d, Route(routes[r["scene"]]["xz"]), scenes[0], wajepa=a == "WA-JEPA")
        except Exception as e:   # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", a, p, rep, sc, e, flush=True)
            continue
        evs.append(ev)
        trs[f"{a}|{p}|{rep}|{sc}"] = tr
    print("done", sc, len(evs), flush=True)
    return evs, trs


def extract(args):
    import pandas as pd
    from multiprocessing import Pool
    from jevdrive.common import n_cpus
    from jevdrive.bench.sets import hugsim_scenarios
    D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
    routes = json.load(open(D / "runs/op_parity/hugsim/routes.json"))
    runs, hd = run_list()
    OUT.mkdir(parents=True, exist_ok=True)
    turn = {Path(s).stem for s in hugsim_scenarios("turn23")}
    H = pd.DataFrame(hd)
    H["turning"] = H.scenario.isin(turn)
    H.to_csv(OUT / "hugsim_hd.csv", index=False)
    by = {}
    for a, p, rep, sc, r in runs:
        by.setdefault(sc, []).append((a, p, rep, r.to_dict()))
    jobs = [(sc, it, routes) for sc, it in by.items()]
    nw = args.workers or max(2, min(16, n_cpus() // 6))
    print(f"{len(runs)} runs over {len(jobs)} scenarios, {nw} workers", flush=True)
    evs, trs = [], {}
    with Pool(nw) as pool:
        for e, t in pool.imap_unordered(_work, jobs):
            evs += e
            trs.update(t)
    E = pd.DataFrame(evs)
    E["turning"] = E.scenario.isin(turn)
    E.to_csv(OUT / "hugsim_events.csv", index=False)
    # per-step table (moving steps of every analysed run) for the curvature figure
    rows = []
    for key, t in trs.items():
        a, p, rep, sc = key.split("|")
        for k in range(len(t["kexe"])):
            rows.append((a, p, rep, sc, k, t["V"][k], t["kneed"][k], t["kreq"][k], t["kdes"][k], t["kexe"][k], t["kplan"][k], t["lat"][k], t["cov"][k]))
    S = pd.DataFrame(rows, columns=["arm", "preset", "rep", "scenario", "k", "v", "k_need", "k_req", "k_des", "k_exe", "k_plan", "lat", "cov"])
    S.to_csv(OUT / "hugsim_steps.csv.gz", index=False, float_format="%.4f")
    with gzip.open(Path(args.cases_out) / "hugsim_traces.json.gz", "wt") as f:
        json.dump(trs, f)
    print(len(E), "events ->", OUT, flush=True)


# ------------------------------------------------------------------------------------------------------------------ report
def label_of(r):
    if r["end"] == "fg_collision":
        return "D3_fg"
    if r["end"] in ("bg_collision", "off_route"):
        return {"sharp": "D1_sharp", "wide": "D2_wide", "straight": "S_straight"}[r["rclass"]]
    return "stuck" if r["end"] == "max_steps" else ("complete" if r["end"] == "complete" else "other")


BUCKETS = ["D1_sharp", "D2_wide", "S_straight", "D3_fg"]


def modal(g):
    c = Counter(g.label)
    top = max(c.values())
    tied = [k for k, v in c.items() if v == top]
    return tied[0] if len(tied) == 1 else g[g.rep == "r0"].label.iloc[0]


def report(args):
    import pandas as pd
    from jevdrive import stats
    E = pd.read_csv(OUT / "hugsim_events.csv")
    H = pd.read_csv(OUT / "hugsim_hd.csv")
    E["label"] = E.apply(label_of, axis=1)
    # side, amended after the first read (definition only, no score involved): a frontal background contact (|centroid| < 0.3 m)
    # has no lateral side of its own, so it takes the side of the ego's final route deviation, as off_route does
    sl = np.where((E.end == "bg_collision") & (E.bg_ly.abs() >= 0.3), E.bg_ly, E.lat_end)
    E["side"] = np.where(~E.end.isin(["bg_collision", "off_route"]), None,
                         np.where(E.rclass == "straight", np.where(sl > 0, "left", "right"),
                                  np.where(np.sign(sl) == E.turn_sign, "inside", "outside")))
    # fg: crossing split into oncoming (|dh| >= 150 deg) and crossing (60-150); refinement after the first read, disclosed
    E.loc[(E.fg_type == "crossing") & (E.fg_dh.abs() >= 150), "fg_type"] = "oncoming"
    E["stopped_at_contact"] = E.v_end < 0.5
    E["t_end"] = E.n_steps * DT
    L = []
    P = L.append
    sm = E[(E.arm.isin(P2H)) & (E.preset == "spec_plan_smooth")]
    lab = sm.groupby(["arm", "scenario"]).apply(modal).rename("label").reset_index()
    stable = sm.groupby(["arm", "scenario"]).label.nunique()
    scen = sorted(H.scenario.unique())
    hd_p2h = H[(H.arm.isin(P2H)) & (H.preset == "spec_plan_smooth")].groupby(["arm", "scenario"]).hd.mean().unstack(0).reindex(scen)
    wa = H[H.arm == "WA-JEPA"].set_index("scenario").hd.reindex(scen)
    best_tab = H[H.arm != "WA-JEPA"].groupby(["arm", "preset", "scenario"]).hd.mean().unstack([0, 1]).reindex(scen)
    best = best_tab.max(axis=1)
    turn = H.drop_duplicates("scenario").set_index("scenario").turning.reindex(scen)
    LB = lab.pivot(index="scenario", columns="arm", values="label").reindex(scen)
    sub = {"oncoming": "D3a_oncoming_crossing", "crossing": "D3a_oncoming_crossing", "lead_stopped": "D3b_lead", "lead_moving": "D3b_lead"}
    ft = sm[sm.end == "fg_collision"].groupby(["arm", "scenario"]).fg_type.agg(lambda x: Counter(x).most_common(1)[0][0])
    LS = LB.copy()
    for (a_, sc_), t_ in ft.items():
        if LS.loc[sc_, a_] == "D3_fg":
            LS.loc[sc_, a_] = sub.get(t_, "D3c_other")
    base = hd_p2h.mean(axis=1)
    gap = wa - base
    P(f"# HUGSIM four directions: tables (generated by fd_hugsim.py report)\n")
    P(f"P2H (P2H10-F-s0/s1, spec_plan_smooth, 3 repeats) HD {base.mean():.3f}, WA-JEPA {wa.mean():.3f}, gap WA - P2H "
      f"{stats.fmt(stats.bootstrap(gap.values))}; turning 23: {base[turn].mean():.3f} vs {wa[turn].mean():.3f}. Best-of-arms oracle source: "
      f"{best_tab.shape[1]} (arm, preset) sets: {', '.join(sorted({f'{a}/{p}' for a, p in best_tab.columns}))}. "
      f"Label stable over the 3 repeats in {int((stable == 1).sum())} / {len(stable)} (arm, scenario) cells.\n")
    # bucket sizes and oracles
    rows, shares = [], {}
    rng_idx = np.random.default_rng(0).integers(0, len(scen), (10000, len(scen)))
    for b in BUCKETS + ["D3a_oncoming_crossing", "D3b_lead", "D3c_other", "stuck", "complete"]:
        M = ((LS if b.startswith("D3") and b != "D3_fg" else LB) == b).astype(float)      # (scen, arm)
        n_cells = int(M.values.sum())
        n_sc = int((M.sum(axis=1) > 0).sum())
        r = dict(bucket=b, cells=n_cells, scenarios=n_sc, turning_sc=int(((M.sum(axis=1) > 0) & turn).sum()),
                 hd_in=float(np.nansum(hd_p2h.values * M.values) / max(n_cells, 1)),
                 hd_loss=stats.fmt(stats.bootstrap(((1 - hd_p2h) * M).mean(axis=1).values), ".3f", 100))
        for nm, ref in (("a_WA", wa), ("b_best", best), ("c_one", pd.Series(1.0, index=scen))):
            x = (M.mul(ref, axis=0) - M * hd_p2h).mean(axis=1)                # per-scenario effect, mean over the two arms
            bs = stats.bootstrap(x.values)
            r[f"{nm}_x100"] = stats.fmt(bs, ".1f", 100)
            if nm == "a_WA":
                g = gap.values
                ratio = x.values[rng_idx].mean(1) / g[rng_idx].mean(1)
                r["a_share_of_gap"] = f"{x.mean() / g.mean():+.2f} [{np.quantile(ratio, .025):+.2f}, {np.quantile(ratio, .975):+.2f}]"
                r["ref_hd_in"] = float(np.nansum(M.mul(wa, axis=0).values) / max(n_cells, 1))
        rows.append(r)
    T1 = pd.DataFrame(rows)
    T1.to_csv(OUT / "hugsim_buckets.csv", index=False)
    P("## T1 buckets (label per (arm, scenario), modal over 3 repeats); oracles in HD x 100 on 64 scenarios, scenario bootstrap B 10000\n")
    P("hd_loss = mean over scenarios of (1 - HD) inside the bucket; a = set to WA-JEPA's HD, b = best HD over our stored arms, c = 1.0. "
      "a_share_of_gap = a / (WA - P2H gap); a ratio, its CI is wide because the gap is small.\n")
    P(T1.to_markdown(index=False, floatfmt=".3f") + "\n")
    # overlaps: scenarios labelled with different buckets by the two arms
    ov = LB[(LB.isin(BUCKETS)).any(axis=1)]
    pairs = Counter(tuple(sorted(set(r))) for _, r in ov.iterrows() if len(set(r)) > 1)
    P("Overlap (two arms disagree on the label of a scenario): " + (", ".join(f"{'/'.join(k)}: {v}" for k, v in pairs.items()) or "none") + "\n")
    # per-scenario table of the bucket scenarios
    W = H[H.arm == "WA-JEPA"].set_index("scenario")
    ex = E[(E.arm.isin(P2H)) & (E.rep == "r0")]
    cmp = []
    for sc in ov.index:
        e0 = sm[(sm.scenario == sc) & (sm.rep == "r0")]
        def end_of(p):
            q = ex[(ex.scenario == sc) & (ex.preset == p)]
            return "/".join(q.sort_values("arm").end.str.replace("_collision", "").str.replace("max_steps", "stuck").tolist())
        cmp.append(dict(scenario=sc, turning=bool(turn[sc]), label="/".join(LB.loc[sc].tolist()), hd_smooth=round(base[sc], 3),
                        end_smooth="/".join(e0.sort_values("arm").end.str.replace("_collision", "").tolist()),
                        R_route=round(e0.R_route.mean(), 1), side="/".join(e0.side.fillna("-").astype(str)),
                        hd_spec=round(H[(H.arm.isin(P2H)) & (H.preset == "spec") & (H.scenario == sc)].hd.mean(), 3), end_spec=end_of("spec"),
                        hd_exam=round(H[(H.arm.isin(P2H)) & (H.preset == "exam") & (H.scenario == sc)].hd.mean(), 3), end_exam=end_of("exam"),
                        hd_wa=round(wa[sc], 3), end_wa=W.loc[sc, "end"].replace("_collision", ""), hd_best=round(best[sc], 3)))
    T2 = pd.DataFrame(cmp).sort_values(["label", "scenario"])
    T2.to_csv(OUT / "hugsim_bucket_scenarios.csv", index=False)
    P("## T2 bucket scenarios: P2H smooth vs spec / exam (P2H10 r0) and WA-JEPA (ends listed s0/s1)\n")
    P(T2.to_markdown(index=False) + "\n")
    # mechanism shares, D1 / D2 / S (events of P2H smooth, all repeats; cluster bootstrap by scenario)
    ed = sm[sm.label.isin(["D1_sharp", "D2_wide", "S_straight"])].copy()
    ed["plan_off"] = ed.plan_off_any.astype(float)
    ed["inside"] = (ed.side == "inside").astype(float)
    ed["under"] = (ed.k_exe < 0.8 * ed.k_need).astype(float)
    ed["req_under"] = (ed.k_req < 0.8 * ed.k_need).astype(float)
    ed["plan_under"] = (ed.k_plan < 0.8 * ed.k_need).astype(float)
    ed["exe_lag"] = (ed.k_exe < 0.8 * ed.k_req).astype(float)
    ed["bg"] = (ed.end == "bg_collision").astype(float)
    ed["fast_entry"] = (ed.v_onset ** 2 / ed.R_route > 4.0).astype(float)     # needed lateral accel at the onset speed > 4 m/s^2
    mrow = []
    for b in ["D1_sharp", "D2_wide", "S_straight"]:
        g = ed[ed.label == b]
        if not len(g):
            continue
        r = dict(bucket=b, runs=len(g), scenarios=g.scenario.nunique())
        for c in ["bg", "inside", "fast_entry", "plan_off", "under", "req_under", "plan_under", "exe_lag"]:
            r[c] = stats.fmt(stats.bootstrap(g[c].values, groups=g.scenario.values), ".2f")
        for c in ["R_route", "v_onset", "a_lat_need", "k_need", "k_plan", "k_req", "k_exe", "plan_cov_m2", "frac_offroad"]:
            r[c] = round(float(g[c].median()), 4)
        mrow.append(r)
    T3 = pd.DataFrame(mrow)
    T3.to_csv(OUT / "hugsim_mech_edge.csv", index=False)
    P("## T3 edge events (D1 / D2 / S), P2H smooth, all 6 runs per scenario; shares with scenario-cluster bootstrap CI; medians of the "
      "approach window (3 s before the departure onset to the end), curvatures signed into the turn (1/m); fast_entry = v_onset^2 / R > 4 m/s^2\n")
    P("inside = contact / departure on the turn's inner side; plan_off = a plan footprint (0.5-3 s) off the drivable ground (HUGSIM DAC "
      "cell rule < 0.5) or into background points at any step in the 2 s before the onset; under = executed curvature < 0.8 x needed; "
      "req_under = requested < 0.8 x needed; plan_under = plan curvature (0.5-1.5 s, from positions) < 0.8 x needed; exe_lag = "
      "executed < 0.8 x requested.\n")
    P(T3.to_markdown(index=False) + "\n")
    cols = ["scenario", "arm", "rep", "end", "rclass", "R_route", "turn_sign", "side", "v_onset", "a_lat_need", "k_need", "k_plan", "k_req",
            "k_des", "k_exe", "clip_frac", "plan_off_any", "plan_cov_m2", "plan_bg_m2", "lat_end", "herr_end", "bg_ly"]
    T3b = ed[ed.rep == "r0"][cols].sort_values(["rclass", "scenario", "arm"])
    T3b.to_csv(OUT / "hugsim_edge_events.csv", index=False)
    P("### T3b edge events, r0 runs\n")
    P(T3b.to_markdown(index=False, floatfmt=".3f") + "\n")
    # same scenarios under spec / exam / WA: the same reading
    oth = E[(E.scenario.isin(ed.scenario.unique())) & (((E.arm.isin(P2H)) & (E.rep == "r0") & E.preset.isin(["spec", "exam"])) | (E.arm == "WA-JEPA"))]
    oth = pd.concat([oth, sm[(sm.rep == "r0") & sm.scenario.isin(ed.scenario.unique())]])
    oth["fails_edge"] = oth.end.isin(["bg_collision", "off_route"])
    T3c = oth.groupby(["arm", "preset"]).agg(runs=("scenario", "size"), edge_fail=("fails_edge", "sum"),
                                             fg=("end", lambda x: (x == "fg_collision").sum()), complete=("end", lambda x: (x == "complete").sum()),
                                             hd=("hd", "mean"), plan_off=("plan_off_any", lambda x: np.nanmean(x.astype(float)))).reset_index()
    T3c.to_csv(OUT / "hugsim_edge_contrast.csv", index=False)
    P("### T3c on the edge-event scenarios: the same runs under spec / exam (P2H10 r0) and WA-JEPA (plan_off on that run's own event "
      "or end window)\n")
    P(T3c.to_markdown(index=False, floatfmt=".3f") + "\n")
    # D3 fg
    ef = E[E.end == "fg_collision"].copy()
    fsm = ef[(ef.arm.isin(P2H)) & (ef.preset == "spec_plan_smooth")]
    few = ef[ef.arm == "WA-JEPA"]
    frow = []
    for nm, g in (("P2H smooth (6 runs / sc)", fsm), ("WA-JEPA", few), ("P2H spec r0", ef[(ef.arm.isin(P2H)) & (ef.preset == "spec")]),
                  ("P2H exam r0", ef[(ef.arm.isin(P2H)) & (ef.preset == "exam")])):
        c = g.fg_type.value_counts()
        r = dict(set=nm, runs=len(g), scenarios=g.scenario.nunique())
        for t in ["oncoming", "crossing", "lead_stopped", "lead_moving", "cut_in", "side", "rear_ended"]:
            r[t] = int(c.get(t, 0))
        r["ego_stopped_at_contact"] = int(g.stopped_at_contact.sum())
        r["t_end_median_s"] = float(g.t_end.median())
        r["turning_route"] = int((g.rclass != "straight").sum())
        frow.append(r)
    T4 = pd.DataFrame(frow)
    T4.to_csv(OUT / "hugsim_fg_types.csv", index=False)
    P("## T4 fg_collision types (rule in the script docstring)\n")
    P(T4.to_markdown(index=False) + "\n")
    fsm = fsm.copy()
    fsm["plan_slow"] = (fsm.plan_v13_min < np.maximum(0.5 * fsm.v_m3, 1.0)).astype(float)
    fsm["plan_clear"] = (fsm.plan_clear_frac >= 0.5).astype(float)
    fsm["exec_fast"] = (fsm.v_end > 2.0).astype(float)
    fsm["stop_not_exec"] = fsm.plan_slow * fsm.exec_fast
    fsm["lead_seen"] = (fsm.lead_prob_m1 > 0.5).astype(float)
    fsm["stopped"] = fsm.stopped_at_contact.astype(float)
    mrow = []
    for t, g in [("all", fsm)] + list(fsm.groupby("fg_type")):
        r = dict(fg_type=t, runs=len(g), scenarios=g.scenario.nunique())
        for c in ["plan_slow", "plan_clear", "exec_fast", "stop_not_exec", "lead_seen", "stopped"]:
            r[c] = stats.fmt(stats.bootstrap(g[c].values, groups=g.scenario.values), ".2f")
        for c in ["v_end", "v_m3", "plan_v13_min", "fg_vo", "fg_dh", "gap_m1", "lead_x_m1", "lead_prob_m1", "dec_exe"]:
            r[c] = round(float(g[c].median()), 2)
        mrow.append(r)
    T5 = pd.DataFrame(mrow)
    T5.to_csv(OUT / "hugsim_fg_mech.csv", index=False)
    P("## T5 fg mechanism, P2H smooth (all runs, scenario-cluster CI): plan_slow = the plan's 1-3 s speed fell below max(0.5 v(-3 s), 1 m/s) "
      "at some step in the last 3 s; plan_clear = >= half of the plans at -1.5..-0.5 s do not touch the actor held at constant velocity "
      "(0.5-2 s); exec_fast = speed at contact > 2 m/s; stop_not_exec = plan_slow and exec_fast; lead_seen = lead_prob > 0.5 at -1 s. "
      "Medians: gap_m1 = actor gap ahead at -1 s (m), lead_x_m1 = lead head's distance at -1 s\n")
    P(T5.to_markdown(index=False) + "\n")
    few = few.copy()
    few["plan_slow"] = (few.plan_v13_min < np.maximum(0.5 * few.v_m3, 1.0)).astype(float)
    wr = []
    for t, g in [("all", few)] + list(few.groupby("fg_type")):
        wr.append(dict(fg_type=t, runs=len(g), plan_slow=round(g.plan_slow.mean(), 2), stopped=int(g.stopped_at_contact.sum()),
                       v_end=round(g.v_end.median(), 2), plan_v13_min=round(g.plan_v13_min.median(), 2), t_end=float(g.t_end.median())))
    P("### T5w the same reading for WA-JEPA (its plan = HUGSIM's stored planned_traj)\n")
    P(pd.DataFrame(wr).to_markdown(index=False) + "\n")
    fcols = ["scenario", "arm", "rep", "fg_type", "fg_part", "rclass", "v_end", "v_m3", "fg_vo", "fg_dh", "fg_lx", "fg_ly", "fg_lat3", "plan_v13_min",
             "plan_clear_frac", "dec_exe", "gap_m1", "lead_prob_m1", "lead_x_m1", "lead_v_m1"]
    T5b = ef[(ef.rep == "r0") & ((ef.arm == "WA-JEPA") | ((ef.arm.isin(P2H)) & (ef.preset == "spec_plan_smooth")))][fcols].sort_values(["scenario", "arm"])
    T5b.to_csv(OUT / "hugsim_fg_events.csv", index=False)
    s_p = set(fsm.scenario)
    s_w = set(few.scenario)
    P(f"fg scenarios: P2H smooth {len(s_p)} (any run), WA-JEPA {len(s_w)}, both {len(s_p & s_w)}; P2H only {sorted(s_p - s_w)}; "
      f"WA only {sorted(s_w - s_p)}\n")
    P("### T5b fg events, r0 (P2H smooth s0 / s1 and WA-JEPA)\n")
    P(T5b.to_markdown(index=False, floatfmt=".2f") + "\n")
    tr_path = Path(args.traces)
    if tr_path.exists():
        trace_tables(E, sm, ed, json.load(gzip.open(tr_path, "rt")), P)
    (OUT / "hugsim_tables.md").write_text("\n".join(L))
    print("\n".join(L))
    if args.figs:
        figures(E, sm, ed, fsm)


def trace_tables(E, sm, ed, T, P):
    """Trace-based readings: entry speed and achieved curvature of every arm on the edge-event turns, turn-in lags, lead head
    calibration against the actor boxes."""
    import pandas as pd
    from jevdrive import stats
    rows = []
    ref = ed[ed.rep == "r0"].groupby("scenario").agg(s_on=("s_onset", "median"), sg=("turn_sign", "first"), lab=("label", "first"))
    for sc, rr in ref.iterrows():
        for key, t in T.items():
            a, p, rep, s_ = key.split("|")
            if s_ != sc or rep != "r0":
                continue
            S_, V_, ke = np.array(t["s"]), np.array(t["V"]), np.array(t["kexe"])
            kn = np.array(t["kneed"])
            kd = np.array(t.get("kdes", [np.nan] * len(ke)), float)
            kq = np.array(t["kreq"], float)
            kp = np.array(t["kplan"], float)
            n = len(ke)
            S_ = S_[:n]
            inw = (S_ > rr.s_on - 10) & (S_ < rr.s_on + 20)
            def at(x):
                i = np.where(S_ >= x)[0]
                return float(V_[i[0]]) if len(i) else np.nan
            e = E[(E.arm == a) & (E.preset == p) & (E.rep == "r0") & (E.scenario == sc)].iloc[0]
            rows.append(dict(scenario=sc, bucket=rr.lab, arm=a, preset=p, end=e.end, hd=round(e.hd, 3), v_entry=at(rr.s_on - 10), v_onset=at(rr.s_on),
                             passed=bool(S_.max() > rr.s_on + 20 and e.end != "off_route" or e.end == "complete"),
                             k_need_max=float(np.nanmax(rr.sg * kn[inw])) if inw.any() else np.nan,
                             k_plan_max=float(np.nanmax(rr.sg * kp[inw])) if inw.any() and np.isfinite(kp[inw]).any() else np.nan,
                             k_req_max=float(np.nanmax(rr.sg * kq[inw])) if inw.any() and np.isfinite(kq[inw]).any() else np.nan,
                             k_des_max=float(np.nanmax(rr.sg * kd[inw])) if inw.any() and np.isfinite(kd[inw]).any() else np.nan,
                             k_exe_max=float(np.nanmax(rr.sg * ke[inw])) if inw.any() else np.nan,
                             a_lat_exe_max=float(np.nanmax(V_[:n][inw] ** 2 * np.abs(ke[inw]))) if inw.any() else np.nan))
    T6 = pd.DataFrame(rows).sort_values(["bucket", "scenario", "arm", "preset"])
    T6.to_csv(OUT / "hugsim_edge_entry.csv", index=False)
    P("## T6 edge-event turns: every arm on the same route segment ([s_onset - 10, s_onset + 20] of the P2H smooth r0 event); speed at "
      "the window entry / onset, max curvature into the turn (1/m) of the route need, plan, request, after clip, executed; passed = "
      "drove beyond the segment\n")
    P(T6.to_markdown(index=False, floatfmt=".3f") + "\n")
    # turn-in lags of the smooth edge events: first time the signed curvature exceeds half the max needed one in the approach
    lags = []
    for _, e in ed.iterrows():
        t = T.get(f"{e.arm}|{e.preset}|{e.rep}|{e.scenario}")
        if t is None:
            continue
        n = len(t["kexe"])
        a0 = max(0, min(int(e.onset), n) - 12)
        sg = e.turn_sign or 1
        ser = {k: sg * np.array(t[k], float)[a0:n] for k in ("kneed", "kplan", "kreq", "kdes", "kexe")}
        thr = 0.5 * np.nanmax(ser["kneed"]) if np.isfinite(ser["kneed"]).any() else np.nan
        first = {k: (np.argmax(np.nan_to_num(v, nan=-9) > thr) if (np.nan_to_num(v, nan=-9) > thr).any() else np.nan) for k, v in ser.items()}
        lags.append(dict(scenario=e.scenario, label=e.label, arm=e.arm, rep=e.rep, thr=thr,
                         **{f"t_{k}": (first[k] * DT if np.isfinite(first[k]) else np.nan) for k in ser},
                         peak_ratio_exe=float(np.nanmax(ser["kexe"]) / np.nanmax(ser["kneed"])), peak_ratio_plan=float(np.nanmax(ser["kplan"]) / np.nanmax(ser["kneed"]))))
    LG = pd.DataFrame(lags)
    LG.to_csv(OUT / "hugsim_edge_lags.csv", index=False)
    P("### T6b turn-in timing of the P2H smooth edge events (s after the approach start = onset - 3 s; NaN = never reached half the "
      "needed peak); peak ratios = max achieved / max needed curvature in the approach\n")
    g = LG.groupby("label")
    P(g[["t_kneed", "t_kplan", "t_kreq", "t_kdes", "t_kexe", "peak_ratio_plan", "peak_ratio_exe"]].median().round(2).to_markdown() + "\n")
    P("never reached half the need: " + ", ".join(f"{k} {int(LG[f't_{k}'].isna().sum())}/{len(LG)}" for k in ("kplan", "kreq", "kdes", "kexe")) + "\n")
    # lead head calibration: steps with an actor ahead in the lane (|lateral| < 1.8 m), P2H runs of all presets
    rows = []
    for key, t in T.items():
        a, p, rep, sc = key.split("|")
        if "lead" not in t:
            continue
        X, Y, YAW, V = map(np.array, (t["X"], t["Y"], t["YAW"], t["V"]))
        for k, ld in enumerate(t["lead"]):
            ob = t["objs"][k]
            if not ob or ld[0] is None:
                continue
            ob = np.array(ob)
            c, s_ = np.cos(YAW[k]), np.sin(YAW[k])
            q = ob[:, :2] - [X[k], Y[k]]
            fx, fy = q[:, 0] * c + q[:, 1] * s_, -q[:, 0] * s_ + q[:, 1] * c
            dh = np.degrees(wrap(ob[:, 4] - YAW[k]))
            m = (fx > 0) & (np.abs(fy) < 1.8)
            if not m.any():
                continue
            j = np.where(m)[0][np.argmin(fx[m])]
            rows.append(dict(scenario=sc, lp=ld[0], lead_x=ld[1], lead_v=ld[2], true_x=fx[j] - ob[j, 3] / 2,
                             kind="same" if abs(dh[j]) < 30 else ("oncoming" if abs(dh[j]) > 150 else "crossing")))
    LC = pd.DataFrame(rows)
    LC["err"] = LC.lead_x - LC.true_x
    LC["bin"] = pd.cut(LC.true_x, [0, 3, 6, 10, 20, 40])
    out = []
    for (kind, b), g in LC.groupby(["kind", "bin"], observed=True):
        det = g[g.lp > 0.5]
        r = stats.bootstrap(det.err.values, groups=det.scenario.values) if len(det) else dict(mean=np.nan, lo=np.nan, hi=np.nan)
        out.append(dict(kind=kind, true_gap_from_camera=str(b), steps=len(g), scenarios=g.scenario.nunique(), detect_rate=round((g.lp > 0.5).mean(), 2),
                        lead_x_err_median=round(det.err.median(), 2) if len(det) else np.nan, lead_x_err_mean_ci=stats.fmt(r, ".2f") if len(det) else "-"))
    T7 = pd.DataFrame(out)
    T7.to_csv(OUT / "hugsim_lead_calib.csv", index=False)
    P("## T7 lead head vs the actor boxes (P2H runs, all presets): nearest actor ahead with |lateral| < 1.8 m; true gap = camera (ego box "
      "centre) to the actor's near end; err = lead_x - true gap on steps with lead_prob > 0.5 (CI: scenario-cluster bootstrap)\n")
    P(T7.to_markdown(index=False) + "\n")


def figures(E, sm, ed, fsm):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "research"))
    try:
        import plot_style
        plot_style.apply()
    except Exception:   # noqa: BLE001
        pass
    F = OUT / "figs"
    F.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(OUT / "hugsim_steps.csv.gz")
    S = S[(S.v > 0.5)]
    S["turn"] = (S.k_need.abs() > 1 / 50) & (S.lat.abs() < 1.0)      # on the route (|lateral| < 1 m) and the route ahead turns
    # summary: needed vs requested / executed / plan curvature by speed, turn steps of P2H smooth and WA-JEPA
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = [0.5, 2, 3, 4, 6, 9, 15]
    for i, (nm, q) in enumerate([("P2H spec_plan_smooth", S[(S.arm.isin(P2H)) & (S.preset == "spec_plan_smooth")]), ("WA-JEPA exam", S[S.arm == "WA-JEPA"])]):
        q = q[q.turn].copy()
        sg = np.sign(q.k_need)
        q["vb"] = pd.cut(q.v, bins)
        for c, col, lab in [("k_need", "#555555", "needed (route, 0.5-1.5 s ahead)"), ("k_plan", "#E69F00", "plan (0.5-1.5 s, positions)"),
                            ("k_req", "#0072B2", "requested (smooth)"), ("k_des", "#56B4E9", "after clip_curvature"),
                            ("k_exe", "#D55E00", "executed (yaw rate / v)")]:
            if q[c].isna().all():
                continue
            g = (q[c] * sg).groupby(q.vb, observed=True).median()
            ax[i].plot([b.mid for b in g.index], g.values, "o-", color=col, label=lab)
        cnt = q.groupby(q.vb, observed=True).size()
        for b, c_ in cnt.items():
            ax[i].annotate(f"n={c_}", (b.mid, 0), fontsize=6, ha="center", va="bottom", color="#555555")
        ax[i].set_title(f"{nm}: on-route steps, route ahead turning (|k_need| > 1/50 m)", fontsize=9)
        ax[i].set_xlabel("ego speed (m/s)")
        ax[i].set_ylabel("curvature into the turn, median (1/m)")
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "hugsim_curvature_by_speed.png", dpi=130)
    plt.close(fig)
    print("fig curvature")


def bev(args):
    """BEV of one run: route, ground / scene points near the event, ego track, plans of the last steps, actors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
    from jevdrive.bench import tables as T
    routes = json.load(open(D / "runs/op_parity/hugsim/routes.json"))
    n_pan = len(args.case)
    fig, axs = plt.subplots(1, n_pan, figsize=(5.2 * n_pan, 5.2))
    axs = np.atleast_1d(axs)
    centre = None
    for ax, case in zip(axs, args.case):
        arm, preset, rep, sc, title = case.split("|")
        key = (arm, preset) if rep == "r0" else (f"run:{arm}_{preset}-rr{rep[1]}",)
        u, _ = T.load("hugsim", *key) if rep == "r0" else T.load("hugsim", key[0])
        r = u.loc[sc]
        d = Path(r.run_dir)
        sc_ = Scene(d)
        R = load_run(d, wajepa=arm == "WA-JEPA")
        route = Route(routes[r.scene]["xz"])
        n = R["n"]
        if centre is None or not args.same_centre:
            centre = (R["X"][-1], R["Y"][-1])
        cx, cy = centre
        rad = args.radius
        gi = sc_.gt.query_ball_point([cx, cy], rad)
        si = sc_.st.query_ball_point([cx, cy], rad)
        k0 = int(np.argmin((R["X"] - cx) ** 2 + (R["Y"] - cy) ** 2))
        si = [i for i in si if R["CY"][k0] < sc_.sy[i] < R["CY"][k0] + EGO_H]
        ax.scatter(*sc_.gxy[gi][::4].T, s=0.3, c="#c8e6c9", rasterized=True)
        ax.scatter(*sc_.sxy[si][::3].T, s=0.3, c="#9e9e9e", rasterized=True)
        ax.plot(*route.p.T, "--", color="k", lw=1, label="recorded route")
        ax.plot(R["X"], R["Y"], "-", color="#D55E00", lw=2, label="ego (executed)")
        kev = min(n - 1, k0) if args.same_centre else n
        for j, k in enumerate(range(max(0, kev - 12), kev, 4)):
            if R["plans"][k] is not None:
                w = R["plans"][k][0]
                ax.plot(np.r_[R["X"][k], w[:, 0]], np.r_[R["Y"][k], w[:, 1]], "-", color="#0072B2", alpha=0.4 + 0.2 * j, lw=1.5,
                        label="plan at -3 / -2 / -1 s" if j == 0 else None)
        from matplotlib.patches import Polygon as MP
        for kk, alpha in ((min(n, kev), 1.0), (max(0, kev - 8), 0.35)):
            for o in R["objs"][kk]:
                ax.add_patch(MP(np.asarray(box_poly(o).exterior.coords), fill=False, ec="#7B1FA2", alpha=alpha, lw=1.2))
            ax.add_patch(MP(np.asarray(box_poly([R["X"][kk], R["Y"][kk], 0, EGO_W, EGO_L, 0, R["YAW"][kk]]).exterior.coords), fill=False,
                            ec="#D55E00", alpha=alpha, lw=1.2))
        ax.set_xlim(cx - rad, cx + rad)
        ax.set_ylim(cy - rad, cy + rad)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{arm} {preset} {rep} {sc}: {r.end}, HD {r.hdscore:.2f}", fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--workers", type=int, default=0)
    e.add_argument("--cases-out", default=os.environ.get("FD_SCRATCH", "/root/autodl-tmp/ujs/runs/op_parity/four_dirs"))
    r = sp.add_parser("report")
    r.add_argument("--figs", action="store_true")
    r.add_argument("--traces", default=os.environ.get("FD_TRACES", "/root/autodl-tmp/ujs/runs/op_parity/four_dirs/hugsim_traces.json.gz"))
    b = sp.add_parser("bev")
    b.add_argument("--case", nargs="+", required=True, help="arm|preset|rep|scenario|title")
    b.add_argument("--out", required=True)
    b.add_argument("--same-centre", action="store_true", help="centre every panel on the first case's end position")
    b.add_argument("--radius", type=float, default=28.0)
    a = ap.parse_args()
    if a.cmd == "extract":
        Path(a.cases_out).mkdir(parents=True, exist_ok=True)
    {"extract": extract, "report": report, "bev": bev}[a.cmd](a)
