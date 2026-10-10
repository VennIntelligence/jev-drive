#!/usr/bin/env python3
"""LOWDIAG, PAI half, analysis (plans/2026-10-10-lowdiag-prereg.md): one row per rollout of the base P2H10-F-s0 served stacks (FIX1 `ab_*_s0`).
Input: lbd_pai_x.py's extraction and lbd_pai_replay.py's replays under $DATA_DIR/runs/lowboard_diag/pai/{x, replay}. CPU, .venv, about a minute.
Simulator boxes, the logged ego path and the map's road edges are labels only.

  units   every zero-score rollout with a kept log: event, reference geometry, raw flags, exclusive class, in-plan and lead time,
          base-right (shipped weights, same tokens), the corridor sub-class; every other rollout from the summary and the driver's own
          records -> <out>/units.csv, summary.json
  bev     one PNG per chosen unit (bird's-eye view + the model frames at the onset) -> <out>/bev/

Conventions (swv1_lib): actors / logged = box-centre poses in the rollout's local frame, plans = rear-axle poses; served plan = the 41-pose
trajectory the driver returned; replay plans = 8 poses in the rear-axle frame at t0. Command index: 0 left, 1 straight, 2 right, 3 unknown.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402

import col1_lib as CL  # noqa: E402
import swv1_lib as S  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
ROOT = DATA / os.environ.get("LBD_PAI_ROOT", "runs/lowboard_diag/pai")      # LOWDIAG2: one root per base seed (default = the seed-0 root of decision 237)
FLAG_K = {"collision_at_fault": "agent", "offroad": "boundary", "left_corridor_laterally": "corridor"}
ON, CORR, A_BRAKE, A_LAT = 1.5, 4.0, 4.0, 4.0                 # onset band, corridor (eval config), counterfactual braking, fast-entry line
CMD = ("left", "straight", "right", "unknown")
deg = np.degrees


class Ref:
    """The logged ego path (swv1_lib's: continued 40 m straight) with heading and curvature by arc length."""

    def __init__(self, path):
        d = np.diff(path, axis=0)
        self.p, self.s = path, np.r_[0.0, np.cumsum(np.hypot(*d.T))]
        psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.psi = np.r_[psi, psi[-1]]

    def heading(self, s):
        return np.interp(s, self.s, self.psi)

    def xy(self, s):
        return np.c_[np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])]

    def window(self, s_on):
        """Heading change and smallest radius over [s_on - 10 m, s_on + 20 m] (curvature = heading difference over +-4 m)."""
        g = np.arange(s_on - 10.0, s_on + 20.01, 1.0)
        k = (self.heading(g + 4) - self.heading(g - 4)) / 8.0
        i = np.abs(k).argmax()
        return float(self.heading(g[-1]) - self.heading(g[0])), float(1 / max(abs(k[i]), 1e-4)), float(np.sign(k[i]))


def ref_type(dpsi, R):
    return "turn" if R < 15 or abs(deg(dpsi)) >= 30 else "bend" if R < 50 or abs(deg(dpsi)) >= 10 else "straight"


def edges_geom(mp):
    return shapely.union_all([shapely.LineString(e[:, :2]) for e in mp["edges"] if len(e) >= 2]) if mp and mp["edges"] else None


def reparam(path, speed):
    """Dense plan (41, 3) with the geometry of `path` and the arc-length-over-time of `speed` (straight continuation past the path's end)."""
    sp = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))]
    ss = np.r_[0.0, np.cumsum(np.hypot(*np.diff(speed[:, :2], axis=0).T))]
    q = np.c_[[np.interp(ss, sp + 1e-9 * np.arange(41), path[:, j]) for j in range(3)]].T
    over = np.maximum(ss - sp[-1], 0)
    q[:, 0] += over * np.cos(path[-1, 2])
    q[:, 1] += over * np.sin(path[-1, 2])
    return q


class Unit:
    def __init__(self, stack, scene, o, mp, rp):
        self.stack, self.scene, self.o, self.mp, self.rp = stack, scene, o, mp, rp
        m = o["summary"]["score_metrics"]
        self.flags = [f for f in FLAG_K if m.get(f)]
        r = self.r = S.Rollout(o, "pai", "base", scene[7:15], "C" if "collision_at_fault" in self.flags else "N")
        ev = {f: CL.first_event(o, f) for f in self.flags}
        ev = {f: t for f, t in ev.items() if t is not None}
        self.ev = ev
        self.flag = min(ev, key=ev.get) if ev else None
        self.K, self.E = (FLAG_K[self.flag], float(ev[self.flag])) if ev else ("none", float(r.T1))
        xy, keep = r.gt_rig[:, 1:3], [0]                      # the logged path thinned to >= 0.5 m steps (a standing log jitters), continued 40 m straight
        for i in range(1, len(xy)):
            if np.hypot(*(xy[i] - xy[keep[-1]])) >= 0.5:
                keep.append(i)
        xy = xy[keep]
        self.L_log = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
        h = np.arctan2(*(xy[-1] - xy[-2])[::-1]) if len(xy) > 1 else r.gt_rig[0, 3]
        r.path = np.r_[xy, (xy[-1] + 40.0 * np.array([np.cos(h), np.sin(h)]))[None]]
        self.ref, self.edges = Ref(r.path), edges_geom(mp)
        self.t = np.arange(r.T0, self.E + 1, 1e5)
        self.rig = r.ego_rig(self.t)
        ls = r.lat(self.rig[:, :2])
        self.lat, self.s = ls[:, 0], ls[:, 1]
        inb = np.flatnonzero(np.abs(self.lat) < ON)
        self.i_on = int(inb[-1]) if len(inb) else 0
        self.t_on, self.s_on = float(self.t[self.i_on]), float(self.s[self.i_on])
        self.dpsi, self.R, self.sgn = self.ref.window(self.s_on)
        self.rtype = ref_type(self.dpsi, self.R)
        self.v = r.ego_v(self.t)
        self.rpi = {} if rp is None else {int(n): i for i, n in enumerate(rp["now"])}

    # ---- plans
    def dec(self, t0, t1):
        return np.flatnonzero((self.r.now >= t0) & (self.r.now < t1))

    def rplan(self, k, name):
        i = self.rpi.get(int(self.r.now[k]))
        return None if i is None else S.dense(self.r.ego_rig(self.rp["t0"][i])[0], self.rp[name][i])

    def hit(self, k, plan=None):
        """Does the plan of decision k (default: the served one) run into the event of this unit's kind?"""
        r = self.r
        plan = r.plan[k] if plan is None else plan
        if self.K == "agent":
            x = r.sweep(k, plan, [self.struck])
            return self.struck in x and x[self.struck][1] >= 0
        c = S.centre(plan, r.off)
        if self.K == "boundary":
            return self.edges is not None and bool(shapely.intersects(S.boxes(c, r.Le, r.We), self.edges).any())
        return bool((np.abs(r.lat(c[:, :2])[:, 0]) >= CORR).any())

    # ---- flags
    def analyse(self):
        r, o, E = self.r, self.o, self.E
        R = dict(stack=self.stack, scene=self.scene[7:15], score=0.0, lost=1.0, flags="+".join(FLAG_K[f] for f in self.flags) or "none", K=self.K,
                 t_event=round((E - r.T0) * 1e-6, 1), t_onset=round((self.t_on - r.T0) * 1e-6, 1), ref=self.rtype, dpsi_deg=round(deg(self.dpsi), 1), R_m=round(min(self.R, 999), 1),
                 lat_E=round(float(self.lat[-1]), 2), v_on=round(float(self.v[self.i_on]), 2), v_E=round(float(self.v[-1]), 2))
        gt = r.gt_rig
        sg = r.lat(gt[:, 1:3])[:, 1]
        j = int(np.clip(np.searchsorted(sg, self.s_on), 1, len(gt) - 2))
        v_log = float(np.hypot(*(gt[j + 1, 1:3] - gt[j - 1, 1:3])) / max((gt[j + 1, 0] - gt[j - 1, 0]) * 1e-6, 1e-3))
        R["v_log_on"] = round(v_log, 2)
        mt = lambda n: float(CL.metric(o, n)[np.abs(CL.metric(o, n)[:, 0] - E).argmin(), 1])  # noqa: E731
        fast = np.flatnonzero(np.hypot(*np.diff(gt[:, 1:3], axis=0).T) / np.maximum(np.diff(gt[:, 0]) * 1e-6, 1e-3) > 0.3)
        R.update(log_len=round(self.L_log, 1), over_end=round(float(self.s[-1]) - self.L_log, 1), log_stop_t=round((gt[fast[-1] + 1, 0] - r.T0) * 1e-6, 1) if len(fast) else 0.0,
                 sc_lat=round(mt("lateral_dist_to_gt_trajectory"), 2), sc_dist=round(mt("dist_to_gt_trajectory"), 2))
        dh = np.abs(deg(CL.wrap(self.rig[self.i_on:, 2] - self.ref.heading(self.s[self.i_on:]))))
        R["hdg_div_deg"] = round(float(dh.max()), 1)
        F = set()
        if self.K == "none":
            F.add("O1")
        wa = np.abs(np.diff(np.unwrap(self.rig[:, 2]))) / 0.1
        iw = np.flatnonzero(wa > 0.3)
        R.update(v_start=round(float(self.v[0]), 1), w_max=round(float(wa.max()), 2) if len(wa) else "", w_t=round(float(iw[0]) * 0.1, 1) if len(iw) else "")
        if self.v[0] >= 20 and len(wa) and wa.max() >= 1.0 and E - r.T0 <= 4e6:   # Amendment 2 (was: > 23 m/s and > 0.3 rad/s within 2 s)
            F.add("O2a")
        self.struck = None
        if self.K == "agent":
            r.t_ev = E
            k = self.struck = S.struck(r)
            po, _ = r.obj_pose(k, E)
            ec = r.ego_c(E)[0]
            rel = CL.into(ec, po[:, :2])[0]
            dho = float(deg(CL.wrap(po[0, 2] - ec[2])))
            R.update(obj=k, obj_label=o["size"][k][2], obj_x=round(rel[0], 2), obj_y=round(rel[1], 2), obj_dh=round(dho, 0), obj_v=round(float(r.obj_v(k, E)[0]), 2))
            on_ref = bool(shapely.intersects(S.boxes(po, *r.size(k))[0], shapely.union_all(S.boxes(S.centre(r.gt_rig[:, 1:], r.off), r.Le, r.We + 0.5))))
            R["obj_on_ref"] = int(on_ref)                    # descriptive: the struck object stands in the band of the logged path
            if rel[0] < 0 and abs(dho) < 45 and R["obj_v"] > 0.5:   # Amendment 1: a rear-ending object moves
                F.add("O2c")
            # L1: in the band of the driven path 2 s before, frontal, and braking along the driven path from E - 3 s clears it
            tt = np.arange(E - 2e6, E + 5e5 + 1, 1e5)
            band = shapely.union_all(S.boxes(r.ego_c(tt), r.Le, r.We + 0.5))
            inpath = bool(shapely.intersects(S.boxes(r.obj_pose(k, E - 2e6)[0], *r.size(k))[0], band))
            t0 = max(E - 3e6, r.T0)
            tau = np.arange(0, 6.01, 0.1)
            td = np.arange(t0, r.T1 + 1, 1e5)
            cd = r.ego_c(td)
            sd = np.r_[0.0, np.cumsum(np.hypot(*np.diff(cd[:, :2], axis=0).T))]
            v0 = float(r.ego_v(t0)[0])
            scf = np.where(tau < v0 / A_BRAKE, v0 * tau - A_BRAKE / 2 * tau**2, v0**2 / (2 * A_BRAKE))
            ccf = np.c_[[np.interp(scf, sd + 1e-9 * np.arange(len(sd)), cd[:, j]) for j in range(3)]].T
            pcf, _ = r.obj_pose(k, t0 + tau * 1e6)
            clears = not bool(shapely.intersects(S.boxes(ccf, r.Le, r.We), S.boxes(pcf, *r.size(k))).any())
            R.update(inpath2s=int(inpath), frontal=int(rel[0] > 0), brake_clears=int(clears))
            if inpath and rel[0] > 0 and clears:
                F.add("L1")
        v_end = float(np.hypot(*(gt[-1, 1:3] - gt[-11, 1:3])) / max((gt[-1, 0] - gt[-11, 0]) * 1e-6, 1e-3))
        self.over = self.K == "corridor" and self.s[-1] > self.L_log and v_end < 0.5
        if self.over:                                        # Amendment 2: the corridor flag fired past the end of a logged path that ends in a standstill
            F.add("L5")
        if self.K in ("boundary", "corridor"):
            if self.rtype != "straight" and self.v[self.i_on] ** 2 / self.R > A_LAT:
                F.add("L2")
            if self.rtype == "turn":
                F.add("R2")
        if self.v[self.i_on] > 1.2 * v_log and v_log > 0.5:
            F.add("L3")
        if abs(self.lat[-1]) >= ON:
            if dh.max() >= 20:
                F.add("R1")
            elif self.rtype == "straight":
                F.add("D")
        if self.K == "agent" and not F & {"L1", "O2c"}:
            F.add("C1")
        if self.K == "boundary" and self.rtype != "turn":
            F.add("C2")
        cls = ("other" if F & {"O1", "O2a", "O2c"} else "longitudinal" if F & {"L1", "L2", "L5"} else "route" if F & {"R1", "R2"} or self.K == "corridor"
               else "clearance")
        R.update(raw="+".join(sorted(F)), cls=cls)
        self.F, self.cls = F, cls
        if self.K != "none":
            self.in_plan(R)
            self.base_right(R)
        if self.K == "corridor" or "left_corridor_laterally" in self.flags:
            self.corridor(R)
        return R

    def in_plan(self, R):
        r, E = self.r, self.E
        ks = self.dec(E - 8e6, E)
        h = np.array([self.hit(k) for k in ks], bool)
        last3 = h[r.now[ks] >= E - 3e6]
        R["in_plan"] = int(last3.any()) if len(last3) else ""
        lead = 0.0
        if len(h) and h.any():
            i = len(h) - 1
            while i >= 0 and not h[i]:
                i -= 1
            while i > 0 and h[i - 1]:
                i -= 1
            lead = (E - r.now[ks[i]]) * 1e-6
        R["lead_s"] = round(float(lead), 1)
        kk = self.dec(r.T0, E - 5e5)
        dev = [np.hypot(*(r.ego_rig(r.now[k] + 5e5)[0, :2] - r.plan[k][5, :2])) for k in kk]
        R["track_med"], R["track_max"] = (round(float(np.median(dev)), 3), round(float(np.max(dev)), 2)) if dev else ("", "")
        self._hit = dict(zip(ks.tolist(), h.tolist()))

    def base_right(self, R):
        r, E = self.r, self.E
        if self.rp is None:
            return
        ks = self.dec(E - 4e6, E - 5e5) if self.K == "agent" else self.dec(max(self.t_on - 2e6, r.T0), min(self.t_on + 2e6, E))
        n = dict(hit=0, any=0, speed=0, path=0, lat=0, dec=0, arc=[])
        for k in ks:
            P = self.rplan(k, "p0")
            if P is None:
                continue
            Sv = r.plan[k]
            n["dec"] += 1
            a_s, a_p = (np.hypot(*np.diff(q[:, :2], axis=0).T).sum() for q in (Sv, P))
            n["arc"].append(a_p / max(a_s, 0.5))
            a = min(a_s, a_p)
            la = [abs(r.lat(S.centre(reparam(q, np.c_[np.linspace(0, a, 41), np.zeros(41), np.zeros(41)]), r.off)[-1:, :2])[0, 0]) for q in (Sv, P)]
            n["lat"] += la[1] <= la[0] - 1.0
            if self._hit.get(int(k), self.hit(k)):
                n["hit"] += 1
                n["any"] += not self.hit(k, P)
                n["speed"] += not self.hit(k, reparam(Sv, P))
                n["path"] += not self.hit(k, reparam(P, Sv))
        R.update(br_dec=n["dec"], br_hit=n["hit"], p0_arc_ratio=round(float(np.median(n["arc"])), 2) if n["arc"] else "")
        for c in ("any", "speed", "path"):
            R[f"br_{c}"] = int(n[c] >= 0.5 * n["hit"]) if n["hit"] else ""
            R[f"br_{c}_share"] = round(n[c] / n["hit"], 2) if n["hit"] else ""
        R["br_lat"] = int(n["lat"] >= 0.5 * n["dec"]) if n["dec"] else ""
        if self.K == "corridor" and n["dec"]:
            R["br_path"] = int(bool(R["br_path"]) or bool(R["br_lat"]))
        lp = lambda z: 1 / (1 + np.exp(-np.asarray(z, float)[:, 0]))  # noqa: E731
        early = self.rp["now"] <= E - 1.5e6
        for nm in ("ft", "p0"):
            R[f"lead_seen_{nm}"] = int((lp(self.rp[f"lp_{nm}"])[early] >= 0.5).any()) if early.any() else ""

    def corridor(self, R):
        r, o = self.r, self.o
        ks = self.dec(self.t_on - 4e6, self.t_on + 1)
        s_a = max(self.s_on - 10.0, 0.0)
        fork_y = float(CL.into(np.r_[self.ref.xy(s_a)[0], self.ref.heading(s_a)], self.ref.xy(self.s_on + 30.0))[0, 1])   # Amendment 1: against the reference's own tangent
        t_g = max(self.t_on - 2e6, r.T0)                    # descriptive: the nearest object ahead in the ego's straight-ahead corridor 2 s before the onset
        eg, gap = r.ego_c(t_g)[0], np.nan
        for k in r.ids:
            p, ok = r.obj_pose(k, t_g)
            if ok[0] and r.o["actors"][k][-1, 0] >= t_g - 1e5:
                q = CL.into(eg, CL.corners(*p[0], *r.size(k)))
                g = q[:, 0].min() - r.Le / 2
                if q[:, 1].min() < r.We / 2 + 0.25 and q[:, 1].max() > -r.We / 2 - 0.25 and -1 < g < 60 and not g >= gap:
                    gap = g
        rt = "fork" if abs(fork_y) >= 3 and abs(deg(self.dpsi)) < 10 else self.rtype
        rts = np.array([q["t"] for q in o["route"]], float)
        fed, true, eff, push, dev = [], [], [], [], []
        for k in ks:
            now = r.now[k]
            rig = r.ego_rig(now)[0]
            wp = o["route"][int(np.abs(rts - now).argmin())]["wp"]
            wp = wp[np.isfinite(wp).all(1)]
            far = wp[np.hypot(*wp.T) >= 5.0]
            d0 = float(np.hypot(*far[0])) if len(far) else 5.0
            s0 = r.lat(rig[None, :2])[0, 1]
            g = np.arange(s0, s0 + 150.0, 0.5)
            q = CL.into(rig, self.ref.xy(g))
            q = q[np.hypot(*q.T) >= d0]
            true.append(1 if not len(q) else 0 if q[0, 1] > 2 else 2 if q[0, 1] < -2 else 1)
            i = self.rpi.get(int(now))
            fed.append(int(self.rp["cmd"][i]) if i is not None else -1)
            if len(wp):
                w = S.compose(rig, np.c_[wp, np.zeros(len(wp))])[:, :2]
                ls = r.lat(w)
                dev.append(float(ls[np.abs(ls[:, 1] - (self.s_on + 30.0)).argmin(), 0]))
            if i is not None:
                y = {n: float(self.rp[n][i][7, 1]) for n in ("ft", "ftL", "ftS", "ftR")}
                ct = true[-1]
                eff.append(y[("ftL", "ftS", "ftR")[ct]] - y["ftS"] if ct != 1 else y["ft"] - y["ftS"])
                push.append((y["ft"] - y["ftS"]) * np.sign(self.lat[-1]))
        mode = lambda x: Counter(x).most_common(1)[0][0] if x else -1  # noqa: E731
        cf, ct = mode([c for c in fed if c >= 0]), mode(true)
        effm = float(np.median(np.abs(eff))) if eff else np.nan
        rdev = float(np.median(dev)) if dev else np.nan
        exit_side = 0 if self.lat[-1] > 0 else 2
        if self.over:
            sub = "0 overran the logged stop"
        elif rt != "straight":
            sub = "1 route info" if cf != ct or abs(rdev) >= 3 else "2 info unused" if effm < 1 else "3 turn not made"
        else:
            sub = "1 route info" if cf != 1 and cf == exit_side else "4 drift"
        side = "" if rt == "straight" else "inside" if np.sign(self.lat[-1]) == self.sgn else "outside"
        R.update(c_ref=rt, fork_y=round(fork_y, 1), gap_ahead=round(float(gap), 1) if np.isfinite(gap) else "", cmd_fed=CMD[cf], cmd_true=CMD[ct], cmd_fed_share=round(float(np.mean(np.array(fed) == cf)), 2), route_dev=round(rdev, 1),
                 cmd_effect=round(effm, 2), cmd_push=round(float(np.median(push)), 2) if push else "", c_side=side, c_sub=sub)


def load(stacks=None):
    U, lite = [], {}
    for d in sorted((ROOT / "x").iterdir()):
        if not (d / "lite.pkl").exists() or (stacks and d.name not in stacks):
            continue
        lite[d.name] = CL.load(d / "lite.pkl")
        if (d / "logs.pkl").exists():
            L, M = CL.load(d / "logs.pkl"), CL.load(d / "map.pkl")
            for s, o in sorted(L.items()):
                f = ROOT / "replay" / f"{s}.npz"
                U.append(Unit(d.name, s, o, M.get(s), dict(np.load(f)) if f.exists() else None))
    return U, lite


def lite_row(stack, scene, o):
    """A rollout without a kept log: the summary row and the driver's own records."""
    m, sc = o["summary"]["score_metrics"], float(o["summary"]["score"])
    R = dict(stack=stack, scene=scene[7:15], score=round(sc, 4), lost=round(1 - sc, 4), flags="+".join(FLAG_K[f] for f in FLAG_K if m.get(f)) or "none")
    rec = o.get("rec", [])
    if sc == 0:
        R.update(K="none", raw="O1", cls="other")
    elif sc < 1 and rec:
        v0 = np.array([q["fix"]["v0"] for q in rec])
        lead = np.array([q["fix"]["lead"]["src"] != "e2e" for q in rec])
        v_log = o["summary"]["metrics"]["gt_dist_traveled_m"] / (len(rec) * 0.1)
        slow = v0 < 0.5 * v_log
        R.update(K="none", raw="L4", cls="longitudinal", l4=("follow" if slow.any() and lead[slow].mean() >= 0.5 else "free"), slow_share=round(float(slow.mean()), 2),
                 lead_share=round(float(lead.mean()), 2), lead_share_slow=round(float(lead[slow].mean()), 2) if slow.any() else "", v_mean=round(float(v0.mean()), 2), v_log_mean=round(v_log, 2),
                 progress=round(float(m["progress_clipped_rel"]), 3))
    else:
        R.update(K="", raw="", cls="pass")
    return R


def cmd_units(a):
    U, lite = load()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for u in U:
        if float(u.o["summary"]["score"]) > 0:
            continue
        rows.append(u.analyse())
        seen.add((u.stack, u.scene))
        print(rows[-1]["scene"], rows[-1]["flags"], rows[-1]["K"], rows[-1]["ref"], rows[-1]["raw"], rows[-1]["cls"], rows[-1].get("c_sub", ""), flush=True)
    for st, L in lite.items():
        rows += [lite_row(st, s, o) for s, o in sorted(L.items()) if (st, s) not in seen]
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(out / "units.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader(), w.writerows(rows)
    lost = sum(r["lost"] for r in rows)
    summ = dict(n=len(rows), mean=1 - lost / len(rows), lost=lost, by_class={c: dict(n=sum(r["cls"] == c for r in rows), lost=round(sum(r["lost"] for r in rows if r["cls"] == c), 3))
                                                                         for c in ("other", "longitudinal", "route", "clearance", "pass")})
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


def cmd_bev(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import col1_pai_clip as CC
    U, _ = load()
    out = Path(a.out) / "bev"
    out.mkdir(parents=True, exist_ok=True)
    for u in U:
        if a.scenes and u.scene[7:15] not in a.scenes or float(u.o["summary"]["score"]) > 0:
            continue
        R, r = u.analyse(), u.r
        fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
        b = ax[0]
        for e in (u.mp or {}).get("edges", []):
            b.plot(e[:, 0], e[:, 1], color="0.35", lw=1.0)
        for ln in (u.mp or {}).get("lanes", []):
            b.plot(ln["center"][:, 0], ln["center"][:, 1], color="0.85", lw=0.6)
        b.plot(*u.ref.p.T, "k--", lw=1.2, label="logged path (corridor reference)")
        b.plot(*u.rig[:, :2].T, color="#E69F00", lw=2, label="driven")
        k_on = int(np.abs(r.now - u.t_on).argmin())
        for kk, al in ((int(np.abs(r.now - (u.t_on - 2e6)).argmin()), 0.35), (k_on, 1.0), (int(np.abs(r.now - (u.E - 1e6)).argmin()), 0.6)):
            b.plot(*r.plan[kk][:, :2].T, color="#0072B2", lw=1.6, alpha=al, label="served plan (onset - 2 s, onset, event - 1 s)" if al == 1.0 else None)
            P = u.rplan(kk, "p0")
            if P is not None:
                b.plot(*P[:, :2].T, color="#CC79A7", lw=1.6, alpha=al, label="shipped weights, same tokens" if al == 1.0 else None)
        if u.rp is not None and "cmd_true" in R:
            P = u.rplan(k_on, ("ftL", "ftS", "ftR")[CMD.index(R["cmd_true"])] if R["cmd_true"] != "unknown" else "ftS")
            if P is not None:
                b.plot(*P[:, :2].T, color="#009E73", lw=1.6, ls=":", label=f"command forced to {R['cmd_true']} (rule on the logged path)")
        rts = np.array([q["t"] for q in u.o["route"]], float)
        wp = u.o["route"][int(np.abs(rts - u.t_on).argmin())]["wp"]
        wp = wp[np.isfinite(wp).all(1)]
        if len(wp):
            w = S.compose(r.ego_rig(u.t_on)[0], np.c_[wp, np.zeros(len(wp))])
            b.plot(w[:, 0], w[:, 1], "o", color="#009E73", ms=3, label="route sent at the onset")
        for t_, fc in ((u.t_on, "none"), (u.E, "#E69F0055")):
            b.add_patch(plt.Polygon(np.array(S.boxes(r.ego_c(t_), r.Le, r.We)[0].exterior.coords), ec="#D55E00", fc=fc, lw=1.5))
        for k in r.ids:
            p, ok = r.obj_pose(k, u.E)
            if ok[0] and np.hypot(*(p[0, :2] - u.rig[-1, :2])) < 60 and r.o["actors"][k][-1, 0] >= u.E - 2e5:
                b.add_patch(plt.Polygon(np.array(S.boxes(p, *r.size(k))[0].exterior.coords), ec="r" if k == u.struck else "0.4", fc="#ff000055" if k == u.struck else "none", lw=1))
        c = u.rig[-1, :2]
        b.set_xlim(c[0] - 45, c[0] + 45), b.set_ylim(c[1] - 45, c[1] + 45), b.set_aspect("equal"), b.legend(fontsize=7, loc="upper left")
        b.set_title(f"{R['scene']} {R['flags']} K={R['K']} ref={R['ref']} dpsi={R['dpsi_deg']} R={R['R_m']} raw={R['raw']} -> {R['cls']}\n"
                    f"in_plan={R.get('in_plan')} lead={R.get('lead_s')}s lat_E={R['lat_E']} v_on={R['v_on']} v_log={R['v_log_on']} hdg_div={R['hdg_div_deg']} "
                    f"br any/speed/path={R.get('br_any')}/{R.get('br_speed')}/{R.get('br_path')}\n"
                    f"{R.get('c_sub', '')} c_ref={R.get('c_ref', '')} fed={R.get('cmd_fed', '')} true={R.get('cmd_true', '')} route_dev={R.get('route_dev', '')} effect={R.get('cmd_effect', '')} push={R.get('cmd_push', '')}", fontsize=8)
        if u.rp is not None and len(u.rp["frame_k"]):
            nows = u.rp["now"][u.rp["frame_k"]]
            f = u.rp["frames"][int(np.abs(nows - u.t_on).argmin())]
            ax[1].imshow(np.vstack([CC.rgb(f[0]), CC.rgb(f[1])]))
            ax[1].set_title("model frames at the onset (road, wide)", fontsize=8)
        ax[1].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{R['cls']}_{R['K']}_{R['scene']}.png", dpi=70)
        plt.close(fig)
        print("bev", R["scene"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("units", "bev")), ap.add_argument("--out", default=str(ROOT / "out")), ap.add_argument("--scenes", nargs="*")
    a = ap.parse_args()
    from jevdrive.run import Run
    with Run("lowboard_diag", f"pai-{a.cmd}", config=vars(a)):
        dict(units=cmd_units, bev=cmd_bev)[a.cmd](a)


if __name__ == "__main__":
    main()
