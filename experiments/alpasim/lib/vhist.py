"""JEV_VHIST: the speed history fed to the adapter at serving time (experiments/route_follow/plans/2026-10-13-vhist-prereg.md). Off by default.

Why: research/next-round/steer-long.md 4.4 / 5 (2), decision 285: the slowness is a ratchet through the policy's own speed-history input (the
plan brakes a little -> the fed history is slower -> the next plan is slower). Only the ego row fed to the adapter changes; the synthesised
frames, the served trajectory and the logged `ego` (the measured row) are untouched. The fed row is logged by the drivers as `vhist`.

Speed-history entries = rm_train's EGO_SPEED: ego_features columns 4-9, 11, 12, 14, 15, 17, 18 (vx, vy, ax, ay, history poses x / y;
`present`, the command and the history yaws stay).

  JEV_VHIST=zero  those entries are 0 at every decision: the serving counterpart of rm_train --ego-drop-speed (for RSGallV-type checkpoints).
  JEV_VHIST=hold  the row of a car that drove the same path at v_fed = max(v_meas, v_ref), v_ref = the running maximum of the measured speed
                  over the session's decisions (updated at every decision). Released (the measured row fed) when the route command is left /
                  right or the previous decision's model plan has a 2 s arc below REL_ARC m (the model asks to stop or crawl). When held and
                  v_fed > v_meas: (vx, vy) scaled by v_fed / v_meas ((v_fed, 0) below 0.5 m/s), ax = 0, ay unchanged, the 3 past poses
                  re-spaced along their own polyline (extended straight beyond the oldest) to arcs v_fed x (0.5, 1.0, 1.5) s, yaw
                  interpolated alike (parity_adapter.cv_history's rule at v_fed).
  JEV_VHIST=holdlead  hold with two more releases (prereg Amendment 1), both read from the PREVIOUS decision's outputs of the same model
                  (the current pass depends on the fed row): (c) a lead inside openpilot's following distance: serve_fix.LeadPlanner.inside
                  on the previous decision's lead / lead_prob outputs at that decision's measured speed (radard's asymmetric probability
                  filter > 0.5 and dRel + v_lead^2 / (2 x 2.5) < v^2 / (2 x 2.5) + 1.45 v + 6; JEV_BASE=2's rule), and (d) the model's
                  previous plan brakes: segment speeds s1 (0 .. 0.5 s) and s4 (1.5 .. 2.0 s) with (s1 - s4) / 1.5 s >= DEC_REL m/s^2.
                  Needs the lead outputs (NEED_LEAD: the drivers set core.lead_out); the served trajectory is unchanged.
Inputs are the driver's own: the measured ego state, the route command and the model's previous outputs (no log, no map).
"""
from __future__ import annotations

import os

import numpy as np

MODE = (os.environ.get("JEV_VHIST", "") or "off").strip().lower()
assert MODE in ("off", "hold", "zero", "holdlead"), f"JEV_VHIST={MODE!r}: off | hold | zero | holdlead"
ON, NEED_LEAD = MODE != "off", MODE == "holdlead"
SUFFIX = f"-vh{MODE}" if ON else ""
EGO_SPEED = [4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]
REL_ARC = 2.0                                             # m: a previous plan shorter than this over 2 s releases the hold
DEC_REL = 1.0                                             # m/s^2: a previous plan braking at least this (s1 -> s4) releases holdlead
T_HIST = np.array([1.5, 1.0, 0.5])                        # s before t0 of the 3 past poses (oldest first)


def new(cam_to_front: float = 2.26) -> dict:
    st = dict(vref=0.0, arc=np.inf, dec=0.0, lead=False, v=0.0)
    if NEED_LEAD:
        from serve_fix import LeadPlanner
        st["lp"] = LeadPlanner(cam_to_front)              # its probability filter and following distance only
    return st


def arc2(poses) -> float:
    """Arc length (m) of a plan from the rear axle through its 0.5 .. 2.0 s poses (rf_follow.plan_arc2)."""
    p = np.r_[np.zeros((1, 2)), np.asarray(poses, float)[:4, :2]]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def respace(P: np.ndarray, v: float) -> np.ndarray:
    """P (4, 3) history poses x, y, yaw (m, rad; oldest first, the last = t0) -> the 3 past poses moved along the polyline newest -> oldest
    (extended 100 m straight beyond the oldest) to arc lengths v x (0.5, 1.0, 1.5) s; yaw interpolated alike."""
    Q = P[::-1].astype(np.float64)
    d = Q[1:, :2] - Q[:-1, :2]
    L = np.linalg.norm(d, axis=1)
    back = np.array([-np.cos(Q[-1, 2]), -np.sin(Q[-1, 2])])
    u = d[-1] / L[-1] if L[-1] > 1e-4 else back
    Qe = np.r_[Q, [[*(Q[-1, :2] + 100.0 * u), Q[-1, 2]]]]
    c = np.r_[0.0, np.cumsum(L), L.sum() + 100.0]
    a = v * T_HIST[::-1]
    out = np.stack([np.interp(a, c, Qe[:, j]) for j in range(3)], 1)    # newest-first arcs 0.5 v, 1.0 v, 1.5 v
    return np.r_[out[::-1], P[3:4]]


def feed(st: dict, ego: np.ndarray, cmd) -> tuple[np.ndarray, dict]:
    """One decision: the measured ego row (20,) -> (the fed row, the log record). Updates st['vref']."""
    v = float(np.hypot(ego[4], ego[5]) * 10.0)
    if MODE == "zero":
        e = ego.copy()
        e[EGO_SPEED] = 0.0
        return e, dict(mode=MODE, v_meas=round(v, 3), v_fed=0.0)
    st["vref"], st["v"] = max(st["vref"], v), v
    rel = "turn" if (cmd[0] > 0 or cmd[2] > 0) else "stop" if st["arc"] < REL_ARC else ""
    if not rel and NEED_LEAD:
        rel = "lead" if st["lead"] else "brake" if st["dec"] >= DEC_REL else ""
    vf = v if rel else max(v, st["vref"])
    info = dict(mode=MODE, v_meas=round(v, 3), v_ref=round(st["vref"], 3), v_fed=round(vf, 3), release=rel)
    if NEED_LEAD:
        info.update(prev_lead=bool(st["lead"]), prev_dec=round(float(st["dec"]), 3), prev_arc=round(float(st["arc"]), 3))
    if vf <= v + 1e-6:
        return ego, info
    e = ego.copy()
    e[4:6] = ego[4:6] * (vf / v) if v >= 0.5 else (vf / 10.0, 0.0)
    e[6] = 0.0
    P = ego[8:20].reshape(4, 3).astype(np.float64) * [10.0, 10.0, 1.0]
    e[8:20] = (respace(P, vf) / [10.0, 10.0, 1.0]).reshape(12)
    return e.astype(np.float32), info


def observe(st: dict, o: dict) -> None:
    """After the plan (o = the core's output, poses = the model's plan): remember what the next decision's release rules read."""
    p = np.r_[np.zeros((1, 2)), np.asarray(o["poses"], float)[:4, :2]]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.5
    st["arc"], st["dec"] = float(seg.sum() * 0.5), float((seg[0] - seg[3]) / 1.5)
    if NEED_LEAD:
        st["lead"] = st["lp"].inside(st["v"], o["lead"], o["lead_prob"], float(o["mu"][0, 3]))
