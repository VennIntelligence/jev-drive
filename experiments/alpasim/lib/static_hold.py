"""JEV_HOLD: serve-side static hold for the nuPlan driver (experiments/alpasim_proxy/plans/2026-10-17-arena-prereg.md, supplement 1;
the scorer rule it targets: experiments/alpasim_proxy/results/corridor_rule.md). Off by default; served by lib/sh30_hold.py, which wraps
sh30_driver.py without changing it (JEV_HOLD unset = sh30_driver.main() untouched).

Why: when the logged human stands still to the end of the scored window, AlpaSim's corridor is a 4 m disc around its rest point, and an
ego that rolls >= 4 m from it is zeroed (RSG-L18-s0: 44 % of the arena's held static windows). Waiting there is free (the progress
score is 1 when the human moves < 5 m).

Latch: at the handoff call (the first drive call >= T_HAND s after the session's first call; serve_fix.ACC_T_HAND_NUPLAN, the call whose
plan is the first one driven), the measured ego speed (the force-GT speed = the human's) below V_LATCH m/s latches the hold for the
session at the ego position there (xy0); otherwise the session is free and nothing changes. While held, the served path is the model's
plan cut at arc s_lim from the ego (poses beyond it collapse onto the point at s_lim; the plan's geometry is kept up to there):
  JEV_HOLD=full       s_lim = 0: the ego is kept stopped for the whole session (the upper bound of the static gain and of the launch cost)
  JEV_HOLD=lead       s_lim = 0 until the lead moves: released for the rest of the session at the first call where the model's own plan
                      wants to move (2 s arc >= REL_ARC m, vhist.REL_ARC) and the checkpoint's own lead output has moved: a present lead
                      (radard's filtered probability > 0.5, lead_long.LeadPlanner.inside's filter) with speed > V_LEAD_GO m/s, or a lead
                      seen while held that is no longer present. No lead ever seen = never released (first in line, parked)
  JEV_HOLD=leash<m>   s_lim = max(0, m - |ego - xy0|) for the whole session: the ego never gets farther than m metres from the latch point
                      (e.g. leash3: 3.0 m, 1 m inside the corridor's 4 m), but launches up to there with the model's plan
Inputs are the driver's own: the measured ego speed and pose, the session clock, the model's plan and lead outputs (no log, no map).
"""
from __future__ import annotations

import os

import numpy as np

_raw = (os.environ.get("JEV_HOLD", "") or "off").strip().lower()
MODE, CAP = ("leash", float(_raw[5:])) if _raw.startswith("leash") else (_raw, 0.0)
assert MODE in ("off", "full", "lead", "leash") and CAP >= 0, f"JEV_HOLD={_raw!r}: off | full | lead | leash<m>"
ON, NEED_LEAD = MODE != "off", MODE == "lead"
SUFFIX = f"-hold{_raw}" if ON else ""
T_HAND = 0.5                                              # s: serve_fix.ACC_T_HAND_NUPLAN (nuPlan's handoff call)
V_LATCH = 0.3                                             # m/s: ego speed at the handoff below which the hold latches
REL_ARC = 2.0                                             # m: the model's 2 s arc that counts as "wants to move" (vhist.REL_ARC)
V_LEAD_GO = 1.0                                           # m/s: a lead faster than this has moved


def new(cam_to_front: float = 2.26) -> dict:
    st = dict(t_first=None, state=None, xy0=None, seen=False, k=0)
    if NEED_LEAD:
        from serve_fix import LeadPlanner
        st["lp"] = LeadPlanner(cam_to_front)              # its probability filter only
    return st


def arc2(poses) -> float:
    p = np.r_[np.zeros((1, 2)), np.asarray(poses, float)[:4, :2]]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def limit_arc(poses: np.ndarray, s_lim: float) -> np.ndarray:
    """Plan poses (8, 3) in the rig frame (x, y, yaw from the ego at the origin) -> the poses beyond arc s_lim along the plan's polyline
    replaced by the point at s_lim (yaw interpolated alike); poses within s_lim are returned unchanged (bit for bit)."""
    P = np.asarray(poses, np.float64)
    p = np.r_[np.zeros((1, 3)), P]
    c = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p[:, :2], axis=0).T))]
    m = c[1:] > s_lim
    if not m.any():
        return poses
    out = P.copy()
    k = int(np.searchsorted(c, s_lim, side="right"))      # s_lim lies on segment k - 1 -> k (c[k - 1] <= s_lim < c[k])
    f = (s_lim - c[k - 1]) / max(c[k] - c[k - 1], 1e-12)
    out[m] = p[k - 1] + f * (p[k] - p[k - 1])
    return out.astype(np.asarray(poses).dtype)


def step(st: dict, poses, v: float, xy, t_us: int, o: dict | None = None) -> tuple[np.ndarray, dict]:
    """One decision: the model's plan (8, 3), the measured ego speed (m/s), the ego position in the rollout's local frame, the call time
    (us), the core's outputs (lead / lead_prob / mu for JEV_HOLD=lead) -> (the served poses, the log record)."""
    st["k"] += 1
    st["t_first"] = t_us if st["t_first"] is None else st["t_first"]
    t_s = (t_us - st["t_first"]) / 1e6
    xy = np.asarray(xy, np.float64)
    if st["state"] is None and t_s >= T_HAND - 1e-6:
        st["state"], st["xy0"] = ("held" if v < V_LATCH else "free"), xy
    info = dict(mode=MODE, t_s=round(t_s, 3), v=round(float(v), 3))
    if NEED_LEAD and o is not None and "lead" in o:
        lp = st["lp"]
        lp.inside(v, o["lead"], o["lead_prob"], float(o["mu"][0, 3]))
        mu = np.asarray(o["lead"], np.float64)[:72].reshape(3, 6, 4)
        present, vl = bool(lp.prob0 > 0.5), max(v + float(mu[0, 0, 2]) - float(o["mu"][0, 3]), 0.0)
        info.update(lead_p=round(float(lp.prob0), 3), lead_v=round(vl, 2), lead_d=round(float(mu[0, 0, 0]) - lp.front, 2))
        if st["state"] == "held":
            moved = (present and vl > V_LEAD_GO) or (st["seen"] and not present)
            st["seen"] |= present
            if moved and arc2(poses) >= REL_ARC:
                st["state"], info["release"] = "free", "lead_moved" if present else "lead_gone"
    info["state"] = st["state"] or "pre"
    if st["state"] != "held":
        return poses, info
    d = float(np.hypot(*(xy - st["xy0"])))
    s_lim = max(0.0, CAP - d) if MODE == "leash" else 0.0
    out = limit_arc(poses, s_lim)
    info.update(d0=round(d, 3), s_lim=round(s_lim, 3), arc_model=round(arc2(poses), 3), cut=out is not poses)
    return out, info
