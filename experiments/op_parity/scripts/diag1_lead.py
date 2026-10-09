"""Lane DIAG1, job 1 (b): lane FIX1's lead-aware longitudinal limit (experiments/alpasim/lib/serve_fix.py, as committed by FIX1; the law is
not re-implemented here) applied open loop to stored WOD val predictions.

What is taken from serve_fix verbatim: LeadPlanner (radard's vision lead, the lead MPC, the planner tick), T_MPC, T_TRACK, and the rule of
Serve.__call__: cap = max(v0 + v_mpc(t) - v_mpc(0), 0), served speed = min(profile without the switch, cap), a_e2e = the profile's
acceleration over the first T_TRACK seconds. What differs, and only because the board differs: the waypoint grid (20 x 0.25 s to 5 s instead
of 8 x 0.5 s to 4 s; the MPC horizon is 10 s, so the cap exists on the whole grid), and the planner has no session: each frame gets a fresh
LeadPlanner, one reset tick, then `ticks` more 20 Hz ticks with the frame's outputs and ego state held (default 20 = 1 s; 0 = the first
decision of a session). Inputs per frame: the stored shipped lead / lead_prob outputs of the same harness run (decoded means and
probabilities, turned back into the raw layout serve_fix reads), the model's own speed (plan velocity at t = 0), the fed ego speed, WOD's
accel_x of the last past state, camera to front bumper = diag1.WOD_FRONT - the camera's x on the vehicle.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R / "experiments/alpasim/lib"), str(_pl.Path(__file__).parent)]

import numpy as np  # noqa: E402

import serve_fix as F  # noqa: E402


def wod_cap(C, p, vcont: float = 0.0, ticks: int = 20, info: dict | None = None):
    """C = diag1.wod_ctx(); p (N, 20, 2) predictions -> the predictions served with FIX1's switches (b) [vcont = 0] or (a) + (b)."""
    import diag1 as D
    out = np.empty_like(p)
    cut, src = np.zeros(len(p)), np.zeros(len(p), bool)
    for i in range(len(p)):
        v0, a0 = max(float(C.vfed[i]), 0.0), float(C.acc[i])
        _, vp, t = D.retime_xy(p[i], v0, 1e-3)
        v = v0 + (vp - v0) * np.minimum(t / vcont, 1.0) if vcont > 0 else vp
        lead = np.concatenate([C.lead[i].ravel(), np.zeros(72)])
        pr = np.clip(C.lp[i], 1e-6, 1 - 1e-6)
        logit = np.log(pr / (1 - pr))
        a_e2e = (float(np.interp(F.T_TRACK, t, v)) - v0) / F.T_TRACK
        pl = F.LeadPlanner(D.WOD_FRONT - float(C.dev[i, 0]))
        vs, li = pl.update(0, v0, a0, lead, logit, float(C.pvel0[i]), a_e2e)
        if ticks:
            vs, li = pl.update(int(ticks * F.DT_MDL * 1e6), v0, a0, lead, logit, float(C.pvel0[i]), a_e2e)
        cap = np.maximum(v0 + np.interp(t, F.T_MPC, vs) - vs[0], 0.0)
        cut[i], src[i] = float(np.sum(np.maximum(v - cap, 0.0)) * 0.05), li["src"] == "mpc"
        out[i] = D.retime_xy(p[i], np.minimum(v, cap))[0]
    if info is not None:
        info.update(cut_m=cut, mpc=src)
    return out
