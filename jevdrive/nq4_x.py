"""Night queue 4, X (mode head -> geometric path -> P7) and the cross-fitted heads the G / X examinees drive with
(todos/2026-09-26-night-queue-4.md, sections G / X and the [F] entries, written before any X number).

The apply side (XState, fold_pick) is numpy only and runs inside the closed-loop agent (scripts/nq4_x_agent.py) and in the
offline check alike.

  XState.step(...)  one 5 Hz plan: mode of the Q2 mode head + the head's trajectory -> the path handed to P7
      bypass_L / bypass_R  PDM-Lite's geometry (team_code/privileged_route_planner.shift_route_smoothly): the dense route
                          shifted to the neighbouring lane's centre (CARLA map get_left_lane / get_right_lane of each route
                          waypoint, the route point itself where there is none), full shift (factor 1), smooth transition
                          over 8 m (PDM-Lite's transition_smoothness_distance), in from where the ego is when the head first
                          says bypass, held until 30 m past the ego's position at the last bypass output, then 8 m back;
                          the trajectory's speed profile (its arc length at 0.25 ... 5 s) is kept and laid along that path
      stop / wait          target speed 0: every point at the origin (P7 brakes with its own law)
      keep                 the head's own trajectory (= CL5), unless a shift is still active (then the shifted path)
  export-q2 / export-mc    (repo .venv, GPU) the Q2 head (lane C's chosen arms) and the M-C heads refitted on one fold's
                           routes (K's route_split.json), written to runs/nq4/gk/heads_xfit/{q2,mc}/R{1,2}, then READY
  check <attempt dir>      rule 8: replay the agent's per-plan dump (x_dump.jsonl) through XState offline, bit for bit
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MODES = ("keep", "stop", "bypass_L", "bypass_R", "wait")
TRANSITION_M = 8.0          # PDM-Lite config.transition_smoothness_distance (8 points at 1 point / m)
HOLD_M = 30.0               # shift held this far past the ego's position at the last bypass output
TIMES = np.arange(1, 21) * 0.25


def smooth(u):
    """PDM-Lite's _smooth_transition: cosine ease, 0 -> 1 over u in [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def fold_pick(split: dict, base: str, pick: str) -> str:
    """The fold whose readout drives this route. K's rule: a recorded route has its own fold; 'unseen' = the other fold,
    'seen' = its own; a route with no fold label is driven by R1."""
    r = split.get(str(base))
    if r is None or "fold" not in r:
        return "R1"
    own = r["fold"]
    return own if pick == "seen" else ("R2" if own == "R1" else "R1")


def base_of(route_id) -> str:
    """Official ids are < 100 000; night-queue-4 G variants are 100 b + 90 + code (jevdrive.nq4_g.vid)."""
    i = int(route_id)
    return str(i // 100) if i >= 100000 else str(i)


def to_ego(P, xy0, yaw0):
    """CARLA world points -> rear-axle ego frame (x forward, y left), as b2d_zeroshot_agent._history."""
    c, s = math.cos(yaw0), math.sin(yaw0)
    d = np.asarray(P, float) - np.asarray(xy0, float)
    return np.stack([c * d[:, 0] + s * d[:, 1], s * d[:, 0] - c * d[:, 1]], -1)


class XState:
    def __init__(self, route_xy, left_xy, right_xy):
        self.P = np.asarray(route_xy, float)
        self.N = {2: np.asarray(left_xy, float), 3: np.asarray(right_xy, float)}      # NaN where there is no lane
        self.s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.P, axis=0).T))]
        self.side = None            # 2 (left) / 3 (right) while a shift is active
        self.s_in = self.s_last = None
        self.ptr = 0

    def _arc(self, xy):
        j = self.ptr + int(np.argmin(np.hypot(self.P[self.ptr:self.ptr + 60, 0] - xy[0], self.P[self.ptr:self.ptr + 60, 1] - xy[1])))
        self.ptr = j
        return float(self.s[j])

    def weight(self, s):
        """Shift weight along the route: 0 before s_in, cosine in over 8 m, 1, cosine out over 8 m after s_last + 30 m."""
        w_in = smooth((s - self.s_in) / TRANSITION_M)
        w_out = 1.0 - smooth((s - (self.s_last + HOLD_M)) / TRANSITION_M)
        return np.minimum(w_in, w_out)

    def step(self, mode, traj, xy0, yaw0):
        """mode (int or None), traj (20, 2) the head's trajectory in the ego frame of pose (xy0, yaw0; rear axle, CARLA
        world, yaw rad) -> (path (20, 2), info). Deterministic given the call sequence."""
        traj = np.asarray(traj, float)
        s0 = self._arc(xy0)
        info = {"mode": None if mode is None else int(mode), "s_ego": round(s0, 3)}
        if mode in (2, 3):
            if self.side is None or s0 > self.s_last + HOLD_M + TRANSITION_M:
                self.side, self.s_in = int(mode), s0
            self.s_last = s0
        if self.side is not None and s0 > self.s_last + HOLD_M + TRANSITION_M:
            self.side = None
        if mode in (1, 4):
            info["x"] = "stop"
            return np.zeros((20, 2)), info
        if self.side is None:
            info["x"] = "head"
            return traj, info
        # the shifted route ahead of the ego, in the ego frame
        j0 = max(self.ptr - 2, 0)
        P, N, s = self.P[j0:], self.N[self.side][j0:], self.s[j0:]
        w = self.weight(s)
        N = np.where(np.isfinite(N), N, P)
        Q = to_ego(P + w[:, None] * (N - P), xy0, yaw0)
        # the head's speed profile (its arc length at each time) laid along the shifted path from the ego's foot point
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], traj], axis=0).T))][1:]
        q = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
        k = int(np.argmin(np.hypot(Q[:, 0], Q[:, 1])))
        target = q[k] + d
        path = np.stack([np.interp(target, q, Q[:, i]) for i in range(2)], -1)
        over = target > q[-1]                         # past the end of the route: straight on along the last segment
        if over.any():
            u = (Q[-1] - Q[-2]) / max(np.hypot(*(Q[-1] - Q[-2])), 1e-9)
            path[over] = Q[-1] + (target[over] - q[-1])[:, None] * u
        info.update(x="shift", side=int(self.side), s_in=round(self.s_in, 3), s_last=round(self.s_last, 3),
                    w_ego=round(float(self.weight(np.array([s0]))[0]), 4))
        return path, info


# ---------------------------------------------------------------- offline check (rule 8)

def check(adir: Path) -> dict:
    """Replay x_dump.jsonl (every plan: mode, head trajectory, pose, the path the agent handed to P7) and x_route.npz
    through a fresh XState; every path must be identical to the bit."""
    z = np.load(adir / "x_route.npz")
    st = XState(z["route"], z["left"], z["right"])
    n = bad = 0
    worst = 0.0
    for line in open(adir / "x_dump.jsonl"):
        r = json.loads(line)
        if r.get("warmup"):
            continue
        path, _ = st.step(r["mode"], np.array(r["traj"]), np.array(r["pose"][:2]), r["pose"][2])
        dif = float(np.abs(path - np.array(r["path"])).max())
        worst = max(worst, dif)
        bad += dif != 0.0
        n += 1
    return {"plans": n, "differing": bad, "max_abs_diff_m": worst}


# ---------------------------------------------------------------- cross-fitted heads

def xfit_root(*p) -> Path:
    from .common import data_dir
    d = data_dir() / "runs" / "nq4" / "gk" / "heads_xfit" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _split():
    from .common import data_dir
    return json.loads((data_dir() / "runs" / "nq4" / "k" / "route_split.json").read_text())["routes"]


def export_q2(rl, placeholder: bool = False):
    """Lane C's closed-loop head (its manifest's trajectory / mode arms), refitted once per fold on the P6 v0 training worlds
    of that fold's routes (K's split: the fold label of each P6 base route), nq3_q2.fit_fold otherwise unchanged.
    placeholder: before lane C's READY, the registered fallbacks (trajectory A1, mode head A2) into heads_xfit/q2_placeholder,
    only for the smoke and the rule-8 check of the X agent; the chain drives with heads_xfit/q2 (lane C's arms) only."""
    import pandas as pd
    from . import nq3_q2 as Q
    from .common import data_dir
    if placeholder:
        man, name = {"trajectory_arm": "A1", "mode_arm": "A2", "model": "cinque", "placeholder": True}, "q2_placeholder"
    else:
        man, name = json.loads((data_dir() / "runs" / "nq3" / "q2" / "closed_loop_head" / "manifest.json").read_text()), "q2"
    arm, mode_arm, model = man["trajectory_arm"], man["mode_arm"], man.get("model", "cinque")
    split = _split()
    D = Q.load("carla_p6", (model,))
    base = D["t"].base_id.astype(str).to_numpy()
    for fk in ("R1", "R2"):
        out = xfit_root(name, fk)
        mine = np.array([split.get(b, {}).get("fold") == fk for b in base])
        fold = np.where(mine, 0, 1)
        head = {}
        chk = np.random.default_rng(0).choice(np.flatnonzero(~mine), 1024, replace=False)
        r = Q.fit_fold(D, fold, 1, model, 0, tuple(sorted({"A0", arm, mode_arm})), None, head, rl, te_rows=chk)
        np.savez(out / "head.npz", **{k: np.asarray(v) for k, v in head.items() if not isinstance(v, (float, int))})
        m = dict(man, fit={"set": "carla_p6", "fold": fk, "routes": sorted(set(base[mine])), "rows": int(r["info"]["n_train"]),
                           "pairs": int(r["info"]["n_pairs"]), **{k: v for k, v in r["info"].items() if k.startswith("lam")}},
                 scalars={k: v for k, v in head.items() if isinstance(v, (float, int))}, written=pd.Timestamp.now().isoformat())
        (out / "manifest.json").write_text(json.dumps(m, indent=1, default=float))
        from .nq3_head import Head
        tr_, md_ = Head(out)(D["op"][model][chk].cpu().numpy(), D["Ego"][chk].cpu().numpy())
        dt, dm = float(np.abs(tr_ - r["pred"][arm]).max()), float((md_ != r["pred"][f"{mode_arm}_mode"]).mean())
        m["check"] = {"rows": len(chk), "traj_max_abs_diff_m": dt, "mode_mismatch_share": dm}
        (out / "manifest.json").write_text(json.dumps(m, indent=1, default=float))
        rl.event("export_q2", fold=fk, rows=int(r["info"]["n_train"]), traj_diff=dt, mode_mismatch=dm)
        assert dt <= 1e-3 and dm <= 1e-3, (fk, dt, dm)
    (xfit_root(name) / "READY").write_text(json.dumps({"arm": arm, "mode_arm": mode_arm, "model": model}))


def export_mc(rl):
    """M-C (prior + `pair` dual stream, lane B's jevdrive.nq3_cl.export) refitted once per fold on the P5 v1 BA training
    rows and pair rows of that fold's routes; heads.npz in lane B's format (the numpy Heads loads it)."""
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from . import elicit_i3 as I, nq3_cl as CL, p5_exam as E, p5_openpilot, p5_pairs as P, planner
    from . import reactivity_mc as MC, waymo_stage_a as sa
    split = _split()
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        op = p5_openpilot.load(t, ("cinque",), sub="op_streams_vis")["op-cinque temporal"]
        Q = P.load_features(t, ("L18_last",))["L18_last"]
    n = len(t)
    base = t.base_id.astype(str).to_numpy()
    ego = E.ego_input(t, past)
    Fc = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    Ego, Xop, Qc = (torch.as_tensor(x, device="cuda") for x in (ego, op, Q))
    pos = pd.Series(np.arange(n), index=t.frame_name)
    ip_all = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    im_all = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    grp_all = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    for fk in ("R1", "R2"):
        mine = np.array([split.get(b, {}).get("fold") == fk for b in base])
        tr = np.flatnonzero((t.role.to_numpy() == "train") & mine)
        sp = SimpleNamespace(train=tr, val=tr, seq=base)
        out = {}

        def stats(X):
            mu, sd = X[tr].double().mean(0), X[tr].double().std(0, correction=0)
            return mu.float().cpu().numpy(), torch.where(sd > 1e-6, sd, 1).float().cpu().numpy()
        Xe = planner.standardize(Ego, tr)
        _, st_e, We = sa.ridge_cv(Xe, Fc, sp, fut)
        b0 = planner.linear_apply(We, Xe, np.arange(n))[0]
        Xi = planner.standardize(Xop, tr)
        R0 = Fc - b0
        _, st_p, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(n, 20, 2).cpu().numpy())
        prior = b0 + planner.linear_apply(Wp, Xi, np.arange(n))[0]
        out["ego_mu"], out["ego_sd"] = stats(Ego)
        out["op_mu"], out["op_sd"] = stats(Xop)
        out["We"], out["Wp"] = We[0].cpu().numpy(), Wp[0].cpu().numpy()
        keep = mine[ip_all] & mine[im_all]
        ip, im, grp = ip_all[keep], im_all[keep], grp_all[keep]
        Rp = (Fc[ip] - Fc[im]) - (prior[ip] - prior[im])
        Z = torch.cat([MC._std(Qc, tr), MC._std(Xop, tr)], 1)
        Zc = Z[tr] - Z[tr].mean(0)
        mu = len(ip) / len(tr)
        Dz = Z[ip] - Z[im]
        score = np.zeros(len(MC.LAMS))
        for a, b in MC._inner_splits(grp):
            Ws = MC._solve_pair(Dz[a], Rp[a], Zc, mu, MC.LAMS)
            score += [float(((Dz[b] @ W - Rp[b]) ** 2).sum()) for W in Ws]
        best = int(np.argmin(score))
        W = MC._solve_pair(Dz, Rp, Zc, mu, [MC.LAMS[best]])[0]
        out["mcq_mu"], out["mcq_sd"] = stats(Qc)
        out["mco_mu"], out["mco_sd"] = stats(Xop)
        out["mc_zbar"], out["mc_W"] = Z[tr].mean(0).cpu().numpy(), W.cpu().numpy()
        mc_pred = prior + (Z - Z[tr].mean(0)) @ W
        d = xfit_root("mc", fk)
        np.savez(d / "heads.npz", **{k: np.asarray(v) for k, v in out.items()})
        H = CL.Heads(d / "heads.npz")
        rows = np.random.default_rng(0).choice(n, 300, replace=False)
        dif = max(float(np.abs(H.predict("mc", ego[r], op[r], q=Q[r]).ravel() - mc_pred[r].cpu().numpy()).max()) for r in rows)
        meta = {"set": I.BA, "fold": fk, "routes": sorted(set(base[mine])), "n_train": int(len(tr)), "n_pair": int(len(ip)),
                "lam_ego": st_e["lam"], "lam_op": st_p["lam"], "lam_mc": float(MC.LAMS[best]),
                "lam_edge": best in (0, len(MC.LAMS) - 1), "apply_check_max_m": dif}
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
        rl.event("export_mc", **{k: v for k, v in meta.items() if k != "routes"})
        assert dif < 1e-2, (fk, dif)
    (xfit_root("mc") / "READY").write_text("ok")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export-q2", "export-q2-placeholder", "export-mc", "check"))
    ap.add_argument("dirs", nargs="*")
    a = ap.parse_args()
    if a.cmd == "check":
        res = {d: check(Path(d)) for d in a.dirs}
        print(json.dumps(res, indent=1))
        raise SystemExit(0 if all(r["differing"] == 0 and r["plans"] > 0 for r in res.values()) else 1)
    rl = RunLog("nq4_x", a.cmd)
    if a.cmd == "export-mc":
        export_mc(rl)
    else:
        export_q2(rl, placeholder=a.cmd == "export-q2-placeholder")
    rl.close()


if __name__ == "__main__":
    main()
