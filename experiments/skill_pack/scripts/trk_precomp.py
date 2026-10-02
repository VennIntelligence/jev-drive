"""Tracker-lag pre-compensation of a submitted NAVSIM trajectory (plan: ../plans/2026-10-04-tracker-precomp-plan.md).

The NAVSIM v1.1 / v2 scorers (identical simulator code) track the 8 submitted poses with a one-step LQR (1 s lookahead,
lateral Q = diag(1, 10, 0)) on a kinematic bicycle whose steering angle starts at 0 and passes a 0.05 s low-pass. This
lags the commanded lateral motion. Here, per token, the submitted poses u are chosen so that the *simulated* trajectory
matches the model's intended plan p (the uncompensated 8 poses, linearly interpolated as the scorer does):

    min_u  sum_t |sim(u)_xy(t) - p_xy(t)|^2 + w_h^2 sum_t wrap(sim(u)_h(t) - p_h(t))^2 + lam |u - p|^2

with Levenberg-Marquardt on finite-difference Jacobians (the devkit's own batched PDMSimulator runs all 25 perturbed
proposals in one call). The submission is u_alpha = p + alpha (u* - p); alpha is fitted on navtrain only.
Reference-free: the only per-token inputs are the plan and the t0 ego velocity / acceleration (the agent's ego_status);
the simulator's start state has steering angle 0 for every token (checked), the vehicle is the devkit's Pacifica.

navsim2 env, CPU.
  check   --data lb_navhard            own dense path == official get_trajectory_as_array, sim equality, tracker response
  make    --data <run> --src <npz> --tag <name> --alphas 0.5 1.0    compensated pose files + tracking diagnostics
  report  ...                          paired bootstrap from the official CSVs (see trk_report.py)
"""
import argparse
import glob
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offroad_lib as L  # noqa: E402

CACHES = {"lb_navhard": "v2_navhard_two_stage", "lb_navtest": "v1_navtest", "lb_navtrain": "v1_navtrain_oplb"}
OUT = L.D / "runs/skill_pack/trk"
W_H, LAM, ITERS = 2.0, 1e-2, 12
EPS = np.array([1e-3, 1e-3, 1e-4])
MODES = ("full",)   # a y-and-heading-only variant was dropped before scoring: it does not isolate lateral (see plan)
# "path" (post hoc, plan addendum 1): realise the plan's path (offset to the plan polyline and heading at the matching arc
# position), keep the arc-length progress of the uncompensated simulation (the tracker's own speed profile).
W_S = 1.0
_W = {}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cache_paths(data):
    return {Path(p).parent.name: p for p in glob.glob(str(L.D / "runs/navsim/metric_cache" / CACHES[data] / "*/*/*/metric_cache.pkl"))}


def proposals(mc, U):
    """U (B, 8, 3) ego-frame poses -> (B, 41, 11) proposal state arrays as the scorer builds them (linear interpolation
    between the t0 state and the poses; only x, y, heading of rows >= 1 are read by the tracker)."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    s0 = ego_state_to_state_array(mc.ego_state)
    o = s0[:3]
    c, s = np.cos(o[2]), np.sin(o[2])
    out = np.zeros((len(U), 41, len(s0)))
    out[:, 0] = s0
    for b, u in enumerate(U):
        d = L.dense_from_poses(u)[1:]
        out[b, 1:, 0] = o[0] + c * d[:, 0] - s * d[:, 1]
        out[b, 1:, 1] = o[1] + s * d[:, 0] + c * d[:, 1]
        out[b, 1:, 2] = wrap(o[2] + d[:, 2])
    return out


def sim_ego(sim, mc, U):
    """Simulated ego-frame (B, 41, 3) of submitted pose sets U."""
    st = sim.simulate_proposals(proposals(mc, U), mc.ego_state)
    return L.to_ego(mc, st[..., :3])


def plan_err(S, target):
    """(.., 41, 2) longitudinal / lateral error of simulated poses in the plan's frame at each time."""
    e = S[..., :2] - target[:, :2]
    c, s = np.cos(target[:, 2]), np.sin(target[:, 2])
    return np.stack([c * e[..., 0] + s * e[..., 1], -s * e[..., 0] + c * e[..., 1]], -1)


def path_proj(S, target):
    """Project (.., 41, 3) poses onto the plan polyline (extended 20 m beyond both ends): arc length s, signed offset d
    (left +), path heading at the foot point."""
    P = target[:, :2]
    ext0 = P[0] - 20 * np.array([np.cos(target[0, 2]), np.sin(target[0, 2])])
    ext1 = P[-1] + 20 * np.array([np.cos(target[-1, 2]), np.sin(target[-1, 2])])
    V = np.vstack([ext0, P, ext1])
    A, B = V[:-1], V[1:]
    seg = B - A
    ln = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-9)
    cum = np.r_[0, np.cumsum(ln)][:-1] - 20.0
    q = S[..., :2][..., None, :] - A                                   # (.., 41, nseg, 2)
    tt = np.clip((q * seg).sum(-1) / ln ** 2, 0, 1)
    foot = A + tt[..., None] * seg
    dd = np.hypot(*np.moveaxis(S[..., :2][..., None, :] - foot, -1, 0))
    k = dd.argmin(-1)
    kk = k[..., None]
    t_k = np.take_along_axis(tt, kk, -1)[..., 0]
    hd = np.arctan2(seg[k, 1], seg[k, 0])
    q_k = np.take_along_axis(q, kk[..., None], -2)[..., 0, :]
    side = np.sign(np.cos(hd) * q_k[..., 1] - np.sin(hd) * q_k[..., 0])
    return cum[k] + t_k * ln[k], side * np.take_along_axis(dd, kk, -1)[..., 0], hd


def residual(S, target, u, p, mode, aux=None):
    if mode == "path":
        s_, d_, hd = path_proj(S, target)
        r = [d_[..., 1:], W_H * wrap(S[..., 1:, 2] - hd[..., 1:]), W_S * (s_[..., 1:] - aux[1:]), np.sqrt(LAM) * (u - p).reshape(len(S), -1)]
        return np.concatenate(r, 1)
    pe = plan_err(S, target)[..., 1:, :]
    r = [pe.reshape(len(S), -1),
         W_H * wrap(S[..., 1:, 2] - target[1:, 2]),
         np.sqrt(LAM) * (u - p).reshape(len(S), -1)]
    return np.concatenate(r, 1)


def compensate(sim, mc, p8, mode):
    """Levenberg-Marquardt over the 24 pose values (mode full: match plan x, y, heading) or the 16 lateral / heading values
    (mode lat: same objective, only y and heading of the submitted poses free, x as planned). Returns u*, cost per iteration, sims."""
    target = L.dense_from_poses(p8)
    u = p8.copy()
    free = np.arange(24) if mode == "full" else np.array([3 * i + j for i in range(8) for j in (1, 2)])
    n = len(free)
    mu = 1e-2
    S = sim_ego(sim, mc, u[None])
    aux = path_proj(S[0], target)[0] if mode == "path" else None
    r = residual(S, target, u[None], p8[None], mode, aux)[0]
    cost = [float(r @ r)]
    base_sim = S[0]
    for _ in range(ITERS):
        steps = np.tile(u, (n, 1, 1))
        e = np.tile(EPS, 8)[free]
        steps.reshape(n, -1)[np.arange(n), free] += e
        Sp = sim_ego(sim, mc, steps)
        J = ((residual(Sp, target, steps, np.broadcast_to(p8, steps.shape), mode, aux) - r) / e[:, None]).T   # (m, n)
        g, H = J.T @ r, J.T @ J
        while True:
            du = -np.linalg.solve(H + mu * np.diag(np.diag(H) + 1e-9), g)
            u2 = u.copy()
            u2.reshape(-1)[free] += du
            S2 = sim_ego(sim, mc, u2[None])
            r2 = residual(S2, target, u2[None], p8[None], mode, aux)[0]
            if r2 @ r2 < r @ r:
                u, r, S, mu = u2, r2, S2, max(mu / 3, 1e-6)
                break
            mu *= 4
            if mu > 1e4:
                break
        cost.append(float(r @ r))
        if mu > 1e4 or cost[-2] - cost[-1] < 1e-4 * cost[-2]:
            break
    return u, cost, base_sim, S[0]


def track_stats(sim_e, target):
    """Tracking error of a simulated ego trajectory against the plan: position error at 0.5..4 s, lateral (plan-normal)."""
    e = plan_err(sim_e, target)
    pos = np.hypot(e[:, 0], e[:, 1])
    s_, d_, hd = path_proj(sim_e, target)
    return dict(pos_mean=float(pos[1:].mean()), pos=[float(pos[i]) for i in (10, 20, 40)], lat_mean=float(np.abs(e[1:, 1]).mean()),
                lat=[float(e[i, 1]) for i in (10, 20, 40)],
                path_d_mean=float(np.abs(d_[1:]).mean()), path_d=[float(d_[i]) for i in (10, 20, 40)], arc=[float(s_[i]) for i in (10, 20, 40)], lon=[float(e[i, 0]) for i in (10, 20, 40)], sim_y=[float(sim_e[i, 1]) for i in (10, 20, 30, 40)],
                plan_y=[float(target[i, 1]) for i in (10, 20, 30, 40)], head_err4=float(wrap(sim_e[40, 2] - target[40, 2])))


# ---------------------------------------------------------------- workers

def _init(data, srcs, modes=MODES):
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    _W.update(sim=PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1)), cp=cache_paths(data),
              P={k: L.poses_by_token(v) for k, v in srcs.items()}, modes=tuple(modes))


def work(token):
    mc = L.load_cache(_W["cp"][token])
    out = {"token": token, "v0": float(mc.ego_state.dynamic_car_state.rear_axle_velocity_2d.x)}
    for (k, P), mode in ((kp, m) for kp in _W["P"].items() for m in _W["modes"]):
        if token not in P:
            continue
        p8 = P[token]
        u, cost, sb, sc = compensate(_W["sim"], mc, p8, mode)
        tgt = L.dense_from_poses(p8)
        out[k, mode] = dict(u=u.astype(np.float32), cost=cost, base=track_stats(sb, tgt), comp=track_stats(sc, tgt),
                      dev=float(np.abs(u - p8)[:, :2].max()), dev_y=[float(u[i, 1] - p8[i, 1]) for i in (1, 3, 5, 7)])
    return out


def cmd_check(a):
    """Own proposal path vs the official one, and simulate equality, on a few tokens of each split."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    srcs = {"native": str(L.D / f"runs/op_lb/{a.data}/preds/gimm-cinque__base.npz")}
    _init(a.data, srcs, a.modes)
    sim = _W["sim"]
    toks = sorted(_W["cp"])[:: max(1, len(_W["cp"]) // a.n)][: a.n]
    mx_arr = mx_sim = 0.0
    t0 = time.time()
    for t in toks:
        mc = L.load_cache(_W["cp"][t])
        p8 = _W["P"]["native"][t]
        arr = get_trajectory_as_array(transform_trajectory(Trajectory(p8), mc.ego_state), sim.proposal_sampling, mc.ego_state.time_point)
        mine = proposals(mc, p8[None])[0]
        mx_arr = max(mx_arr, float(np.abs(mine[1:, :2] - arr[1:, :2]).max()), float(np.abs(wrap(mine[1:, 2] - arr[1:, 2])).max()))
        s1 = sim.simulate_proposals(arr[None], mc.ego_state)[0]
        s2 = sim.simulate_proposals(mine[None], mc.ego_state)[0]
        mx_sim = max(mx_sim, float(np.abs(s1[:, :3] - s2[:, :3]).max()))
    print(f"{len(toks)} tokens: max |pose| own vs official proposal {mx_arr:.2e}, max |sim| {mx_sim:.2e}, {time.time() - t0:.1f}s")
    t0 = time.time()
    for t in toks[:5]:
        mc = L.load_cache(_W["cp"][t])
        for m in _W["modes"]:
            u, cost, sb, sc = compensate(sim, mc, _W["P"]["native"][t], m)
            tg = L.dense_from_poses(_W["P"]["native"][t])
            b, c = track_stats(sb, tg), track_stats(sc, tg)
            print(t, m, "cost", round(cost[0], 2), "->", round(cost[-1], 2), len(cost), "path d 1/2/4 s base", np.round(b["path_d"], 2), "comp",
                  np.round(c["path_d"], 2), "arc base", np.round(b["arc"], 2), "comp", np.round(c["arc"], 2), "lon base", np.round(b["lon"], 2), "comp", np.round(c["lon"], 2))
    print(f"compensate: {(time.time() - t0) / 5:.2f}s per token")


def cmd_make(a):
    srcs = dict(s.split("=", 1) for s in a.src)
    OUT.mkdir(parents=True, exist_ok=True)
    tokens = sorted(set().union(*(np.load(f)["tokens"].tolist() for f in srcs.values())))
    t0 = time.time()
    res = {}
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.data, srcs, a.modes)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tokens, chunksize=8)):
            res[r["token"]] = r
            if i % 1000 == 0:
                print(f"{a.data} {i}/{len(tokens)} {time.time() - t0:.0f}s", flush=True)
    sfx = "" if tuple(a.modes) == ("full",) else "_" + "-".join(a.modes)
    pickle.dump(res, open(OUT / f"diag_{a.data}{sfx}.pkl", "wb"), protocol=4)
    for k, f in srcs.items():
        z = np.load(f)
        p = z["poses"].astype(np.float64)
        for m in a.modes:
            U = np.stack([res[t][k, m]["u"] for t in z["tokens"]]).astype(np.float64)
            for al in a.alphas:
                g = OUT / a.data / f"{k}_{m}_a{al:g}.npz"
                g.parent.mkdir(parents=True, exist_ok=True)
                np.savez(g, tokens=z["tokens"], poses=(p + al * (U - p)).astype(np.float32))
                print("wrote", g)
    print(f"done {len(res)} tokens in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "make"])
    ap.add_argument("--data", default="lb_navhard", choices=list(CACHES))
    ap.add_argument("--src", nargs="+", default=[], help="name=poses.npz")
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=["full", "path"])
    a = ap.parse_args()
    {"check": cmd_check, "make": cmd_make}[a.cmd](a)
