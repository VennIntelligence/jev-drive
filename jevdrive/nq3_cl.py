"""Night queue 3, lane B: our heads in closed loop (todos/2026-09-26-night-queue-3.md, section CL, and the [B] entries
under it, written before any closed-loop number).

  export   (repo .venv, GPU) refit the P5 v1 BA heads on all of the set's training rows and pair rows, with the fold
           code of the P5 tables unchanged (reactivity_mc.fit_fold's prior and `pair` arm, night2_n4's arm B student,
           seed 0), and store them as plain arrays: runs/nq3/b/heads/heads.npz (+ student_b.pt for the check)
  check    offline path of the equivalence check (rule 8): the dumped inputs of a closed-loop route (JPEG bytes, desire,
           the agent's pose track) -> features and head outputs with the offline code, compared bit for bit with what
           the closed-loop servers returned (scripts/nq3_cl_server.py dumps both)

The apply side (`Heads`, `ego_past`, `intent`) is numpy only: it runs inside the head server (envs/openpilot) and in the
offline check alike, so the two paths share it by construction; what the check tests is everything upstream of it
(camera rig, JPEG, model frames, openpilot stepping, Qwen / YOLO features, the ego history from the pose track).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

TICK, STEP_TICKS, CAM_TICKS = 0.05, 5, 4            # p4_carla: 20 Hz ticks, 0.25 s history steps, 5 Hz cameras
REAR_AXLE_X = -1.388633220
INTENT_LOOKAHEAD_M = 15.0                           # p5_pairs.INTENT_LOOKAHEAD_M
HEAD_ARMS = ("ridge_late", "mc", "student_b")


def head_dir() -> Path:
    return Path(os.environ["DATA_DIR"]) / "runs" / "nq3" / "b" / "heads"


# ---------------------------------------------------------------- inputs from the agent's own sensors

def ego_past(ra: np.ndarray, th: np.ndarray, k: int) -> np.ndarray:
    """(16, 6) past state at tick k exactly as p4_carla.route_rows builds it, from a per-tick rear-axle track in a
    right-handed world (ra (n, 2), heading th (n,), ticks 0..n-1 from the route start). Needs tick k + 1 (the
    central-difference velocity), so the closed-loop agent evaluates frame k one tick later."""
    n = len(ra)
    assert k + 1 < n, "ego_past needs tick k + 1"
    v = np.zeros_like(ra)
    v[1:-1] = (ra[2:] - ra[:-2]) / (2 * TICK)
    v[-1] = 0.0                                     # route_rows leaves the last tick at zero; k + 1 <= n - 1 is never read
    dv = v - np.roll(v, STEP_TICKS, axis=0)
    dv[:STEP_TICKS] = 0.0
    pk = k + STEP_TICKS * np.arange(-15, 1)
    pad = pk < 0
    pk = np.maximum(pk, 0)

    def rot(x, a):
        c, s = np.cos(a), np.sin(a)
        return np.stack([c * x[..., 0] + s * x[..., 1], -s * x[..., 0] + c * x[..., 1]], -1)
    p = rot(ra[pk] - ra[k], th[k])
    vv, aa = rot(v[pk], th[k]), rot(dv[pk], th[k])
    vv[pad], aa[pad] = 0.0, 0.0
    vv[-2], aa[-2] = vv[-1], aa[-1]
    return np.concatenate([p, vv, aa], -1).astype(np.float32)


def rh_track(xy_carla: np.ndarray, yaw_carla_rad: np.ndarray):
    """Rear-axle track in CARLA world (y right, yaw clockwise) -> right-handed (ra, th) as route_rows uses."""
    return np.stack([xy_carla[:, 0], -xy_carla[:, 1]], -1), -np.asarray(yaw_carla_rad)


def intent(route_xy: np.ndarray, route_cmd: np.ndarray, route_s: np.ndarray, xy_carla: np.ndarray) -> int:
    """WOD-E2E intent (1 straight, 2 left, 3 right) as p4_carla.route_intent with P5's 15 m lookahead; progress is the
    nearest dense route point (p4_carla._progress, all points)."""
    p = int(np.argmin(((route_xy - xy_carla) ** 2).sum(1)))
    ahead = np.flatnonzero((route_s >= route_s[p]) & (route_s <= route_s[p] + INTENT_LOOKAHEAD_M)
                           & np.isin(route_cmd, (1, 2)))
    return (2 if route_cmd[ahead[0]] == 1 else 3) if len(ahead) else 1


def ego_input(past: np.ndarray, it: int) -> np.ndarray:
    """p5_exam.ego_input for one row: the flat past (96) and the 4-way intent one-hot."""
    return np.concatenate([past.reshape(-1), np.eye(4, dtype=np.float32)[it]]).astype(np.float32)


# ---------------------------------------------------------------- heads (numpy apply)

def _gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


class Heads:
    """prior (`ridge ego` + `ridge_late` on Cinque `temporal`), M-C `pair` Delta on [Qwen L18_last | temporal] and the
    N4 arm-B student Delta on [temporal | image-plane tokens]; every output (20, 2) rear-axle, x forward, y left,
    at 0.25 ... 5.0 s."""

    def __init__(self, path: Path | None = None):
        z = np.load(path or head_dir() / "heads.npz")
        self.p = {k: z[k] for k in z.files}

    @staticmethod
    def _std(x, mu, sd):
        return (x - mu) / sd

    def prior(self, ego: np.ndarray, op: np.ndarray) -> np.ndarray:
        p = self.p
        xe = self._std(ego, p["ego_mu"], p["ego_sd"])
        base = xe @ p["We"][:-1] + p["We"][-1]
        xi = self._std(op, p["op_mu"], p["op_sd"])
        return base + xi @ p["Wp"][:-1] + p["Wp"][-1]

    def mc_delta(self, q: np.ndarray, op: np.ndarray) -> np.ndarray:
        p = self.p
        z = np.concatenate([self._std(q, p["mcq_mu"], p["mcq_sd"]) / math.sqrt(len(q)),
                            self._std(op, p["mco_mu"], p["mco_sd"]) / math.sqrt(len(op))])
        return (z - p["mc_zbar"]) @ p["mc_W"]

    def student_delta(self, op: np.ndarray, tok: np.ndarray) -> np.ndarray:
        p = self.p
        zo = (op - p["sb_op_mu"]) / p["sb_op_sd"] / math.sqrt(len(op))
        zb = np.where(p["sb_mask"], tok, (tok - p["sb_b_mu"]) / p["sb_b_sd"]) / math.sqrt(len(tok))
        h = np.concatenate([zo, zb]).astype(np.float32)
        h = _gelu(h @ p["sb_w0"].T + p["sb_b0"])
        h = _gelu(h @ p["sb_w2"].T + p["sb_b2"])
        return h @ p["sb_w4"].T + p["sb_b4"]

    def predict(self, arm: str, ego, op, q=None, tok=None) -> np.ndarray:
        y = self.prior(ego, op)
        if arm == "mc":
            y = y + self.mc_delta(q, op)
        elif arm == "student_b":
            y = y + self.student_delta(op, tok)
        else:
            assert arm == "ridge_late", arm
        return np.asarray(y, np.float64).reshape(20, 2)


# ---------------------------------------------------------------- export (repo .venv, GPU)

def export(rl):
    """Full-data refit. Every choice the fold code makes internally (lambda by inner CV, mu = n_pair / n_train, the
    student's early stopping on a group holdout of pair rows) is made the same way on all rows."""
    from types import SimpleNamespace

    import pandas as pd
    import torch

    from . import elicit_e5 as E5, elicit_i3 as I, night2_n4 as N4, p5_exam as E, p5_openpilot, p5_pairs as P, planner
    from . import reactivity_mc as MC, waymo_stage_a as sa
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        op = p5_openpilot.load(t, ("cinque",), sub="op_streams_vis")["op-cinque temporal"]
        Q = P.load_features(t, ("L18_last",))["L18_last"]
    n = len(t)
    role = t.role.to_numpy()
    tr = np.flatnonzero(role == "train")
    Fc = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    ego = E.ego_input(t, past)
    Ego, Xop, Qc = (torch.as_tensor(x, device="cuda") for x in (ego, op, Q))
    sp = SimpleNamespace(train=tr, val=tr, seq=t.base_id.to_numpy())
    out = {}

    def stats(X):
        mu, sd = X[tr].double().mean(0), X[tr].double().std(0, correction=0)
        return mu.float().cpu().numpy(), torch.where(sd > 1e-6, sd, 1).float().cpu().numpy()
    # prior: reactivity_mc.fit_fold's first half on all training rows
    Xe = planner.standardize(Ego, tr)
    _, st_e, We = sa.ridge_cv(Xe, Fc, sp, fut)
    base = planner.linear_apply(We, Xe, np.arange(n))[0]
    Xi = planner.standardize(Xop, tr)
    R0 = Fc - base
    _, st_p, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(n, 20, 2).cpu().numpy())
    prior = base + planner.linear_apply(Wp, Xi, np.arange(n))[0]
    out["ego_mu"], out["ego_sd"] = stats(Ego)
    out["op_mu"], out["op_sd"] = stats(Xop)
    out["We"], out["Wp"] = We[0].cpu().numpy(), Wp[0].cpu().numpy()
    rl.event("export_prior", lam_ego=st_e["lam"], lam_op=st_p["lam"])
    # M-C `pair` (dual stream), every pair row
    pos = pd.Series(np.arange(n), index=t.frame_name)
    ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    grp = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    Rp = (Fc[ip] - Fc[im]) - (prior[ip] - prior[im])
    Z = torch.cat([MC._std(Qc, tr), MC._std(Xop, tr)], 1)
    Zc = Z[tr] - Z[tr].mean(0)
    mu = len(ip) / len(tr)
    D = Z[ip] - Z[im]
    score = np.zeros(len(MC.LAMS))
    for a, b in MC._inner_splits(grp):
        Ws = MC._solve_pair(D[a], Rp[a], Zc, mu, MC.LAMS)
        score += [float(((D[b] @ W - Rp[b]) ** 2).sum()) for W in Ws]
    best = int(np.argmin(score))
    W = MC._solve_pair(D, Rp, Zc, mu, [MC.LAMS[best]])[0]
    out["mcq_mu"], out["mcq_sd"] = stats(Qc)
    out["mco_mu"], out["mco_sd"] = stats(Xop)
    out["mc_zbar"], out["mc_W"] = Z[tr].mean(0).cpu().numpy(), W.cpu().numpy()
    mc_pred = prior + (Z - Z[tr].mean(0)) @ W
    rl.event("export_mc", lam=float(MC.LAMS[best]), lam_edge=best in (0, len(MC.LAMS) - 1), n_pair=len(ip),
             n_train=len(tr), mu=mu)
    # N4 arm B student, seed 0, pair-difference loss only
    Bt = torch.as_tensor(np.load(N4.out("tokB.npy")), device="cuda")
    mB = torch.as_tensor(np.arange(Bt.shape[1]) % N4.TOK_D == N4.TOK_D - 1, device="cuda")

    def zz(X, mask=None):
        m_, s_ = X[tr].mean(0), X[tr].std(0, correction=0).clamp_min(1e-6)
        y = (X - m_) / s_
        return (y if mask is None else torch.where(mask, X, y)) / np.sqrt(X.shape[1]), m_, s_
    zo, o_mu, o_sd = zz(Xop)
    zb, b_mu, b_sd = zz(Bt, mB)
    X = torch.cat([zo, zb], 1).float()
    keep = {}
    d = E5.train_student(X, ip, im, Rp, tr, grp, None, 0, rl, "cl6 B", keep)
    sd_ = keep["cl6 B", 0]
    out.update(sb_op_mu=o_mu.cpu().numpy(), sb_op_sd=o_sd.cpu().numpy(), sb_b_mu=b_mu.cpu().numpy(),
               sb_b_sd=b_sd.cpu().numpy(), sb_mask=mB.cpu().numpy())
    for k_ in ("0", "2", "4"):
        out[f"sb_w{k_}"] = sd_[f"{k_}.weight"].numpy()
        out[f"sb_b{k_}"] = sd_[f"{k_}.bias"].numpy()
    pca = np.load(N4.out("pca.npz"))
    out["pca_mu"], out["pca_V"] = pca["mu"], pca["V"]
    hd = head_dir()
    hd.mkdir(parents=True, exist_ok=True)
    np.savez(hd / "heads.npz", **{k: np.asarray(v) for k, v in out.items()})
    # the numpy apply against the torch fits on a sample of rows (float32 GEMM order: ~1e-4 m)
    H = Heads(hd / "heads.npz")
    rows = np.random.default_rng(0).choice(n, 300, replace=False)
    tok = np.load(N4.out("tokB.npy"), mmap_mode="r")
    dif = {"ridge_late": 0.0, "mc": 0.0, "student_b": 0.0}
    for r in rows:
        dif["ridge_late"] = max(dif["ridge_late"], float(np.abs(H.predict("ridge_late", ego[r], op[r]).ravel()
                                                                 - prior[r].cpu().numpy()).max()))
        dif["mc"] = max(dif["mc"], float(np.abs(H.predict("mc", ego[r], op[r], q=Q[r]).ravel()
                                                 - mc_pred[r].cpu().numpy()).max()))
        dif["student_b"] = max(dif["student_b"], float(np.abs(H.predict("student_b", ego[r], op[r], tok=np.asarray(tok[r])).ravel()
                                                               - (prior[r] + d[r]).cpu().numpy()).max()))
    rl.event("export_apply_check", **dif)
    rl.log.info("heads -> %s; numpy apply vs torch fit, max |diff| (m): %s", hd, dif)
    assert max(dif.values()) < 1e-2, dif
    (hd / "meta.json").write_text(json.dumps({"set": I.BA, "n_train": int(len(tr)), "n_pair": int(len(ip)),
                                              "lam_ego": st_e["lam"], "lam_op": st_p["lam"],
                                              "lam_mc": float(MC.LAMS[best]), "apply_check": dif}, indent=1))


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("export",))
    a = ap.parse_args()
    rl = RunLog("nq3_cl", a.step)
    export(rl)
    rl.close()


if __name__ == "__main__":
    main()
