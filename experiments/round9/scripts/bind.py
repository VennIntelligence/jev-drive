#!/usr/bin/env python3
"""ROUND9 BINDING (research/body-schema/binding.md section 0, registered before any read): is the policy's ego state, its own action and its
frozen visual representation "disconnected"? Diagnosis only: no training of the policy. Data: on1_PAID (R5LC-L18-s0's DAgger own states on 718
PAI train scenes, 10 Hz consecutive decisions), the body-schema lane's prep.npz / fwd plans (read only).

  peek    (CPU)  model output slices, token-store parity with on1_PAID@warp/front.npy, forward timing
  prep    (CPU)  per-row executed motion (0.5 s ahead, from row k+5's own pose history), cue labels (stop / lead brake / lead start / road edge),
                 slot-mean token features (n, 8, 512) -> $O/prep.npz, $O/feat.npy
  probe   (CPU)  scene-fold linear probes: cues, ego speed, executed action (forward: tok_k; inverse: tok_k + tok_{k+5} - tok_k) -> $O/probe.json
  dirs    (CPU)  partial-regression token directions (full (8, 32, 512)) for ego speed and each cue, fitted on folds != the eval fold -> $O/dirs_f<k>.npz
  steer   (GPU)  --ckpt TAG: forwards on the eval rows: base, ego-row speed x f, token speed direction x dv, both, cue directions added /
                 removed, cue x speed grid; plan + own lead / pose heads -> $O/steer/<TAG>.npz
  report  (CPU)  every line of section 0 -> $O/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("JEV_REPO", HERE.parents[2]))
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round9/scripts"), str(REPO / "experiments/body1/lib"),
                str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/route_follow/scripts"),
                str(REPO / "experiments/op_probe/scripts"), str(REPO / "experiments/op_parity/scripts"),
                str(REPO / "experiments/defect_fix/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/binding"
BS = DATA / "runs/round9/body_schema"                 # the body-schema lane (read only)
CR = DATA / "runs/op_parity/cache"
POOL = CR / "on1_PAID"
META = DATA / "runs/round9/pai_s_gate/meta.npz"
NF = 4


def fold_of(scene: str) -> int:                       # the body-schema lane's folds (bsl.fold_of)
    return int(hashlib.sha256(f"bsl|{scene}".encode()).hexdigest(), 16) % NF


# ---------------------------------------------------------------- the policy, built as the trainer builds it (bsl.cmd_fwd)
def load_policy(ckpt, threads=0):
    import torch
    import rtc
    from jevdrive.common import n_cpus
    torch.set_num_threads(threads or n_cpus())
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    argv0 = sys.argv
    sys.argv = [str(rt), "rely", *rtc.R6DAG_ARGV.format(tag=ckpt, data=DATA).split()]
    exec(compile(src.replace(tail, "_BIND_ARGS = a"), str(rt), "exec"), g)
    sys.argv = argv0
    ra = g["_BIND_ARGS"]
    ra.rm_from = ckpt
    ns, *_ = g["build"](ra, torch.device("cpu"))
    import pp_train as T
    S = ns.S
    pn = S.tab["names"][ns.on0:].astype(str)
    names = np.load(POOL / "tab.npz")["names"].astype(str)
    key = lambda x: ":".join(x.split(":")[-2:])                                    # noqa: E731  (scene:k)
    pos = {key(x): i for i, x in enumerate(pn) if x.startswith("on:pd:")}
    st = np.array([pos[key(x)] for x in names]) + ns.on0                           # PAID row -> store row
    return ns, ns.model.eval(), S, ns.losses(ns.mode), T, st


class Policy:
    """Forward of a checkpoint on PAID rows with optionally edited tokens / ego rows; returns rear-axle plans (B, 8, 3) and the raw output."""

    def __init__(self, ckpt, dev, threads=0):
        import torch
        self.ns, self.model, self.S, self.LS, self.T, self.st = load_policy(ckpt, threads)
        self.dev = torch.device(dev)
        self.model.to(self.dev)
        self.pi = torch.as_tensor(self.S.pi, device=self.dev)
        self.W = self.LS.W.to(self.dev)
        self.sl = self.model.net.slices

    def tokens(self, rows):
        import torch
        x = self.S.front[self.st[rows]]
        return x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))

    def __call__(self, rows, V=None, E=None):
        """rows: PAID row indices (B,); V (B, 8, 32, 512) tokens (None = stored); E (B, 20+) fed ego rows (None = stored)."""
        import torch
        r = torch.as_tensor(self.st[rows])
        V = (self.tokens(rows) if V is None else V).to(self.dev)
        E = (self.S.ego[r] if E is None else E).to(self.dev)
        nv = None if self.ns.nv is None else self.ns.nv[r].to(self.dev)
        with torch.no_grad():
            o = self.model(V, E, self.S.tc[r].to(self.dev), None, None, nv=nv).float()
            x, y, psi = self.T.rear(o[:, self.pi].view(-1, 33, 15), self.S.cam_x[r].to(self.dev), self.W)
        return torch.stack([x, y, psi], -1).cpu().numpy(), o.cpu().numpy()


def cmd_peek(a):
    import torch
    t = time.time()
    P = Policy(a.ckpt, "cuda" if torch.cuda.is_available() and not a.cpu else "cpu", a.threads)
    print(f"build {time.time() - t:.0f} s; device {P.dev}", flush=True)
    print("slices", {k: (v.start, v.stop) for k, v in P.sl.items()}, flush=True)
    print("ego store", tuple(P.S.ego.shape), "nv", None if P.ns.nv is None else tuple(P.ns.nv.shape), "front", type(P.S.front).__name__, flush=True)
    fr = np.load(CR / "on1_PAID@warp/front.npy", mmap_mode="r")
    tab = np.load(POOL / "tab.npz")
    rows = np.array([0, 1, 5, 1000, 54321, 110980])
    Vs = P.tokens(rows).float().numpy()
    print("token parity max |diff|", float(np.abs(Vs - fr[rows].astype(np.float32)).max()), flush=True)
    print("ego parity max |diff|", float(np.abs(P.S.ego[torch.as_tensor(P.st[rows])][:, :20].cpu().numpy() - tab["ego"][rows]).max()), flush=True)
    rows = np.arange(64)
    t = time.time()
    p, o = P(rows)
    print(f"64 rows {time.time() - t:.2f} s, out {o.shape}", flush=True)
    fw = np.load(BS / "fwd" / f"{a.ckpt}_0.npz") if (BS / "fwd" / f"{a.ckpt}_0.npz").exists() else None
    if fw is not None:
        idx = fw["idx"][:64]
        p2, _ = P(idx)
        print("parity with the body-schema fwd plans: max |dxy|", float(np.abs(p2[..., :2] - fw["poses"][:64, :, :2]).max()), flush=True)
    print("served parity (R5LC states, this ckpt's plan, not expected equal): median |d end|",
          float(np.median(np.linalg.norm(p[:, -1, :2] - tab["served"][rows, -1, :2], axis=-1))), flush=True)
    for k in ("lead", "lead_prob", "pose"):
        if k in P.sl:
            print(k, np.round(o[0, P.sl[k]][:12], 3), flush=True)



# ---------------------------------------------------------------- plan metrics
def seg_speeds(P, origin=None):
    """P (..., 8, 3) poses at 0.5 .. 4 s (origin = zeros or (..., 3)) -> (..., 8) interval speeds."""
    P = np.asarray(P, np.float64)
    o = np.zeros(P.shape[:-2] + (1, 2)) if origin is None else np.asarray(origin, np.float64)[..., None, :2]
    xy = np.concatenate([o, P[..., :2]], -2)
    return np.linalg.norm(np.diff(xy, axis=-2), axis=-1) / 0.5


def pmetrics(P, origin=None):
    s = seg_speeds(P, origin)
    return dict(A4=s.sum(-1) * 0.5, a02=(s[..., 3] - s[..., 0]) / 1.5, v24=s[..., 4:].mean(-1))


def inv_pose(p):
    """p (..., 3) pose of frame A in frame B -> pose of B in A."""
    c, s = np.cos(p[..., 2]), np.sin(p[..., 2])
    x, y = -(c * p[..., 0] + s * p[..., 1]), -(-s * p[..., 0] + c * p[..., 1])
    return np.stack([x, y, -p[..., 2]], -1)


def to_frame(q, e):
    """points q (..., 2) in frame k -> frame k+5 whose pose in k is e (..., 3)."""
    c, s = np.cos(e[..., 2])[..., None], np.sin(e[..., 2])[..., None]
    d = q - e[..., None, :2] if q.ndim > e.ndim else q - e[..., :2]
    return np.stack([c[..., 0] * d[..., 0] + s[..., 0] * d[..., 1], -s[..., 0] * d[..., 0] + c[..., 0] * d[..., 1]], -1)


# ---------------------------------------------------------------- prep (CPU)
def cmd_prep(a):
    from concurrent.futures import ThreadPoolExecutor
    O.mkdir(parents=True, exist_ok=True)
    tab, M, B = np.load(POOL / "tab.npz"), np.load(META), np.load(BS / "prep.npz")
    names = tab["names"].astype(str)
    assert np.array_equal(M["names"].astype(str), names) and np.array_equal(B["names"].astype(str), names)
    n = len(names)
    scene, k = M["scene"].astype(str), M["k"].astype(np.int64)
    fold = B["fold"].astype(np.int8)
    assert all(fold_of(s) == f for s, f in zip(scene[:2000:97], fold[:2000:97]))
    ego, speed = tab["ego"].astype(np.float64), tab["speed"].astype(np.float64)
    idx = {(s, int(j)): i for i, (s, j) in enumerate(zip(scene, k))}
    nx5 = np.array([idx.get((s, int(j) + 5), -1) for s, j in zip(scene, k)])
    pv5 = np.array([idx.get((s, int(j) - 5), -1) for s, j in zip(scene, k)])
    ex = np.full((n, 3), np.nan)
    ok5 = nx5 >= 0
    ex[ok5] = inv_pose(ego[nx5[ok5], 14:17] * [10.0, 10.0, 1.0])                 # pose of k+5 in frame k (executed 0.5 s)
    dv = np.where(ok5, speed[np.maximum(nx5, 0)] - speed, np.nan)
    a_past = np.where(pv5 >= 0, (speed - speed[np.maximum(pv5, 0)]) / 0.5, np.nan)
    w_past = -ego[:, 16] / 0.5                                                   # yaw of the -0.5 s pose in the current frame
    sv = tab["served"].astype(np.float64)
    sm = pmetrics(sv)
    s1 = seg_speeds(sv)
    dv_plan = 0.5 * (s1[:, 0] + s1[:, 1]) - speed                                # plan-implied v(0.5 s) - v0
    # ---- lead (replay boxes, row frame at t0)
    Z = np.load(POOL / "agent.npz")
    box, val = Z["box"], Z["valid"]
    x0, y0 = box[:, 0, :, 0], box[:, 0, :, 1]
    cand = val[:, 0] & (x0 > 0) & (x0 <= 60) & (np.abs(y0) < 1.8)
    xm = np.where(cand, x0, np.inf)
    li = xm.argmin(1)
    has_lead = np.isfinite(xm.min(1))
    r = np.arange(n)
    L = box[r, :, li]                                                            # (n, 9, 5)
    Lv = val[r, :, li]
    gap = np.where(has_lead, L[:, 0, 0] - 4.05 - L[:, 0, 3] / 2, np.inf)
    vL0 = np.where(Lv[:, 0] & Lv[:, 1], np.linalg.norm(L[:, 1, :2] - L[:, 0, :2], axis=1) / 0.5, np.nan)
    vL1 = np.where(Lv[:, 1] & Lv[:, 2], np.linalg.norm(L[:, 2, :2] - L[:, 1, :2], axis=1) / 0.5, np.nan)
    aL = (vL1 - vL0) / 0.5
    disp2 = np.where(Lv[:, 0] & Lv[:, 4], np.linalg.norm(L[:, 4, :2] - L[:, 0, :2], axis=1), np.nan)
    v0 = speed
    LB = has_lead & (gap <= 40) & (v0 >= 2) & (aL <= -1.5)
    LC = has_lead & (gap <= 40) & (v0 >= 2) & (np.abs(aL) < 0.5)
    LS = has_lead & (gap <= 25) & (v0 < 1) & (vL0 < 0.5) & (disp2 > 1.5)
    LH = has_lead & (gap <= 25) & (v0 < 1) & (vL0 < 0.5) & (disp2 < 0.3)
    lead40 = has_lead & (gap <= 40)
    stop = B["stop"].astype(np.float64)
    ST = (stop[:, 3] > 0.5) & ~lead40
    d_stop = np.where(ST, np.hypot(stop[:, 0], stop[:, 1]), np.nan)
    # ---- the human (log0 / log_fut in the row frame)
    l0, lf, lok = M["log0"].astype(np.float64), M["log_fut"].astype(np.float64), M["log_fut_ok"].all(1)
    hs = seg_speeds(lf, l0)
    A4h, vh, a02h = hs.sum(1) * 0.5, hs[:, 0], (hs[:, 3] - hs[:, 0]) / 1.5
    near = lok & (np.hypot(l0[:, 0], l0[:, 1]) < 2.0) & (np.abs(l0[:, 2]) < 0.1)
    dstop_h = np.where(ST, np.hypot(stop[:, 0] - l0[:, 0], stop[:, 1] - l0[:, 1]), np.nan)
    hturn = np.abs(lf[:, 7, 2] - l0[:, 2])
    has_lead60 = has_lead & (gap <= 60)
    OP = (v0 >= 2) & ~has_lead60 & lok & (hs.min(1) > 1.0) & (hturn < 0.1) & ~ST
    cls = np.full(n, "", "<U2")
    for nm, m in (("OP", OP), ("ST", ST), ("LH", LH), ("LC", LC), ("LS", LS), ("LB", LB)):   # later wins: LB > LS > LC > LH > ST > OP
        cls[m] = nm
    # ---- slot-mean token features
    fr = np.load(CR / "on1_PAID@warp/front.npy", mmap_mode="r")
    feat = np.lib.format.open_memmap(O / "feat.npy", "w+", np.float16, (n, 8, 512))
    t = time.time()

    def job(i):
        feat[i:i + 1024] = fr[i:i + 1024].astype(np.float32).mean(2).astype(np.float16)
    with ThreadPoolExecutor(a.workers) as tp:
        list(tp.map(job, range(0, n, 1024)))
    feat.flush()
    print(f"features {time.time() - t:.0f} s", flush=True)
    np.savez(O / "prep.npz", names=names, scene=scene, k=k, fold=fold, nx5=nx5, pv5=pv5, ex=ex, dv=dv, a_past=a_past, w_past=w_past,
             v0=v0, ego6=ego[:, 6], dv_plan=dv_plan, A4s=sm["A4"], a02s=sm["a02"], has_lead=has_lead, gap=gap, vL0=vL0, aL=aL, disp2=disp2,
             cls=cls, ST=ST, LB=LB, LC=LC, LS=LS, LH=LH, OP=OP, lead40=lead40, d_stop=d_stop, near=near, A4h=A4h, vh=vh, a02h=a02h,
             dstop_h=dstop_h, served=sv.astype(np.float32))
    fin = np.isfinite(a_past)
    info = dict(rows=n, scenes=len(set(scene)), nx5=int(ok5.sum()), classes={c: int((cls == c).sum()) for c in ("OP", "ST", "LH", "LC", "LS", "LB")},
                classes_f0={c: int(((cls == c) & (fold == 0)).sum()) for c in ("OP", "ST", "LH", "LC", "LS", "LB")},
                near=float(near.mean()), lead=float(has_lead.mean()),
                exec_vs_plan_05_med_m=float(np.nanmedian(np.hypot(ex[:, 0] - sv[:, 0, 0], ex[:, 1] - sv[:, 0, 1]))),
                exec_dx_vs_speed_med=float(np.nanmedian(ex[:, 0] / np.maximum(0.5 * (v0 + np.where(ok5, speed[np.maximum(nx5, 0)], np.nan)), 0.1) / 0.5)),
                corr_ego6_apast=float(np.corrcoef(ego[fin, 6], a_past[fin])[0, 1]))
    print(json.dumps(info), flush=True)
    (O / "prep.json").write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------- ridge probes (CPU)
class Ridge:
    """Centred ridge via the eigendecomposition of the train Gram: w(lam) for a grid at the cost of one eigh."""

    def __init__(self, X, Y):
        X = np.asarray(X, np.float64)
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mx) / self.sx
        Y = np.asarray(Y, np.float64).reshape(len(X), -1)
        self.my = Y.mean(0)
        self.ev, self.Q = np.linalg.eigh(Z.T @ Z)
        self.b = self.Q.T @ (Z.T @ (Y - self.my))
        self.n = len(X)

    def w(self, lam):
        return self.Q @ (self.b / (self.ev[:, None] + lam * self.n))

    def predict(self, X, lam):
        return ((np.asarray(X, np.float64) - self.mx) / self.sx) @ self.w(lam) + self.my


LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def r2(y, p):
    y, p = np.asarray(y).reshape(len(y), -1), np.asarray(p).reshape(len(p), -1)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def auc(y, s):
    y = np.asarray(y, bool)
    o = np.argsort(s)
    rk = np.empty(len(s))
    rk[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def cv(X, Y, fold, rows, metric, sel=0):
    """lam chosen on fold `sel` per target column, the metric averaged over the other folds (train = every other fold).
    Y (n,) or (n, m); metric(y (k, c), p (k, c)) -> float. Returns (mean, per fold, lam) or a list of them when cols=True via cvm."""
    return cvm(X, np.asarray(Y).reshape(len(Y), -1), fold, rows, metric, sel, joint=True)[0]


def cvm(X, Y, fold, rows, metric, sel=0, joint=False):
    """joint: one metric over all columns (e.g. pooled R2 of a multi-output target); else one result per column."""
    res = {}
    for f in range(NF):
        tr, te = rows[fold[rows] != f], rows[fold[rows] == f]
        m = Ridge(X[tr], Y[tr])
        res[f] = {}
        for lam in LAMS:
            p = m.predict(X[te], lam)
            res[f][lam] = [metric(Y[te], p)] if joint else [metric(Y[te][:, j:j + 1], p[:, j:j + 1]) for j in range(Y.shape[1])]
    out = []
    for j in range(len(res[0][LAMS[0]])):
        lam = max(LAMS, key=lambda l_: res[sel][l_][j])
        per = [res[f][lam][j] for f in range(NF) if f != sel]
        out.append((round(float(np.mean(per)), 4), [round(x, 4) for x in per], lam))
    return out


def cmd_probe(a):
    P = dict(np.load(O / "prep.npz"))
    F = np.load(O / "feat.npy", mmap_mode="r")
    fold, n = P["fold"], len(P["fold"])
    rng = np.random.default_rng(0)
    out = {}
    sub = lambda m, cap: np.sort(rng.choice(np.flatnonzero(m), min(cap, int(m.sum())), replace=False))   # noqa: E731
    f2 = lambda r: F[r].reshape(len(r), -1).astype(np.float32)                                         # noqa: E731
    # A1 speed
    r = sub(np.ones(n, bool), a.cap)
    X = np.zeros((n, 4096), np.float32)
    X[r] = f2(r)
    out["A1_speed_r2"] = cv(X, P["v0"], fold, r, r2)
    out["A1_pose_from_ego_row"] = "n/a"
    # A2 executed dv / dpsi: forward (f_k), inverse (f_k, f_{k+5} - f_k), ego row baseline, served-plan implied
    ok = (P["nx5"] >= 0) & np.isfinite(P["dv"])
    r = sub(ok, a.cap)
    nx = P["nx5"]
    Xi = np.zeros((n, 8192), np.float32)
    Xi[r, :4096] = f2(r)
    Xi[r, 4096:] = f2(nx[r]) - Xi[r, :4096]
    ego = np.load(POOL / "tab.npz")["ego"].astype(np.float32)
    tg = ("dv", "dpsi", "dx")
    Y = np.nan_to_num(np.c_[P["dv"], P["ex"][:, 2], P["ex"][:, 0]])
    for nm, X_ in (("forward", Xi[:, :4096]), ("idm", Xi), ("egorow", ego)):
        for t_, v_ in zip(tg, cvm(X_, Y, fold, r, r2)):
            out[f"A2_{t_}_{nm}_r2"] = v_
    out["A3_plan_dv_r2"] = r2(P["dv"][r], P["dv_plan"][r])
    out["A3_exec_vs_plan05_med_m"] = float(np.median(np.hypot(P["ex"][r, 0] - P["served"][r, 0, 0], P["ex"][r, 1] - P["served"][r, 0, 1])))
    # cue probes (G2 for ST / LS / LB; weights from folds != 0 kept for the steering check)
    W = {}
    for c, pos, neg in (("ST", "ST", "OP"), ("LB", "LB", "LC"), ("LS", "LS", "LH")):
        m = P[pos] | P[neg]
        rr = np.flatnonzero(m)
        Xc = np.zeros((n, 4096), np.float32)
        Xc[rr] = f2(rr)
        out[f"probe_{c}_auc"] = cv(Xc, P[pos].astype(np.float64), fold, rr, lambda y, s: auc(y[:, 0] > 0.5, s[:, 0]))
        lam = out[f"probe_{c}_auc"][2]
        tr = rr[fold[rr] != 0]
        mdl = Ridge(Xc[tr], P[pos][tr].astype(np.float64))
        W[c] = dict(w=(mdl.w(lam)[:, 0] / mdl.sx).astype(np.float32), mx=mdl.mx.astype(np.float32))
        del Xc
    np.savez(O / "probe_w.npz", **{f"{c}_{k_}": v for c, d in W.items() for k_, v in d.items()})
    # C1: forward model of the token change, PCA-64 of df; predictors f_k PCA-256, + executed action, + served plan (first 2 points)
    df = Xi[:, 4096:]
    tr0 = r[fold[r] != 0]
    mu = df[tr0].mean(0)
    _, sv_, Vt = np.linalg.svd(df[tr0[:20000]] - mu, full_matrices=False)
    Ydf = (df - mu) @ Vt[:64].T
    muf = Xi[tr0, :4096].mean(0)
    _, _, Vf = np.linalg.svd(Xi[tr0[:20000], :4096] - muf, full_matrices=False)
    Xf = (Xi[:, :4096] - muf) @ Vf[:256].T
    act = np.nan_to_num(np.c_[P["ex"], P["dv"], P["v0"]])
    pl = np.c_[P["served"][:, :2].reshape(n, -1), P["v0"]]
    wr2 = lambda y, p: r2(y, p)                                                     # noqa: E731  (pooled over components = variance-weighted)
    out["C1_df_from_f"] = cv(Xf, Ydf, fold, r, wr2)
    out["C1_df_from_f_act"] = cv(np.c_[Xf, act], Ydf, fold, r, wr2)
    out["C1_df_from_f_plan"] = cv(np.c_[Xf, pl], Ydf, fold, r, wr2)
    out["C1_df_from_act"] = cv(act, Ydf, fold, r, wr2)
    out["C1_df_var_top64"] = float((sv_[:64] ** 2).sum() / (sv_ ** 2).sum())
    out["rows"] = dict(A=int(len(r)))
    print(json.dumps(out, indent=1), flush=True)
    (O / "probe.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- token directions (CPU)
DCOLS = ("one", "v", "v2", "a_past", "w_past", "lead40", "LB", "LS", "ST")


def design(P, rows):
    v = P["v0"][rows]
    return np.c_[np.ones(len(rows)), v, v * v, np.nan_to_num(P["a_past"][rows]), P["w_past"][rows], P["lead40"][rows], P["LB"][rows], P["LS"][rows],
                 P["ST"][rows]].astype(np.float64)


def cmd_dirs(a):
    P = dict(np.load(O / "prep.npz"))
    fr = np.load(CR / "on1_PAID@warp/front.npy", mmap_mode="r")
    ok = (P["fold"] != a.fold) & np.isfinite(P["a_past"])
    rng = np.random.default_rng(1)
    pool = np.flatnonzero(ok)
    cue = np.flatnonzero(ok & (P["LB"] | P["LS"] | P["ST"] | P["LC"] | P["LH"]))   # every cue / host row, then a uniform fill
    rows = np.unique(np.r_[cue, rng.choice(pool, min(a.cap, len(pool)), replace=False)])
    X = design(P, rows)
    XtX = X.T @ X
    XtT = np.zeros((X.shape[1], 8 * 32 * 512))
    t = time.time()
    for i in range(0, len(rows), 2048):
        rr = rows[i:i + 2048]
        XtT += X[i:i + 2048].T @ fr[rr].reshape(len(rr), -1).astype(np.float64)
    Bc = np.linalg.solve(XtX, XtT)
    sd = np.zeros(8 * 32 * 512)
    for i in range(0, min(len(rows), 8192), 2048):                                   # residual sd on a subset (scale reference)
        rr = rows[i:i + 2048]
        sd += ((fr[rr].reshape(len(rr), -1).astype(np.float64) - X[i:i + 2048] @ Bc) ** 2).sum(0)
    sd = np.sqrt(sd / min(len(rows), 8192))
    np.savez(O / f"dirs_f{a.fold}.npz", B=Bc.reshape(-1, 8, 32, 512).astype(np.float32), cols=np.array(DCOLS), sd=sd.reshape(8, 32, 512).astype(np.float32),
             n=len(rows), counts={c: int(X[:, j].sum()) for j, c in enumerate(DCOLS) if c in ("lead40", "LB", "LS", "ST")})
    print(json.dumps(dict(rows=len(rows), sec=round(time.time() - t), norms={c: float(np.linalg.norm(Bc[j])) for j, c in enumerate(DCOLS)},
                          rel_sd={c: float(np.linalg.norm(Bc[j]) / np.linalg.norm(sd)) for j, c in enumerate(DCOLS)})), flush=True)


# ---------------------------------------------------------------- steering forwards (CPU or GPU)
def scale_row(ego, f):
    """inertia.scale_row: the fed row's speed history x f (vx vy, ax, the 3 past poses re-spaced along the own polyline)."""
    import warp_loop as WL
    if f == 1.0:
        return ego.copy()
    Pp = ego[8:20].reshape(4, 3).astype(np.float64) * [10.0, 10.0, 1.0]
    arc = np.cumsum(np.linalg.norm(np.diff(Pp[::-1, :2], axis=0), axis=1))
    Q = WL.respace(Pp, f * arc)
    r = ego.copy()
    r[4:6] = ego[4:6] * f
    r[6] = ego[6] * f
    r[8:17] = (Q[:3] / [10.0, 10.0, 1.0]).reshape(9)
    return r


CUE_ARMS = {"OP": ("ST",), "ST": ("ST",), "LC": ("LB",), "LB": ("LB",), "LH": ("LS",), "LS": ("LS",)}


def steer_sets(P, per, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for c in CUE_ARMS:
        m = np.flatnonzero((P["cls"] == c) & (P["fold"] == 0) & np.isfinite(P["a_past"]))
        out[c] = np.sort(rng.choice(m, min(per, len(m)), replace=False))
    m = np.flatnonzero(((P["cls"] == "ST") | (P["cls"] == "OP")) & (P["fold"] == 0) & (P["nx5"] >= 0))
    out["PAIR"] = np.sort(rng.choice(m, min(per, len(m)), replace=False))
    return out


def cmd_steer(a):
    import torch
    P = dict(np.load(O / "prep.npz"))
    D = np.load(O / "dirs_f0.npz")
    cols = list(D["cols"])
    Bd = {c: torch.as_tensor(D["B"][cols.index(c)]) for c in cols}
    sets = steer_sets(P, a.per)
    pol = Policy(a.ckpt, a.dev, a.threads)
    sl = pol.sl
    rec = {}
    t0 = time.time()
    for sname, rows in sets.items():
        if sname == "PAIR":
            arms = [("base", 0, 1.0, None, 0)]
            rows = np.r_[rows, P["nx5"][rows]]
        else:
            arms = [("base", 0, 1.0, None, 0), ("R+", 0, 1.2, None, 0), ("R-", 0, 0.8, None, 0), ("T+", 0.2, 1.0, None, 0), ("T-", -0.2, 1.0, None, 0),
                    ("B+", 0.2, 1.2, None, 0), ("B-", -0.2, 0.8, None, 0)]
            cue = CUE_ARMS[sname][0]
            host = sname in ("OP", "LC", "LH")
            if host:
                arms += [("C+", 0, 1.0, cue, 1.0), ("C2+", 0, 1.0, cue, 2.0)]
                if sname == "OP":
                    arms += [("C+B+", 0.2, 1.2, cue, 1.0), ("C+B-", -0.2, 0.8, cue, 1.0)]
            else:
                arms += [("C-", 0, 1.0, cue, -1.0)]
        E0 = pol.S.ego[torch.as_tensor(pol.st[rows])].clone()
        v = torch.as_tensor(P["v0"][rows], dtype=torch.float32)
        for name, ts, rf, cue, cs in arms:
            plans, heads = [], []
            for j in range(0, len(rows), a.bs):
                rr = rows[j:j + a.bs]
                V = pol.tokens(rr)
                if ts or cue:
                    Vf = V.float()
                    if ts:
                        vv = v[j:j + a.bs]
                        dvec = (Bd["v"][None] + 2 * Bd["v2"][None] * vv[:, None, None, None]) * (ts * vv)[:, None, None, None]
                        Vf = Vf + dvec
                    if cue:
                        Vf = Vf + cs * Bd[cue][None]
                    V = Vf.to(V.dtype)
                E = E0[j:j + a.bs].clone()
                if rf != 1.0:
                    e20 = E[:, :20].numpy()
                    E[:, :20] = torch.as_tensor(np.stack([scale_row(e, rf) for e in e20]), dtype=E.dtype)
                p, o = pol(rr, V, E)
                plans.append(p)
                ld = o[:, sl["lead"]][:, :72].reshape(-1, 3, 6, 4)[:, 0, 0]          # lead 0 at t = 0: x, y, v, a
                heads.append(np.c_[o[:, sl["pose"]][:, :3], ld, o[:, sl["lead_prob"]][:, :1]])
            rec[f"{sname}/{name}"] = dict(plan=np.concatenate(plans).astype(np.float32), heads=np.concatenate(heads).astype(np.float32))
        rec[f"{sname}/rows"] = dict(rows=rows)
        print(f"{sname}: {len(rows)} rows x {len(arms)} arms, {time.time() - t0:.0f} s", flush=True)
    (O / "steer").mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, d in rec.items():
        for k2, v2 in d.items():
            flat[f"{key}/{k2}"] = v2
    np.savez(O / "steer" / f"{a.ckpt}.npz", **flat)
    print(json.dumps(dict(ckpt=a.ckpt, sec=round(time.time() - t0))), flush=True)


HCOLS = ("one", "v", "v2", "lead40", "LB", "LS", "ST")


def hdesign(P, rows, v):
    return np.c_[np.ones(len(rows)), v, v * v, P["lead40"][rows], P["LB"][rows], P["LS"][rows], P["ST"][rows]].astype(np.float64)


def ols_boot(X, y, scene, B=500, seed=0):
    """OLS coefficients and a scene-cluster bootstrap (per-scene Gram sums) -> (beta, lo, hi)."""
    us, si = np.unique(scene, return_inverse=True)
    G = np.zeros((len(us), X.shape[1], X.shape[1]))
    b = np.zeros((len(us), X.shape[1]))
    np.add.at(G, si, X[:, :, None] * X[:, None, :])
    np.add.at(b, si, X * y[:, None])
    beta = np.linalg.lstsq(G.sum(0), b.sum(0), rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        c = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)).astype(np.float64)
        bs.append(np.linalg.lstsq(np.tensordot(c, G, 1), c @ b, rcond=None)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    return beta, lo, hi


def boot_ratio(num, den, scene, B=2000, seed=0):
    """sum(num) / sum(den) with a scene-cluster bootstrap -> (est, lo, hi)."""
    us, si = np.unique(scene, return_inverse=True)
    sn, sd = np.bincount(si, num, len(us)), np.bincount(si, den, len(us))
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)) for _ in range(B)]).astype(np.float64)
    r = (w @ sn) / (w @ sd)
    return [round(float(sn.sum() / sd.sum()), 4), round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def cmd_report(a):
    P = dict(np.load(O / "prep.npz"))
    scene, n = P["scene"], len(P["scene"])
    R = dict(probe=json.loads((O / "probe.json").read_text()) if (O / "probe.json").exists() else None)
    # ---- the human reference: A4_h, a02_h on [v_h, v_h^2, lead40, LB, LS, ST] (rows where the human is near the ego)
    hr = np.flatnonzero(P["near"] & np.isfinite(P["A4h"]))
    Xh = hdesign(P, hr, P["vh"][hr])
    H = {}
    for tgt in ("A4h", "a02h"):
        beta, lo, hi = ols_boot(Xh, P[tgt][hr], scene[hr])
        H[tgt] = {c: [round(float(beta[j]), 4), round(float(lo[j]), 4), round(float(hi[j]), 4)] for j, c in enumerate(HCOLS)}
    R["human"] = dict(rows=int(len(hr)), **H)
    bh = {c: H["A4h"][c][0] for c in HCOLS}
    bha = {c: H["a02h"][c][0] for c in HCOLS}
    # ---- B4 observational: plans on the same design with the ego's v0
    obs = {}
    plans = {"served(R5LC)": (np.arange(n), P["served"])}
    for c in ("NTV0-L18-s0", "PCL1-F-s0"):
        fs = sorted((BS / "fwd").glob(f"{c}_*.npz"))
        if fs:
            z = [np.load(f) for f in fs]
            plans[c] = (np.concatenate([x["idx"] for x in z]), np.concatenate([x["poses"] for x in z]))
    for nm, (rows, pl) in plans.items():
        m = pmetrics(pl)
        X = hdesign(P, rows, P["v0"][rows])
        d = {}
        for tgt, key, ref in (("A4", "A4", bh), ("a02", "a02", bha)):
            beta, lo, hi = ols_boot(X, m[key], scene[rows])
            d[tgt] = {c: [round(float(beta[j]), 4), round(float(lo[j]), 4), round(float(hi[j]), 4)] for j, c in enumerate(HCOLS)}
            d[f"{tgt}_gain_vs_human"] = {c: round(float(beta[j] / ref[c]), 3) for j, c in enumerate(HCOLS) if c in ("LB", "LS", "ST", "lead40")}
        d["rows"] = int(len(rows))
        # B3: within ST rows A4 ~ [1, v0, d_stop]; within OP rows A4 ~ [1, v0]
        st, op = rows[P["ST"][rows]], rows[P["OP"][rows]]
        mi = {r_: i for i, r_ in enumerate(rows)}
        a4 = m["A4"]
        if len(st) > 50 and len(op) > 50:
            bst = ols_boot(np.c_[np.ones(len(st)), P["v0"][st], P["d_stop"][st]], a4[[mi[x] for x in st]], scene[st])
            bop = ols_boot(np.c_[np.ones(len(op)), P["v0"][op]], a4[[mi[x] for x in op]], scene[op])
            d["B3"] = dict(b_ST_v=[round(float(x[1]), 3) for x in bst], b_ST_dstop=[round(float(x[2]), 3) for x in bst],
                           b_OP_v=[round(float(x[1]), 3) for x in bop], kappa=round(float(bst[0][1] / bop[0][1]), 3), n_ST=int(len(st)))
        # exploratory (not registered): lead binding, A4 ~ [1, v0, gap, vL0] on lead rows (LC | LB)
        lr = rows[(P["LC"] | P["LB"])[rows] & np.isfinite(P["vL0"][rows])]
        if len(lr) > 50:
            bl = ols_boot(np.c_[np.ones(len(lr)), P["v0"][lr], P["gap"][lr], P["vL0"][lr]], a4[[mi[x] for x in lr]], scene[lr])
            d["B3_lead_expl"] = dict(b_v=[round(float(x[1]), 3) for x in bl], b_gap=[round(float(x[2]), 3) for x in bl],
                                     b_vL=[round(float(x[3]), 3) for x in bl], n=int(len(lr)))
        obs[nm] = d
    st = np.flatnonzero(P["ST"] & P["near"])
    op = np.flatnonzero(P["OP"] & P["near"])
    bst = ols_boot(np.c_[np.ones(len(st)), P["vh"][st], P["dstop_h"][st]], P["A4h"][st], scene[st])
    bop = ols_boot(np.c_[np.ones(len(op)), P["vh"][op]], P["A4h"][op], scene[op])
    R["human"]["B3"] = dict(b_ST_v=[round(float(x[1]), 3) for x in bst], b_ST_dstop=[round(float(x[2]), 3) for x in bst],
                            b_OP_v=[round(float(x[1]), 3) for x in bop], kappa=round(float(bst[0][1] / bop[0][1]), 3), n_ST=int(len(st)))
    lr = np.flatnonzero((P["LC"] | P["LB"]) & P["near"] & np.isfinite(P["vL0"]))
    gh = P["gap"][lr] - P["A4h"][lr] * 0                                               # gap from the ego; the human is within 2 m of it
    bl = ols_boot(np.c_[np.ones(len(lr)), P["vh"][lr], gh, P["vL0"][lr]], P["A4h"][lr], scene[lr])
    R["human"]["B3_lead_expl"] = dict(b_v=[round(float(x[1]), 3) for x in bl], b_gap=[round(float(x[2]), 3) for x in bl],
                                      b_vL=[round(float(x[3]), 3) for x in bl], n=int(len(lr)))
    R["B4_obs"] = obs
    # ---- C2 on the served plans: plan_{k+5} 3.5 s point vs plan_k 4 s point in frame k+5, along x
    ok = P["nx5"] >= 0
    rr = np.flatnonzero(ok)
    q = to_frame(P["served"][rr, 7, :2].astype(np.float64), P["ex"][rr])
    inc = P["served"][P["nx5"][rr], 6, 0] - q[:, 0]
    R["C2_served"] = {c: dict(n=int(m.sum()), med=round(float(np.median(inc[m])), 3), pos=round(float((inc[m] > 0).mean()), 3),
                              med_over_advance=round(float(np.median(inc[m] / np.maximum(P["ex"][rr][m, 0], 0.1))), 3))
                      for c, m in (("ST", P["ST"][rr]), ("OP", P["OP"][rr]), ("LB", P["LB"][rr]), ("all", np.ones(len(rr), bool)))}
    # ---- steering
    F = np.load(O / "feat.npy", mmap_mode="r")
    D = np.load(O / "dirs_f0.npz")
    cols = list(D["cols"])
    W = np.load(O / "probe_w.npz") if (O / "probe_w.npz").exists() else None
    R["dirs"] = dict(n=int(D["n"]), norm={c: round(float(np.linalg.norm(D["B"][j])), 3) for j, c in enumerate(cols)},
                     norm_rel_resid_sd={c: round(float(np.linalg.norm(D["B"][j]) / np.linalg.norm(D["sd"])), 4) for j, c in enumerate(cols)})
    if W is not None:                                                            # G2 for ST / LS / LB: probe shift of the pushed direction
        g2 = {}
        for c, pos, neg in (("ST", "ST", "OP"), ("LB", "LB", "LC"), ("LS", "LS", "LH")):
            w, mx = W[f"{c}_w"].astype(np.float64), W[f"{c}_mx"].astype(np.float64)
            shift = float(D["B"][cols.index(c)].mean(1).reshape(-1) @ w)
            f0 = P["fold"] == 0
            sp = F[np.flatnonzero(P[pos] & f0)].reshape(-1, 4096).astype(np.float64) @ w
            sn = F[np.flatnonzero(P[neg] & f0)].reshape(-1, 4096).astype(np.float64) @ w
            g2[c] = dict(shift=round(shift, 4), natural=round(float(sp.mean() - sn.mean()), 4), ratio=round(shift / float(sp.mean() - sn.mean()), 3))
        R["G2_probe"] = g2
    R["steer"] = {}
    for f in sorted((O / "steer").glob("*.npz")):
        z = np.load(f)
        S = {}
        get = lambda s_, arm, k_="plan": z[f"{s_}/{arm}/{k_}"]                     # noqa: E731
        for sname in CUE_ARMS:
            if f"{sname}/rows/rows" not in z:
                continue
            rows = z[f"{sname}/rows/rows"]
            sc = scene[rows]
            v0 = P["v0"][rows]
            base = pmetrics(get(sname, "base"))
            d = dict(n=int(len(rows)), A4_base=round(float(base["A4"].mean()), 2))
            A = {arm: pmetrics(get(sname, arm)) for arm in ("R+", "R-", "T+", "T-", "B+", "B-")}
            for x in ("R", "T", "B"):
                dA = (A[f"{x}+"]["A4"] - A[f"{x}-"]["A4"]) / 2
                d[f"dA4_{x}"] = boot_ratio(dA, np.ones(len(rows)), sc)
                d[f"G_{x}"] = boot_ratio(dA / 0.2, base["A4"], sc)
            req = 0.2 * v0 * (bh["v"] + 2 * bh["v2"] * v0)
            d["rho_v"] = boot_ratio((A["B+"]["A4"] - A["B-"]["A4"]) / 2, req, sc)
            d["R_share"] = boot_ratio((A["R+"]["A4"] - A["R-"]["A4"]) / 2, (A["B+"]["A4"] - A["B-"]["A4"]) / 2, sc)
            hb, ht = get(sname, "base", "heads"), get(sname, "T+", "heads")
            d["pose_vx_shift_per_req"] = boot_ratio(ht[:, 0] - hb[:, 0], 0.2 * v0, sc)
            d["pose_vx_base_over_v0"] = boot_ratio(hb[:, 0], v0, sc)
            cue = CUE_ARMS[sname][0]
            for arm in ("C+", "C2+", "C-"):
                if f"{sname}/{arm}/plan" not in z:
                    continue
                m = pmetrics(get(sname, arm))
                sgn = -1.0 if arm == "C-" else (2.0 if arm == "C2+" else 1.0)
                d[f"rho_{cue}_{arm}"] = boot_ratio(m["A4"] - base["A4"], np.full(len(rows), sgn * bh[cue]), sc)
                d[f"dA4_{arm}"] = boot_ratio(m["A4"] - base["A4"], np.ones(len(rows)), sc)
                d[f"rho_a02_{cue}_{arm}"] = boot_ratio(m["a02"] - base["a02"], np.full(len(rows), sgn * bha[cue]), sc)
                hc = get(sname, arm, "heads")
                d[f"lead_head_{arm}"] = dict(da=round(float((hc[:, 6] - hb[:, 6]).mean()), 3), dv=round(float((hc[:, 5] - hb[:, 5]).mean()), 3),
                                             dx=round(float((hc[:, 3] - hb[:, 3]).mean()), 3), dprob=round(float((hc[:, 7] - hb[:, 7]).mean()), 3))
            if sname == "OP" and "OP/C+B+/plan" in z:
                cb = (pmetrics(get(sname, "C+B+"))["A4"] - pmetrics(get(sname, "C+B-"))["A4"]) / 2
                d["interaction_speed_slope_with_ST_over_without"] = boot_ratio(cb, (A["B+"]["A4"] - A["B-"]["A4"]) / 2, sc)
            d["head_base_mean"] = dict(lead_a=round(float(hb[:, 6].mean()), 3), lead_v=round(float(hb[:, 5].mean()), 3), lead_x=round(float(hb[:, 3].mean()), 2),
                                       lead_prob_logit=round(float(hb[:, 7].mean()), 3))
            S[sname] = d
        # G2 for LB through the model's own lead head: shift under C+ on LC vs the natural LB - LC difference of the head
        if "LC" in S and "LB" in S:
            nat = S["LB"]["head_base_mean"]["lead_a"] - S["LC"]["head_base_mean"]["lead_a"]
            S["G2_lead_head"] = dict(natural_da=round(nat, 3), pushed_da=S["LC"]["lead_head_C+"]["da"],
                                     ratio=round(S["LC"]["lead_head_C+"]["da"] / nat, 3) if nat else None)
        if "LH" in S and "LS" in S:
            nat = S["LS"]["head_base_mean"]["lead_v"] - S["LH"]["head_base_mean"]["lead_v"]
            S["G2_lead_head_LS_v"] = dict(natural_dv=round(nat, 3), pushed_dv=S["LH"]["lead_head_C+"]["dv"],
                                          ratio=round(S["LH"]["lead_head_C+"]["dv"] / nat, 3) if nat else None)
        if "ST" in S and "OP" in S:
            S["B3_steer_GT_ST_over_OP"] = round(S["ST"]["G_T"][0] / S["OP"]["G_T"][0], 3)
            S["B3_steer_GB_ST_over_OP"] = round(S["ST"]["G_B"][0] / S["OP"]["G_B"][0], 3)
        if "PAIR/base/plan" in z:                                                   # C2 on this checkpoint's own plans (R5LC's states)
            rows = z["PAIR/rows/rows"]
            h = len(rows) // 2
            r0, r1 = rows[:h], rows[h:]
            pl = get("PAIR", "base")
            q = to_frame(pl[:h, 7, :2].astype(np.float64), P["ex"][r0])
            inc = pl[h:, 6, 0] - q[:, 0]
            S["C2"] = {c: dict(n=int(m.sum()), med=round(float(np.median(inc[m])), 3), pos=round(float((inc[m] > 0).mean()), 3))
                       for c, m in (("ST", P["ST"][r0]), ("OP", P["OP"][r0]))}
        R["steer"][f.stem] = S
    (O / "report.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("peek")
    p.add_argument("--ckpt", default="NTV0-L18-s0")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p = sub.add_parser("prep")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("probe")
    p.add_argument("--cap", type=int, default=60000)
    p = sub.add_parser("dirs")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--cap", type=int, default=40000)
    p = sub.add_parser("steer")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dev", default="cpu")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--per", type=int, default=400)
    p.add_argument("--bs", type=int, default=64)
    p = sub.add_parser("report")
    a = ap.parse_args()
    dict(peek=cmd_peek, prep=cmd_prep, probe=cmd_probe, dirs=cmd_dirs, steer=cmd_steer, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
