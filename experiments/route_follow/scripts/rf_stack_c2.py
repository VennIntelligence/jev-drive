#!/usr/bin/env python3
"""STACK check C2 (experiments/route_follow/plans/2026-10-16-stack-prereg.md, section 1 "C2"): can a small consequence head (lib/conshead.py)
learn the per-time contact of ANY candidate plan? No policy training. Box; envs/op-train; one GPU (pool job).

  rf_stack_c2.py [--smoke] labels   candidates + contact labels + frozen token slots of both row sets, and the hinge reference rates (3a, 3b)
  rf_stack_c2.py [--smoke] train    the head (BCE, fixed schedule and seed) -> $O/head.pt
  rf_stack_c2.py [--smoke] eval     the registered gates (1, 2), the baselines, the cons-constraint shares (3c) -> $O/c2.json
  rf_stack_c2.py [--smoke] all      the three in order
  --smoke: 2 % of the rows, a short schedule, 200 bootstrap draws, everything under $O/smoke (the full run's files untouched).

Rows. own = pool on1_L18RSG (RSG-L18-s0's own states); dev = rf_recov.dev_of (crc32(log) % 100 < 15), train = the rest. human = the 12
navtrain_full shards; train / dev = pp_train.split_rows(tab, navsim/op-parity-full); rows with a finite log future, agent boxes
(navtrain_all-k32) and the road raster joint_train.ROAD; a seeded subsample of 34 000 train + 6 000 dev rows.
Candidates (lib/stack_cand.py, cands_t): own: 0 = RSG-L18-s0's forward plan on the row's own input (rf_recov.py fwd -> $D/runs/route_follow/
stack/fwd; --smoke without it falls back to the stored served plan), 1 .. 8 = CANDS; human: 0 = the shipped teacher plan (cache teacher.npz
`plan`, pp_train.rear with the Losses' W), 1 .. 8 = CANDS, 9 = the log future.
Labels (rf_dagger.contacts' geometry): AgentHinge margin 0 / side margin 0, contact = signed distance < 0 to a counted box; drivable_hinge.Hinge
on the road raster, contact = a Pacifica corner with sdf < 0; 41 dense steps (interp matrix, step 0 = t0). Per candidate (2, 8): [agent,
road] x 0.5 s bins over steps 1 .. 40 (bin b = steps 5 b + 1 .. 5 b + 5). Depth = sum over steps 1 .. 40 of (max over counted objects of
max(0, -distance)) + (sum over the 4 corners of max(0, -sdf)). Rows without both labels, and rows in contact at t0, are dropped.
Head training: train-log rows only (an inner 10 % of the train logs, crc32("inner:" + log) % 10 == 0, is held out of the gradient and
only monitored); every step 128 own + 128 human rows with all their candidates plus EXTRA random candidates per row (a random registered
candidate of the row, then lat offset U(-1.5, 1.5) m, curvature gain U(0.7, 1.3), arc scale U(0.6, 1.1), the stack_cand transforms with
per-row values), labelled on the fly with the same geometry; BCE on the 2 x 8 bins, own and human halves weighted equally; AdamW lr 3e-4,
wd 0.05, 200 warm-up steps then cosine, STEPS steps, seed 0; the last step is the head (no checkpoint selection).
Gates and reads (eval): see the prereg and experiments/route_follow/results/stack/checks.md "C2".
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R_), str(HERE), str(R_ / "lib"), str(R_ / "experiments/op_parity/scripts"), str(R_ / "experiments/alpasim/lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = D / "runs/op_parity/cache"
OWN = "on1_L18RSG"
NAV = [f"navtrain_full.s{k}of12" for k in range(12)]
AGENT = D / "runs/op_parity/agent_labels/navtrain_all-k32.npz"
ROAD_REL = "runs/body1/labels_road/navtrain_s01234567891011.npz"
SPLIT = "navsim/op-parity-full"
BASE = "RSG-L18-s0"
FWD = D / "runs/route_follow/stack/fwd"
O_FULL = D / "runs/route_follow/stack/cons"
N_HUM = (34000, 6000)
EXTRA, BS, STEPS, LR, WD, WARM, SEED = 2, 128, 8000, 3e-4, 0.05, 200, 0
STEPS_RMS = (0.05, 0.10, 0.20)


def odir(a):
    return O_FULL / "smoke" if a.smoke else O_FULL


def road_file():
    """joint_train.ROAD (the raster the trainer uses for hinge-only rows), read from the source without importing the trainer."""
    src = (R_ / "experiments/alpasim/scripts/joint_train.py").read_text()
    v = next(n.value.value for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ROAD")
    assert v == ROAD_REL, v
    return D / v


# ---------------------------------------------------------------- geometry
def contact(ah, rh, P, rows):
    """P (B, 8, 3) torch, rows (B,) long -> agent (B, 41) bool, road (B, 41) bool, depth (B,) over steps 1 .. 40 (margin-0 hinges)."""
    import torch
    with torch.no_grad():
        dist, cnt, _ = ah.distances(P[..., 0], P[..., 1], P[..., 2], rows)
        oa, orr = ah.ok[rows][:, None], rh.ok[rows][:, None]
        ag = ((dist < 0) & cnt).any(-1) & oa
        pen = (-dist).clamp_min(0).masked_fill(~cnt, 0.0).amax(-1) * oa
        m = rh.margins(P[..., 0], P[..., 1], P[..., 2], rows).view(len(rows), 41, 4)
        ro = (m < 0).any(-1) & orr
        rd = (-m).clamp_min(0).sum(-1) * orr
    return ag, ro, (pen + rd)[:, 1:].sum(1)


def bins(x):
    return x[:, 1:].reshape(len(x), 8, 5).any(-1)


def label(ah, rh, C, rows, dev, bs=2048):
    """C (n, K, 8, 3) numpy, rows (n,) hinge rows -> lab (n, K, 2, 8) bool, depth (n, K) f4, t0 (n,) bool."""
    import torch
    n, K = C.shape[:2]
    lab, dep, t0 = np.zeros((n, K, 2, 8), bool), np.zeros((n, K), np.float32), np.zeros(n, bool)
    for i in range(0, n, bs):
        r = torch.as_tensor(rows[i:i + bs], device=dev)
        Pc = torch.as_tensor(C[i:i + bs], dtype=torch.float32, device=dev)
        for k in range(K):
            ag, ro, d = contact(ah, rh, Pc[:, k], r)
            lab[i:i + bs, k, 0], lab[i:i + bs, k, 1], dep[i:i + bs, k] = bins(ag).cpu().numpy(), bins(ro).cpu().numpy(), d.cpu().numpy()
            if k == 0:
                t0[i:i + bs] = (ag[:, 0] | ro[:, 0]).cpu().numpy()
    return lab, dep, t0


def viol03(ah3, rh, P, rows, dev, bs=4096):
    """The trainer's hinges at margin 0.3 (AgentHinge side margin 0: per_step > 0 at any of the 41 steps; road: relu(0.3 - margins) > 0
    anywhere) -> agent (n,) bool, road (n,) bool (False where unlabelled; the callers mask with ok)."""
    import torch
    va, vr = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs], device=dev)
            Q = torch.as_tensor(P[i:i + bs], dtype=torch.float32, device=dev)
            va[i:i + bs] = (ah3.per_step(Q[..., 0], Q[..., 1], Q[..., 2], r) > 0).any(1).cpu().numpy()
            vr[i:i + bs] = (torch.relu(0.3 - rh.margins(Q[..., 0], Q[..., 1], Q[..., 2], r)) > 0).any(1).cpu().numpy()
    return va, vr


def cands(P, dev):
    import torch
    import stack_cand as SC
    return SC.cands_t(torch.as_tensor(np.asarray(P), dtype=torch.float64, device=dev)).float().cpu().numpy()


def branch_vec(P, g, s):
    """stack_cand._branch_t with per-row g, s (B,) tensors (training-time random candidates)."""
    import torch
    import stack_cand as SC
    z = torch.zeros_like(P[:, :1, :2])
    xy = torch.cat([z, P[:, :, :2]], 1)
    d = xy[:, 1:] - xy[:, :-1]
    ds = torch.linalg.norm(d, dim=-1)
    z1 = torch.zeros_like(P[:, :1, 0])
    c = g[:, None] * SC._unwrap_t(torch.cat([z1, torch.atan2(d[..., 1], d[..., 0])], 1))[:, 1:]
    yaw = g[:, None] * SC._unwrap_t(torch.cat([z1, P[..., 2]], 1))
    xy = torch.cat([z, torch.cumsum(ds[..., None] * torch.stack([torch.cos(c), torch.sin(c)], -1), 1)], 1)
    S = torch.cat([z1, torch.cumsum(ds, 1)], 1)
    u = torch.cummax(S + 1e-6 * torch.arange(S.shape[1], device=P.device, dtype=P.dtype), 1).values
    w = s[:, None] * u[:, 1:]
    j = torch.searchsorted(u.contiguous(), w.contiguous()).clamp(1, u.shape[1] - 1)
    u0, u1 = u.gather(1, j - 1), u.gather(1, j)
    f = ((w - u0) / (u1 - u0).clamp_min(1e-12)).clamp(0, 1)
    lerp = lambda V: V.gather(1, j - 1) + f * (V.gather(1, j) - V.gather(1, j - 1))  # noqa: E731
    return torch.stack([lerp(xy[..., 0]), lerp(xy[..., 1]), lerp(yaw)], -1)


def rand_cands(P, gen):
    """P (B, 8, 3) -> one random candidate per row: branch (gain, arc) then a lateral offset with the stack_cand ramp."""
    import torch
    import stack_cand as SC
    B = len(P)
    u = torch.rand(B, 3, generator=gen, device=P.device)
    g, s, dl = 0.7 + 0.6 * u[:, 0], 0.6 + 0.5 * u[:, 1], -1.5 + 3.0 * u[:, 2]
    Q = branch_vec(P, g, s)
    a = dl[:, None] * torch.as_tensor(SC.RAMP, device=P.device, dtype=P.dtype)[None]
    return torch.stack([Q[..., 0] - a * torch.sin(Q[..., 2]), Q[..., 1] + a * torch.cos(Q[..., 2]), Q[..., 2]], -1)


def checks():
    """stack_cand selftest (torch vs numpy) and branch_np == perturb.branch_poses bit for bit."""
    import stack_cand as SC
    import perturb as PB
    worst = SC.selftest()
    rng = np.random.default_rng(1)
    eq = True
    for i in range(300):
        arc = np.cumsum(rng.uniform(0, 4, 8)) if i % 10 else np.zeros(8)
        th = np.linspace(0, rng.uniform(-1.2, 1.2), 8)
        P = np.c_[arc * np.cos(th), arc * np.sin(th), th + rng.normal(0, 0.05, 8)]
        for g, s in ((0.8, 1.0), (1.2, 1.0), (1.0, 0.75), (1.0, 0.9), (0.9, 0.8)):
            eq &= bool(np.array_equal(SC.branch_np(P, g, s), PB.branch_poses(P, g, s)))
    assert eq, "branch_np != perturb.branch_poses"
    return dict(cands_t_vs_np_max=worst, branch_np_equals_perturb=eq)


def gather_front(f, rows, slots, chunk=2048):
    from conshead import SLOTS
    F = np.load(f, mmap_mode="r")
    out = np.empty((len(rows), len(slots or SLOTS), 32, 512), np.float16)
    sl = np.asarray(slots or SLOTS)
    o = np.argsort(rows, kind="stable")
    for i in range(0, len(rows), chunk):
        j = o[i:i + chunk]
        out[j] = F[rows[j][:, None], sl[None]]
    return out


def nav_tab():
    tabs = [np.load(CR / d / "tab.npz") for d in NAV]
    T = {k: np.concatenate([t[k] for t in tabs]) for k in ("names", "log", "ego", "fut", "cam", "speed")}
    T["shard"] = np.concatenate([np.full(len(t["names"]), i) for i, t in enumerate(tabs)])
    T["local"] = np.concatenate([np.arange(len(t["names"])) for t in tabs])
    return T


def nav_front(T, rows, slots=None):
    out = np.empty((len(rows), 2, 32, 512), np.float16)
    for s in np.unique(T["shard"][rows]):
        m = T["shard"][rows] == s
        out[m] = gather_front(CR / f"{NAV[s]}@warp" / "front.npy", T["local"][rows][m], slots)
    return out


def ci(x, g, B):
    from jevdrive import stats
    r = stats.bootstrap(np.asarray(x, float), None if g is None else np.asarray(g), n_boot=B)
    return dict(mean=round(float(r["mean"]), 4), lo=round(float(r["lo"]), 4), hi=round(float(r["hi"]), 4), n=int(r["n"]), units=int(r["units"]))


# ---------------------------------------------------------------- labels
def cmd_labels(a, run):
    import torch
    import pp_train as T
    import rf_recov as RC
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    dev = torch.device("cuda")
    O = odir(a)
    O.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rec = dict(checks=checks())
    run.info(f"checks {rec['checks']}")
    # ---- own rows
    tb = dict(np.load(CR / OWN / "tab.npz"))
    names = tb["names"]
    N = len(names)
    dv = RC.dev_of(tb)
    P0 = np.full((N, 8, 3), np.nan, np.float32)
    src = "fwd"
    for rs in ("dev-rec", "train-rec"):
        f = FWD / f"L18RSG-{rs}" / f"plans_{BASE}.npz"
        if f.exists():
            z = np.load(f)
            P0[z["rows"]] = z["poses"]
    if not np.isfinite(P0).all():
        assert a.smoke, f"forward plans missing for {int((~np.isfinite(P0).all((1, 2))).sum())} rows ({FWD})"
        P0, src = tb["served"].astype(np.float32), "served (smoke fallback)"
    d_sv = np.hypot(*(P0[..., :2] - tb["served"][..., :2]).transpose(2, 0, 1)).mean(1)
    rows = np.arange(N) if not a.smoke else np.sort(rng.choice(N, max(N // 50, 200), replace=False))
    ah, rh = AgentHinge(CR / OWN / "agent.npz", names, dev, 0.0, 0.0), Hinge(CR / OWN / "road.npz", names, dev, 0.0)
    C = cands(P0[rows], dev)
    lab, dep, t0 = label(ah, rh, C, rows, dev)
    okA, okR = ah.ok.cpu().numpy()[rows], rh.ok.cpu().numpy()[rows]
    keep = okA & okR & ~t0
    # 3 (descriptive): the trainer's margin-0.3 hinges on RSG-L18-s0's own plans, own train rows
    ah3 = AgentHinge(CR / OWN / "agent.npz", names, dev, 0.3, 0.0)
    tr = rows[~dv[rows]]
    va, vr = viol03(ah3, rh, P0[tr], tr, dev)
    lg = tb["log"]
    mA, mR = ah.ok.cpu().numpy()[tr], rh.ok.cpu().numpy()[tr]
    rec["own"] = dict(pool=OWN, centre=src, centre_vs_served_mean_pose_m=dict(median=float(np.median(d_sv)), p95=float(np.percentile(d_sv, 95))),
                      rows=int(len(rows)), no_agent_label=int((~okA).sum()), no_road_label=int((~okR).sum()), t0_contact=int((t0 & okA & okR).sum()),
                      kept=int(keep.sum()), kept_dev=int((keep & dv[rows]).sum()), kept_train=int((keep & ~dv[rows]).sum()),
                      logs_dev=int(len(set(lg[rows][keep & dv[rows]]))), logs_train=int(len(set(lg[rows][keep & ~dv[rows]]))),
                      contact_rate_by_cand=lab[keep].any((2, 3)).mean(0).round(4).tolist(),
                      agent_rate_by_cand=lab[keep][:, :, 0].any(-1).mean(0).round(4).tolist(),
                      road_rate_by_cand=lab[keep][:, :, 1].any(-1).mean(0).round(4).tolist(),
                      rates03_own_plans_train=dict(rows=int(len(tr)), agent=ci(va[mA], lg[tr][mA], a.boot), road=ci(vr[mR], lg[tr][mR], a.boot),
                                                   agent_excl_t0=ci(va[mA & ~t0[~dv[rows]]], lg[tr][mA & ~t0[~dv[rows]]], a.boot)))
    r = rows[keep]
    run.info(f"own: {len(rows)} rows, kept {keep.sum()} ({rec['own']['kept_dev']} dev), centre {src}")
    np.savez(O / "own.npz", rows=r, names=names[r], log=lg[r], dev=dv[r], ego=tb["ego"][r].astype(np.float32), speed=tb["speed"][r].astype(np.float32),
             cands=C[keep], lab=lab[keep], dep=dep[keep], plan=P0[r])
    np.save(O / "own_front.npy", gather_front(CR / f"{OWN}@warp" / "front.npy", r, None))
    del ah, rh, ah3
    torch.cuda.empty_cache()
    # ---- human rows
    NT = nav_tab()
    names = NT["names"]
    trr, dvr, sp = T.split_rows(dict(names=names, log=NT["log"]), SPLIT)
    for s in sp:
        run.use_split(s)
    ahn, rhn = AgentHinge(AGENT, names, dev, 0.0, 0.0), Hinge(road_file(), names, dev, 0.0)
    ah_ok, rh_ok = ahn.ok.cpu().numpy(), rhn.ok.cpu().numpy()
    has = np.isfinite(NT["fut"]).all((1, 2))
    good = has & ah_ok & rh_ok
    ntr, ndv = (680, 120) if a.smoke else N_HUM
    pick_tr = np.sort(rng.choice(trr[good[trr]], min(ntr, int(good[trr].sum())), replace=False))
    pick_dv = np.sort(rng.choice(dvr[good[dvr]], min(ndv, int(good[dvr].sum())), replace=False))
    hr = np.r_[pick_tr, pick_dv]
    hdev = np.r_[np.zeros(len(pick_tr), bool), np.ones(len(pick_dv), bool)]
    W = torch.as_tensor(T.R2.t_weights(T.T8), dtype=torch.float32, device=dev)
    tp = []
    for s in range(12):
        m = NT["shard"][hr] == s
        tf = CR / f"{NAV[s]}@warp" / "teacher.npz"
        tf = tf if tf.exists() else CR / NAV[s] / "teacher.npz"
        pl = np.load(tf)["plan"][NT["local"][hr][m]]
        x, y, psi = T.rear(torch.as_tensor(pl, device=dev), torch.as_tensor(NT["cam"][hr][m, 0], device=dev), W)
        tp.append((np.flatnonzero(m), torch.stack([x, y, psi], -1).cpu().numpy()))
    Pt = np.zeros((len(hr), 8, 3), np.float32)
    for m, p in tp:
        Pt[m] = p
    C = np.concatenate([cands(Pt, dev), NT["fut"][hr][:, None].astype(np.float32)], 1)
    lab, dep, t0 = label(ahn, rhn, C, hr, dev)
    keep = ~t0
    h = hr[keep]
    ld = NT["log"]
    rec["human"] = dict(split=SPLIT, split_rows=dict(train=int(len(trr)), dev=int(len(dvr))),
                        eligible=dict(train=int(good[trr].sum()), dev=int(good[dvr].sum())), picked=dict(train=int(len(pick_tr)), dev=int(len(pick_dv))),
                        t0_contact=int(t0.sum()), kept=int(keep.sum()), kept_dev=int((keep & hdev).sum()),
                        logs_dev=int(len(set(ld[hr][keep & hdev]))), logs_train=int(len(set(ld[hr][keep & ~hdev]))),
                        dev_logs_also_in_train=int(len(set(ld[trr]) & set(ld[dvr]))),
                        contact_rate_by_cand=lab[keep].any((2, 3)).mean(0).round(4).tolist(),
                        agent_rate_by_cand=lab[keep][:, :, 0].any(-1).mean(0).round(4).tolist(),
                        road_rate_by_cand=lab[keep][:, :, 1].any(-1).mean(0).round(4).tolist())
    run.info(f"human: picked {len(hr)}, kept {keep.sum()}")
    np.savez(O / "hum.npz", rows=h, names=names[h], log=ld[h], dev=hdev[keep], ego=NT["ego"][h].astype(np.float32), speed=NT["speed"][h].astype(np.float32),
             cands=C[keep], lab=lab[keep], dep=dep[keep], plan=Pt[keep])
    np.save(O / "hum_front.npy", nav_front(NT, h))
    # ---- 3a / 3b: navtrain TRAIN rows' log futures, the trainer's margin-0.3 geometry
    ah3 = AgentHinge(AGENT, names, dev, 0.3, 0.0)
    rt = trr[has[trr]]
    if a.smoke:
        rt = np.sort(rng.choice(rt, len(rt) // 50, replace=False))
    va, vr = viol03(ah3, rhn, NT["fut"][rt], rt, dev)
    _, _, t0f = label(ahn, rhn, NT["fut"][rt][:, None].astype(np.float32), rt, dev)
    mA, mR = ah_ok[rt], rh_ok[rt]
    rec["rates03_human_logfut_train"] = dict(rows=int(len(rt)), agent_rows=int(mA.sum()), road_rows=int(mR.sum()),
                                             agent=ci(va[mA], ld[rt][mA], a.boot), road=ci(vr[mR], ld[rt][mR], a.boot),
                                             agent_excl_t0=ci(va[mA & ~t0f], ld[rt][mA & ~t0f], a.boot), road_excl_t0=ci(vr[mR & ~t0f], ld[rt][mR & ~t0f], a.boot),
                                             t0_contact_m0=float(t0f[mA & mR].mean()))
    (O / "labels.json").write_text(json.dumps(rec, indent=1))
    run.summary.update(own_kept=rec["own"]["kept"], hum_kept=rec["human"]["kept"])
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- train
class Data:
    """One row set on the card: tokens, ego, candidates, labels, and the margin-0 hinges for on-the-fly labels."""

    def __init__(self, O, which, dev, hinges=True):
        import torch
        from agent_hinge import AgentHinge
        from drivable_hinge import Hinge
        z = dict(np.load(O / f"{which}.npz"))
        self.z = z
        self.n = len(z["rows"])
        t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), device=dev, dtype=dt)  # noqa: E731
        self.front = t(np.load(O / f"{which}_front.npy"))
        self.ego, self.C, self.lab = t(z["ego"]), t(z["cands"]), t(z["lab"], torch.float32)
        self.rows = t(z["rows"], torch.long)
        self.log, self.dev = z["log"], z["dev"].astype(bool)
        self.inner = np.array([zlib.crc32(("inner:" + str(x)).encode()) % 10 == 0 for x in self.log])
        if hinges:
            names = np.load(CR / OWN / "tab.npz")["names"] if which == "own" else nav_tab()["names"]
            ag, ro = (CR / OWN / "agent.npz", CR / OWN / "road.npz") if which == "own" else (AGENT, road_file())
            self.ah, self.rh = AgentHinge(ag, names, dev, 0.0, 0.0), Hinge(ro, names, dev, 0.0, bank=True)


def cmd_train(a, run):
    import torch
    import torch.nn.functional as F
    from conshead import ConsHead
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    O = odir(a)
    Do, Dh = Data(O, "own", dev), Data(O, "hum", dev)
    tr_o, tr_h = np.flatnonzero(~Do.dev & ~Do.inner), np.flatnonzero(~Dh.dev & ~Dh.inner)
    iv_o = np.flatnonzero(~Do.dev & Do.inner)
    E = torch.cat([Do.ego[torch.as_tensor(tr_o, device=dev)], Dh.ego[torch.as_tensor(tr_h, device=dev)]])
    head = ConsHead().to(dev)
    head.ego_mu.copy_(E.mean(0))
    head.ego_sd.copy_(E.std(0).clamp_min(1e-3))
    steps = 60 if a.smoke else a.steps
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    gen = torch.Generator(device=dev).manual_seed(SEED)
    tro, trh = torch.as_tensor(tr_o, device=dev), torch.as_tensor(tr_h, device=dev)
    run.info(f"train rows own {len(tr_o)} human {len(tr_h)} (inner-val own {len(iv_o)}); params {sum(p.numel() for p in head.parameters())}")

    def batch(Dx, tri):
        i = tri[torch.randint(len(tri), (BS,), generator=gen, device=dev)]
        C, L = Dx.C[i], Dx.lab[i]
        ex, el = [], []
        for _ in range(EXTRA):
            k = torch.randint(C.shape[1], (BS,), generator=gen, device=dev)
            Pr = rand_cands(C[torch.arange(BS, device=dev), k], gen)
            ag, ro, _ = contact(Dx.ah, Dx.rh, Pr, Dx.rows[i])
            ex.append(Pr)
            el.append(torch.stack([bins(ag), bins(ro)], 1).float())
        return i, torch.cat([C, torch.stack(ex, 1)], 1), torch.cat([L, torch.stack(el, 1)], 1)

    def inner_auc():
        p = predict(head, Do, iv_o, dev)
        return auc(Do.z["lab"][iv_o].any((2, 3)).ravel(), p.max((2, 3)).ravel()) if len(iv_o) else float("nan")
    t0 = time.time()
    head.train()
    for s in range(steps):
        lo = []
        for Dx, tri in ((Do, tro), (Dh, trh)):
            i, C, L = batch(Dx, tri)
            lo.append(F.binary_cross_entropy_with_logits(head(Dx.front[i], Dx.ego[i], C), L))
        loss = 0.5 * (lo[0] + lo[1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == steps - 1:
            run.scalar("loss/own", float(lo[0]), s)
            run.scalar("loss/hum", float(lo[1]), s)
        if (s + 1) % 1000 == 0 or s == steps - 1:
            head.eval()
            v = inner_auc()
            head.train()
            run.scalar("inner_val/own_auc", v, s)
            run.info(f"step {s + 1}/{steps} loss own {float(lo[0]):.4f} hum {float(lo[1]):.4f} inner-val own AUC {v:.4f} ({time.time() - t0:.0f} s)")
    head.eval()
    meta = dict(steps=steps, bs=BS, extra=EXTRA, lr=LR, wd=WD, warm=WARM, seed=SEED, train_own=int(len(tr_o)), train_hum=int(len(tr_h)),
                inner_val_own_auc=inner_auc(), wall_s=round(time.time() - t0, 1), slots=list(__import__("conshead").SLOTS))
    torch.save(dict(cfg=head.cfg, state=head.state_dict(), meta=meta), O / "head.pt")
    (O / "train.json").write_text(json.dumps(meta, indent=1))
    run.summary.update(meta)
    run.info(f"saved {O / 'head.pt'} {meta}")


# ---------------------------------------------------------------- eval
def predict(head, Dx, idx, dev, poses=None, bs=512):
    """Probabilities (n, K, 2, 8) of the rows idx on their registered candidates (or on poses (n, K, 8, 3))."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = torch.as_tensor(idx[i:i + bs], device=dev)
            P = Dx.C[j] if poses is None else torch.as_tensor(poses[i:i + bs], device=dev)
            out.append(torch.sigmoid(head(Dx.front[j], Dx.ego[j], P)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, Dx.C.shape[1], 2, 8), np.float32)


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    npos, nneg = y.sum(), (~y).sum()
    if not npos or not nneg:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_ci(y, s, g, B, seed=0):
    """AUC with the log-clustered percentile bootstrap (jevdrive.stats' resampling: default_rng(seed).integers over clusters, B draws):
    each draw weights every item by the number of times its log was drawn."""
    import pandas as pd
    y, s = np.asarray(y, bool), np.asarray(s, float)
    codes, uniq = pd.factorize(np.asarray(g))
    o = np.argsort(s, kind="stable")
    y, s, codes = y[o], s[o], codes[o]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    rng = np.random.default_rng(seed)
    U = len(uniq)
    idx = rng.integers(U, size=(B, U))
    vals = []
    chunk = max(1, int(4e6 // len(y)))
    for i in range(0, B, chunk):
        cnt = np.zeros((min(chunk, B - i), U))
        np.add.at(cnt, (np.arange(len(cnt))[:, None], idx[i:i + chunk]), 1)
        w = cnt[:, codes]
        wp, wn = np.add.reduceat(w * y, starts, 1), np.add.reduceat(w * ~y, starts, 1)
        below = np.cumsum(wn, 1) - wn
        vals.append((wp * (below + 0.5 * wn)).sum(1) / np.maximum(wp.sum(1) * wn.sum(1), 1e-12))
    b = np.concatenate(vals)
    lo, hi = np.quantile(b, [0.025, 0.975])
    return dict(auc=round(auc(y, s), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n=int(len(y)), pos=int(y.sum()), units=int(U))


def within(y, s, g, B):
    """y, s (n, K): AUC among the K candidates of each state with both outcomes; mean over states, log-clustered CI."""
    v = np.array([auc(y[i], s[i]) for i in range(len(y))])
    m = np.isfinite(v)
    return dict(ci(v[m], g[m], B), states=int(m.sum()), of=int(len(y)))


def logreg(Xtr, ytr, Xte, dev):
    """Logistic regression (L2 1e-4, LBFGS) -> test probabilities."""
    import torch
    X, y = torch.as_tensor(Xtr, dtype=torch.float32, device=dev), torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")

    def f():
        opt.zero_grad()
        L = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y) + 1e-4 * (w ** 2).sum()
        L.backward()
        return L
    opt.step(f)
    with torch.no_grad():
        return torch.sigmoid(torch.as_tensor(Xte, dtype=torch.float32, device=dev) @ w).cpu().numpy()


def base_feats(K, speed, mu, sd):
    n = len(speed)
    X = np.zeros((n, K, K + 1), np.float32)
    X[:, np.arange(K), np.arange(K)] = 1
    X[..., K] = ((speed - mu) / sd)[:, None]
    return X.reshape(n * K, K + 1)


def grad_read(head, Dx, idx, dev, B, bs=256):
    """Gate 2 on the dev candidates with true contact: one xy step of every pose along -grad(sum of predicted probabilities) (yaw kept),
    scaled to an RMS displacement of h per pose; true depth strictly lower after the step? Same for a random xy direction of the same size."""
    import torch
    K = Dx.C.shape[1]
    dep = Dx.z["dep"][idx]
    ii, kk = np.nonzero(dep > 0)
    gen = torch.Generator(device=dev).manual_seed(1)
    res = {h: dict(g=[], r=[], gd=[], rd=[], ps=[]) for h in STEPS_RMS}
    for i in range(0, len(ii), bs):
        j = torch.as_tensor(idx[ii[i:i + bs]], device=dev)
        P = Dx.C[j, torch.as_tensor(kk[i:i + bs], device=dev)].clone().requires_grad_(True)
        p = torch.sigmoid(head(Dx.front[j], Dx.ego[j], P))
        (g,) = torch.autograd.grad(p.sum(), P)
        d = -g[..., :2]
        rd = torch.randn(d.shape, generator=gen, device=dev)
        rms = lambda v: v.pow(2).sum(-1).mean(-1, keepdim=True).sqrt()[..., None].clamp_min(1e-12)  # noqa: E731
        d0 = depth_of(Dx, j, P.detach())
        for h in STEPS_RMS:
            for key, v in (("g", d), ("r", rd)):
                Q = P.detach().clone()
                Q[..., :2] += h * v / rms(v)
                d1 = depth_of(Dx, j, Q)
                res[h][key].append((d1 < d0 - 1e-6).cpu().numpy() & (rms(v)[:, 0, 0] > 1e-11).cpu().numpy())
                res[h][key + "d"].append((d1 - d0).cpu().numpy())
                if key == "g":
                    with torch.no_grad():
                        res[h]["ps"].append((torch.sigmoid(head(Dx.front[j], Dx.ego[j], Q)).sum((1, 2)) < p.detach().sum((1, 2))).cpu().numpy())
    g = Dx.log[idx][ii]
    out = dict(candidates=int(len(ii)), rows=int(len(set(ii))), by_cand=np.bincount(kk, minlength=K).tolist())
    for h in STEPS_RMS:
        R = {k: np.concatenate(v) for k, v in res[h].items()}
        out[f"rms_{h:.2f}"] = dict(grad=ci(R["g"], g, B), random=ci(R["r"], g, B), grad_mean_depth_change=float(R["gd"].mean()),
                                   random_mean_depth_change=float(R["rd"].mean()), grad_pred_sum_decreases=float(R["ps"].mean()))
    return out


def depth_of(Dx, j, Q):
    return contact(Dx.ah, Dx.rh, Q, Dx.rows[j])[2]


def cmd_eval(a, run):
    import torch
    import conshead as CH
    import pp_train as T
    dev = torch.device("cuda")
    O = odir(a)
    B = a.boot
    head = CH.load(O / "head.pt", dev)
    Do, Dh = Data(O, "own", dev), Data(O, "hum", dev, hinges=False)
    rec = dict(labels=json.loads((O / "labels.json").read_text()), train=json.loads((O / "train.json").read_text()), boot=B)
    out = {}
    for nm, Dx in (("own", Do), ("human", Dh)):
        dv, trn = np.flatnonzero(Dx.dev), np.flatnonzero(~Dx.dev)
        K = Dx.C.shape[1]
        p = predict(head, Dx, dv, dev)
        L = Dx.z["lab"][dv]
        g = np.repeat(Dx.log[dv], K)
        y_any, s_any = L.any((2, 3)), p.max((2, 3))
        R = dict(rows=int(len(dv)), logs=int(len(set(Dx.log[dv]))), K=K,
                 pooled=auc_ci(y_any.ravel(), s_any.ravel(), g, B),
                 agent=auc_ci(L[:, :, 0].any(-1).ravel(), p[:, :, 0].max(-1).ravel(), g, B),
                 road=auc_ci(L[:, :, 1].any(-1).ravel(), p[:, :, 1].max(-1).ravel(), g, B),
                 perturbed_only=auc_ci(y_any[:, 1:9].ravel(), s_any[:, 1:9].ravel(), np.repeat(Dx.log[dv], 8), B),
                 executed=auc_ci(y_any[:, 0], s_any[:, 0], Dx.log[dv], B),
                 within_state=within(y_any, s_any, Dx.log[dv], B),
                 bin_level=auc(L.ravel(), p.ravel()),
                 calib=dict(rate_by_cand=y_any.mean(0).round(4).tolist(), mean_maxp_by_cand=s_any.mean(0).round(4).tolist(),
                            share_maxp_gt_05_by_cand=(s_any > 0.5).mean(0).round(4).tolist()))
        if nm == "human":
            R["log_future"] = auc_ci(y_any[:, 9], s_any[:, 9], Dx.log[dv], B)
            R["without_log_future"] = auc_ci(y_any[:, :9].ravel(), s_any[:, :9].ravel(), np.repeat(Dx.log[dv], 9), B)
        # candidate type + speed baseline (trained on the same set's train rows)
        sp = Dx.z["speed"]
        mu, sd = float(sp[trn].mean()), float(sp[trn].std() + 1e-6)
        pb = logreg(base_feats(K, sp[trn], mu, sd), Dx.z["lab"][trn].any((2, 3)).ravel(), base_feats(K, sp[dv], mu, sd), dev).reshape(len(dv), K)
        R["baseline_type_speed"] = dict(pooled=auc_ci(y_any.ravel(), pb.ravel(), g, B), executed=auc_ci(y_any[:, 0], pb[:, 0], Dx.log[dv], B),
                                        within_state=within(y_any, pb, Dx.log[dv], B))
        ptr = predict(head, Dx, trn[:: max(1, len(trn) // 4000)], dev)
        R["train_rows_auc_descriptive"] = auc(Dx.z["lab"][trn[:: max(1, len(trn) // 4000)]].any((2, 3)).ravel(), ptr.max((2, 3)).ravel())
        out[nm] = R
        run.info(f"{nm}: pooled {R['pooled']} executed {R['executed']} baseline {R['baseline_type_speed']['pooled']}")
    out["gradient"] = grad_read(head, Do, np.flatnonzero(Do.dev), dev, B)
    run.info(f"gradient {out['gradient']}")
    # ---- 3c: share of rows with max predicted contact probability > 0.5 (the cons constraint)
    NT = nav_tab()
    trr, _, sp_ = T.split_rows(dict(names=NT["names"], log=NT["log"]), SPLIT)
    for s in sp_:
        run.use_split(s)
    rt = trr[np.isfinite(NT["fut"][trr]).all((1, 2))]
    if a.smoke:
        rt = np.sort(np.random.default_rng(SEED).choice(rt, len(rt) // 50, replace=False))
    mx = np.zeros(len(rt), np.float32)
    ch = 8192
    with torch.no_grad():
        for i in range(0, len(rt), ch):
            r = rt[i:i + ch]
            Fr = torch.as_tensor(nav_front(NT, r), device=dev)
            for k in range(0, len(r), 1024):
                p = torch.sigmoid(head(Fr[k:k + 1024], torch.as_tensor(NT["ego"][r[k:k + 1024]], device=dev),
                                       torch.as_tensor(NT["fut"][r[k:k + 1024]], dtype=torch.float32, device=dev)))
                mx[i + k:i + k + len(p)] = p.amax((1, 2)).cpu().numpy()
    inh = np.isin(rt, Dh.z["rows"][~Dh.dev])
    out["cons_share_human_logfut_train"] = dict(all=ci(mx > 0.5, NT["log"][rt], B), head_train_rows=ci((mx > 0.5)[inh], NT["log"][rt][inh], B),
                                                other_rows=ci((mx > 0.5)[~inh], NT["log"][rt][~inh], B), mean_maxp=float(mx.mean()))
    tb = dict(np.load(CR / OWN / "tab.npz"))
    import rf_recov as RC
    otr = np.flatnonzero(~RC.dev_of(tb))
    P0 = np.full((len(tb["names"]), 8, 3), np.nan, np.float32)
    for rs in ("dev-rec", "train-rec"):
        f = FWD / f"L18RSG-{rs}" / f"plans_{BASE}.npz"
        if f.exists():
            z = np.load(f)
            P0[z["rows"]] = z["poses"]
    if not np.isfinite(P0).all():
        P0 = tb["served"].astype(np.float32)
    if a.smoke:
        otr = np.sort(np.random.default_rng(SEED).choice(otr, len(otr) // 50, replace=False))
    Fo = torch.as_tensor(gather_front(CR / f"{OWN}@warp" / "front.npy", otr, None), device=dev)
    mo = []
    with torch.no_grad():
        for k in range(0, len(otr), 1024):
            r = otr[k:k + 1024]
            mo.append(torch.sigmoid(head(Fo[k:k + 1024], torch.as_tensor(tb["ego"][r], device=dev), torch.as_tensor(P0[r], device=dev))).amax((1, 2)).cpu().numpy())
    mo = np.concatenate(mo)
    out["cons_share_own_plans_train"] = dict(all=ci(mo > 0.5, tb["log"][otr], B), mean_maxp=float(mo.mean()))
    o, gr = out["own"], out["gradient"]["rms_0.10"]
    out["gates"] = dict(auc_pooled_own_dev=dict(value=o["pooled"]["auc"], ci=[o["pooled"]["lo"], o["pooled"]["hi"]], threshold=0.85,
                                                passed=bool(o["pooled"]["auc"] >= 0.85)),
                        gradient_alignment_0_10=dict(value=gr["grad"]["mean"], ci=[gr["grad"]["lo"], gr["grad"]["hi"]], threshold=0.70,
                                                     passed=bool(gr["grad"]["mean"] >= 0.70)))
    out["gates"]["passed"] = all(v["passed"] for v in out["gates"].values())
    rec.update(out)
    (O / "c2.json").write_text(json.dumps(rec, indent=1, default=float))
    run.summary.update(gates=out["gates"])
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("cmd", choices=["labels", "train", "eval", "all"])
    a = ap.parse_args()
    a.boot = a.boot or (200 if a.smoke else 10000)
    from jevdrive.run import Run
    for c in (["labels", "train", "eval"] if a.cmd == "all" else [a.cmd]):
        with Run("route_follow", f"stack-c2-{c}{'-smoke' if a.smoke else ''}", seed=SEED, config=vars(a)) as run:
            dict(labels=cmd_labels, train=cmd_train, eval=cmd_eval)[c](a, run)
