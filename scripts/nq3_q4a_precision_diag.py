#!/usr/bin/env python
"""Q4a lam_r = 0 reproduction diagnosis: where does |pred - stored M-C| come from? (tmp/2026-09-27-q4a-fp64.md)

Usage (box): P5_SET=carla_p5v1_ba CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/nq3_q4a_precision_diag.py <model> <seed> <folds,...>
       (or `orig-fit <model> <seed>`: nq3_q4a.fit as it is, outputs under runs/nq3/q4a/precision_diag/)
Prints one JSON line per fold: the stored M-C code path, the Q4a code path in float32 (GPU / CPU eigh) and in float64,
each as max |pred - stored preds_obs.npz| (m), plus the selected lam and the inner-CV scores."""
import json, logging, os, sys, time
import numpy as np, pandas as pd, torch
os.environ.setdefault("P5_SET", "carla_p5v1_ba")
from jevdrive import elicit_e1 as E1, elicit_i3 as I, night2_n3 as N3, p5_exam as E, p5_openpilot, p5_pairs as P
from jevdrive import reactivity_mc as MC, nq3_q4a as Q
from jevdrive.common import data_dir

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
logging.basicConfig(level=logging.WARNING)
class RL:
    log = logging.getLogger("diag")
    def event(self, *a, **k): pass
rl = RL()
if sys.argv[1] == "orig-fit":      # orig-fit <model> <seed>: nq3_q4a.fit as it is, outputs under precision_diag/
    from pathlib import Path
    from jevdrive.runlog import RunLog
    base = data_dir() / "runs/nq3/q4a/precision_diag" / f"fit_rerun_{sys.argv[2]}_s{sys.argv[3]}"

    orig_out = Q.out

    def _out(*p):         # fit outputs go to the side directory, prep inputs are read from the real one
        if not p or p[0] != "fit":
            return orig_out(*p)
        d = base / Path(*p)
        d.parent.mkdir(parents=True, exist_ok=True)
        return d
    Q.out = _out
    Q._stamp = lambda msg: None
    Q.fit(RunLog("nq3", "q4a-precision-fit-rerun"), (sys.argv[2],), (int(sys.argv[3]),))
    sys.exit(0)
PRIOR_MODE = sys.argv[1] == "prior"      # prior <model> <seed> <folds>: where the float64 prior departs from the stored one
if PRIOR_MODE:
    sys.argv.pop(1)
M_, S_ = sys.argv[1], int(sys.argv[2])
FOLDS = [int(x) for x in sys.argv[3].split(",")]
dev = "cuda"
with I.p5_set(I.BA):
    t, past, fut, obs, null, pairs = E.load()
    Qf = P.load_features(t, ("L18_last",))["L18_last"]
    op = p5_openpilot.load(t, (M_,), sub="op_streams_vis")
with I.p5_set(I.I3):
    t3a, past3a, _, obs3, null3, pairs3 = E.load()
    keep = t3a.frame_name.isin(set(I.needed())).to_numpy()
    t3, past3 = t3a[keep].reset_index(drop=True), past3a[keep]
    Q3 = P.load_features(t3, ("L18_last",))["L18_last"]
    op3 = p5_openpilot.load(t3, (M_,), sub="op_streams")
n, n3 = len(t), len(t3)
ta = pd.concat([t[["frame_name", "role", "base_id", "intent"]],
                t3[["frame_name", "base_id", "intent"]].assign(role="i3", base_id="i3:" + t3.base_id)], ignore_index=True)
# stored-run tensors (n rows) and Q4a tensors (n + n3 rows)
Fn = torch.as_tensor(fut.reshape(n, -1), device=dev)
Egon = torch.as_tensor(E.ego_input(t, past), device=dev)
Qn = torch.as_tensor(Qf, device=dev)
Xn = torch.as_tensor(op[f"op-{M_} temporal"], device=dev)
Fa = torch.as_tensor(np.r_[fut.reshape(n, -1), np.zeros((n3, 40), np.float32)], device=dev)
Egoa = torch.as_tensor(np.r_[E.ego_input(t, past), E.ego_input(t3, past3)], device=dev)
Qa = torch.as_tensor(np.r_[Qf, Q3], device=dev)
Xa = torch.as_tensor(np.r_[op[f"op-{M_} temporal"], op3[f"op-{M_} temporal"]], device=dev)
pos = pd.Series(np.arange(n), index=t.frame_name)
pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
fold = E.folds(t, pairs, S_)
fold_a = np.r_[fold, np.full(n3, -2)]
ref_run = data_dir() / N3.MC_RUNS[S_]
ref = np.load(ref_run / "preds_obs.npz")
ref_at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
ev_ref = {e["fold"]: e for e in map(json.loads, open(ref_run / "events.jsonl"))
          if e.get("kind") == "mc_fold" and e.get("arm") == "pair" and e.get("model") == M_}


def solve32(D, R, M, lams, dev):
    """nq3_q4a._solve before the 2026-09-27 fix: float32 gram, float64 eigh cast back to float32."""
    ev_, V = torch.linalg.eigh((D.T @ D + M).double().to(dev))
    ev_, V = ev_.float().to(D.device), V.float().to(D.device)
    B = V.T @ (D.T @ R)
    return [V @ (B / (ev_[:, None] + lam)) for lam in lams]


def zmap32(X, mu, sd):
    return (X - mu.float()) / sd.float() / np.sqrt(X.shape[1])


def solve64(D, R, M, lams, eig_dev="cpu"):
    """Everything in float64: gram, eigh, back-substitution."""
    ev_, V = torch.linalg.eigh((D.T @ D + M).to(eig_dev))
    ev_, V = ev_.to(D.device), V.to(D.device)
    B = V.T @ (D.T @ R)
    return [V @ (B / (ev_[:, None] + lam)) for lam in lams]


def q4a_delta(f, prior, Qx, Xx, F, fa, tr, variant):
    keepp = fold[pr_ip] != f
    ip, im, grp = pr_ip[keepp], pr_im[keepp], pr_group[keepp]
    Rp = (F[ip] - F[im]) - (prior[ip] - prior[im])
    inner = MC._inner_splits(grp)
    sq, so = E1._stats(Qx, tr), E1._stats(Xx, tr)
    if variant.startswith("fp64"):
        Z = torch.cat([(Qx.double() - sq[0]) / sq[1] / np.sqrt(Qx.shape[1]), (Xx.double() - so[0]) / so[1] / np.sqrt(Xx.shape[1])], 1)
        Rp = Rp.double()
    else:
        Z = torch.cat([zmap32(Qx, *sq), zmap32(Xx, *so)], 1)
    zbar = Z[tr].mean(0)
    Zc = Z[tr] - zbar
    mu = len(ip) / len(tr)
    D = Z[ip] - Z[im]
    M = mu * (Zc.T @ Zc)
    if variant.startswith("fp64"):
        sol = lambda D_, R_, lams: solve64(D_, R_, M, lams, "cpu" if variant == "fp64" else "cuda")  # noqa: E731
    else:
        edev = variant.split("-")[1]
        sol = lambda D_, R_, lams: solve32(D_, R_, M, lams, edev)  # noqa: E731
    score = np.zeros(len(MC.LAMS))
    for a, b in inner:
        Ws = sol(D[a], Rp[a], MC.LAMS)
        score += [float(((D[b] @ W - Rp[b]) ** 2).sum()) for W in Ws]
    best = int(np.argmin(score))
    W = sol(D, Rp, [MC.LAMS[best]])[0]
    delta = (Z - zbar) @ W
    return delta, MC.LAMS[best], score / len(ip)


if PRIOR_MODE:
    from types import SimpleNamespace
    from jevdrive import planner as PL, waymo_stage_a as sa
    lg = logging.getLogger("jevdrive.planner")
    for f in FOLDS:
        ev = obs_rows[fold[obs_rows] == f]
        tr = np.flatnonzero((t.role.to_numpy() == "train") & (fold != f))
        sp = SimpleNamespace(train=tr, val=ev, seq=t.base_id.to_numpy())
        stored_prior = ref[f"prior [{M_}]"][ref_at[ev].to_numpy()].reshape(len(ev), -1).astype(np.float64)
        row = {"model": M_, "seed": S_, "fold": f, "n_train": len(tr)}
        for tag, cm in (("fp32", None), ("fp64", Q._float64_planner)):
            dt = torch.float32 if tag == "fp32" else torch.float64
            F_, Eg, X_ = Fn.to(dt), Egon.to(dt), Xn.to(dt)
            ctx = cm() if cm else __import__("contextlib").nullcontext()
            with ctx:
                Xe = PL.standardize(Eg, tr)
                _, st_e, We = sa.ridge_cv(Xe, F_, sp, F_.reshape(n, 20, 2).cpu().numpy())
                base = PL.linear_apply(We, Xe, np.arange(n))[0]
                Xi = PL.standardize(X_, tr)
                R0 = F_ - base
                _, st_p, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(n, 20, 2).cpu().numpy())
                prior = base + PL.linear_apply(Wp, Xi, np.arange(n))[0]
                A = Xe[tr] - Xe[tr].mean(0)
                g = (A.T @ A).double().cpu()
            row.update({f"{tag}_ego_lam": st_e["lam"], f"{tag}_op_lam": st_p["lam"], f"{tag}_ego_sel_ade": st_e["sel_ade"],
                        f"{tag}_prior_vs_stored": float(np.abs(prior[ev].double().cpu().numpy() - stored_prior).max())})
            if tag == "fp64":
                e64 = torch.linalg.eigvalsh(g)
                row["ego_gram_eig_min_max"] = [float(e64[0]), float(e64[-1])]
                row["ego_gram_eigs_below_lam_n"] = int((e64 < st_e["lam"] * len(tr)).sum())
                row["ego_dim"] = int(g.shape[0])
            else:
                g32 = g
        A64 = Q._float64_planner
        with A64():
            A = PL.standardize(Egon.double(), tr)[tr]
        A = A - A.mean(0)
        gt = (A.T @ A).cpu()
        row["ego_gram_fp32_abs_err_max"] = float((g32 - gt).abs().max())
        print(json.dumps(row), flush=True)
    sys.exit(0)

res = []
for f in FOLDS:
    ev = obs_rows[fold[obs_rows] == f]
    tr_a = np.flatnonzero((ta.role.to_numpy() == "train") & (fold_a != f))
    stored = ref[f"M-C pair [{M_}]"][ref_at[ev].to_numpy()].astype(np.float64)
    stored_prior = ref[f"prior [{M_}]"][ref_at[ev].to_numpy()].astype(np.float64)
    # prior on the Q4a (n + n3) tensors and on the stored (n) tensors
    pa = MC.fit_fold(f, fold_a, ta, Fa, Egoa, Xa, Qa, pr_ip, pr_im, pr_group, rl, M_, arms=())["prior"]
    pn = MC.fit_fold(f, fold, t, Fn, Egon, Xn, Qn, pr_ip, pr_im, pr_group, rl, M_, arms=())["prior"]
    pr_a, pr_n = pa[ev].reshape(-1, 20, 2).cpu().double().numpy(), pn[ev].reshape(-1, 20, 2).cpu().double().numpy()
    row = {"fold": f, "lam_stored": ev_ref[f]["lam"], "prior_a_vs_stored": np.abs(pr_a - stored_prior).max(),
           "prior_n_vs_stored": np.abs(pr_n - stored_prior).max()}
    # stored code path exactly (reactivity_mc.fit_fold pair arm, n rows) with CPU / CUDA eigh
    for edev in ("cpu", "cuda"):
        MC.EIGH_DEVICE = edev
        t0 = time.time()
        o = MC.fit_fold(f, fold, t, Fn, Egon, Xn, Qn, pr_ip, pr_im, pr_group, rl, M_, arms=("pair",))
        p = o["M-C pair"][ev].reshape(-1, 20, 2).cpu().double().numpy()
        row[f"mc_path_n_eigh{edev}"] = np.abs(p - stored).max()
        row[f"mc_path_n_eigh{edev}_s"] = time.time() - t0
    MC.EIGH_DEVICE = "cpu"
    # the Q4a code path (n + n3 rows)
    for var in ("fp32-cuda", "fp32-cpu", "fp64", "fp64-cuda"):
        t0 = time.time()
        d, lam, sc = q4a_delta(f, pa, Qa, Xa, Fa, fold_a, tr_a, var)
        p = (pa.double() + d.double())[ev].reshape(-1, 20, 2).cpu().numpy()
        row[f"q4a_{var}"] = np.abs(p - stored).max()
        row[f"q4a_{var}_lam"] = lam
        row[f"q4a_{var}_s"] = time.time() - t0
        if var == "fp64":
            row["fp64_inner_mse"] = sc.tolist()
            row["stored_inner_mse"] = ev_ref[f]["inner_mse"]
            p64 = p
            row["delta_med_m"] = float(np.median(np.linalg.norm((p - pr_a), axis=-1).mean(-1)))
            row["delta_max_m"] = float(np.abs(p - pr_a).max())
        if var == "fp32-cuda":
            p32 = p
    row["fp32cuda_vs_fp64"] = np.abs(p32 - p64).max()
    # the fix: prior and Delta in float64 (nq3_q4a._float64_planner around reactivity_mc.fit_fold, float64 inputs)
    if hasattr(Q, "_float64_planner"):
        t0 = time.time()
        d64 = lambda x: x.to(torch.float64)  # noqa: E731
        with Q._float64_planner():
            p64a = MC.fit_fold(f, fold_a, ta, d64(Fa), d64(Egoa), d64(Xa), Qa, pr_ip, pr_im, pr_group, rl, M_, arms=())["prior"]
        d, lam, sc = q4a_delta(f, p64a, Qa, d64(Xa), d64(Fa), fold_a, tr_a, "fp64-cuda")
        p = (p64a + d)[ev].reshape(-1, 20, 2).cpu().numpy()
        row["fix_prior_vs_stored"] = np.abs(p64a[ev].reshape(-1, 20, 2).cpu().numpy() - stored_prior).max()
        row["fix_vs_stored"] = np.abs(p.astype(np.float32).astype(np.float64) - stored).max()
        row["fix_lam"] = lam
        row["fix_s"] = time.time() - t0
        np.save(data_dir() / "runs/nq3/q4a/precision_diag" / f"fix_{M_}_s{S_}_f{f}_th{os.environ.get('OMP_NUM_THREADS')}.npy", p)
    # stored path fp32 (cpu eigh, n rows) against fp64
    print(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()}), flush=True)
