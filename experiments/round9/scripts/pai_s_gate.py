#!/usr/bin/env python3
"""ROUND9 PAI-S-GATE (plans/2026-10-22-pai-s-gate.md): can S consequence hinges on the policy's own PAI states (DAgger states, road_pai
raster + replay-agent boxes) join the next DAgger joint arm? CPU only, stored files; no training, no closed loop, no pool jobs.
Chain: pai_s_gate.sh. Outputs: $DATA_DIR/runs/round9/pai_s_gate/ -> experiments/round9/results/pai_s_gate/gate.json.

  build  (AlpaSim venv)  road_pai.py's evaluator-rule surface (pai_raster) at every on1_PAID row's anchor (row files runs/round6/dag/rows/PAID)
                         -> $O/road/{parts/, sdf.npy, keys.npz} and the Hinge-layout index $O/road/road_on1_PAID.npz (tokens, ok, ix, sdf_file)
  meta   (any venv)      per on1_PAID row from its row file: human log pose / future in the row frame, served plan, offroad / collision event
                         times, scene outcome -> $O/meta.npz
  fwd    (op-train)      --ckpt TAG: the trainer's build with R6DAG's argv (rtc.py band path), eval-mode plans on a seeded --frac of the on1_PAID
                         rows (pd part of on1_JTPHD) + the trainer's own agent / lat_side terms on them -> $O/fwd_<TAG>.npz
  gate   (op-train)      every term on plan / teacher label / human log, G1-G5 tables -> $O/gate.json (+ copied to the results dir)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/round9/scripts"),
                str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/op_probe/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/pai_s_gate"
RD = O / "road"
CR = DATA / "runs/op_parity/cache"
ROWS = DATA / "runs/round6/dag/rows/PAID"
CFG_GLOB = str(DATA / "runs/round6/dag/roll/runs/*/sim/generated-user-config-0.yaml")
RES = REPO / "experiments/round9/results/pai_s_gate"
HO_W, LAM, SIDE_W, BUF, MARGIN = 3.0, 10.0, 0.5, 0.4, 0.3     # R6DAG: --ho-w 3, agent / road lam 10, --lat-side 0.5, --lat-buf 0.4, margins 0.3
W_AG, W_RD, W_SD = HO_W * LAM, HO_W * LAM, HO_W * 10.0 * SIDE_W  # trainer weights: 30 / 30 / 15 x raw
STORED = {"PAID2A": ("on1_PAID2A", DATA / "runs/round6/dag2/s0/fwd/PAID2A-train-rec/plans_R6DAG-L18-s0.npz", "R6DAG-L18-s0"),
          "PAID2B": ("on1_PAID2B", DATA / "runs/round6/dag2/s1/fwd/PAID2B-train-rec/plans_R6DAG-L18-s1.npz", "R6DAG-L18-s1")}


def pool_keys(pool="on1_PAID"):
    names = np.load(CR / pool / "tab.npz")["names"].astype(str)
    f = [n.split(":") for n in names]
    return names, np.array([x[-2] for x in f]), np.array([int(x[-1]) for x in f])


# ---------------------------------------------------------------- build (AlpaSim venv)
def cmd_build(a):
    import road_pai as RP
    from jevdrive import par
    from jevdrive.run import Run
    names, sc, kk = pool_keys()
    cfg = RP.scene_cfgs(CFG_GLOB)
    scenes = sorted(set(sc))[:a.limit or None]
    assert set(scenes) <= set(cfg), f"no config for {sorted(set(scenes) - set(cfg))[:3]}"
    (RD / "parts").mkdir(parents=True, exist_ok=True)
    with Run("round9", "pai-s-gate-build", config=vars(a)) as run:
        jobs = [(s, cfg[s], sorted(kk[sc == s].tolist()), ROWS / f"{s}.npz", RD / "parts" / f"{s}.npz") for s in scenes]
        t0 = time.time()
        from jevdrive.common import n_cpus
        res = par.pmap(RP._build_one, jobs, workers=max(n_cpus() // 2, 1), run=run, desc="scenes", skip=RP._part_done)
        res.raise_if_failed()
        n = sum(len(j[2]) for j in jobs)
        sdf = np.lib.format.open_memmap(RD / "sdf.tmp.npy", "w+", np.float16, (n, 128, 96))
        S, K, E, Fr, o = [], [], [], [], 0
        for j in jobs:
            with np.load(j[4]) as z:
                m = len(z["k"])
                sdf[o:o + m] = z["sdf"]
                S += [j[0]] * m
                K.append(z["k"]), Fr.append(z["frac"])
                E += [bool(z["empty"])] * m
                o += m
        sdf.flush()
        del sdf
        os.replace(RD / "sdf.tmp.npy", RD / "sdf.npy")
        S, K, E, Fr = np.array(S), np.concatenate(K), np.array(E), np.concatenate(Fr)
        np.savez(RD / "keys.npz", scene=S, k=K, empty=E, frac=Fr)
        at = {(s, int(k)): i for i, (s, k) in enumerate(zip(S.tolist(), K.tolist()))}
        ix = np.array([at.get((s, int(k)), -1) for s, k in zip(sc, kk)], np.int64)
        ok = (ix >= 0) & ~E[np.maximum(ix, 0)]
        np.savez(RD / "road_on1_PAID.npz", tokens=names, ok=ok, ix=np.maximum(ix, 0), sdf_file="sdf.npy")
        rec = dict(rows=len(names), scenes=len(scenes), rasters=n, ok=float(ok.mean()), empty_scenes=int(len(set(S[E]))),
                   inside_gt90=float((Fr > 0.9).mean()), inside_med=float(np.median(Fr)), build_s=round(time.time() - t0, 1))
        (RD / "build.json").write_text(json.dumps(rec, indent=1))
        run.summary |= rec
        print(json.dumps(rec))


# ---------------------------------------------------------------- meta (row files)
def _meta_one(s):
    with np.load(ROWS / f"{s}.npz") as z:
        zk, ev = z["zero_kind"], z["t_event_us"].astype(np.int64)
        return s, dict(k=z["k"].astype(np.int64), t0=z["t0_us"].astype(np.int64), log0=z["log0"], log_fut=z["log_fut"], log_fut_ok=z["log_fut_ok"],
                       served=z["poses"], v0=z["v0"], ev_col=np.int64(ev[0]) if zk[0] else -1, ev_off=np.int64(ev[1]) if zk[1] else -1,
                       ev_cor=np.int64(ev[2]) if zk[2] else -1, score=float(z["score"]), fail=str(z["fail_reason"]))


def cmd_meta(a):
    from jevdrive import par
    names, sc, kk = pool_keys()
    res = par.pmap(_meta_one, sorted(set(sc)), threads=True)
    res.raise_if_failed()
    D = dict(v for v in res.values)
    n = len(names)
    out = dict(log0=np.full((n, 3), np.nan, np.float32), log_fut=np.full((n, 8, 3), np.nan, np.float32), log_fut_ok=np.zeros((n, 8), bool),
               served=np.zeros((n, 8, 3), np.float32), v0=np.zeros(n, np.float32), t0=np.zeros(n, np.int64), ev_col=np.zeros(n, np.int64),
               ev_off=np.zeros(n, np.int64), ev_cor=np.zeros(n, np.int64), score=np.zeros(n, np.float32), fail=np.empty(n, "U40"))
    for s in np.unique(sc):
        j = np.flatnonzero(sc == s)
        d = D[s]
        pos = {int(k): i for i, k in enumerate(d["k"])}
        i = np.array([pos[int(k)] for k in kk[j]])
        for q in ("log0", "log_fut", "log_fut_ok", "served", "v0", "t0"):
            out[q][j] = d[q][i]
        for q in ("ev_col", "ev_off", "ev_cor", "score", "fail"):
            out[q][j] = d[q]
    O.mkdir(parents=True, exist_ok=True)
    np.savez(O / "meta.npz", names=names, scene=sc, k=kk, **out)
    print(json.dumps(dict(rows=n, scenes=len(D), off_scenes=int(len({s for s, d in D.items() if d["ev_off"] >= 0})),
                          col_scenes=int(len({s for s, d in D.items() if d["ev_col"] >= 0})))))


# ---------------------------------------------------------------- fwd (op-train, CPU)
def cmd_fwd(a):
    import torch
    import rtc
    from jevdrive.common import n_cpus
    torch.set_num_threads(a.threads or n_cpus())
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    sys.argv = [str(rt), "rely", *rtc.R6DAG_ARGV.format(tag=a.ckpt, data=DATA).split()]
    exec(compile(src.replace(tail, "_PSG_ARGS = a"), str(rt), "exec"), g)
    ra = g["_PSG_ARGS"]
    ra.rm_from = a.ckpt
    dev = torch.device("cpu")
    ns, *_ = g["build"](ra, dev)
    LS = ns.losses(ns.mode)
    import pp_train as T
    S, model = ns.S, ns.model.eval()
    pn = S.tab["names"][ns.on0:].astype(str)
    names, _, _ = pool_keys()
    key = {":".join(x.split(":")[-2:]): i for i, x in enumerate(names)}
    pd_rows = np.flatnonzero(np.char.startswith(pn, "on:pd:"))
    assert len(pd_rows) == len(names), (len(pd_rows), len(names))
    pid = np.array([key[":".join(x.split(":")[-2:])] for x in pn[pd_rows]])
    sel = np.sort(np.random.default_rng(a.seed).choice(len(pd_rows), int(round(a.frac * len(pd_rows))), replace=False))
    pi_ = torch.as_tensor(S.pi, device=dev)
    P, AG, SD = [], [], []
    t0 = time.time()
    for j in range(0, len(sel), a.bs):
        r = torch.as_tensor(ns.on0 + pd_rows[sel[j:j + a.bs]], device=dev)
        with torch.no_grad():
            o = model(S.front[r.cpu().numpy()], S.ego[r], S.tc[r], None, None, nv=None if ns.nv is None else ns.nv[r]).float()
            x, y, psi = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[r], LS.W)
            AG.append((W_AG * LS.agent2.per_step(x, y, psi, r).mean(1) * LS.on_aok[r]).numpy())
            lt = LS.lon.loss(torch.stack([x, y, psi], -1), r - LS.on0, LS.pack)
            SD.append((HO_W * lt["lat_side"]).numpy())
        P.append(torch.stack([x, y, psi], -1).numpy())
        if j == 0:
            print(f"first batch {time.time() - t0:.1f} s, rows {len(sel)}", flush=True)
    O.mkdir(parents=True, exist_ok=True)
    np.savez(O / f"fwd_{a.ckpt}.npz", idx=pid[sel], poses=np.concatenate(P), agent_tr=np.concatenate(AG), side_tr=np.concatenate(SD))
    print(json.dumps(dict(ckpt=a.ckpt, rows=len(sel), sec=round(time.time() - t0, 1))))


# ---------------------------------------------------------------- gate (op-train, CPU)
class Terms:
    """agent / side / road (trainer weights) on (n, 8, 3) rear-axle poses of pool-local rows."""

    def __init__(self, pool):
        import torch
        import nr1_lat
        from agent_hinge import AgentHinge
        from drivable_hinge import Hinge, interp_matrix
        self.names = np.load(CR / pool / "tab.npz")["names"].astype(str)
        self.ah = AgentHinge(CR / pool / "agent.npz", self.names, "cpu", MARGIN, None)
        self.ah0 = AgentHinge(CR / pool / "agent.npz", self.names, "cpu", 0.0, None)          # contact only (G2 overlap)
        self.M = torch.as_tensor(interp_matrix(), dtype=torch.float32)
        f = RD / f"road_{pool}.npz"
        self.road = Hinge(f, self.names, "cpu", MARGIN) if f.exists() else None
        self.lat = nr1_lat

    def __call__(self, poses, rows, bs=4096, step0=False):
        import torch
        out = {k: [] for k in ("agent", "side", "road", "m_min", "m_late", "ag0")}
        for i in range(0, len(rows), bs):
            p = torch.as_tensor(np.asarray(poses[i:i + bs], np.float32))
            r = torch.as_tensor(rows[i:i + bs])
            x, y, psi = p[..., 0], p[..., 1], p[..., 2]
            with torch.no_grad():
                out["agent"].append((W_AG * self.ah.per_step(x, y, psi, r).mean(1) * self.ah.ok[r]).numpy())
                out["ag0"].append((self.ah0.per_step(x, y, psi, r)[:, 0] * self.ah0.ok[r]).numpy())
                d = self.lat.dense(p, self.M)
                out["side"].append((W_SD * self.lat.side(d, r, self.ah, BUF)).numpy())
                if self.road is not None:
                    m = self.road.margins(x, y, psi, r)
                    ok = self.road.ok[r].float()
                    out["road"].append((W_RD * torch.relu(MARGIN - m).mean(1) * ok).numpy())
                    out["m_min"].append(torch.where(ok > 0, m.amin(1), torch.full_like(ok, np.nan)).numpy())
                    out["m_late"].append(torch.where(ok > 0, m[:, 20:].amin(1), torch.full_like(ok, np.nan)).numpy())   # steps >= 0.5 s
        return {k: np.concatenate(v) for k, v in out.items() if v}

    def standing(self, rows, pose=None):
        """terms of a standing pose (default the row origin = the student at t0); with a pose, m_late is its footprint's min SDF (the
        interpolation's first 0.5 s runs from the origin)."""
        z = np.zeros((len(rows), 8, 3), np.float32)
        if pose is not None:
            z[:] = np.asarray(pose, np.float32)[:, None]
        return self(z, rows)


def grad_step(T, poses, rows, term, step=0.2):
    """Shape-only gradient of one term on the plan, step of max 0.2 m per pose -> (new poses, term before, after)."""
    import torch
    from nr1_lat import shape_only
    p = torch.as_tensor(np.asarray(poses, np.float32))
    r = torch.as_tensor(rows)
    x, y, psi = p[..., 0].clone().requires_grad_(True), p[..., 1].clone().requires_grad_(True), p[..., 2]

    def f(x_, y_):
        xs, ys = shape_only(x_, y_, psi)
        if term == "road":
            return W_RD * torch.relu(MARGIN - T.road.margins(xs, ys, psi, r)).mean(1)
        return W_AG * T.ah.per_step(xs, ys, psi, r).mean(1)

    v = f(x, y)
    v.sum().backward()
    gx, gy = x.grad, y.grad
    n = torch.hypot(gx, gy).amax(1, keepdim=True).clamp_min(1e-9)
    nx, ny = (x - step * gx / n).detach(), (y - step * gy / n).detach()
    with torch.no_grad():
        v1 = f(nx, ny)
    return torch.stack([nx, ny, psi], -1).numpy(), v.detach().numpy(), v1.numpy()


def arc(P):
    P = np.asarray(P, np.float64)[..., :2]
    seg = np.hypot(*np.diff(np.concatenate([np.zeros_like(P[:, :1]), P], 1), axis=1).transpose(2, 0, 1))
    return seg.sum(1), seg[:, -1] / 0.5


def _sh(x):
    return float(np.mean(x)) if len(x) else None


def table(df, w=None):
    """activity table on candidate rows (df: agent / side / road terms on plan, unlearn flags)."""
    act = (df[["agent", "side", "road"]].fillna(0) > 0)
    any_ = act.any(axis=1)
    la = any_ & ~df.unlearn
    mass = df[["agent", "side", "road"]].fillna(0).sum(axis=1)
    out = dict(rows=int(len(df)), any_active=_sh(any_), la=_sh(la), unlearn=_sh(df.unlearn), t0_off=_sh(df.t0_off),
               mass_mean=float(mass.mean()), mass_on_learnable=float(mass[~df.unlearn].sum() / max(mass.sum(), 1e-12)),
               mass_on_la=float(mass[la].sum() / max(mass.sum(), 1e-12)))
    for c in ("agent", "side", "road"):
        a = act[c]
        out[f"{c}_active"] = _sh(a)
        out[f"{c}_la"] = _sh(a & ~df.unlearn)
        out[f"{c}_mass_share"] = float(df[c].fillna(0).sum() / max(mass.sum(), 1e-12))
        out[f"{c}_label_also"] = _sh(df.loc[a, f"{c}_lab"] > 0) if f"{c}_lab" in df and a.any() else None
    return out, la


def cover(df, la):
    v, dpsi = df.speed.to_numpy(), np.degrees(np.abs(df.dpsi4.to_numpy()))
    L = la.to_numpy()
    n = max(L.sum(), 1)
    road_side = ((df.road.fillna(0) > 0) | (df.side > 0)).to_numpy() & L
    ag = (df.agent > 0).to_numpy() & L
    return dict(la_rows=int(L.sum()), gt13=float((L & (v > 13)).sum() / n), gt8=float((L & (v > 8)).sum() / n),
                turn_4_10=float((L & (dpsi > 30) & (v >= 4) & (v <= 10)).sum() / n),
                turn_inside_band_road=float((road_side & (dpsi > 30) & (v >= 2.6) & (v <= 12.2)).sum() / n),
                straight_offroad_band=float((road_side & (dpsi < 5) & (v >= 1.3) & (v <= 24.2)).sum() / n),
                drift_band=float((road_side & (dpsi < 5) & (v >= 17.2)).sum() / n),
                collision_band=float((ag & (v >= 2.7) & (v <= 16.1)).sum() / n),
                speed_q=[float(x) for x in np.percentile(v[L], [10, 50, 90])] if L.any() else None)


def cmd_gate(a):
    import pandas as pd
    import torch
    from jevdrive.common import n_cpus
    torch.set_num_threads(n_cpus())
    G = {}
    T = Terms("on1_PAID")
    tab = np.load(CR / "on1_PAID/tab.npz")
    dg = np.load(CR / "on1_PAID/dagger.npz")
    M = np.load(O / "meta.npz")
    n = len(T.names)
    assert np.array_equal(M["names"], T.names) and np.array_equal(dg["names"].astype(str), T.names)
    rows = np.arange(n)
    draw, speed, fut = dg["draw"], tab["speed"], tab["fut"]
    # ---- G1 raster validity (all PAID rows)
    ok = T.road.ok.numpy()
    Z0 = T.standing(rows)
    m_stu = Z0["m_min"]
    hl = np.isfinite(M["log0"]).all(1)
    m_hum = np.full(n, np.nan)
    m_hum[hl] = T.standing(rows[hl], M["log0"][hl])["m_late"]
    served = T(M["served"], rows)
    t0, evo = M["t0"], M["ev_off"]
    sc = M["scene"]
    ag_rows, ctrl = [], []
    for s in np.unique(sc):
        j = np.flatnonzero(sc == s)
        e = evo[j[0]]
        if e < 0:
            ctrl.append(j)
            continue
        at = j[(t0[j] >= e)]
        pre = j[(t0[j] < e) & (t0[j] >= e - 2_500_000)]
        if len(at):
            i = at[np.argmin(t0[at])]
            ag_rows.append(dict(scene=s, at_ev=True, off=bool(m_stu[i] < 0), plan=bool(served["road"][i] > 0), v0=float(M["v0"][i])))
        elif len(pre):
            ag_rows.append(dict(scene=s, at_ev=False, off=bool((m_stu[pre] < 0).any()), plan=bool((served["road"][pre] > 0).any()),
                                v0=float(M["v0"][pre].max()), plan_rate=float((served["road"][pre] > 0).mean()), t0off_rate=float((m_stu[pre] < 0).mean())))
    ctrl = np.concatenate(ctrl)
    A = pd.DataFrame(ag_rows)
    G["G1"] = dict(rows=n, raster_ok=float(ok.mean()), human_rows=int(hl.sum()),
                   human_t0_inside=float((m_hum[hl & ok] >= 0).mean()), human_t0_inside_0p3=float((m_hum[hl & ok] >= MARGIN).mean()),
                   human_t0_inside_close=float((m_hum[hl & ok & (np.hypot(*M["log0"][:, :2].T) < 3)] >= 0).mean()),
                   student_t0_off_all=float((m_stu[ok] < 0).mean()), student_t0_off_cand=float((m_stu[ok & draw] < 0).mean()),
                   offroad_scenes=int((A.shape[0])), offroad_event_rows_in_pool=int(A.at_ev.sum()) if len(A) else 0,
                   agree_off_or_plan=float((A.off | A.plan).mean()) if len(A) else None, agree_t0_off=float(A.off.mean()) if len(A) else None,
                   agree_at_event_only=float((A[A.at_ev].off | A[A.at_ev].plan).mean()) if len(A) and A.at_ev.any() else None,
                   pre_event_plan_rate=float(A.plan_rate.mean()) if "plan_rate" in A else None,
                   pre_event_t0off_rate=float(A.t0off_rate.mean()) if "t0off_rate" in A else None,
                   control_rows=int(len(ctrl)), control_t0_off=float((m_stu[ctrl][ok[ctrl]] < 0).mean()),
                   control_served_road_active=float((served["road"][ctrl] > 0).mean()))
    print("G1", json.dumps(G["G1"]), flush=True)
    # ---- G2 agent boxes
    val0 = T.ah.val[:, 0].numpy().any(1) & T.ah.ok.numpy()
    evc = M["ev_col"]
    no_col = evc < 0
    lab = T(fut, rows)
    hum_ok = np.isfinite(M["log_fut"]).all((1, 2)) & M["log_fut_ok"].all(1)
    hum = {k: np.full(n, np.nan) for k in ("agent", "road", "side")}
    hh = T(np.nan_to_num(M["log_fut"][hum_ok]), rows[hum_ok])
    for k in hum:
        hum[k][hum_ok] = hh[k]
    G["G2"] = dict(agent_ok=float(T.ah.ok.numpy().mean()), box_t0_any=float(val0.mean()),
                   boxes_t0_mean=float(T.ah.val[:, 0].numpy().sum(1).mean()),
                   t0_overlap_nocol=float((Z0["ag0"] > 0)[no_col].mean()),
                   t0_overlap_colscenes=float((Z0["ag0"] > 0)[~no_col].mean()) if (~no_col).any() else None,
                   collision_scenes=int(len(set(sc[~no_col]))))
    # ---- per-row frames for the student plans
    dpsi4 = fut[:, -1, 2]
    base = pd.DataFrame(dict(row=rows, scene=sc, speed=speed, draw=draw, dpsi4=dpsi4, t0_off=(m_stu < 0) & ok, agent_lab=lab["agent"],
                             road_lab=lab["road"], side_lab=lab["side"], hum_agent=hum["agent"], hum_road=hum["road"]))

    def frame(terms, idx):
        d = base.iloc[idx].copy().reset_index(drop=True)
        for k in ("agent", "side", "road"):
            d[k] = terms[k] if k in terms else np.nan
        ag_u = (d.agent > 0) & (d.agent_lab > 0)
        rd_u = (d.road.fillna(0) > 0) & (d.road_lab > 0)
        d["unlearn"] = d.t0_off | ag_u | rd_u
        return d[d.draw].reset_index(drop=True)

    F = {}
    sv = frame(served, rows)
    F["R5LC-s0 served (PAID)"] = sv
    for ck in a.ckpts.split(","):
        z = np.load(O / f"fwd_{ck}.npz")
        tt = T(z["poses"], z["idx"])
        d_ag = np.abs(tt["agent"] - z["agent_tr"]).max()
        d_sd = np.abs(tt["side"] - z["side_tr"]).max()
        G.setdefault("check_trainer_terms", {})[ck] = dict(agent_maxdiff=float(d_ag), side_maxdiff=float(d_sd))
        d = frame(tt, z["idx"])
        d["poses"] = list(z["poses"][base.draw.to_numpy()[z["idx"]]])
        F[f"{ck} (PAID)"] = d
    G2l = []
    for key, d in F.items():
        a_ = d.agent > 0
        r_ = d.road.fillna(0) > 0
        G2l.append(dict(src=key, agent_active=_sh(a_), label_also=_sh(d.agent_lab[a_] > 0), human_also=float((d.hum_agent[a_] > 0).mean()),
                        road_active=_sh(r_), road_label_also=_sh(d.road_lab[r_] > 0), road_human_also=float((d.hum_road[r_] > 0).mean()),
                        road_label_trig_all=_sh(d.road_lab > 0), road_human_trig_all=float((d.hum_road > 0).mean())))
    G["G2"]["by_source"] = G2l
    # ---- PAID2 (agent / side on the stored GPU plans of the students' own states)
    for k, (pool, f, ck) in STORED.items():
        T2 = Terms(pool)
        t2 = np.load(CR / pool / "tab.npz")
        d2 = np.load(CR / pool / "dagger.npz")
        z = np.load(f)
        tt = T2(z["poses"], z["rows"])
        lb = T2(t2["fut"][z["rows"]], z["rows"])
        d = pd.DataFrame(dict(speed=t2["speed"][z["rows"]], draw=d2["draw"][z["rows"]], dpsi4=t2["fut"][z["rows"], -1, 2], agent=tt["agent"],
                              side=tt["side"], road=np.nan, agent_lab=lb["agent"], side_lab=lb["side"], t0_off=False))
        d["unlearn"] = (d.agent > 0) & (d.agent_lab > 0)
        F[f"{ck} own states ({k}, no road)"] = d[d.draw].reset_index(drop=True)
        del T2
    # ---- G3 / G4
    G["G3"], G["G4"] = {}, {}
    for key, d in F.items():
        t_, la = table(d)
        G["G3"][key] = t_
        c = cover(d, la)
        for dose in (8, 16):
            c[f"la_per_batch_{dose}"] = dose * t_["la"]
            c[f"gt13_la_per_batch_{dose}"] = dose * t_["la"] * c["gt13"]
        c["candidate_speed_gt13"] = _sh(d.speed > 13)
        G["G4"][key] = c
        print(key, json.dumps(t_), json.dumps(c), flush=True)
    # ---- G5 gradient direction (primary checkpoint, PAID)
    d = F[f"{a.ckpts.split(',')[0]} (PAID)"]
    rng = np.random.default_rng(0)
    G["G5"] = {}
    for term in ("road", "agent"):
        cand = np.flatnonzero(((d[term].fillna(0) > 0) & ~d.unlearn).to_numpy())
        act = np.flatnonzero((d[term].fillna(0) > 0).to_numpy())
        pick = rng.choice(cand, min(4000, len(cand)), replace=False) if len(cand) else cand
        if not len(pick):
            continue
        P = np.stack(d.poses.iloc[pick].to_list())
        r = d.row.to_numpy()[pick]
        P1, v0, v1 = grad_step(T, P, r, term)
        a0, e0 = arc(P)
        a1, e1 = arc(P1)
        L = fut[r]
        dv = (P1 - P)[..., :2].reshape(len(P), -1)
        to = (L - P)[..., :2].reshape(len(P), -1)
        cos = (dv * to).sum(1) / np.maximum(np.linalg.norm(dv, axis=1) * np.linalg.norm(to, axis=1), 1e-9)
        lat = ((P1 - P)[..., 1]).mean(1)                                   # mean lateral move (+ left)
        G["G5"][term] = dict(rows=len(pick), active=len(act), la=len(cand), label_also_violates=_sh(d[f"{term}_lab"].to_numpy()[act] > 0),
                             arc_rel_change_med=float(np.median((a1 - a0) / np.maximum(a0, 1e-3))),
                             arc_rel_change_p10_p90=[float(x) for x in np.percentile((a1 - a0) / np.maximum(a0, 1e-3), [10, 90])],
                             end_speed_change_med=float(np.median(e1 - e0)), end_speed_change_p10_p90=[float(x) for x in np.percentile(e1 - e0, [10, 90])],
                             term_reduced=float((v1 < v0).mean()), term_rel_reduction_med=float(np.median((v0 - v1) / np.maximum(v0, 1e-9))),
                             toward_label=float((cos > 0).mean()), cos_med=float(np.median(cos)),
                             label_dist_before_med=float(np.median(np.linalg.norm((L - P)[..., :2], axis=2).mean(1))),
                             lat_left_share=float((lat > 0).mean()),
                             plan_vs_label_arc_ratio_med=float(np.median(a0 / np.maximum(arc(L)[0], 1e-3))))
        print("G5", term, json.dumps(G["G5"][term]), flush=True)
    G["meta"] = dict(weights=dict(agent=W_AG, road=W_RD, side=W_SD), margin=MARGIN, buf=BUF, ckpts=a.ckpts, frac_note="fwd on a seeded sample")
    O.mkdir(parents=True, exist_ok=True)
    (O / "gate.json").write_text(json.dumps(G, indent=1, default=float))
    A.to_csv(O / "g1_offroad_events.csv", index=False)
    print("written", O / "gate.json")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--limit", type=int, default=0)
    sp.add_parser("meta")
    f = sp.add_parser("fwd")
    f.add_argument("--ckpt", required=True)
    f.add_argument("--frac", type=float, default=0.3)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--bs", type=int, default=256)
    f.add_argument("--threads", type=int, default=0)
    gt = sp.add_parser("gate")
    gt.add_argument("--ckpts", default="R6DAG-L18-s1,R6DAG-L18-s0")
    a = p.parse_args()
    dict(build=cmd_build, meta=cmd_meta, fwd=cmd_fwd, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
