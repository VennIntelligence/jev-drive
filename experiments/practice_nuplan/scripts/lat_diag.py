#!/usr/bin/env python3
"""LATERAL-DIAG (results/lateral_diag.md): why NTVX (NTV0's recipe on on1_JTX = on1_JT + the PN0-5 start-state S rows) is worse laterally on
held 460 than NTV0. Diagnosis on stored data, no training. Box, repo root.

  $TR lat_diag.py rows      (CPU, op-train) every S row (mode self) of on1_JTX: the two pool hinges exactly as the trainer sets them up
        (on_train.build: agent hinge 0.3 m in the corridor / 0 beside it, road-and-lane raster 0.3 m, rows off that raster at t0 on
        <POOL>/drivable.npz; shape-only poses, loss43.shape_only) on three paths in the row frame: the stored own plan of the rolling policy
        (`plan`), the imitation target `fut` (the human's path blended from the ego's offset, the row's only human-consistent path) and,
        at k <= 1 (the ego at the log's state), the log's own future `log` (label gate: the human must not activate his own labels).
        Per row and path: hinge value, active, and the norm of its gradient on the plan's (x, y) times the trainer's weight (ho_w 3 x lam 10;
        shape-only, so this is a sideways / turning push); per-row context: speed, k, nearest counted object ahead and its 4 s motion.
        -> $W/rows.parquet, $W/rows.json
  $TR lat_diag.py replay --tags T .. (GPU pool job, op-train) open-loop plans of each checkpoint on the stored native-gap dumps (arena held
        windows, RSG and JTN-L18-s1 closed loops: the same frames, ego row and route memory for every checkpoint; sh30_core.Core, imported
        unchanged) -> $W/replay.parquet (one row per dump x tag: plan poses (8, 3) rear axle, row frame)
  $AP lat_diag.py held      (CPU, AlpaSim venv) held 460 @18 zero scenes of NTVX / NTV0 / RSG / JTN with failure type and, from the
        driver's anchors (2 Hz) against the route of rf_follow (its own functions), the lateral offset at the zero event and whether the
        route turns near it -> $W/held.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "scripts"), str(REPO / "experiments/body1/lib"), str(REPO / "experiments/op_parity/scripts"),
                 str(REPO / "experiments/op_adapt_r2/lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/practice_nuplan/lat_diag"
POOL = DATA / "runs/op_parity/cache/on1_JTX"
ROWS = DATA / "runs/onpol_nuplan/rows"
WFILE = DATA / "runs/defect_fix/w/JTX00S.npz"
PARTS = dict(rb0="L18RB0", rtb1="L18RTB1", ys="L18YS", y="L18Y", **{f"pn{k}": f"PN{k}" for k in range(6)})
WEIGHT = 3.0 * 10.0                                  # ho_w x agent_lam = ho_w x road_lam (NTVX / NTV0 command)


def cmd_rows(a):
    import pandas as pd
    import torch
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    from loss43 import shape_only
    from jevdrive.run import Run
    torch.set_num_threads(a.threads)
    W.mkdir(parents=True, exist_ok=True)
    with Run("practice_nuplan", "lat-diag-rows", config=vars(a)) as run:
        tab = np.load(POOL / "tab.npz")
        sel = np.flatnonzero(tab["mode"].astype(str) == "self")
        if a.limit:
            sel = sel[np.linspace(0, len(sel) - 1, a.limit).astype(int)]
        names, pool = tab["names"][sel], tab["pool"][sel].astype(str)
        wz = np.load(WFILE)
        assert np.array_equal(wz["names"], tab["names"])
        grp = wz["grp"][sel].astype(str)
        # per-row context and paths from the rows dirs (meta.npz), joined by name
        plan = np.full((len(sel), 8, 3), np.nan, np.float32)
        logf = np.full((len(sel), 8, 3), np.nan, np.float32)
        k_ = np.full(len(sel), -1)
        for p, rd in PARTS.items():
            m = np.load(ROWS / rd / "meta.npz", allow_pickle=True)
            pos = {n: i for i, n in enumerate(m["names"].tolist())}
            idx = np.flatnonzero(pool == p)
            j = np.array([pos[n.replace(f"on:{p}:", "on:", 1)] for n in names[idx].tolist()])
            plan[idx], logf[idx], k_[idx] = m["poses"][j], m["logfut"][j, 1:], m["k"][j]
            run.info(f"{p}: {len(idx)} rows joined")
        assert (k_ >= 0).all()
        fut = tab["fut"][sel].astype(np.float32)
        dev = torch.device("cpu")
        rows_all = torch.arange(len(sel))
        ah = AgentHinge(POOL / "agent.npz", names, dev, 0.3, 0.0)
        road = Hinge(POOL / "road.npz", names, dev, 0.3, bank=True)
        z = torch.zeros(len(sel), 8)
        off0 = torch.cat([road.margins(z[i:i + 8192], z[i:i + 8192], z[i:i + 8192], rows_all[i:i + 8192]).amin(1) < 0
                          for i in range(0, len(sel), 8192)]) & road.ok
        fb = torch.nonzero(off0)[:, 0]
        if len(fb):                                                              # on_train: rows off the road raster at t0 -> drivable.npz
            dr = Hinge(POOL / "drivable.npz", names, dev, 0.3, bank=True)
            road.sdf[fb], road.ok[fb] = dr.sdf[fb], dr.ok[fb]
            del dr
        run.info(f"labels: agent ok {float(ah.ok.float().mean()):.4f}, road ok {float(road.ok.float().mean()):.4f}, off the road raster at t0 {len(fb)}")
        out = dict(name=names, pool=pool, new=np.char.startswith(pool.astype(str), "pn"), grp=grp, k=k_, speed=tab["speed"][sel],
                   off_t0=off0.numpy())

        def score(P, key, rows_mask=None):
            av, rv, ga, gr, gs = (np.zeros(len(sel), np.float32) for _ in range(5))
            ok = np.isfinite(P).all((1, 2)) if rows_mask is None else rows_mask & np.isfinite(P).all((1, 2))
            for i in range(0, len(sel), a.bs):
                r = np.arange(i, min(i + a.bs, len(sel)))
                r = r[ok[r]]
                if not len(r):
                    continue
                rt = torch.as_tensor(r)
                T = torch.as_tensor(P[r]).double().float()
                x, y, psi = (T[..., j].clone().requires_grad_(True) for j in range(3))
                xs, ys = shape_only(x, y, psi)
                aok, rok = ah.ok[rt], road.ok[rt]
                va = ah.per_step(xs, ys, psi, rt).mean(1) * aok
                vr = torch.relu(0.3 - road.margins(xs, ys, psi, rt)).mean(1) * rok
                g1 = torch.autograd.grad(va.sum() * WEIGHT, (x, y), retain_graph=True)
                g2 = torch.autograd.grad(vr.sum() * WEIGHT, (x, y))
                n = torch.stack([-torch.sin(psi.detach()), torch.cos(psi.detach())], -1)          # left normal of each pose
                la = -(g1[0] * n[..., 0] + g1[1] * n[..., 1])                                     # descent direction, + = push left
                lr = -(g2[0] * n[..., 0] + g2[1] * n[..., 1])
                av[r], rv[r] = va.detach().numpy(), vr.detach().numpy()
                ga[r], gr[r] = la.norm(dim=1).numpy(), lr.norm(dim=1).numpy()
                gs[r] = (la + lr).sum(1).numpy()
            out.update({f"{key}_agent": av, f"{key}_road": rv, f"{key}_gagent": ga, f"{key}_groad": gr, f"{key}_gleft": gs, f"{key}_ok": ok})

        score(plan, "plan")
        score(fut, "fut")
        score(logf, "log", k_ <= 1)
        # context: the nearest counted object ahead at t0 of the stored plan's sweep (distance, in corridor) and its motion over 4 s
        dmin, inc, mv = (np.full(len(sel), np.nan, np.float32) for _ in range(3))
        with torch.no_grad():
            for i in range(0, len(sel), a.bs):
                r = torch.arange(i, min(i + a.bs, len(sel)))
                T = torch.as_tensor(np.nan_to_num(plan[r.numpy()]))
                d, cnt, cor = ah.distances(T[..., 0], T[..., 1], T[..., 2], r)
                dd = d.masked_fill(~cnt, 1e3)
                j = dd.amin(1).argmin(1)                                                        # the object closest over the 41 steps
                b = ah.box[r, :, :, :2]
                bj = b[torch.arange(len(r)), :, j]                                               # (B, 9, 2)
                vj = ah.val[r][torch.arange(len(r)), :, j]
                dmin[r.numpy()] = dd.amin((1, 2)).numpy()
                inc[r.numpy()] = cor.any(1)[torch.arange(len(r)), j].float().numpy()
                mv[r.numpy()] = np.where((vj[:, 0] & vj[:, 8]).numpy(), (bj[:, 8] - bj[:, 0]).norm(dim=1).numpy(), np.nan)
        out |= dict(obj_d=dmin, obj_corridor=inc, obj_move4=mv,
                    plan_arc4=np.hypot(plan[:, -1, 0], plan[:, -1, 1]), fut_arc4=np.hypot(fut[:, -1, 0], fut[:, -1, 1]),
                    plan_y4=plan[:, -1, 1], fut_y4=fut[:, -1, 1], fut_dpsi4=fut[:, -1, 2])
        df = pd.DataFrame(out)
        df.to_parquet(W / ("rows.parquet" if not a.limit else "rows_lim.parquet"))
        run.summary.update(rows=len(df), off_t0=int(len(fb)))
        run.info(f"-> {W}/rows.parquet ({len(df)} rows)")


def cmd_replay(a):
    import glob
    import pandas as pd
    import torch
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib"), str(REPO / "experiments/alpasim/scripts")]
    import sh30_core as SC
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.run import Run
    NG = DATA / "runs/defect_fix/native_gap/dump"
    W.mkdir(parents=True, exist_ok=True)
    with Run("practice_nuplan", "lat-diag-replay", config=vars(a)) as run:
        recs = []
        for src in a.src:
            for f in sorted(glob.glob(str(NG / src / "c*/run/driver-logs/drive.jsonl"))):
                sess = {}
                for line in open(f):
                    r = json.loads(line)
                    if r["kind"] == "start":
                        sess[r["session"]] = (r["n"], Path(f).parent / "dump")
                    elif r["kind"] == "drive":
                        n, d = sess[r["session"]]
                        recs.append((src, r["scene"], int(r["k"]), d / f"s{n:02d}_k{r['k']}.npz", r.get("rmem")))
        recs = [x for x in recs if x[3].exists()]
        if a.limit:
            recs = recs[:a.limit]
        run.info(f"{len(recs)} dumped decisions from {a.src}")
        cores = {t: SC.Core(t, "cuda", "backwarp") for t in a.tags}
        out = []
        for i, (src, sc, k, f, rmem) in enumerate(recs):
            d = np.load(f)
            ego, cur, valid = d["ego"].astype(np.float32), d["cur"], d["valid"]
            prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
            row = dict(src=src, id=sc, k=k, v=float(np.hypot(ego[4], ego[5]) * 10), served=np.asarray(d["poses"])[:, :3].astype(np.float32).ravel())
            with torch.no_grad():
                for t, core in cores.items():
                    pp, cu = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
                    Hf = core.model.net.run_batched(core.A.vision_feeds(pp, cu), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *core.A.H_SHAPE)
                    r_ = np.r_[ego, np.asarray(rmem, np.float32)] if core.rm and rmem is not None else ego
                    o = core.model(Hf, torch.from_numpy(r_[None].astype(np.float32)).to(core.dev), torch.tensor([[1.0, 0.0]], device=core.dev))
                    o = (o[0] if isinstance(o, tuple) else o).float()
                    mu = o[0, core.pi].reshape(33, 15).cpu().numpy()
                    row[t] = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, d["cam_t"][:2].astype(np.float64), Z.T_OUT, "lever")[:, :3].astype(np.float32).ravel()
            out.append(row)
            if i % 500 == 0:
                run.info(f"{i} / {len(recs)}")
        df = pd.DataFrame(out)
        df.to_parquet(W / "replay.parquet")
        gate = {t: float(np.abs(np.stack(df[t]).reshape(-1, 8, 3)[..., :2] - np.stack(df.served).reshape(-1, 8, 3)[..., :2]).max(axis=(1, 2)).mean())
                for t in a.tags}
        run.summary.update(decisions=len(df), served_gap=gate)
        print(json.dumps(dict(decisions=len(df), mean_max_gap_to_served=gate)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rows")
    p.add_argument("--limit", type=int, default=0), p.add_argument("--bs", type=int, default=4096), p.add_argument("--threads", type=int, default=16)
    p = sp.add_parser("replay")
    p.add_argument("--tags", nargs="+", default=["NTV0-L18-s0", "NTVX-L18-s0", "RSG-L18-s0", "JTN-L18-s1"])
    p.add_argument("--src", nargs="+", default=["JTN1", "RSG"]), p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dict(rows=cmd_rows, replay=cmd_replay)[a.cmd](a)
