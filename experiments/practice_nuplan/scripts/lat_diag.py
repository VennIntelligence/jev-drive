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
  closed-loop paths: experiments/route_follow/scripts/rf_follow.py (unchanged, AlpaSim venv, --cover none) on the stored held 460 @18 reads
        -> $W/follow (NTVX, NTV0, RSG, JTN-s0 @18) and $W/follow1k (JTX1K, JTNC1K, NTV0-s1, JTN-s1, RSG-s1 @18): per decision the anchor's
        signed lateral offset to AlpaSim's own route (left +), the first zero-score event k_z, turns >= 60 deg.
  $TR lat_diag.py trainee --ckpt TAG   (GPU pool job, op-train, training-sized host memory) the trainer's own setup for NTVX's command
        (rm_train.py executed unedited up to its dispatch, then its build() with the checkpoint TAG loaded as `rely` does): TAG's eval-mode
        plans on every S row of on1_JTX and, on them, the pool terms exactly as the loss sees them (agent and road hinge shape-only x ho_w x
        lam, the nr1_lat terms x ho_w) with the signed sideways push of each term per row -> $W/trainee_<TAG>.parquet
  $TR lat_diag.py wns       (CPU) prereg supplement 2's draw JTX00SNS = JTX00S with the new start-state S rows (pool pn*) at tab speed
        < 0.5 m/s set to weight 0, the rest renormalised -> runs/defect_fix/w/JTX00SNS.npz (+ .json)
  $TR lat_diag.py confirm   (CPU) supplement 2's readout on $W/replay_ld.parquet -> results/lateral_confirm.{json,md}
  $TR lat_diag.py report    (CPU) the tables of results/lateral_diag.md from rows.parquet, follow*/ and replay.parquet -> stdout (markdown)
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
        for P in (plan, logf, fut):                                              # meta logfut keeps the log's unwrapped yaw (2 pi offsets)
            P[..., 2] = (P[..., 2] + np.pi) % (2 * np.pi) - np.pi
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
            av, rv, ga, gr, gs, gsa = (np.zeros(len(sel), np.float32) for _ in range(6))
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
                gs[r], gsa[r] = (la + lr).sum(1).numpy(), la.sum(1).numpy()
            out.update({f"{key}_agent": av, f"{key}_road": rv, f"{key}_gagent": ga, f"{key}_groad": gr, f"{key}_gleft": gs, f"{key}_gleft_agent": gsa,
                        f"{key}_ok": ok})

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
        df.to_parquet(W / a.out)
        gate = {t: float(np.abs(np.stack(df[t]).reshape(-1, 8, 3)[..., :2] - np.stack(df.served).reshape(-1, 8, 3)[..., :2]).max(axis=(1, 2)).mean())
                for t in a.tags}
        run.summary.update(decisions=len(df), served_gap=gate)
        print(json.dumps(dict(decisions=len(df), mean_max_gap_to_served=gate)))


def cmd_trainee(a):
    """The trainee's view of the S rows (docstring): rm_train's main block up to the dispatch, its build(), TAG's weights."""
    import pandas as pd
    import torch
    line = next(x for x in (DATA / "runs/practice_nuplan/ntvx/stages.txt").read_text().splitlines() if " train NTVX-L18-s0: " in x)
    argv = line.split("rm_train.py train ", 1)[1].split()
    i = argv.index("--tag")
    argv[i + 1] = a.ckpt
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    sys.argv = [str(rt), "rely", *argv]
    exec(compile(src.replace(tail, "_LATDIAG_ARGS = a"), str(rt), "exec"), g)
    ra = g["_LATDIAG_ARGS"]
    ra.rm_from = a.ckpt                                                          # cmd_rely: the checkpoint with its branch
    dev = torch.device("cuda")
    ns, *_ = g["build"](ra, dev)
    LS = ns.losses(ns.mode)
    import pp_train as T
    from loss43 import shape_only
    S, model = ns.S, ns.model.eval()
    pt = np.load(POOL / "tab.npz")
    assert len(pt["names"]) == ns.n_on and np.array_equal(S.tab["names"][ns.on0:], pt["names"]), "pool rows are not on1_JTX's"
    sel = np.flatnonzero(pt["mode"].astype(str) == "self")
    pi_ = torch.as_tensor(S.pi, device=dev)
    wA, wR, wL = ra.ho_w * ra.agent_lam, ra.ho_w * ra.road_lam, ra.ho_w * ra.on_lon_lam
    cols = {}
    for j in range(0, len(sel), a.bs):
        r = torch.as_tensor(ns.on0 + sel[j:j + a.bs], device=dev)
        with torch.no_grad():
            o = model(S.front[r.cpu().numpy()], S.ego[r], S.tc[r], None, None, nv=None if ns.nv is None else ns.nv[r]).float()
            x0, y0, p0 = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[r], LS.W)
        x, y, psi = x0.detach().requires_grad_(True), y0.detach().requires_grad_(True), p0.detach()
        xs, ys = shape_only(x, y, psi) if LS.shape else (x, y)
        terms = {}
        ok = LS.on_aok[r]
        terms["agent"] = wA * LS.agent2.per_step(xs, ys, psi, r).mean(1) * ok
        ko = LS.road.ok[r]
        terms["road"] = wR * torch.relu(LS.road.margin - LS.road.margins(xs, ys, psi, r)).mean(1) * ko
        if LS.lon is not None:
            for q, v in LS.lon.loss(torch.stack([x, y, psi], -1), r - LS.on0, LS.pack).items():
                terms[q] = wL * v
        n = torch.stack([-torch.sin(psi), torch.cos(psi)], -1)
        rec = dict(arc4=torch.hypot(x0[:, -1], y0[:, -1]), y4=y0[:, -1], y2=y0[:, 3])
        for q, v in terms.items():
            gx, gy = torch.autograd.grad(v.sum(), (x, y), retain_graph=True, allow_unused=True)
            gx, gy = (torch.zeros_like(x) if t is None else t for t in (gx, gy))
            lat = -(gx * n[..., 0] + gy * n[..., 1])                                       # descent direction, + = left
            rec[q] = v.detach()
            rec[f"{q}_left"] = lat.sum(1)
            rec[f"{q}_abs"] = lat.abs().sum(1)
        for q, v in rec.items():
            cols.setdefault(q, []).append(v.detach().cpu().numpy())
    df = pd.DataFrame({q: np.concatenate(v) for q, v in cols.items()})
    df.insert(0, "name", pt["names"][sel]), df.insert(1, "pool", pt["pool"][sel].astype(str)), df.insert(2, "speed", pt["speed"][sel])
    W.mkdir(parents=True, exist_ok=True)
    df.to_parquet(W / f"trainee_{a.ckpt}.parquet")
    print(json.dumps(dict(ckpt=a.ckpt, rows=len(df), weights=dict(agent=wA, road=wR, lat=wL), terms=[q for q in cols if not q.endswith(("_left", "_abs"))])))


def contexts(R):
    """Decision contexts of the replay rows (native-gap decision table of the dumped closed loop): standing, slow lead <= 15 m, turning, moving."""
    import pandas as pd
    A_ = pd.read_csv(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").set_index("id")
    R = R.assign(stratum=R.id.map(A_.stratum), log=R.id.map(A_.log))
    ng = pd.read_parquet(DATA / "runs/defect_fix/native_gap/dec.parquet")
    ng = ng.assign(src=ng.tag)[["src", "id", "k", "lead_gap", "lead_v", "turn20"]]
    R = R.merge(ng, on=["src", "id", "k"], how="left")
    R["ctx"] = np.select([R.v < 0.5, (R.lead_gap < 15) & (R.lead_v < 1.0), R.turn20.abs() > 20], ["standing", "slow lead <= 15 m", "turning"], "moving")
    return R


def cmd_wns(a):
    z, tb = np.load(WFILE), np.load(POOL / "tab.npz")
    assert np.array_equal(z["names"], tb["names"])
    w, grp = z["w"].astype(np.float64).copy(), z["grp"].astype("U12").copy()
    drop = np.char.startswith(tb["pool"].astype(str), "pn") & (tb["speed"] < 0.5) & (grp == "S")
    s0 = w[grp == "S"].sum()
    w[drop], grp[drop] = 0.0, "S_static0"
    w[grp == "S"] *= s0 / w[grp == "S"].sum()
    w /= w.sum()
    out = DATA / "runs/defect_fix/w/JTX00SNS.npz"
    assert not out.exists(), f"{out} exists"
    np.savez(out, names=z["names"], w=w, grp=grp)
    rec = dict(src=str(WFILE), dropped=int(drop.sum()), s_rows_left=int((grp == "S").sum()), held=int((grp == "S_held").sum()),
               share={g: float(w[grp == g].sum()) for g in np.unique(grp)})
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


def cmd_confirm(a):
    import pandas as pd
    R = contexts(pd.read_parquet(W / "replay_ld.parquet"))
    Y = {t: np.stack(R[t]).reshape(-1, 8, 3) for t in ("LDC-L18-s0", "LDX-L18-s0", "LDXS-L18-s0")}
    out, L = {}, ["# LATERAL-DIAG confirmatory run (PN prereg supplement 2)", "",
                  "Open-loop plans on the 5 400 arena held dumps; y = plan lateral (left +), arm minus LDC on the same decision, log-clustered 95 % CI.", "",
                  "| arm - LDC | context | decisions | y(2 s) | y(4 s) |", "|---|---|--:|--:|--:|"]
    for t in ("LDX-L18-s0", "LDXS-L18-s0"):
        for c in ("moving", "turning", "slow lead <= 15 m", "standing"):
            q = ((R.k >= 2) & (R.ctx == c)).to_numpy()
            d2, d4 = _boot(Y[t][q, 3, 1] - Y["LDC-L18-s0"][q, 3, 1], R.log[q]), _boot(Y[t][q, 7, 1] - Y["LDC-L18-s0"][q, 7, 1], R.log[q])
            out[f"{t}|{c}"] = dict(n=int(q.sum()), y2=d2, y4=d4)
            L.append(f"| {t} | {c} | {q.sum()} | {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}] | {d4[0]:+.4f} [{d4[1]:+.4f}, {d4[2]:+.4f}] |")
    x, xs = out["LDX-L18-s0|moving"]["y4"], out["LDXS-L18-s0|moving"]["y4"]
    rep = x[0] <= -0.010 and x[2] < 0
    if not rep:
        verdict = "not decidable: the 1 000-step continuation does not reproduce the right bias"
    elif (xs[1] <= 0 <= xs[2]) or abs(xs[0]) <= 0.5 * abs(x[0]):
        verdict = "the new static rows are the cause"
    elif xs[0] <= -0.010 and xs[2] < 0 and abs(xs[0]) > 0.5 * abs(x[0]):
        verdict = "not the static rows (new moving rows / dilution)"
    else:
        verdict = "not decidable by the registered rule"
    L += ["", f"Registered rule (moving, y(4 s)): LDX - LDC {x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}], LDXS - LDC {xs[0]:+.4f} [{xs[1]:+.4f}, {xs[2]:+.4f}] "
          f"-> **{verdict}**."]
    res = REPO / "experiments/practice_nuplan/results"
    (res / "lateral_confirm.json").write_text(json.dumps(dict(cells=out, verdict=verdict), indent=1))
    (res / "lateral_confirm.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _boot(x, g, n=2000, seed=0):
    """mean of x and its 95 % CI, bootstrap clustered by g (log)."""
    L, inv = np.unique(np.asarray(g), return_inverse=True)
    rng, x = np.random.default_rng(seed), np.asarray(x, float)
    b = [(w * x).sum() / w.sum() for w in (np.bincount(rng.integers(0, len(L), len(L)), minlength=len(L))[inv] for _ in range(n))]
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def cmd_report(a):
    import pandas as pd
    f3 = lambda m, lo, hi: f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"  # noqa: E731
    # ---- 1. the S rows the two draws see (stored plans of the rolling policies)
    d = pd.read_parquet(W / "rows.parquet")
    d = d[d.grp == "S"].assign(static=lambda x: x.speed < 0.5)
    d["cls"] = np.where(d.new, "new", "old") + np.where(d.static, " static", " moving")
    d["gl_road"] = d.plan_gleft - d.plan_gleft_agent
    print("## 1. S rows (trainable, JTX00S): hinges on the stored own plan, the imitation target and the log's own future\n")
    print("| rows | n | share of NTVX's S draw | plan 4 s arc (med, m) | human 4 s arc (med) | agent hinge active: plan / fut / log (k <= 1) | "
          "road hinge active: plan / fut / log | lateral push per row, signed (+ left): agent / road | |push| agent / road |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for c, g in [*d.groupby("cls"), ("NTV0 draw (old)", d[~d.new]), ("NTVX draw (all)", d)]:
        lg = g[g.log_ok]
        print(f"| {c} | {len(g)} | {len(g) / len(d):.3f} | {g.plan_arc4.median():.2f} | {g.fut_arc4.median():.2f} | "
              f"{(g.plan_agent > 0).mean():.3f} / {(g.fut_agent > 0).mean():.3f} / {(lg.log_agent > 0).mean():.4f} | "
              f"{(g.plan_road > 0).mean():.3f} / {(g.fut_road > 0).mean():.3f} / {(lg.log_road > 0).mean():.3f} | "
              f"{g.plan_gleft_agent.mean():+.4f} / {g.gl_road.mean():+.4f} | {g.plan_gagent.mean():.3f} / {g.plan_groad.mean():.3f} |")
    a_ = d[(d.plan_agent > 0)]
    for c, g in a_.groupby("new"):
        print(f"\nagent-active rows, {'new' if c else 'old'}: {len(g)}; ego static {g.static.mean():.3f}; object in the corridor {g.obj_corridor.mean():.3f}; "
              f"object moves < 0.5 m in 4 s {(g.obj_move4 < 0.5).mean():.3f}; imitation target also active {(g.fut_agent > 0).mean():.3f}")
    print(f"\nrows off the road raster at t0 (drivable fallback): old {d[~d.new].off_t0.mean():.4f}, new {d[d.new].off_t0.mean():.4f}")
    # ---- 1b. the trainee's own plans (lat_diag.py trainee): the pool terms as the loss sees them
    wz = np.load(WFILE)
    gmap = dict(zip(wz["names"].tolist(), wz["grp"].astype(str).tolist()))
    for tag in ("RB0-F-s0", "NTV0-L18-s0", "NTVX-L18-s0"):
        f = W / f"trainee_{tag}.parquet"
        if not f.exists():
            continue
        t = pd.read_parquet(f)
        t = t[t.name.map(gmap) == "S"]
        t["cls"] = np.where(t.pool.str.startswith("pn"), "new", "old") + np.where(t.speed < 0.5, " static", " moving")
        terms = [c[:-5] for c in t.columns if c.endswith("_left")]
        print(f"\n### Trainee plans of {tag} on the trainable S rows: signed sideways push per row (+ left) of each pool term, x its loss weight\n")
        print("| rows | n | share of NTVX's draw | plan 4 s arc (med) | agent active | " + " | ".join(f"push {q}" for q in terms) + " | push, all terms | |push|, all |")
        print("|---|--:|--:|--:|--:|" + "--:|" * (len(terms) + 2))
        for c, g in [*t.groupby("cls"), ("NTV0 draw (old)", t[~t.pool.str.startswith("pn")]), ("NTVX draw (all)", t)]:
            tot, ab = sum(g[f"{q}_left"] for q in terms), sum(g[f"{q}_abs"] for q in terms)
            print(f"| {c} | {len(g)} | {len(g) / len(t):.3f} | {g.arc4.median():.2f} | {(g.agent > 0).mean():.3f} | "
                  + " | ".join(f"{g[f'{q}_left'].mean():+.4f}" for q in terms) + f" | {tot.mean():+.4f} | {ab.mean():.3f} |")
    # ---- 2. closed-loop lateral drift on held 460 @18
    D = pd.concat([pd.read_parquet(W / f / "decisions.parquet") for f in ("follow", "follow1k")])
    S = pd.concat([pd.read_csv(W / f / "scenes.csv") for f in ("follow", "follow1k")])
    P = D.pivot_table(index=["scene", "k"], columns="label", values="lat").reset_index()
    KZ = S.pivot_table(index="scene", columns="label", values="kz")
    turn = set(S[S.turn == True].scene)  # noqa: E712
    print("\n## 2. Held 460 @18 closed loop: signed lateral offset to the route (left +), A - B on the same (scene, decision), "
          "both before their first zero event; log-clustered 95 % CI\n")
    print("| A - B | non-turn, k 5-10 | non-turn, k 11-17 | turn scenes, k 11-17 |")
    print("|---|--:|--:|--:|")
    for A, B in [("NTVX@18", "NTV0@18"), ("NTV0s1@18", "NTV0@18"), ("JTNs1@18", "JTN@18"), ("RSGs1@18", "RSG@18"), ("NTV0@18", "RSG@18"),
                 ("NTV0s1@18", "RSG@18"), ("JTN@18", "RSG@18"), ("NTVX@18", "RSG@18"), ("JTX1K@18", "JTNC1K@18")]:
        m = P.merge(KZ[[A, B]].rename(columns={A: "za", B: "zb"}).reset_index(), on="scene")
        m = m[((m.za < 0) | (m.k < m.za)) & ((m.zb < 0) | (m.k < m.zb))].dropna(subset=[A, B])
        cells = []
        for ks, tr in ((range(5, 11), False), (range(11, 18), False), (range(11, 18), True)):
            q = m[m.k.isin(ks) & (m.scene.isin(turn) == tr)]
            cells.append(f3(*_boot(q[A] - q[B], q.scene.str.rsplit("-", n=1).str[0])) + f" ({len(q)})")
        print(f"| {A} - {B} | " + " | ".join(cells) + " |")
    # ---- 3. open-loop replay on the arena held dumps
    if (W / "replay.parquet").exists():
        R = pd.read_parquet(W / "replay.parquet")
        tags = [c for c in R.columns if c.endswith(("-s0", "-s1"))]
        Y = {t: np.stack(R[t]).reshape(-1, 8, 3) for t in tags}
        A_ = pd.read_csv(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").set_index("id")
        R["stratum"] = R.id.map(A_.stratum)
        R["log"] = R.id.map(A_.log)
        ng = pd.read_parquet(DATA / "runs/defect_fix/native_gap/dec.parquet")
        ng = ng.assign(src=ng.tag)[["src", "id", "k", "lead_gap", "lead_v", "turn20"]]
        R = R.merge(ng, on=["src", "id", "k"], how="left")
        R["ctx"] = np.select([R.v < 0.5, (R.lead_gap < 15) & (R.lead_v < 1.0), R.turn20.abs() > 20], ["standing", "slow lead <= 15 m", "turning"], "moving")
        print("\n## 3. Open loop on the same inputs (arena held dumps of RSG and JTN-L18-s1, k >= 2): plan lateral y at 2 s / 4 s (left +), A - B\n")
        print("| A - B | context | decisions | y(2 s) [CI] | y(4 s) [CI] | |y(4 s) diff| mean |")
        print("|---|---|--:|--:|--:|--:|")
        for A, B in [("NTVX-L18-s0", "NTV0-L18-s0"), ("NTV0-L18-s0", "RSG-L18-s0"), ("JTN-L18-s1", "RSG-L18-s0"), ("JTX1K-L18-s0", "JTNC1K-L18-s0")]:
            if A not in Y or B not in Y:
                continue
            for c in ["all", "moving", "turning", "slow lead <= 15 m", "standing"]:
                q = (R.k >= 2) & ((R.ctx == c) if c != "all" else True)
                q = q.to_numpy()
                d2, d4 = Y[A][q, 3, 1] - Y[B][q, 3, 1], Y[A][q, 7, 1] - Y[B][q, 7, 1]
                print(f"| {A} - {B} | {c} | {q.sum()} | {f3(*_boot(d2, R.log[q]))} | {f3(*_boot(d4, R.log[q]))} | {np.abs(d4).mean():.3f} |")
        # restoring gain: the plan's lateral y(4 s) against the ego's own signed offset to the human track (left +), moving non-turn decisions
        import pickle
        hum = pickle.load(open(DATA / "runs/defect_fix/native_gap/hum.pkl", "rb"))
        xy = pd.read_parquet(DATA / "runs/defect_fix/native_gap/dec.parquet", columns=["tag", "id", "k", "x", "y", "yaw"]).rename(columns={"tag": "src"})
        R2 = R.merge(xy, on=["src", "id", "k"], how="left")

        def slat(i, x, y, yaw):
            if i not in hum or not np.isfinite(x):
                return np.nan, np.nan
            P = np.asarray(hum[i]["P"], float)
            q, a_, b_ = np.array([x, y]), P[:-1], P[1:]
            ab = b_ - a_
            L2 = np.maximum((ab ** 2).sum(1), 1e-9)
            u = np.clip(((q - a_) * ab).sum(1) / L2, 0, 1)
            j = np.hypot(*(a_ + u[:, None] * ab - q).T).argmin()
            t = ab[j] / np.sqrt(L2[j])
            v = q - a_[j]
            return float(t[0] * v[1] - t[1] * v[0]), float(np.arctan2(np.sin(yaw - np.arctan2(t[1], t[0])), np.cos(yaw - np.arctan2(t[1], t[0]))))
        e = np.array([slat(i, x, y, w) for i, x, y, w in zip(R2.id, R2.x, R2.y, R2.yaw)])
        R2["e"], R2["epsi"] = e[:, 0], e[:, 1]
        q = ((R2.k >= 2) & (R2.ctx == "moving") & R2.e.notna() & (R2.e.abs() < 1.5)).to_numpy()
        print("\nRestoring gain (moving decisions, k >= 2, |offset| < 1.5 m): least squares y_plan(4 s) = b0 + b1 e + b2 psi_e, e = the ego's signed offset to "
              "the human track (left +), psi_e its heading error; b1 = -1 re-centres fully within 4 s\n")
        print("| tag | decisions | b0 (m) | b1 | b2 (m / rad) | mean y(4 s) at e in [-0.3, 0.3] m |")
        print("|---|--:|--:|--:|--:|--:|")
        X = np.c_[np.ones(q.sum()), R2.e[q], R2.epsi[q]]
        for t in tags:
            b = np.linalg.lstsq(X, Y[t][q, 7, 1], rcond=None)[0]
            c0 = q & (R2.e.abs() < 0.3).to_numpy()
            print(f"| {t} | {q.sum()} | {b[0]:+.3f} | {b[1]:+.3f} | {b[2]:+.2f} | {Y[t][c0, 7, 1].mean():+.3f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rows")
    p.add_argument("--limit", type=int, default=0), p.add_argument("--bs", type=int, default=4096), p.add_argument("--threads", type=int, default=16)
    p = sp.add_parser("replay")
    p.add_argument("--tags", nargs="+", default=["NTV0-L18-s0", "NTVX-L18-s0", "RSG-L18-s0", "JTN-L18-s1"])
    p.add_argument("--src", nargs="+", default=["JTN1", "RSG"]), p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="replay.parquet")
    sp.add_parser("report")
    sp.add_parser("wns"), sp.add_parser("confirm")
    p = sp.add_parser("trainee")
    p.add_argument("--ckpt", required=True), p.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    dict(rows=cmd_rows, replay=cmd_replay, report=cmd_report, trainee=cmd_trainee, wns=cmd_wns, confirm=cmd_confirm)[a.cmd](a)
