#!/usr/bin/env python3
"""ROAD-PROBE (plans/2026-10-12-road-probe-prereg.md): is the road geometry of nuPlan city scenes (ego-lane boundaries, road edges, lane
heading at 10 / 20 / 40 m, especially at junctions and bends) readable from the FROZEN Cinque vision tokens the driver already computes,
although the shipped lane-line / road-edge heads do not decode it (decision 269)? Capability probe, offline; nothing is fed to a policy.

Groups of decisions (row order fixed by each group's token file):
  train  the on-policy rows of onpol_long (on1_{L18Y,L18YS,L18P}: tokens already cached by on_rows.py tok), 1020 train scenes, 28 log units
  held   a fresh held read of YS-L18-s1 (rp_chain.sh roll, SH30_DUMP): every decision of the 460 held scenes
  zero   decision 269's 98 corridor + offroad zeros of YS-L18-s1 and YR10m10-F-s0: every decision of those rollouts, served tokens by replay

  tok     (envs/op-train, GPU)   held dumps -> tokens (pp_prep.encoder, on_rows make_front's pairing), P0 heads, gate G-tok
  replay  (envs/op-train, GPU)   zero rollouts: stored driver messages through the driver class (ev.py replay) -> served tokens, P0 heads, G-rep
  labels  (AlpaSim venv, CPU)    map / logged-path labels of every decision of a group (ev.py Geo, zm_read Lanes) -> labels/<group>.parquet
  train   (envs/op-train, GPU)   --arm P|blind|lin [--frac F] [--seed S]: the probe on the frozen tokens -> pred/<name>.npz
  read    (AlpaSim venv, CPU)    gates, registered tables, lines, verdict, learning curve -> $W/read/<ts>/ (copied to results/ by report)
  bev     (AlpaSim venv, CPU)    BEV payloads of picked decisions -> read dir
  report  (Mac)                  rp_report.py

Frame: rollout local frame; ego = rear axle, x forward, y LEFT (as ev.py / decision 269). A lateral target at distance d is the y where
something crosses the line x = cam_x + d.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("labels", "read", "bev"):    # one BLAS / polars thread per worker
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/eyes_vs_head/scripts"), str(REPO / "experiments/zeros_mech/scripts"),
                str(REPO / "experiments/corridor/scripts"), str(REPO / "experiments/op_parity/scripts")]
import ev as EV  # noqa: E402  (sets the AlpaSim / driver paths; its read helpers are used unedited)

DATA = EV.DATA
W = DATA / "runs/road_probe"
CR = DATA / "runs/op_parity/cache"
ROWS = DATA / "runs/onpol_nuplan/rows"
TRAIN_SETS = ("L18Y", "L18YS", "L18P")
HELD_TAG = "YS-L18-s1"
ZERO_LABELS = ("YS-L18-s1", "YR10m10-F-s0")
EVENTS = REPO / "experiments/eyes_vs_head/results/events_P0.csv"
DS = (10, 20, 40)
SIDES = ("L", "R", "EL", "ER")
OUTS = [f"{s}{d}" for d in DS for s in SIDES] + ["h20", "onc"]        # 12 lateral y (m), lane heading at 20 m (rad), oncoming logit
NREG = 13
LANE_W = (2.0, 6.0)
SCALE = EV.SCALE


def logu(scene):
    return "_".join(scene.split("_")[:2])                                 # c0b_report.log_of


def is_val(log):
    return int(hashlib.sha256(f"rpval|{log}".encode()).hexdigest(), 16) % 7 == 0


def frac_logs(logs, frac):
    lg = sorted(set(logs), key=lambda l: hashlib.sha256(f"rplc|{l}".encode()).hexdigest())
    return set(lg[: max(1, int(np.ceil(frac * len(lg))))])


def sessions(run):
    """drive.jsonl -> {scene: (session n, [drive records sorted by k])}, the last session of a scene (ao_e1.drive_records' rule)."""
    S = {}
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "start":
            S[r["session"]] = (r["scene"], r["n"], [])
        elif r["kind"] == "drive" and r["session"] in S:
            S[r["session"]][2].append(r)
    best = {}
    for sc, n, rr in S.values():
        if rr and (sc not in best or n > best[sc][0]):
            best[sc] = (n, sorted(rr, key=lambda r: r["k"]))
    return best


def zero_events():
    import csv
    return [r for r in csv.DictReader(open(EVENTS)) if r["group"] in ZERO_LABELS and r["type"] in ("corridor", "offroad")]


def p0_raw(out, sl):
    """Policy output rows (B, n_out) -> the raw head slices ev.heads reads."""
    return {k: out[:, sl[k]].cpu().numpy().astype(np.float32) for k in ("lane_lines", "lane_lines_prob", "road_edges")}


# ---------------------------------------------------------------- tok: held dumps -> tokens (GPU)
def tok(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    import pp_prep as PP
    import pp_train as T
    from jevdrive import navsim_zs as Z
    from jevdrive import op_adapt as A
    from jevdrive import op_interp as I
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = W / "tok/held"
    out.mkdir(parents=True, exist_ok=True)
    held = splits.load("alpasim_nuplan/onpol-held")
    with Run("road_probe", "tok", config=vars(a)) as run:
        run.use_split(held)
        items = []
        for rd in sorted(W.glob("roll/c*/run")):
            for scene, (n, rr) in sorted(sessions(rd).items()):
                assert scene in set(held.members), scene
                for r in rr:
                    items.append(dict(run=str(rd), scene=scene, k=int(r["k"]), t0=int(r["t0"]), anchor=np.asarray(r["anchor"], float),
                                      dump=str(rd / "driver-logs/dump" / f"s{n:02d}_k{r['k']}.npz")))
        if a.limit:
            keep = sorted({i["scene"] for i in items})[: a.limit]
            items = [i for i in items if i["scene"] in keep]
        N = len(items)
        run.info(f"{N} held decisions, {len({i['scene'] for i in items})} scenes")
        dev = torch.device("cuda")
        net, enc = PP.encoder(dev)
        front = np.lib.format.open_memmap(out / "front.npy.tmp", "w+", np.float16, (N, 8) + A.H_SHAPE)
        meta = {q: [] for q in ("ego", "cam", "poses", "mu", "valid")}

        def load(rows):
            zs = [np.load(items[i]["dump"]) for i in rows]
            for z, i in zip(zs, rows):
                assert str(z["scene"]) == items[i]["scene"], items[i]["dump"]
            cur = np.stack([z["cur"] for z in zs])
            m = [{q: z[q] for q in ("ego", "cam_t", "poses", "mu", "valid")} for z in zs]
            return rows, np.concatenate([np.zeros_like(cur[:, :1]), cur[:, :-1]], 1).reshape(-1, *cur.shape[2:]), cur.reshape(-1, *cur.shape[2:]), m
        chunks = [list(range(i, min(i + 32, N))) for i in range(0, N, 32)]
        with ThreadPoolExecutor(min(16, n_cpus())) as ex, torch.no_grad():
            for rows, prev, cur, m in run.tqdm(PP._bounded(ex, load, chunks, 8), total=len(chunks), desc="encode"):
                front[rows[0]:rows[-1] + 1] = enc(prev, cur).reshape(len(rows), 8, *A.H_SHAPE)
                for x in m:
                    for q, k in (("ego", "ego"), ("cam", "cam_t"), ("poses", "poses"), ("mu", "mu"), ("valid", "valid")):
                        meta[q].append(np.asarray(x[k]))
        front.flush()
        meta = {q: np.stack(v) for q, v in meta.items()}
        assert meta["valid"].all(), "a held decision with an invalid slot"
        # P0 heads and gate G-tok (the served checkpoint's plan from these tokens against the dumped served plan)
        p0, ms = T.load_pmodel("P0", dev), T.load_pmodel(HELD_TAG, dev)
        pi = A.plan_index(ms.net.slices)
        tc = torch.tensor([[1.0, 0.0]], device=dev)
        raw, d8 = {k: [] for k in ("lane_lines", "lane_lines_prob", "road_edges")}, []
        with torch.no_grad():
            for i in range(0, N, 256):
                r = slice(i, min(i + 256, N))
                H = torch.from_numpy(np.ascontiguousarray(front[r])).to(dev)
                e = torch.from_numpy(meta["ego"][r].astype(np.float32)).to(dev)
                t = tc.repeat(len(H), 1)
                for k, v in p0_raw(p0(H, e, t).float(), p0.net.slices).items():
                    raw[k].append(v)
                mu = ms(H, e, t).float()[:, pi].view(-1, 33, 15).cpu().numpy()
                ex_ = np.stack([I.to_rear(q[:, 0:3], q[:, 11], I.T_IDXS, c[:2], Z.T_OUT, "lever") for q, c in zip(mu, meta["cam"][r].astype(np.float64))])
                d8.append(np.abs(ex_[..., :2] - meta["poses"][r][..., :2]).max((1, 2)))
        d8 = np.concatenate(d8)
        os.replace(out / "front.npy.tmp", out / "front.npy")
        np.savez(out / "p0.npz", **{k: np.concatenate(v) for k, v in raw.items()})
        np.savez(out / "tab.npz", scene=np.array([i["scene"] for i in items]), k=np.array([i["k"] for i in items]), t0=np.array([i["t0"] for i in items]),
                 anchor=np.stack([i["anchor"] for i in items]), run=np.array([i["run"] for i in items]), **meta)
        g = dict(n=N, share=float((d8 <= 0.10).mean()), median=float(np.median(d8)), max=float(d8.max()))
        g["pass"] = bool(g["share"] >= 0.95)
        (out / "G_tok.json").write_text(json.dumps(g, indent=1))
        run.summary.update(G_tok=g)
        run.info(f"G-tok {json.dumps(g)}")


# ---------------------------------------------------------------- replay: zero rollouts -> served tokens (GPU)
def replay(a):
    import tempfile
    import torch
    import pp_train as T
    import sh30_driver as D
    from ao_e1 import drive_records
    from jevdrive.run import Run
    out = W / "tok/zero"
    out.mkdir(parents=True, exist_ok=True)
    ev = zero_events()
    with Run("road_probe", "replay", config=vars(a)) as run:
        p0 = T.load_pmodel("P0", torch.device("cuda"))
        pb, ctx = D.egodriver_pb2, EV.Ctx()
        V, tab, raw, grep = [], {q: [] for q in ("label", "scene", "k", "t0", "anchor", "cam", "ego", "poses", "valid")}, {k: [] for k in ("lane_lines", "lane_lines_prob", "road_edges")}, []
        for lab in ZERO_LABELS:
            scenes = sorted({e["scene"] for e in ev if e["group"] == lab})[: a.limit or None]
            core = D.C.Core(lab, "cuda", "backwarp", 1.0, "gpu")
            logd = Path(tempfile.mkdtemp())
            drv = D.Driver(core, logd, 0, False)
            cap = {}
            core.model.register_forward_hook(lambda m, args, o: cap.update(H=args[0], ego=args[1], tc=args[2]))
            recs = []
            for scene in run.tqdm(scenes, desc=lab):
                uuid = None
                for kind, rawm in pickle.load(open(EV.OUT / "msgs" / lab / f"{scene}.pkl", "rb")):
                    if kind == "driver_session_request":
                        req = pb.DriveSessionRequest.FromString(rawm)
                        uuid = req.session_uuid
                        drv.start_session(req, ctx)
                    elif kind == "driver_camera_image":
                        drv.submit_image_observation(pb.RolloutCameraImage.FromString(rawm), ctx)
                    elif kind == "driver_ego_trajectory":
                        drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(rawm), ctx)
                    elif kind == "route_request":
                        drv.submit_route(pb.RouteRequest.FromString(rawm), ctx)
                    elif kind == "driver_request":
                        req = pb.DriveRequest.FromString(rawm)
                        try:
                            drv.drive(req, ctx)
                        except RuntimeError as e:
                            run.info(f"{lab} {scene} t {req.time_now_us}: {e}")
                            continue
                        H = cap["H"].float()
                        v = np.zeros((8, 32, 512), np.float16)
                        v[8 - H.shape[1]:] = H[0].cpu().numpy().astype(np.float16)
                        with torch.no_grad():
                            o0 = p0(cap["H"], cap["ego"], cap["tc"]).float()
                        V.append(v)
                        for k, x in p0_raw(o0, p0.net.slices).items():
                            raw[k].append(x[0])
                        recs.append(dict(scene=scene, now=int(req.time_now_us), cam=np.asarray(drv.sessions[uuid].cam["t"], np.float64).copy(),
                                         ego=cap["ego"][0].float().cpu().numpy(), valid=np.arange(8) >= 8 - H.shape[1]))
                drv.sessions.pop(uuid, None)
            drv.out.flush()
            drives = [json.loads(x) for x in open(logd / "drive.jsonl") if '"kind": "drive"' in x]
            assert len(drives) == len(recs), f"{lab}: {len(drives)} driver records for {len(recs)} decisions"
            stored = drive_records(EV.RUNS[lab])
            for r, d in zip(recs, drives):
                assert int(d["now"]) == r["now"] and d["scene"] == r["scene"]
                for q, x in (("label", lab), ("scene", r["scene"]), ("k", int(d["k"])), ("t0", int(d["t0"])), ("anchor", np.asarray(d["anchor"], float)),
                             ("cam", r["cam"]), ("ego", r["ego"]), ("poses", np.asarray(d["poses"], np.float32)), ("valid", r["valid"])):
                    tab[q].append(x)
                s = {int(x["k"]): x for x in stored.get(r["scene"], [])}.get(int(d["k"]))
                grep.append(np.nan if s is None else float(np.hypot(*(np.asarray(s["poses"], float)[7, :2] - np.asarray(d["poses"], float)[7, :2]))))
            del core, drv
            torch.cuda.empty_cache()
        np.save(out / "front.npy", np.stack(V))
        np.savez(out / "p0.npz", **{k: np.stack(v) for k, v in raw.items()})
        np.savez(out / "tab.npz", **{q: np.stack(v) if q not in ("label", "scene") else np.array(v) for q, v in tab.items()})
        gr = np.array(grep)
        g = dict(n=len(gr), missing=int(np.isnan(gr).sum()), share=float(np.nanmean(gr <= 0.10)), median=float(np.nanmedian(gr)))
        g["share"] = float((gr[~np.isnan(gr)] <= 0.10).mean())
        g["pass"] = bool(g["share"] >= 0.95)
        (out / "G_rep.json").write_text(json.dumps(g, indent=1))
        run.summary.update(G_rep=g)
        run.info(f"G-rep {json.dumps(g)}")


# ---------------------------------------------------------------- labels (AlpaSim venv, CPU)
def group_rows(g):
    """-> list of dict(row, scene, k, t0, anchor, cam) of a group, in token order."""
    if g == "train":
        out, off = [], 0
        for s in TRAIN_SETS:
            tb, M = dict(np.load(CR / f"on1_{s}/tab.npz")), dict(np.load(ROWS / s / "meta.npz", allow_pickle=True))
            assert (M["names"] == tb["names"]).all()
            out += [dict(row=off + i, scene=str(M["scene"][i]), k=int(M["k"][i]), t0=int(M["t_us"][i]), anchor=M["anchor"][i].astype(float),
                         cam=tb["cam"][i].astype(float), src=s, tag=str(M["tag"][i])) for i in range(len(tb["names"]))]
            off += len(tb["names"])
        return out
    tb = dict(np.load(W / f"tok/{g}/tab.npz"))
    return [dict(row=i, scene=str(tb["scene"][i]), k=int(tb["k"][i]), t0=int(tb["t0"][i]), anchor=tb["anchor"][i].astype(float), cam=tb["cam"][i].astype(float),
                 src=g, tag=str(tb["label"][i]) if "label" in tb.files else HELD_TAG) for i in range(len(tb["k"]))]


def arc(P):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def proj_s(P, S, p):
    """Arc length of the nearest point of polyline P (arc S) to p."""
    d = P[1:] - P[:-1]
    t = np.clip(((p - P[:-1]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    q = P[:-1] + t[:, None] * d
    i = int(np.argmin(np.hypot(*(q - p).T)))
    return S[i] + t[i] * (S[i + 1] - S[i])


def path_at(P, S, Y, s):
    """Point and heading of the polyline at arc length s (clipped)."""
    s = float(np.clip(s, 0, S[-1]))
    return np.array([np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])]), float(np.interp(s, S, Y))


def lab_unit(job):
    """(group, scene, [rows]) -> label dicts."""
    g, scene, rows = job
    import zm_read as ZM
    from ao_e1 import loader
    ds = loader(str(EV.CFG)).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg3, Yg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.unwrap(np.asarray(tr.yaws, float))
    Pg = Pg3[:, :2]
    LP = np.vstack([Pg, Pg[-1] + np.outer(np.linspace(2.0, 60.0, 30), [np.cos(Yg[-1]), np.sin(Yg[-1])])])
    keep = np.r_[True, np.hypot(*np.diff(LP, axis=0).T) > 1e-3]
    LP = LP[keep]
    S = arc(LP)
    seg = np.unwrap(np.arctan2(*np.diff(LP, axis=0).T[::-1]))
    Y = np.r_[seg, seg[-1]]                                               # heading at each vertex (of the segment leaving it)
    A = np.array([r["anchor"] for r in rows])
    allp = np.vstack([Pg, A[:, :2]])
    G = EV.Geo(ds, (allp.min(0) - 90, allp.max(0) + 90))
    lanes = ZM.Lanes(ds.map)
    out = []
    for r in rows:
        a, cam = r["anchor"], r["cam"]
        z = float(np.interp(r["t0"], Tg, Pg3[:, 2]))
        rec = dict(row=r["row"], group=g, scene=scene, k=r["k"], src=r["src"], tag=r["tag"],
                   frame_err=float(np.hypot(*(a[:2] - [np.interp(r["t0"], Tg, Pg[:, 0]), np.interp(r["t0"], Tg, Pg[:, 1])]))),
                   e_log=EV.signed_off(LP, a[:2]), cam_x=cam[0], cam_y=cam[1])
        for d in DS:
            xs = cam[0] + d
            yl = EV.log_y(LP, a, xs)
            B, R = EV.cross(*G.SE, a, xs), EV.cross(*G.SR, a, xs)
            rec[f"ylog{d}"], rec[f"nB{d}"], rec[f"nR{d}"] = yl, len(B), len(R)
            L = Rr = EL = ER = np.nan
            if np.isfinite(yl):
                up, dn = B[B > yl], B[B < yl]
                if len(up) and len(dn) and LANE_W[0] <= up.min() - dn.max() <= LANE_W[1]:
                    L, Rr = float(up.min()), float(dn.max())
                EL = float(R[R > yl].min()) if (R > yl).any() else np.nan
                ER = float(R[R < yl].max()) if (R < yl).any() else np.nan
            rec[f"L{d}"], rec[f"R{d}"], rec[f"EL{d}"], rec[f"ER{d}"] = L, Rr, EL, ER
            if d == 20:
                rec["xB20"] = np.r_[B, np.full(max(0, 24 - len(B)), np.nan)][:24].astype(np.float32)        # crossings, for the nearest-boundary error
                rec["xR20"] = np.r_[R, np.full(max(0, 24 - len(R)), np.nan)][:24].astype(np.float32)
        # route: logged path heading change over 40 m from the ego's projection; junction distance along it
        s0 = proj_s(LP, S, a[:2])
        _, y0 = path_at(LP, S, Y, s0)
        _, y40 = path_at(LP, S, Y, s0 + 40.0)
        rec["turn40"] = float(np.degrees(abs(EV.wrap(y40 - y0))))
        rec["turn_sign"] = float(np.sign(EV.wrap(y40 - y0)))
        jd = np.nan
        for s in np.arange(0.0, 60.01, 2.0):
            p, _ = path_at(LP, S, Y, s0 + s)
            if G.junction(p[0], p[1]):
                jd = float(s)
                break
        rec["jdist"] = jd
        # lane-centre heading at 20 m: the lane containing the route point whose tangent is nearest the logged path's
        rec["h20"] = np.nan
        if np.isfinite(rec["ylog20"]) and G.tree is not None:
            q = EV.to_local(a, [[cam[0] + 20.0, rec["ylog20"]]])[0]
            _, ylq = path_at(LP, S, Y, proj_s(LP, S, q))
            best = None
            for i in G.tree.query(G.sh.Point(*q), predicate="within"):
                C = G.lanes[int(i)][1]
                j = min(int(np.argmin(np.hypot(*(C - q).T))), len(C) - 2)
                yw = float(np.arctan2(C[j + 1, 1] - C[j, 1], C[j + 1, 0] - C[j, 0]))
                dd = abs(EV.wrap(yw - ylq))
                if best is None or dd < best[0]:
                    best = (dd, yw)
            if best is not None:
                rec["h20"] = float(EV.wrap(best[1] - a[2]))
        c, junc = lanes.cls(a[0], a[1], z, a[2])
        rec["lane_cls"], rec["junc_now"] = int(c), bool(junc)
        out.append(rec)
    return out


def labels(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("road_probe", f"labels-{a.group}", config=vars(a)) as run:
        run.use_split(splits.load("alpasim_nuplan/onpol-train" if a.group == "train" else "alpasim_nuplan/onpol-held"))
        rows = group_rows(a.group)
        by = {}
        for r in rows:
            by.setdefault(r["scene"], []).append(r)
        jobs = [(a.group, s, by[s]) for s in sorted(by)][: a.limit or None]
        run.info(f"{a.group}: {len(rows)} decisions, {len(by)} scenes, {len(jobs)} units")
        res = par.pmap(lab_unit, jobs, workers=a.workers or n_cpus(), run=run, desc="scenes")
        D = pd.DataFrame([x for v in res.values if v for x in v]).sort_values("row").reset_index(drop=True)
        (W / "labels").mkdir(parents=True, exist_ok=True)
        f = W / "labels" / f"{a.group}{'-smoke' if a.limit else ''}.parquet"
        D.to_parquet(f, index=False)
        f0 = D[D.k == 0].groupby("scene").frame_err.first()
        g = dict(units=len(jobs), errors=len(res.errors), err_first=[res.errors[i].strip().splitlines()[-1][:300] for i in sorted(res.errors)[:5]],
                 rows=len(D), G_frame=dict(scenes=int(len(f0)), share=float((f0 <= 0.10).mean()), max=float(f0.max())))
        g["G_frame"]["pass"] = bool(g["G_frame"]["share"] >= 0.99)
        (W / "labels" / f"{f.stem}.gates.json").write_text(json.dumps(g, indent=1, default=float))
        run.summary.update(g)
        run.info(json.dumps(g, default=float))


# ---------------------------------------------------------------- train (envs/op-train, GPU)
def load_group(g, dev, n_threads=8):
    """-> (V (n, 8, 32, 512) fp16 on the device, ego (n, 20) fp32, labels DataFrame in row order)."""
    import pandas as pd
    import torch
    from concurrent.futures import ThreadPoolExecutor
    D = pd.read_parquet(W / "labels" / f"{g}.parquet")
    if g == "train":
        srcs = [(CR / f"on1_{s}@warp/front.npy", np.load(CR / f"on1_{s}/tab.npz")["ego"]) for s in TRAIN_SETS]
    else:
        srcs = [(W / f"tok/{g}/front.npy", np.load(W / f"tok/{g}/tab.npz")["ego"])]
    n = sum(len(e) for _, e in srcs)
    assert len(D) == n and (D.row.values == np.arange(n)).all(), f"{g}: labels do not cover the token rows"
    V = torch.empty((n, 8, 32, 512), dtype=torch.float16, device=dev)
    jobs, o = [], 0
    for f, e in srcs:
        jobs += [(f, c, min(c + 1024, len(e)), o + c) for c in range(0, len(e), 1024)]
        o += len(e)

    def one(j):
        f, c0, c1, oo = j
        return oo, np.ascontiguousarray(np.load(f, mmap_mode="r")[c0:c1])
    with ThreadPoolExecutor(n_threads) as ex:
        for oo, x in ex.map(one, jobs):
            V[oo:oo + len(x)] = torch.from_numpy(x).to(dev)
    return V, np.concatenate([e for _, e in srcs]).astype(np.float32), D


def targets(D):
    """-> Y (n, 14) with NaN where no label: 12 lateral y, h20, oncoming (1 / 0, NaN outside any lane)."""
    Y = np.stack([D[c].values.astype(np.float32) for c in OUTS[:NREG]], 1)
    onc = np.where(D.lane_cls.values == 0, np.nan, (D.lane_cls.values == 2).astype(np.float32))
    return np.c_[Y, onc].astype(np.float32)


def train(a):
    import time
    import torch
    import torch.nn.functional as F
    import head1_train as H1
    from jevdrive.data import splits
    from jevdrive.run import Run
    name = f"{a.arm}-p{int(round(a.frac * 1000)):04d}-s{a.seed}" + ("-smoke" if a.smoke else "")
    with Run("road_probe", f"train-{name}", seed=a.seed, config=vars(a)) as run:
        s_tr, s_he = splits.load("alpasim_nuplan/onpol-train"), splits.load("alpasim_nuplan/onpol-held")
        run.use_split(s_tr), run.use_split(s_he)
        splits.check_disjoint(s_tr, s_he)
        dev = torch.device("cuda")
        t0 = time.time()
        Vt, Et, Dt = load_group("train", dev)
        Vh, Eh, Dh = load_group("held", dev)
        Vz, Ez, Dz = load_group("zero", dev)
        run.info("tokens on the card: %.1f GB in %.0f s", (Vt.numel() + Vh.numel() + Vz.numel()) * 2 / 2 ** 30, time.time() - t0)
        assert s_tr.mask(Dt.scene.values).all() and s_he.mask(Dh.scene.values).all() and s_he.mask(Dz.scene.values).all()
        logs = np.array([logu(s) for s in Dt.scene])
        assert not set(logs) & {logu(s) for s in Dh.scene}, "a held log in training"
        val = np.array([is_val(l) for l in logs])
        fit = ~val
        if a.frac < 1:
            keep = frac_logs(logs[fit].tolist(), a.frac)
            fit &= np.array([l in keep for l in logs])
        if a.smoke:
            rng = np.random.default_rng(0)
            for m in (fit, val):
                m[rng.permutation(np.flatnonzero(m))[3000:]] = False
        i_fit, i_val = np.flatnonzero(fit), np.flatnonzero(val)
        run.info("fit %d rows / %d logs / %d scenes; val %d / %d logs", len(i_fit), len(set(logs[fit])), Dt.scene[fit].nunique(), len(i_val), len(set(logs[val])))
        Y = targets(Dt)
        mu, sd = np.nanmean(Y[i_fit, :NREG], 0), np.nanstd(Y[i_fit, :NREG], 0) + 1e-6
        Ys = Y.copy()
        Ys[:, :NREG] = (Y[:, :NREG] - mu) / sd
        emu, esd = Et[i_fit].mean(0), Et[i_fit].std(0) + 1e-6
        G = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        E = {g: G((e - emu) / esd) for g, e in (("train", Et), ("held", Eh), ("zero", Ez))}
        V = {"train": Vt, "held": Vh, "zero": Vz}
        if a.arm == "lin":
            P = ridge(run, Vt, E, V, Ys, i_fit, i_val, dev)
        else:
            P = head(run, H1, a, V, E, Ys, i_fit, i_val, dev, F)
        Pd = {g: np.c_[p[:, :NREG] * sd + mu, p[:, NREG:]] for g, p in P.items()}
        (W / "pred").mkdir(parents=True, exist_ok=True)
        f = W / "pred" / f"{name}.npz"
        np.savez(str(f) + ".tmp.npz", held=Pd["held"], zero=Pd["zero"], val=Pd["val"], val_rows=i_val, outs=np.array(OUTS), arm=a.arm, frac=a.frac, seed=a.seed,
                 n_fit=len(i_fit), logs_fit=len(set(logs[fit])), scenes_fit=int(Dt.scene[fit].nunique()))
        os.replace(str(f) + ".tmp.npz", f)
        run.summary.update(pred=str(f), n_fit=len(i_fit), logs_fit=len(set(logs[fit])))


def head(run, H1, a, V, E, Ys, i_fit, i_val, dev, F):
    import time
    import torch
    vision = a.arm != "blind"
    Yt, ok = torch.as_tensor(np.nan_to_num(Ys), device=dev), torch.as_tensor(np.isfinite(Ys), device=dev).float()

    def loss_of(o, i):
        y, m = Yt[i], ok[i]
        reg = (F.huber_loss(o[:, :NREG], y[:, :NREG], delta=1.0, reduction="none") * m[:, :NREG]).sum() / m[:, :NREG].sum().clamp_min(1)
        cls = (F.binary_cross_entropy_with_logits(o[:, NREG], y[:, NREG], reduction="none") * m[:, NREG]).sum() / m[:, NREG].sum().clamp_min(1)
        return reg + 0.2 * cls
    torch.manual_seed(a.seed)
    net = H1.build(vision, nq=len(OUTS), nc=1).to(dev)
    fwd = lambda g, i: net(V[g][i] if vision else None, E[g][i])[..., 0]  # noqa: E731
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps, warm, ev = (60, 10, 30) if a.smoke else (a.steps, a.warm, a.eval)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    gf, gv = torch.as_tensor(i_fit, device=dev), torch.as_tensor(i_val, device=dev)
    run.info("%s: %.2f M parameters", a.arm, sum(x.numel() for x in net.parameters()) / 1e6)
    best, best_v, curve, t1 = None, np.inf, [], time.time()
    for step in run.tqdm(range(steps), desc=f"train {a.arm}"):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, step / steps)))
        b = gf[torch.randint(len(gf), (a.batch,), device=dev, generator=gen)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = fwd("train", b).float()
        loss = loss_of(o, b)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            run.scalar("loss/train", float(loss), step)
        if (step + 1) % ev == 0 or step + 1 == steps:
            net.eval()
            num = den = 0.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for c in range(0, len(gv), 512):
                    i = gv[c:c + 512]
                    num, den = num + float(loss_of(fwd("train", i).float(), i)) * len(i), den + len(i)
            net.train()
            v = num / max(den, 1)
            curve.append(dict(step=step + 1, val=v, train=float(loss)))
            run.scalar("loss/val", v, step + 1)
            run.info("step %d: train %.4f val %.4f | %.1f it/s", step + 1, float(loss), v, (step + 1) / (time.time() - t1))
            if v < best_v:
                best_v, best = v, {k: x.detach().clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best)
    net.eval()

    def pred(g, idx):
        O = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for c in range(0, len(idx), 512):
                O.append(fwd(g, torch.as_tensor(idx[c:c + 512], device=dev)).float().cpu().numpy())
        return np.concatenate(O)
    P = dict(held=pred("held", np.arange(len(E["held"]))), zero=pred("zero", np.arange(len(E["zero"]))), val=pred("train", i_val))
    run.path("curve.json").write_text(json.dumps(curve, indent=1))
    run.summary.update(best_step=int(min(curve, key=lambda c: c["val"])["step"]), best_val=best_v, its=steps / max(time.time() - t1, 1e-6),
                       vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
    return P


def ridge(run, Vt, E, V, Ys, i_fit, i_val, dev, lams=(1e1, 1e2, 1e3, 1e4, 1e5)):
    """Linear floor: newest slot's 32 x 512 tokens flattened + standardised ego -> each output, ridge, lambda chosen on val."""
    import torch
    feat = lambda g, i: torch.cat([V[g][i, 7].flatten(1).float(), E[g][i]], 1)  # noqa: E731
    mu = torch.zeros(32 * 512 + 20, device=dev)
    for c in range(0, len(i_fit), 4096):
        mu += feat("train", torch.as_tensor(i_fit[c:c + 4096], device=dev)).sum(0)
    mu /= len(i_fit)
    nout = Ys.shape[1]
    Y = torch.as_tensor(np.nan_to_num(Ys), device=dev, dtype=torch.float32)
    M = torch.as_tensor(np.isfinite(Ys), device=dev, dtype=torch.float32)
    Wb = []
    for j in range(nout):
        rows = i_fit[np.isfinite(Ys[i_fit, j])]
        G = torch.zeros((mu.numel(), mu.numel()), device=dev)
        b = torch.zeros(mu.numel(), device=dev)
        ym = float(Y[torch.as_tensor(rows, device=dev), j].mean())
        for c in range(0, len(rows), 4096):
            r = torch.as_tensor(rows[c:c + 4096], device=dev)
            X = feat("train", r) - mu
            G += X.T @ X
            b += X.T @ (Y[r, j] - ym)
        tr = float(G.diagonal().mean())
        best = None
        for lam in lams:
            w = torch.linalg.solve(G + lam * tr * 1e-3 * torch.eye(len(b), device=dev), b)
            vr = i_val[np.isfinite(Ys[i_val, j])]
            r = torch.as_tensor(vr, device=dev)
            p = (feat("train", r) - mu) @ w + ym
            err = float(((p - Y[r, j]) ** 2).mean())
            if best is None or err < best[0]:
                best = (err, lam, w, ym)
        run.info("ridge output %s: lambda %.0e (relative), val mse %.4f", OUTS[j], best[1], best[0])
        Wb.append((best[2], best[3]))
        del G
        torch.cuda.empty_cache()

    def pred(g, idx):
        O = []
        with torch.no_grad():
            for c in range(0, len(idx), 2048):
                X = feat(g, torch.as_tensor(idx[c:c + 2048], device=dev)) - mu
                O.append(torch.stack([X @ w + ym for w, ym in Wb], 1).cpu().numpy())
        return np.concatenate(O)
    _ = M
    return dict(held=pred("held", np.arange(len(E["held"]))), zero=pred("zero", np.arange(len(E["zero"]))), val=pred("train", i_val))


# ---------------------------------------------------------------- read (CPU): gates, registered tables, lines, verdict
CURVE = (0.125, 0.25, 0.5, 1.0)
DB = (0, 10, 20, 40, 60)


def shipped(g, D, scale=False):
    """P0 heads on a group's tokens -> (n, 13) predictions in OUTS order (h20 from the ego-line centre slope), and (n, 4) line probabilities."""
    raw, tb = dict(np.load(W / f"tok/{g}/p0.npz")), dict(np.load(W / f"tok/{g}/tab.npz"))
    n = len(tb["k"])
    out, prob = np.full((n, NREG), np.nan, np.float32), np.zeros((n, 4), np.float32)
    for i in range(n):
        cam = tb["cam"][i].astype(float)
        h = EV.heads({k: raw[k][i] for k in raw}, cam)
        ll, re = h["ll"], h["re"]
        if scale:
            ll, re = cam[1] + (ll - cam[1]) / SCALE, cam[1] + (re - cam[1]) / SCALE
        prob[i] = h["pr"]
        for j, d in enumerate(DS):
            x = cam[0] + d
            out[i, 4 * j:4 * j + 4] = [EV.at(ll[1], x, cam[0]), EV.at(ll[2], x, cam[0]), EV.at(re[0], x, cam[0]), EV.at(re[1], x, cam[0])]
        c = lambda x: 0.5 * (EV.at(ll[1], x, cam[0]) + EV.at(ll[2], x, cam[0]))  # noqa: E731
        out[i, 12] = np.arctan2(c(cam[0] + 22.0) - c(cam[0] + 18.0), 4.0)
    return out, prob


def boot_q(e, cl, q=0.5, B=2000, seed=0):
    """Quantile q of e with a percentile CI from resampling clusters cl (scenes)."""
    m = np.isfinite(e)
    e, cl = e[m], cl[m]
    if not len(e):
        return dict(n=0, v=np.nan, lo=np.nan, hi=np.nan, clusters=0)
    u, inv = np.unique(cl, return_inverse=True)
    groups = [e[inv == k] for k in range(len(u))]
    rng = np.random.default_rng(seed)
    bs = [np.quantile(np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))]), q) for _ in range(B)]
    return dict(n=int(len(e)), clusters=int(len(u)), v=float(np.quantile(e, q)), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)))


def auc(y, s):
    from scipy.stats import rankdata
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return dict(auc=np.nan, pos=n1, neg=n0)
    r = rankdata(s)
    return dict(auc=float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), pos=n1, neg=n0)


def r3_windows(err_l, err_r, Dz, ev):
    """Per registered window: per-side median of the 20 m error over its decisions -> window error (mean of the sides), both-within flag."""
    out = []
    for e in ev:
        k_ev = int(e["k_ev"])
        m = ((Dz.tag == e["group"]) & (Dz.scene == e["scene"]) & (Dz.k >= max(0, k_ev - 4)) & (Dz.k <= k_ev)).values
        ml = float(np.nanmedian(err_l[m])) if np.isfinite(err_l[m]).any() else np.nan
        mr = float(np.nanmedian(err_r[m])) if np.isfinite(err_r[m]).any() else np.nan
        sides = [x for x in (ml, mr) if np.isfinite(x)]
        out.append(dict(group=e["group"], scene=e["scene"], type=e["type"], k_ev=k_ev, n_dec=int(m.sum()), med_L=ml, med_R=mr,
                        werr=float(np.mean(sides)) if sides else np.nan, both=bool(np.isfinite(ml) and np.isfinite(mr) and ml <= 0.5 and mr <= 0.5),
                        both_eval=bool(np.isfinite(ml) and np.isfinite(mr))))
    return out


def r3_summary(W_, B=2000):
    import pandas as pd
    X = pd.DataFrame(W_)
    w = X.werr.values
    ok = np.isfinite(w)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(ok)
    bm = [np.median(w[rng.choice(idx, len(idx))]) for _ in range(B)]
    be = X.both_eval.values
    sh = X.both.values[be]
    bs = [rng.choice(sh, len(sh)).mean() for _ in range(B)] if len(sh) else [np.nan]
    return dict(windows=len(X), eval=int(ok.sum()), both_eval=int(be.sum()), median=float(np.median(w[ok])), med_lo=float(np.quantile(bm, 0.025)),
                med_hi=float(np.quantile(bm, 0.975)), share=float(sh.mean()) if len(sh) else np.nan, share_lo=float(np.quantile(bs, 0.025)),
                share_hi=float(np.quantile(bs, 0.975)))


def nearest(B, y):
    """|y - nearest crossing| per row; B (n, 24) NaN-padded."""
    d = np.abs(B - y[:, None])
    d[~np.isfinite(d)] = np.inf
    m = d.min(1)
    return np.where(np.isfinite(m) & np.isfinite(y), m, np.nan)


def read(a):
    import pandas as pd
    from jevdrive.run import Run
    with Run("road_probe", "read", config=vars(a)) as run:
        out = Path(run.dir)
        Dh, Dz = pd.read_parquet(W / "labels/held.parquet"), pd.read_parquet(W / "labels/zero.parquet")
        ev = zero_events()
        G = {}
        # gates
        sc = []
        for rd in sorted(W.glob("roll/c*/run")):
            sc += [float(r["score"]) for r in json.loads((rd / "aggregate/results-summary.json").read_text())["rollouts"] if r.get("score") is not None]
        held = set(np.loadtxt(REPO / "experiments/onpol_nuplan/results/split/held.txt", dtype=str).tolist())
        st = [float(r["score"]) for r in json.loads((DATA / "runs/onpol_long/read" / HELD_TAG / "run/aggregate/results-summary.json").read_text())["rollouts"]
              if r.get("score") is not None and r["clipgt_id"] in held]
        G["G_roll"] = dict(rollouts=len(sc), mean=float(np.mean(sc)), stored_rollouts=len(st), stored_mean=float(np.mean(st)))
        G["G_roll"]["pass"] = bool(len(sc) >= 450 and abs(G["G_roll"]["mean"] - G["G_roll"]["stored_mean"]) < 0.03)
        G["G_tok"] = json.loads((W / "tok/held/G_tok.json").read_text())
        G["G_rep"] = json.loads((W / "tok/zero/G_rep.json").read_text())
        for g in ("train", "held", "zero"):
            G[f"G_frame_{g}"] = json.loads((W / f"labels/{g}.gates.json").read_text())["G_frame"]
        st_ = (Dh.turn40 < 5).values
        tu_ = (Dh.turn40 >= 20).values
        G["G_lab"] = dict(straight=int(st_.sum()), share=float((Dh.L20.notna() & Dh.R20.notna())[st_].mean()))
        G["G_lab"]["pass"] = bool(G["G_lab"]["share"] >= 0.70)
        G["G_sign"] = dict(L20=float(Dh.L20[st_].median()), R20=float(Dh.R20[st_].median()))
        G["G_sign"]["pass"] = bool(G["G_sign"]["L20"] > 0 > G["G_sign"]["R20"])
        G["pass"] = all(v["pass"] for k, v in G.items() if k.startswith("G_"))
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        run.info(json.dumps(G, default=float))
        if not G["pass"] and not a.force:
            run.info("a gate failed: no registered read")
            return
        # predictions
        P, meta = {}, {}
        for f in sorted((W / "pred").glob("*.npz")):
            if "smoke" in f.stem:
                continue
            z = np.load(f)
            P[f.stem] = dict(held=z["held"], zero=z["zero"])
            meta[f.stem] = dict(logs_fit=int(z["logs_fit"]), scenes_fit=int(z["scenes_fit"]), n_fit=int(z["n_fit"]))
        P["P"] = {g: 0.5 * (P["P-p1000-s0"][g] + P["P-p1000-s1"][g]) for g in ("held", "zero")}
        sh = {g: shipped(g, D) for g, D in (("held", Dh), ("zero", Dz))}
        shk = {g: shipped(g, D, True)[0] for g, D in (("held", Dh), ("zero", Dz))}
        P["P0"] = {g: sh[g][0] for g in sh}
        P["P0K"] = shk
        Ldt = {g: np.stack([D[c].values for c in OUTS[:NREG]], 1) for g, D in (("held", Dh), ("zero", Dz))}
        err = {k: {g: np.abs(v[g][:, :NREG] - Ldt[g]) for g in ("held", "zero")} for k, v in P.items()}
        for k in err:
            for g in ("held", "zero"):
                err[k][g][:, 12] = np.degrees(np.abs(EV.wrap(P[k][g][:, 12] - Ldt[g][:, 12])))
        names = ["P", "P-p1000-s0", "P-p1000-s1", "blind-p1000-s0", "lin-p1000-s0", "P0", "P0K"]
        cl_h = Dh.scene.values
        # R1 / R2
        rows = []
        for sub, m in (("straight", st_), ("turn", tu_), ("gentle", ~st_ & ~tu_), ("all", np.ones(len(Dh), bool))):
            for k in names:
                e = err[k]["held"][m]
                for qn, cols in (("L20", [4]), ("R20", [5]), ("LR20", [4, 5]), ("LR10", [0, 1]), ("LR40", [8, 9]), ("E20", [6, 7]), ("E10", [2, 3]), ("E40", [10, 11]), ("h20_deg", [12])):
                    ee = e[:, cols].ravel()
                    cc = np.repeat(cl_h[m], len(cols)).reshape(-1, len(cols)).ravel() if len(cols) > 1 else cl_h[m]
                    if len(cols) > 1:
                        cc = np.repeat(cl_h[m][:, None], len(cols), 1).ravel()
                    md, p9 = boot_q(ee, cc, 0.5, B=a.boot), boot_q(ee, cc, 0.9, B=a.boot)
                    rows.append(dict(subset=sub, pred=k, q=qn, n=md["n"], scenes=md["clusters"], median=md["v"], med_lo=md["lo"], med_hi=md["hi"],
                                     p90=p9["v"], p90_lo=p9["lo"], p90_hi=p9["hi"]))
        T = pd.DataFrame(rows)
        T.to_csv(out / "r12.csv", index=False)
        # nearest-boundary error (decision 269's measure) at 20 m, L and R predictions pooled
        B20 = {g: np.stack(D.xB20.values) for g, D in (("held", Dh), ("zero", Dz))}
        nb = []
        for sub, m in (("straight", st_), ("turn", tu_)):
            for k in names:
                e = np.r_[nearest(B20["held"][m], P[k]["held"][m, 4]), nearest(B20["held"][m], P[k]["held"][m, 5])]
                r = boot_q(e, np.r_[cl_h[m], cl_h[m]], B=a.boot)
                nb.append(dict(subset=sub, pred=k, n=r["n"], median=r["v"], lo=r["lo"], hi=r["hi"], p90=float(np.nanquantile(e, 0.9))))
        pd.DataFrame(nb).to_csv(out / "nearest.csv", index=False)
        # P0 presence (both ego lines > 0.5)
        pres = {s: float((sh["held"][1][m][:, 1:3] > 0.5).all(1).mean()) for s, m in (("straight", st_), ("turn", tu_))}
        # R3
        r3, r3w = {}, {}
        for k in names + [n for n in P if n.startswith("P-p") and n.endswith("-s0")]:
            r3w[k] = r3_windows(err[k]["zero"][:, 4], err[k]["zero"][:, 5], Dz, ev)
            r3[k] = r3_summary(r3w[k], B=a.boot)
        pd.DataFrame(r3w["P"]).assign(P0=[w["werr"] for w in r3w["P0"]], P0K=[w["werr"] for w in r3w["P0K"]]).to_csv(out / "r3_windows.csv", index=False)
        R = r3["P"]
        verdict = ("eyes repairable" if R["median"] < 0.3 and R["share"] >= 0.6 else "frozen eyes do not see city roads" if R["median"] > 1.0 else "undetermined")
        # learning curve
        lc = []
        for f in CURVE:
            k = f"P-p{int(round(f * 1000)):04d}-s0"
            if k not in P:
                continue
            e = err[k]["held"]
            lc.append(dict(frac=f, logs=meta[k]["logs_fit"], scenes=meta[k]["scenes_fit"], rows=meta[k]["n_fit"],
                           straight=float(np.nanmedian(e[st_][:, 4:6])), turn=float(np.nanmedian(e[tu_][:, 4:6])), r3=r3[k]["median"], r3_share=r3[k]["share"]))
        LC = pd.DataFrame(lc)
        slope = {}
        if len(LC) >= 3:
            x = np.log2(LC.logs.values.astype(float))
            rng = np.random.default_rng(0)
            for col in ("turn", "r3"):
                b0 = float(np.polyfit(x, LC[col].values, 1)[0])
                bs = []
                if col == "r3":
                    ws = {k: pd.DataFrame(r3w[f"P-p{int(round(f * 1000)):04d}-s0"]).werr.values for k, f in zip(range(len(LC)), LC.frac)}
                    nW = len(ws[0])
                    for _ in range(a.boot):
                        i = rng.integers(0, nW, nW)
                        bs.append(np.polyfit(x, [np.nanmedian(ws[j][i]) for j in range(len(LC))], 1)[0])
                else:
                    us = np.unique(cl_h[tu_])
                    E_ = {j: err[f"P-p{int(round(f * 1000)):04d}-s0"]["held"][tu_][:, 4:6] for j, f in enumerate(LC.frac)}
                    gi = {u: np.flatnonzero(cl_h[tu_] == u) for u in us}
                    for _ in range(min(a.boot, 500)):
                        idx = np.concatenate([gi[u] for u in rng.choice(us, len(us))])
                        bs.append(np.polyfit(x, [np.nanmedian(E_[j][idx]) for j in range(len(LC))], 1)[0])
                slope[col] = dict(per_doubling=b0, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)))
        LC.to_csv(out / "curve.csv", index=False)
        # turn error by distance to the junction (held turn decisions)
        jb = []
        jd = Dh.jdist.values
        bins = [("in junction", jd == 0)] + [(f"{lo}-{hi} m", (jd > lo) & (jd <= hi)) for lo, hi in zip(DB[:-1], DB[1:])] + [("none within 60 m", np.isnan(jd))]
        for lab, mb in bins:
            for sub, m in (("turn", tu_), ("straight", st_)):
                mm = mb & m
                for k in ("P", "P0", "P0K", "blind-p1000-s0", "lin-p1000-s0"):
                    e = err[k]["held"][mm][:, 4:6].ravel()
                    jb.append(dict(bin=lab, subset=sub, pred=k, n=int(np.isfinite(e).sum()), decisions=int(mm.sum()), median=float(np.nanmedian(e)) if np.isfinite(e).any() else np.nan))
        pd.DataFrame(jb).to_csv(out / "junction.csv", index=False)
        # exploratory: oncoming-lane classifier
        onc = {}
        yh = np.where(Dh.lane_cls.values == 0, np.nan, (Dh.lane_cls.values == 2).astype(float))
        yz = np.where(Dz.lane_cls.values == 0, np.nan, (Dz.lane_cls.values == 2).astype(float))
        for k in ("P", "P-p1000-s0", "lin-p1000-s0", "blind-p1000-s0"):
            onc[k] = dict(held=auc(yh, P[k]["held"][:, 13]), zero=auc(yz, P[k]["zero"][:, 13]))
        # per-decision errors for the figures
        keep = ["row", "scene", "k", "tag", "turn40", "jdist", "lane_cls"] + OUTS[:NREG]
        for g, D in (("held", Dh), ("zero", Dz)):
            X = D[keep].copy()
            for k in ("P", "P0", "P0K", "blind-p1000-s0", "lin-p1000-s0"):
                for j, c in enumerate(OUTS):
                    if j < NREG:
                        X[f"{k}:{c}"] = P[k][g][:, j]
            X.to_parquet(out / f"dec_{g}.parquet", index=False)
        S = dict(gates=G, verdict=verdict, r3=r3, slope=slope, presence_P0=pres, onc=onc, meta=meta,
                 counts=dict(held=len(Dh), held_scenes=int(Dh.scene.nunique()), straight=int(st_.sum()), turn=int(tu_.sum()), zero=len(Dz), windows=len(ev)))
        (out / "rp.json").write_text(json.dumps(S, indent=1, default=float))
        run.summary.update(verdict=verdict, r3=R)
        run.info(f"verdict: {verdict}; R3 {json.dumps(R)}")


# ---------------------------------------------------------------- bev payloads (AlpaSim venv)
def bev_unit(job):
    g, scene, rows = job
    from ao_e1 import loader
    ds = loader(str(EV.CFG)).get_data_source(scene)
    tr = ds.rig.trajectory
    Pg, Yg = np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
    LP = np.vstack([Pg, Pg[-1] + np.outer(np.linspace(2.0, 60.0, 30), [np.cos(Yg[-1]), np.sin(Yg[-1])])])
    A = np.array([r["anchor"] for r in rows])
    allp = np.vstack([Pg, A[:, :2]])
    G = EV.Geo(ds, (allp.min(0) - 90, allp.max(0) + 90))
    out = {}
    box = lambda p: ((np.abs(p[:, 1]) < 25) & (p[:, 0] > -15) & (p[:, 0] < 60)).any()  # noqa: E731
    for r in rows:
        a = r["anchor"]
        Lg = EV.in_ego(a, LP)
        out[(g, scene, r["k"])] = dict(edges=[p.astype(np.float32) for p in (EV.in_ego(a, q) for q in G.edges) if box(p)],
                                       road=[p.astype(np.float32) for p in (EV.in_ego(a, q) for q in G.road) if box(p)],
                                       log=Lg[(Lg[:, 0] > -15) & (Lg[:, 0] < 60) & (np.abs(Lg[:, 1]) < 25)].astype(np.float32))
    return out


def bev(a):
    from jevdrive import par
    from jevdrive.run import Run
    picks = json.loads(Path(a.picks).read_text())                       # [[group, scene, k], ...]
    with Run("road_probe", "bev", config=vars(a)) as run:
        jobs = []
        for g in sorted({p[0] for p in picks}):
            rows = {(r["scene"], r["k"]): r for r in group_rows(g)}
            by = {}
            for p in picks:
                if p[0] == g:
                    by.setdefault(p[1], []).append(rows[(p[1], int(p[2]))])
            jobs += [(g, s, rr) for s, rr in by.items()]
        res = par.pmap(bev_unit, jobs, workers=min(len(jobs), 16), run=run, desc="bev")
        res.raise_if_failed()
        pay = {}
        for v in res.values:
            pay.update(v)
        for g in sorted({p[0] for p in picks}):                        # served plan and the P0 heads' curves (ego frame, left +)
            tb, raw = dict(np.load(W / f"tok/{g}/tab.npz")), dict(np.load(W / f"tok/{g}/p0.npz"))
            idx = {(str(s_), int(k_)): i for i, (s_, k_) in enumerate(zip(tb["scene"], tb["k"]))}
            for key in [k for k in pay if k[0] == g]:
                i = idx[(key[1], int(key[2]))]
                cam = tb["cam"][i].astype(float)
                h = EV.heads({q: raw[q][i] for q in raw}, cam)
                sel = EV.X_IDXS <= 60
                pay[key].update(row=i, plan=tb["poses"][i].astype(np.float32), cam=cam, xs=cam[0] + EV.X_IDXS[sel], ll=h["ll"][:, sel], re=h["re"][:, sel], pr=h["pr"])
        pickle.dump(pay, open(Path(a.read) / "bev.pkl", "wb"), protocol=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tok")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("replay")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("labels")
    p.add_argument("--group", required=True, choices=("train", "held", "zero")), p.add_argument("--workers", type=int, default=0), p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("train")
    p.add_argument("--arm", required=True, choices=("P", "blind", "lin")), p.add_argument("--frac", type=float, default=1.0), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8000), p.add_argument("--batch", type=int, default=256), p.add_argument("--eval", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4), p.add_argument("--wd", type=float, default=0.05), p.add_argument("--warm", type=int, default=300)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--boot", type=int, default=2000), p.add_argument("--force", action="store_true")
    p = sub.add_parser("bev")
    p.add_argument("--picks", required=True), p.add_argument("--read", required=True)
    a = ap.parse_args()
    dict(tok=tok, replay=replay, labels=labels, train=train, read=read, bev=bev)[a.cmd](a)
