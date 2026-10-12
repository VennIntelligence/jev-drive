#!/usr/bin/env python3
"""ROAD-PROBE-REAL (plans/2026-10-12-road-probe-prereg.md, amendment 8): is the junction blur of decision 275 (turn error at 20 m 1.1-1.7 m
inside / just before a junction, 0.42-0.47 m when it is 20-60 m away) a property of the frozen encoder on REAL nuPlan camera frames too, or an
artefact of the MTGS-reconstructed images of the AlpaSim loop? Same labels (rp.lab_rows), same head (rp.head), real navsim frames.

  labels  (AlpaSim venv, CPU)   --split navtest|navtrain: map = the trajdata nuPlan vector maps AlpaSim loads (after160 cache, world frame),
                                logged path = the navsim log segment's rear-axle poses (2 Hz, Hermite-densified) -> $W/real/labels/<split>.parquet
  train   (envs/op-train, GPU)  --arm A|B|blind|C --seed S   A = decision 275's probe re-trained on the on-policy rows (weights kept),
                                applied to the navtest tokens of the held log units; B = the same head on navtrain tokens; blind = B without
                                tokens; C = B on 27 fit units (exploratory); --arm P0 = the shipped heads on the navtest tokens
                                -> $W/real/pred/<arm>-s<S>.npz, $W/tok/navtest/{p0,tab}.npz
  read    (AlpaSim venv, CPU)   gates, bucket curves, near / far / ratio, verdict -> $W/real/read/<ts>/
  report  (Mac .venv)           --read DIR -> results/rp_real.md, results/rp_real_buckets.csv, figs/rp_real.png

Frame: the token's rear axle, x forward, y LEFT (as rp.py). Units of a navsim log name: rp.logu (date_vehicle = one drive log).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("labels", "read"):                # one BLAS / polars thread per worker
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/road_probe/scripts")]
OUT, FIGS = REPO / "experiments/road_probe/results", REPO / "experiments/road_probe/figs"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/road_probe"
R = W / "real"
CR = DATA / "runs/op_parity/cache"
LOGS = DATA / "datasets/navsim/navsim_logs"
MAPC = DATA / "datasets/alpasim_nuplan_long/after160"
MAPS = {"us-nv-las-vegas-strip": "las_vegas", "us-ma-boston": "boston", "us-pa-pittsburgh-hazelwood": "pittsburgh", "sg-one-north": "singapore"}
VMP = dict(collate=False, incl_road_areas=True, incl_road_edges=True, incl_road_lanes=True, incl_traffic_signs=True, incl_wait_lines=True,
           keep_in_memory=True)                                           # the vector_map_params of the AlpaSim runs (generated-user-config)
NSH = 12
REC_READ = W / "read/20261012-175622"                                     # decision 275's read: dec_held.parquet
BINS = ("in junction", "0-10 m", "10-20 m", "20-40 m", "40-60 m", "none within 60 m")
NEAR, FAR = BINS[:3], BINS[3:5]
REF_TURN = 0.877                                                          # decision 275: held turn, 20 m, probe (2 seeds)
NFIT_C = 27                                                               # decision 275's fit log units


def dirs(split):
    return [f"navtrain_full.s{k}of{NSH}" for k in range(NSH)] if split == "navtrain" else ["lb_navtest"]


def tab(split):
    """Token-cache rows of a split, in cache order -> dict(names, log, cam, ego)."""
    ts = [np.load(CR / d / "tab.npz") for d in dirs(split)]
    return {k: np.concatenate([t[k] for t in ts]) for k in ("names", "log", "cam", "ego")}


def logu(s):
    return "_".join(str(s).split("_")[:2])


def held_units():
    from jevdrive.data import splits
    return {logu(s) for s in splits.load("alpasim_nuplan/onpol-held").members}


def train_units():
    from jevdrive.data import splits
    return {logu(s) for s in splits.load("alpasim_nuplan/onpol-train").members}


# ---------------------------------------------------------------- labels (AlpaSim venv, CPU)
_VM = {}


def vmap(loc):
    if loc not in _VM:
        from trajdata.maps.map_api import MapAPI
        _VM[loc] = MapAPI(MAPC).get_map(f"nuplan_test:{MAPS[loc]}", **VMP)
    return _VM[loc]


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def densify(T, P, Y, n=10):
    """2 Hz log poses -> cubic Hermite path through them with tangents along the logged heading (chord length scaled); standstill
    and reversing steps stay straight. -> (T, P, Y) at n sub-steps per step."""
    Td, Pd, Yd = [T[:1]], [P[:1]], [Y[:1]]
    u = np.linspace(0, 1, n + 1)[1:, None]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    for i in range(len(P) - 1):
        d = P[i + 1] - P[i]
        L = float(np.hypot(*d))
        t0, t1 = np.array([np.cos(Y[i]), np.sin(Y[i])]), np.array([np.cos(Y[i + 1]), np.sin(Y[i + 1])])
        if L < 0.05 or d @ t0 < 0 or d @ t1 < 0:
            q = P[i] + u * d
        else:
            q = h00 * P[i] + h10 * L * t0 + h01 * P[i + 1] + h11 * L * t1
        Td.append(T[i] + u[:, 0] * (T[i + 1] - T[i]))
        Pd.append(q)
        Yd.append(Y[i] + u[:, 0] * (Y[i + 1] - Y[i]))
    return np.concatenate(Td), np.concatenate(Pd), np.concatenate(Yd)


def lab_log(job):
    """(split, log, [(row, token, cam)]) -> label dicts (rp.lab_rows) + in_road."""
    import pickle
    import types
    import shapely
    import rp
    import zm_read as ZM
    from trajdata.maps.vec_map_elements import MapElementType as MT
    split, log, rows = job
    fr = pickle.load(open(LOGS / ("test" if split == "navtest" else "trainval") / f"{log}.pkl", "rb"))
    T = np.array([f["timestamp"] for f in fr], float)
    P = np.array([f["ego2global_translation"][:2] for f in fr], float)
    Y = np.unwrap([yaw_of(f["ego2global_rotation"]) for f in fr])
    pos = {f["token"]: i for i, f in enumerate(fr)}
    vm = vmap(fr[0]["map_location"])
    Td, Pd, Yd = densify(T, P, Y)
    Pg3 = np.c_[Pd, np.zeros(len(Pd))]                                     # map z is 0 in the world frame
    R_ = [dict(row=int(r), scene=log, k=pos[t], t0=int(T[pos[t]]), anchor=np.r_[P[pos[t]], Y[pos[t]]], cam=np.asarray(c, float), src=split, tag=split)
          for r, t, c in rows]
    lo, hi = P.min(0) - 90, P.max(0) + 90
    out = rp.lab_rows(split, log, R_, Td, Pg3, Yd, rp.EV.Geo(types.SimpleNamespace(map=vm), (lo, hi)), ZM.Lanes(vm))
    polys = []
    for ra in vm.elements.get(MT.ROAD_AREA, {}).values():
        E = np.asarray(ra.exterior_polygon.points, float)[:, :2]
        if len(E) > 2 and ((E >= lo) & (E <= hi)).all(1).any():
            polys.append(shapely.Polygon(E, [np.asarray(h.points, float)[:, :2] for h in ra.interior_holes]).buffer(0))
    road = shapely.union_all(polys) if polys else None
    tok = {int(r): t for r, t, _ in rows}
    for o, r in zip(out, R_):
        o["token"], o["unit"] = tok[r["row"]], logu(log)
        o["in_road"] = bool(road is not None and road.contains(shapely.Point(*r["anchor"][:2])))
    return out


def labels(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    import rp  # noqa: F401  (sets the AlpaSim / zeros_mech paths for the workers)
    with Run("road_probe", f"real/labels-{a.split}", config=vars(a)) as run:
        sp = splits.load(f"navsim/{a.split}")
        run.use_split(sp)
        tb = tab(a.split)
        assert sp.mask(tb["names"]).all(), "a cached token outside the split"
        by = {}
        for i, (t, l, c) in enumerate(zip(tb["names"], tb["log"], tb["cam"])):
            by.setdefault(str(l), []).append((i, str(t), c))
        jobs = [(a.split, l, by[l]) for l in sorted(by)][: a.limit or None]
        run.info(f"{a.split}: {len(tb['names'])} tokens, {len(by)} log segments, {len(jobs)} jobs")
        res = par.pmap(lab_log, jobs, workers=a.workers or n_cpus(), run=run, desc="logs")
        D = pd.DataFrame([x for v in res.values if v for x in v]).sort_values("row").reset_index(drop=True)
        (R / "labels").mkdir(parents=True, exist_ok=True)
        f = R / "labels" / f"{a.split}{'-smoke' if a.limit else ''}.parquet"
        D.to_parquet(f, index=False)
        g = dict(jobs=len(jobs), errors=len(res.errors), err_first=[res.errors[i].strip().splitlines()[-1][:300] for i in sorted(res.errors)[:5]], rows=len(D),
                 G_frame=dict(in_lane=float((D.lane_cls != 0).mean()), in_road=float(D.in_road.mean()), share=float(((D.lane_cls != 0) | D.in_road).mean())))
        g["G_frame"]["pass"] = bool(g["G_frame"]["share"] >= 0.95 and not a.limit and len(D) == len(tb["names"]))
        (R / "labels" / f"{f.stem}.gates.json").write_text(json.dumps(g, indent=1, default=float))
        run.summary.update(g)
        run.info(json.dumps(g, default=float))


# ---------------------------------------------------------------- train (envs/op-train, GPU)
def load_rows(split, rows, dev, n_threads=8):
    """(len(rows), 8, 32, 512) fp16 on the device: rows of a split's concatenated @warp front caches."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    off = np.cumsum([0] + [len(np.load(CR / d / "tab.npz")["names"]) for d in dirs(split)])
    V = torch.empty((len(rows), 8, 32, 512), dtype=torch.float16, device=dev)
    jobs = []
    for k, d in enumerate(dirs(split)):
        m = np.flatnonzero((rows >= off[k]) & (rows < off[k + 1]))
        jobs += [(d, rows[m[c:c + 1024]] - off[k], m[c:c + 1024]) for c in range(0, len(m), 1024)]

    def one(j):
        d, r, o = j
        return o, np.ascontiguousarray(np.load(CR / f"{d}@warp" / "front.npy", mmap_mode="r")[r])
    with ThreadPoolExecutor(n_threads) as ex:
        for o, x in ex.map(one, jobs):
            V[torch.as_tensor(o, device=dev)] = torch.from_numpy(x).to(dev)
    return V


def train(a):
    import time
    import torch
    import torch.nn.functional as F
    import head1_train as H1
    import rp
    from jevdrive.data import splits
    from jevdrive.run import Run
    name = f"{a.arm}-s{a.seed}" + ("-smoke" if a.smoke else "")
    with Run("road_probe", f"real/train-{name}", seed=a.seed, config=vars(a)) as run:
        s_nt = splits.load("navsim/navtest")
        run.use_split(s_nt)
        dev = torch.device("cuda")
        tn = tab("navtest")
        t0 = time.time()
        if a.arm == "P0":
            import pp_train as T
            p0 = T.load_pmodel("P0", dev)
            raw = {k: [] for k in ("lane_lines", "lane_lines_prob", "road_edges")}
            tc = torch.tensor([[1.0, 0.0]], device=dev)
            idx = np.arange(len(tn["names"]))
            with torch.no_grad():
                for c in range(0, len(idx), 512):
                    i = idx[c:c + 512]
                    H = load_rows("navtest", i, dev)
                    for k, v in rp.p0_raw(p0(H, torch.as_tensor(tn["ego"][i], dtype=torch.float32, device=dev), tc.repeat(len(i), 1)).float(), p0.net.slices).items():
                        raw[k].append(v)
            (W / "tok/navtest").mkdir(parents=True, exist_ok=True)
            np.savez(W / "tok/navtest/p0.npz", **{k: np.concatenate(v) for k, v in raw.items()})
            np.savez(W / "tok/navtest/tab.npz", k=np.zeros(len(idx), int), cam=tn["cam"], scene=tn["log"], names=tn["names"])
            run.info(f"P0 heads on {len(idx)} navtest tokens in {time.time() - t0:.0f} s")
            return
        Dn = __import__("pandas").read_parquet(R / "labels/navtest.parquet")
        assert len(Dn) == len(tn["names"]) and (Dn.token.values == tn["names"]).all(), "navtest labels are not in token-cache order"
        un = np.array([logu(l) for l in tn["log"]])
        if a.arm == "A":                                                   # decision 275's probe, re-trained; its own train / held groups
            s_tr, s_he = splits.load("alpasim_nuplan/onpol-train"), splits.load("alpasim_nuplan/onpol-held")
            run.use_split(s_tr), run.use_split(s_he)
            Vt, Et, Dt = rp.load_group("train", dev)
            Vh, Eh, _ = rp.load_group("held", dev)
            logs = np.array([rp.logu(s) for s in Dt.scene])
            hu = held_units()
            ev_rows = np.flatnonzero(np.isin(un, list(hu)))
            assert not set(un[ev_rows]) & set(logs), "an evaluated navtest unit is in the on-policy training rows"
            val = np.array([rp.is_val(l) for l in logs])
            fit = ~val
            Y = rp.targets(Dt)
            V = {"train": Vt, "held": Vh, "navtest": load_rows("navtest", ev_rows, dev)}
            En = {"held": Eh, "navtest": tn["ego"][ev_rows].astype(np.float32)}
            groups = ("held", "navtest")
        else:                                                              # navtrain fit, log-unit validation
            s_tr = splits.load("navsim/navtrain")
            run.use_split(s_tr)
            tt = tab("navtrain")
            Dt = __import__("pandas").read_parquet(R / "labels/navtrain.parquet")
            assert len(Dt) == len(tt["names"]) and (Dt.token.values == tt["names"]).all(), "navtrain labels are not in token-cache order"
            logs = np.array([logu(l) for l in tt["log"]])
            assert not set(logs) & set(un), "navtest shares a drive log with navtrain"
            val = np.array([rp.is_val(l) for l in logs])
            fit = ~val
            if a.arm == "C":
                keep = set(sorted(set(logs[fit]), key=lambda l: __import__("hashlib").sha256(f"rplc|{l}".encode()).hexdigest())[:NFIT_C])
                fit &= np.isin(logs, list(keep))
            ev_rows = np.arange(len(tn["names"]))
            Et = tt["ego"].astype(np.float32)
            Y = rp.targets(Dt)
            vision = a.arm != "blind"
            V = {"train": load_rows("navtrain", np.arange(len(Dt)), dev) if vision else None,
                 "navtest": load_rows("navtest", ev_rows, dev) if vision else None}
            En = {"navtest": tn["ego"].astype(np.float32)}
            groups = ("navtest",)
        if a.smoke:
            rng = np.random.default_rng(0)
            for m in (fit, val):
                m[rng.permutation(np.flatnonzero(m))[3000:]] = False
        i_fit, i_val = np.flatnonzero(fit), np.flatnonzero(val)
        run.info("tokens in %.0f s; fit %d rows / %d units, val %d / %d units; eval navtest %d rows", time.time() - t0, len(i_fit), len(set(logs[fit])),
                 len(i_val), len(set(logs[val])), len(ev_rows))
        mu, sd = np.nanmean(Y[i_fit, :rp.NREG], 0), np.nanstd(Y[i_fit, :rp.NREG], 0) + 1e-6
        Ys = Y.copy()
        Ys[:, :rp.NREG] = (Y[:, :rp.NREG] - mu) / sd
        emu, esd = Et[i_fit].mean(0), Et[i_fit].std(0) + 1e-6
        G = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        E = {"train": G((Et - emu) / esd), **{g: G((e - emu) / esd) for g, e in En.items()}}
        a.arm_head = "blind" if a.arm == "blind" else "P"
        P = rp.head(run, H1, argparse.Namespace(**{**vars(a), "arm": a.arm_head}), V, E, Ys, i_fit, i_val, dev, F, groups=groups)
        Pd = {g: np.c_[p[:, :rp.NREG] * sd + mu, p[:, rp.NREG:]] for g, p in P.items()}
        nt = np.full((len(tn["names"]), len(rp.OUTS)), np.nan, np.float32)
        nt[ev_rows] = Pd["navtest"]
        (R / "pred").mkdir(parents=True, exist_ok=True)
        f = R / "pred" / f"{name}.npz"
        np.savez(str(f) + ".tmp.npz", navtest=nt, held=Pd.get("held", np.zeros((0, len(rp.OUTS)))), val=Pd["val"], val_rows=i_val, outs=np.array(rp.OUTS),
                 arm=a.arm, seed=a.seed, n_fit=len(i_fit), units_fit=len(set(logs[fit])), weights=str(run.path("best.pt")))
        os.replace(str(f) + ".tmp.npz", f)
        run.summary.update(pred=str(f), n_fit=len(i_fit), units_fit=len(set(logs[fit])))


# ---------------------------------------------------------------- read (AlpaSim venv, CPU)
def jbin(jd):
    return np.select([jd == 0, (jd > 0) & (jd <= 10), (jd > 10) & (jd <= 20), (jd > 20) & (jd <= 40), (jd > 40) & (jd <= 60)], list(BINS[:5]), BINS[5])


def curve(eL, eR, jd, cl, B, seed=0):
    """Pooled L / R 20 m error per junction bucket, near / far / ratio, with a cluster (scene / log segment) bootstrap."""
    b = jbin(jd)
    e = np.r_[eL, eR]
    bb, cc = np.r_[b, b], np.r_[cl, cl]
    ok = np.isfinite(e)
    e, bb, cc = e[ok], bb[ok], cc[ok]
    u, inv = np.unique(cc, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(u))]

    def stats(ix):
        ee, b_ = e[ix], bb[ix]
        s = {k: float(np.median(ee[b_ == k])) if (b_ == k).any() else np.nan for k in BINS}
        nm, fm = np.isin(b_, NEAR), np.isin(b_, FAR)
        s["near"] = float(np.median(ee[nm])) if nm.any() else np.nan
        s["far"] = float(np.median(ee[fm])) if fm.any() else np.nan
        s["ratio"] = s["near"] / s["far"] if s["far"] > 0 else np.nan
        return s
    pt = stats(np.arange(len(e)))
    rng = np.random.default_rng(seed)
    bs = [stats(np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))])) for _ in range(B)]
    out = {}
    for k in pt:
        v = np.array([x[k] for x in bs], float)
        out[k] = dict(v=pt[k], lo=float(np.nanquantile(v, 0.025)) if np.isfinite(v).any() else np.nan,
                      hi=float(np.nanquantile(v, 0.975)) if np.isfinite(v).any() else np.nan)
    for k in BINS:
        out[k]["n"] = int((bb == k).sum())
    out["near"]["n"], out["far"]["n"], out["clusters"] = int(np.isin(bb, NEAR).sum()), int(np.isin(bb, FAR).sum()), int(len(u))
    return out


def verdict(c):
    near, ratio = c["near"]["v"], c["ratio"]["v"]
    if not np.isfinite(near):
        return "no read"
    if near < 0.6:
        return "reconstruction artefact"
    return "same shape on real frames" if ratio > 2 else "undetermined"


def read(a):
    import pandas as pd
    from jevdrive.run import Run
    import rp
    with Run("road_probe", "real/read", config=vars(a)) as run:
        out = Path(run.dir)
        Dn = pd.read_parquet(R / "labels/navtest.parquet")
        Dh = pd.read_parquet(W / "labels/held.parquet")
        G = {f"G_frame_{s}": json.loads((R / f"labels/{s}.gates.json").read_text())["G_frame"] for s in ("navtest", "navtrain")}
        st, tu = (Dn.turn40 < 5).values, (Dn.turn40 >= 20).values
        G["G_lab"] = dict(straight=int(st.sum()), share=float((Dn.L20.notna() & Dn.R20.notna())[st].mean()))
        G["G_lab"]["pass"] = bool(G["G_lab"]["share"] >= 0.70)
        G["G_sign"] = dict(L20=float(Dn.L20[st].median()), R20=float(Dn.R20[st].median()))
        G["G_sign"]["pass"] = bool(G["G_sign"]["L20"] > 0 > G["G_sign"]["R20"])
        Pz = {f.stem: dict(np.load(f)) for f in sorted((R / "pred").glob("*.npz")) if "smoke" not in f.stem}
        Ah = 0.5 * (Pz["A-s0"]["held"] + Pz["A-s1"]["held"])
        th = (Dh.turn40 >= 20).values
        eh = np.abs(Ah[:, 4:6] - Dh[["L20", "R20"]].values)[th]
        G["G_repro"] = dict(held_turn20=float(np.nanmedian(eh)), stored=REF_TURN, seeds=[float(np.nanmedian(np.abs(Pz[f"A-s{s}"]["held"][:, 4:6] - Dh[["L20", "R20"]].values)[th]))
                                                                                       for s in (0, 1)])
        G["G_repro"]["pass"] = bool(abs(G["G_repro"]["held_turn20"] - REF_TURN) <= 0.05)
        G["pass"] = all(v["pass"] for v in G.values())
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        run.info(json.dumps(G, default=float))
        if not G["pass"] and not a.force:
            run.info("a gate failed: no registered read")
            return
        P = {"A": 0.5 * (Pz["A-s0"]["navtest"] + Pz["A-s1"]["navtest"]), "B": 0.5 * (Pz["B-s0"]["navtest"] + Pz["B-s1"]["navtest"]),
             **{k: v["navtest"] for k, v in Pz.items()}}
        sh, prob = rp.shipped("navtest", Dn)
        P["P0"], P["P0K"] = sh, rp.shipped("navtest", Dn, True)[0]
        hu = np.isin(Dn.unit.values, list(held_units()))
        cl = Dn.scene.values.astype(str)
        jd = Dn.jdist.values
        rows, curves = [], {}
        subsets = {"turn": tu, "straight": st, "all": np.ones(len(Dn), bool)}
        for k, p in P.items():
            for where, wm in (("navtest", np.ones(len(Dn), bool)), ("held units", hu)):
                if k.startswith("A") and where == "navtest":
                    continue
                for sub, sm in subsets.items():
                    m = wm & sm & np.isfinite(p[:, 4])
                    c = curve(np.abs(p[m, 4] - Dn.L20.values[m]), np.abs(p[m, 5] - Dn.R20.values[m]), jd[m], cl[m], a.boot if sub == "turn" else 200)
                    curves[(k, where, sub)] = c
                    rows += [dict(pred=k, frames="real", where=where, subset=sub, bin=b, n=c[b]["n"], median=c[b]["v"], lo=c[b]["lo"], hi=c[b]["hi"])
                             for b in BINS + ("near", "far")] + [dict(pred=k, frames="real", where=where, subset=sub, bin="ratio", n=c["clusters"],
                                                                     median=c["ratio"]["v"], lo=c["ratio"]["lo"], hi=c["ratio"]["hi"])]
        # the reconstructed-frame reference: decision 275's stored read, same quantities
        X = pd.read_parquet(REC_READ / "dec_held.parquet")
        assert (X.row.values == Dh.row.values).all()
        for k in ("P", "blind-p1000-s0", "P0", "P0K"):
            for sub, sm in (("turn", (X.turn40 >= 20).values), ("straight", (X.turn40 < 5).values), ("all", np.ones(len(X), bool))):
                c = curve(np.abs(X[f"{k}:L20"].values - X.L20.values)[sm], np.abs(X[f"{k}:R20"].values - X.R20.values)[sm], X.jdist.values[sm],
                          X.scene.values.astype(str)[sm], a.boot if sub == "turn" else 200)
                curves[(f"rec:{k}", "held", sub)] = c
                rows += [dict(pred=k, frames="reconstructed", where="held", subset=sub, bin=b, n=c[b]["n"], median=c[b]["v"], lo=c[b]["lo"], hi=c[b]["hi"])
                         for b in BINS + ("near", "far")] + [dict(pred=k, frames="reconstructed", where="held", subset=sub, bin="ratio", n=c["clusters"],
                                                                 median=c["ratio"]["v"], lo=c["ratio"]["lo"], hi=c["ratio"]["hi"])]
        pd.DataFrame(rows).to_csv(out / "buckets.csv", index=False)
        V = {"b_navtest": verdict(curves[("B", "navtest", "turn")]), "b_held_units": verdict(curves[("B", "held units", "turn")]),
             "a_held_units (descriptive)": verdict(curves[("A", "held units", "turn")]), "reconstructed (descriptive)": verdict(curves[("rec:P", "held", "turn")])}
        cnt = dict(navtest=len(Dn), held_units=int(hu.sum()), turn=int(tu.sum()), turn_held_units=int((tu & hu).sum()), straight=int(st.sum()),
                   segments=int(Dn.scene.nunique()), units=int(Dn.unit.nunique()), jdist_known=float(np.isfinite(jd).mean()),
                   lr20_eval=float((Dn.L20.notna() & Dn.R20.notna()).mean()), lr20_eval_turn=float((Dn.L20.notna() & Dn.R20.notna())[tu].mean()),
                   meta={k: dict(n_fit=int(v["n_fit"]), units_fit=int(v["units_fit"])) for k, v in Pz.items() if "n_fit" in v})
        keep = ["row", "token", "scene", "unit", "turn40", "jdist", "lane_cls", "L20", "R20"]
        D = Dn[keep].copy()
        D["held_unit"] = hu
        for k in ("A", "B", "blind-s0", "C-s0", "P0", "P0K"):
            if k in P:
                D[f"{k}:L20"], D[f"{k}:R20"] = P[k][:, 4], P[k][:, 5]
        D.to_parquet(out / "dec_navtest.parquet", index=False)
        S = dict(gates=G, verdict=V, counts=cnt, presence_P0_turn=float((prob[tu][:, 1:3] > 0.5).all(1).mean()),
                 curves={"|".join(k): v for k, v in curves.items()})
        (out / "rp_real.json").write_text(json.dumps(S, indent=1, default=float))
        run.summary.update(verdict=V)
        run.info(f"verdict {json.dumps(V)}")


# ---------------------------------------------------------------- report (Mac .venv)
LBL = {("rec:P", "held"): "reconstructed frames, probe (decision 275, held AlpaSim)",
       ("A", "held units"): "(a) same probe on real frames, same 16 logs",
       ("B", "navtest"): "(b) probe trained on navtrain, all navtest",
       ("B", "held units"): "(b) on the same 16 logs",
       ("blind-s0", "navtest"): "blind floor (navtrain, ego + command)",
       ("C-s0", "navtest"): "(c) navtrain, 27 units only (exploratory)",
       ("P0", "navtest"): "shipped heads, raw", ("P0K", "navtest"): "shipped heads, 0.689 undone",
       ("rec:blind-p1000-s0", "held"): "blind floor on reconstructed (decision 275)", ("rec:P0", "held"): "shipped heads on reconstructed, raw"}


def report(a):
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Rd = Path(a.read)
    S = json.loads((Rd / "rp_real.json").read_text())
    C = {tuple(k.split("|")): v for k, v in S["curves"].items()}
    shutil.copy(Rd / "buckets.csv", OUT / "rp_real_buckets.csv")
    shutil.copy(Rd / "gates.json", OUT / "rp_real_gates.json")
    f3 = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.2f}"  # noqa: E731
    cell = lambda c: f"{f3(c['v'])} [{f3(c['lo'])}, {f3(c['hi'])}]"  # noqa: E731
    # figure: turn decisions (main) and straight decisions, bucket curves
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    sty = {("rec:P", "held"): ("#d1495b", "-", "o"), ("A", "held units"): ("#edae49", "-", "s"), ("B", "navtest"): ("#00798c", "-", "D"),
           ("B", "held units"): ("#00798c", "--", "D"), ("blind-s0", "navtest"): ("#8d8d8d", ":", "v")}
    x = np.arange(len(BINS))
    for ax, sub in zip(axs, ("turn", "straight")):
        for j, (key, (col, ls, mk)) in enumerate(sty.items()):
            c = C.get(key + (sub,))
            if c is None:
                continue
            y = np.array([c[b]["v"] for b in BINS])
            lo, hi = np.array([c[b]["lo"] for b in BINS]), np.array([c[b]["hi"] for b in BINS])
            xx = x + (j - 2) * 0.06
            ax.errorbar(xx, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], color=col, ls=ls, marker=mk, ms=4, lw=1.4, capsize=2,
                        label=LBL[key])
        ax.axhline(0.6, color="k", lw=0.6, ls=(0, (2, 3)))
        ax.set_xticks(x, [b.replace("none within 60 m", "none <60 m") for b in BINS], fontsize=8)
        ax.set_title(f"{'turn decisions (route turns >= 20 deg in 40 m)' if sub == 'turn' else 'straight decisions (< 5 deg)'}", fontsize=10)
        ax.set_xlabel("route distance to the junction")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("ego-lane boundary error at 20 m, median (m)")
    axs[0].legend(fontsize=7.5, loc="upper right", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"rp_real.{ext}", dpi=150)
    # tables
    G, V, N = S["gates"], S["verdict"], S["counts"]
    L = ["# road_probe, real frames: is the junction blur in the frozen encoder or in the reconstructed images?", "",
         "Pre-registration: [plans/2026-10-12-road-probe-prereg.md](../plans/2026-10-12-road-probe-prereg.md) section 8 (pushed before any real-frame label, "
         "training or read). Script `scripts/rp_real.py`; box `$DATA_DIR/runs/road_probe/real/` (labels, pred, read "
         f"`{Rd.name}`). Figure [figs/rp_real.png](../figs/rp_real.png).", "",
         "## Gates", ""]
    L += [f"- {k}: " + ", ".join(f"{kk} {f3(vv) if isinstance(vv, float) else vv}" for kk, vv in v.items()) for k, v in G.items() if isinstance(v, dict)]
    L += ["", f"navtest {N['navtest']} tokens ({N['segments']} log segments, {N['units']} drive logs), {N['turn']} turn decisions, of which "
          f"{N['turn_held_units']} in the 16 held drive logs (the same roads as the reconstructed held read); junction distance known for "
          f"{f3(N['jdist_known'])} of all tokens; both 20 m boundaries defined on {f3(N['lr20_eval_turn'])} of the turns. Fit sizes: "
          + ", ".join(f"{k} {v['n_fit']} rows / {v['units_fit']} units" for k, v in N["meta"].items()) + ".", "", "## Turn decisions: error at 20 m by distance to the junction", ""]
    cols = ["predictor"] + list(BINS) + ["near (in + 0-20)", "far (20-60)", "near / far"]
    L += ["| " + " | ".join(cols) + " |", "|:--" + "|--:" * (len(cols) - 1) + "|"]
    for key in LBL:
        c = C.get(key + ("turn",))
        if c is None:
            continue
        L.append("| " + " | ".join([LBL[key]] + [f"{f3(c[b]['v'])} ({c[b]['n']})" for b in BINS] + [cell(c["near"]), cell(c["far"]), cell(c["ratio"])]) + " |")
    L += ["", "Cells: median error in m (pooled L / R errors in the bucket); brackets: 95% CI from resampling log segments (real) or scenes "
          "(reconstructed), B = 2000.", "", "## Straight decisions (descriptive)", ""]
    L += ["| " + " | ".join(cols) + " |", "|:--" + "|--:" * (len(cols) - 1) + "|"]
    for key in LBL:
        c = C.get(key + ("straight",))
        if c is None:
            continue
        L.append("| " + " | ".join([LBL[key]] + [f"{f3(c[b]['v'])} ({c[b]['n']})" for b in BINS] + [cell(c["near"]), cell(c["far"]), cell(c["ratio"])]) + " |")
    L += ["", "## Verdict (registered: (b) on all navtest turn decisions)", "",
          "Line: near < 0.6 m -> reconstruction artefact; else near / far > 2 -> same shape on real frames (the encoder); else undetermined.", ""]
    L += [f"- {k}: **{v}**" for k, v in V.items()]
    (OUT / "rp_real.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("labels")
    p.add_argument("--split", required=True, choices=("navtest", "navtrain")), p.add_argument("--workers", type=int, default=0), p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("train")
    p.add_argument("--arm", required=True, choices=("A", "B", "blind", "C", "P0")), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8000), p.add_argument("--batch", type=int, default=256), p.add_argument("--eval", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4), p.add_argument("--wd", type=float, default=0.05), p.add_argument("--warm", type=int, default=300)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--boot", type=int, default=2000), p.add_argument("--force", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--read", required=True)
    a = ap.parse_args()
    dict(labels=labels, train=train, read=read, report=report)[a.cmd](a)
