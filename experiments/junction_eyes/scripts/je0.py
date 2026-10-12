#!/usr/bin/env python3
"""junction_eyes step 0 (plans/2026-10-12-prereg-A.md): new rulers R-A / R-B / R-C on the FROZEN Cinque tokens, Z-0 line.

  rows    (AlpaSim venv, CPU)  row tables J_nav / J_on / val / P / navtest / held / zero + G-split -> $W/rows/*.parquet (results/rows.md)
  labels  (AlpaSim venv, CPU)  R-A (n, 30) and R-B (n, 48, 48) for every row of rows/all.parquet -> $W/labels/{ra.npy, rb.npy, meta.parquet}
  glab    (AlpaSim venv, CPU)  G-lab (iii): navsim-side labels against the AlpaSim scene loader's map + rig path on 20 held scenes
  wa-rows (any python)         rows files of the J_nav tokens missing a WA-Cf feature, per navtrain shard -> $W/wa/rows/
  probe   (envs/op-train, GPU) --ruler A|B|old --arm Z|blind|wa|s7 --seed S -> $W/pred/<ruler>-<arm>-s<S>.npz (pool job JEP-...)
  read    (AlpaSim venv, CPU)  gates, R-C, bucket tables with cluster-bootstrap CIs, Z-0 verdict -> $W/read/<ts>/
  report  (Mac .venv)          --read DIR -> results/step0.md, results/*.csv, figs/je0_ra.png, figs/je0_rb.png

Frames: labels in the decision's rear-axle frame, x forward, y LEFT (lib/je_ruler.py). AlpaSim scene frames are the world frame shifted by
the logged position of navsim frame (scene token index - 3), no rotation, time base = that frame's timestamp (amendment 2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("rows", "labels", "glab", "read"):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/junction_eyes/lib"), str(REPO / "experiments/road_probe/scripts"),
                str(REPO / "experiments/corridor/scripts")]
TOPIC = REPO / "experiments/junction_eyes"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = Path(os.environ.get("JE0_W", DATA / "runs/junction_eyes/step0"))
RP = DATA / "runs/road_probe"
CR = DATA / "runs/op_parity/cache"
LOGS = DATA / "datasets/navsim/navsim_logs"
ONROWS = DATA / "runs/onpol_nuplan/rows"
WA = DATA / "runs/op_probe/feats/WA"
NSH = 12
ON_SETS = ("L18Y", "L18YS")                    # d275's train sets in J_on; L18P (P2H10S-F-s0) is not (amendment 1 item 1)
ON_ALL = ("L18Y", "L18YS", "L18P")             # row order of road_probe/labels/train.parquet
BINS = ("in junction", "0-10 m", "10-20 m", "20-40 m", "40-60 m", "none within 60 m")
NEAR, FAR = BINS[:3], BINS[3:5]


def logu(s):
    return "_".join(str(s).split("_")[:2])


def h(salt, x):
    return int(hashlib.sha256(f"{salt}|{x}".encode()).hexdigest(), 16)


def seglog(scene):
    """AlpaSim scene id -> (navsim log segment, start token)."""
    a, b = str(scene).rsplit("-", 1)
    return a, b


# ---------------------------------------------------------------- rows (CPU)
def rows(a):
    import pandas as pd
    import rp
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("junction_eyes", "step0/rows", config=vars(a)) as run:
        for s in ("navsim/navtrain", "navsim/navtest", "alpasim_nuplan/onpol-train", "alpasim_nuplan/onpol-held"):
            run.use_split(splits.load(s))
        held_u = {logu(s) for s in splits.load("alpasim_nuplan/onpol-held").members}
        T = []
        # J_nav: navtrain tokens (real/labels/navtrain.parquet is in navtrain_full cache order)
        n = pd.read_parquet(RP / "real/labels/navtrain.parquet", columns=["row", "scene", "unit", "token", "jdist", "turn40"])
        off = np.cumsum([0] + [len(np.load(CR / f"navtrain_full.s{k}of{NSH}/tab.npz")["names"]) for k in range(NSH)])
        cand = ((n.jdist <= 60) | (n.turn40 >= 10)).values
        keep = cand | (~cand & (n.turn40 < 5).values)
        n = n[keep].copy()
        sh = np.searchsorted(off, n.row.values, side="right") - 1
        T.append(pd.DataFrame(dict(set="J_nav", source="navtrain", scene=n.scene.values, log=n.scene.values, unit=n.unit.values,
                                   token=n.token.values, tag="log", k=-1, jdist=n.jdist.values, turn40=n.turn40.values, cand=cand[keep],
                                   tok_file=[f"navtrain_full.s{k}of{NSH}@warp" for k in sh], tok_row=n.row.values - off[sh],
                                   tab_file=[f"navtrain_full.s{k}of{NSH}" for k in sh], old_lab="real/labels/navtrain", old_row=n.row.values)))
        # J_on: on-policy rows of L18Y / L18YS (labels/train.parquet rows follow L18Y, L18YS, L18P)
        o = pd.read_parquet(RP / "labels/train.parquet", columns=["row", "scene", "src", "tag", "k", "jdist", "turn40"])
        offo = dict(zip(ON_ALL, np.cumsum([0] + [len(np.load(CR / f"on1_{s}/tab.npz")["names"]) for s in ON_ALL])[:-1]))
        o = o[o.src.isin(ON_SETS)]
        cand = ((o.jdist <= 60) | (o.turn40 >= 10)).values
        o = o[cand].copy()
        meta = {s: dict(np.load(ONROWS / s / "meta.npz", allow_pickle=True)) for s in ON_SETS}
        lr = o.row.values - np.array([offo[s] for s in o.src])
        names = np.array([meta[s]["names"][i] for s, i in zip(o.src, lr)])
        tlog = np.array([int(meta[s]["t_log_us"][i]) for s, i in zip(o.src, lr)])
        dump = np.array([str(meta[s]["dump"][i]) for s, i in zip(o.src, lr)])
        assert all(str(meta[s]["scene"][i]) == sc for s, i, sc in zip(o.src, lr, o.scene)), "on1 meta and labels disagree"
        T.append(pd.DataFrame(dict(set="J_on", source="onpol", scene=o.scene.values, log=[seglog(s)[0] for s in o.scene], unit=[logu(s) for s in o.scene],
                                   token=names, tag=o.tag.values, k=o.k.values, jdist=o.jdist.values, turn40=o.turn40.values, cand=True,
                                   tok_file=[f"on1_{s}@warp" for s in o.src], tok_row=lr, tab_file=[f"on1_{s}" for s in o.src], old_lab="labels/train",
                                   old_row=o.row.values, t_log_us=tlog, dump=dump)))
        # navtest (all tokens; navtest-16 = the held drive logs)
        x = pd.read_parquet(RP / "real/labels/navtest.parquet", columns=["row", "scene", "unit", "token", "jdist", "turn40"])
        T.append(pd.DataFrame(dict(set=np.where(x.unit.isin(held_u), "navtest16", "navtest_rest"), source="navtest", scene=x.scene.values,
                                   log=x.scene.values, unit=x.unit.values, token=x.token.values, tag="log", k=-1, jdist=x.jdist.values,
                                   turn40=x.turn40.values, cand=False, tok_file="lb_navtest@warp", tok_row=x.row.values, tab_file="lb_navtest",
                                   old_lab="real/labels/navtest", old_row=x.row.values)))
        # held-452 and zero (served tokens of the 98 windows' rollouts)
        for g in ("held", "zero"):
            d = pd.read_parquet(RP / f"labels/{g}.parquet", columns=["row", "scene", "k", "tag", "jdist", "turn40"])
            tb = dict(np.load(RP / f"tok/{g}/tab.npz"))
            assert len(d) == len(tb["k"]) and (d.row.values == np.arange(len(d))).all()
            st = np.full(len(d), "held452" if g == "held" else "zero_rest", dtype=object)
            if g == "zero":
                for e in rp.zero_events():
                    kv = int(e["k_ev"])
                    st[((d.tag == e["group"]) & (d.scene == e["scene"]) & (d.k >= max(0, kv - 4)) & (d.k <= kv)).values] = "zero98"
            T.append(pd.DataFrame(dict(set=st, source=g, scene=d.scene.values, log=[seglog(s)[0] for s in d.scene], unit=[logu(s) for s in d.scene],
                                       token=[f"{g}:{t}:{s}:{k}" for t, s, k in zip(d.tag, d.scene, d.k)], tag=d.tag.values, k=d.k.values,
                                       jdist=d.jdist.values, turn40=d.turn40.values, cand=False, tok_file=f"road_probe/tok/{g}", tok_row=d.row.values,
                                       tab_file=f"road_probe/tok/{g}", old_lab=f"labels/{g}", old_row=d.row.values)))
        D = pd.concat(T, ignore_index=True)
        D.insert(0, "rid", np.arange(len(D)))
        # val / fit / read
        split = np.full(len(D), "none", dtype=object)
        jn, jo = (D.set == "J_nav").values, (D.set == "J_on").values
        split[jn] = np.where([h("jeval", u) % 10 == 0 for u in D.unit[jn]], "val", "fit")
        ou = sorted(set(D.unit[jo]), key=lambda u: h("jeval", u))
        vu = set(ou[:3])
        split[jo] = np.where(D.unit[jo].isin(vu), "val", "fit")
        split[D.set.isin(["navtest16", "held452", "zero98"]).values] = "read"
        D["split"] = split
        D["w_scene"] = 1.0
        D.loc[jo, "w_scene"] = 1.0 / D[jo].groupby("scene").rid.transform("size").values
        D["t_log_us"] = D["t_log_us"].fillna(-1).astype(np.int64)
        D["dump"] = D["dump"].fillna("")
        # G-split
        J, R_ = D[D.split.isin(["fit", "val"])], D[D.split == "read"]
        g = {}
        for lev in ("token", "scene", "log", "unit"):
            x = set(J[lev]) & set(R_[lev])
            g[lev] = dict(overlap=len(x), first=sorted(x)[:3])
        g["pass"] = all(v["overlap"] == 0 for v in g.values())
        W.joinpath("rows").mkdir(parents=True, exist_ok=True)
        D.to_parquet(W / "rows/all.parquet", index=False)
        views = {"J_nav": D.set == "J_nav", "J_on": D.set == "J_on", "val": D.split == "val", "P": D.split == "fit",
                 "navtest16": D.set == "navtest16", "held452": D.set == "held452", "zero98": D.set == "zero98"}
        for k, m in views.items():
            D[m.values].to_parquet(W / f"rows/{k}.parquet", index=False)
        cnt = {k: dict(rows=int(m.sum()), scenes=int(D[m.values].scene.nunique()), units=int(D[m.values].unit.nunique())) for k, m in views.items()}
        S = dict(G_split=g, counts=cnt, val_units_on=sorted(vu), n=len(D))
        (W / "rows/rows.json").write_text(json.dumps(S, indent=1, default=str))
        run.summary.update(G_split=g["pass"], counts=cnt)
        run.info(json.dumps(S, default=str))


# ---------------------------------------------------------------- labels (CPU)
_VM = {}


def vmap(loc):
    import rp_real as RR
    if loc not in _VM:
        from trajdata.maps.map_api import MapAPI
        _VM[loc] = MapAPI(RR.MAPC).get_map(f"nuplan_test:{RR.MAPS[loc]}", **RR.VMP)
    return _VM[loc]


def log_frames(log, kind):
    import pickle
    import rp_real as RR
    fr = pickle.load(open(LOGS / kind / f"{log}.pkl", "rb"))
    T = np.array([f["timestamp"] for f in fr], float)
    P = np.array([f["ego2global_translation"][:2] for f in fr], float)
    Y = np.unwrap([RR.yaw_of(f["ego2global_rotation"]) for f in fr])
    return fr, T, P, Y


def lab_job(job):
    """(log segment, kind, items) -> list of (rid, ra (30,), rb (48, 48), s0, s_cam, frame_err). items: dict(rid, token) for navsim tokens,
    dict(rid, scene, t_rel, anchor (local), cam) for AlpaSim rows."""
    import je_ruler as JR
    import rp_real as RR
    log, kind, items = job
    fr, T, P, Y = log_frames(log, kind)
    pos = {f["token"]: i for i, f in enumerate(fr)}
    Td, Pd, _ = RR.densify(T, P, Y)
    LP, S, H, TT = JR.logged_path(Td, Pd)
    rows_ = []
    for it in items:
        if "scene" in it:
            i0 = pos[seglog(it["scene"])[1]] - 3
            assert i0 >= 0, it["scene"]
            a = np.asarray(it["anchor"], float).copy()
            a[:2] += P[i0]
            t = T[i0] + it["t_rel"]
            cam = np.asarray(it["cam"], float)
        else:
            i = pos[it["token"]]
            a, t, cam = np.r_[P[i], Y[i]], T[i], np.asarray(it["cam"], float)
        rows_.append((it["rid"], a, t, cam))
    A = np.array([r[1][:2] for r in rows_])
    lo, hi = np.minimum(LP.min(0), A.min(0)) - 100, np.maximum(LP.max(0), A.max(0)) + 100
    m = JR.LocalMap(vmap(fr[0]["map_location"]), lo, hi)
    rt = m.route(LP, S, H)
    out = []
    for rid, a, t, cam in rows_:
        ra, s0, sc = JR.ra_label(rt, LP, S, TT, a, cam, t)
        fe = float(np.hypot(*(a[:2] - [np.interp(t, Td, Pd[:, 0]), np.interp(t, Td, Pd[:, 1])])))
        out.append((rid, ra.reshape(-1), m.rb_label(a), s0, sc, fe))
    return out


def lab_items(D):
    """rows table -> {(log, kind): [items]}."""
    by = {}
    tabs = {}

    def tab(f):
        if f not in tabs:
            p = CR / f / "tab.npz" if not f.startswith("road_probe") else DATA / "runs" / f / "tab.npz"
            tabs[f] = dict(np.load(p))
        return tabs[f]
    for r in D.itertuples(index=False):
        tb = tab(r.tab_file)
        cam = tb["cam"][r.tok_row].astype(float)
        if r.source in ("navtrain", "navtest"):
            assert str(tb["names"][r.tok_row]) == r.token
            it = dict(rid=int(r.rid), token=r.token, cam=cam)
            key = (r.log, "trainval" if r.source == "navtrain" else "test")
        else:
            if r.source == "onpol":
                M = tab(r.tab_file)
                mm = _onmeta(r.tab_file)
                anchor, t_rel = mm["anchor"][r.tok_row], float(mm["t_us"][r.tok_row])
                assert str(mm["names"][r.tok_row]) == r.token and str(M["names"][r.tok_row]) == r.token
            else:
                anchor, t_rel = tb["anchor"][r.tok_row], float(tb["t0"][r.tok_row])
                assert str(tb["scene"][r.tok_row]) == r.scene and int(tb["k"][r.tok_row]) == r.k
            it = dict(rid=int(r.rid), scene=r.scene, t_rel=t_rel, anchor=np.asarray(anchor, float), cam=cam)
            key = (r.log, "test")
        by.setdefault(key, []).append(it)
    return by


_OM = {}


def _onmeta(tab_file):
    s = tab_file.split("_", 1)[1]
    if s not in _OM:
        _OM[s] = dict(np.load(ONROWS / s / "meta.npz", allow_pickle=True))
    return _OM[s]


def labels(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    with Run("junction_eyes", "step0/labels", config=vars(a)) as run:
        D = pd.read_parquet(W / "rows/all.parquet")
        if a.limit:
            D = D[D.log.isin(sorted(D.log.unique(), key=lambda l: h("jesmoke", l))[: a.limit])]
        by = lab_items(D)
        jobs = sorted(((l, k, v) for (l, k), v in by.items()), key=lambda j: -len(j[2]))
        run.info(f"{len(D)} rows, {len(jobs)} log segments")
        res = par.pmap(lab_job, jobs, workers=a.workers or n_cpus(), run=run, desc="segments")
        N = int(D.rid.max()) + 1 if not a.limit else len(D)
        idx = {r: i for i, r in enumerate(D.rid.values)} if a.limit else None
        ra, rb = np.full((N, 30), np.nan, np.float32), np.zeros((N, 48, 48), np.uint8)
        meta = np.full((N, 4), np.nan)
        got = np.zeros(N, bool)
        for v in res.values:
            for rid, x, b, s0, sc, fe in v or []:
                i = idx[rid] if idx else rid
                ra[i], rb[i], meta[i], got[i] = x, b, (rid, s0, sc, fe), True
        sfx = "-smoke" if a.limit else ""
        (W / "labels").mkdir(parents=True, exist_ok=True)
        np.save(W / f"labels/ra{sfx}.npy", ra)
        np.save(W / f"labels/rb{sfx}.npy", rb)
        pd.DataFrame(dict(rid=D.rid.values if a.limit else np.arange(N), ok=got, s0=meta[:, 1], s_cam=meta[:, 2], frame_err=meta[:, 3])).to_parquet(
            W / f"labels/meta{sfx}.parquet", index=False)
        g = dict(jobs=len(jobs), errors=len(res.errors), err_first=[res.errors[i].strip().splitlines()[-1][:300] for i in sorted(res.errors)[:5]],
                 rows=int(got.sum()), of=len(D), frame_err_p99=float(np.nanquantile(meta[:, 3], 0.99)))
        (W / f"labels/labels{sfx}.json").write_text(json.dumps(g, indent=1, default=float))
        run.summary.update(g)
        run.info(json.dumps(g, default=float))


# ---------------------------------------------------------------- glab: the AlpaSim scene loader against the navsim-side labels
def glab_job(scene):
    import je_ruler as JR
    import pandas as pd
    import ev as EV
    from ao_e1 import loader
    log, tok = seglog(scene)
    fr, T, P, Y = log_frames(log, "test")
    i0 = [i for i, f in enumerate(fr) if f["token"] == tok][0] - 3
    ds = loader(str(EV.CFG)).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2]
    LP, S, H, TT = JR.logged_path(Tg, Pg)
    D = pd.read_parquet(W / "rows/all.parquet", columns=["rid", "source", "token", "log"])
    D = D[(D.source == "navtest") & (D.log == log)]
    pos = {f["token"]: i for i, f in enumerate(fr)}
    tb = np.load(CR / "lb_navtest/tab.npz")
    cam_of = dict(zip(tb["names"], tb["cam"]))
    ra_all = np.load(W / "labels/ra.npy", mmap_mode="r")
    rb_all = np.load(W / "labels/rb.npy", mmap_mode="r")
    items = []
    for rid, t_ in zip(D.rid, D.token):
        i = pos[t_]
        tr_ = T[i] - T[i0]
        if 0 <= tr_ <= Tg[-1]:
            a = np.r_[P[i] - P[i0], Y[i]]
            items.append((int(rid), a, tr_, np.asarray(cam_of[t_], float)))
    if not items:
        return []
    A = np.array([x[1][:2] for x in items])
    m = JR.LocalMap(ds.map, np.minimum(LP.min(0), A.min(0)) - 100, np.maximum(LP.max(0), A.max(0)) + 100)
    rt = m.route(LP, S, H)
    out = []
    for rid, a, t_, cam in items:
        ra, _, _ = JR.ra_label(rt, LP, S, TT, a, cam, t_)
        rb = m.rb_label(a)
        out.append(dict(scene=scene, rid=rid, ra_alp=ra.reshape(-1), ra_nav=np.asarray(ra_all[rid]), rb_agree=float((rb == rb_all[rid]).mean())))
    return out


def glab(a):
    import pandas as pd
    import rp  # noqa: F401  (AlpaSim / zeros_mech paths for the scene loader)
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("junction_eyes", "step0/glab", config=vars(a)) as run:
        held = splits.load("alpasim_nuplan/onpol-held")
        run.use_split(held)
        sc = [s for s in sorted(held.members) if (LOGS / "test" / f"{seglog(s)[0]}.pkl").exists()][:20]
        res = par.pmap(glab_job, sc, workers=20, run=run, desc="scenes")
        res.raise_if_failed()
        X = [x for v in res.values for x in v]
        d = np.concatenate([np.hypot(*(x["ra_alp"].reshape(-1, 2) - x["ra_nav"].reshape(-1, 2)).T) for x in X])
        g = dict(scenes=len(sc), tokens=len(X), points=int(np.isfinite(d).sum()), median=float(np.nanmedian(d)), p90=float(np.nanquantile(d, 0.9)),
                 both_defined_share=float(np.isfinite(d).mean()), rb_agree=float(np.mean([x["rb_agree"] for x in X])))
        g["pass"] = bool(g["median"] < 0.2)
        (W / "labels/glab.json").write_text(json.dumps(g, indent=1))
        pd.DataFrame([dict(scene=x["scene"], rid=x["rid"], rb_agree=x["rb_agree"]) for x in X]).to_parquet(W / "labels/glab.parquet", index=False)
        run.summary.update(g)
        run.info(json.dumps(g))


# ---------------------------------------------------------------- WA-Cf rows to extract
def wa_have():
    """{navtrain shard: {row: (file, index)}} of the cached WA-Cf features."""
    have = {}
    for f in sorted(WA.glob("navtrain_full.s*of12*.npz")):
        sh = f.name.split(".")[1]
        z = np.load(f, mmap_mode="r")
        for j, r in enumerate(z["rows"]):
            have.setdefault(sh, {})[int(r)] = (str(f), j)
    return have


def wa_rows(a):
    import pandas as pd
    D = pd.read_parquet(W / "rows/all.parquet", columns=["set", "split", "tok_file", "tok_row"])
    D = D[(D.set == "J_nav") & D.split.isin(["fit", "val"])]
    have = wa_have()
    (W / "wa/rows").mkdir(parents=True, exist_ok=True)
    tot = 0
    for k in range(NSH):
        sh = f"s{k}of{NSH}"
        r = np.sort(D.tok_row[D.tok_file == f"navtrain_full.{sh}@warp"].values)
        miss = r[~np.isin(r, list(have.get(sh, {})))]
        if len(miss):
            np.save(W / f"wa/rows/navtrain_full.{sh}.npy", miss)
        tot += len(miss)
        print(sh, len(r), "missing", len(miss))
    print("total missing", tot)


# ---------------------------------------------------------------- probes (GPU)
def load_tok(D, dev, slot7_only=False, threads=8):
    """(len(D), 8, 32, 512) fp16 on the device for the rows of D (columns tok_file, tok_row)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    V = torch.zeros((len(D), 8, 32, 512), dtype=torch.float16, device=dev)
    jobs = []
    for f, g in D.reset_index(drop=True).groupby("tok_file"):
        p = DATA / "runs" / f / "front.npy" if f.startswith("road_probe") else CR / f / "front.npy"
        o, r = g.index.values, g.tok_row.values
        jobs += [(p, r[c:c + 1024], o[c:c + 1024]) for c in range(0, len(r), 1024)]

    def one(j):
        p, r, o = j
        srt = np.argsort(r)
        x = np.ascontiguousarray(np.load(p, mmap_mode="r")[r[srt]])
        return o[srt], x
    with ThreadPoolExecutor(threads) as ex:
        for o, x in ex.map(one, jobs):
            V[torch.as_tensor(o, device=dev)] = torch.from_numpy(x).to(dev)
    if slot7_only:
        V[:, :7] = 0
    return V


def load_wa(D, dev):
    """WA-Cf tokens (32 x 512) of navtrain / navtest rows -> (len(D), 8, 32, 512) fp16 with slot 7 = Cf, other slots 0."""
    import torch
    idx = {}
    for f in sorted(WA.glob("*.npz")):
        if f.name.startswith("navtrain_full.s") or (f.name.startswith("lb_navtest.s") and "of3" in f.name):
            src = f.name.split(".s")[0] if f.name.startswith("lb_") else "navtrain_full." + f.name.split(".")[1]
            for j, r in enumerate(np.load(f, mmap_mode="r")["rows"]):
                idx[(src, int(r))] = (f, j)
    V = torch.zeros((len(D), 8, 32, 512), dtype=torch.float16, device=dev)
    miss = 0
    by = {}
    for i, (tf, tr) in enumerate(zip(D.tok_file, D.tok_row)):
        key = (tf.replace("@warp", ""), int(tr))
        if key not in idx:
            miss += 1
            continue
        f, j = idx[key]
        by.setdefault(f, []).append((i, j))
    for f, ij in by.items():
        z = np.load(f)["Cf"]
        ij = np.array(ij)
        V[torch.as_tensor(ij[:, 0], device=dev), 7] = torch.from_numpy(z[ij[:, 1]].astype(np.float16)).to(dev)
    return V, miss


def ego_of(D):
    tabs, out = {}, np.zeros((len(D), 20), np.float32)
    for i, (f, r) in enumerate(zip(D.tab_file, D.tok_row)):
        if f not in tabs:
            p = DATA / "runs" / f / "tab.npz" if f.startswith("road_probe") else CR / f / "tab.npz"
            tabs[f] = np.load(p)["ego"]
        out[i] = tabs[f][r]
    return out


def probe(a):
    import time
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import head1_train as H1
    import rp
    from jevdrive.data import splits
    from jevdrive.run import Run
    name = f"{a.ruler}-{a.arm}-s{a.seed}" + ("-smoke" if a.smoke else "")
    with Run("junction_eyes", f"step0/probe-{name}", seed=a.seed, config=vars(a)) as run:
        for s in ("navsim/navtrain", "navsim/navtest", "alpasim_nuplan/onpol-train", "alpasim_nuplan/onpol-held"):
            run.use_split(splits.load(s))
        dev = torch.device("cuda")
        D = pd.read_parquet(W / "rows/all.parquet")
        nav_only = a.arm in ("wa", "s7")
        tr = D[D.split.isin(["fit", "val"]) & ((D.set == "J_nav") if nav_only else True)].reset_index(drop=True)
        rd = D[D.set.isin(["navtest16", "navtest_rest"] if nav_only else ["navtest16", "navtest_rest", "held452", "zero98", "zero_rest"])].reset_index(drop=True)
        assert not set(tr.token) & set(rd.token) and not set(tr.unit) & set(rd[rd.split == "read"].unit), "a read row in the probe rows"
        if a.smoke:
            tr = tr.sample(n=6000, random_state=0).reset_index(drop=True)
            rd = rd.sample(n=2000, random_state=0).reset_index(drop=True)
        fit, val = (tr.split == "fit").values, (tr.split == "val").values
        i_fit, i_val = np.flatnonzero(fit), np.flatnonzero(val)
        # targets
        if a.ruler == "A":
            Y = np.load(W / "labels/ra.npy", mmap_mode="r")[tr.rid.values].astype(np.float32)
            mu, sd = np.nanmean(Y[i_fit], 0), np.nanstd(Y[i_fit], 0) + 1e-6
            Yt = (Y - mu) / sd
            nq, nc = 15, 2
        elif a.ruler == "B":
            Y = np.load(W / "labels/rb.npy", mmap_mode="r")[tr.rid.values]
            Yt = np.ascontiguousarray(Y.reshape(len(Y), 6, 8, 6, 8).transpose(0, 1, 3, 2, 4).reshape(len(Y), 36, 64))   # (n, block, cell) uint8
            nq, nc = 36, 192
        else:                                                             # old ruler: rp.OUTS on the decision 275 / rp_real labels
            parts = []
            for f, g in tr.groupby("old_lab", sort=False):
                L = pd.read_parquet(RP / f"{f}.parquet")
                parts.append((g.index.values, rp.targets(L.iloc[g.old_row.values])))
            Y = np.zeros((len(tr), len(rp.OUTS)), np.float32)
            for o, y in parts:
                Y[o] = y
            mu, sd = np.nanmean(Y[i_fit, :rp.NREG], 0), np.nanstd(Y[i_fit, :rp.NREG], 0) + 1e-6
            Yt = Y.copy()
            Yt[:, :rp.NREG] = (Y[:, :rp.NREG] - mu) / sd
            nq, nc = len(rp.OUTS), 1
        t0 = time.time()
        vision = a.arm != "blind"
        V = {}
        if vision:
            if a.arm == "wa":
                V["tr"], m1 = load_wa(tr, dev)
                V["rd"], m2 = load_wa(rd, dev)
                assert m1 == 0 and m2 == 0, f"WA-Cf features missing: {m1} probe rows, {m2} read rows"
            else:
                V["tr"], V["rd"] = load_tok(tr, dev, a.arm == "s7"), load_tok(rd, dev, a.arm == "s7")
        E_tr, E_rd = ego_of(tr), ego_of(rd)
        emu, esd = E_tr[i_fit].mean(0), E_tr[i_fit].std(0) + 1e-6
        G = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        E = {"tr": G((E_tr - emu) / esd), "rd": G((E_rd - emu) / esd)}
        run.info("rows: fit %d, val %d, read %d; tokens in %.0f s", len(i_fit), len(i_val), len(rd), time.time() - t0)
        keep = torch.arange(224, 256, device=dev) if a.arm in ("wa", "s7") else None
        if a.ruler == "B":
            Yg = torch.as_tensor(Yt, device=dev)

            def loss_of(o, i):
                return F.cross_entropy(o.view(len(i), 36, 64, 3).reshape(-1, 3), Yg[i].reshape(-1).long())
        else:
            nreg = Yt.shape[1] if a.ruler == "A" else rp.NREG
            Yg, ok = torch.as_tensor(np.nan_to_num(Yt), device=dev), torch.as_tensor(np.isfinite(Yt), device=dev).float()

            def loss_of(o, i):
                o = o.reshape(len(i), -1)
                y, m = Yg[i], ok[i]
                reg = (F.huber_loss(o[:, :nreg], y[:, :nreg], delta=1.0, reduction="none") * m[:, :nreg]).sum() / m[:, :nreg].sum().clamp_min(1)
                if a.ruler == "A":
                    return reg
                cls = (F.binary_cross_entropy_with_logits(o[:, nreg], y[:, nreg], reduction="none") * m[:, nreg]).sum() / m[:, nreg].sum().clamp_min(1)
                return reg + 0.2 * cls
        torch.manual_seed(a.seed)
        net = H1.build(vision, nq=nq, nc=nc).to(dev)
        fwd = lambda g, i: net(V[g][i] if vision else None, E[g][i], keep=keep)  # noqa: E731
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        steps, warm, ev = (60, 10, 30) if a.smoke else (a.steps, a.warm, a.eval)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        gf, gv = torch.as_tensor(i_fit, device=dev), torch.as_tensor(i_val, device=dev)
        run.info("%s: %.3f M parameters", name, sum(x.numel() for x in net.parameters()) / 1e6)
        best, best_v, curve, t1 = None, np.inf, [], time.time()
        for step in run.tqdm(range(steps), desc=name):
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, step / steps)))
            b = gf[torch.randint(len(gf), (a.batch,), device=dev, generator=gen)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = fwd("tr", b).float()
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
                        num, den = num + float(loss_of(fwd("tr", i).float(), i)) * len(i), den + len(i)
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
                    o = fwd(g, torch.as_tensor(idx[c:c + 512], device=dev)).float()
                    if a.ruler == "B":
                        o = torch.softmax(o.view(len(o), 6, 6, 8, 8, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(o), 48, 48, 3), -1).half()
                    O.append(o.cpu().numpy())
            return np.concatenate(O)
        P_rd, P_val = pred("rd", np.arange(len(rd))), pred("tr", i_val)
        if a.ruler == "A":
            P_rd, P_val = P_rd.reshape(len(P_rd), -1) * sd + mu, P_val.reshape(len(P_val), -1) * sd + mu
        elif a.ruler == "old":
            P_rd, P_val = P_rd[..., 0], P_val[..., 0]
            P_rd[:, :rp.NREG] = P_rd[:, :rp.NREG] * sd + mu
            P_val[:, :rp.NREG] = P_val[:, :rp.NREG] * sd + mu
        (W / "pred").mkdir(parents=True, exist_ok=True)
        f = W / "pred" / f"{name}.npz"
        np.savez(str(f) + ".tmp.npz", rid=rd.rid.values, pred=P_rd, val_rid=tr.rid.values[i_val], val=P_val, ruler=a.ruler, arm=a.arm, seed=a.seed,
                 n_fit=len(i_fit), n_val=len(i_val), best_val=best_v)
        os.replace(str(f) + ".tmp.npz", f)
        torch.save(best, run.path("best.pt"))
        run.path("curve.json").write_text(json.dumps(curve, indent=1))
        run.summary.update(pred=str(f), n_fit=len(i_fit), best_step=int(min(curve, key=lambda c: c["val"])["step"]), best_val=best_v,
                           its=steps / max(time.time() - t1, 1e-6), vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows")
    p = sub.add_parser("labels")
    p.add_argument("--workers", type=int, default=0), p.add_argument("--limit", type=int, default=0)
    sub.add_parser("glab")
    sub.add_parser("wa-rows")
    p = sub.add_parser("probe")
    p.add_argument("--ruler", required=True, choices=("A", "B", "old")), p.add_argument("--arm", default="Z", choices=("Z", "blind", "wa", "s7"))
    p.add_argument("--seed", type=int, default=0), p.add_argument("--steps", type=int, default=8000), p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval", type=int, default=500), p.add_argument("--lr", type=float, default=3e-4), p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warm", type=int, default=300), p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--boot", type=int, default=2000), p.add_argument("--force", action="store_true")
    p.add_argument("--gdump", default=str(W / "gdump.json"))
    p = sub.add_parser("report")
    p.add_argument("--read", required=True)
    a = ap.parse_args()
    cmds = dict(rows=rows, labels=labels, glab=glab, probe=probe, **{"wa-rows": wa_rows})
    if a.cmd in ("read", "report"):
        import je0_read as RD
        cmds.update(read=RD.read, report=RD.report)
    cmds[a.cmd](a)
