"""op-adapt H sample sets: every domain in one layout so that the training loader and the probe warp the same way.

A sample = 10 packed model frames (2, 6, 128, 256) uint8 on the 5 Hz context lattice, oldest first, the last at t0; the 9 policy
context slots are the pairs (img k, img k+1), k = 0..8 (Cinque's vision pair is (t - 0.2 s, t)). Per sample: source time of every
frame (s, <= 0; the yaw / offset warps use it), slot validity, camera position on the vehicle (x fwd, y left, z up; the warp
lever), traffic convention, t0 speed and the expert future on the 0.25 s grid (20, 2) rear-axle x fwd / y left.

  domain   source                                                                    split
  nav      runs/op_lb/lb_h1train (h_nav_pool.py): 4 keys + 6 GIMM frames, op_lb's     navsim/op-adapt-h-nav-{train,dev}
           step schedule (the context of `op_lb run` / navtest), slot 0 a zero hidden
  wod      WOD-E2E train frames of op_adapt_l's prep table (its train / dev carve)  wod/r2-train rows, op_adapt_l train / dev
  carla    processed/carla_p6_v1 frames whose route and base route are outside        b2d/op-adapt-h-carla-{train,dev}
           bench2drive220 (BehaviorAgent / P6 expert), 5 Hz streams
  lwod     WOD-E2E train / dev launch events (round 2): after >= 1.8 s at v < 0.1 m/s the first frame with v > 0.1 is the onset;
           t0 = the m-th moving 5 Hz frame, m = 1, 2, 3 (table column m)
  lcarla   the same on carla_p6_v1 streams outside bench2drive220, routes split by b2d/op-adapt-h-carla-{train,dev}
  pnav / pwod / pcarla   the decision-92 probe samples (runs/op_common_cause/samples/*.json): readout (a) only, never trained

Output $DATA_DIR/runs/op_adapt_H/samples/<domain>/{imgs.npy, tab.npz}.

  CUDA_VISIBLE_DEVICES= taskset -c 0-49 $DATA_DIR/envs/op-train/bin/python experiments/op_adapt_h/scripts/h_prep.py nav wod carla pnav pwod pcarla
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.data import splits  # noqa: E402
from jevdrive.run import Run  # noqa: E402

OUT = data_dir() / "runs" / "op_adapt_H" / "samples"
NIMG = 10
FRAME = (2, 6, 128, 256)
T5 = np.round(-0.2 * np.arange(NIMG - 1, -1, -1), 3)          # -1.8 .. 0
QUOTA = {"stop": 700, "low": 800, "mid": 600, "high": 300}
DEV_QUOTA = {"stop": 100, "low": 120, "mid": 100, "high": 60}
PER_CLUSTER = 4
T_FUT = 0.25 * np.arange(1, 21)


def speed_bin(v):
    return np.select([np.asarray(v) < 0.5, np.asarray(v) < 3.0, np.asarray(v) < 8.0], ["stop", "low", "mid"], "high")


def stratify(rng, cand_bins, clusters, quota):
    """Indices into the candidate list: per bin up to quota[bin], at most PER_CLUSTER per cluster, random order."""
    keep, per = [], {}
    for name, q in quota.items():
        got = 0
        for k in rng.permutation(np.flatnonzero(cand_bins == name)):
            if got >= q:
                break
            if per.get(clusters[k], 0) >= PER_CLUSTER:
                continue
            per[clusters[k]] = per.get(clusters[k], 0) + 1
            keep.append(k)
            got += 1
    return np.sort(np.array(keep, int))


def write(dom, imgs_fn, n, tab, run, workers=0, jobs=None, init=None, initargs=()):
    d = OUT / dom
    d.mkdir(parents=True, exist_ok=True)
    if (d / "tab.npz").exists():
        run.info(f"{dom}: exists, skipped")
        return
    tmp = d / "imgs.tmp.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.uint8, (n, NIMG) + FRAME)
    t0 = time.time()
    if jobs is None:
        for i in run.tqdm(range(n), desc=dom):
            out[i] = imgs_fn(i)
    else:
        with ProcessPoolExecutor(workers, initializer=init, initargs=initargs) as ex:
            for i, fr in enumerate(run.tqdm(ex.map(imgs_fn, jobs, chunksize=4), total=n, desc=dom)):
                out[i] = fr
    out.flush()
    del out
    tmp.replace(d / "imgs.npy")
    tab = {k: np.asarray(v) for k, v in tab.items()}
    np.savez(d / "tab.npz", **tab)
    run.info(f"{dom}: {n} samples in {time.time() - t0:.0f} s; bins {dict(zip(*np.unique(tab['bin'], return_counts=True)))}; "
             f"splits {dict(zip(*np.unique(tab['split'], return_counts=True)))}")
    run.summary[dom] = n


def base_tab(n):
    return {"slot_valid": np.ones((n, 9), bool), "img_valid": np.ones((n, NIMG), bool), "img_t": np.tile(T5, (n, 1)),
            "tc": np.tile(np.array([1.0, 0.0], np.float32), (n, 1))}


# ---------------------------------------------------------------- nav (op_lb protocol)
class NavSrc:
    """op_lb frames of one run dir on the 10-image lattice: image j = step 4 j - 6 of the 31-step schedule (steps < 0: zero image)."""

    def __init__(self, data):
        import op_lb as B
        self.B, self.mt = B, B.meta(data)
        self.keys = B.Keys(data)
        self.syn = np.load(B.root(data) / "gimm.npy", mmap_mode="r")
        ts, src = B._steps(0.0, False)
        from jevdrive.op_interp import T_KEY
        self.steps = 4 * np.arange(NIMG) - 6
        self.src = [src[s] if s >= 0 else None for s in self.steps]
        self.t = np.array([0.0 if s is None else T_KEY[s[1]] if s[0] == "k" else B.SYN_T[s[1]] for s in self.src])
        self.valid = np.array([s is not None for s in self.src])

    def __call__(self, row):
        kf, sf = self.keys[row], np.asarray(self.syn[row])
        out = np.zeros((NIMG,) + FRAME, np.uint8)
        for j, s in enumerate(self.src):
            if s is not None:
                out[j] = kf[s[1]] if s[0] == "k" else sf[s[1]]
        return out


def nav_future(tokens):
    from jevdrive import navsim_zs as Z
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        pos = dict(zip(f["tokens"].tolist(), range(len(f["tokens"]))))
        P = f["poses"]
    t8 = 0.5 * np.arange(1, 9)
    out = np.zeros((len(tokens), 20, 2), np.float32)
    for i, tk in enumerate(tokens):
        p = np.asarray(P[pos[tk]], float)
        tt, xy = np.r_[0.0, t8], np.r_[np.zeros((1, 2)), p[:, :2]]
        for c in range(2):
            v = np.interp(T_FUT, tt, xy[:, c])
            ext = T_FUT > 4.0                                        # linear extrapolation of the last 0.5 s
            v[ext] = xy[-1, c] + (T_FUT[ext] - 4.0) * (xy[-1, c] - xy[-2, c]) / 0.5
            out[i, :, c] = v
    return out


def prep_nav(run, rng):
    src = NavSrc("lb_h1train")
    mt = src.mt
    tr, dv = splits.load("navsim/op-adapt-h-nav-train"), splits.load("navsim/op-adapt-h-nav-dev")
    run.use_split(tr), run.use_split(dv)
    names = list(mt["names"])
    n = len(names)
    v = np.asarray(mt["speed"], float)
    tab = base_tab(n) | {"id": np.array(names), "split": np.array(["train" if t in tr else "dev" if t in dv else "?" for t in names]),
                         "v0": v, "bin": speed_bin(v), "fut20": nav_future(names), "cam": np.asarray(mt["cam"], np.float32),
                         "cluster": np.array([str(i) for i in mt["index"]])}
    from jevdrive import navsim_zs as Z
    idx = Z.load_index("navtrain", slim=True)
    tab["cluster"] = np.array([idx[i]["log_name"] for i in mt["index"]])
    tab["img_t"] = np.tile(src.t, (n, 1))
    tab["img_valid"] = np.tile(src.valid, (n, 1))
    tab["slot_valid"][:, 0] = False
    tab["tc"] = np.array([[0.0, 1.0] if l else [1.0, 0.0] for l in mt["lht"]], np.float32)
    assert (tab["split"] != "?").all()
    write("nav", src, n, tab, run)


def prep_pnav(run, rng):
    S = json.loads((data_dir() / "runs/op_common_cause/samples/nav.json").read_text())
    src = NavSrc("lb_navtrain")
    n = len(S)
    rows = [s["row"] for s in S]
    tab = base_tab(n) | {"id": np.array([s["id"] for s in S]), "split": np.full(n, "probe"), "v0": np.array([s["v"] for s in S]),
                         "bin": np.array([s["bin"] for s in S]), "fut20": np.array([s["fut"] for s in S], np.float32),
                         "cam": np.array([s["cam"] for s in S], np.float32), "cluster": np.array([s["cluster"] for s in S]),
                         "cmd": np.array([s["cmd"] for s in S])}
    tab["img_t"] = np.tile(src.t, (n, 1))
    tab["img_valid"] = np.tile(src.valid, (n, 1))
    tab["slot_valid"][:, 0] = False
    tab["tc"] = np.array([[0.0, 1.0] if s["lht"] else [1.0, 0.0] for s in S], np.float32)
    write("pnav", lambda i: src(rows[i]), n, tab, run)


# ---------------------------------------------------------------- WOD
def _wod_init(kind):
    import wod_zeroshot_openpilot as WZ
    from jevdrive import drive_backbones as DB
    from jevdrive import wod_zeroshot as Z
    if kind == "probe":
        spans, _ = Z.load_spans()
        calib = json.loads((Z.root() / "op_calib.json").read_text())
    else:
        spans = json.loads((DB.root() / DB.plan_name("trainval")).read_text())["spans"]
        calib = json.loads((Z.root() / "op_calib.json").read_text()) | json.loads((DB.root() / "op_calib_trainval.json").read_text())
    WZ._init(spans, calib, str(data_dir() / "datasets" / "waymo_e2e" / "front3"))


def _wod_render(names):
    import drive_backbones_openpilot as DB
    return DB.render(list(names))


def wod_calib():
    from jevdrive import drive_backbones as DB
    from jevdrive import wod_zeroshot as Z
    return json.loads((Z.root() / "op_calib.json").read_text()) | json.loads((DB.root() / "op_calib_trainval.json").read_text())


def prep_wod(run, rng):
    from jevdrive import drive_backbones as DB
    z = np.load(data_dir() / "runs/op_adapt_L/prep/wod.npz", allow_pickle=True)
    run.use_split(splits.load("wod/r2-train"))
    spans = set(json.loads((DB.root() / DB.plan_name("trainval")).read_text())["spans"])
    calib = wod_calib()
    names, seq, split = z["name"].astype(str), z["seq"].astype(str), z["split"].astype(str)
    hist = lambda n: [f"{n.rsplit('-', 1)[0]}-{int(n.rsplit('-', 1)[1]) - 2 * j:03d}" for j in range(NIMG - 1, -1, -1)]  # noqa: E731
    cand = np.flatnonzero(z["has"] & np.isin(split, ["train", "dev"]))
    cand = cand[rng.permutation(len(cand))[:120000]]
    cand = np.array([k for k in cand if seq[k] in calib and all(h in spans for h in hist(names[k]))])
    v = z["v0"][cand].astype(float)
    b = speed_bin(v)
    sel = []
    for sp, q in (("train", QUOTA), ("dev", DEV_QUOTA)):
        m = split[cand] == sp
        sel.append(cand[m][stratify(rng, b[m], seq[cand][m], q)])
    sel = np.sort(np.concatenate(sel))
    n = len(sel)
    tab = base_tab(n) | {"id": names[sel], "split": split[sel], "v0": z["v0"][sel].astype(float), "bin": speed_bin(z["v0"][sel]),
                         "fut20": z["fut20"][sel].astype(np.float32), "cluster": seq[sel],
                         "cam": np.array([np.array(calib[s]["1"]["extrinsic"]).reshape(4, 4)[:3, 3] for s in seq[sel]], np.float32)}
    write("wod", _wod_render, n, tab, run, workers=40, jobs=[hist(names[k]) for k in sel], init=_wod_init, initargs=("train",))


LAUNCH_V, LAUNCH_M = 0.1, (1, 2, 3)


def prep_lwod(run, rng):
    from collections import defaultdict
    from jevdrive import drive_backbones as DB
    z = np.load(data_dir() / "runs/op_adapt_L/prep/wod.npz", allow_pickle=True)
    run.use_split(splits.load("wod/r2-train"))
    spans = set(json.loads((DB.root() / DB.plan_name("trainval")).read_text())["spans"])
    calib = wod_calib()
    names, seq, split, v = z["name"].astype(str), z["seq"].astype(str), z["split"].astype(str), z["v0"].astype(float)
    fr = np.array([int(n.rsplit("-", 1)[1]) for n in names])
    by = defaultdict(dict)
    for k, (q, f) in enumerate(zip(seq, fr)):
        by[q][f] = k
    hist = lambda n: [f"{n.rsplit('-', 1)[0]}-{int(n.rsplit('-', 1)[1]) - 2 * j:03d}" for j in range(NIMG - 1, -1, -1)]  # noqa: E731
    sel, ms = [], []
    for q, d in by.items():
        if q not in calib:
            continue
        for f in sorted(d):
            prev = [d.get(f - 2 * j) for j in range(1, NIMG)]
            if v[d[f]] <= LAUNCH_V or any(p is None or v[p] >= LAUNCH_V for p in prev):
                continue
            for m in LAUNCH_M:
                k = d.get(f + 2 * (m - 1))
                if k is not None and z["has"][k] and split[k] in ("train", "dev") and all(h in spans for h in hist(names[k])):
                    sel.append(k), ms.append(m)
    sel, ms = np.array(sel), np.array(ms)
    n = len(sel)
    tab = base_tab(n) | {"id": names[sel], "split": split[sel], "v0": v[sel], "bin": speed_bin(v[sel]), "m": ms,
                         "fut20": z["fut20"][sel].astype(np.float32), "cluster": seq[sel],
                         "cam": np.array([np.array(calib[s]["1"]["extrinsic"]).reshape(4, 4)[:3, 3] for s in seq[sel]], np.float32)}
    write("lwod", _wod_render, n, tab, run, workers=40, jobs=[hist(names[k]) for k in sel], init=_wod_init, initargs=("train",))


def prep_lcarla(run, rng):
    import pandas as pd
    import p5_openpilot as PO
    from jevdrive import p5_openpilot as P
    D = data_dir() / "processed/carla_p6_v1"
    t = pd.read_parquet(D / "index.parquet")
    past, fut = np.load(D / "past.npy", mmap_mode="r"), np.load(D / "future.npy", mmap_mode="r")
    tr, dv = splits.load("b2d/op-adapt-h-carla-train"), splits.load("b2d/op-adapt-h-carla-dev")
    run.use_split(tr), run.use_split(dv)
    rid_all = t.route_id.astype(str).to_numpy()
    plan = json.loads((D / "op_plan.json").read_text())
    pos = {n: k for k, n in enumerate(t.frame_name.astype(str))}
    v = np.linalg.norm(np.asarray(past[:, -1, 2:4]), axis=1)
    sel, files, ms = [], [], []
    for s in plan["streams"]:
        fr = [int(x.rsplit("-", 1)[1]) for x in s["names"]]
        ks = [pos.get(x) for x in s["names"]]
        for j in range(NIMG - 1, len(ks)):
            if ks[j] is None or v[ks[j]] <= LAUNCH_V or any(ks[q] is None or v[ks[q]] >= LAUNCH_V for q in range(j - NIMG + 1, j)):
                continue
            for m in LAUNCH_M:
                e = j + m - 1
                if e >= len(ks) or ks[e] is None or np.any(np.diff(fr[e - NIMG + 1: e + 1]) != 4):
                    continue
                if rid_all[ks[e]] in tr or rid_all[ks[e]] in dv:
                    sel.append(ks[e]), files.append(s["files"][e - NIMG + 1: e + 1]), ms.append(m)
    k = np.array(sel)
    n = len(k)
    rid = rid_all[k]
    tab = base_tab(n) | {"id": t.frame_name.astype(str).to_numpy()[k], "split": np.array(["dev" if r in dv else "train" for r in rid]),
                         "v0": v[k], "bin": speed_bin(v[k]), "m": np.array(ms), "fut20": np.asarray(fut[k], np.float32), "cluster": rid,
                         "cam": np.tile(np.array(P.RIG[0][1:4], np.float32), (n, 1))}
    write("lcarla", _carla_render, n, tab, run, workers=40, jobs=[(f, PO.SEQ) for f in files], init=_carla_init, initargs=("train",))


def prep_pwod(run, rng):
    S = json.loads((data_dir() / "runs/op_common_cause/samples/wod.json").read_text())
    S = [s for s in S if len(s["hist"]) >= NIMG]
    n = len(S)
    tab = base_tab(n) | {"id": np.array([s["id"] for s in S]), "split": np.full(n, "probe"), "v0": np.array([s["v"] for s in S]),
                         "bin": np.array([s["bin"] for s in S]), "fut20": np.array([s["fut"] for s in S], np.float32),
                         "cam": np.array([s["cam"] for s in S], np.float32), "cluster": np.array([s["cluster"] for s in S]),
                         "cmd": np.array([s["cmd"] for s in S])}
    write("pwod", _wod_render, n, tab, run, workers=40, jobs=[s["hist"][-NIMG:] for s in S], init=_wod_init, initargs=("probe",))


# ---------------------------------------------------------------- CARLA
def _carla_init(kind):
    import wod_zeroshot_openpilot as WZ
    from jevdrive import p5_openpilot as P
    if kind == "probe":
        WZ._init({}, {"carla": P.carla_calib()}, ".")
    else:
        plan = json.loads((data_dir() / "processed/carla_p6_v1/op_plan.json").read_text())
        import p5_openpilot as PO
        WZ._init({}, {PO.SEQ: plan["calib"]}, ".")


def _carla_render(job):
    import p5_openpilot as PO
    files, seq = job
    return PO.render(files, seq)


def prep_carla(run, rng):
    import pandas as pd
    import p5_openpilot as PO
    from jevdrive import p5_openpilot as P
    D = data_dir() / "processed/carla_p6_v1"
    t = pd.read_parquet(D / "index.parquet")
    past, fut = np.load(D / "past.npy", mmap_mode="r"), np.load(D / "future.npy", mmap_mode="r")
    b220 = splits.load("b2d/bench2drive220")
    run.use_split(b220)
    ok = ~t.base_id.astype(str).isin(set(b220)) & ~t.route_id.astype(str).isin(set(b220))
    routes = np.array(sorted(t.route_id[ok].astype(str).unique()))
    dev_r = set(np.random.default_rng(20261004).permutation(routes)[: int(round(0.12 * len(routes)))])
    origin = "processed/carla_p6_v1 routes whose route_id and base_id are outside b2d/bench2drive220; dev = 12% of routes (seed 20261004)"
    for nm, mem in (("op-adapt-h-carla-train", [r for r in routes if r not in dev_r]), ("op-adapt-h-carla-dev", sorted(dev_r))):
        run.use_split(splits.define("b2d", nm, mem, unit="route", origin=origin, status="frozen", used_by=["op_adapt_h"],
                                    notes="op_adapt_h layer-3 pilot CARLA pool; disjoint from bench2drive220 and the decision-92 P4 probe"))
    plan = json.loads((D / "op_plan.json").read_text())
    pos = {n: k for k, n in enumerate(t.frame_name.astype(str))}
    cand, files = [], []
    for s in plan["streams"]:
        fr = [int(x.rsplit("-", 1)[1]) for x in s["names"]]
        for j in range(NIMG - 1, len(s["names"])):
            k = pos.get(s["names"][j])
            if k is None or not ok.iloc[k] or np.any(np.diff(fr[j - NIMG + 1: j + 1]) != 4):
                continue
            cand.append(k)
            files.append(s["files"][j - NIMG + 1: j + 1])
    cand = np.array(cand)
    v = np.linalg.norm(past[cand, -1, 2:4], axis=1)
    b = speed_bin(v)
    rid = t.route_id.astype(str).to_numpy()[cand]
    sel = []
    for sp, q in (("train", QUOTA), ("dev", DEV_QUOTA)):
        m = np.array([(r in dev_r) == (sp == "dev") for r in rid])
        sel.append(np.flatnonzero(m)[stratify(rng, b[m], rid[m], q)])
    sel = np.sort(np.concatenate(sel))
    n = len(sel)
    k = cand[sel]
    tab = base_tab(n) | {"id": t.frame_name.astype(str).to_numpy()[k], "split": np.array(["dev" if r in dev_r else "train" for r in rid[sel]]),
                         "v0": v[sel], "bin": b[sel], "fut20": np.asarray(fut[k], np.float32), "cluster": rid[sel],
                         "cam": np.tile(np.array(P.RIG[0][1:4], np.float32), (n, 1))}
    write("carla", _carla_render, n, tab, run, workers=40, jobs=[(files[i], PO.SEQ) for i in sel], init=_carla_init, initargs=("train",))


def prep_pcarla(run, rng):
    S = json.loads((data_dir() / "runs/op_common_cause/samples/carla.json").read_text())
    S = [s for s in S if len(s["files"]) >= NIMG]
    n = len(S)
    tab = base_tab(n) | {"id": np.array([s["id"] for s in S]), "split": np.full(n, "probe"), "v0": np.array([s["v"] for s in S]),
                         "bin": np.array([s["bin"] for s in S]), "fut20": np.array([s["fut"] for s in S], np.float32),
                         "cam": np.array([s["cam"] for s in S], np.float32), "cluster": np.array([s["cluster"] for s in S]),
                         "cmd": np.array([s["cmd"] for s in S])}
    write("pcarla", _carla_render, n, tab, run, workers=40, jobs=[(s["files"][-NIMG:], "carla") for s in S], init=_carla_init,
          initargs=("probe",))


def main():
    doms = sys.argv[1:] or ["nav", "wod", "carla", "pnav", "pwod", "pcarla"]
    with Run("op_adapt_H", "prep", seed=0) as run:
        for d in doms:
            globals()[f"prep_{d}"](run, np.random.default_rng(0))


if __name__ == "__main__":
    main()
