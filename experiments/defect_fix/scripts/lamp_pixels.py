#!/usr/bin/env python3
"""LAMP-PIXELS (plans/2026-10-19-lamp-pixels-prereg.md): is the traffic-lamp colour in the AlpaSim-rendered CAM_F0 pixels at all? A lamp
classifier of our own (ImageNet ResNet-18, fine-tuned on real NAVSIM trainval CAM_F0, red vs green for the route connector ahead, cells of
speed x distance balanced) applied unchanged to (i) held trainval logs, (ii) the real CAM_F0 of the arena held windows (same log, same time),
(iii) the AlpaSim render of the same windows (JTN-L18-s1 native SH30_DUMP JPEGs). Against the frozen Cinque tokens on the same frames.
Results: results/lamp_pixels.md. No policy training.

  $TR lamp_pixels.py select      (CPU) TV rows (census, balanced per cell) + arena rows (window x decision, raw-DB lamp state, paired real
                                 frame) + dump lists of the windows with no reusable render -> $O/{tv,arena}.parquet, $O/cams.npz, $O/lists/
  $TR lamp_pixels.py frames      (CPU) every needed render JPEG + its serve pack (slot 7) out of the dumps -> $O/render/; own dump npz deleted
  $TR lamp_pixels.py cnn --var N|L   (GPU) train on TV train, score TV hold / arena real / arena render -> $O/cnn_<var>/scores.npz
  $TR lamp_pixels.py ft          (GPU) CNN-N fine-tuned on the render k = 0 rows, 5 log folds -> $O/ft/scores.npz
  $TR lamp_pixels.py tok         (GPU) Cinque view_39 tokens (single frame, prev = cur) + token probe -> $O/tok/scores.npz
  $TR lamp_pixels.py eval        (CPU) AUC tables, gates, branch -> $O/eval.json
  python lamp_pixels.py report [--src F]   eval.json -> markdown (stdout)
SMOKE=1 writes under $O/smoke with a few rows and steps.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import pickle
import sqlite3
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/wait_cap/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
O = DATA / "runs/defect_fix/lamp_pixels" / ("smoke" if SMOKE else "")
NAV = DATA / "datasets/navsim"
CENSUS = DATA / "runs/self_iter/tv_census/frames.npz"
ACEN = DATA / "runs/defect_fix/native_gap/set_census.parquet"
ASET = DATA / "runs/alpasim_proxy/arena/set/set.csv.gz"
RAW = DATA / "datasets/nuplan_raw/test"
SRC_DUMPS = [(DATA / "runs/defect_fix/native_gap/dump/JTN1", "c*/run/driver-logs/drive.jsonl"),
             (DATA / "runs/defect_fix/redline_probe/dump", "*/run/driver-logs/drive.jsonl")]
WEIGHTS = DATA / "models/imagenet/resnet18-f37072fd.pth"
SPLIT_TR, SPLIT_HO, SPLIT_AR = "navsim/tv-clean-train-logs", "navsim/tv-clean-hold-logs", "alpasim_nuplan/onpol-held"
VB, DB = (0.3, 3.0, 8.0), (10.0, 25.0)               # cell edges: speed (m/s) x front-bumper distance to the line (m)
D_LO, D_HI, NK = -3.0, 60.0, 10
CAP_TR, CAP_HO = (40, 20) if SMOKE else (1500, 300)
CROP_H, JIT = 640, 24
FOLDS, B_BOOT, PER_LIST = 5, 2000, 60


def cell_of(v, d):
    return np.digitize(v, VB) * 3 + np.digitize(d, DB)


def fold_of(logs):
    return np.array([int(hashlib.sha1(x.encode()).hexdigest(), 16) % FOLDS for x in logs])


def cam_arr(c):
    """OpenScene CAM_F0 dict -> (K 9, R 9, t 3, D 5) float32 row."""
    return np.r_[np.ravel(c["cam_intrinsic"]), np.ravel(c["sensor2lidar_rotation"]), np.ravel(c["sensor2lidar_translation"]), np.ravel(c["distortion"])].astype(np.float32)


# ---------------------------------------------------------------- select (CPU)
def _tv_paths(job):
    log, toks = job
    fr = pickle.load(open(NAV / "navsim_logs/trainval" / f"{log}.pkl", "rb"))
    want = set(toks)
    return {f["token"]: (str(NAV / "sensor_blobs/trainval" / f["cams"]["CAM_F0"]["data_path"]), cam_arr(f["cams"]["CAM_F0"]))
            for f in fr if f["token"] in want}


def _arena_log(job):
    """One held log: per window and decision the raw-DB lamp state of the census connector, the human's speed and front distance to the
    line, and the nearest 2 Hz OpenScene CAM_F0 frame of the same instant."""
    import nx_census as NX
    log, W = job
    L = NX.Log(RAW / f"{log}.db")
    c = sqlite3.connect(f"file:{RAW / f'{log}.db'}?mode=ro", uri=True)
    ts = np.array([r[0] for r in c.execute("select timestamp from lidar_pc order by timestamp")], float)
    fr = pickle.load(open(NAV / "navsim_logs/test" / f"{log}.pkl", "rb"))
    pts = np.array([f["timestamp"] for f in fr], float)
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(L.xy, axis=0).T))]
    out = []
    for w in W:
        i0 = int(w["start_idx"])
        assert abs(L.t[i0] - w["t0"]) < 0.06, (w["id"], L.t[i0], w["t0"])
        cid = w["tl_cid"]
        for k in range(NK):
            i = i0 + 10 * k
            if i >= len(L.t):
                break
            st = L.status(i, cid)
            d = float(w["tl_d_front"] - (arc[i] - arc[i0]))
            j = int(np.argmin(np.abs(pts - ts[i])))
            dt = abs(pts[j] - ts[i]) * 1e-6
            ok = dt <= 0.26
            out.append(dict(id=w["id"], log=log, k=k, st=int(st), dF=d, v=float(L.v[i]), real_dt=float(dt),
                            real=str(NAV / "sensor_blobs/test" / fr[j]["cams"]["CAM_F0"]["data_path"]) if ok else "",
                            cam=cam_arr(fr[j]["cams"]["CAM_F0"]) if ok else np.zeros(26, np.float32)))
    return out


def dumped(root, pattern):
    out = {}
    for f in sorted(glob.glob(str(root / pattern))):
        for line in open(f):
            if line.startswith('{"kind": "start"'):
                r = json.loads(line)
                out.setdefault(r["scene"], (f, r["n"]))
    return out


def render_src(ids_ks):
    """{(id, k): npz path} over the reusable dumps and this lane's own."""
    out = {}
    for root, pat in SRC_DUMPS + [(O / "dump", "*/run/driver-logs/drive.jsonl")]:
        for sc, (f, n) in dumped(root, pat).items():
            for k in ids_ks.get(sc, ()):
                p = Path(f).parent / "dump" / f"s{n:02d}_k{k}.npz"
                if (sc, k) not in out and p.exists():
                    out[sc, k] = str(p)
    return out


def cmd_select(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    O.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "lamp-pixels-select" + ("-smoke" if SMOKE else ""), config=vars(a)) as run:
        tr, ho, ar = splits.load(SPLIT_TR), splits.load(SPLIT_HO), splits.load(SPLIT_AR)
        for s in (tr, ho, ar):
            run.use_split(s)
        # TV rows
        z = np.load(CENSUS)
        log = z["log"].astype(str)
        m = (z["ln_type"] == 2) & z["ln_conn"] & (z["ln_dF"] >= D_LO) & (z["ln_dF"] < D_HI) & np.isin(z["ln_st0"], [0, 1]) & z["cam4"]
        cell = cell_of(z["v0"], z["ln_dF"])
        rng = np.random.default_rng(0)
        rows = []
        for part, S, cap in (("train", tr, CAP_TR), ("hold", ho, CAP_HO)):
            logs = sorted(S.members)
            if SMOKE:
                logs = logs[:60]
            q = m & np.isin(log, logs)
            for c in range(12):
                a_, b_ = np.flatnonzero(q & (cell == c) & (z["ln_st0"] == 1)), np.flatnonzero(q & (cell == c) & (z["ln_st0"] == 0))
                n = min(len(a_), len(b_), cap)
                rows += [(i, part) for i in np.r_[rng.choice(a_, n, replace=False), rng.choice(b_, n, replace=False)]]
        idx = np.array([r[0] for r in rows])
        T = pd.DataFrame(dict(token=z["token"][idx].astype(str), log=log[idx], part=[r[1] for r in rows], y=(z["ln_st0"][idx] == 1).astype(int),
                              v=z["v0"][idx], dF=z["ln_dF"][idx], cell=cell[idx]))
        res = par.pmap(_tv_paths, list(T.groupby("log").token.apply(list).items()), run=run, workers=a.workers, desc="tv pickles")
        res.raise_if_failed()
        P = {k: v for d in res.values for k, v in d.items()}
        T["path"] = [P[t][0] for t in T.token]
        cam_tv = np.stack([P[t][1] for t in T.token])
        run.info(f"TV rows {len(T)}: " + T.groupby("part").y.agg(["size", "sum"]).to_string().replace("\n", "; "))
        # arena rows
        C = pd.read_parquet(ACEN).drop_duplicates("id").set_index("id")
        S = pd.read_csv(ASET).set_index("id")
        C = C[(C.split == "held") & C.tl.fillna(False).astype(bool) & C.tl_cid.notna() & (C.tl_d_front >= D_LO) & (C.tl_d_front < D_HI)]
        assert set(C.src) <= set(ar.members), "an arena window outside the registered held split"
        C = C.assign(tl_cid=[str(int(float(x))) if str(x).replace(".", "").isdigit() else str(x) for x in C.tl_cid], id=C.index)
        if SMOKE:
            C = C[C.log.isin(sorted(set(C.log))[:3])]
        jobs = [(lg, g[["id", "start_idx", "t0", "tl_cid", "tl_d_front"]].to_dict("records")) for lg, g in C.groupby("log")]
        res = par.pmap(_arena_log, jobs, run=run, workers=a.workers, desc="arena logs")
        res.raise_if_failed()
        R = pd.DataFrame([r for v in res.values for r in v])
        cam_ar = np.stack(R.pop("cam").values)
        keep = R.st.isin([0, 1]).values & (R.dF >= D_LO).values & (R.dF < D_HI).values
        R, cam_ar = R[keep].reset_index(drop=True), cam_ar[keep]
        R["y"] = (R.st == 1).astype(int)
        R["cell"] = cell_of(R.v.values, R.dF.values)
        for c_ in ("red_stop", "red_cls", "red_start", "stratum", "lead"):
            src = S if c_ in ("stratum", "lead") else C
            R[c_] = src[c_].reindex(R.id).values
        mixed = R.groupby("id").y.agg(lambda x: x.nunique() > 1)
        R["mixed"] = R.id.map(mixed).astype(bool)
        # render sources: k = 0 for every window, every row's k for the mixed windows
        need = {i: set(g.k) if mixed[i] else {0} & set(g.k) for i, g in R.groupby("id")}
        need = {i: ks for i, ks in need.items() if ks}
        have = render_src(need)
        new = sorted(i for i, ks in need.items() if not all((i, k) in have for k in ks))
        (O / "lists").mkdir(exist_ok=True)
        for f in (O / "lists").glob("*.txt"):
            f.unlink()
        sh = S.shard.reindex(new)
        for k, g in sh.groupby(sh):
            g = sorted(g.index)[:4] if SMOKE else sorted(g.index)
            for j, cch in enumerate(np.array_split(np.array(g), int(np.ceil(len(g) / PER_LIST)))):
                (O / "lists" / f"s{k}_{j}.txt").write_text("\n".join(cch) + "\n")
            if SMOKE:
                break
        T.to_parquet(O / "tv.parquet")
        R.to_parquet(O / "arena.parquet")
        np.savez(O / "cams.npz", tv=cam_tv, arena=cam_ar)
        s = dict(tv=T.groupby("part").y.agg(["size", "sum"]).to_dict(), arena_rows=len(R), arena_windows=int(R.id.nunique()), arena_logs=int(R.log.nunique()),
                 k0_red=int(((R.k == 0) & (R.y == 1)).sum()), k0_green=int(((R.k == 0) & (R.y == 0)).sum()), mixed_windows=int(mixed.sum()),
                 real_missing=int((R.real == "").sum()), real_dt_p99=float(R.real_dt.quantile(0.99)), new_dump_windows=len(new),
                 lists=sorted(p.stem for p in (O / "lists").glob("*.txt")))
        run.summary.update(s)
        print(json.dumps(s, indent=1, default=str))


# ---------------------------------------------------------------- frames (CPU)
def cmd_frames(a):
    import pandas as pd
    from jevdrive.run import Run
    R = pd.read_parquet(O / "arena.parquet")
    with Run("defect_fix", "lamp-pixels-frames" + ("-smoke" if SMOKE else ""), config=vars(a)) as run:
        need = {i: set(g.k) if g.mixed.iloc[0] else {0} & set(g.k) for i, g in R.groupby("id")}
        have = render_src(need)
        (O / "render").mkdir(exist_ok=True)
        keys = sorted(have)
        packs = np.lib.format.open_memmap(O / "render/packs.npy", "w+", np.uint8, (len(keys), 2, 6, 128, 256))
        for j, (i, k) in enumerate(keys):
            d = np.load(have[i, k])
            assert str(d["scene"]) == i, (i, k, have[i, k])
            (O / "render" / f"{i}_k{k}.jpg").write_bytes(d["jpeg"].tobytes())
            packs[j] = d["cur"][7]
        packs.flush()
        pd.DataFrame(dict(id=[x[0] for x in keys], k=[x[1] for x in keys])).to_parquet(O / "render/index.parquet")
        own = sorted(glob.glob(str(O / "dump/*/run/driver-logs/dump/*.npz")))
        for f in own:
            os.remove(f)
        run.summary.update(frames=len(keys), windows=len({x[0] for x in keys}), own_npz_deleted=len(own))
        run.info(f"render frames {len(keys)}, own dump npz deleted {len(own)}")


def sets():
    """The three evaluation sets as (paths, frame keys): TV hold, arena real, arena render (rows of arena.parquet with a frame)."""
    import pandas as pd
    T = pd.read_parquet(O / "tv.parquet")
    R = pd.read_parquet(O / "arena.parquet")
    ix = pd.read_parquet(O / "render/index.parquet")
    rk = {(i, k): j for j, (i, k) in enumerate(zip(ix.id, ix.k))}
    R["render_j"] = [rk.get((i, k), -1) for i, k in zip(R.id, R.k)]
    R["render"] = [str(O / "render" / f"{i}_k{k}.jpg") if j >= 0 else "" for i, k, j in zip(R.id, R.k, R.render_j)]
    return T, R


# ---------------------------------------------------------------- CNN (GPU)
class Frames:
    def __init__(self, paths, y=None, train=False, down=1, seed=0):
        self.p, self.y, self.train, self.down, self.seed = list(paths), y, train, down, seed

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        import torch
        from PIL import Image, ImageFilter
        im = Image.open(self.p[i]).convert("RGB")
        W = im.size[0]
        if self.train:
            g = np.random.default_rng()
            ox, oy = int(g.integers(-JIT, JIT + 1)), int(g.integers(0, JIT + 1))
            im = im.crop((ox, oy, ox + W, oy + CROP_H))
            if g.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            x = np.asarray(im, np.float32) * g.uniform(0.8, 1.2)
            x = (x - x.mean()) * g.uniform(0.8, 1.2) + x.mean()
            im = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
            if g.random() < 0.5:
                im = im.filter(ImageFilter.GaussianBlur(g.uniform(0, 1.5)))
            if g.random() < 0.3:
                f = g.uniform(1, 2)
                im = im.resize((int(W / f), int(CROP_H / f)), Image.BILINEAR).resize((W, CROP_H), Image.BILINEAR)
        else:
            im = im.crop((0, 0, W, CROP_H))
        if self.down > 1:
            im = im.resize((W // self.down, CROP_H // self.down), Image.BILINEAR, reducing_gap=None)
        x = (np.asarray(im, np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))
        return x, (float(self.y[i]) if self.y is not None else 0.0)


def resnet():
    import torch
    import torchvision
    m = torchvision.models.resnet18()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.fc = torch.nn.Linear(512, 1)
    return m


def fit(m, ds, epochs, lr, bs, workers, run, pos_weight=1.0, max_steps=None):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, bs, shuffle=True, num_workers=workers, drop_last=len(ds) > bs, persistent_workers=True, pin_memory=True)
    steps = max(1, min(epochs * len(dl), max_steps or 10**9))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    pw = torch.tensor(pos_weight, device="cuda")
    m.train()
    t = 0
    for ep in range(epochs):
        tot = 0.0
        for x, y in dl:
            with torch.autocast("cuda", torch.bfloat16):
                lo = m(x.cuda(non_blocking=True).to(memory_format=torch.channels_last))[:, 0].float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, y.cuda().float(), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            tot += loss.item()
            t += 1
            if t % 200 == 0:
                run.info(f"epoch {ep} step {t} / {steps} loss {tot / max(1, t - ep * len(dl)):.4f}")
            if t >= steps:
                return
        run.info(f"epoch {ep} mean loss {tot / len(dl):.4f}")


def score(m, paths, down, workers):
    import torch
    from torch.utils.data import DataLoader
    if not len(paths):
        return np.zeros(0)
    m.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for x, _ in DataLoader(Frames(paths, down=down), 32, num_workers=workers):
            out.append(m(x.cuda().to(memory_format=torch.channels_last))[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def cmd_cnn(a):
    import torch
    from jevdrive.run import Run
    from jevdrive.data import splits
    T, R = sets()
    down = {"N": 1, "L": 4}[a.var]
    out = O / f"cnn_{a.var}"
    out.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", f"lamp-pixels-cnn{a.var}" + ("-smoke" if SMOKE else ""), config=vars(a)) as run:
        run.use_split(splits.load(SPLIT_TR))
        torch.manual_seed(0)
        m = resnet().cuda().to(memory_format=torch.channels_last)
        tr = T[T.part == "train"]
        fit(m, Frames(tr.path.values, tr.y.values, train=True, down=down), 3, 3e-4, 32, a.workers, run, max_steps=60 if SMOKE else None)
        torch.save(m.state_dict(), out / "model.pt")
        ho = T[T.part == "hold"]
        S = dict(tv_hold=score(m, ho.path.values, down, a.workers))
        q = R.real != ""
        S["real"] = np.where(q, 0.0, np.nan)
        S["real"][q.values] = score(m, R.real[q].values, down, a.workers)
        q = R.render != ""
        S["render"] = np.where(q, 0.0, np.nan)
        S["render"][q.values] = score(m, R.render[q].values, down, a.workers)
        np.savez(out / "scores.npz", **S)
        run.summary.update(tv_hold=len(ho), real=int(np.isfinite(S["real"]).sum()), render=int(np.isfinite(S["render"]).sum()))


def cmd_ft(a):
    """CNN-N fine-tuned on the render k = 0 rows, 5 log folds: does a render-side reader find red vs green there at all?"""
    import torch
    from jevdrive.run import Run
    T, R = sets()
    out = O / "ft"
    out.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "lamp-pixels-ft" + ("-smoke" if SMOKE else ""), config=vars(a)) as run:
        q = np.flatnonzero(((R.k == 0) & (R.render != "")).values)
        fo = fold_of(R.log.values[q])
        sc = np.full(len(R), np.nan)
        for f in range(FOLDS):
            tr, te = q[fo != f], q[fo == f]
            if not len(te) or len(set(R.y.values[tr])) < 2:
                continue
            torch.manual_seed(f)
            m = resnet().cuda().to(memory_format=torch.channels_last)
            m.load_state_dict(torch.load(O / "cnn_N/model.pt", map_location="cuda"))
            y = R.y.values[tr]
            fit(m, Frames(R.render.values[tr], y, train=True), 4, 1e-4, 16, a.workers, run, pos_weight=float((1 - y).sum() / max(y.sum(), 1)),
                max_steps=10 if SMOKE else None)
            sc[te] = score(m, R.render.values[te], 1, a.workers)
            run.info(f"fold {f}: train {len(tr)}, scored {len(te)}")
        np.savez(out / "scores.npz", render_ft=sc)


# ---------------------------------------------------------------- frozen tokens (GPU)
class Packs:
    """Real CAM_F0 -> openpilot serve pack (navsim_zs.OpenpilotMaps with the frame's own calibration)."""

    def __init__(self, paths, cams):
        self.p, self.c, self.maps = list(paths), cams, {}

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        from jevdrive import navsim_zs as Z
        c = self.c[i]
        cam = dict(K=c[:9].reshape(3, 3), R=c[9:18].reshape(3, 3), t=c[18:21], D=c[21:26])
        key = Z.calib_key({"CAM_F0": cam})
        if key not in self.maps:
            self.maps[key] = Z.OpenpilotMaps(cam)
        mp = self.maps[key]
        return mp(mp.decode(self.p[i]))


def cmd_tok(a):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib"), str(REPO / "lib")]
    import sh30_core as SC
    from jevdrive.run import Run
    from jevdrive.data import splits
    T, R = sets()
    cams = np.load(O / "cams.npz")
    out = O / "tok"
    out.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "lamp-pixels-tok" + ("-smoke" if SMOKE else ""), config=vars(a)) as run:
        run.use_split(splits.load(SPLIT_TR))
        core = SC.Core("JTN-L18-s1", "cuda", "backwarp")
        net, A = core.model.net, core.A

        def enc(x):
            x = torch.as_tensor(x).cuda()
            with torch.no_grad():
                return net.run_batched(A.vision_feeds(x, x), ["view_39"])["view_39"].reshape(len(x), *A.H_SHAPE).half().cpu().numpy()

        def toks(ds):
            return np.concatenate([enc(b) for b in DataLoader(ds, 32, num_workers=a.workers)]) if len(ds) else np.zeros((0, *A.H_SHAPE), np.float16)

        H_tv = toks(Packs(T.path.values, cams["tv"]))
        q = (R.real != "").values
        H_real = toks(Packs(R.real.values[q], cams["arena"][q]))
        packs = np.load(O / "render/packs.npy", mmap_mode="r")
        qr = (R.render_j >= 0).values
        H_ren = np.concatenate([enc(np.ascontiguousarray(packs[R.render_j.values[qr][i:i + 32]])) for i in range(0, qr.sum(), 32)])
        run.info(f"tokens: tv {len(H_tv)}, real {len(H_real)}, render {len(H_ren)}")
        # probe: per-token LN -> 16 -> MLP 256 (redline_probe.fit_tok without the kinematic row), 10 epochs
        torch.manual_seed(0)
        pr = nn.ModuleDict(dict(ln=nn.LayerNorm(512, elementwise_affine=False), red=nn.Linear(512, 16),
                                mlp=nn.Sequential(nn.Linear(32 * 16, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)))).cuda()
        f = lambda t: pr["mlp"](pr["red"](pr["ln"](t.float())).flatten(1))[:, 0]  # noqa: E731
        tr = (T.part == "train").values
        Xt, yt = torch.from_numpy(H_tv[tr]).cuda(), torch.from_numpy(T.y.values[tr].astype(np.float32)).cuda()
        opt = torch.optim.AdamW(pr.parameters(), lr=1e-3, weight_decay=1e-2)
        g = np.random.default_rng(0)
        for ep in range(2 if SMOKE else 10):
            pr.train()
            for b in np.array_split(g.permutation(len(yt)), max(1, len(yt) // 256)):
                b = torch.from_numpy(b).cuda()
                loss = nn.functional.binary_cross_entropy_with_logits(f(Xt[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            run.info(f"tok probe epoch {ep} loss {loss.item():.4f}")
        pr.eval()

        def sc(H):
            with torch.no_grad():
                return np.concatenate([f(torch.from_numpy(H[i:i + 1024]).cuda()).cpu().numpy() for i in range(0, len(H), 1024)]) if len(H) else np.zeros(0)
        S = dict(tv_hold=sc(H_tv[~tr]), real=np.full(len(R), np.nan), render=np.full(len(R), np.nan))
        S["real"][q], S["render"][qr] = sc(H_real), sc(H_ren)
        np.savez(out / "scores.npz", **S)


# ---------------------------------------------------------------- eval (CPU)
def sauc(s, y, strata, grp, seed=0, C=None):
    """AUC over (positive, negative) pairs inside the same stratum; log-cluster bootstrap (B_BOOT draws of grp with replacement).
    -> (auc, lo, hi), draws, n pairs. Ties count 1/2."""
    s, y, strata, grp = (np.asarray(x) for x in (s, y, strata, grp))
    ok = np.isfinite(s)
    s, y, strata, grp = s[ok], y[ok].astype(bool), strata[ok], grp[ok]
    u, g = np.unique(grp, return_inverse=True)
    if C is None:
        rng = np.random.default_rng(seed)
        C = np.vstack([np.ones(len(u)), np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B_BOOT)])])
    num, den, npair = np.zeros(len(C)), np.zeros(len(C)), 0
    for c in np.unique(strata):
        q = strata == c
        if y[q].all() or not y[q].any():
            continue
        ss, yy, gg = s[q], y[q], g[q]
        o = np.argsort(ss, kind="stable")
        ss, yy, gg = ss[o], yy[o], gg[o]
        W = C[:, gg]
        Wp, Wn = W * yy, W * ~yy
        _, first, cnt = np.unique(ss, return_index=True, return_counts=True)       # tie blocks (sorted ascending)
        blk = np.repeat(np.arange(len(first)), cnt)
        Pb = np.add.reduceat(Wp, first, axis=1)
        above = Pb.sum(1, keepdims=True) - np.cumsum(Pb, axis=1)                  # positive weight strictly above each block
        num += (Wn * (above[:, blk] + 0.5 * Pb[:, blk])).sum(1)
        den += Wn.sum(1) * Wp.sum(1)
        npair += int(yy.sum() * (~yy).sum())
    if npair == 0:
        return None, None, 0
    a = num / np.maximum(den, 1e-12)
    return (float(a[0]), float(np.quantile(a[1:], 0.025)), float(np.quantile(a[1:], 0.975))), a[1:], npair


def cmd_eval(a):
    import pandas as pd
    T, R = sets()
    SC = {}
    for nm, d in (("CNN-N", "cnn_N"), ("CNN-L", "cnn_L"), ("TOK", "tok")):
        p = O / d / "scores.npz"
        if p.exists():
            SC[nm] = dict(np.load(p))
    ft = np.load(O / "ft/scores.npz")["render_ft"] if (O / "ft/scores.npz").exists() else None
    ho = T[T.part == "hold"]
    J = dict(counts={}, tables={})
    J["counts"]["tv"] = {p: dict(rows=int((T.part == p).sum()), red=int(T.y[T.part == p].sum()), logs=int(T.log[T.part == p].nunique())) for p in ("train", "hold")}
    k0 = (R.k == 0).values & (R.real != "").values & (R.render != "").values
    J["counts"]["arena_k0"] = dict(windows=int(k0.sum()), red=int(R.y.values[k0].sum()), green=int((1 - R.y.values[k0]).sum()),
                                   logs=int(R.log[k0].nunique()), logs_green=int(R.log[k0 & (R.y == 0).values].nunique()))
    f3 = lambda t: None if t is None else [round(x, 4) for x in t]  # noqa: E731

    def tab(rows, y, strata, grp, cols):
        """cols {name: scores on rows}; same bootstrap draws for every column -> AUCs + paired differences."""
        u = np.unique(grp)
        rng = np.random.default_rng(0)
        C = np.vstack([np.ones(len(u)), np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B_BOOT)])])
        res, dr = {}, {}
        for k, s in cols.items():
            if s is None or not np.isfinite(s[rows]).all():
                continue
            r, d, n = sauc(s[rows], y[rows], strata[rows], grp[rows], C=C)
            if r is not None:
                res[k], dr[k] = dict(auc=f3(r), pairs=n), d
        return res, dr

    def diff(dr, res, a_, b_):
        if a_ in dr and b_ in dr:
            d = dr[a_] - dr[b_]
            return f3((res[a_]["auc"][0] - res[b_]["auc"][0], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))))
        return None

    # TV hold
    for strat in ("cell", "natural"):
        st = ho.cell.values if strat == "cell" else np.zeros(len(ho))
        res, _ = tab(np.ones(len(ho), bool), ho.y.values, st, ho.log.values, {k: v["tv_hold"] for k, v in SC.items()})
        J["tables"][f"tv_hold {strat}"] = res
    # arena k = 0 (rows with both frames): real vs render, every reader
    y, cell, lg = R.y.values, R.cell.values, R.log.values
    band = np.digitize(R.dF.values, DB)
    for strat, st in (("cell", cell), ("natural", np.zeros(len(R)))):
        cols = {f"{k} {src}": v[src] for k, v in SC.items() for src in ("real", "render")}
        if ft is not None:
            cols["CNN-N-FT render"] = ft
        res, dr = tab(k0, y, st, lg, cols)
        res["diff render - real (CNN-N)"] = diff(dr, res, "CNN-N render", "CNN-N real")
        res["diff CNN-N - TOK (render)"] = diff(dr, res, "CNN-N render", "TOK render")
        res["diff CNN-N - TOK (real)"] = diff(dr, res, "CNN-N real", "TOK real")
        res["diff CNN-N - CNN-L (real)"] = diff(dr, res, "CNN-N real", "CNN-L real")
        J["tables"][f"arena k0 {strat}"] = res
    for b, nm in enumerate(("[-3, 10) m", "[10, 25) m", "[25, 60) m")):
        res, _ = tab(k0 & (band == b), y, cell, lg, {f"{k} {src}": v[src] for k, v in SC.items() for src in ("real", "render")})
        J["tables"][f"arena k0 cell, line {nm}"] = res
    # tracking: within-window pairs on the windows whose lamp changes, every decision with both frames
    both = (R.real != "").values & (R.render != "").values & R.mixed.values
    res, dr = tab(both, y, R.id.values, lg, {f"{k} {src}": v[src] for k, v in SC.items() for src in ("real", "render")})
    res["diff render - real (CNN-N)"] = diff(dr, res, "CNN-N render", "CNN-N real")
    J["tables"]["tracking (within window)"] = res
    J["counts"]["tracking"] = dict(windows=int(R.id[both].nunique()), rows=int(both.sum()), red=int(y[both].sum()), logs=int(R.log[both].nunique()))
    # baked-lamp shares at the TV-hold balanced-accuracy threshold
    shares = {}
    if "CNN-N" in SC:
        sh, yh = SC["CNN-N"]["tv_hold"], ho.y.values
        cand = np.unique(sh)
        ba = [(np.mean(sh[yh == 1] >= t) + np.mean(sh[yh == 0] < t)) / 2 for t in cand[::max(1, len(cand) // 2000)]]
        tau = float(cand[::max(1, len(cand) // 2000)][int(np.argmax(ba))])
        J["tau"], J["tau_bal_acc_tv_hold"] = tau, float(max(ba))
        rc, rs = R.red_cls.fillna("").values, R.red_stop.fillna(False).astype(bool).values
        grp = {"red wait (C), k0": k0 & rs & (rc == "C"), "red wait (C), k0, no lead": k0 & rs & (rc == "C") & ~R.lead.fillna(False).astype(bool).values,
               "launch (G), k0": k0 & (rc == "G"), "launch (G), log green, any k": (R.real != "").values & (R.render != "").values & (rc == "G") & (y == 0),
               "log red, k0": k0 & (y == 1), "log green, k0": k0 & (y == 0)}
        for nm, q in grp.items():
            shares[nm] = dict(n=int(q.sum()), logs=int(R.log[q].nunique()), real=float(np.mean(SC["CNN-N"]["real"][q] >= tau)) if q.any() else None,
                              render=float(np.mean(SC["CNN-N"]["render"][q] >= tau)) if q.any() else None)
    J["shares_called_red"] = shares
    # gates and branch (prereg)
    g = lambda t, k: (J["tables"][t].get(k) or {}).get("auc", [np.nan])[0]  # noqa: E731
    g1, g2 = g("tv_hold cell", "CNN-N"), g("arena k0 cell", "CNN-N real")
    Rr, Tr = g("arena k0 cell", "CNN-N render"), g("arena k0 cell", "TOK render")
    dd = J["tables"]["arena k0 cell"].get("diff CNN-N - TOK (render)") or [np.nan] * 3
    ftr = g("arena k0 cell", "CNN-N-FT render")
    trk = J["tables"]["tracking (within window)"]
    tr_real, tr_ren = (trk.get(k, {}).get("auc", [np.nan])[0] for k in ("CNN-N real", "CNN-N render"))
    if not (g1 >= 0.85 and g2 >= 0.80):
        br = "inconclusive: probe too weak (G1 / G2)"
    elif Rr >= 0.85:
        br = "(b) colour in the pixels, not carried by the frozen eye" if dd[0] >= 0.05 and dd[1] > 0 else "(b0) colour in the pixels and in the tokens"
    elif Rr <= 0.65:
        br = "(m) present but domain-shifted (render fine-tune >= 0.80)" if ftr >= 0.80 else (
            "(a) colour absent in the render; lamp frozen" if (tr_ren <= 0.65 and tr_real >= 0.80) else "(a) colour absent in the render")
    else:
        br = "(m) intermediate"
    J["gates"] = dict(G1_tv_hold=g1, G2_real_arena=g2, R_render=Rr, TOK_render=Tr, diff=dd, ft_render=ftr, track_real=tr_real, track_render=tr_ren)
    J["branch"] = br
    json.dump(J, open(O / "eval.json", "w"), indent=1, default=float)
    print(json.dumps(dict(gates=J["gates"], branch=br), indent=1, default=float))


def cmd_report(a):
    J = json.load(open(a.src))
    f = lambda t: "-" if not t else f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"  # noqa: E731
    L = [f"branch: **{J['branch']}**", "", f"gates: {J['gates']}", "", f"counts: {J['counts']}", ""]
    for nm, r in J["tables"].items():
        L += [f"### {nm}", "", "| reader | AUC [95 % log CI] | pairs |", "|---|--:|--:|"]
        L += [f"| {k} | {f(v['auc'])} | {v['pairs']} |" if isinstance(v, dict) else f"| **{k}** | {f(v)} | |" for k, v in r.items()]
        L.append("")
    L += [f"### share called red (tau {J.get('tau', float('nan')):.3f}, TV-hold balanced accuracy {J.get('tau_bal_acc_tv_hold', float('nan')):.3f})", "",
          "| rows | n | logs | real | render |", "|---|--:|--:|--:|--:|"]
    p = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    L += [f"| {k} | {v['n']} | {v['logs']} | {p(v['real'])} | {p(v['render'])} |" for k, v in J["shares_called_red"].items()]
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "frames", "cnn", "ft", "tok", "eval", "report"))
    ap.add_argument("--var", default="N", choices=("N", "L"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--src", default=str(O / "eval.json"))
    a = ap.parse_args()
    dict(select=cmd_select, frames=cmd_frames, cnn=cmd_cnn, ft=cmd_ft, tok=cmd_tok, eval=cmd_eval, report=cmd_report)[a.cmd](a)
