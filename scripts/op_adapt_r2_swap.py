"""op-adapt round 2, §4.4 pedestrian-region swap inputs (package C; op-train venv, one GPU): every x+ frame with its
pedestrian region (mask dilated by 24 px, a 49 x 49 square as the Cosmos controls' k24) replaced by the x- pixels, then
the frozen trunk again. Every context frame is swapped with its own mask, so the 9-slot context carries no pedestrian.

  sim  Cosmos test pairs (T's cosC / cosK readout rows): C+ <- C- and K+ <- K- on the gt.npz hazard-walker mask
       -> R2/cache-swap/sim/<pair>.npy fp16 (48, 1024, 8, 16), rows = stream (0 C+swap, 1 K+swap) * 24 + slot
  p5   the P5 v1 BA plus streams holding the pedestrian-scope observation frames (index/p5_obs.parquet fn_plus): the
       segmentation view is not stored, so the walker mask is rebuilt per camera (front, front_left, front_right) as
       the x+ / x- difference (max |dYCbCr| > 30, 3x3 opening) inside the walker hazards' projected 3D boxes (+ 8 px),
       frames matched by tick k (pre-divergence, so the worlds differ only by the hazard)
       -> R2/cache-swap/p5/<stream>.npy fp16 (n, 1024, 8, 16), rows = stream slots
  index  index/{simC,simK,p5}_swap.parquet: one row per swapped x+ readout row, `src_uid` = the unswapped row, plus the
       usual file / row / ctx / tc columns; checks/C_swap.json (pixels outside the dilated mask unchanged, inside = x-)

  CUDA_VISIBLE_DEVICES=3 taskset -c 24-47 python scripts/op_adapt_r2_swap.py --workers 20
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["P5_SET"] = "carla_p5v1_ba"
import op_adapt_r2_cache as RC  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_r2_data as C  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

K24 = np.ones((49, 49), np.uint8)
DIFF, BOX_PAD = 30, 8
_S = {}


def _check(plus, minus, out, dil):
    """Unit check: out == plus outside the dilated mask, == minus inside (counts of violating pixels)."""
    return int((out[~dil] != plus[~dil]).sum()), int((out[dil] != minus[dil]).sum())


# ---------------------------------------------------------------- sim
def sim_job(pair):
    import cv2
    import cosmos_openpilot as CO
    from jevdrive.cosmos_full import main_dir
    d = main_dir() / "pairs" / pair
    gt = np.load(d / "gt.npz")
    n = RC.H_ * RC.W_ // 8
    ts = range(0, RC.T_, C.STEP)
    dil = np.stack([cv2.dilate(np.unpackbits(gt["mask"][t])[: RC.H_ * RC.W_].reshape(RC.H_, RC.W_), K24) > 0 for t in ts])
    sup = np.stack([np.unpackbits(gt["support"][t * n:(t + 1) * n]).reshape(RC.H_, RC.W_) for t in ts]).astype(bool)
    cp, cm, kp, km = (RC.read_mp4_every(d / f"{k}.mp4") for k in ("carla_plus", "carla_minus", "cosmos_plus", "cosmos_minus"))
    kp = np.where(sup[..., None], kp, km)
    bad, frames = [0, 0], []
    for p, m in ((cp, cm), (kp, km)):
        o = np.where(dil[..., None], m, p)
        b = _check(p, m, o, np.broadcast_to(dil[..., None], o.shape))
        bad = [bad[0] + b[0], bad[1] + b[1]]
        frames.append(CO.model_frames(o, _S["idx"]))
    return pair, np.stack(frames), bad, float(dil.mean())


def _sim_init(idx):
    _S["idx"] = idx


# ---------------------------------------------------------------- p5
def _p5_init():
    import wod_zeroshot_openpilot as WZ
    import p5_openpilot as P
    from jevdrive import p5_openpilot as PP
    plan = json.loads((PP.root() / "op_plan.json").read_text())
    calibs = plan.get("calibs") or {P.SEQ: plan.get("calib")}
    WZ._init({}, calibs, ".")
    _S["calib"] = {int(c): v for c, v in calibs[P.SEQ].items()}


def _ycc(path):
    from PIL import Image
    im = Image.open(path)
    im.draft("YCbCr", im.size)
    return np.asarray(im.convert("YCbCr"))


def _boxes(W, k, cam, ego):
    """Projected 2D boxes (u0, v0, u1, v1) of the walker hazards at tick k in one P5 camera (CARLA world -> Waymo rear-axle
    frame -> camera); None when behind the camera."""
    from jevdrive import camgeom as G
    from jevdrive.cosmos_pilot import box_corners
    act = W["act"]
    sel = act["k"] == k
    out = []
    y = np.radians(ego.yaw)
    c, s = np.cos(y), np.sin(y)
    rear = np.array([ego.x, ego.y]) + C.REAR_AXLE_X * np.array([c, s])
    t = np.asarray(cam["extrinsic"], np.float64).reshape(4, 4)[:3, 3]
    for h, ty in zip(W["hazards"], W["hazard_types"]):
        if not str(ty).startswith("walker."):
            continue
        m = sel & (act["id"] == h)
        if not m.any():
            continue
        P = box_corners(act["xyz"][m][0], act["yaw"][m][0], W["kinds"][str(h)][2])
        d = P[:, :2] - rear
        v = np.c_[c * d[:, 0] + s * d[:, 1], -(-s * d[:, 0] + c * d[:, 1]), P[:, 2] - ego.z]     # x fwd, y left, z up
        u, vv, ok = G.waymo_project(np, v - t, cam)
        if (np.asarray(v - t) @ np.asarray(cam["extrinsic"]).reshape(4, 4)[:3, 0] > 0.05).all():
            out.append((u.min() - BOX_PAD, vv.min() - BOX_PAD, u.max() + BOX_PAD, vv.max() + BOX_PAD))
    return out


def p5_job(job):
    import cv2
    import p5_openpilot as P
    from jevdrive import p5_pairs as PP
    key, names, files, plus_dir, minus_dir = job
    WP, WM = PP.load_world(Path(plus_dir)), PP.load_world(Path(minus_dir))
    f2k = {int(r.frame): int(k) for k, r in WP["frames"].iterrows()}
    k2fm = {int(k): int(r.frame) for k, r in WM["frames"].iterrows()}
    idx = P._maps(P.SEQ)
    out = np.empty((len(names), 2, 6, 128, 256), np.uint8)
    bad, nsw, area = [0, 0], 0, []
    for j, (nm, trip) in enumerate(zip(names, files)):
        fnum = int(nm.rsplit("-", 1)[1])
        k = f2k.get(fnum)
        planes = []
        for ci, f in enumerate(trip):
            yp = _ycc(f)
            if k is not None and k in k2fm and k in WP["pose"].index:
                fm = Path(minus_dir) / "cams" / Path(f).parent.name / f"{k2fm[k]:07d}.jpg"
                if fm.exists():
                    ym = _ycc(fm)
                    diff = np.abs(yp.astype(np.int16) - ym).max(-1) > DIFF
                    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
                    box = np.zeros(diff.shape, bool)
                    for u0, v0, u1, v1 in _boxes(WP, k, _S["calib"][ci + 1], WP["pose"].loc[k]):
                        box[max(0, int(v0)):max(0, int(np.ceil(v1)) + 1), max(0, int(u0)):max(0, int(np.ceil(u1)) + 1)] = True
                    dil = cv2.dilate((diff & box).astype(np.uint8), K24) > 0
                    o = np.where(dil[..., None], ym, yp)
                    b = _check(yp, ym, o, np.broadcast_to(dil[..., None], o.shape))
                    bad = [bad[0] + b[0], bad[1] + b[1]]
                    nsw += int(dil.any())
                    area.append(float(dil.mean()))
                    yp = o
            planes.append(yp.reshape(-1, 3))
        cat = np.concatenate(planes + [P.BLACK])
        for m, kk in enumerate(("road", "wide")):
            out[j, m] = P.WZ._pack(cat[idx[kk]].reshape(128 * 2, 256 * 2, 3))
    return key, out, bad, nsw, float(np.mean(area)) if area else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parts", nargs="+", default=["sim", "p5", "index"])
    a = ap.parse_args()
    log = RunLog("op_adapt_r2", "logs", "C-swap")
    chk = {}
    net = None
    if "sim" in a.parts:
        import cosmos_openpilot as CO
        x = C.load_index("simC")
        pairs = sorted(x[x.split == "test"].key.unique())[: a.limit or None]
        out = C.root("cache-swap", "sim")
        todo = [p for p in pairs if not (out / f"{p}.npy").exists()]
        bad, t0 = [0, 0], time.time()
        with ProcessPoolExecutor(a.workers, initializer=_sim_init, initargs=(CO.maps(),)) as ex:
            list(ex.map(int, range(a.workers)))
            net = A.load("cinque", torch.float16).cuda()
            for i, (p, fr, b, ar) in enumerate(RC.bounded(ex, sim_job, todo, 2 * a.workers)):
                cur = torch.as_tensor(fr.reshape(-1, 2, 6, 128, 256)).cuda()
                prev = torch.as_tensor(np.concatenate([np.zeros_like(fr[:, :1]), fr[:, :-1]], 1).reshape(-1, 2, 6, 128, 256)).cuda()
                np.save(out / f"{p}.tmp.npy", RC.trunk(net, prev, cur, a.batch).cpu().numpy())
                (out / f"{p}.tmp.npy").replace(out / f"{p}.npy")
                bad = [bad[0] + b[0], bad[1] + b[1]]
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    log.info(f"sim [{i + 1}/{len(todo)}] {(time.time() - t0) / 60:.1f} min, mask area {ar:.4f}, violations {bad}")
        chk["sim"] = {"pairs": len(pairs), "outside_changed_px": bad[0], "inside_not_minus_px": bad[1]}
    if "p5" in a.parts:
        from jevdrive import p5_exam as E
        from jevdrive import p5_openpilot as PP
        t, *_ , obs, _n, _p = E.load()
        o = pd.read_parquet(C.root("index") / "p5_obs.parquet")
        p5 = C.load_index("p5")
        tf = t.set_index("frame_name")
        keys = p5.drop_duplicates("name").set_index("name").key.reindex(o.fn_plus).to_numpy()
        adir = lambda fn: str(Path([f for f in tf.at[fn, "files"] if "/front/" in f][-1]).parents[2])  # noqa: E731
        plan = {s["key"]: s for s in json.loads((PP.root() / "op_plan.json").read_text())["streams"]}
        jobs = []
        for key, g in o.assign(key=keys).groupby("key"):
            st = plan[key]
            jobs.append((key, st["names"], st["files"], adir(g.fn_plus.iloc[0]), adir(g.fn_minus.iloc[0])))
        jobs = jobs[: a.limit or None]
        out = C.root("cache-swap", "p5")
        todo = [j for j in jobs if not (out / f"{j[0]}.npy").exists()]
        bad, nsw, t0 = [0, 0], 0, time.time()
        if "equiv" in a.parts:                  # the swap path with nothing swapped must equal p5_openpilot.render
            import p5_openpilot as P
            global DIFF
            _p5_init()
            d0, DIFF = DIFF, 256
            key, fr, *_ = p5_job(jobs[0])
            DIFF = d0
            ref = P.render(jobs[0][2], P.SEQ)
            chk["p5_equiv"] = {"stream": key, "maxdiff": int(np.abs(fr.astype(int) - ref).max())}
            log.info(f"p5 equivalence (no swap vs render): {chk['p5_equiv']}")
        with ProcessPoolExecutor(a.workers, initializer=_p5_init) as ex:
            list(ex.map(int, range(a.workers)))
            net = net or A.load("cinque", torch.float16).cuda()
            for i, (key, fr, b, n, ar) in enumerate(RC.bounded(ex, p5_job, todo, 2 * a.workers)):
                cur = torch.as_tensor(fr).cuda()
                prev = torch.cat([torch.zeros_like(cur[:1]), cur[:-1]])
                np.save(out / f"{key}.tmp.npy", RC.trunk(net, prev, cur, a.batch).cpu().numpy())
                (out / f"{key}.tmp.npy").replace(out / f"{key}.npy")
                bad, nsw = [bad[0] + b[0], bad[1] + b[1]], nsw + n
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    log.info(f"p5 [{i + 1}/{len(todo)}] {(time.time() - t0) / 60:.1f} min, swapped camera frames {nsw}, violations {bad}")
        chk["p5"] = {"streams": len(jobs), "outside_changed_px": bad[0], "inside_not_minus_px": bad[1], "swapped_camera_frames": nsw}
    if "index" in a.parts:
        for dom, s0 in (("simC", 0), ("simK", 1)):
            x = C.load_index(dom)
            x = x[(x.split == "test") & (x.sign > 0)].copy()
            x = x[[(C.root("cache-swap", "sim") / f"{k}.npy").exists() for k in x.key]]
            x["src_uid"] = x.uid.to_numpy()
            x["file"] = [C.rel(C.root("cache-swap", "sim") / f"{k}.npy") for k in x.key]
            x["row"] = s0 * C.NSLOT + x.slot.to_numpy()
            x["ctx"] = [s0 * C.NSLOT + np.arange(j - C.CTX + 1, j + 1) for j in x.slot]
            x = C._finish(x.drop(columns=["uid", "domain"]), f"{dom}_swap")
            x.to_parquet(C.root("index") / f"{dom}_swap.parquet", index=False)
            log.info(f"index {dom}_swap: {len(x)} rows")
        p5 = C.load_index("p5")
        o = pd.read_parquet(C.root("index") / "p5_obs.parquet")
        x = p5[p5.uid.isin(o.uid_plus)].copy()
        x = x[[(C.root("cache-swap", "p5") / f"{k}.npy").exists() for k in x.key]]
        x["src_uid"] = x.uid.to_numpy()
        x["file"] = [C.rel(C.root("cache-swap", "p5") / f"{k}.npy") for k in x.key]
        x = C._finish(x.drop(columns=["uid", "domain"]), "p5_swap")
        x.to_parquet(C.root("index") / "p5_swap.parquet", index=False)
        log.info(f"index p5_swap: {len(x)} rows")
    if chk:
        chk["pass"] = all(v["outside_changed_px"] == 0 and v["inside_not_minus_px"] == 0 for v in chk.values() if isinstance(v, dict))
        (C.root("checks") / "C_swap.json").write_text(json.dumps(chk, indent=1))
        log.info(f"check {chk}")
    log.event("end")


if __name__ == "__main__":
    main()
