"""WA-JEPA (NAVSIM specialist, released weights) zero-shot on WOD-E2E val (plans/2026-10-08-wa-xboard-prereg.md, results/wa_xboard.md).
Box, jevdrive env (.venv), CPU; the model itself runs in the wajepa env (wa_xboard_run.py).

  g0       renderer checks: NAVSIM undistort/project round trip, WOD identity render, coverage of the virtual cameras -> results/wa_xboard/g0.json
  render   --variant V1|V2 [--limit N]  the 3-camera 512 x 256 images of every history frame of the targets -> $DATA_DIR/runs/op_parity/wa_xboard/<variant>/
  req      --limit N                    the request files of all arms (V1, V2, V3, IMG0, STATE0) -> .../req_<arm>.npz
  convert  --tags ...                   wa_xboard_run output -> preds/op_cinque_<tag>/<name>.npz (`wod`: 20 x 2, xcv) and <tag>-xca
  montage                               the input images of 10 targets under V1 / V2 -> figs/wa_xboard/inputs.png
  report                                results/wa_xboard/*.csv and the numbers for results/wa_xboard.md

Mapping (all arms): 2 Hz x 4 frames = WOD frames f, f-5, f-10, f-15 (oldest first), cameras [L0, F0, R0, B0] per frame (B0 black); history poses = pp_wod.wod_ego
pose (past_states 9 / 11 / 13 / 15, already relative to the current pose); ego = vx (from positions), vy 0, ax, ay (given); command intent -> [left, straight, right, unknown].
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, io, json, os  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

H, W = 256, 512
SS = 2                                                        # supersampling of the reprojection before INTER_AREA
NAV_CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")
NAV_K = np.array([[1545.0, 0, 960.0], [0, 1545.0, 560.0], [0, 0, 1]])
NAV_D = np.array([-0.356, 0.173, -0.002, 0.0, -0.052])
NAV_WH = (1920, 1080)
STRIDE, NHIST = 5, 4
ARMS = ("V1", "V2", "V3", "IMG0", "STATE0")
OUT = _R / "experiments/op_parity/results/wa_xboard"


def work_dir():
    from jevdrive.common import data_dir
    d = data_dir() / "runs/op_parity/wa_xboard"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hist_names(name):
    seq, f = name.rsplit("-", 1)
    return [f"{seq}-{int(f) - STRIDE * k}" for k in range(NHIST - 1, -1, -1)]               # oldest first


def targets():
    from jevdrive import wod_zeroshot as Z
    S = Z.load_sets()
    return S, [str(n) for k in ("rater", "extra") for n in S[k]["name"]]


# ---------------------------------------------------------------- geometry

def nav_rays(cam_R, ss=SS):
    """Unit rays in the ego frame (ss*H, ss*W, 3) of a NAVSIM camera (K, Brown-Conrady distortion, sensor2ego rotation cam_R) through the 512 x 256 grid
    that WA-JEPA's cv2.resize of the 1920 x 1080 frame samples (pixel centres)."""
    import cv2
    h, w = H * ss, W * ss
    u, v = np.meshgrid((np.arange(w) + 0.5) / w * NAV_WH[0] - 0.5, (np.arange(h) + 0.5) / h * NAV_WH[1] - 0.5)
    pts = np.stack([u, v], -1).reshape(-1, 1, 2)
    xy = cv2.undistortPointsIter(pts, NAV_K, NAV_D, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-10)).reshape(h, w, 2)
    ray = np.concatenate([xy, np.ones((h, w, 1))], -1) @ np.asarray(cam_R, np.float64).T
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


_RAYS = {}


def nav_rig():
    """Rotations of L0 / F0 / R0 (the box's navtest index: one frame is enough, the rig is constant) as constants measured on 2021.08.30 log."""
    if not _RAYS:
        from jevdrive import navsim_zs as Z
        e = Z.load_index("navtest")[100]
        cams = e["cams"][-1]
        for c in NAV_CAMS:
            _RAYS[c] = nav_rays(np.asarray(cams[c]["R"], np.float64))
    return _RAYS


def wod_maps(calib, rays_by_cam):
    """Per sequence: source camera index and pixel (U, V) of every ray of the virtual cameras, over the WOD FRONT / FRONT_LEFT / FRONT_RIGHT."""
    from jevdrive import camgeom as G
    cal = {c: {"intrinsic": np.asarray(calib[str(c)]["intrinsic"], np.float64), "extrinsic": np.asarray(calib[str(c)]["extrinsic"], np.float64),
               "width": calib[str(c)]["width"], "height": calib[str(c)]["height"]} for c in (1, 2, 3)}
    return {k: G.choose_sources(np, r, cal) for k, r in rays_by_cam.items()}


def reproject(maps, imgs):
    """V1: remap the three WOD images into one virtual camera, supersampled then INTER_AREA to 512 x 256 (as WA-JEPA's cv2.resize of the source frame)."""
    import cv2
    from jevdrive import camgeom as G
    src, U, V = maps
    out = G.render_np(src, U, V, imgs)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)


def crop2(img, cy):
    """V2: the band of 2:1 aspect around the principal row, full width, INTER_AREA to 512 x 256."""
    import cv2
    h, w = img.shape[:2]
    bh = w // 2
    y0 = int(np.clip(round(cy - bh / 2), 0, h - bh))
    return cv2.resize(img[y0:y0 + bh], (W, H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- render

_ctx = {}


def _init(spans, calib, shard_dir, variant):
    _ctx.update(spans=spans, calib=calib, shard_dir=_pl.Path(shard_dir), variant=variant, rays=None, seq=None, maps=None, fh={})


def _decode(span):
    from PIL import Image
    sp = span
    f = _ctx["fh"].get(sp[0]) or _ctx["fh"].setdefault(sp[0], open(_ctx["shard_dir"] / sp[0], "rb"))
    ims = []
    for k in range(3):
        f.seek(sp[1 + 2 * k])
        ims.append(np.asarray(Image.open(io.BytesIO(f.read(sp[2 + 2 * k]))).convert("RGB")))
    return ims


def render_unit(unit):
    """unit = (sequence, [frame names]) -> [(name, (3, H, W, 3) uint8: L0 F0 R0)]"""
    seq, names = unit
    cal = _ctx["calib"][seq]
    if _ctx["variant"] == "V1":
        if _ctx["rays"] is None:
            _ctx["rays"] = nav_rig()
        maps = wod_maps(cal, _ctx["rays"])
    out = []
    for n in names:
        ims = _decode(_ctx["spans"][n])                                                 # FRONT, FRONT_LEFT, FRONT_RIGHT
        if _ctx["variant"] == "V1":
            out.append((n, np.stack([reproject(maps[c], ims) for c in NAV_CAMS])))
        else:
            cy = [cal[str(c)]["intrinsic"][3] for c in (1, 2, 3)]
            out.append((n, np.stack([crop2(ims[1], cy[1]), crop2(ims[0], cy[0]), crop2(ims[2], cy[2])])))   # L0 <- FRONT_LEFT, F0 <- FRONT, R0 <- FRONT_RIGHT
    return out


def frame_list(limit=0):
    S, names = targets()
    if limit:
        names = names[:limit]
    fr = sorted({h for n in names for h in hist_names(n)})
    return names, fr


def cmd_render(a):
    from jevdrive import wod_zeroshot as Z
    from jevdrive.common import data_dir, n_cpus
    from jevdrive.run import Run
    spans, _ = Z.load_spans()
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    names, fr = frame_list(a.limit)
    assert all(f in spans for f in fr), [f for f in fr if f not in spans][:5]
    d = work_dir() / (a.variant + (f"_n{a.limit}" if a.limit else ""))
    d.mkdir(exist_ok=True)
    with Run("op_parity", f"wa_xboard_render_{a.variant}", config=vars(a)) as run:
        by = {}
        for f in fr:
            by.setdefault(f.rsplit("-", 1)[0], []).append(f)
        units = list(by.items())
        mm = np.lib.format.open_memmap(d / "cache.npy.tmp", "w+", np.uint8, (len(fr) * 3, H, W, 3))
        row = {f: i for i, f in enumerate(fr)}
        shard_dir = data_dir() / "datasets/waymo_e2e/front3"
        nw = a.workers or max(1, min(64, n_cpus() - 2))
        run.info("%d targets, %d frames in %d sequences, %d workers", len(names), len(fr), len(units), nw)
        with ProcessPoolExecutor(nw, initializer=_init, initargs=(spans, calib, str(shard_dir), a.variant)) as ex:
            for res in run.tqdm(ex.map(render_unit, units, chunksize=1), total=len(units), desc="sequences"):
                for n, arr in res:
                    mm[row[n] * 3:row[n] * 3 + 3] = arr
        mm.flush()
        del mm
        # row i * 3 + c <-> key "<frame>|<cam>"; the runner's cache wants (unique path array, memmap) with rows in the order of the array
        keys = np.array([f"{f}|{c}" for f in fr for c in ("L0", "F0", "R0")])
        os.replace(d / "cache.npy.tmp", d / "cache.npy")
        np.save(d / "paths.npy", keys)
        (d / "done").touch()
        run.summary.update(frames=len(fr), variant=a.variant)


# ---------------------------------------------------------------- requests

def cmd_req(a):
    import pp_wod as PW
    from jevdrive import waymo as W_
    from jevdrive.run import Run
    S, names = targets()
    if a.limit:
        names = names[:a.limit]
    past = np.concatenate([S["rater"]["past"], S["extra"]["past"]])[:len(names)]
    intent = np.concatenate([S["rater"]["intent"], S["extra"]["intent"]])[:len(names)]
    _, pose = PW.wod_ego(past, intent)                                              # (n, 4, 3) relative to the current pose
    kin = W_.past_kinematics(past)
    ego = np.stack([kin["v"], np.zeros(len(names)), past[:, -1, 4], past[:, -1, 5]], 1).astype(np.float32)
    cmd = np.full(len(names), 3, np.int64)
    for i, c in ((1, 1), (2, 0), (3, 2)):                                           # WOD intent -> NAVSIM [left, straight, right, unknown]
        cmd[intent == i] = c
    hn = [hist_names(n) for n in names]
    base = {"keys": np.array(names), "hist": pose.astype(np.float32), "ego": ego, "cmd": cmd}

    def img(cams):                                                                  # (n, 16): frame-major, [L0, F0, R0, B0]
        return np.array([[f"{h}|{c}" if c in cams else "" for h in hs for c in ("L0", "F0", "R0", "B0")] for hs in hn])
    d = work_dir()
    z0 = dict(base, hist=np.zeros_like(base["hist"]), ego=np.zeros_like(ego), cmd=np.full(len(names), 3, np.int64))
    arms = {"V1": dict(base, img=img(("L0", "F0", "R0"))), "V2": dict(base, img=img(("L0", "F0", "R0"))), "V3": dict(base, img=img(("F0",))),
            "IMG0": dict(base, img=img(())), "STATE0": dict(z0, img=img(("L0", "F0", "R0")))}
    with Run("op_parity", "wa_xboard_req", config=vars(a)) as run:
        for k, z in arms.items():
            np.savez(d / f"req_{k}{a.suffix}.npz", **z)
        run.summary.update(n=len(names), cmd_counts=np.bincount(cmd, minlength=4).tolist())
        run.info("requests for %d targets: %s", len(names), arms.keys())


# ---------------------------------------------------------------- horizon

def interp16(traj):
    """(n, 8, 3) WA-JEPA poses at 0.5 ... 4.0 s -> (n, 20, 2) with the first 16 lattice points (0.25 ... 4.0 s) linearly interpolated from the origin."""
    xy = np.concatenate([np.zeros((len(traj), 1, 2)), traj[..., :2].astype(np.float64)], 1)       # t = 0, 0.5, ..., 4.0
    t, tp = np.arange(1, 17) * 0.25, np.arange(9) * 0.5
    p = np.zeros((len(traj), 20, 2))
    for i in range(len(traj)):
        for j in range(2):
            p[i, :16, j] = np.interp(t, tp, xy[i, :, j])
    return p


def to_wod(traj, how="xcv"):
    """4 s -> 5 s: constant velocity (xcv, primary) or constant acceleration (xca) continuation of the 3.5 -> 4.0 s motion (wod_launch_report operators)."""
    from wod_launch_report import xca, xcv
    return {"xcv": xcv, "xca": xca}[how](interp16(traj))


def cmd_convert(a):
    from jevdrive import wod_zeroshot as Z
    for tag in a.tags:
        z = np.load(work_dir() / f"out_{tag}.npz")
        for how, suf in (("xcv", ""), ("xca", "-xca")):
            q = to_wod(z["traj"], how)
            d = Z.root("preds", f"op_cinque_wa-{tag}{suf}")
            for n, w, tr in zip(z["keys"].astype(str), q, z["traj"]):
                np.savez(d / f"{n}.npz", wod=w, traj=tr)
        print(f"{tag}: {len(q)} plans, 5 s displacement median {np.median(np.linalg.norm(q[:, -1], axis=-1)):.2f} m")


# ---------------------------------------------------------------- G0 and the input montage

def cmd_g0(a):
    import cv2
    from jevdrive import camgeom as G
    from jevdrive import navsim_zs as NZ
    from jevdrive import wod_zeroshot as Z
    from jevdrive.run import Run
    spans, _ = Z.load_spans()
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    S, names = targets()
    with Run("op_parity", "wa_xboard_g0", config=vars(a)) as run:
        e = NZ.load_index("navtest")[100]["cams"][-1]
        res = {}
        # (1) NAVSIM side: rays from the undistortion, projected back by the verified navsim_zs.project_nuplan, must hit the pixel they came from
        for c in NAV_CAMS:
            cam = {"R": e[c]["R"], "K": NAV_K, "D": NAV_D}
            rays = nav_rays(e[c]["R"], ss=1)
            uv, ok, _ = NZ.project_nuplan(rays, cam)
            u, v = np.meshgrid((np.arange(W) + 0.5) / W * NAV_WH[0] - 0.5, (np.arange(H) + 0.5) / H * NAV_WH[1] - 0.5)
            err = np.linalg.norm(uv - np.stack([u, v], -1), axis=-1)[ok]
            res[f"roundtrip_{c}"] = {"valid": float(ok.mean()), "max_px": float(err.max()), "mean_px": float(err.mean())}
        # (2) WOD side identity: render a virtual camera that is WOD FRONT itself, central +-20 degrees, must reproduce the image
        diffs = []
        for n in names[:6]:
            seq = n.rsplit("-", 1)[0]
            cal = calib[seq]
            fr = cal["1"]
            fu, fv, cu, cv_, k1, k2, p1, p2, k3 = fr["intrinsic"]
            K = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]])
            w, h = fr["width"], fr["height"]
            u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            xy = cv2.undistortPointsIter(np.stack([u, v], -1).reshape(-1, 1, 2), K, np.array([k1, k2, p1, p2, k3]), None, None,
                                         (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-10)).reshape(h, w, 2)
            Rc = np.asarray(fr["extrinsic"], np.float64).reshape(4, 4)[:3, :3]
            ray = np.stack([np.ones((h, w)), -xy[..., 0], -xy[..., 1]], -1) @ Rc.T
            ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
            maps = wod_maps(cal, {"x": ray})["x"]
            ims = _decode_one(spans[n])
            out = G.render_np(*maps, ims)
            mid = np.abs(xy[..., 0]) < np.tan(np.radians(20))
            ok = mid & (maps[0] == 0)
            diffs.append(float(np.abs(out.astype(float) - ims[0].astype(float))[ok].mean()))
        res["wod_identity_mean_abs_gray"] = {"per_target": diffs, "max": max(diffs)}
        # (3) coverage of the virtual cameras by WOD front3 (share of pixels with a source)
        cov = {c: [] for c in NAV_CAMS}
        rig = nav_rig()
        for n in names[:: max(1, len(names) // 12)][:12]:
            maps = wod_maps(calib[n.rsplit("-", 1)[0]], rig)
            for c in NAV_CAMS:
                cov[c].append(float((maps[c][0] >= 0).mean()))
        res["coverage"] = {c: {"mean": float(np.mean(v)), "min": float(np.min(v))} for c, v in cov.items()}
        res["pass"] = bool(res["wod_identity_mean_abs_gray"]["max"] < 2.0 and all(r["max_px"] < 0.05 for k, r in res.items() if k.startswith("roundtrip")))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "g0.json").write_text(json.dumps(res, indent=1))
        run.info("G0 %s", json.dumps(res))
        run.summary.update(res)
        if not res["pass"]:
            raise SystemExit("G0 failed: " + json.dumps(res))


def _decode_one(span):
    from PIL import Image
    with open(__import__("jevdrive.common", fromlist=["x"]).data_dir() / "datasets/waymo_e2e/front3" / span[0], "rb") as f:
        out = []
        for k in range(3):
            f.seek(span[1 + 2 * k])
            out.append(np.asarray(Image.open(io.BytesIO(f.read(span[2 + 2 * k]))).convert("RGB")))
    return out


def cmd_montage(a):
    """The model inputs of a few targets: rows = target, columns = V1 [L0 F0 R0] | V2 [L0 F0 R0], newest frame."""
    import cv2
    d = work_dir()
    z = np.load(d / f"req_V1{a.suffix}.npz")
    pa = {v: np.load(d / v / "paths.npy") for v in ("V1", "V2")}
    mm = {v: np.load(d / v / "cache.npy", mmap_mode="r") for v in ("V1", "V2")}
    rows = []
    for i in np.linspace(0, len(z["keys"]) - 1, a.n).astype(int):
        tiles = []
        for v in ("V1", "V2"):
            row = {p: j for j, p in enumerate(pa[v])}
            for c in (0, 1, 2):
                tiles.append(np.array(mm[v][row[str(z["img"][i][12 + c])]]))
        rows.append(np.concatenate(tiles, 1))
    img = np.concatenate(rows, 0)
    (_R / "experiments/op_parity/figs/wa_xboard").mkdir(parents=True, exist_ok=True)
    out = _R / "experiments/op_parity/figs/wa_xboard/inputs.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("render")
    p.add_argument("--variant", required=True, choices=["V1", "V2"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p = sp.add_parser("req")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--suffix", default="")
    p = sp.add_parser("convert")
    p.add_argument("--tags", nargs="+", required=True)
    sp.add_parser("g0")
    p = sp.add_parser("montage")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--suffix", default="")
    a = ap.parse_args()
    {"render": cmd_render, "req": cmd_req, "convert": cmd_convert, "g0": cmd_g0, "montage": cmd_montage}[a.cmd](a)
