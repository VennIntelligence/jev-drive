"""junction_eyes field-of-view arms (plan 4.1 / 5.1): the 90 deg wide slot of L-W90 and the three-view pixels of L-3V.
Paths, layouts and check numbers: experiments/junction_eyes/results/pixels.md section 4.

  coverage --datas D ...  per calibration of CAM_L0 / F0 / R0: which camera each pixel of the 90 deg wide frame takes (frames.wide_K(256),
                          512 x 256: CAM_F0 wherever it sees, else the side camera nearest its axis), seam bearings; the slot mask = the
                          rows of 2 x 2 blocks every calibration sees across the full width (a horizontal band) -> $RUN/px90/{mask.npy, coverage.json}
  build --data D          CPU render of the t0 pair's wide slot at 90 deg -> $RUN/px90/<data>/w90-t0.npy (N, 2, 6, 128, 256) uint8
                          = [prev, cur] wide frames, rows as px/<data>/frames-t0.npy (the road slot stays that file's). cur = the t0 keys
                          of the three cameras reprojected by pure rotation (nearest sampling of libjpeg YCbCr, chroma 2 x 2 mean, as
                          navsim_zs.OpenpilotMaps); prev = cur warped to the ego pose 0.2 s earlier (op_interp.warp_view at the same K,
                          the F0 cache's warp); outside the mask black (Y 0, Cb Cr 128), the same pixels for every row
  keep                    the keep set's frames with the wide slot warped from comma's wide camera (openpilot's pinhole model, focal 567)
                          straight to wide_K(256), masked -> $RUN/keep_w90/frames.npy (KeepStore layout; rows.npz and teacher files are
                          links to keep/: the teacher stays the factory model on the 58.7 deg frames)
  link3v --datas D ...    hard links of the vis_train side cache (vt_side.py build: CAM_L0 / CAM_R0 t0 pairs, same rows) -> $RUN/px_side/<data>/
  check --datas D ...     the checks of pixels.md section 4 -> $RUN/px90/check.json, figure $RUN/px90/px_w90_3v.png

RUN = $DATA_DIR/runs/junction_eyes. D = navtrain_full.s<i>of12 | lb_navtest. Workers = the cores of the pool's grant.
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
if _sys.argv[1:2] in (["build"], ["keep"]):     # one thread per render worker (set before numpy / OpenCV load)
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_k] = "1"
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/vis_train/scripts")]
import argparse, json, pickle, shutil, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.openpilot.frames import SBIGMODEL_K, VIEW_FROM_DEVICE, MODEL_W, MODEL_H, wide_K  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

RUN = data_dir() / "runs" / "junction_eyes"
PX, P90, P3V, KEEP = RUN / "px", RUN / "px90", RUN / "px_side", RUN / "keep"
SIDE = data_dir() / "runs" / "vis_train" / "px_side"           # vt_side.py build (2026-10-11)
CR = data_dir() / "runs" / "op_parity" / "cache"
K90 = wide_K(256)                                              # +-45 deg, horizon row 151.8 (53.1 deg of rows)
VIEWS = ("CAM_F0", "CAM_L0", "CAM_R0")
FRAME1 = (6, 128, 256)                                         # one packed view
FB1 = int(np.prod(FRAME1))
VERSION = "w90-1"
FLOOR_GB = 300                                                 # this line's disk floor (plan 5.1)
HALF_FOV_F0 = 29.4                                             # the factory wide slot: atan(256 / 455)


def model_rays(K) -> np.ndarray:
    """(256, 512, 3) ego-frame rays (x fwd, y left, z up) of the integer model pixels, exactly as navsim_zs.OpenpilotMaps."""
    uu, vv = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
    ray_dev = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(K @ VIEW_FROM_DEVICE).T
    return ray_dev * np.array([1., -1., -1.])


def half(c):
    """2 x 2 mean rounded half to even in integers (OpenpilotMaps' chroma)."""
    c = c.astype(np.uint16)
    s = c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2]
    return ((s + 1 + ((s >> 2) & 1)) >> 2).astype(np.uint8)


def pack1(Y, U, V):
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], U, V])


class StitchMaps:
    """One calibration {CAM_F0, CAM_L0, CAM_R0: nuPlan cam dict} -> one packed model view at intrinsics K: every pixel's ray, rotated
    only (no parallax), sampled nearest from CAM_F0 if CAM_F0 sees it, else from the side camera that sees it nearest its optical
    axis; pixels no camera sees take CAM_F0's clipped border (as OpenpilotMaps; the slot mask blacks them out)."""

    def __init__(self, cams: dict, K=K90, views=VIEWS):
        from jevdrive import navsim_zs as Z
        w, h = Z.NUPLAN_WH
        rays = model_rays(K)
        self.pick = np.full((MODEL_H, MODEL_W), -1, np.int8)
        idx, best = np.zeros((MODEL_H, MODEL_W), np.int64), np.full((MODEL_H, MODEL_W), -2.0)
        okF = None
        for j, c in enumerate(views):
            uv, ok, cs = Z.project_nuplan(rays, cams[c], 1)
            xi = np.clip(np.rint(uv[..., 0]), 0, w - 1).astype(np.int64)
            yi = np.clip(np.rint(uv[..., 1]), 0, h - 1).astype(np.int64)
            flat = j * w * h + yi * w + xi
            if c == "CAM_F0":
                okF, take = ok, ok
                idx[:] = flat
            else:
                take = ok & ~okF & (cs > best)
            idx[take], best[take], self.pick[take] = flat[take], cs[take], j
        self.n = len(views)
        self._take = [3 * idx + c for c in range(3)]

    def __call__(self, yccs) -> np.ndarray:
        """Full-res YCbCr images of the views, in `views` order -> (6, 128, 256) uint8."""
        f = np.concatenate([np.ascontiguousarray(x).reshape(-1) for x in yccs[:self.n]])
        Y, U, V = (f.take(i) for i in self._take)
        return pack1(Y, half(U), half(V))


def apply_mask(p, m):
    """Packed view(s) (..., 6, 128, 256), block mask m (128, 256) True = seen: black outside (Y 0, Cb Cr 128), in place."""
    p[..., :4, ~m] = 0
    p[..., 4:, ~m] = 128
    return p


def load_mask():
    f = P90 / "mask.npy"
    assert f.exists(), f"{f}: run `coverage` first"
    return np.load(f)


def ents(data):
    """F0 inputs (vt_px ents: t0 key path, camera position, pose / vel) and side inputs (vt_side ents: L0 / R0 key dicts, the warp's
    poses), in the tab.npz row order."""
    with open(PX / data / "ents.pkl", "rb") as f:
        ef = pickle.load(f)
    with open(SIDE / data / "ents.pkl", "rb") as f:
        es = pickle.load(f)
    assert [e["token"] for e in ef] == [e["token"] for e in es] == np.load(CR / data / "tab.npz")["names"].tolist(), data
    return ef, es


def cams_of(ef, es) -> dict:
    return {"CAM_F0": ef["cams"][-1]["CAM_F0"], "CAM_L0": es["cams"]["CAM_L0"][1], "CAM_R0": es["cams"]["CAM_R0"][1]}


_MAPS = {}


def stitch(cams: dict, K=K90, views=VIEWS) -> StitchMaps:
    from jevdrive import navsim_zs as Z
    k = (Z.calib_key({c: cams[c] for c in views}), np.asarray(K).tobytes(), views)
    return _MAPS.get(k) or _MAPS.setdefault(k, StitchMaps(cams, K, views))


def decode_all(cams: dict, views=VIEWS):
    from jevdrive import navsim_zs as Z
    return [Z.OpenpilotMaps.decode(cams[c]["path"]) for c in views]


def render_row(ef, es, mask, ycc=None):
    """One row -> (2, 6, 128, 256) [prev, cur] 90 deg wide frames, masked."""
    from jevdrive import op_interp as I
    cams = cams_of(ef, es)
    cur = stitch(cams)(ycc if ycc is not None else decode_all(cams))
    prev = I.warp_view(cur, "wide", ef["cam"], es["pose"][0], es["pose"][1], K90)
    return apply_mask(np.stack([prev, cur]), mask)


# ---------------------------------------------------------------- coverage: the slot mask
def cmd_coverage(a):
    from jevdrive import navsim_zs as Z
    with Run("junction_eyes", "w90-coverage", config=vars(a)) as run:
        cal = {}
        for d in a.datas:
            ef, es = ents(d)
            for x, y in zip(ef, es):
                cams = cams_of(x, y)
                r = cal.setdefault(Z.calib_key(cams), {"cams": cams, "n": 0})
                r["n"] += 1
        rays = model_rays(K90)
        seen_all = np.ones((MODEL_H, MODEL_W), bool)
        hr = int(round(K90[1, 2]))
        bear = np.degrees(np.arctan2(rays[hr, :, 1], rays[hr, :, 0]))
        rows = []
        for r in cal.values():
            S = StitchMaps(r["cams"])
            seen = S.pick >= 0
            seen_all &= seen
            p = S.pick[hr]
            ch = np.flatnonzero(p[1:] != p[:-1])
            seams = [dict(at=float((bear[i] + bear[i + 1]) / 2), left=int(p[i]), right=int(p[i + 1])) for i in ch]
            rows.append(dict(n=r["n"], seams=seams, seen=float(seen.mean()), seen_horizon=float(seen[hr].mean()),
                             yaw={c: Z.cam_yaw_deg(r["cams"][c]) for c in VIEWS},
                             reach={c: view_reach(r["cams"][c], c) for c in VIEWS}))
        seen_blk = seen_all[0::2, 0::2] & seen_all[1::2, 0::2] & seen_all[0::2, 1::2] & seen_all[1::2, 1::2]
        blk = np.repeat(seen_blk.all(1, keepdims=True), seen_blk.shape[1], 1)    # a horizontal band: the block rows seen across the width
        np.save(P90 / "mask.npy", blk)
        colc = blk[:, MODEL_W // 4]                                          # the centre column of the block grid
        vr = np.flatnonzero(colc)
        el = lambda v: float(np.degrees(np.arctan2(K90[1, 2] - v, K90[1, 1])))  # noqa: E731
        reachL = min(x["reach"]["CAM_L0"][1] for x in rows)
        reachR = max(x["reach"]["CAM_R0"][0] for x in rows)
        allseam = [s["at"] for x in rows for s in x["seams"]]
        res = dict(datas=a.datas, tokens=int(sum(x["n"] for x in rows)), calibrations=len(rows), K=K90.tolist(),
                   mask_share=float(blk.mean()), seen_by_every_calibration_share=float(seen_blk.mean()), band_rows_share=float(colc.mean()), band_rows=[int(2 * vr[0]), int(2 * vr[-1] + 1)],
                   band_elev_deg=[el(2 * vr[-1] + 1.5), el(2 * vr[0] - 0.5)], mask_full_width_rows_share=float(blk.all(1).mean()),
                   reach_three_views_deg=[float(reachR), float(reachL)],
                   seams_horizon_deg=dict(min=float(min(allseam)), max=float(max(allseam)),
                                          left=sorted({round(s["at"], 1) for x in rows for s in x["seams"] if s["at"] > 0}),
                                          right=sorted({round(s["at"], 1) for x in rows for s in x["seams"] if s["at"] < 0})))
        run.info(json.dumps({k: v for k, v in res.items() if k not in ("K",)}))
        run.summary |= {k: res[k] for k in ("tokens", "calibrations", "mask_share", "band_rows_share")}
        (P90 / "coverage.json").write_text(json.dumps(res | dict(rows=[{k: v for k, v in x.items()} for x in rows]), indent=1))


def view_reach(cam, name):
    """Bearing range (deg, left +) a nuPlan camera sees on the horizon (native image's rays at its principal row)."""
    from jevdrive import navsim_zs as Z
    import cv2
    w, h = Z.NUPLAN_WH
    K, D = np.asarray(cam["K"], np.float64), np.asarray(cam["D"], np.float64)
    u = np.linspace(0, w - 1, 961)
    pts = np.stack([u, np.full_like(u, K[1, 2])], -1).reshape(-1, 1, 2)
    xy = cv2.undistortPointsIter(pts, K, D, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-10)).reshape(-1, 2)
    ray = np.c_[xy, np.ones(len(xy))] @ Z.cam_to_ego(cam).T
    b = np.degrees(np.arctan2(ray[:, 1], ray[:, 0]))
    return [float(b.min()), float(b.max())]


# ---------------------------------------------------------------- build
_G = {}


def _job(rows):
    if "out" not in _G:
        _G["out"] = np.lib.format.open_memmap(_G["path"], mode="r+")
    for i in rows:
        _G["out"][i] = render_row(_G["ef"][i], _G["es"][i], _G["mask"])
    return rows


def cmd_build(a):
    import pp_prep as P
    d = a.data
    ef, es = ents(d)
    N = len(ef) if not a.limit else min(a.limit, len(ef))
    mask = load_mask()
    (P90 / d).mkdir(parents=True, exist_ok=True)
    out = P90 / d / ("w90-t0.npy" if not a.limit else f"w90-t0-first{a.limit}.npy")
    with Run("junction_eyes", f"w90-build-{d}", config=vars(a)) as run:
        key = cache.key(params=dict(data=d, n=N, names=cache.key(params=dict(n=[e["token"] for e in ef])), version=VERSION, K=K90.tolist(),
                                    mask=cache.key(params=dict(m=mask.tobytes().hex()))), code=[_job, render_row, StitchMaps.__init__, StitchMaps.__call__])
        W = a.workers or max(1, len(_os.sched_getaffinity(0)))
        need, free = N * 2 * FB1 / 2 ** 30, shutil.disk_usage(P90).free / 2 ** 30
        run.info(f"{d}: {N} rows, {need:.1f} GB to write, {free:.0f} GB free (floor {FLOOR_GB} GB), {W} workers")
        if not cache.valid(out, key) and free - need < FLOOR_GB:
            raise RuntimeError(f"{d}: {free:.0f} GB free - {need:.1f} GB < {FLOOR_GB} GB floor; not written")
        st = {}

        def make():
            proot = P90 / d / (".part" + (str(a.limit) if a.limit else ""))
            proot.mkdir(exist_ok=True)
            part = st["part"] = P.Part(proot, key, N, a.force)
            part.open("w90", (N, 2) + FRAME1, np.uint8).flush()
            _G.pop("out", None)
            _G.update(ef=ef, es=es, mask=mask, path=str(proot / "w90.part.npy"))
            todo = np.flatnonzero(~part.done)
            chunks = [todo[i:i + 8] for i in range(0, len(todo), 8)]
            t0 = time.time()
            with ProcessPoolExecutor(W) as ex:
                futs = [ex.submit(_job, c) for c in chunks]
                for f in run.tqdm(as_completed(futs), total=len(futs), desc=f"w90 {d}"):
                    part.mark(f.result())
            part.save()
            run.summary |= {"rows": int(len(todo)), "workers": W, "rows_per_s": len(todo) / max(time.time() - t0, 1e-9)}
            run.info(f"{d}: {len(todo)} rows in {time.time() - t0:.0f} s")
            return None
        cache.cached(out, key, make, force=a.force, writer=lambda p, o: st["part"].commit("w90", p), reader=lambda p: None)
        run.summary |= {"n": N, "file": str(out), "gb": out.stat().st_size / 2 ** 30}


# ---------------------------------------------------------------- keep: comma wide camera straight to K90
def _keep_seg(job):
    """One segment: decode the wide camera to the last kept frame; per kept frame the K90 wide view (masked) into the out memmap;
    the factory-K warp of the window's first frame must reproduce keep/frames.npy's wide bytes."""
    from jevdrive.openpilot.frames import CAMERAS, Warper, decode_hevc, get_warp_matrix, intrinsics, load_segment_meta
    seg, wins, mask = job
    root = data_dir() / "datasets" / "comma1M" / seg
    meta = load_segment_meta(root)
    wh = meta["fcam_wh"]
    camK = intrinsics(*wh, CAMERAS[wh][1])
    w90 = Warper(get_warp_matrix(meta["rpy_calib"], camK, True, model_K=K90), wh)
    wfac = Warper(get_warp_matrix(meta["rpy_calib"], camK, True), wh)
    out = np.load(RUN / "keep_w90" / "frames.npy", mmap_mode="r+")
    ref = np.load(KEEP / "frames.npy", mmap_mode="r")
    hi = max(w["hi"] for w in wins)
    eq, n = 0, 0
    for k, planes in enumerate(decode_hevc(root / "ecamera.hevc")):
        if k >= hi:
            break
        for w in wins:
            if w["start"] <= k < w["hi"]:
                g = w["off"] + k - w["start"]
                out[g, 0] = ref[g, 0]
                out[g, 1] = apply_mask(w90(*planes), mask)
                if k == w["start"]:
                    eq, n = eq + int(np.array_equal(wfac(*planes), ref[g, 1])), n + 1
    out.flush()
    assert k + 1 >= hi, f"{seg}: decoded {k + 1} frames, need {hi}"
    return seg, eq, n


def cmd_keep(a):
    from jevdrive import par
    s = json.loads((KEEP / "selection.json").read_text())
    mask = load_mask()
    d = RUN / "keep_w90"
    d.mkdir(exist_ok=True)
    with Run("junction_eyes", "w90-keep", config=vars(a)) as run:
        F = np.load(KEEP / "frames.npy", mmap_mode="r").shape
        free = shutil.disk_usage(d).free / 2 ** 30
        run.info(f"{F}: {np.prod(F) / 2 ** 30:.1f} GB, {free:.0f} GB free")
        assert free - np.prod(F) / 2 ** 30 > FLOOR_GB, "disk floor"
        tmp = d / "frames.tmp.npy"
        np.lib.format.open_memmap(tmp, "w+", np.uint8, F).flush()
        (d / "frames.npy").unlink(missing_ok=True)
        tmp.rename(d / "frames.npy")
        by = {}
        for w in s["windows"]:
            by.setdefault(w["seg"], []).append(w)
        res = par.pmap(_keep_seg, [(g, ws, mask) for g, ws in by.items()], workers=a.workers or par.cpus(), run=run, desc="segments")
        res.raise_if_failed()
        eq, n = sum(v[1] for v in res.values), sum(v[2] for v in res.values)
        for f in ("rows.npz", "teacher_raw.npy", "teacher_temporal.npy", "teacher_vision.npy", "teacher_slices.json", "selection.json", "census.json"):
            if (KEEP / f).exists() and not (d / f).exists():
                (d / f).symlink_to(_os.path.relpath(KEEP / f, d))
        r = dict(frames=list(F), segments=len(by), factory_k_first_frames_equal=eq, of=n)
        (d / "w90_keep.json").write_text(json.dumps(r))
        run.summary |= r
        run.info(json.dumps(r))
        assert eq == n, "the factory-K warp does not reproduce keep/frames.npy"


# ---------------------------------------------------------------- 3V: the side cache under this run dir
def cmd_link3v(a):
    for d in a.datas:
        (P3V / d).mkdir(parents=True, exist_ok=True)
        assert np.load(SIDE / d / "meta.npz")["names"].tolist() == np.load(CR / d / "tab.npz")["names"].tolist(), d
        for f in ("side_t0.npy", "side_t0.npy.key", "meta.npz"):
            t = P3V / d / f
            if not t.exists():
                _os.link(SIDE / d / f, t)
        print(d, "linked", (P3V / d / "side_t0.npy").stat().st_nlink, "links")


# ---------------------------------------------------------------- check
def resample_luma(Y, K_src, K_dst, interp):
    """Luma (256, 512) of a frame at K_src -> the frame at K_dst (same rotation): pure intrinsics change."""
    import cv2
    uu, vv = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
    x, y = (uu - K_dst[0, 2]) / K_dst[0, 0], (vv - K_dst[1, 2]) / K_dst[1, 1]
    mx, my = (K_src[0, 0] * x + K_src[0, 2]).astype(np.float32), (K_src[1, 1] * y + K_src[1, 2]).astype(np.float32)
    return cv2.remap(Y, mx, my, interp, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def cmd_check(a):
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.openpilot.frames import unpack_luma
    mask = load_mask()
    lm = np.repeat(np.repeat(mask, 2, 0), 2, 1)                                # luma mask (256, 512)
    vm = resample_luma(lm.astype(np.uint8) * 255, K90, SBIGMODEL_K, cv2.INTER_LINEAR) == 255   # factory-K pixels inside the W90 mask
    res = {}
    with Run("junction_eyes", "w90-check", config=vars(a)) as run:
        run.info(f"factory-K wide pixels inside the W90 mask: {vm.mean():.4f}")
        for d in a.datas:
            ef, es = ents(d)
            f0 = np.load(PX / d / "frames-t0.npy", mmap_mode="r")
            w90 = np.load(P90 / d / "w90-t0.npy", mmap_mode="r")
            sd = np.load(P3V / d / "side_t0.npy", mmap_mode="r")
            assert len(f0) == len(w90) == len(sd) == len(ef), d
            rows = np.sort(np.random.default_rng(0).choice(len(ef), min(a.rows, len(ef)), replace=False))

            def one(i):
                cams = cams_of(ef[i], es[i])
                ycc = decode_all(cams)
                r = {}
                # (a) the renderer at the factory K with CAM_F0 only = the F0 cache's wide slot (cur and prev)
                cf = stitch(cams, SBIGMODEL_K, ("CAM_F0",))(ycc[:1])
                pf = I.warp_view(cf, "wide", ef[i]["cam"], es[i]["pose"][0], es[i]["pose"][1])
                r["a_cur"], r["a_prev"] = np.array_equal(cf, f0[i, 1, 1]), np.array_equal(pf, f0[i, 0, 1])
                # (b) stored W90 = a fresh render
                fresh = render_row(ef[i], es[i], mask, ycc)
                r["b"] = np.array_equal(fresh, w90[i])
                # (c) the central +-29.4 deg: W90 resampled back to the factory K vs the F0 wide frame (luma; cur and prev)
                for k, nm in ((1, "cur"), (0, "prev")):
                    Yf = unpack_luma(f0[i, k, 1]).astype(np.float32)
                    Yw = unpack_luma(w90[i, k])
                    back = resample_luma(Yw, K90, SBIGMODEL_K, cv2.INTER_LINEAR)
                    trip = resample_luma(resample_luma(unpack_luma(f0[i, k, 1]), SBIGMODEL_K, K90, cv2.INTER_AREA).astype(np.uint8),
                                         K90, SBIGMODEL_K, cv2.INTER_LINEAR)
                    r[f"c_{nm}"] = np.abs(back - Yf)[vm]
                    r[f"c_{nm}_trip"] = np.abs(trip - Yf)[vm]
                r["c_Yf"] = unpack_luma(f0[i, 1, 1]).astype(np.float32)[vm]
                r["c_back"] = resample_luma(unpack_luma(w90[i, 1]), K90, SBIGMODEL_K, cv2.INTER_LINEAR)[vm]
                # (d) inside +-29.4 deg the W90 pixels are CAM_F0's own: = the CAM_F0-only render at K90
                cF = apply_mask(stitch(cams, K90, ("CAM_F0",))(ycc[:1]), mask)
                yc, yw = unpack_luma(cF), unpack_luma(w90[i, 1])
                cen = np.abs(np.arctan2(np.arange(MODEL_W) - K90[0, 2], K90[0, 0])) <= np.radians(HALF_FOV_F0)
                r["d"] = int((yc[:, cen] != yw[:, cen]).sum())
                # (e) seams: luma step across each seam column vs neighbouring column steps, inside the mask
                S = stitch(cams)
                Y = yw.astype(np.float32)
                g = np.abs(np.diff(Y, axis=1))                                 # step between column j and j + 1
                hr = int(round(K90[1, 2]))
                seam_d, base_d = [], []
                for j in np.flatnonzero(S.pick[hr, 1:] != S.pick[hr, :-1]):
                    band = lm[:, j] & lm[:, j + 1]
                    seam_d.append(float(g[band, j].mean()))
                    nb = [c for c in list(range(j - 12, j - 3)) + list(range(j + 4, j + 13)) if 0 <= c < MODEL_W - 1]
                    base_d.append(float(g[band][:, nb].mean()))
                r["e"] = (seam_d, base_d, [float(np.degrees(np.arctan2(j + 0.5 - K90[0, 2], K90[0, 0]))) for j in
                                           np.flatnonzero(S.pick[hr, 1:] != S.pick[hr, :-1])])
                # (f) 3V: the CAM_F0 view through vt_side's pairing (key rendered along the ego axis, warp_frame in its yawed frame,
                #     yaw 0) = the F0 t0 cache bytes; the side views = a fresh render (cur) / warp (prev)
                import side_store as SD
                import vt_side as VS
                key = Z.calib_key({"CAM_F0": cams["CAM_F0"]})
                mF = _MAPS.get(("op", key)) or _MAPS.setdefault(("op", key), Z.OpenpilotMaps(cams["CAM_F0"]))
                cur = mF(ycc[0])
                cam_v, pd = SD.virtual(np.asarray(cams["CAM_F0"]["t"], float), 0.0, es[i]["pose"][0])
                _, ps = SD.virtual(np.asarray(cams["CAM_F0"]["t"], float), 0.0, es[i]["pose"][1])
                r["f_F0"] = np.array_equal(np.stack([I.warp_frame(cur, cam_v, pd, ps), cur]), f0[i])
                r["f_F0_cam"] = bool(np.allclose(np.asarray(cams["CAM_F0"]["t"], float), ef[i]["cam"]))
                side_cur = [VS._render(cams[c], c) for c in ("CAM_L0", "CAM_R0")]
                r["f_side_cur"] = all(np.array_equal(side_cur[c], sd[i, c, 1]) for c in range(2))
                return i, r

            with ThreadPoolExecutor(a.threads) as ex:
                out = dict(ex.map(one, rows))
            R = [out[i] for i in rows]
            cf = lambda k: np.stack([x[k] for x in R])  # noqa: E731
            st = lambda e: dict(mean_abs=float(e.mean()), p99=float(np.percentile(e, 99)), max=float(e.max()))  # noqa: E731
            sd_, bd_, at_ = [], [], []
            for x in R:
                sd_ += x["e"][0]
                bd_ += x["e"][1]
                at_ += x["e"][2]
            other = np.abs(np.roll(cf("c_back"), 1, 0) - cf("c_Yf"))
            r = dict(rows=int(len(rows)),
                     a_factory_K_F0_only_equals_F0_cache_wide=dict(cur=int(sum(x["a_cur"] for x in R)), prev=int(sum(x["a_prev"] for x in R))),
                     b_stored_equals_fresh=int(sum(x["b"] for x in R)),
                     c_central_back_to_factory_K=dict(cur=st(cf("c_cur")), prev=st(cf("c_prev")), floor_area_round_trip=st(cf("c_cur_trip")),
                                                      other_row=st(other)),
                     d_central_pixels_not_CAM_F0=int(sum(x["d"] for x in R)),
                     e_seams=dict(n=len(sd_), seam_step=float(np.mean(sd_)), neighbour_step=float(np.mean(bd_)),
                                  ratio=float(np.mean(sd_) / np.mean(bd_)), at_deg=sorted({round(v, 1) for v in at_})),
                     f_3v=dict(F0_view_equals_F0_cache=int(sum(x["f_F0"] for x in R)), F0_cam_position_same=int(sum(x["f_F0_cam"] for x in R)),
                               side_cur_equals_fresh=int(sum(x["f_side_cur"] for x in R)),
                               side_names_equal_tab=bool(np.load(P3V / d / "meta.npz")["names"].tolist() == [e["token"] for e in ef])))
            r["c_central_back_to_factory_K"]["share_of_factory_frame_compared"] = float(vm.mean())
            res[d] = r
            run.info(f"{d}: {json.dumps(r)}")
            if a.fig and d == a.fig_data:
                figure(d, ef, es, f0, w90, sd, a.fig_token, mask)
        run.summary |= {d: {k: v for k, v in r.items() if k != "c_central_back_to_factory_K"} for d, r in res.items()}
        (P90 / f"check{'-' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1))


def figure(d, ef, es, f0, w90, sd, token, mask):
    """One turn row: F0 wide (58.7 deg), W90 wide (+-29.4 deg and the seams marked), CAM_L0 wide, CAM_R0 wide (cur frames, RGB)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from jevdrive import op_interp as I
    names = [e["token"] for e in ef]
    if not token:                                                            # the sharpest junction turn of the data's read rows
        import pandas as pd
        df = pd.read_parquet(RUN / "step0" / "rows" / "all.parquet", columns=["tok_file", "token", "jdist", "turn40"])
        df = df[(df.tok_file == d) & (df.jdist <= 15)]
        token = df.token.values[np.argmax(np.abs(df.turn40.values))]
    i = names.index(token)
    S = stitch(cams_of(ef[i], es[i]))
    rgb = lambda p: I.to_rgb(torch.from_numpy(np.ascontiguousarray(p))).permute(1, 2, 0).numpy()  # noqa: E731
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.2))
    panels = [(f0[i, 1, 1], "CAM_F0 wide slot, factory K (f 455, +-29.4 deg)"), (w90[i, 1], "W90 wide slot (f 256, +-45 deg): L0 | F0 | R0"),
              (sd[i, 0, 1, 1], "CAM_L0 wide, along its mounting yaw"), (sd[i, 1, 1, 1], "CAM_R0 wide, along its mounting yaw")]
    for k, (p, t) in enumerate(panels):
        A = ax.flat[k]
        A.imshow(rgb(p))
        A.set_title(t, fontsize=10)
        A.set_xticks([]), A.set_yticks([])
    A = ax.flat[1]
    for b in (-HALF_FOV_F0, HALF_FOV_F0):
        A.axvline(K90[0, 2] - K90[0, 0] * np.tan(np.radians(b)), color="#f2c14e", lw=1, ls="--")
    hr = int(round(K90[1, 2]))
    for j in np.flatnonzero(S.pick[hr, 1:] != S.pick[hr, :-1]):
        A.axvline(j + 0.5, color="#e4572e", lw=0.8)
    fig.suptitle(f"{d} token {token} (row {i}): dashed = the factory slot's +-29.4 deg, red = seams (F0 -> side camera)", fontsize=10)
    fig.tight_layout()
    fig.savefig(P90 / "px_w90_3v.png", dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("coverage")
    p.add_argument("--datas", nargs="+", required=True)
    cli_args(p)
    p = sp.add_parser("build")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    cli_args(p)
    p = sp.add_parser("keep")
    p.add_argument("--workers", type=int, default=0)
    cli_args(p)
    p = sp.add_parser("link3v")
    p.add_argument("--datas", nargs="+", required=True)
    p = sp.add_parser("check")
    p.add_argument("--datas", nargs="+", required=True)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--tag", default="")
    p.add_argument("--fig", action="store_true")
    p.add_argument("--fig-data", default="lb_navtest")
    p.add_argument("--fig-token", default="")
    cli_args(p)
    a = ap.parse_args()
    {"coverage": cmd_coverage, "build": cmd_build, "keep": cmd_keep, "link3v": cmd_link3v, "check": cmd_check}[a.cmd](a)
