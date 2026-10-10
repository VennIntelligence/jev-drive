"""vis_train arm W: the t0 side-camera pixel cache (lib/side_store.py reads it), its field of view and its checks.

  split  --datas D ...     per data dir, in its tab.npz row order: the render inputs (CAM_L0 / CAM_R0 camera dicts of the t0 and
                           t0 - 0.5 s keys, CAM_F0's calibration) and the warp poses -> px_side/<data>/{ents.pkl, meta.npz}
                           (one job: navtrain's full index unpickles to ~30 GB)
  build  --data D          CPU render -> px_side/<data>/side_t0.npy (N, 2, 2, 2, 6, 128, 256) uint8: per camera the protocol-W pair of
                           the t0 slot (t0 key warped to the pose 0.2 s earlier, t0 key); rows written by the workers, resumable
  fov    --datas D ...     horizontal field of the three views (CAM_L0, CAM_F0, CAM_R0 model frames) per calibration, from the
                           rendering code's rays -> px_side/fov.json
  check  --datas D ...     cached bytes against a fresh render; the frozen encoder on (t0 - 0.5 s key, cached t0 key) against P3's
                           cache/<data>/side.npy (same pairing), and on the cached protocol-W pair (the expected difference)

D = navtrain_full.s<i>of12 | lb_navtest | lb_navhard. px_side = $DATA_DIR/runs/vis_train/px_side. Workers = the cores of the pool's grant.
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
if _sys.argv[1:2] == ["build"]:                 # one thread per render worker (set before numpy / OpenCV load)
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_k] = "1"
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, pickle, shutil, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

import pixel_store as PX  # noqa: E402
import side_store as SD  # noqa: E402
from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

CR = data_dir() / "runs" / "op_parity" / "cache"
VERSION = "sd1"
FRONT = [f"navtrain_full.s{i}of12" for i in range(12)] + ["lb_navtest", "lb_navhard"]    # the first wave's pixel cache (vt_px.py)
VIEWS = ("CAM_L0", "CAM_F0", "CAM_R0")


def sdir(data) -> _pl.Path:
    d = SD.side_root() / data
    d.mkdir(parents=True, exist_ok=True)
    return d


def names_of(data) -> list:
    return np.load(CR / data / "tab.npz")["names"].tolist()


def kbase(data, names) -> dict:
    return dict(data=data, n=len(names), names_sha=cache.key(params=dict(n=names)), version=VERSION)


def t_prev() -> float:
    """Time of the frame before t0 in the policy's t0 slot (protocol W: step 26 of the 31-step 20 Hz rollout), from op_lb's step
    sources: a synthesized lattice frame, while t0 itself is the last key."""
    import op_lb as OL
    import pp_prep as P
    _, src = OL._steps(0.0, False)
    (k1, j1), (k0, j0) = src[P.STEPS[-1]], src[P.STEPS[-1] - 4]
    assert (k1, j1) == ("k", 3) and k0 == "s", (src[P.STEPS[-1]], src[P.STEPS[-1] - 4])
    return float(OL.SYN_T[j0])


def _render(cam, name):
    """One side-camera key as pp_prep.render_side renders it: model frames [road, wide] along the camera's mounting yaw."""
    import pp_prep as P
    from jevdrive import navsim_zs as Z
    k = (name, Z.calib_key({name: cam}))
    m = P._MAPS.get(k) or P._MAPS.setdefault(k, Z.OpenpilotMaps(cam, yaw_deg=Z.cam_yaw_deg(cam)))
    return m(m.decode(cam["path"]))


# ---------------------------------------------------------------- split: render inputs per data dir
def cmd_split(a):
    import op_lb as OL
    import pp_prep as P
    from jevdrive import navsim_zs as Z
    tp = t_prev()
    with Run("vis_train", "side-split", config=vars(a)) as run:
        idx = {}
        for d in a.datas:
            names = names_of(d)
            full = d.startswith("navtrain_full")
            mt = None if full else OL.meta(d)
            split = "navtrain" if full else mt["split"]
            if split not in idx:
                idx.clear()                                                         # one index in RAM at a time
                idx[split] = Z.load_index(split)
                idx[split + "/by"] = {e["token"]: k for k, e in enumerate(idx[split])}
            ix, by = idx[split], idx[split + "/by"]
            if not full:
                assert mt["names"] == names, f"{d}: op_lb meta and tab.npz disagree"

            def make():
                out = []
                for i, t in enumerate(names):
                    e = ix[by[t]]
                    pose, vel = (np.asarray(e[k], float) for k in ("pose", "vel")) if full else (mt["pose"][i], mt["vel"][i])
                    sk, pp = P.warp_plan(pose, vel, [tp])
                    assert sk[0] == 3, "the frame before t0 is a warp of the t0 key"
                    out.append({"token": t, "cams": {c: (e["cams"][-2][c], e["cams"][-1][c]) for c in SD.CAMS},
                                "f0": {k: v for k, v in e["cams"][-1]["CAM_F0"].items() if k != "path"}, "pose": pp[0]})
                return out
            ents = cache.cached(sdir(d) / "ents.pkl", cache.key(params=kbase(d, names) | dict(t_prev=tp), code=[P.warp_plan]), make, force=a.force)
            np.savez(sdir(d) / "meta.npz", names=np.array(names), t_prev=tp, pose=np.stack([e["pose"] for e in ents]),
                     cam_t=np.array([[np.asarray(e["cams"][c][1]["t"], float) for c in SD.CAMS] for e in ents]),
                     yaw=np.array([[Z.cam_yaw_deg(e["cams"][c][1]) for c in SD.CAMS] for e in ents]))
            run.info(f"{d}: {len(ents)} rows, frame before t0 at {tp:+.2f} s")


# ---------------------------------------------------------------- build: CPU render, the workers write the rows
_G = {}


def _job(rows):
    """Rows -> per camera [t0 key warped to the pose at t_prev (op_interp.warp_frame in the camera's yawed vehicle frame), t0 key]."""
    from jevdrive import op_interp as I
    if "out" not in _G:
        _G["out"] = np.lib.format.open_memmap(_G["path"], mode="r+")
    cam, pd, ps = _G["virt"]
    for i in rows:
        for c, name in enumerate(SD.CAMS):
            cur = _render(_G["ents"][i]["cams"][name][1], name)
            _G["out"][i, c, 0], _G["out"][i, c, 1] = I.warp_frame(cur, cam[i, c], pd[i, c], ps[i, c]), cur
    return rows


def virt(data):
    """meta.npz -> (camera position, pose at t_prev, pose at t0), each (N, 2, 3), in every camera's yawed vehicle frame."""
    z = np.load(sdir(data) / "meta.npz")
    cam, pd = SD.virtual(z["cam_t"], z["yaw"], z["pose"][:, None, 0])
    return cam, pd, SD.virtual(z["cam_t"], z["yaw"], z["pose"][:, None, 1])[1]


def disk_after(need: int) -> float:
    """GiB that stay free on the data disk after `need` more bytes and after the first wave's pixel cache (vt_px.py: FRONT) has
    written what it has not yet allocated (planned size - blocks on disk)."""
    left = 0
    for d in FRONT:
        fs = [PX.px_root() / d / "frames.npy", PX.px_root() / d / ".part" / "frames.part.npy"]
        have = sum(f.stat().st_blocks * 512 for f in fs if f.exists())
        left += max(0, len(names_of(d)) * PX.NF * PX.FB - have)
    return (shutil.disk_usage(data_dir()).free - left - need) / 2 ** 30


def cmd_build(a):
    import pp_prep as P
    d, names = a.data, names_of(a.data)
    N = len(names) if not a.limit else min(a.limit, len(names))
    out = sdir(d) / ("side_t0.npy" if not a.limit else f"side_t0-first{a.limit}.npy")
    with Run("vis_train", f"side-build-{d}", config=vars(a)) as run:
        with open(sdir(d) / "ents.pkl", "rb") as f:
            ents = pickle.load(f)
        assert [e["token"] for e in ents] == names
        key = cache.key(params=kbase(d, names) | dict(limit=a.limit, t_prev=t_prev()), code=[_job, _render, SD.virtual])
        W = a.workers or PX.cores()
        st = {}

        def make():
            proot = sdir(d) / (".part" if not a.limit else f".part{a.limit}")
            proot.mkdir(exist_ok=True)
            part = st["part"] = P.Part(proot, key, N, a.force)
            todo = np.flatnonzero(~part.done)
            free = disk_after(len(todo) * SD.RB)
            run.info(f"{d}: {len(todo)} of {N} rows to render, {W} workers; {free:.0f} GiB stay free after this file and the front cache")
            assert free >= a.floor_gb, f"disk: {free:.0f} GiB would stay free, floor {a.floor_gb} GiB"
            part.open("side", (N, len(SD.CAMS), 2) + SD.FRAME, np.uint8).flush()
            todo = np.flatnonzero(~part.done)
            _G.update(ents=ents, virt=virt(d), path=str(proot / "side.part.npy"))
            chunks = [todo[i:i + 8] for i in range(0, len(todo), 8)]
            t0 = time.time()
            with ProcessPoolExecutor(W) as ex:
                futs = [ex.submit(_job, c) for c in chunks]
                for f in run.tqdm(as_completed(futs), total=len(futs), desc=f"render {d}"):
                    part.mark(f.result())
            part.save()
            run.summary |= {"rows": int(len(todo)), "workers": W, "tokens_per_s": len(todo) / max(time.time() - t0, 1e-9)}
            return None
        cache.cached(out, key, make, force=a.force, writer=lambda p, o: st["part"].commit("side", p), reader=lambda p: None)
        run.summary |= {"n": N, "file": str(out), "gb": out.stat().st_size / 2 ** 30}


# ---------------------------------------------------------------- fov: the horizontal field of the three views
def view_fov(cam, yaw_deg: float) -> dict:
    """Per model frame (road, wide) of a camera rendered along yaw_deg: the bearings (deg in the ego frame, left +) of the pixel
    centres of the horizon row that the native camera sees, from OpenpilotMaps' own rays; and the share of the frame it sees."""
    from jevdrive import navsim_zs as Z
    from jevdrive.openpilot.frames import MEDMODEL_K, SBIGMODEL_K, VIEW_FROM_DEVICE, MODEL_W, MODEL_H
    uu, vv = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    out = {}
    for nm, Km in (("road", MEDMODEL_K), ("wide", SBIGMODEL_K)):
        rays = (np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(Km @ VIEW_FROM_DEVICE).T) * np.array([1., -1., -1.])
        rays = rays @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        _, ok, _ = Z.project_nuplan(rays, cam, 1)
        row = int(round(Km[1, 2]))                                                  # the horizon row (level camera)
        b = np.degrees(np.arctan2(rays[row, :, 1], rays[row, :, 0]))
        seen = b[ok[row]]
        out[nm] = dict(lo=float(seen.min()), hi=float(seen.max()), full_lo=float(b.min()), full_hi=float(b.max()),
                       cover=float(ok.mean()), cover_row=float(ok[row].mean()))
    return out


def cmd_fov(a):
    from jevdrive import navsim_zs as Z
    with Run("vis_train", "side-fov", config=vars(a)) as run:
        cal = {}
        for d in a.datas:
            with open(sdir(d) / "ents.pkl", "rb") as f:
                for e in pickle.load(f):
                    cams = {"CAM_F0": e["f0"]} | {c: e["cams"][c][1] for c in SD.CAMS}
                    r = cal.setdefault(Z.calib_key(cams), {"cams": cams, "n": 0})
                    r["n"] += 1
        rows = []
        for r in cal.values():
            yaw = {c: 0.0 if c == "CAM_F0" else Z.cam_yaw_deg(r["cams"][c]) for c in VIEWS}       # the front frame is rendered along the ego x axis
            v = {c: view_fov(r["cams"][c], yaw[c]) for c in VIEWS}
            L, F, Rr = (v[c]["wide"] for c in VIEWS)
            rows.append(dict(n=r["n"], yaw_L=yaw["CAM_L0"], yaw_R=yaw["CAM_R0"], mount_yaw_F=Z.cam_yaw_deg(r["cams"]["CAM_F0"]),
                             hfov_wide=F["full_hi"] - F["full_lo"], hfov_road=v["CAM_F0"]["road"]["full_hi"] - v["CAM_F0"]["road"]["full_lo"],
                             L_lo=L["lo"], L_hi=L["hi"], F_lo=F["lo"], F_hi=F["hi"], R_lo=Rr["lo"], R_hi=Rr["hi"],
                             overlap_LF=F["hi"] - L["lo"], overlap_FR=Rr["hi"] - F["lo"], reach_left=L["hi"], reach_right=-Rr["lo"],
                             cover_wide_min=min(x["cover"] for x in (L, F, Rr)), cover_row_min=min(x["cover_row"] for x in (L, F, Rr)),
                             cover_road_min=min(v[c]["road"]["cover"] for c in VIEWS)))
        w = np.array([r["n"] for r in rows], float)
        agg = {}
        for k in rows[0]:
            if k != "n":
                x = np.array([r[k] for r in rows])
                o = np.argsort(x)
                agg[k] = dict(min=float(x.min()), median=float(x[o][np.searchsorted(np.cumsum(w[o]), w.sum() / 2)]), max=float(x.max()))
        res = dict(datas=a.datas, tokens=int(w.sum()), calibrations=len(rows), agg=agg,
                   reach_min=float(min(agg["reach_left"]["min"], agg["reach_right"]["min"])),
                   gap=bool(agg["overlap_LF"]["min"] < 0 or agg["overlap_FR"]["min"] < 0))
        for k, x in agg.items():
            run.info(f"{k:16s} min {x['min']:+8.2f}  median {x['median']:+8.2f}  max {x['max']:+8.2f}")
        run.info(f"{res['tokens']} tokens, {res['calibrations']} calibrations: reach >= +-{res['reach_min']:.1f} deg, gap between views: {res['gap']}")
        run.summary |= {k: res[k] for k in ("tokens", "calibrations", "reach_min", "gap")}
        (SD.side_root() / f"fov{'-' + a.tag if a.tag else ''}.json").write_text(json.dumps(res | dict(rows=rows), indent=1))


# ---------------------------------------------------------------- check: bytes, P3's side tokens, the protocol-W pair
def cmd_check(a):
    import torch
    import pp_prep as P
    from jevdrive import op_adapt as A
    from jevdrive import op_interp as I
    dev = torch.device("cuda")
    res = {}
    with Run("vis_train", "side-check", config=vars(a)) as run:
        _, enc = P.encoder(dev)
        for d in a.datas:
            SS = SD.SideStore([d], dev)
            assert SS.names.tolist() == names_of(d)
            with open(sdir(d) / "ents.pkl", "rb") as f:
                ents = pickle.load(f)
            rows = np.sort(np.random.default_rng(0).choice(len(SS), min(a.rows, len(SS)), replace=False))
            prev, cur = (x.cpu().numpy() for x in SS.t0(rows))                      # (B, 2, 2, 6, 128, 256)
            with ThreadPoolExecutor(8) as ex:                                       # the t0 - 0.5 s and t0 keys, rendered now
                k2, k3 = (np.stack(list(ex.map(lambda i: np.stack([_render(ents[i]["cams"][c][f], c) for c in SD.CAMS]), rows))) for f in (0, 1))
            cam, pd, ps = (x[rows[:16]] for x in virt(d))
            wp = np.stack([[I.warp_frame(k3[j, c], cam[j, c], pd[j, c], ps[j, c]) for c in range(2)] for j in range(len(cam))])
            r = {"rows": int(len(rows)), "cur_bytes_equal": bool((k3 == cur).all()), "prev_bytes_equal_first16": bool((wp == prev[:16]).all()),
                 "pair_mean_abs_px": float(np.abs(prev.astype(np.int16) - cur).mean()), "p3_pair_mean_abs_px": float(np.abs(k2.astype(np.int16) - cur).mean())}
            e = lambda p, c: enc(p.reshape(-1, *SD.FRAME), c.reshape(-1, *SD.FRAME)).reshape(len(rows), 2, *A.H_SHAPE).astype(np.float32)  # noqa: E731
            hp, hw = e(k2, cur), e(prev, cur)
            ref = CR / d / "side.npy"
            if ref.exists():
                x = np.load(ref, mmap_mode="r")[rows][:, :2, -1].astype(np.float32)   # CAM_L0, CAM_R0 at the pair (key 2, key 3)
                st = lambda y: dict(mean_abs=float(np.abs(y - x).mean()), max_abs=float(np.abs(y - x).max()))  # noqa: E731
                r |= {"p3_tokens_rms": float(np.sqrt((x ** 2).mean())), "p3_pairing": st(hp), "w_pairing": st(hw), "other_row": st(np.roll(x, 1, 0))}
            r["w_vs_p3_pairing_mean_abs"] = float(np.abs(hw - hp).mean())
            res[d] = r
            run.info(f"{d}: {json.dumps(r)}")
            SS.close()
        run.summary |= res
        (SD.side_root() / f"check{'-' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1))
        assert all(r["cur_bytes_equal"] and r["prev_bytes_equal_first16"] for r in res.values()), "cached bytes differ from a fresh render"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("split", "fov", "check"):
        p = sp.add_parser(c)
        p.add_argument("--datas", nargs="+", required=True)
        p.add_argument("--rows", type=int, default=256)
        p.add_argument("--tag", default="")
        cli_args(p)
    p = sp.add_parser("build")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--floor-gb", type=float, default=50.0, help="stop when the data disk would keep less than this free")
    cli_args(p)
    a = ap.parse_args()
    {"split": cmd_split, "build": cmd_build, "fov": cmd_fov, "check": cmd_check}[a.cmd](a)
