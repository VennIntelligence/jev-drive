"""vis_train pixel cache: the protocol-W model frames of every token, rendered once (lib/pixel_store.py reads them).

  split  --datas D ...     the per-token render inputs of each data dir, in its tab.npz row order (one job: navtrain's full index
                           unpickles to ~30 GB, so the shard jobs must not load it) -> px/<data>/{ents.pkl, meta.npz}
  build  --data D          CPU render (pp_prep's exact W path: 4 CAM_F0 keys + 6 ego-motion warps) -> px/<data>/frames.npy
                           (N, 8, 2, 6, 128, 256) uint8, written row-wise by the workers, resumable (pp_prep.Part)
  check  --datas D ...     equivalence: the frozen encoder on the cached pixels against cache/<data>@warp/front.npy; the future index

D = navtrain_full.s<i>of12 | lb_navtest | lb_navhard. px = $DATA_DIR/runs/vis_train/px. Workers = the cores of the pool's grant.

t0-only mode (junction_eyes; the protocol above is unchanged): `build --t0` writes only the t0 image pair of every row,
px/<data>/frames-t0.npy (N, 2, 2, 6, 128, 256) uint8 = frames 6:8 of the full row (786 432 bytes), byte-identical to what the full build
writes there. It decodes and warps only the keys those two frames need (for the t0 pair: the t0 key). `--root DIR` puts px (split, build,
check) somewhere else. `check --t0` encodes those pairs and compares with the t0 slot of front.npy; `--pixels n` also re-renders n rows
through the full path and requires the t0 bytes to be equal.
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
if _sys.argv[1:2] == ["build"]:                 # one thread per render worker (set before numpy / OpenCV load)
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_k] = "1"
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, pickle, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

import pixel_store as PX  # noqa: E402
from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

CR = data_dir() / "runs" / "op_parity" / "cache"
VERSION = "px1"
T0_FILE = "frames-t0.npy"
_ROOT = [None]                                  # --root override of PX.px_root()
MIN_FREE_GB = 50                                # shared data disk: a build that would leave less free space does not start


def pdir(data) -> _pl.Path:
    d = (_ROOT[0] or PX.px_root()) / data
    d.mkdir(parents=True, exist_ok=True)
    return d


def names_of(data) -> list:
    return np.load(CR / data / "tab.npz")["names"].tolist()


def kbase(data, names) -> dict:
    return dict(data=data, n=len(names), names_sha=cache.key(params=dict(n=names)), version=VERSION)


# ---------------------------------------------------------------- split: render inputs per data dir
def cmd_split(a):
    import op_lb as OL
    from jevdrive import navsim_zs as Z
    with Run("vis_train", "px-split", config=vars(a)) as run:
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
                    out.append({"token": t, "log": e["log_name"], "ts": int(e["timestamp"]),
                                "cams": [{"CAM_F0": c["CAM_F0"]} for c in e["cams"]],
                                "pose": np.asarray(e["pose"], float) if full else mt["pose"][i],
                                "vel": np.asarray(e["vel"], float) if full else mt["vel"][i],
                                "cam": np.asarray(e["cams"][-1]["CAM_F0"]["t"], float).tolist() if full else mt["cam"][i]})
                return out
            ents = cache.cached(pdir(d) / "ents.pkl", cache.key(params=kbase(d, names)), make, force=a.force)
            np.savez(pdir(d) / "meta.npz", names=np.array(names), log=np.array([e["log"] for e in ents]), ts=np.array([e["ts"] for e in ents], np.int64))
            run.info(f"{d}: {len(ents)} rows, {len({e['log'] for e in ents})} logs")


# ---------------------------------------------------------------- build: CPU render, the workers write the rows
_G = {}


def _job(rows):
    """Rows -> their 8 frames (steps 2, 6, .., 30: op_lb's step sources over the 4 keys and the 6 warps) written into the part file."""
    import navsim_zs_openpilot as NZ
    import pp_prep as P
    if "out" not in _G:
        _G["out"] = np.lib.format.open_memmap(_G["path"], mode="r+")
    for i in rows:
        e = _G["ents"][i]
        kf = NZ.render_token(e)
        sf = P._warp_job((kf, e["pose"], e["vel"], e["cam"], _G["syn_t"]))
        _G["out"][i] = np.stack([kf[j] if k == "k" else sf[j] for k, j in _G["src"]])
    return rows


def _job_t0(rows):
    """Rows -> their t0 image pair (the last two of the 8 steps' frames), rendered from the keys those frames need only."""
    import navsim_zs_openpilot as NZ
    import pp_prep as P
    if "out" not in _G:
        _G["out"] = np.lib.format.open_memmap(_G["path"], mode="r+")
    keys, syn_idx, src = _G["t0"]
    for i in rows:
        e = _G["ents"][i]
        cams = e["cams"][-1]
        ck = NZ.Z.calib_key({"CAM_F0": cams["CAM_F0"]})
        m = NZ._maps.get(ck) or NZ._maps.setdefault(ck, NZ.Z.OpenpilotMaps(cams["CAM_F0"]))
        kf = np.zeros((4, 2, 6, 128, 256), np.uint8)
        for f in keys:                                                              # same call as NZ.render_token, per needed key
            kf[f] = m(m.decode(e["cams"][f]["CAM_F0"]["path"]))
        sf = P._warp_job((kf, e["pose"], e["vel"], e["cam"], _G["syn_t"][syn_idx])) if len(syn_idx) else None
        _G["out"][i] = np.stack([kf[j] if k == "k" else sf[syn_idx.index(j)] for k, j in src])
    return rows


def t0_needs(src, syn_t):
    """For the (kind, index) sources of the t0 pair: the key frames to decode, the warp times (indices into syn_t) to synthesize."""
    from jevdrive import op_interp as I
    keys, syn = set(), []
    for k, j in src:
        if k == "k":
            keys.add(j)
        else:
            i0, i1, s = I.neighbours(float(syn_t[j]))
            keys.add(i0 if s <= 0.5 else i1)                                       # op_interp.synth_cpu's `warp` source key
            syn.append(j)
    return sorted(keys), sorted(set(syn))


def step_sources():
    """The (kind, index) frame source of each of the 8 steps, and the check that slot j's prev frame is step j - 1's frame
    (pp_unfreeze._slots: prev = img(s - 4), a zero image before the stream)."""
    import op_lb as OL
    import pp_prep as P
    _, src = OL._steps(0.0, False)
    assert all(P.STEPS[j] - 4 == P.STEPS[j - 1] for j in range(1, 8)) and P.STEPS[0] - 4 < 0 and len(P.STEPS) == PX.NF
    return [src[s] for s in P.STEPS], np.asarray(OL.SYN_T)


def cmd_build(a):
    import pp_prep as P
    d, names = a.data, names_of(a.data)
    N = len(names) if not a.limit else min(a.limit, len(names))
    t0only = a.t0
    nf = 2 if t0only else PX.NF
    out = pdir(d) / ((T0_FILE if t0only else "frames.npy") if not a.limit else f"frames{'-t0' if t0only else ''}-first{a.limit}.npy")
    with Run("vis_train", f"px-build-{d}", config=vars(a)) as run:
        with open(pdir(d) / "ents.pkl", "rb") as f:
            ents = pickle.load(f)
        assert [e["token"] for e in ents] == names
        src, syn_t = step_sources()
        t0need = t0_needs(src[-2:], syn_t) if t0only else None
        key = cache.key(params=kbase(d, names) | dict(limit=a.limit, src=[list(map(str, s)) for s in src]) | (dict(t0=True) if t0only else {}),
                        code=[_job_t0 if t0only else _job])
        W = a.workers or PX.cores()
        st = {}
        import shutil
        need, free = N * nf * PX.FB / 2 ** 30, shutil.disk_usage(pdir(d)).free / 2 ** 30
        run.info(f"{d}: {need:.1f} GB to write, {free:.0f} GB free on the data disk (floor {MIN_FREE_GB} GB)")
        if not cache.valid(out, key) and free - need < MIN_FREE_GB:               # sparse part files: the other builds are not counted
            raise RuntimeError(f"{d}: {free:.0f} GB free - {need:.1f} GB < {MIN_FREE_GB} GB floor; not written")

        def make():
            proot = pdir(d) / (".part" + ("-t0" if t0only else "") + (str(a.limit) if a.limit else ""))
            proot.mkdir(exist_ok=True)
            part = st["part"] = P.Part(proot, key, N, a.force)
            part.open("frames", (N, nf) + PX.FRAME, np.uint8).flush()
            _G.pop("out", None)
            _G.update(ents=ents, src=src, syn_t=syn_t, path=str(proot / "frames.part.npy"))
            if t0only:
                _G["t0"] = (t0need[0], t0need[1], src[-2:])
            todo = np.flatnonzero(~part.done)
            chunks = [todo[i:i + 8] for i in range(0, len(todo), 8)]
            run.info(f"{d}: {len(todo)} of {N} rows to render, {W} workers")
            t0 = time.time()
            with ProcessPoolExecutor(W) as ex:
                futs = [ex.submit(_job_t0 if t0only else _job, c) for c in chunks]
                for f in run.tqdm(as_completed(futs), total=len(futs), desc=f"render {d}"):
                    part.mark(f.result())
            part.save()
            run.summary |= {"rows": int(len(todo)), "workers": W, "tokens_per_s": len(todo) / max(time.time() - t0, 1e-9)}
            return None
        cache.cached(out, key, make, force=a.force, writer=lambda p, o: st["part"].commit("frames", p), reader=lambda p: None)
        run.summary |= {"n": N, "file": str(out), "gb": out.stat().st_size / 2 ** 30}


# ---------------------------------------------------------------- check: frozen encoder on the cached pixels = front.npy
def cmd_check_t0(a):
    """The t0-only cache: the frozen encoder on its pairs against the t0 slot (7) of front.npy; --pixels n: n rows re-rendered through
    the full 8-frame path (the original `build` job) and compared byte for byte with the stored pair."""
    import torch
    import pp_prep as P
    from jevdrive import op_adapt as A
    dev = torch.device("cuda")
    res = {}
    with Run("vis_train", "px-check-t0", config=vars(a)) as run:
        net, _ = P.encoder(dev)
        for d in a.datas:
            arr = np.load(pdir(d) / T0_FILE, mmap_mode="r")
            ref = np.load(CR / f"{d}@warp" / "front.npy", mmap_mode="r")
            assert arr.shape == (len(ref), 2) + PX.FRAME and arr.dtype == np.uint8, f"{d}: {arr.shape} vs front {ref.shape}"
            rows = np.sort(np.random.default_rng(0).choice(len(arr), min(a.rows, len(arr)), replace=False))
            dm, mx, sq, n = 0.0, 0.0, 0.0, 0
            for i in range(0, len(rows), 128):                                     # 128 pairs per pass, as pp_prep's batches
                r = rows[i:i + 128]
                p0 = torch.from_numpy(np.ascontiguousarray(arr[r, 0])).to(dev)
                c0 = torch.from_numpy(np.ascontiguousarray(arr[r, 1])).to(dev)
                h = P.enc_dev(net, p0, c0).reshape(len(r), *A.H_SHAPE).astype(np.float32)
                x = ref[r][:, 7].astype(np.float32)
                e = np.abs(h - x)
                dm, mx, sq, n = dm + e.sum(), max(mx, float(e.max())), sq + (x ** 2).sum(), n + e.size
            res[d] = {"rows": int(len(rows)), "mean_abs": float(dm / n), "max_abs": mx, "front_rms": float(np.sqrt(sq / n))}
            run.info(f"{d}: t0 pixels -> encoder vs front.npy[:, 7] on {len(rows)} rows: mean |d| {dm / n:.2e}, max {mx:.4f}, rms {np.sqrt(sq / n):.3f}")
            if a.pixels:                                                           # the full path's t0 bytes
                with open(pdir(d) / "ents.pkl", "rb") as f:
                    ents = pickle.load(f)
                src, syn_t = step_sources()
                _G.update(ents=ents, src=src, syn_t=syn_t)
                import navsim_zs_openpilot as NZ
                import pp_prep as P2
                bad, pr = 0, np.sort(np.random.default_rng(1).choice(len(arr), min(a.pixels, len(arr)), replace=False))
                for i in pr:
                    kf = NZ.render_token(ents[i])
                    sf = P2._warp_job((kf, ents[i]["pose"], ents[i]["vel"], ents[i]["cam"], syn_t))
                    full = np.stack([kf[j] if k == "k" else sf[j] for k, j in src])
                    bad += int(not np.array_equal(full[-2:], arr[i]))
                res[d] |= {"pixel_rows": int(len(pr)), "pixel_rows_differing": bad}
                run.info(f"{d}: full-path t0 bytes vs stored pair: {bad} of {len(pr)} rows differ")
        run.summary |= res
        (pdir_root() / f"check-t0{'-' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1))


def pdir_root() -> _pl.Path:
    return _ROOT[0] or PX.px_root()


def cmd_check(a):
    if a.t0:
        return cmd_check_t0(a)
    import torch
    import pp_prep as P
    from jevdrive import op_adapt as A
    dev = torch.device("cuda")
    res = {}
    with Run("vis_train", "px-check", config=vars(a)) as run:
        net, _ = P.encoder(dev)
        for d in a.datas:
            PS = PX.PixelStore([d], dev, mode="all", root=_ROOT[0])
            ref = np.load(CR / f"{d}@warp" / "front.npy", mmap_mode="r")
            rows = np.sort(np.random.default_rng(0).choice(len(PS), min(a.rows, len(PS)), replace=False))
            dm, mx, sq, n, t0d = 0.0, 0.0, 0.0, 0, 0.0
            for i in range(0, len(rows), 32):                                      # pp_prep's batches: 32 rows = 256 pairs, 128 per pass
                r = rows[i:i + 32]
                prev, cur = PS.all(r)
                h = P.enc_dev(net, prev.flatten(0, 1), cur.flatten(0, 1)).reshape(len(r), 8, *A.H_SHAPE).astype(np.float32)
                p0, c0 = PS.t0(r)
                assert torch.equal(p0, prev[:, 7]) and torch.equal(c0, cur[:, 7]) and not prev[:, 0].any()
                x = ref[r].astype(np.float32)
                e = np.abs(h - x)
                dm, mx, sq, n = dm + e.sum(), max(mx, float(e.max())), sq + (x ** 2).sum(), n + e.size
                t0d = max(t0d, float(e[:, 7].max()))
            res[d] = {"rows": int(len(rows)), "mean_abs": float(dm / n), "max_abs": mx, "max_abs_t0": t0d, "front_rms": float(np.sqrt(sq / n))}
            run.info(f"{d}: cached pixels -> encoder vs front.npy on {len(rows)} rows: mean |d| {dm / n:.2e}, max {mx:.4f}, rms {np.sqrt(sq / n):.3f}")
            PS.close()
        train = [d for d in a.datas if d.startswith("navtrain_full")]
        if len(train) > 1:                                                         # the future-token index of arm B over the given shards
            PS = PX.PixelStore(train, dev, root=_ROOT[0])
            res["future"] = {f"k{k}": float(PS.future_index(k)[1].mean()) for k in (1, 2, 3, 4)} | {"rows": len(PS), "datas": len(train)}
            run.info(f"future-token coverage over {len(train)} shards ({len(PS)} rows): {res['future']}")
        run.summary |= res
        (pdir_root() / f"check{'-' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("split")
    p.add_argument("--datas", nargs="+", required=True)
    cli_args(p)
    p = sp.add_parser("build")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--t0", action="store_true", help="only the t0 image pair: frames-t0.npy (N, 2, 2, 6, 128, 256)")
    cli_args(p)
    p = sp.add_parser("check")
    p.add_argument("--datas", nargs="+", required=True)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--tag", default="")
    p.add_argument("--t0", action="store_true", help="check frames-t0.npy (the t0-only cache)")
    p.add_argument("--pixels", type=int, default=0, help="with --t0: rows re-rendered through the full path and compared byte for byte")
    cli_args(p)
    for q in sp.choices.values():
        q.add_argument("--root", default="", help="px root (default $DATA_DIR/runs/vis_train/px)")
    a = ap.parse_args()
    _ROOT[0] = _pl.Path(a.root) if a.root else None
    {"split": cmd_split, "build": cmd_build, "check": cmd_check}[a.cmd](a)
