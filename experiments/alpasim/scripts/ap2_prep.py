"""AP2 token cache: Cinque's frozen vision tokens of the policy slots a decision WITHOUT full history sees, built with the serving
functions themselves (experiments/alpasim/lib/sh30_core.py: fill_history + lattice), for every token of an op_parity cache dir.

An AlpaSim rollout has no history before t = 0, so decisions 0 / 1 / 2 have m = 1 / 2 / 3 keyframes (experiments/alpasim/lib/ap2_inputs.py).
Per token and m, from the m newest keyframes and their states (missing poses: the oldest state run backwards at constant body velocity and
the track's yaw rate):

  cold.npy  (N, 10, 32, 512) fp16  rule `zero` (AP2 training / serving): only the real slots, the pair of the oldest starts from a zero
                                   image; packed [m = 1: 1 slot | m = 2: 3 | m = 3: 6], oldest first (COLD_AT)
  bw.npy    (N, 3, 8, 32, 512) fp16  rule `backwarp` (SH30 as served today; --bw): all 8 slots, the ones older than the oldest keyframe
                                   are that keyframe re-projected to the back-extrapolated poses

m = 4 is the pp_prep W cache (cache/<data>@warp/front.npy). Out: $DATA_DIR/runs/alpasim/ap2/cache/<data>/. The 4 CAM_F0 keyframes are
rendered from the JPEGs (navsim_zs_openpilot.render_token, the op_lb / pp_prep key rendering); their paths and calibration are read from the
NAVSIM logs here (the token's frame and the 3 before it, as the devkit's AgentInput), so the navsim_zs index is not needed. Ego poses,
velocities and the camera position come from the op_parity tab. CPU rendering and warps on a process pool, the encoder on one GPU.

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/ap2_prep.py --data lb_navtest --bw [--limit 64]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib")]
import argparse, json, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import ap2_inputs as AI  # noqa: E402
import pp_prep as PP  # noqa: E402
import sh30_core as C  # noqa: E402
from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir, n_cpus  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

COLD_AT, N_COLD = AI.COLD_AT, 10


def aroot(data, *p) -> _pl.Path:
    d = data_dir() / "runs/alpasim/ap2/cache" / data / _pl.Path(*p)
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


def _job(args):
    """One token -> cold frames (10, 2, 6, 128, 256) and, with bw, backwarp frames (3, 8, 2, 6, 128, 256): sh30_core.Core.plan's frame path."""
    import navsim_zs_openpilot as NZ
    from jevdrive import op_interp as I
    ent, pose, vel, w, cam, bw = args
    if C._POOL is None:
        import cv2
        cv2.setNumThreads(1)
        C._POOL = ThreadPoolExecutor(1)                 # serial warps: the process pool is the parallelism here
    kf = NZ.render_token(ent)
    cold = np.zeros((N_COLD,) + C.FRAME, np.uint8)
    back = np.zeros((3, 8) + C.FRAME, np.uint8) if bw else None
    cam = np.asarray(cam, np.float64)
    for m in (1, 2, 3):
        e = 4 - m
        P, V = C.fill_history(pose[e:], vel[e:], float(w[e]))
        K = np.zeros((4,) + C.FRAME, np.uint8)
        K[e:] = kf[e:]
        tr = I.track_navsim(P, V)
        cur, valid = C.lattice(K, e, tr, cam, "zero")
        cold[COLD_AT[m]] = cur[valid]
        if bw:
            back[m - 1] = C.lattice(K, e, tr, cam, "backwarp")[0]
    return cold, back


def _full(args):
    """One token with its full history -> the 8 slot frames of the W protocol (parity check against cache/<data>@warp/front.npy)."""
    import navsim_zs_openpilot as NZ
    from jevdrive import op_interp as I
    ent, pose, vel, cam = args
    C._POOL = C._POOL or ThreadPoolExecutor(1)
    return C.lattice(NZ.render_token(ent), 0, I.track_navsim(pose, vel), np.asarray(cam, np.float64), "zero")[0]


def _log_ents(u):
    """One NAVSIM log: (split dir, log, tokens) -> [(token, {"cams": 4 x {"CAM_F0": path + calibration}})], the index entry render_token reads."""
    import pickle
    from jevdrive import navsim_zs as Z
    split_dir, log, toks = u
    base = data_dir() / "datasets/navsim"
    with open(base / "navsim_logs" / split_dir / f"{log}.pkl", "rb") as f:
        L = pickle.load(f)
    pos = {x["token"]: i for i, x in enumerate(L)}
    return [(t, {"cams": [{"CAM_F0": Z.cams_of(L[j]["cams"], base / "sensor_blobs" / split_dir)["CAM_F0"]} for j in range(pos[t] - 3, pos[t] + 1)]}) for t in toks]


def entries(names, logs, run) -> list:
    from jevdrive import par
    split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
    by = {}
    for t, lg in zip(names, logs):
        by.setdefault(lg, []).append(t)
    res = par.pmap(_log_ents, [(split_dir, lg, ts) for lg, ts in sorted(by.items())], run=run, desc="log entries")
    res.raise_if_failed()
    ent = dict(x for v in res.values for x in v)
    return [ent[t] for t in names]


def pairs(cur: np.ndarray):
    """cur (b, n, ...) slot frames, oldest first -> the image pairs (prev, cur) of sh30_core: the oldest pair starts from a zero image."""
    return np.concatenate([np.zeros_like(cur[:, :1]), cur[:, :-1]], 1).reshape(-1, *C.FRAME), cur.reshape(-1, *C.FRAME)


def main(a):
    import torch
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    tab = np.load(data_dir() / "runs/op_parity/cache" / a.data / "tab.npz")
    names = tab["names"][: a.limit].tolist() if a.limit else tab["names"].tolist()
    N = len(names)
    tag = a.data + (f"-first{a.limit}" if a.limit else "")
    pose, vel, cam = tab["pose"][:N].astype(float), tab["vel"][:N].astype(float), tab["cam"][:N].astype(float)
    W = AI.yaw_rates(tab["pose"][:N], tab["fut"][:N])
    with Run("alpasim", f"ap2-prep-{tag}", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest" if a.data == "lb_navtest" else "navsim/navtrain"))
        _, enc0 = PP.encoder(torch.device("cuda"))
        enc = lambda prev, cur: enc0(prev, cur, bs=a.bs)  # noqa: E731
        kbase = dict(data=a.data, n=N, names_sha=cache.key(params=dict(n=names)), v="ap2-1")
        root = aroot(tag)
        root.mkdir(parents=True, exist_ok=True)
        out = {}

        def make():
            t0 = time.time()
            cold = np.zeros((N, N_COLD) + A.H_SHAPE, np.float16)
            back = np.zeros((N, 3, 8) + A.H_SHAPE, np.float16) if a.bw else None
            chunks = [np.arange(i, min(i + 32, N)) for i in range(0, N, 32)]
            ents = entries(names, tab["log"][:N].tolist(), run)
            d = np.abs(np.array([e["cams"][-1]["CAM_F0"]["t"] for e in ents[:64]]) - cam[:64]).max()
            assert d < 1e-4, f"camera position of the logs and of the op_parity tab differ by {d} m"
            with ProcessPoolExecutor(a.workers or max(1, n_cpus() - 6)) as pool, ThreadPoolExecutor(6) as ex:
                k = min(N, 32)                          # this rendering path against the stored W tokens of the same rows
                ref = np.load(data_dir() / "runs/op_parity/cache" / f"{a.data}@warp" / "front.npy", mmap_mode="r")[:k].astype(np.float32)
                got = enc(*pairs(np.stack(list(pool.map(_full, [(ents[i], pose[i], vel[i], cam[i]) for i in range(k)]))))).reshape(ref.shape).astype(np.float32)
                out["parity_rel"] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                run.info(f"W-token parity on {k} rows: relative error {out['parity_rel']:.5f}")
                assert out["parity_rel"] < 0.01, "the rendered keys do not reproduce the pp_prep W tokens"

                def load(rows):
                    res = list(pool.map(_job, [(ents[i], pose[i], vel[i], W[i], cam[i], a.bw) for i in rows]))
                    return rows, np.stack([r[0] for r in res]), (np.stack([r[1] for r in res]) if a.bw else None)
                for rows, cf, bf in run.tqdm(PP._bounded(ex, load, chunks, 12), total=len(chunks), desc="cold slots"):
                    b = len(rows)
                    for m, sl in COLD_AT.items():
                        cold[rows, sl] = enc(*pairs(cf[:, sl])).reshape(b, AI.N_SLOT[m], *A.H_SHAPE)
                    if a.bw:
                        back[rows] = enc(*pairs(bf.reshape(b * 3, 8, *C.FRAME))).reshape(b, 3, 8, *A.H_SHAPE)
            out["tokens_per_s"] = N / (time.time() - t0)
            if a.bw:
                np.save(root / "bw.tmp.npy", back)
                (root / "bw.tmp.npy").replace(root / "bw.npy")
            return cold
        cold = cache.cached(root / "cold.npy", cache.key(params=kbase | dict(bw=a.bw), code=[_job, pairs]), make, force=a.force)
        run.summary |= {"n": N, "cold_shape": list(cold.shape), "bw": a.bw, "rms": float(np.sqrt((cold[: min(N, 2000)].astype(np.float32) ** 2).mean())), **out}
        (root / "meta.json").write_text(json.dumps({"names_sha": kbase["names_sha"], "n": N, "cold_at": {m: [s.start, s.stop] for m, s in COLD_AT.items()}, **out}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--bw", action="store_true", help="also the backwarp tokens (SH30 as served; arm AB)")
    ap.add_argument("--bs", type=int, default=128, help="encoder batch (image pairs): 29 GB of VRAM at 128, about a quarter at 32")
    cli_args(ap)
    main(ap.parse_args())
