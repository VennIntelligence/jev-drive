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
velocities and the camera position come from the op_parity tab. The keys are rendered on a process pool, the encoder runs on one GPU.

--synth gpu (default): the workers return the 4 keys and the warp poses (_keys), the slot warps of a chunk of 32 tokens run on the card
(lattices: pp_prep.warp_keys, the bytes of the CPU warp), each lattice once (the real slots of rule `zero` are those of `backwarp`), and
the encoder sees the batches of the CPU path, so cold.npy / bw.npy are the same files (scripts/prep_check.py). --synth cpu is the
reference: all warps in the workers (_job). The 34 image pairs of a token hold 27 distinct ones, but the encoder's output depends on
the batch size, so encoding the 27 once changes tokens and is not done. Both arrays are written row-wise (pp_prep.Part): a killed
build continues from its last checkpoint. Under the GPU pool pass --workers (jevdrive.common.n_cpus reports the box, not the pinned
cores; it also sizes the log-entry pool, whose forked workers otherwise show as ~90 GB of RSS).

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/ap2_prep.py --data lb_navtest --bw [--limit 64]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib")]
import argparse, functools, json, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import ap2_inputs as AI  # noqa: E402
import pp_prep as PP  # noqa: E402
import sh30_core as C  # noqa: E402
from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir, n_cpus  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

COLD_AT, N_COLD = AI.COLD_AT, 10
OUT = None                                      # --out: another cache root (checks and benchmarks next to the real cache)


def aroot(data, *p) -> _pl.Path:
    d = (OUT or data_dir() / "runs/alpasim/ap2/cache") / data / _pl.Path(*p)
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


@functools.lru_cache
def slot_plan(ms: tuple, bw: bool):
    """The slot sources of sh30_core.lattice_gpu for decisions with m in ms keyframes (rule `backwarp` with bw, else `zero`) -> copies,
    warps: int rows (index of m in ms, slot, source key). A real slot at a key time is that key, any other real slot the nearest key
    warped; with bw the slots older than the oldest key are that key warped."""
    from jevdrive import op_interp as I
    cp, wp = [], []
    for i, m in enumerate(ms):
        e = 4 - m
        for j, t in enumerate(C.SLOT_T):
            real, k = t >= I.T_KEY[e] - 1e-9, np.flatnonzero(np.isclose(I.T_KEY, t))
            if real and len(k):
                cp.append((i, j, k[0]))
            elif real or bw:
                i0, i1, s = I.neighbours(t) if real else (e, e, 0.0)
                wp.append((i, j, i0 if s <= 0.5 else i1))
    return np.array(cp).reshape(-1, 3), np.array(wp).reshape(-1, 3)


def _keys(args):
    """One token -> its 4 keyframes and the destination / source poses (n, 2, 3) of slot_plan's warps: _job (ms = 1, 2, 3) or _full
    (ms = 4) without the pixels of the warps."""
    import navsim_zs_openpilot as NZ
    from jevdrive import op_interp as I
    ent, pose, vel, w, ms, bw = args
    tr = [I.track_navsim(*C.fill_history(pose[4 - m:], vel[4 - m:], float(w[4 - m]))) for m in ms]
    return NZ.render_token(ent), np.array([[tr[i](C.SLOT_T[j]), tr[i](I.T_KEY[k])] for i, j, k in slot_plan(ms, bw)[1]]).reshape(-1, 2, 3)


def lattices(kf, pp, cam, ms: tuple, bw: bool, dev):
    """Keyframes (b, 4, 2, 6, 128, 256), _keys' warp poses (b, n, 2, 3) and camera positions (b, 3) of a chunk -> the slot frames
    (b, len(ms), 8, 2, 6, 128, 256) uint8 on dev: sh30_core.lattice per token and m (rule `backwarp` with bw, else `zero`), all warps
    of the chunk batched on the card. Slots that are not valid stay zero."""
    import torch
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
    (ci, cj, ck), (wi, wj, wk), K, b = slot_plan(ms, bw)[0].T, slot_plan(ms, bw)[1].T, t(kf), len(kf)
    lat = torch.zeros((b, len(ms), 8) + C.FRAME, dtype=torch.uint8, device=dev)
    lat[:, t(ci), t(cj)] = K[:, t(ck)]
    if len(wk):
        row = np.repeat(np.arange(b), len(wk))
        lat[t(row), t(np.tile(wi, b)), t(np.tile(wj, b))] = PP.warp_keys(K, row, np.tile(wk, b), cam, pp.reshape(-1, 2, 3))
    return lat


def pairs_dev(cur):
    """pairs for a tensor on the card."""
    import torch
    return torch.cat([torch.zeros_like(cur[:, :1]), cur[:, :-1]], 1).flatten(0, 1), cur.flatten(0, 1)


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


def entries(names, logs, run, workers=None) -> list:
    from jevdrive import par
    split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
    by = {}
    for t, lg in zip(names, logs):
        by.setdefault(lg, []).append(t)
    res = par.pmap(_log_ents, [(split_dir, lg, ts) for lg, ts in sorted(by.items())], workers=workers, run=run, desc="log entries")
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
        dev, gpu = torch.device("cuda"), a.synth == "gpu"
        net, enc0 = PP.encoder(dev)
        enc = (lambda prev, cur: PP.enc_dev(net, prev, cur, a.bs)) if gpu else (lambda prev, cur: enc0(prev, cur, bs=a.bs))  # noqa: E731
        prs = pairs_dev if gpu else pairs
        kbase = dict(data=a.data, n=N, names_sha=cache.key(params=dict(n=names)), v="ap2-1")
        root = aroot(tag)
        root.mkdir(parents=True, exist_ok=True)
        key, out, st = cache.key(params=kbase | dict(bw=a.bw), code=[_job, pairs]), {}, {}

        def make():
            t0 = time.time()
            part = st["part"] = PP.Part(root, key, N, a.force)
            cold = part.open("cold", (N, N_COLD) + A.H_SHAPE)
            back = part.open("bw", (N, 3, 8) + A.H_SHAPE) if a.bw else None
            chunks = [c for c in (np.arange(i, min(i + 32, N)) for i in range(0, N, 32)) if not part.done[c].all()]
            ents = entries(names, tab["log"][:N].tolist(), run, a.workers or None)
            d = np.abs(np.array([e["cams"][-1]["CAM_F0"]["t"] for e in ents[:64]]) - cam[:64]).max()
            assert d < 1e-4, f"camera position of the logs and of the op_parity tab differ by {d} m"
            nw = a.workers or (min(8, max(1, n_cpus() - 2)) if gpu else max(1, n_cpus() - 6))    # gpu: the encoder keeps up with 2-3 workers
            with ProcessPoolExecutor(nw) as pool, ThreadPoolExecutor(6) as ex:
                def keys(rows, ms, bw):                 # --synth gpu: the keys and the warp poses of a chunk; the pixels are warped on the card
                    res = list(pool.map(_keys, [(ents[i], pose[i], vel[i], W[i], ms, bw) for i in rows]))
                    return rows, np.stack([r[0] for r in res]), np.stack([r[1] for r in res])

                def load(rows):
                    if gpu:
                        return keys(rows, (1, 2, 3), a.bw)
                    res = list(pool.map(_job, [(ents[i], pose[i], vel[i], W[i], cam[i], a.bw) for i in rows]))
                    return rows, np.stack([r[0] for r in res]), (np.stack([r[1] for r in res]) if a.bw else None)
                k = min(N, 32)                          # this rendering path against the stored W tokens of the same rows
                ref = np.load(data_dir() / "runs/op_parity/cache" / f"{a.data}@warp" / "front.npy", mmap_mode="r")[:k].astype(np.float32)
                if gpu:
                    _, kf, pp = keys(range(k), (4,), False)
                    full = lattices(kf, pp, cam[:k], (4,), False, dev)[:, 0]
                else:
                    full = np.stack(list(pool.map(_full, [(ents[i], pose[i], vel[i], cam[i]) for i in range(k)])))
                got = enc(*prs(full)).reshape(ref.shape).astype(np.float32)
                out["parity_rel"] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                run.info(f"W-token parity on {k} rows: relative error {out['parity_rel']:.5f}")
                assert out["parity_rel"] < 0.01, "the rendered keys do not reproduce the pp_prep W tokens"
                t1 = time.time()
                for rows, x, y in run.tqdm(PP._bounded(ex, load, chunks, 12), total=len(chunks), desc="cold slots"):
                    b = len(rows)
                    if gpu:                             # one lattice per m: the real slots of rule `zero` are those of `backwarp`
                        y = lattices(x, y, cam[rows], (1, 2, 3), a.bw, dev)
                    for m, sl in COLD_AT.items():
                        cf = y[:, m - 1, 8 - AI.N_SLOT[m]:] if gpu else x[:, sl]
                        cold[rows, sl] = enc(*prs(cf)).reshape(b, AI.N_SLOT[m], *A.H_SHAPE)
                    if a.bw:
                        back[rows] = enc(*prs(y.reshape(b * 3, 8, *C.FRAME))).reshape(b, 3, 8, *A.H_SHAPE)
                    part.mark(rows)
            import resource
            rss = sum(resource.getrusage(w).ru_maxrss for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)) / 2 ** 20
            out.update(tokens_per_s=N / (time.time() - t0), loop_tokens_per_s=sum(map(len, chunks)) / (time.time() - t1), synth=a.synth, workers=nw,
                       bs=a.bs, vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30, rss_main_plus_largest_worker_gb=rss)
            return cold

        def commit(path, cold):
            if a.bw:
                st["part"].commit("bw", root / "bw.npy")
            st["part"].commit("cold", path)
        cold = cache.cached(root / "cold.npy", key, make, force=a.force, writer=commit)
        run.summary |= {"n": N, "cold_shape": list(cold.shape), "bw": a.bw, "rms": float(np.sqrt((cold[: min(N, 2000)].astype(np.float32) ** 2).mean())), **out}
        (root / "meta.json").write_text(json.dumps({"names_sha": kbase["names_sha"], "n": N, "cold_at": {m: [s.start, s.stop] for m, s in COLD_AT.items()}, **out}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--bw", action="store_true", help="also the backwarp tokens (SH30 as served; arm AB)")
    ap.add_argument("--bs", type=int, default=128, help="encoder batch (image pairs); the tokens depend on it at the 7e-4 level (a cache is one batch size)")
    ap.add_argument("--synth", default="gpu", choices=PP.SYNTH, help="slot warps batched on the card (same bytes), or in the CPU workers")
    ap.add_argument("--out", type=_pl.Path, default=None, help="cache root instead of $DATA_DIR/runs/alpasim/ap2/cache (checks)")
    cli_args(ap)
    a = ap.parse_args()
    OUT = a.out
    main(a)
