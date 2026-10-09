"""op_parity feature cache: everything the trainer and the NAVSIM readout read, computed once with Cinque's FROZEN vision encoder.

Per op_lb data dir (lb_navtrain, lb_h1train: navtrain tokens with GIMM context frames; lb_navtest: all 12 146 navtest tokens) ->
$DATA_DIR/runs/op_parity/cache/<data>/:

  front.npy  (N, 8, 32, 512) fp16  hidden tokens `view_39` of the 8 valid policy context frames, exactly the op_lb / port navtest protocol
                                   (experiments/op_adapt_r2/lib/op_adapt_r2_readout.nav_plans: steps 2, 6, .., 30 of the 31-step 20 Hz
                                   rollout, image pair (step s - 4, step s), keys + GIMM frames; slot 0 = zero hidden, not stored)
  side.npy   (N, 3, 3, 32, 512) fp16  CAM_L0 / CAM_R0 / CAM_B0, each rendered as an openpilot road + wide pair along its own mounting yaw
                                   (jevdrive.navsim_zs.OpenpilotMaps(yaw_deg)), pair (key k - 1, key k) at 2 Hz, k = 1..3
  tab.npz    names, log, ego (N, 20) lib/parity_adapter.ego_features, raw pose / vel / acc / cmd, fut (N, 8, 3) logged future
             (x, y, yaw at 0.5 .. 4 s, rear axle), cam (N, 3) camera position, lht, speed
  <data>@<frames>/front.npy  the same tokens under another frame protocol (prereg addendum 1, Stage B): warp (8 slots, CPU ego-motion
             warp lattice frames), real (8 slots, real 10 Hz lattice frames; lb_hq_navtestX), keys (N: (N, 4, 32, 512), 4 slots at the 2 Hz keys,
             pairs (key k - 1, key k), the first with a zero image)
  teacher.npz  shipped Cinque (port, fp16) on the same rows: out (N, Dd) distilled output positions, plan (N, 33, 15) MDN mean

Rendering runs on a CPU process pool (all cores of the job), the encoder on one GPU, overlapped (rendered chunks stream into the GPU
loop). Every file is a jevdrive.cache artifact (key: data dir, token list, code), so a rerun skips what is done.

Protocol `warp`, --synth gpu (default): the workers only read the keys and compute the warp poses, the 6 lattice frames of a chunk are
warped on the card (warp_keys: jevdrive.op_interp.warp_gpu, the bytes of the CPU warp) and go to the encoder in the batches of the CPU
path, so front.npy is the same file (experiments/alpasim/scripts/prep_check.py); --synth cpu keeps the warps in the workers. The warp
front is written row-wise (Part): a killed build continues from its last checkpoint.

  python experiments/op_parity/scripts/pp_prep.py --data lb_navtest [--limit 64] [--workers N]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib")]
import argparse, json, os, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import parity_adapter as PA  # noqa: E402
from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir, n_cpus  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

FRAME = (2, 6, 128, 256)
STEPS = np.arange(2, 31, 4)                     # 2, 6, .., 30: the 8 valid context steps (oldest first), as nav_plans
VERSION = "pp1"
SYNTH = ("gpu", "cpu")                          # where the `warp` lattice frames are made: batched on the card, or in the CPU workers
WARP_BS = 16                                    # frames per op_interp.warp_gpu call (float64 maps, ~30 MB of VRAM per frame; 16 is also the fastest)
OUT = None                                      # --out: another cache root (checks and benchmarks next to the real cache)


def croot(data, *p):
    d = (OUT or data_dir() / "runs" / "op_parity" / "cache") / data / _pl.Path(*p)
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- side cameras (CPU workers)
_MAPS = {}


def _bounded(ex, fn, items, depth):
    """ex.map(fn, items) in order with at most `depth` items in flight (Executor.map submits everything at once: host RAM)."""
    from collections import deque
    it, q = iter(items), deque()
    for x in it:
        q.append(ex.submit(fn, x))
        if len(q) >= depth:
            break
    while q:
        yield q.popleft().result()
        for x in it:
            q.append(ex.submit(fn, x))
            break


def full_meta(data: str):
    """'navtrain_full.s<i>of<n>' -> (op_lb-style meta of shard i of the navtrain tokens with a logged future, the full index list, entries)."""
    import op_lb as OL
    from jevdrive import navsim_zs as Z
    from jevdrive import par
    i, n = map(int, data.split(".s", 1)[1].split("of"))
    idx = Z.load_index("navtrain")
    fut = set(np.load(Z.root("index") / "navtrain_future.npz")["tokens"].tolist())
    by = {e["token"]: e for e in idx}
    names = par.shards(sorted(t for t in by if t in fut), n, i)
    ents = [by[t] for t in names]
    mt = {"split": "navtrain", "names": names, "cam": [np.asarray(e["cams"][-1]["CAM_F0"]["t"], float).tolist() for e in ents],
          "pose": [np.asarray(e["pose"], float) for e in ents], "vel": [np.asarray(e["vel"], float) for e in ents],
          "speed": [float(np.linalg.norm(e["vel"][-1])) for e in ents], "lht": [e["map"] in OL.LHT_MAPS for e in ents],
          "syn_t": OL.SYN_T.tolist()}
    return mt, idx, ents


def _full_job(args):
    """One full-navtrain token: the 4 CAM_F0 keys (navsim_zs_openpilot.render_token, the op_lb key rendering) + the 6 warped lattice frames."""
    import navsim_zs_openpilot as NZ
    e, pose, vel, cam, times = args
    kf = NZ.render_token(e)
    return kf, _warp_job((kf, pose, vel, cam, times))


def _warp_job(args):
    """CPU ego-motion warp of the 6 lattice frames (jevdrive.op_interp.synth_cpu), as op_lb's `warp` synthesis."""
    from jevdrive import op_interp as I
    kf, pose, vel, cam, times = args
    return I.synth_cpu(kf, "warp", times, I.track_navsim(pose, vel), cam)


# ---------------------------------------------------------------- `warp` lattice frames on the card
def warp_plan(pose, vel, times):
    """_warp_job without the pixels (op_interp.synth_cpu `warp` at lattice times between the keys): per time its source key (n,) and
    the destination / source pose (n, 2, 3)."""
    from jevdrive import op_interp as I
    times = np.asarray(times)
    assert times.min() > I.T_KEY[0] and not np.isclose(times[:, None], I.T_KEY).any(), "lattice times between the keys only"
    tr = I.track_navsim(pose, vel)
    src = [i0 if s <= 0.5 else i1 for i0, i1, s in map(I.neighbours, times)]
    return np.array(src), np.array([[tr(t), tr(I.T_KEY[k])] for t, k in zip(times, src)])


def _key_job(args):
    """_full_job without the pixels of the warps: the 4 keys and warp_plan's source keys and poses."""
    import navsim_zs_openpilot as NZ
    e, pose, vel, times = args
    return (NZ.render_token(e),) + warp_plan(pose, vel, times)


def warp_keys(K, row, key, cam, pose, bs: int = 0):
    """n warps of a chunk on the card. K (b, 4, 2, 6, 128, 256) uint8 keyframes on the device; per warp its row and source key (n,) and
    the destination / source pose (n, 2, 3); cam (b, 3) camera position per row -> (n, 2, 6, 128, 256) uint8, op_interp.warp_frame's
    bytes. One op_interp.warp_gpu call per camera position and WARP_BS frames."""
    import torch
    from jevdrive import op_interp as I
    row, key, cam, bs = np.asarray(row), torch.as_tensor(np.asarray(key), device=K.device), np.asarray(cam, np.float64), bs or WARP_BS
    out = torch.empty((len(row),) + FRAME, dtype=torch.uint8, device=K.device)
    cams, inv = np.unique(cam, axis=0, return_inverse=True)
    inv = inv.reshape(-1)[row]
    for c in range(len(cams)):
        idx = np.flatnonzero(inv == c)
        for i in range(0, len(idx), bs):
            q = torch.as_tensor(idx[i:i + bs], device=K.device)
            out[q] = I.warp_gpu(K[torch.as_tensor(row[idx[i:i + bs]], device=K.device), key[q]], cams[c], pose[idx[i:i + bs], 0], pose[idx[i:i + bs], 1])
    return out


def enc_dev(net, prev, cur, bs: int = 128) -> np.ndarray:
    """encoder()'s enc for uint8 pairs (n, 2, 6, 128, 256) that are already on the card: the same batches, so the same tokens."""
    import torch
    from jevdrive import op_adapt as A
    with torch.no_grad():
        out = [net.run_batched(A.vision_feeds(prev[i:i + bs], cur[i:i + bs]), ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE).cpu().numpy()
               for i in range(0, len(cur), bs)]
    return np.concatenate(out).astype(np.float16)


class Part:
    """Row-wise resumable .npy outputs of one cache artifact: <name>.part.npy memmaps and a done-rows mask (done.part.npy), bound to the
    artifact's cache key (part.key). A killed build reopens them and skips the rows of its last checkpoint; another key, a missing part
    or force starts over. commit() renames a part to its final path (as a jevdrive.cache writer)."""

    def __init__(self, root, key: str, n: int, force: bool = False):
        self.root, self.arr, self.todo, self.t = _pl.Path(root), {}, [], time.time()
        kf, self.mask = self.root / "part.key", self.root / "done.part.npy"
        self.resume = not force and self.mask.exists() and kf.exists() and kf.read_text() == key
        self.done = np.load(self.mask) if self.resume else np.zeros(n, bool)
        if len(self.done) != n:
            self.resume, self.done = False, np.zeros(n, bool)
        if not self.resume:
            self.mask.unlink(missing_ok=True)
            kf.write_text(key)

    def open(self, name: str, shape, dtype=np.float16):
        p = self.root / f"{name}.part.npy"
        if self.resume and p.exists():
            a = np.lib.format.open_memmap(p, mode="r+")
            if a.shape == tuple(shape) and a.dtype == dtype:
                self.arr[name] = a
                return a
            del a
        self.done[:] = False
        self.arr[name] = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=tuple(shape))
        return self.arr[name]

    def mark(self, rows, every: float = 60.0):
        """rows are written; every `every` seconds the parts are flushed and the mask saved (the checkpoint a resume starts from)."""
        self.todo.append(rows)
        if time.time() - self.t >= every:
            self.save()

    def save(self):
        for a in self.arr.values():
            a.flush()
        for r in self.todo:
            self.done[r] = True
        tmp = self.mask.with_name(".done.part.tmp.npy")
        np.save(tmp, self.done)
        os.replace(tmp, self.mask)
        self.todo, self.t = [], time.time()

    def commit(self, name: str, path):
        self.arr[name].flush()
        os.replace(self.root / f"{name}.part.npy", path)
        if all(not (self.root / f"{k}.part.npy").exists() for k in self.arr):
            self.mask.unlink(missing_ok=True), (self.root / "part.key").unlink(missing_ok=True)


def render_side(cams4: list) -> np.ndarray:
    """cams4: 4 history frames (oldest first) of {cam name: nuPlan cam dict} -> (3, 4, 2, 6, 128, 256) uint8 openpilot frames."""
    from jevdrive import navsim_zs as Z
    out = np.zeros((len(PA.SIDE_CAMS), 4) + FRAME, np.uint8)
    for c, name in enumerate(PA.SIDE_CAMS):
        for f in range(4):
            cam = cams4[f][name]
            k = (name, Z.calib_key({name: cam}))
            m = _MAPS.get(k) or _MAPS.setdefault(k, Z.OpenpilotMaps(cam, yaw_deg=Z.cam_yaw_deg(cam)))
            out[c, f] = m(m.decode(cam["path"]))
    return out


# ---------------------------------------------------------------- GPU encoder
def encoder(dev):
    import torch
    from jevdrive import op_adapt as A
    net = A.load("cinque", torch.float16).to(dev).eval()

    @torch.no_grad()
    def enc(prev: np.ndarray, cur: np.ndarray, bs=128) -> np.ndarray:
        """(n, 2, 6, 128, 256) uint8 pairs -> (n, 32, 512) fp16 hidden tokens."""
        out = []
        for i in range(0, len(cur), bs):
            p, c = (torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(dev) for x in (prev, cur))
            out.append(net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), *A.H_SHAPE).cpu().numpy())
        return np.concatenate(out).astype(np.float16)
    return net, enc


def main(a):
    import torch
    import op_lb as OL
    from jevdrive import navsim_zs as Z
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    full_idx, ents = None, None
    if a.data.startswith("navtrain_full"):                         # full navtrain shard "navtrain_full.s<i>of<n>": no op_lb dir, keys rendered here
        assert a.frames == "warp", "full navtrain: protocol W only (Stage B decision)"
        mt, full_idx, ents = full_meta(a.data)
    else:
        mt = OL.meta(a.data)
    names = mt["names"][: a.limit] if a.limit else mt["names"]
    N = len(names)
    tag = a.data + (f"-first{a.limit}" if a.limit else "")
    with Run("op_parity", f"prep-{tag}", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{mt['split']}"))
        dev = torch.device("cuda")
        net, enc = encoder(dev)
        kbase = dict(data=a.data, n=N, names_sha=cache.key(params=dict(n=names)), version=VERSION)
        root = croot(tag)
        timing = {}

        # ---- tab
        def make_tab():
            idx = {e["token"]: e for e in (full_idx or Z.load_index(mt["split"], slim=True))}
            ents = [idx[t] for t in names]
            fz = np.load(Z.root("index") / f"{mt['split']}_future.npz")
            fpos = dict(zip(fz["tokens"].tolist(), range(len(fz["tokens"]))))
            fut = np.full((N, 8, 3), np.nan, np.float32)
            for i, t in enumerate(names):
                if t in fpos:
                    fut[i] = fz["poses"][fpos[t]]
            pose, vel, acc, cmd = (np.stack([np.asarray(e[k], np.float32) for e in ents]) for k in ("pose", "vel", "acc", "cmd"))
            ego = PA.ego_features(pose, vel, acc, cmd[:, -1])
            return dict(names=np.array(names), log=np.array([e["log_name"] for e in ents]), ego=ego, pose=pose, vel=vel, acc=acc, cmd=cmd,
                        fut=fut, cam=np.asarray(mt["cam"], np.float32)[:N], lht=np.asarray(mt["lht"], bool)[:N],
                        speed=np.asarray(mt["speed"], np.float32)[:N])
        tab = cache.cached(root / "tab.npz", cache.key(params=kbase, code=[PA.ego_features]), make_tab, force=a.force)
        run.info(f"tab: {N} rows, future missing {int(np.isnan(tab['fut'][:, 0, 0]).sum())}")

        # ---- front hidden tokens, one frame protocol (prereg addendum 1)
        fparams = kbase | dict(steps=STEPS.tolist()) if a.frames == "gimm" else kbase | dict(frames=a.frames, steps=STEPS.tolist(), v="pp2")
        froot = root if a.frames == "gimm" else croot(f"{tag}@{a.frames}")
        froot.mkdir(parents=True, exist_ok=True)
        fkey, st = cache.key(params=fparams, code=[encoder]), {}

        def make_front():
            keys = OL.Keys(a.data) if ents is None else None
            _, src = OL._steps(0.0, False)
            t0 = time.time()
            pool = None
            gpu = a.frames == "warp" and a.synth == "gpu"
            if a.frames == "keys":                                  # N: 4 slots at the 2 Hz keys, pairs (key k - 1, key k); key -1 is a zero image
                def load(rows):
                    kf = keys[rows]                                 # (b, 4, 2, 6, 128, 256)
                    prev = np.concatenate([np.zeros_like(kf[:, :1]), kf[:, :-1]], 1)
                    return rows, prev.reshape(-1, *FRAME), kf.reshape(-1, *FRAME)
                n_slot = 4
            else:                                                   # Q8: steps 2, 6, .., 30 of the op_lb rollout, keys + 6 lattice frames
                syn = None if a.frames == "warp" else np.load(OL.root(a.data) / f"{a.frames}.npy", mmap_mode="r")
                pool = ProcessPoolExecutor(a.workers or max(1, n_cpus() - 4)) if a.frames == "warp" and (ents is not None or not gpu) else None
                if gpu:
                    times, mcam = np.asarray(mt["syn_t"]), np.asarray(mt["cam"], np.float64)
                    # cur then prev frame of the 8 steps, as indices into [4 keys, 6 lattice frames, a zero image]
                    at = [int(src[s][1] if src[s][0] == "k" else 4 + src[s][1]) if s >= 0 else 10 for s in np.r_[STEPS, STEPS - 4]]

                def load_keys(rows):                                # --synth gpu: the keys and the warp poses; the pixels are warped on the card
                    if ents is not None:
                        kf, sk, pp = (np.stack(x) for x in zip(*pool.map(_key_job, [(ents[i], mt["pose"][i], mt["vel"][i], times) for i in rows])))
                    else:
                        kf = np.array(keys[rows])                   # read here, off the encoder thread
                        sk, pp = (np.stack(x) for x in zip(*[warp_plan(mt["pose"][i], mt["vel"][i], times) for i in rows]))
                    return rows, kf, sk, pp

                def pairs_dev(rows, kf, sk, pp):
                    K, b = torch.from_numpy(kf).to(dev), len(rows)
                    sf = warp_keys(K, np.repeat(np.arange(b), sk.shape[1]), sk.reshape(-1), mcam[rows], pp.reshape(-1, 2, 3))
                    fr = torch.cat([K, sf.view(b, -1, *FRAME), K.new_zeros((b, 1) + FRAME)], 1)[:, at]
                    return fr[:, 8:].flatten(0, 1), fr[:, :8].flatten(0, 1)

                def load(rows):
                    if ents is not None:                            # full navtrain: render the 4 CAM_F0 keys and warp in one CPU job per token
                        kf, sf = (np.stack(x) for x in zip(*pool.map(_full_job, [(ents[i], mt["pose"][i], mt["vel"][i], mt["cam"][i],
                                                                                     np.asarray(mt["syn_t"])) for i in rows])))
                    elif syn is None:
                        kf = keys[rows]
                        sf = np.stack(list(pool.map(_warp_job, [(kf[j], mt["pose"][i], mt["vel"][i], mt["cam"][i], np.asarray(mt["syn_t"]))
                                                                for j, i in enumerate(rows)])))
                    else:
                        kf = keys[rows]
                        sf = np.asarray(syn[rows[0]:rows[-1] + 1])
                    img = lambda j, s: (kf[j][src[s][1]] if src[s][0] == "k" else sf[j][src[s][1]]) if s >= 0 else np.zeros(FRAME, np.uint8)  # noqa: E731
                    cur = np.stack([[img(j, s) for s in STEPS] for j in range(len(rows))])
                    prev = np.stack([[img(j, s - 4) for s in STEPS] for j in range(len(rows))])
                    return rows, prev.reshape(-1, *FRAME), cur.reshape(-1, *FRAME)
                n_slot = 8
            part = st["part"] = Part(froot, fkey, N, a.force) if a.frames == "warp" else None
            out = part.open("front", (N, n_slot) + A.H_SHAPE) if part else np.zeros((N, n_slot) + A.H_SHAPE, np.float16)
            chunks = [c for c in (np.arange(i, min(i + 32, N)) for i in range(0, N, 32)) if not (part and part.done[c].all())]
            with ThreadPoolExecutor(8) as ex:
                for rows, *x in run.tqdm(_bounded(ex, load_keys if gpu else load, chunks, 16), total=len(chunks), desc=f"front {a.frames}"):
                    out[rows] = (enc_dev(net, *pairs_dev(rows, *x)) if gpu else enc(*x)).reshape(len(rows), n_slot, *A.H_SHAPE)
                    if part:
                        part.mark(rows)
            if pool:
                pool.shutdown()
            timing.update({f"front_{a.frames}_pairs_per_s": n_slot * sum(map(len, chunks)) / (time.time() - t0)})
            if a.frames == "warp":
                import resource
                rss = sum(resource.getrusage(w).ru_maxrss for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)) / 2 ** 20
                timing.update(synth=a.synth, vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30, rss_main_plus_largest_worker_gb=rss)
            return out
        front = cache.cached(froot / "front.npy", fkey, make_front, force=a.force,
                             writer=(lambda p, o: st["part"].commit("front", p)) if a.frames == "warp" else None)
        if a.frames != "gimm" and ents is None:                    # tab / side / teacher live in the base dir (the G protocol's)
            run.summary |= {"n": N, "front_shape": list(front.shape), **timing}
            if timing:
                (froot / "timing.json").write_text(json.dumps(timing, indent=1))
            return

        # ---- side cameras
        def make_side():
            full = full_idx or Z.load_index(mt["split"])
            pos = {e["token"]: k for k, e in enumerate(full)}
            items = [[{c: full[pos[t]]["cams"][f][c] for c in PA.SIDE_CAMS} for f in range(4)] for t in names]
            out = np.zeros((N, len(PA.SIDE_CAMS), PA.SIDE_T) + A.H_SHAPE, np.float16)
            t0, tb = time.time(), []
            W = a.workers or max(1, n_cpus() - 4)
            with ProcessPoolExecutor(W) as ex:
                buf, rows = [], []
                for i, fr in enumerate(run.tqdm(ex.map(render_side, items, chunksize=4), total=N, desc="side")):
                    buf.append(fr), rows.append(i)
                    if i < 4:                                                       # a few rendered tokens for the visual check
                        np.save(root / f"side_sample{i}.npy", fr)
                    if len(buf) == 32 or i == N - 1:
                        x = np.stack(buf)                                           # (b, 3, 4, ...)
                        prev, cur = x[:, :, :-1].reshape(-1, *FRAME), x[:, :, 1:].reshape(-1, *FRAME)
                        t1 = time.time()
                        out[rows] = enc(prev, cur).reshape(len(rows), len(PA.SIDE_CAMS), PA.SIDE_T, *A.H_SHAPE)
                        tb.append(time.time() - t1)
                        buf, rows = [], []
            timing.update({"side_tokens_per_s": N / (time.time() - t0), "side_gpu_busy": sum(tb) / (time.time() - t0), "render_workers": W})
            return out
        if a.no_side:
            return
        side = cache.cached(root / "side.npy", cache.key(params=kbase | dict(cams=PA.SIDE_CAMS, t=PA.SIDE_T), code=[render_side, encoder]),
                            make_side, force=a.force)

        # ---- teacher (shipped Cinque, port fp16) on the same rows
        def make_teacher():
            di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
            out, plan = np.zeros((N, len(di)), np.float32), np.zeros((N, 33, 15), np.float32)
            tc_all = np.where(tab["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)
            with torch.no_grad():
                for i in range(0, N, 256):
                    r = slice(i, min(i + 256, N))
                    H = torch.from_numpy(front[r]).to(dev)
                    H = torch.cat([torch.zeros_like(H[:, :1]), H], 1)
                    valid = torch.ones(H.shape[:2], dtype=torch.bool, device=dev)
                    valid[:, 0] = False
                    o = A._policy(net, H, (0.275, 0.525), torch.from_numpy(tc_all[r]).to(dev), valid)["outputs"].float()
                    out[r], plan[r] = o[:, di].cpu().numpy(), o[:, pi].cpu().numpy().reshape(-1, 33, 15)
            return dict(out=out, plan=plan, di=di, pi=pi)
        troot = froot if ents is not None else root                  # full navtrain (W): teacher = shipped on the W frames, next to its front
        cache.cached(troot / "teacher.npz", cache.key(params=kbase, inputs=[froot / "front.npy"]),
                     make_teacher, force=a.force)
        run.summary |= {"n": N, "front_shape": list(front.shape), "side_shape": list(side.shape), **timing}
        if timing:
            (root / "timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--frames", default="gimm", choices=["gimm", "warp", "keys", "real"],
                    help="front frame protocol: gimm / warp / real = 8 slots at 0.2 s with that lattice source (real: lb_hq_navtestX only), "
                         "keys = N, 4 slots at the 2 Hz keys; non-gimm fronts go to cache/<data>@<frames>/, tab and side stay in cache/<data>/")
    ap.add_argument("--no-side", action="store_true", help="skip the side / rear cameras (and the teacher): eval-only subsets")
    ap.add_argument("--synth", default="gpu", choices=SYNTH, help="--frames warp: the lattice warps batched on the card (same bytes), or in the CPU workers")
    ap.add_argument("--out", type=_pl.Path, default=None, help="cache root instead of $DATA_DIR/runs/op_parity/cache (checks)")
    cli_args(ap)
    a = ap.parse_args()
    OUT = a.out
    main(a)
