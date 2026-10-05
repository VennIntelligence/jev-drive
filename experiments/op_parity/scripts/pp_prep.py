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
  teacher.npz  shipped Cinque (port, fp16) on the same rows: out (N, Dd) distilled output positions, plan (N, 33, 15) MDN mean

Rendering runs on a CPU process pool (all cores of the job), the encoder on one GPU, overlapped (rendered chunks stream into the GPU
loop). Every file is a jevdrive.cache artifact (key: data dir, token list, code), so a rerun skips what is done.

  python experiments/op_parity/scripts/pp_prep.py --data lb_navtest [--limit 64] [--workers N]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib")]
import argparse, json, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import parity_adapter as PA  # noqa: E402
from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir, n_cpus  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

FRAME = (2, 6, 128, 256)
STEPS = np.arange(2, 31, 4)                     # 2, 6, .., 30: the 8 valid context steps (oldest first), as nav_plans
VERSION = "pp1"


def croot(data, *p):
    d = data_dir() / "runs" / "op_parity" / "cache" / data / _pl.Path(*p)
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- side cameras (CPU workers)
_MAPS = {}


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
    def enc(prev: np.ndarray, cur: np.ndarray, bs=256) -> np.ndarray:
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
            idx = {e["token"]: e for e in Z.load_index(mt["split"], slim=True)}
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

        # ---- front hidden tokens (keys + GIMM, nav_plans protocol)
        def make_front():
            keys = OL.Keys(a.data)
            syn = np.load(OL.root(a.data) / "gimm.npy", mmap_mode="r")
            _, src = OL._steps(0.0, False)
            out = np.zeros((N, 8) + A.H_SHAPE, np.float16)
            t0 = time.time()

            def load(rows):
                kf, sf = keys[rows], np.asarray(syn[rows[0]:rows[-1] + 1])
                img = lambda j, s: (kf[j][src[s][1]] if src[s][0] == "k" else sf[j][src[s][1]]) if s >= 0 else np.zeros(FRAME, np.uint8)  # noqa: E731
                cur = np.stack([[img(j, s) for s in STEPS] for j in range(len(rows))])
                prev = np.stack([[img(j, s - 4) for s in STEPS] for j in range(len(rows))])
                return rows, prev.reshape(-1, *FRAME), cur.reshape(-1, *FRAME)
            chunks = [np.arange(i, min(i + 32, N)) for i in range(0, N, 32)]
            with ThreadPoolExecutor(8) as ex:
                for rows, prev, cur in run.tqdm(ex.map(load, chunks), total=len(chunks), desc="front"):
                    out[rows] = enc(prev, cur).reshape(len(rows), 8, *A.H_SHAPE)
            timing["front_pairs_per_s"] = 8 * N / (time.time() - t0)
            return out
        front = cache.cached(root / "front.npy", cache.key(params=kbase | dict(steps=STEPS.tolist()), code=[encoder]), make_front, force=a.force)

        # ---- side cameras
        def make_side():
            full = Z.load_index(mt["split"])
            pos = {e["token"]: k for k, e in enumerate(full)}
            items = [[{c: full[pos[t]]["cams"][f][c] for c in PA.SIDE_CAMS} for f in range(4)] for t in names]
            del full
            out = np.zeros((N, len(PA.SIDE_CAMS), PA.SIDE_T) + A.H_SHAPE, np.float16)
            t0, tb = time.time(), []
            W = a.workers or max(1, n_cpus() - 4)
            with ProcessPoolExecutor(W) as ex:
                buf, rows = [], []
                for i, fr in enumerate(run.tqdm(ex.map(render_side, items, chunksize=4), total=N, desc="side")):
                    buf.append(fr), rows.append(i)
                    if len(buf) == 32 or i == N - 1:
                        x = np.stack(buf)                                           # (b, 3, 4, ...)
                        prev, cur = x[:, :, :-1].reshape(-1, *FRAME), x[:, :, 1:].reshape(-1, *FRAME)
                        t1 = time.time()
                        out[rows] = enc(prev, cur).reshape(len(rows), len(PA.SIDE_CAMS), PA.SIDE_T, *A.H_SHAPE)
                        tb.append(time.time() - t1)
                        buf, rows = [], []
            timing |= {"side_tokens_per_s": N / (time.time() - t0), "side_gpu_busy": sum(tb) / (time.time() - t0), "render_workers": W}
            return out
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
        cache.cached(root / "teacher.npz", cache.key(params=kbase, inputs=[root / "front.npy"]),
                     make_teacher, force=a.force)
        run.summary |= {"n": N, "front_shape": list(front.shape), "side_shape": list(side.shape), **timing}
        if timing:
            (root / "timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    cli_args(ap)
    main(ap.parse_args())
