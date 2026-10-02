"""Image-channel route command, zero-shot on navtrain (openpilot env, one GPU). Plan: ../plans/2026-10-04-img-cmd-prereg.md.

Frozen Cinque (op_lb's TensorRT backend, action_t, step schedule), zero state per run, desire off, the leaderboard
history protocol (4 keyframes + GIMM frames, op_common_cause NavFrames). Per sample of geom/nav.pkl (img_geom_nav.py):
`none`, every overlay family x every commanded branch class, and the route-free controls once. Every distinct source
frame of the 31-step history is drawn with the ego pose at its source time (keys: logged poses; GIMM frames: op_interp's
EgoTrack), so the overlay stays fixed on the road across the history.

Output $DATA_DIR/runs/op_img_cmd/raw/nav/<tag>.npz: one row per run: token, fam, cmd, plan_pos (33, 3) / plan_yaw (33)
converted to the t0 rear-axle frame (x fwd, y left, yaw ccw), plan_v (33) forward speed, hidden (512) fp16.

  CUDA_VISIBLE_DEVICES=1 $DATA_DIR/envs/openpilot/bin/python experiments/op_img_cmd/scripts/img_run.py --shard 0/1 [--limit 10]
"""
import argparse, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments" / "op_common_cause" / "scripts"), str(Path(__file__).resolve().parent)]
import img_overlay as O  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_img_cmd"
ACTION_T = (0.275, 0.525)
CMDS = ("left", "straight", "right")
STRAIGHT_FAMS = ("band", "lines", "arrow_road", "sign")


def variants(s, fams=None):
    if s["kind"] == "straight":
        v = [("none", "")] + [(f, "straight") for f in STRAIGHT_FAMS]
    else:
        cls = [c for c in CMDS if any(b["cls"] == c for b in s["branches"])]
        v = [("none", "")] + [(f, c) for f in O.FAMILIES if f not in O.ROUTE_FREE for c in cls] + [(f, "") for f in O.ROUTE_FREE]
    return [x for x in v if fams is None or x[0] in fams]


_W = {}


def _init():
    from cc_run import NavFrames
    _W["nav"] = NavFrames()
    _W["meta"] = json.loads((data_dir() / "runs" / "op_lb" / "lb_navtrain" / "meta.json").read_text())


def render(s, fams=None):
    """All variants of one sample: {(fam, cmd): (frames (k, 2, 6, 128, 256), step -> frame index (31,))}."""
    nav, mt = _W["nav"], _W["meta"]
    r = s["row"]
    fr, src_t = nav.steps(r)
    cam = np.asarray(mt["cam"][r], float)
    tr = I.track_navsim(mt["pose"][r], mt["vel"][r])
    uniq, idx = {}, []
    for f, t in zip(fr, src_t):
        k = uniq.setdefault(float(t), (len(uniq), f, t))[0]      # one source frame per source time
        idx.append(k)
    base = sorted(uniq.values(), key=lambda z: z[0])
    poses = []
    for _, _, t in base:
        kk = np.flatnonzero(np.isclose(I.T_KEY, t))
        poses.append(np.asarray(mt["pose"][r][kk[0]], float) if len(kk) else tr(t))
    out = {}
    for fam, c in variants(s, fams):
        lay = O.primitives(s, fam, c or None)
        out[(fam, c)] = np.stack([O.draw(f, lay, p, cam) for (_, f, _), p in zip(base, poses)])
    return s["token"], out, np.array(idx)


def to_rear(pp, py, cam):
    """openpilot plan (calib frame at the camera, y right, yaw clockwise) -> t0 rear-axle frame (y left, yaw ccw)."""
    p = np.stack([pp[:, 0], -pp[:, 1]], -1).astype(np.float64)
    psi = -np.asarray(py, np.float64)
    d = np.asarray(cam[:2], np.float64)
    Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
    return d + p - Rd, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--kind", default="all", choices=("all", "junction", "straight"))
    ap.add_argument("--fams", nargs="*", help="only these families (+ none); default all")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    from drive_backbones_openpilot import bounded_map
    from jevdrive.data import splits
    from jevdrive.openpilot.model import OPModel, decode
    from jevdrive.run import Run
    import op_lb as B
    G = [s for s in pickle.load(open(ROOT / "geom" / "nav.pkl", "rb")) if O.valid(s) and a.kind in ("all", s["kind"])]
    G = [G[k] for k in np.random.default_rng(0).permutation(len(G))][si::sn][: a.limit or None]
    mt = json.loads((data_dir() / "runs" / "op_lb" / "lb_navtrain" / "meta.json").read_text())
    out = ROOT / "raw" / "nav"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"nav-{a.kind}-{si}of{sn}" + (f"-lim{a.limit}" if a.limit else "") + (f"-{'+'.join(a.fams)}" if a.fams else "")
    fams = None if not a.fams else set(a.fams) | {"none"}
    with Run("op_img_cmd", tag, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        ex = ProcessPoolExecutor(a.workers, initializer=_init)        # fork before the TensorRT session exists
        list(ex.map(int, range(a.workers)))
        m = OPModel("cinque", B.BACKENDS["cinque"], cache=data_dir() / "runs" / "op_interp" / "trt_cache" / f"cinque-{B.BACKENDS['cinque']}",
                    context_rate=False)
        zero = np.zeros(8, np.float32)
        R = {k: [] for k in ("token", "fam", "cmd", "plan_pos", "plan_yaw", "plan_v", "hidden")}
        t0, nrun = time.time(), 0
        for i, (tok, var, idx) in enumerate(run.tqdm(bounded_map(ex, partial(render, fams=fams), G, 2 * a.workers), total=len(G), desc=tag)):
            s = G[i]
            assert s["token"] == tok
            tc = (0, 1) if mt["lht"][s["row"]] else (1, 0)
            cam = mt["cam"][s["row"]]
            for (fam, c), frames in var.items():
                m.reset()
                for k in idx:
                    raw = m.step(frames[k], desire=zero, traffic=tc, action_t=ACTION_T)
                d = decode(raw, m.slices, max(s["v"], 0.0), ACTION_T)
                xy, yaw = to_rear(d["plan_pos"], d["plan_yaw"], cam)
                R["token"].append(tok); R["fam"].append(fam); R["cmd"].append(c)
                R["plan_pos"].append(xy.astype(np.float32)); R["plan_yaw"].append(yaw.astype(np.float32))
                R["plan_v"].append(d["plan_vel"][:, 0].astype(np.float32))
                R["hidden"].append(raw[m.slices["hidden_state"]].astype(np.float16))
                nrun += 1
            if i == 0 or (i + 1) % 25 == 0:
                rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
                run.info(f"[{i + 1}/{len(G)}] {nrun} runs, {(time.time() - t0) / (i + 1):.2f} s/sample, rss {rss:.1f} GB")
        ex.shutdown()
        np.savez(out / f"{tag}.tmp.npz", **{k: np.array(v) for k, v in R.items()})
        os.replace(out / f"{tag}.tmp.npz", out / f"{tag}.npz")
        run.summary.update(n=len(G), runs=nrun, loop_s=time.time() - t0)
    sys.stdout.flush()
    os._exit(0)          # TensorRT teardown can hang


if __name__ == "__main__":
    main()
