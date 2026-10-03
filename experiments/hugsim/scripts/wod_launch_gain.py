"""Local launch gain and launch lean of openpilot on REAL WOD launches (CPU onnxruntime).  Prereg: experiments/hugsim/plans/2026-10-04-wod-launch-gain-prereg.txt
Same measurement as spin_attr_cpu_gain.py / lean_probe.py (decisions 100, 110) on the decision-106 `lwod` pool: rows of 10 packed 5 Hz model frames
(oldest first, the last at t0 = the m-th moving frame, m = 1..3 = HUGSIM step m), the first 10 - m frames static (v < 0.1 m/s for >= 1.8 s).
Per row and fake yaw w in (0, +W, -W) deg / model-s: the history frames get yaw w * (j - 9) * 0.2 deg (left +, t0 unchanged), frame 0 is held WARM
reps (HUGSIM's static warm-up), the others PER reps; read phi1 = direction of the HUGSIM 1 s plan point (lean_probe.metrics, DIL 1.25).
    python wod_launch_gain.py <model: cinque | <onnx name>> <out.jsonl> --shard K --nshard N [--threads 4] [--limit E] [--w 2]
Events are visited dev-split first, then in a fixed random order, so a partial run is a random subset; resumable (rows already in out.jsonl are skipped)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments/hugsim/scripts"), str(REPO / "experiments/op_adapt_h/lib")]
import lean_probe as P  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

SAMPLES = data_dir() / "runs" / "op_adapt_H" / "samples" / "lwod"


def warp(f, cam, dpsi):
    import cv2
    from jevdrive import op_interp as I
    if abs(dpsi) < 1e-9:
        return f
    out = np.empty_like(f)
    c = np.round(np.asarray(cam, float), 2)
    for k, view in enumerate(("road", "wide")):
        mx, my = I.warp_map(view, c, np.array([0.0, 0.0, dpsi]), np.zeros(3))
        hx, hy = (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25
        Y, U, V = I.unpack(f[k])
        out[k] = I.pack(cv2.remap(Y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                        cv2.remap(np.ascontiguousarray(U), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                        cv2.remap(np.ascontiguousarray(V), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def events(tab):
    """event key (seq, onset frame) -> {m: row}; events in a fixed random order, dev-split events first."""
    ev = {}
    for r, (i, m) in enumerate(zip(tab["id"], tab["m"])):
        q, f = str(i).rsplit("-", 1)
        ev.setdefault((q, int(f) - 2 * (int(m) - 1)), {})[int(m)] = r
    keys = sorted(ev)
    order = [ev[keys[i]] for i in np.random.default_rng(0).permutation(len(keys))]
    dev = lambda e: tab["split"][next(iter(e.values()))] == "dev"  # noqa: E731
    return [e for e in order if dev(e)] + [e for e in order if not dev(e)]      # dev (held out of adapted models) first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="events in total (all shards)")
    ap.add_argument("--w", type=float, default=2.0)
    a = ap.parse_args()
    from jevdrive.openpilot.model import OPModel
    name = "cinque" if a.model == "cinque" else str(data_dir() / "runs" / "op_adapt_H" / "onnx" / f"{a.model}.onnx")
    m = OPModel(name, "cpu", threads=a.threads)
    z = np.load(SAMPLES / "tab.npz", allow_pickle=True)
    tab = {k: z[k] for k in z.files}
    imgs = np.load(SAMPLES / "imgs.npy", mmap_mode="r")
    evs = events(tab)[: a.limit or None][a.shard::a.nshard]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["row"] for x in open(a.out)}
    t0 = time.time()
    with open(a.out, "a") as fo:
        for n, ev in enumerate(evs):
            for mm, r in sorted(ev.items()):
                if r in done:
                    continue
                fr = np.array(imgs[r])
                t = tab["img_t"][r]
                res = {"row": int(r), "m": mm, "id": str(tab["id"][r]), "cluster": str(tab["cluster"][r]), "split": str(tab["split"][r]),
                       "v0": float(tab["v0"][r]), "model": a.model}
                for w in (0.0, a.w, -a.w):
                    seq = [(warp(fr[j], tab["cam"][r], np.radians(w * t[j])), P.WARM if j == 0 else P.PER, 0, j == 9) for j in range(10)]
                    res[f"{w:+g}"] = P.feed(m, seq, [float(x) for x in tab["tc"][r]])[0]
                fo.write(json.dumps(res) + "\n")
                fo.flush()
            print(f"{a.model} shard {a.shard}: event {n + 1}/{len(evs)} {time.time() - t0:.0f}s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
