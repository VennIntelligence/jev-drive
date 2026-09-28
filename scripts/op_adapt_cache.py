"""op-adapt: per-stream caches of Cinque's frozen trunk output (stage 3, `permute_73`) for training and evaluation
(op-train venv). Every cached slot j is one image pair; a sample at slot j reads the 9 context slots
j, j - c, .., j - 8c (zero hidden where < 0, like a stream started from a zero state).

  nusc  per scene, the nuScenes protocol (jevdrive.op_adapt_data): slots = even 20 Hz steps, pair (step s-4, s), c = 2
  wod   WOD val streams of processed/drive_backbones/op_plan.json (every frame fed twice): slots = frames,
        pair (f-2, f), c = 2
  p5    the P5 v1 BA streams (processed/carla_p5v1_ba/op_plan.json, frames held 4 steps): slots = frames, pair (f-1, f), c = 1

Output: processed/op_adapt/<dataset>/<key>.npz with trunk (n, 1024, 8, 16) fp16, stride, and per-slot metadata.

  CUDA_VISIBLE_DEVICES=2 python scripts/op_adapt_cache.py nusc --workers 24
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_data as D  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402


@torch.no_grad()
def trunk(net, prev, cur, batch=64):
    """prev / cur: (n, 2, 6, 128, 256) uint8 numpy -> (n, 1024, 8, 16) fp16 numpy."""
    out = []
    for i in range(0, len(cur), batch):
        p = torch.as_tensor(prev[i:i + batch]).cuda(non_blocking=True)
        c = torch.as_tensor(cur[i:i + batch]).cuda(non_blocking=True)
        o = net.run_batched(A.vision_feeds(p, c), [A.TRUNK_OUT])[A.TRUNK_OUT]
        out.append(o[:, 0].to(torch.float16).cpu())
    return torch.cat(out).numpy()


# ---- dataset jobs: each yields (key, prev frames, cur frames, meta) ----
def nusc_job(name):
    name, plan, frames, inv, cov = D.render_scene(name)
    steps = np.arange(0, len(plan["t"]), 2)
    zero = np.zeros_like(frames[:1])
    cur = frames[inv[steps]]
    prev = np.concatenate([zero, frames])[np.where(steps >= 4, inv[np.maximum(steps - 4, 0)] + 1, 0)]
    key_slot = plan["key_step"] // 2
    return name, prev, cur, {"stride": 2, "steps": steps, "key_slot": key_slot, "tokens": np.array(plan["tokens"]),
                             "traffic": np.array(plan["traffic"], np.float32), "coverage": cov}


def wod_job(st):
    import drive_backbones_openpilot as R
    names = st["names"]
    fr = R.render(names)
    prev = np.concatenate([np.zeros_like(fr[:2]), fr[:-2]])
    return st["key"], prev, fr, {"stride": 2, "names": np.array(names), "targets": np.array(st["targets"]),
                                 "traffic": np.array([1.0, 0.0], np.float32)}


def p5_job(st):
    import p5_openpilot as P
    fr = P.render(st["files"], st.get("seq", P.SEQ))
    prev = np.concatenate([np.zeros_like(fr[:1]), fr[:-1]])
    return st["key"], prev, fr, {"stride": 1, "names": np.array(st["names"]), "targets": np.array(st["targets"]),
                                 "traffic": np.array([1.0, 0.0], np.float32)}


def items(a):
    if a.dataset == "nusc":
        from jevdrive import nuscenes_zs as Z
        idx = Z.load_index("trainval")
        sc = sorted(idx["scenes"])
        return sc, nusc_job, D._init, ()
    if a.dataset == "wod":
        import wod_zeroshot_openpilot as WZ
        from jevdrive import drive_backbones as DB
        from jevdrive import wod_zeroshot as Z
        plan = json.loads((DB.root() / DB.plan_name("subset")).read_text())
        sts = [dict(s, key=f"{i:04d}_{s['sequence']}") for i, s in enumerate(plan["streams"])]
        rng = np.random.default_rng(0)
        sts = [sts[i] for i in sorted(rng.choice(len(sts), a.limit, replace=False))] if a.limit else sts
        calib = json.loads((Z.root() / "op_calib.json").read_text())
        return sts, wod_job, WZ._init, (plan["spans"], calib, str(data_dir() / "datasets" / "waymo_e2e" / "front3"))
    os.environ["P5_SET"] = "carla_p5v1_ba"
    import wod_zeroshot_openpilot as WZ
    import p5_openpilot as P
    from jevdrive import p5_openpilot as PP
    plan = json.loads((PP.root() / "op_plan.json").read_text())
    calibs = plan.get("calibs") or {P.SEQ: plan.get("calib")}
    return plan["streams"], p5_job, WZ._init, ({}, calibs, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=("nusc", "wod", "p5"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="wod: number of streams (random, seed 0); others: first n")
    a = ap.parse_args()
    log = RunLog("op_adapt", f"cache-{a.dataset}")
    its, job, init, initargs = items(a)
    if a.dataset != "wod" and a.limit:
        its = its[: a.limit]
    out = D.root(a.dataset)
    key = (lambda x: x) if a.dataset == "nusc" else (lambda x: x["key"])
    its = [x for x in its if not (out / f"{key(x)}.npz").exists()]
    log.info(f"{len(its)} streams to cache -> {out}")
    t0, n, nslot = time.time(), 0, 0
    from drive_backbones_openpilot import bounded_map
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=initargs) as ex:
        list(ex.map(int, range(a.workers)))       # fork the render workers before CUDA exists in this process
        net = A.load("cinque", torch.float16).cuda()
        for k, prev, cur, meta in bounded_map(ex, job, its, 2 * a.workers):
            T = trunk(net, prev, cur)
            tmp = out / f"{k}.tmp.npz"
            np.savez(tmp, trunk=T, **meta)
            tmp.replace(out / f"{k}.npz")
            n, nslot = n + 1, nslot + len(T)
            if n % 20 == 0 or n == len(its):
                el = time.time() - t0
                log.info(f"[{n}/{len(its)}] {nslot} slots, {el / 60:.1f} min, ETA {(len(its) - n) * el / n / 60:.0f} min")
                log.event("progress", n=n, slots=nslot, wall_s=el)
    log.event("end", streams=n, slots=nslot, wall_s=time.time() - t0)


if __name__ == "__main__":
    main()
