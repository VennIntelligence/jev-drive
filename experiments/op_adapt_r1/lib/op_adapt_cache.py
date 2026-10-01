"""op-adapt: per-stream caches of Cinque's frozen trunk output (stage 3, `permute_73`) for training and evaluation
(op-train venv). Every cached slot j is one image pair; a sample at slot j reads the 9 context slots
j, j - c, .., j - 8c (zero hidden where < 0, like a stream started from a zero state).

  nusc  per scene, the nuScenes protocol (jevdrive.op_adapt_data): slots = even 20 Hz steps, pair (step s-4, s), c = 2
  wod   WOD val streams of processed/drive_backbones/op_plan.json (every frame fed twice): slots = frames,
        pair (f-2, f), c = 2
  p5    the P5 v1 BA streams (processed/carla_p5v1_ba/op_plan.json, frames held 4 steps): slots = frames, pair (f-1, f), c = 1

  wodtrain  WOD train streams of processed/op_adapt/wod_train_plan.json (even frame numbers only): slots = those
            frames, pair (f-2, f), c = 1 (labels: wod_train_labels.parquet)
  wodval    WOD-E2E val, every sequence, even frame numbers, one stream per contiguous run of frames (op-adapt L: the full
            held-out set; streams from wod_val_plan.json written by experiments/op_adapt_l/scripts/op_adapt_l_prep.py), same protocol as wodtrain
  navtrain  NAVSIM navtrain on its 2 Hz sample-and-hold protocol, one file per log: the unique image pairs of the
            tokens' 9-slot contexts (t = -1.6 .. 0 at 0.2 s, each the latest 2 Hz frame; before -1.5 s a zero hidden
            state / zero image), `ctx` (n_tokens, 9) = slot index per context position (-1 = zero hidden state)

Output: processed/op_adapt/<dataset>/<key>.npz with trunk (n, 1024, 8, 16) fp16, stride (or ctx), per-slot metadata.

  CUDA_VISIBLE_DEVICES=2 python experiments/op_adapt_r1/lib/op_adapt_cache.py nusc --workers 24
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("scripts",)]
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
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


def wod_train_job(st):
    import drive_backbones_openpilot as R
    fr = R.render(st["names"])
    prev = np.concatenate([np.zeros_like(fr[:1]), fr[:-1]])
    return st["key"], prev, fr, {"stride": 1, "names": np.array(st["names"]), "targets": np.array(st["targets"]),
                                 "traffic": np.array([1.0, 0.0], np.float32)}


def p5_job(st):
    import p5_openpilot as P
    fr = P.render(st["files"], st.get("seq", P.SEQ))
    prev = np.concatenate([np.zeros_like(fr[:1]), fr[:-1]])
    return st["key"], prev, fr, {"stride": 1, "names": np.array(st["names"]), "targets": np.array(st["targets"]),
                                 "traffic": np.array([1.0, 0.0], np.float32)}


def navtrain_job(job):
    import navsim_zs_openpilot as NZ
    from jevdrive import navsim_zs as Z
    log, entries = job
    T = np.round(np.arange(-8, 1) * 0.2, 3)
    slot = lambda t: int(np.searchsorted(Z.T_HIST2, t + 1e-6) - 1) if t >= -1.5 - 1e-6 else -1  # noqa: E731
    paths, frames, pairs, ctx, toks, tcs = {}, [], {}, [], [], []
    for e in entries:
        cams = e["cams"][-1]
        key = Z.calib_key({"CAM_F0": cams["CAM_F0"]})
        m = NZ._maps.get(key) or NZ._maps.setdefault(key, Z.OpenpilotMaps(cams["CAM_F0"]))

        def fidx(k):
            p = e["cams"][k]["CAM_F0"]["path"]
            if p not in paths:
                paths[p] = len(frames)
                frames.append(m(m.decode(p)))
            return paths[p]
        row = []
        for t in T:
            c = slot(t)
            if c < 0:
                row.append(-1)
                continue
            pr = slot(round(t - 0.2, 3))
            pk = (fidx(pr) if pr >= 0 else -1, fidx(c))
            row.append(pairs.setdefault(pk, len(pairs)))
        ctx.append(row)
        toks.append(e["token"])
        tcs.append((0.0, 1.0) if e["map"] in NZ.LHT_MAPS else (1.0, 0.0))
    F = np.concatenate([np.zeros((1, 2, 6, 128, 256), np.uint8), np.stack(frames)])      # row 0 = zero image
    pk = np.array(list(pairs), np.int64) + 1
    return log, F[pk[:, 0]], F[pk[:, 1]], {"ctx": np.array(ctx, np.int32), "tokens": np.array(toks),
                                           "traffic": np.array(tcs, np.float32)}


def items(a):
    if a.dataset == "wodtrain":
        import wod_zeroshot_openpilot as WZ
        from jevdrive import drive_backbones as DB
        from jevdrive import wod_zeroshot as Z
        plan = json.loads((DB.root() / DB.plan_name("trainval")).read_text())
        calib = json.loads((Z.root() / "op_calib.json").read_text()) | json.loads((DB.root() / "op_calib_trainval.json").read_text())
        sts = json.loads((D.root() / "wod_train_plan.json").read_text())["streams"]
        return sts, wod_train_job, WZ._init, (plan["spans"], calib, str(data_dir() / "datasets" / "waymo_e2e" / "front3"))
    if a.dataset == "wodval":
        import wod_zeroshot_openpilot as WZ
        from jevdrive import drive_backbones as DB
        from jevdrive import wod_zeroshot as Z
        plan = json.loads((DB.root() / DB.plan_name("trainval")).read_text())
        calib = json.loads((Z.root() / "op_calib.json").read_text()) | json.loads((DB.root() / "op_calib_trainval.json").read_text())
        sts = json.loads((D.root() / "wod_val_plan.json").read_text())["streams"]
        return sts, wod_train_job, WZ._init, (plan["spans"], calib, str(data_dir() / "datasets" / "waymo_e2e" / "front3"))
    if a.dataset == "navtrain":
        from collections import defaultdict
        import pandas as pd
        from jevdrive import navsim_zs as Z
        keep = set(pd.read_parquet(D.root() / "navtrain_labels.parquet").token)
        by = defaultdict(list)
        for e in Z.load_index("navtrain", slim=True):
            if e["token"] in keep:
                by[e["log_name"]].append(e)
        return sorted(by.items()), navtrain_job, None, ()
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
    ap.add_argument("dataset", choices=("nusc", "wod", "p5", "wodtrain", "wodval", "navtrain"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64, help="image pairs per trunk forward (64: ~15 GB)")
    ap.add_argument("--out-sub", default="", help="write under processed/op_adapt/<out-sub> instead of <dataset> (pilots)")
    ap.add_argument("--limit", type=int, default=0, help="wod: number of streams (random, seed 0); others: first n")
    a = ap.parse_args()
    log = RunLog("op_adapt", f"cache-{a.dataset}")
    its, job, init, initargs = items(a)
    if a.dataset != "wod" and a.limit:
        its = its[: a.limit]
    out = D.root(a.out_sub or a.dataset)
    key = (lambda x: x) if a.dataset == "nusc" else (lambda x: x[0]) if a.dataset == "navtrain" else (lambda x: x["key"])
    its = [x for x in its if not (out / f"{key(x)}.npz").exists()]
    log.info(f"{len(its)} streams to cache -> {out}")
    t0, n, nslot = time.time(), 0, 0
    from drive_backbones_openpilot import bounded_map
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=initargs) if init else ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(int, range(a.workers)))       # fork the render workers before CUDA exists in this process
        net = A.load("cinque", torch.float16).cuda()
        for k, prev, cur, meta in bounded_map(ex, job, its, 2 * a.workers):
            T = trunk(net, prev, cur, a.batch)
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
