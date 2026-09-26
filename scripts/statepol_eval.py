"""Roll out BehaviorBench planners on the statepol scene variants and log per-step ego/partner states.

For each (variant split, ego planner, traffic) it runs pufferlib's Evaluator on shards of maps in parallel
processes (CPU inference: the policies are ~0.6M-parameter MLP/LSTMs and batch 1 for the ego) and writes
<out>/<variant>__<planner>__<traffic>.pkl = {map_id: dict(ego=(T,6) [x,y,vx,vy,heading,collision_state],
partner=(T,3) [x,y,valid], metrics=MapMetrics fields)}.
Usage (statepol env, cwd = behavior-bench repo):
  python statepol_eval.py --data <root with variant splits> --out <run dir> --variants base,nopartner \
      --planners ppo,idm --traffic expert --workers 60
"""
import argparse
import dataclasses
import json
import logging
import os
import pickle
import sys
import time
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

W = Path(os.environ.get("BB_ROOT", Path.home() / "data/third_party/statepol/behavior-bench"))
PLANNER_ARGS = {
    "ppo": ["--planner.type", "ppo", "--planner.ppo.weights-path", str(W / "weights/simple_ppo.pt")],
    "cond_normal": ["--planner.type", "conditioned_normal", "--planner.conditioned-normal.weights-path", str(W / "weights/conditioned_ppo.pt")],
    "cond_caut": ["--planner.type", "conditioned_caut", "--planner.conditioned-caut.weights-path", str(W / "weights/conditioned_ppo.pt")],
    "cond_aggr": ["--planner.type", "conditioned_aggr", "--planner.conditioned-aggr.weights-path", str(W / "weights/conditioned_ppo.pt")],
    "idm": ["--planner.type", "idm"],
    "pdm": ["--planner.type", "pdm"],
    "cv": ["--planner.type", "constant_velocity"],
}
TRAFFIC_ARGS = {
    "expert": ["--traffic.type", "expert"],
    "ppo": ["--traffic.type", "ppo", "--traffic.ppo.weights-path", str(W / "weights/simple_ppo.pt")],
}


def run_shard(job):
    variant_dir, planner, traffic, map_ids, partners = job
    import torch
    torch.set_num_threads(1)
    os.environ["DRIVE_BINARIES_DATA_ROOT"] = str(variant_dir)
    sys.argv = ["eval", "--eval.split", "validation_interactive", *PLANNER_ARGS[planner], *TRAFFIC_ARGS[traffic]]
    for k in ("planner", "traffic"):
        for t in ("ppo", "conditioned-normal", "conditioned-caut", "conditioned-aggr"):
            sys.argv += [f"--{k}.{t}.device", "cpu"]
    import pufferlib.ocean.drive.drive as D

    # Neural planners build a throwaway Drive env per map only to read obs/action shapes; that env probes
    # hundreds of maps (7-8 s of the ~9 s per episode). Cache it per kwargs and make close() a no-op.
    orig_drive, cache = D.Drive, {}

    def drive_factory(*args, **kw):
        if kw.get("max_controlled_agents") == 1 and "map_id" not in kw:
            key = repr(sorted(kw.items()))
            if key not in cache:
                env = orig_drive(*args, **kw)
                env.close = lambda: None
                cache[key] = env
            return cache[key]
        return orig_drive(*args, **kw)

    D.Drive = drive_factory
    from pufferlib.evaluation import Evaluator
    import pufferlib.ocean.benchmark.eval as E

    logging.disable(logging.CRITICAL)
    rec = {}

    class RecEvaluator(Evaluator):
        def _create_env(self, map_id, human_agent_idx=0):
            env = super()._create_env(map_id, human_agent_idx)
            if env is None:
                return None
            ego_e, par_e = human_agent_idx, partners.get(map_id, -1)
            log = rec.setdefault(map_id, dict(ego=[], partner=[]))
            step = env.step

            def rec_step(actions):
                out = step(actions)
                st = env.get_state()
                st = st[0] if isinstance(st, list) else st
                ents = st.get("entities", [])
                e = ents[ego_e]
                log["ego"].append((e["x"], e["y"], e["vx"], e["vy"], e["heading"], e.get("collision_state", 0)))
                if 0 <= par_e < len(ents):
                    p = ents[par_e]
                    log["partner"].append((p["x"], p["y"], p.get("valid", 1)))
                return out

            env.step = rec_step
            return env

        def _print_dashboard(self, *a, **k):
            pass

    # Reuse eval.py's config -> EvaluatorConfig logic by patching Evaluator and running its main().
    captured = {}

    class Capture(RecEvaluator):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["ev"] = self

    E.Evaluator = Capture
    tmp_out = Path(variant_dir) / "_evaltmp"
    sys.argv += ["--output-dir", str(tmp_out), "--map-ids", ",".join(map(str, map_ids))]
    null = open(os.devnull, "w")
    so, se = sys.stdout, sys.stderr
    if not os.environ.get("STATEPOL_DEBUG"):
        sys.stdout = sys.stderr = null
    try:
        E.main()
    except BaseException as ex:  # argparse exits or crashes must not hang the pool
        sys.stdout, sys.stderr = so, se
        print(f"shard {map_ids[:3]}... failed: {ex!r}", flush=True)
        return {}
    finally:
        sys.stdout, sys.stderr = so, se
    ev = captured["ev"]
    mets = {m.map_id: dataclasses.asdict(m) for m in ev.all_metrics}
    res = {}
    for mid, log in rec.items():
        res[mid] = dict(ego=np.asarray(log["ego"], np.float32), partner=np.asarray(log["partner"], np.float32),
                        metrics=mets.get(mid))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="base,nopartner,nullrm,obstacle,obsctrl,shift")
    ap.add_argument("--planners", default="ppo,cond_normal,cond_caut,cond_aggr,idm,pdm,cv")
    ap.add_argument("--traffic", default="expert")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--shard", type=int, default=8)
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_log = open(out / "events.jsonl", "a")
    ctx = get_context("spawn")
    from tqdm import tqdm
    for traffic in a.traffic.split(","):
        for variant in a.variants.split(","):
            vdir = Path(a.data) / variant
            metas = pickle.load(open(vdir / "meta.pkl", "rb"))
            if a.limit:
                metas = metas[: a.limit]
            partners = {m["map_id"]: (m["partner"] - (m["partner"] > m["null_removed"]) if variant == "nullrm"
                                      else m["partner"] if variant in ("base", "shift") else -1) for m in metas}
            ids = [m["map_id"] for m in metas]
            for planner in a.planners.split(","):
                dst = out / f"{variant}__{planner}__{traffic}.pkl"
                if dst.exists():
                    continue
                t0 = time.time()
                jobs = [(str(vdir), planner, traffic, ids[i:i + a.shard], partners) for i in range(0, len(ids), a.shard)]
                res = {}
                with ctx.Pool(a.workers, maxtasksperchild=4) as pool:
                    for r in tqdm(pool.imap_unordered(run_shard, jobs), total=len(jobs), desc=dst.stem):
                        res.update(r)
                pickle.dump(res, open(dst, "wb"))
                rec = dict(variant=variant, planner=planner, traffic=traffic, n=len(res), sec=round(time.time() - t0, 1))
                ev_log.write(json.dumps(rec) + "\n"); ev_log.flush()
                print(rec, flush=True)


if __name__ == "__main__":
    main()
