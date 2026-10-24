#!/usr/bin/env python3
"""Rescore stored PAI 281 reads with AlpaSim's fixed ground-truth corridor scorer (e2e_challenge 1b024f0, PR #215).

The fix changes only `eval/scorers/ground_truth.py` (`_end_direction` uses the last >= 1 m of GT motion; the lateral distance is
min(plain, perpendicular to the forward ray from the GT end) whenever the ego is ahead of the end). Stored reads keep `rollout.asl`
only for zero-score rollouts, so those are re-evaluated offline with AlpaSim's own `eval` code (the same calls as `eval.main` /
`eval/reeval.py`: ASL -> ScenarioEvaluator -> per-rollout metrics.parquet -> `_run_aggregation_core`); every other rollout keeps its
stored metrics.parquet in a mirror tree. Nothing under the stored reads or the live AlpaSim checkout is written.

A nonzero rollout without an ASL can only change at the one kept step where the plain GT distance first reaches 4 m (the truncation
event): `risk` lists those rollouts, with the scene's GT end shape (old vs new end direction) when an ASL of that scene exists anywhere.

Box, repo root (score runs in the AlpaSim venv; the eval package is chosen by PYTHONPATH and checked against --variant):
  PY=/root/autodl-tmp/ujs/third_party/alpasim/.venv/bin/python NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
  $PY experiments/alpasim/scripts/scorer_v2.py score --variant old --root $R [--reads A,B] [--limit 3]
  PYTHONPATH=$NEW $PY experiments/alpasim/scripts/scorer_v2.py score --variant new --root $R
  .venv/bin/python experiments/alpasim/scripts/scorer_v2.py report --root $R --out experiments/alpasim/results/scorer_v2
"""
import argparse
import glob
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

# Each eval worker is single-threaded: polars / BLAS would otherwise start a thread per core in every worker, and 64 workers hit the
# box's pids limit (polars panics with EAGAIN, the worker dies; 10-23 first run).
for _k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

RUNS = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data"))) / "runs"
FACTS = RUNS / "pai_fix/facts"
# label -> (stored read dir under $DATA_DIR/runs, pai_fix facts file stem or "")
READS = {
    "NTV0-L18-s0-ag": ("defect_fix/pai_accgap/NTV0-L18-s0", "NTV0-L18-s0-281ag"),
    "NTV0-L18-s0-agH": ("round7/harmcl/NTV0-L18-s0-agH", "NTV0-L18-s0-281agH"),
    "RSG-L18-s0-ag": ("route_follow/acc_gap/read/s2", ""),
    "R6DAG-L18-s0-ag": ("round6/dag/pai281/R6DAG-L18-s0-ag", "R6DAG-L18-s0-281ag"),
    "R6DAG-L18-s0-agH": ("round7/harmcl/R6DAG-L18-s0-agH", "R6DAG-L18-s0-281agH"),
    "R6DAG-L18-s1-ag": ("round6/dag/pai281/R6DAG-L18-s1-ag", "R6DAG-L18-s1-281ag"),
    "R6DAG-L18-s1-agH": ("round7/harmcl/R6DAG-L18-s1-agH", "R6DAG-L18-s1-281agH"),
    "R6DAG-L18-s2-ag": ("round8/s2/pai281/R6DAG-L18-s2-ag", "R6DAG-L18-s2-281ag"),
    "R6DAG-L18-s2-agH": ("round8/s2/pai281/R6DAG-L18-s2-agH", "R6DAG-L18-s2-281agH"),
    "R8K-L18-s0-ag": ("round8/r8k/pai281/R8K-L18-s0-ag", "R8K-L18-s0-281ag"),
    "R8K-L18-s0-agH": ("round8/r8k/pai281/R8K-L18-s0-agH", "R8K-L18-s0-281agH"),
    "R8K-L18-s1-ag": ("round8/r8k/pai281/R8K-L18-s1-ag", "R8K-L18-s1-281ag"),
    "R8K-L18-s1-agH": ("round8/r8k/pai281/R8K-L18-s1-agH", "R8K-L18-s1-281agH"),
    "R8K-L18-s2-ag": ("round8/s2/pai281/R8K-L18-s2-ag", "R8K-L18-s2-281ag"),
    "R8K-L18-s2-agH": ("round8/s2/pai281/R8K-L18-s2-agH", "R8K-L18-s2-281agH"),
    # other stored 281 ACC + GAP reads (cheap: only their zero rollouts are re-evaluated)
    "NTV0-L18-s1-ag": ("defect_fix/pai_accgap/NTV0-L18-s1", ""),
    "JTN-L18-s0-ag": ("defect_fix/pai_accgap/JTN-L18-s0", ""),
    "JTN-L18-s1-ag": ("defect_fix/pai_accgap/JTN-L18-s1", ""),
    "BNC2K-L18-s1-ag": ("defect_fix/pai_accgap/BNC2K-L18-s1", ""),
    "JTXS-L18-s1-ag": ("defect_fix/pai_accgap/JTXS-L18-s1", ""),
    "NTVX-L18-s0-ag": ("defect_fix/pai_accgap/NTVX-L18-s0", ""),
    "NTVXS-L18-s0-ag": ("defect_fix/pai_accgap/NTVXS-L18-s0", ""),
    "R2C-L18-s0-ag": ("round2/pai281/R2C-L18-s0-ag", "R2C-L18-s0-281ag"),
    "R2J-L18-s0-ag": ("round2/pai281/R2J-L18-s0-ag", "R2J-L18-s0-281ag"),
    "R3MV-L18-s0-ag": ("round5/r5_281/pai281/R3MV-L18-s0-ag", "R3MV-L18-s0-281ag"),
    "R3MV-L18-s1-ag": ("round5/r5/pai281/R3MV-L18-s1-ag", "R3MV-L18-s1-281ag"),
    "R5A-L18-s0-ag": ("round5/r5_281_R5A/pai281/R5A-L18-s0-ag", "R5A-L18-s0-281ag"),
    "R5LC-L18-s0-ag": ("round5/r5/pai281/R5LC-L18-s0-ag", "R5LC-L18-s0-281ag"),
    "R5LC-L18-s1-ag": ("round5/r5/pai281/R5LC-L18-s1-ag", "R5LC-L18-s1-281ag"),
    "R6DAG2-L18-s0-ag": ("round6/dag2/pai281/R6DAG2-L18-s0-ag", "R6DAG2-L18-s0-281ag"),
    "R6DAG2-L18-s1-ag": ("round6/dag2/pai281/R6DAG2-L18-s1-ag", "R6DAG2-L18-s1-281ag"),
    "R6FIX-L18-s0-ag": ("round6/fix/pai281/R6FIX-L18-s0-ag", "R6FIX-L18-s0-281ag"),
    "R6FIX-L18-s1-ag": ("round6/fix/pai281/R6FIX-L18-s1-ag", "R6FIX-L18-s1-281ag"),
    "R6REC-L18-s0-ag": ("round6/rec/pai281/R6REC-L18-s0-ag", "R6REC-L18-s0-281ag"),
    "R6REC-L18-s1-ag": ("round6/rec/pai281/R6REC-L18-s1-ag", "R6REC-L18-s1-281ag"),
    "R7J-L18-s0-ag": ("round7/joint/pai281/R7J-L18-s0-ag", "R7J-L18-s0-281ag"),
    "R7J-L18-s0-agH": ("round7/joint/pai281/R7J-L18-s0-agH", "R7J-L18-s0-281agH"),
    "NTV0NS-L18-s0-ag": ("round9/ntv0_nos/pai281/NTV0NS-L18-s0-ag", "NTV0NS-L18-s0-281ag"),   # round9 ntv0_nos (scored with --reads only)
    "RB0L-F-s0-ag": ("round9/rb0_long/pai281/RB0L-F-s0-ag", "RB0L-F-s0-281ag"),   # round9 rb0_long (scored with --reads only)
    "PCL1-F-s0-ag": ("round9/pool_pilot/pai281/PCL1-F-s0-ag", "PCL1-F-s0-281ag"),   # round9 pool_pilot (scored with --reads only)
    "PCL1B-F-s0-ag": ("round9/pool_pilot/pai281/PCL1B-F-s0-ag", "PCL1B-F-s0-281ag"),   # round9 pool_pilot (scored with --reads only)
    "PCL0-L18-s0-ag": ("round9/pool_pilot/pai281/PCL0-L18-s0-ag", "PCL0-L18-s0-281ag"),   # round9 pool_pilot (scored with --reads only)
    "PCL1-F-s1-ag": ("round9/p1_s1/pai281/PCL1-F-s1-ag", "PCL1-F-s1-281ag"),   # round9 p1_s1 (scored with --reads only)
    "PCL1L-F-s0-ag": ("round9/p1l/pai281/PCL1L-F-s0-ag", "PCL1L-F-s0-281ag"),   # round9 p1l (scored with --reads only)
    "PCL1LA-F-s0-ag": ("round9/p1l_split/pai281/PCL1LA-F-s0-ag", "PCL1LA-F-s0-281ag"),   # round9 p1l_split (scored with --reads only)
    "PCL1LB-F-s0-ag": ("round9/p1l_split/pai281/PCL1LB-F-s0-ag", "PCL1LB-F-s0-281ag"),
    "SSNH-L18-s0-ag": ("round9/s_split/pai281/SSNH-L18-s0-ag", "SSNH-L18-s0-281ag"),   # round9 s_split (scored with --reads only)
    "SSND-L18-s0-ag": ("round9/s_split/pai281/SSND-L18-s0-ag", "SSND-L18-s0-281ag"),
    "HF-s0-ag": ("round9/cmd_pilot/pai281/HF-s0-ag", "HF-s0-281ag"),   # round9 cmd_pilot (scored with --reads only)
    "INTP50-NR-ag": ("round9/interp/pai281/INTP50-NR-ag", "INTP50-NR-281ag"),   # round9 interp (scored with --reads only)
    "INTP75-NR-ag": ("round9/interp/pai281/INTP75-NR-ag", "INTP75-NR-281ag"),
    "PCL1L-F-s0-agH": ("round9/diag_reads/pai281/PCL1L-F-s0-agH", "PCL1L-F-s0-281agH"),   # round9 diag_reads (scored with --reads only)
    "INTP75-NR-ag-rr": ("round9/diag_reads/pai281/INTP75-NR-ag-rr", "INTP75-NR-281ag-rr"),   # same checkpoint and serve as INTP75-NR-ag, re-read
    "PCL1-F-s0x10000-ag": ("round9/snap10k/pai281/PCL1-F-s0x10000-ag", "PCL1-F-s0x10000-281ag"),   # round9 snap10k (scored with --reads only)
    "HP50-s0-ag": ("round9/cmd_pilot/pai281/HP50-s0-ag", "HP50-s0-281ag"),
    "NT1N1-F-s0-ag": ("round9/ntv1_step1/pai281/NT1N1-F-s0-ag", "NT1N1-F-s0-281ag"),   # round9 ntv1_step1 (scored with --reads only)
    "NT1N2A-F-s0-ag": ("round9/ntv1_step1/pai281/NT1N2A-F-s0-ag", "NT1N2A-F-s0-281ag"),
    "NT1N2B-F-s0-ag": ("round9/ntv1_step1/pai281/NT1N2B-F-s0-ag", "NT1N2B-F-s0-281ag"),
    "NT3D1-F-s0-ag": ("round9/ntv1_step3/pai281/NT3D1-F-s0-ag", "NT3D1-F-s0-281ag"),   # round9 ntv1_step3 (scored with --reads only; s1 / agH = the candidate tail)
    "NT3D1-F-s0-agH": ("round9/ntv1_step3/pai281/NT3D1-F-s0-agH", "NT3D1-F-s0-281agH"),
    "NT3D1-F-s1-ag": ("round9/ntv1_step3/pai281/NT3D1-F-s1-ag", "NT3D1-F-s1-281ag"),
    "NT3D1-F-s1-agH": ("round9/ntv1_step3/pai281/NT3D1-F-s1-agH", "NT3D1-F-s1-281agH"),
    "NT3D2-F-s0-ag": ("round9/ntv1_step3/pai281/NT3D2-F-s0-ag", "NT3D2-F-s0-281ag"),
    "NT3D2-F-s0-agH": ("round9/ntv1_step3/pai281/NT3D2-F-s0-agH", "NT3D2-F-s0-281agH"),
    "NT3D2-F-s1-ag": ("round9/ntv1_step3/pai281/NT3D2-F-s1-ag", "NT3D2-F-s1-281ag"),
    "NT3D2-F-s1-agH": ("round9/ntv1_step3/pai281/NT3D2-F-s1-agH", "NT3D2-F-s1-281agH"),
    "R7J-L18-s1-ag": ("round7/joint/pai281/R7J-L18-s1-ag", "R7J-L18-s1-281ag"),
    "R7J-L18-s1-agH": ("round7/joint/pai281/R7J-L18-s1-agH", "R7J-L18-s1-281agH"),
    "R9DL-L18-s0-ag": ("round9/dag_lane/pai281/R9DL-L18-s0-ag", "R9DL-L18-s0-281ag"),
    "R9DL-L18-s0-agH": ("round9/dag_lane/pai281/R9DL-L18-s0-agH", "R9DL-L18-s0-281agH"),
    "R9DL-L18-s1-ag": ("round9/dag_lane/pai281/R9DL-L18-s1-ag", "R9DL-L18-s1-281ag"),
    "R9DL-L18-s1-agH": ("round9/dag_lane/pai281/R9DL-L18-s1-agH", "R9DL-L18-s1-281agH"),
    "RB0-F-s0-ag": ("round9/pai_base/pai281/RB0-F-s0-ag", "RB0-F-s0-281ag"),
    "RB0-P30-s0-ag": ("round9/pai_base/pai281/RB0-P30-s0-ag", "RB0-P30-s0-281ag"),
    "RB0-P50-s0-ag": ("round9/pai_base/pai281/RB0-P50-s0-ag", "RB0-P50-s0-281ag"),
    "RB0-P70-s0-ag": ("round9/pai_base/pai281/RB0-P70-s0-ag", "RB0-P70-s0-281ag"),
    # round9 VAL160 (152 PAI val160 scenes, ACC + GAP, Harmonizer off; scored with --reads only)
    "NT3D1-F-s0-v160": ("round9/val160/reads/NT3D1-F-s0-ag", "NT3D1-F-s0-v160ag"),
    "R8K-L18-s2-v160": ("round9/val160/reads/R8K-L18-s2-ag", "R8K-L18-s2-v160ag"),
    "PCL1L-F-s0-v160": ("round9/val160/reads/PCL1L-F-s0-ag", "PCL1L-F-s0-v160ag"),
    # PAI-PCS-V2 anchor re-reads (third-party drivers, zero ASLs kept): garage model_0014 and the starter-kit go-straight
    "G0014-v2": ("alpasim_proxy/pai_pcs_v2/G0014", ""),
    "GS-v2": ("alpasim_proxy/pai_pcs_v2/GS", ""),
    "GS5-v2": ("alpasim_proxy/pai_pcs_v2/GS5", ""),            # GS-v2's 5 thread-cap failures re-read (KEEP_ASL=1)
}
CORRIDOR_M = 4.0


def chunks(label):
    """Stored chunk sim dirs of a read, in order (a later chunk's rollout of the same scene wins, as in pai_report.rollouts)."""
    d = RUNS / READS[label][0]
    return [Path(p).parents[2] for p in sorted(glob.glob(str(d / "runs/*/sim/aggregate/results-summary.json")))]


# ---------------------------------------------------------------- score (AlpaSim venv)
_CACHE = {}


def _cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def _end_shape(gt):
    """GT end (positions as the scorer's linestring): last segment length, and the angle between the old (last segment) and new
    (last >= 1 m) end directions."""
    import numpy as np
    c = np.asarray(gt.positions, float)[:, :2]
    last = float(np.linalg.norm(c[-1] - c[-2])) if len(c) > 1 else 0.0
    new = next((c[-1] - p for p in c[-2::-1] if np.linalg.norm(c[-1] - p) >= 1.0), None)
    if last <= 0 or new is None:
        return last, float("nan")
    old = (c[-1] - c[-2]) / last
    new = new / np.linalg.norm(new)
    return last, float(np.degrees(np.arccos(np.clip(old @ new, -1, 1))))


TRANSIENT = ("can't start new thread", "Resource temporarily unavailable")   # box-wide pids limit, set by other jobs


def eval_one(task):
    """eval_try with retries on thread-start failures (the box's pids cgroup is shared with pool jobs of ~940 threads each)."""
    import time
    for k in range(8):
        r = eval_try(task)
        if r["ok"] or not any(t in r["err"] for t in TRANSIENT):
            return r | dict(tries=k + 1)
        time.sleep(5 + 10 * k)
    return r | dict(tries=k + 1)


def eval_try(task):
    """One ASL in the mirror -> metrics.parquet next to it (eval.main.process_asl_file minus video) + GT end shape."""
    asl, cfg_path, usdz_glob, stored_parquet = task
    try:
        import asyncio
        import polars as pl
        from alpasim_utils.artifact import Artifact
        from eval.asl_loader import load_scenario_eval_input_from_asl
        from eval.metadata import get_metadata
        from eval.scenario_evaluator import ScenarioEvaluator
        from eval.schema import EvalConfig
        from omegaconf import OmegaConf
        cfg = _cached(("cfg", cfg_path), lambda: OmegaConf.merge(EvalConfig, OmegaConf.load(cfg_path)))
        arts = _cached(("art", usdz_glob), lambda: Artifact.discover_from_glob(usdz_glob))
        meta = _cached(("meta", cfg_path), lambda: get_metadata(Path(cfg_path).parent))
        inp = asyncio.run(load_scenario_eval_input_from_asl(asl, cfg, arts, meta))
        df = ScenarioEvaluator(cfg).evaluate(inp).metrics_df
        df.write_parquet(Path(asl).parent / "metrics.parquet")
        last, ang = _end_shape(inp.ego_recorded_ground_truth_trajectory)
        same = None
        if stored_parquet:                                  # reproduction check: every metric value at every step
            key = ["name", "timestamps_us"]
            a = pl.read_parquet(stored_parquet).select(key + ["values", "valid"]).sort(key)
            b = df.select(key + ["values", "valid"]).sort(key)
            same = a.shape == b.shape and a.equals(b)
        return dict(asl=asl, ok=True, gt_last_seg_m=last, end_dir_change_deg=ang, parquet_same=same)
    except Exception:
        return dict(asl=asl, ok=False, err=traceback.format_exc()[-1500:])


def agg_one(sim):
    try:
        from eval.aggregation.main import _run_aggregation_core
        from eval.schema import EvalConfig
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(EvalConfig, OmegaConf.load(Path(sim) / "eval-config.yaml"))
        _run_aggregation_core([Path(sim)], Path(sim) / "aggregate", cfg)
        return dict(sim=sim, ok=True)
    except Exception:
        return dict(sim=sim, ok=False, err=traceback.format_exc()[-1500:])


def event_step(parquet):
    """Kept truncation step of a rollout (first step with plain GT distance >= 4 m): (plain, lateral, progress) there, or None."""
    import polars as pl
    w = pl.read_parquet(parquet).filter(pl.col("name").is_in(["dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory", "progress"]))
    w = w.pivot(on="name", index="timestamps_us", values="values").sort("timestamps_us")
    e = w.filter(pl.col("dist_to_gt_trajectory") >= CORRIDOR_M)
    if not len(e):
        return None
    r = e.row(0, named=True)
    return r["dist_to_gt_trajectory"], r["lateral_dist_to_gt_trajectory"], r["progress"]


def n_cpus():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        from jevdrive.common import n_cpus as f
        return f()
    except Exception:
        return len(os.sched_getaffinity(0))


def run_shards(cmd, items, key_in, key_out, wdir, workers, attempts=3):
    """Run items through `scorer_v2.py <cmd> <in.json> <out.jsonl>` in `workers` subprocesses (contiguous blocks); returns {key: result}."""
    import subprocess
    if wdir.exists():
        shutil.rmtree(wdir)                                 # our own scratch from an earlier run
    wdir.mkdir(parents=True)
    res = {}
    for att in range(attempts):
        todo = [x for x in items if key_in(x) not in res]
        if not todo:
            break
        n = min(workers, len(todo))
        procs = []
        for k in range(n):
            blk = todo[k * len(todo) // n:(k + 1) * len(todo) // n]
            fi, fo = wdir / f"a{att}_{k:03d}.json", wdir / f"a{att}_{k:03d}.jsonl"
            fi.write_text(json.dumps(blk))
            procs.append((subprocess.Popen([sys.executable, __file__, cmd, str(fi), str(fo)], stdout=subprocess.DEVNULL,
                                           stderr=open(wdir / f"a{att}_{k:03d}.err", "w")), fo))
        print(f"{cmd}: attempt {att + 1}, {len(todo)} items in {n} shards", flush=True)
        for pr, fo in procs:
            rc = pr.wait()
            if rc:
                print(f"{cmd}: shard {fo.name} exited {rc}", flush=True)
            for ln in (fo.read_text().splitlines() if fo.exists() else []):
                r = json.loads(ln)
                if r["ok"] or att == attempts - 1:
                    res[key_out(r)] = r
                elif not r["ok"]:
                    print(f"{cmd} ERROR {key_out(r)} {r['err'][-300:]}", flush=True)
        print(f"{cmd}: {len(res)} / {len(items)} done", flush=True)
    return res


def cmd_endshape(a):
    """GT end shape of every scene from its usdz (the runtime's GT is `artifact.rig.trajectory`) -> <root>/end_shape.csv."""
    import yaml
    import pandas as pd
    from alpasim_utils.artifact import Artifact
    globs = set()
    for lab in READS:
        for ch in chunks(lab):
            wz = yaml.safe_load((ch / "sim/wizard-config.yaml").read_text())["scenes"]
            globs.add(str(Path(wz["scene_cache"]) / wz["sceneset_path"] / "*.usdz"))
    paths = {os.path.realpath(p) for g in globs for p in glob.glob(g)}
    rows = []
    for p in sorted(paths):
        art = Artifact(p)
        last, ang = _end_shape(art.rig.trajectory)
        rows.append(dict(scene=art.scene_id, gt_last_seg_m=last, end_dir_change_deg=ang))
    pd.DataFrame(rows).drop_duplicates("scene").to_csv(Path(a.root) / "end_shape.csv", index=False)
    print(f"{len(rows)} usdz -> {Path(a.root) / 'end_shape.csv'}")


def cmd_shard(a):
    fn = {"eval-shard": eval_one, "agg-shard": agg_one}[a.cmd]
    with open(a.out, "a") as f:
        for x in json.loads(Path(a.inp).read_text()):
            f.write(json.dumps(fn(tuple(x) if isinstance(x, list) else x), default=float) + "\n")
            f.flush()


def cmd_score(a):
    import inspect
    import eval.scorers.ground_truth as gt
    fixed = "1 m of displacement" in inspect.getsource(gt)
    assert fixed == (a.variant == "new"), f"variant {a.variant} but {gt.__file__} fixed={fixed}"
    print(f"eval code: {gt.__file__} (fixed={fixed})", flush=True)
    root = Path(a.root) / a.variant
    labels = a.reads.split(",") if a.reads else list(READS)
    tasks, sims, meta = [], [], {}
    for lab in labels:
        for ch in chunks(lab):
            src = ch / "sim"
            dst = root / lab / ch.name / "sim"
            if dst.exists():
                shutil.rmtree(dst)                          # only our own mirror
            (dst / "rollouts").mkdir(parents=True)
            for f in ("eval-config.yaml", "run_metadata.yaml"):
                shutil.copy2(src / f, dst / f)
            import yaml
            wz = yaml.safe_load((src / "wizard-config.yaml").read_text())["scenes"]
            usdz = str(Path(wz["scene_cache"]) / wz["sceneset_path"] / "**" / "*.usdz")
            for rd in sorted(src.glob("rollouts/*/*")):
                if not (rd / "_complete").exists() or not (rd / "metrics.parquet").exists():
                    continue
                o = dst / "rollouts" / rd.parent.name / rd.name
                o.mkdir(parents=True)
                (o / "_complete").touch()
                has_asl = (rd / "rollout.asl").exists()
                meta[str(o)] = dict(read=lab, chunk=ch.name, scene=rd.parent.name, rollout=rd.name, has_asl=has_asl, stored=str(rd))
                if has_asl and (a.limit is None or len(tasks) < a.limit):
                    (o / "rollout.asl").symlink_to(rd / "rollout.asl")
                    tasks.append((str(o / "rollout.asl"), str(dst / "eval-config.yaml"), usdz, str(rd / "metrics.parquet") if a.variant == "old" else ""))
                    meta[str(o)]["evaluated"] = True
                else:
                    (o / "metrics.parquet").symlink_to(rd / "metrics.parquet")
            sims.append(str(dst))
    workers = a.workers or n_cpus()
    print(f"{len(labels)} reads, {len(sims)} chunks, {len(meta)} rollouts, {len(tasks)} ASLs to evaluate, {workers} workers", flush=True)
    # Shards are plain subprocesses writing JSON lines (no multiprocessing queues: a ProcessPoolExecutor hung twice on the box,
    # 10-23); a shard that dies only loses its unfinished items, which the next attempt reruns.
    ev = run_shards("eval-shard", [list(t) for t in sorted(tasks, key=lambda t: t[2])], lambda t: t[0], lambda r: r["asl"], root / "shards_eval", workers)
    bad = [r for r in ev.values() if not r["ok"]] + [dict(asl=t[0], ok=False, err="never evaluated") for t in tasks if t[0] not in ev]
    if bad:
        (root / "ERROR").write_text(json.dumps(bad, indent=1))
        sys.exit(f"{len(bad)} ASL evaluations failed; see {root / 'ERROR'}")
    ev = list(ev.values())
    ag = list(run_shards("agg-shard", sims, lambda x: x, lambda r: r["sim"], root / "shards_agg", workers).values())
    ag += [dict(sim=x, ok=False, err="never aggregated") for x in sims if x not in {r["sim"] for r in ag}]
    bad = [r for r in ag if not r["ok"]]
    if bad:
        (root / "ERROR").write_text(json.dumps(bad, indent=1))
        sys.exit(f"{len(bad)} aggregations failed; see {root / 'ERROR'}")
    evd = {str(Path(r["asl"]).parent): r for r in ev}
    rows = []
    for sim in sims:
        summ = {r["rollout_id"]: r for r in json.loads((Path(sim) / "aggregate/results-summary.json").read_text())["rollouts"]}
        for o, m in meta.items():
            if not o.startswith(sim + "/"):
                continue
            s = summ.get(m["rollout"])
            e = evd.get(o, {})
            es = event_step(Path(o) / "metrics.parquet")
            rows.append(m | dict(score=None if s is None else s["score"], failure_reason=None if s is None else s.get("failure_reason"),
                                 lateral=None if s is None else s["metrics"].get("lateral_dist_to_gt_trajectory"),
                                 ev_plain=es and es[0], ev_lateral=es and es[1], ev_progress=es and es[2],
                                 gt_last_seg_m=e.get("gt_last_seg_m"), end_dir_change_deg=e.get("end_dir_change_deg"),
                                 parquet_same=e.get("parquet_same")))
    import pandas as pd
    pd.DataFrame(rows).drop(columns=["stored"]).assign(stored=[m["stored"] for m in rows]).to_csv(root / "rollouts.csv", index=False)
    (root / "DONE").write_text(f"{len(rows)} rollouts, {len(tasks)} evaluated\n")
    print(f"DONE {root / 'rollouts.csv'}", flush=True)


# ---------------------------------------------------------------- report (repo venv)
def boot(d, n=10000, seed=0):
    import numpy as np
    g = np.random.default_rng(seed)
    m = d[g.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def nfr(x):
    return "" if x is None or x != x else str(x)


def stored_scores(label):
    out = {}
    for ch in chunks(label):
        for r in json.loads((ch / "sim/aggregate/results-summary.json").read_text())["rollouts"]:
            out[r["clipgt_id"]] = (r["rollout_id"], r["score"], r.get("failure_reason"))
    return out


def cmd_report(a):
    import numpy as np
    import pandas as pd
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    O = pd.read_csv(root / "old/rollouts.csv")
    N = pd.read_csv(root / "new/rollouts.csv")
    labels = [lab for lab in READS if lab in set(N.read)]
    # ---- reproduction gate: old variant == stored, every rollout
    rep, rpc_failed = [], []
    for lab in labels:
        st = stored_scores(lab)
        o = O[O.read == lab].drop_duplicates("scene", keep="last").set_index("scene")
        for sc, (rid, s, fr) in st.items():
            if sc not in o.index:
                if "RpcError" in nfr(fr) or any(t in nfr(fr) for t in TRANSIENT):   # failed rollout (RPC, or the runtime hit the pids cap),
                                                            # no metrics: nothing to rescore (excluded like the facts' rpc rows)
                    rpc_failed.append(dict(read=lab, scene=sc))
                else:
                    rep.append(dict(read=lab, scene=sc, issue="missing in old rescore", stored=s))
            elif o.at[sc, "rollout"] != rid or not np.isclose(o.at[sc, "score"], s, rtol=0, atol=1e-12) or nfr(o.at[sc, "failure_reason"]) != nfr(fr):
                rep.append(dict(read=lab, scene=sc, issue="differs", stored=s, old=o.at[sc, "score"], fr=fr, fr_old=o.at[sc, "failure_reason"]))
    ev = O[O.evaluated == True]  # noqa: E712
    pq_bad = int((ev.parquet_same != True).sum())  # noqa: E712
    gate = dict(rollouts=int(len(O)), asl_evaluated=int(len(ev)), parquet_identical=int((ev.parquet_same == True).sum()),  # noqa: E712
                parquet_differs=pq_bad, scene_score_mismatch=len(rep),
                rpc_failed_rollouts=len(rpc_failed), mismatches=rep[:50], rpc_failed=rpc_failed)
    # 6 / 2315 re-evaluated parquets differ at one step by < 2e-6 m (float noise, checked 10-23); the gate is the scene score
    (out / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    if rep:
        (out / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        sys.exit(f"reproduction gate FAILED: {len(rep)} scene mismatches, {pq_bad} parquet differences; {out / 'gate.json'}")
    # ---- per scene table: old / new score per read
    def per_scene(D):
        D = D.drop_duplicates(["read", "scene"], keep="last")
        return D.pivot(index="scene", columns="read", values="score"), D.pivot(index="scene", columns="read", values="failure_reason")
    So, Fo = per_scene(O)
    Sn, Fn = per_scene(N)
    prim = {}
    for lab in labels:
        f = READS[lab][1]
        if f and (FACTS / f"{f}.csv").exists():
            x = pd.read_csv(FACTS / f"{f}.csv").set_index("scene")
            prim[lab] = x
    rows, flips = [], []
    for lab in labels:
        so, sn = So[lab].dropna(), Sn[lab].dropna()
        sc = so.index.intersection(sn.index)
        so, sn = so[sc], sn[sc]
        d = sn - so
        rpc = prim[lab].rpc.reindex(sc).fillna(False).astype(bool) if lab in prim else pd.Series(False, sc)
        risk = N[(N.read == lab) & (N.score > 0) & (~N.evaluated.fillna(False).astype(bool)) & N.ev_plain.notna()]
        zo = O[(O.read == lab)].drop_duplicates("scene", keep="last")
        zo = zo[zo.score == 0]
        rows.append(dict(read=lab, partial=bool((~zo.has_asl.astype(bool)).any()), zeros_with_asl=f"{int(zo.has_asl.sum())}/{len(zo)}", scenes=len(sc), old_mean=so.mean(), new_mean=sn.mean(), new_minus_old=d.mean(), old_zeros=int((so == 0).sum()),
                         new_zeros=int((sn == 0).sum()), flipped_up=int(((so == 0) & (sn > 0)).sum()), worse=int((d < 0).sum()),
                         changed=int((d != 0).sum()), unverified_nonzero_with_trunc=len(risk), rpc_scenes=int(rpc.sum())))
        for s in sc[(d != 0).to_numpy()]:
            p = prim[lab].loc[s] if lab in prim and s in prim[lab].index else None
            flips.append(dict(read=lab, scene=s, old=so[s], new=sn[s], fr_old=Fo.at[s, lab], fr_new=Fn.at[s, lab],
                              primary_old=None if p is None else p["primary"], fclass_old=None if p is None else p["fclass"]))
    R = pd.DataFrame(rows)
    F = pd.DataFrame(flips)
    R.round(4).to_csv(out / "reads.csv", index=False)
    F.round(4).to_csv(out / "changed_scenes.csv", index=False)
    pd.concat({"old": So, "new": Sn}, axis=1).round(4).to_csv(out / "per_scene.csv")
    # ---- unverifiable rollouts: nonzero, no ASL, a truncation event (GT end shape from any ASL of the scene)
    shape = pd.concat([O, N])[lambda x: x.gt_last_seg_m.notna()].groupby("scene")[["gt_last_seg_m", "end_dir_change_deg"]].first()
    risk = N[(N.score > 0) & (~N.evaluated.fillna(False).astype(bool)) & N.ev_plain.notna()][["read", "scene", "score", "ev_plain", "ev_lateral", "ev_progress"]]
    risk = risk.join(shape, on="scene")
    risk.round(4).to_csv(out / "unverified.csv", index=False)
    # ---- pairs on the new scorer (common scenes, no RPC error in any read)
    def common(labs):
        c = set.intersection(*[set(Sn[l].dropna().index) for l in labs])
        for l in labs:
            if l in prim:
                c -= set(prim[l].index[prim[l].rpc.fillna(False).astype(bool)])
        return sorted(c)
    pairs = {}
    seeds = ("s0", "s1", "s2")
    rec = lambda r, m, S, c: np.mean([S[f"{r}-L18-{sd}-{m}"].reindex(c).to_numpy(float) for sd in seeds], 0)  # noqa: E731
    labs3 = [f"{r}-L18-{sd}-{m}" for r in ("R6DAG", "R8K") for sd in seeds for m in ("ag", "agH")] + ["NTV0-L18-s0-ag", "NTV0-L18-s0-agH"]
    c = common(labs3) if set(labs3) <= set(labels) else []
    for S, tag in ((So, "old"), (Sn, "new")) if c else ():   # a root with other reads only (e.g. the anchors): gate and tables only
        n_off, n_on = S["NTV0-L18-s0-ag"].reindex(c).to_numpy(float), S["NTV0-L18-s0-agH"].reindex(c).to_numpy(float)
        for r in ("R6DAG", "R8K"):
            off, on = rec(r, "ag", S, c), rec(r, "agH", S, c)
            pairs[f"{tag}: {r} 3-seed - NTV0-s0"] = dict(OFF=boot(off - n_off), ON=boot(on - n_on), AVG=boot((off + on) / 2 - (n_off + n_on) / 2),
                                                         means=dict(OFF=float(off.mean()), ON=float(on.mean()), AVG=float(((off + on) / 2).mean())))
        g = {q: rec("R8K", m, S, c) - rec("R6DAG", m, S, c) for q, m in (("OFF", "ag"), ("ON", "agH"))}
        pairs[f"{tag}: R8K - R6DAG 3-seed"] = dict(OFF=boot(g["OFF"]), ON=boot(g["ON"]), AVG=boot((g["OFF"] + g["ON"]) / 2))
        pairs[f"{tag}: NTV0-s0"] = dict(means=dict(OFF=float(n_off.mean()), ON=float(n_on.mean()), AVG=float(((n_off + n_on) / 2).mean())))
        for sd in seeds:
            for r in ("R6DAG", "R8K"):
                pairs[f"{tag}: {r}-{sd} - NTV0-s0"] = {q: boot(S[f"{r}-L18-{sd}-{m}"].reindex(c).to_numpy(float) - S[f"NTV0-L18-s0-{m}"].reindex(c).to_numpy(float))
                                                    for q, m in (("OFF", "ag"), ("ON", "agH"))}
    cg = common(["NTV0-L18-s0-ag", "RSG-L18-s0-ag"]) if {"NTV0-L18-s0-ag", "RSG-L18-s0-ag"} <= set(labels) else []
    for S, tag in ((So, "old"), (Sn, "new")) if cg else ():
        pairs[f"{tag}: NTV0-s0 - RSG-s0 (ACC + GAP, OFF)"] = boot(S["NTV0-L18-s0-ag"].reindex(cg).to_numpy(float) - S["RSG-L18-s0-ag"].reindex(cg).to_numpy(float)) + (
            float(S["NTV0-L18-s0-ag"].reindex(cg).mean()), float(S["RSG-L18-s0-ag"].reindex(cg).mean()))
    # Can a nonzero rollout without an ASL get worse? Only at its kept truncation step (plain GT distance first >= 4 m), and only if
    # the new lateral there reaches 4 m while the old one was < 4 m. Rate of that crossing over the re-evaluated rollouts:
    J = O.merge(N, on=["read", "scene", "rollout"], suffixes=("_o", "_n"))
    x = J[J.evaluated_o.fillna(False).astype(bool)].dropna(subset=["ev_lateral_o"])
    cross = dict(evaluated_with_event_old_below_4m=int((x.ev_lateral_o < CORRIDOR_M).sum()),
                 of_which_new_at_or_above_4m=int(((x.ev_lateral_o < CORRIDOR_M) & (x.ev_lateral_n >= CORRIDOR_M)).sum()),
                 evaluated_max_lateral_up=int((J.lateral_n > J.lateral_o + 1e-6)[J.evaluated_o.fillna(False).astype(bool)].sum()),
                 unverified_nonzero_with_event=int(len(risk)))
    res = dict(crossing=cross, gate={k: v for k, v in gate.items() if k not in ("mismatches", "rpc_failed")}, common_scenes=len(c), common_scenes_rsg=len(cg), pairs=pairs)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["gate"]))
    print(R.round(4).to_string())
    print(json.dumps(pairs, indent=0)[:6000])
    print("changed classes:", F.groupby(["fr_old", "fr_new"], dropna=False).size().to_string() if len(F) else "none")
    print("unverified:", len(risk), "| with jittery end (dir change > 1 deg):", int((risk.end_dir_change_deg > 1).sum()),
          "| end shape unknown:", int(risk.end_dir_change_deg.isna().sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("score")
    q.add_argument("--variant", choices=("old", "new"), required=True)
    q.add_argument("--root", required=True)
    q.add_argument("--reads", default="")
    q.add_argument("--limit", type=int, default=None, help="evaluate only the first N ASLs (smoke)")
    q.add_argument("--workers", type=int, default=0)
    q = s.add_parser("endshape")
    q.add_argument("--root", required=True)
    for c in ("eval-shard", "agg-shard"):
        q = s.add_parser(c)
        q.add_argument("inp")
        q.add_argument("out")
    q = s.add_parser("report")
    q.add_argument("--root", required=True)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    {"score": cmd_score, "report": cmd_report, "eval-shard": cmd_shard, "agg-shard": cmd_shard, "endshape": cmd_endshape}[a.cmd](a)


if __name__ == "__main__":
    main()
