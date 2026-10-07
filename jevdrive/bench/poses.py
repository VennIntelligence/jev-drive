"""Pose scoring: per-token NAVSIM devkit pdm_score of arbitrary pose arrays plus two DAC diagnostics, as a pool primitive.

    python -m jevdrive.bench score-poses --poses f.npz [--keys a b] [--tokens t.txt] --out o.csv [--wait]

f.npz: `tokens` (N,) and one or more (N, 8, 3) pose arrays (rear axle at t0, 0.5 .. 4 s); every key is scored on the selected
tokens with the v2 navtest metric cache and run_pdm_score.py's simulator / scorer / reactive (IDM) traffic. Per (key, token):
the v2 sub-scores, `score` = the per-token EPDMS without extended comfort (EC needs the neighbouring frame), `raw_out` / `raw_depth`
(the raw plan, linearly interpolated to 0.1 s with the ego footprint, leaves the scorer's drivable polygons: no tracker) and
`lqr_out` / `out_depth` (the same on the LQR-simulated states; depth = largest distance of a footprint corner outside, m).
This is experiments/op_probe/scripts/opb_score.py moved here unchanged in its arithmetic (that script is now a thin wrapper).

No sub-score needs a neighbouring frame, so the unit is the token: the run is a claim queue of token chunks
(<run>/claims, O_EXCL files, mtime heartbeat, stale after 300 s) pulled by K concurrent pool jobs of C cores each. K and C come
from the box (cgroup quota / pool CPU budget: K = budget // C, fewer for small sets); jobs the pool admits late find less work,
so a busy box degrades to fewer jobs instead of waiting. Inside a job the devkit objects are built once and forked into one
worker per granted core (sched_getaffinity, i.e. the pool's taskset), each token runs all keys (the metric cache is loaded once
per token). The collect job checks every (key, token) is present exactly once and writes the CSV in token-then-key order.

Run dir $DATA_DIR/runs/bench/poses/<poses stem>-<hash of poses bytes, keys, tokens>/: config.json, tokens.txt, claims/,
chunks/c<j>.pkl (rows + per-token cost), workers/w<i>.DONE, score.csv (= --out), summary.json, STATUS, DONE.
"""
from __future__ import annotations

import glob
import hashlib
import json
import lzma
import os
import pickle
import shutil
import time
from pathlib import Path

from . import runner as R
from .models import data_dir

SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort"]
COLUMNS = ["key", "token", *SUBS, "score", "raw_out", "raw_depth", "lqr_out", "out_depth"]
CORES_PER_JOB = 12                    # pool job size: small enough to backfill a busy box (as the navtest log shards)
CHUNK = 24                            # tokens per claim
MIN_TOKENS_PER_WORKER = 16            # fewer jobs for small token sets
RAM_BASE_GB, RAM_PER_WORKER_GB = 4.0, 1.0
STALE_S = 300.0
_W = {}


def mcache_dir() -> Path:
    return data_dir() / "runs/navsim/metric_cache/v2_navtest"


def devkit_env(force_threads: bool = False) -> dict:
    """Env of the devkit (navsim2): map / data roots, OPENBLAS_CORETYPE=Haswell (the box's correctness setting), one BLAS thread."""
    D = data_dir()
    e = dict(NUPLAN_MAP_VERSION="nuplan-maps-v1.0", NUPLAN_MAPS_ROOT=str(D / "datasets/navsim/maps"),
             OPENSCENE_DATA_ROOT=str(D / "datasets/navsim"), NAVSIM_EXP_ROOT=str(D / "runs/navsim/eval"),
             NAVSIM_DEVKIT_ROOT=str(D / "third_party/navsim"), OPENBLAS_CORETYPE="Haswell")
    th = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return e | th if force_threads else e | {k: os.environ.get(k, v) for k, v in th.items()}


def granted_cores() -> int:
    """Cores this process may run on: the pool's taskset (sched_getaffinity), not the host or the cgroup quota."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                                          # macOS
        return os.cpu_count() or 1


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- devkit scoring (envs/navsim2)
def init(poses_file, keys) -> None:
    """Devkit objects + poses + metric-cache index into _W, once per process (worker pools fork after this)."""
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    import numpy as np
    GlobalHydra.instance().clear()
    with initialize_config_module("navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose("default_run_pdm_score", overrides=["train_test_split=navtest", f"metric_cache_path={mcache_dir()}",
                                                          "experiment_name=op_probe"])
    sim, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    z = np.load(poses_file)
    _W.update(sim=sim, scorer=scorer, policy=instantiate(cfg.traffic_agents_policy.reactive, sim.proposal_sampling),
              samp=sim.proposal_sampling, P={k: z[k] for k in keys}, row={t: i for i, t in enumerate(z["tokens"].tolist())},
              cp={Path(p).parent.name: p for p in glob.glob(str(mcache_dir() / "*/*/*/metric_cache.pkl"))})


def _dense(p8):
    import numpy as np
    T_POSE, T_DENSE = np.arange(1, 9) * 0.5, np.arange(0, 41) * 0.1
    P = np.vstack([[0, 0, 0], p8])
    t = np.r_[0, T_POSE]
    h = np.unwrap(P[:, 2])
    return np.stack([np.interp(T_DENSE, t, P[:, 0]), np.interp(T_DENSE, t, P[:, 1]), np.interp(T_DENSE, t, h)], -1)


def _corners(mc, states):
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    return state_array_to_coords_array(states[None], mc.ego_state.car_footprint.vehicle_parameters)[0]   # (41, 5, 2), last = centre


def _out_depth(mc, cor, idc, area):
    import shapely
    ins = mc.drivable_area_map.points_in_polygons(cor[None, :, :-1, :])[idc].any(0)[0]     # (41, 4)
    if ins.all():
        return False, 0.0
    pts = shapely.points(cor[:, :-1][~ins])
    return True, float(shapely.distance(area(), pts).max())


def work_prof(token):
    """(rows of every key for one token, cost: load / pdm / diag / union seconds, worker max RSS GB)."""
    import resource
    import numpy as np
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from shapely.ops import unary_union
    W = _W
    t0 = time.perf_counter()
    with lzma.open(W["cp"][token], "rb") as f:
        mc = pickle.load(f)
    am = mc.drivable_area_map
    idc = am.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    prof = dict(load=time.perf_counter() - t0, pdm=0.0, diag=0.0, union=0.0)
    memo = []

    def area():                                                     # union of the drivable polygons, only when a corner is out
        if not memo:
            u0 = time.perf_counter()
            memo.append(unary_union([am._geometries[k] for k in idc]))
            prof["union"] += time.perf_counter() - u0
        return memo[0]
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    out = []
    for k, P in W["P"].items():
        p8 = np.asarray(P[W["row"][token]], np.float64)
        t1 = time.perf_counter()
        row, st = pdm_score(metric_cache=mc, model_trajectory=Trajectory(p8), future_sampling=W["samp"], simulator=W["sim"],
                            scorer=W["scorer"], traffic_agents_policy=W["policy"])
        t2 = time.perf_counter()
        r = row.iloc[0] if hasattr(row, "iloc") else row
        res = {"key": k, "token": token, **{m: float(r[m]) for m in SUBS}}
        X = np.array([res[m] for m in SUBS])
        res["score"] = float(np.prod(X[:4]) * (5 * X[4] + 5 * X[5] + 2 * X[6] + 2 * X[7]) / 14)
        d = _dense(p8)                                                     # raw plan, ego frame -> global states
        g = np.zeros((41, st.shape[-1]))
        g[:, 0], g[:, 1], g[:, 2] = o[0] + c * d[:, 0] - s * d[:, 1], o[1] + s * d[:, 0] + c * d[:, 1], o[2] + d[:, 2]
        res["raw_out"], res["raw_depth"] = _out_depth(mc, _corners(mc, g), idc, area)
        lq = _out_depth(mc, _corners(mc, np.asarray(st)), idc, area)
        res["lqr_out"], res["out_depth"] = lq
        out.append(res)
        prof["pdm"] += t2 - t1
        prof["diag"] += time.perf_counter() - t2
    prof["diag"] -= prof["union"]
    prof["rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    return out, prof


def work(token):
    return work_prof(token)[0]


def score_iter(tokens, procs: int):
    """Yield (token, rows, cost) as workers finish; `init` must have run. One forked worker per core, a bounded in-flight window."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    it = iter(tokens)
    with ProcessPoolExecutor(procs, mp_context=mp.get_context("fork")) as ex:
        live = {}
        for t in it:
            live[ex.submit(work_prof, t)] = t
            if len(live) >= 2 * procs:
                break
        while live:
            done, _ = wait(live, return_when=FIRST_COMPLETED)
            for f in done:
                t = live.pop(f)
                rows, prof = f.result()
                yield t, rows, prof
                nxt = next(it, None)
                if nxt is not None:
                    live[ex.submit(work_prof, nxt)] = nxt


def frame(rows, tokens, keys):
    """DataFrame in token-then-key order with opb_score.py's columns; every (key, token) exactly once or it raises."""
    import pandas as pd
    df = pd.DataFrame(rows, columns=COLUMNS)
    dup = df.duplicated(["key", "token"])
    if dup.any():
        raise RuntimeError(f"{int(dup.sum())} duplicate (key, token) rows, e.g. {df[dup].iloc[0][['key', 'token']].tolist()}")
    want = len(tokens) * len(keys)
    if len(df) != want:
        have = set(df.token)
        miss = [t for t in tokens if t not in have]
        raise RuntimeError(f"{want - len(df)} of {want} (key, token) rows missing ({len(miss)} tokens absent, e.g. {miss[:3]})")
    ti, ki = {t: i for i, t in enumerate(tokens)}, {k: i for i, k in enumerate(keys)}
    order = sorted(range(len(df)), key=lambda i: (ti[df.token.iat[i]], ki[df.key.iat[i]]))
    return df.iloc[order].reset_index(drop=True)


def write_csv(df, out) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)


def score_local(poses, keys, tokens, out, procs: int = 0, status=print):
    """In-process scoring on this process's cores (opb_score.py; a leased pool job). Returns (DataFrame, cost summary)."""
    procs = procs or granted_cores()
    t0 = time.time()
    init(poses, keys)
    t_init = time.time() - t0
    rows, costs = [], []
    for n, (_, r, p) in enumerate(score_iter(tokens, procs), 1):
        rows.extend(r)
        costs.append(p)
        if n % 500 == 0:
            status(f"{n}/{len(tokens)} tokens, {time.time() - t0:.0f} s")
    df = frame(rows, tokens, keys)
    write_csv(df, out)
    return df, dict(cost_summary(costs), procs=procs, init_s=t_init, wall_s=time.time() - t0)


def cost_summary(costs) -> dict:
    if not costs:
        return {}
    n = len(costs)
    s = {f"{k}_s_per_token": sum(c[k] for c in costs) / n for k in ("load", "pdm", "diag", "union")}
    return s | dict(n_tokens=n, union_frac=sum(c["union"] > 0 for c in costs) / n, worker_rss_gb_max=max(c["rss_gb"] for c in costs))


# ---------------------------------------------------------------- validation (any env with numpy)
def read_tokens(path) -> list:
    return [t.strip() for t in Path(path).read_text().splitlines() if t.strip()]


def check_inputs(poses, keys=(), tokens=None, mcache=True) -> tuple:
    """(keys, tokens) after loud checks: keys are (N, 8, 3) arrays, tokens unique, all in the poses file and the metric cache."""
    import numpy as np
    z = np.load(poses)
    if "tokens" not in z.files:
        raise ValueError(f"{poses}: no `tokens` array")
    ptok = z["tokens"].tolist()
    if len(set(ptok)) != len(ptok):
        raise ValueError(f"{poses}: {len(ptok) - len(set(ptok))} duplicate tokens in the poses file")
    keys = list(keys) or [k for k in z.files if z[k].ndim == 3 and z[k].shape[1:] == (8, 3)]
    bad = [k for k in keys if k not in z.files or z[k].shape != (len(ptok), 8, 3)]
    if not keys or bad:
        raise ValueError(f"{poses}: keys {bad or keys} are not ({len(ptok)}, 8, 3) pose arrays (has {z.files})")
    tokens = list(tokens) if tokens is not None else ptok
    if not tokens:
        raise ValueError("no tokens to score")
    if len(set(tokens)) != len(tokens):
        seen, dup = set(), []
        for t in tokens:
            (dup.append(t) if t in seen else seen.add(t))
        raise ValueError(f"{len(dup)} duplicate tokens requested, e.g. {dup[:3]}")
    have = set(ptok)
    miss = [t for t in tokens if t not in have]
    if miss:
        raise ValueError(f"{len(miss)} requested tokens are not in {poses}, e.g. {miss[:3]}")
    if mcache:
        mc = {Path(p).parent.name for p in glob.glob(str(mcache_dir() / "*/*/*/metric_cache.pkl"))}
        miss = [t for t in tokens if t not in mc]
        if miss:
            raise ValueError(f"{len(miss)} tokens have no v2 navtest metric cache ({mcache_dir()}), e.g. {miss[:3]}")
    return keys, tokens


# ---------------------------------------------------------------- the pool run
def run_key(poses, keys, tokens) -> str:
    h = hashlib.sha256(json.dumps([file_sha(poses), keys, tokens]).encode()).hexdigest()[:12]
    return f"{Path(poses).stem}-{h}"


def plan_jobs(n_tokens: int, cpu: int = 0, jobs: int = 0, budget: float = 0) -> tuple:
    """(jobs K, cores per job C) from the pool's CPU budget (the cgroup quota): K x C fills the box, fewer jobs for few tokens."""
    c = cpu or CORES_PER_JOB
    budget = budget or pool_budget()
    k = jobs or max(1, min(int(budget // c), -(-n_tokens // (c * MIN_TOKENS_PER_WORKER))))
    return k, c


def pool_budget() -> float:
    """The pool's CPU budget (cores x overcommit, from its status.json), else the box probe's cores."""
    try:
        from ..cl import pool as P
        b = (P.snapshot()[1].get("cpu") or {}).get("budget")
        if b:
            return float(b)
    except Exception:                                               # noqa: BLE001 - off the box
        pass
    return R_cores()


def R_cores() -> float:
    from .navsim import R_cores as rc
    return rc()


def stages(poses, out, keys=(), tokens=None, cpu: int = 0, jobs: int = 0, chunk: int = CHUNK, root: Path = None) -> tuple:
    """(run dir, stages): K worker jobs on the shared chunk queue, then collect. Writes config.json / tokens.txt."""
    poses = str(Path(poses).resolve())
    keys, tokens = check_inputs(poses, keys, tokens)
    d = (root or R.bench_root("poses")) / run_key(poses, keys, tokens)
    d.mkdir(parents=True, exist_ok=True)
    n_chunks = -(-len(tokens) // chunk)
    k, c = plan_jobs(len(tokens), cpu, jobs)
    cf = d / "config.json"
    old = json.loads(cf.read_text()) if cf.exists() else {}
    if old and old.get("chunk") != chunk:
        chunk, n_chunks = old["chunk"], old["n_chunks"]             # resume: keep the chunking of the existing chunk files
    outs = list(dict.fromkeys(old.get("outs", []) + [str(Path(out).resolve())]))
    cfg = dict(poses=poses, poses_sha=file_sha(poses), keys=keys, n=len(tokens), chunk=chunk, n_chunks=n_chunks, jobs=k, cpu=c,
               outs=outs, t_submit=time.strftime("%F %T"))
    if not (d / "tokens.txt").exists():
        R.atomic_write(d / "tokens.txt", "\n".join(tokens) + "\n")
    R.atomic_write(cf, json.dumps(cfg, indent=1))
    if (d / "DONE").exists():                                       # finished: only the new --out needs a copy
        for o in outs:
            if not Path(o).exists():
                _copy(d / "score.csv", Path(o))
    env = devkit_env(force_threads=True)
    S = []
    left = [j for j in range(n_chunks) if not (d / "chunks" / f"c{j:05d}.pkl").exists()]
    if left:
        k = min(k, len(left))
        for f in [d / "DONE"] + list((d / "workers").glob("w*.DONE")):
            if f.exists():
                f.rename(f.with_name(f"{f.name}.{time.strftime('%Y%m%d-%H%M%S')}"))
        for i in range(k):
            S.append(R.Stage(f"w{i}", R.stage_cmd("navsim2", "poses-score", d, i), done=str(d / "workers" / f"w{i}.DONE"), vram=0.5,
                             cpu=c, ram=RAM_BASE_GB + RAM_PER_WORKER_GB * c, env=env, tries=2))
    S.append(R.Stage("collect", R.stage_cmd("navsim2", "poses-collect", d), done=str(d / "DONE"), vram=0.5, cpu=2, ram=8, env=env,
                     after=[s.name for s in S]))
    return d, S


def submit(poses, out, keys=(), tokens=None, cpu: int = 0, jobs: int = 0, priority: float = 0.0, dry: bool = False,
           owner: str = "bench") -> Path:
    """Queue a pose-scoring run (idempotent: finished chunks are kept, live jobs reused); returns the run dir."""
    d, S = stages(poses, out, keys, tokens, cpu, jobs)
    R.submit(d, f"bn-poses-{d.name}", S, owner=owner, dry=dry, priority=priority)
    if not dry:
        cfg = json.loads((d / "config.json").read_text())
        R.status(d, f"submitted {cfg['n']} tokens x {len(cfg['keys'])} keys: {sum(s.name.startswith('w') for s in S)} jobs x "
                    f"{cfg['cpu']} cores, {cfg['n_chunks']} chunks of {cfg['chunk']}")
    return d


class Chunks:
    """The shared chunk queue: claims/c<j>.claim (O_EXCL; mtime = heartbeat), a chunk is finished when chunks/c<j>.pkl exists."""

    def __init__(self, run_dir: Path, n: int, i: int):
        self.d, self.n, self.i = Path(run_dir), n, i
        (self.d / "claims").mkdir(parents=True, exist_ok=True)
        (self.d / "chunks").mkdir(parents=True, exist_ok=True)
        self.mine, self.beat_t = set(), 0.0
        for f in (self.d / "claims").glob("*.claim"):                 # this worker's claims from an earlier try
            try:
                if json.loads(f.read_text()).get("worker") == i:
                    f.unlink()
            except (OSError, ValueError):
                pass

    def out(self, j: int) -> Path:
        return self.d / "chunks" / f"c{j:05d}.pkl"

    def claim_path(self, j: int) -> Path:
        return self.d / "claims" / f"c{j:05d}.claim"

    def _claim(self, j: int) -> bool:
        f = self.claim_path(j)
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - f.stat().st_mtime > STALE_S:
                    f.rename(f.with_name(f"{f.name}.stale.{time.time():.0f}"))
                    return self._claim(j)
            except OSError:
                pass
            return False
        os.write(fd, json.dumps(dict(worker=self.i, job=os.environ.get("CL_POOL_JOB", ""), pid=os.getpid(), t=time.time())).encode())
        os.close(fd)
        if self.out(j).exists():                                     # finished between the scan and the claim
            f.unlink(missing_ok=True)
            return False
        self.mine.add(j)
        return True

    def next(self):
        for j in range(self.n):
            if not self.out(j).exists() and j not in self.mine and self._claim(j):
                return j
        return None

    def pending(self) -> bool:
        return any(not self.out(j).exists() for j in range(self.n))

    def release(self, j: int) -> None:
        self.mine.discard(j)
        self.claim_path(j).unlink(missing_ok=True)

    def beat(self, every_s: float = 30.0) -> None:
        if time.time() - self.beat_t < every_s:
            return
        self.beat_t = time.time()
        for j in list(self.mine):
            try:
                os.utime(self.claim_path(j))
            except OSError:
                pass


def worker(run_dir, i) -> None:
    """Pool job: claim chunks, score their tokens on every granted core, write each finished chunk atomically."""
    run_dir, i = Path(run_dir), int(i)
    cfg = json.loads((run_dir / "config.json").read_text())
    tokens = read_tokens(run_dir / "tokens.txt")
    q = Chunks(run_dir, cfg["n_chunks"], i)
    wdone = run_dir / "workers" / f"w{i}.DONE"
    t0 = time.time()
    first = q.next()
    stats = dict(chunks=0, tokens=0, procs=granted_cores())
    if first is not None:
        if file_sha(cfg["poses"]) != cfg["poses_sha"]:
            raise RuntimeError(f"{cfg['poses']} changed after submission (sha differs); resubmit to score the new file")
        init(cfg["poses"], cfg["keys"])
        stats["init_s"] = round(time.time() - t0, 2)
        ch = cfg["chunk"]
        tok_chunk, acc = {}, {}

        def gen():                                                  # claims the next chunk only when the workers need tokens
            j = first
            while j is not None:
                acc[j] = dict(rows=[], cost=[], left=len(tokens[j * ch:(j + 1) * ch]))
                for t in tokens[j * ch:(j + 1) * ch]:
                    tok_chunk[t] = j
                    yield t
                j = q.next()
        for t, rows, cost in score_iter(gen(), stats["procs"]):
            j = tok_chunk.pop(t)
            a = acc[j]
            a["rows"].extend(rows)
            a["cost"].append(dict(cost, token=t))
            a["left"] -= 1
            q.beat()
            if a["left"] == 0:
                _atomic_pickle(q.out(j), dict(rows=a["rows"], cost=a["cost"], worker=i, job=os.environ.get("CL_POOL_JOB", "")))
                q.release(j)
                del acc[j]
                stats["chunks"] += 1
                stats["tokens"] += len(a["cost"])
                ndone = sum(q.out(x).exists() for x in range(cfg["n_chunks"]))
                R.status(run_dir, f"w{i}: {ndone} / {cfg['n_chunks']} chunks done ({time.time() - t0:.0f} s, this job {stats['tokens']} tokens)")
        if acc:
            raise RuntimeError(f"w{i}: chunks {sorted(acc)} unfinished")
    R.atomic_write(wdone, json.dumps(dict(t=time.strftime("%F %T"), wall_s=round(time.time() - t0, 1), t0=t0, **stats)) + "\n")


def _atomic_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def collect(run_dir) -> None:
    """Merge the chunks, check completeness, write score.csv and every requested --out, summary.json, DONE."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    tokens = read_tokens(run_dir / "tokens.txt")
    miss = [j for j in range(cfg["n_chunks"]) if not (run_dir / "chunks" / f"c{j:05d}.pkl").exists()]
    if miss:
        raise RuntimeError(f"{len(miss)} of {cfg['n_chunks']} chunks missing, e.g. {miss[:5]} (a worker job failed; resubmit)")
    rows, costs = [], []
    for j in range(cfg["n_chunks"]):
        with open(run_dir / "chunks" / f"c{j:05d}.pkl", "rb") as f:
            c = pickle.load(f)
        rows.extend(c["rows"])
        costs.extend(c["cost"])
    df = frame(rows, tokens, cfg["keys"])
    write_csv(df, run_dir / "score.csv")
    for o in cfg["outs"]:
        _copy(run_dir / "score.csv", Path(o))
    ws = [json.loads(f.read_text()) for f in sorted((run_dir / "workers").glob("w*.DONE"))]
    busy = [w for w in ws if w.get("tokens")]
    t_first = min((w["t0"] for w in busy), default=0)
    t_last = max((w["t0"] + w["wall_s"] for w in busy), default=0)
    g = df.groupby("key")[["drivable_area_compliance", "score", "raw_out", "lqr_out"]].mean()
    summ = dict(n=len(tokens), keys=cfg["keys"], outs=cfg["outs"], jobs_used=len(busy), jobs=cfg["jobs"], cpu=cfg["cpu"],
                wall_s_first_to_last=round(t_last - t_first, 1), init_s=[w.get("init_s") for w in busy],
                cost=cost_summary(costs), means=g.to_dict())
    R.atomic_write(run_dir / "summary.json", json.dumps(summ, indent=1))
    R.status(run_dir, f"done: {len(tokens)} tokens x {len(cfg['keys'])} keys, {len(busy)} jobs, "
                      f"{summ['wall_s_first_to_last']:.0f} s first job start -> last job end")
    R.atomic_write(run_dir / "DONE", json.dumps(dict(t=time.strftime("%F %T"), n=len(tokens), keys=cfg["keys"])) + "\n")
