#!/usr/bin/env python
"""op_parity nt-cache: the v2 NAVSIM metric cache of navtrain (turn tokens first), built shard by shard and restartable.

  nt_cache.py plan                                  token lists + shards (whole logs) -> $O/plan/            (.venv, ~1 min)
  nt_cache.py run --stage turn|rest [--limit N]     submit worker jobs to the pool, wait, collect          (.venv; re-run = resume)
  nt_cache.py control                               rebuild 10 navtest tokens with this pipeline, compare with v2_navtest
  nt_cache.py status                                shards done / left, bytes and core-s per token so far

Cache: $DATA_DIR/runs/navsim/metric_cache/v2_navtrain/<log>/<scene type>/<token>/metric_cache.pkl (the devkit layout, so the
v2 scorer reads it like v2_navtest). Same devkit (`third_party/navsim` = navsim main @0a380a9, env navsim2, OPENBLAS_CORETYPE=Haswell)
and the same `run_metric_caching.py` defaults as navsim_zs_score.sh cache v2; only `train_test_split=navtrain` and the token filter differ.

Shards = whole logs of one stage (turn: the |dyaw| >= 20 deg tokens, rest: the other navtrain tokens). A worker claims a shard
(claims/<shard>.claim, mtime heartbeat), runs the devkit into $O/stage/<shard>/ and, only when it exits 0, moves each token dir
into the final cache (rename) and writes done/<shard>.json atomically. A shard without its done file is redone from scratch;
stale claims (no heartbeat for STALE_S) are taken over. Nothing depends on tmux or a live dispatcher: re-running the same
`run` command after a reboot resubmits workers for the shards that are not done.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from jevdrive.common import data_dir, n_cpus

STALE_S = 300.0
SHARD_TOKENS = 600                      # target tokens per shard (about 24 turn logs / 9 rest logs)
C_JOB = 12                              # cores of one pool job (as the navtest scoring shards)
SPLIT_NAVTRAIN, SPLIT_TURN_DEG = "navsim/navtrain", 20.0


def O() -> Path:
    return data_dir() / "runs/op_parity/nt_cache"


def final_dir(name="v2_navtrain") -> Path:
    return data_dir() / "runs/navsim/metric_cache" / name


def write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- plan
def cmd_plan(a):
    import numpy as np
    from jevdrive.bench import tables as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    D = data_dir()
    with Run("op_parity", "nt_cache/plan", config=vars(a)) as run:
        tr = splits.load(SPLIT_NAVTRAIN)
        run.use_split(tr)
        names, dy = [], []
        for d in sorted(glob.glob(str(D / "runs/op_parity/cache/navtrain_full.s*of12/tab.npz"))):
            z = np.load(d)
            names += z["names"].tolist()
            dy += [T.motion(np.asarray(f, float), float(v))[0] for f, v in zip(z["fut"], z["speed"])]
        names, dy = np.array(names), np.abs(np.array(dy))
        assert tr.mask(names).all() and len(set(names.tolist())) == len(names) == len(tr), "token cache != navtrain split"
        # token -> log from the v1 navtrain cache directory tree (<log>/<scene type>/<token>)
        tlog = {}
        for p in glob.glob(str(D / "runs/navsim/metric_cache/v1_navtrain/*/*/*")):
            q = Path(p)
            tlog[q.name] = q.parent.parent.name
        miss = [t for t in names.tolist() if t not in tlog]
        assert not miss, f"{len(miss)} navtrain tokens not in the v1 cache tree, e.g. {miss[:3]}"
        turn = dy >= SPLIT_TURN_DEG
        out = O() / "plan"
        res = {}
        for stage, m in (("turn", turn), ("rest", ~turn)):
            toks = sorted(names[m].tolist())
            by_log = {}
            for t in toks:
                by_log.setdefault(tlog[t], []).append(t)
            shards, cur, n = [], [], 0
            for lg in sorted(by_log):
                cur.append(lg)
                n += len(by_log[lg])
                if n >= SHARD_TOKENS:
                    shards.append(cur)
                    cur, n = [], 0
            if cur:
                shards.append(cur)
            S = [dict(name=f"{stage}-{i:04d}", logs=ls, tokens=[t for lg in ls for t in by_log[lg]]) for i, ls in enumerate(shards)]
            write(out / f"tokens_{stage}.txt", "\n".join(toks) + "\n")
            write(out / f"shards_{stage}.json", json.dumps(S))
            res[stage] = dict(tokens=len(toks), logs=len(by_log), shards=len(S))
        write(out / "token_log.json", json.dumps({t: tlog[t] for t in names.tolist()}))
        run.summary.update(res)
        run.info("plan: %s", json.dumps(res))


def load_shards(stage):
    return json.loads((O() / f"plan/shards_{stage}.json").read_text())


def select(stage, limit):
    """Shards of a run: all, or `limit` of them spread evenly over the date-sorted logs (pilots)."""
    S = load_shards(stage)
    if not limit or limit >= len(S):
        return S
    return [S[i * len(S) // limit + len(S) // (2 * limit)] for i in range(limit)]


def done_file(name) -> Path:
    return O() / "done" / f"{name}.json"


# ---------------------------------------------------------------- the devkit (envs/navsim2)
def devkit_argv(split, cache, logs, tokens, cores):
    D = data_dir()
    q = lambda xs: "[" + ",".join(f"'{x}'" for x in xs) + "]"            # quoted: a hex token of digits only must stay a string
    syn = [f"synthetic_sensor_path={D}/datasets/navsim/navhard_two_stage/sensor_blobs",
           f"synthetic_scenes_path={D}/datasets/navsim/navhard_two_stage/synthetic_scene_pickles"]
    return [str(D / "envs/navsim2/bin/python"), str(D / "third_party/navsim/navsim/planning/script/run_metric_caching.py"),
            f"train_test_split={split}", f"metric_cache_path={cache}", "worker=ray_distributed_no_torch",
            f"worker.threads_per_node={cores}", f"train_test_split.scene_filter.log_names={q(logs)}",
            f"train_test_split.scene_filter.tokens={q(tokens)}", *syn]


def devkit_env():
    from jevdrive.bench.poses import devkit_env as de
    return {**os.environ, **de(force_threads=True)}


def build_shard(split, name, logs, tokens, cores, stage_root, final):
    """Run the devkit into stage_root/name, then move the token dirs into `final`. Returns the done record (raises on rc != 0)."""
    sd = Path(stage_root) / name
    shutil.rmtree(sd, ignore_errors=True)
    sd.mkdir(parents=True)
    t0 = time.time()
    with open(sd.parent / f"{name}.log", "w") as lf:
        rc = subprocess.run(devkit_argv(split, sd, logs, tokens, cores), env=devkit_env(), stdout=lf, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc:
        raise RuntimeError(f"{name}: devkit rc {rc} (log {sd.parent / (name + '.log')})")
    got = {Path(p).parent.name: Path(p).parent for p in glob.glob(str(sd / "*/*/*/metric_cache.pkl"))}
    want = set(tokens)
    extra = sorted(set(got) - want)
    nbytes = 0
    for t, src in got.items():
        if t not in want:
            continue
        nbytes += sum(f.stat().st_size for f in src.iterdir())
        dst = Path(final) / src.parent.parent.name / src.parent.name / t
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        os.rename(src, dst)
    miss = sorted(want - set(got))
    rec = dict(shard=name, t=time.strftime("%F %T"), tokens=len(want), cached=len(want) - len(miss), missing=miss, extra=len(extra), bytes=nbytes,
               wall_s=round(wall, 1), cores=cores, core_s=round(wall * cores, 1), job=os.environ.get("CL_POOL_JOB", ""))
    shutil.rmtree(sd, ignore_errors=True)
    return rec


class Claim:
    """claims/<shard>.claim, O_EXCL; the mtime is the heartbeat (a daemon thread touches it while the devkit runs)."""

    def __init__(self, name):
        self.f = O() / "claims" / f"{name}.claim"
        self.f.parent.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()

    def take(self) -> bool:
        try:
            fd = os.open(self.f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - self.f.stat().st_mtime > STALE_S:
                    self.f.rename(self.f.with_name(f"{self.f.name}.stale.{time.time():.0f}"))
                    return self.take()
            except OSError:
                pass
            return False
        os.write(fd, json.dumps(dict(pid=os.getpid(), job=os.environ.get("CL_POOL_JOB", ""), t=time.time())).encode())
        os.close(fd)
        threading.Thread(target=self._beat, daemon=True).start()
        return True

    def _beat(self):
        while not self.stop.wait(30):
            try:
                os.utime(self.f)
            except OSError:
                pass

    def release(self):
        self.stop.set()
        self.f.unlink(missing_ok=True)


def cmd_worker(a):
    sel = json.loads(Path(a.sel).read_text())
    cores = len(os.sched_getaffinity(0))
    t0, n = time.time(), 0
    for sh in sel:
        if done_file(sh["name"]).exists():
            continue
        c = Claim(sh["name"])
        if not c.take():
            continue
        try:
            if done_file(sh["name"]).exists():
                continue
            rec = build_shard("navtrain", sh["name"], sh["logs"], sh["tokens"], cores, O() / "stage", final_dir())
            write(done_file(sh["name"]), json.dumps(rec) + "\n")
            n += 1
            write(O() / f"worker_{a.i}.STATUS", f"{time.strftime('%F %T')} w{a.i}: {n} shards, last {sh['name']} {rec['cached']}/{rec['tokens']} tokens "
                  f"{rec['wall_s']:.0f} s on {cores} cores\n")
            print(json.dumps(rec), flush=True)
        finally:
            c.release()
    write(Path(a.done), json.dumps(dict(t=time.strftime("%F %T"), wall_s=round(time.time() - t0, 1), shards=n, cores=cores)) + "\n")


# ---------------------------------------------------------------- run / collect / status
def records(sel):
    return [json.loads(done_file(s["name"]).read_text()) for s in sel if done_file(s["name"]).exists()]


def summarize(sel, recs):
    tok = sum(r["cached"] for r in recs)
    by = sum(r["bytes"] for r in recs)
    cs = sum(r["core_s"] for r in recs)
    return dict(shards=len(sel), done=len(recs), tokens=sum(r["tokens"] for r in recs), cached=tok, missing=sum(len(r["missing"]) for r in recs),
                bytes=by, core_s=round(cs, 1), core_s_per_token=round(cs / max(tok, 1), 3), bytes_per_token=round(by / max(tok, 1)),
                wall_s_sum=round(sum(r["wall_s"] for r in recs), 1))


def cmd_collect(a):
    sel = json.loads(Path(a.sel).read_text())
    recs = records(sel)
    s = summarize(sel, recs)
    write(Path(a.out) / "summary.json", json.dumps(s, indent=1) + "\n")
    if s["done"] != s["shards"]:
        raise RuntimeError(f"{s['shards'] - s['done']} shards not done: {s}")
    write(Path(a.out) / "DONE", json.dumps(s) + "\n")


def cmd_run(a):
    from jevdrive.run import Run
    with Run("op_parity", f"nt_cache/run-{a.stage}", config=vars(a)) as run:
        ok = _run(a, run)
    sys.exit(0 if ok else 1)


def _run(a, run):
    from jevdrive.bench import runner as R
    from jevdrive.bench.poses import pool_budget
    sel = select(a.stage, a.limit)
    tag = f"{a.stage}" + (f"-n{a.limit}" if a.limit else "")
    rd = O() / "runs" / tag
    rd.mkdir(parents=True, exist_ok=True)
    selp = rd / "sel.json"
    write(selp, json.dumps(sel))
    left = [s for s in sel if not done_file(s["name"]).exists()]
    k = a.jobs or max(1, min(len(left), int(pool_budget() // (2 * C_JOB))))
    for f in [rd / "DONE", rd / "ERROR", *(rd / "workers").glob("w*.DONE")]:
        if f.exists():
            f.rename(f.with_name(f"{f.name}.{time.strftime('%Y%m%d-%H%M%S')}"))
    env = {k_: v for k_, v in devkit_env().items() if k_ in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    me = Path(__file__).resolve()
    S = []
    for i in range(min(k, len(left))):
        S.append(R.Stage(f"w{i}", [R.py("jev"), str(me), "worker", "--sel", str(selp), "--i", str(i), "--done", str(rd / "workers" / f"w{i}.DONE")],
                         done=str(rd / "workers" / f"w{i}.DONE"), cpu=C_JOB, ram=24, env=env, tries=2))
    S.append(R.Stage("collect", [R.py("jev"), str(me), "collect", "--sel", str(selp), "--out", str(rd)], done=str(rd / "DONE"), cpu=2, ram=4,
                     after=[s.name for s in S]))
    R.submit(rd, f"ntc-{tag}", S, owner="op_parity", priority=-1.0)
    R.status(rd, f"submitted {len(left)} / {len(sel)} shards on {len(S) - 1} jobs x {C_JOB} cores")
    ok = R.wait([rd], poll_s=60, quiet=True)
    summ = json.loads((rd / "summary.json").read_text()) if (rd / "summary.json").exists() else {}
    run.summary.update(summ)
    run.info("%s", json.dumps(summ))
    if not ok:
        raise RuntimeError(f"nt-cache {tag} not done: see {rd}/STATUS and the pool logs; re-run the same command to resume")
    return ok


def cmd_status(a):
    for stage in ("turn", "rest"):
        sel = load_shards(stage)
        s = summarize(sel, records(sel))
        s["est_total_GB_if_all"] = round(s["bytes_per_token"] * sum(len(x["tokens"]) for x in sel) / 1e9, 1)
        print(stage, json.dumps(s))


# ---------------------------------------------------------------- control: same pipeline on navtest tokens vs the existing v2_navtest cache
def cmd_control(a):
    import lzma
    import pickle
    sys.path.insert(0, str(data_dir() / "third_party/navsim"))
    old = final_dir("v2_navtest")
    toks = sorted(Path(p).parent.name for p in glob.glob(str(old / "*/*/*/metric_cache.pkl")))
    import random
    toks = random.Random(0).sample(toks, a.n)
    lg = {t: Path(glob.glob(str(old / f"*/*/{t}"))[0]).parent.parent.name for t in toks}
    logs = sorted(set(lg.values()))
    new = O() / "control"
    shutil.rmtree(new, ignore_errors=True)
    rec = build_shard("navtest", "ctl", logs, toks, len(os.sched_getaffinity(0)), new / "stage", new / "cache")
    bad = []
    for t in toks:
        a_ = pickle.loads(lzma.decompress(open(glob.glob(str(old / f"*/*/{t}/metric_cache.pkl"))[0], "rb").read()))
        b_ = pickle.loads(lzma.decompress(open(glob.glob(str(new / f"cache/*/*/{t}/metric_cache.pkl"))[0], "rb").read()))
        for k in sorted(vars(a_)):
            if k == "file_path":
                continue
            if pickle.dumps(getattr(a_, k)) != pickle.dumps(getattr(b_, k)):
                bad.append((t, k))
    res = dict(tokens=len(toks), cached=rec["cached"], field_mismatches=len(bad), first=bad[:5], bytes_new=rec["bytes"],
               bytes_old=sum(f.stat().st_size for t in toks for f in Path(glob.glob(str(old / f"*/*/{t}"))[0]).iterdir()))
    write(O() / "control.json", json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    sys.exit(1 if bad or rec["missing"] else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    p = sp.add_parser("run")
    p.add_argument("--stage", required=True, choices=["turn", "rest"])
    p.add_argument("--limit", type=int, default=0, help="pilot: this many shards spread over the logs (0 = all)")
    p.add_argument("--jobs", type=int, default=0, help="worker jobs (default: half of the pool's CPU budget / 12 cores)")
    p = sp.add_parser("worker")
    p.add_argument("--sel", required=True)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--done", required=True)
    p = sp.add_parser("collect")
    p.add_argument("--sel", required=True)
    p.add_argument("--out", required=True)
    sp.add_parser("status")
    p = sp.add_parser("control")
    p.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
