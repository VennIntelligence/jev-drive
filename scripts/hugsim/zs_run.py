#!/usr/bin/env python
"""Batch runner of HUGSIM's official closed_loop.py for the zero-shot exam (todos/2026-09-25-hugsim-exam).

One job = (scenario, agent, controller). Agents: alpamayo / cinque / lebowski (scripts/hugsim/zs_agent.py against a
resident scripts/hugsim_zs_server.py), cv / route (scripts/hugsim/agent_client.py), ltf (the official LTF client).
Controllers run from two private copies of the patched HUGSIM tree, so nobody else's apply / revert of the optional
patch can touch a running job:  official = patches/hugsim/*.patch,  fixed = + optional/lqr-heading-fix.patch.
The patch state of both trees is verified before every job.

    python scripts/hugsim/zs_run.py setup-trees
    python scripts/hugsim/zs_run.py run --out $DATA_DIR/runs/hugsim-exam --agent cinque --controller official \
        --socket $DATA_DIR/runs/hugsim-exam/cinque.sock --scenarios <list.txt> --workers 2 --gpu 0

Writes <out>/<agent>-<controller>/<ad>/<scene>_<mode>/ (closed_loop.py's outputs + zs_steps.jsonl) and appends one
row per finished job to <out>/results.csv (resumable: finished jobs are skipped).
"""
import argparse
import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
DATA = D / "datasets" / "hugsim"
PY = D / "envs" / "hugsim" / "bin" / "python"
TREES = {"official": D / "third_party" / "HUGSIM-zs" / "official", "fixed": D / "third_party" / "HUGSIM-zs" / "fixed"}
FIX = REPO / "patches" / "hugsim" / "optional" / "lqr-heading-fix.patch"
AD = {"alpamayo": "zs", "cinque": "zs", "lebowski": "zs", "cv": "jev", "route": "jev", "ltf": "ltf"}
FIELDS = ["scenario", "dataset", "difficulty", "agent", "controller", "tag", "hdscore", "rc", "nc", "dac", "ttc", "c",
          "pdms", "steps", "end", "wall_s", "rc_code", "finished", "scene", "run_dir"]
END = [("Collision with background", "bg_collision"), ("Collision with foreground", "fg_collision"),
       ("Far from preset trajectory", "off_route"), ("Complete", "complete")]


def sh(*a, **k):
    return subprocess.run(a, check=True, capture_output=True, text=True, **k).stdout


def setup_trees():
    src = D / "third_party" / "HUGSIM"
    for name, dst in TREES.items():
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            sh("git", "clone", "-q", str(src), str(dst))
            sh("git", "-C", str(dst), "checkout", "-q", sh("git", "-C", str(src), "rev-parse", "HEAD").strip())
            for p in sorted((REPO / "patches" / "hugsim").glob("*.patch")):
                sh("git", "-C", str(dst), "apply", str(p))
            if name == "fixed":
                sh("git", "-C", str(dst), "apply", str(FIX))
        check_tree(name)
        print(name, dst, "ok")


def check_tree(name):
    t = str(TREES[name])
    fwd = subprocess.run(["git", "-C", t, "apply", "--check", str(FIX)], capture_output=True).returncode == 0
    rev = subprocess.run(["git", "-C", t, "apply", "-R", "--check", str(FIX)], capture_output=True).returncode == 0
    ok = (fwd and not rev) if name == "official" else (rev and not fwd)
    if not ok:
        raise SystemExit(f"tree {t}: optional LQR patch state wrong for controller '{name}'")


def traffic_map():
    rows = csv.DictReader(open(REPO / "research" / "results" / "hugsim-exam" / "scenarios.csv"))
    return {r["scene"]: ([0, 1] if r["location"].startswith("singapore") else [1, 0]) for r in rows}


def unpack(ds, scene):
    d = DATA / "scenes" / ds / scene
    if (d / "scene.pth").exists():
        return
    with open(DATA / "scenes" / ds / f".{scene}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not (d / "scene.pth").exists():
            zipfile.ZipFile(DATA / "scenes" / ds / f"{scene}.zip").extractall(DATA / "scenes" / ds)


def done_set(results):
    if not results.exists():
        return set()
    return {(r["scenario"], r["tag"]) for r in csv.DictReader(open(results))}


def append(results, row):
    with open(results, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        new = f.tell() == 0
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_job(a, scen, tag_dir, traffic):
    import yaml
    ds = scen.parent.name
    cfg = yaml.safe_load(scen.read_text())
    scene, mode = cfg["scene_name"], cfg["mode"]
    unpack(ds, scene)
    ad = AD[a.agent]
    base = tag_dir / f"base_{ds}.yaml"
    base.write_text(f"realcar_path: {DATA}/3DRealCar\nmodel_base: {DATA}/scenes/{ds}\n"
                    f"zs_path: {REPO}/scripts/hugsim/zs_agent_e2e.sh\njev_path: {REPO}/scripts/hugsim/agent_e2e.sh\n"
                    f"ltf_path: {REPO}/scripts/hugsim/ltf_e2e.sh\noutput_dir: {tag_dir}/\n"
                    f"HD_map:\n  path: {DATA}/nusc_map_cache\n  version: nusc_trainval\n")
    run_dir = tag_dir / ad / f"{scene}_{mode}"
    if run_dir.exists():
        subprocess.run(["rm", "-rf", str(run_dir)])
    run_dir.mkdir(parents=True)
    tree = TREES[a.controller]
    check_tree(a.controller)
    opts = dict(json.loads(a.opts), traffic=traffic)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HUGSIM_ZS_MODEL=a.agent, HUGSIM_ZS_SOCKET=a.socket or "", HUGSIM_ZS_DATASET=ds,
               HUGSIM_ZS_CAMYAML=str(tree / "configs" / "sim" / f"{ds}_camera.yaml"), HUGSIM_ZS_OPTS=json.dumps(opts),
               HUGSIM_POLICY=a.agent if ad == "jev" else "", HUGSIM_SCENE_DIR=str(DATA / "scenes" / ds / scene))
    cmd = [str(PY), "-u", "closed_loop.py", "--scenario_path", str(scen), "--base_path", str(base),
           "--camera_path", f"configs/sim/{ds}_camera.yaml", "--kinematic_path", "configs/sim/kinematic.yaml",
           "--ad", ad, "--ad_cuda", str(a.gpu)]
    t0 = time.time()
    with open(run_dir / "sim.log", "w") as log:
        p = subprocess.Popen(cmd, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            code = "timeout"
    wall = time.time() - t0
    txt = (run_dir / "sim.log").read_text(errors="replace")
    end = next((e for k, e in END if k in txt), "max_steps" if txt.count("ego pose") > 400 else "other")
    try:
        ev = json.loads((run_dir / "eval.json").read_text())
    except (OSError, ValueError):
        ev = {}
    row = dict(scenario=scen.stem, dataset=ds, difficulty=mode.split("_")[0], agent=a.agent, controller=a.controller,
               tag=tag_dir.name, steps=txt.count("ego pose"), end=end if ev else "crash", wall_s=round(wall, 1),
               rc_code=code, finished=time.strftime("%F %T"), scene=scene, run_dir=str(run_dir),
               **{k: ev.get(k, "") for k in ("hdscore", "rc", "nc", "dac", "ttc", "c", "pdms")})
    return row


def run(a):
    tag = a.tag or f"{a.agent}-{a.controller}"
    out = Path(a.out)
    tag_dir = out / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    scen = [Path(s) for s in (open(a.scenarios).read().split() if a.scenarios.endswith(".txt") else [a.scenarios])]
    scen = [s if s.is_absolute() else DATA / "scenarios" / s for s in scen]
    done = done_set(results)
    todo = [s for s in scen if (s.stem, tag) not in done]
    traffic = traffic_map()
    print(f"{tag}: {len(todo)} of {len(scen)} scenarios to run, {a.workers} workers, gpu {a.gpu}", flush=True)
    lock, n = threading.Lock(), [0]
    fails = []

    def one(s):
        for attempt in range(1 + a.retries):
            row = run_job(a, s, tag_dir, traffic.get(s.stem.rsplit("-", 2)[0], [1, 0]))
            if row["end"] != "crash":
                break
            print(f"  {s.stem}: crash (attempt {attempt + 1}), see {tag_dir}", flush=True)
        append(results, row)
        with lock:
            n[0] += 1
            if row["end"] == "crash":
                fails.append(s.stem)
            print(f"[{n[0]}/{len(todo)}] {time.strftime('%H:%M:%S')} {tag} {s.stem}: HD {row['hdscore']} "
                  f"RC {row['rc']} {row['end']} {row['steps']} steps {row['wall_s']} s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    print(f"{tag}: finished, {len(fails)} crashed: {fails}", flush=True)
    return 1 if len(fails) > a.max_fail else 0


def derive(a):
    """A scenario yaml with some keys replaced (values parsed as YAML), for checklist-only scenarios."""
    import yaml
    c = yaml.safe_load(open(a.src))
    for kv in a.set:
        k, v = kv.split("=", 1)
        c[k] = yaml.safe_load(v)
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(c, open(a.dst, "w"), sort_keys=False)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup-trees")
    dv = sub.add_parser("derive")
    dv.add_argument("src")
    dv.add_argument("dst")
    dv.add_argument("set", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--agent", required=True, choices=list(AD))
    r.add_argument("--controller", default="official", choices=list(TREES))
    r.add_argument("--scenarios", required=True, help="a .txt list (paths relative to scenarios/) or one yaml")
    r.add_argument("--socket", default="")
    r.add_argument("--opts", default="{}")
    r.add_argument("--tag", default="")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--gpu", default="0")
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--retries", type=int, default=1)
    r.add_argument("--max-fail", type=int, default=3)
    a = ap.parse_args()
    sys.exit({"setup-trees": lambda a: setup_trees(), "derive": derive, "run": run}[a.cmd](a))
