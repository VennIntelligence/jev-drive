#!/usr/bin/env python
"""Batch runner of HUGSIM's official closed_loop.py for the zero-shot exam (experiments/hugsim/results/hugsim-exam-plan).

One job = (scenario, agent, controller). Agents: alpamayo / cinque / lebowski (experiments/hugsim/lib/zs_agent.py against a
resident experiments/hugsim/archive/hugsim_zs_server.py), cv / route (experiments/hugsim/archive/agent_client.py), ltf (the official LTF client),
wajepa (WA-JEPA's shipped client, experiments/hugsim/scripts/wajepa_e2e.sh; ad slot `wj`), preset (the scene's logged trajectory as the plan, experiments/hugsim/archive/preset_agent.py; controller acceptance).
Controllers run from private copies of the patched HUGSIM tree, so nobody else's apply / revert of the optional
patch can touch a running job:  official = patches/hugsim/*.patch,  fixed = + optional/lqr-heading-fix.patch,
ideal = + optional/ideal-tracker.patch (no iLQR: the ego moves exactly along the plan; reference for acceptance,
created only by `setup-trees ideal`), fixed2 = fixed + optional/lqr-tracker-v2.patch (iLQR at the 0.25 s simulator step,
steering-rate cost 1; created only by `setup-trees fixed2`), lowspeed = fixed + optional/lowspeed-ctrl.patch (steering low-pass
and curvature-rate limit below 3 m/s, lib/lowspeed_ctrl.py, parameters in env LOWSPEED_CTRL; created only by `setup-trees lowspeed`), lowsel = lowspeed + optional/lowspeed-sel-ctrl.patch (selective plan-direction rule, env LOWSPEED_SEL; `setup-trees lowsel`), opctrl = fixed + optional/op-ctrl.patch (openpilot's lateral path from the model's desired curvature, lib/op_ctrl.py, env OP_CTRL + OP_CTRL_LIB, agent opt op_ctrl; `setup-trees opctrl`), opctrl_long = opctrl + optional/op-ctrl-long.patch (openpilot's longitudinal path too, lib/op_ctrl.py OpLongitudinal, env OP_CTRL_LONG, agent opt op_long; `setup-trees opctrl_long`). The patch state of the job's tree is verified before every job.

    python experiments/hugsim/archive/zs_run.py setup-trees
    python experiments/hugsim/archive/zs_run.py run --out $DATA_DIR/runs/hugsim-exam --agent cinque --controller official \
        --socket $DATA_DIR/runs/hugsim-exam/cinque.sock --scenarios <list.txt> --workers 2 --gpu 0

Interface presets (jevdrive/openpilot/interface.py HUGSIM_PRESETS, docs/openpilot-interface.md): --preset spec (the default for
the openpilot agents cinque / lebowski: openpilot's lateral path in tree opctrl, 5 s static warm-up, dilate clock = decision 118's
arm) | spec_cold (no static warm-up: does not launch) | spec_hold (hold clock) | opctrl_d118 (alias of spec) | exam (legacy: --controller and --opts taken literally; every result before 2026-10-05; the default for the other
agents). A preset sets the controller tree, OP_CTRL / OP_CTRL_LIB and the agent opts; --opts are merged on top. The agent writes
interface.json into every run dir.

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

REPO = Path(__file__).resolve().parents[3]
D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
DATA = D / "datasets" / "hugsim"
PY = D / "envs" / "hugsim" / "bin" / "python"
TREES = {c: D / "third_party" / "HUGSIM-zs" / c for c in ("official", "fixed", "ideal", "fixed2", "lowspeed", "lowsel", "opctrl", "opctrl_long", "fixedc")}
FIX = REPO / "patches" / "hugsim" / "optional" / "lqr-heading-fix.patch"
IDEAL = REPO / "patches" / "hugsim" / "optional" / "ideal-tracker.patch"
LOWSPEED = REPO / "patches" / "hugsim" / "optional" / "lowspeed-ctrl.patch"
LOWSEL = REPO / "patches" / "hugsim" / "optional" / "lowspeed-sel-ctrl.patch"
V2 = REPO / "patches" / "hugsim" / "optional" / "lqr-tracker-v2.patch"
OPCTRL = REPO / "patches" / "hugsim" / "optional" / "op-ctrl.patch"
CLAMP = REPO / "patches" / "hugsim" / "optional" / "command-index-clamp.patch"
OPCTRL_LONG = REPO / "patches" / "hugsim" / "optional" / "op-ctrl-long.patch"
AD = {"alpamayo": "zs", "cinque": "zs", "lebowski": "zs", "cv": "jev", "route": "jev", "ltf": "ltf", "preset": "pre", "wajepa": "wj"}
FIELDS = ["scenario", "dataset", "difficulty", "agent", "controller", "tag", "hdscore", "rc", "nc", "dac", "ttc", "c",
          "pdms", "steps", "end", "wall_s", "rc_code", "finished", "scene", "run_dir"]
END = [("Collision with background", "bg_collision"), ("Collision with foreground", "fg_collision"),
       ("Far from preset trajectory", "off_route"), ("Complete", "complete")]


def sh(*a, **k):
    return subprocess.run(a, check=True, capture_output=True, text=True, **k).stdout


def setup_trees(names=("official", "fixed")):
    src = D / "third_party" / "HUGSIM"
    for name in names:
        dst = TREES[name]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            sh("git", "clone", "-q", str(src), str(dst))
            sh("git", "-C", str(dst), "checkout", "-q", sh("git", "-C", str(src), "rev-parse", "HEAD").strip())
            for p in sorted((REPO / "patches" / "hugsim").glob("*.patch")):
                sh("git", "-C", str(dst), "apply", str(p))
            if name in ("fixed", "fixed2", "lowspeed", "lowsel", "opctrl", "opctrl_long", "fixedc"):
                sh("git", "-C", str(dst), "apply", str(FIX))
            if name == "fixedc":
                sh("git", "-C", str(dst), "apply", str(CLAMP))
            if name in ("opctrl", "opctrl_long"):
                sh("git", "-C", str(dst), "apply", str(OPCTRL))
            if name == "opctrl_long":
                sh("git", "-C", str(dst), "apply", str(OPCTRL_LONG))
            if name in ("lowspeed", "lowsel"):
                sh("git", "-C", str(dst), "apply", str(LOWSPEED))
            if name == "lowsel":
                sh("git", "-C", str(dst), "apply", str(LOWSEL))
            if name == "fixed2":
                sh("git", "-C", str(dst), "apply", str(V2))
            if name == "ideal":
                sh("git", "-C", str(dst), "apply", str(IDEAL))
        check_tree(name)
        print(name, dst, "ok")


def applied(t, patch):
    fwd = subprocess.run(["git", "-C", t, "apply", "--check", str(patch)], capture_output=True).returncode == 0
    rev = subprocess.run(["git", "-C", t, "apply", "-R", "--check", str(patch)], capture_output=True).returncode == 0
    return rev if fwd != rev else None                     # None: neither or both apply (a broken tree)


def check_tree(name):
    t = str(TREES[name])
    if applied(t, FIX) is not (name in ("fixed", "fixed2", "lowspeed", "lowsel", "opctrl", "opctrl_long", "fixedc")):
        raise SystemExit(f"tree {t}: optional LQR patch state wrong for controller '{name}'")
    if bool(applied(t, CLAMP)) is not (name == "fixedc"):
        raise SystemExit(f"tree {t}: command-index-clamp patch state wrong for controller '{name}'")
    if name == "ideal" and not applied(t, IDEAL):
        raise SystemExit(f"tree {t}: ideal-tracker patch not applied")
    if name == "lowsel" and not applied(t, LOWSEL):
        raise SystemExit(f"tree {t}: lowspeed-sel-ctrl patch not applied")
    if name in ("lowspeed", "lowsel") and not applied(t, LOWSPEED):
        raise SystemExit(f"tree {t}: lowspeed-ctrl patch not applied")
    if name == "opctrl" and not applied(t, OPCTRL):          # opctrl_long: the long patch extends op-ctrl's hunks, so op-ctrl alone no longer reverses cleanly
        raise SystemExit(f"tree {t}: op-ctrl patch not applied")
    if bool(applied(t, OPCTRL_LONG)) is not (name == "opctrl_long"):
        raise SystemExit(f"tree {t}: op-ctrl-long patch state wrong for controller '{name}'")
    if name == "opctrl_long" and "op_kappa" not in (Path(t) / "closed_loop.py").read_text():
        raise SystemExit(f"tree {t}: op-ctrl patch not applied")
    if name == "fixed2" and not applied(t, V2):
        raise SystemExit(f"tree {t}: tracker-v2 patch not applied")


def traffic_map():
    rows = csv.DictReader(open(REPO / "experiments/hugsim/results/hugsim-exam/scenarios.csv"))
    return {r["scene"]: ([0, 1] if r["location"].startswith("singapore") else [1, 0]) for r in rows}


def unpack(ds, scene):
    d = DATA / "scenes" / ds / scene
    if (d / "scene.pth").exists():
        return
    with open(DATA / "scenes" / ds / f".{scene}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        nested = DATA / "scenes" / ds / ds / scene     # later releases zip as <ds>/<scene>/..., older as <scene>/...
        if not (d / "scene.pth").exists() and not (nested / "scene.pth").exists():
            zipfile.ZipFile(DATA / "scenes" / ds / f"{scene}.zip").extractall(DATA / "scenes" / ds)
        if not (d / "scene.pth").exists() and (nested / "scene.pth").exists():
            nested.rename(d)


def done_set(results):
    """Finished (scenario, tag) pairs; infrastructure crashes do not count, so a rerun retries them."""
    if not results.exists():
        return set()
    return {(r["scenario"], r["tag"]) for r in csv.DictReader(open(results)) if r["end"] != "crash"}


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
                    f"zs_path: {REPO}/experiments/hugsim/archive/zs_agent_e2e.sh\njev_path: {REPO}/experiments/hugsim/archive/agent_e2e.sh\n"
                    f"ltf_path: {REPO}/experiments/hugsim/archive/ltf_e2e.sh\nwj_path: {REPO}/experiments/hugsim/scripts/wajepa_e2e.sh\npre_path: {REPO}/experiments/hugsim/archive/preset_agent_e2e.sh\n"
                    f"output_dir: {tag_dir}/\n"
                    f"HD_map:\n  path: {DATA}/nusc_map_cache\n  version: nusc_trainval\n")
    run_dir = tag_dir / ad / f"{scene}_{mode}"
    if run_dir.exists():
        subprocess.run(["rm", "-rf", str(run_dir)])
    run_dir.mkdir(parents=True)
    tree = TREES[a.controller]
    check_tree(a.controller)
    opts = dict(a.preset_opts, **json.loads(a.opts), traffic=traffic)
    env = dict(os.environ, **a.preset_env)
    for k, v in json.loads(getattr(a, "controller_env", "{}") or "{}").items():
        env[k] = json.dumps(v) if isinstance(v, dict) else str(v)
    if a.controller in ("lowspeed", "lowsel"):
        env["LOWSPEED_LIB"] = str(REPO / "lib")
    if a.controller in ("opctrl", "opctrl_long"):
        env["OP_CTRL_LIB"] = str(REPO / "lib")
    env = dict(env, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HUGSIM_ZS_PRESET=a.preset, HUGSIM_ZS_CONTROLLER=a.controller,
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
    end = next((e for k, e in END if k in txt), "max_steps" if txt.count("ego pose") >= 400 else "other")
    try:
        ev = json.loads((run_dir / "eval.json").read_text())
    except (OSError, ValueError):
        ev = {}
    row = dict(scenario=scen.stem, dataset=ds, difficulty=mode.split("_")[0], agent=a.agent, controller=a.controller,
               tag=tag_dir.name, steps=txt.count("ego pose"), end=end if ev else "crash", wall_s=round(wall, 1),
               rc_code=code, finished=time.strftime("%F %T"), scene=scene, run_dir=str(run_dir),
               **{k: ev.get(k, "") for k in ("hdscore", "rc", "nc", "dac", "ttc", "c", "pdms")})
    return row


def apply_preset(a):
    """Resolve --preset into a.controller / a.preset_env / a.preset_opts (jevdrive.openpilot.interface.HUGSIM_PRESETS)."""
    sys.path.insert(0, str(REPO))
    from jevdrive.openpilot import interface as IF
    a.preset = a.preset or ("spec" if a.agent in ("cinque", "lebowski") else "exam")
    p = IF.HUGSIM_PRESETS[a.preset]
    a.preset_env, a.preset_opts = {}, dict(p["opts"])
    if p["controller"] is None:                       # legacy: everything as given
        a.controller = a.controller or "official"
        return
    if a.controller not in (None, p["controller"]):
        raise SystemExit(f"--preset {a.preset} runs tree {p['controller']}, not --controller {a.controller}")
    a.controller = p["controller"]
    for k, v in p["env"].items():
        a.preset_env[k] = json.dumps(v)
    if "OP_CTRL" in p["env"]:
        a.preset_env["OP_CTRL_LIB"] = str(REPO / "lib")


def run(a):
    apply_preset(a)
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
    st = sub.add_parser("setup-trees")
    st.add_argument("names", nargs="*", default=["official", "fixed"], choices=list(TREES))
    dv = sub.add_parser("derive")
    dv.add_argument("src")
    dv.add_argument("dst")
    dv.add_argument("set", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--agent", required=True, choices=list(AD))
    r.add_argument("--controller", default=None, choices=list(TREES), help="exam preset: default official; other presets set it")
    r.add_argument("--preset", default=None, help="interface preset: spec | spec_cold | spec_hold | opctrl_d118 | exam (see the docstring)")
    r.add_argument("--scenarios", required=True, help="a .txt list (paths relative to scenarios/) or one yaml")
    r.add_argument("--socket", default="")
    r.add_argument("--opts", default="{}")
    r.add_argument("--controller-env", default="{}", help="explicit controller parameter objects, applied after preset defaults")
    r.add_argument("--tag", default="")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--gpu", default="0")
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--retries", type=int, default=1)
    r.add_argument("--max-fail", type=int, default=3)
    a = ap.parse_args()
    sys.exit({"setup-trees": lambda a: setup_trees(a.names), "derive": derive, "run": run}[a.cmd](a))
