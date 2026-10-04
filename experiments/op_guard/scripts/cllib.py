"""Closed-loop guard lines (hugsim, b2d_turns, b2d_ds): unit sets, lane jobs, the blocking runner and the readers.

Every closed-loop unit runs as a `jevdrive.cl` lane job (lane file `cl_lane.py`, root `<run_dir>/cl`) on a lease:
  - env OP_GUARD_LANE set: that lane is already leased by the caller (guard.py); it is used as is, never released here;
  - unset: `op-guard-cl-<candidate>` is leased with `--cards N` cards and released at the end.
`--cl-lines a,b,c` packs several lines' jobs into one lane run so the cards stay busy (guard.py passes all three on its
first call; the later lines then find their units done and only collect). Units are resumable: b2d_run resumes its
`--out`, zs_run skips finished scenarios, finished lane jobs stay done in `<root>/state.json`.

Harnesses, unchanged: HUGSIM `experiments/hugsim/archive/zs_run.py --preset spec` behind one resident
`hugsim_zs_server.py cinque [--onnx]` (guard_hugsim.sh); B2D `experiments/op_closed_loop/archive/op_arb.sh arm spec`
with `SRV_ONNX` (the candidate serving ONNX) and agent `lib/op_arb_agent.py` (the rig122 lane's unit, verbatim env).
Scorers, reused: HUGSIM spin = `experiments/hugsim/scripts/spin_analysis.analyse` (decision 118 definition), end
class from zs_run's results.csv; B2D turns = `junction_rig122_report.score_attempt` (junction_cl_report turn geometry,
branch / leaves-lane / collision-window definitions); B2D DS = the official record (`vlm_arb_common.route_row`).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardlib as G  # noqa: E402

sys.path.insert(0, str(G.REPO))

REPO = G.REPO
DATA = G.data_dir()
LANEFILE = HERE / "cl_lane.py"
NOZ = '"zones": false, "div_m": 1e9'
OP_ARB = "experiments/op_closed_loop/archive/op_arb.sh"

# ---------------------------------------------------------------- unit sets
HUGSIM_SETS = {"subset": REPO / "experiments/leaderboard_audit/scripts/unified_hugsim_small.txt",   # decision 124's 11 scenes
               "full": REPO / "experiments/hugsim/scripts/derot_all64.txt"}                       # all 64
HUGSIM_TAG = "guard-spec"
# decision 127's 25 turns: 20 val routes (junction_rig122_lane NEW + OLD6), seed 2; the turn keys are the labelled turns
# scored there (results/junction_rig122_per_turn.csv, arm olnz)
TURN_ROUTES = ("10255 15102 28147 5423 334 26872 25051 27994 26153 26723 26365 24758 28008 24416 "
               "28180 24944 27297 9196 6999 34183").split()
TURN_CSV = REPO / "experiments/op_closed_loop/results/junction_rig122_per_turn.csv"
TURN_SEED = 2
# decision 107's 19 routes (vlm_arb_common.ROUTES = DEV + TGT; the `drive` arm of od2_b2d_lane.py, seeds 2 and 3)
DS_ROUTES = ("27043 15102 24944 27870 22535 37969 24497 27297 9196 28147 "
             "16390 15612 15483 17280 16529 16508 19324 2520 19832").split()
DS_SEEDS = {"subset": (2,), "full": (2, 3)}
SHARD = 3                                     # routes (= CARLA workers) per B2D job: two jobs fill a 6-worker card
# DS tolerance: decision 38 measured the 95% band of the difference of two independent evaluations of one checkpoint at
# +-2.3 DS on 209 routes (route-level SD 0.80 x sqrt(209) = 11.6 DS); scaled to n paired route runs: 2.3 x sqrt(209 / n)
DS_BAND_209 = 2.3


def ds_tol(n: int) -> float:
    return DS_BAND_209 * (209 / max(n, 1)) ** 0.5


def turn_keys():
    return sorted({(r["route"], int(r["turn"])) for r in csv.DictReader(open(TURN_CSV)) if r["arm"] == "olnz"})


# ---------------------------------------------------------------- cache (shipped only, checked to be the same arm)
CACHE_HUGSIM = DATA / "runs/unified/hugsim"          # tag uni-d118 = preset opctrl_d118 = alias of spec (11 scenes)
CACHE_TURNS = DATA / "runs/rig122/arms"              # olnz-s2-k* = arm olnz (spec, zones off, open-loop camera, seed 2)


def hugsim_cache_ok(run_dir: Path) -> bool:
    """A cached shipped HUGSIM run counts only if its interface.json is the spec resolution (lateral op-path, delay 0.2 model
    s, static warm-up, dilate clock) and no candidate ONNX was served (the unified chain serves shipped only)."""
    try:
        f = json.loads((run_dir / "interface.json").read_text())
    except (OSError, ValueError):
        return False
    r = f.get("resolved", {})
    return (f.get("preset") in ("spec", "opctrl_d118") and r.get("lateral.exec") == "op-path" and r.get("lateral.delay_s") == 0.2
            and r.get("history.warmup") == "static" and r.get("history.clock") == "dilate")


def turns_cache_ok(udir: Path) -> bool:
    """A cached rig122 unit counts only if its agent config is arm olnz as the guard runs it: preset spec, zones off,
    no camera override (open-loop-aligned default), desire on, tm seed 2, shipped server (no SRV_ONNX)."""
    try:
        d = json.loads(next((udir / "done").glob("*.json")).read_text())
        cfg = json.loads(Path(d["config"]["agent_config"]).read_text())
        mid = (Path(d["config"]["agent_config"]).parents[1] / "srv/model_id").read_text()
    except (OSError, ValueError, StopIteration, KeyError):
        return False
    return (cfg.get("arb") == {"preset": "spec", "zones": False, "div_m": 1e9} and "op_mount" not in cfg and cfg.get("desire") is True
            and d["config"].get("tm_seed") == TURN_SEED and mid.startswith("base:"))


# ---------------------------------------------------------------- lane jobs
def b2d_unit(name, ids, out, seed, nz, onnx, priority=0.0, route_adapter=None):
    from jevdrive.cl import Job
    e = dict(GPU="{gpu}", IDX0="{idx}", WORKERS="{workers}", CPUS="{cpus}", SEED=str(seed), OP_ARB_DIR="{job_dir}/op",
             OP_ARB_ARMS=str(Path(out).parent), SRV_NO_TWIN="1", OPENBLAS_CORETYPE="Haswell", OP_ARB_AGENT="lib/op_arb_agent.py", DESIRE="true")
    if nz:
        e["DRIVE_ARGS"] = NOZ
    if onnx:
        e["SRV_ONNX"] = onnx
    if route_adapter:                         # op_route_ft: the agent sends the route polyline, the server adds the adapter's bias
        e["TOP_ARGS"] = '"route_adapter": "%s"' % route_adapter
    n = len(ids)
    return Job(name, ["bash", OP_ARB, "arm", "spec", ",".join(ids), str(out)], workers=n, vram_gb=7.5, cores=12, tries=2, env=e,
               out=str(out), priority=priority, ok=lambda j, out=Path(out), ids=tuple(ids): all((out / "done" / (r + ".json")).exists() for r in ids))


def hugsim_job(name, out, scen_file, onnx, priority=0.0):
    from jevdrive.cl import Job
    e = dict(GPU="{gpu}", OUT=str(out), SCEN=str(scen_file), TAG=HUGSIM_TAG, WORKERS="5")
    if onnx:
        e["ONNX"] = onnx

    def ok(j, out=Path(out), scen=Path(scen_file)):
        want = {Path(s).stem for s in scen.read_text().split()}
        return want <= {(r["scenario"]) for r in hugsim_rows(out) if r["end"] != "crash"}
    # 5 HUGSIM workers + the server count as 3 CARLA-worker slots (~30 GB VRAM measured as budget, see the line's provenance)
    return Job(name, ["bash", str(HERE / "guard_hugsim.sh")], workers=3, vram_gb=10, cores=12, tries=2, env=e, out=str(out),
               priority=priority, ok=ok)


def hugsim_rows(out: Path, tag: str = HUGSIM_TAG):
    p = Path(out) / "results.csv"
    return [r for r in csv.DictReader(open(p)) if r["tag"] == tag] if p.exists() else []


def shards(ids, stage):
    """Routes dealt round-robin over ceil(n / SHARD) jobs; smoke = the first route alone, in shard 0's dir (the batch reuses it)."""
    n = -(-len(ids) // SHARD)
    return [(0, ids[:1])] if stage == "smoke" else [(k, ids[k::n]) for k in range(n)]


def lane_jobs(candidate: str, mode: str, lines, stage: str = "all", force: bool = False):
    c = G.resolve(candidate)
    rd = G.run_dir(c["name"], mode)
    onnx, ra = c.get("onnx"), c.get("route_adapter")
    jobs = []
    if "hugsim" in lines and not (candidate == G.SHIPPED and mode == "subset" and not force and hugsim_cached()):
        scen = HUGSIM_SETS[mode]
        if stage == "smoke":
            scen = rd / "hugsim_smoke.txt"
            scen.write_text(HUGSIM_SETS[mode].read_text().split()[0] + "\n")
        jobs.append(hugsim_job("hugsim" + ("-smoke" if stage == "smoke" else ""), rd / "hugsim", scen, onnx, priority=3))
    if "b2d_turns" in lines and not (candidate == G.SHIPPED and not force and turns_cached()):
        for k, ids in shards(TURN_ROUTES, stage):
            jobs.append(b2d_unit("turns-s%d-k%d%s" % (TURN_SEED, k, "-smoke" if stage == "smoke" else ""), ids,
                                 rd / "b2d" / ("turns-s%d-k%d" % (TURN_SEED, k)), TURN_SEED, True, onnx, priority=2, route_adapter=ra))
    if "b2d_ds" in lines:
        for s in DS_SEEDS[mode][:1] if stage == "smoke" else DS_SEEDS[mode]:
            for k, ids in shards(DS_ROUTES, stage):
                jobs.append(b2d_unit("ds-s%d-k%d%s" % (s, k, "-smoke" if stage == "smoke" else ""), ids,
                                     rd / "b2d" / ("ds-s%d-k%d" % (s, k)), s, False, onnx, priority=1, route_adapter=ra))
    return jobs


# ---------------------------------------------------------------- blocking runner
def add_args(ap):
    ap.add_argument("--cl-lines", default="", help="closed-loop lines whose jobs this call runs in one lane (default: this line only)")
    ap.add_argument("--cards", type=int, default=2, help="cards to lease when OP_GUARD_LANE is unset")
    ap.add_argument("--stage", choices=("smoke", "all"), default="all", help="smoke = one unit per line (same dirs, reused by all)")
    ap.add_argument("--collect-only", action="store_true", help="do not run anything; read the finished units")
    return ap


def lanes_cmd(*a):
    return subprocess.run([str(REPO / ".venv/bin/python"), "-m", "jevdrive.cl", *a], cwd=REPO, text=True, capture_output=True)


def run_lines(a, line: str) -> dict:
    """Lease (or use OP_GUARD_LANE), run the lane for --cl-lines (or `line`) of a.candidate / a.mode, block until it ends.
    Returns run provenance (lane, root, rc, wall_s, per-job wall)."""
    lines = [x for x in (a.cl_lines or line).split(",") if x]
    c = G.resolve(a.candidate)
    rd = G.run_dir(c["name"], a.mode)
    root = rd / "cl"
    if a.force:                                        # recompute: forget the lane state and these lines' own units
        kill = [root] + ([rd / "hugsim"] if "hugsim" in lines else []) + [
            p for p in (rd / "b2d").glob("*") if (p.name.startswith("turns-") and "b2d_turns" in lines) or (p.name.startswith("ds-") and "b2d_ds" in lines)]
        for p in kill:
            if p.exists():
                shutil.rmtree(p)
    elif (root / "state.json").exists():               # a rerun retries jobs that failed before (their units resume)
        st = json.loads((root / "state.json").read_text())
        for v in st["jobs"].values():
            if v.get("state") == "failed":
                v.update(state="queued", tries=0)
        (root / "state.json").write_text(json.dumps(st, indent=1))
    jobs = lane_jobs(a.candidate, a.mode, lines, a.stage, a.force)
    prov = dict(lines=lines, root=str(root), stage=a.stage, jobs=[j.name for j in jobs])
    if not jobs:
        return dict(prov, rc=0, wall_s=0.0, note="every unit cached or finished")
    lane = os.environ.get("OP_GUARD_LANE")
    own = not lane
    if own:
        lane = "op-guard-cl-" + c["name"]
        r = lanes_cmd("lease", lane, "--gpus", str(a.cards), "--status", "op_guard closed-loop lines %s for %s" % (",".join(lines), c["name"]))
        print(r.stdout + r.stderr, flush=True)
        if r.returncode:
            raise SystemExit("lease %s failed (rc %d); free cards are needed, see `python -m jevdrive.cl probe`" % (lane, r.returncode))
    t0 = time.time()
    try:
        cmd = [str(REPO / ".venv/bin/python"), "-m", "jevdrive.cl", "run", str(LANEFILE), "--lane", lane, "--root", str(root),
               "--arg", "candidate=" + a.candidate, "--arg", "mode=" + a.mode, "--arg", "lines=" + ",".join(lines), "--arg", "stage=" + a.stage]
        if a.force:
            cmd += ["--arg", "force=1"]
        print(" ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
    finally:
        if own:
            print(lanes_cmd("release", lane, "done: op_guard %s %s" % (c["name"], ",".join(lines))).stdout, flush=True)
    st = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {"jobs": {}}
    return dict(prov, lane=lane, rc=rc, wall_s=round(time.time() - t0, 1),
                job_wall_s={n: st["jobs"].get(n, {}).get("wall_s") for n in prov["jobs"]},
                failed=[n for n in prov["jobs"] if st["jobs"].get(n, {}).get("state") == "failed"])


# ---------------------------------------------------------------- readers
_HUGSIM_CACHED = None


def hugsim_cached() -> bool:
    global _HUGSIM_CACHED
    if _HUGSIM_CACHED is None:
        rows = hugsim_rows(CACHE_HUGSIM, "uni-d118")
        want = {Path(s).stem for s in HUGSIM_SETS["subset"].read_text().split()}
        _HUGSIM_CACHED = want <= {r["scenario"] for r in rows if r["end"] != "crash" and hugsim_cache_ok(Path(r["run_dir"]))}
    return _HUGSIM_CACHED


def hugsim_scenes(candidate: str, mode: str, force: bool = False) -> dict:
    """scenario -> {spin, max_abs_e, end, hd, rc, steps, wall_s, run_dir, src} for every finished scene of the line's set."""
    sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
    from spin_analysis import analyse, load_run
    want = [Path(s).stem for s in HUGSIM_SETS[mode].read_text().split()]
    rows = {r["scenario"]: (r, "run") for r in hugsim_rows(G.run_dir(candidate, mode) / "hugsim") if r["end"] != "crash"}
    if candidate == G.SHIPPED and mode == "subset" and not force:
        for r in hugsim_rows(CACHE_HUGSIM, "uni-d118"):
            if r["scenario"] not in rows and r["end"] != "crash" and hugsim_cache_ok(Path(r["run_dir"])):
                rows[r["scenario"]] = (r, "cache:" + str(CACHE_HUGSIM) + " tag uni-d118")
    routes = json.load(open(DATA / "tmp_spin/routes.json"))
    out = {}
    for s in want:
        if s not in rows:
            continue
        r, src = rows[s]
        d = Path(r["run_dir"])
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        out[s] = dict(spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1), end=r["end"], hd=float(r["hdscore"]), rc=float(r["rc"]),
                      steps=int(r["steps"]), wall_s=float(r["wall_s"]), run_dir=str(d), src=src)
    return out


def turns_cached() -> bool:
    dirs = sorted(CACHE_TURNS.glob("olnz-s%d-k*" % TURN_SEED))
    have = {p.stem for d in dirs if turns_cache_ok(d) for p in (d / "done").glob("*.json")}
    return set(TURN_ROUTES) <= have


def attempt_of(udirs, rid):
    for u in udirs:
        f = Path(u) / "done" / (rid + ".json")
        if f.exists():
            d = json.loads(f.read_text())
            return Path(u) / "attempts" / rid / str(d.get("attempt", 1)), d
    return None, None


def b2d_dirs(candidate, mode, kind, seed, force=False):
    """Unit dirs holding (kind, seed) runs of the candidate, own first; shipped turns fall back to the checked rig122 cache."""
    own = sorted((G.run_dir(candidate, mode) / "b2d").glob("%s-s%d-k*" % (kind, seed)))
    if kind == "turns" and candidate == G.SHIPPED and not force:
        own += [d for d in sorted(CACHE_TURNS.glob("olnz-s%d-k*" % seed)) if turns_cache_ok(d)]
    return own


def turn_rows(candidate, mode, force=False):
    """Per-turn rows (junction_rig122_report.score_attempt) of the 25 turns, plus run collisions, route DS and provenance."""
    sys.path[:0] = [str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
    import junction_forced_report as F
    import junction_rig122_report as J
    lab, keys = F.load_labels(), set(turn_keys())
    dirs = b2d_dirs(candidate, mode, "turns", TURN_SEED, force)
    rows, cols, routes, src = [], [], {}, {}
    for rid in TURN_ROUTES:
        att, d = attempt_of(dirs, rid)
        if att is None or not (att / "route.json").exists():
            continue
        s = J.score_attempt(rid, "cand", att, J.route_geometry(att, rid, lab), lab)
        if s is None:
            continue
        rows += [r for r in s["rows"] if (r["route"], r["turn"]) in keys]
        cols += s["cols"]
        routes[rid] = dict(ds=s["rr"][0], rc=s["rr"][1], status=s["rr"][2], wall_s=d.get("wall_s"))
        src[rid] = str(att)
    return rows, cols, routes, src


def ds_rows(candidate, mode):
    """(seed, route) -> official route row (vlm_arb_common.route_row: DS, RC, status, infractions) of the 19-route spec runs."""
    sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
    from vlm_arb_common import route_row
    out = {}
    for s in DS_SEEDS[mode]:
        for rid in DS_ROUTES:
            for u in b2d_dirs(candidate, mode, "ds", s):
                r = route_row(u, rid)
                if r is not None:
                    d = json.loads((u / "done" / (rid + ".json")).read_text())
                    out[(s, rid)] = dict(r, wall_s=d.get("wall_s"))
                    break
    return out


# ---------------------------------------------------------------- line entry
def main_line(line: str, doc: str, collect) -> int:
    """The blocking entry of a closed-loop line script: run its lane units (unless --collect-only), then
    `collect(candidate, mode, force) -> (rows, provenance, status)` into lines/<line>.json."""
    a = add_args(G.line_args(line, doc)).parse_args()
    c = G.resolve(a.candidate)
    out = G.run_dir(c["name"], a.mode)
    lines = [x for x in (a.cl_lines or line).split(",") if x]
    if not (a.force or a.collect_only or a.stage == "smoke") and all((G.load_line(out, x) or {}).get("status") == "ok" for x in lines):
        print("%s: lines %s already done for %s / %s (--force recomputes)" % (line, lines, c["name"], a.mode))
        return 0
    t0 = time.time()
    run = {} if a.collect_only else run_lines(a, line)
    rows, prov, status = collect(c["name"], a.mode, a.force)
    if run.get("failed") or run.get("rc"):
        status = "error"
    if a.stage == "smoke":
        print(json.dumps(dict(run=run, rows=rows, provenance=prov), indent=1, default=float))
        return 0
    p = G.write_line(out, line, c["name"], a.mode, rows, t0, status, dict(prov, run=run))
    for r in rows:
        print("%-16s %-34s %10s  ref %10s  ci %-22s pass %-5s %s" % (r["id"], r["metric"][:34], _f(r["value"]), _f(r["ref"]),
                                                                  "" if r["ci"] is None else "[%s, %s]" % tuple(_f(x) for x in r["ci"]), r["pass"], r["note"]))
    print("wrote", p, "status", status)
    return 0 if status == "ok" else 1


def _f(x):
    return "-" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))


def paired_ci(a, b, groups=None):
    """jevdrive.stats.paired of (a - b): (mean, [lo, hi]) or (None, None) without pairs."""
    import numpy as np
    from jevdrive import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return None, None
    r = stats.paired(a, b, groups=groups)
    return float(r["mean"]), [float(r["lo"]), float(r["hi"])]
