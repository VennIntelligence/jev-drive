"""Closed-loop guard lines (hugsim, b2d_turns, b2d_ds): unit sets, GPU-pool units, the blocking runner and the readers.

Every closed-loop unit is one GPU-pool job (`jevdrive.cl.pool`, the dispatcher in tmux jev:pool picks the card), log dir
`<run_dir>/cl/<unit>/` (log.txt, STATUS, DONE / ERROR, pool_id). `--cl-lines a,b,c` submits several lines' units at once so
the pool can spread them (guard.py passes all three on its first call; the later lines then find their units done and only
collect). Units are resumable: b2d_run resumes its `--out`, zs_run skips finished scenarios, a finished unit is not resubmitted
and a unit still in the pool is waited on, not submitted twice. `--dry-run` prints the pool specs.

Harnesses, unchanged: HUGSIM `experiments/hugsim/archive/zs_run.py --preset spec` behind one resident
`hugsim_zs_server.py cinque [--onnx]` (guard_hugsim.sh); B2D `experiments/op_closed_loop/archive/op_arb.sh arm spec`
with `SRV_ONNX` (the candidate serving ONNX) and agent `lib/op_arb_agent.py` (the rig122 unit, verbatim env).
Scorers, reused: HUGSIM spin = `experiments/hugsim/scripts/spin_analysis.analyse` (decision 118 definition), end
class from zs_run's results.csv; B2D turns = `junction_rig122_report.score_attempt` (junction_cl_report turn geometry,
branch / leaves-lane / collision-window definitions); B2D DS = the official record (`vlm_arb_common.route_row`).
"""
from __future__ import annotations

import csv
import json
import shlex
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardlib as G  # noqa: E402

sys.path.insert(0, str(G.REPO))

REPO = G.REPO
DATA = G.data_dir()
NOZ = '"zones": false, "div_m": 1e9'
OP_ARB = "experiments/op_closed_loop/archive/op_arb.sh"

# ---------------------------------------------------------------- unit sets
HUGSIM_SETS = {"subset": REPO / "experiments/leaderboard_audit/scripts/unified_hugsim_small.txt",   # decision 124's 11 scenes
               "full": REPO / "experiments/hugsim/scripts/derot_all64.txt"}                       # all 64
HUGSIM_TAG = "guard-spec"
# decision 127's 25 turns: 20 val routes (junction_rig122_report NEW + OLD6), seed 2; the turn keys are the labelled turns
# scored there (results/junction_rig122_per_turn.csv, arm olnz)
TURN_ROUTES = ("10255 15102 28147 5423 334 26872 25051 27994 26153 26723 26365 24758 28008 24416 "
               "28180 24944 27297 9196 6999 34183").split()
TURN_CSV = REPO / "experiments/op_closed_loop/results/junction_rig122_per_turn.csv"
TURN_SEED = 2
# decision 107's 19 routes (vlm_arb_common.ROUTES = DEV + TGT; the `drive` arm of op_adapt_h's od2 B2D run, seeds 2 and 3)
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


# ---------------------------------------------------------------- pool units
def b2d_unit(name, ids, out, seed, nz, onnx, priority=0.0, route_adapter=None) -> dict:
    """One op_arb.sh `arm spec` invocation on len(ids) CARLA workers (7.5 GB each, 12 pinned cores); done when every route has
    out/done/<id>.json (the command fails otherwise, so a retry resumes the unfinished routes)."""
    e = dict(GPU="{gpu}", IDX0="{idx}", WORKERS="{carla}", CPUS="{cpus}", SEED=str(seed), OP_ARB_DIR="{job_dir}/op",
             OP_ARB_ARMS=str(Path(out).parent), SRV_NO_TWIN="1", OPENBLAS_CORETYPE="Haswell", OP_ARB_AGENT="lib/op_arb_agent.py", DESIRE="true")
    if nz:
        e["DRIVE_ARGS"] = NOZ
    if onnx:
        e["SRV_ONNX"] = onnx
    if route_adapter:                         # op_route_ft: the agent sends the route polyline, the server adds the adapter's bias
        e["TOP_ARGS"] = '"route_adapter": "%s"' % route_adapter
    out, ids = Path(out), [str(r) for r in ids]
    check = " && ".join("test -f %s" % shlex.quote(str(out / "done" / (r + ".json"))) for r in ids)
    cmd = "%s && %s" % (shlex.join(["bash", OP_ARB, "arm", "spec", ",".join(ids), str(out)]), check)
    return dict(name=name, cmd=cmd, env=e, vram_gb=7.5 * len(ids), carla=len(ids), cpu=12, tries=2, priority=priority,
                out=str(out), done=lambda out=out, ids=tuple(ids): all((out / "done" / (r + ".json")).exists() for r in ids))


def hugsim_unit(name, out, scen_file, onnx, priority=0.0) -> dict:
    """guard_hugsim.sh: one resident Cinque server + 5 HUGSIM workers (no CARLA; ~30 GB VRAM measured as budget); done when every
    scenario of scen_file has a non-crash row (checked after the run by `cllib.py hugsim-check`)."""
    e = dict(GPU="{gpu}", OUT=str(out), SCEN=str(scen_file), TAG=HUGSIM_TAG, WORKERS="5")
    if onnx:
        e["ONNX"] = onnx
    cmd = "bash %s && %s %s hugsim-check %s %s" % (shlex.quote(str(HERE / "guard_hugsim.sh")), shlex.quote(sys.executable),
                                                  shlex.quote(str(HERE / "cllib.py")), shlex.quote(str(out)), shlex.quote(str(scen_file)))
    return dict(name=name, cmd=cmd, env=e, vram_gb=30.0, carla=0, cpu=12, tries=2, priority=priority, out=str(out),
                done=lambda out=Path(out), scen=Path(scen_file): hugsim_complete(out, scen))


def hugsim_complete(out: Path, scen: Path) -> bool:
    want = {Path(s).stem for s in Path(scen).read_text().split()}
    return want <= {r["scenario"] for r in hugsim_rows(out) if r["end"] != "crash"}


def hugsim_rows(out: Path, tag: str = HUGSIM_TAG):
    p = Path(out) / "results.csv"
    return [r for r in csv.DictReader(open(p)) if r["tag"] == tag] if p.exists() else []


def shards(ids, stage):
    """Routes dealt round-robin over ceil(n / SHARD) units; smoke = the first route alone, in shard 0's dir (the batch reuses it)."""
    n = -(-len(ids) // SHARD)
    return [(0, ids[:1])] if stage == "smoke" else [(k, ids[k::n]) for k in range(n)]


def submit_units(units, log_root, owner: str, dry_run: bool = False, force: bool = False) -> dict:
    """Submit each unit to the GPU pool (jevdrive.cl.pool) with log_dir <log_root>/<name> (log.txt, STATUS, DONE / ERROR, pool_id).
    Resumable: a finished unit (its `done` check) is skipped, a unit whose recorded pool job is still queued / running is reused,
    so rerunning a submitter never double-books. Returns name -> pool id ("" = already finished). dry_run prints the specs."""
    from jevdrive.cl import pool as P
    out = {}
    for u in units:
        ld = Path(log_root) / u["name"]
        if "{job_dir}/op" in u["env"].values() and len(str(ld / "op/srv/op.sock")) > 107:   # op_arb.sh's AF_UNIX socket (<= 108 bytes)
            raise SystemExit("unit %s: socket path %s too long for AF_UNIX; shorten the root or the unit name" % (u["name"], ld / "op/srv/op.sock"))
        if not force and u["done"]():
            out[u["name"]] = ""
            print("finished  %s" % u["name"])
            continue
        old = (ld / "pool_id").read_text().strip() if (ld / "pool_id").exists() else ""
        if old and P.job_states([old])[old] in ("inbox", "queued", "running"):
            out[u["name"]] = old
            print("in pool   %s  %s" % (u["name"], old))
            continue
        kw = dict(owner=owner, vram_gb=u["vram_gb"], carla=u["carla"], cpu=u["cpu"], tries=u["tries"], priority=u["priority"],
                  env=u["env"], log_dir=str(ld), cwd=str(REPO))
        if dry_run:
            print("would submit %s\n  %s\n  %s" % (u["name"], json.dumps(kw), u["cmd"]))
            continue
        ld.mkdir(parents=True, exist_ok=True)
        jid = P.submit(u["cmd"], name=u["name"], **kw)
        (ld / "pool_id").write_text(jid + "\n")
        out[u["name"]] = jid
        print("submitted %s  %s" % (u["name"], jid))
    return out


def wait_units(ids: dict, poll_s: float = 60.0) -> list:
    """Block until every submitted unit is final (jevdrive.cl.pool.wait); returns the names that did not finish done."""
    from jevdrive.cl import pool as P
    live = {n: i for n, i in ids.items() if i}
    st = P.wait(list(live.values()), poll_s) if live else {}
    return [n for n, i in live.items() if st.get(i) != "done"]


def line_units(candidate: str, mode: str, lines, stage: str = "all", force: bool = False):
    c = G.resolve(candidate)
    rd = G.run_dir(c["name"], mode)
    onnx, ra = c.get("onnx"), c.get("route_adapter")
    jobs = []
    if "hugsim" in lines and not (candidate == G.SHIPPED and mode == "subset" and not force and hugsim_cached()):
        scen = HUGSIM_SETS[mode]
        if stage == "smoke":
            scen = rd / "hugsim_smoke.txt"
            scen.parent.mkdir(parents=True, exist_ok=True)
            scen.write_text(HUGSIM_SETS[mode].read_text().split()[0] + "\n")
        jobs.append(hugsim_unit("hugsim" + ("-smoke" if stage == "smoke" else ""), rd / "hugsim", scen, onnx, priority=3))
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
    ap.add_argument("--cl-lines", default="", help="closed-loop lines whose units this call submits together (default: this line only)")
    ap.add_argument("--stage", choices=("smoke", "all"), default="all", help="smoke = one unit per line (same dirs, reused by all)")
    ap.add_argument("--collect-only", action="store_true", help="do not run anything; read the finished units")
    ap.add_argument("--dry-run", action="store_true", help="print the pool specs of the units this call would submit, run nothing")
    return ap


def run_lines(a, line: str) -> dict:
    """Submit the units of --cl-lines (or `line`) of a.candidate / a.mode to the GPU pool (log dirs <run_dir>/cl/<unit>), block until
    they are final. Returns run provenance (root, pool ids, wall, per-unit wall, failed units)."""
    lines = [x for x in (a.cl_lines or line).split(",") if x]
    c = G.resolve(a.candidate)
    rd = G.run_dir(c["name"], a.mode)
    root = rd / "cl"
    if a.force and not a.dry_run:                      # recompute: forget these lines' own units
        kill = ([rd / "hugsim"] if "hugsim" in lines else []) + [
            p for p in (rd / "b2d").glob("*") if (p.name.startswith("turns-") and "b2d_turns" in lines) or (p.name.startswith("ds-") and "b2d_ds" in lines)]
        for p in kill:
            if p.exists():
                shutil.rmtree(p)
    units = line_units(a.candidate, a.mode, lines, a.stage, a.force)
    prov = dict(lines=lines, root=str(root), stage=a.stage, jobs=[u["name"] for u in units])
    if not units:
        return dict(prov, rc=0, wall_s=0.0, note="every unit cached or finished")
    t0 = time.time()
    ids = submit_units(units, root, "op_guard " + c["name"], dry_run=a.dry_run)
    if a.dry_run:
        return dict(prov, rc=0, wall_s=0.0, note="dry run")
    failed = wait_units(ids)
    wall = {}
    for u in units:
        try:
            wall[u["name"]] = json.loads((root / u["name"] / "DONE").read_text()).get("wall_s")
        except (OSError, ValueError):
            wall[u["name"]] = None
    return dict(prov, pool_ids=ids, rc=1 if failed else 0, wall_s=round(time.time() - t0, 1), job_wall_s=wall, failed=failed)


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
    if a.dry_run:
        print(json.dumps(run_lines(a, line), indent=1, default=str))
        return 0
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


if __name__ == "__main__":                 # `cllib.py hugsim-check <out> <scen_file>`: the HUGSIM unit's output check
    if sys.argv[1:2] == ["hugsim-check"] and len(sys.argv) == 4:
        ok = hugsim_complete(Path(sys.argv[2]), Path(sys.argv[3]))
        print("hugsim-check", "ok" if ok else "missing scenarios")
        sys.exit(0 if ok else 1)
    sys.exit("usage: cllib.py hugsim-check <out> <scen_file>")
