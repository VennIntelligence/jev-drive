"""CARLA / Bench2Drive closed-loop readouts: a thin layer over what docs/closed-loop-runbook.md standardises (scripts/b2d_run.py,
`jevdrive.cl.pool.b2d_cmd`, the GPU pool's CARLA placement). Nothing here starts a server or picks a port: every worker job is one
b2d_run invocation on the pool's placeholders, and all K jobs share one output dir, so b2d_run's own route claims
(claims/<id>.lock, done/<id>.json) spread the routes over the cards and a rerun skips finished routes.

Generic on purpose: any leaderboard agent (a .py with get_entry_point(), its config, its python) plugs in through `B2DAgent`;
a data collector (e.g. PDM-Lite with a recording agent) is the same call with its agent / extra b2d_run args.

    from jevdrive.bench import b2d
    a = b2d.B2DAgent("pdm-lite", agent="", agent_config="")          # "" = b2d_run's default stub / built-in expert
    b2d.submit(a, routes="bench2drive220", route_ids=None, workers=6)   # -> run dir; units.csv on collect
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import runner as R
from .models import data_dir

B2D220 = "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
INFRACTIONS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle", "red_light", "stop_infraction", "outside_route_lanes",
               "min_speed_infractions", "yield_emergency_vehicle_infractions", "scenario_timeouts", "route_dev", "vehicle_blocked",
               "route_timeout")


@dataclass
class B2DAgent:
    name: str                       # run-dir name
    agent: str = ""                 # leaderboard agent .py ("" = b2d_run's default)
    agent_config: str = ""          # its config / checkpoint
    python: str = ""                # interpreter of the agent's env ("" = envs/carla)
    extra: list = field(default_factory=list)   # further b2d_run.py args (e.g. --rig, --record-dir, --drive)
    vram_per_worker: float = 9.0    # GB per CARLA server + its client (runbook: 8-10 GB with six cameras)
    cpu_per_worker: float = 2.0
    env: dict = field(default_factory=dict)


def route_ids(routes: str, ids=None) -> tuple:
    """(routes xml, ids). routes = 'bench2drive220' | a jevdrive.data.splits b2d split ('b2d/dev10') | an xml path."""
    xml = str(data_dir() / B2D220)
    if routes in ("", "bench2drive220", "b2d220"):
        rid = ids
    elif routes.endswith(".xml"):
        xml, rid = routes, ids
    else:
        from ..data import splits
        s = splits.load(routes if "/" in routes else f"b2d/{routes}")
        rid = list(ids or s.members)
    if rid is None:
        import re
        rid = re.findall(r'<route[^>]*\bid="(\d+)"', Path(xml).read_text())
    return xml, [str(r) for r in rid]


def stages(a: B2DAgent, run_dir: Path, routes: str = "bench2drive220", ids=None, workers: int = 6, jobs: int = 0, tm_seed: int = 0,
           max_attempts: int = 3) -> list:
    from ..cl import pool as P
    xml, rid = route_ids(routes, ids)
    run_dir = Path(run_dir)
    out = run_dir / "b2d"
    todo = [r for r in rid if not (out / "done" / f"{r}.json").exists()]
    w = max(1, min(workers, len(todo) or 1))
    k = jobs or max(1, min(len(R.box_cards()), -(-len(todo) // w)))
    R.atomic_write(run_dir / "config.json", json.dumps(dict(agent=a.__dict__, routes=xml, ids=rid, workers=w, jobs=k, tm_seed=tm_seed),
                                                       indent=1))
    S = []
    for i in range(k if todo else 0):
        cmd = P.b2d_cmd(out, rid, routes=xml, agent=a.agent, agent_config=a.agent_config, python=a.python or None, tm_seed=tm_seed,
                        max_attempts=max_attempts, extra=a.extra)
        # every job gets the whole id list; b2d_run's claims hand each route to one card; done marker per job:
        cmd = " ".join(_q(x) for x in cmd) + f" && touch {_q(str(run_dir / f'w{i}.DONE'))}"
        S.append(R.Stage(f"w{i}", cmd, done=str(run_dir / f"w{i}.DONE"), vram=a.vram_per_worker * w, cpu=int(a.cpu_per_worker * w + 2),
                         ram=8 * w, env=dict(a.env), tries=1, carla=w))
    S.append(R.Stage("collect", R.stage_cmd("jev", "b2d-collect", run_dir), done=str(run_dir / "DONE"), vram=0.5, cpu=2, ram=8,
                     after=[s.name for s in S]))
    return S


def submit(a: B2DAgent, routes: str = "bench2drive220", route_ids=None, workers: int = 6, jobs: int = 0, tm_seed: int = 0, max_attempts: int = 3,
           run_dir=None, dry: bool = False, priority: float = 0.0) -> Path:
    """The call the module docstring names (first real run 2026-10-07, route 24211 with the default agent: DS 50, units.csv written): every
    stage of one agent on a route list to the pool at once; returns the run dir (default $DATA_DIR/runs/bench/b2d/<agent name>). Idempotent."""
    d = Path(run_dir) if run_dir else R.bench_root("b2d", a.name)
    R.submit(d, f"bn-b2d-{a.name}", stages(a, d, routes, route_ids, workers, jobs, tm_seed, max_attempts), dry=dry, priority=priority)
    return d


def _q(x: str) -> str:
    import shlex
    x = str(x)
    return f'"{x}"' if x.startswith("{") and x.endswith("}") else shlex.quote(x)   # pool placeholders: filled before bash parses


def route_row(out: Path, rid: str):
    """Official result of one finished route (vlm_arb_common.route_row): DS, RC, status, infraction counts; None if unfinished."""
    done = out / "done" / f"{rid}.json"
    if not done.exists():
        return None
    a = out / "attempts" / rid / str(json.loads(done.read_text()).get("attempt", 1))
    if not (a / "results.json").exists():
        return None
    recs = json.loads((a / "results.json").read_text())["_checkpoint"]["records"]
    if not recs:
        return None
    rec = recs[0]
    st = rec["status"]
    row = dict(route=rid, status=st, crash=st == "Failed" or any(m in st for m in ("Simulation crashed", "Agent crashed",
               "Agent couldn't be set up", "Agent's sensors were invalid")), DS=float(rec["scores"]["score_composed"]),
               RC=float(rec["scores"]["score_route"]), attempt=str(a))
    for k in INFRACTIONS:
        row[k] = len(rec.get("infractions", {}).get(k, []))
    return row


def collect(run_dir: str) -> None:
    import pandas as pd
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    out = run_dir / "b2d"
    rows = [r for r in (route_row(out, i) for i in cfg["ids"]) if r]
    u = pd.DataFrame(rows)
    u.to_csv(run_dir / "units.csv", index=False)
    summ = dict(agent=cfg["agent"]["name"], n=len(u), requested=len(cfg["ids"]), DS=float(u.DS.mean()) if len(u) else float("nan"),
                RC=float(u.RC.mean()) if len(u) else float("nan"), crashes=int(u.crash.sum()) if len(u) else 0)
    sj = out / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text())
        summ |= {k: s.get(k) for k in ("routes_never_finished", "restarts")}
    R.atomic_write(run_dir / "summary.json", json.dumps(summ, indent=1, default=str))
    R.status(run_dir, f"done: {len(u)} / {len(cfg['ids'])} routes, DS {summ['DS']:.2f}")
    R.atomic_write(run_dir / "DONE", json.dumps(dict(t=time.strftime("%F %T"), **summ), default=str) + "\n")
