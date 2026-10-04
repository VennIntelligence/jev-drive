#!/usr/bin/env python
"""Route-choice fine-tunes on decision 127's 25 B2D junction turns (results/harness.md): the guard line `b2d_turns`
(experiments/op_guard/scripts/line_b2d_turns.py: op_arb.sh arm `spec` at the open-loop-aligned camera, zones off
`"zones": false, "div_m": 1e9`, turn desire on, seed 2, 20 val routes = rig122 arm olnz) run per candidate ONNX, plus a per-candidate
per-turn report against shipped.

A candidate is a serving ONNX from route_onnx.py build; its adapter is `<stem>.adapter.npz` next to it (guardlib.resolve picks it up and
the B2D units pass it to the agent as "route_adapter"); no adapter file = zero bias (arm rc-ctl). Shipped is decision 127's olnz run itself
(cllib.turns_cached: same agent config, shipped server; a candidate run differs only by SRV_ONNX and route_adapter).

  .venv/bin/python experiments/op_route_ft/scripts/route_turns.py run X.onnx [Y.onnx ...] [--cards 1] [--stage smoke|all]
        [--wait-h 3] [--force]
      leases lane `op-route-ft-turns` (retries every 5 min up to --wait-h while no card is free), runs the candidates one after the
      other through it (OP_GUARD_LANE), releases it, then writes the reports. Run it in tmux (scripts/tmux_run.sh).
  .venv/bin/python experiments/op_route_ft/scripts/route_turns.py report X.onnx [...]      reports only (finished units)

Outputs: units and line files under $DATA_DIR/runs/op_guard/<stem>/subset/ (b2d/turns-s2-k*/, lines/b2d_turns.json); the report
$DATA_DIR/runs/op_route_ft/turns/<stem>.{json,md} (+ shipped.{json,md}): per turn took exit / leaves lane / collisions / forced / R_min,
the paired took-exit difference vs shipped, and per route a 5 Hz step series (t, v, route index, desired curvature act_k,
adapter features) in the json
for BEV panels / GIFs. smoke = route 10255 (one choice turn) only, line not written, report from what exists.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GUARD = REPO / "experiments/op_guard/scripts"
sys.path[:0] = [str(GUARD), str(REPO)]
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

LANE = "op-route-ft-turns"
PY = str(REPO / ".venv/bin/python")
OUT = G.data_dir() / "runs/op_route_ft/turns"


def lanes(*a):
    return subprocess.run([PY, "-m", "jevdrive.cl", *a], cwd=REPO, text=True, capture_output=True)


def lease(cards, wait_h):
    t_end = time.time() + 3600 * wait_h
    while True:
        r = lanes("lease", LANE, "--gpus", str(cards), "--status", "op_route_ft 25 junction turns")
        print(r.stdout + r.stderr, flush=True)
        if r.returncode == 0:
            return
        if time.time() > t_end:
            raise SystemExit("no free card for %.1f h" % wait_h)
        time.sleep(300)


def run(a):
    lease(a.cards, a.wait_h)
    env = dict(os.environ, OP_GUARD_LANE=LANE)
    rcs = {}
    try:
        for x in a.cand:
            cmd = [PY, str(GUARD / "line_b2d_turns.py"), "--candidate", x, "--mode", "subset", "--stage", a.stage] + (["--force"] if a.force else [])
            print(" ".join(cmd), flush=True)
            rcs[x] = subprocess.run(cmd, cwd=REPO, env=env).returncode
    finally:
        print(lanes("release", LANE, "done: op_route_ft turns").stdout, flush=True)
    report(a)
    print("line rc", rcs)
    return int(any(rcs.values()))


# ---------------------------------------------------------------- report
def steps(att, every=4):
    """Per-plan series of one attempt at 5 Hz (every 4th 20 Hz plan): t, v, route index ri, desired curvature k (act_k), desire, lateral
    owner, adapter features f (None without an adapter)."""
    out = []
    for i, line in enumerate(open(att / "plans.jsonl")):
        if i % every:
            continue
        p = json.loads(line)
        if p.get("warm"):
            continue
        out.append(dict(t=round(p["t"], 2), v=round(p["v"], 2), ri=p.get("ri"), k=p.get("act_k"), desire=p.get("desire"), lat=p.get("lat"),
                        f=(p.get("ra") or {}).get("f")))
    return out


def one(cand):
    """(name, rows, collisions, per_route, attempts, route_adapter) of a candidate ('shipped' or an ONNX path)."""
    if cand == G.SHIPPED:
        name, ra = G.SHIPPED, None
    else:
        c = G.resolve(cand)
        name, ra = c["name"], c["route_adapter"]
    rows, cols, routes, src = C.turn_rows(name, "subset")
    return name, rows, cols, routes, src, ra


def report(a):
    sys.path[:0] = [str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
    OUT.mkdir(parents=True, exist_ok=True)
    keys = C.turn_keys()
    sname, srows, scols, sroutes, ssrc, _ = one(G.SHIPPED)
    sx = {(r["route"], r["turn"]): r for r in srows}
    for cand in [G.SHIPPED] + list(a.cand):
        name, rows, cols, routes, src, ra = one(cand)
        ix = {(r["route"], r["turn"]): r for r in rows}
        have = [k for k in keys if k in ix]
        res = dict(candidate=name, onnx=None if cand == G.SHIPPED else G.resolve(cand)["onnx"], route_adapter=ra,
                   arm="op_arb.sh spec, zones off, open-loop camera, desire on, seed 2 (= rig122 olnz)", turns=len(keys), scored=len(have),
                   per_turn=[], collisions=cols, per_route=routes, attempts=src)
        for k in keys:
            r, s = ix.get(k), sx.get(k)
            d = dict(route=k[0], turn=k[1])
            if r is not None:
                d.update({x: r[x] for x in ("forced", "kind", "angle", "rmin", "need", "entered", "branch", "peak", "head_pk", "vmin", "coll", "red", "leaves")})
                d["took"] = int(r["branch"] == "yes")
            if s is not None:
                d["shipped_took"] = int(s["branch"] == "yes")
            res["per_turn"].append(d)
        if a.steps:                                      # per route, for BEV panels / GIFs (turn geometry: route.json of the attempt)
            res["steps"] = {rid: steps(Path(p)) for rid, p in src.items()}
        summ = {}
        for g, sel in (("choice", lambda d: d.get("forced") == 0), ("forced", lambda d: d.get("forced") == 1), ("all", lambda d: "forced" in d)):
            dd = [d for d in res["per_turn"] if sel(d)]
            ent = [d for d in dd if d["entered"]]
            pair = [d for d in dd if "shipped_took" in d]
            summ[g] = dict(n=len(dd), took=sum(d["took"] for d in dd), leaves=sum(int(d["leaves"] == 1) for d in ent), entered=len(ent),
                           coll=sum(d["coll"] for d in dd), shipped_took=sum(d["shipped_took"] for d in pair),
                           gained=sum(d["took"] > d["shipped_took"] for d in pair), lost=sum(d["took"] < d["shipped_took"] for d in pair))
        if cand != G.SHIPPED and len(have) == len(keys) and all(k in sx for k in keys):
            y = [ix[k]["branch"] == "yes" for k in keys]
            yr = [sx[k]["branch"] == "yes" for k in keys]
            summ["all"]["paired_diff"], summ["all"]["paired_ci"] = C.paired_ci(y, yr, [k[0] for k in keys])
        res["summary"] = summ
        res["route_ds"] = sum(v["ds"] for v in routes.values()) / max(len(routes), 1)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
        (OUT / f"{name}.md").write_text(md(res))
        print("wrote", OUT / f"{name}.md", json.dumps(summ, default=float))


def md(r):
    s = r["summary"]
    L = ["# %s: 25 B2D junction turns (decision 127 set)" % r["candidate"], "",
         "ONNX `%s`, route adapter `%s`. Arm: %s. Scored %d / %d turns; route DS mean %.1f over %d routes." %
         (r["onnx"], r["route_adapter"], r["arm"], r["scored"], r["turns"], r["route_ds"], len(r["per_route"])), "",
         "| turns | n | took exit | shipped took | gained / lost vs shipped | leaves lane (of entered) | collisions in window |", "|---|--:|--:|--:|--:|--:|--:|"]
    for g in ("choice", "forced", "all"):
        x = s[g]
        L.append("| %s | %d | %d | %d | %d / %d | %d / %d | %d |" % (g, x["n"], x["took"], x["shipped_took"], x["gained"], x["lost"], x["leaves"],
                                                                     x["entered"], x["coll"]))
    if s["all"].get("paired_diff") is not None:
        lo, hi = s["all"]["paired_ci"]
        L += ["", "Took-exit rate minus shipped, paired, route-cluster bootstrap: %+.3f [%+.3f, %+.3f] (n 25, one seed)." % (s["all"]["paired_diff"], lo, hi)]
    L += ["", "| route | turn | forced | kind | angle | R_min m | entered | branch | took | shipped took | leaves | peak m | head peak 1/m | need 1/m | coll |",
          "|---|--:|--:|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|"]
    for d in r["per_turn"]:
        if "forced" not in d:
            L.append("| %s | %d | not scored |" % (d["route"], d["turn"]))
            continue
        L.append("| %s | %d | %d | %s | %.0f | %.1f | %d | %s | %d | %s | %s | %.2f | %s | %.3f | %d |" % (
            d["route"], d["turn"], d["forced"], d["kind"], d["angle"], d["rmin"], d["entered"], d["branch"], d["took"], d.get("shipped_took", "-"),
            "-" if d["leaves"] != d["leaves"] else int(d["leaves"]), d["peak"], d["head_pk"], d["need"], d["coll"]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "report"):
        p = sp.add_parser(n)
        p.add_argument("cand", nargs="+", help="serving ONNX paths (route_onnx.py build)")
        p.add_argument("--no-steps", dest="steps", action="store_false", help="leave the per-plan step series out of the json")
        if n == "run":
            p.add_argument("--cards", type=int, default=1)
            p.add_argument("--stage", choices=("smoke", "all"), default="all")
            p.add_argument("--wait-h", type=float, default=3.0)
            p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == "run" else report(a) or 0)
