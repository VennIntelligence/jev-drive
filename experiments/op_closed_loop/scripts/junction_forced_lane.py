"""Forced turns and the sky arrow in closed loop (results/junction_forced_and_arrow.md).

  scripts/tmux_run.sh jfa .venv/bin/python -m jevdrive.cl run experiments/op_closed_loop/scripts/junction_forced_lane.py \
      --lane jfa --workers-per-card 6 --arg stage=smoke|few|forced|all [--arg seeds=2] [--arg arms=A,D,N,Z,NA,SA,NA0] [--arg routes=<ids>]
      [--arg gif=<arm>:<route>:<xml>]            (stage=gif: one chase-camera unit with every overlaid model frame dumped)

All arms are the shipped `drive` agent (lib/vlm_arb_agent.py -> op_arb_agent.py), never "resume": "nored" (no privileged light), `zones: false`
and `div_m: 1e9` everywhere but A, so only the action head steers. `desire` is the route desire input of the head (A, D as in jcl) or off (the
image-command arms and their controls):
  A    shipped drive: dense route steers in the junction zones and on divergence (route desire on)
  D    action head steers everywhere, route desire on (= jclD of junction_cl_lane.py)
  N    D with the route desire off, no command at all (control of the arrow arms)
  Z    N + the sky arrow drawn on every frame, shipped Cinque (zero-shot)
  NA   fine-tune q3NA-s0 + arrow (op_img_cmd decision 102), desire off
  SA   fine-tune q3SA-s0 + arrow, desire off
  NA0  q3NA-s0 without the arrow (separates the fine-tune from the arrow)
The arrow comes from lib/op_arb_agent.py img_cmd(): next LEFT / RIGHT / STRAIGHT command of the official route and the route distance to it
(none beyond 60 m, none on plain road curves: the leaderboard route carries no turn command there). No HD geometry.
Routes: the 36 routes of the 38 earlier turns + every route holding a forced turn (junction_forced_turns.csv); A and D rerun only on the
new routes (the 36 already exist as v2-jclA/D-s2-q*, same seed). Bench2Drive val and the 220 exam xml are separate units (B2D_XML).
Units v2-jfa<arm>-s<seed>-<shard> under $DATA_DIR/runs/vlm_arb/arms; hand-offs in $DATA_DIR/runs/jfa.
"""
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402,F401  (sets SLOT_WORKERS / SLOT_CORES = 3, 9)
import vmerge_chain as vm  # noqa: E402

NAME, ROOT = "jfa", "jfa"
R36 = ("10255 10364 15102 24944 25051 27043 27297 27870 28147 28180 34183 34391 35330 4104 4183 4721 5423 6999 7157 7616 7841 8859 "
       "9102 9196 9218 9646 35243 334 17280 15612 16390 15483 16508 16529 26872 27787").split()
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
XML_DIR = DATA / "third_party/Bench2Drive/leaderboard/data"
XMLS = {"v": "bench2drive_0.0.4_val.xml", "x": "bench2drive220.xml"}
ONNX = DATA / "runs/op_img_cmd/cl/onnx"
SRV = "experiments/op_img_cmd/scripts/img_cl_server.py"
NOZ = '"zones": false, "div_m": 1e9'
ARMS = {"A": dict(args="", desire=True), "D": dict(args=NOZ, desire=True), "N": dict(args=NOZ, desire=False),
        "Z": dict(args=NOZ + ', "img_cmd": true', desire=False, srv=SRV),
        "NA": dict(args=NOZ + ', "img_cmd": true', desire=False, srv=SRV, onnx="q3NA-s0"),
        "SA": dict(args=NOZ + ', "img_cmd": true', desire=False, srv=SRV, onnx="q3SA-s0"),
        "NA0": dict(args=NOZ, desire=False, onnx="q3NA-s0")}


def route_sets():
    """({xml key: routes} of the arrow arms, {xml key: new routes} of A / D): 36 earlier routes + routes with a forced turn."""
    rows = list(csv.DictReader(open(REPO / "experiments/op_closed_loop/results/junction_forced_turns.csv")))
    forced = {r["route"]: ("v" if r["xml"].startswith("bench2drive_0") else "x") for r in rows if r["forced"] == "1"}
    new = {"v": sorted(r for r, k in forced.items() if k == "v" and r not in R36), "x": sorted(r for r, k in forced.items() if k == "x")}
    return {"v": R36 + new["v"], "x": new["x"]}, new


def unit(arm, seed, shard, ids, xml, extra=None, kind="", prio=1, deps=(), tag="jfa"):
    c = ARMS[arm]
    e = dict(OP_DET_HEAD=vm.DET, B2D_XML=str(XML_DIR / XMLS[xml]), DESIRE="true" if c["desire"] else "false")
    if c["args"]:
        e["DRIVE_ARGS"] = c["args"]
    if c.get("srv"):
        e["SRV_PY"] = c["srv"]
    if c.get("onnx"):
        e["SRV_ONNX"] = str(ONNX / (c["onnx"] + ".onnx"))
    e.update(extra or {})
    return base.unit(tag + arm, seed, shard, ids, e, kind, prio, deps=deps, base="drive")


def split(ids, n):
    return [ids[k::n] for k in range(n) if ids[k::n]]


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "2").split(",")]
    arms = args.get("arms", "N,Z,NA,SA,NA0,A,D").split(",")
    stage = args.get("stage", "smoke")
    allr, new = route_sets()
    dump = lambda a, sub, every: dict(IMG_CL_DUMP=str(DATA / "runs/jfa/dump" / sub / a), IMG_CL_DUMP_EVERY=str(every)) if ARMS[a].get("srv") else {}  # noqa: E731
    if stage == "gif":                                      # gif=<arm>:<route>:<v|x>[,...]: chase camera + every overlaid frame, seed 2
        out = []
        for g in args["gif"].split(","):
            a, rid, xml = g.split(":")
            out.append(unit(a, seeds[0], "g" + rid, [rid], xml, dict(OP_ARB_AGENT="experiments/vlm_arb/scripts/vlm_arb_record_agent.py", GIF_BASE="vlm",
                                                                    **dump(a, "gif_" + rid, 1)), prio=0, tag="jfagif"))
        return out
    smoke = [unit("NA", 2, "dbg", ["28180"], "v", dump("NA", "smoke", 100), tag="jfadbg"),         # a forced T-stem
             unit("D", 2, "dbg", ["28180"], "v", tag="jfadbg")]
    if stage == "smoke":
        return smoke
    few_ids = ["28180", "24944", "27297", "9196", "6999", "34183"]                              # 2 forced + 4 choice routes, all val
    few = [unit(a, 2, "few", few_ids, "v", dump(a, "few", 100), tag="jfafew") for a in ("Z", "NA", "SA", "N") if a in arms]
    if stage == "few":
        return smoke + few
    if stage == "forced":                                   # D and A on chosen forced-turn routes (arg routes=a,b,c; all val), 2 routes per unit
        rids = args["routes"].split(",")
        return [unit(a, s, "f%d" % i, ids, "v") for s in seeds for a in ("D", "A") for i, ids in enumerate(split(rids, (len(rids) + 1) // 2))]
    out = list(smoke) + few
    dep = [j.name for j in smoke]
    for s in seeds:
        for a in arms:
            if a in ("A", "D"):
                out += [unit(a, s, "f%s%d" % (k, i), ids, k, deps=dep) for k in "vx" for i, ids in enumerate(split(new[k], 2 if k == "x" else 1))]
            else:
                out += [unit(a, s, "q%d" % i, ids, "v", dump(a, "all", 400), deps=dep) for i, ids in enumerate(split(allr["v"], 6))]
                out += [unit(a, s, "x%d" % i, ids, "x", dump(a, "all", 400), deps=dep) for i, ids in enumerate(split(allr["x"], 2))]
    return out
