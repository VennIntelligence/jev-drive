"""B2D read of decision 149: does steering from the smoothed plan curvature (preset spec_plan_smooth) make openpilot turn at junctions?

Arms: P2-F-s0, P2H10-F-s0 (op_parity checkpoints: the arm's serving ONNX + bias server, ego inputs from lib/op_arb_agent.parity_ego) and shipped
cinque, each under `action` (op_arb.sh arm spec) or `plan_smooth` (arm spec_plan_smooth). Everything else is decision 127's `olnz`: zones off,
open-loop camera (1.59, 0, 1.86), route turn desire on, tm seed 2. Units are GPU-pool jobs (cllib.b2d_unit, 3 routes per unit); the shipped
`action` reference is decision 127's cached olnz run. Resumable.

  b2d_plan_smooth.py run    [--routes small|all|b2d220|ID,ID,..] [--arms P2-F-s0,P2H10-F-s0,cinque] [--execs action,plan_smooth] [--wait] [--dry-run]
  b2d_plan_smooth.py gif    --gif ARM:EXEC:ROUTE[,..] [--wait]   one recorded run (chase camera + every model input frame dump) per case, then
                            experiments/op_closed_loop/scripts/junction_forced_gif.py <attempt> <dump> <out.gif> --route-turn <mi>
  b2d_plan_smooth.py report [--routes ...] [--out DIR]     per-turn table (class, requested curvature vs needed) + route DS, markdown + csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_closed_loop/scripts"),
                str(REPO / "experiments/vlm_arb/scripts")]
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

DATA = G.data_dir() / "runs/op_parity/b2d_plan_smooth"
ONNX = G.data_dir() / "runs/op_parity/hugsim/onnx"
ROUTES = {"small": "28008 5423 10255".split(), "all": list(C.TURN_ROUTES)}
ROUTES["b2d220"] = sorted(__import__("jevdrive.data.splits", fromlist=["load"]).load("b2d/bench2drive220").members)   # the 220 DS routes (B2D P2 prereg)
ARM_ARB = {"action": "spec", "plan_smooth": "spec_plan_smooth"}


def route_list(s):
    return ROUTES.get(s) or s.split(",")


def units(arms, execs, routes):
    out = []
    for arm in arms:
        for ex in execs:
            if arm == "cinque" and ex == "action":
                continue                                    # decision 127's cached olnz run
            onnx = None if arm == "cinque" else str(ONNX / f"pp-{arm}.onnx")
            env = {} if arm == "cinque" else {"PARITY_TAG": arm}
            for k, ids in C.shards(routes, "all"):
                out.append(C.b2d_unit("ps-%s-%s-k%d" % (arm.replace("-F-s0", "").replace("cinque", "C"), ex[:3], k), ids,
                                      DATA / "arms" / ("%s-%s-s%d-k%d" % (arm, ex, C.TURN_SEED, k)), C.TURN_SEED, True, onnx, priority=3,
                                      arm=ARM_ARB[ex], env=env))
    return out


def gif_units(spec):
    """ARM:EXEC:ROUTE -> one-route record unit (rft_gif_lane.gif_unit's recipe: vlm_arb record agent + img_cl_server frame dump)."""
    out = []
    for g in spec.split(","):
        arm, ex, rid = g.split(":")
        onnx = None if arm == "cinque" else str(ONNX / f"pp-{arm}.onnx")
        tag = "%s-%s-%s" % (arm, ex, rid)
        u = C.b2d_unit("psg-%s-%s-%s" % (arm.replace("-F-s0", "").replace("cinque", "C"), ex[:3], rid), [rid], DATA / "gif" / tag, C.TURN_SEED, True, onnx,
                       priority=3, arm=ARM_ARB[ex], env={} if arm == "cinque" else {"PARITY_TAG": arm})
        u["env"].update(OP_ARB_AGENT="experiments/vlm_arb/scripts/vlm_arb_record_agent.py", GIF_BASE="vlm", VLM_ARM="drive",
                        SRV_PY="experiments/op_img_cmd/scripts/img_cl_server.py", IMG_CL_DUMP=str(DATA / "gif" / "dump" / tag), IMG_CL_DUMP_EVERY="1")
        out.append(u)
    return out


def attempts(arm, ex, routes):
    """rid -> attempt dir of an arm (the shipped action arm = the rig122 olnz cache)."""
    if (arm, ex) == ("cinque", "action"):
        root, pat = C.CACHE_TURNS, "olnz-s%d-k*" % C.TURN_SEED
    else:
        root, pat = DATA / "arms", "%s-%s-s%d-k*" % (arm, ex, C.TURN_SEED)
    out = {}
    for rid in routes:
        for u in sorted(root.glob(pat)):
            f = u / "done" / (rid + ".json")
            if f.exists():
                out[rid] = u / "attempts" / rid / str(json.loads(f.read_text()).get("attempt", 1))
                break
    return out


def classify(r):
    """One class per labelled turn: turned / collision / off-route / went straight / never entered."""
    if not r["entered"]:
        return "never entered"
    if r["branch"] == "yes":
        return "turned"
    if r["coll"] > 0:
        return "collision"
    return "off-route" if r["leaves"] == 1 else "went straight"


def report(arms, execs, routes, out):
    import junction_forced_report as F
    import junction_rig122_report as J
    import numpy as np
    sys.path.insert(0, str(REPO / "experiments/op_route_ft/scripts"))
    import rft_split as RS                                   # steering split of a turn: signed requested curvature, turn-in arc, cause
    rows_of = RS.rows_of
    lab = F.load_labels()
    keys = set(C.turn_keys())
    rows, ds = [], []
    for arm in arms:
        for ex in execs:
            for rid, att in attempts(arm, ex, routes).items():
                if not (att / "route.json").exists():
                    continue
                s = J.score_attempt(rid, f"{arm}/{ex}", att, J.route_geometry(att, rid, lab), lab)
                if s is None:
                    continue
                ds.append(dict(arm=arm, exec=ex, route=rid, DS=round(s["rr"][0], 1), RC=round(s["rr"][1], 1), status=s["rr"][2],
                               hits=sum(c["kind"].startswith("hit") for c in s["cols"])))
                RS.rows_of = lambda q, ex=ex: [dict(p, act_k=p.get("k_sm", p["act_k"])) if ex == "plan_smooth" and Path(q).name == "plans.jsonl" else p
                                               for p in rows_of(q)]          # split_one reads act_k: for plan_smooth the executed curvature is k_sm
                sp = {d["turn"]: d for d in RS.split_one(rid, f"{arm}/{ex}", att, J.route_geometry(att, rid, lab), lab)}
                for r in s["rows"]:
                    if (r["route"], r["turn"]) not in keys:
                        continue
                    d = sp[r["turn"]]
                    sg = d.get("s_peak")
                    rows.append(dict(arm=arm, exec=ex, route=rid, turn=r["turn"], kind=r["kind"], forced=r["forced"], angle=r["angle"], need=r["need"],
                                     req_peak=None if sg is None else round(sg, 3), ratio=None if sg is None else round(sg / r["need"], 2),
                                     turn_in_m=d.get("tin_m"), v_med=d.get("v_med"), stop_frac=d.get("stop_frac"), cause=RS.classify(d),
                                     cls=classify(r), vmin=r["vmin"], coll=r["coll"]))
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("turns", rows), ("routes", ds)):
        if data:
            with open(out / f"b2d_plan_smooth_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, list(data[0]))
                w.writeheader()
                w.writerows(data)
    L = ["| arm | exec | turns | turned | went straight | off-route | collision | never entered | req / need median | route DS mean (n) |", "|" + "---|" * 10]
    for arm in arms:
        for ex in execs:
            R = [r for r in rows if r["arm"] == arm and r["exec"] == ex]
            D = [d["DS"] for d in ds if d["arm"] == arm and d["exec"] == ex]
            if not R:
                continue
            n = lambda c: sum(r["cls"] == c for r in R)  # noqa: E731
            rat = [r["ratio"] for r in R if r["ratio"] is not None]
            L.append("| %s | %s | %d | %d | %d | %d | %d | %d | %s | %s |" % (arm, ex, len(R), n("turned"), n("went straight"), n("off-route"), n("collision"),
                     n("never entered"), "%.2f" % np.median(rat) if rat else "n/a", "%.1f (%d)" % (np.mean(D), len(D)) if D else "n/a"))
    L += ["", "| arm | exec | route:turn | kind | need 1/m | signed req peak 1/m | ratio | class | steering cause | turn-in m | v median | stopped frac |", "|" + "---|" * 12]
    for r in rows:
        L.append("| %s | %s | %s:%s | %s | %.3f | %s | %s | %s | %s | %s | %s | %s |" % (r["arm"], r["exec"], r["route"], r["turn"], "forced" if r["forced"] else "choice",
                 r["need"], r["req_peak"], r["ratio"], r["cls"], r["cause"], r["turn_in_m"], r["v_med"], r["stop_frac"]))
    (out / "b2d_plan_smooth_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "report", "gif"))
    ap.add_argument("--routes", default="small")
    ap.add_argument("--arms", default="P2-F-s0,P2H10-F-s0,cinque")
    ap.add_argument("--execs", default="action,plan_smooth")
    ap.add_argument("--gif", default="")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(REPO / "experiments/op_parity/results/b2d_plan_smooth"))
    a = ap.parse_args()
    arms, execs, routes = a.arms.split(","), a.execs.split(","), route_list(a.routes)
    if a.cmd in ("run", "gif"):
        us = units(arms, execs, routes) if a.cmd == "run" else gif_units(a.gif)
        ids = C.submit_units(us, DATA / "pool", "op_parity b2d_plan_smooth", dry_run=a.dry_run)
        if a.wait and not a.dry_run:
            bad = C.wait_units(ids)
            print("units not done: %s" % bad if bad else "all units done", flush=True)
            sys.exit(1 if bad else 0)
    else:
        report(arms, execs, routes, Path(a.out))


if __name__ == "__main__":
    main()
