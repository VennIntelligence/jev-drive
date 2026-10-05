"""Per-tick longitudinal dump of the 25 B2D turn windows (decision 128 point 5): joins ticks.jsonl and plans.jsonl of every
entered turn, so crawl.py can attribute each stopped interval. Runs on the box (needs the attempt dirs).

    .venv/bin/python experiments/op_route_ft/scripts/crawl_dump.py --out $DATA_DIR/runs/op_route_ft/crawl_dump.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
import cllib as C  # noqa: E402
import junction_forced_report as F  # noqa: E402
import junction_rig122_report as J  # noqa: E402
import junction_cl_report as R  # noqa: E402

ARMS = {"shipped": None, "rc-ctl-s0": None, "rc-bear-s0": None, "rc-poly-s0": None, "rc-all-s0": None, "zones-on": "olz", "vmin0": None}
PRE_S, WIN_S = 6.0, 15.0


def rows(p):
    return [json.loads(x) for x in open(p)]


def dirs_of(arm):
    if arm == "vmin0":
        return sorted((C.DATA / "runs/op_route_ft/vmin0/b2d").glob("turns-s%d-k*" % C.TURN_SEED))
    if arm == "zones-on":
        return sorted(Path(C.CACHE_TURNS).glob("ol-s%d-k*" % C.TURN_SEED))
    return C.b2d_dirs(arm, "subset", "turns", C.TURN_SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="", help="comma list, default all")
    ap.add_argument("--routes", default="", help="comma list of route ids, default the 20 turn routes")
    a = ap.parse_args()
    arms = a.arms.split(",") if a.arms else list(ARMS)
    routes = a.routes.split(",") if a.routes else C.TURN_ROUTES
    lab = F.load_labels()
    out, geo = [], {}
    for arm in arms:
        dirs = dirs_of(arm)
        for rid in routes:
            att, _ = C.attempt_of(dirs, rid)
            if att is None or not (att / "route.json").exists() or not (att / "ticks.jsonl").exists():
                continue
            g = geo.setdefault(rid, J.route_geometry(att, rid, lab))
            D, gd, psiD, turns = g
            tk = [t for t in rows(att / "ticks.jsonl") if "truth" in t]
            tr = np.array([t["truth"][:2] for t in tk])
            yaw = np.array([t["truth"][2] for t in tk])
            pl = {p["frame"]: p for p in rows(att / "plans.jsonl")}
            near, dev = R.project(tr, D)
            for T in turns:
                if (rid, T["mi"]) not in set(C.turn_keys()):
                    continue
                e = R.eval_turn(tr, yaw, D, psiD, T)
                L = lab[(rid, T["mi"])]
                rec = dict(arm=arm, route=rid, turn=T["mi"], forced=int(L["forced"]), kind=L["kind"], rmin=T["rmin"], angle=T["angle"],
                           entered=bool(e["entered"]), took=e["branch"] == "yes", i0=T["i0"])
                if e["entered"]:
                    s0, s1 = e["span"]
                    t0 = tk[s0]["t"]
                    i_pre = max(0, int(s0 - PRE_S * 20))
                    i_end = min(s0 + int(WIN_S * 20), len(tk) - 1)
                    tick = []
                    for i in range(i_pre, i_end + 1):
                        t = tk[i]
                        p = pl.get(t["frame"], {})
                        c = p.get("ctx") or t.get("ctx") or {}
                        tick.append(dict(t=round(t["t"] - t0, 2), v=t["v"], thr=t.get("throttle"), brk=t.get("brake"), reason=t.get("reason"),
                                         arc=round(float((near[i] - T["i0"]) * 0.25), 1), dev=round(float(dev[i]), 2),
                                         src=p.get("src"), latch=p.get("latch"), rel=p.get("rel"), rb=p.get("rb"), go=p.get("go"), zone=p.get("zone"),
                                         lat=p.get("lat"), lat_why=p.get("lat_why"), vplan=p.get("vplan"), aplan=p.get("aplan"), act_a=p.get("act_a"),
                                         act_k=p.get("act_k"), s=p.get("s"), s2=p.get("s2"), lead=p.get("lead"), lp=p.get("lp"), gas=p.get("gas"), brkp=p.get("brk"),
                                         junc=c.get("junc"), tl=c.get("tl"), tl_dist=c.get("tl_dist"), lead_gap=c.get("lead_gap"), lead_v=c.get("lead_v"),
                                         warm=p.get("warm"), near=c.get("near"), ped_gap=c.get("ped_gap"), ped_v=c.get("ped_v"), desire=p.get("desire"), cmd=p.get("cmd"), pose_v=p.get("pose_v")))
                    rec.update(span_s=round(float(tk[s1]["t"] - t0), 1), win_i=int(s0 - i_pre), ticks=tick)
                out.append(rec)
    Path(a.out).write_text(json.dumps(out, default=float))
    print(len(out), "turns", sum(r["entered"] for r in out), "entered")


if __name__ == "__main__":
    main()
