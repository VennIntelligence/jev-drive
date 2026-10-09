#!/usr/bin/env python3
"""Lane COL1, nuPlan half: every at-fault collision of the lambda-10 recipes on the 1491 public AlpaSim scenes, read back from finished runs.
Runs on the box with AlpaSim's venv ($DATA_DIR/third_party/alpasim/.venv/bin/python); state in $DATA_DIR/runs/alpasim/col1/.
Map, other actors and the logged future are privileged: labels only. WA-JEPA and the other drivers appear as reference columns only.

  inv      which run holds which scene for each driver, the at-fault-collision rollouts and whether their rollout.asl survived -> inv.json
  extract  c1_extract records (one_log) of every collision rollout with an .asl -> cases.pkl; add --rerun <manifest.json> to take the
           missing ones from re-runs ({driver: [run dirs]})
  tax      per-case measurements and class counts -> results/collisions/nuplan_cases.csv, nuplan_tables.md      (--out DIR)

Classes. N1 (decision 153, c1_review.collision): A1 stopped lead, A2 moving lead, B cut-in, C crossing, D side contact, E oncoming.
`kind` refines them with facts the N1 rule does not use:
  static     the struck object moves less than 1 m in the whole scene (parked car, queue that never moves, barrier)
  in_path    the object box overlaps the logged ego's swept corridor (ego width; the logged path extended 40 m straight beyond its end) at
             some time of the scene: it is in the lane the log drives
  kind       object slower than 0.5 m/s at impact: lead stopped = in_path; parked / static beside the path = static and not in_path;
             standing neighbour beside the path = the rest (it moved earlier in the scene: a queue in the next lane, a car that pulled up).
             Object moving at impact, by N1 class: slow lead = A2; cut-in = B; crossing = C; oncoming = E; side contact while turning = D
             and |log turn| >= 20 deg; side contact (moving neighbour) = the remaining D
Visibility: the first sim time (>= 0, 0.1 s grid) at which the object's centre is within 60 m of the CAM_F0 position and within the
camera's horizontal field (atan(960 / fx) = 31.4 deg) of the simulated ego heading; no occlusion test. TTC there = range / closing range
rate (inf when opening). Plan slowing: decision k's planned 4 s arc against 4 s at the speed the ego has at that decision.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RA = DATA / "runs/alpasim"
O = RA / "col1"
TARGETS = ["P2H10-F-s0", "P2H10-F-s1", "APY10m10-AB-s0", "APY10m10-AB-s1", "AP2H10-AB-s0", "AP2H10-AB-s1"]
REFS = ["WA-JEPA (reference)", "SH30-F-s0", "AP2-AB-s0", "OT30-F-s0", "OT30-F-s1", "YR10m10-F-s0", "YR10m10-F-s1"]
HFOV = float(np.degrees(np.arctan(960 / 1573.5)))
CAM_X = 1.786                                             # CAM_F0 ahead of the rear axle (docs/alpasim.md)
KINDS = ["lead stopped", "slow lead", "cut-in", "crossing", "parked / static beside the path", "standing neighbour beside the path", "oncoming",
         "side contact while turning", "side contact (moving neighbour)", "other"]


def manifests() -> dict:
    """driver -> run dirs, finished lanes only (cf1/b is running and is not read)."""
    o3, cf, c0 = (json.loads((RA / f).read_text()) for f in ("ot3/a/manifest.json", "cf1/a/manifest.json", "c0c/manifest.json"))
    M = {}
    for k in TARGETS + REFS:
        M[k] = o3.get(k, []) + o3.get(k + "@new", []) + cf.get(k + "@new", []) + cf.get(k + "@fresh", []) + c0.get(k, [])
    return M


def asl(run, scene):
    g = glob.glob(f"{run}/rollouts/{scene}/*/rollout.asl")
    return g[0] if g else None


def cmd_inv(a):
    M, inv = manifests(), dict(drivers={}, scores={}, cases=[])
    for k, dirs in M.items():
        seen = {}
        for D in dirs:
            for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]:
                seen.setdefault(r["clipgt_id"], (D, r))
        inv["scores"][k] = {s: [r["score"], [f for f in ("collision_at_fault", "offroad", "left_corridor_laterally") if r["score_metrics"].get(f)]]
                            for s, (D, r) in seen.items()}
        col = [(s, D) for s, (D, r) in seen.items() if r["score_metrics"].get("collision_at_fault")]
        have = [bool(asl(D, s)) for s, D in col]
        inv["drivers"][k] = dict(runs=dirs, scenes=len(seen), zeros=sum(r["score"] == 0 for _, r in seen.values()), collisions=len(col), asl=sum(have))
        print(f"{k:22s} scenes {len(seen):5d} zeros {inv['drivers'][k]['zeros']:4d} at-fault collision {len(col):3d} with rollout.asl {sum(have):3d}")
        if k in TARGETS:
            inv["cases"] += [dict(driver=k, scene=s, run=D, asl=h) for (s, D), h in zip(col, have)]
    O.mkdir(parents=True, exist_ok=True)
    (O / "inv.json").write_text(json.dumps(inv))
    miss = [c for c in inv["cases"] if not c["asl"]]
    print(len(inv["cases"]), "target collision rollouts,", len(miss), "without rollout.asl")
    for c in miss:
        print("  missing", c["driver"], c["scene"], c["run"])


def cmd_extract(a):
    import c1_extract as X
    inv = json.loads((O / "inv.json").read_text())
    re = json.loads(Path(a.rerun).read_text()) if a.rerun else {}
    jobs, src = [], {}
    for c in inv["cases"]:
        run, origin = c["run"], "original"
        if not c["asl"]:
            run = next((D for D in re.get(c["driver"], []) if asl(D, c["scene"])), None)
            origin = "re-run"
            if run is None:
                print("no log for", c["driver"], c["scene"])
                continue
        jobs.append((run, c["scene"]))
        src[len(jobs) - 1] = (c["driver"], c["scene"], run, origin)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(X.one_log, jobs))
    out = {}
    for i, (scene, o) in enumerate(res):
        drv, _, run, origin = src[i]
        S = {r["clipgt_id"]: r for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]}
        o["summary"], o["run"], o["origin"] = S[scene], run, origin
        o["rec"] = []
        out[drv, scene] = o
    by = defaultdict(set)
    for (drv, scene), o in out.items():
        by[o["run"]].add(scene)
    for run, sc in by.items():                            # the driver's own records (command, route0) of these scenes
        for line in open(f"{run}/driver-logs/drive.jsonl"):
            if '"drive"' in line:
                r = json.loads(line)
                if r["kind"] == "drive" and r["scene"] in sc:
                    r.pop("ms", None)
                    for (drv, scene), o in out.items():
                        if o["run"] == run and scene == r["scene"]:
                            o["rec"].append(r)
    for o in out.values():
        o["rec"].sort(key=lambda r: r["k"])
    pickle.dump(out, open(O / "cases.pkl", "wb"), protocol=4)
    print("extract", len(out), "cases ->", O / "cases.pkl", os.path.getsize(O / "cases.pkl") >> 20, "MB")


# ---------------------------------------------------------------- per-case measurements
def measure(o, L, RV):
    from shapely.geometry import LineString
    c = RV.collision(o)
    t = L.first_event(o, "collision_at_fault")
    e, g = L.ego(o), L.gt(o)
    aid = next(k for k in o["actors"] if k != "EGO" and k[:6] == c["obj"] and
               L.box(*L.interp_pose(e, t)).distance(L.box(*L.interp_pose(o["actors"][k], t), *o["size"][k][:2])) <= c["gap"] + 1e-6)
    tr, (sx, sy, lab) = o["actors"][aid], o["size"][aid]
    lg = next(x for x in o["logged"] if x["id"] == aid)["traj"]
    # the logged human against the same object
    ts = np.arange(max(g[0, 0], lg[0, 0]), min(g[-1, 0], lg[-1, 0]) + 1, 1e5)
    hd = [L.box(*L.interp_pose(g, x)).distance(L.box(*L.interp_pose(lg, x), sx, sy)) for x in ts]
    h_end = RV.path_heading(g, L.arc(g[:, 1:3])) if L.arc(g[:, 1:3]) > 1.0 else g[0, 3]
    ext = g[-1, 1:3] + 40.0 * np.array([np.cos(h_end), np.sin(h_end)])
    cor = LineString(([tuple(x) for x in g[:, 1:3]] if L.arc(g[:, 1:3]) > 1.0 else [tuple(g[0, 1:3])]) + [tuple(ext)]).buffer(L.EGO_W / 2)
    obj_lat = L.signed_lat(np.r_[g[:, 1:3], ext[None]] if L.arc(g[:, 1:3]) > 1.0 else np.array([g[0, 1:3], ext]), L.interp_pose(tr, t)[:2])[0]
    in_path = any(cor.intersects(L.box(*L.interp_pose(lg, x), sx, sy)) for x in ts[::2])
    static = L.arc(lg[:, 1:3]) < 1.0
    # visibility from the simulated ego
    vis, ttc_vis, rng_vis = None, None, None
    grid = np.arange(0, t + 1, 1e5)
    rng = []
    for x in grid:
        pe = L.interp_pose(e, x)
        rear = pe[:2] - L.CENTER * np.array([np.cos(pe[2]), np.sin(pe[2])])
        cam = rear + CAM_X * np.array([np.cos(pe[2]), np.sin(pe[2])])
        d = L.interp_pose(tr, x)[:2] - cam
        rng.append((float(np.hypot(*d)), float(np.degrees(L.wrap(np.arctan2(d[1], d[0]) - pe[2])))))
    rng = np.array(rng)
    ok = (rng[:, 0] < 60) & (np.abs(rng[:, 1]) < HFOV)
    if ok.any():
        i = int(np.argmax(ok))
        j = min(i + 5, len(grid) - 1)
        rate = (rng[j, 0] - rng[i, 0]) / max((grid[j] - grid[i]) * 1e-6, 1e-3) if j > i else 0.0
        vis, rng_vis = float(grid[i] * 1e-6), float(rng[i, 0])
        ttc_vis = float(rng[i, 0] / -rate) if rate < -0.1 else float("inf")
    # plans before the event: 4 s arc against the current speed, and against the log
    sl = []
    for k, d in enumerate(o["drive"]):
        if d["now"] >= t or "traj" not in d:
            break
        p = d["traj"]
        T = (p[-1, 0] - p[0, 0]) * 1e-6
        v = L.speed_at(e, d["now"], 0.25e6)
        sl.append((L.arc(p[:, 1:3]) / max(v * T, 0.5), float(RV.plan_vs_log(o, k)[0]), v, L.arc(p[:, 1:3]), T))
    sl = np.array(sl).reshape(-1, 5)
    turn = L.turn_deg(o)
    v0 = L.speed_at(g, 0.1e6)
    n1 = c["cls"]
    kind = (("lead stopped" if in_path else "parked / static beside the path" if static else "standing neighbour beside the path") if c["obj_v"] < 0.5 else
            "slow lead" if n1.startswith("A2") else "cut-in" if n1.startswith("B") else "crossing" if n1.startswith("C") else
            "oncoming" if n1.startswith("E") else "side contact while turning" if n1.startswith("D") and abs(turn) >= 20 else
            "side contact (moving neighbour)" if n1.startswith("D") else "other")
    cmds = [r["cmd"] for r in o.get("rec", [])]
    return dict(n1=n1, kind=kind, obj=aid, obj_label=str(lab), obj_L=round(sx, 1), obj_W=round(sy, 1), obj_v=c["obj_v"], obj_static=static, obj_in_log_path=in_path,
                obj_dy0=c["dy0"], obj_lat_from_log_path=round(obj_lat, 2), toward_obj=bool(np.sign(c["lat"]) == np.sign(obj_lat) and abs(c["lat"]) >= 0.3), dh_deg=c["dh"], obj_dx=c["dx"], obj_dy=c["dy"], contact_front=c["front"], contact_lateral=c["lateral"],
                t_event=c["t"], decisions_before=c["k"], ego_v=c["ego_v"], log_v=c["log_v"], v0=round(v0, 2), start_from_rest=bool(v0 < 1.0),
                log_turn_deg=round(turn, 1), turn_bucket=L.turn_bucket(turn), cmds="".join("LSRU"[x] for x in cmds),
                lat_off=c["lat"], yaw_off_deg=c["yaw_off"], ahead_of_log=c["ahead"], on_path=bool(abs(c["lat"]) < 0.5),
                clear_on_log_path=c["clear_on_log_path"], clear_at_log_speed=c["clear_at_log_speed"],
                log_gap_at_event=c["log_gap"], log_min_gap=round(float(min(hd)), 2) if hd else None,
                t_visible=vis, vis_to_impact_s=None if vis is None else round(c["t"] - vis, 2), range_visible=None if rng_vis is None else round(rng_vis, 1),
                ttc_visible=None if ttc_vis is None else (round(ttc_vis, 2) if np.isfinite(ttc_vis) else "inf"),
                plan_v_ratio_last=round(float(sl[-1, 0]), 2) if len(sl) else None, plan_v_ratio_min=round(float(sl[:, 0].min()), 2) if len(sl) else None,
                plan_slowed=bool(len(sl) and sl[:, 0].min() < 0.9), plan_log_ratio_last=round(float(sl[-1, 1]), 2) if len(sl) and np.isfinite(sl[-1, 1]) else None,
                plan_log_ratio_max=c["ratio_max"], lead_gap_1s_before=c["lead_gap"], lead_v_1s_before=c["lead_v"])


def share(n, d):
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def cmd_tax(a):
    import c1_lib as L
    import c1_review as RV
    inv, C = json.loads((O / "inv.json").read_text()), pickle.load(open(O / "cases.pkl", "rb"))
    S = inv["scores"]
    rows = []
    for (drv, scene), o in sorted(C.items()):
        r = dict(driver=drv, recipe=drv.rsplit("-s", 1)[0], seed=int(drv[-1]), scene=scene, log="_".join(scene.split("_")[:2]), origin=o["origin"],
                 flags="+".join(L.SHORT[f] for f in L.zero_flags(o)))
        r.update(measure(o, L, RV))
        other = drv[:-1] + str(1 - r["seed"])
        r["other_seed_score"] = S[other].get(scene, [None])[0]
        for k in ("WA-JEPA (reference)", "SH30-F-s0", "P2H10-F-s0", "P2H10-F-s1", "APY10m10-AB-s0", "APY10m10-AB-s1", "AP2H10-AB-s0", "AP2H10-AB-s1"):
            v = S[k].get(scene)
            r["score|" + k] = None if v is None else v[0]
            r["flags|" + k] = None if v is None else "+".join(L.SHORT[f] for f in v[1])
        rows.append(r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "nuplan_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    D = inv["drivers"]
    md = ["# COL1: at-fault collisions on the 1491 public nuPlan scenes (generated by scripts/col1_nuplan.py tax)", "",
          "Rules and definitions: the docstring of `scripts/col1_nuplan.py`. One row per (driver, scene) rollout whose `collision_at_fault` flag is set; per-case rows: `nuplan_cases.csv`.", "",
          "## Inventory", "", "| driver | scenes scored | zeros | at-fault collision rollouts | with rollout.asl | read here |", "|:--|--:|--:|--:|--:|--:|"]
    for k in TARGETS + REFS:
        md.append(f"| {k} | {D[k]['scenes']} | {D[k]['zeros']} | {D[k]['collisions']} | {D[k]['asl']} | {sum(r['driver'] == k for r in rows) if k in TARGETS else ''} |")
    groups = [("P2H10-F (s0 + s1)", [r for r in rows if r["recipe"] == "P2H10-F"]), ("P2H10-F-s0", [r for r in rows if r["driver"] == "P2H10-F-s0"]),
              ("P2H10-F-s1", [r for r in rows if r["driver"] == "P2H10-F-s1"]), ("APY10m10-AB (s0 + s1)", [r for r in rows if r["recipe"] == "APY10m10-AB"]),
              ("AP2H10-AB (s0 + s1, 700 scenes)", [r for r in rows if r["recipe"] == "AP2H10-AB"]), ("all six", rows)]
    groups = [(n, g) for n, g in groups if g]
    md += ["", "## Struck object, by kind", "", "| kind | " + " | ".join(n for n, _ in groups) + " |", "|:--|" + "--:|" * len(groups)]
    for kd in KINDS:
        md.append(f"| {kd} | " + " | ".join(share(sum(r["kind"] == kd for r in g), len(g)) for _, g in groups) + " |")
    md.append("| n | " + " | ".join(str(len(g)) for _, g in groups) + " |")
    md += ["", "## N1 class (decision 153 rule, as in C1)", "", "| class | " + " | ".join(n for n, _ in groups) + " |", "|:--|" + "--:|" * len(groups)]
    for cl in sorted({r["n1"] for r in rows}):
        md.append(f"| {cl} | " + " | ".join(share(sum(r["n1"] == cl for r in g), len(g)) for _, g in groups) + " |")
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    tests = [("ego off the logged path by >= 0.5 m at impact", lambda r: not r["on_path"]),
             ("clear if it had stayed on the logged path at the driven speed (drift consequence)", lambda r: r["clear_on_log_path"]),
             ("clear at the log's position along the path with the driven lateral offset", lambda r: r["clear_at_log_speed"]),
             ("neither counterfactual clears it", lambda r: not r["clear_on_log_path"] and not r["clear_at_log_speed"]),
             ("on path (< 0.5 m) and not cleared by staying on the logged path", lambda r: r["on_path"] and not r["clear_on_log_path"]),
             ("ego ahead of the log at impact (> 1 m)", lambda r: r["ahead_of_log"] > 1.0),
             ("ego offset (>= 0.3 m) is toward the struck object's side", lambda r: r["toward_obj"]),
             ("ego below 3 m/s at impact", lambda r: r["ego_v"] < 3.0),
             ("scene starts from rest (log < 1 m/s)", lambda r: r["start_from_rest"]),
             ("log turn >= 20 deg", lambda r: abs(r["log_turn_deg"]) >= 20),
             ("log turn < 5 deg", lambda r: abs(r["log_turn_deg"]) < 5),
             ("object static all scene", lambda r: r["obj_static"]),
             ("object in the logged ego's corridor at some time", lambda r: r["obj_in_log_path"]),
             ("object faster than the ego at impact", lambda r: r["obj_v"] > r["ego_v"]),
             ("some plan before impact slows (4 s arc < 0.9 x speed x 4 s)", lambda r: r["plan_slowed"]),
             ("last plan before impact longer than 1.1 x the log", lambda r: num(r["plan_log_ratio_last"]) and r["plan_log_ratio_last"] > 1.1),
             ("object visible (60 m, camera field) for >= 2 s before impact", lambda r: num(r["vis_to_impact_s"]) and r["vis_to_impact_s"] >= 2.0),
             ("object never inside the camera field before impact", lambda r: r["t_visible"] is None),
             ("logged human within 1 m of the same object at some time", lambda r: num(r["log_min_gap"]) and r["log_min_gap"] < 1.0),
             ("WA-JEPA (reference) passes the scene (score > 0)", lambda r: num(r["score|WA-JEPA (reference)"]) and r["score|WA-JEPA (reference)"] > 0),
             ("SH30-F-s0 passes the scene", lambda r: num(r["score|SH30-F-s0"]) and r["score|SH30-F-s0"] > 0),
             ("the recipe's other seed passes the scene", lambda r: num(r["other_seed_score"]) and r["other_seed_score"] > 0)]
    md += ["", "## Circumstances (count and share of each column's cases)", "", "| | " + " | ".join(n for n, _ in groups) + " |", "|:--|" + "--:|" * len(groups)]
    for name, fn in tests:
        md.append(f"| {name} | " + " | ".join(share(sum(bool(fn(r)) for r in g), len(g)) for _, g in groups) + " |")
    med = lambda g, k: (lambda v: f"{np.median(v):.2f}" if len(v) else "")([r[k] for r in g if num(r[k])])  # noqa: E731
    for k, name in (("ego_v", "median ego speed at impact m/s"), ("lat_off", "median signed lateral offset m"), ("vis_to_impact_s", "median s from first visible to impact"),
                    ("t_event", "median impact time s"), ("log_min_gap", "median closest approach of the logged human to the object m")):
        md.append(f"| {name} | " + " | ".join(med(g, k) if k != "lat_off" else (lambda v: f"{np.median(np.abs(v)):.2f} (abs)")([r[k] for r in g]) for _, g in groups) + " |")
    md += ["", "## Kind x drift, P2H10-F both seeds", "", "| kind | n | off path >= 0.5 m | clear on logged path | WA-JEPA passes | other seed passes | median ego m/s | static object |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    g0 = groups[0][1]
    for kd in KINDS:
        g = [r for r in g0 if r["kind"] == kd]
        if g:
            md.append(f"| {kd} | {len(g)} | {sum(not r['on_path'] for r in g)} | {sum(r['clear_on_log_path'] for r in g)} | "
                      f"{sum(bool(tests[19][1](r)) for r in g)} | {sum(bool(tests[21][1](r)) for r in g)} | {np.median([r['ego_v'] for r in g]):.1f} | {sum(r['obj_static'] for r in g)} |")
    sc = Counter(r["scene"] for r in g0)
    md += ["", f"P2H10-F: {len(g0)} collision rollouts in {len(sc)} distinct scenes ({sum(v == 2 for v in sc.values())} scenes collide in both seeds), "
               f"{len({r['log'] for r in g0})} logs.", "", "## Cases", "",
           "| driver | scene | origin | N1 | kind | t s | ego m/s | obj m/s | lat m | yaw deg | turn deg | vis->impact s | TTC at vis | plan/v last | on-path clear | log-speed clear | human min gap m | WA-JEPA | other seed |",
           "|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|--:|--:|--:|"]
    for r in rows:
        md.append(f"| {r['driver']} | {r['scene'][-16:]} ({r['log'][5:]}) | {r['origin']} | {r['n1'][:2]} | {r['kind']} ({r['obj_label']}) | {r['t_event']} | {r['ego_v']} | {r['obj_v']} | {r['lat_off']} | "
                  f"{r['yaw_off_deg']} | {r['log_turn_deg']} | {r['vis_to_impact_s']} | {r['ttc_visible']} | {r['plan_v_ratio_last']} | {r['clear_on_log_path']} | {r['clear_at_log_speed']} | "
                  f"{r['log_min_gap']} | {r['score|WA-JEPA (reference)']} | {r['other_seed_score']} |")
    (out / "nuplan_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:90]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("inv", cmd_inv), ("extract", cmd_extract), ("tax", cmd_tax)):
        p = sub.add_parser(n)
        p.add_argument("--rerun"), p.add_argument("--out"), p.add_argument("--jobs", type=int, default=16)
        p.set_defaults(fn=f)
    a = ap.parse_args()
    a.fn(a)
