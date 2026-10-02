"""Markdown tables for experiments/hugsim/results/controller_spin.md from the outputs of spin_analysis.py, spin_controller_replay.py
and spin_scene_context.py.
    python spin_tables.py <spin_episodes.csv> <replay.csv> <ctx.csv> <runs_root> <out.md>
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

AG = ["cinque", "lebowski", "ltf", "cv"]
CT = ["official", "fixed"]


def main(ep_csv, replay_csv, ctx_csv, root, out):
    R = list(csv.DictReader(open(ep_csv)))
    L = []
    # 1. spin share
    L += ["## T1 spin-outs by agent and controller (64 scenarios each; spin = heading error to the route reaches 60 deg)", "",
          "| agent | controller | complete | failed (not complete) | spin >= 60 deg | of failed | spin >= 90 deg | spin >= 45 deg | left / right | share of failures that are spins |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in AG:
        for c in CT:
            X = [r for r in R if r["agent"] == a and r["controller"] == c]
            fail = [r for r in X if r["end"] != "complete"]
            sp = lambda th: sum(float(r["max_abs_e"]) >= th for r in X)  # noqa: E731
            spf = sum(float(r["max_abs_e"]) >= 60 for r in fail)
            lr = Counter(r["first_dir"] for r in X if r["spin"] == "True" and r["first_dir"])
            L.append(f"| {a} | {c} | {len(X) - len(fail)} | {len(fail)} | {sp(60)} | {spf} | {sp(90)} | {sp(45)} | "
                     f"{lr.get('left', 0)} / {lr.get('right', 0)} | {100 * spf / max(len(fail), 1):.0f}% |")
    # 2. pairs
    d = {(r["agent"], r["controller"], r["scenario"]): r for r in R}
    L += ["", "## T2 paired by scenario: spin under the official controller vs under PR #57 (same agent, same scenario)", "",
          "| agent | spin official & not PR#57 | spin both | spin PR#57 only | no spin either |", "|---|---|---|---|---|"]
    for a in ("cinque", "lebowski", "cv"):
        c = Counter()
        for (ag, ct, sc), r in d.items():
            if ag == a and ct == "official":
                c[(r["spin"] == "True", d[(a, "fixed", sc)]["spin"] == "True")] += 1
        L.append(f"| {a} | {c[(True, False)]} | {c[(True, True)]} | {c[(False, True)]} | {c[(False, False)]} |")
    # 3. PR#57 openpilot spin episodes
    L += ["", "## T3 the PR #57 openpilot spin-outs (Cinque and Lebowski)", "",
          "| agent | scenario | dataset | end | steps | first direction | divergence step (|e| > 5 deg) | speed there (m/s) | plan error before divergence (deg, signed toward the turn) | share of the yaw built on steps whose plan pointed >= 10 deg off the route on the turn side |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        if r["controller"] == "fixed" and r["agent"] in ("cinque", "lebowski") and r["spin"] == "True":
            L.append(f"| {r['agent']} | {r['scenario']} | {r['dataset']} | {r['end']} | {r['steps']} | {r['first_dir']} | {r['start']} | "
                     f"{float(r['v_start']):.1f} | {float(r['plan_err_signed_deg']):.0f} | {float(r['yaw_share_plan_led']):.2f} |")
    # 4. spin by dataset
    L += ["", "## T4 PR #57 spin-outs by dataset (16 scenarios per cell)", "", "| agent | " + " | ".join(["nuscenes", "waymo", "kitti360", "pandaset"]) + " |", "|---|---|---|---|---|"]
    for a in ("cinque", "lebowski"):
        row = []
        for ds in ("nuscenes", "waymo", "kitti360", "pandaset"):
            X = [r for r in R if r["agent"] == a and r["controller"] == "fixed" and r["dataset"] == ds]
            row.append(f"{sum(r['spin'] == 'True' for r in X)} / {len(X)}")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    # 5. controller replay
    P = list(csv.DictReader(open(replay_csv)))
    for r in P:
        for k in r:
            if k not in ("run", "tag"):
                r[k] = float(r[k])
    L += ["", "## T5 controller replay on the logged states and plans (one 0.25 s step from each logged state)", "",
          "| run set | moving steps | steps whose plan points < 5 deg off the ego axis | of those, executed |yaw change| > 3 deg in the step | share |",
          "|---|---|---|---|---|"]
    for tag in ("cinque-official", "lebowski-official", "cinque-fixed", "lebowski-fixed"):
        X = [r for r in P if r["tag"] == tag and r["phi05"] == r["phi05"] and r["plan_len"] > 1]
        ph = np.array([abs(r["phi05"]) for r in X])
        dd = np.array([abs(r["dth_logged"]) for r in X])
        s = ph < np.radians(5)
        L.append(f"| {tag} | {len(X)} | {s.sum()} | {(dd[s] > np.radians(3)).sum()} | {100 * (dd[s] > np.radians(3)).mean():.1f}% |")
    L += ["", "| run set | steps with plan direction at 0.5 s > 20 deg | median executed yaw change / plan direction: as run | official | PR #57 | fixed2 |", "|---|---|---|---|---|---|"]
    for tag in ("cinque-fixed", "lebowski-fixed"):
        X = [r for r in P if r["tag"] == tag and r["phi05"] == r["phi05"] and r["plan_len"] > 1]
        ph = np.array([r["phi05"] for r in X])
        big = abs(ph) > np.radians(20)
        med = lambda k: np.median(np.array([r[k] for r in X])[big] / ph[big])  # noqa: E731
        L.append(f"| {tag} | {big.sum()} | {med('dth_logged'):.2f} | {med('dth_official'):.2f} | {med('dth_fixed'):.2f} | {med('dth_fixed2'):.2f} |")
    # 6. left bias
    L += ["", "## T6 plan direction relative to the route, PR #57 openpilot runs, steps with heading error < 5 deg (still aligned)", "",
          "| agent | dataset | steps | mean plan error at 1 s (deg, - = left of the route direction) | median | share >= 10 deg left | share >= 10 deg right |", "|---|---|---|---|---|---|---|"]
    sys.path.insert(0, str(Path(__file__).parent))
    import spin_analysis as A
    routes = json.load(open(Path(ep_csv).parent.parent / "routes.json"))
    acc = defaultdict(list)
    for r in csv.DictReader(open(Path(root) / "scored-op" / "results.csv")):
        if r["controller"] != "fixed":
            continue
        dd = Path(r["run_dir"])
        dd = Path(root) / "scored-op" / r["tag"] / dd.parent.name / dd.name
        pos, th, v, st, pl = A.load_run(dd, r["agent"])
        _, s = A.analyse(pos, th, v, st, pl, routes[r["scene"]])
        err = np.degrees(A.wrap(s["phi"] - s["des"]))
        m = s["moving"] & (np.abs(s["e"]) < np.radians(5))
        acc[(r["agent"], r["dataset"])] += list(err[m])
    for (a, ds), v in sorted(acc.items()):
        v = np.array(v)
        L.append(f"| {a} | {ds} | {len(v)} | {v.mean():.1f} | {np.median(v):.1f} | {100 * (v < -10).mean():.0f}% | {100 * (v > 10).mean():.0f}% |")
    # 7. scene context
    C = list(csv.DictReader(open(ctx_csv)))
    L += ["", "## T7 scene context at the divergence step (cinque + lebowski, PR #57): drivable ground left vs right, geometry in the lane ahead", "",
          "| group | n | mean (L - R) / (L + R) of ground points 3-25 m ahead | share with scene points in the lane 3-15 m ahead |", "|---|---|---|---|"]
    for name, sel in (("spin, turned left", lambda r: r["spin"] == "True" and r["first_dir"] == "left"),
                      ("spin, turned right", lambda r: r["spin"] == "True" and r["first_dir"] == "right"),
                      ("no spin (step 8)", lambda r: r["spin"] == "False")):
        X = [r for r in C if sel(r)]
        L.append(f"| {name} | {len(X)} | {np.mean([float(r['asym']) for r in X]):.2f} | {100 * np.mean([float(r['blocked_pts']) > 0 for r in X]):.0f}% |")
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:6])
