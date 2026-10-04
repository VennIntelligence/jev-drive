"""openpilot's lateral path in HUGSIM closed loop (plans/2026-10-05-op-control-stack-prereg.md): spins, launch events, HD paired vs the exam base and
the same-day reruns, and the closed-loop c of the arm (lowspeed_ctrl.md readout). Existing arms come from results/lowspeed_ctrl_selective/lowsel_runs.csv
(base, base_rerun, base_rerun2); new arms (cinque-opctrl, cinque-fixed-base3 if present) are scored from the run dir.  Box, envs/hugsim python:
    python experiments/hugsim/scripts/opctrl_report.py $DATA_DIR/runs/opctrl/closed $DATA_DIR/tmp_spin/routes.json experiments/hugsim/results/op_control_stack
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derot_report import SPIN10, runs  # noqa: E402
from lowspeed_report import boot, events  # noqa: E402
from spin_analysis import analyse, load_run  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
NEWTAG = {"cinque-opctrl": "opctrl", "cinque-fixed-base3": "base_rerun3"}
REFS = ("base", "base_rerun", "base_rerun2", "base_rerun3")
BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (0.0, 3.0)]


def c_logged(dirs):
    """Heading over the next step per degree of the sent plan's 1 s direction (|phi1| < 15, |p1| >= 0.3), run-cluster bootstrap, by speed bin."""
    rows = []
    for name, d in dirs:
        recs = [json.loads(x) for x in open(Path(d) / "zs_steps.jsonl") if '"step"' in x]
        for r, nx in zip(recs, recs[1:]):
            p = np.array(r["plan"], float)
            if len(p) < 2 or np.linalg.norm(p[1]) < 0.3:
                continue
            a = math.degrees(math.atan2(p[1, 0], p[1, 1]))
            if abs(a) < 15:
                rows.append((name, float(r["v"]), a, math.degrees(nx["theta"] - r["theta"])))
    out, rng = {}, np.random.default_rng(0)
    for lo, hi in BINS:
        sel = [x for x in rows if lo <= x[1] < hi]
        names = sorted({x[0] for x in sel})
        if len(sel) < 5:
            out[f"{lo:g}-{hi:g}"] = None
            continue
        by = {n: np.array([(x[2], x[3]) for x in sel if x[0] == n]) for n in names}
        s = lambda ks: float(sum((by[k][:, 0] * by[k][:, 1]).sum() for k in ks) / max(sum((by[k][:, 0] ** 2).sum() for k in ks), 1e-12))  # noqa: E731
        bs = [s(rng.choice(names, len(names))) for _ in range(1000)]
        out[f"{lo:g}-{hi:g}"] = dict(c=round(s(names), 4), lo=round(float(np.percentile(bs, 2.5)), 4), hi=round(float(np.percentile(bs, 97.5)), 4), n=len(sel))
    return out


def main(lo, routes, out):
    lo, out = Path(lo), Path(out)
    routes = json.load(open(routes))
    by = {}
    for r in csv.DictReader(open(REPO / "experiments/hugsim/results/lowspeed_ctrl_selective/lowsel_runs.csv")):
        if r["arm"] in REFS:
            by.setdefault(r["arm"], {})[r["scenario"]] = dict(r, spin=r["spin"] == "True", hdscore=float(r["hdscore"]), rc=float(r["rc"]), steps=int(r["steps"]),
                                                               n_init=int(r["n_init"]), spin_init=int(r["spin_init"]), n_relaunch=int(r["n_relaunch"]), spin_relaunch=int(r["spin_relaunch"]))
    R = runs(lo / "results.csv", lo, set(NEWTAG))
    dirs = {}
    for (tag, scen), (r, d) in sorted(R.items()):
        arm = NEWTAG[tag]
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        ev = events(v, res["spin"], res.get("start", -1))
        by.setdefault(arm, {})[scen] = dict(arm=arm, scenario=scen, spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1), hdscore=float(r["hdscore"]),
                                            rc=float(r["rc"]), nc=r["nc"], end=r["end"], steps=int(r["steps"]), n_init=1, spin_init=int(ev[0][2]),
                                            n_relaunch=len(ev) - 1, spin_relaunch=sum(e[2] for e in ev[1:]))
        dirs.setdefault(arm, []).append((scen, d))
    base = by["base"]
    ns_all = [s for s in base if not base[s]["spin"]]
    refs = [a for a in REFS if a in by]
    L = ["| arm | n | spins | on the 10 | new spins | initial-launch spins / n | re-launch spins / n | fg_collision | bg_collision | max_steps | HD mean | RC mean | "
         + " | ".join(f"non-spin HD paired vs {a}" for a in refs) + " |", "|" + "---|" * (12 + len(refs))]
    summ = {}
    for arm in refs + (["opctrl"] if "opctrl" in by else []):
        a = by[arm]
        sc = sorted(a)
        ns = [s for s in ns_all if s in a]
        new = [s for s in sc if a[s]["spin"] and not base[s]["spin"]]
        cells = []
        for ref in refs:
            if arm in ("opctrl", "base_rerun3") and ref != arm:
                dl = np.array([a[s]["hdscore"] - by[ref][s]["hdscore"] for s in ns if s in by[ref]])
                m = boot(dl)
                summ[f"{arm}_vs_{ref}"] = list(m)
                cells.append(f"{m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}]")
            else:
                cells.append("")
        summ[arm] = dict(n=len(sc), spins=sum(a[s]["spin"] for s in sc), spins10=sum(a[s]["spin"] for s in sc if s in SPIN10), new=new,
                         fg=sum(a[s]["end"] == "fg_collision" for s in sc), bg=sum(a[s]["end"] == "bg_collision" for s in sc),
                         max_steps=sum(a[s]["end"] == "max_steps" for s in sc), crash=sum(a[s]["end"] == "crash" for s in sc),
                         hd=float(np.mean([a[s]["hdscore"] for s in sc])), rc=float(np.mean([a[s]["rc"] for s in sc])),
                         ev_init=[sum(a[s]["n_init"] for s in sc), sum(a[s]["spin_init"] for s in sc)], ev_re=[sum(a[s]["n_relaunch"] for s in sc), sum(a[s]["spin_relaunch"] for s in sc)])
        k = summ[arm]
        L.append(f"| {arm} | {len(sc)} | {k['spins']} | {k['spins10']} / {len(SPIN10)} | {len(new)} {new or ''} | {k['ev_init'][1]} / {k['ev_init'][0]} | {k['ev_re'][1]} / {k['ev_re'][0]} | "
                 f"{k['fg']} | {k['bg']} | {k['max_steps']} | {k['hd']:.3f} | {k['rc']:.3f} | " + " | ".join(cells) + " |")
    if "opctrl" in by:
        d = by["opctrl"]
        spins, lo_ci = summ["opctrl"]["spins"], summ["opctrl_vs_base"][1]
        L += ["", f"Line (i) spins <= 2: **{spins}** -> {'PASS' if spins <= 2 else 'FAIL'}",
              f"Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **{lo_ci:+.3f}** -> {'PASS' if lo_ci > -0.02 else 'FAIL'}"]
        worst = sorted([s for s in ns_all if s in d], key=lambda s: d[s]["hdscore"] - base[s]["hdscore"])[:6]
        best = sorted([s for s in ns_all if s in d], key=lambda s: -(d[s]["hdscore"] - base[s]["hdscore"]))[:4]
        L += ["", "Largest non-spin HD losses vs exam base: " + "; ".join(f"{s} {base[s]['hdscore']:.3f} -> {d[s]['hdscore']:.3f} ({d[s]['end']})" for s in worst),
              "Largest gains: " + "; ".join(f"{s} {base[s]['hdscore']:.3f} -> {d[s]['hdscore']:.3f} ({d[s]['end']})" for s in best),
              "Spins: " + "; ".join(f"{s} (max err {d[s]['max_abs_e']} deg, base spin {base[s]['spin']})" for s in d if d[s]["spin"])]
        summ["c_opctrl"] = c_logged(dirs["opctrl"])
        L += ["", "Closed-loop c of the opctrl arm (heading over the next step per degree of the plan's 1 s direction): "
              + "; ".join(f"{b} {v['c']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] (n {v['n']})" for b, v in summ["c_opctrl"].items() if v)]
    per = ["| scenario | " + " | ".join(refs) + (" | opctrl |" if "opctrl" in by else " |"), "|" + "---|" * (1 + len(refs) + ("opctrl" in by))]
    for s in SPIN10:
        per.append(f"| {s} | " + " | ".join(f"{'SPIN ' if by[a][s]['spin'] else ''}{float(by[a][s]['max_abs_e']):.0f} deg, HD {by[a][s]['hdscore']:.3f}, {by[a][s]['end']}"
                                             for a in refs + (["opctrl"] if "opctrl" in by else []) if s in by[a]) + " |")
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for a in by.values() for r in a.values()]
    keys = sorted({k for r in rows for k in r})
    with open(out / "opctrl_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text("\n".join(L) + "\n\nPer baseline-spin scenario:\n\n" + "\n".join(per) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(L))
    print("\n".join(per))


if __name__ == "__main__":
    main(*sys.argv[1:4])
