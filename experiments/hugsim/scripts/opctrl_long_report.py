"""openpilot's lateral + longitudinal path in HUGSIM closed loop (plans/2026-10-04-op-control-stack-long-prereg.md). Reference arms (base, base_rerun 1-3,
opctrl = decision 118) are read from results/op_control_stack/opctrl_runs.csv; the new arms (cinque-opctrl-long, cinque-fixed-base4 if present) are scored
from the run dir like opctrl_report.py. Spin / launch-event / HD-paired definitions are the same as there. Box, envs/hugsim python:
    python experiments/hugsim/scripts/opctrl_long_report.py $DATA_DIR/runs/opctrl_long/closed $DATA_DIR/tmp_spin/routes.json experiments/hugsim/results/op_control_stack_long
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derot_report import SPIN10, runs  # noqa: E402
from lowspeed_report import boot, events  # noqa: E402
from opctrl_report import c_logged  # noqa: E402
from spin_analysis import analyse, load_run  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
NEWTAG = {"cinque-opctrl-long": "opctrl_long", "cinque-fixed-base4": "base_rerun4"}
REFS = ("base", "base_rerun", "base_rerun2", "base_rerun3", "opctrl")


def main(lo, routes, out):
    lo, out = Path(lo), Path(out)
    routes = json.load(open(routes))
    by = {}
    for r in csv.DictReader(open(REPO / "experiments/hugsim/results/op_control_stack/opctrl_runs.csv")):
        if r["arm"] in REFS:
            by.setdefault(r["arm"], {})[r["scenario"]] = dict(r, spin=r["spin"] == "True", hdscore=float(r["hdscore"]), rc=float(r["rc"]), steps=int(float(r["steps"])),
                                                               n_init=int(float(r["n_init"])), spin_init=int(float(r["spin_init"])),
                                                               n_relaunch=int(float(r["n_relaunch"])), spin_relaunch=int(float(r["spin_relaunch"])))
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
    arms = [a for a in REFS + ("base_rerun4", "opctrl_long") if a in by]
    refs = [a for a in arms if a != "opctrl_long"]
    L = ["| arm | n | spins | on the 10 | initial-launch spins / n | re-launch spins / n | fg | bg | off_route | max_steps | complete | HD mean | RC mean | "
         + " | ".join(f"non-spin HD paired vs {a}" for a in refs) + " |", "|" + "---|" * (13 + len(refs))]
    summ = {}
    for arm in arms:
        a = by[arm]
        sc = sorted(a)
        ns = [s for s in ns_all if s in a]
        cells = []
        for ref in refs:
            if arm == "opctrl_long" and ref != arm:
                m = boot(np.array([a[s]["hdscore"] - by[ref][s]["hdscore"] for s in ns if s in by[ref]]))
                summ[f"{arm}_vs_{ref}"] = list(m)
                cells.append(f"{m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}]")
            else:
                cells.append("")
        cnt = lambda e: sum(a[s]["end"] == e for s in sc)  # noqa: E731
        summ[arm] = dict(n=len(sc), spins=sum(a[s]["spin"] for s in sc), spins10=sum(a[s]["spin"] for s in sc if s in SPIN10), fg=cnt("fg_collision"), bg=cnt("bg_collision"),
                         off=cnt("off_route"), max_steps=cnt("max_steps"), complete=cnt("complete"), crash=cnt("crash"),
                         hd=float(np.mean([a[s]["hdscore"] for s in sc])), rc=float(np.mean([a[s]["rc"] for s in sc])),
                         ev_init=[sum(a[s]["n_init"] for s in sc), sum(a[s]["spin_init"] for s in sc)], ev_re=[sum(a[s]["n_relaunch"] for s in sc), sum(a[s]["spin_relaunch"] for s in sc)])
        k = summ[arm]
        L.append(f"| {arm} | {len(sc)} | {k['spins']} | {k['spins10']} / {len(SPIN10)} | {k['ev_init'][1]} / {k['ev_init'][0]} | {k['ev_re'][1]} / {k['ev_re'][0]} | "
                 f"{k['fg']} | {k['bg']} | {k['off']} | {k['max_steps']} | {k['complete']} | {k['hd']:.3f} | {k['rc']:.3f} | " + " | ".join(cells) + " |")
    if "opctrl_long" in by:
        d = by["opctrl_long"]
        spins, lo_ci = summ["opctrl_long"]["spins"], summ["opctrl_long_vs_base"][1]
        L += ["", f"Line (i) spins <= 2: **{spins}** -> {'PASS' if spins <= 2 else 'FAIL'}",
              f"Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **{lo_ci:+.3f}** -> {'PASS' if lo_ci > -0.02 else 'FAIL'}"]
        for ref in ("base_rerun3", "base_rerun4"):
            if ref in by:
                L.append(f"  vs {ref}: {summ[f'opctrl_long_vs_{ref}'][1]:+.3f} -> {'PASS' if summ[f'opctrl_long_vs_{ref}'][1] > -0.02 else 'FAIL'}")
        stuck = lambda a: {s for s in by[a] if by[a][s]["end"] == "max_steps"}  # noqa: E731
        sl, so, sb = stuck("opctrl_long"), stuck("opctrl"), stuck("base_rerun3")
        L += ["", f"Stuck runs (max_steps): opctrl_long {len(sl)}, opctrl {len(so)}, exam base {len(stuck('base'))}, same-day base {len(sb)}. "
              f"opctrl_long stuck and not in same-day base: {len(sl - sb)}; opctrl stuck and not in opctrl_long: {len(so - sl)}; opctrl_long stuck and not in opctrl: {len(sl - so)}"]
        worst = sorted([s for s in ns_all if s in d], key=lambda s: d[s]["hdscore"] - base[s]["hdscore"])[:6]
        best = sorted([s for s in ns_all if s in d], key=lambda s: -(d[s]["hdscore"] - base[s]["hdscore"]))[:4]
        L += ["", "Largest non-spin HD losses vs exam base: " + "; ".join(f"{s} {base[s]['hdscore']:.3f} -> {d[s]['hdscore']:.3f} ({d[s]['end']})" for s in worst),
              "Largest gains: " + "; ".join(f"{s} {base[s]['hdscore']:.3f} -> {d[s]['hdscore']:.3f} ({d[s]['end']})" for s in best),
              "Spins: " + "; ".join(f"{s} (max err {d[s]['max_abs_e']} deg, base spin {base[s]['spin']})" for s in d if d[s]["spin"])]
        summ["c_opctrl_long"] = c_logged(dirs["opctrl_long"])
        L += ["", "Closed-loop c of the arm (heading over the next step per degree of the plan's 1 s direction): "
              + "; ".join(f"{b} {v['c']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] (n {v['n']})" for b, v in summ["c_opctrl_long"].items() if v)]
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for a in by.values() for r in a.values()]
    keys = sorted({k for r in rows for k in r})
    with open(out / "opctrl_long_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:4])
