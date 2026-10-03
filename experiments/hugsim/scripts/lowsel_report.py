"""Selective low-speed rule in HUGSIM closed loop (plans/2026-10-04-lowspeed-ctrl-selective-prereg.md): spins, HD paired vs the exam baseline and
vs the same-day base reruns, per-launch-event spins, the pre-registered lines. Existing arms (base, base_rerun, lowspeed) come from
results/lowspeed_ctrl/lowspeed_runs.csv; new arms are scored from the run dir.  Box, envs/hugsim python:
    python experiments/hugsim/scripts/lowsel_report.py $DATA_DIR/runs/hugsim-lowsel/closed $DATA_DIR/runs/hugsim-exam/scored-op \
        $DATA_DIR/tmp_spin/routes.json experiments/hugsim/results/lowspeed_ctrl_selective
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derot_report import SPIN10, runs  # noqa: E402
from lowspeed_report import boot, events  # noqa: E402
from spin_analysis import analyse, load_run  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
NEW = {"cinque-lowsel": "lowsel", "cinque-fixed-base2": "base_rerun2"}
D1, D0 = 4.0, 2.0


def main(lo, exam, routes, out):
    lo, exam, out = Path(lo), Path(exam), Path(out)
    routes = json.load(open(routes))
    old = list(csv.DictReader(open(REPO / "experiments/hugsim/results/lowspeed_ctrl/lowspeed_runs.csv")))
    by = {}
    for r in old:
        by.setdefault(r["arm"], {})[r["scenario"]] = {k: (float(v) if k in ("hdscore", "rc", "max_abs_e") else v) for k, v in r.items()}
        by[r["arm"]][r["scenario"]].update(spin=r["spin"] == "True", start=int(r["start"]), n_init=int(r["n_init"]), spin_init=int(r["spin_init"]),
                                           n_relaunch=int(r["n_relaunch"]), spin_relaunch=int(r["spin_relaunch"]), steps=int(r["steps"]))
    R = runs(lo / "results.csv", lo, set(NEW))
    last_small = {}
    for (tag, scen), (r, d) in sorted(R.items()):
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        ev = events(v, res["spin"], res.get("start", -1))
        by.setdefault(NEW[tag], {})[scen] = dict(arm=NEW[tag], scenario=scen, spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1),
                                                start=res.get("start", -1), hdscore=float(r["hdscore"]), rc=float(r["rc"]), nc=r["nc"], end=r["end"],
                                                steps=int(r["steps"]), n_init=1, spin_init=int(ev[0][2]), n_relaunch=len(ev) - 1,
                                                spin_relaunch=sum(e[2] for e in ev[1:]), vmax=round(float(v.max()), 2))
        if tag == "cinque-lowsel" and res["spin"]:   # for a surviving spin: |plan 1 s direction| over the 12 steps before the divergence start
            import math
            s0 = int(res["start"])
            a = [abs(math.degrees(math.atan2(p[1][0], p[1][1]))) if len(p) > 1 else float("nan") for p in plans[max(0, s0 - 12):s0 + 1]]
            last_small[scen] = dict(start=s0, raw_plan_dir_deg=[round(x, 1) for x in a])
    base = by["base"]
    ns_all = [s for s in base if not base[s]["spin"]]
    L = ["| arm | n | spins | spins on the 10 | new spins | initial-launch spins / n | re-launch spins / n | max_steps | HD mean | non-spin HD paired vs exam base (95% CI) | vs same-day rerun | n non-spin |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm, d in by.items():
        sc = sorted(d)
        s10 = [s for s in sc if s in SPIN10]
        ns = [s for s in ns_all if s in d]
        new = [s for s in sc if d[s]["spin"] and s in base and not base[s]["spin"]]
        cells = []
        for ref in ("base", "base_rerun"):
            if arm != ref and arm != "base" and ref in by and ns:
                dl = np.array([d[s]["hdscore"] - by[ref][s]["hdscore"] for s in ns if s in by[ref]])
                m = boot(dl)
                summ[f"{arm}_vs_{ref}"] = list(m)
                cells.append(f"{m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}]")
            else:
                cells.append("")
        summ[arm] = dict(n=len(sc), spins=sum(d[s]["spin"] for s in sc), spins10=sum(d[s]["spin"] for s in s10), new=new,
                         hd=float(np.mean([d[s]["hdscore"] for s in sc])), max_steps=sum(d[s]["end"] == "max_steps" for s in sc),
                         rc=float(np.mean([d[s]["rc"] for s in sc])),
                         ev_init=[sum(d[s]["n_init"] for s in sc), sum(d[s]["spin_init"] for s in sc)],
                         ev_re=[sum(d[s]["n_relaunch"] for s in sc), sum(d[s]["spin_relaunch"] for s in sc)])
        k = summ[arm]
        L.append(f"| {arm} | {len(sc)} | {k['spins']} | {k['spins10']} / {len(s10)} | {len(new)} {new or ''} | {k['ev_init'][1]} / {k['ev_init'][0]} | "
                 f"{k['ev_re'][1]} / {k['ev_re'][0]} | {k['max_steps']} | {k['hd']:.3f} | {cells[0]} | {cells[1]} | {len(ns)} |")
    if "lowsel" in by:
        d = by["lowsel"]
        miss = [s for s in base if s not in d]
        m2 = boot(np.array([(d[s]["hdscore"] if s in d else 0.0) - base[s]["hdscore"] for s in ns_all]))
        spins, lo_ci = summ["lowsel"]["spins"], summ["lowsel_vs_base"][1]
        L += ["", f"Line (i) spins <= 2 (base {summ['base']['spins']}): **{spins}** -> {'PASS' if spins <= 2 else 'FAIL'}",
              f"Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **{lo_ci:+.3f}** -> {'PASS' if lo_ci > -0.02 else 'FAIL'}"
              f" (crashed {miss or 'none'}; crash = 0: {m2[0]:+.3f} [{m2[1]:+.3f}, {m2[2]:+.3f}])",
              f"Near gate (spins <= 4 and lower bound > -0.04): {'yes' if spins <= 4 and lo_ci > -0.04 else 'no'}"]
        summ["missing"] = miss
        worst = sorted(ns_all, key=lambda s: d[s]["hdscore"] - base[s]["hdscore"] if s in d else 0)[:6]
        L += ["", "Largest non-spin HD losses vs exam base: " + "; ".join(f"{s} {base[s]['hdscore']:.3f} -> {d[s]['hdscore']:.3f} ({d[s]['end']})" for s in worst if s in d)]
        L += ["", "Surviving spins (raw plan 1 s direction |a| deg, 12 steps up to the divergence start): " + json.dumps(last_small)]
    per = ["| scenario | " + " | ".join(by) + " |", "|---|" + "---|" * len(by)]
    for s in SPIN10:
        per.append(f"| {s} | " + " | ".join(f"{'SPIN ' if by[a][s]['spin'] else ''}{by[a][s]['max_abs_e']:.0f} deg, HD {by[a][s]['hdscore']:.3f}, "
                                            f"{by[a][s]['end']} {by[a][s]['steps']} st" if s in by[a] else "-" for a in by) + " |")
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for a in by.values() for r in a.values()]
    with open(out / "lowsel_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text("\n".join(L) + "\n\nPer baseline-spin scenario (max heading error vs route, HD, end, steps):\n\n" + "\n".join(per) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(L))
    print("\n".join(per))


if __name__ == "__main__":
    main(*sys.argv[1:5])
