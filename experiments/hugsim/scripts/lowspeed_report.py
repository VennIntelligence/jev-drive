"""Low-speed lateral transfer limit in HUGSIM closed loop (plans/2026-10-04-lowspeed-ctrl-prereg.md): spins, HD paired against the exam
baseline, per-launch-event spins, the two pre-registered lines.  Box, envs/hugsim python:
    python experiments/hugsim/scripts/lowspeed_report.py $DATA_DIR/runs/hugsim-lowspeed/closed $DATA_DIR/runs/hugsim-exam/scored-op \
        $DATA_DIR/tmp_spin/routes.json experiments/hugsim/results/lowspeed_ctrl
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derot_report import SPIN10, runs  # noqa: E402
from spin_analysis import analyse, load_run  # noqa: E402

ARMS = {"cinque-fixed": "base", "cinque-lowspeed": "lowspeed", "cinque-fixed-base": "base_rerun"}


def events(v, spin, start):
    """Launch events as spin_attribution: initial (step 0) and re-launches (v >= 1 after < 0.4 within 8 steps); spin = start within 0-12 steps."""
    end = int(start) if spin else len(v)
    ev = [("initial", 0, bool(spin) and start <= 12)]
    last = -99
    for k in range(8, end):
        if v[k] >= 1.0 and v[max(0, k - 8):k].min() < 0.4 and k - last > 8:
            last = k
            ev.append(("relaunch", k, bool(spin) and 0 <= start - k <= 12))
    return ev


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(lo, exam, routes, out):
    lo, exam, out = Path(lo), Path(exam), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    routes = json.load(open(routes))
    R = runs(exam / "results.csv", exam, {"cinque-fixed"}) | runs(lo / "results.csv", lo, {"cinque-lowspeed", "cinque-fixed-base"})
    rows = []
    for (tag, scen), (r, d) in sorted(R.items()):
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        ev = events(v, res["spin"], res.get("start", -1))
        rows.append(dict(arm=ARMS[tag], scenario=scen, spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1), start=res.get("start", -1),
                         hdscore=float(r["hdscore"]), rc=float(r["rc"]), nc=r["nc"], end=r["end"], steps=int(r["steps"]),
                         n_init=1, spin_init=int(ev[0][2]), n_relaunch=len(ev) - 1, spin_relaunch=sum(e[2] for e in ev[1:]),
                         vmax=round(float(v.max()), 2)))
    with open(out / "lowspeed_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["scenario"]] = r
    base = by["base"]
    L = ["| arm | n | spins (all) | spins on the 10 | new spins | initial-launch spins / n | re-launch spins / n | stood to max_steps | HD mean | HD on non-spin (paired delta, 95% CI) | n non-spin |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm, d in by.items():
        sc = sorted(d)
        s10 = [s for s in sc if s in SPIN10]
        ns = [s for s in sc if s not in SPIN10 and s in base]
        new = [s for s in sc if d[s]["spin"] and s in base and not base[s]["spin"]]
        ci = ""
        dl = np.array([d[s]["hdscore"] - base[s]["hdscore"] for s in ns])
        if arm != "base" and len(dl):
            m, a, b = boot(dl)
            ci = f"{m:+.3f} [{a:+.3f}, {b:+.3f}]"
            summ[arm + "_dl_nonspin"] = [m, a, b]
        summ[arm] = dict(n=len(sc), spins=sum(d[s]["spin"] for s in sc), spins10=sum(d[s]["spin"] for s in s10), new=new,
                         hd=float(np.mean([d[s]["hdscore"] for s in sc])), max_steps=sum(d[s]["end"] == "max_steps" for s in sc),
                         ev_init=[sum(d[s]["n_init"] for s in sc), sum(d[s]["spin_init"] for s in sc)],
                         ev_re=[sum(d[s]["n_relaunch"] for s in sc), sum(d[s]["spin_relaunch"] for s in sc)])
        k = summ[arm]
        L.append(f"| {arm} | {len(sc)} | {k['spins']} | {k['spins10']} / {len(s10)} | {len(new)} {new or ''} | {k['ev_init'][1]} / {k['ev_init'][0]} | "
                 f"{k['ev_re'][1]} / {k['ev_re'][0]} | {k['max_steps']} | {k['hd']:.3f} | {ci} | {len(ns)} |")
    # the two lines (vs exam base); crash handling: scenarios missing from the arm are dropped (reading 1) or scored 0 (reading 2)
    if "lowspeed" in by:
        d = by["lowspeed"]
        miss = [s for s in base if s not in d]
        ns_all = [s for s in base if not base[s]["spin"]]
        dl2 = np.array([(d[s]["hdscore"] if s in d else 0.0) - base[s]["hdscore"] for s in ns_all])
        m2 = boot(dl2)
        summ["line_nonspin_crash_as_zero"] = list(m2)
        summ["missing"] = miss
        spins = summ["lowspeed"]["spins"]
        lo_ci = summ.get("lowspeed_dl_nonspin", [None, None])[1]
        L += ["", f"Line (i) spins <= 2 (base {summ['base']['spins']}): **{spins}** -> {'PASS' if spins <= 2 else 'FAIL'}",
              f"Line (ii) non-spin HD paired CI lower bound > -0.02: **{lo_ci:+.3f}** -> {'PASS' if lo_ci is not None and lo_ci > -0.02 else 'FAIL'}"
              f" (crashed scenarios {miss or 'none'}; reading 2, crash = 0: {m2[0]:+.3f} [{m2[1]:+.3f}, {m2[2]:+.3f}])"]
    per = ["| scenario | " + " | ".join(by) + " |", "|---|" + "---|" * len(by)]
    for s in SPIN10:
        per.append(f"| {s} | " + " | ".join(f"{'SPIN ' if by[a][s]['spin'] else ''}{by[a][s]['max_abs_e']:.0f} deg, HD {by[a][s]['hdscore']:.3f}, "
                                            f"{by[a][s]['end']} {by[a][s]['steps']} st" if s in by[a] else "-" for a in by) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n\nPer baseline-spin scenario (max heading error vs route, HD, end, steps):\n\n" + "\n".join(per) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(L))
    print("\n".join(per))


if __name__ == "__main__":
    main(*sys.argv[1:5])
