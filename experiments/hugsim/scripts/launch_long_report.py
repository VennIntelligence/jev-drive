"""Longitudinal launch assist in HUGSIM closed loop (plans/2026-10-04-launch-long-prereg.md): spins, HD paired vs the exam baseline and the two
same-day reruns, front collisions, time below 3 m/s in the first 10 s, assist reasons. Existing arms come from
results/lowspeed_ctrl_selective/lowsel_runs.csv (base, base_rerun, base_rerun2); the new arm is scored from the run dir.  Box, envs/hugsim python:
    python experiments/hugsim/scripts/launch_long_report.py $DATA_DIR/runs/hugsim-launchlong/closed $DATA_DIR/runs/hugsim-exam/scored-op \
        $DATA_DIR/tmp_spin/routes.json experiments/hugsim/results/launch_long
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derot_report import SPIN10, runs  # noqa: E402
from lowspeed_report import boot, events  # noqa: E402
from spin_analysis import analyse, load_run  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TAG = "cinque-launchlong"
NEWTAG = {"cinque-launchlong": "launch_long"}
D = Path(__import__("os").environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
DIRS = {"base": (D / "runs/hugsim-exam/scored-op", "cinque-fixed"), "base_rerun": (D / "runs/hugsim-lowspeed/closed", "cinque-fixed-base"),
        "base_rerun2": (D / "runs/hugsim-lowsel/closed", "cinque-fixed-base2")}


def low_time(d, n=41):
    """Seconds below 3 m/s in the first 10 s (41 states at 0.25 s)."""
    v = [json.loads(x)["v"] for x in open(Path(d) / "zs_steps.jsonl") if '"step"' in x][:n]
    return 0.25 * sum(x < 3 for x in v), (v[4] if len(v) > 4 else np.nan), (v[8] if len(v) > 8 else np.nan), (v[12] if len(v) > 12 else np.nan)


def main(lo, exam, routes, out):
    lo, out = Path(lo), Path(out)
    routes = json.load(open(routes))
    by = {}
    for r in csv.DictReader(open(REPO / "experiments/hugsim/results/lowspeed_ctrl_selective/lowsel_runs.csv")):
        if r["arm"] in ("base", "base_rerun", "base_rerun2"):
            by.setdefault(r["arm"], {})[r["scenario"]] = dict(r, spin=r["spin"] == "True", hdscore=float(r["hdscore"]), rc=float(r["rc"]), steps=int(r["steps"]),
                                                               n_init=int(r["n_init"]), spin_init=int(r["spin_init"]), n_relaunch=int(r["n_relaunch"]), spin_relaunch=int(r["spin_relaunch"]))
    # low-speed time of the existing arms (from their run dirs)
    low = {a: {} for a in DIRS}
    for a, (root, tag) in DIRS.items():
        for (t, scen), (r, d) in runs(root / "results.csv", root, {tag}).items():
            low[a][scen] = low_time(d)
    R = runs(lo / "results.csv", lo, set(NEWTAG))
    ll, why = {}, Counter()
    low["launch_long"] = {}
    for (tag, scen), (r, d) in sorted(R.items()):
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        ev = events(v, res["spin"], res.get("start", -1))
        by.setdefault("launch_long", {})[scen] = dict(arm="launch_long", scenario=scen, spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1), hdscore=float(r["hdscore"]),
                                                      rc=float(r["rc"]), nc=r["nc"], end=r["end"], steps=int(r["steps"]), n_init=1, spin_init=int(ev[0][2]),
                                                      n_relaunch=len(ev) - 1, spin_relaunch=sum(e[2] for e in ev[1:]), vmax=round(float(v.max()), 2))
        low["launch_long"][scen] = low_time(d)
        recs = [json.loads(x) for x in open(d / "zs_steps.jsonl") if '"step"' in x]
        c = Counter(x.get("ll", "n/a") for x in recs)
        ll[scen] = dict(c)
        why.update(c)
        by["launch_long"][scen]["ll_on"] = c.get("on", 0)
        # front collision in the first 15 s with the rule on during the preceding 3 s
        by["launch_long"][scen]["fg_near_on"] = int(r["end"] == "fg_collision" and r["steps"] and int(r["steps"]) <= 60 and any(x.get("ll") == "on" for x in recs[-12:]))
    base = by["base"]
    ns_all = [s for s in base if not base[s]["spin"]]
    d = by["launch_long"]
    L = ["| arm | n | spins | on the 10 | new spins | initial-launch spins / n | re-launch spins / n | fg_collision | max_steps | HD mean | RC mean | below 3 m/s in first 10 s, median s | v at +1 / +2 / +3 s, median | non-spin HD paired vs exam base | vs rerun 1 | vs rerun 2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm in ("base", "base_rerun", "base_rerun2", "launch_long"):
        a = by[arm]
        sc = sorted(a)
        ns = [s for s in ns_all if s in a]
        new = [s for s in sc if a[s]["spin"] and not base[s]["spin"]]
        cells = []
        for ref in ("base", "base_rerun", "base_rerun2"):
            if arm == "launch_long":
                dl = np.array([a[s]["hdscore"] - by[ref][s]["hdscore"] for s in ns if s in by[ref]])
                m = boot(dl)
                summ[f"launch_long_vs_{ref}"] = list(m)
                cells.append(f"{m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}]")
            else:
                cells.append("")
        lt = np.array([low[arm][s] for s in sc if s in low[arm]])
        summ[arm] = dict(n=len(sc), spins=sum(a[s]["spin"] for s in sc), spins10=sum(a[s]["spin"] for s in sc if s in SPIN10), new=new,
                         fg=sum(a[s]["end"] == "fg_collision" for s in sc), max_steps=sum(a[s]["end"] == "max_steps" for s in sc),
                         hd=float(np.mean([a[s]["hdscore"] for s in sc])), rc=float(np.mean([a[s]["rc"] for s in sc])),
                         ev_init=[sum(a[s]["n_init"] for s in sc), sum(a[s]["spin_init"] for s in sc)], ev_re=[sum(a[s]["n_relaunch"] for s in sc), sum(a[s]["spin_relaunch"] for s in sc)],
                         low_med=float(np.median(lt[:, 0])), v_med=[float(np.nanmedian(lt[:, k])) for k in (1, 2, 3)])
        k = summ[arm]
        L.append(f"| {arm} | {len(sc)} | {k['spins']} | {k['spins10']} / {len(SPIN10)} | {len(new)} {new or ''} | {k['ev_init'][1]} / {k['ev_init'][0]} | {k['ev_re'][1]} / {k['ev_re'][0]} | "
                 f"{k['fg']} | {k['max_steps']} | {k['hd']:.3f} | {k['rc']:.3f} | {k['low_med']:.2f} | {k['v_med'][0]:.2f} / {k['v_med'][1]:.2f} / {k['v_med'][2]:.2f} | {cells[0]} | {cells[1]} | {cells[2]} |")
    spins, lo_ci = summ["launch_long"]["spins"], summ["launch_long_vs_base"][1]
    fg_new = {ref: [s for s in d if d[s]["end"] == "fg_collision" and by[ref][s]["end"] != "fg_collision"] for ref in ("base", "base_rerun", "base_rerun2")}
    fg_new_near = [s for s in d if d[s]["fg_near_on"] and all(by[ref][s]["end"] != "fg_collision" for ref in ("base", "base_rerun", "base_rerun2"))]
    L += ["", f"Line (i) spins <= 2 (base {summ['base']['spins']}, reruns {summ['base_rerun']['spins']}, {summ['base_rerun2']['spins']}): **{spins}** -> {'PASS' if spins <= 2 else 'FAIL'}",
          f"Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **{lo_ci:+.3f}** -> {'PASS' if lo_ci > -0.02 else 'FAIL'}",
          f"Front collisions: {summ['launch_long']['fg']} (base {summ['base']['fg']}, reruns {summ['base_rerun']['fg']}, {summ['base_rerun2']['fg']}); scenes with fg_collision where a baseline arm has none: "
          + "; ".join(f"vs {k}: {v or 'none'}" for k, v in fg_new.items()) + f"; of those, collision within 15 s with the assist on in the last 3 s: {fg_new_near or 'none'} "
          f"-> safety check {'FAIL' if len(fg_new_near) > 2 else 'ok'}",
          f"Assist reasons over all steps of the 64 runs: {dict(why)}"]
    worst = sorted(ns_all, key=lambda s: d[s]["hdscore"] - base[s]["hdscore"])[:6]
    best = sorted(ns_all, key=lambda s: -(d[s]["hdscore"] - base[s]["hdscore"]))[:4]
    L += ["", "Largest non-spin HD losses vs exam base: " + "; ".join(f"{s} {base[s]['hdscore']:.3f} -> {d[s]['hdscore']:.3f} ({d[s]['end']}, ll on {d[s]['ll_on']} steps)" for s in worst),
          "Largest gains: " + "; ".join(f"{s} {base[s]['hdscore']:.3f} -> {d[s]['hdscore']:.3f} ({d[s]['end']})" for s in best),
          "Surviving / new spins: " + "; ".join(f"{s} (max err {d[s]['max_abs_e']} deg, base spin {base[s]['spin']}, ll on {d[s]['ll_on']})" for s in d if d[s]["spin"])]
    per = ["| scenario | base | rerun 1 | rerun 2 | launch_long |", "|---|---|---|---|---|"]
    for s in SPIN10:
        per.append(f"| {s} | " + " | ".join(f"{'SPIN ' if by[a][s]['spin'] else ''}{float(by[a][s]['max_abs_e']):.0f} deg, HD {by[a][s]['hdscore']:.3f}, {by[a][s]['end']}" for a in ("base", "base_rerun", "base_rerun2", "launch_long")) + " |")
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for a in by.values() for r in a.values()]
    keys = sorted({k for r in rows for k in r})
    with open(out / "launch_long_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text("\n".join(L) + "\n\nPer baseline-spin scenario:\n\n" + "\n".join(per) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(summ, ll=ll, fg_new=fg_new), indent=1))
    print("\n".join(L))
    print("\n".join(per))


if __name__ == "__main__":
    main(*sys.argv[1:5])
