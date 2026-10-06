"""Joint action-head lane, stage 1 gate (plans/2026-10-07-joint-action-prereg.md). Box, envs/hugsim (reads bench run dirs).

  python experiments/op_parity/scripts/pp_joint_report.py [--seed 0] [--arms JC JL JW] [--base HP]
  -> experiments/op_parity/results/joint_action/{stage1.md, stage1_runs.csv, gate_s<seed>.json}

Per (model, preset) on turn23 and spin10: HD, classes, spins (bench units.csv), heading-rate sign flips per 100 moving steps and
oscillating runs (definition of pp_specplan_report.osc), requested curvature |kappa| and its mean step change (zs_steps.jsonl).
Gate per arm (prereg): offline gain_all >= 0.85, |dev ADE / drift_off - base| <= 0.03 m; d1 = arm spec - base spec (turn23 HD) >= 0.03;
spins (turn23 + spin10) <= base spec + 1; mean flips <= 0.5 / 100 moving steps; no oscillating run; g_hist_cmd < 0.5.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/hugsim/scripts"), str(REPO / "experiments/op_parity/scripts")]
OUT = REPO / "experiments/op_parity/results/joint_action"
PRESETS = ("spec", "spec_plan_smooth")
SETS = ("turn23", "spin10")


def runs_of(model, preset):
    from jevdrive.bench.sets import hugsim_scenarios
    d = Path(os.environ["DATA_DIR"]) / "runs/bench/hugsim" / f"{model}_{preset}"
    if not (d / "units.csv").exists():
        return []
    member = {s: {Path(p).stem for p in hugsim_scenarios(s)} for s in SETS}
    rows = []
    for r in csv.DictReader(open(d / "units.csv")):
        sets = [s for s in SETS if r["scenario"] in member[s]]
        if sets:
            rows.append(r | {"model": model, "preset": preset, "sets": "+".join(sets)})
    return rows


def behaviour(r):
    import spin_analysis as SA
    from pp_specplan_report import osc
    d = Path(r["run_dir"])
    try:
        _, th, v, *_ = SA.load_run(d, "cinque")
    except Exception as e:  # noqa: BLE001
        print("no trace", d, e)
        return {}
    w = np.degrees(np.diff(th))
    fl, fr = osc(w, v)
    k = []
    zf = d / "zs_steps.jsonl"
    if zf.exists():
        k = [x["kappa"] for x in (json.loads(l) for l in open(zf)) if x.get("kappa") is not None]
    k = np.asarray(k, float)
    return dict(flips=fl, flip_rate=fr, oscillates=bool(fl >= 8 and fr >= 0.15),
                k_abs=float(np.abs(k).mean()) if len(k) else np.nan, k_step=float(np.abs(np.diff(k)).mean()) if len(k) > 1 else np.nan)


def boot(d, B=10000, seed=0):
    d = np.asarray(d, float)
    i = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    return float(d.mean()), *np.percentile(d[i].mean(1), [2.5, 97.5]).tolist()


def train_summary(tag):
    root = Path(os.environ["DATA_DIR"]) / "runs/op_parity" / f"train-{tag}"
    for d in sorted(root.glob("*"), reverse=True):
        if (d / "DONE").exists():
            return json.loads((d / "DONE").read_text())
    return {}


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    base = f"{a.base}-F-s{a.seed}"
    models = [base] + [f"{x}-F-s{a.seed}" for x in a.arms]
    rows = []
    for m in models:
        for p in PRESETS:
            for r in runs_of(m, p):
                rows.append(r | behaviour(r))
    keys = ["model", "preset", "sets", "scenario", "hdscore", "rc", "cls", "spin", "launch_stall", "stuck", "flips", "flip_rate", "oscillates", "k_abs", "k_step"]
    with open(OUT / "stage1_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    probe = json.loads((OUT / f"probe_s{a.seed}.json").read_text()) if (OUT / f"probe_s{a.seed}.json").exists() else {}
    tsum = {m: train_summary(m) for m in models}

    def sel(m, p, s=None):
        return {r["scenario"]: r for r in rows if r["model"] == m and r["preset"] == p and (s is None or s in r["sets"].split("+"))}

    def summ(m, p, s):
        d = list(sel(m, p, s).values())
        if not d:
            return {}
        hd = np.array([float(r["hdscore"]) for r in d])
        return dict(n=len(d), HD=float(hd.mean()), complete=sum(r["cls"] == "complete" for r in d), spins=sum(r["spin"] == "True" for r in d),
                    fg=sum(r["cls"] == "fg_coll" for r in d), bg=sum(r["cls"] == "bg_coll" for r in d), off=sum(r["cls"] == "off_route" for r in d),
                    flips100=100 * float(np.nanmean([r.get("flip_rate", np.nan) for r in d])), osc=sum(bool(r.get("oscillates")) for r in d),
                    k_abs=float(np.nanmean([r.get("k_abs", np.nan) for r in d])), k_step=float(np.nanmean([r.get("k_step", np.nan) for r in d])))

    S = {(m, p, s): summ(m, p, s) for m in models for p in PRESETS for s in SETS + (None,)}

    def paired(m1, p1, m2, p2, s="turn23"):
        x, y = sel(m1, p1, s), sel(m2, p2, s)
        common = sorted(set(x) & set(y))
        if not common:
            return None
        return boot([float(x[c]["hdscore"]) - float(y[c]["hdscore"]) for c in common]) + (len(common),)

    bs = S[(base, "spec", None)]
    gate = {}
    for x in a.arms:
        m = f"{x}-F-s{a.seed}"
        pr, tb, tm = probe.get(m, {}), tsum[base], tsum[m]
        all_ = S[(m, "spec", None)]
        d1, d2 = paired(m, "spec", base, "spec"), paired(m, "spec", m, "spec_plan_smooth")
        d3 = paired(m, "spec", base, "spec_plan_smooth")
        chk = dict(
            gain=pr.get("gain_all", 0) >= 0.85,
            plan_kept=all(abs(tm.get(k, 9) - tb.get(k, 0)) <= 0.03 for k in ("dev_ade", "dev_drift_off")),
            d1=bool(d1 and d1[0] >= 0.03),
            spins=bool(all_ and bs and all_["spins"] <= bs["spins"] + 1),
            flips=bool(all_ and all_["flips100"] <= 0.5),
            no_osc=bool(all_ and all_["osc"] == 0),
            g_hist=pr.get("g_hist_cmd", 9) < 0.5)
        guard_ok = all(chk[k] for k in ("spins", "flips", "no_osc", "g_hist"))
        verdict = "pass" if chk["gain"] and chk["plan_kept"] and chk["d1"] and guard_ok else \
                  "feedback" if chk["d1"] and not guard_ok else "fail"
        gate[m] = dict(checks=chk, verdict=verdict, d1=d1, d2=d2, d3=d3,
                       turn23_spec=S[(m, "spec", "turn23")].get("HD"), turn23_smooth=S[(m, "spec_plan_smooth", "turn23")].get("HD"))
    passing = [m for m, g in gate.items() if g["verdict"] == "pass"]
    best = None
    if passing:
        top = max(gate[m]["turn23_spec"] for m in passing)
        near = [m for m in passing if top - gate[m]["turn23_spec"] < 0.02]
        best = max(near, key=lambda m: (gate[m]["d2"] or (-9,))[0])
    (OUT / f"gate_s{a.seed}.json").write_text(json.dumps(dict(arms=gate, best=best, base=base), indent=1, default=float))

    f = lambda v, d=3: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
    L = [f"# Joint action head, stage 1 small read (seed {a.seed})\n",
         "Generated by `scripts/pp_joint_report.py` from the bench runs; plan: [../../plans/2026-10-07-joint-action-prereg.md](../../plans/2026-10-07-joint-action-prereg.md).\n",
         "## HUGSIM per (model, preset, set)\n",
         "| model | preset | set | n | HD | complete | spins | fg | bg | off | flips/100 | osc | abs kappa | kappa step |", "|" + "---|" * 14]
    for m in models:
        for p in PRESETS:
            for s in SETS:
                d = S[(m, p, s)]
                if d:
                    L.append(f"| {m} | {p} | {s} | {d['n']} | {f(d['HD'])} | {d['complete']} | {d['spins']} | {d['fg']} | {d['bg']} | {d['off']} | "
                             f"{f(d['flips100'], 2)} | {d['osc']} | {f(d['k_abs'], 4)} | {f(d['k_step'], 4)} |")
    L += ["\n## Offline probe (dev rows, v0 > 3 m/s) and plan guard\n",
          "| model | gain all | gain turn | gain 3-8 | gain >8 | cmd on plan | corr | g_hist cmd | g_hist plan | latdrop slope | dev ADE | drift_off |", "|" + "---|" * 12]
    for m in models:
        pr, t = probe.get(m, {}), tsum[m]
        L.append(f"| {m} | " + " | ".join(f(pr.get(k)) for k in ("gain_all", "gain_turn", "gain_v3_8", "gain_v8", "cmd_on_plan", "cmd_plan_corr", "g_hist_cmd",
                                                                   "g_hist_plan", "latdrop_slope")) + f" | {f(t.get('dev_ade'))} | {f(t.get('dev_drift_off'))} |")
    L += ["\n## Gate (turn23 HD, paired over scenarios, bootstrap 95% CI)\n",
          "| arm | spec - base spec | spec - own smooth | spec - base smooth | checks failed | verdict |", "|---|---|---|---|---|---|"]
    ci = lambda d: "-" if not d else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] (n {d[3]})"  # noqa: E731
    for m, g in gate.items():
        L.append(f"| {m} | {ci(g['d1'])} | {ci(g['d2'])} | {ci(g['d3'])} | {', '.join(k for k, v in g['checks'].items() if not v) or 'none'} | {g['verdict']} |")
    L.append(f"\nBase {base}: turn23 spec {f(S[(base, 'spec', 'turn23')].get('HD'))}, spec_plan_smooth {f(S[(base, 'spec_plan_smooth', 'turn23')].get('HD'))}. "
             f"Best passing arm: {best or 'none'}.")
    (OUT / "stage1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["JC", "JL", "JW"])
    ap.add_argument("--base", default="HP")
    main(ap.parse_args())
