"""History de-rotation in HUGSIM closed loop (plans/2026-10-03-history-derotate-plan.md): spin count and HD-Score per arm,
paired against the exam baseline (cinque-fixed), and the amplification-chain figure (history yaw vs plan direction per step).
Box, envs/hugsim python:
    python experiments/hugsim/scripts/derot_report.py $DATA_DIR/runs/hugsim-derot $DATA_DIR/runs/hugsim-exam/scored-op \
        $DATA_DIR/tmp_spin/routes.json experiments/hugsim/results/derot
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spin_analysis import analyse, load_run  # noqa: E402

SPIN10 = [Path(x).stem for x in open(Path(__file__).resolve().parent / "derot_spin10.txt").read().split()]


def runs(results, root, tag_filter):
    out = {}
    for r in csv.DictReader(open(results)):
        if r["tag"] not in tag_filter or r["end"] == "crash":
            continue
        d = Path(r["run_dir"])
        d = root / r["tag"] / d.parent.name / d.name
        out[(r["tag"], r["scenario"])] = (r, d)
    return out


def chain(d):
    """Per step: heading change over the last 1.5 s (6 steps, + = right), plan direction at 1 s (+ = right), speed, derot flag."""
    recs = [json.loads(x) for x in open(d / "zs_steps.jsonl")][1:]
    th = np.unwrap([r["theta"] for r in recs])
    k = np.maximum(np.arange(len(th)) - 6, 0)
    p = np.array([r["plan"][1] for r in recs])
    return dict(t=np.array([r["t"] for r in recs]), hyaw=np.degrees(th - th[k]), heading=np.degrees(th - th[0]),
                phi=np.degrees(np.arctan2(p[:, 0], np.maximum(p[:, 1], 1e-3))), v=np.array([r["v"] for r in recs]),
                derot=np.array([bool(r.get("derot")) for r in recs]))


def main(dero, exam, routes, out):
    dero, exam, out = Path(dero), Path(exam), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    routes = json.load(open(routes))
    R = runs(exam / "results.csv", exam, {"cinque-fixed"}) | runs(dero / "results.csv", dero,
                                                                  {"cinque-fixed-derot3", "cinque-fixed-replay3", "cinque-fixed-base"})
    rows = []
    for (tag, scen), (r, d) in sorted(R.items()):
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        recs = [json.loads(x) for x in open(d / "zs_steps.jsonl")][1:]
        rows.append(dict(arm={"cinque-fixed": "base", "cinque-fixed-derot3": "derot3", "cinque-fixed-replay3": "replay3",
                              "cinque-fixed-base": "base_rerun"}[tag], scenario=scen, dataset=r["dataset"], spin10=scen in SPIN10,
                         spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1), hdscore=float(r["hdscore"]), rc=float(r["rc"]),
                         end=r["end"], steps=int(r["steps"]), derot_steps=sum(bool(x.get("derot")) for x in recs)))
    with open(out / "derot_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["scenario"]] = r
    base = by["base"]
    lines = ["| arm | n | spins (all) | spins on the 10 | new spins | HD mean | HD on the 10 | HD on non-spin (paired delta, 95% CI) | n non-spin |",
             "|---|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    summ = {}
    for arm, d in by.items():
        sc = sorted(d)
        s10 = [s for s in sc if s in SPIN10]
        ns = [s for s in sc if s not in SPIN10]
        new = [s for s in sc if d[s]["spin"] and not base[s]["spin"]]
        dl = np.array([d[s]["hdscore"] - base[s]["hdscore"] for s in ns])
        ci = ""
        if len(dl) and arm != "base":
            b = dl[rng.integers(0, len(dl), (10000, len(dl)))].mean(1)
            ci = f"{dl.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"
        summ[arm] = dict(n=len(sc), spins=sum(d[s]["spin"] for s in sc), spins10=sum(d[s]["spin"] for s in s10), n10=len(s10), new=new,
                         hd=float(np.mean([d[s]["hdscore"] for s in sc])), hd10=float(np.mean([d[s]["hdscore"] for s in s10])) if s10 else None,
                         hd10_base=float(np.mean([base[s]["hdscore"] for s in s10])) if s10 else None,
                         dl_nonspin=float(dl.mean()) if len(dl) else None)
        lines.append(f"| {arm} | {len(sc)} | {summ[arm]['spins']} | {summ[arm]['spins10']} / {len(s10)} | {len(new)} {new if new else ''} | "
                     f"{summ[arm]['hd']:.3f} | {summ[arm]['hd10'] if s10 else float('nan'):.3f} | {ci} | {len(ns)} |")
    per = ["| scenario | " + " | ".join(by) + " |", "|---|" + "---|" * len(by)]
    for s in SPIN10:
        per.append(f"| {s} | " + " | ".join(f"{'SPIN ' if by[a][s]['spin'] else ''}{by[a][s]['max_abs_e']:.0f} deg, HD {by[a][s]['hdscore']:.3f}, "
                                            f"{by[a][s]['end']} {by[a][s]['steps']} st" if s in by[a] else "-" for a in by) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n\nPer spin scenario (max heading error vs route, HD, end, steps):\n\n" + "\n".join(per) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(lines))
    print("\n".join(per))
    fig(R, out)


def fig(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [s for s in ("scene-0013-medium-00", "scene-0528-medium-00", "scene-152217047339-medium-00") if ("cinque-fixed", s) in R]
    arms = [("cinque-fixed", "base (PR #57)", "#c0392b", 2.5), ("cinque-fixed-replay3", "replay, no rotation", "#7f8c8d", 1.0),
            ("cinque-fixed-derot3", "de-rotated < 3 m/s", "#2471a3", 1.5)]
    f, ax = plt.subplots(3, len(cases), figsize=(5.5 * len(cases), 9), squeeze=False, sharex=True)
    for j, s in enumerate(cases):
        for tag, lab, col, lw in arms:
            if (tag, s) not in R:
                continue
            c = chain(R[(tag, s)][1])
            n = int((c["t"] <= 10).sum())
            t = c["t"][:n]
            ax[0, j].plot(t, c["hyaw"][:n], color=col, lw=lw, label=f"{lab}: yaw over the last 1.5 s")
            ax[0, j].plot(t, c["phi"][:n], color=col, lw=lw, ls="--", label=f"{lab}: plan direction at 1 s")
            ax[1, j].plot(t, c["heading"][:n], color=col, lw=lw, label=lab)
            ax[2, j].plot(t, c["v"][:n], color=col, lw=lw, label=lab)
            if tag == "cinque-fixed-derot3":
                on = c["derot"][:n]
                ax[1, j].scatter(t[on], c["heading"][:n][on], color=col, s=6)
        ax[0, j].set(title=s, ylabel="deg (+ right)")
        ax[1, j].set(ylabel="heading since start (deg)\ndots: de-rotated input")
        ax[2, j].set(xlabel="t (s)", ylabel="speed (m/s)")
        for a in ax[:2, j]:
            a.axhline(0, color="k", lw=.5)
        ax[2, j].axhline(3, color="k", lw=.5, ls=":")
    ax[0, 0].legend(fontsize=6.5)
    ax[2, 0].legend(fontsize=7)
    f.tight_layout()
    f.savefig(out / "derot-chain.png", dpi=110)
    print("figure", out / "derot-chain.png")


if __name__ == "__main__":
    main(*sys.argv[1:5])
