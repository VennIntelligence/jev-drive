"""Readouts (a) and (b) of op-adapt H round 2 (plans/2026-10-04-launch-pairs-prereg.md), op-train python on the box:

    python experiments/op_adapt_h/scripts/h_launch_report.py <out dir> <tag> [<tag> ...]

(a) launch large-signal gain on decision 100's 9 HUGSIM spin logs: s = (phi1 logged history - phi1 de-rotated) / H on steps with
    1 <= |H| <= 15 deg, pooled median (lean_report.py's definition); O and it_dw3-s0 come from $DATA_DIR/runs/hugsim-lean/replay.json,
    new tags from $H/launch/<tag>/replay.json; loop growth z of the window kernel with lean_report's controller transfer c. Local gain
    at step 1 (w 2) from local.json likewise.
(b) small-rate G ratio (tag / shipped, stop + low, 0.5 / 1 / 2 deg/s) from $H/rate_probe/ratio_launch.csv.
Dev: $H/runs/<tag>/ldev.json against $H/dev/O_ldev.json. Writes <out>/launch_ab.md, launch_ab.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
sys.path.insert(0, str(REPO / "research"))
from lean_report import controller_c  # noqa: E402

DD = Path(os.environ["DATA_DIR"])
LEAN = DD / "runs" / "hugsim-lean"
H = DD / "runs" / "op_adapt_H"
REF = ("O", "it_dw3-s0")


def zwin(cs):
    r = np.roots(np.r_[1.0, -(1 + cs), np.zeros(5), cs])
    r = r[np.abs(r - 1) > 1e-6]
    return float(np.max(np.abs(r)))


def large(R, m):
    """Per (log, step) large-signal gain of model m in a replay result; NaN off the 1 <= |H| <= 15 window."""
    out = {}
    for key in [k for k in R if not k.startswith("_")]:
        th = np.array(R[key]["theta"])
        Hd = np.array([-np.degrees(th[k] - th[max(0, k - 6)]) for k in range(len(th))])
        pn = np.array([x["phi1"] for x in R[key][f"{m}|normal"]])
        pdr = np.array([x["phi1"] for x in R[key][f"{m}|derot"]])
        sel = (np.abs(Hd) >= 1) & (np.abs(Hd) <= 15)
        out[key] = np.where(sel, (pn - pdr) / np.where(sel, Hd, 1.0), np.nan)
    return out


def local1(Lc, m, w=2.0):
    return float(np.median([(Lc[k][f"{m}|{w:+g}"][0]["phi1"] - Lc[k][f"{m}|{-w:+g}"][0]["phi1"]) / 2 / (1.2 * w)
                            for k in Lc if not k.startswith("_")]))


def main(out, *tags):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = Path(json.load(open(LEAN / "jobs64.json"))[0]["run_dir"]).parents[2]
    c, nc = controller_c(D / "results.csv")
    R0, L0 = json.load(open(LEAN / "replay.json")), json.load(open(LEAN / "local.json"))
    S = {m: large(R0, m) for m in REF}
    LO = {m: local1(L0, m) for m in REF}
    for t in tags:
        p = H / "launch" / t
        if (p / "replay.json").exists():
            S[t] = large(json.load(open(p / "replay.json")), t)
        if (p / "local.json").exists():
            LO[t] = local1(json.load(open(p / "local.json")), t)
    pooled = {m: np.concatenate(list(v.values())) for m, v in S.items()}
    s0 = float(np.nanmedian(pooled["O"]))
    md = [f"# op-adapt H round 2: readouts (a) and (b)\n\nController transfer c = {c:.3f} (n {nc}); shipped large-signal gain s = {s0:.2f}.\n",
          "## (a) Launch large-signal gain (9 HUGSIM spin logs, replay)\n",
          "| model | s (pooled median) | s / shipped | median per-step ratio | window-kernel growth z | local gain step 1 (w 2) | local / shipped |",
          "|---|---|---|---|---|---|---|"]
    res = {"c": c, "a": {}, "b": {}}
    for m in S:
        s = float(np.nanmedian(pooled[m]))
        ratio_step = float(np.nanmedian(pooled[m] / pooled["O"]))
        res["a"][m] = dict(s=s, ratio=s / s0, ratio_step=ratio_step, z=zwin(c * s), local1=LO.get(m), local1_ratio=(LO.get(m) or np.nan) / LO["O"])
        r = res["a"][m]
        md.append(f"| {m} | {s:.2f} | {r['ratio']:.2f} | {ratio_step:.2f} | {r['z']:.2f} | {r['local1']:.2f} | {r['local1_ratio']:.2f} |")
    md.append(f"\nGrowth z at s = 0.2 x shipped: {zwin(c * 0.2 * s0):.2f}; z = 1 needs c s < ~0.15, s < {0.15 / c:.2f}.\n")
    rp = H / "rate_probe" / "ratio_launch.csv"
    if rp.exists():
        ra = pd.read_csv(rp)
        ra = ra[(ra.bin == "low+stop") & ra.rate.isin([0.5, 1.0, 2.0, 10.0])]
        md.append("## (b) Small-rate G, ratio to shipped (decision-92 probe pools, stop + low, cluster bootstrap 95% CI)\n")
        md.append("| model | domain | 0.5 deg/s | 1 deg/s | 2 deg/s | 10 deg/s |\n|---|---|---|---|---|---|")
        for m in ra.model.unique():
            for d in ("pnav", "pwod", "pcarla"):
                q = ra[(ra.model == m) & (ra.domain == d)].set_index("rate")
                md.append(f"| {m} | {d[1:]} | " + " | ".join(f"{q.ratio[r]:.2f} [{q.lo[r]:.2f}, {q.hi[r]:.2f}]" for r in (0.5, 1.0, 2.0, 10.0)) + " |")
            mean = {r: float(ra[(ra.model == m) & (ra.rate == r)].ratio.mean()) for r in (0.5, 1.0, 2.0, 10.0)}
            res["b"][m] = mean
            md.append(f"| {m} | **mean** | " + " | ".join(f"**{mean[r]:.2f}**" for r in (0.5, 1.0, 2.0, 10.0)) + " |")
    rows = {}
    for m in ("O",) + tags:
        p = (H / "dev" / "O_ldev.json") if m == "O" else (H / "runs" / m / "ldev.json")
        if p.exists():
            for d, r in json.load(open(p)).items():
                for k, v in r.items():
                    rows.setdefault((d, k), {})[m] = v
    if rows:
        df = pd.DataFrame(rows).T
        md.append("\n## Launch dev (pool dev splits; GL = plan response per deg of launch fake yaw, G1 = per 1 deg/s on stop + low)\n")
        md.append(df.round(3).to_markdown())
        df.to_csv(out / "ldev.csv")
    (out / "launch_ab.md").write_text("\n".join(md) + "\n")
    (out / "launch_ab.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:])
