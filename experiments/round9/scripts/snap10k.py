#!/usr/bin/env python3
"""ROUND9 SNAP10K (plans/2026-10-24-snap10k.md). report --dir O: PCL1-F-s0x10000 (PAI 281 ACC + GAP, Harmonizer OFF) next to RB0-F-s0 (10 000 steps, own
schedule end), RB0L-F-s0 (19 500 offline), PCL1-F-s0 (+ S rows from step 10 000), PCL1L-F-s0 (same + lr2 schedule) on common scenes: means (new scorer
primary, old facts in brackets), zero classes, paired CIs -> O/results/snap10k.{md,json}. Box, repo root, .venv."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
SNAP, RB, RBL, P1, P1L = "PCL1-F-s0x10000", "RB0-F-s0", "RB0L-F-s0", "PCL1-F-s0", "PCL1L-F-s0"
LABS = [SNAP, RB, RBL, P1, P1L]
LANE = {RBL: "rb0_long", P1: "pool_pilot", P1L: "p1l"}


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"])


def gpu_hours(o):
    from jevdrive.cl import pool as P
    J = P.snapshot()[0]["jobs"]
    tot = rd = 0.0
    for ln in (o / "jobs.txt").read_text().splitlines():
        j = J.get(ln.split()[0]) if ln.split() else None
        if not j or not j.get("t0") or j["spec"]["vram_gb"] <= 1:
            continue
        h = ((j.get("t1") or time.time()) - j["t0"]) / 3600
        tot += h
        rd += h if j["spec"]["name"].startswith("pai-") else 0.0
    return dict(gpu_job_h=tot, read_h=rd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub", choices=["report"])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    PS = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]

    def rolls(f, lab):
        R = pd.read_csv(f)
        y = R[R.read == f"{lab}-ag"]
        assert len(y), f"no new-scorer rollouts for {lab} in {f}"
        return y.drop_duplicates("scene", keep="last").set_index("scene").score

    NEW = {t: rolls(DD / "runs/round9" / LANE[t] / "sv2/new/rollouts.csv", t) for t in LANE}
    NEW[RB] = PS[f"{RB}-ag"]
    NEW[SNAP] = rolls(o / "sv2/new/rollouts.csv", SNAP)
    OLD = {t: pd.read_csv(DD / "runs/pai_fix/facts" / f"{t}-281ag.csv").set_index("scene") for t in LABS}
    rpc = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    com = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in LABS]) & set.intersection(*[set(NEW[t].dropna().index) for t in LABS]) - rpc)
    sn = {t: NEW[t].reindex(com).to_numpy(float) for t in LABS}
    so = {t: OLD[t].score.reindex(com).to_numpy(float) for t in LABS}
    res = dict(scenes=len(com), rpc_excluded=len(rpc), reads={}, pairs={})
    for t in LABS:
        m = PF.metrics(OLD[t].reindex(com))
        res["reads"][t] = dict(new_mean=float(sn[t].mean()), old_mean=float(so[t].mean()), new_zeros=int((sn[t] == 0).sum()), old_zeros=int((so[t] == 0).sum()),
                               zcls={c: int(m.get(f"z_{c}", 0)) for c in PF.ZCLS})
    for x, y in ((SNAP, RB), (SNAP, RBL), (SNAP, P1), (SNAP, P1L), (P1, SNAP), (P1L, SNAP), (RBL, RB), (P1, RBL), (P1L, P1)):
        res["pairs"][f"{x} - {y}"] = dict(new=stats.paired(sn[x], sn[y]), old=stats.paired(so[x], so[y]), zeros_old=int((so[x] == 0).sum() - (so[y] == 0).sum()))
    res["cost"] = gpu_hours(o)
    L = ["# SNAP10K: PCL1-F-s0x10000 (= RB0L-F-s0x10000, tensor-identical), PAI 281 ACC + GAP, Harmonizer OFF", "",
         f"{res['scenes']} common scenes ({res['rpc_excluded']} RPC scenes excluded). New scorer (scorer_v2 new) primary, old in brackets; zero classes from old-scorer facts.", "",
         "| read | new mean (old) | zeros new (old) | " + " | ".join(PF.ZCLS) + " |", "|:--|--:|--:|" + "--:|" * len(PF.ZCLS)]
    for k, v in res["reads"].items():
        L.append(f"| {k} | {v['new_mean']:.3f} ({v['old_mean']:.3f}) | {v['new_zeros']} ({v['old_zeros']}) | " + " | ".join(str(v["zcls"][c]) for c in PF.ZCLS) + " |")
    L += ["", "| pair (95 % bootstrap CI, scenes as units) | new | old | zeros (old) |", "|:--|--:|--:|--:|"]
    for k, p in res["pairs"].items():
        L.append(f"| {k} | {f3(p['new'])} | {f3(p['old'])} | {p['zeros_old']:+d} |")
    L += ["", f"GPU: {res['cost']['gpu_job_h']:.2f} card-hours of job time ({res['cost']['read_h']:.2f} in read chunks)."]
    (o / "results").mkdir(exist_ok=True)
    (o / "results/snap10k.md").write_text("\n".join(L) + "\n")
    (o / "results/snap10k.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
