#!/usr/bin/env python3
"""GARAGE020: the host's py123d_garage v0.2.0 baseline on our PAI 281 exam (Harmonizer off), job bodies of garage020.sh.

  rescore --variant old|new --read DIR --root DIR   every rollout.asl of the read re-evaluated with AlpaSim's eval (scorer_v2.cmd_score on this
                                                    read; run in the AlpaSim venv, PYTHONPATH=<1b024f0 eval src> for new)
  report --facts CSV --rescore DIR --theirs DIR --out DIR   r2.py r281 tables (garage vs NTV0 / R6DAG / R8K, ACC + GAP, Harmonizer off),
                                                    new zeros on NTV0-perfect scenes, old vs fixed corridor scorer, the host's own per-scene
                                                    results (Harmonizer on, e121e59) on the same scenes; per-scene csv
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
FACTS = DATA / "runs/pai_fix/facts"
LABEL = "GARAGE020"
REFS = {"NTV0-ag": "NTV0-L18-s0-281ag", **{f"{a}-s{s}-ag": f"{a}-L18-s{s}-281ag" for a in ("R6DAG", "R8K") for s in (0, 1, 2)}}
PERFECT = 0.98          # pai_fix partial line: 0 < score < 0.98 is a partial


def cmd_rescore(a):
    sys.path.insert(0, str(HERE))
    import scorer_v2 as S
    S.READS[LABEL] = (str(Path(a.read).resolve().relative_to(S.RUNS)), "")
    S.cmd_score(argparse.Namespace(variant=a.variant, root=a.root, reads=LABEL, limit=None, workers=a.workers))


def scores_json(path):
    """clipgt_id -> score of one results-summary.json (one rollout per scene)."""
    return {r["clipgt_id"]: r["score"] for r in json.load(open(path))["rollouts"]}


def cmd_report(a):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(REPO))
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reads = [f"{LABEL}={a.facts}"] + [f"{k}={FACTS / v}.csv" for k, v in REFS.items()]
    pairs = [f"{LABEL}:{k}" for k in REFS]
    subprocess.run([sys.executable, str(REPO / "experiments/round2/scripts/r2.py"), "r281", "--read", *reads, "--pairs", *pairs,
                    "--out", str(out / "pai281_garage.md"), "--title",
                    "PAI 281, Harmonizer OFF: py123d_garage v0.2.0 (ltf-base-joint64-68M posttrain seed 0, conditional, as shipped) vs our ACC + GAP reads"],
                   check=True, stdout=subprocess.DEVNULL)
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (r.split("=", 1) for r in reads)}
    g, n0 = F[LABEL], F["NTV0-ag"]
    L = ["# GARAGE020 extras", ""]
    # new zeros / partials on scenes NTV0 drives perfectly (decision 443's count)
    ok = lambda F: F.score.notna() & ~F.rpc.fillna(False).astype(bool)  # noqa: E731
    perf = n0.index[ok(n0) & (n0.score >= PERFECT)]
    L += [f"## On the {len(perf)} scenes NTV0-ag scores >= {PERFECT}", "", "| read | n | new zeros | new partials | zero classes |", "|:--|--:|--:|--:|:--|"]
    rows = []
    for k, X in F.items():
        x = X.reindex(perf)
        x = x[ok(x)]
        z = x[x.score == 0]
        rows.append(dict(read=k, n=len(x), new_zeros=len(z), new_partials=int(((x.score > 0) & (x.score < PERFECT)).sum()),
                         classes=", ".join(f"{c} {n}" for c, n in z.fclass.value_counts().items())))
        L.append(f"| {k} | {rows[-1]['n']} | {rows[-1]['new_zeros']} | {rows[-1]['new_partials']} | {rows[-1]['classes']} |")
    pd.DataFrame(rows).to_csv(out / "perfect_ntv0.csv", index=False)
    # old vs fixed corridor scorer on the garage read (every rollout re-evaluated from its ASL)
    per = pd.DataFrame({"garage_old": g.score})
    rs = {}
    for v in ("old", "new"):
        f = Path(a.rescore) / v / "rollouts.csv"
        if f.exists():
            r = pd.read_csv(f)
            rs[v] = r
            per[f"garage_rescore_{v}"] = r.groupby("scene").score.last()
            per[f"garage_fail_{v}"] = r.groupby("scene").failure_reason.last()
    L += ["", "## Corridor scorer (garage read, every rollout re-evaluated from its rollout.asl)", ""]
    if "old" in rs:
        same = rs["old"].parquet_same
        L.append(f"- old scorer re-evaluation reproduces the stored metrics.parquet: {int(same.sum())} / {int(same.notna().sum())} rollouts")
    if "new" in rs:
        m = per.dropna(subset=["garage_old", "garage_rescore_new"])
        d = stats.paired(m.garage_rescore_new.to_numpy(float), m.garage_old.to_numpy(float))
        ch = m[(m.garage_rescore_new - m.garage_old).abs() > 1e-9]
        L.append(f"- fixed (1b024f0) - old: {len(m)} scenes, mean {m.garage_old.mean():.3f} -> {m.garage_rescore_new.mean():.3f} "
                 f"({stats.fmt(d, '+.3f')}); zeros {int((m.garage_old == 0).sum())} -> {int((m.garage_rescore_new == 0).sum())}; {len(ch)} scenes change")
        L.append(f"- fixed-scorer failure reasons (zeros): " + ", ".join(f"{k} {n}" for k, n in per[per.garage_rescore_new == 0].garage_fail_new.value_counts().items()))
        nr = Path(a.ref_rescore) / "new/rollouts.csv"
        if nr.exists():                                    # the stored reads rescored by scorer_v2 (zero rollouts re-evaluated)
            R = pd.read_csv(nr)
            L += ["", "Paired on the fixed scorer (stored reads: scorer_v2's rescoring; their nonzero rollouts keep the stored score):", "",
                  "| ref | n | garage | ref | garage - ref [95 % CI] |", "|:--|--:|--:|--:|--:|"]
            for k, v in REFS.items():
                lab = v.replace("-281ag", "-ag")
                s = R[R.read == lab].groupby("scene").score.last()
                if not len(s):
                    continue
                per[f"{k}_new"] = s
                j = per[["garage_rescore_new", f"{k}_new"]].dropna()
                j = j[j.index.isin(g.index[ok(g)]) & j.index.isin(F[k].index[ok(F[k])])]
                d = stats.paired(j.garage_rescore_new.to_numpy(float), j[f"{k}_new"].to_numpy(float))
                L.append(f"| {k} | {len(j)} | {j.garage_rescore_new.mean():.3f} | {j[f'{k}_new'].mean():.3f} | {stats.fmt(d, '+.3f')} |")
    # the host's own per-scene results (Harmonizer ON, AlpaSim e121e59 = fixed corridor scorer, their 2 Hz camera / skip timing)
    th = {}
    for s in (0, 1, 2):
        f = Path(a.theirs) / f"seed_{s}/results-summary.json"
        if f.exists():
            th[s] = pd.Series(scores_json(f))
            per[f"host_s{s}_harmon"] = th[s]
    if th:
        L += ["", "## The host's own per-scene results (Harmonizer ON, e121e59, their timing), on our scenes", "",
              "| host seed | n (ours scored) | host mean | ours (fixed scorer) | ours - host [95 % CI] | host mean on all their scenes |", "|:--|--:|--:|--:|--:|--:|"]
        col = "garage_rescore_new" if "garage_rescore_new" in per else "garage_old"
        for s, t in th.items():
            j = per[[col, f"host_s{s}_harmon"]].dropna()
            j = j[j.index.isin(g.index[ok(g)])]
            d = stats.paired(j[col].to_numpy(float), j[f"host_s{s}_harmon"].to_numpy(float))
            L.append(f"| {s} | {len(j)} | {j[f'host_s{s}_harmon'].mean():.3f} | {j[col].mean():.3f} | {stats.fmt(d, '+.3f')} | {t.mean():.3f} ({len(t)}) |")
        if 0 in th:
            j = per[[col, "host_s0_harmon"]].dropna()
            L.append(f"\n- seed 0, same scene: both zero {int(((j[col] == 0) & (j.host_s0_harmon == 0)).sum())}, ours only {int(((j[col] == 0) & (j.host_s0_harmon > 0)).sum())}, "
                     f"host only {int(((j[col] > 0) & (j.host_s0_harmon == 0)).sum())}; score correlation {np.corrcoef(j[col], j.host_s0_harmon)[0, 1]:.2f}")
    per["garage_fclass"] = g.fclass
    per["ntv0_ag"] = n0.score
    per.index.name = "scene"
    per.round(4).to_csv(out / "per_scene.csv")
    (out / "extras.md").write_text("\n".join(L) + "\n")
    print((out / "pai281_garage.md").read_text().split("\n## ")[0])
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("rescore")
    q.add_argument("--variant", choices=("old", "new"), required=True), q.add_argument("--read", required=True), q.add_argument("--root", required=True)
    q.add_argument("--workers", type=int, default=0)
    q = sp.add_parser("report")
    q.add_argument("--facts", required=True), q.add_argument("--rescore", required=True), q.add_argument("--theirs", required=True)
    q.add_argument("--ref-rescore", default=str(DATA / "runs/alpasim/scorer_v2")), q.add_argument("--out", required=True)
    a = ap.parse_args()
    {"rescore": cmd_rescore, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
