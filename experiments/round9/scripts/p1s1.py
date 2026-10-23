#!/usr/bin/env python3
"""ROUND9 P1 SEED 1 (plans/2026-10-23-pool-design.md, section 6 supplement; lane scripts/p1s1.sh). Box, repo root, repo venv. Reuses pcl.py.
  p1s1.py verdict --dir O --out P    PCL1-F-s1 vs PCL1-F-s0, NTV0-L18-s0 / s1, RB0-F-s0 on common scenes: paired CIs (both scorers), 2-seed means,
                                     gap shares, zero classes, dev ADE, cost -> P.{json,md}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl as C  # noqa: E402

DD = C.DD
S1, S0, N0, N1, RB = "PCL1-F-s1", "PCL1-F-s0", "NTV0-L18-s0", "NTV0-L18-s1", "RB0-F-s0"
LABS = [S1, S0, N0, N1, RB]
PP = DD / "runs/round9/pool_pilot"


def cmd_verdict(a):
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(C.REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    OLD = {t: pd.read_csv(DD / "runs/pai_fix/facts" / f"{t}-281ag.csv").set_index("scene") for t in LABS}
    NEW = C.new_scores(PP)                                    # NTV0-s0 / RB0-F (scorer_v2 report), PCL1-F-s0 (pool_pilot sv2)
    P = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    NEW[N1] = P[f"{N1}-ag"]
    x = pd.read_csv(o / "sv2/new/rollouts.csv")
    NEW[S1] = x[x.read == f"{S1}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    rpc = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    common = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in LABS])
                    & set.intersection(*[set(NEW[t].dropna().index) for t in LABS]) - rpc)
    n = len(common)
    Sc = dict(old={t: OLD[t].score.reindex(common).to_numpy(float) for t in LABS}, new={t: NEW[t].reindex(common).to_numpy(float) for t in LABS})
    res = dict(scenes=n, reads={}, pairs={}, seeds2={}, g={})
    for t in LABS:
        res["reads"][t] = dict(old=PF.metrics(OLD[t].reindex(common)), new_mean=float(Sc["new"][t].mean()), new_zeros=int((Sc["new"][t] == 0).sum()),
                               dev_ade=C.dev_ade(t))
    for y in (S0, N0, N1, RB):
        res["pairs"][f"{S1}:{y}"] = {sc: stats.paired(Sc[sc][S1], Sc[sc][y]) for sc in ("new", "old")}
    for sc in ("new", "old"):
        p2, n2 = (Sc[sc][S1] + Sc[sc][S0]) / 2, (Sc[sc][N0] + Sc[sc][N1]) / 2
        res["seeds2"][sc] = dict(P1=float(p2.mean()), NTV0=float(n2.mean()), diff=stats.paired(p2, n2), vs_RB0F=stats.paired(p2, Sc[sc][RB]))
        b, nt = Sc[sc][RB].mean(), Sc[sc][N0].mean()
        nt2 = n2.mean()
        res["g"][sc] = dict(rb0f=float(b), ntv0_s0=float(nt), ntv0_2seed=float(nt2), s1=float((Sc[sc][S1].mean() - b) / (nt - b)),
                            s0=float((Sc[sc][S0].mean() - b) / (nt - b)), two_seed=float((p2.mean() - b) / (nt - b)),
                            two_seed_vs_ntv0_2seed=float((p2.mean() - b) / (nt2 - b)))
    f3 = C.f3
    L = [f"# P1 seed 1 (PCL1-F-s1) vs PCL1-F-s0 (PAI 281, ACC + GAP, Harmonizer OFF)", "",
         f"{n} scenes scored by all five reads (both scorers) with no driver RPC error in any of them (a different set from the pool pilot's seven-read set).", "",
         "## Reads", "", "| read | old mean | new mean | zeros old / new | dev ADE |", "|:--|--:|--:|--:|--:|"]
    for t, r in res["reads"].items():
        m = r["old"]
        L.append(f"| {t} | {m['mean']:.3f} | {r['new_mean']:.3f} | {m['zeros']} / {r['new_zeros']} | " + (f"{r['dev_ade']:.4f} |" if r["dev_ade"] is not None else "- |"))
    L += ["", "## Paired differences (95 % bootstrap CI, scenes as units)", "", "| pair | new scorer | old scorer |", "|:--|--:|--:|"]
    for k, p in res["pairs"].items():
        L.append(f"| {k.replace(':', ' - ')} | {f3(p['new'])} | {f3(p['old'])} |")
    L += ["", "## Two seeds", "", "| scorer | P1 2-seed mean | NTV0 2-seed mean | P1 - NTV0 | P1 - RB0-F-s0 |", "|:--|--:|--:|--:|--:|"]
    for sc, q in res["seeds2"].items():
        L.append(f"| {sc} | {q['P1']:.3f} | {q['NTV0']:.3f} | {f3(q['diff'])} | {f3(q['vs_RB0F'])} |")
    L += ["", "## Gap share g = (arm - RB0-F) / (NTV0-s0 - RB0-F)", "", "| scorer | RB0-F | NTV0-s0 | s1 | s0 | 2-seed | 2-seed vs NTV0 2-seed |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for sc, q in res["g"].items():
        L.append(f"| {sc} | {q['rb0f']:.3f} | {q['ntv0_s0']:.3f} | {q['s1']:.0%} | {q['s0']:.0%} | {q['two_seed']:.0%} | {q['two_seed_vs_ntv0_2seed']:.0%} |")
    cls = PF.ZCLS + PF.PCLS
    L += ["", "## Zero / partial classes (old scorer, counts on the common scenes)", "", "| read | zeros | " + " | ".join(cls) + " |", "|:--|--:|" + "--:|" * len(cls)]
    for t in LABS:
        m = res["reads"][t]["old"]
        L.append(f"| {t} | {m['zeros']} | " + " | ".join(str(m.get(f'z_{c}', m.get(f'p_{c}', '-'))) for c in cls) + " |")
    c = C.cost(o)
    L += ["", "## Cost", "", f"GPU jobs {c['gpu_job_h']:.1f} job-hours (training {c['train_h']:.2f}, reads {c['read_h']:.2f}, smoke {c['gate_h']:.2f}); wall {c['wall_h']:.1f} h."]
    res["cost"] = c
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", res)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_verdict(a)


if __name__ == "__main__":
    main()
