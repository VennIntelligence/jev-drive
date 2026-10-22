"""Knows-vs-does read-out for any subject (lib/know_do.py; plans/2026-10-23-p1-knowdo-prereg.md).

  .venv/bin/python experiments/wm_surplus/scripts/p1_knowdo.py --feats <tokens, X, B npz> --plans PROBE-wajepa-ctx-off \
      --name wajepa --out <dir>
--plans: a bench model whose navtest pred file holds the subject's plans (or a path to a (tokens, poses) npz).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/wm_surplus/lib"), str(REPO)]
import know_do as KD  # noqa: E402
from jevdrive.bench.models import data_dir, resolve  # noqa: E402
from jevdrive.bench.navsim import pred_file  # noqa: E402
from jevdrive.bench.tables import motion, navtest_strata  # noqa: E402
from jevdrive.bench.sets import navsim_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = np.load(a.feats)
    pf = Path(a.plans) if a.plans.endswith(".npz") else pred_file(resolve(a.plans), "navtest")
    P = np.load(pf)
    plan = dict(zip(P["tokens"].tolist(), P["poses"]))
    G = np.load(data_dir() / "runs/navsim_zs/index/navtest_future.npz")
    gt = dict(zip(G["tokens"].tolist(), G["poses"]))
    S = navtest_strata()
    toks, logs = navsim_tokens("navtest")
    lg = dict(zip(toks, logs))
    T = [t for t in F["tokens"].tolist() if t in plan and t in gt and t in S.index]
    idx = {t: i for i, t in enumerate(F["tokens"].tolist())}
    rows = [idx[t] for t in T]
    v0 = S.loc[T, "v0"].to_numpy(float)
    gm = [motion(np.asarray(gt[t], float), v) for t, v in zip(T, v0)]
    pm = [motion(np.asarray(plan[t], float), v) for t, v in zip(T, v0)]
    keep = np.array([m[1] != "unknown" for m in gm])
    T = [t for t, k in zip(T, keep) if k]
    rows = np.array(rows)[keep]
    gm = [m for m, k in zip(gm, keep) if k]
    pm = [m for m, k in zip(pm, keep) if k]
    r = KD.knows_vs_does(F["X"][rows], F["B"][rows], [m[1] for m in gm], [m[1] for m in pm], [lg[t] for t in T],
                         gt_dyaw=[m[0] for m in gm], gt_lat=[gt[t][-1, 1] for t in T],
                         plan_dyaw=[m[0] for m in pm], plan_lat=[plan[t][-1, 1] for t in T])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = r["table"].sort_values("n", ascending=False)
    tab.to_csv(out / f"knowdo_{a.name}.csv", index=False)
    summ = dict(name=a.name, tokens=len(T), balanced_accuracy=r["balanced_accuracy"], regression_r2=r["regression_r2"],
                verdicts=dict(zip(tab.cls, tab.verdict)))
    (out / f"knowdo_{a.name}.json").write_text(json.dumps(summ, indent=1))
    L = [f"# Knows vs does: {a.name} ({len(T)} navtest tokens, 5 log folds)", "",
         "| GT manoeuvre | n | knows (feat+blind) | feat only | blind | does (plan) | knows - does [95 % CI] | knows - blind [95 % CI] | verdict |",
         "|---|--:|--:|--:|--:|--:|---|---|---|"]
    for x in tab.itertuples():
        L.append(f"| {x.cls} | {x.n} | {x.knows:.3f} | {x.feat_only:.3f} | {x.blind:.3f} | {x.does:.3f} | "
                 f"{x.knows_minus_does:+.3f} [{x.kd_lo:+.3f}, {x.kd_hi:+.3f}] | {x.knows_minus_blind:+.3f} [{x.kb_lo:+.3f}, {x.kb_hi:+.3f}] | {x.verdict} |")
    L += ["", "Balanced accuracy over these classes: " + ", ".join(f"{k} {v:.3f}" for k, v in r["balanced_accuracy"].items()),
          "", "R^2 (out of fold): " + "; ".join(f"{k}: " + ", ".join(f"{n} {v:.3f}" for n, v in d.items()) for k, d in r["regression_r2"].items()),
          "", "Recall per GT class; CIs = log-cluster bootstrap (B 10000, seed 0); verdict lines from plans/2026-10-23-p1-knowdo-prereg.md."]
    (out / f"knowdo_{a.name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
