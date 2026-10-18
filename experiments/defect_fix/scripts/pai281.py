#!/usr/bin/env python3
"""Paired PAI 281 read-out of one pai_box.sh read against another (box, repo root, .venv python; read-only on stored runs).

  pai281.py --j <read root> --ref <read root> [--jl JTN] [--rl RSG] [--out FILE.md]

A read root holds runs/*/sim/aggregate/results-summary.json (pai_box.sh layout). Scenes: those both reads scored without a driver RPC error
(decision 327's rule: the RPC-error rollout is scored 0 by the scorer but is infrastructure, not behaviour). Paired difference of the scene score:
jevdrive.stats.paired, 10 000 bootstrap draws, seed 0, clusters = the log ('_'-split prefix of the clip id, as wt.xpair; every PAI scene is its own).
Zero types: nx_regress.failure_type (AF-front / AF-side / AF-rear, offroad, corridor, route-fail, zero-other); partial scores 0 < s < 0.98 are hit
(split: rear-hit when collision_rear, else other contact), no-launch (log > 5 m, policy < 2 m), creep (barely / crawl, < 30 % of the log), slow.
Strata (PAI has no per-scene window table, so they come from the rollout itself): static = the log's distance < 1 m; launch = the speed at the
simulator handoff (controller csv, rel. time 1.6 s, the force-GT part is the human's) < 1 m/s and the log moves; moving = the rest. Braking and
turn strata need the human's trace, which the stored results do not hold.
New zero = zero for J, scored for the reference; fixed zero = the reverse.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/wait_cap/scripts")]
from jevdrive import stats  # noqa: E402
from nx_regress import failure_type  # noqa: E402

PARTIAL = ["hit", "rear-hit", "no-launch", "creep", "slow"]


def load(root: str) -> tuple[dict, dict]:
    """-> {scene: rollout row + path of its controller csv}, {scene: reason} of RPC-error rollouts."""
    R, bad = {}, {}
    for sim in sorted(glob.glob(f"{root}/runs/*/sim")):
        f = Path(sim) / "aggregate/results-summary.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["rollouts"]:
            s = r["clipgt_id"]
            if "RpcError" in str(r.get("failure_reason") or ""):
                bad[s] = str(r["failure_reason"]).splitlines()[0][:60]
            r["_csv"] = str(Path(sim) / "controller" / f"alpasim_controller_{r['rollout_id']}.csv")
            R.setdefault(s, r)
    return R, bad


def ptype(r: dict) -> str:
    t = failure_type(r)
    if t == "hit" and (r.get("metrics") or {}).get("collision_rear"):
        return "rear-hit"
    return {"barely": "creep", "crawl": "creep"}.get(t, t)


def zclass(t: str) -> str:
    return t.split("+")[0] if t.startswith("AF-") else "offroad" if "offroad" in t else t


def handoff_speed(r: dict) -> float:
    try:
        rows = list(csv.DictReader(open(r["_csv"])))
        t = np.array([float(x["timestamp_us"]) for x in rows]) / 1e6
        i = int(np.searchsorted(t - t.min() + 0.1, 1.6))
        return float(rows[min(i, len(rows) - 1)]["vx"])
    except OSError:
        return float("nan")


def stratum(r: dict) -> str:
    gt = float((r.get("metrics") or {}).get("gt_dist_traveled_m") or 0.0)
    if gt < 1.0:
        return "static"
    return "launch" if handoff_speed(r) < 1.0 else "moving"


def summ(R: dict, S: list) -> dict:
    sc = np.array([float(R[s].get("score") or 0.0) for s in S])
    ty = [ptype(R[s]) for s in S]
    m = [R[s].get("metrics") or {} for s in S]
    gt = np.array([float(x.get("gt_dist_traveled_m") or 0.0) for x in m])
    d = np.array([float(x.get("dist_traveled_m") or 0.0) for x in m])
    return dict(n=len(S), mean=sc.mean(), zeros=int((sc == 0).sum()), zc=Counter(zclass(t) for t, x in zip(ty, sc) if x == 0),
                pc=Counter(t for t, x in zip(ty, sc) if 0 < x < 0.98), prog=float(np.nanmean([x.get("progress_rel", np.nan) for x in m])),
                dist_ratio=float(d[gt > 5].sum() / gt[gt > 5].sum()), scored1=int((sc >= 0.98).sum()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--j", required=True), a.add_argument("--ref", required=True)
    a.add_argument("--jl", default="J"), a.add_argument("--rl", default="REF"), a.add_argument("--out", default="")
    a = a.parse_args()
    J, bj = load(a.j)
    C, bc = load(a.ref)
    S = sorted((set(J) & set(C)) - set(bj) - set(bc))
    sj, sc = summ(J, S), summ(C, S)
    sa = np.array([float(J[s].get("score") or 0.0) for s in S])
    sb = np.array([float(C[s].get("score") or 0.0) for s in S])
    grp = np.array(["_".join(s.split("_")[:2]) for s in S])
    p = stats.paired(sa, sb, groups=grp)
    z = stats.paired((sa == 0).astype(float), (sb == 0).astype(float), groups=grp)
    L = [f"# PAI 281: {a.jl} against {a.rl}, paired", "",
         f"J = `{a.j}`, reference = `{a.ref}`. Scenes scored by both without a driver RPC error: {len(S)} of {len(set(J) | set(C))} "
         f"(excluded: J {sorted(x[-8:] for x in bj)} {sorted(set(bj.values()))}, reference {sorted(x[-8:] for x in bc)}).", "",
         "| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |", "|---|--:|--:|--:|--:|--:|--:|"]
    for lab, x in ((a.jl, sj), (a.rl, sc)):
        L.append(f"| {lab} | {x['n']} | {x['mean']:.4f} | {x['zeros']} | {x['scored1']} | {x['prog']:.3f} | {x['dist_ratio']:.3f} |")
    L += ["", f"Paired mean ({a.jl} - {a.rl}): **{stats.fmt(p, '+.4f')}** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). "
          f"Zero-rate difference: {stats.fmt(z, '+.4f')} ({sj['zeros'] - sc['zeros']:+d} zeros).", "",
          "## Zero types", "", f"| type | {a.jl} | {a.rl} |", "|---|--:|--:|"]
    for k in sorted(set(sj["zc"]) | set(sc["zc"])):
        L.append(f"| {k} | {sj['zc'][k]} | {sc['zc'][k]} |")
    L += ["", "## Partial scores (0 < s < 0.98)", "", f"| type | {a.jl} | {a.rl} |", "|---|--:|--:|"]
    for k in PARTIAL:
        L.append(f"| {k} | {sj['pc'][k]} | {sc['pc'][k]} |")
    L += ["", "## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)", "",
          f"| stratum | scenes | {a.jl} mean | {a.rl} mean | diff | contribution | {a.jl} zeros | {a.rl} zeros | new | fixed |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    st = np.array([stratum(C[s]) for s in S])
    for k in ("static", "launch", "moving"):
        i = st == k
        if not i.any():
            continue
        new = int(((sa[i] == 0) & (sb[i] > 0)).sum())
        fix = int(((sa[i] > 0) & (sb[i] == 0)).sum())
        L.append(f"| {k} | {i.sum()} | {sa[i].mean():.4f} | {sb[i].mean():.4f} | {(sa[i] - sb[i]).mean():+.4f} | {(sa[i] - sb[i]).sum() / len(S):+.4f} | "
                 f"{int((sa[i] == 0).sum())} | {int((sb[i] == 0).sum())} | {new} | {fix} |")
    for title, cond in (("New zeros (zero for " + a.jl + ", scored for " + a.rl + ")", lambda x, y: x == 0 < y),
                        ("Fixed zeros (zero for " + a.rl + ", scored for " + a.jl + ")", lambda x, y: y == 0 < x)):
        idx = [i for i in range(len(S)) if cond(sa[i], sb[i])]
        L += ["", f"## {title}: {len(idx)}", "", f"| scene | stratum | {a.jl} | {a.rl} |", "|---|---|---|---|"]
        for i in idx:
            L.append(f"| {S[i][-8:]} | {st[i]} | {sa[i]:.3f} {ptype(J[S[i]])} | {sb[i]:.3f} {ptype(C[S[i]])} |")
    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        Path(a.out).write_text(out)


if __name__ == "__main__":
    main()
