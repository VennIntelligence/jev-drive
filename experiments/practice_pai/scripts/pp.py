#!/usr/bin/env python3
"""PRACTICE-PAI (plans/2026-10-19-pai-practice-prereg.md): split registration, the pool's input-path check, draw weights and the closed-loop gate.

  split   --curated <sim_suites_curated.csv>        (Mac, repo root) registers physical_ai_av/{curated-train, curated-val, test281, practice-pai}@v1
                                                    and checks: practice-pai disjoint from test281 and from curated-train's complement rules below
  r1      --pool N --cache DIR --fwd DIR --tag T --control C --out FILE.json
                                                    the checkpoint's plans recomputed on the pool's inputs (rf_recov fwd, train-rec + dev-rec) against
                                                    the rollout's logged `poses_model`, 8 poses (<= 4 s); line as d333 Amendment 1: median row max-diff
                                                    <= 0.02 m and share over max(0.03 m, 4 fp16 ulp) <= 0.10; wrong model share over >= 0.9 (exit 2)
  weights --pool N --arm pilot|control              the arm's --on-weight on the merged pool (pilot: PAI rows `p` get W = n_s / n_p, i.e. half of the
                                                    drawn pool rows; control: W = 0, the draw of the nuPlan part alone bit for bit)
  gate    --pilot R --control R --ref R --chunks c000,.. --out DIR
                                                    the registered gate on the fixed chunks -> DIR/gate.{json,md}, DIR/VERDICT (PASS / FAIL)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "experiments/defect_fix/scripts"), str(R_ / "experiments/wait_cap/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = D / "runs/op_parity/cache"
DS = "physical_ai_av"


def tsv_ids(p):
    return [l.split("\t")[0] for l in Path(p).read_text().splitlines() if l.startswith("clipgt-")]


def cmd_split(a):
    from jevdrive.data import splits
    suite = {}
    for r in csv.DictReader(open(a.curated)):
        suite.setdefault(r["test_suite_id"], []).append(r["scene_id"])
    S = "experiments/onpol_pai/results/split"
    tr = splits.define(DS, "curated-train", suite["nurec_curated_train"], "scene", "alpasim data/scenes/sim_suites_curated.csv, suite "
                       "nurec_curated_train (organisers' curated train split, 26.01 + 26.04 releases)", status="official", used_by=["experiments/practice_pai"])
    va = splits.define(DS, "curated-val", suite["nurec_curated_val"], "scene", "alpasim data/scenes/sim_suites_curated.csv, suite nurec_curated_val",
                       status="official", used_by=["experiments/practice_pai"], notes="= test281 + train160 (experiments/onpol_pai/results/split)")
    te = splits.define(DS, "test281", tsv_ids(f"{S}/test281.tsv"), "scene", f"{S}/test281.tsv (onpol_pai prereg 2026-10-11: the PAI 281 exam)",
                       status="frozen", used_by=["experiments/practice_pai"], notes="never rolled for training rows (research/onpol-scaleup.md 3.1)")
    pr = splits.define(DS, "practice-pai", tsv_ids(f"{S}/train160.tsv"), "scene", f"{S}/train160.tsv: the PAI scenes with a usdz on the GPU box "
                       "($DATA_DIR/datasets/nurec/all-usdzs) outside test281 on 2026-10-19 (no curated-train scene is on disk; the mirror ran at ~3 MB/s)",
                       status="frozen", used_by=["experiments/practice_pai"],
                       notes="PRACTICE-PAI pilot pool scenes (= train160, a part of curated-val); disjoint from test281 and from curated-train")
    splits.check_disjoint(tr, va)
    splits.check_disjoint(pr, te)
    assert set(te) | set(pr) == set(va) and not set(te) & set(pr), "curated-val != test281 + practice-pai"
    for f in ("hard_exam.txt", "control.txt", "exam_turn26.txt"):           # subsets of the 281 exam stay out
        sub = {x.strip() for x in open(R_ / "experiments/alpasim/results/pai_exam" / f) if x.strip().startswith("clipgt-")}
        assert not sub & set(pr), f
    print(json.dumps({s.id: len(s) for s in (tr, va, te, pr)}, indent=1))


def cmd_r1(a):
    tb = np.load(CR / f"on1_{a.pool}" / "tab.npz")
    C = np.load(Path(a.cache) / "tab.npz")
    key = {(s, int(k)): i for i, (s, k) in enumerate(zip(C["scenes"][C["sid"]].tolist(), C["k"].tolist()))}
    ref = C["poses_model"][np.array([key[(n.split(":")[2], int(n.split(":")[3]))] for n in tb["names"].tolist()])][:, :8, :2].astype(np.float64)
    tol = np.maximum(4 * 2.0 ** (np.floor(np.log2(np.maximum(np.abs(ref), 2.0 ** -14))) - 10), 0.03)
    res = {}
    for t in (a.tag, a.control):
        P = np.full(ref.shape, np.nan)
        for rs in ("train-rec", "dev-rec"):
            f = Path(a.fwd) / f"{a.pool}-{rs}" / f"plans_{t}.npz"
            if f.exists():
                z = np.load(f)
                P[z["rows"]] = z["poses"][:, :8, :2]
        d = np.abs(P - ref).reshape(len(ref), -1).max(1)
        res[t] = dict(share_over=float((np.abs(P - ref) > tol).reshape(len(ref), -1).any(1).mean()), median_m=float(np.median(d)),
                      p99_m=float(np.percentile(d, 99)), rows=int(len(d)), missing=int(np.isnan(d).sum()))
    r = dict(pool=a.pool, tag=a.tag, control=a.control, model=res[a.tag], wrong=res[a.control],
             pass_=bool(res[a.tag]["missing"] == 0 and res[a.tag]["median_m"] <= 0.02 and res[a.tag]["share_over"] <= 0.10
                        and res[a.control]["share_over"] >= 0.9))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["pass_"] else 2)


def cmd_weights(a):
    pre = np.load(CR / f"on1_{a.pool}" / "tab.npz")["pool"].astype(str)
    assert set(pre) == {"s", "p"} and (pre[: (pre == "s").sum()] == "s").all(), "merged pool: s rows first, then p"
    w = (pre == "s").sum() / (pre == "p").sum() if a.arm == "pilot" else 0.0
    print(f"pool=p:{w:.6f}")


def cmd_gate(a):
    import pai281 as P
    from jevdrive import stats
    ch = a.chunks.split(",")
    want = set()
    for c in ch:
        want |= set(Path(a.ref, "chunks", f"{c}.txt").read_text().split())
    (J, bj), (K, bk), (C, bc) = (P.load(r) for r in (a.pilot, a.control, a.ref))
    S = sorted((want & set(J) & set(K)) - set(bj) - set(bk))           # the gate: pilot vs control, both scored without a driver RPC error
    sj = np.array([float(J[s].get("score") or 0.0) for s in S])
    sk = np.array([float(K[s].get("score") or 0.0) for s in S])
    grp = np.array(S)                                                  # every PAI clip is its own cluster (no log ids): scene-clustered bootstrap
    p = stats.paired(sj, sk, groups=grp)
    zj, zk = int((sj == 0).sum()), int((sk == 0).sum())
    Sr = sorted(set(S) & set(C) - set(bc))                             # reported: both arms against the stored reference
    sr = {s: float(C[s].get("score") or 0.0) for s in Sr}
    rep = {}
    for lab, X in (("pilot", J), ("control", K)):
        x = np.array([float(X[s].get("score") or 0.0) for s in Sr])
        y = np.array([sr[s] for s in Sr])
        rep[lab] = dict(stats.paired(x, y, groups=np.array(Sr)), zeros=int((x == 0).sum()), ref_zeros=int((y == 0).sum()), n=len(Sr))
    ok_z, ok_m = zj <= zk + 2, p["mean"] >= 0
    v = "PASS" if ok_z and ok_m else "FAIL"
    missing = sorted(want - set(J)) + sorted(want - set(K))
    G = dict(verdict=v, chunks=ch, scenes_in_chunks=len(want), scenes=len(S), missing_or_unscored=len(missing), rpc_excluded=dict(pilot=sorted(bj), control=sorted(bk)),
             pilot=dict(mean=float(sj.mean()), zeros=zj), control=dict(mean=float(sk.mean()), zeros=zk), paired_pilot_minus_control=p,
             line_zeros=f"{zj} <= {zk} + 2: {ok_z}", line_mean=f"{p['mean']:+.4f} >= 0: {ok_m}", vs_reference=rep)
    md = [f"# PRACTICE-PAI gate: {v}", "", f"Chunks {', '.join(ch)} of the PAI 281 exam ({len(want)} scenes); scored by both arms without an RPC error: {len(S)}.", "",
          "| arm | mean | zeros |", "|:--|--:|--:|", f"| pilot (+ PAI pool) | {sj.mean():.4f} | {zj} |", f"| control (no new pool) | {sk.mean():.4f} | {zk} |", "",
          f"- zeros: {G['line_zeros']}", f"- paired mean pilot - control: {stats.fmt(p, '+.4f')} (scene-clustered bootstrap 95 % CI, 10 000 draws, seed 0); "
          f"line mean >= 0: {ok_m}", "", "Reported, not gating: against the stored reference RSG-L18-s0 + ACC + GAP (acc_gap/read/s2) on the same scenes:", ""]
    for lab, r in rep.items():
        md.append(f"- {lab} - reference: {stats.fmt(r, '+.4f')}, zeros {r['zeros']} vs {r['ref_zeros']} (n {r['n']})")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(G, indent=1, default=float))
    (out / "gate.md").write_text("\n".join(md) + "\n")
    (out / "VERDICT").write_text(v + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("split")
    p.add_argument("--curated", required=True)
    p = sp.add_parser("r1")
    for k in ("--pool", "--cache", "--fwd", "--tag", "--control", "--out"):
        p.add_argument(k, required=True)
    p = sp.add_parser("weights")
    p.add_argument("--pool", required=True), p.add_argument("--arm", required=True, choices=("pilot", "control"))
    p = sp.add_parser("gate")
    for k in ("--pilot", "--control", "--ref", "--chunks", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
