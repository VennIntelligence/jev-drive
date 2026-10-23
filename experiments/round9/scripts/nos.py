#!/usr/bin/env python3
"""ROUND9 NTV0-NOS checks and report (plans/2026-10-23-ntv0-nos.md; lane scripts/nos.sh). Box, repo root, repo venv.
  nos.py smoke --tag T --out J        60-step smoke: DONE, every scalar finite, batch 115 normal / 13 yaw-rate / 0 on-policy, no S / pool term nonzero
  nos.py ident --tag T --ref R --out J  (informational) T = R's argv 60 steps on HEAD: loss scalars at steps 25 / 50 and on_hash_60 vs R's stored run
  nos.py verdict --dir O --out P      the registered reading line (old scorer, vs NTV0-L18-s0), class table, dev ADE, scorer_v2 new pairs -> P.{json,md}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

DD = data_dir()
RUNS = DD / "runs/op_parity/runs"
LINE_HI, LINE_LO = 0.72, 0.67
OLD_NTV0, OLD_RB0F = 0.744, 0.650                                      # registered old-scorer reference values
SH_HI, SH_LO = (LINE_HI - OLD_RB0F) / (OLD_NTV0 - OLD_RB0F), (LINE_LO - OLD_RB0F) / (OLD_NTV0 - OLD_RB0F)   # amendment 1: gap shares
S_TERM = re.compile(r"(^|/)on/|_ho(_pos)?$|^loss/act/pool_")


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def run_dir(tag):
    ds = sorted((DD / "runs/route_mem").glob(f"train-{tag}/*/"))
    assert ds, f"no run dir of {tag}"
    return ds[-1]


def cmd_smoke(a):
    d = run_dir(a.tag)
    log = (d / "log.txt").read_text()
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy \(mode (\w+)", log)
    ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]
    sc = [e for e in ev if e.get("kind") == "scalar" and not e["tag"].startswith("throughput/")]
    vals = np.array([e["value"] for e in sc], float)
    s_terms = {}
    for e in sc:
        if S_TERM.search(e["tag"]):
            s_terms[e["tag"]] = max(s_terms.get(e["tag"], 0.0), abs(float(e["value"])))
    steps = [int(m) for m in re.findall(r"step (\d+): ", log)]
    ck = dict(done=(d / "DONE").exists(), scalars=int(len(vals)), finite=bool(len(vals) and np.isfinite(vals).all()),
              batch=list(b.groups()) if b else None, batch_want=["115", "13", "0", "mixed"], last_logged_step=max(steps) if steps else None,
              s_terms_max_abs=s_terms, nan_in_log="nan" in log.lower().replace("nonfinite", ""))
    ok = ck["done"] and ck["finite"] and ck["batch"] == ck["batch_want"] and ck["last_logged_step"] == 60 and all(v == 0 for v in s_terms.values())
    rec = dict(tag=a.tag, run=str(d), checks=ck, pass_=bool(ok))
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def scalars(d, steps):
    ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]
    return {(e["tag"], e["step"]): float(e["value"]) for e in ev if e.get("kind") == "scalar" and e["step"] in steps and e["tag"].startswith("loss/")}


def cmd_ident(a):
    d, r = run_dir(a.tag), run_dir(a.ref)
    x, y = scalars(d, (25, 50)), scalars(r, (25, 50))
    keys = sorted(set(x) & set(y))
    diff = {f"{k[0]}@{k[1]}": abs(x[k] - y[k]) for k in keys}
    h = json.loads((d / "DONE").read_text()).get("on_hash_60"), json.loads((r / "DONE").read_text()).get("on_hash_60")
    rec = dict(tag=a.tag, ref=a.ref, scalars_compared=len(keys), only_new=sorted(f"{k[0]}@{k[1]}" for k in set(x) - set(y)),
               only_ref=sorted(f"{k[0]}@{k[1]}" for k in set(y) - set(x)), identical=int(sum(v == 0 for v in diff.values())),
               max_abs_diff=max(diff.values()) if diff else None, worst=sorted(diff.items(), key=lambda kv: -kv[1])[:8],
               on_hash_60=dict(new=h[0], ref=h[1], equal=h[0] == h[1]))
    rec["bit_identical"] = bool(keys and rec["identical"] == len(keys) and rec["on_hash_60"]["equal"])
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))


def dev_ade(tag):
    f = RUNS / tag / "route_mem.json"
    return json.loads(f.read_text()).get("dev_ade") if f.exists() else None


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"]) if d else "-"


def sv2_pairs(o):
    """scorer_v2 new: the arm's rescored scenes vs the stored new-scorer per-scene table (decision 453); RPC scenes of the arm / ref dropped."""
    import pandas as pd
    f = o / "sv2/new/rollouts.csv"
    if not f.exists():
        return None
    N = pd.read_csv(f).drop_duplicates("scene", keep="last").set_index("scene")
    arm = N.score
    P = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    rpc = set()
    for fa in ("NTV0NS-L18-s0-281ag", "NTV0-L18-s0-281ag", "RB0-F-s0-281ag"):
        x = pd.read_csv(DD / "runs/pai_fix/facts" / f"{fa}.csv")
        rpc |= set(x.scene[x.rpc.fillna(False).astype(bool)])
    out = dict(arm_mean_all=float(arm.mean()), arm_scenes=int(arm.notna().sum()), pairs={})
    for lab, partial in (("NTV0-L18-s0-ag", False), ("RB0-F-s0-ag", False), ("NTV0-L18-s1-ag", True)):
        ref = P[lab]
        sc = sorted(set(arm.dropna().index) & set(ref.dropna().index) - rpc)
        u, v = arm.reindex(sc).to_numpy(float), ref.reindex(sc).to_numpy(float)
        out["pairs"][lab] = dict(scenes=len(sc), arm=float(u.mean()), ref=float(v.mean()), diff=stats.paired(u, v), zeros_arm=int((u == 0).sum()),
                                 zeros_ref=int((v == 0).sum()), partial_ref=partial, sc=sc)
    c = sorted(set(out["pairs"]["NTV0-L18-s0-ag"]["sc"]) & set(out["pairs"]["RB0-F-s0-ag"]["sc"]))   # amendment 1: lines as gap shares
    u, n0, r0 = arm.reindex(c).to_numpy(float), P["NTV0-L18-s0-ag"].reindex(c).to_numpy(float), P["RB0-F-s0-ag"].reindex(c).to_numpy(float)
    hi, lo = r0.mean() + SH_HI * (n0.mean() - r0.mean()), r0.mean() + SH_LO * (n0.mean() - r0.mean())
    m = float(u.mean())
    out["lines"] = dict(scenes=len(c), arm=m, ntv0=float(n0.mean()), rb0f=float(r0.mean()), share_hi=SH_HI, share_lo=SH_LO, line_hi=float(hi),
                        line_lo=float(lo), arm_gap_share=float((m - r0.mean()) / (n0.mean() - r0.mean())),
                        line=("the S rows are not where NTV0's gain comes from" if m >= hi else "the S rows carry the gain" if m <= lo
                              else "undecided (propose seed 1, not run)"))
    for p in out["pairs"].values():
        p.pop("sc")
    return out


def cmd_verdict(a):
    o = Path(a.dir)
    R = json.loads((o / "results/pai281.json").read_text())
    R1 = json.loads((o / "results/pai281_s1.json").read_text())
    ns = R["reads"]["NTV0NS-ag"]
    m = ns["mean"]
    line = ("the S rows are not where NTV0's gain comes from" if m >= LINE_HI else "the S rows carry the gain" if m <= LINE_LO
            else "undecided (propose seed 1, not run)")
    pr = R["pairs"]["NTV0NS-ag:NTV0s0-ag"]
    cls = ["zeros", "z_offroad", "z_zero-other", "z_side", "z_route-turn", "z_overshoot", "z_AF-front", "p_rear-hit", "p_slow", "launch"]
    reads = {"NTV0NS-L18-s0": ns, "NTV0-L18-s0": R["reads"]["NTV0s0-ag"], "NTV0-L18-s1": R1["reads"]["NTV0s1-ag"], "RB0-F-s0": R["reads"]["RB0F-ag"]}
    ade = {k: dev_ade(k) for k in reads}
    sv = sv2_pairs(o)
    L = ["# NTV0-NOS verdict (PAI 281, ACC + GAP, Harmonizer OFF)", "",
         f"Old scorer (the read's own, as registered), {R['scenes']} scenes common to NTV0NS / NTV0-s0 / RB0-F-s0 without RPC error.", "",
         f"- NTV0NS-L18-s0 mean **{m:.3f}**; registered line: >= {LINE_HI} / <= {LINE_LO} -> **{line}**",
         f"- NTV0NS - NTV0-L18-s0: {f3(pr['mean'])}", f"- NTV0NS - RB0-F-s0: {f3(R['pairs']['NTV0NS-ag:RB0F-ag']['mean'])}",
         f"- NTV0-L18-s0 - RB0-F-s0: {f3(R['pairs']['NTV0s0-ag:RB0F-ag']['mean'])}",
         f"- NTV0NS - NTV0-L18-s1 ({R1['scenes']} scenes): {f3(R1['pairs']['NTV0NS-ag:NTV0s1-ag']['mean'])}; "
         f"s1 - s0: {f3(R1['pairs']['NTV0s1-ag:NTV0s0-ag']['mean'])}", "",
         "| read | mean | " + " | ".join(cls) + " | navtrain dev ADE |", "|:--|--:|" + "--:|" * (len(cls) + 1)]
    for k, r in reads.items():
        L.append(f"| {k} | {r['mean']:.3f} | " + " | ".join(str(r[c]) for c in cls) + (f" | {ade[k]:.4f} |" if ade[k] is not None else " | - |"))
    L += ["", "NTV0-L18-s1's row is on its own common set with NTV0NS / NTV0-s0 (pai281_s1.md).", ""]
    if sv:
        L += ["## New scorer (scorer_v2, decision 453)", "", f"NTV0NS rescored: mean {sv['arm_mean_all']:.3f} over {sv['arm_scenes']} scenes.", "",
              "| ref | scenes | NTV0NS | ref | NTV0NS - ref | zeros NTV0NS / ref |", "|:--|--:|--:|--:|--:|--:|"]
        for lab, p in sv["pairs"].items():
            L.append(f"| {lab}{' (partial: zero ASLs mostly deleted)' if p['partial_ref'] else ''} | {p['scenes']} | {p['arm']:.3f} | {p['ref']:.3f} | "
                     f"{f3(p['diff'])} | {p['zeros_arm']} / {p['zeros_ref']} |")
        q = sv["lines"]
        same = q["line"] == line
        L += ["", f"Amendment 1: lines as shares of the RB0-F -> NTV0-s0 gap ({q['share_hi']:.0%} / {q['share_lo']:.0%} from the old-scorer values "
              f"{OLD_RB0F} / {OLD_NTV0}), on {q['scenes']} common scenes: RB0-F {q['rb0f']:.3f}, NTV0-s0 {q['ntv0']:.3f} -> lines {q['line_hi']:.3f} / "
              f"{q['line_lo']:.3f}; NTV0NS {q['arm']:.3f} = {q['arm_gap_share']:.0%} of the gap -> **{q['line']}**.",
              f"Old-scorer gap share for comparison: {(m - reads['RB0-F-s0']['mean']) / (reads['NTV0-L18-s0']['mean'] - reads['RB0-F-s0']['mean']):.0%}. "
              + ("Both scorers land on the same line." if same else "**The two scorers land on different lines.**")]
    else:
        L += ["scorer_v2 new: not available (see sv2.log)."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    dump(f"{a.out}.json", dict(mean=m, line=line, lines=[LINE_HI, LINE_LO], scenes=R["scenes"], pair_ntv0s0=pr["mean"], reads=reads, dev_ade=ade, sv2=sv))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="cmd", required=True)
    p = s.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--out", required=True)
    p = s.add_parser("ident")
    p.add_argument("--tag", required=True), p.add_argument("--ref", required=True), p.add_argument("--out", required=True)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(smoke=cmd_smoke, ident=cmd_ident, verdict=cmd_verdict)[a.cmd](a)


if __name__ == "__main__":
    main()
