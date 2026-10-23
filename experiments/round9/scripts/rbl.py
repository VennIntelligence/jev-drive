#!/usr/bin/env python3
"""ROUND9 RB0-LONG gates and report (plans/2026-10-23-rb0-long.md; lane scripts/rbl.sh). Box, repo root, repo venv.
  rbl.py ident --tag T --ref R --upto N --out J -- CMD..   run CMD (R's argv, tag T), stop it once a scalar past step N + 25 exists, then every
                                                           scalar at steps <= N (throughput/* aside) must equal R's stored run bit for bit
  rbl.py smoke --tag T --snap S --out J                    60-step smoke: DONE, every scalar finite, last logged step 60, snapshot S written
  rbl.py verdict --dir O --out P                           the registered reading line, paired CIs, class table, dev ADE -> P.{json,md}
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

DD = data_dir()
RUNS = DD / "runs/op_parity/runs"
LINE_HI, LINE_LO = 0.72, 0.67
OLD_RBF, OLD_NTV0 = 0.650, 0.744                 # the registered old-scorer anchors
SHARE_HI, SHARE_LO = [(x - OLD_RBF) / (OLD_NTV0 - OLD_RBF) for x in (LINE_HI, LINE_LO)]   # the lines as shares of the RB0-F -> NTV0 gap


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def run_dirs(tag):
    return sorted((DD / "runs/route_mem").glob(f"train-{tag}/*/"))


def events(d):
    return [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]


def scalars(d, upto):
    return {(e["tag"], e["step"]): float(e["value"]) for e in events(d)
            if e.get("kind") == "scalar" and e["step"] <= upto and not e["tag"].startswith("throughput/")}


def cmd_ident(a):
    assert not run_dirs(a.tag), f"{a.tag} has a run dir already"
    p = subprocess.Popen(a.cmd, start_new_session=True)
    d, stop = None, a.upto + 25
    while p.poll() is None:                     # wait for the first scalar past step upto + 25: every scalar <= upto is flushed by then
        time.sleep(5)
        d = d or (run_dirs(a.tag) or [None])[-1]
        if d and (d / "events.jsonl").exists() and any(e.get("kind") == "scalar" and e["step"] >= stop for e in events(d)):
            break
    rc = p.poll()
    if rc is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL), p.wait()
    assert d, "no run dir"
    x, y = scalars(d, a.upto), scalars(run_dirs(a.ref)[-1], a.upto)
    keys = sorted(set(x) & set(y))
    diff = {f"{k[0]}@{k[1]}": abs(x[k] - y[k]) for k in keys}
    rec = dict(tag=a.tag, ref=a.ref, ref_run=str(run_dirs(a.ref)[-1]), run=str(d), trainer_rc_before_stop=rc, upto=a.upto,
               steps=sorted({k[1] for k in keys}), scalars_compared=len(keys), only_new=sorted(f"{k[0]}@{k[1]}" for k in set(x) - set(y)),
               only_ref=sorted(f"{k[0]}@{k[1]}" for k in set(y) - set(x)), identical=int(sum(v == 0 for v in diff.values())),
               max_abs_diff=max(diff.values()) if diff else None, worst=sorted(diff.items(), key=lambda kv: -kv[1])[:8])
    rec["bit_identical"] = bool(keys and rec["identical"] == len(keys) and not rec["only_new"] and not rec["only_ref"]
                                and {25, 50, 75} <= set(rec["steps"]))
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["bit_identical"] else 2)


def cmd_smoke(a):
    d = run_dirs(a.tag)[-1]
    log = (d / "log.txt").read_text()
    vals = np.array([e["value"] for e in events(d) if e.get("kind") == "scalar"], float)
    steps = [int(m) for m in re.findall(r"step (\d+): ", log)]
    ck = dict(done=(d / "DONE").exists(), scalars=int(len(vals)), finite=bool(len(vals) and np.isfinite(vals).all()),
              last_logged_step=max(steps) if steps else None, snapshot=(RUNS / a.snap / "ckpt-final.pt").exists(),
              final=(RUNS / a.tag / "ckpt-final.pt").exists())
    ok = ck["done"] and ck["finite"] and ck["last_logged_step"] == 60 and ck["snapshot"] and ck["final"]
    rec = dict(tag=a.tag, run=str(d), checks=ck, pass_=bool(ok))
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def dev_ade(tag, step=None):
    if step is None:
        f = RUNS / tag / "route_mem.json"
        return json.loads(f.read_text()).get("dev_ade") if f.exists() else None
    v = [e["value"] for e in events(run_dirs(tag)[-1]) if e.get("kind") == "scalar" and e["tag"] == "dev/ade" and e["step"] == step]
    return float(v[-1]) if v else None


def sv2(o):
    """New scorer (scorer_v2, decision 453): the arm rescored vs the stored new-scorer RB0-F-s0 / NTV0-L18-s0; RPC scenes of any of the three dropped."""
    import pandas as pd
    f = o / "sv2/new/rollouts.csv"
    if not f.exists():
        return None
    arm = pd.read_csv(f).drop_duplicates("scene", keep="last").set_index("scene").score
    P = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    rpc = set()
    for fa in ("RB0L-F-s0-281ag", "RB0-F-s0-281ag", "NTV0-L18-s0-281ag"):
        x = pd.read_csv(DD / "runs/pai_fix/facts" / f"{fa}.csv")
        rpc |= set(x.scene[x.rpc.fillna(False).astype(bool)])
    rb, nt = P["RB0-F-s0-ag"], P["NTV0-L18-s0-ag"]
    sc = sorted(set(arm.dropna().index) & set(rb.dropna().index) & set(nt.dropna().index) - rpc)
    u, b, n = (x.reindex(sc).to_numpy(float) for x in (arm, rb, nt))
    share = (u.mean() - b.mean()) / (n.mean() - b.mean())
    line = ("a longer offline single run nearly matches NTV0" if share >= SHARE_HI else "more offline steps do not give NTV0's gain" if share <= SHARE_LO
            else "undecided")
    return dict(scenes=len(sc), arm_mean_all=float(arm.mean()), arm=float(u.mean()), rbf=float(b.mean()), ntv0=float(n.mean()),
                zeros=dict(arm=int((u == 0).sum()), rbf=int((b == 0).sum()), ntv0=int((n == 0).sum())),
                d_rbf=stats.paired(u, b), d_ntv0=stats.paired(u, n), share=float(share), line=line)


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"]) if d else "-"


def cmd_verdict(a):
    o = Path(a.dir)
    R = json.loads((o / "results/pai281.json").read_text())
    fn = o / "results/pai281_ns.json"
    RN = json.loads(fn.read_text()) if fn.exists() else None
    m = R["reads"]["RB0L-ag"]["mean"]
    line = ("a longer offline single run nearly matches NTV0" if m >= LINE_HI else "more offline steps do not give NTV0's gain" if m <= LINE_LO
            else "undecided (reported; no more seeds run)")
    P = R["pairs"]
    cls = ["zeros", "z_offroad", "z_zero-other", "z_side", "z_route-turn", "z_overshoot", "z_AF-front", "p_rear-hit", "p_slow", "launch"]
    reads = {"RB0L-F-s0": R["reads"]["RB0L-ag"], "RB0-F-s0": R["reads"]["RB0F-ag"], "NTV0-L18-s0": R["reads"]["NTV0s0-ag"]}
    ade = {"RB0L-F-s0 @ 10000": dev_ade("RB0L-F-s0", 10000), "RB0L-F-s0 @ 19500": dev_ade("RB0L-F-s0", 19500),
           "RB0-F-s0 @ 10000": dev_ade("RB0-F-s0"), "NTV0-L18-s0 (10 000 + 9 500)": dev_ade("NTV0-L18-s0")}
    L = ["# RB0-LONG verdict (PAI 281, ACC + GAP, Harmonizer OFF)", "",
         f"Old scorer (the read's own, as registered), {R['scenes']} scenes common to RB0L / RB0-F-s0 / NTV0-L18-s0 without RPC error.", "",
         f"- RB0L-F-s0 mean **{m:.3f}**; registered line: >= {LINE_HI} / <= {LINE_LO} -> **{line}**",
         f"- RB0L - RB0-F-s0: {f3(P['RB0L-ag:RB0F-ag']['mean'])}", f"- RB0L - NTV0-L18-s0: {f3(P['RB0L-ag:NTV0s0-ag']['mean'])}",
         f"- NTV0-L18-s0 - RB0-F-s0: {f3(P['NTV0s0-ag:RB0F-ag']['mean'])}"]
    if RN:
        L.append(f"- RB0L - NTV0NS-L18-s0 ({RN['scenes']} scenes common to the four reads): {f3(RN['pairs']['RB0L-ag:NTV0NS-ag']['mean'])}; "
                 f"NTV0NS mean {RN['reads']['NTV0NS-ag']['mean']:.3f}, RB0L {RN['reads']['RB0L-ag']['mean']:.3f} on that set")
        reads["NTV0NS-L18-s0 (own common set)"] = RN["reads"]["NTV0NS-ag"]
    else:
        L.append("- NTV0NS-L18-s0: no facts at report time (not waited for)")
    L += ["", "| read | mean | " + " | ".join(cls) + " |", "|:--|--:|" + "--:|" * len(cls)]
    for k, r in reads.items():
        L.append(f"| {k} | {r['mean']:.3f} | " + " | ".join(str(r[c]) for c in cls) + " |")
    L += ["", "drift / lane change = zero-other + side; rear-ended / slow = partials (0 < score < 0.98); launch = launched / launch scenes.", "",
          "| navtrain dev ADE | m |", "|:--|--:|"] + [f"| {k} | {v:.4f} |" if v is not None else f"| {k} | - |" for k, v in ade.items()]
    sv = sv2(o)
    if sv:
        L += ["", "## New scorer (scorer_v2, decision 453; reported next to the registered old-scorer line)", "",
              f"{sv['scenes']} scenes common to the three new-scorer reads without RPC error (RB0L rescored mean over all its scenes "
              f"{sv['arm_mean_all']:.3f}). RB0L {sv['arm']:.3f}, RB0-F-s0 {sv['rbf']:.3f}, NTV0-L18-s0 {sv['ntv0']:.3f}; zeros "
              f"{sv['zeros']['arm']} / {sv['zeros']['rbf']} / {sv['zeros']['ntv0']}.", "",
              f"- RB0L - RB0-F-s0: {f3(sv['d_rbf'])}; RB0L - NTV0-L18-s0: {f3(sv['d_ntv0'])}",
              f"- share of the RB0-F -> NTV0 gap: **{sv['share']:.0%}**; the registered lines as shares: >= {SHARE_HI:.0%} / <= {SHARE_LO:.0%} -> "
              f"**{sv['line']}**",
              f"- old and new scorer {'land on the same line' if sv['line'].split(' (')[0] == line.split(' (')[0] else 'land on DIFFERENT lines'}"]
    else:
        L += ["", "New scorer: not available (see sv2.log)."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    dump(f"{a.out}.json", dict(mean=m, line=line, lines=[LINE_HI, LINE_LO], scenes=R["scenes"], pairs={k: P[k]["mean"] for k in P},
                               ns=RN and dict(scenes=RN["scenes"], pair=RN["pairs"]["RB0L-ag:NTV0NS-ag"]["mean"]), reads=reads, dev_ade=ade, sv2=sv))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("ident")
    p.add_argument("--tag", required=True), p.add_argument("--ref", required=True), p.add_argument("--upto", type=int, default=75)
    p.add_argument("--out", required=True), p.add_argument("cmd", nargs=argparse.REMAINDER)
    p = s.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--snap", required=True), p.add_argument("--out", required=True)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.sub == "ident":
        a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        assert a.cmd, "ident needs the trainer command after --"
    dict(ident=cmd_ident, smoke=cmd_smoke, verdict=cmd_verdict)[a.sub](a)

if __name__ == "__main__":
    main()
