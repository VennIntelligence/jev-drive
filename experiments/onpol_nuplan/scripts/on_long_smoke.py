#!/usr/bin/env python3
"""ONPOL-NUPLAN round 2, item A: the smoke's read-out and gate A1 (plans/2026-10-12-onpol2-prereg.md), from on_long.sh's run dirs.

  python on_long_smoke.py --stage $DATA_DIR/runs/onpol_nuplan/long/smoke --list results/long/smoke10_train.txt --out <smoke.json>

Per arm (ship / a80 / b160 / c160new): pool result, rollouts, decisions per scene (driver `close` records), driver errors, runtime wall per
scene, peak host RSS and VRAM (native_summary.json), disk per scene before pruning, the scene's time span and the clipping line of the
runtime's log. Gate A1:
  (1) a80 against ship: no zero flip and every scene score within 0.01;
  (2) the wider-window arm (b160): every rollout ends, 0 driver inference / input errors, and decisions 0..9 are served the same inputs as
      in a80: the largest distance between the two runs' plan points (the 8 poses of drive.jsonl, x and y) is <= 0.03 m on >= 99 % of the
      scene x decision pairs. ship against a80 is printed beside it as the run-to-run reference.
Also `extends`: the number of scenes with more than 10 decisions in any wider-window arm. numpy only.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("ship", "a80", "b160", "c160new")
WIDE = ("b160", "c160new")


def load(D, scenes):
    D = Path(D)
    o = dict(dir=str(D), pool="DONE" if (D / "pool/DONE").exists() else "ERROR" if (D / "pool/ERROR").exists() else "none")
    if (D / "pool/ERROR").exists():
        o["pool_error"] = (D / "pool/ERROR").read_text()[:300].strip()
    ns = D / "native_summary.json"
    if ns.exists():
        n = json.loads(ns.read_text())
        o |= dict(rc=n["rc"], error=n["times"].get("error"), runtime_s=n["times"].get("runtime_s"), total_s=n["times"].get("total_s"),
                  peak_rss_gib={k: round(v["rss_gib"], 2) for k, v in n["peak"].items()}, peak_gpu_mib={k: v["gpu_mib"] for k, v in n["peak"].items()},
                  overrides=[x for x in n["overrides"] if "trajdata" in x])
    if (D / "du_mb_before_prune").exists():
        o["disk_mb"] = int((D / "du_mb_before_prune").read_text())
    for name in ("wizard", "runtime-0", "renderer-0"):                               # the failure lines of an arm that did not run
        f = D / "native-logs" / f"{name}.log"
        if f.exists():
            L = f.read_text(errors="replace").splitlines()
            bad = [x[:300] for x in L if re.search(r"Error|Traceback|not found, skipping|not installed", x)]
            if bad:
                o.setdefault("log_lines", {})[name] = dict(n=len(bad), first=bad[:3], last=bad[-2:])
            clip = [x for x in L if "Clipping n_sim_steps" in x]
            if clip:
                o["clip_lines"], o["clip_example"] = len(clip), clip[0][-220:]
    rs = D / "aggregate/results-summary.json"
    o["scene"] = {}
    if rs.exists():
        for r in json.loads(rs.read_text())["rollouts"]:
            m = r.get("metrics") or {}
            o["scene"][r["clipgt_id"]] = dict(score=r.get("score"), fail=r.get("failure_reason"), gt_dist=m.get("dist_to_gt_trajectory"),
                                              dist_m=m.get("dist_traveled_m"), gt_m=m.get("gt_dist_traveled_m"), duration_frac_20s=m.get("duration_frac_20s"))
    plans, ie, ip = defaultdict(dict), 0, 0
    dj = D / "driver-logs/drive.jsonl"
    if dj.exists():
        for ln in dj.open():
            r = json.loads(ln)
            if r["kind"] == "drive":
                plans[r["scene"]][r["k"]] = np.array(r["poses"], float)
            elif r["kind"] == "close":
                o["scene"].setdefault(r["scene"], {})["decisions"] = r.get("drive", 0)
                ie, ip = ie + r.get("inference_error", 0), ip + r.get("input_error", 0)
    o |= dict(rollouts=sum(v.get("score") is not None for v in o["scene"].values()), inference_errors=ie, input_errors=ip,
              decisions=sorted({v.get("decisions") for v in o["scene"].values() if v.get("decisions") is not None}),
              missing=sorted(set(scenes) - {s for s, v in o["scene"].items() if v.get("score") is not None}))
    return o, plans


def plan_diff(pa, pb, scenes, K=10):
    """Largest xy distance between two runs' plan points, per scene x decision 0..K-1."""
    d = {}
    for s in scenes:
        for k in range(K):
            if k in pa.get(s, {}) and k in pb.get(s, {}):
                d[(s, k)] = float(np.abs(pa[s][k][:, :2] - pb[s][k][:, :2]).max()) if pa[s][k].shape == pb[s][k].shape else float("inf")
    v = np.array(list(d.values()))
    by_k = [float(np.max([x for (s, k), x in d.items() if k == kk] or [np.nan])) for kk in range(K)]
    return dict(pairs=len(v), expected=len(scenes) * K, share_le_0p03=float((v <= 0.03).mean()) if len(v) else None, max_m=float(v.max()) if len(v) else None,
                median_m=float(np.median(v)) if len(v) else None, identical=int((v == 0).sum()), max_by_decision=by_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True), ap.add_argument("--list", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scenes = Path(a.list).read_text().split()
    R, P = {}, {}
    for arm in ARMS:
        if (Path(a.stage) / "runs" / arm).exists():
            R[arm], P[arm] = load(Path(a.stage) / "runs" / arm, scenes)
    G = {}
    if "ship" in R and "a80" in R:
        sa, sb = R["ship"]["scene"], R["a80"]["scene"]
        both = [s for s in scenes if sa.get(s, {}).get("score") is not None and sb.get(s, {}).get("score") is not None]
        flips = [s for s in both if (sa[s]["score"] == 0) != (sb[s]["score"] == 0)]
        dmax = max((abs(sa[s]["score"] - sb[s]["score"]) for s in both), default=None)
        G["shipped_window"] = dict(scenes=len(both), zero_flips=flips, max_abs_score_diff=dmax, plans=plan_diff(P["ship"], P["a80"], scenes),
                                   passed=len(both) == len(scenes) and not flips and dmax is not None and dmax <= 0.01)
    if "a80" in R and "b160" in R:
        b = R["b160"]
        pd_ = plan_diff(P["a80"], P["b160"], scenes)
        G["wider_window"] = dict(rollouts=b["rollouts"], missing=b["missing"], inference_errors=b["inference_errors"], input_errors=b["input_errors"], plans=pd_,
                                 passed=not b["missing"] and not b["inference_errors"] and not b["input_errors"] and pd_["pairs"] == pd_["expected"]
                                 and pd_["share_le_0p03"] >= 0.99)
    ext = {arm: sorted(s for s, v in R[arm]["scene"].items() if (v.get("decisions") or 0) > 10) for arm in WIDE if arm in R}
    out = dict(list=a.list, scenes=scenes, arms=R, gate_A1=G | dict(passed=bool(G) and len(G) == 2 and all(g["passed"] for g in G.values())),
               extends=ext, any_scene_beyond_10_decisions=any(ext.values()))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("| arm | pool | rollouts | decisions per scene | driver errors | runtime s per scene | runtime RSS GiB | renderer VRAM GiB | disk MB per scene | mean score |")
    print("|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|")
    for arm, r in R.items():
        sc = [v["score"] for v in r["scene"].values() if v.get("score") is not None]
        f = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"
        print(f"| {arm} | {r['pool']} | {r['rollouts']} / {len(scenes)} | {r['decisions'] or '-'} | {r['inference_errors'] + r['input_errors']} | "
              f"{f(r.get('runtime_s') and r['runtime_s'] / len(scenes))} | {f(r.get('peak_rss_gib', {}).get('runtime-0'))} | "
              f"{f(r.get('peak_gpu_mib', {}).get('renderer-0') and r['peak_gpu_mib']['renderer-0'] / 1024)} | {f(r.get('disk_mb') and r['disk_mb'] / len(scenes))} | "
              f"{f(np.mean(sc) if sc else None, 4)} |")
    print(json.dumps(dict(gate_A1=out["gate_A1"], extends=ext), indent=1))


if __name__ == "__main__":
    main()
