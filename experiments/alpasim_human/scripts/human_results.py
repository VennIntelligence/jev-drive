#!/usr/bin/env python3
"""Per-scene table of a human AlpaSim run (human_stack.sh): scene score, pass / zero reason, progress, distance, sim s, wall per sim s,
stale calls, start wait; mean over scenes; the models' scores on the same scenes next to it where a reference exists.

    python3 experiments/alpasim_human/scripts/human_results.py <run dir> [--ref label=<results-summary.json | run dir | dir of *.json>]

Default references: PAI (clipgt- scenes) = P2H10-F-s0 in pai_run.sh (/data/runs/alpasim/pai1/runs/a10_c4, stage-1 scenes) and the
organisers' alpamayo1 (mean of its 3 rollouts, /data/runs/alpasim/pai1/ref); nuPlan = native runs in /data/runs/alpasim/native_ref
(SH30-F-s0 sh30_full48_c8, AP2-AB-s0 ap2_full48). Writes <run>/results.md and results.json. Plain python3.
"""
import argparse
import json
from pathlib import Path

FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")
R = Path("/data/runs/alpasim")
REFS = {"pai": {"P2H10 (model)": R / "pai1/runs/a10_c4", "alpamayo1 (organisers)": R / "pai1/ref/alpamayo1.json"},
        "nuplan": {"SH30-F-s0 (model)": R / "native_ref/sh30_full48_c8", "AP2-AB-s0 (model)": R / "native_ref/ap2_full48"}}


def summary(p: Path):
    for c in (p, p / "aggregate/results-summary.json", p / "sim/aggregate/results-summary.json"):
        if c.is_file():
            return json.loads(c.read_text())
    return None


def scores(p: Path) -> dict:
    """scene -> mean score over its rollouts (a dir of *.json = several runs pooled)."""
    out = {}
    docs = [summary(p)] if summary(p) else [json.loads(f.read_text()) for f in sorted(p.glob("*.json"))] if p.is_dir() else []
    for d in docs:
        for r in d["rollouts"]:
            out.setdefault(r["clipgt_id"], []).append(r["score"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def why(r) -> str:
    m = r.get("score_metrics") or {}
    z = [f for f in FAIL if m.get(f)]
    if r.get("failure_reason") and not z:
        return "FAILED: " + str(r["failure_reason"])[:50]
    return ", ".join(z) if z else "pass" if r["score"] >= 1 else "slow"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", nargs="*", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    rolls = summary(run)["rollouts"]
    drv = [json.loads(x) for x in (run / "driver/drive.jsonl").read_text().splitlines()] if (run / "driver/drive.jsonl").exists() else []
    close = {r["scene"]: r for r in drv if r["kind"] == "close"}
    track = "pai" if rolls and rolls[0]["clipgt_id"].startswith("clipgt-") else "nuplan"
    refs = dict(x.split("=", 1) for x in a.ref) if a.ref is not None else REFS[track]
    ref = {k: scores(Path(v)) for k, v in refs.items()}
    rows = []
    for r in rolls:
        m, c = r.get("metrics") or {}, close.get(r["clipgt_id"], {})
        rows.append({"scene": r["clipgt_id"], "score": round(r["score"], 4), "why": why(r),
                     "progress": round((r.get("score_metrics") or {}).get("progress_clipped_rel", float("nan")), 3),
                     "driven_m": round(m.get("dist_traveled_m", float("nan")), 1), "gt_m": round(m.get("gt_dist_traveled_m", float("nan")), 1),
                     "sim_s": c.get("sim_s"), "wall_per_sim": c.get("wall_per_sim"), "stale": c.get("stale"), "wait_s": c.get("wait_s"),
                     **{k: (round(v[r["clipgt_id"]], 4) if r["clipgt_id"] in v else None) for k, v in ref.items()}})
    mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 4) if rows and all(r[k] is not None for r in rows) else None  # noqa: E731
    wps = [r["wall_per_sim"] for r in rows if r["wall_per_sim"]]
    out = {"run": str(run), "track": track, "n": len(rows), "mean_score": mean("score"), "refs": {k: mean(k) for k in ref},
           "ref_paths": {k: str(v) for k, v in refs.items()}, "wall_per_sim_mean": round(sum(wps) / len(wps), 3) if wps else None, "scenes": rows}
    (run / "results.json").write_text(json.dumps(out, indent=1))
    L = ["| scene | human | why | progress | driven / GT m | sim s | wall per sim s | stale | wait s | " + " | ".join(ref) + " |",
         "|---|--:|---|--:|--:|--:|--:|--:|--:|" + "--:|" * len(ref)]
    for r in rows:
        L.append(f"| {r['scene'][-16:]} | {r['score']:.3f} | {r['why']} | {r['progress']} | {r['driven_m']} / {r['gt_m']} | {r['sim_s']} | "
                 f"{r['wall_per_sim']} | {r['stale']} | {r['wait_s']} | " + " | ".join("-" if r[k] is None else f"{r[k]:.3f}" for k in ref) + " |")
    L += ["", f"{track}, {len(rows)} scenes: human mean {out['mean_score']}; " + "; ".join(f"{k} {v}" for k, v in out["refs"].items())
          + f"; wall s per sim s (after the start) {out['wall_per_sim_mean']}"]
    (run / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
