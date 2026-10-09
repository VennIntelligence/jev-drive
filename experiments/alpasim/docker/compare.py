#!/usr/bin/env python3
"""Scene scores of one AlpaSim run against a reference run of the same scenes (both `aggregate/results-summary.json`), plus the driver's
own latency log. stdlib only. Writes <out>/smoke.json and <out>/compare.md (summary line, then one row per scene that differs).

  compare.py <results-summary.json> [--ref <results-summary.json>] [--drive <drive.jsonl>] [--label NAME] [--meta k=v ...] --out DIR
"""
import argparse
import json
from pathlib import Path


def scores(f) -> dict:
    """scene id -> (score, failure reason or the non-zero zeroing metrics)."""
    out = {}
    for r in json.load(open(f))["rollouts"]:
        m = r.get("score_metrics") or {}
        why = r.get("failure_reason") or ",".join(k for k in ("collision_at_fault", "offroad", "left_corridor_laterally") if m.get(k)) or ""
        out[r["clipgt_id"]] = (float(r["score"]) if r.get("score") is not None else float("nan"), str(why)[:60])
    return out


def stats(s: dict) -> dict:
    v = [x for x, _ in s.values()]
    return {"n": len(v), "mean": round(sum(v) / len(v), 4), "zeros": sum(x == 0 for x in v), "ones": sum(x == 1 for x in v)}


def latency(f) -> dict:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    dr = [r["ms"] for r in rows if r.get("kind") == "drive"]
    if not dr:
        return {}
    q = lambda k, p: round(sorted(d[k] for d in dr)[min(len(dr) - 1, int(p * len(dr)))], 1)  # noqa: E731
    cl = [r for r in rows if r.get("kind") == "close"]
    return {"drives": len(dr), "sessions": len(cl), "inference_errors": sum(r.get("inference_error", 0) for r in cl),
            "input_errors": sum(r.get("input_error", 0) for r in cl),
            **{f"{k}_ms": {"p50": q(k, .5), "p90": q(k, .9), "p99": q(k, .99), "max": q(k, 1)} for k in ("total", "wait", "frames", "encode", "policy")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--ref")
    ap.add_argument("--drive")
    ap.add_argument("--label", default="run")
    ap.add_argument("--meta", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = scores(a.run)
    res = {"label": a.label, **dict(m.split("=", 1) for m in a.meta), "run": stats(run)}
    lines = [f"# {a.label}", "", f"containerised: n {res['run']['n']}, mean scene score {res['run']['mean']:.4f}, zeros {res['run']['zeros']}, at 1 {res['run']['ones']}"]
    lat = latency(a.drive) if a.drive and Path(a.drive).stat().st_size else {}
    if lat:
        res["driver"] = lat
        lines += [f"driver: {lat['drives']} drive calls, {lat['inference_errors']} inference errors, total p50 / p90 / p99 / max "
                  f"{lat['total_ms']['p50']} / {lat['total_ms']['p90']} / {lat['total_ms']['p99']} / {lat['total_ms']['max']} ms "
                  f"(encode p50 {lat['encode_ms']['p50']}, frames p50 {lat['frames_ms']['p50']}, wait p50 {lat['wait_ms']['p50']})"]
    if a.ref:
        ref = scores(a.ref)
        both = sorted(set(run) & set(ref))
        diff = [(s, run[s], ref[s]) for s in both if abs(run[s][0] - ref[s][0]) > 1e-9]
        res["ref"], res["common"], res["differ"] = stats({s: ref[s] for s in both}), len(both), len(diff)
        res["max_abs_diff"] = round(max((abs(r[0] - f[0]) for _, r, f in diff), default=0.0), 4)
        res["missing_in_run"], res["missing_in_ref"] = sorted(set(ref) - set(run)), sorted(set(run) - set(ref))
        lines += [f"native reference: n {res['ref']['n']}, mean {res['ref']['mean']:.4f}, zeros {res['ref']['zeros']}, at 1 {res['ref']['ones']}",
                  f"scenes that differ: {len(diff)} of {len(both)} (max |diff| {res['max_abs_diff']})", ""]
        if diff:
            lines += ["| scene | containerised | native | why (containerised / native) |", "|---|--:|--:|---|"]
            lines += [f"| {s} | {r[0]:.4f} | {f[0]:.4f} | {r[1] or '-'} / {f[1] or '-'} |" for s, r, f in diff]
        res["scenes"] = {s: [run[s][0], ref[s][0]] for s in both}
    else:
        res["scenes"] = {s: [v[0]] for s, v in sorted(run.items())}
    out = Path(a.out)
    (out / "smoke.json").write_text(json.dumps(res, indent=1))
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
