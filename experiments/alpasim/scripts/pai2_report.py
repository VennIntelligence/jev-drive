"""PAI2 read-outs for native runs on the GPU box (pai_native.sh) and their comparison with containerised Tokyo runs. Plain python3.

  parity    per-scene agreement of runs with a reference run; classes: score 1, zero by kind, progress (0 < s < 1)
              pai2_report.py parity --ref tokyo=<run dir | results-summary.json> --runs a=<dir> b=<dir> --out parity.md
  baseline  per-seed table (mean scene score, score 1, zeros by kind, slow, wall), the pooled mean, and the organisers' reference rows
            on the same scenes
              pai2_report.py baseline --runs s0=<dir>[,<dir>...] s1=<dir>[,<dir>...] --ref <dir with <subject>.json> --out base.md
  stats     wall time and per-service VRAM / RSS / cores peaks of native runs (native_summary.json)
              pai2_report.py stats --runs name=<dir> ... --out stats.md
A run dir is a native run (aggregate/results-summary.json, native_summary.json, usage.jsonl) or a Tokyo pai_run.sh dir (sim/aggregate/...).
Several dirs under one label (comma) are chunks of one seed.
"""
import argparse
import json
from pathlib import Path

FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")


def summary(p: Path) -> Path:
    for c in (p, p / "results-summary.json", p / "aggregate/results-summary.json", p / "sim/aggregate/results-summary.json"):
        if c.is_file():
            return c
    raise FileNotFoundError(p)


def load(spec: str) -> dict:
    """{scene id: rollout} over comma-separated dirs."""
    out = {}
    for p in spec.split(","):
        for r in json.loads(summary(Path(p)).read_text())["rollouts"]:
            out[r["clipgt_id"]] = r
    return out


def kind(r) -> str:
    m = r["score_metrics"]
    z = [f for f in FAIL if m.get(f)]
    if r["score"] >= 1:
        return "pass"
    if z:
        return z[0]
    return "broken" if r.get("failure_reason") and not r.get("metrics") else "slow"


def short(s: str) -> str:
    return s.replace("clipgt-", "")[:8]


def cmd_parity(a):
    ref = load(a.ref.split("=", 1)[1]); rn = a.ref.split("=", 1)[0]
    runs = {k: load(v) for k, v in (x.split("=", 1) for x in a.runs)}
    sc = sorted(set(ref).intersection(*runs.values()), key=lambda s: ref[s]["score"])
    L = [f"Scenes: {len(sc)}. Reference run: {rn}. Class = pass (score 1) / zero kind / slow (0 < s < 1).", "",
         "| Run | Mean | Pass | Zeros | Same class as ref | Same class, same score +-0.01 | Max |score - ref| |", "|---|--:|--:|--:|--:|--:|--:|"]
    for k, R in {rn: ref, **runs}.items():
        sam = sum(kind(R[s]) == kind(ref[s]) for s in sc); eq = sum(abs(R[s]["score"] - ref[s]["score"]) <= 0.01 for s in sc)
        L.append(f"| {k} | {sum(R[s]['score'] for s in sc) / len(sc):.4f} | {sum(R[s]['score'] >= 1 for s in sc)} | {sum(R[s]['score'] == 0 for s in sc)} | "
                 f"{sam}/{len(sc)} | {eq}/{len(sc)} | {max(abs(R[s]['score'] - ref[s]['score']) for s in sc):.3f} |")
    L += ["", "| Scene | " + " | ".join(f"{k} score | {k} class" for k in (rn, *runs)) + " |", "|---|" + "--:|---|" * (1 + len(runs))]
    for s in sc:
        L.append(f"| {short(s)} | " + " | ".join(f"{R[s]['score']:.3f} | {kind(R[s])}" for R in (ref, *runs.values())) + " |")
    ch = [(k, [short(s) for s in sc if kind(R[s]) != kind(ref[s])]) for k, R in runs.items()]
    L += [""] + [f"Class changes vs {rn}, {k}: {len(c)} ({', '.join(c) or 'none'})" for k, c in ch]
    if len(runs) > 1:
        ks = list(runs); r0, r1 = runs[ks[0]], runs[ks[1]]
        L.append(f"{ks[0]} vs {ks[1]} (same code, same box): class changes {sum(kind(r0[s]) != kind(r1[s]) for s in sc)}, "
                 f"scores equal to 0.01 on {sum(abs(r0[s]['score'] - r1[s]['score']) <= 0.01 for s in sc)}/{len(sc)}")
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


def refs(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")):
        s = {}
        for r in json.loads(p.read_text())["rollouts"]:
            s.setdefault(r["clipgt_id"], []).append(r)
        out[p.stem] = s
    return out


def row(name, scores, kinds) -> str:
    n = len(scores)
    z = {f: sum(k == f for k in kinds) for f in FAIL}
    return (f"| {name} | {n} | {sum(scores) / n:.4f} | {sum(s >= 1 for s in scores)} | {sum(s == 0 for s in scores)} | {z[FAIL[0]]} / {z[FAIL[1]]} / {z[FAIL[2]]} | "
            f"{sum(0 < s < 1 for s in scores)} |")


def cmd_baseline(a):
    runs = {k: load(v) for k, v in (x.split("=", 1) for x in a.runs)}
    sc = sorted(set.intersection(*(set(v) for v in runs.values())))
    L = [f"Scenes scored in every seed: {len(sc)}.", "", "| Row | Scenes | Mean scene score | Score 1 | Zeros | Zeros: at-fault collision / offroad / left corridor | Slow (0 < s < 1) |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for k, R in runs.items():
        L.append(row(k, [R[s]["score"] for s in sc], [kind(R[s]) for s in sc]))
    own = {k: R for k, R in runs.items() if not k.startswith("tokyo")}
    if len(own) > 1:
        L.append(row("pooled over the seeds above (not Tokyo rows)", [R[s]["score"] for R in own.values() for s in sc], [kind(R[s]) for R in own.values() for s in sc]))
    if a.ref:
        for k, R in refs(Path(a.ref)).items():
            ss = [x for s in sc if s in R for x in R[s]]
            if ss:
                L.append(row(f"{k} (reference, {len(ss)} rollouts)", [x["score"] for x in ss], [kind(x) for x in ss]))
    # per-scene table
    L += ["", "| Scene | " + " | ".join(runs) + " |", "|---|" + "--:|" * len(runs)]
    for s in sc:
        L.append(f"| {short(s)} | " + " | ".join(f"{R[s]['score']:.3f} {kind(R[s]) if kind(R[s]) not in ('pass', 'slow') else ''}" for R in runs.values()) + " |")
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


def cmd_stats(a):
    """Wall time and per-service peaks of native runs (native_summary.json): one row per run, then the maximum over runs."""
    L = ["| Run | Scenes | Wall s (runtime) | s per scene | Renderer VRAM GiB | Driver VRAM GiB | Physics VRAM GiB | Stack VRAM GiB (sum of peaks) | Host RSS GiB (sum of peaks) | Mean cores |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    mx = [0.0] * 6
    for spec in a.runs:
        k, d = spec.split("=", 1); d = Path(d)
        j = json.loads((d / "native_summary.json").read_text()); t, pk = j["times"], j["peak"]
        n = t.get("rollouts") or len(summary_rows(d)); g = lambda s_: next((v["gpu_mib"] for kk, v in pk.items() if kk.startswith(s_)), 0) / 1024
        v = [g("renderer"), g("driver"), g("physics"), sum(x["gpu_mib"] for x in pk.values()) / 1024, sum(x["rss_gib"] for x in pk.values()),
             sum(x["cpu_s"] for x in pk.values()) / t["total_s"]]
        mx = [max(m, x) for m, x in zip(mx, v)]
        L.append(f"| {k} | {n} | {t['runtime_s']:.0f} | {t['runtime_s'] / n:.1f} | " + " | ".join(f"{x:.1f}" for x in v[:5]) + f" | {v[5]:.1f} |")
    L.append("| max over runs | | | | " + " | ".join(f"{x:.1f}" for x in mx[:5]) + f" | {mx[5]:.1f} |")
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


def summary_rows(d: Path) -> list:
    return json.loads(summary(d).read_text())["rollouts"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("parity"); p.add_argument("--ref", required=True); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("baseline"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--ref"); p.add_argument("--out", required=True)
    p = sp.add_parser("stats"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(); {"parity": cmd_parity, "baseline": cmd_baseline, "stats": cmd_stats}[a.cmd](a)
