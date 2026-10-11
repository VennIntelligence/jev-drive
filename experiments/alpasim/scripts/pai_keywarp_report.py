#!/usr/bin/env python3
"""Lane PAI-KW read-out (plans/2026-10-11-pai-keywarp-prereg.md): the PAI slot switch PAI_KEYWARP, on against off, per arm and for S - base.

  collect  on the Tokyo box (python3 + numpy): one compact JSON per run dir of pai_run.sh / pai_eval.sh: per-scene score and scorer flags,
           the driver's latency per inference, counters, run.json
             pai_keywarp_report.py collect --runs base-s0-on=<run dir> ... --scenes <s_b1a.tsv> --out <dir>
  report   on the Mac (jevdrive.stats): tables from those JSONs -> <dir>/{runs,onoff,sbase,latency,scenes}.{md,csv} and report.md
             python experiments/alpasim/scripts/pai_keywarp_report.py report --dir experiments/alpasim/results/pai_keywarp
Run names: <arm>-s<seed>-<on|off>[-stored] with arm in base, S. `-stored` = a rollout made before the switch existed (decision 247's runs).
Reads: (a) on - off per arm (scene score of an arm = mean of its two checkpoint seeds), paired over scenes; (b) S - base under off, under
on, and the difference of the two. Counts: fixed = zero with the switch off and scored with it on, broken = the reverse.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")


def collect(a):
    ids = [x.split("\t")[0] for x in Path(a.scenes).read_text().splitlines()[1:] if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.runs:
        name, pat = spec.split("=", 1)
        S, lat, cnt, meta = {}, [], [], []
        for d in sorted(q for p in pat.split(",") for q in Path("/").glob(p.lstrip("/"))):
            f = d / "sim/aggregate/results-summary.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["rollouts"]:
                if r["clipgt_id"] in ids:
                    m = r["score_metrics"]
                    S[r["clipgt_id"]] = {"score": r["score"], **{k: bool(m.get(k)) for k in FAIL}, "progress": m.get("progress_score"),
                                         "failure_reason": r.get("failure_reason") or ""}
            for ln in (d / "driver/drive.jsonl").read_text().splitlines():
                r = json.loads(ln)
                if r["kind"] == "drive" and r.get("infer") and r["scene"] in ids:
                    lat.append([r["total_ms"], r["n_real"], r["ms"]["slots"], r["ms"]["encode"], r["ms"]["policy"], r["ms"]["wait"], int(bool(r.get("keywarp")))])
                elif r["kind"] == "close" and r["scene"] in ids:
                    cnt.append({k: r[k] for k in ("drive", "inference", "inference_error", "input_error", "cold")})
            meta.append({"dir": str(d), **(json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {})})
        L = np.array(lat, float).reshape(-1, 7)
        w = L[L[:, 1] == 8]
        pc = lambda x: [float(np.percentile(x, p)) for p in (50, 95, 99, 100)] if len(x) else None  # noqa: E731
        (out / f"{name}.json").write_text(json.dumps({
            "name": name, "runs": meta, "scenes": S, "missing": [i for i in ids if i not in S],
            "latency": {"inferences": int(len(L)), "keywarp_inferences": int(L[:, 6].sum()), "total_ms_p50_p95_p99_max": pc(L[:, 0]),
                        "warm_total_ms_p50_p95_p99_max": pc(w[:, 0]), "share_over_100ms": float((L[:, 0] > 100).mean()) if len(L) else None,
                        "warm_median_ms": dict(zip(("slots", "encode", "policy", "wait"), np.median(w[:, 2:6], 0).tolist())) if len(w) else None},
            "counters": {k: int(sum(c[k] for c in cnt)) for k in ("drive", "inference", "inference_error", "input_error", "cold")}}, indent=1))
        print(name, len(S), "scenes,", len(L), "inferences", "missing", len(ids) - len(S))


def report(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from jevdrive import stats
    D = Path(a.dir)
    R = {p.stem: json.loads(p.read_text()) for p in sorted(D.glob("*-s[01]-*.json"))}
    sc = sorted(set.intersection(*(set(r["scenes"]) for r in R.values())))
    v = {k: np.array([r["scenes"][s]["score"] for s in sc]) for k, r in R.items()}
    off = lambda arm, s: f"{arm}-s{s}-off" if f"{arm}-s{s}-off" in v else f"{arm}-s{s}-off-stored"  # noqa: E731
    L = [f"Scenes scored in every run: {len(sc)}. Bootstrap: jevdrive.stats.paired, 10 000 resamples of scenes, seed 0, percentile 95 % CI.", ""]

    def table(rows, stem, head, fmt=".4f"):
        stats.write_table(rows, D / stem, floatfmt=fmt)
        L.extend(["", f"### {head}", "", (D / f"{stem}.md").read_text().strip(), ""])

    # every run
    table([dict(run=k, scenes=len(sc), mean=float(v[k].mean()), zeros=int((v[k] == 0).sum()),
                **{f: int(sum(R[k]["scenes"][s][f] and R[k]["scenes"][s]["score"] == 0 for s in sc)) for f in FAIL},
                mean_scored=float(v[k][v[k] > 0].mean()) if (v[k] > 0).any() else float("nan"),
                inference_error=R[k]["counters"]["inference_error"], input_error=R[k]["counters"]["input_error"]) for k in R], "runs", "Runs")

    def cmp(name, x, y, rollouts=None):
        """Row of x - y: paired CI over scenes, scene counts, and the decomposition of the mean by zero / scored status."""
        r, d = stats.paired(x, y), x - y
        fx, br, both = (y == 0) & (x > 0), (y > 0) & (x == 0), (y > 0) & (x > 0)
        row = dict(contrast=name, n=r["n"], mean_a=r["mean_a"], mean_b=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"],
                   better=int((d > 0.01).sum()), worse=int((d < -0.01).sum()), within=int((abs(d) <= 0.01).sum()))
        if rollouts is None:                                             # single rollouts on both sides: zero <-> scored is defined
            row.update(fixed=int(fx.sum()), broken=int(br.sum()), both_scored=int(both.sum()), both_zero=int(((x == 0) & (y == 0)).sum()),
                       from_fixed=float(d[fx].sum() / len(d)), from_broken=float(d[br].sum() / len(d)), from_both_scored=float(d[both].sum() / len(d)))
        else:                                                            # a seed mean: counts over the (scene, seed) rollouts
            row.update(fixed=int(sum(((q == 0) & (p > 0)).sum() for p, q in rollouts)), broken=int(sum(((q > 0) & (p == 0)).sum() for p, q in rollouts)),
                       both_scored=int(sum(((q > 0) & (p > 0)).sum() for p, q in rollouts)), both_zero=int(sum(((q == 0) & (p == 0)).sum() for p, q in rollouts)))
        return row
    arm = lambda a_, sw: (v[f"{a_}-s0-on"] + v[f"{a_}-s1-on"]) / 2 if sw == "on" else (v[off(a_, 0)] + v[off(a_, 1)]) / 2  # noqa: E731
    rows = []
    for a_ in ("base", "S"):
        rows.append(cmp(f"{a_} on - off (2-seed mean)", arm(a_, "on"), arm(a_, "off"), [(v[f"{a_}-s{s}-on"], v[off(a_, s)]) for s in (0, 1)]))
        rows += [cmp(f"{a_} s{s} on - off", v[f"{a_}-s{s}-on"], v[off(a_, s)]) for s in (0, 1)]
    allon, alloff = (arm("base", "on") + arm("S", "on")) / 2, (arm("base", "off") + arm("S", "off")) / 2
    rows.append(cmp("all four checkpoints on - off", allon, alloff, [(v[f"{a_}-s{s}-on"], v[off(a_, s)]) for a_ in ("base", "S") for s in (0, 1)]))
    table(rows, "onoff", "(a) Switch on minus off, per arm")
    rows = []
    for sw in ("off", "on"):
        k = (lambda a_, s: f"{a_}-s{s}-on") if sw == "on" else off
        rows.append(cmp(f"S - base, {sw} (2-seed means)", arm("S", sw), arm("base", sw), [(v[k("S", s)], v[k("base", s)]) for s in (0, 1)]))
        rows += [cmp(f"S s{s} - base s{s}, {sw}", v[k("S", s)], v[k("base", s)]) for s in (0, 1)]
    did = cmp("difference of differences: (S - base) on - (S - base) off", arm("S", "on") - arm("base", "on"), arm("S", "off") - arm("base", "off"), [])
    did.update(fixed="", broken="", both_scored="", both_zero="")
    rows.append(did)
    for s in (0, 1):
        d1 = cmp(f"difference of differences, seed {s}", v[f"S-s{s}-on"] - v[f"base-s{s}-on"], v[off("S", s)] - v[off("base", s)], [])
        d1.update(fixed="", broken="", both_scored="", both_zero="")
        rows.append(d1)
    table(rows, "sbase", "(b) S minus base under each frame source")
    # identity / repeatability of the off side
    rows = []
    for k in sorted(R):
        st = k + "-stored" if not k.endswith("-stored") else None
        if st in R:
            d = v[k] - v[st]
            rows.append(dict(rerun=k, stored=st, stored_scene_list=R[st]["runs"][0].get("scenes", ""), scenes=len(sc), identical=int((d == 0).sum()),
                             within_0p01=int((abs(d) <= 0.01).sum()), zero_flips=int(((v[k] == 0) != (v[st] == 0)).sum()), mean_rerun=float(v[k].mean()),
                             mean_stored=float(v[st].mean()), max_abs=float(abs(d).max())))
    if rows:
        table(rows, "offcheck", "Switch-off reruns against the stored rollouts")
    table([dict(run=k, inferences=r["latency"]["inferences"], keywarp=r["latency"]["keywarp_inferences"],
                **dict(zip(("p50", "p95", "p99", "max"), r["latency"]["total_ms_p50_p95_p99_max"])), share_over_100ms=r["latency"]["share_over_100ms"],
                **{f"warm_{q}": x for q, x in (r["latency"]["warm_median_ms"] or {}).items()}) for k, r in R.items()], "latency", "Driver latency per inference (ms, `total_ms` of drive.jsonl)", ".2f")
    ks = list(R)
    (D / "scenes.tsv").write_text("scene_id\t" + "\t".join(ks) + "\n" + "".join(s + "\t" + "\t".join(f"{R[k]['scenes'][s]['score']:.4f}" for k in ks) + "\n" for s in sc))
    (D / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", nargs="+", required=True), c.add_argument("--scenes", required=True), c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    a = ap.parse_args()
    {"collect": collect, "report": report}[a.cmd](a)
