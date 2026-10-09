"""PAI1 scene sample: N scenes of the curated NuRec validation split (441), one per quantile bin of the organisers' reference
difficulty, so a small set spans easy to hard. Difficulty of a scene = mean scene score over the bundled reference subjects
(e2e_challenge/local_evaluation/data/pai, 8 subjects x 3 rollouts). The first `--first` scenes of the output are a nested subsample
(every N / first-th bin), for the staged 10 -> N launch. Plain python3 (stdlib), run on the Tokyo box.

  pai_scenes.py --src <alpasim checkout> --ref <dir with <subject>.json> --n 40 --first 10 --out pai_scenes_40.tsv
The reference summaries are git-lfs files; <dir> holds them as <subject>.json (docs/alpasim.md, PAI section).
Output: tab-separated scene_id, uuid, hf path, hf revision, reference mean, alpamayo1, vavam-nonlinear, stage (1 = first subsample).
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def ref_scores(ref: Path) -> dict:
    """{subject: {scene: mean score over its rollouts}}."""
    out = {}
    for p in sorted(ref.glob("*.json")):
        if p.stem == "per_scene":
            continue
        s = defaultdict(list)
        for r in json.loads(p.read_text())["rollouts"]:
            s[r["clipgt_id"]].append(float(r["score"]))
        out[p.stem] = {k: sum(v) / len(v) for k, v in s.items()}
    return out


def catalog(src: Path, suite: str = "nurec_curated_val") -> dict:
    """{scene_id: row of the scenes csv} for the suite, resolved as curated_val.yaml does (26.01 catalogue first, then 26.04)."""
    d = src / "data/scenes"
    ids = [r["scene_id"] for r in csv.DictReader(open(d / "sim_suites_curated.csv")) if r["test_suite_id"] == suite]
    rows = {}
    for f in ("sim_scenes.csv", "sim_scenes_2604.csv"):
        for r in csv.DictReader(open(d / f)):
            rows.setdefault(r["scene_id"], r)
    return {i: rows[i] for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True), ap.add_argument("--ref", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40), ap.add_argument("--first", type=int, default=10), ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R, cat = ref_scores(Path(a.ref)), catalog(Path(a.src))
    mean = {s: sum(R[k][s] for k in R) / len(R) for s in cat if all(s in R[k] for k in R)}
    order = sorted(mean, key=lambda s: (mean[s], s))
    rng, pick = random.Random(a.seed), []
    for b in range(a.n):
        pick.append(rng.choice(order[b * len(order) // a.n:(b + 1) * len(order) // a.n]))
    step = a.n // a.first
    first = pick[step // 2::step][:a.first]
    rows = [(s, 1) for s in first] + [(s, 2) for s in pick if s not in first]
    with open(a.out, "w") as f:
        f.write("scene_id\tuuid\tpath\trevision\tref_mean\talpamayo1\tvavam_nonlinear\tstage\n")
        for s, st in rows:
            c = cat[s]
            f.write(f"{s}\t{c['uuid']}\t{c['path']}\t{c.get('hf_revision') or ''}\t{mean[s]:.4f}\t{R['alpamayo1'][s]:.4f}\t"
                    f"{R['vavam-nonlinear'][s]:.4f}\t{st}\n")
    print(f"{len(cat)} suite scenes, {len(mean)} with all {len(R)} references; subject means on the suite: "
          + ", ".join(f"{k} {sum(v[s] for s in mean) / len(mean):.4f}" for k, v in R.items()))
    for n, sel in ((len(first), first), (len(pick), pick)):
        print(f"sample {n}: reference mean {sum(mean[s] for s in sel) / n:.4f} (suite {sum(mean.values()) / len(mean):.4f}), "
              + ", ".join(f"{k} {sum(R[k][s] for s in sel) / n:.4f}" for k in ("alpamayo1", "vavam-nonlinear")))


if __name__ == "__main__":
    main()
