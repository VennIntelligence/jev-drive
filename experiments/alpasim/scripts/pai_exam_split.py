#!/usr/bin/env python3
"""PAI exam split (user decision 2026-10-13: PAI becomes the exam). Over the scenes every reference read covers: hard = scored zero in any
reference; control = a seeded random half of the never-zero scenes; trainable = the other never-zero half. A zero caused by infrastructure
(results-summary failure_reason other than offroad / left_corridor_laterally / collision_at_fault, e.g. an RPC crash before any drive call)
does not count as a zero; such scenes are excluded from all three lists and listed. Plain python3.

  pai_exam_split.py --refs ybase_g20=/data/runs/route_night/r281/ybase_g20 ys0_g20=... rsg=... ys1=/data/runs/onpol_long/pai281/ys1 rbsg2=... \
      --out experiments/alpasim/results/pai_exam [--seed 0]
"""
import argparse, glob, json, os, random
from pathlib import Path

KINDS = ("offroad", "left_corridor_laterally", "collision_at_fault")


def read(root):
    out = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "sim", "aggregate", "results-summary.json")):
        for r in json.load(open(f))["rollouts"]:
            fr = r.get("failure_reason") or ""
            out[r["clipgt_id"]] = "infra" if (not r["score"] and fr not in KINDS) else ("zero:" + fr if not r["score"] else "ok")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = {k: read(v) for k, v in (x.split("=", 1) for x in a.refs)}
    cover = sorted(set.intersection(*[set(d) for d in R.values()]))
    infra = sorted(s for s in cover if any(d[s] == "infra" for d in R.values()))
    ok = [s for s in cover if s not in infra]
    hard = sorted(s for s in ok if any(d[s].startswith("zero") for d in R.values()))
    never = sorted(s for s in ok if s not in hard)
    rnd = random.Random(a.seed)
    sh = never[:]
    rnd.shuffle(sh)
    control, trainable = sorted(sh[:len(sh) // 2]), sorted(sh[len(sh) // 2:])
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    for n, l in (("hard", hard), ("control", control), ("trainable", trainable), ("infra_excluded", infra)):
        (o / f"{n}.txt").write_text("".join(s + "\n" for s in l))
    per = {k: dict(zeros=sum(d[s].startswith("zero") for s in cover), infra=sum(d[s] == "infra" for s in cover),
                   **{x: sum(d[s] == "zero:" + x for s in cover) for x in KINDS}) for k, d in R.items()}
    rec = dict(refs=dict(x.split("=", 1) for x in a.refs), seed=a.seed, covered=len(cover), infra_excluded=len(infra), hard=len(hard),
               control=len(control), trainable=len(trainable), per_ref=per)
    (o / "split.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
