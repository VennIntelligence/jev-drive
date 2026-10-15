#!/usr/bin/env python3
"""OTHERS-TURN side read (plans/2026-10-15-others-turn-prereg.md, score-only, PAI): the organisers' bundled PAI reference subjects
(alpasim 0bb4c4b e2e_challenge/local_evaluation/data/pai/<subject>/aggregate/results-summary.json, git-lfs; alternative_* also in Tokyo's
/data/runs/alpasim/pai1/ref) on our 36 strict PAI turns, next to RSG-L18-s0's PAI 281 read (path 0, results/pai281/RV2a-L18-s0.turns.json label rsg).
No ego path in the summaries: pass = score > 0 per rollout (3 per scene), not turn completion. Plain python3.

  ot_pai_ref.py --ref <dir with <subject>.json> --out results/pai_ref.md
"""
import argparse
import json
from pathlib import Path

R_ = Path(__file__).resolve().parents[3]
RF = R_ / "experiments/route_follow/results"


def main(a):
    strict = RF.joinpath("pai281/strict36.txt").read_text().split()
    rows = [r for r in json.loads(RF.joinpath("pai281/RV2a-L18-s0.turns.json").read_text())["rows"] if r["label"] == "rsg"]
    rsg = {r["scene"]: r for r in rows}
    subj = {}
    for f in sorted(Path(a.ref).glob("*.json")):
        d = json.loads(f.read_text())
        S = {}
        for r in d["rollouts"]:
            S.setdefault(r["clipgt_id"], []).append(r)
        subj[f.stem] = S
    sc = [s for s in strict if s in rsg and all(s in S for S in subj.values())]
    md = ["# OTHERS-TURN side read: organisers' PAI reference subjects on the 36 strict turns (score only)", "",
          f"Scenes: {len(sc)} of {len(strict)} strict turns present in every subject and in RSG's PAI 281 rows. Subjects: 3 rollouts each (organisers' runs); "
          "RSG-L18-s0: 1 rollout, path 0 (JEV_TRACK 0, gains 20), completion from rf_pai281 (ego path). Pass = score > 0. Fail kinds per rollout.", "",
          "| subject | mean score | pass (rollouts) | scenes passed >= 2 / 3 | offroad | corridor | at-fault collision | mean progress_rel | all-441 mean |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    res = {}
    for k, S in sorted(subj.items(), key=lambda kv: -sum(float(r["score"]) for s in sc for r in kv[1][s])):
        R = [r for s in sc for r in S[s]]
        m = lambda q: sum(float((r.get("metrics") or {}).get(q) or 0) > 0 for r in R)  # noqa: E731
        allm = sum(float(r["score"]) for v in S.values() for r in v) / sum(len(v) for v in S.values())
        res[k] = dict(mean=sum(float(r["score"]) for r in R) / len(R), pass_=sum(float(r["score"]) > 0 for r in R), n=len(R),
                      scenes2=sum(sum(float(r["score"]) > 0 for r in S[s]) >= 2 for s in sc), off=m("offroad"), cor=m("left_corridor_laterally"),
                      col=m("collision_at_fault"), prog=sum(float((r.get("metrics") or {}).get("progress_rel") or 0) for r in R) / len(R), all441=allm)
        x = res[k]
        md.append(f"| {k} | {x['mean']:.3f} | {x['pass_']} / {x['n']} | {x['scenes2']} / {len(sc)} | {x['off']} | {x['cor']} | {x['col']} | {x['prog']:.2f} | {x['all441']:.3f} |")
    g = [rsg[s] for s in sc]
    res["RSG-L18-s0"] = dict(mean=sum(r["score"] for r in g) / len(g), pass_=sum(r["score"] > 0 for r in g), n=len(g),
                             completed=sum(bool(r["completed"]) for r in g))
    x = res["RSG-L18-s0"]
    md += ["", f"RSG-L18-s0 (ours, 1 rollout): mean {x['mean']:.3f}, pass {x['pass_']} / {x['n']}, turn completed {x['completed']} / {x['n']} (ego path).", "",
           "Per scene (subjects: passes of 3; RSG: score / completed):", "",
           "| scene | " + " | ".join(sorted(subj)) + " | RSG |", "|:--|" + "--:|" * (len(subj) + 1)]
    for s in sc:
        md.append(f"| {s[7:15]} | " + " | ".join(str(sum(float(r['score']) > 0 for r in subj[k][s])) for k in sorted(subj))
                  + f" | {rsg[s]['score']:.2f} / {'Y' if rsg[s]['completed'] else 'n'} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    print("\n".join(md[:16]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
