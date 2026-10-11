"""ONPOL-PAI: the pre-registered lines H and V on the 281 test scenes (plans/2026-10-11-onpol-pai-prereg.md, section 8), from the run dirs.
Plain python3 on the Tokyo box; reads scores and scorer flags through experiments/alpasim/scripts/pai_report.py, as pai_eval_report.py does.

  onp_lines.py [--arms self imit ft] [--read /data/runs/onpol_pai/read281] [--out lines.md]

At-fault zero = zero-score scene with the at-fault collision or the offroad flag. Fixed = zero for the control seed and scored for the arm,
broken = the reverse. An arm whose 281 scenes are not all read is listed as incomplete and gets no verdict.
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "alpasim" / "scripts"))
import pai_report as P  # noqa: E402

A, NULL = "/data/runs/alpasim", 12
CTRL = {"s0": [A + "/pai_full/runs/p2h10s-f-s0_*"], "s1": [A + "/pai2c/runs/s1_b1a_c4", A + "/pai_s1/runs/*"],
        "s2": [A + "/pai281_ext/queue/s2/runs/*"], "s3": [A + "/pai281_ext/queue/s3/runs/*"]}
KIND = ("collision_at_fault", "offroad", "left_corridor_laterally")


def load(pats, test):
    R = {}
    for p in pats:
        for d in sorted(glob.glob(p)):
            if (Path(d) / "sim/aggregate/results-summary.json").exists():
                R |= P.rollouts(Path(d))
    return {s: v for s, v in R.items() if s in test}


def row(R):
    z = [v for v in R.values() if v["score"] == 0]
    k = [sum(1 for v in z if v["score_metrics"].get(n)) for n in KIND]
    af = sum(1 for v in z if v["score_metrics"].get(KIND[0]) or v["score_metrics"].get(KIND[1]))
    return dict(n=len(R), mean=sum(v["score"] for v in R.values()) / max(len(R), 1), zeros=len(z), coll=k[0], off=k[1], corr=k[2], af=af)


def fb(a, b):
    c = sorted(set(a) & set(b))
    return sum(b[s]["score"] == 0 < a[s]["score"] for s in c), sum(a[s]["score"] == 0 < b[s]["score"] for s in c)


ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["self", "imit", "ft"]), ap.add_argument("--read", default="/data/runs/onpol_pai/read281")
ap.add_argument("--split", default="/data/runs/onpol_pai/split/test281.tsv"), ap.add_argument("--out")
a = ap.parse_args()
test = {ln.split("\t")[0] for ln in open(a.split).read().splitlines()[1:]}
R = {k: load(v, test) for k, v in CTRL.items()} | {k: load([f"{a.read}/{k}/runs/*"], test) for k in a.arms}
S = {k: row(v) for k, v in R.items()}
L = ["| Row | Scenes | Mean | Zeros | at-fault collision | offroad | left corridor | At-fault zeros |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
L += [f"| {k} | {s['n']} | {s['mean']:.4f} | {s['zeros']} | {s['coll']} | {s['off']} | {s['corr']} | {s['af']} |" for k, s in S.items()]
L += ["", "| Arm | fixed / broken / net against s0 (parent) | s1 | s2 | s3 |", "|---|---|---|---|---|"]
net = {}
for k in a.arms:
    c = [fb(R[k], R[s]) for s in CTRL]
    net[k] = c[0][0] - c[0][1]
    L.append(f"| {k} | " + " | ".join(f"{f} / {b} / {f - b:+d}" for f, b in c) + " |")
full = {k: S[k]["n"] == len(test) for k in a.arms}
hmax = max(S[s]["af"] for s in CTRL)
L += ["", f"Control at-fault zeros: " + " / ".join(str(S[s]['af']) for s in CTRL) + f" (parent s0 {S['s0']['af']}, largest {hmax}); seed-pair null of fixed minus broken: {NULL}.", ""]
for k in a.arms:
    L.append(f"- **H, {k}**: " + (f"at-fault zeros {S[k]['af']} against the line {min(S['s0']['af'], hmax)}: {'PASS' if S[k]['af'] <= min(S['s0']['af'], hmax) else 'NEGATIVE'}"
                                  if full[k] else f"incomplete ({S[k]['n']} of {len(test)} scenes), no verdict"))
if all(full.get(k) for k in ("self", "imit")):
    s, i = S["self"], S["imit"]
    v = dict(a=all(s[x] < i[x] for x in ("coll", "off", "corr")), b=s["af"] < i["af"], c=net["self"] > NULL, d=s["af"] <= min(S["s0"]["af"], hmax))
    L.append(f"- **V (self beats imitation)**: (a) fewer zeros of each kind {s['coll']} / {s['off']} / {s['corr']} against {i['coll']} / {i['off']} / {i['corr']}: {v['a']}; "
             f"(b) fewer at-fault zeros {s['af']} against {i['af']}: {v['b']}; (c) fixed minus broken against the parent {net['self']:+d} > {NULL}: {v['c']}; (d) H: {v['d']}. "
             f"**{'MET' if all(v.values()) else 'NOT MET'}**")
out = "\n".join(L) + "\n"
print(out)
if a.out:
    Path(a.out).write_text(out)
