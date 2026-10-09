"""FIX1, PAI track: helper of fix1_pai_clips.sh (one review GIF per zero-score rollout, rendered by COL1's pipeline).

  fix1_pai_clips.py zeros --run <pai_run.sh out dir>                        plain python; prints scene8 <tab> score <tab> why <tab> has_rollout
  fix1_pai_clips.py cases --x <col1_pai_extract.py dir> --zeros <tsv from zeros> --run <name> --out <dir>
                                                                            numpy (trusted image); writes <out>/cases.csv for col1_pai_clip.py
                                                                            (scene, run, t_from, t_to, name) and <out>/why.tsv (scene8 <tab> why)
why = collision | offroad | corridor | other; the window is 8 s before to 1.5 s after the first scored step of the flag (COL1's choice
for the at-fault collisions), or the last 9.5 s of the rollout when no flag fired.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

FLAGS = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor"}


def zeros(a):
    S = json.loads((Path(a.run) / "sim/aggregate/results-summary.json").read_text())["rollouts"]
    for r in sorted(S, key=lambda r: r["clipgt_id"]):
        if r.get("score") is None or float(r["score"]) != 0.0:
            continue
        m = r.get("score_metrics") or {}
        fl = [FLAGS[f] for f in FLAGS if m.get(f)] or ([FLAGS[r["failure_reason"]]] if r.get("failure_reason") in FLAGS else [])
        has = (Path(a.run) / "sim/rollouts" / r["clipgt_id"]).is_dir()
        print(r["clipgt_id"][7:15], r["score"], fl[0] if fl else "other", int(has), sep="\t")


def cases(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import col1_lib as L
    logs = L.load(f"{a.x}/logs.pkl")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, whys = [], []
    for line in open(a.zeros):
        scene, score, why0, has = line.rstrip("\n").split("\t")
        s = next((k for k in logs if k[7:15] == scene), None)
        if s is None:
            print("no extract for", scene, flush=True)
            continue
        o = logs[s]
        ev = {n: L.first_event(o, f) for f, n in FLAGS.items()}
        ev = {n: t for n, t in ev.items() if t is not None}
        T0 = o["actors"]["EGO"][0, 0]
        if ev:
            why = min(ev, key=ev.get)
            t = (ev[why] - T0) * 1e-6
            t_from, t_to = max(0.0, t - 8.0), t + 1.5
        else:
            why, tend = "other", (o["actors"]["EGO"][-1, 0] - T0) * 1e-6
            t_from, t_to = max(0.0, tend - 9.5), tend
        rows.append([scene, a.run, round(t_from, 1), round(t_to, 1), why]), whys.append((scene, why))
    with open(out / "cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "run", "t_from", "t_to", "name"]), w.writerows(rows)
    (out / "why.tsv").write_text("".join(f"{s}\t{w}\n" for s, w in whys))
    print("cases", len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("zeros")
    p.add_argument("--run", required=True), p.set_defaults(f=zeros)
    p = sp.add_parser("cases")
    for k in ("x", "zeros", "run", "out"):
        p.add_argument(f"--{k}", required=True)
    p.set_defaults(f=cases)
    a = ap.parse_args()
    a.f(a)
