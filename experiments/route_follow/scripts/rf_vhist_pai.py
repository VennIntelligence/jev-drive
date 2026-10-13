#!/usr/bin/env python3
"""VHIST on PAI (plans/2026-10-13-vhist-prereg.md, Amendment 2): an arm's PAI 281 read against the stored reference, paired per scene.
numpy only (Tokyo python3 or the Mac .venv); pai_eval.sh roots are read, never written.

  rf_vhist_pai.py --ref LABEL=<root> --arm LABEL=<root>[,<root>..] --lists all=<file> exam=<file> .. --out DIR

Per label and scene list: mean score; zeros by kind (rf_pai281's order: collision = collision_at_fault, else offroad, else corridor =
left_corridor_laterally, else other); front-collision zeros (score 0 and collision_front > 0, any kind); rear-ended = rollouts with
collision_rear > 0 (all / scored > 0; PAI ends a rollout on rear contact); progress score (score_metrics.progress_score); GT distance
(metrics.dist_to_gt_trajectory); spins (steer_long sf_detect.py on the controller CSV); plan 2 s deficit (rf_pai281's PAI rule: served plan
arc through its 0.5 .. 2.0 s poses minus the measured speed x 2 s, inferences with speed >= 1 m/s, whole rollout): pooled decision median
and mean of rollout means. A rollout without metrics (infrastructure failure) is dropped, a later root replaces an earlier one per scene;
the paired set is the scenes both labels drove. Paired differences arm - ref: mean with a scene bootstrap 95 % CI (10 000 draws, seed 0).
-> DIR/vhist_pai.json, DIR/tables.md, DIR/scenes.csv
"""
import argparse, csv, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../steer_long/scripts"))
from sf_detect import detect  # noqa: E402


def rollouts(roots):
    out = {}
    for root in roots.split(","):
        for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))):
            f = os.path.join(sim, "aggregate", "results-summary.json")
            if not os.path.exists(f):
                continue
            for r in json.load(open(f))["rollouts"]:
                m = r.get("metrics") or {}
                if not m:
                    continue
                sc, z = float(r.get("score") or 0), not r.get("score")
                kind = "" if not z else "collision" if m.get("collision_at_fault") else "offroad" if m.get("offroad") else \
                    "corridor" if m.get("left_corridor_laterally") else "other"
                spin = None
                c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
                if os.path.exists(c):
                    rows = list(csv.DictReader(open(c)))
                    t = (np.array([float(x["timestamp_us"]) for x in rows]) - float(rows[0]["timestamp_us"])) / 1e6 + 0.1
                    g = lambda k: np.array([float(x[k]) for x in rows])  # noqa: E731
                    spin = bool(detect(t, g("vx"), g("vy"), g("wz"), g("front_steering_angle"))["spin"])
                out[r["clipgt_id"]] = dict(score=sc, kind=kind, front_zero=bool(z and (m.get("collision_front") or 0) > 0),
                                           rear=bool((m.get("collision_rear") or 0) > 0), prog=float((r.get("score_metrics") or {}).get("progress_score", np.nan)),
                                           gt=float(m.get("dist_to_gt_trajectory", np.nan)), spin=spin)
    return out


def deficit(roots):
    """{scene: per-inference deficits (m)} of the last driver session of each scene."""
    out = {}
    for root in roots.split(","):
        for f in sorted(glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl"))):
            n = {r["session"]: r["n"] for r in map(json.loads, open(os.path.join(os.path.dirname(f), "start.jsonl")))}
            S = {}
            for line in open(f):
                if '"infer": true' not in line:
                    continue
                r = json.loads(line)
                v = 10.0 * float(np.hypot(r["ego"][4], r["ego"][5]))
                P = np.array(r["poses"])[:4, :2]
                arc = float(np.hypot(*P[0]) + np.hypot(*np.diff(P, axis=0).T).sum())
                S.setdefault(r["session"], (r["scene"], []))[1].append(arc - 2.0 * v if v >= 1 else np.nan)
            best = {}
            for ss, (sc, d) in S.items():
                if sc not in best or n.get(ss, -1) > best[sc][0]:
                    best[sc] = (n.get(ss, -1), np.array(d))
            out.update({sc: d[np.isfinite(d)] for sc, (_, d) in best.items()})
    return out


def boot(x, rng):
    b = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (rl, rr), (al, ar) = a.ref.split("=", 1), a.arm.split("=", 1)
    D = {rl: rollouts(rr), al: rollouts(ar)}
    F = {rl: deficit(rr), al: deficit(ar)}
    os.makedirs(a.out, exist_ok=True)
    res, md = {}, []
    for spec in a.lists:
        name, f = spec.split("=", 1)
        listed = open(f).read().split()
        sc = [s for s in listed if all(s in d for d in D.values())]
        rng = np.random.default_rng(0)
        R = dict(scenes_listed=len(listed), scenes_paired=len(sc), labels={}, diff={})
        for lab, d in D.items():
            z = [d[s]["kind"] for s in sc if d[s]["kind"]]
            dec = np.concatenate([F[lab].get(s, np.array([])) for s in sc])
            rm = [F[lab][s].mean() for s in sc if len(F[lab].get(s, []))]
            R["labels"][lab] = dict(mean=float(np.mean([d[s]["score"] for s in sc])), zeros=len(z), collision=z.count("collision"),
                                    offroad=z.count("offroad"), corridor=z.count("corridor"), other=z.count("other"),
                                    front_zeros=sum(d[s]["front_zero"] for s in sc), rear=sum(d[s]["rear"] for s in sc),
                                    rear_scored=sum(d[s]["rear"] and d[s]["score"] > 0 for s in sc),
                                    prog=float(np.nanmean([d[s]["prog"] for s in sc])), gt=float(np.nanmean([d[s]["gt"] for s in sc])),
                                    spins=sum(bool(d[s]["spin"]) for s in sc), def_med=float(np.median(dec)) if len(dec) else np.nan,
                                    def_mean=float(np.mean(rm)) if rm else np.nan, decisions=int(len(dec)))
        x = lambda k: np.array([float(D[al][s][k]) - float(D[rl][s][k]) for s in sc])  # noqa: E731
        R["diff"] = dict(score=boot(x("score"), rng), prog=boot(x("prog"), rng), gt=boot(x("gt"), rng),
                         zeros=int(sum(D[al][s]["score"] == 0 for s in sc) - sum(D[rl][s]["score"] == 0 for s in sc)),
                         front_zeros=R["labels"][al]["front_zeros"] - R["labels"][rl]["front_zeros"],
                         rear=R["labels"][al]["rear"] - R["labels"][rl]["rear"])
        both = [s for s in sc if len(F[al].get(s, [])) and len(F[rl].get(s, []))]
        R["diff"]["deficit"] = boot(np.array([F[al][s].mean() - F[rl][s].mean() for s in both]), rng) if both else None
        R["flips"] = dict(to_zero=[s for s in sc if D[al][s]["score"] == 0 < D[rl][s]["score"]],
                          from_zero=[s for s in sc if D[rl][s]["score"] == 0 < D[al][s]["score"]])
        res[name] = R
        L = R["labels"]
        md += [f"## {name}: {len(sc)} paired scenes (of {len(listed)} listed)", "",
               "| label | mean | zeros (coll / off / corr / other) | front-collision zeros | rear-ended (all / scored > 0) | progress score | "
               "GT (m) | spins | deficit median / mean of rollout means (m) |", "|:--|--:|:--|--:|:--|--:|--:|--:|:--|"]
        md += [f"| {k} | {v['mean']:.4f} | {v['zeros']} ({v['collision']} / {v['offroad']} / {v['corridor']} / {v['other']}) | {v['front_zeros']} | "
               f"{v['rear']} / {v['rear_scored']} | {v['prog']:.4f} | {v['gt']:.3f} | {v['spins']} | {v['def_med']:+.3f} / {v['def_mean']:+.3f} |"
               for k, v in L.items()]
        d = R["diff"]
        ci = lambda t, f: f"{t[0]:{f}} [{t[1]:{f}}, {t[2]:{f}}]"  # noqa: E731
        md += ["", f"{al} - {rl}: score {ci(d['score'], '+.4f')}, zeros {d['zeros']:+d}, front-collision zeros {d['front_zeros']:+d}, "
               f"rear-ended {d['rear']:+d}, progress score {ci(d['prog'], '+.4f')}, GT {ci(d['gt'], '+.3f')} m, deficit (rollout means) "
               f"{ci(d['deficit'], '+.3f') if d['deficit'] else '-'} m; to zero {len(R['flips']['to_zero'])}, from zero {len(R['flips']['from_zero'])}", ""]
    json.dump(res, open(os.path.join(a.out, "vhist_pai.json"), "w"), indent=1)
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "scenes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "scene", "score", "kind", "front_zero", "rear", "prog", "gt", "spin", "deficit_mean"])
        for lab, d in D.items():
            for s, v in sorted(d.items()):
                f = F[lab].get(s, np.array([]))
                w.writerow([lab, s, v["score"], v["kind"], int(v["front_zero"]), int(v["rear"]), v["prog"], v["gt"], v["spin"],
                            round(float(f.mean()), 4) if len(f) else ""])
    print("\n".join(md))


if __name__ == "__main__":
    main()
