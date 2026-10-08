"""C0 read-out: several AlpaSim runs on the same public scenes -> zero-score table, slow scenes, at-fault events,
run-to-run difference, best-of-two. Run with plain python3 on the box (stdlib + numpy).

  c0_report.py --sh30 <run dir> --ap2a <run dir> --ap2b <run dir> [--ref wajepa=<run dir> ...] --out results/c0_public400.md --json c0.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")


def load(run):
    d = json.loads((Path(run) / "aggregate/results-summary.json").read_text())
    return {r["clipgt_id"]: r for r in d["rollouts"]}


def zero_kinds(r):
    m = r["score_metrics"]
    return [f for f in FAIL if m.get(f)]


def stats(R, scenes):
    s = np.array([R[x]["score"] for x in scenes])
    zk = {x: zero_kinds(R[x]) for x in scenes}
    z = [x for x in scenes if zk[x] or R[x]["score"] == 0]
    slow = [x for x in scenes if R[x]["score"] < 1 and x not in z]
    ev = sum(float(R[x]["metrics"].get("offroad_or_collision_at_fault", 0) or 0) for x in scenes if R[x].get("metrics"))
    return dict(n=len(scenes), mean=float(s.mean()), n1=int((s >= 1).sum()), nz=len(z), zk=zk, zero=z, slow=slow,
                collision=sum("collision_at_fault" in zk[x] for x in scenes), offroad=sum("offroad" in zk[x] for x in scenes),
                corridor=sum("left_corridor_laterally" in zk[x] for x in scenes),
                zero_other=sum(1 for x in z if not zk[x]), events=ev,
                failed=sum(1 for x in scenes if R[x].get("failure_reason")),
                prog=float(np.mean([R[x]["score_metrics"].get("progress_clipped_rel", 0) for x in scenes])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sh30"), ap.add_argument("--ap2a"), ap.add_argument("--ap2b")
    ap.add_argument("--out"), ap.add_argument("--json")
    a = ap.parse_args()
    R = {"SH30-F-s0": load(a.sh30), "AP2-AB-s0 run 1": load(a.ap2a), "AP2-AB-s0 run 2": load(a.ap2b)}
    scenes = sorted(set.intersection(*(set(r) for r in R.values())))
    S = {k: stats(r, scenes) for k, r in R.items()}
    sc = {k: np.array([r[x]["score"] for x in scenes]) for k, r in R.items()}
    L = [f"Scenes common to all three runs: {len(scenes)}.", "",
         "| run | scenes | mean scene score | score 1 | score 0 | at-fault collision | offroad | left corridor | zero, other | slow (0 < score < 1) | at-fault events | mean progress | rollouts with failure_reason (= the zeros) |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, s in S.items():
        L.append(f"| {k} | {s['n']} | {s['mean']:.4f} | {s['n1']} | {s['nz']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['zero_other']} | "
                 f"{len(s['slow'])} | {s['events']:.0f} | {s['prog']:.3f} | {s['failed']} |")
    out = dict(stats={k: {kk: vv for kk, vv in s.items() if kk not in ("zk",)} for k, s in S.items()}, scenes=len(scenes))
    # AP2 repeat
    d = sc["AP2-AB-s0 run 2"] - sc["AP2-AB-s0 run 1"]
    ch = [(x, sc["AP2-AB-s0 run 1"][i], sc["AP2-AB-s0 run 2"][i]) for i, x in enumerate(scenes) if abs(d[i]) > 1e-9]
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    L += ["", f"AP2 repeat: mean {sc['AP2-AB-s0 run 1'].mean():.4f} vs {sc['AP2-AB-s0 run 2'].mean():.4f}, difference {d.mean():+.4f} "
          f"(paired s.e. {se:.4f}); {len(ch)} of {len(scenes)} scenes differ; mean |diff| {np.abs(d).mean():.4f}; "
          f"zero in run 1 only {sum(1 for x, a1, b1 in ch if a1 == 0 and b1 > 0)}, in run 2 only {sum(1 for x, a1, b1 in ch if b1 == 0 and a1 > 0)}, "
          f"both {sum(1 for i in range(len(scenes)) if sc['AP2-AB-s0 run 1'][i] == 0 and sc['AP2-AB-s0 run 2'][i] == 0)}.", "",
          "| scene | AP2 run 1 | AP2 run 2 | zero reasons run 1 / run 2 |", "|:--|--:|--:|:--|"]
    for x, a1, b1 in sorted(ch, key=lambda t: -abs(t[2] - t[1]))[:40]:
        L.append(f"| {x} | {a1:.3f} | {b1:.3f} | {','.join(S['AP2-AB-s0 run 1']['zk'][x]) or '-'} / {','.join(S['AP2-AB-s0 run 2']['zk'][x]) or '-'} |")
    # best-of-two
    ap_mean = (sc["AP2-AB-s0 run 1"] + sc["AP2-AB-s0 run 2"]) / 2
    ap1, ap2, sh = sc["AP2-AB-s0 run 1"], sc["AP2-AB-s0 run 2"], sc["SH30-F-s0"]
    best = {"SH30": sh.mean(), "AP2 run 1": ap1.mean(), "AP2 run 2": ap2.mean(), "AP2 mean of runs": ap_mean.mean()}
    single = max(sh.mean(), ap_mean.mean())
    orc = {"SH30 vs AP2 run 1": np.maximum(sh, ap1).mean(), "SH30 vs AP2 run 2": np.maximum(sh, ap2).mean(),
           "SH30 vs AP2 (mean of runs)": np.maximum(sh, ap_mean).mean(), "AP2 run 1 vs AP2 run 2 (same driver)": np.maximum(ap1, ap2).mean()}
    L += ["", "Best-of-two (per scene maximum, an oracle selector):", "", "| pair | oracle mean | oracle minus best single driver (by mean) |", "|:--|--:|--:|"]
    for k, v in orc.items():
        base = max(sh.mean(), ap1.mean()) if k == "SH30 vs AP2 run 1" else max(sh.mean(), ap2.mean()) if k == "SH30 vs AP2 run 2" else \
            single if "mean of runs" in k else max(ap1.mean(), ap2.mean())
        L.append(f"| {k} | {v:.4f} | {v - base:+.4f} |")
    gain = orc["SH30 vs AP2 (mean of runs)"] - single
    noise = orc["AP2 run 1 vs AP2 run 2 (same driver)"] - max(ap1.mean(), ap2.mean())
    net = float(np.maximum(sh, ap1).mean() - max(sh.mean(), ap1.mean()) - noise)
    L += ["", f"Same-driver oracle (AP2 run 1 vs run 2) gains {noise:+.4f}: that is what a selector gets from run-to-run noise alone. "
          f"Cross-driver oracle (SH30 vs AP2 run 1) minus best single, net of that noise: {net:+.4f}. C2 line: >= 0.015 -> "
          f"{'MET' if net >= 0.015 else 'NOT MET'}."]
    both0 = [x for i, x in enumerate(scenes) if sh[i] == 0 and ap1[i] == 0 and ap2[i] == 0]
    only = lambda a_, b_: [x for i, x in enumerate(scenes) if a_[i] == 0 and b_[i] > 0]
    L += ["", f"Zero in all three runs: {len(both0)}; zero in SH30 only (AP2 both runs > 0): {len([x for i, x in enumerate(scenes) if sh[i] == 0 and ap1[i] > 0 and ap2[i] > 0])}; "
          f"zero in AP2 both runs only (SH30 > 0): {len([x for i, x in enumerate(scenes) if sh[i] > 0 and ap1[i] == 0 and ap2[i] == 0])}."]
    L += ["", "Zero-score scenes:", "", "| scene | SH30 | AP2 run 1 | AP2 run 2 |", "|:--|:--|:--|:--|"]
    for x in scenes:
        zs = [S[k]["zk"][x] if S[k]["zk"][x] else ("zero" if R[k][x]["score"] == 0 else None) for k in R]
        if any(zs):
            L.append(f"| {x} | " + " | ".join((",".join(z) if isinstance(z, list) else z) if z else f"{R[k][x]['score']:.2f}" for z, k in zip(zs, R)) + " |")
    out.update(oracle={k: float(v) for k, v in orc.items()}, noise_same_driver=float(noise), net_gain=net, repeat_diff=float(d.mean()), repeat_se=se,
               repeat_changed=len(ch), c2_met=bool(net >= 0.015), best=best)
    if a.out:
        Path(a.out).write_text("\n".join(L) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
