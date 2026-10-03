#!/usr/bin/env python
"""Real-car launch readouts from real_launch_replay.py output + comma1M localizer.  Prints a Markdown block.

  python real_launch_report.py <events.json> <replay_dir> [--out summary.json]
c   = heading change over the next 0.25 s (localizer yaw rate, left +) per degree of the plan's direction at 0.8 s model time
      (phi1, the HUGSIM 1 s point at DIL 1.25), through the origin, by speed bin; cluster bootstrap over events.
lean = |phi1| at launch steps m = 1..4 (frame onset + 5 (m - 1)).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.openpilot.frames import load_segment_meta  # noqa: E402
from jevdrive.openpilot.model import T_IDXS  # noqa: E402

ROOT = Path.home() / "data/datasets/comma1M"
RNG = np.random.default_rng(0)


def phi(pos, tau):
    x, y = np.interp(tau, T_IDXS, pos[:, 0]), np.interp(tau, T_IDXS, pos[:, 1])
    return -np.degrees(np.arctan2(y, np.maximum(x, 1e-3)))


def phi_series(P, tau):
    return np.array([phi(p, tau) for p in P])


def boot(vals_by_event, fn, n=2000):
    """cluster bootstrap: vals_by_event = list of arrays (one per event)."""
    k = len(vals_by_event)
    est = fn(np.concatenate(vals_by_event)) if k else np.nan
    bs = []
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        c = [vals_by_event[i] for i in idx if len(vals_by_event[i])]
        bs.append(fn(np.concatenate(c)) if c else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi


def fmt(t):
    return f"{t[0]:.3g} [{t[1]:.3g}, {t[2]:.3g}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("rep")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ev = json.load(open(a.events))
    metas, E = {}, []
    for f in sorted(Path(a.rep).glob("*.npz")):
        sid, j = f.stem.rsplit("_", 1)
        z = np.load(f)
        if sid not in metas:
            metas[sid] = load_segment_meta(ROOT / sid)
        m = metas[sid]
        lo, j = int(z["lo"]), int(z["onset"])
        v = np.linalg.norm(m["vel"], axis=1)
        yr = -np.degrees(m["omega_dev"][:, 2])                      # left +, deg / s
        t = m["t_loc"]
        E.append(dict(sid=sid, j=j, lo=lo, v=v, yr=yr, t=t, ph={k: {"08": phi_series(z[k], 0.8), "10": phi_series(z[k], 1.0)} for k in ("real", "frozen")},
                      static_s=[e["static_s"] for e in ev[sid] if e["onset"] == j][0]))
    print(f"events {len(E)} in {len(metas)} segments\n")
    out = {"n_events": len(E)}

    # ---- sign check: model plan direction vs vehicle yaw rate over the launch windows
    xs = np.concatenate([E_["ph"]["real"]["08"][E_["j"] - E_["lo"]: E_["j"] - E_["lo"] + 60] for E_ in E])
    ys = np.concatenate([E_["yr"][E_["j"]: E_["j"] + 60][: 60] for E_ in E if len(E_["yr"]) >= E_["j"] + 60])
    print(f"yaw-rate sign check: corr(phi1_08, yaw_rate_left+) over 3 s after onset = "
          f"{np.corrcoef(xs[: len(ys)], ys)[0, 1] if len(ys) == len(xs) else float('nan'):.2f}\n")

    # ---- (2) launch lean
    print("## launch lean |phi1| (deg), steps m = 1..4 at onset + 5(m-1) frames; median [cluster CI], mean, share >= 1 deg\n")
    print("| cond | m | n | median | mean | >=1 deg |\n|---|---|---|---|---|---|")
    lean = {}
    for c in ("real", "frozen"):
        for mm in (1, 2, 3, 4, "1-2"):
            ms = (1, 2) if mm == "1-2" else (mm,)
            vals = [np.array([abs(e["ph"][c]["08"][e["j"] - e["lo"] + 5 * (q - 1)]) for q in ms]) for e in E]
            med, mean, sh = boot(vals, np.median), boot(vals, np.mean), boot(vals, lambda x: np.mean(x >= 1))
            lean[f"{c}_{mm}"] = dict(median=med, mean=mean, share=sh)
            print(f"| {c} | {mm} | {sum(len(v) for v in vals)} | {fmt(med)} | {fmt(mean)} | {fmt(sh)} |")
    out["lean"] = lean
    d = [np.array([abs(e["ph"]["real"]["08"][e["j"] - e["lo"] + 5 * (q - 1)]) - abs(e["ph"]["frozen"]["08"][e["j"] - e["lo"] + 5 * (q - 1)]) for q in (1, 2)]) for e in E]
    print(f"\nreal - frozen history, |phi1| steps 1-2, paired: median {fmt(boot(d, np.median))}, mean {fmt(boot(d, np.mean))}")
    dp = [np.array([e["ph"]["real"]["08"][e["j"] - e["lo"] + 5 * (q - 1)] - e["ph"]["frozen"]["08"][e["j"] - e["lo"] + 5 * (q - 1)] for q in (1, 2)]) for e in E]
    print(f"signed phi1 real - frozen, steps 1-2: rms {np.sqrt(np.mean(np.concatenate(dp) ** 2)):.2f} deg, median |diff| {np.median(np.abs(np.concatenate(dp))):.2f} deg")
    out["real_minus_frozen_abs"] = [fmt(boot(d, np.median)), fmt(boot(d, np.mean))]

    # ---- (1) c
    print("\n## c: heading change over the next 0.25 s per deg of plan direction (phi1), through the origin\n")
    print("| source | cond | bin | steps | c [cluster CI] |\n|---|---|---|---|---|")
    bins = [(0, 1), (1, 2), (2, 3), (0, 3)]
    cres = {}
    for c in ("real", "frozen"):
        for b in bins:
            xs_ev, ys_ev = [], []
            for e in E:
                X, Y = [], []
                for k in range(0, 40):                           # 0.25 s steps from onset until 10 s... capped by window
                    i = e["j"] + 5 * k
                    if i + 5 >= e["lo"] + len(e["ph"][c]["08"]) or i + 5 >= len(e["yr"]):
                        break
                    vk = e["v"][i]
                    if not (b[0] <= vk < b[1]):
                        continue
                    dth = np.trapezoid(e["yr"][i:i + 6], e["t"][i:i + 6])
                    p = e["ph"][c]["08"][i - e["lo"]]
                    if abs(p) < 15:
                        X.append(p), Y.append(dth)
                xs_ev.append(np.array([X, Y]).T if X else np.zeros((0, 2)))
            fn = lambda A: float((A[:, 0] * A[:, 1]).sum() / max((A[:, 0] ** 2).sum(), 1e-9))   # noqa: E731
            ok = [x for x in xs_ev if len(x)]
            if not ok:
                continue
            r = boot(ok, fn)
            cres[f"{c}_{b}"] = r
            print(f"| comma1M launches | {c} | {b[0]}-{b[1]} m/s | {sum(len(x) for x in ok)} | {fmt(r)} |")
    out["c"] = cres

    # ---- (3) history content before onset
    print("\n## history before the onset (real stream), localizer motion\n")
    def stat(fn, lo_s, hi_s):
        return np.array([fn(e, lo_s, hi_s) for e in E])
    win = lambda e, a_, b_: slice(e["j"] - int(b_ * 20), e["j"] - int(a_ * 20))   # noqa: E731
    sd = stat(lambda e, a_, b_: np.std(e["yr"][win(e, a_, b_)]), 0, 2)
    net = stat(lambda e, a_, b_: abs(np.trapezoid(e["yr"][win(e, a_, b_)], e["t"][win(e, a_, b_)])), 0, 2)
    mx = stat(lambda e, a_, b_: np.max(np.abs(e["yr"][win(e, a_, b_)] - np.mean(e["yr"][win(e, a_, b_)]))), 0, 2)
    vmax = stat(lambda e, a_, b_: np.max(e["v"][win(e, a_, b_)]), 0, 2)
    print(f"2 s before onset: yaw-rate std median {np.median(sd):.3f} deg/s (p90 {np.percentile(sd, 90):.3f}); net |heading change| median {np.median(net):.3f} deg (p90 {np.percentile(net, 90):.3f}); "
          f"max |yaw rate - mean| median {np.median(mx):.3f} deg/s; max speed in the window median {np.median(vmax):.3f} m/s, share > 0.05 m/s {np.mean(vmax > 0.05):.2f}")
    print(f"static before onset (s): median {np.median([e['static_s'] for e in E]):.1f}, min {min(e['static_s'] for e in E):.1f}")
    # model output while still static, real vs frozen history
    dstat = np.concatenate([e["ph"]["real"]["08"][max(e["j"] - e["lo"] - 40, 0): e["j"] - e["lo"]] - e["ph"]["frozen"]["08"][max(e["j"] - e["lo"] - 40, 0): e["j"] - e["lo"]] for e in E])
    sstat = np.concatenate([e["ph"]["real"]["08"][max(e["j"] - e["lo"] - 40, 0): e["j"] - e["lo"]] for e in E])
    print(f"phi1 while static (last 2 s before onset), real history: |phi1| median {np.median(np.abs(sstat)):.2f} deg; real - frozen: rms {np.sqrt(np.mean(dstat ** 2)):.2f} deg (frozen = identical frame repeated, HUGSIM warm-up)")
    out["hist"] = dict(yr_std_med=float(np.median(sd)), net_med=float(np.median(net)), vmax_share_gt_005=float(np.mean(vmax > 0.05)))

    # ---- (4) car behaviour right after onset
    print("\n## the real car right after the onset\n")
    for T in (1, 2, 3):
        i = [min(e["j"] + 20 * T, len(e["v"]) - 1) for e in E]
        dth = [abs(np.trapezoid(e["yr"][e["j"]: ii + 1], e["t"][e["j"]: ii + 1])) for e, ii in zip(E, i)]
        print(f"+{T} s: v median {np.median([e['v'][ii] for e, ii in zip(E, i)]):.2f} m/s; |net heading change| median {np.median(dth):.2f} deg, p90 {np.percentile(dth, 90):.2f}, max {np.max(dth):.1f}")

    # ---- realised loop: phi1 on actual heading change in the previous 1.25 s (decision 100 kernel s_gain)
    print("\n## realised kernel: phi1(t) on real heading change over the previous 1.25 s (0.25 s steps m = 1..4 and 5..12), through the origin\n")
    for c in ("real", "frozen"):
        for lab, ks in (("m 1-4", range(0, 4)), ("m 5-12", range(4, 12))):
            A = []
            for e in E:
                X = []
                for k in ks:
                    i = e["j"] + 5 * k
                    if i - e["lo"] >= len(e["ph"][c]["08"]) or i >= len(e["yr"]) or i < 25:
                        continue
                    th = np.trapezoid(e["yr"][i - 25:i + 1], e["t"][i - 25:i + 1])
                    X.append([th, e["ph"][c]["08"][i - e["lo"]]])
                A.append(np.array(X) if X else np.zeros((0, 2)))
            A = [x for x in A if len(x)]
            fn = lambda M: float((M[:, 0] * M[:, 1]).sum() / max((M[:, 0] ** 2).sum(), 1e-9))   # noqa: E731
            print(f"- {c}, {lab}: slope phi1 per deg heading change {fmt(boot(A, fn))} (n {sum(len(x) for x in A)})")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
