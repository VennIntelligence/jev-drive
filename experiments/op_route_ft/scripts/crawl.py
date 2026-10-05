"""Attribute the stopped time in the B2D 25-turn windows (decision 128 point 5) to one cause per stopped interval.

    .venv/bin/python experiments/op_route_ft/scripts/crawl.py DUMP.json --out experiments/op_route_ft/results/crawl [--figs experiments/op_route_ft/figs]
    DUMP.json from crawl_dump.py. -> <out>.json (intervals, counts) and the figures.

Window = entry .. +15 s of each entered turn (as rft_split.py); warm-up ticks (5 s static start, reason no_trajectory) are reported apart.
A stopped interval is a run of v < 0.3 m/s of >= 1 s. Cause, first match over the interval's ticks (> 50% of them):
  warm-up   the harness's 5 s static start
  blocker   ground-truth actor in the way: lead gap < 10 m at v < 1.5 m/s, pedestrian within 8 m, or any actor within 8 m ahead and 2 m aside
  red light ground-truth light red / yellow within -5 .. 30 m of the stop line (the harness ignores lights; the model sees them)
  model     the model's own plan: planned speed at 3 s < 1 m/s (plan head), binding source plan / latch / lead
  harness   the plan wants to go (planned speed at 3 s >= 1 m/s) and the car is still stopped: latch hold, a ghost lead, the base profile, launch ramp
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

V_STOP, MIN_S, DT = 0.3, 1.0, 0.05


def blocker(t):
    g, lv = t.get("lead_gap"), t.get("lead_v")
    if g is not None and g < 10 and (lv or 0) < 1.5:
        return True
    if t.get("ped_gap") is not None and t["ped_gap"] < 8:
        return True
    return any(0 < n[2] < 8 and abs(n[3]) < 2 for n in (t.get("near") or []))


def red(t):
    return t.get("tl") in (1, 2) and t.get("tl_dist") is not None and -5 < t["tl_dist"] < 30


def is_warm(t):
    return bool(t.get("warm")) or t.get("reason") == "no_trajectory"


def want_go(t):
    vp = t.get("vplan")
    return vp is not None and vp[3] >= 1.0


def intervals(r):
    """Stopped intervals (>= MIN_S) inside the 15 s window: [(i0, i1)] over r['ticks'][win_i:]."""
    tk = r["ticks"][r["win_i"]:]
    out, i = [], 0
    while i < len(tk):
        if tk[i]["v"] < V_STOP:
            j = i
            while j + 1 < len(tk) and tk[j + 1]["v"] < V_STOP:
                j += 1
            if (j - i + 1) * DT >= MIN_S:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return tk, out


def cause_of(seg):
    n = len(seg)
    f = lambda fn: sum(bool(fn(t)) for t in seg) / n  # noqa: E731
    if f(is_warm) > 0.5:
        return "warm-up"
    if f(blocker) > 0.5:
        return "blocker"
    if f(red) > 0.5:
        return "red light"
    if f(lambda t: not want_go(t)) > 0.5:
        return "model"
    return "harness"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.dump))
    res, summ = [], {}
    for arm in dict.fromkeys(r["arm"] for r in D):
        R = [r for r in D if r["arm"] == arm and r["entered"]]
        S = collections.Counter()
        tot = stop = warm_stop = 0
        vm, vplan1, nstop_iv = [], [], 0
        for r in R:
            tk, iv = intervals(r)
            tot += len(tk)
            stop += sum(t["v"] < V_STOP for t in tk)
            warm_stop += sum(t["v"] < V_STOP and is_warm(t) for t in tk)
            nm = [t for t in tk if not is_warm(t)]
            vm += [t["v"] for t in nm]
            vplan1 += [t["vplan"][1] for t in nm if t.get("vplan")]
            for i, j in iv:
                seg = tk[i:j + 1]
                c = cause_of(seg)
                S[c] += len(seg) * DT
                srcs = collections.Counter("latch" if t["latch"] else t["src"] for t in seg)
                res.append(dict(arm=arm, route=r["route"], turn=r["turn"], t0=seg[0]["t"], dur=round(len(seg) * DT, 1), cause=c,
                                src=dict(srcs), plan_go=round(float(np.mean([want_go(t) for t in seg])), 2),
                                act_a_med=round(float(np.median([t["act_a"] for t in seg if t["act_a"] is not None] or [np.nan])), 2),
                                v_before=round(float(tk[i - 1]["v"]), 1) if i else None, dev=seg[0]["dev"], junc=seg[0]["junc"], took=r["took"]))
        summ[arm] = dict(n_turns=len(R), ticks=tot, stopped_share=round(stop / max(tot, 1), 3), warm_stopped_share=round(warm_stop / max(tot, 1), 3),
                         v_med_all=round(float(np.median([t["v"] for r in R for t in r["ticks"][r["win_i"]:]])), 2),
                         v_med_nowarm=round(float(np.median(vm)), 2), vplan1_med_nowarm=round(float(np.median(vplan1)), 2),
                         stopped_s_by_cause={k: round(v / len(R), 2) for k, v in S.items()})
    Path(a.out + ".json").write_text(json.dumps(dict(summary=summ, intervals=res), indent=1))
    for k, v in summ.items():
        print(k, v)


if __name__ == "__main__":
    main()
