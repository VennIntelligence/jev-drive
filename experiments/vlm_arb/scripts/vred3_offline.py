"""Offline readings that fix the parameters of the vred3 yellow rule (plan 2026-10-02-pbyp2-vred2.md section 11). No CARLA, no GPU.

  python vred3_offline.py            -> results/vred3_yellow_offline.md (+ yellow_durations.csv, yellow_onsets.csv)

Reads the logged Qwen3-VL-4B answers with the simulator's light state of the shadow runs (`cal`, `cal2`, `cal3`: shadow `drive`, the car drives
as drive, seed 1, 19 routes, three repeats) and of the in-loop runs `vred` (decision 86). Each answer line carries the truth of the same frame:
ego light state `tl` (0 green, 1 yellow, 2 red), `tl_dist`, `tl_id`, and the state of every light within 60 m.

1. Yellow duration of every light, from complete green -> yellow -> red sequences (samples every 0.5 s).
2. Option (b) of the brief, inferring yellow from "red_or_yellow answer right after a green answer": for every true green -> yellow onset of the
   ego light within 50 m, was the first yellow frame answered non-green (detection lag in frames), how old is the yellow when the first non-green
   answer arrives (truth) against the age the rule assumes (time since the last green answer's frame), i.e. is the assumption conservative.
   Option (a), a yellow option in the question, needs a changed prompt and server and a new Phase A; the number of yellow frames available is
   reported so that it is clear why it was not run.
3. The cost of reacting to the first non-green answer: green -> non-green answer transitions while the truth is green, single (next answer
   green again) or persistent.
4. What the rule would have decided at the yellow encounters of the vred runs (hypothetical, from the logged speed and distances).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "lib"))
import yellow_rule  # noqa: E402
from vlm_arb_common import RUN, jsonl  # noqa: E402

GREEN, RED = "green_for_ego", "red_or_yellow_for_ego"
SHADOW = ("v2-cal-s1-q0", "v2-cal-s1-q1", "v2-cal-s1-q2", "v2-cal2-s1-q0", "v2-cal2-s1-q1", "v2-cal2-s1-q2", "v2-cal3-s1-q0", "v2-cal3-s1-q1", "v2-cal3-s1-q2")
INLOOP = tuple("v2-vred-s%d-q%d" % (s, k) for s in (0, 1) for k in range(3))
OUT = RUN / "v3/results"


def attempts(units):
    for u in units:
        d = RUN / "arms" / u
        for done in sorted(d.glob("done/*.json")):
            yield u, done.stem, d / "attempts" / done.stem / str(json.loads(done.read_text()).get("attempt", 1))


def answers(a):
    rows = [r for r in jsonl(a / "vlm_decisions.jsonl") if r.get("k") == "a" and r["ans"].get("ok")]
    return sorted(rows, key=lambda r: r["t_q"])


def durations():
    """Yellow durations of every light: green sample, then consecutive yellow samples, then red sample, spacing <= 0.6 s."""
    ser = collections.defaultdict(list)
    for u, rid, a in attempts(SHADOW + INLOOP):
        for r in answers(a):
            for lid, st, dist in r["gt"].get("lights", []):
                ser[(rid, lid, u)].append((r["t_q"], st))
    out = []
    for (rid, lid, u), v in ser.items():
        v.sort()
        i = 0
        while i < len(v):
            if v[i][1] == 1:
                j = i
                while j + 1 < len(v) and v[j + 1][1] == 1 and v[j + 1][0] - v[j][0] < 0.6:
                    j += 1
                if i > 0 and j + 1 < len(v) and v[i - 1][1] == 0 and v[j + 1][1] == 2 and v[i][0] - v[i - 1][0] < 0.6 and v[j + 1][0] - v[j][0] < 0.6:
                    # onset in (t_{i-1}, t_i], red onset in (t_j, t_{j+1}]: the yellow lasted between t_j - t_i and t_{j+1} - t_{i-1}
                    out.append(dict(route=rid, light=lid, unit=u, t=v[i][0], lo=v[j][0] - v[i][0], hi=v[j + 1][0] - v[i - 1][0], n_yellow=j - i + 1))
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(out)


def onsets():
    """True green -> yellow onsets of the ego light within 50 m with the answer sequence around them."""
    out, flick = [], []
    green_n = green_n_ans = approaches = 0
    for u, rid, a in attempts(SHADOW):
        rows = answers(a)
        seen_green_ids = set()
        for i in range(1, len(rows)):
            g0, g1, a1 = rows[i - 1]["gt"], rows[i]["gt"], rows[i]["ans"]["Q_light"]
            t0, t1 = rows[i - 1]["t_q"], rows[i]["t_q"]
            near = lambda g: g.get("tl") is not None and g.get("tl_dist") is not None and -5 <= g["tl_dist"] < 50   # noqa: E731
            if near(g1) and g1["tl"] == 0:
                green_n += 1
                if g1.get("tl_id") not in seen_green_ids:
                    seen_green_ids.add(g1.get("tl_id"))
                    approaches += 1
                # a green -> non-green answer transition while the truth is green
                if rows[i - 1]["ans"]["Q_light"] == GREEN and a1 == RED:
                    nxt = rows[i + 1]["ans"]["Q_light"] if i + 1 < len(rows) else None
                    flick.append(dict(unit=u, route=rid, t=t1, single=(nxt == GREEN), nxt=nxt, tl_dist=g1["tl_dist"]))
            # true onset: previous sample green, this sample yellow, same ego light, consecutive samples
            if near(g0) and near(g1) and g0["tl"] == 0 and g1["tl"] == 1 and g0.get("tl_id") == g1.get("tl_id") and t1 - t0 < 0.6:
                j = i
                while j < len(rows) and rows[j]["gt"].get("tl") == 1 and rows[j]["gt"].get("tl_id") == g1.get("tl_id") and rows[j]["ans"]["Q_light"] != RED:
                    j += 1                                                  # first yellow frame answered non-green
                if j >= len(rows) or rows[j]["gt"].get("tl") != 1 or rows[j]["gt"].get("tl_id") != g1.get("tl_id"):
                    out.append(dict(unit=u, route=rid, t_onset_lo=t0, t_onset_hi=t1, detected=False, lag_frames=np.nan, tl_dist=g1["tl_dist"]))
                    continue
                tj = rows[j]["t_q"]
                k = j - 1
                while k >= 0 and rows[k]["ans"]["Q_light"] != GREEN:
                    k -= 1
                last_green = rows[k]["t_q"] if k >= 0 else np.nan
                age_assumed = tj - last_green                                  # frame age the rule assumes at the first non-green frame
                age_true = tj - 0.5 * (t0 + t1)                                # truth: the onset is somewhere in (t0, t1]
                out.append(dict(unit=u, route=rid, t_onset_lo=t0, t_onset_hi=t1, detected=True, lag_frames=j - i, tl_dist=g1["tl_dist"], age_assumed=age_assumed,
                                age_true_mid=age_true, age_true_max=tj - t0, age_true_min=tj - t1, deficit_mid=age_true - age_assumed,
                                deficit_max=(tj - t0) - age_assumed))
    return pd.DataFrame(out), pd.DataFrame(flick), green_n, approaches


def yellow_samples():
    n = hit = 0
    for u, rid, a in attempts(SHADOW):
        for r in answers(a):
            g = r["gt"]
            if g.get("tl") == 1 and g.get("tl_dist") is not None and -5 <= g["tl_dist"] < 50:
                n += 1
                hit += r["ans"]["Q_light"] == RED
    return n, hit


def hypothetical():
    """At the yellow encounters of the vred runs: what the rule decides with the logged speed and distances (not an outcome)."""
    out = []
    for u, rid, a in attempts(INLOOP):
        rows = answers(a)
        st = [r for r in jsonl(a / "vlm_decisions.jsonl") if r.get("k") == "s"]
        for i in range(1, len(rows)):
            g0, g1 = rows[i - 1]["gt"], rows[i]["gt"]
            if rows[i - 1]["ans"]["Q_light"] == GREEN and rows[i]["ans"]["Q_light"] == RED and g1.get("tl") == 1 and g1.get("tl_dist") is not None and 0 <= g1["tl_dist"] < 50:
                t = rows[i]["t_eff"]
                s = min(st, key=lambda r: abs(r["t"] - t))
                v = s["v"]
                d_stop = g1["tl_dist"] - 0.5
                d_line = s["junc_dist"] - 3.8394
                age = t - rows[i - 1]["t_q"]
                rem = yellow_rule.remaining_yellow(age)
                out.append(dict(unit=u, route=rid, t=round(t, 1), v=round(v, 2), d_stop_target=round(d_stop, 2), d_line=round(d_line, 2), age=round(age, 2), remaining=round(rem, 2),
                                decision=yellow_rule.decide(d_stop, d_line, v, rem, 8.0), stop_need=round(yellow_rule.stop_distance(v), 2)))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dur = durations()
    dur.to_csv(OUT / "yellow_durations.csv", index=False)
    ons, flick, green_n, approaches = onsets()
    ons.to_csv(OUT / "yellow_onsets.csv", index=False)
    ny, hy = yellow_samples()
    hyp = hypothetical()
    D = ["# vred3: offline readings that fix the yellow rule's parameters", "",
         "Plan: [plans/2026-10-02-pbyp2-vred2.md](../plans/2026-10-02-pbyp2-vred2.md) section 11. Logged answers of Qwen3-VL-4B (the `vred` serving) with the simulator's light state of the same frame: "
         "shadow runs `cal`, `cal2`, `cal3` (the car drives as `drive`; seed 1, 19 routes, three repeats of the same traffic seed, so repeats are not independent) and the in-loop `vred` runs. "
         "Code: `scripts/vred3_offline.py`.", ""]
    # 1
    d = dur
    D += ["## 1. Yellow duration", "", "Complete green -> yellow -> red sequences of every light in the logs (%d, over %d routes and %d lights; samples every 0.5 s, so each duration is known to within a bracket of 1.0 s)." % (
        len(d), d.route.nunique() if len(d) else 0, d.groupby(["route", "light"]).ngroups if len(d) else 0), ""]
    if len(d):
        mid = 0.5 * (d.lo + d.hi)
        D += ["Yellow samples per sequence (0.5 s each; a 3.0 s yellow gives 6, a 2.5 s one 5 or 6, a 3.5 s one 6 or 7): %s. Duration bracket [%.1f, %.1f] s over all sequences; share of "
              "sequences whose bracket contains 3.0 s: %.1f%%; contains 2.0 s: %.1f%%; contains 4.0 s: %.1f%%." % (
                  dict(d.n_yellow.value_counts().sort_index()), d.lo.min(), d.hi.max(), 100 * ((d.lo <= 3.0 + 1e-9) & (d.hi >= 3.0 - 1e-9)).mean(),
                  100 * ((d.lo <= 2.0 + 1e-9) & (d.hi >= 2.0 - 1e-9)).mean(), 100 * ((d.lo <= 4.0 + 1e-9) & (d.hi >= 4.0 - 1e-9)).mean()), "",
              d.groupby("route").agg(sequences=("t", "count"), n_yellow_min=("n_yellow", "min"), n_yellow_max=("n_yellow", "max")).reset_index().to_markdown(index=False), "",
              "Reading: the sequences that are complete and sampled without gaps give 6 yellow samples, a 3.0 s yellow; the registered `yellow_s` = 3.0 s. Sequences with other counts are "
              "listed in `yellow_durations.csv` (gaps in the sampling when the ego stopped answering or left the 60 m window).", ""]
    # 2
    D += ["## 2. Inferring the yellow from the answer sequence (option b)", "",
          "Truth frames of yellow (ego light within 50 m): %d; answered `red_or_yellow_for_ego`: %d (%.1f%%). Option (a), a yellow option in the question, would need a changed prompt, a changed "
          "server and a new Phase A on %d yellow frames: not run; the inference needs no model change." % (ny, hy, 100 * hy / max(ny, 1), ny), ""]
    if len(ons):
        det = ons[ons.detected]
        D += ["True green -> yellow onsets of the ego light within 50 m: %d (%d detected, i.e. a yellow frame was answered non-green before the yellow ended; %d never). Detection lag in frames "
              "(0 = the first yellow frame is answered non-green): %s." % (len(ons), len(det), int((~ons.detected).sum()), dict(det.lag_frames.value_counts().sort_index())), ""]
        if len(det):
            D += ["Every onset (`lag` = yellow frames answered green before the first non-green answer; deficit = true age with the earliest possible onset minus the age the rule assumes):", "",
                  det[["unit", "route", "t_onset_hi", "tl_dist", "lag_frames", "age_assumed", "age_true_min", "age_true_max", "deficit_max"]].round(2).to_markdown(index=False), ""]
            D += ["Age of the yellow at the first non-green answer: the rule assumes `age_assumed` = frame time of the first non-green answer minus the frame time of the last green answer (the "
                  "yellow may have started right after it). Truth: the onset lies between the last green sample and the first yellow sample. The deficit `true - assumed` is positive when the rule "
                  "assumes a younger yellow than the truth (not conservative): this happens when the model keeps answering green into the yellow.", "",
                  "| quantity | median | p90 | max |", "|:--|--:|--:|--:|",
                  "| age assumed [s] | %.2f | %.2f | %.2f |" % (det.age_assumed.median(), np.percentile(det.age_assumed, 90), det.age_assumed.max()),
                  "| true age, mid of the bracket [s] | %.2f | %.2f | %.2f |" % (det.age_true_mid.median(), np.percentile(det.age_true_mid, 90), det.age_true_mid.max()),
                  "| deficit, true (mid) - assumed [s] | %.2f | %.2f | %.2f |" % (det.deficit_mid.median(), np.percentile(det.deficit_mid, 90), det.deficit_mid.max()),
                  "| deficit, true (earliest onset) - assumed [s] | %.2f | %.2f | %.2f |" % (det.deficit_max.median(), np.percentile(det.deficit_max, 90), det.deficit_max.max()), "",
                  "The registered `lag_margin_s` = 0.5 s is subtracted from the remaining yellow time: it covers the p90 of the earliest-onset deficit (%.2f s) when that is <= 0.5 s." % np.percentile(det.deficit_max, 90), ""]
    # 3
    fl = flick
    fb = fl[fl.tl_dist > 0] if len(fl) else fl
    D += ["## 3. What reacting to the first non-green answer costs in false starts", "",
          "Green frames of the ego light within 50 m: %d in %d (run, light) approaches. Answer transitions green -> `red_or_yellow_for_ego` while the truth is green: %d (%.2f per 100 green "
          "frames, %.1f per 100 approaches); %d of them single (the next answer is green again: the second answer cancels the tentative stop), %d persistent. **Before the stop line "
          "(`tl_dist` > 0, where the rule can start a stop): %d transitions, %d single.** The others are after the front bumper has passed the line, where the commit rule starts nothing." % (
              green_n, approaches, len(fl), 100 * len(fl) / max(green_n, 1), 100 * len(fl) / max(approaches, 1), int(fl.single.sum()) if len(fl) else 0, int((~fl.single).sum()) if len(fl) else 0,
              len(fb), int(fb.single.sum()) if len(fb) else 0), ""]
    if len(fl):
        D += ["Distance of the ego to the stop line at these false transitions (m): median %.1f, p10 %.1f, min %.1f; %d within 15 m." % (
            fl.tl_dist.median(), np.percentile(fl.tl_dist, 10), fl.tl_dist.min(), int((fl.tl_dist < 15).sum())), "",
            "Cost model: a false first answer starts a tentative R2 hold (the IDM stop constraint) for one answer period (0.5 s plus the answer delay) before the second answer cancels it; "
            "a persistent one is the same as a K = 2 hold started 0.5 s earlier. The speed cost of a tentative hold is measured in the batch (`vred3.md`: tentative starts, cancellations).", ""]
    # 4
    D += ["## 4. What the rule would have decided at the yellow encounters of the vred runs (hypothetical)", "",
          "The logged speed, distance and answer times of the `vred` in-loop runs at each green -> non-green answer transition with a yellow truth frame. These cars braked under `vred`'s rules and "
          "had no R1 cap, so speeds differ from what `vred3` would drive; the table checks the decision function on real states, it is not an outcome.", "",
          hyp.to_markdown(index=False) if len(hyp) else "none", ""]
    (OUT / "vred3_yellow_offline.md").write_text("\n".join(D) + "\n")
    print("\n".join(D))


if __name__ == "__main__":
    main()
