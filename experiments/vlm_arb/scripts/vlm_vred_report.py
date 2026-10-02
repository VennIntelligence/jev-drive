"""Calibration, staged-launch gate and final report of the vred lane (plan 2026-10-02-vlm-vred.md).

  python vlm_vred_report.py calibrate     in-loop latency of the three shadow units `cal-s1-q*` -> gates/vred_cal.json
  python vlm_vred_report.py few           checklist over the first three batch units -> gates/vred_few.json
  python vlm_vred_report.py report        all tables and figures -> $DATA_DIR/runs/vlm_arb/vred/results/

Conventions of results/report.md: official infractions, paired differences per (route, seed) averaged over seeds, resampling
routes (2000 draws, seed 0), 95% percentile interval. Every number is a diagnostic read (19 routes); registered
confirmation lines are marked "not evaluated" and never "passed".
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vlm_arb_checks import route_checks, vlm_rows  # noqa: E402
from vlm_arb_common import (DATA, LIGHT_ROUTES, N_BOOT, BOOT_SEED, ROUTES, RUN, SEEDS, TGT, boot_ratio, drive_dir, fmt,  # noqa: E402
                            jsonl, route_row, unit_dir, write_json)
from vlm_arb_report import paired  # noqa: E402
from vlm_thin_common import RED, light_masks  # noqa: E402

OUT = RUN / "vred/results"
SHARDS = ("q0", "q1", "q2")
TTL, GRID = 2.5, 0.05
TARGETS = {"light routes": LIGHT_ROUTES, "other routes": [r for r in ROUTES if r not in LIGHT_ROUTES]}


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def attempts(arm, seed, shards=SHARDS):
    sh = json.loads((RUN / "gates/vred_shards.json").read_text())
    out = []
    for k in shards:
        for rid in sh[k]:
            d = unit_dir(arm, seed, k)
            r = route_row(d, rid)
            if r:
                out.append((rid, seed, k, r))
    return out


def answers(arm, seed):
    """One row per answered request of every route of the arm (seed): latency parts, answer, truth columns."""
    rows = []
    for rid, s, k, r in attempts(arm, seed):
        ans, st, head = vlm_rows(r["attempt"])
        for d in ans:
            g, a = d["gt"], d["ans"]
            tl = g.get("tl")
            oth = [x for x in g.get("lights", []) if x[0] != g.get("tl_id")]
            rows.append(dict(route=rid, seed=s, unit=k, L=head.get("L", np.nan), t_q=d["t_q"], t_eff=d["t_eff"], ok=bool(a.get("ok")),
                             lat=a.get("latency_ms", np.nan), rtt=a.get("rtt_ms", np.nan), q_ms=a.get("srv_queue_ms", np.nan),
                             svc=a.get("srv_svc_ms", np.nan), depth=a.get("srv_depth", np.nan), ans=a.get("Q_light", ""),
                             tl=-1 if tl is None else tl, tl_dist=g["tl_dist"] if tl is not None else np.nan,
                             other_red=float(any(x[1] == 2 for x in oth)), any_light=float(bool(g.get("lights"))),
                             other_green=float(any(x[1] == 0 for x in oth))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- calibration
def calibrate(_):
    df = pd.concat([answers("cal", 1)])
    if df.empty:
        raise SystemExit("no calibration answers")
    lat = df.lat[df.ok].to_numpy()
    L = float(np.ceil(pct(lat, 95) / 50.0) * GRID)
    per = {u: dict(n=int(len(g)), p50=pct(g.lat[g.ok], 50), p95=pct(g.lat[g.ok], 95), q95=pct(g.q_ms[g.ok], 95), depth_max=float(g.depth.max()))
           for u, g in df.groupby("unit")}
    gate = dict(done=True, workers_per_card=3, n=int(len(df)), ok_rate=float(df.ok.mean()), p50_ms=pct(lat, 50), p95_ms=pct(lat, 95),
                p99_ms=pct(lat, 99), max_ms=float(lat.max()), over_0p5=float(np.mean(lat > 500)), over_ttl=float(np.mean(lat > 2500)),
                queue_p95_ms=pct(df.q_ms[df.ok], 95), svc_p50_ms=pct(df.svc[df.ok], 50), L_s=L, line_ok=bool(pct(lat, 95) <= 600.0), per_unit=per)
    write_json(RUN / "gates/vred_cal.json", gate)
    txt = ("Calibration (shadow `drive`, Qwen3-VL-4B servers, 3 workers per card on 3 cards, seed 1, 19 routes): %d requests, %.1f%% answered. "
           "In-loop latency (submit to answer, JPEG encoding included): p50 %.0f ms, p95 %.0f ms, p99 %.0f ms, max %.0f ms; server queue wait p95 %.0f ms, "
           "service p50 %.0f ms. Share > 0.5 s %.1f%%, share > TTL 2.5 s %.1f%%. L = ceil(p95 / 50 ms) x 0.05 s = %.2f s; registered line "
           "(p95 <= 600 ms): %s.\n" % (len(df), 100 * gate["ok_rate"], gate["p50_ms"], gate["p95_ms"], gate["p99_ms"], gate["max_ms"], gate["queue_p95_ms"],
                                       gate["svc_p50_ms"], 100 * gate["over_0p5"], 100 * gate["over_ttl"], L, "yes" if gate["line_ok"] else "NO"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.md").write_text(txt + "\nper unit: " + json.dumps(per) + "\n")
    print(txt, json.dumps(per, indent=1))


# ---------------------------------------------------------------------------------------------------- staged gate
def few(_):
    """The checklist over the first three batch units (seed 0): a stop driven by a VLM answer, a release after green, no crash."""
    rows, stops, rolls, crashes, r5 = [], 0, 0, 0, 0
    for rid, s, k, r in attempts("vred", 0):
        c, o = route_checks(r["attempt"], "red_stop")[1], route_checks(r["attempt"], "")[0]
        stops += bool(c["stop_before_line"])
        rolls += bool(c["rolls_after_release"])
        crashes += bool(r["crash"])
        r5 += o.get("n_R5", 0) > 0
        rows.append(dict(route=rid, unit=k, n_R2=o.get("n_R2", 0), n_R5=o.get("n_R5", 0), stop_before_line=bool(c["stop_before_line"]),
                         rolls_after_release=bool(c["rolls_after_release"]), crash=bool(r["crash"]), DS=r["DS"]))
    an = answers("vred", 0)
    lat = an.lat[an.ok].to_numpy()
    lat_ok = bool(len(lat) and pct(lat, 95) <= 600.0 and np.mean(lat > 1e3 * TTL) <= 0.01 and an.ok.mean() >= 0.98)
    gate = dict(passed=bool(stops >= 1 and rolls >= 1 and crashes == 0 and lat_ok), latency_ok=lat_ok, p50_ms=pct(lat, 50), p95_ms=pct(lat, 95),
                over_ttl=float(np.mean(lat > 1e3 * TTL)) if len(lat) else None, routes_with_stop=stops, routes_with_release=rolls, crashes=crashes,
                routes_with_R5=r5, routes=rows)
    write_json(RUN / "gates/vred_few.json", gate)
    print(json.dumps(gate, indent=1))


# ---------------------------------------------------------------------------------------------------- report
def table(df, cols=None):
    return df.to_markdown(index=False) if cols is None else df[cols].to_markdown(index=False)


def runs():
    rows = []
    sh = json.loads((RUN / "gates/vred_shards.json").read_text())
    for arm in ("drive", "pred", "vred"):
        for s in SEEDS:
            for rid in ROUTES:
                if arm == "drive":
                    d = drive_dir(rid, s)
                elif arm == "pred":
                    d = unit_dir("pred", s, "tgt" if rid in TGT else "dev")
                else:
                    d = unit_dir("vred", s, next(k for k in SHARDS if rid in sh[k]))
                r = route_row(d, rid)
                if r:
                    rows.append(dict(arm=arm, seed=s, **r))
    return pd.DataFrame(rows)


def boot_ret(df, num, den, routes, col, n_boot=N_BOOT, seed=BOOT_SEED):
    """(num - drive) / (den - drive) as ratio of route means, route-cluster bootstrap; col is a per-run column."""
    piv = {a: df[(df.arm == a) & ~df.crash].groupby("route")[col].mean() for a in ("drive", num, den)}
    idx = [r for r in routes if all(r in piv[a].index for a in piv)]
    if not idx:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0)
    n_, d_ = np.array([piv[num][r] - piv["drive"][r] for r in idx]), np.array([piv[den][r] - piv["drive"][r] for r in idx])
    ix = np.random.default_rng(seed).integers(0, len(idx), (n_boot, len(idx)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = n_[ix].mean(1) / d_[ix].mean(1)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(est=float(n_.mean() / d_.mean()) if abs(d_.mean()) > 1e-9 else np.nan, lo=float(lo), hi=float(hi), n=len(idx),
                num=float(n_.mean()), den=float(d_.mean()))


def fmt_ret(r):
    return "n/a" if not r["n"] or not np.isfinite(r["est"]) else "%.2f [%.2f, %.2f] (%+.2f / %+.2f, %d routes)" % (r["est"], r["lo"], r["hi"], r["num"], r["den"], r["n"])


def episodes(st, key):
    """Rising edges of a rule in the 0.5 s state rows: [(t_start, t_end)]."""
    out, cur = [], None
    for r in st:
        on = key in r.get("rules", []) if key != "r5" else bool(r.get("r5"))
        if on and cur is None:
            cur = r["t"]
        if not on and cur is not None:
            out.append((cur, r["t"]))
            cur = None
    if cur is not None:
        out.append((cur, st[-1]["t"]))
    return out


def green_release(ans, st):
    """Per ego-light red -> green change while R2 holds: seconds to the 2nd consecutive green answer in force, to the end of R2, to rolling."""
    out = []
    a = sorted(ans, key=lambda d: d["t_q"])
    gt = [(d["t_q"], d["gt"].get("tl")) for d in a]
    for i in range(1, len(gt)):
        if gt[i][1] == 0 and gt[i - 1][1] in (1, 2):
            tg = gt[i][0]
            at = [r for r in st if r["t"] >= tg]
            if not at or "R2" not in at[0].get("rules", []):
                continue
            end = next((r["t"] for r in at if "R2" not in r.get("rules", [])), np.nan)
            roll = next((r["t"] for r in at if "R2" not in r.get("rules", []) and r["v"] > 1.0), np.nan)
            greens = [d for d in a if d["t_q"] >= tg and d["ans"].get("Q_light") == "green_for_ego"]
            two = np.nan
            for d1, d2 in zip(greens, greens[1:]):
                if abs(d2["t_q"] - d1["t_q"] - 0.5) < 1e-6:
                    two = d2["t_eff"] - tg
                    break
            out.append(dict(t_green=tg, answer=two, r2_end=end - tg, roll=roll - tg))
    return out


def infraction_events(rid, seed, r):
    """Red-light infractions of one run with the crossing time of the logged ego light (red, distance to the stop line from > 0 to <= 0)."""
    a = Path(r["attempt"])
    rec = json.loads((a / "results.json").read_text())["_checkpoint"]["records"][0]
    msgs = rec["infractions"].get("red_light", [])
    if not msgs:
        return []
    ticks = jsonl(a / "ticks.jsonl")
    cross, prev, cur = [], None, None
    for t in ticks:
        c = t.get("ctx", {})
        if c.get("tl") is not None and c.get("tl_dist") is not None:
            if prev is not None and prev[0] > 0 >= c["tl_dist"] and c["tl"] in (1, 2):    # the ego light's stop line passed on yellow / red
                cur = [t["t"], c.get("tl_id"), t["v"], c["tl"], None]
                cross.append(cur)
            if cur is not None and cur[4] is None and c["tl"] == 2 and c["tl_dist"] <= 0 and t["v"] > 0.5:
                cur[4] = t["t"]                                                   # first moment past the line on red while rolling
            prev = (c["tl_dist"], c["tl"])
        elif prev is not None and prev[0] <= 0 and cur is not None and cur[4] is None and prev[1] == 2 and t["v"] > 0.5:
            cur[4] = t["t"]                                                       # inside the junction (ctx without a light), light last seen red
            prev = None
        else:
            prev = None
    cross = [(x[4] if x[4] is not None else x[0], x[1], x[2], x[3], x[0]) for x in cross]      # (T, light id, v at the line, state at the line, t at the line)
    return [dict(route=rid, seed=seed, msg=m, cross=cross[i] if i < len(cross) else None, n_cross=len(cross)) for i, m in enumerate(msgs)]


def timeline(r, T, back=9.0, fwd=0.5):
    ans, st, head = vlm_rows(r["attempt"])
    A = {round(d["t_q"], 2): d for d in ans}
    rows = []
    for s in st:
        if T - back <= s["t"] <= T + fwd:
            q = [d for d in ans if d["t_eff"] <= s["t"] + 1e-6]
            last = q[-1] if q else None
            rows.append(dict(t=round(s["t"], 1), v=round(s["v"], 1), line_m=s.get("line_r2", s.get("line")), rules="+".join(s.get("rules", [])) or "-", fresh=s.get("fresh"),
                             last_K=",".join(x[:3] for x in s.get("light", [])),
                             answer_in_force=(last["ans"].get("Q_light", "")[:3] + " (q %.1f, lat %.2f s)" % (last["t_q"], last["ans"].get("latency_ms", 0) / 1e3)) if last else "-",
                             truth=(None if not last else "tl %s @ %s m" % (last["gt"].get("tl"), last["gt"].get("tl_dist")))))
    return rows, head


def classify(r, T):
    """Machine label of one infraction from the logs (evidence is printed next to it and checked by hand)."""
    ans, st, head = vlm_rows(r["attempt"])
    lo = T - 12.0
    win = [s for s in st if lo <= s["t"] <= T + 0.25]
    held = [s for s in win if "R2" in s.get("rules", [])]
    r5 = [s for s in win if s.get("r5") or "R5" in s.get("rules", [])]
    g = [d for d in ans if lo <= d["t_q"] <= T and d["gt"].get("tl") in (1, 2) and (d["gt"].get("tl_dist") or 999) < 50]
    red_ans = [d for d in g if d["ans"].get("Q_light") == RED]
    if held and (not [s for s in win if s["t"] >= T - 0.5 and "R2" in s.get("rules", [])]):
        if r5:
            return "released early by R5 (fallback)"
        return "released early (R2 dropped while the light was red)"
    if held:
        return "R2 held at the line, crossed anyway (pushed through)"
    if not g:
        return "light beyond range (no red ego light within 50 m in the 12 s before)"
    if not red_ans:
        c = pd.Series([d["ans"].get("Q_light") for d in g]).value_counts().to_dict()
        return "not answered red (answers on red frames: %s)" % c
    first = min(d["t_eff"] for d in red_ans)
    after = [s for s in st if s["t"] >= first]
    sp = after[0] if after else None
    if sp is not None and not sp.get("fresh", True):
        return "stale answer"
    if sp is not None:
        return "answered late (first red answer in force at t=%.1f, v=%.1f, line %.1f m)" % (sp["t"], sp["v"], sp.get("line_r2", sp.get("line", np.nan)))
    return "unclassified"


def report(_):
    OUT.mkdir(parents=True, exist_ok=True)
    df = runs()
    df.to_csv(OUT / "runs.csv", index=False)
    cal, few_g = (RUN / "gates/vred_cal.json").read_text(), None
    D = ["# vred: zero-shot Qwen3-VL-4B reads the traffic light, closed loop (19 routes x 2 traffic seeds)", "",
         "Diagnostic batch. Every number is a read, not a confirmation: registered confirmation lines are listed with their value and marked "
         "not evaluated (19 routes < 30). `drive` and `pred` are the existing runs of the earlier batch (not rerun); `vred` ran with the "
         "same base configuration (decision-82 resume setting, deviation D5), rows R2 + R5, light question answered by zero-shot Qwen3-VL-4B "
         "(one forward pass, option scoring, both cameras, 1153 visual tokens); R3 off (deviation D18).", ""]
    g = df.groupby("arm")
    ARMT = g.agg(runs=("route", "count"), crashes=("crash", "sum"), DS=("DS", "mean"), RC=("RC", "mean"), red_light=("red_light", "sum"),
              stop_sign=("stop_infraction", "sum"), collisions=("collisions", "sum"), blocked=("vehicle_blocked", "sum"),
              timeouts=("route_timeout", "sum"), v_mean=("v_mean", "mean")).reindex(["drive", "pred", "vred"]).round(2)
    D += ["## Arms", "", ARMT.to_markdown(), "", "Official infraction counts summed over the runs; DS, RC, mean speed averaged over runs; expected runs per arm: %d. "
          "`crashes` are program crashes (excluded from the paired reads)." % (len(ROUTES) * len(SEEDS)), ""]
    ans = pd.concat([answers("vred", 0), answers("vred", 1)], ignore_index=True)
    ans.to_csv(OUT / "answers.csv", index=False)
    logged = sorted(set(ans.route[(ans.tl != -1) & (ans.tl_dist < 50)]))      # sensitivity set: an ego light was seen within 50 m in the vred logs
    # paired reads
    P = []
    sets = [("all routes", ROUTES)] + list(TARGETS.items()) + [("logged-light routes", logged), ("no logged light", [r for r in ROUTES if r not in logged])]
    for a in ("vred", "pred"):
        for name, routes in sets:
            for col in ("DS", "RC", "red_light", "vehicle_blocked", "collisions", "v_mean"):
                P.append(dict(contrast=a + " - drive", routes=name, metric=col, **paired(df, a, "drive", routes, col)))
    P = pd.DataFrame(P)
    P.to_csv(OUT / "paired.csv", index=False)
    D += ["## Paired differences to `drive` (mean over routes of the per-route mean over the two seeds; [95% route-cluster CI])", "",
          "`light routes` = %s (scenario types with a traffic light, as in results/report.md; primary); `other routes` = the remaining %d. `logged-light routes` = %s (sensitivity: an ego light "
          "was seen within 50 m in the vred logs); `no logged light` = the rest. Counts are per run "
          "(difference of the per-route mean over seeds), so +0.50 is one extra event in one of two runs of a route.\n" % (", ".join(LIGHT_ROUTES), len(ROUTES) - len(LIGHT_ROUTES), ", ".join(logged)),
          "| contrast | routes | n routes | DS | RC | red light | blocked | collisions | mean speed m/s |", "|:--|:--|--:|:--|:--|:--|:--|:--|:--|"]
    for (c, rs), g_ in P.groupby(["contrast", "routes"], sort=False):
        f = lambda col: fmt(g_[g_.metric == col].iloc[0].to_dict())   # noqa: E731
        D.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (c, rs, g_.groups.iloc[0], f("DS"), f("RC"), f("red_light"), f("vehicle_blocked"), f("collisions"), f("v_mean")))
    # event counts on the other routes
    V, B = df[df.arm == "vred"].set_index(["route", "seed"]), df[df.arm == "drive"].set_index(["route", "seed"])
    idx = [i for i in V.index.intersection(B.index) if i[0] in TARGETS["other routes"]]
    newb = int(sum(V.loc[i, "vehicle_blocked"] > B.loc[i, "vehicle_blocked"] for i in idx))
    newc = int(sum(V.loc[i, "collisions"] > B.loc[i, "collisions"] for i in idx))
    D += ["", "Harm on the other routes (%d runs): runs with more blocked events than their `drive` pair %d; runs with more collisions %d; summed collisions vred %d vs drive %d; "
          "summed blocked %d vs %d." % (len(idx), newb, newc, V.loc[idx, "collisions"].sum(), B.loc[idx, "collisions"].sum(), V.loc[idx, "vehicle_blocked"].sum(), B.loc[idx, "vehicle_blocked"].sum()), ""]
    # retention
    D += ["## Retention of the privileged gain: (vred - drive) / (pred - drive)", "", "Ratio of the route means of the paired differences, [95% route-cluster CI]. "
          "The denominator is itself small and its interval includes zero in several rows: read the point estimate with that in mind.", "",
          "| quantity | routes | retention |", "|:--|:--|:--|"]
    for name, routes in sets:
        for lab, col in (("DS", "DS"), ("red-light infractions per run", "red_light")):
            D.append("| %s | %s | %s |" % (lab, name, fmt_ret(boot_ret(df, "vred", "pred", routes, col))))
    # in-loop reading and latency
    lat = ans.lat[ans.ok].to_numpy()
    L = float(ans.L.iloc[0]) if len(ans) else float("nan")
    D += ["", "## In-batch latency of the light answers", "",
          "`latency_ms` = from the hand-over of the frames to the answer (frame conversion, JPEG encoding, wait for a pool thread, HTTP, server queue and service). "
          "L = %.2f s: an answer is used at t_q + max(L, latency). TTL = 2.5 s." % L, "",
          "| requests | answered ok | p50 ms | p95 ms | p99 ms | max ms | share > L | share > 1 s | share > TTL (2.5 s) | server queue p95 ms | server service p50 ms | effective delay p95 s |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
          "| %d | %.2f%% | %.0f | %.0f | %.0f | %.0f | %.1f%% | %.2f%% | %.2f%% | %.0f | %.0f | %.2f |" % (
              len(ans), 100 * ans.ok.mean(), pct(lat, 50), pct(lat, 95), pct(lat, 99), lat.max(), 100 * np.mean(lat > 1e3 * L), 100 * np.mean(lat > 1000),
              100 * np.mean(lat > 2500), pct(ans.q_ms[ans.ok], 95), pct(ans.svc[ans.ok], 50), pct(np.maximum(L, lat / 1e3), 95)), "",
          "Calibration for L (shadow run, same concurrency): see `calibration.md`; p95 there %.0f ms." % json.loads(cal)["p95_ms"], ""]
    # in-loop reading quality
    a2 = ans[ans.ok].reset_index(drop=True)
    m = light_masks(a2)
    near = lambda lo, hi: (a2.tl_dist >= lo) & (a2.tl_dist < hi)   # noqa: E731
    red = a2.tl.isin([1, 2])
    rate = lambda mask, hit: boot_ratio((hit & mask).astype(float), mask.astype(float), a2.route)   # noqa: E731
    rows = [("ego red / yellow answered red, 0-50 m", rate(m["red"], a2.ans == RED), ">= 80%"),
            ("... 0-20 m", rate(red & near(-5, 20), a2.ans == RED), ""), ("... 20-50 m", rate(red & near(20, 50), a2.ans == RED), ""),
            ("ego red / yellow answered green, 0-50 m", rate(m["red"], a2.ans == "green_for_ego"), ""),
            ("ego green answered green, 0-50 m", rate(m["green"], a2.ans == "green_for_ego"), ""),
            ("another direction red, ego not red: answered red", rate(m["other"], a2.ans == RED), "<= 10%"),
            ("no light: answered red", rate(m["nolight"], a2.ans == RED), "<= 2%")]
    D += ["## In-loop reading quality (answers logged during the vred runs against the simulator's light state)", "",
          "Same label rows as Phase A (`vlm_arb_phase_a`), but on the frames the car saw while the VLM was driving: the car stops at red lights, so the frames "
          "are not the shadow distribution. Estimate [95% route-cluster CI].", "", "| readout | value | requests | routes | registered line |", "|:--|:--|--:|--:|:--|"]
    for lab, r, line in rows:
        D.append("| %s | %s | %d | %d | %s |" % (lab, fmt(r, True), r["n"], r["groups"], line))
    # per-route and event tables
    sh = json.loads((RUN / "gates/vred_shards.json").read_text())
    ev, per_route, rel, r5_rows, infr = [], [], [], [], []
    for rid in ROUTES:
        row = dict(route=rid, light=("yes" if rid in LIGHT_ROUTES else "") + ("*" if rid in logged else ""))
        for s in SEEDS:
            r = route_row(unit_dir("vred", s, next(k for k in SHARDS if rid in sh[k])), rid)
            if not r:
                continue
            a, st, head = vlm_rows(r["attempt"])
            row["R2_stops_s%d" % s] = len(episodes(st, "R2"))
            e5 = episodes(st, "r5")
            for (t0, t1) in e5:
                pre = [x for x in st if x["t"] <= t0][-2:]
                cause = "answer not fresh" if any(not x.get("fresh", True) for x in pre) else "held > T_max"
                r5_rows.append(dict(route=rid, seed=s, t=round(t0, 1), duration_s=round(t1 - t0, 1), cause=cause, light=("%s" % [x["gt"].get("tl") for x in a if abs(x["t_eff"] - t0) < 1][:1])))
            for x in green_release(a, st):
                rel.append(dict(route=rid, seed=s, **x))
            for e in infraction_events(rid, s, r):
                T = e["cross"][0] if e["cross"] else None
                lab = classify(r, T) if T is not None else "crossing not found in the log (n_cross=%d)" % e["n_cross"]
                tl, _ = timeline(r, T) if T is not None else ([], None)
                infr.append(dict(route=rid, seed=s, light_id=re.findall(r"light (\d+)", e["msg"])[0], t_line=None if T is None else round(e["cross"][4], 1),
                                 state_at_line={1: "yellow", 2: "red"}.get(e["cross"][3]) if T is not None else None, t_red_rolling=None if T is None else round(T, 1),
                                 v_line=None if T is None else round(e["cross"][2], 1), cause=lab, timeline=tl))
        for arm in ("drive", "pred", "vred"):
            x = df[(df.arm == arm) & (df.route == rid)]
            row.update({"DS_" + arm: round(x.DS.mean(), 1), "red_" + arm: int(x.red_light.sum())})
        row["blocked_vred"] = int(df[(df.arm == "vred") & (df.route == rid)].vehicle_blocked.sum())
        row["coll_drive"] = int(df[(df.arm == "drive") & (df.route == rid)].collisions.sum())
        row["coll_vred"] = int(df[(df.arm == "vred") & (df.route == rid)].collisions.sum())
        row["v_drive"] = round(df[(df.arm == "drive") & (df.route == rid)].v_mean.mean(), 2)
        row["v_vred"] = round(df[(df.arm == "vred") & (df.route == rid)].v_mean.mean(), 2)
        per_route.append(row)
    pr = pd.DataFrame(per_route)
    pr["dDS"] = (pr.DS_vred - pr.DS_drive).round(1)
    D += ["", "## Per route (DS and red-light counts are means / sums over the two seeds; R2 stops = episodes of R2 holding the car; light: yes = scenario set, * = ego light seen in the vred logs)", "",
          pr[["route", "light", "DS_drive", "DS_vred", "dDS", "DS_pred", "red_drive", "red_vred", "red_pred", "R2_stops_s0", "R2_stops_s1", "blocked_vred",
              "coll_drive", "coll_vred", "v_drive", "v_vred"]].to_markdown(index=False), ""]
    rl = pd.DataFrame(rel)
    D += ["## Green after red", ""]
    if len(rl):
        D += ["%d red-to-green changes of the ego light while R2 held the car. Seconds after the light turned green (median [p5, p95]): "
              "second consecutive green answer in force %.1f [%.1f, %.1f] (%d never reached); R2 released %.1f [%.1f, %.1f]; car rolling (> 1 m/s) %.1f [%.1f, %.1f] (%d never)." % (
                  len(rl), np.nanmedian(rl.answer), pct(rl.answer.dropna(), 5), pct(rl.answer.dropna(), 95), int(rl.answer.isna().sum()),
                  np.nanmedian(rl.r2_end), pct(rl.r2_end.dropna(), 5), pct(rl.r2_end.dropna(), 95), np.nanmedian(rl.roll), pct(rl.roll.dropna(), 5),
                  pct(rl.roll.dropna(), 95), int(rl.roll.isna().sum())), "", rl.round(1).to_markdown(index=False), ""]
    else:
        D += ["No red-to-green change of the ego light occurred while R2 held the car.", ""]
    D += ["## R5 fallback", "", "%d episodes in the %d vred runs." % (len(r5_rows), len(df[df.arm == "vred"])) + (("\n\n" + pd.DataFrame(r5_rows).to_markdown(index=False)) if r5_rows else ""), ""]
    D += ["## Remaining red-light infractions in vred (%d)" % len(infr), "",
          "Crossing time = the tick where the logged ego light, red, passed from distance > 0 to <= 0 from the stop line (ticks.jsonl); the cause label is the "
          "machine reading of the logs and is checked by hand below the table.", "",
          pd.DataFrame([{k: v for k, v in x.items() if k != "timeline"} for x in infr]).to_markdown(index=False) if infr else "none", ""]
    for x in infr:
        if x["timeline"]:
            D += ["", "Route %s seed %s, light %s, stop line at t = %s s (%s), rolling on red at t = %s s: %s" % (x["route"], x["seed"], x["light_id"], x["t_line"], x["state_at_line"], x["t_red_rolling"], x["cause"]), "",
                  pd.DataFrame(x["timeline"]).to_markdown(index=False)]
    # registered lines
    pv = lambda a, rs, col: paired(df, a, "drive", rs, col)   # noqa: E731
    non = TARGETS["other routes"]
    r_ds, r_nl = pv("vred", non, "DS"), pv("vred", LIGHT_ROUTES, "DS")
    D += ["", "## Registered lines (plan 4.3), vred", "", "| line | read | status |", "|:--|:--|:--|",
          "| harmless on non-target routes: DS CI lower bound >= -5, no new blocked, no new collision | DS %s; blocked %+d, collisions %+d | not evaluated at this size (%d routes < 30) |" % (
              fmt(r_ds), r_ds["d_vehicle_blocked"], r_ds["d_collisions"], r_ds["groups"]),
          "| useful on target routes: red-light infractions fall, DS CI lower bound > 0, retention >= 50%% | DS %s; red light %+d runs; retention (DS, light routes) %s | not evaluated at this size (%d routes < 30) |" % (
              fmt(r_nl), r_nl["d_red_light"], fmt_ret(boot_ret(df, "vred", "pred", LIGHT_ROUTES, "DS")), r_nl["groups"])]
    figures(P, ans, L)
    D += ["", "![paired differences](vred_paired.png)", "", "Figure: paired difference to `drive` per route set for `vred` and the privileged `pred` (dot: mean over routes, bar: 95% route-cluster CI), "
          "DS on the left and red-light infractions per run on the right. Look at whether `vred` sits near `pred` and whether any bar clears zero.", "",
          "![in-batch latency](vred_latency.png)", "", "Figure: cumulative distribution of the in-batch answer latency, with L and the TTL. Look at how far the curve stays left of L "
          "and the size of the tail beyond it."]
    (OUT / "vred.md").write_text("\n".join(D) + "\n")
    write_json(OUT / "summary.json", dict(arms=ARMT.reset_index().to_dict("records"), L=L))
    print("\n".join(D))


def figures(P, ans, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"vred - drive": "#3b6ea5", "pred - drive": "#b5651d"}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for ax, metric, title in ((axs[0], "DS", "DS"), (axs[1], "red_light", "red-light infractions per run")):
        k = 0
        for rs in ("all routes", "light routes", "other routes"):
            for c in ("vred - drive", "pred - drive"):
                r = P[(P.contrast == c) & (P.routes == rs) & (P.metric == metric)].iloc[0]
                if r.groups:
                    ax.plot([r.lo, r.hi], [-k, -k], color=col[c], lw=2, solid_capstyle="round")
                    ax.plot(r.est, -k, "o", color=col[c], ms=6, label=c if rs == "all routes" else None)
                k += 1
            k += 0.6
        ax.axvline(0, color="#222", lw=1)
        ax.set_yticks([-0.5, -3.1, -5.7])
        ax.set_yticklabels(["all", "light", "other"] if ax is axs[0] else [])
        ax.set_xlabel("paired difference to drive (%s)" % title, fontsize=8)
        ax.grid(axis="x", color="#eee", lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "vred_paired.png")
    plt.close(fig)
    lat = np.sort(ans.lat[ans.ok].to_numpy()) / 1e3
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=150)
    ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat), color="#3b6ea5", lw=2)
    ax.axvline(L, color="#222", lw=1)
    ax.axvline(2.5, color="#b5651d", lw=1, ls="--")
    ax.text(L, 0.05, " L = %.2f s" % L, fontsize=7)
    ax.text(2.5, 0.05, " TTL 2.5 s", fontsize=7, color="#b5651d")
    ax.set_xscale("log")
    ax.set_xlabel("answer latency, s (log)", fontsize=8)
    ax.set_ylabel("share of answers <= x", fontsize=8)
    ax.grid(color="#eee", lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "vred_latency.png")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calibrate", "few", "report"])
    a = ap.parse_args()
    {"calibrate": calibrate, "few": few, "report": report}[a.cmd](a)
