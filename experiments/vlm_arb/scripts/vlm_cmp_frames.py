"""Frames, labels, the ground-truth directive and the frame selection of the vlm_cmp study (Qwen3-VL-4B vs 8B).

  python vlm_cmp_frames.py counts      print the category counts of every framed request (no model output is read)

Plan and pre-registration: experiments/vlm_arb/plans/2026-10-03-vlm-4b-vs-8b.md. Everything here is derived from the logged
ground truth (vlm_decisions.jsonl: `gt` of every answered request and the arbitration step at the same instant); no model
answer enters any definition in this file.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vlm_arb_common import DATA, REPO, RUN, XML, obstacle_kinds  # noqa: E402

ARMS = RUN / "arms"
SPLIT = dict(                                   # the registered route splits b2d/vlm-thin-{train,val,test} v1 (unchanged)
    test="27043 15483 16529 9196 28147 17280 2520 19324".split(),
    val="15612 27297 334 19832".split(),
    train="15102 16390 16508 24944 27787 27870 22535 24497 37969".split())
DIRECTIVES = ["proceed", "slow", "stop_at_line", "go_now", "pass_left", "pass_right", "wait"]
PREV_DT = 1.0                                   # seconds between the earlier frame and the current one (two-frame variant)


# ---------------------------------------------------------------------------------------------------- all framed requests
def load_all():
    """One row per answered request that has both camera frames on disk, in every unit, with the logged truth, the ego speed
    at that instant, and the truth / path of the request about PREV_DT earlier of the same attempt (NaN / '' when none)."""
    rows = []
    for vd in sorted(ARMS.glob("*/attempts/*/*/vlm_frames")):
        a = vd.parent
        unit, route, att = a.parent.parent.parent.name, a.parent.name, a.name
        p = a / "vlm_decisions.jsonl"
        if not p.exists():
            continue
        have = set(os.listdir(vd))
        S, A = {}, []
        for line in open(p):
            d = json.loads(line)
            if d.get("k") == "s":
                S[round(d["t"], 2)] = d["v"]
            elif d.get("k") == "a" and d["ans"].get("ok"):
                tq = d["t_q"]
                if "%08.2f_wide.jpg" % tq in have and "%08.2f_road.jpg" % tq in have:
                    A.append(d)
        by_t = {round(d["t_q"], 2): d for d in A}
        for d in A:
            t, g = round(d["t_q"], 2), d["gt"]
            tl, others = g.get("tl"), [x for x in g.get("lights", []) if x[0] != g.get("tl_id")]
            pv = by_t.get(round(t - PREV_DT, 2))
            row = dict(
                id="%s/%s/%s/%.2f" % (unit, route, att, t), unit=unit, route=route, attempt=att, t=t,
                v=S.get(t, np.nan), tl=-1 if tl is None else tl, tl_dist=g["tl_dist"] if tl is not None else np.nan,
                junc_dist=np.nan if g.get("junc_dist") is None else g["junc_dist"],
                any_light=float(bool(g.get("lights"))), other_red=float(any(x[1] == 2 for x in others)),
                other_green=float(any(x[1] == 0 for x in others)),
                stop_dist=np.nan if g.get("stop_dist") is None else g["stop_dist"], block=g["block"], side=g["side"],
                block_dist=np.nan if g.get("block_dist") is None else g["block_dist"],
                lead_gap=np.nan if g.get("lead_gap") is None else g["lead_gap"],
                lead_v=np.nan if g.get("lead_v") is None else g["lead_v"],
                wide=str(vd / ("%08.2f_wide.jpg" % d["t_q"])), road=str(vd / ("%08.2f_road.jpg" % d["t_q"])),
                prev_t=np.nan, prev_wide="", prev_road="", prev_tl=-9, prev_block="", prev_v=np.nan, prev_lead_v=np.nan)
            if pv is not None:
                pt = round(pv["t_q"], 2)
                gp = pv["gt"]
                row.update(prev_t=pt, prev_wide=str(vd / ("%08.2f_wide.jpg" % pv["t_q"])),
                           prev_road=str(vd / ("%08.2f_road.jpg" % pv["t_q"])), prev_tl=-1 if gp.get("tl") is None else gp["tl"],
                           prev_block=gp["block"], prev_v=S.get(pt, np.nan),
                           prev_lead_v=np.nan if gp.get("lead_v") is None else gp["lead_v"])
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    return label(df)


# ---------------------------------------------------------------------------------------------------- labels
def label(df):
    kinds = obstacle_kinds()
    df["kind"] = df.route.map(kinds).fillna("other")                       # cones / vehicle / other: the route's static obstacle
    df["part"] = df.route.map({r: p for p, rs in SPLIT.items() for r in rs}).fillna("none")
    near = (df.tl_dist >= -5) & (df.tl_dist < 50)
    df["ego_red"] = df.tl.isin([1, 2]) & near
    df["ego_green"] = (df.tl == 0) & near
    df["ego_light"] = np.where(df.ego_red, "red", np.where(df.ego_green, "green", np.where(df.tl == -1, "none", "far")))
    # position of the car relative to the light's stop line / the junction entrance, from the logged gt fields
    inside = df.junc_dist <= 0.01
    df["pos"] = np.where(df.tl == -1, "", np.where(inside, "C", np.where(df.tl_dist <= 0, "B", "A")))
    df["yellow"] = df.tl == 1
    df["lead_state"] = np.where(df.lead_gap.isna(), "none", np.where(df.lead_v < 0.5, "stopped", "moving"))
    df["ego_moving"] = df.v >= 0.5
    prev_ok = df.prev_t.notna()
    # events visible only across two instants (one second apart)
    df["ev_yellow_onset"] = prev_ok & (df.prev_tl == 0) & (df.tl == 1) & near
    df["ev_red_onset"] = prev_ok & (df.prev_tl == 0) & (df.tl == 2) & near
    df["ev_red_to_green"] = prev_ok & df.prev_tl.isin([1, 2]) & (df.tl == 0) & near
    df["ev_hold"] = df.ego_red & (df.v < 0.5)
    df["sign_near"] = df.stop_dist <= 25
    df["sign_far"] = df.stop_dist > 25
    df["dir"], df["dir_ok"], df["dir_note"] = zip(*[directive(r) for r in df.itertuples()])
    return df


def directive(r):
    """Ground-truth directive of one request: (primary, set of accepted directives, ambiguity note). Registered in the plan
    (section 3) before any model answer was read; the accepted set is the strict one when the note is empty.
    Priority: ego light red/yellow > ego green > stop sign > static block > moving lead > junction ahead > proceed."""
    stopped = bool(r.v < 1.0)
    side_dir = {"left_free": "pass_left", "right_free": "pass_right", "none_free": "wait"}.get(r.side, "wait")
    if r.ego_red:
        if r.tl_dist < 0:
            return "stop_at_line", {"stop_at_line", "proceed"}, "red_past_line"        # cannot stop before the line any more
        if r.yellow:
            return "stop_at_line", {"stop_at_line", "proceed", "slow"}, "yellow"        # yellow: comfortable stop depends on speed
        return "stop_at_line", {"stop_at_line"}, ""
    if r.ego_green:
        if stopped:
            return "go_now", {"go_now"}, ""
        return "proceed", {"proceed", "slow", "go_now"}, "green_moving"                # slow vs proceed is a style choice
    if r.sign_near:
        if not stopped:
            return "stop_at_line", {"stop_at_line"}, ""
        return "stop_at_line", {"stop_at_line", "go_now"}, "sign_stopped"              # the log does not say if the stop is done
    if r.block == "static_block":
        if r.kind in ("cones", "vehicle"):
            return side_dir, {side_dir}, ""
        return side_dir, {side_dir, "wait", "slow", "proceed"}, "block_not_a_scenario"  # a queue or a car stopped by traffic
    if r.block == "moving_lead":
        return "proceed", {"proceed", "slow"}, "moving_lead"
    if 0 < r.junc_dist < 40:
        return "slow", {"slow"}, ""
    return "proceed", {"proceed"}, ""


# ---------------------------------------------------------------------------------------------------- selection
def waterfill(counts, quota):
    """Per-group caps summing to <= quota, as equal as possible (groups with fewer items than the cap keep all of them)."""
    caps = {k: 0 for k in counts}
    left, active = quota, [k for k in counts if counts[k] > 0]
    while left > 0 and active:
        share = max(1, left // len(active))
        nxt = []
        for k in active:
            take = min(share, counts[k] - caps[k], left)
            caps[k] += take
            left -= take
            if caps[k] < counts[k]:
                nxt.append(k)
        active = nxt
    return caps


def spread(idx, k):
    """k indices evenly spread over the sorted list idx (first and last included when k >= 2)."""
    n = len(idx)
    if k >= n:
        return list(idx)
    if k <= 0:
        return []
    return [idx[int(round(i * (n - 1) / max(k - 1, 1)))] for i in range(k)]


# (name, mask function, quota); a request can belong to several categories, it is kept once. The order does not matter:
# every category is filled independently, evenly over routes (water-filling) and over time inside a route.
CATS = [
    ("red_A", lambda d: d.ego_red & (d.pos == "A"), 150),
    ("red_BC", lambda d: d.ego_red & d.pos.isin(["B", "C"]), 100),
    ("green_A", lambda d: d.ego_green & (d.pos == "A"), 70),
    ("green_BC", lambda d: d.ego_green & d.pos.isin(["B", "C"]), 70),
    ("hold", lambda d: d.ev_hold, 40),
    ("far_light", lambda d: (d.ego_light == "far"), 25),
    ("other_light_only", lambda d: (d.tl == -1) & (d.any_light == 1), 60),
    ("no_light_clear", lambda d: (d.tl == -1) & (d.any_light == 0) & (d.block == "clear") & d.sign_near.eq(False) & d.sign_far.eq(False), 60),
    ("sign_near", lambda d: d.sign_near, 80),
    ("sign_far", lambda d: d.sign_far, 30),
    ("cones", lambda d: (d.block == "static_block") & (d.kind == "cones"), 60),
    ("vehicle", lambda d: (d.block == "static_block") & (d.kind == "vehicle"), 80),
    ("block_other", lambda d: (d.block == "static_block") & (d.kind == "other"), 50),
    ("moving_lead", lambda d: d.block == "moving_lead", 60),
    ("lead_stopped_clear", lambda d: (d.block == "clear") & (d.lead_state == "stopped"), 40),
    ("lead_moving_clear", lambda d: (d.block == "clear") & (d.lead_state == "moving"), 40),
    ("ev_yellow_onset", lambda d: d.ev_yellow_onset, 10 ** 6),                   # every one of these
    ("ev_red_onset", lambda d: d.ev_red_onset, 10 ** 6),
    ("ev_red_to_green", lambda d: d.ev_red_to_green, 10 ** 6),
    ("junction_slow", lambda d: (d.junc_dist > 0) & (d.junc_dist < 40) & (d.tl == -1) & ~d.sign_near & (d.block == "clear"), 40),
]


def select(df):
    """The core set: per category its quota, spread over routes and then over time. Adds `cats` (all categories of a kept
    request, comma separated). Deterministic; no model output involved."""
    df = df.sort_values(["route", "unit", "attempt", "t"]).reset_index(drop=True)
    keep, cats = set(), {}
    for name, fn, quota in CATS:
        m = fn(df)
        by_route = {r: list(g.index) for r, g in df[m].groupby("route")}
        caps = waterfill({r: len(v) for r, v in by_route.items()}, quota)
        for r, idx in by_route.items():
            for i in spread(idx, caps[r]):
                keep.add(i)
                cats.setdefault(i, []).append(name)
    out = df.loc[sorted(keep)].copy()
    out["cats"] = [",".join(cats[i]) for i in out.index]
    out = out.reset_index(drop=True)
    out["tf"] = two_frame_subset(out)
    return out


TF_HALF = {"cones", "vehicle", "moving_lead", "lead_stopped_clear", "lead_moving_clear", "hold", "red_BC", "green_BC", "junction_slow"}


def two_frame_subset(sel):
    """Instants of the two-frame run (an earlier frame must exist): every event instant, every 2nd request of the categories
    in TF_HALF, every 3rd of the rest (counted per route in the sorted order of the core set)."""
    out = np.zeros(len(sel), bool)
    cnt = {}
    for i, r in enumerate(sel.itertuples()):
        if not np.isfinite(r.prev_t):
            continue
        cs = set(r.cats.split(","))
        if cs & {"ev_yellow_onset", "ev_red_onset", "ev_red_to_green"}:
            out[i] = True
            continue
        step = 2 if cs & TF_HALF else 3
        key = (r.route, step)
        cnt[key] = cnt.get(key, 0) + 1
        out[i] = cnt[key] % step == 1
    return out


def counts():
    df = load_all()
    print("framed requests", len(df), "routes", df.route.nunique(), "units", df.unit.nunique(), "attempts",
          df.groupby(["unit", "route", "attempt"]).ngroups)
    print("parts", df.part.value_counts().to_dict())
    print("with an earlier frame", int(df.prev_t.notna().sum()), "speed known", int(df.v.notna().sum()))
    print(df.dir.value_counts().to_string())
    print("ambiguity notes", df.dir_note.value_counts().to_string())
    for name, fn, q in CATS:
        m = fn(df)
        print("%-20s total %6d routes %2d quota %s" % (name, m.sum(), df[m].route.nunique(), q if q < 10 ** 6 else "all"))
    sel = select(df)
    print("selected", len(sel), "routes", sel.route.nunique(), sel.part.value_counts().to_dict())
    print(sel.dir.value_counts().to_string())
    print(sel.groupby("route").size().to_string())
    print("prev available in selection", int(sel.prev_t.notna().sum()), "two-frame instants", int(sel.tf.sum()), sel[sel.tf].route.nunique())
    print(pd.crosstab(df.lead_state, df.ego_moving))
    print(pd.crosstab(df[df.block == "static_block"].kind, df[df.block == "static_block"].side))
    print(df[df.ego_red].groupby("pos").size().to_dict(), df[df.ego_green].groupby("pos").size().to_dict())


if __name__ == "__main__":
    {"counts": counts}[sys.argv[1]]()
