"""Analysis 2: decide "static obstacle ahead" from time (ego stopped behind something for T s) instead of from a single frame.

  python3 bypass_timedet.py EXTRACT_DIR RESULTS_DIR

Works on the 38 `drive` runs. Writes results/bypass_stop_episodes.csv, bypass_timedet_grid.csv and bypass_timedet.md.
Definitions: plans/2026-10-02-bypass-offline.md.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bypass_common import OBST_TYPE, TARGET, ego_s_series, junction_table, load_all, plan_arr  # noqa: E402

T_REF = 6.5          # the ego is held until about 6 s (planner leaves warm-up at 5 s); stops are counted from 6.5 s
V_STOP = 0.3         # m/s, episode definition
MIN_EP = 1.0         # s, shortest stop episode kept
MERGE_GAP = 0.6      # s, gaps shorter than this inside an episode are bridged


def episodes(t, v, v_stop=V_STOP):
    """Maximal runs of plan steps with v < v_stop, clipped to start at T_REF, bridged over short gaps, at least MIN_EP long.
    Returns list of (t_start, t_end, truncated)."""
    stopped = (v < v_stop) & (t >= T_REF)
    out, i, n = [], 0, len(t)
    while i < n:
        if not stopped[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if stopped[j + 1]:
                j += 1
                continue
            k = j + 1
            while k < n and not stopped[k] and t[k] - t[j] <= MERGE_GAP:
                k += 1
            if k < n and stopped[k] and t[k] - t[j] <= MERGE_GAP:
                j = k
            else:
                break
        out.append((float(t[i]), float(t[j]), j >= n - 2))
        i = j + 1
    return [e for e in out if e[1] - e[0] >= MIN_EP]


OBS_AHEAD_M = 45.0   # a scenario obstacle this close ahead (arc length, rear axle to obstacle start) is "the" cause of a stop on a target route
RED_X_TRUTH = 50.0   # m, ego light red or yellow closer than this = red-light stop in the ground-truth label
LEAD_D = 40.0        # m, lead head distance (from the camera) counted as "a lead is reported ahead"
LEAD_P = 0.5         # lead probability threshold (the agent's own lead_p)
VLM_TTL = 2.5        # s, age after which a logged VLM answer is not used


class Drive:
    """One drive run: plan-step signals, ego arc position, junction distance and the logged VLM answers."""

    def __init__(self, d, junc):
        self.d = d
        pl = plan_arr(d)
        self.t, self.v, self.tl, self.td = pl["t"], pl["v"], pl["tl"], pl["tl_dist"]
        self.lead_gap_truth = pl["lead_gap"]
        lead0 = pl["lead0"]
        self.lead_x = np.array([np.nan if r is None else r[0] for r in lead0], float)
        self.lead_vh = np.array([np.nan if r is None else r[2] for r in lead0], float)
        self.lp0 = np.array([np.nan if r is None else r[0] for r in pl["lp"]], float)
        ts, ss = ego_s_series(d)
        self.es = np.interp(self.t, ts, ss)
        sl = sorted({e["t"]: e["junc_dist"] for e in d["vlm"] if e["k"] == "s" and e["junc_dist"] is not None}.items())
        st = np.array([a for a, _ in sl], float)
        sv = np.array([b for _, b in sl], float)
        idx = np.clip(np.searchsorted(st, self.t, side="right") - 1, 0, len(st) - 1)
        self.jd = sv[idx] if len(st) else np.full(len(self.t), 999.0)
        # VLM answers, deduplicated by query time (old-format logs repeat the latest answer at every plan step)
        ans = {}
        for e in d["vlm"]:
            if e["k"] == "a" and e.get("ok", True) and e.get("tq") is not None:
                ans[e["tq"]] = e
        self.ans = [ans[k] for k in sorted(ans)]
        self.at = np.array([a["t"] for a in self.ans], float)

    def vlm_at(self, t):
        i = int(np.searchsorted(self.at, t, side="right")) - 1
        if i < 0 or t - self.at[i] > VLM_TTL:
            return None
        return self.ans[i]

    def lead_ok(self, k, static_head=False, mode="head"):
        """A lead/blocker is reported ahead. mode head: openpilot lead head (lead_prob[0] > 0.5, 0 < x < 40 m; with static_head the head's own
        speed estimate |v| < 0.5 m/s as well); vlm: the latest logged Q_block answer is moving_lead or static_block."""
        if mode == "vlm":
            a = self.vlm_at(self.t[k])
            return a is not None and a.get("Qb") in ("moving_lead", "static_block")
        ok = bool(self.lp0[k] > LEAD_P and 0.0 < self.lead_x[k] < LEAD_D)
        return ok and (not static_head or abs(self.lead_vh[k]) < 0.5)


def obstacle_ranges(runs):
    out = {}
    for (arm, seed, route), d in runs.items():
        if arm == "pbyp" and route in TARGET:
            for p in d["priv"]:
                if p["st"] and p["st"]["start_s"] > 20:      # the first state past the spawn area is the scenario obstacle
                    out[(seed, route)] = (p["st"]["start_s"], p["st"]["end_s"])
                    break
    return out


def label_episodes(dr, v_stop, obs):
    """Stop episodes of a drive run with ground-truth cause: obstacle (target route, scenario obstacle within OBS_AHEAD_M ahead),
    red_light (ego light red or yellow within 50 m for at least half of the episode), junction (within 15 m of an entrance or inside one),
    lead (a vehicle within 40 m ahead per the simulator), other (no cause in the logs)."""
    out = []
    d = dr.d
    for a, b, trunc in episodes(dr.t, dr.v, v_stop):
        m = (dr.t >= a) & (dr.t <= b)
        k0 = int(np.flatnonzero(m)[0])
        es = float(dr.es[k0])
        o = obs.get((d["seed"], d["route"]))
        od = None if o is None else o[0] - es
        redf = float(np.mean(((dr.tl[m] == 1) | (dr.tl[m] == 2)) & (dr.td[m] < RED_X_TRUTH)))
        if od is not None and 0 <= od <= OBS_AHEAD_M:
            cause = "obstacle"
        elif redf >= 0.5:
            cause = "red_light"
        elif dr.jd[k0] <= 15.0:
            cause = "junction"
        elif np.nanmin(np.where(np.isnan(dr.lead_gap_truth[m]), 1e9, dr.lead_gap_truth[m])) < LEAD_D:
            cause = "lead"
        else:
            cause = "other"
        out.append(dict(route=d["route"], seed=d["seed"], t0=a, t1=b, dur=b - a, trunc=trunc, ego_s=es, obs_d=od, cause=cause, k0=k0,
                        k1=int(np.flatnonzero(m)[-1])))
    return out


def detect(dr, ep, T, light, N, X=None, static_head=False, lead_mode="head"):
    """First time in the episode at which the time-based detector fires, else None.
    Fires when the ego has been below the episode's speed bound for T s, a lead is reported ahead by the openpilot lead head
    (probability > 0.5, x < 40 m; with static_head also |lead speed| < 0.5 m/s), and neither suppression holds:
    light: None | 'truth' (ctx light red or yellow closer than X m) | 'vlm' (latest logged Q_light answer is red_or_yellow_for_ego)
    N: None | no firing within N m before a junction entrance or inside one (from the route map)."""
    for k in range(ep["k0"], ep["k1"] + 1):
        t = dr.t[k]
        if t - ep["t0"] < T:
            continue
        if not dr.lead_ok(k, static_head, lead_mode):
            continue
        if light == "truth" and dr.tl[k] in (1, 2) and dr.td[k] < X:
            continue
        if light == "vlm":
            a = dr.vlm_at(t)
            if a is not None and a.get("Ql") == "red_or_yellow_for_ego":
                continue
        if N is not None and dr.jd[k] <= N:
            continue
        return float(t)
    return None


def step_cause(dr, k, obs):
    """Ground-truth label of a plan step (for fires of the window variant)."""
    d = dr.d
    o = obs.get((d["seed"], d["route"]))
    if o is not None and 0 <= o[0] - dr.es[k] <= OBS_AHEAD_M:
        return "obstacle"
    if dr.tl[k] in (1, 2) and dr.td[k] < RED_X_TRUTH:
        return "red_light"
    if dr.jd[k] <= 15.0:
        return "junction"
    if not np.isnan(dr.lead_gap_truth[k]) and dr.lead_gap_truth[k] < LEAD_D:
        return "lead"
    return "other"


def suppressed(dr, k, light, N, X):
    t = dr.t[k]
    if light == "truth" and dr.tl[k] in (1, 2) and dr.td[k] < X:
        return True
    if light == "vlm":
        a = dr.vlm_at(t)
        if a is not None and a.get("Ql") == "red_or_yellow_for_ego":
            return True
    return N is not None and dr.jd[k] <= N


def window_fires(dr, vbar, T, light, N, X=50.0, static_head=True, refractory=10.0, lead_mode="head"):
    """Creep-tolerant variant: fires when the mean ego speed of the last T s is below vbar, a lead is reported ahead (static_head: and the
    lead head's own speed estimate is below 0.5 m/s) and no suppression holds; at most one fire per refractory period."""
    out, last = [], -1e9
    cs = np.r_[0.0, np.cumsum(dr.v)]
    for k in range(len(dr.t)):
        t = dr.t[k]
        if t < T_REF + T or t - last < refractory:
            continue
        j = int(np.searchsorted(dr.t, t - T))
        if (cs[k + 1] - cs[j]) / max(k + 1 - j, 1) >= vbar:
            continue
        if not dr.lead_ok(k, static_head, lead_mode) or suppressed(dr, k, light, N, X):
            continue
        out.append(k)
        last = t
    return out


def score_stop(drs, eps_by_v, v_stop, T, light, N, X, static_head, lead_mode="head"):
    """Per stop episode scoring of the strict detector (eps_by_v[v_stop]: labelled episodes of every run)."""
    return [dict(e, fire=detect(drs[(e["seed"], e["route"])], e, T, light, N, X, static_head, lead_mode)) for e in eps_by_v[v_stop]]


def summarize_stop(res, T, eps_all=None):
    by = defaultdict(Counter)
    for r in res:
        by[r["cause"]]["fire" if r["fire"] is not None else "no"] += 1
        if r["cause"] == "obstacle" and r["dur"] >= T:
            by["obstacle_long"]["fire" if r["fire"] is not None else "no"] += 1
    runs = defaultdict(list)
    for r in res:
        if r["cause"] == "obstacle":
            runs[(r["seed"], r["route"])].append(r)
    delays, hit = [], 0
    for k, g in runs.items():
        g.sort(key=lambda r: r["t0"])
        f = [r["fire"] for r in g if r["fire"] is not None]
        if f:
            hit += 1
            delays.append(min(f) - g[0]["t0"])
    fp_routes = Counter(r["route"] for r in res if r["cause"] != "obstacle" and r["fire"] is not None)
    n_ge = Counter(r["cause"] for r in res if r["dur"] >= T)
    fp = {c: by[c]["fire"] for c in ("red_light", "junction", "lead", "other")}
    return dict(by=by, runs_n=len(runs), runs_hit=hit, delays=delays, fp_routes=fp_routes, n_ge=n_ge, fp=fp, fp_total=sum(fp.values()),
                ob_all=(by["obstacle"]["fire"], by["obstacle"]["fire"] + by["obstacle"]["no"]),
                ob_long=(by["obstacle_long"]["fire"], by["obstacle_long"]["fire"] + by["obstacle_long"]["no"]),
                delay_med=float(np.median(delays)) if delays else float("nan"), delay_max=float(max(delays)) if delays else float("nan"))


def vlm_behaviour(drs, obs, runs):
    """Logged openjev answers (Q_block, Q_side) at the time the frame was queried, with the ego stopped (v < 0.3) and the scenario
    obstacle ahead within OBS_AHEAD_M, by obstacle type and distance bucket; plus Q_block during non-obstacle stops."""
    side_gt = {}
    for (arm, seed, route), d in runs.items():
        if arm == "pbyp" and route in TARGET:
            for p in d["priv"]:
                if p["st"] and p["st"]["start_s"] > 20:
                    side_gt[route] = "right_free" if p["st"]["offset"] < 0 else "left_free"
                    break
    buckets = [(0, 5), (5, 10), (10, 20), (20, 45)]
    rows = defaultdict(list)       # (type, bucket) -> list of (Qb, Qs, Pb_static)
    other = defaultdict(list)      # cause -> list of Qb
    moving = defaultdict(list)     # type -> list of Qb while the ego approaches (v >= 0.3, obstacle within 45 m)
    for (seed, route), dr in sorted(drs.items()):
        o = obs.get((seed, route))
        eps = label_episodes(dr, V_STOP, obs)
        for a in dr.ans:
            tq = a["tq"]
            k = int(np.abs(dr.t - tq).argmin())
            od = None if o is None else o[0] - dr.es[k]
            ps = (a.get("Pb") or {}).get("static_block")
            if o is not None and od is not None and 0 <= od <= OBS_AHEAD_M:
                if dr.v[k] < V_STOP:
                    b = next(i for i, (lo, hi) in enumerate(buckets) if lo <= od < hi or (hi == 45 and od <= hi))
                    rows[(OBST_TYPE[route], route, b)].append((a["Qb"], a["Qs"], ps, side_gt[route]))
                else:
                    moving[OBST_TYPE[route]].append(a["Qb"])
            elif dr.v[k] < V_STOP:
                ep = next((e for e in eps if e["t0"] <= dr.t[k] <= e["t1"] and e["dur"] >= 2.0), None)
                if ep is not None:
                    other[ep["cause"]].append(a["Qb"])
    return rows, other, moving, buckets


GRID_V = (0.3, 0.5, 1.0)
GRID_T = (2, 3, 5, 8, 12)
GRID_LIGHT = (("none", None, None), ("truth", "truth", 20.0), ("truth", "truth", 50.0), ("vlm", "vlm", None))
GRID_N = (None, 15, 30)
GRID_LEAD = (("head", False, "head"), ("head_static", True, "head"), ("vlm", False, "vlm"))


def name_cfg(v, T, lname, N, lead):
    return "v%.1f T%d light=%s N=%s lead=%s" % (v, T, lname, "off" if N is None else N, lead)


def pareto(rows):
    """Rows with all obstacle runs hit that no other row beats on (fp_total, delay_med, obstacle recall >= T)."""
    keep = []
    cand = [r for r in rows if r["S"]["runs_hit"] == r["S"]["runs_n"]]
    for r in cand:
        a = (r["S"]["fp_total"], r["S"]["delay_med"], -r["S"]["ob_long"][0] / max(r["S"]["ob_long"][1], 1))
        dom = any(((q["S"]["fp_total"], q["S"]["delay_med"], -q["S"]["ob_long"][0] / max(q["S"]["ob_long"][1], 1)) <= a) and
                  ((q["S"]["fp_total"], q["S"]["delay_med"], -q["S"]["ob_long"][0] / max(q["S"]["ob_long"][1], 1)) != a) for q in cand)
        if not dom:
            keep.append(r)
    return keep


def fmt(x, n=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%.*f" % (n, x) if isinstance(x, float) else str(x))


def mdtable(head, rows, left=(0,)):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(":--" if i in left else "--:" for i in range(len(head))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main(ex_dir, res_dir):
    runs = load_all(ex_dir)
    obs = obstacle_ranges(runs)
    drs = {}
    for (arm, seed, route), d in sorted(runs.items()):
        if arm == "drive":
            drs[(seed, route)] = Drive(d, junction_table(runs, route))
    eps_by_v = {v: [e for _, dr in sorted(drs.items()) for e in label_episodes(dr, v, obs)] for v in GRID_V}
    md = []
    # ----------------------------------------------------------------------------------------- episodes
    e0 = eps_by_v[0.3]
    md.append("## 1. Stop episodes in the 38 drive runs (ego v < 0.3 m/s for at least 1 s, counted from 6.5 s)\n")
    body = []
    for c in ("obstacle", "red_light", "junction", "lead", "other"):
        g = [e for e in e0 if e["cause"] == c]
        body.append([c, len(g), sum(e["dur"] >= 3 for e in g), sum(e["dur"] >= 5 for e in g), sum(e["dur"] >= 8 for e in g),
                     fmt(float(np.median([e["dur"] for e in g])) if g else float("nan")), fmt(max([e["dur"] for e in g]) if g else float("nan")),
                     sum(e["trunc"] for e in g), len(set(e["route"] for e in g))])
    body.append(["all", len(e0), sum(e["dur"] >= 3 for e in e0), sum(e["dur"] >= 5 for e in e0), sum(e["dur"] >= 8 for e in e0),
                 fmt(float(np.median([e["dur"] for e in e0]))), fmt(max(e["dur"] for e in e0)), sum(e["trunc"] for e in e0),
                 len(set(e["route"] for e in e0))])
    md.append(mdtable(["cause (ground truth)", "episodes", ">= 3 s", ">= 5 s", ">= 8 s", "median [s]", "max [s]", "ran to end of run", "routes"], body))
    md.append("\nObstacle episodes by route (stop episodes with the scenario obstacle within 45 m ahead; durations in s, `->` = never left):\n")
    body = []
    for route in sorted(TARGET):
        for seed in (0, 1):
            g = [e for e in e0 if e["route"] == route and e["seed"] == seed and e["cause"] == "obstacle"]
            body.append([route, seed, OBST_TYPE[route], len(g), ", ".join(("%.0f->" % e["dur"]) if e["trunc"] else "%.0f" % e["dur"] for e in g),
                         ", ".join("%.0f" % e["obs_d"] for e in g)])
    md.append(mdtable(["route", "seed", "obstacle", "episodes", "durations", "obstacle distance at start [m]"], body, left=(0, 2, 4, 5)))
    md.append("\nHow to read: 30 of the 111 stop episodes have the scenario obstacle ahead, but only 17 last 3 s or longer. The drive ego approaches an obstacle in "
              "stop-and-go steps (a 2 s stop, a creep at 2 to 4 m/s, the next stop), then sits behind it until the 200 s run limit. Red-light stops "
              "(25) and junction stops (21) dominate the rest; 35 stops have no cause in the logs (mostly 2 s stalls).")

    # ----------------------------------------------------------------------------------------- grid
    rows = []
    for v in GRID_V:
        for T in GRID_T:
            for lname, light, X in GRID_LIGHT:
                for N in GRID_N:
                    for lead, sh, lm in GRID_LEAD:
                        res = score_stop(drs, eps_by_v, v, T, light, N, X, sh, lm)
                        rows.append(dict(cfg=(v, T, lname, N, lead), S=summarize_stop(res, T)))
    with open(res_dir / "bypass_timedet_grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["v_stop", "T_s", "light", "N_m", "lead", "obstacle_runs_hit", "obstacle_runs", "obstacle_ep_fired", "obstacle_ep", "obstacle_ep_ge_T_fired",
                    "obstacle_ep_ge_T", "delay_median_s", "delay_max_s", "fp_red_light", "fp_junction", "fp_lead", "fp_other", "fp_total", "nonobstacle_ep_ge_T"])
        for r in rows:
            v, T, lname, N, lead = r["cfg"]
            S = r["S"]
            w.writerow([v, T, lname, "off" if N is None else N, lead, S["runs_hit"], S["runs_n"], S["ob_all"][0], S["ob_all"][1], S["ob_long"][0], S["ob_long"][1],
                        "%.1f" % S["delay_med"], "%.1f" % S["delay_max"], S["fp"]["red_light"], S["fp"]["junction"], S["fp"]["lead"], S["fp"]["other"],
                        S["fp_total"], sum(n for c, n in S["n_ge"].items() if c != "obstacle")])
    md.append("\n## 2. Time-based detector, per stop episode\n")
    md.append("Detector: ego below the speed bound v for T s, a lead reported ahead, and neither suppression. Lead: openpilot lead head "
              "(`plans.jsonl` fields `lead[0] = [x, y, v, a]` and `lp[0]`; reported when `lp[0] > 0.5` and `0 < x < 40 m`; `head_static` also needs the head's own "
              "speed estimate below 0.5 m/s; `vlm` uses the logged openjev Q_block answer being moving_lead or static_block). Light suppression: `truth` = "
              "`ctx.tl` red or yellow and `ctx.tl_dist` below X (privileged), `vlm` = latest logged openjev Q_light answer (at most 2.5 s old) is "
              "red_or_yellow_for_ego, `none`. Junction suppression N: route-map distance to the next junction entrance at most N m (or inside one), from the "
              "logged `junc_dist` (not privileged). Counts are raw; a false trigger is a stop episode that is not an obstacle stop in which the detector fires.\n")
    sel = [(0.5, 3, "none", None, "head"), (0.5, 5, "none", None, "head"), (0.5, 5, "truth", 15, "head"), (0.5, 5, "vlm", 15, "head"),
           (0.5, 5, "vlm", None, "head"), (0.5, 5, "truth", None, "head"), (0.5, 5, "vlm", 15, "head_static"), (0.5, 5, "vlm", 15, "vlm"),
           (0.5, 3, "vlm", 15, "head"), (0.5, 8, "vlm", 15, "head"), (0.5, 12, "vlm", 15, "head"), (0.5, 2, "vlm", 15, "head"),
           (0.3, 5, "vlm", 15, "head"), (1.0, 5, "vlm", 15, "head"), (0.5, 5, "truth", 30, "head")]
    lookup = {r["cfg"]: r for r in rows}
    body = []
    for cfg in sel:
        S = lookup[cfg]["S"]
        v, T, lname, N, lead = cfg
        nn = sum(n for c, n in S["n_ge"].items() if c != "obstacle")
        body.append(["v<%.1f" % v, T, lname, "off" if N is None else N, lead,
                     "%d / %d" % S["ob_long"], "%d / %d" % S["ob_all"], "%d / %d" % (S["runs_hit"], S["runs_n"]), fmt(S["delay_med"]), fmt(S["delay_max"]),
                     "%d / %d / %d / %d" % (S["fp"]["red_light"], S["fp"]["junction"], S["fp"]["lead"], S["fp"]["other"]), nn,
                     ", ".join("%s %d" % kv for kv in sorted(S["fp_routes"].items())) or "-"])
    md.append(mdtable(["stop bound", "T [s]", "light", "N [m]", "lead", "obstacle episodes >= T fired", "all obstacle episodes fired", "obstacle runs hit",
                       "delay first obstacle stop to first trigger, median [s]", "max [s]", "false triggers: red / junction / lead / other",
                       "non-obstacle episodes >= T", "false-trigger routes (count)"], body, left=(0, 2, 4, 12)))
    md.append("\nHow to read: with T of 3 to 5 s every obstacle stop that lasts that long fires (15 of 17 at 5 s) in all 8 obstacle runs, and the false triggers "
              "are few because the lead head rarely reports a lead at a red light or in a junction; the suppression terms remove the rest (red 1 -> 0, junction 2 -> 0 at T = 5 "
              "s, N = 15 m). Ground-truth light and the logged openjev light answer give the same counts here. The delay from the first obstacle stop is the "
              "T itself when the first stop is a long one (5 of 8 runs) and 20 to 25 s when the ego first takes 2 s stalls on the way in. The trigger cannot fire inside a "
              "2 s stall: 13 of the 30 obstacle episodes (v < 0.3) are shorter than 3 s by construction. The one false trigger left at T = 5 s with both suppressions is 9196 seed 1: the ego stands for 60 s "
              "right after its two official collisions with the fire truck (a collision-induced standstill, so arguably a real blocker, but not a scenario obstacle). "
              "Taking the lead from the VLM answer instead of the lead "
              "head fires on 17 of 17 but adds false triggers (3 at T = 5 s).")
    pf = pareto(rows)
    groups = {}
    for r in pf:
        v, T, lname, N, lead = r["cfg"]
        S = r["S"]
        key = (v, T, N, lead, S["ob_long"], round(S["delay_med"], 1), round(S["delay_max"], 1), S["fp_total"])
        groups.setdefault(key, []).append(lname)
    body = []
    for key, ln in sorted(groups.items(), key=lambda kv: (kv[0][7], kv[0][5], kv[0][1])):
        v, T, N, lead, ol, dm, dx, fpt = key
        body.append(["v<%.1f" % v, T, "off" if N is None else N, lead, "%d / %d" % ol, fmt(dm), fmt(dx), fpt, ", ".join(sorted(set(ln)))])
    md.append("\nNon-dominated settings over the full grid (540 settings, `bypass_timedet_grid.csv`) among those that hit all 8 obstacle runs, ordered by false triggers, then delay:\n")
    md.append(mdtable(["stop bound", "T [s]", "N [m]", "lead", "obstacle episodes >= T fired", "median delay [s]", "max [s]", "false triggers",
                       "light settings with the same result"], body, left=(0, 3, 8)))

    # ---- creep-tolerant window variant
    first = {}
    for key, dr in drs.items():
        g = [e for e in label_episodes(dr, 0.3, obs) if e["cause"] == "obstacle"]
        first[key] = g[0]["t0"] if g else None
    body = []
    for vbar, T, light, N, sh in ((0.5, 5, "vlm", 15, True), (1.0, 5, "vlm", 15, True), (1.5, 5, "vlm", 15, True), (1.0, 8, "vlm", 15, True), (1.0, 5, "vlm", 15, False)):
        fp, hit, dl = Counter(), 0, []
        for key, dr in drs.items():
            fires = window_fires(dr, vbar, T, light, N, 50.0, sh)
            lab = [(dr.t[k], step_cause(dr, k, obs)) for k in fires]
            for t, c in lab:
                if c != "obstacle":
                    fp[c] += 1
            ob = [t for t, c in lab if c == "obstacle"]
            if first[key] is not None and ob:
                hit += 1
                dl.append(min(ob) - first[key])
        body.append(["mean v < %.1f" % vbar, T, light, N, "head_static" if sh else "head", "%d / 8" % hit, fmt(float(np.median(dl))) if dl else "-",
                     fmt(float(max(dl))) if dl else "-", sum(fp.values()), ", ".join("%s %d" % kv for kv in sorted(fp.items()))])
    md.append("\nCreep-tolerant variant (mean ego speed over the last T s below a bound instead of a continuous stop; at most one fire per 10 s; a fire is false when no "
              "obstacle is within 45 m ahead; scored per fire, not per episode):\n")
    md.append(mdtable(["rule", "T [s]", "light", "N [m]", "lead", "obstacle runs hit", "median delay [s]", "max [s]", "false fires", "by cause"], body, left=(0, 2, 4, 9)))
    md.append("\nHow to read: averaging over the creep removes the 20 s delays only by loosening the stop test, and then slow following of a moving lead fires "
              "(cause `other`); the strict per-episode detector is the better trade on these logs.")

    # ----------------------------------------------------------------------------------------- VLM behaviour
    vr, other, moving, buckets = vlm_behaviour(drs, obs, runs)
    md.append("\n## 3. What the single-frame model answers once the ego is stopped behind the obstacle\n")
    md.append("Logged openjev answers (the only model in these logs), frame time inside an obstacle stop (ego v < 0.3 m/s, scenario obstacle within 45 m ahead). "
              "Truth for Q_block is static_block; truth for Q_side is left_free on all four routes (the path offset is negative, which is the left lane in CARLA's "
              "left-handed frame, and the new-format logs say left_free; the old-format `gt_side` of the seed-0 logs says right_free, an inconsistent label that "
              "was not used). Answers in one stopped scene are near-duplicates, so the number of runs is the effective sample.\n")
    body = []
    agg = defaultdict(list)
    for (typ, route, b), lst in vr.items():
        agg[(typ, b)] += [(x, route) for x in lst]
    for typ in ("ConstructionObstacle", "Accident", "ParkedObstacle"):
        for b, (lo, hi) in enumerate(buckets):
            lst = agg.get((typ, b))
            if not lst:
                continue
            qb = Counter(x[0][0] for x in lst)
            qs = Counter(x[0][1] for x in lst)
            sb = [x[0] for x in lst if x[0][0] == "static_block"]
            nrun = len(set((x[1]) for x in lst))
            body.append([typ, "%d-%d" % (lo, hi), len(lst), nrun, "%d (%.0f%%)" % (qb["static_block"], 100.0 * qb["static_block"] / len(lst)),
                         qb["moving_lead"], qb["clear"], "%d (%.0f%%)" % (qs["left_free"], 100.0 * qs["left_free"] / len(lst)), qs["none_free"], qs["right_free"],
                         "%d / %d" % (sum(1 for x in sb if x[1] == "left_free"), len(sb))])
    md.append(mdtable(["obstacle", "distance [m]", "answers", "routes", "Q_block static_block", "moving_lead", "clear", "Q_side left_free (correct)", "none_free",
                       "right_free", "left_free among static_block answers"], body, left=(0, 1)))
    md.append("\nQ_block during other stops (ego v < 0.3 m/s inside a stop episode of at least 2 s without the obstacle):\n")
    body = []
    for c in ("red_light", "junction", "other"):
        q = Counter(other.get(c, []))
        n = sum(q.values())
        body.append([c, n, "%d (%.0f%%)" % (q["static_block"], 100.0 * q["static_block"] / max(n, 1)), q["moving_lead"], q["clear"]])
    md.append(mdtable(["stop cause", "answers", "static_block", "moving_lead", "clear"], body))
    mv = {k: Counter(v) for k, v in moving.items()}
    md.append("\nWhile the ego is still moving towards the obstacle (v >= 0.3 m/s, obstacle within 45 m): " + "; ".join(
        "%s %d answers, static_block %d" % (k, sum(c.values()), c["static_block"]) for k, c in sorted(mv.items())) + ".")
    md.append("\nHow to read: for the cone-and-barrier construction zones the model names the obstacle (static_block) in 99% of the stopped frames within 10 m "
              "and in 37% of those 20 to 45 m away; for the accident vehicles and the parked vehicle it answers moving_lead in 100% of the stopped frames at every "
              "distance, for stops of up to about 170 s. The free side is almost never named: Q_side is none_free in 87% to 100% of the construction frames, "
              "left_free (correct) in 13% at 0 to 5 m, 7% at 5 to 10 m and never beyond, and none_free for every vehicle-obstacle frame. So a stopped ego "
              "does not make the single frame readable; the time-based trigger has to supply the static decision, and the side has to come from elsewhere "
              "(inferred: these are openjev answers; Qwen3-VL-4B was not run in closed loop).")
    return md, rows, lookup, eps_by_v, drs


HEADER = """# bypass_timedet: can "static" be decided by time instead of by a single-frame model?

Offline analysis of the 38 `drive` runs (19 routes x 2 traffic seeds; the reference arm that stops in front of obstacles and queues alike).
No CARLA, no GPU, no new driving. Definitions (stop episode, causes, detector) are in
[plans/2026-10-02-bypass-offline.md](../plans/2026-10-02-bypass-offline.md); code `scripts/bypass_timedet.py` (+ `bypass_common.py`,
`bypass_extract.py`); the full 540-setting grid is `bypass_timedet_grid.csv`. Verified = computed from the logs; inferred is stated in the text.

"""


if __name__ == "__main__":
    res_dir = Path(sys.argv[2])
    md, rows, lookup, eps_by_v, drs = main(sys.argv[1], res_dir)
    (res_dir / "bypass_timedet.md").write_text(HEADER + "\n".join(md) + "\n")
    print("written")
