"""Tables for results/release_replay.md from tmp/vlm_arb_offline/release_replay.pkl (numpy, Python 3.9).

  python3 release_replay_report.py   -> prints the markdown tables and writes results/release_replay_policies.csv, release_replay_episodes.csv,
                                        release_replay_infractions.csv
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_replay as rr  # noqa: E402
import route_common as rc  # noqa: E402

ROOT = rr.ROOT
RES = os.path.join(ROOT, "experiments/vlm_arb/results")
POSN = ["A before stop line", "B between stop line and entrance", "C past entrance"]


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(":--" if i == 0 else "--:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, n=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%.*f" % (n, x))


def pol_table(P, split=None):
    """One row per policy over the original (non-rerun) episodes."""
    rows = defaultdict(list)
    for r in P:
        rows[r["policy"]].append(r)
    out = []
    for name, rs in rows.items():
        n = len(rs)
        ff = sum(r["first_false"] for r in rs)
        fe = sum(r["n_false_events"] for r in rs)
        g = [r for r in rs if r["has_green"]]
        rel = [r["delay"] for r in g if r["delay"] is not None]
        miss = len(g) - len(rel)
        gt1 = sum(1 for r in g if r["delay"] is None or r["delay"] > 1.0)
        gt2 = sum(1 for r in g if r["delay"] is None or r["delay"] > 2.0)
        out.append(dict(policy=name, episodes=n, first_false=ff, false_events=fe, green_eps=len(g), released=len(rel), missed=miss,
                        med=float(np.median(rel)) if rel else float("nan"), p90=float(np.percentile(rel, 90)) if rel else float("nan"),
                        gt1=gt1, gt2=gt2))
    return out


def infraction_audit(docs, infos):
    rows = []
    for d in docs:
        for m in d["infractions"].get("red_light", []):
            x, y = map(float, re.findall(r"x=([-\d.]+), y=([-\d.]+)", m)[0])
            ego = np.array(d["ego"], float)
            ts, ss = rc.ego_s_at(d)
            t_pole = float(ego[np.hypot(ego[:, 1] - x, ego[:, 2] - y).argmin(), 0])
            info = infos[d["route"]]
            # the junction whose entrance the car passes closest to the pole time
            es_pole = float(np.interp(t_pole, ts, ss))
            cand = sorted((abs(E - es_pole), j) for j, E in info["entries"].items())
            jid = cand[0][1] if cand else None
            E = info["entries"].get(jid)
            L = rc.governing_light(info, jid)
            TAIL_M = 2.4508 - 1.3886          # rear axle -> tail
            def first(arc):
                idx = np.nonzero(ss - TAIL_M >= arc)[0]
                return float(ts[idx[0]]) if len(idx) else None
            t_line = first(L["arc"]) if L is not None else None          # tail passes the stop line
            t_cross = first(E) if E is not None else None                # tail passes the junction entrance
            st_line = rc.light_state(L, d["attempt"], t_line, 1.0) if (L and t_line) else None
            st_cross = rc.light_state(L, d["attempt"], t_cross, 1.0) if (L and t_cross) else None
            # the scorer's line is not known exactly: take the first of the two candidates at which the light was red
            t_x = next((tt for tt, st in ((t_line, st_line), (t_cross, st_cross)) if tt is not None and st == 2), None)
            if t_x is None:
                t_x = t_cross if t_cross is not None else t_pole
            eps = [e for e in rr.episodes_of(d) if e["jid"] == jid]
            tref = t_x
            prev = [(e, c) for e in eps for c in e["causes"] if c[1] <= tref + 0.6]
            last_seg = prev[-1][1] if prev else None
            active = any(e["t0"] <= tref <= e["t1"] + 0.6 for e in eps)
            false_rel = []
            for e in eps:
                for c in e["causes"]:
                    if c[0] == "green":
                        st = rc.light_state(L, d["attempt"], c[2]) if L else None
                        false_rel.append((c[2], st))
            if active:
                cat = "held (R2 active) while the car crossed"
            elif last_seg is None:
                cat = "no R2 hold before the crossing"
            elif last_seg[0] == "green" and any(t == last_seg[2] and s not in (0, None) for t, s in false_rel):
                cat = "false green release, then crossed"
            elif last_seg[0] == "r5":
                cat = "R5 25 s fallback released a red light"
            elif last_seg[0] == "green":
                cat = "green release, light not green at crossing"
            else:
                cat = "other (%s)" % last_seg[0]
            if cat == "green release, light not green at crossing":
                cat = "correct green release, light red again before the tail cleared"
            # could a faster release have saved it? same motion after the release, release moved to the green onset (instant detection)
            margin = None
            if cat.startswith("correct green release") and L is not None and last_seg is not None and last_seg[2] is not None:
                tt, vv = L["tl"][d["attempt"]]
                ch = [tt[k] for k in range(1, len(tt)) if vv[k] != 0 and vv[k - 1] == 0 and tt[k] <= t_x + 0.3]
                gs = [tt[k] for k in range(1, len(tt)) if vv[k] == 0 and vv[k - 1] != 0 and tt[k] <= last_seg[2] + 0.3]
                if ch and gs:
                    red_on, t_g0 = ch[-1], gs[-1]
                    margin = round((red_on - (t_x - last_seg[2])) - t_g0, 2)      # >= 0: an instant release at the green onset would have cleared in time
            rows.append(dict(arm=d["arm"], margin=margin, seed=d["seed"], route=d["route"], unit=d["unit"], t_pole=round(t_pole, 1),
                             t_line=None if t_line is None else round(t_line, 1), t_cross=None if t_cross is None else round(t_cross, 1), t_x=round(t_x, 1),
                             truth_line=st_line, truth_cross=st_cross, category=cat, last_segment=None if last_seg is None else (last_seg[0], round(last_seg[1], 1)),
                             false_releases="; ".join("%.1f(truth %s)" % (t, s) for t, s in false_rel if s not in (0, None)),
                             rerun=int(d["arm"].startswith("gif_"))))
    return rows


def main():
    P = pickle.load(open(os.path.join(ROOT, "tmp/vlm_arb_offline/release_replay.pkl"), "rb"))
    eps = P["eps"]
    orig = [e for e in eps if not e["rerun"]]
    pol = P["pol_rows"]
    lines = []
    # ---- episode inventory
    inv = defaultdict(lambda: Counter())
    for e in orig:
        a = inv[e["arm"]]
        a["episodes"] += 1
        a["no_light"] += e["no_light"]
        if not e["no_light"]:
            a["green_onset"] += int(e["t_g"] is not None)
            a["logged_releases"] += e["logged_releases"]
            a["logged_false"] += e["logged_false"]
            a["start_nongreen"] += int(e["truth_start"] in (1, 2))
            a["start_yellow"] += int(e["truth_start"] == 1)
            a["r5_end"] += int("r5" in e["end_causes"])
    lines.append("### Episodes (original runs)\n")
    lines.append(md([[a, v["episodes"], v["no_light"], v["start_nongreen"], v["start_yellow"], v["green_onset"], v["logged_releases"], v["logged_false"], v["r5_end"]]
                     for a, v in inv.items()],
                    ["arm", "episodes", "hold without a governing light", "truth not green at start", "of which yellow", "truth turned green in window", "logged K=2 releases",
                     "logged false releases", "episodes with an R5 end"]))
    # ---- policies, overall
    T = pol_table(pol)
    lines.append("\n### Policies, all positions\n")
    lines.append(md([[t["policy"], t["episodes"], t["first_false"], t["false_events"], t["green_eps"], t["released"], t["missed"], f(t["med"]), f(t["p90"]), t["gt1"], t["gt2"]] for t in T],
                    ["policy", "episodes", "episodes with a false first release", "false release events (re-armed)", "episodes with a green window", "released in it", "missed",
                     "delay median s", "delay p90 s", "n > 1 s (missed incl.)", "n > 2 s (missed incl.)"]))
    with open(os.path.join(RES, "release_replay_policies.csv"), "w") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "episodes", "first_false", "false_events", "green_eps", "released", "missed", "delay_med", "delay_p90", "gt1", "gt2"])
        for t in T:
            w.writerow([t["policy"], t["episodes"], t["first_false"], t["false_events"], t["green_eps"], t["released"], t["missed"], f(t["med"], 3), f(t["p90"], 3), t["gt1"], t["gt2"]])
    # ---- policies by position
    lines.append("\n### Policies by position\n")
    lines.append("False = first release while the truth light was red or yellow, split by the position of the car at that answer. Delay and missed split by the position when the light turned green.\n")
    rows = []
    for name in dict.fromkeys(r["policy"] for r in pol):
        rs = [r for r in pol if r["policy"] == name]
        cells = [name]
        for grp in ((0,), (1, 2)):
            ff = sum(1 for r in rs if r["first_false"] and r["first_pos"] in grp)
            g = [r for r in rs if r["has_green"] and r["pos_g"] in grp]
            rel = [r["delay"] for r in g if r["delay"] is not None]
            cells += [ff, len(g), len(g) - len(rel), f(float(np.median(rel)) if rel else float("nan")), sum(1 for r in g if r["delay"] is None or r["delay"] > 2.0)]
        rows.append(cells)
    lines.append(md(rows, ["policy", "A: false first rel.", "A: green eps", "A: missed", "A: delay med s", "A: n > 2 s",
                           "B+C: false first rel.", "B+C: green eps", "B+C: missed", "B+C: delay med s", "B+C: n > 2 s"]))
    # ---- by route
    lines.append("\n### Selected policies by route\n")
    sel = ["K=1", "K=2 (current)", "K=3", "cusum >= 5", "single p>=0.99", "position-gated: K=2 before the line, K=4 and p>=0.99 past it"]
    routes = sorted({r["route"] for r in pol})
    rows = []
    for ro in routes:
        cells = [ro, len({(r["arm"], r["seed"], r["t0"]) for r in pol if r["route"] == ro})]
        for s in sel:
            rs = [r for r in pol if r["route"] == ro and r["policy"] == s]
            g = [r for r in rs if r["has_green"]]
            cells.append("%d / %d" % (sum(r["first_false"] for r in rs), sum(1 for r in g if r["delay"] is None)))
        rows.append(cells)
    lines.append("Cells: false first releases / missed green windows.\n")
    lines.append(md(rows, ["route", "episodes"] + sel))
    # ---- by arm
    lines.append("\n### Selected policies by arm\n")
    sel2 = ["K=1", "K=2 (current)", "K=3", "cusum >= 5", "single p>=0.99", "position-gated: K=2 before the line, K=4 and p>=0.99 past it"]
    rows = []
    for arm in rr.ORIGINAL:
        for s in sel2:
            rs = [r for r in pol if r["arm"] == arm and r["policy"] == s]
            g = [r for r in rs if r["has_green"]]
            rel = [r["delay"] for r in g if r["delay"] is not None]
            rows.append([arm, s, len(rs), sum(r["first_false"] for r in rs), len(g) - len(rel), f(float(np.median(rel)) if rel else float("nan")), f(float(np.percentile(rel, 90)) if rel else float("nan")),
                         sum(1 for r in g if r["delay"] is None or r["delay"] > 2.0)])
    lines.append(md(rows, ["arm", "policy", "episodes", "false first releases", "missed", "delay median s", "delay p90 s", "n > 2 s (missed incl.)"]))
    with open(os.path.join(RES, "release_replay_by_episode.csv"), "w") as fh:
        keys = ["policy", "arm", "route", "seed", "t0", "n_events", "n_false_events", "first_false", "first_pos", "first_te", "first_truth", "has_green", "tg", "delay", "pos_g", "green_len"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(pol)
    # ---- model green probability on truly red
    def tab(ans, title):
        rows = []
        for p in (0, 1, 2):
            red = [a for a in ans if a["pos"] == p and a["truth"] in (1, 2)]
            grn = [a for a in ans if a["pos"] == p and a["truth"] == 0]
            if not red and not grn:
                continue
            pg = np.array([a["pg"] for a in red]) if red else np.array([])
            pgg = np.array([a["pg"] for a in grn]) if grn else np.array([])
            rows.append([POSN[p], len(red), len({(a["arm"], a["route"]) for a in red}),
                         *(["%.1f%%" % (100 * (pg >= th).mean()) if len(pg) else "-" for th in (0.5, 0.9, 0.99)]),
                         "%.1f%%" % (100 * np.mean([a["lab"] == "green_for_ego" for a in red])) if red else "-",
                         len(grn), "%.1f%%" % (100 * (pgg >= 0.5).mean()) if len(pgg) else "-", "%.1f%%" % (100 * (pgg >= 0.9).mean()) if len(pgg) else "-"])
        return "\n### %s\n\n" % title + md(rows, ["position", "answers on a red or yellow light", "arm-route cells", "p_green >= 0.5", "p_green >= 0.9", "p_green >= 0.99",
                                                  "answered green (argmax)", "answers on a green light", "p_green >= 0.5", "p_green >= 0.9"])
    lines.append(tab(P["hold_ans"], "Model green probability on a truly red or yellow light: answers during holds"))
    lines.append(tab(P["all_ans"], "Same, all logged answers with a signalised junction ahead (stop line <= 50 m ahead or up to 6 m past the entrance)"))
    by_arm = []
    for arm in rr.ORIGINAL:
        for p in (0, 1, 2):
            red = [a for a in P["hold_ans"] if a["arm"] == arm and a["pos"] == p and a["truth"] in (1, 2)]
            if red:
                by_arm.append([arm, POSN[p], len(red), "%.1f%%" % (100 * np.mean([a["pg"] >= 0.5 for a in red])), "%.1f%%" % (100 * np.mean([a["pg"] >= 0.9 for a in red]))])
    lines.append("\n### Hold answers on a red or yellow light, by arm and position\n\n" + md(by_arm, ["arm", "position", "answers", "p_green >= 0.5", "p_green >= 0.9"]))
    # ---- infractions
    docs = rc.load_docs(rr.EX)
    by_route = defaultdict(list)
    for d in docs:
        by_route[d["route"]].append(d)
    infos = {r: rc.route_info(v) for r, v in by_route.items()}
    vd = [d for d in docs if d["arm"] in rr.ORIGINAL or d["arm"].startswith("gif_vred")]
    aud = infraction_audit(vd, infos)
    with open(os.path.join(RES, "release_replay_infractions.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(aud[0].keys()))
        w.writeheader()
        w.writerows(aud)
    lines.append("\n### Green window lengths (original runs; time from the green onset to the next non-green sample, censored ones dropped)\n")
    gl = [e["green_len"] for e in orig if e.get("green_len") is not None and e["green_len"] < 900]
    lines.append(md([[len(gl), f(min(gl)), f(float(np.median(gl))), f(float(np.percentile(gl, 90))), f(max(gl)), sum(1 for x in gl if x <= 3.5), sum(1 for x in gl if x <= 5.0)]],
                    ["windows", "min s", "median s", "p90 s", "max s", "n <= 3.5 s", "n <= 5 s"]))
    lines.append("\n### Red-light infractions of the vred family\n\n")
    lines.append(md([[a["arm"], a["seed"], a["route"], a["t_line"], a["t_cross"], a["truth_line"], a["truth_cross"], a["category"], a["last_segment"], a["false_releases"] or "-"] for a in
                     sorted(aud, key=lambda a: (a["rerun"], a["arm"], a["route"], a["seed"]))],
                    ["arm", "seed", "route", "tail at stop line s", "tail at entrance s", "truth then (stop line)", "truth then (entrance)", "category", "last R2 segment ended", "false K=2 releases in the approach (s, truth)"]))
    lines.append("\nAvoidability of the correct-release infractions (same motion after the release, release moved to the green onset): margin s = latest release time that still clears before the light turns red, minus the green onset; negative = impossible.\n")
    lines.append(md([[a["arm"], a["seed"], a["route"], a["margin"]] for a in sorted(aud, key=lambda a: (a["arm"], a["route"], a["seed"])) if a["margin"] is not None and not a["rerun"]], ["arm", "seed", "route", "margin s"]))
    # episodes csv
    with open(os.path.join(RES, "release_replay_episodes.csv"), "w") as fh:
        keys = ["arm", "rerun", "seed", "route", "unit", "jid", "t0", "t1", "t_end", "end_causes", "no_light", "n_ans", "truth_start", "t_g", "logged_releases", "logged_false", "logged_release_times"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(eps)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
