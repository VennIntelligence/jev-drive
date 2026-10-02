"""Did the vehicle that hit the pulling-out ego react? (numpy, Python 3.9)

Input : tmp/vlm_arb_offline/tmp_c_ex/*.json.gz (bypass_extract.py on the box: route polyline, 0.2 s privileged snapshots of ego and all actors
        within 70 m, contacts, bypass state) for the arms pbyp, pbyp2, pbyp2ng, drive on the four obstacle routes.
Output: prints markdown tables; writes results/pullout_collisions.csv (one row per vehicle contact), pullout_events.csv (one row per pull-out),
        pullout_headways.csv (one row per vehicle passing the reference line in the adjacent lane).
Frame: arc position s and signed lateral offset (left positive) of every object on the official route polyline (the ego's own lane centre line
while the ego is in lane). Definitions: plans/2026-10-02-offline-analyses-definitions.md section C.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bypass_common as bc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EX = os.path.join(ROOT, "tmp/vlm_arb_offline/tmp_c_ex")
RES = os.path.join(ROOT, "experiments/vlm_arb/results")
EGO_HL, EGO_HW = 2.4508, 0.92          # ego half length and half width, m
LANE_FLOOR = 3.0                        # m, fallback lane separation
WIN = 6.0                               # s before contact
# contacts.jsonl 't' is the world clock, privileged.jsonl 't' the scenario clock: constant per route (measured from the frame numbers of both logs)
CLOCK_OFFSET = {"19324": 1.1, "19832": 1.0, "24497": 1.35, "2520": 1.4}


def tangent_yaw(xy, s_arr, s):
    i = int(np.clip(np.searchsorted(s_arr, s) - 1, 0, len(xy) - 2))
    d = xy[i + 1] - xy[i]
    return float(np.arctan2(d[1], d[0]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Run:
    def __init__(self, d):
        self.d = d
        self.xy = np.array(d["route_xy"], float)
        self.s_arr = bc.route_s(self.xy)
        pr = d["priv"]
        self.t = np.array([p["t"] for p in pr], float)
        self.ego = np.array([p["ego"] for p in pr], float).reshape(-1, 4)
        self.es, self.el = bc.project(self.ego[:, :2], self.xy) if len(pr) else (np.array([]), np.array([]))
        self.tracks = bc.actor_tracks(d)
        for k in self.tracks.values():
            if len(k["t"]):
                k["s"], k["lat"] = bc.project(np.c_[k["x"], k["y"]], self.xy)
                k["hd"] = np.array([abs(wrap(y - tangent_yaw(self.xy, self.s_arr, s))) for y, s in zip(k["yaw"], k["s"])])
        offs = [p["st"]["offset"] for p in pr if p.get("st") and p["st"].get("offset") is not None]
        self.W = abs(float(np.median(offs))) if offs else None
        self.side = float(np.sign(np.median(offs))) if offs else None   # `offset` < 0 is the right side (lat < 0): checked against the hitting vehicles


def contact_episodes(d):
    """First contact time per vehicle id (type vehicle.*), on the scenario clock of privileged.jsonl."""
    first = {}
    off = CLOCK_OFFSET[d["route"]]
    for c in d["contacts"]:
        if str(c.get("type", "")).startswith("vehicle"):
            first[c["id"]] = min(first.get(c["id"], 1e9), c["t"] - off)
    return first


def interp(t, tt, v):
    return float(np.interp(t, tt, v))


def extremity(R, side, W):
    """Lateral position (towards the adjacent lane, from the own lane centre) of the ego body's outermost corner on that side."""
    tang = np.array([tangent_yaw(R.xy, R.s_arr, s) for s in R.es])
    dpsi = wrap(R.ego[:, 2] - tang) * side
    lat = R.el * side
    return lat + EGO_HW * np.cos(dpsi) + EGO_HL * np.maximum(np.sin(dpsi), 0.0), lat


def pullouts(R, W, side):
    """Pull-out events: the body's outermost corner crosses the lane line (W/2) coming from inside the own lane for at least 1 s.
    Returns [(t_pull, t_enter)] with t_pull = last time before the entry at which the ego centre was within 0.15 m of the lane centre."""
    out = []
    if not len(R.t):
        return out
    ext, lat = extremity(R, side, W)
    inside = ext < W / 2.0
    i, last_out = 0, -1
    while i < len(ext):
        if not inside[i]:
            if i > 0 and inside[max(i - 5, 0):i].all():
                j = i
                while j > 0 and lat[j - 1] > 0.15:
                    j -= 1
                out.append((float(R.t[max(j - 1, 0)]), float(R.t[i])))
            while i < len(ext) and not inside[i]:
                i += 1
        else:
            i += 1
    return out


def lane_vehicles(R, t, side, W, ahead_limit=70.0):  # noqa: E302
    """Same-direction vehicles in the adjacent lane band at time t: list of (id, ds, speed, ext0) with ds = other_s - ego_s (signed, centre to centre)."""
    i = int(np.abs(R.t - t).argmin())
    es = R.es[i]
    out = []
    for aid, k in R.tracks.items():
        if not str(k["type"]).startswith("vehicle") or not len(k["t"]):
            continue
        j = int(np.abs(k["t"] - R.t[i]).argmin())
        if abs(k["t"][j] - R.t[i]) > 0.15:
            continue
        lat_rel = k["lat"][j] * side            # towards the adjacent lane positive
        if abs(lat_rel - W) < 1.75 and k["hd"][j] < np.pi / 4 and abs(k["s"][j] - es) < ahead_limit:
            out.append((aid, float(k["s"][j] - es), float(k["v"][j]), float(k["ext"][0])))
    return out


def analyse():
    runs = bc.load_all(EX)
    R = {k: Run(d) for k, d in runs.items() if d["priv"]}
    # lane geometry per route: separation from the bypass offset, side from the hitting vehicles' lateral position
    routeW = {}
    for route in bc.TARGET:
        ws = [r.W for k, r in R.items() if k[2] == route and r.W]
        routeW[route] = float(np.median(ws)) if ws else LANE_FLOOR
    coll, ev = [], []
    for k, r in sorted(R.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        arm, seed, route = k
        d = r.d
        if arm == "drive":
            continue
        W = routeW[route]
        first = contact_episodes(d)
        # side from this run's own offset convention verified on a hitting vehicle (see below); default from the sign of the lateral position of the other
        for oid, tc in sorted(first.items(), key=lambda x: x[1]):
            trk = r.tracks.get(oid)
            row = dict(arm=arm, seed=seed, route=route, other=oid, t_contact=round(tc, 2))
            if trk is None or not len(trk["t"]) or tc < r.t[0]:
                row.update(cls="no track", note="the actor is not in the 70 m snapshots")
                coll.append(row)
                continue
            ie = int(np.abs(r.t - (tc - 0.2)).argmin())
            j = int(np.abs(trk["t"] - (tc - 0.2)).argmin())
            ds = float(trk["s"][j] - r.es[ie])
            lat_rel = float(trk["lat"][j] - r.el[ie])
            hd = float(trk["hd"][j])
            if hd > 3 * np.pi / 4:
                cls = "oncoming"
            elif hd < np.pi / 4:
                cls = "same direction"
            else:
                cls = "crossing / other"
            row.update(cls=cls, ds_at_contact=round(ds, 1), lat_rel=round(lat_rel, 2), v_other=round(float(trk["v"][j]), 2), v_ego=round(float(r.ego[ie, 3]), 2),
                       ego_lat=round(float(r.el[ie]), 2), other_lat=round(float(trk["lat"][j]), 2), bypass=int(bool(d["priv"][ie]["bypass"])), W=W)
            coll.append(row)
    return runs, R, routeW, coll


def bumper_gap(R, ie, trk, j):
    """Bumper-to-bumper gap along the route between the ego and a vehicle behind it (negative: overlapping or alongside / ahead)."""
    return float((R.es[ie] - trk["s"][j]) - EGO_HL - trk["ext"][0])


def accel(trk):
    """Acceleration over 0.6 s windows (three 0.2 s steps), assigned to the window centre (0.2 s differences of the snapshots are too noisy)."""
    v, t = trk["v"], trk["t"]
    a = np.full(len(v), np.nan)
    for n in range(1, len(v) - 2):
        if t[n + 2] - t[n - 1] < 0.9:
            a[n] = (v[n + 2] - v[n - 1]) / (t[n + 2] - t[n - 1])
    return a


def collisions_table(R, routeW):
    rows = []
    for k, r in sorted(R.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        arm, seed, route = k
        if arm == "drive":
            continue
        W = routeW[route]
        side = r.side if r.side else -1.0
        ext_, lat_ = extremity(r, side, W)
        pos = pullouts(r, W, side)
        for oid, tc in sorted(contact_episodes(r.d).items(), key=lambda x: x[1]):
            trk = r.tracks.get(oid)
            if trk is None or not len(trk["t"]) or tc < r.t[0] + 0.3:
                continue
            ie = int(np.abs(r.t - (tc - 0.2)).argmin())
            j = int(np.abs(trk["t"] - (tc - 0.2)).argmin())
            hd = float(trk["hd"][j])
            v_o = float(trk["v"][j])
            if not (hd < np.pi / 4 and v_o >= 3.0 and abs(trk["lat"][j] * side - W) < 1.75):
                continue                               # not a moving same-direction vehicle of the adjacent lane
            tp = [x for x in pos if x[1] <= tc + 0.3]
            t_pull, t_enter = (tp[-1] if tp else (None, None))
            m = (trk["t"] >= tc - WIN) & (trk["t"] <= tc - 0.2)
            a = accel(trk)
            row = dict(arm=arm, seed=seed, route=route, other=oid, t_contact=round(tc, 2), v_other_contact=round(v_o, 1), v_ego_contact=round(float(r.ego[ie, 3]), 1),
                       ego_lat_contact=round(float(lat_[ie]), 2), gap_contact=round(bumper_gap(r, ie, trk, j), 1),
                       t_pull=None if t_pull is None else round(t_pull, 2), t_enter=None if t_enter is None else round(t_enter, 2),
                       misfire_start=int(t_pull is not None and t_pull < 8.0 and arm == "pbyp"))
            for lab, tt in (("pull", t_pull), ("enter", t_enter)):
                if tt is None:
                    continue
                ii = int(np.abs(r.t - tt).argmin())
                jj = int(np.abs(trk["t"] - tt).argmin())
                g = bumper_gap(r, ii, trk, jj)
                vo = float(trk["v"][jj])
                clos = vo - float(r.ego[ii, 3])
                row.update({"gap_m_" + lab: round(g, 1), "v_other_" + lab: round(vo, 1), "v_ego_" + lab: round(float(r.ego[ii, 3]), 1),
                            "time_gap_s_" + lab: None if g <= 0 or vo < 0.5 else round(g / vo, 2),
                            "ttc_s_" + lab: None if g <= 0 or clos <= 0.1 else round(g / clos, 2)})
            if t_enter is not None:
                row["enter_to_contact_s"] = round(tc - t_enter, 2)
            if t_pull is not None:
                row["pull_to_contact_s"] = round(tc - t_pull, 2)
            # braking of the other vehicle
            m2 = m & (trk["t"] <= tc - 0.5)                  # central differences must not reach the impact sample
            vv, tt, aa = trk["v"][m], trk["t"][m], np.where(m2, a, np.nan)[m]
            if len(vv):
                row["v_other_minus6"] = round(float(vv[0]), 1)
                row["min_accel_6s"] = round(float(np.nanmin(aa)), 2) if np.isfinite(aa).any() else None
                row["min_accel_t"] = None if not np.isfinite(aa).any() else round(float(tt[np.nanargmin(aa)] - tc), 2)
                if t_pull is not None:
                    after = tt >= t_pull
                    row["min_accel_after_pull"] = round(float(np.nanmin(aa[after])), 2) if after.any() and np.isfinite(aa[after]).any() else None
                    row["speed_change_pull_to_contact"] = round(float(vv[-1] - trk["v"][int(np.abs(trk["t"] - t_pull).argmin())]), 1)
                    first_brake = tt[after & (aa <= -1.0)]
                    row["first_decel_ge1_after_pull_s"] = None if not len(first_brake) else round(float(first_brake[0] - t_pull), 2)
                    first_hard = tt[after & (aa <= -3.0)]
                    row["first_decel_ge3_after_pull_s"] = None if not len(first_hard) else round(float(first_hard[0] - t_pull), 2)
                if t_enter is not None:
                    row["speed_change_enter_to_contact"] = round(float(vv[-1] - trk["v"][int(np.abs(trk["t"] - t_enter).argmin())]), 1)
                # speed change in the last 1.0 s before contact
                last = tt >= tc - 1.2
                row["speed_change_last_1s"] = round(float(vv[-1] - vv[last][0]), 1) if last.sum() >= 2 else None
            # ego lateral series
            for lag in (6, 4, 2, 1):
                ii = int(np.abs(r.t - (tc - lag)).argmin())
                row["ego_lat_minus%d" % lag] = round(float(lat_[ii]), 2)
            rows.append(row)
    return rows


def events_table(R, routeW, coll):
    """All pull-outs (not only those that ended in a contact). At the pull start and at the lane entry: a vehicle alongside (|centre distance| < ego + other
    half length) and the nearest vehicle approaching from behind (bumper gap > 0) of the adjacent lane, same direction. hit = a moving-vehicle contact
    within 12 s after the entry. t_exit = body back inside the own lane for 0.4 s (None: did not leave before the log ended or a contact)."""
    cset = defaultdict(list)
    for c in coll:
        cset[(c["arm"], c["seed"], c["route"])].append(c["t_contact"])
    rows = []
    for k, r in sorted(R.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        arm, seed, route = k
        if arm == "drive":
            continue
        W = routeW[route]
        side = r.side if r.side else -1.0
        ext_, lat_ = extremity(r, side, W)
        for t_pull, t_enter in pullouts(r, W, side):
            ie = int(np.abs(r.t - t_enter).argmin())
            out = np.nonzero((ext_[ie:] < W / 2.0 - 0.1))[0]
            t_exit = None
            for n in out:
                if (ext_[ie + n:ie + n + 2] < W / 2.0 - 0.1).all():
                    t_exit = float(r.t[ie + n])
                    break
            row = dict(arm=arm, seed=seed, route=route, t_pull=round(t_pull, 2), t_enter=round(t_enter, 2),
                       hit=int(any(t_enter - 0.5 <= tc <= t_enter + 12.0 for tc in cset[k])),
                       misfire_start=int(arm == "pbyp" and t_pull < 8.0), lane_time_s=None if t_exit is None else round(t_exit - t_enter, 1))
            for lab, tt in (("pull", t_pull), ("enter", t_enter)):
                ii = int(np.abs(r.t - tt).argmin())
                lv = lane_vehicles(r, r.t[ii], side, W)
                along = [x for x in lv if abs(x[1]) < EGO_HL + x[3]]
                behind = [(-(x[1]) - EGO_HL - x[3], x[2], x[0]) for x in lv if x[1] < 0 and -(x[1]) - EGO_HL - x[3] > 0]
                ve = float(r.ego[ii, 3])
                row["v_ego_" + lab] = round(ve, 1)
                row["alongside_" + lab] = int(bool(along))
                if behind:
                    g, vo, fid = min(behind, key=lambda z: z[0])
                    if lab == "enter":
                        trk = r.tracks[fid]
                        tend = min([tc for tc in cset[k] if tc >= t_enter - 0.5] + [t_enter + 5.0]) - 0.5
                        mm = (trk["t"] >= t_enter) & (trk["t"] <= min(tend, t_enter + 5.0))
                        aa = accel(trk)[mm]
                        row["follower_id"] = fid
                        row["follower_peak_decel"] = None if not np.isfinite(aa).any() else round(float(-np.nanmin(aa)), 1)
                        row["follower_v_change"] = None if mm.sum() < 2 else round(float(trk["v"][mm][-1] - trk["v"][mm][0]), 1)
                    row.update({"gap_m_" + lab: round(g, 1), "v_other_" + lab: round(vo, 1), "time_gap_s_" + lab: None if vo < 0.5 else round(g / vo, 2),
                                "ttc_s_" + lab: None if vo - ve < 0.1 else round(g / (vo - ve), 2)})
                else:
                    row.update({"gap_m_" + lab: None, "v_other_" + lab: None, "time_gap_s_" + lab: None, "ttc_s_" + lab: None})
            rows.append(row)
    return rows


def headways(R, routeW):
    """Time headway of same-direction vehicles of the adjacent lane at a reference line (the obstacle start), from all runs (ego within 60 m of the line)."""
    rows = []
    for route in bc.TARGET:
        runs = {k: r for k, r in R.items() if k[2] == route}
        srefs = [p["st"]["start_s"] for r in runs.values() for p in r.d["priv"] if p.get("st") and p["st"].get("start_s") is not None]
        if not srefs:
            continue
        sref = float(np.median(srefs))
        W = routeW[route]
        side = next((r.side for r in runs.values() if r.side), -1.0)
        for k, r in sorted(runs.items()):
            if not len(r.t):
                continue
            near = np.abs(r.es - sref) < 60.0
            cross = []
            for aid, tr in r.tracks.items():
                if not str(tr["type"]).startswith("vehicle") or len(tr["t"]) < 2:
                    continue
                for n in range(len(tr["t"]) - 1):
                    if tr["s"][n] < sref <= tr["s"][n + 1] and tr["t"][n + 1] - tr["t"][n] < 0.45:
                        f = (sref - tr["s"][n]) / max(tr["s"][n + 1] - tr["s"][n], 1e-6)
                        tcx = tr["t"][n] + f * (tr["t"][n + 1] - tr["t"][n])
                        latx = (tr["lat"][n] + f * (tr["lat"][n + 1] - tr["lat"][n])) * side
                        ii = int(np.abs(r.t - tcx).argmin())
                        if abs(latx - W) < 1.75 and tr["hd"][n] < np.pi / 4 and near[ii]:
                            cross.append((tcx, float(np.hypot(tr["vx"][n] if "vx" in tr else tr["v"][n], 0.0)) if False else float(tr["v"][n]), aid, float(tr["ext"][0])))
                        break
            cross.sort()
            for a, b in zip(cross[:-1], cross[1:]):
                rows.append(dict(arm=k[0], seed=k[1], route=route, t=round(b[0], 2), headway_s=round(b[0] - a[0], 2), v_follower=round(b[1], 1), v_leader=round(a[1], 1),
                                 gap_m=round((b[0] - a[0]) * b[1] - a[3] - b[3], 1)))
            rows.append(dict(arm=k[0], seed=k[1], route=route, t=None, headway_s=None, n_cross=len(cross), observed_s=round(float(near.sum() * 0.2), 0)))
    return rows


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(":--" if i == 0 else "--:" for i in range(len(head))) + "|"]
    return "\n".join(out + ["| " + " | ".join("-" if x is None else str(x) for x in r) + " |" for r in rows])


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def num(x):
    return None if x in (None, "") else float(x)


def classify(c):
    """Situation at the start of the pull-out (distance to the nearest vehicle behind in the adjacent lane, bumper to bumper)."""
    if c.get("misfire_start"):
        return "old pbyp misfire at the route start, vehicle alongside"
    g = c.get("gap_m_pull")
    if c["v_ego_contact"] < 0.5 and (c.get("pull_to_contact_s") or 0) > 8:
        return "ego standing partly in the lane for > 8 s"
    if g is None or g <= 6.5:
        return "pulled out alongside / within 6.5 m of the vehicle"
    return "pulled out with the vehicle >= 15 m behind"


def report(coll, ev, hw):
    out = []
    thr = 3.0
    rows = []
    for c in coll:
        cls = classify(c)
        c["cls"] = cls
        peak = -c["min_accel_after_pull"] if c.get("min_accel_after_pull") is not None and c["min_accel_after_pull"] < 0 else 0.0
        c["peak_decel"] = round(peak, 1)
        c["braked"] = int(peak >= thr)
        rows.append([c["arm"], c["seed"], c["route"], c["t_contact"], cls, c.get("gap_m_pull"), c.get("time_gap_s_pull"), c.get("gap_m_enter"), c.get("time_gap_s_enter"),
                     c["v_ego_contact"], c["ego_lat_contact"], c.get("pull_to_contact_s"), c.get("enter_to_contact_s"), c.get("v_other_pull"), c["v_other_contact"],
                     c["peak_decel"], c.get("first_decel_ge1_after_pull_s"), c.get("speed_change_pull_to_contact"), "yes" if c["braked"] else "no"])
    out.append(md(rows, ["arm", "seed", "route", "contact t s", "situation", "gap at pull-out start m", "time gap s", "gap at lane entry m", "time gap s", "ego v at contact m/s",
                         "ego lateral into the lane m", "pull start -> contact s", "lane entry -> contact s", "other v at pull start m/s", "other v at contact m/s",
                         "other peak decel after pull start m/s2", "first decel >= 1 m/s2, s after pull start", "other speed change pull start -> contact m/s", "braked (>= 3 m/s2)"]))
    out.append("")
    by = defaultdict(list)
    for c in coll:
        by[c["cls"]].append(c)
    srows = []
    for k, v in sorted(by.items()):
        srows.append([k, len(v), sum(c["braked"] for c in v), "%.1f" % float(np.median([c["peak_decel"] for c in v])),
                      "%.1f" % float(np.median([c["v_other_contact"] for c in v])),
                      "-" if not [c for c in v if c.get("time_gap_s_pull")] else "%.2f / %.2f" % (min(c["time_gap_s_pull"] for c in v if c.get("time_gap_s_pull")), max(c["time_gap_s_pull"] for c in v if c.get("time_gap_s_pull")))])
    srows.append(["all", len(coll), sum(c["braked"] for c in coll), "%.1f" % float(np.median([c["peak_decel"] for c in coll])), "%.1f" % float(np.median([c["v_other_contact"] for c in coll])), "-"])
    out.append(md(srows, ["situation", "contacts", "other vehicle braked >= 3 m/s2 after the pull-out started", "median peak decel m/s2", "median other v at contact m/s", "time gap at pull start, min / max s"]))
    # all pull-outs: hit vs not
    out.append("")
    erows = []
    for e in ev:
        erows.append([e["arm"], e["seed"], e["route"], e["t_enter"], "yes" if e["hit"] else "no", "yes" if e["misfire_start"] else "", e["v_ego_pull"], e["v_ego_enter"], e.get("gap_m_pull"), e.get("time_gap_s_pull"),
                      e.get("gap_m_enter"), e.get("time_gap_s_enter"), e.get("v_other_enter"), e.get("follower_peak_decel"), e.get("follower_v_change"), e.get("lane_time_s"),
                      "yes" if e.get("alongside_pull") else "no"])
    out.append(md(erows, ["arm", "seed", "route", "lane entry t s", "contact within 12 s", "start-of-route misfire", "ego v at pull start", "ego v at entry", "nearest follower gap at pull start m",
                          "time gap s", "gap at entry m", "time gap s", "follower v at entry m/s", "follower peak decel in 5 s after entry m/s2", "follower speed change in 5 s after entry m/s",
                          "ego time in the adjacent lane s", "vehicle alongside at pull start"]))
    # headways
    out.append("")
    hrows = []
    series = defaultdict(list)
    for r in hw:
        if r.get("headway_s") not in (None, ""):
            series[(r["route"], r["seed"], r["arm"])].append(r)
    meta = {(r["route"], r["seed"], r["arm"]): r for r in hw if r.get("headway_s") in (None, "")}
    # one series per (route, seed): the run that observed the lane longest
    best = {}
    for key, m in meta.items():
        if float(m["observed_s"]) < 100.0:                 # the stream needs ~30 s to build up; short observations are skewed
            continue
        k2 = (key[0], key[1])
        if k2 not in best or float(m["observed_s"]) > float(meta[best[k2]]["observed_s"]):
            best[k2] = key
    allh, allg = [], []
    for k2, key in sorted(best.items()):
        hs = [float(r["headway_s"]) for r in series[key]]
        gs = [float(r["gap_m"]) for r in series[key]]
        allh += hs
        allg += gs
        if hs:
            hrows.append([k2[0], k2[1], key[2], meta[key]["observed_s"], len(hs) + 1] + ["%.2f" % q(hs, p) for p in (10, 50, 90)] + [min(hs), sum(1 for x in hs if x < 3.0), "%.1f" % q(gs, 50)])
    hrows.append(["all", "", "", "", len(allh)] + ["%.2f" % q(allh, p) for p in (10, 50, 90)] + [min(allh), sum(1 for x in allh if x < 3.0), "%.1f" % q(allg, 50)])
    out.append(md(hrows, ["route", "traffic seed", "run used", "lane observed s", "vehicles", "headway p10 s", "median s", "p90 s", "min s", "headways < 3 s", "median bumper gap m"]))
    out.append("")
    out.append("headway percentiles over the %d headways of the series above: p5 %.2f, p25 %.2f, p75 %.2f, p95 %.2f, max %.2f s; share < 2 s %.1f%%, < 3 s %.1f%%, < 4 s %.1f%%, > 5 s %.1f%%" % (
        len(allh), q(allh, 5), q(allh, 25), q(allh, 75), q(allh, 95), max(allh), 100 * np.mean(np.array(allh) < 2), 100 * np.mean(np.array(allh) < 3), 100 * np.mean(np.array(allh) < 4),
        100 * np.mean(np.array(allh) > 5)))
    lt = [float(e["lane_time_s"]) for e in ev if e.get("lane_time_s") not in (None, "") and not e["hit"]]
    out.append("ego time in the adjacent lane for the pull-outs that left it without a contact: n %d, min %.1f, median %.1f, max %.1f s" % (len(lt), min(lt), float(np.median(lt)), max(lt)))
    return "\n".join(out)


if __name__ == "__main__":
    runs, R, routeW, _ = analyse()
    coll = collisions_table(R, routeW)
    ev = events_table(R, routeW, coll)
    hw = headways(R, routeW)
    text = report(coll, ev, hw)
    write_csv(os.path.join(RES, "pullout_collisions.csv"), coll)
    write_csv(os.path.join(RES, "pullout_events.csv"), ev)
    write_csv(os.path.join(RES, "pullout_headways.csv"), hw)
    print(text)
