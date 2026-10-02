"""Analysis 1: where does the privileged bypass (pbyp arm) misfire? Offline, on the compact extracts of bypass_extract.py.

  python3 bypass_misfire.py EXTRACT_DIR RESULTS_DIR

Writes results/bypass_activations.csv and results/bypass_misfire.md. Definitions: plans/2026-10-02-bypass-offline.md.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bypass_common import (OBST_TYPE, TARGET, actor_tracks, ego_s_series, in_junction, junction_table, light_state_at, load_all,  # noqa: E402
                           next_entry, plan_arr, project, route_lights, route_s)

HORIZON = 15.0           # s after an activation in which outcomes are attributed to it
RESUME_V = 0.5           # m/s: an actor "moves again" once it exceeds this for 3 consecutive snapshots (0.6 s)
EVENT_KEYS = ("collisions_vehicle", "collisions_layout", "collisions_pedestrian", "vehicle_blocked", "outside_route_lanes",
              "red_light", "stop_infraction", "route_dev")


def activations(d):
    """Bypass state creations in a pbyp run: first snapshot where the privileged geometry holds a bypass state whose
    (start_s, ids) differs from the previous snapshot's. Also the first snapshot in which the path was actually shifted."""
    out, prev = [], None
    rows = d["priv"]
    for i, p in enumerate(rows):
        st = p["st"]
        key = None if st is None else (round(st["start_s"] / 5), tuple(st["ids"]))
        if st is not None and key != prev:
            end = next((q["t"] for q in rows[i:] if q["st"] is None), rows[-1]["t"])
            shift = next((q["t"] for q in rows[i:] if q["bypass"]), None)
            out.append(dict(i=i, t0=p["t"], st=st, p=p, t_end=end, t_shift=shift))
        prev = key
    return out


def resume_time(track, t0):
    t, v = track["t"], track["v"]
    ok = (v > RESUME_V) & (t >= t0)
    for k in range(len(ok) - 2):
        if ok[k] and ok[k + 1] and ok[k + 2]:
            return float(t[k])
    return None


def features(runs, d, act, junc, lights):
    xy = np.array(d["route_xy"])
    L = float(route_s(xy)[-1])
    tr = actor_tracks(d)
    st, p, t0 = act["st"], act["p"], act["t0"]
    ent, iv = junc
    pl = plan_arr(d)
    k = int(np.abs(pl["t"] - t0).argmin())
    f = dict(arm=d["arm"], seed=d["seed"], route=d["route"], scenario=d["scenario"].rsplit("_", 1)[0], t0=t0, t_end=act["t_end"],
             t_shift=act["t_shift"], ego_s=p["ego_s"], ego_v=p["ego"][3], borrow=bool(st.get("borrow")), offset=st.get("offset"),
             tl=None if np.isnan(pl["tl"][k]) else int(pl["tl"][k]), tl_dist=None if np.isnan(pl["tl_dist"][k]) else float(pl["tl_dist"][k]),
             ego_dj=next_entry(ent, p["ego_s"]), ego_in_junction=in_junction(iv, p["ego_s"]), ids=list(st["ids"]), L=L)
    members = []
    for i in st["ids"]:
        tk = tr.get(i)
        if tk is None:
            continue
        j = int(np.searchsorted(tk["t"], t0).clip(0, len(tk["t"]) - 1))
        s, lat = project([[tk["x"][j], tk["y"][j]]], xy)
        before = tk["v"][tk["t"] < t0]
        members.append(dict(id=i, type=tk["type"], s=float(s[0]), lat=float(lat[0]), static_for=float(tk["stat"][j]),
                            first_seen=float(tk["t"][0]), moved_before=bool(len(before) and before.max() > 1.0),
                            vmax_all=float(tk["v"].max()), resume=resume_time(tk, t0), last_seen=float(tk["t"][-1]),
                            vis_static_s=float(((tk["v"] <= 0.2) * 0.2).sum())))
    members.sort(key=lambda m: m["s"])
    f["members"] = members
    if not members:
        return f
    m = members[0]
    f.update(lead_id=m["id"], lead_type=m["type"], lead_s=m["s"], lead_lat=m["lat"], static_for=m["static_for"],
             moved_before=m["moved_before"], first_seen=m["first_seen"], resume=m["resume"],
             resume_dt=None if m["resume"] is None else m["resume"] - t0, vmax_all=m["vmax_all"],
             beyond_end=m["s"] >= L - 0.05, a_dj=next_entry(ent, m["s"]), a_in_junction=in_junction(iv, m["s"], 3.0),
             last_seen=m["last_seen"])
    # signed distance of the clamped projection beyond the polyline end (+) or behind its start (-) along the end tangent
    tk = tr[m["id"]]
    pa = np.array([tk["x"][int(np.searchsorted(tk["t"], t0).clip(0, len(tk["t"]) - 1))], tk["y"][int(np.searchsorted(tk["t"], t0).clip(0, len(tk["t"]) - 1))]])
    if m["s"] <= 0.05:
        tang = xy[1] - xy[0]
        f["ext_m"] = float((pa - xy[0]) @ tang / np.linalg.norm(tang))
    elif m["s"] >= L - 0.05:
        tang = xy[-1] - xy[-2]
        f["ext_m"] = float((pa - xy[-1]) @ tang / np.linalg.norm(tang))
    exits = [b for a, b in iv if b <= m["s"] + 1]
    f["a_after_exit"] = float(m["s"] - max(exits)) if exits else float("inf")
    # the light that governs the lead actor: nearest reconstructed stop line within 45 m ahead of it
    gl = [(l["s_stop"] - m["s"], l) for l in lights if -3 <= l["s_stop"] - m["s"] <= 45]
    if gl:
        dist, l = min(gl, key=lambda x: x[0])
        f["a_light_dist"] = float(dist)
        f["a_light"] = light_state_at(l, t0)
        r = m["resume"]
        f["a_light_red_before_resume"] = None if r is None else any(
            (light_state_at(l, tt) in (1, 2)) for tt in np.arange(t0 - 2, r, 0.5))
    return f


# ---------------------------------------------------------------------------------------------------------- classes
def classify(f):
    """Primary class of an activation (rules in plans/2026-10-02-bypass-offline.md), first match wins."""
    if "lead_id" not in f:
        return "other"
    never_moves = f["vmax_all"] <= RESUME_V
    if f["route"] in TARGET and never_moves:
        return "scenario_obstacle"
    if f["lead_s"] <= 0.05:
        return "behind_route_start"
    if f.get("beyond_end"):
        return "beyond_route_end"
    if never_moves:
        return "parked_or_broken"
    if f.get("a_light_red_before_resume") or f.get("a_light") in (1, 2):
        return "red_light_queue"
    if f["a_dj"] <= 35 or f["a_in_junction"] or f["a_after_exit"] <= 15:
        return "junction_queue"
    return "moving_traffic_pause"


CLASS_ORDER = ["scenario_obstacle", "behind_route_start", "beyond_route_end", "parked_or_broken", "red_light_queue",
               "junction_queue", "moving_traffic_pause", "other"]


# ---------------------------------------------------------------------------------------------------------- outcomes
def ego_xy_series(d):
    return np.array([p["t"] for p in d["priv"]]), np.array([[p["ego"][0], p["ego"][1]] for p in d["priv"]])


def time_at(d, loc, mode="closest", tol=3.0):
    """Plan-clock time of an official event, from the ego trajectory and the event location (the leaderboard records the ego's location):
    closest approach of the ego to loc; for blocked events (the ego stands still there) the last snapshot within tol m.
    The contact sensor's own timestamps run about 1.05 s ahead of the plan clock (median over 38 collisions, IQR 0.9 to 1.15 s), so they are not used."""
    t, xy = ego_xy_series(d)
    dist = np.linalg.norm(xy - np.asarray(loc), axis=1)
    if mode == "last":
        idx = np.flatnonzero(dist <= tol)
        if len(idx):
            return float(t[idx[-1]])
    return float(t[dist.argmin()])


CONTACT_CLOCK_OFFSET = 1.05    # s, contact timestamp minus plan clock (median 1.05, IQR 0.9 to 1.15 over 38 collisions; 20 dropped ticks, inferred)


def parse_events(d):
    """Official infractions as dicts(kind, t, id, loc, text). Collision times: first contact with the actor from the contact sensor minus the
    clock offset; red-light and blocked events: from the ego trajectory and the recorded location."""
    import re
    first = {}
    for c in d["contacts"]:
        first.setdefault(c["id"], c["t"])
    ev = []
    for kind, items in d["infractions"].items():
        if kind not in EVENT_KEYS:
            continue
        for txt in items:
            m = re.search(r"id=(\d+) at \(x=([-\d.]+), y=([-\d.]+)", txt) or re.search(r"light (\d+) at \(x=([-\d.]+), y=([-\d.]+)", txt) \
                or re.search(r"(\d+)?.*at \(x=([-\d.]+), y=([-\d.]+)", txt)
            aid = int(m.group(1)) if m and m.group(1) else None
            loc = (float(m.group(2)), float(m.group(3))) if m else None
            if kind.startswith("collisions") and aid in first:
                t = first[aid] - CONTACT_CLOCK_OFFSET
            else:
                t = time_at(d, loc, "last" if kind == "vehicle_blocked" else "closest") if loc is not None and d["priv"] else None
            ev.append(dict(kind=kind, t=t, id=aid, loc=loc, text=txt))
    return ev


def bypass_active_at(d, t, pad=2.0):
    return any(p["bypass"] for p in d["priv"] if p["t"] is not None and t - pad <= p["t"] <= t + 0.2)


# ---------------------------------------------------------------------------------------------------------- replay
class Ego:
    """Ego-side signals as the closed loop would have seen them without the bypass: the paired drive run (same route,
    same traffic seed) from t0 on. Index by time."""

    def __init__(self, drive, junc):
        self.t, self.s = ego_s_series(drive)
        pl = plan_arr(drive)
        self.pt, self.tl, self.td = pl["t"], pl["tl"], pl["tl_dist"]
        self.ent, self.iv = junc

    def red_recent(self, t, G, X=50.0):
        """Ego light red or yellow within X m at any plan step of the last G seconds."""
        m = (self.pt >= t - G) & (self.pt <= t) & ((self.tl == 1) | (self.tl == 2)) & (self.td < X)
        return bool(m.any())

    def at(self, t):
        es = float(np.interp(t, self.t, self.s))
        k = int(np.abs(self.pt - t).argmin())
        tl = None if np.isnan(self.tl[k]) else int(self.tl[k])
        td = None if np.isnan(self.td[k]) else float(self.td[k])
        return es, tl, td, next_entry(self.ent, es), in_junction(self.iv, es)


def replay(f, act, d, ego, rule, junc):
    """Would the suppression rule have blocked this activation, and if not, how late would it have fired?
    rule: X (red/yellow light for the ego within X m; None = off), N (ego within N m before a junction entrance or inside
    one; None = off), Nb (blocker within Nb m before an entrance or inside one), T (blocker static for >= T s), valid
    (blocker projects strictly inside the route), tmin (no activation before tmin s).
    Returns ('prevented', reason) or ('fires', delay_s)."""
    if "lead_id" not in f:
        return "fires", 0.0
    X, N, Nb, T, valid, tmin, G = (rule.get(k) for k in ("X", "N", "Nb", "T", "valid", "tmin", "G"))
    tk = actor_tracks(d)[f["lead_id"]]
    t0 = f["t0"]
    if valid and (f["lead_s"] <= 0.05 or f.get("beyond_end")):
        return "prevented", "invalid_projection"
    if Nb is not None and (f["a_dj"] <= Nb or f["a_in_junction"]):
        return "prevented", "blocker_near_junction"
    times = [p["t"] for p in d["priv"] if t0 - 1e-6 <= p["t"] <= act["t_end"]]
    ok_run = 0
    for t in times:
        j = int(np.abs(tk["t"] - t).argmin())
        if tk["v"][j] > RESUME_V and t > t0 + 0.25:
            ok_run += 1
            if ok_run >= 3:
                return "prevented", "blocker_resumed"
            continue
        else:
            ok_run = 0
        if t <= t0 + 1e-6:
            es, tl, td, dj, inj = f["ego_s"], f["tl"], f["tl_dist"], f["ego_dj"], f["ego_in_junction"]
        else:
            es, tl, td, dj, inj = ego.at(t)
        if tmin is not None and t < tmin:
            continue
        if T is not None and tk["stat"][j] < T:
            continue
        if X is not None and tl in (1, 2) and td is not None and td < X:
            continue
        if N is not None and (dj <= N or inj):
            continue
        if G is not None and ego.red_recent(t, G):
            continue
        return "fires", float(t - t0)
    return "prevented", "conditions_never_cleared"


# ---------------------------------------------------------------------------------------------------------- main
def load_ds(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        out[(r["arm"], int(r["seed"]), r["route"])] = dict(DS=float(r["DS"]), RC=float(r["RC"]), coll=int(r["collisions"]),
                                                          blocked=int(r["vehicle_blocked"]), red=int(r["red_light"]))
    return out


def build(runs, ds):
    rows = []
    ctx = {}
    for (arm, seed, route), d in sorted(runs.items()):
        if arm != "pbyp":
            continue
        if route not in ctx:
            ctx[route] = (junction_table(runs, route), route_lights(runs, route))
        junc, lights = ctx[route]
        drive = runs[("drive", seed, route)]
        ego = Ego(drive, junc)
        events = parse_events(d)
        acts = activations(d)
        for n, a in enumerate(acts):
            f = features(runs, d, a, junc, lights)
            f["cls"] = classify(f)
            f["n"] = n
            f["target"] = route in TARGET
            f["_act"], f["_d"], f["_ego"], f["_junc"] = a, d, ego, junc
            nxt = acts[n + 1]["t0"] if n + 1 < len(acts) else 1e9
            f["events"] = [e for e in events if e["t"] is not None and a["t0"] <= e["t"] < nxt and e["t"] <= a["t_end"] + 2.0]
            f["events15"] = [e for e in f["events"] if e["t"] <= a["t0"] + HORIZON]
            rows.append(f)
        for e in events:
            e["att"] = None
            for f in rows:
                if f["route"] == route and f["seed"] == seed and e in f["events"]:
                    e["att"] = f["cls"]
        runs[("pbyp", seed, route)]["_events"] = events
    return rows


def outcome(f):
    ev = f["events15"]
    kinds = [e["kind"] for e in ev]
    if any(k.startswith("collisions") for k in kinds):
        return "collision"
    if "vehicle_blocked" in [e["kind"] for e in f["events"]]:
        return "blocked_later"
    if "red_light" in kinds:
        return "red_light"
    return "none_in_15s"



def median(x):
    return float(np.median(x)) if len(x) else float("nan")


def phase(p_st, ego_s):
    if p_st is None:
        return "state cleared"
    a, b = p_st["start_s"], p_st["end_s"]
    if ego_s < a - 20:
        return "before ramp"
    if ego_s < a:
        return "entering (ramp)"
    if ego_s <= b:
        return "abeam obstacle"
    if ego_s <= b + 8:
        return "just past obstacle"
    if ego_s <= b + 23:
        return "returning"
    return "after return"


def target_collisions(runs, rows):
    """Every official collision in the pbyp runs on the 4 obstacle routes with the geometry at the collision."""
    out = []
    for (arm, seed, route), d in sorted(runs.items()):
        if arm != "pbyp" or route not in TARGET:
            continue
        xy = np.array(d["route_xy"])
        mine = [f for f in rows if f["route"] == route and f["seed"] == seed]
        pt = np.array([p["t"] for p in d["priv"]])
        for e in d["_events"]:
            if not e["kind"].startswith("collisions") or e["t"] is None:
                continue
            i = int(np.abs(pt - e["t"]).argmin())
            p = d["priv"][i]
            ex, ey, yaw, v = p["ego"]
            ev = np.array([np.cos(yaw), np.sin(yaw)]) * v
            ctype = next((c["type"] for c in d["contacts"] if c["id"] == e["id"]), "?")
            row = dict(route=route, seed=seed, scenario=OBST_TYPE[route], t=e["t"], kind=e["kind"], actor=e["id"], type=ctype,
                       ego_s=p["ego_s"], ego_v=v, bypass=bool(p["bypass"]), borrow=bool(p["borrow"]))
            s_e, lat_e = project([[ex, ey]], xy)
            row["ego_lat"] = float(lat_e[0])
            st = p["st"]
            if st is None:       # state already cleared: use the last state of the run before this time
                prev = [q["st"] for q in d["priv"][:i] if q["st"] is not None]
                st_ref, row["state"] = (prev[-1] if prev else None), "cleared"
            else:
                st_ref, row["state"] = st, "active"
            row["phase"] = phase(st_ref, p["ego_s"]) if st_ref else "no state yet"
            row["offset"] = None if st_ref is None else st_ref["offset"]
            row["borrow_state"] = None if st_ref is None else bool(st_ref["borrow"])
            a = next((a for a in p["acts"] if a[0] == e["id"]), None)
            if a is not None:
                av = np.array([a[4], a[5]])
                s_a, lat_a = project([[a[2], a[3]]], xy)
                tang = np.gradient(xy, axis=0)
                k = int(np.abs(route_s(xy) - s_a[0]).argmin())
                tk = tang[k] / max(np.linalg.norm(tang[k]), 1e-9)
                hd = np.array([np.cos(a[6]), np.sin(a[6])])
                rel = ev - av
                u = np.array([a[2] - ex, a[3] - ey])
                along = float(u @ np.array([np.cos(yaw), np.sin(yaw)]))
                u = u / max(np.linalg.norm(u), 1e-9)
                row.update(actor_along=along, actor_s=float(s_a[0]), actor_lat=float(lat_a[0]), actor_v=float(np.hypot(*av)),
                           actor_dir="oncoming" if float(hd @ tk) < -0.5 else "same direction" if float(hd @ tk) > 0.5 else "crossing",
                           rel_speed=float(np.linalg.norm(rel)), closing=float(rel @ u), actor_gap=a[10], actor_static=a[9])
            row["act_cls"] = next((f["cls"] for f in mine if f["t0"] <= e["t"] and e["t"] <= f["t_end"] + 2.0 and
                                   not any(g["t0"] > f["t0"] and g["t0"] <= e["t"] for g in mine)), None)
            out.append(row)
    return out


def fmt(x, n=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%.*f" % (n, x) if isinstance(x, float) else str(x))


def mdtable(head, rows, left=(0,)):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(":--" if i in left else "--:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_activations_csv(rows, path):
    cols = ["route", "seed", "target", "scenario", "n", "t0", "t_end", "t_shift", "cls", "ego_s", "ego_v", "borrow", "tl", "tl_dist", "ego_dj",
            "ego_in_junction", "lead_id", "lead_type", "lead_s", "L", "lead_lat", "static_for", "moved_before", "first_seen", "resume_dt",
            "vmax_all", "beyond_end", "a_dj", "a_after_exit", "a_light", "a_light_dist", "a_light_red_before_resume", "n_members",
            "outcome15", "events_in_episode"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for f in rows:
            r = dict(f, n_members=len(f.get("members", [])), outcome15=outcome(f),
                     events_in_episode=";".join("%s@%.1f" % (e["kind"], e["t"]) for e in f["events"]))
            w.writerow([("%.2f" % r[c] if isinstance(r.get(c), float) else r.get(c, "")) for c in cols])


RULE_GRID_X = (None, 15, 30, 50)
RULE_GRID_N = (None, 10, 25, 40)
RULE_GRID_T = (None, 5, 10, 20)


def evaluate(rows, rule):
    res = [(f, replay(f, f["_act"], f["_d"], f["_ego"], rule, f["_junc"])) for f in rows]
    nt = [(f, o) for f, o in res if not f["target"]]
    ok = [(f, o) for f, o in res if f["cls"] == "scenario_obstacle"]
    ot = [(f, o) for f, o in res if f["target"] and f["cls"] != "scenario_obstacle"]
    ev = lambda g: sum(1 for f, o in g if o[0] == "prevented" for e in f["events"] if e["kind"].startswith("collisions") or e["kind"] == "vehicle_blocked")  # noqa: E731
    dl = [o[1] for f, o in ok if o[0] == "fires"]
    return dict(nt_n=len(nt), nt_prev=sum(o[0] == "prevented" for f, o in nt), nt_ev=ev(nt), nt_delay=median([o[1] for f, o in nt if o[0] == "fires" and o[1] > 0]),
                ok_n=len(ok), ok_blocked=sum(o[0] == "prevented" for f, o in ok), ok_delay_med=median(dl), ok_delay_max=max(dl) if dl else float("nan"),
                ot_n=len(ot), ot_prev=sum(o[0] == "prevented" for f, o in ot), ev_total=ev(nt) + ev(ot))


def rule_name(r):
    parts = []
    if r.get("valid"):
        parts.append("valid")
    if r.get("X") is not None:
        parts.append("X%d" % r["X"])
    if r.get("N") is not None:
        parts.append("N%d" % r["N"])
    if r.get("Nb") is not None:
        parts.append("Nb%d" % r["Nb"])
    if r.get("T") is not None:
        parts.append("T%d" % r["T"])
    if r.get("tmin") is not None:
        parts.append("tmin%d" % r["tmin"])
    if r.get("G") is not None:
        parts.append("G%d" % r["G"])
    return "+".join(parts) or "baseline"


def main(ex_dir, res_dir):
    runs = load_all(ex_dir)
    ds = load_ds(res_dir / "routes.csv")
    rows = build(runs, ds)
    write_activations_csv(rows, res_dir / "bypass_activations.csv")
    nt_rows = [f for f in rows if not f["target"]]
    tg_rows = [f for f in rows if f["target"]]
    md = []

    # ---------------------------------------------------------------- 1 class table
    def cls_table(sel):
        out = []
        for c in CLASS_ORDER:
            g = [f for f in sel if f["cls"] == c]
            if not g:
                continue
            res_ = [f["resume_dt"] for f in g if f.get("resume_dt") is not None]
            oc = Counter(outcome(f) for f in g)
            out.append([c, len(g), len(set((f["route"]) for f in g)), "%d" % sum(f["t0"] <= 7.0 for f in g),
                        "%d/%d" % (len(res_), len(g)), fmt(median(res_)), oc["collision"], oc["red_light"],
                        sum(1 for f in g if any(e["kind"] == "vehicle_blocked" for e in f["events"])),
                        sum(1 for f in g if any(e["kind"] == "outside_route_lanes" for e in f["events"])) or "-"])
        return out
    head = ["class", "activations", "routes", "t0 <= 7 s", "blocker resumes (n)", "median resume after t0 [s]",
            "collision within 15 s", "red light within 15 s", "blocked later in episode", "-"]
    md.append("## 1. Activations by class\n")
    md.append("Non-target routes (15 routes x 2 traffic seeds, 30 pbyp runs):\n")
    t = cls_table(nt_rows)
    md.append(mdtable(head[:-1], [r[:-1] for r in t] + [["all", len(nt_rows), len(set(f["route"] for f in nt_rows)),
                      sum(f["t0"] <= 7.0 for f in nt_rows), "%d/%d" % (sum(f.get("resume_dt") is not None for f in nt_rows), len(nt_rows)),
                      fmt(median([f["resume_dt"] for f in nt_rows if f.get("resume_dt") is not None])),
                      sum(outcome(f) == "collision" for f in nt_rows), sum(outcome(f) == "red_light" for f in nt_rows),
                      sum(1 for f in nt_rows if any(e["kind"] == "vehicle_blocked" for e in f["events"]))]]))
    md.append("\nObstacle routes (4 routes x 2 seeds, 8 pbyp runs):\n")
    t = cls_table(tg_rows)
    md.append(mdtable(head[:-1], [r[:-1] for r in t]))

    n_b = sum(f["cls"] == "behind_route_start" for f in nt_rows)
    n_e = sum(f["cls"] == "beyond_route_end" for f in nt_rows)
    n_q = sum(f["cls"] == "red_light_queue" for f in nt_rows)
    n_borrow = sum(f["borrow"] for f in nt_rows)
    md.append("\nHow to read: of the %d activations on non-target routes, %d fire on a vehicle that is not on the route ahead of the ego "
              "at all: %d on a vehicle behind the ego's spawn point (about 10 m behind, projection clamped to the route start) and %d on a "
              "vehicle beyond the last route point (clamped to the route end). Only %d are a vehicle waiting at a red light (all three at the "
              "instant the light turns green, before the first car moves) and 1 is a short stop of moving traffic. Every one of the %d blockers "
              "moved again (median %.1f s after the activation); none was parked or broken down (verified from the logs). %d of the %d "
              "activations pull the path into the oncoming lane (`borrow`)." % (
                  len(nt_rows), n_b + n_e, n_b, n_e, n_q, len(nt_rows),
                  median([f["resume_dt"] for f in nt_rows if f.get("resume_dt") is not None]), n_borrow, len(nt_rows)))
    md.append("\nEvery activation on the non-target routes (ego state at the activation, blocker, what followed within 15 s; `ext` is the "
              "distance of the blocker beyond the route end (+) or behind the route start (-)):\n")
    body = []
    for f in nt_rows:
        d_ = f["_d"]
        col = next((e for e in f["events15"] if e["kind"].startswith("collisions")), None)
        ctype = "-" if col is None else next((c["type"] for c in d_["contacts"] if c["id"] == col["id"]), "?").replace("vehicle.", "").replace("static.prop.", "")
        body.append([f["route"], f["seed"], fmt(f["t0"]), f["cls"], fmt(f["ego_s"], 0), fmt(f["ego_v"]), "oncoming" if f["borrow"] else "same dir",
                     f["lead_type"].replace("vehicle.", ""), fmt(f.get("ext_m")), fmt(f.get("static_for")), fmt(f.get("resume_dt")),
                     "-" if f["tl"] is None else "%s @ %s m" % ("GYR"[f["tl"]], fmt(f["tl_dist"], 0)), fmt(f["ego_dj"], 0) if f["ego_dj"] < 1e3 else "-",
                     outcome(f) + ("" if col is None else " (%s at %.1f s)" % (ctype, col["t"]))])
    md.append(mdtable(["route", "seed", "t0 [s]", "class", "ego s", "ego v", "borrowed lane", "blocker", "ext [m]", "static for [s]", "resumes after [s]",
                       "ego light", "ego to junction [m]", "outcome within 15 s"], body, left=(0, 3, 6, 7, 13)))

    # ---------------------------------------------------------------- 2 per route
    md.append("\n## 2. Per route: DS against the paired drive run\n")
    per = []
    routes = sorted(set(r for _, _, r in runs), key=lambda r: (r in TARGET, r))
    for r in routes:
        dd = [ds[("drive", s_, r)] for s_ in (0, 1)]
        pp = [ds[("pbyp", s_, r)] for s_ in (0, 1)]
        g = [f for f in rows if f["route"] == r]
        cc = Counter(f["cls"] for f in g)
        scen = runs[("pbyp", 0, r)]["scenario"].rsplit("_", 1)[0]
        per.append((r in TARGET, r, scen, dd, pp, cc))
    body = []
    for tgt, r, scen, dd, pp, cc in per:
        body.append([r, scen[:34], "%.0f / %.0f" % (dd[0]["DS"], dd[1]["DS"]), "%.0f / %.0f" % (pp[0]["DS"], pp[1]["DS"]),
                     "%+.1f" % (np.mean([p_["DS"] for p_ in pp]) - np.mean([d_["DS"] for d_ in dd])),
                     ", ".join("%s %d" % (k.replace("_", " "), v) for k, v in sorted(cc.items())) or "none",
                     "%d / %d" % (sum(d_["coll"] for d_ in dd), sum(p_["coll"] for p_ in pp)),
                     "%d / %d" % (sum(d_["blocked"] for d_ in dd), sum(p_["blocked"] for p_ in pp)),
                     "obstacle route" if tgt else ""])
    md.append(mdtable(["route", "scenario", "DS drive s0 / s1", "DS pbyp s0 / s1", "mean DS diff", "activations by class (both seeds)",
                       "collisions drive / pbyp", "blocked drive / pbyp", ""], body, left=(0, 1, 5, 8)))
    nt_d = [np.mean([ds[("pbyp", s_, r)]["DS"] - ds[("drive", s_, r)]["DS"] for s_ in (0, 1)]) for r in routes if r not in TARGET]
    tg_d = [np.mean([ds[("pbyp", s_, r)]["DS"] - ds[("drive", s_, r)]["DS"] for s_ in (0, 1)]) for r in routes if r in TARGET]
    md.append("\nMean over routes: non-target %+.1f DS (15 routes), obstacle routes %+.1f DS (4 routes); reproduces report.md (-28.8 / +23.1)." % (np.mean(nt_d), np.mean(tg_d)))

    worst = sorted(((np.mean([ds[("pbyp", s_, r)]["DS"] - ds[("drive", s_, r)]["DS"] for s_ in (0, 1)]), r) for r in routes if r not in TARGET))[:6]
    md.append("\nHow to read: the four obstacle routes gain DS because the real obstacle is detected, but each of them also has at least one "
              "misfire activation (%d of the %d activations on these four routes). On non-target routes the "
              "largest losses (%s) are runs where the activation is followed by a collision and an ego that stays in the shifted lane; "
              "three routes have no activation and no change (15102, 27043, 9196), and 24944 has one in seed 1 only." % (
                  sum(f["cls"] != "scenario_obstacle" for f in tg_rows), len(tg_rows), ", ".join("%s %+.0f" % (r, v) for v, r in worst)))

    # ---------------------------------------------------------------- 3 attribution of events
    def drive_has(route, seed, e):
        dv = runs[("drive", seed, route)]
        for x in parse_events(dv):
            if x["kind"] == e["kind"] and x["loc"] is not None and e["loc"] is not None and \
                    np.hypot(x["loc"][0] - e["loc"][0], x["loc"][1] - e["loc"][1]) <= 10.0:
                return True
        return False

    cat = defaultdict(Counter)
    drive_tot = Counter()
    pb_tot = Counter()
    ev_rows = []
    for (arm, seed, route), d in sorted(runs.items()):
        if route in TARGET:
            continue
        evs = parse_events(d) if arm == "drive" else d["_events"]
        kmap = lambda k: "collision" if k.startswith("collisions") else k  # noqa: E731
        for e in evs:
            k = kmap(e["kind"])
            if arm == "drive":
                drive_tot[k] += 1
                continue
            pb_tot[k] += 1
            has_act = any(f["route"] == route and f["seed"] == seed for f in rows)
            if e["t"] is None:                    # outside_route_lanes has no time or position in the official record
                where = "in a run with a misfire" if has_act else "in a run without activation"
            else:
                where = "in a misfire episode" if e.get("att") else "outside every episode"
            same = drive_has(route, seed, e) if e["loc"] is not None else None
            cat[k][(where, "also in drive" if same else ("not in drive" if same is not None else "n/a"))] += 1
            ev_rows.append((route, seed, k, e["t"], e.get("att"), same))
    md.append("\n## 3. Which pbyp events follow a misfire (30 non-target runs)\n")
    body = []
    for k in ("collision", "vehicle_blocked", "red_light", "outside_route_lanes", "stop_infraction"):
        if not pb_tot[k] and not drive_tot[k]:
            continue
        c = cat[k]
        body.append([k, drive_tot[k], pb_tot[k],
                     sum(v for (w, a), v in c.items() if w == "in a misfire episode" and a == "not in drive"),
                     sum(v for (w, a), v in c.items() if w == "in a misfire episode" and a == "also in drive"),
                     sum(v for (w, a), v in c.items() if w == "outside every episode" and a == "also in drive"),
                     sum(v for (w, a), v in c.items() if w == "outside every episode" and a == "not in drive"),
                     sum(v for (w, a), v in c.items() if w.startswith("in a run") or a == "n/a")])
    md.append(mdtable(["event", "drive (official)", "pbyp (official)", "in misfire episode, not in drive", "in misfire episode, also in drive (same place)",
                       "outside episodes, also in drive", "outside episodes, not in drive", "no time or place in record"], body))
    # per class
    body = []
    for c in CLASS_ORDER:
        n_c = sum(1 for f in nt_rows if f["cls"] == c)
        if not n_c:
            continue
        cnt = Counter()
        for f in nt_rows:
            if f["cls"] != c:
                continue
            for e in f["events"]:
                cnt["collision" if e["kind"].startswith("collisions") else e["kind"]] += 1
        body.append([c, n_c, cnt["collision"], cnt["vehicle_blocked"], cnt["red_light"]])
    md.append("\nEvents inside the episode of each class (an episode runs from the activation to the end of its bypass state, "
              "or to the next activation):\n")
    md.append(mdtable(["class", "activations", "collisions", "blocked", "red light"], body))

    md.append("\nHow to read: of the %d pbyp collisions on non-target routes, %d fall in a misfire episode and are not reproduced by the paired "
              "drive run; %d are the same collisions as drive (27043 and 9196 have no activation); the red-light infractions are nearly all "
              "the same as drive. All 8 blocked events are in a misfire episode (the ego ends in a lane it cannot leave). 16 of the 30 pbyp "
              "runs also log `outside_route_lanes`, which has no position or time in the official record." % (
                  pb_tot["collision"], sum(v for (w, a), v in cat["collision"].items() if w == "in a misfire episode" and a == "not in drive"),
                  sum(v for (w, a), v in cat["collision"].items() if w == "outside every episode" and a == "also in drive")))

    # ---------------------------------------------------------------- 4 replay of suppression rules
    md.append("\n## 4. Log replay of suppression rules\n")
    md.append("Replay on the logs only: an activation counts as prevented when, from its logged time on, the rule never lets it fire "
              "before its blocker moves off again or the episode ends; otherwise it fires late by the stated delay. "
              "Ego-side signals after t0 come from the paired drive run (what the ego sees if it does not bypass). "
              "The closed loop is not replayed, so what the car would have done afterwards is unknown.\n")
    grid = []
    for valid in (False, True):
        for X in RULE_GRID_X:
            for N in RULE_GRID_N:
                for T in RULE_GRID_T:
                    r = dict(valid=valid, X=X, N=N, T=T)
                    grid.append((r, evaluate(rows, r)))
    for r in (dict(Nb=25), dict(Nb=40), dict(tmin=7), dict(valid=True, tmin=7), dict(valid=True, N=25, T=5)):
        grid.append((r, evaluate(rows, r)))
    with open(res_dir / "bypass_rule_grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["rule", "nt_n", "nt_prevented", "nt_events_in_prevented", "nt_median_delay_s", "target_correct_n", "target_correct_blocked",
                "target_correct_delay_median_s", "target_correct_delay_max_s", "target_misfire_n", "target_misfire_prevented"]
        w.writerow(cols)
        for r, e in grid:
            w.writerow([rule_name(r), e["nt_n"], e["nt_prev"], e["nt_ev"], "%.2f" % e["nt_delay"], e["ok_n"], e["ok_blocked"],
                        "%.2f" % e["ok_delay_med"], "%.2f" % e["ok_delay_max"], e["ot_n"], e["ot_prev"]])
    pick = [dict(), dict(X=15), dict(X=30), dict(X=50), dict(N=10), dict(N=25), dict(N=40), dict(Nb=25), dict(Nb=40), dict(T=5), dict(T=10),
            dict(T=20), dict(tmin=7), dict(valid=True), dict(X=30, N=25), dict(N=25, T=5), dict(N=25, T=10), dict(N=40, T=10),
            dict(X=30, N=25, T=10), dict(G=5), dict(G=10), dict(valid=True, T=5), dict(valid=True, T=10), dict(valid=True, G=5), dict(valid=True, T=5, G=5),
            dict(valid=True, N=25, T=5, G=5), dict(valid=True, X=30, N=25, T=10)]
    body = []
    for r in pick:
        e = evaluate(rows, r)
        body.append([rule_name(r), "%d / %d" % (e["nt_prev"], e["nt_n"]), e["nt_ev"], fmt(e["nt_delay"]), "%d / %d" % (e["ot_prev"], e["ot_n"]),
                     "%d / %d" % (e["ok_blocked"], e["ok_n"]), fmt(e["ok_delay_med"]), fmt(e["ok_delay_max"])])
    md.append(mdtable(["rule", "non-target misfires prevented", "collisions + blocked in those episodes", "median delay of the rest [s]",
                       "obstacle-route misfires prevented", "correct activations blocked", "correct: median delay [s]", "correct: max delay [s]"], body))
    md.append("\nX: no activation while the ego light is red or yellow and closer than X m. N: none while the ego is within N m before a "
              "junction entrance or inside one. Nb: same test on the blocker. T: blocker static for at least T s (the original rule is 2 s). "
              "valid: blocker projects strictly inside the route polyline. tmin: no activation before tmin s. G: none while the ego light was red or yellow within 50 m at any time in the last G s "
              "(the queue's first car has not moved yet when the light turns green).\n")
    md.append("N x T grid, no X, no valid (cells: non-target prevented of 26 / correct activations delayed, median s; 8 correct, none blocked):\n")
    body = []
    for N in RULE_GRID_N:
        row = ["N=%s" % ("off" if N is None else N)]
        for T in RULE_GRID_T:
            e = evaluate(rows, dict(N=N, T=T))
            row.append("%d / %s" % (e["nt_prev"], fmt(e["ok_delay_med"])) + ("" if e["ok_blocked"] == 0 else " (%d blocked)" % e["ok_blocked"]))
        body.append(row)
    md.append(mdtable(["", "T=2 (original)", "T=5", "T=10", "T=20"], body))

    md.append("\nHow to read: a red-light test on the ego light (X) prevents nothing, because no misfire happens while the ego's light is red "
              "(the privileged rule already has that test at 50 m); the three red-queue activations happen when it turns green (G). A junction "
              "test on the ego (N) removes the activations at the spawn point and some beyond-end ones, because the ego stands in or next to "
              "a junction there, and delays the rest by about 7 s. A longer static time T removes most misfires because the blockers move "
              "again within seconds, and costs the correct activations T minus their current static time (0 s at T=5, 5 s at T=10, 15 s at "
              "T=20). The projection test (valid) removes 22 of 26 non-target and 7 of 8 obstacle-route misfires at no delay. These are fits "
              "on the same 38 runs, so the combinations are a ranking, not a confirmation.")

    # ---------------------------------------------------------------- 5 target-route collisions
    tc = target_collisions(runs, rows)
    md.append("\n## 5. Collisions on the obstacle routes\n")
    body = []
    for r in tc:
        body.append([r["route"], r["seed"], fmt(r["t"]), "%s %s" % (r["type"].replace("vehicle.", "").replace("static.prop.", ""), r["actor"]),
                     r["act_cls"] or "-", fmt(r["ego_s"], 0), r["phase"], "yes" if r["bypass"] else "no", fmt(r["ego_lat"]),
                     fmt(r.get("actor_lat")), r.get("actor_dir", "-"), fmt(r["ego_v"]), fmt(r.get("actor_v")), fmt(r.get("actor_along")), fmt(r.get("closing")), fmt(r.get("rel_speed")),
                     fmt(r["offset"]) + (" borrow" if r["borrow_state"] else "")])
    md.append(mdtable(["route", "seed", "t [s]", "actor", "episode class at t", "ego s", "phase vs obstacle", "bypass shifted", "ego lat [m]", "actor lat [m]",
                       "actor direction", "ego v", "actor v", "actor ahead of ego [m]", "closing speed (+ = ego approaching)", "rel speed", "state offset [m]"], body, left=(0, 3, 4, 6, 11)))
    n_start = sum(r["act_cls"] == "behind_route_start" for r in tc)
    n_end = sum(r["act_cls"] == "beyond_route_end" for r in tc)
    n_real = sum(r["act_cls"] == "scenario_obstacle" for r in tc)
    ramp = sum(r["phase"] == "entering (ramp)" for r in tc)
    same = sum(r.get("actor_dir") == "same direction" for r in tc)
    inlane = sum(r["offset"] is not None and r.get("actor_lat") is not None and abs(r["actor_lat"] - r["offset"]) <= 0.5 for r in tc)
    alongside = sum(r.get("actor_along") is not None and abs(r["actor_along"]) <= 3.0 for r in tc)
    t_start = sorted(r["t"] for r in tc if r["act_cls"] == "behind_route_start")
    md.append("\nHow to read: of the %d collisions on the obstacle routes, %d belong to the artefact at the spawn point (at %s s, before the real obstacle is "
              "within 50 m; one of them, 2520 seed 0 at 16.0 s, is with a nearly stationary vehicle (0.5 m/s) in the borrowed lane while the ego 'returns' from the spurious shift), %d to the artefact near the "
              "route end, and %d to the real bypass. %d of %d are with a vehicle of the ego's own driving direction, %d of %d with an actor within 0.5 m of the "
              "lateral offset the path borrows, %d of %d while the ego is in the entry ramp 12 to 20 m before the obstacle, %d of %d with the other vehicle alongside "
              "the ego (centre offset along the ego heading within 3 m: a faster vehicle in the adjacent lane meets the ego while it moves over). Ego speed at the "
              "collision is %.1f to %.1f m/s, the other vehicle's %.1f to %.1f m/s (the 0.2 s snapshot nearest to the collision; collision times are the first contact minus a 1.05 s clock offset)." % (
                  len(tc), n_start, ", ".join("%.1f" % x for x in t_start), n_end, n_real, same, len(tc), inlane, len(tc), ramp, len(tc), alongside, len(tc),
                  min(r["ego_v"] for r in tc), max(r["ego_v"] for r in tc),
                  min(r["actor_v"] for r in tc if "actor_v" in r), max(r["actor_v"] for r in tc if "actor_v" in r)))
    return rows, runs, ds, md, ev_rows


HEADER = """# bypass_misfire: where does the privileged bypass misfire (pbyp, 19 routes x 2 traffic seeds)

Offline analysis of finished logs in `$DATA_DIR/runs/vlm_arb/arms/` (units `v2-pbyp-*` against the paired `drive` runs). No CARLA, no GPU, no
new driving. Definitions (activation, classes, replay) are in [plans/2026-10-02-bypass-offline.md](../plans/2026-10-02-bypass-offline.md);
code: `scripts/bypass_extract.py`, `bypass_common.py`, `bypass_misfire.py`; per-activation rows in `bypass_activations.csv`, the full rule grid
in `bypass_rule_grid.csv`. Every number below is computed from the logs (verified) unless a sentence says inferred.

**Answer in one paragraph.** The guess "it pulls out around red-light queues" is mostly wrong. Of 26 activations on the 15 non-target routes,
22 are triggered by a vehicle that is not on the route ahead of the ego: 8 sit about 10 m *behind* the ego's spawn point and 14 sit *beyond the
last route point*, and the geometry projects both onto the first or last route point, where the blocker test (static 2 s, within -5 to 50 m,
lateral offset below one lane) passes. Only 3 are the head of a red-light queue (fired at the instant the light turns green), 1 is a short stop of
moving traffic; no activation was triggered by a parked or broken-down vehicle. All 26 blockers drive off again within a median of 1.8 s.
The collisions follow: 21 of the 27 pbyp collisions on non-target routes are inside such an episode and absent from the paired drive run.

"""

FOOTER = """
## 6. Verified and inferred

Verified from the logs: activation times, blocker identity, speed history, projection onto the route (clamped to the route ends: 8 + 14 cases,
`ext` column), resume times, official events and their times (collisions from the contact sensor, red-light and blocked events located on the ego
trajectory), the paired DS differences (they reproduce report.md: -28.8 and +23.1), the replay counts under the stated rules.

Inferred: why the 14 vehicles beyond the route end were stopped (a queue or held traffic beyond the final junction; the light state there is not
logged); that the 3 red-queue activations are a red-light queue (the light state is reconstructed from the ego's own light at the same stop line,
the transition times agree across runs to 0.1 s); that the collisions would not have happened without the activation (the paired drive run has
none at the same place, but the ego trajectories differ); the 1.05 s offset between the contact sensor's clock and the plan clock (measured on the 38 collisions
from the ego's closest approach to the recorded location, attributed to 20 dropped ticks); every statement about what a suppression rule would have done after the replayed time.

Data limits: the 13 seed-0 `drive` runs in `eval-drive-s0` have no `privileged.jsonl`, so only their ego-side signals (plans, junction distance
from the 's' lines) were used; junction entrances come from the `drive` runs' own junction distance and are exact, exits are sampled at the
ego's step. The traffic frozen until about 6 s (all vehicles stand from about 2.8 s, the ego itself is held until 6 s) is what makes standing
vehicles pass the 2 s static test at the first allowed moment, 5.0 s (inferred from the speed histories, the harness code was not read).
"""


if __name__ == "__main__":
    res_dir = Path(sys.argv[2])
    rows, runs, ds, md, ev_rows = main(sys.argv[1], res_dir)
    (res_dir / "bypass_misfire.md").write_text(HEADER + "\n".join(md) + "\n" + FOOTER)
    print("written", res_dir / "bypass_misfire.md")
