"""C1 zero-score and slow-scene review of SH30-F-s0 and AP2-AB-s0 on the 400 public AlpaSim scenes (decision 202).
Runs on the Mac with the repo .venv from the pickles of c1_extract.py in tmp/c1/ (logs of both runs, map, CAM_F0 frames).
Map, other actors and the logged future are privileged: used for labels only.

  rows    per (driver, zero or slow scene) measurements -> experiments/alpasim/results/c1/rows.json
            c1_review.py rows
  sheets  one review sheet per scene (bird's-eye view of both drivers + the CAM_F0 frames each driver received) -> tmp/c1/sheets/
            c1_review.py sheets [--set zero|slow] [--scene <suffix> ...]

Collision classes (rules of decision 153 / the N1 taxonomy, on AlpaSim's logs): the object whose box meets the ego box at the first
at-fault step; stopped = object speed < 0.5 m/s at the event and heading difference < 30 deg; side = object centroid at the event not
beyond the front bumper (dx < 4.05 m in the rear-axle frame) and |dy| >= 1.0 m; first match of: stopped and not side -> A1;
stopped and side, or a lateral-only contact -> D; heading difference > 150 deg -> E oncoming; >= 30 deg -> C crossing; object more than
1.5 m beside the ego's path at t0 and heading difference < 45 deg -> B cut-in; else A2 moving lead.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_lib as L  # noqa: E402

RES = L.ROOT / "experiments/alpasim/results/c1"
SIDE_DX, SIDE_DY = 4.05, 1.0


def plan_vs_log(o, k):
    """Decision k: plan arc / logged arc over the common horizon (the log ends with the scene), and the plan's 4 s arc."""
    p, g = o["drive"][k]["traj"], L.gt(o)
    t0, t1 = p[0, 0], min(p[-1, 0], g[-1, 0])
    if t1 - t0 < 0.9e6:
        return np.nan, L.arc(p[:, 1:3]), 0.0
    pa = L.arc(p[p[:, 0] <= t1 + 1, 1:3])
    ts = np.arange(t0, t1 + 1, 1e5)
    ga = L.arc(np.array([L.interp_pose(g, t)[:2] for t in ts]))
    return pa / max(ga, 0.5), L.arc(p[:, 1:3]), (t1 - t0) * 1e-6


def base(o, name, scene):
    g, e = L.gt(o), L.ego(o)
    r = dict(driver=name, scene=scene, score=o["summary"]["score"], flags=L.zero_flags(o), turn=round(L.turn_deg(o), 1),
             v0=round(L.speed_at(g, 0.1e6), 2), log_dist=round(L.arc(g[:, 1:3]), 1), log_v_end=round(L.speed_at(g, g[-1, 0] - 0.2e6), 2),
             ego_dist=round(L.arc(e[:, 1:3]), 1), cmd=[x["cmd"] for x in o["rec"]], progress=o["summary"]["score_metrics"].get("progress_clipped_rel"))
    r["turn_bucket"] = L.turn_bucket(r["turn"])
    ry = [x["route0"][1] for x in o["rec"] if x["route0"]]
    r["route_y"] = [round(float(y), 1) for y in ry]
    r["bend"] = "left" if max(ry) > 2 else "right" if min(ry) < -2 else "none"      # the route's first waypoint (40 m ahead) moves sideways
    r["ratio"] = [round(float(plan_vs_log(o, k)[0]), 2) for k in range(len(o["drive"]))]
    return r


def path_heading(g, s):
    """Heading of the logged path at arc position s."""
    ls = LineString(g[:, 1:3])
    a, b = np.array(ls.interpolate(max(s - 0.75, 0)).coords[0]), np.array(ls.interpolate(min(s + 0.75, ls.length)).coords[0])
    return float(np.arctan2(*(b - a)[::-1])) if np.hypot(*(b - a)) > 0.05 else float(np.unwrap(g[:, 3])[-1])


def lead_on_path(o, t, horizon=40.0, half=1.3):
    """Nearest object on the logged path ahead of the simulated ego at time t: (bumper gap m along the path, object speed m/s)."""
    g, e = L.gt(o), L.interp_pose(L.ego(o), t)
    ls = LineString(g[:, 1:3])
    if ls.length < 1.0:                                           # a stopped log: use the straight line ahead of the logged pose
        h = g[0, 3]
        ls = LineString([g[0, 1:3], g[0, 1:3] + horizon * np.array([np.cos(h), np.sin(h)])])
    else:                                                         # extend the path straight beyond its end
        h = path_heading(g, ls.length)
        ls = LineString(list(ls.coords) + [tuple(np.array(ls.coords[-1]) + horizon * np.array([np.cos(h), np.sin(h)]))])
    s0 = ls.project(Point(e[:2]))
    best = (None, None)
    for aid, tr in o["actors"].items():
        if aid == "EGO" or not (tr[0, 0] <= t <= tr[-1, 0]):
            continue
        p = L.interp_pose(tr, t)
        s = ls.project(Point(p[:2]))
        if 0 < s - s0 < horizon and ls.distance(Point(p[:2])) < half:
            gap = s - s0 - L.EGO_L / 2 - o["size"][aid][0] / 2
            if best[0] is None or gap < best[0]:
                lg = next(x for x in o["logged"] if x["id"] == aid)["traj"]
                best = (round(float(gap), 1), round(L.speed_at(lg, t), 2))
    return best


def event_common(o, t):
    """Shared event measurements at scored time t (us)."""
    e, g = L.ego(o), L.gt(o)
    k = sum(d["now"] < t for d in o["drive"])                 # decisions taken before the event
    pe, pg = L.interp_pose(e, t), L.interp_pose(g, t)
    lat, s = L.signed_lat(g[:, 1:3], pe[:2])
    sg = L.signed_lat(g[:, 1:3], pg[:2])[1]
    rat = [plan_vs_log(o, j)[0] for j in range(k)]
    lead = lead_on_path(o, max(t - 1.0e6, e[1, 0]))
    yaw_off = float(np.degrees(L.wrap(pe[2] - L.interp_pose(g, g[0, 0] + 1e6 * 0)[2]))) if False else float(np.degrees(L.wrap(pe[2] - path_heading(g, s))))
    lat_hist = [round(L.signed_lat(g[:, 1:3], L.interp_pose(e, t - d * 1e6)[:2])[0], 2) for d in (2.0, 1.0, 0.5) if t - d * 1e6 >= 0]
    return dict(lead_gap=lead[0], lead_v=lead[1], yaw_off=round(yaw_off, 1), lat_hist=lat_hist, t=round(t * 1e-6, 2), k=k, ego_v=round(L.speed_at(e, t - 0.25e6, 0.25e6), 2), log_v=round(L.speed_at(g, t), 2),
                lat=round(lat, 2), ahead=round(s - sg, 2), ratio_max=round(float(np.nanmax(rat)), 2) if len(rat) and np.isfinite(rat).any() else None,
                ratio_k0=round(float(rat[0]), 2) if rat and np.isfinite(rat[0]) else None)


def collision(o):
    t = L.first_event(o, "collision_at_fault")
    c = event_common(o, t)
    e = L.interp_pose(L.ego(o), t)
    eb = L.box(*e)
    best = None
    for aid, tr in o["actors"].items():
        if aid == "EGO" or not (tr[0, 0] <= t <= tr[-1, 0]):
            continue
        p = L.interp_pose(tr, t)
        sx, sy, lab = o["size"][aid]
        d = eb.distance(L.box(*p, sx, sy))
        if best is None or d < best[0]:
            best = (d, aid, p, lab, (sx, sy))
    d, aid, p, lab, sz = best
    lg = next(x for x in o["logged"] if x["id"] == aid)["traj"]
    rear = e[:2] - L.CENTER * np.array([np.cos(e[2]), np.sin(e[2])])
    dx, dy = L.rot(e[2]).T @ (p[:2] - rear)
    dh = abs(np.degrees(L.wrap(p[2] - e[2])))
    ov = L.speed_at(lg, t)
    g = L.gt(o)
    p0 = L.interp_pose(lg, max(lg[0, 0], 0))
    dy0 = L.signed_lat(g[:, 1:3], p0[:2])[0]                   # object beside the logged ego path at the scene start
    front, latc = (L.metric(o, n) for n in ("collision_front", "collision_lateral"))
    f_at = bool(front[front[:, 0] == t, 1].max() > 0)
    l_at = bool(latc[latc[:, 0] == t, 1].max() > 0)
    stopped = ov < 0.5 and dh < 30
    side = dx < SIDE_DX and abs(dy) >= SIDE_DY
    cls = ("A1 stopped lead" if stopped and not side else "D side contact" if (stopped and side) or (l_at and not f_at) else
           "E oncoming" if dh > 150 else "C crossing" if dh >= 30 else "B cut-in" if abs(dy0) > 1.5 and dh < 45 else "A2 moving lead")
    # would the logged ego have been clear of this object at the same time, and how far the object moved in the scene
    gd = L.box(*L.interp_pose(g, t)).distance(L.box(*p, *sz))
    c.update(clear_at_log_speed=hybrid_clear(o, "log_speed", t), clear_on_log_path=hybrid_clear(o, "log_path", t))
    c.update(cls=cls, obj=aid[:6], obj_v=round(ov, 2), dh=round(dh), dx=round(float(dx), 2), dy=round(float(dy), 2), dy0=round(dy0, 2), front=f_at,
             lateral=l_at, gap=round(d, 2), log_gap=round(gd, 2), obj_dist=round(L.arc(lg[:, 1:3]), 1), obj_size=[round(x, 1) for x in sz])
    return c


def hybrid_clear(o, mode, t_end):
    """Counterfactual for attribution (privileged): would the ego box have stayed clear of every object up to t_end if it had
    kept the log's position along the path with the driven lateral / heading offset ("log_speed"), or the driven position along the
    path with no lateral offset ("log_path"). Contacts from an object behind the ego (a non-reactive follower) do not count."""
    e, g = L.ego(o), L.gt(o)
    ls = LineString(g[:, 1:3])
    for t in np.arange(e[1, 0], t_end + 1, 1e5):
        pe, pg = L.interp_pose(e, t), L.interp_pose(g, t)
        lat, s_e = L.signed_lat(g[:, 1:3], pe[:2])
        s_g = ls.project(Point(pg[:2]))
        s_ = s_g if mode == "log_speed" else s_e
        h = path_heading(g, s_)
        c = np.array(ls.interpolate(s_).coords[0])
        if mode == "log_speed":
            c, h = c + lat * np.array([-np.sin(h), np.cos(h)]), h + L.wrap(pe[2] - path_heading(g, s_e))
        b = L.box(c[0], c[1], h)
        for aid, tr in o["actors"].items():
            if aid == "EGO" or not (tr[0, 0] <= t <= tr[-1, 0]):
                continue
            p = L.interp_pose(tr, t)
            if np.hypot(*(p[:2] - c)) > 12:
                continue
            sx, sy, _ = o["size"][aid]
            if b.intersects(L.box(*p, sx, sy)):
                dx, dy = L.rot(h).T @ (p[:2] - c)
                if not (dx < -1.5 and abs(dy) < 1.0):
                    return False
    return True


def lateral(o, flag, rd):
    """Offroad / corridor event: where, which side, and whether the plans or the tracking left the road / corridor."""
    t = L.first_event(o, flag)
    c = event_common(o, t)
    e, g = L.ego(o), L.gt(o)
    turn = L.turn_deg(o)
    gl = LineString(g[:, 1:3])
    hold = turn if abs(turn) >= 5 else 0.0
    inside = None if not hold else bool(np.sign(c["lat"]) == np.sign(hold))     # the ego is on the inner side of the logged turn
    # each plan on its own: first horizon at which its footprint fails the test, had it been tracked exactly
    plan_out, plan_out_full, track = [], [], []
    for k, d in enumerate(o["drive"]):
        if d["now"] >= t:
            break
        pc = L.plan_c(o, k)
        if flag == "offroad":
            bad = [not L.onroad_scorer(rd, q[1:4]) for q in pc[::5]]
            bad_full = [not L.onroad(rd, q[1:4]) for q in pc[::5]]
        else:
            bad = [gl.distance(Point(q[1:3])) >= 4.0 for q in pc[::5]]
            bad_full = bad
        plan_out.append(round(0.5 * int(np.argmax(bad)), 1) if any(bad) else None)
        plan_out_full.append(round(0.5 * int(np.argmax(bad_full)), 1) if any(bad_full) else None)
        tn = d["now"] + 0.5e6                                    # tracking: executed pose 0.5 s later against the plan's 0.5 s pose
        if tn <= e[-1, 0]:
            q, x = pc[5, 1:4], L.interp_pose(e, tn)
            dv = L.rot(q[2]).T @ (x[:2] - q[:2])
            track.append([round(float(dv[0]), 2), round(float(dv[1]), 2)])
    # heading of the last plan before the event against the logged heading change still to come
    pe = L.interp_pose(e, t)
    end, eh = g[-1, 1:3], np.unwrap(g[:, 3])[-1]
    dv = L.rot(eh).T @ (pe[:2] - end)
    beyond = bool(gl.project(Point(pe[:2])) >= gl.length - 1e-6)   # the ego has passed the end of the logged path
    seg = g[-1, 1:3] - g[-2, 1:3]                                 # the scorer's end direction: the last logged segment
    c.update(beyond=beyond, over=round(float(dv[0]), 2), lat_end=round(float(dv[1]), 2), log_len=round(gl.length, 1),
             end_dir_err=round(float(abs(np.degrees(L.wrap(np.arctan2(seg[1], seg[0]) - eh)))), 1), last_seg=round(float(np.hypot(*seg)), 4))
    full_in = L.onroad(rd, pe) if flag == "offroad" else None
    gt_flag = (not L.onroad_scorer(rd, L.interp_pose(g, t))) if flag == "offroad" else None
    plan_turn = [round(float(np.degrees(L.wrap(d["traj"][-1, 3] - d["traj"][0, 3]))), 1) for d in o["drive"] if d["now"] < t]
    c.update(flag=L.SHORT[flag], inside=inside, plan_out=plan_out, plan_out_full=plan_out_full, track=track, plan_turn=plan_turn,
             inside_full_area=full_in, gt_flagged=gt_flag,
             track_lat_max=round(max((abs(x[1]) for x in track), default=0.0), 2), n_plan_out=sum(x is not None for x in plan_out))
    return c


def lead_ahead(o, t, horizon=40.0):
    """Nearest object in the ego's own lane ahead at time t: (gap m, object speed), by the ego footprint swept straight ahead."""
    e = L.interp_pose(L.ego(o), t)
    best = (None, None)
    for aid, tr in o["actors"].items():
        if aid == "EGO" or not (tr[0, 0] <= t <= tr[-1, 0]):
            continue
        p = L.interp_pose(tr, t)
        dx, dy = L.rot(e[2]).T @ (p[:2] - e[:2])
        if 0 < dx < horizon and abs(dy) < 1.6 and (best[0] is None or dx < best[0]):
            lg = next(x for x in o["logged"] if x["id"] == aid)["traj"]
            best = (round(float(dx - L.EGO_L / 2 - o["size"][aid][0] / 2), 1), round(L.speed_at(lg, t), 2))
    return best


def slow(o):
    """A slow scene (0 < score < 1): how the driven speed profile differs from the log's."""
    e, g = L.ego(o), L.gt(o)
    ts = e[2:, 0]
    ve = np.array([L.speed_at(e, t - 0.25e6, 0.25e6) for t in ts])
    vg = np.array([L.speed_at(g, t - 0.25e6, 0.25e6) for t in ts])
    lead = [lead_ahead(o, t) for t in (e[1, 0], e[len(e) // 2, 0], e[-1, 0])]
    ctrl, cols = o.get("ctrl"), o.get("ctrl_cols")
    return dict(ve=[round(float(x), 1) for x in ve], vg=[round(float(x), 1) for x in vg], ego_v_end=round(float(ve[-1]), 2),
                dv_end=round(float(ve[-1] - vg[-1]), 2), log_accel=round(float(vg[-1] - vg[0]), 2), ego_accel=round(float(ve[-1] - ve[0]), 2),
                lead=lead, ratio_k0=round(float(plan_vs_log(o, 0)[0]), 2), plan4_k0=round(plan_vs_log(o, 0)[1], 1),
                lat_max=round(float(L.metric(o, "lateral_dist_to_gt_trajectory")[:, 1].max()), 2),
                acc_min=None if ctrl is None else round(float(ctrl[:, cols.index("acceleration")].min()), 2))


def cmd_rows(a):
    R, M = L.runs(), L.load("map")
    rows = []
    for n, S in R.items():
        for s, o in sorted(S.items()):
            sc = o["summary"]["score"]
            if sc >= 1:
                continue
            r = base(o, n, s)
            if sc == 0:
                rd = L.road(M[s])
                r["events"] = [collision(o) | {"flag": "collision"} if f == "collision_at_fault" else lateral(o, f, rd) for f in r["flags"]]
            else:
                r["slow"] = slow(o)
            rows.append(r)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rows.json").write_text(json.dumps(rows, indent=0, default=float))
    print(len(rows), "rows ->", RES / "rows.json")


# ---------------------------------------------------------------- review sheets
def bev(ax, o, rd, t_ev=None, title="", hit=None, w=16.0):
    import matplotlib.patches as mp
    e, g = L.ego(o), L.gt(o)
    for geom in getattr(rd["area"], "geoms", [rd["area"]]):
        if geom.geom_type == "Polygon":
            ax.add_patch(mp.Polygon(np.array(geom.exterior.coords), fc="#ececec", ec="#9a9a9a", lw=0.5, zorder=0))
            for h in geom.interiors:
                ax.add_patch(mp.Polygon(np.array(h.coords), fc="white", ec="#9a9a9a", lw=0.5, zorder=0.1))
    for cl in rd["centers"]:
        ax.plot(*np.array(cl.coords).T, color="#c8c8c8", lw=0.5, zorder=0.2)
    ax.plot(g[:, 1], g[:, 2], color="#555555", lw=1.6, ls="--", zorder=3)
    for k, d in enumerate(o["drive"]):
        pc = L.plan_c(o, k)
        ax.plot(pc[:, 1], pc[:, 2], color="#0072B2", lw=0.8, alpha=0.3 + 0.07 * k, zorder=2)
    ax.plot(e[:, 1], e[:, 2], color="#D55E00", lw=1.8, zorder=4)
    tt = t_ev if t_ev is not None else e[-1, 0]
    for aid, tr in o["actors"].items():
        if aid == "EGO":
            continue
        ax.plot(tr[:, 1], tr[:, 2], color="#56B4E9", lw=0.6, zorder=1)
        sx, sy, _ = o["size"][aid]
        for t, al in ((tt - 1e6, 0.25), (tt, 0.85)):
            if tr[0, 0] <= t <= tr[-1, 0]:
                red = hit and aid.startswith(hit)
                ax.add_patch(mp.Polygon(np.array(L.box(*L.interp_pose(tr, t), sx, sy).exterior.coords), fc="#CC79A7" if red else "#56B4E9",
                                        ec="#0b5e8a", lw=0.5, alpha=al, zorder=5))
    for t, al in ((tt - 1e6, 0.3), (tt, 0.9)):
        ax.add_patch(mp.Polygon(np.array(L.box(*L.interp_pose(e, max(t, 0))).exterior.coords), fc="#D55E00", ec="k", lw=0.6, alpha=al, zorder=6))
    ax.add_patch(mp.Polygon(np.array(L.box(*L.interp_pose(g, tt)).exterior.coords), fc="none", ec="#222222", lw=1.0, ls="--", zorder=7))
    c = L.interp_pose(e, tt)[:2]
    ax.set_xlim(c[0] - w, c[0] + w), ax.set_ylim(c[1] - w, c[1] + w)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=6.5)


def sheet(scene, rows, out, frames_set):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    R, M = L.runs(), L.load("map")
    rd = L.road(M[scene])
    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.5, 1, 1], wspace=0.03, hspace=0.1)
    for i, n in enumerate(R):
        o = R[n][scene]
        r = next((x for x in rows if x["driver"] == n and x["scene"] == scene), None)
        ev = (r or {}).get("events") or []
        t_ev = int(round(ev[0]["t"] * 1e6)) if ev else None
        bits = [f"{n} score {o['summary']['score']:.2f} turn {L.turn_deg(o):.0f} v0 {L.speed_at(L.gt(o), 1e5):.1f} cmd {''.join(str(x['cmd']) for x in o['rec'])}"]
        for x in ev:
            bits.append(f"{x['flag']} t={x['t']} k={x['k']} v={x['ego_v']} log_v={x['log_v']} lat={x['lat']} ahead={x['ahead']}" + (
                f" {x['cls']} obj_v={x['obj_v']} dh={x['dh']} dx={x['dx']} dy={x['dy']}" if x["flag"] == "collision" else
                f" plan_out={x['n_plan_out']}/{x['k']} trk={x['track_lat_max']} beyond={x['beyond']} over={x['over']}"))
        bev(fig.add_subplot(gs[0, 2 * i:2 * i + 2]), o, rd, t_ev, "\n".join(bits), hit=next((x["obj"] for x in ev if x["flag"] == "collision"), None))
        fr = L.load(f"{n.lower()}_{frames_set}_frames")[scene]
        ks = sorted(fr)
        te = t_ev or ks[-1]
        tsel = [ks[1]] + [min(ks, key=lambda q: abs(q - (te - d))) for d in (1.0e6, 0.5e6, 0)]
        for j, t in enumerate(tsel):
            ax = fig.add_subplot(gs[1 + j // 2, 2 * i + j % 2])
            ax.imshow(Image.open(io.BytesIO(fr[t])))
            ax.set_title(f"{n} CAM_F0 t={t * 1e-6:.2f}s", fontsize=6.5, pad=1), ax.axis("off")
    fig.suptitle(scene, fontsize=7, y=0.94)
    fig.savefig(out, dpi=78, bbox_inches="tight")
    plt.close(fig)


def cmd_sheets(a):
    rows = json.loads((RES / "rows.json").read_text())
    scenes = (L.TMP / f"{a.set}.txt").read_text().split()
    if a.scene:
        scenes = [s for s in scenes if any(s.endswith(x) for x in a.scene)]
    out = L.TMP / "sheets"
    out.mkdir(exist_ok=True)
    for s in scenes:
        sheet(s, rows, out / f"{a.set}_{s[-16:]}.jpg", a.set)
        print(s, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows").set_defaults(fn=cmd_rows)
    p = sub.add_parser("sheets")
    p.add_argument("--set", default="zero"), p.add_argument("--scene", nargs="*")
    p.set_defaults(fn=cmd_sheets)
    a = ap.parse_args()
    a.fn(a)
