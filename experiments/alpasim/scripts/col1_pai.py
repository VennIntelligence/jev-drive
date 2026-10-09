"""COL1, PAI track: collision taxonomy and the pre-registered lead-output read (plans/2026-10-09-col1-lead-prereg.md) from
col1_pai_extract.py (logs.pkl) and col1_pai_replay.py (replay/<scene>.npz). Simulator state = labels only.

  col1_pai.py --x <extract dir> [<extract dir> ...] --ref <dir of reference json> --out <dir>      (numpy + shapely)

Writes <out>/pai_cases.csv (one row per rollout), pai_decisions.npz (per-decision signals and labels), pai_tables.md.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import col1_lib as L  # noqa: E402

FLAGS = ("collision_at_fault", "offroad", "left_corridor_laterally")
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def boxes(o, t):
    """{actor id: polygon} at time t, the ego included."""
    out = {}
    for k, tr in o["actors"].items():
        if tr[0, 0] - 1e5 <= t <= tr[-1, 0] + 1e5:
            p = L.interp(tr, t)[0]
            out[k] = Polygon(L.corners(*p, *o["size"][k][:2]))
    return out


def struck(o, t):
    """The actor nearest to the ego box around the first at-fault step -> (id, distance)."""
    best = (None, 1e9)
    for dt in (0, -1e5, 1e5):
        b = boxes(o, t + dt)
        e = b.pop("EGO")
        for k, p in b.items():
            d = e.distance(p)
            if d < best[1]:
                best = (k, d)
    return best


def rel(o, k, t, off):
    """Actor k seen from the ego at times t -> gap (bumper to nearest corner along the ego heading), lateral span (min, max), closing
    speed along the ego heading, object speed, heading difference."""
    ego, tr, (Lo, Wo, _) = o["actors"]["EGO"], o["actors"][k], o["size"][k]
    Le, We = o["size"]["EGO"][:2]
    pe, po = L.interp(ego, t), L.interp(tr, t)
    ve, vo = L.speed(ego, t), L.speed(tr, t)
    out = []
    for a, b, v1, v2, tt in zip(pe, po, ve, vo, np.atleast_1d(t)):
        c = L.into(a, L.corners(*b, Lo, Wo))
        ok = tr[0, 0] - 1e5 <= tt <= tr[-1, 0] + 1e5
        out.append([c[:, 0].min() - Le / 2 if ok else np.nan, c[:, 1].min(), c[:, 1].max(), v1 - v2 * np.cos(b[2] - a[2]), v2, L.wrap(b[2] - a[2])])
    return np.array(out)


def lead_label(o, t, off, reach=80.0):
    """Controls: the nearest actor whose box overlaps the ego's straight-ahead corridor (ego width + 0.5 m) -> gap, closing speed per t."""
    ego = o["actors"]["EGO"]
    Le, We = o["size"]["EGO"][:2]
    g, c = np.full(len(t), np.nan), np.zeros(len(t))
    pe, ve = L.interp(ego, t), L.speed(ego, t)
    for k, tr in o["actors"].items():
        if k == "EGO":
            continue
        po, vo = L.interp(tr, t), L.speed(tr, t)
        for i in range(len(t)):
            if not tr[0, 0] - 1e5 <= t[i] <= tr[-1, 0] + 1e5:
                continue
            q = L.into(pe[i], L.corners(*po[i], *o["size"][k][:2]))
            gap = q[:, 0].min() - Le / 2
            if q[:, 1].min() < We / 2 + 0.25 and q[:, 1].max() > -We / 2 - 0.25 and 0 < gap + 1.0 and gap < reach and (np.isnan(g[i]) or gap < g[i]):
                g[i], c[i] = gap, ve[i] - vo[i] * np.cos(po[i, 2] - pe[i, 2])
    return g, c


def a_need(g, c):
    c = np.maximum(c, 0)
    return np.where(np.isnan(g), 0.0, c**2 / (2 * np.maximum(g - 0.5 * c - 1.0, 0.1)))


def classify(o, k, t_evt, off, gt_rig):
    """Struck-object class at the first at-fault step (rules in the module of results/collisions.md)."""
    tr = o["actors"][k]
    r = rel(o, k, np.array([t_evt - 3e6, t_evt - 2e6, t_evt]), off)
    ylog = L.lat_to_path(gt_rig[:, 1:3], L.interp(tr, np.array([t_evt - 3e6, t_evt]))[:, :2])[:, 0]
    dh, vo = abs(r[2, 5]), r[2, 4]
    moved = np.hypot(*(tr[-1, 1:3] - tr[0, 1:3])) > 1.0
    lab = o["size"][k][2]
    if lab not in ("automobile", "heavy_truck", "bus", "trailer", "truck", "vehicle", "car", ""):
        cls = f"other ({lab})"
    elif dh > np.radians(135) and vo >= 1.0:
        cls = "oncoming"
    elif dh > np.radians(45) and vo >= 1.0:
        cls = "crossing"
    elif abs(ylog[1]) < 1.5 and abs(ylog[0]) >= 1.5 and moved:
        cls = "cut-in"
    elif abs(ylog[1]) < 1.5:
        cls = "lead stopped" if vo < 0.5 else "slow lead"
    elif not moved:
        cls = "parked / static"
    else:
        cls = "adjacent-lane vehicle"
    return cls, float(ylog[1]), float(vo), bool(moved), lab


def counterfactual(o, off, gt_rig, t_evt):
    """c1's two counterfactuals up to 0.3 s after the event: (a) the logged path at the driven arc position, (b) the log's arc position
    with the driven lateral offset. -> (clear_a, clear_b)."""
    ego = o["actors"]["EGO"]
    Le, We = o["size"]["EGO"][:2]
    path = gt_rig[:, 1:3]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    ts = ego[(ego[:, 0] <= t_evt + 3e5), 0]
    er = L.to_rig(L.interp(ego, ts), off)
    ls = L.lat_to_path(path, er[:, :2])
    lg = L.lat_to_path(path, L.interp(gt_rig, ts)[:, :2])

    def pose(s, lat):
        s = np.clip(s, 0, seg[-1] - 1e-3)
        i = min(np.searchsorted(seg, s, "right") - 1, len(path) - 2)
        d = path[i + 1] - path[i]
        h = np.arctan2(d[1], d[0])
        p = path[i] + d * (s - seg[i]) / max(seg[i + 1] - seg[i], 1e-6) + lat * np.array([-np.sin(h), np.cos(h)])
        return p[0] + off * np.cos(h), p[1] + off * np.sin(h), h
    clear = [True, True]
    for i, t in enumerate(ts):
        b = boxes(o, t)
        b.pop("EGO")
        for j, (s, lat) in enumerate(((ls[i, 1], 0.0), (lg[i, 1], ls[i, 0]))):
            e = Polygon(L.corners(*pose(s, lat), Le, We))
            if any(e.intersects(p) for p in b.values()):
                clear[j] = False
    return clear


def scene_rows(o, rp, ref):
    """One rollout -> (case row, per-decision arrays)."""
    off, ego, gt = L.center_off(o), o["actors"]["EGO"], o["logged"][0]["traj"]
    gt_rig = np.c_[gt[:, 0], L.to_rig(gt, off)]
    Le = o["size"]["EGO"][0]
    S = o["summary"]
    m = S["metrics"]
    ev = {f: L.first_event(o, f) for f in FLAGS}
    t0, T0 = rp["t0"].astype(float), ego[0, 0]
    er = L.to_rig(L.interp(ego, t0), off)
    ve = L.speed(ego, t0)
    lat = L.lat_to_path(gt_rig[:, 1:3], er[:, :2])
    arc = lambda p: np.hypot(*np.diff(np.concatenate([np.zeros((len(p), 1, 2)), p[:, :, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)  # noqa: E731
    cam_x = float(o["start"]["cam_t"][0])
    bump = off + Le / 2 - cam_x
    D = dict(t=(t0 - T0) * 1e-6, ve=ve, lat=lat[:, 0], cmd=rp["cmd"], arc_ft=arc(rp["ft"]), arc_ftS=arc(rp["ftS"]), arc_p0=arc(rp["p0"]),
             y4_ft=rp["ft"][:, -1, 1], y4_ftS=rp["ftS"][:, -1, 1], y4_p0=rp["p0"][:, -1, 1])
    D["x_ft"], D["x_p0"] = rp["ft"][:, :, 0], rp["p0"][:, :, 0]                  # plan x at 0.5 .. 4 s in the rig frame of t0
    for n in ("ft", "p0"):
        ld = rp[f"lead_{n}"][:, :72].reshape(-1, 3, 6, 4)
        D[f"p_{n}"], D[f"d_{n}"], D[f"vl_{n}"], D[f"y_{n}"] = sig(rp[f"lp_{n}"][:, 0]), ld[:, 0, 0, 0] - bump, ld[:, 0, 0, 2], ld[:, 0, 0, 1]
    rec = {r["k"]: r for r in o["rec"] if r.get("infer")}
    logged = np.array([rec[k]["poses"] for k in sorted(rec)][: len(t0)])
    D["replay_err"] = np.hypot(*(rp["run"][: len(logged), -1, :2] - logged[:, -1, :2]).T)
    row = dict(scene=o["summary"]["clipgt_id"][7:15], score=S["score"], flag=S.get("failure_reason") or "", collision_any=m["collision_any"],
               collision_rear=m["collision_rear"], v0=float(ve[0]), log_turn_deg=float(np.degrees(np.unwrap(gt[:, 3])[-1] - gt[0, 3])),
               log_dist=m["gt_dist_traveled_m"], driven=m["dist_traveled_m"], ref_alpamayo1=ref.get("alpamayo1", {}).get(o["summary"]["clipgt_id"]),
               ref_mean=np.mean([v[o["summary"]["clipgt_id"]] for v in ref.values() if o["summary"]["clipgt_id"] in v]) if ref else None)
    ho = (D["t"] > 1.45) & (D["t"] < 1.75)                        # forced-replay decisions with 8 real slots: the state is the log's
    if ho.any():
        row.update(v_handover=float(ve[ho].mean()), ft_v05=float(D["x_ft"][ho, 0].mean() / 0.5), p0_v05=float(D["x_p0"][ho, 0].mean() / 0.5),
                   ft_v4=float(D["arc_ft"][ho].mean() / 4), p0_v4=float(D["arc_p0"][ho].mean() / 4))
    te = ev["collision_at_fault"]
    if te is None:
        g, c = lead_label(o, t0, off)
        D["g"], D["c"], D["kind"] = g, c, np.zeros(len(t0), int)
        far = np.flatnonzero(np.abs(lat[:, 0]) > 4.0)
        D["ok"] = np.arange(len(t0)) < (far[0] if len(far) else len(t0))
        return row, D
    k, dist = struck(o, te)
    r = rel(o, k, t0, off)
    cls, ylog, vo, moved, lab = classify(o, k, te, off, gt_rig)
    D["g"], D["c"], D["ylo"], D["yhi"], D["kind"], D["ok"] = r[:, 0], r[:, 3], r[:, 1], r[:, 2], np.ones(len(t0), int), t0 <= te
    i_e = int(np.searchsorted(t0, te))
    ie = min(i_e, len(t0) - 1)
    # first decision at which the object is within 60 m ahead and inside the front-wide camera's field of view (60 deg half angle)
    cx, cy = r[:, 0] + Le / 2 + off - cam_x, (r[:, 1] + r[:, 2]) / 2
    vis = (cx > 0) & (r[:, 0] < 60) & (np.abs(np.arctan2(cy, np.maximum(cx, 0.1))) < np.radians(60)) & (t0 <= te)
    iv = int(np.flatnonzero(vis)[0]) if vis.any() else ie
    ttc = r[iv, 0] / r[iv, 3] if r[iv, 3] > 0.1 else np.inf
    pre = slice(max(ie - 50, 0), ie + 1)
    slow = D["arc_ft"][pre] / np.maximum(4 * ve[pre], 1.0)
    glog = L.to_rig(L.interp(gt, t0), off)
    hl = min(Polygon(L.corners(*L.interp(gt, t)[0], *o["size"]["EGO"][:2])).distance(Polygon(L.corners(*L.interp(o["actors"][k], t)[0], *o["size"][k][:2])))
             for t in gt[::2, 0] if o["actors"][k][0, 0] <= t <= o["actors"][k][-1, 0])
    ca, cb = counterfactual(o, off, gt_rig, te)
    front = m["collision_front"] > 0
    # does a plan reach the object within 4 s if the object keeps its speed (plan x minus the object's travel against the gap)
    th = np.arange(1, 9) * 0.5
    for n in ("ft", "p0"):
        D[f"thru_{n}"] = ((D[f"x_{n}"] - np.maximum(ve - r[:, 3], 0)[:, None] * th) >= r[:, 0][:, None]).any(1)
    w4 = slice(max(ie - 40, 0), ie)
    row.update(thru_ft_last4s=float(D["thru_ft"][w4].mean()) if ie else np.nan, thru_p0_last4s=float(D["thru_p0"][w4].mean()) if ie else np.nan,
               arc_p0_over_ft_last4s=float(np.median(D["arc_p0"][w4] / np.maximum(D["arc_ft"][w4], 0.5))) if ie else np.nan)
    w8 = slice(max(ie - 80, 0), ie)
    We = o["size"]["EGO"][1]
    ahead = (r[w8, 0] > 0) & (r[w8, 1] < We / 2 + 0.25) & (r[w8, 2] > -We / 2 - 0.25)      # inside the ego's own straight-ahead corridor
    hitp = (D["p_p0"][w8] >= 0.5) & (np.abs(D["d_p0"][w8] - r[w8, 0]) <= np.maximum(2.0, 0.3 * r[w8, 0]))
    row.update(ahead_s=float(ahead.sum() * 0.1), ahead_hit_p0=float((hitp & ahead).sum() / max(ahead.sum(), 1)))
    row.update(t_evt=(te - T0) * 1e-6, obj=k, obj_label=lab, cls=cls, contact="front" if front else "lateral", v_ego_evt=float(ve[ie]), v_obj_evt=vo,
               obj_lat_to_log_path=ylog, obj_moved=moved, driven_to_evt=float(np.hypot(*np.diff(ego[ego[:, 0] <= te, 1:3], axis=0).T).sum()),
               ego_lat_evt=float(lat[ie, 0]), ego_ahead_of_log=float(lat[ie, 1] - L.lat_to_path(gt_rig[:, 1:3], glog[ie:ie + 1, :2])[0, 1]),
               t_visible_before=float((te - t0[iv]) * 1e-6), gap_first_visible=float(r[iv, 0]), ttc_first_visible=float(ttc),
               plan_ratio_min=float(slow.min()), plan_ratio_last1s=float(np.median(slow[-10:])), plan_slowed=bool((slow < 0.9).any()),
               plan_arc_vs_log_4s=float(np.median(D["arc_ft"][pre] / np.maximum(np.hypot(*(L.interp(gt_rig, t0[pre] + 4e6)[:, :2] - glog[pre, :2]).T), 1.0))),
               human_min_dist=float(hl), clear_on_log_path=ca, clear_at_log_position=cb, standstill_start=bool(ve[0] < 1.0))
    return row, D


def s_time(D, src, te_idx):
    """Seen in time (prereg quantity 1) for one class-A rollout -> (seen, lead time s, n hit)."""
    w = slice(max(te_idx - 80, 0), te_idx + 1)
    g, c, p, d, t = D["g"][w], D["c"][w], D[f"p_{src}"][w], D[f"d_{src}"][w], D["t"][w]
    hit = (p >= 0.5) & (np.abs(d - g) <= np.maximum(2.0, 0.3 * g))
    ok = hit & (a_need(g, c) <= 4.0)
    return bool(ok.any()), float(t[-1] - t[ok][0]) if ok.any() else np.nan, int(hit.sum())


def guard(D, src, ps, a_s):
    p, d, vl, ve = D[f"p_{src}"], D[f"d_{src}"], D[f"vl_{src}"], D["ve"]
    req = np.maximum(ve - vl, 0) ** 2 / (2 * np.maximum(d - 4.0, 0.5))
    return (p >= ps) & ((req >= a_s) | ((ve < 0.5) & (d < 6.0)))


def run_len(x, n):
    """True if x has a run of n consecutive True."""
    c = 0
    for v in x:
        c = c + 1 if v else 0
        if c >= n:
            return True
    return False


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if not len(v):
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    return tuple(np.percentile(v[r.integers(0, len(v), (n, len(v)))].mean(1), [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", nargs="+", required=True), ap.add_argument("--ref", default=""), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {}
    for f in sorted(Path(a.ref).glob("*.json")) if a.ref else []:
        try:
            j = json.loads(f.read_text())
            rows = j["rollouts"] if isinstance(j, dict) and "rollouts" in j else j
            sc = {}
            for r in rows:
                sc.setdefault(r["clipgt_id"], []).append(r["score"])
            ref[f.stem] = {k: float(np.mean(v)) for k, v in sc.items()}
        except Exception as e:
            print("ref", f.name, "not read:", e)
    rows, Ds = [], {}
    for x in a.x:
        logs = L.load(f"{x}/logs.pkl")
        for s, o in sorted(logs.items()):
            f = Path(x) / "replay" / f"{s}.npz"
            if not f.exists() or "rec" not in o:
                print("skip", s)
                continue
            row, D = scene_rows(o, dict(np.load(f)), ref)
            row["run"] = Path(x).name
            rows.append(row), Ds.__setitem__(row["scene"], D)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "pai_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(out / "pai_decisions.npz", **{f"{s}/{k}": v for s, D in Ds.items() for k, v in D.items()})

    md = []
    P = md.append
    n = len(rows)
    col = [r for r in rows if r["flag"] == "collision_at_fault"]
    P(f"# COL1 PAI tables ({n} rollouts of P2H10-F-s0, one plan per call, Harmonizer off)\n")
    P(f"Mean scene score {np.mean([r['score'] for r in rows]):.4f}; zeros {sum(r['score'] == 0 for r in rows)}: "
      + ", ".join(f"{f} {sum(r['flag'] == f for r in rows)}" for f in FLAGS) + f"; replay reproduces the run's plan (4 s end point within 0.05 m) in "
      f"{np.mean(np.concatenate([D['replay_err'] for D in Ds.values()]) <= 0.05):.1%} of decisions "
      f"(max {max(D['replay_err'].max() for D in Ds.values()):.3f} m).\n")
    P("## At-fault collisions, one row per rollout\n")
    P("| scene | class | contact | t s | driven m | ego m/s | object m/s | ego off the logged path m | ego ahead of the log m | first visible: s before / gap m / TTC s | "
      "plan 4 s arc / (4 v): min, last 1 s | plan / log 4 s | last 4 s: served plan reaches the object / shipped plan reaches it / shipped arc over served | in the ego's own corridor ahead, s of the last 8 / shipped lead head on it then | log's own min distance m | clear on the logged path | clear at the log's position | alpamayo1 | start m/s |")
    P("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|")
    for r in col:
        P(f"| {r['scene']} | {r['cls']} | {r['contact']} | {r['t_evt']:.1f} | {r['driven_to_evt']:.0f} | {r['v_ego_evt']:.1f} | {r['v_obj_evt']:.1f} | "
          f"{r['ego_lat_evt']:+.2f} | {r['ego_ahead_of_log']:+.1f} | {r['t_visible_before']:.1f} / {r['gap_first_visible']:.1f} / {r['ttc_first_visible']:.1f} | "
          f"{r['plan_ratio_min']:.2f}, {r['plan_ratio_last1s']:.2f} | {r['plan_arc_vs_log_4s']:.2f} | "
          f"{r['thru_ft_last4s']:.0%} / {r['thru_p0_last4s']:.0%} / {r['arc_p0_over_ft_last4s']:.2f} | {r['ahead_s']:.1f} / {r['ahead_hit_p0']:.0%} | {r['human_min_dist']:.2f} | "
          f"{'yes' if r['clear_on_log_path'] else 'no'} | {'yes' if r['clear_at_log_position'] else 'no'} | "
          f"{'' if r['ref_alpamayo1'] is None else format(r['ref_alpamayo1'], '.2f')} | {r['v0']:.1f} |")
    P("\n## Class counts\n\n| class | n | share | ego off the logged path > 1 m | ego ahead of the log > 2 m | clear at the log's position | alpamayo1 mean |\n|---|--:|--:|--:|--:|--:|--:|")
    for c in sorted({r["cls"] for r in col}):
        q = [r for r in col if r["cls"] == c]
        al = [r["ref_alpamayo1"] for r in q if r["ref_alpamayo1"] is not None]
        P(f"| {c} | {len(q)} | {len(q) / max(len(col), 1):.0%} | {sum(abs(r['ego_lat_evt']) > 1 for r in q)} | {sum(r['ego_ahead_of_log'] > 2 for r in q)} | "
          f"{sum(r['clear_at_log_position'] for r in q)} | {np.mean(al) if al else float('nan'):.2f} |")

    # ---- pre-registered lead read
    CA = [r for r in col if r["cls"] in ("lead stopped", "slow lead")]
    N = [r for r in rows if r["collision_any"] == 0]
    P(f"\n## Lead outputs (pre-registered)\n\nC = {len(col)} at-fault-collision rollouts, C_A = {len(CA)} (class A), N = {len(N)} control rollouts "
      f"({sum(int(Ds[r['scene']]['ok'].sum()) for r in N)} decisions).\n")
    P("### Quantity 1: seen in time, per class-A rollout\n\n| scene | class | P0 seen in time | P0 lead time s | P0 hit decisions in the last 8 s | FT seen in time | FT lead time s | FT hits |\n|---|---|---|--:|--:|---|--:|--:|")
    st = {"p0": [], "ft": []}
    for r in CA:
        D = Ds[r["scene"]]
        ie = int(D["ok"].sum()) - 1
        v = {s: s_time(D, s, ie) for s in ("p0", "ft")}
        for s in v:
            st[s].append(v[s][0])
        P(f"| {r['scene']} | {r['cls']} | {'yes' if v['p0'][0] else 'no'} | {v['p0'][1]:.1f} | {v['p0'][2]} | {'yes' if v['ft'][0] else 'no'} | {v['ft'][1]:.1f} | {v['ft'][2]} |")
    for s in ("p0", "ft"):
        lo, hi = boot(st[s])
        P(f"\nS_time ({s.upper()}) = {np.mean(st[s]) if st[s] else float('nan'):.2f} [{lo:.2f}, {hi:.2f}] (n = {len(st[s])})")
    P("\n### Lead head against the label by gap (all decisions with a labelled object ahead: collision rollouts up to impact, controls)\n")
    P("| gap m | decisions | P0: p >= 0.5 | P0: median d - g m | P0: |d - g| <= max(2, 0.3 g) | FT: p >= 0.5 | FT: median d - g m | FT: within |\n|---|--:|--:|--:|--:|--:|--:|--:|")
    allD = {k: np.concatenate([Ds[r["scene"]][k][Ds[r["scene"]]["ok"]] for r in col + N]) for k in ("g", "p_p0", "d_p0", "p_ft", "d_ft")}
    for lo, hi in ((-2, 5), (5, 10), (10, 20), (20, 40), (40, 80)):
        q = (allD["g"] >= lo) & (allD["g"] < hi)
        if q.sum():
            g = allD["g"][q]
            cell = lambda s: (f"{np.mean(allD[f'p_{s}'][q] >= 0.5):.0%} | {np.median(allD[f'd_{s}'][q] - g):+.1f} | "  # noqa: E731
                              f"{np.mean(np.abs(allD[f'd_{s}'][q] - g) <= np.maximum(2, 0.3 * g)):.0%}")
            P(f"| {max(lo, 0)}-{hi} | {int(q.sum())} | {cell('p0')} | {cell('ft')} |")
    q = np.isnan(allD["g"])
    P(f"| no object in the corridor | {int(q.sum())} | {np.mean(allD['p_p0'][q] >= 0.5):.0%} | | | {np.mean(allD['p_ft'][q] >= 0.5):.0%} | | |")
    P("\n### Quantity 2: guard ceiling G(p*, a*)\n\n| source | p* | a* | turned / C | turned among class A | post hoc: turned with the lead head on the struck object | too late (trigger, no stop possible) | no trigger | "
      "false-alarm decisions | false-alarm scenes (>= 1.0 s) / N | supported triggers on controls (scenes) |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    notes = []
    for s in ("p0", "ft"):
        for ps in (0.3, 0.5, 0.7, 0.9):
            for a_s in (1.0, 1.5, 2.5):
                turned, late, none, tA, strict = [], 0, 0, 0, 0
                for r in col:
                    D = Ds[r["scene"]]
                    ok = D["ok"]
                    tr = guard(D, s, ps, a_s)[ok]
                    an = a_need(D["g"][ok], D["c"][ok])
                    t = bool((tr & (an <= 4.0)).any())
                    hit = np.abs(D[f"d_{s}"][ok] - D["g"][ok]) <= np.maximum(2.0, 0.3 * D["g"][ok])
                    strict += bool((tr & hit & (an <= 4.0)).any())          # post hoc: the lead head is on the struck object at the trigger
                    turned.append(t)
                    tA += t and r["cls"] in ("lead stopped", "slow lead")
                    late += (not t) and bool(tr.any())
                    none += not tr.any()
                fa_d, fa_s, n_d, sup = 0, [], 0, 0
                for r in N:
                    D = Ds[r["scene"]]
                    ok = D["ok"]
                    tr = guard(D, s, ps, a_s)[ok]
                    an = a_need(D["g"][ok], D["c"][ok])
                    false = tr & (an <= 0.5)
                    fa_d += int(false.sum())
                    n_d += int(ok.sum())
                    fa_s.append(run_len(false, 10))
                    sup += bool((tr & (an > 0.5)).any())
                star = " **(primary)**" if (ps, a_s) == (0.5, 1.5) else ""
                if star:
                    notes.append(f"{s.upper()} primary point, turned: " + ", ".join(f"{r['scene']} ({r['cls']})" for r, t in zip(col, turned) if t)
                                 + "; not turned: " + ", ".join(f"{r['scene']} ({r['cls']})" for r, t in zip(col, turned) if not t))
                ci = boot(turned) if star else None
                ci2 = boot(fa_s) if star else None
                P(f"| {s.upper()}{star} | {ps} | {a_s} | {sum(turned)} / {len(col)}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "") +
                  f" | {tA} / {len(CA)} | {strict} / {len(col)} | {late} | {none} | {fa_d} / {n_d} ({fa_d / max(n_d, 1):.1%}) | {sum(fa_s)} / {len(N)}"
                  + (f" [{ci2[0]:.2f}, {ci2[1]:.2f}]" if ci2 else "") + f" | {sup} |")
    P("\n" + "\n\n".join(notes))
    P("\n## Command and shipped-plan sensitivity (offline replay, same tokens)\n")
    P("| scene | flag | plan 4 s end, fed command vs straight: median / p95 abs lateral m | 4 s arc, fed / straight (median ratio) | "
      "4 s arc, shipped P0 / served (median ratio) | served arc / (4 v) median | P0 arc / (4 v) median |\n|---|---|--:|--:|--:|--:|--:|")
    for r in rows:
        D = Ds[r["scene"]]
        ok = D["ok"] & (D["t"] > 1.5)
        if not ok.any():
            continue
        dy = np.abs(D["y4_ft"] - D["y4_ftS"])[ok]
        v4 = np.maximum(4 * D["ve"][ok], 1.0)
        P(f"| {r['scene']} | {r['flag']} | {np.median(dy):.2f} / {np.percentile(dy, 95):.2f} | {np.median(D['arc_ft'][ok] / np.maximum(D['arc_ftS'][ok], 0.5)):.2f} | "
          f"{np.median(D['arc_p0'][ok] / np.maximum(D['arc_ft'][ok], 0.5)):.2f} | {np.median(D['arc_ft'][ok] / v4):.2f} | {np.median(D['arc_p0'][ok] / v4):.2f} |")
    P("\n## Plan speed against the ego's speed just before the hand-over (decisions at 1.45-1.75 s, the state is the log's)\n")
    P("| scene | ego m/s | served: first 0.5 s, 4 s mean (m/s, % of ego) | shipped P0: first 0.5 s, 4 s mean | flag |\n|---|--:|--:|--:|---|")
    pc = lambda v, r: f"{v:.1f} ({100 * (v / max(r, 0.5) - 1):+.0f}%)"  # noqa: E731
    for r in sorted((r for r in rows if "v_handover" in r), key=lambda r: r["v_handover"]):
        P(f"| {r['scene']} | {r['v_handover']:.1f} | {pc(r['ft_v05'], r['v_handover'])}, {pc(r['ft_v4'], r['v_handover'])} | "
          f"{pc(r['p0_v05'], r['v_handover'])}, {pc(r['p0_v4'], r['v_handover'])} | {r['flag']} |")
    for lo, hi in ((-1, 2), (2, 5), (5, 13), (13, 23), (23, 99)):
        q = [r for r in rows if "v_handover" in r and lo <= r["v_handover"] < hi]
        if q:
            f = lambda k: np.median([r[k] / max(r["v_handover"], 0.5) - 1 for r in q])  # noqa: E731
            P(f"\nego {max(lo, 0)}-{hi} m/s (n = {len(q)}): median first-0.5 s speed against the ego's, served {100 * f('ft_v05'):+.0f}%, shipped {100 * f('p0_v05'):+.0f}%; "
              f"4 s mean, served {100 * f('ft_v4'):+.0f}%, shipped {100 * f('p0_v4'):+.0f}%.")
    with open(out / "pai_lead_classA.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "cls", "t_to_impact", "ve", "gap", "closing", "p_p0", "d_p0", "vl_p0", "p_ft", "d_ft", "arc_ft", "arc_p0"])
        for r in col:
            D = Ds[r["scene"]]
            for i in np.flatnonzero(D["ok"]):
                w.writerow([r["scene"], r["cls"]] + [round(float(x), 3) for x in (D["t"][i] - r["t_evt"], D["ve"][i], D["g"][i], D["c"][i], D["p_p0"][i],
                                                                               D["d_p0"][i], D["vl_p0"][i], D["p_ft"][i], D["d_ft"][i], D["arc_ft"][i], D["arc_p0"][i])])
    (out / "pai_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
