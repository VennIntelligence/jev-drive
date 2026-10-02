"""Map-side labels for the stop-position probe: junction entrance and stop line of a route from the OpenDRIVE file of
its town, no CARLA process (plan: experiments/vlm_arb/plans/2026-10-03-stoppos-probe.md, section Labels).

  parse(town)         driving-lane centre samples (0.5 m, CARLA frame x, -y) with road id / junction flag, plus the painted
                      stop lines ("StopLine*"), traffic-light posts (type 1000001) and stop signs (type 206)
  label_route(...)    entrance arc (first route point whose nearest driving-lane sample lies on a junction road), the
                      painted stop line on the route in [entrance - 15, entrance + 1] m, light / stop-sign presence near
                      the entrance
The labels are used as training / evaluation targets only, never as model input. Validation against the logged CARLA truth
of the 13 closed-loop junctions is `stoppos_labels.py validate`.
"""
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

XODR = {t: Path("/home/ujs/carlaCache/0.9.15/Carla/Maps") / (f"{t}/OpenDrive/{t}.xodr" if t in ("Town11", "Town12", "Town13", "Town15")
                                                                 else f"OpenDrive/{t}.xodr")
        for t in ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD", "Town11", "Town12",
                  "Town13", "Town15")}
STEP = 0.5
LIGHT_R, SIGN_R = 35.0, 35.0             # a junction "has" a light / stop sign if one stands within this radius of its entrance
LINE_WINDOW = (-15.0, 1.0)               # the agent's rule: stop line within [entrance - 15, entrance + 1] m
REAR_TO_BUMPER = 1.3886 + 2.4508         # lib/op_arb_agent.py


def cache_dir() -> Path:
    d = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "processed/vlm_arb_stoppos/xodr"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _poly(c, x):
    return c[0] + x * (c[1] + x * (c[2] + x * c[3]))


def _f(e, k, d=0.0):
    return float(e.get(k, d))


def _ref_line(geoms, s_q):
    """Reference line (x, y, hdg) at arc positions s_q (ascending) from the planView geometry list."""
    out = np.zeros((len(s_q), 3))
    starts = np.array([g[0] for g in geoms])
    idx = np.clip(np.searchsorted(starts, s_q, side="right") - 1, 0, len(geoms) - 1)
    for gi in np.unique(idx):
        s0, x0, y0, h0, ln, kind, par = geoms[gi]
        m = idx == gi
        u = np.clip(s_q[m] - s0, 0, ln)
        if kind == "line":
            x, y, h = x0 + u * np.cos(h0), y0 + u * np.sin(h0), np.full_like(u, h0)
        elif kind == "arc":
            k = par[0]
            if abs(k) < 1e-12:
                x, y, h = x0 + u * np.cos(h0), y0 + u * np.sin(h0), np.full_like(u, h0)
            else:
                h = h0 + k * u
                x, y = x0 + (np.sin(h) - np.sin(h0)) / k, y0 - (np.cos(h) - np.cos(h0)) / k
        elif kind == "spiral":
            k0, k1 = par
            n = max(int(ln / 0.05), 2)
            uu = np.linspace(0, ln, n + 1)
            hh = h0 + k0 * uu + 0.5 * (k1 - k0) / max(ln, 1e-9) * uu ** 2
            dx, dy = np.cos(hh), np.sin(hh)
            cx = np.r_[0, np.cumsum(0.5 * (dx[1:] + dx[:-1]) * np.diff(uu))]
            cy = np.r_[0, np.cumsum(0.5 * (dy[1:] + dy[:-1]) * np.diff(uu))]
            x, y, h = x0 + np.interp(u, uu, cx), y0 + np.interp(u, uu, cy), np.interp(u, uu, hh)
        elif kind == "ppoly":
            au, bu, cu, du, av, bv, cv, dv, norm = par
            p = u / ln if norm else u
            U = _poly((au, bu, cu, du), p)
            V = _poly((av, bv, cv, dv), p)
            x, y = x0 + U * np.cos(h0) - V * np.sin(h0), y0 + U * np.sin(h0) + V * np.cos(h0)
            dU = bu + p * (2 * cu + 3 * du * p)
            dV = bv + p * (2 * cv + 3 * dv * p)
            h = h0 + np.arctan2(dV, dU)
        else:  # poly3 in local frame
            a, b, c, d = par
            U, V = u, _poly((a, b, c, d), u)
            x, y = x0 + U * np.cos(h0) - V * np.sin(h0), y0 + U * np.sin(h0) + V * np.cos(h0)
            h = h0 + np.arctan(b + u * (2 * c + 3 * d * u))
        out[m] = np.stack([x, y, h], 1)
    return out


def _geoms(road):
    g = []
    for e in road.find("planView").findall("geometry"):
        c = e[0]
        kind = c.tag
        if kind == "line":
            par = ()
        elif kind == "arc":
            par = (_f(c, "curvature"),)
        elif kind == "spiral":
            par = (_f(c, "curvStart"), _f(c, "curvEnd"))
        elif kind == "paramPoly3":
            par = tuple(_f(c, k) for k in ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV")) + (c.get("pRange", "normalized") == "normalized",)
            kind = "ppoly"
        else:
            par = tuple(_f(c, k) for k in "abcd")
            kind = "poly3"
        g.append((_f(e, "s"), _f(e, "x"), _f(e, "y"), _f(e, "hdg"), _f(e, "length"), kind, par))
    return g


def _piece(items, s):
    """Last (s0, a, b, c, d) with s0 <= s (OpenDRIVE piecewise cubic), evaluated at s."""
    cur = items[0]
    for it in items:
        if it[0] <= s + 1e-9:
            cur = it
    return _poly(cur[1:], s - cur[0])


def parse(town: str) -> dict:
    f = cache_dir() / f"{town}.npz"
    if f.exists():
        return {k: v for k, v in np.load(f, allow_pickle=True).items()}
    tree = ET.parse(XODR[town])
    xs, ys, zs, rid, jun, ss, lid, hd = [], [], [], [], [], [], [], []
    sig = []   # kind, x, y, road, s, t
    for road in tree.getroot().findall("road"):
        L = float(road.get("length"))
        r_id, junction = int(road.get("id")), int(road.get("junction", -1)) != -1
        s_q = np.unique(np.r_[np.arange(0.0, L, STEP), L])
        ref = _ref_line(_geoms(road), s_q)
        el = [(_f(e, "s"), _f(e, "a"), _f(e, "b"), _f(e, "c"), _f(e, "d")) for e in
              (road.find("elevationProfile").findall("elevation") if road.find("elevationProfile") is not None else [])] or [(0, 0, 0, 0, 0)]
        off = [(_f(e, "s"), _f(e, "a"), _f(e, "b"), _f(e, "c"), _f(e, "d")) for e in road.find("lanes").findall("laneOffset")] or [(0, 0, 0, 0, 0)]
        secs = road.find("lanes").findall("laneSection")
        s_sec = [_f(e, "s") for e in secs] + [L + 1.0]
        for si, sec in enumerate(secs):
            m = (s_q >= s_sec[si] - 1e-9) & (s_q < s_sec[si + 1] - 1e-9) if si + 1 < len(secs) else (s_q >= s_sec[si] - 1e-9)
            if not m.any():
                continue
            sq = s_q[m]
            t_off = np.array([_piece(off, s) for s in sq])
            for side, sign in (("left", 1), ("right", -1)):
                node = sec.find(side)
                if node is None:
                    continue
                lanes = sorted(node.findall("lane"), key=lambda e: abs(int(e.get("id"))))
                inner = t_off.copy()
                for ln in lanes:
                    ws = [(_f(w, "sOffset"), _f(w, "a"), _f(w, "b"), _f(w, "c"), _f(w, "d")) for w in ln.findall("width")]
                    if not ws:
                        continue
                    w = np.array([_piece(ws, s - s_sec[si]) for s in sq])
                    ctr = inner + sign * w / 2
                    inner = inner + sign * w
                    if ln.get("type") != "driving":
                        continue
                    h = ref[m, 2]
                    xs.append(ref[m, 0] - ctr * np.sin(h))
                    ys.append(-(ref[m, 1] + ctr * np.cos(h)))
                    zs.append(np.array([_piece(el, s) for s in sq]))
                    rid.append(np.full(len(sq), r_id))
                    jun.append(np.full(len(sq), junction))
                    ss.append(sq)
                    lid.append(np.full(len(sq), int(ln.get("id"))))
                    hd.append(-h)
        sg = road.find("signals")
        if sg is not None:
            for e in sg.findall("signal"):
                kind = ("line" if e.get("name", "").startswith("StopLine") else "light" if e.get("type") == "1000001"
                        else "sign" if e.get("type") == "206" else None)
                if kind is None:
                    continue
                s, t = _f(e, "s"), _f(e, "t")
                p = _ref_line(_geoms(road), np.array([s]))[0]
                sig.append((kind, p[0] - t * np.sin(p[2]), -(p[1] + t * np.cos(p[2])), r_id, s, t))
    out = dict(x=np.concatenate(xs), y=np.concatenate(ys), z=np.concatenate(zs), road=np.concatenate(rid).astype(np.int32),
               junction=np.concatenate(jun), s=np.concatenate(ss).astype(np.float32), lane=np.concatenate(lid).astype(np.int16),
               hdg=np.concatenate(hd).astype(np.float32),
               sig_kind=np.array([k[0] for k in sig]), sig_xy=np.array([[k[1], k[2]] for k in sig]).reshape(-1, 2),
               sig_road=np.array([k[3] for k in sig]), sig_s=np.array([k[4] for k in sig]), sig_t=np.array([k[5] for k in sig]))
    np.savez(f.with_suffix(".tmp.npz"), **out)
    os.replace(f.with_suffix(".tmp.npz"), f)
    return out


@lru_cache(maxsize=None)
def _trees(town: str):
    from scipy.spatial import cKDTree
    m = parse(town)
    return m, cKDTree(np.c_[m["x"], m["y"], 3.0 * m["z"]]), cKDTree(np.c_[m["x"], m["y"]])


def label_route(town: str, xy: np.ndarray, z: np.ndarray | None = None) -> dict:
    """xy: (n, 2) dense route points in the CARLA frame (z optional, helps at bridges). Arc length is the polyline arc of xy.

    Returns dict(arc, flag, E (first flagged arc, None if no junction), S (painted stop line arc in the window, None),
    has_light, has_sign, line_cands)."""
    m, tree3, tree2 = _trees(town)
    xy = np.asarray(xy, float)
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    d, i = tree2.query(xy) if z is None else tree3.query(np.c_[xy, 3.0 * np.asarray(z, float)])
    flag = m["junction"][i].astype(bool)
    res = dict(arc=arc, flag=flag, near=d, E=None, S=None, has_light=False, has_sign=False, line_cands=[])
    if not flag.any():
        return res
    j0 = int(np.flatnonzero(flag)[0])
    E = float(arc[j0])
    res["E"] = E
    pe = xy[j0]
    k, sx = m["sig_kind"], m["sig_xy"]
    if len(sx):
        dd = np.hypot(*(sx - pe).T)
        res["has_light"] = bool(((k == "light") & (dd <= LIGHT_R)).any())
        res["has_sign"] = bool(((k == "sign") & (dd <= SIGN_R)).any())
        cand = []
        for p in sx[(k == "line") & (dd <= 60.0)]:
            dj = np.hypot(*(xy - p).T)
            a = int(dj.argmin())
            if dj[a] <= 1.8 and LINE_WINDOW[0] <= arc[a] - E <= LINE_WINDOW[1]:
                cand.append(float(arc[a]))
        res["line_cands"] = sorted(cand)
        res["S"] = cand and max(cand) or None
    return res


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or XODR:
        m = parse(t)
        print(t, len(m["x"]), "samples", int(m["junction"].sum()), "junction samples", len(m["sig_kind"]), "signals", flush=True)
