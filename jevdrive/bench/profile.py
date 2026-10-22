"""HUGSIM capability profile: what a driver can do, per manoeuvre class of the scenario's recorded route (P2, PD4/PD7).

HUGSIM has no lane map in its scenes (`load_HD_map: false`) and no traffic-light state or rule in its score (NC, DAC, TTC, COM, RC
only), so the classes come from the route geometry alone (routes.json: x right, z forward, yaw positive = turning right):

  turn L / turn R   heading range of the route >= 30 deg (the bench `turn23` rule), side = sign of the net heading change;
                    sharpness 30-60 / 60-100 / > 100 deg (a 60-100 deg turn is a junction turn; no map says junction)
  bend              heading range 10-30 deg
  lateral shift     heading range < 10 deg and the route ends >= 2 m to the side of the start heading line: the lane-change
                    proxy (no lane geometry, so a shift is not proven to be a lane change)
  straight          the rest

Per class: scenarios, HD-Score mean, the share of each end (complete / fg_coll / bg_coll / off_route / stuck / spin; the runner's
end class, spin first), all with 95 % CIs from a bootstrap clustered by scene (the difficulty variants of one scene resampled
together); for turns also where the run ended relative to the turn (progress RC x route length against the arc where 10 % / 90 %
of the heading change is reached: before / inside / through the turn); and example scenario ids per class x end for a video page.
"""
from __future__ import annotations

import numpy as np

ENDS = ("complete", "fg_coll", "bg_coll", "off_route", "stuck", "spin")


def route_class(v: dict) -> dict:
    """routes.json entry -> manoeuvre class and turn geometry."""
    xz = np.asarray(v["xz"], float)
    y = np.unwrap(np.asarray(v["yaw"], float))
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    rng, net = np.degrees(np.ptp(y)), np.degrees(y[-1] - y[0])
    h = np.array([np.sin(y[0]), np.cos(y[0])])                     # start heading in (x, z)
    side = float(np.cross(h, xz[-1] - xz[0]))                        # > 0: end point to the left of the start heading line
    out = dict(length=float(s[-1]), heading_range=float(rng), net_turn=float(net), end_offset=side)   # end_offset > 0: left
    if rng >= 30:
        c = np.abs(y - y[0]) / max(abs(y[-1] - y[0]), 1e-6)
        out |= dict(manoeuvre="turn " + ("R" if net > 0 else "L"), sharp="30-60" if rng < 60 else "60-100" if rng < 100 else ">100",
                    s_in=float(s[np.argmax(c >= 0.1)]), s_out=float(s[np.argmax(c >= 0.9)]))
    elif rng >= 10:
        out |= dict(manoeuvre="bend", sharp="")
    elif abs(side) >= 2.0:
        out |= dict(manoeuvre="lateral shift", sharp="")
    else:
        out |= dict(manoeuvre="straight", sharp="")
    return out


def classify(units, routes: dict):
    """units (one row per scenario: scene, rc, cls (runner end class, spin first), end, hdscore) -> + manoeuvre, sharp, turn_pass, end_class."""
    import pandas as pd
    R = pd.DataFrame({k: route_class(routes[k]) for k in set(units.scene) if k in routes}).T
    d = units.join(R, on="scene")
    end = d["cls"].where(d["cls"].isin(ENDS), d["end"].map({"fg_collision": "fg_coll", "bg_collision": "bg_coll", "off_route": "off_route",
                                                            "max_steps": "stuck", "complete": "complete"}))
    d["end_class"] = end.fillna("other")
    s_end = d["rc"].astype(float) * d["length"].astype(float)
    turn = d["manoeuvre"].astype(str).str.startswith("turn")
    d["turn_pass"] = np.where(~turn, "", np.where(s_end >= d.get("s_out", np.nan), "through",
                                                 np.where(s_end >= d.get("s_in", np.nan), "inside", "before")))
    return d


def _ci(x, g):
    from .. import stats
    r = stats.bootstrap(np.asarray(x, float), np.asarray(g))
    return r["mean"], r["lo"], r["hi"]


def profile(units, routes: dict, label: str, n_examples: int = 3) -> list:
    """Rows of the profile table of one arm (classes, then turn sharpness, then all)."""
    d = classify(units, routes)
    rows = []
    groups = [("class", c, d[d.manoeuvre == c]) for c in ("straight", "lateral shift", "bend", "turn L", "turn R") if (d.manoeuvre == c).any()]
    groups += [("turn sharpness", f"{sh} deg", d[d.sharp == sh]) for sh in ("30-60", "60-100", ">100") if (d.sharp == sh).any()]
    groups += [("all", "all", d)]
    for kind, val, g in groups:
        sc = g.scene.astype(str).to_numpy()
        m, lo, hi = _ci(g.hdscore, sc)
        r = {"arm": label, "by": kind, "class": val, "scenarios": len(g), "scenes": g.scene.nunique(), "HD": m, "HD lo": lo, "HD hi": hi}
        for e in ENDS:
            m, lo, hi = _ci(g.end_class == e, sc)
            r[e] = f"{int((g.end_class == e).sum())} ({m:.2f} [{lo:.2f}, {hi:.2f}])"
        t = g[g.turn_pass != ""]
        r["turn: before / inside / through"] = " / ".join(str(int((t.turn_pass == k).sum())) for k in ("before", "inside", "through")) if len(t) else ""
        ex = [f"{e}: " + ", ".join(g[g.end_class == e].scenario.astype(str)[:n_examples]) for e in ENDS if (g.end_class == e).any()]
        r["examples"] = "; ".join(ex) if kind == "class" else ""
        rows.append(r)
    return rows


NOTE = ("HUGSIM capability profile (jevdrive/bench/profile.py): classes from the recorded route only (no lane map, no traffic "
        "lights in HUGSIM); 'lateral shift' is the lane-change proxy; end shares with 95 % CIs clustered by scene; turn passage = "
        "where progress (RC x route length) ended against the arc holding 10-90 % of the heading change.")
