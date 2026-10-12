#!/usr/bin/env python3
"""CAP-AUDIT read (plans/2026-10-12-cap-audit-prereg.md): per-scene capability measures from ca.py collect, deficits against the log,
our excess deficit over the reference drivers by capability x scenario group, flags, classification, levers, heatmaps, BEV picks.

  ca_report.py tables --pub DIR --held DIR [--out experiments/cap_audit/results] [--figs experiments/cap_audit/figs]
  ca_report.py bevfig --bev bev.pkl --out experiments/cap_audit/figs/ca_bev        (after ca.py bev on the box)

DIR = a copy of a collect run dir (dec / log / roll / scene parquet). Mac, project venv.
CIs: log-clustered bootstrap in jevdrive.stats' convention (default_rng(0), 10 000 draws of logs, ratio of sums); one draw of logs for
every driver and cell of a set, so differences and the minimum over references are under one resample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "research"))
OURS = {"YR10m10-F-s0": ["YR10m10-F-s0"], "YS2G": ["YS2G-L18-s0", "YS2G-L18-s1"], "YS-L18": ["YS-L18-s0", "YS-L18-s1"]}
REFS = ["WA-JEPA", "LTF", "DD", "G0014"]
DESC = ["APY10m10-AB-s0"]
THW_CAP, GAP_CAP, Z_STRONG, B = 4.0, 5.0, 3.75, 10_000
# measure -> (class, unit, needs a log value); every measure is oriented so that larger = worse
MEAS = {
    "lat_log": ("lateral", "m", False), "lane_ctr": ("lateral", "m", True), "head_lane": ("lateral", "deg", True),
    "head_path": ("lateral", "deg", False), "drift": ("lateral", "m/dec", False),
    "speed_err": ("longitudinal", "m/s", False), "behind": ("longitudinal", "m", False), "ahead": ("longitudinal", "m", False),
    "lead_close": ("longitudinal", "s", False), "stop_speed": ("longitudinal", "m/s", False), "brake_miss": ("longitudinal", "m/s", False),
    "onc": ("legality", "share", True), "nolane": ("legality", "share", True), "margin": ("legality", "m", True), "cor_half": ("legality", "share", False),
    "gap_min": ("interaction", "m", True), "near_miss": ("interaction", "share", False),
}
TAGF = {
    "intersection": lambda t: t in ("on_intersection", "traversing_intersection", "on_all_way_stop_intersection") or "traffic_light_intersection" in t
    or (t.startswith("starting_") and t.endswith("_intersection_traversal")),
    "turn_start": lambda t: t in ("starting_left_turn", "starting_right_turn", "starting_high_speed_turn") or (t.startswith("starting_") and t.endswith("_turn")),
    "stop_tl": lambda t: t.startswith("on_stopline_") or t.startswith("stationary_at_traffic_light") or t.startswith("stopping_")
    or t.startswith("accelerating_at_traffic_light") or t in ("accelerating_at_stop_sign", "accelerating_at_crosswalk"),
    "pedestrian": lambda t: t.startswith("near_pedestrian_") or t == "traversing_crosswalk",
    "pickup": lambda t: t in ("on_pickup_dropoff", "traversing_pickup_dropoff"),
    "lead": lambda t: t.startswith("following_lane_with_") or t == "stationary_in_traffic" or t.endswith("_with_lead") or t in ("behind_long_vehicle", "near_multiple_vehicles"),
    "stationary": lambda t: t == "stationary",
    "obstacle": lambda t: t in ("near_construction_zone_sign", "near_trafficcone_on_driveable", "near_long_vehicle", "near_high_speed_vehicle"),
    "lane_change": lambda t: t.startswith("changing_lane"),
}
SPEEDTAG = lambda t: t.endswith("_magnitude_speed") or t in ("high_lateral_acceleration", "high_magnitude_jerk")    # noqa: E731
GEOM = ["straight", "gentle", "kerb_turn", "across_turn"]
SPEED = ["v<0.5", "v0.5-5", "v5-10", "v>=10"]
NONTURN = {"straight", "gentle", "plain", *SPEED}
TURNISH = {"kerb_turn", "across_turn", "intersection", "turn_start", "pickup"}


# ---------------------------------------------------------------- per-scene measures
def load(d):
    d = Path(d)
    D, L, Ro, S = (pd.read_parquet(d / f"{n}.parquet") for n in ("dec", "log", "roll", "scene"))
    return D, L, Ro, S


def per_scene(D, L, Ro):
    """-> long table (label, scene, measure, val, log) on each driver's kept decisions."""
    L = L.rename(columns={c: f"l_{c}" for c in L.columns if c not in ("scene", "k")})
    X = D[D.kept].merge(L, on=["scene", "k"], how="left")
    f = lambda c: pd.to_numeric(X[c], errors="coerce").astype(float)                      # noqa: E731
    X["lat_log"] = f("lat_eval")
    X["lane_ctr_e"], X["lane_ctr_l"] = f("lane_off").abs(), f("l_lane_off").abs()
    X["head_lane_e"], X["head_lane_l"] = f("lane_dh").abs(), f("l_lane_dh").abs()
    X["head_path_e"] = f("head_path").abs()
    X["speed_err_e"] = (f("v") - f("l_v")).abs()
    X["dv"] = f("v") - f("l_v")
    thw = lambda g, v: np.where(np.isfinite(g), np.minimum(g / np.maximum(v, 0.5), THW_CAP), THW_CAP)           # noqa: E731
    lead = np.isfinite(f("lead_gap"))
    X["lead_close_e"] = np.where(lead, np.maximum(0, thw(f("l_lead_gap"), f("l_v")) - thw(f("lead_gap"), f("v"))), np.nan)
    X["stop_speed_e"] = np.where(f("l_v") < 0.5, f("v"), np.nan)
    brk = f("l_v1") <= f("l_v") - 1.0
    X["brake_miss_e"] = np.where(brk, np.maximum(0, (f("v1") - f("v")) - (f("l_v1") - f("l_v"))), np.nan)
    X["onc_e"], X["onc_l"] = (X.cls == 2).astype(float), (X.l_cls == 2).astype(float)
    X["nolane_e"], X["nolane_l"] = (X.cls == 0).astype(float), (X.l_cls == 0).astype(float)
    g = X.groupby(["label", "scene"])
    out = pd.DataFrame(index=g.size().index)
    for m in ("lane_ctr", "head_lane", "onc", "nolane"):
        out[f"{m}|e"], out[f"{m}|l"] = g[f"{m}_e"].mean(), g[f"{m}_l"].mean()
    for m in ("head_path", "speed_err", "lead_close", "stop_speed", "brake_miss"):
        out[f"{m}|e"] = g[f"{m}_e"].mean()
    out["lat_log|e"] = g.lat_log.mean()
    out["dv"] = g.dv.mean()
    out["margin|e"], out["margin|l"] = g.margin.min(), g.l_margin.min()
    out["gap_min|e"], out["gap_min|l"] = g.gap.min().clip(0, GAP_CAP), g.l_gap.min().clip(0, GAP_CAP)
    out["near_miss|e"] = ((out["gap_min|e"] < 0.5) & (out["gap_min|l"] >= 1.0)).astype(float)
    out["cor_half|e"] = (g.lat_log.max() >= 2.0).astype(float)
    last = X.loc[g.k.idxmax()].set_index(["label", "scene"])
    out["behind|e"] = np.maximum(0, last.l_s - last.s).reindex(out.index)
    out["ahead|e"] = np.maximum(0, last.s - last.l_s).reindex(out.index)

    def slope(x):
        x = x.dropna(subset=["lat_log"])
        return np.polyfit(x.k, x.lat_log, 1)[0] if len(x) >= 3 else np.nan
    out["drift|e"] = g[["k", "lat_log"]].apply(slope)
    # in-lane-ahead flag of the closest vehicle at the decision of the minimum gap (classification of interaction cells)
    gi = X.loc[X.groupby(["label", "scene"]).gap.idxmin()].set_index(["label", "scene"])
    out["gap_inlane"] = ((gi.gap_x > 0) & (gi.gap_y.abs() < 1.5)).astype(float).reindex(out.index)
    # serving (ours): |plan - executed| at +1 s and |plan offset 10 m ahead|
    if "plan_exec" in X:
        out["plan_exec"] = g.plan_exec.mean()
        X["plan_off10a"] = f("plan_off10").abs()
        out["plan_off10"] = g.plan_off10a.mean()
    out = out.reset_index()
    # deficits, oriented larger = worse
    for m, (_, _, haslog) in MEAS.items():
        if m in ("margin", "gap_min"):
            out[f"{m}|d"] = out[f"{m}|l"] - out[f"{m}|e"]
        elif haslog:
            out[f"{m}|d"] = out[f"{m}|e"] - out[f"{m}|l"]
        else:
            out[f"{m}|d"] = out[f"{m}|e"]
    sc = Ro[Ro.has].set_index(["label", "scene"]).score.astype(float)
    out["score"] = sc.reindex(pd.MultiIndex.from_frame(out[["label", "scene"]])).to_numpy()
    return out


def seed_means(P):
    """Add the seed-mean labels (per-scene NaN-aware mean of the seeds)."""
    cols = [c for c in P.columns if c not in ("label", "scene")]
    extra = []
    for name, labs in OURS.items():
        if len(labs) > 1:
            x = P[P.label.isin(labs)].groupby("scene")[cols].mean().reset_index()
            x["score_min"] = P[P.label.isin(labs)].groupby("scene").score.min().to_numpy()
            x["label"] = name
            extra.append(x)
    P = pd.concat([P] + extra, ignore_index=True)
    P["score_min"] = P.score_min.fillna(P.score) if "score_min" in P else P.score
    return P


# ---------------------------------------------------------------- groups
def groups(S, T):
    S = S.copy()
    S["log"] = S.scene.str.rsplit("-", n=1).str[0]
    dp = S.dpsi_log.to_numpy()
    turn = np.abs(dp) >= 30
    across = np.sign(dp) * S.s_onc.to_numpy() > 0             # left turn in right-hand traffic, right turn in Singapore
    S["geom"] = np.where(np.abs(dp) < 15, "straight", np.where(~turn, "gentle", np.where(across, "across_turn", "kerb_turn")))
    S["speed"] = pd.cut(S.v0.fillna(0), [-1, 0.5, 5, 10, 1e9], right=False, labels=SPEED).astype(str)
    T = T.set_index("scene")
    tl = S.scene.map(lambda s: [t for t in str(T.tags.get(s, "") or "").split("|") if t])
    G = {"all": np.ones(len(S), bool)}
    G |= {g: (S.geom == g).to_numpy() for g in GEOM}
    G |= {g: (S.speed == g).to_numpy() for g in SPEED}
    G |= {f: tl.map(lambda ts, fn=fn: any(fn(t) for t in ts)).to_numpy() for f, fn in TAGF.items()}
    G["plain"] = tl.map(lambda ts: all(SPEEDTAG(t) for t in ts)).to_numpy()
    return S, {g: set(S.scene[m]) for g, m in G.items()}


# ---------------------------------------------------------------- bootstrap
class Boot:
    def __init__(self, logs, n=B, seed=0):
        self.logs = np.array(sorted(set(logs)))
        self.ix = {g: i for i, g in enumerate(self.logs)}
        idx = np.random.default_rng(seed).integers(len(self.logs), size=(n, len(self.logs)))
        self.M = np.stack([np.bincount(r, minlength=len(self.logs)) for r in idx]).astype(float)

    def draws(self, x, logs):
        """Per-scene values -> (point, (B,) bootstrap means, n scenes, n logs)."""
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        i = np.array([self.ix[g] for g in np.asarray(logs)[ok]], int)
        s, c = np.bincount(i, x[ok], len(self.logs)), np.bincount(i, minlength=len(self.logs)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (s.sum() / c.sum() if c.sum() else np.nan), (self.M @ s) / (self.M @ c), int(c.sum()), int((c > 0).sum())


def ci(b):
    b = b[np.isfinite(b)]
    return (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) else (np.nan, np.nan)


def cell_table(P, S, G, boot, survived=False):
    """All cells: per driver value / deficit, per our row the excess over each reference and m* = min over references."""
    logs = S.set_index("scene").log
    W = {lab: x.set_index("scene") for lab, x in P.groupby("label")}
    rows = []
    for g, members in G.items():
        for m in MEAS:
            c = f"{m}|d"
            base = dict(group=g, measure=m)
            for lab in list(OURS) + REFS + DESC:
                x = W[lab][c].reindex(sorted(members & set(W[lab].index)))
                pt, _, n, nl = boot.draws(x, logs.reindex(x.index))
                lv = W[lab][f"{m}|l"].reindex(x.index) if MEAS[m][2] else None
                rows.append(dict(base, row="driver", label=lab, value=pt, n=n, logs=nl,
                                 log_value=float(np.nanmean(lv)) if lv is not None and len(lv) else (0.0 if not MEAS[m][2] else np.nan)))
            for o in OURS:
                diffs, pts, ns = [], [], []
                for r in REFS:
                    sc = sorted(members & set(W[o].index) & set(W[r].index))
                    if survived:
                        sc = [s for s in sc if W[o].score_min.get(s, 0) > 0 and W[r].score.get(s, 0) > 0]
                    dx = W[o][c].reindex(sc) - W[r][c].reindex(sc)
                    pt, bd, n, nl = boot.draws(dx, logs.reindex(sc))
                    diffs.append(bd)
                    pts.append(pt)
                    ns.append((n, nl))
                    lo, hi = ci(bd)
                    rows.append(dict(base, row="excess", label=o, ref=r, value=pt, lo=lo, hi=hi, n=n, logs=nl))
                Dm = np.nanmin(np.stack(diffs), 0) if all(np.isfinite(d).any() for d in diffs) else np.full(B, np.nan)
                lo, hi = ci(Dm)
                sd = np.nanstd(Dm)
                n_min = min(n for n, _ in ns)
                l_min = min(nl for _, nl in ns)
                ok = n_min >= 30 and l_min >= 10
                z = float(np.nanmean(Dm) / sd) if sd > 0 else np.nan
                rows.append(dict(base, row="mstar", label=o, value=float(np.nanmin(pts)) if np.isfinite(pts).any() else np.nan, lo=lo, hi=hi,
                                 z=z, n=n_min, logs=l_min, testable=ok, flag=bool(ok and lo > 0), strong=bool(ok and lo > 0 and z > Z_STRONG)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- classification
def classify(F, P, S, G, boot, perc):
    """Registered rules (plan section 4) per flagged cell -> class, lever, evidence."""
    logs = S.set_index("scene").log
    out = []
    for _, f in F.iterrows():
        m, g, o = f.measure, f.group, f.label
        cls_m = MEAS[m][0]
        ev = {}
        X = P[P.label == o].set_index("scene")
        sc = sorted(G[g] & set(X.index))
        if cls_m == "longitudinal":
            klass = "longitudinal"
        elif cls_m == "interaction":
            share = float(X.gap_inlane.reindex(sc).mean())
            ev["inlane_share"] = share
            klass = "longitudinal" if share > 0.5 else "geometry"
        else:
            # serving: plan_exec higher in the cell than on all scenes (CI excludes 0) while |plan_off10| is not
            for q in ("plan_exec", "plan_off10"):
                pc, bc, _, _ = boot.draws(X[q].reindex(sc), logs.reindex(sc))
                pa, ba, _, _ = boot.draws(X[q], logs.reindex(X.index))
                ev[q] = dict(cell=pc, all=pa, diff=pc - pa, ci=ci(bc - ba))
            serving = ev["plan_exec"]["ci"][0] > 0 and not ev["plan_off10"]["ci"][0] > 0
            pz = perc[perc.scene.isin(sc)]
            ev["percep"] = dict(n_dec=int(len(pz)), scenes=int(pz.scene.nunique()), err20_med=float(pz.err20.median()) if len(pz) else np.nan,
                                both=float(pz.both.mean()) if len(pz) else np.nan)
            eyes_bad = len(pz) >= 30 and (ev["percep"]["err20_med"] >= 0.30 or ev["percep"]["both"] < 0.80)
            klass = "serving" if serving else "perception" if (g in NONTURN and eyes_bad) else "geometry"
        if klass == "longitudinal":
            lever = "new label course L (lead / must-stop point / entry-speed hinges, research/self-consequence)"
        elif klass == "perception":
            lever = "perception-side fix (calibration / pitch d263; a road head on frozen tokens is not enough, d275)"
        elif klass == "serving":
            lever = "serving fix (outside the three levers)"
        elif m in ("margin", "nolane", "onc", "gap_min", "near_miss"):
            lever = "hinge-label fine-tuning (S road / agent hinges on on-policy rows; the directional raster failed, d274)"
        else:
            lever = "new label course (short-range anchor labels E3 d266 / turn-geometry labels)"
        out.append(dict(label=o, group=g, measure=m, klass=klass, lever=lever, evidence=json.dumps(ev, default=float)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures
def heatmap(C, P, path, labels=("YS2G", "YR10m10-F-s0")):
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    cols = ["all"] + GEOM + SPEED + list(TAGF) + ["plain"]
    meas = list(MEAS)
    sd = {m: float(np.nanstd(P[P.label.isin(REFS + sum(OURS.values(), []))][f"{m}|d"])) for m in meas}
    fig, axs = plt.subplots(1, len(labels), figsize=(ps.DOUBLE_COLUMN_IN, 3.9), sharey=True)
    for ax, o in zip(np.atleast_1d(axs), labels):
        M = C[(C.row == "mstar") & (C.label == o)].set_index(["measure", "group"])
        Z = np.array([[M.value.get((m, g), np.nan) / sd[m] if sd[m] > 0 else np.nan for g in cols] for m in meas])
        T = np.array([[bool(M.testable.get((m, g), False)) for g in cols] for m in meas])
        Zp = np.where(T, Z, np.nan)
        im = ax.imshow(np.clip(Zp, -0.6, 0.6), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        for i, m in enumerate(meas):
            for j, g in enumerate(cols):
                if M.flag.get((m, g), False):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.2 if M.strong.get((m, g), False) else 0.7, ec="k"))
                elif not T[i, j]:
                    ax.text(j, i, "·", ha="center", va="center", fontsize=6, color="#999999")
        ax.set_xticks(range(len(cols)), [c.replace("_", " ") for c in cols], rotation=60, ha="right")
        ax.set_yticks(range(len(meas)), [m.replace("_", " ") for m in meas])
        ax.grid(False)
        ax.set_title({"YS2G": "YS2G-L18 (seed mean)", "YR10m10-F-s0": "YR10m10-F-s0 (sub. #2)"}.get(o, o), fontsize=8)
        for x in (0.5, 4.5, 8.5):
            ax.axvline(x, color="w", lw=1.5)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.93, wspace=0.06)
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.25, 0.012, 0.68]))
    cb.set_label("min over references of (ours - ref) / SD", fontsize=7.5)
    return ps.save(fig, path)


def tables(a):
    out, figs = Path(a.out), Path(a.figs)
    T = pd.read_csv(out / "tags.csv")
    perc = pd.read_csv(out / "percep_held.csv.gz")
    res, md = {}, ["# CAP-AUDIT: capability scorecard on every stored decision", "",
                   "Pre-registration: [../plans/2026-10-12-cap-audit-prereg.md](../plans/2026-10-12-cap-audit-prereg.md). Code: `scripts/ca.py` (box), "
                   "`scripts/ca_report.py` (Mac). Every measure is oriented larger = worse; deficit = driver - log (margin, gap: log - driver); "
                   "m* = min over references (WA-JEPA, LTF, DD, G0014) of (ours - reference), paired per scene; flag = 2.5 % bootstrap "
                   "quantile of m* > 0 (log-clustered, 10 000 draws); strong = bootstrap z > 3.75; testable = >= 30 scenes and >= 10 logs.", ""]
    for st, d in (("pub", a.pub), ("held", a.held)):
        D, L, Ro, S = load(d)
        P = seed_means(per_scene(D, L, Ro))
        S, G = groups(S, T)
        boot = Boot(S.log)
        C = cell_table(P, S, G, boot)
        C.to_csv(out / f"cells_{st}.csv", index=False)
        P.to_csv(R / f"tmp/cap_audit/per_scene_{st}.csv.gz", index=False)          # 2-4 MB: kept out of git
        res[st] = dict(P=P, S=S, G=G, boot=boot, C=C)
        md += [f"## {st}: {S.scene.nunique()} scenes, {S.log.nunique()} logs, groups: " + ", ".join(f"{g} {len(v)}" for g, v in G.items()), ""]
    # primary = pub; sensitivity (survived) and held replication of the flagged cells
    p = res["pub"]
    C = p["C"]
    F = C[(C.row == "mstar") & C.flag].copy()
    Cs = cell_table(p["P"], p["S"], {g: p["G"][g] for g in F.group.unique()}, p["boot"], survived=True) if len(F) else pd.DataFrame()
    Ch = res["held"]["C"]
    if len(F):
        ms = Cs[Cs.row == "mstar"].set_index(["label", "group", "measure"])
        mh = Ch[Ch.row == "mstar"].set_index(["label", "group", "measure"])
        key = list(zip(F.label, F.group, F.measure))
        F["surv_value"], F["surv_lo"], F["surv_flag"] = [ms.value.get(k) for k in key], [ms.lo.get(k) for k in key], [ms.flag.get(k) for k in key]
        F["held_value"], F["held_same_sign"] = [mh.value.get(k) for k in key], [bool(mh.value.get(k, np.nan) > 0) for k in key]
        K = classify(F, p["P"], p["S"], p["G"], p["boot"], perc)
        F = F.merge(K, on=["label", "group", "measure"])
        sd = {m: float(np.nanstd(p["P"][p["P"].label.isin(REFS + sum(OURS.values(), []))][f"{m}|d"])) for m in MEAS}
        F["std_lo"] = [lo / sd[m] if sd[m] > 0 else np.nan for lo, m in zip(F.lo, F.measure)]
    F.to_csv(out / "flags.csv", index=False)
    # tables
    dv = C[C.row == "driver"].set_index(["group", "measure", "label"])
    ex = C[C.row == "excess"].set_index(["group", "measure", "label", "ref"])
    mst = C[C.row == "mstar"].set_index(["group", "measure", "label"])
    md += ["## Flagged cells (pub, 10 decisions)", "",
           "| our row | group | measure | m* [95 % CI] | z | n / logs | survived-only m* (flag) | held 18 m* | class | lever |",
           "|:--|:--|:--|--:|--:|--:|--:|--:|:--|:--|"]
    for _, f in F.sort_values(["label", "std_lo"], ascending=[True, False]).iterrows():
        md.append(f"| {f.label} | {f.group} | {f.measure} | {f.value:+.3f} [{f.lo:+.3f}, {f.hi:+.3f}]{' **strong**' if f.strong else ''} | {f.z:.1f} | "
                  f"{f.n} / {f.logs} | {f.surv_value:+.3f} ({'yes' if f.surv_flag else 'no'}) | {f.held_value:+.3f} | {f.klass} | {f.lever} |")
    md += ["", "## All cells, our rows (pub): driver value (deficit vs log), m* [CI]; log value in brackets after the measure", ""]
    for m in MEAS:
        md += [f"### {m} ({MEAS[m][0]}, {MEAS[m][1]})", "",
               "| group | n | log | " + " | ".join(list(OURS) + REFS + DESC) + " | m* YR | m* YS2G | m* YS-L18 |",
               "|:--|--:|--:|" + "--:|" * (len(OURS) + len(REFS) + len(DESC) + 3)]
        for g in p["G"]:
            r = [f"{dv.value.get((g, m, lab), np.nan):.3f}" for lab in list(OURS) + REFS + DESC]
            s = []
            for o in OURS:
                q = mst.loc[(g, m, o)]
                t = f"{q.value:+.3f} [{q.lo:+.3f}, {q.hi:+.3f}]" if q.testable else f"({q.value:+.3f})"
                s.append(f"**{t}**" if q.flag else t)
            lv = dv.log_value.get((g, m, "YR10m10-F-s0"), np.nan)
            md.append(f"| {g} | {dv.n.get((g, m, 'YR10m10-F-s0'), 0)} | {lv:.3f} | " + " | ".join(r) + " | " + " | ".join(s) + " |")
        md.append("")
    (out / "cap_table.md").write_text("\n".join(md) + "\n")
    _ = ex
    heatmap(C, p["P"], figs / "ca_heatmap")
    # BEV picks: the 5 worst flagged cells of the YS2G row by standardized lower bound, then YR, then unflagged by the same order
    picks = []
    sd = {m: float(np.nanstd(p["P"][p["P"].label.isin(REFS + sum(OURS.values(), []))][f"{m}|d"])) for m in MEAS}
    Mx = C[(C.row == "mstar") & C.testable].copy()
    Mx["std_lo"] = [lo / sd[m] if sd[m] > 0 else np.nan for lo, m in zip(Mx.lo, Mx.measure)]
    Mx["flagged"] = Mx.flag.astype(bool)
    order = pd.concat([Mx[(Mx.label == "YS2G") & Mx.flagged].sort_values("std_lo", ascending=False),
                       Mx[(Mx.label == "YR10m10-F-s0") & Mx.flagged].sort_values("std_lo", ascending=False),
                       Mx[(Mx.label == "YS2G") & ~Mx.flagged].sort_values("std_lo", ascending=False)])
    E = C[(C.row == "excess") & (C.lo > 0) & (C.n >= 30) & (C.logs >= 10)]
    E.to_csv(out / "pairwise_worse.csv", index=False)       # exploratory: cells where one of our rows is worse than a single reference
    seen = set()
    Dpub = load(a.pub)[0]
    for _, f in order.iterrows():
        if (f.group, f.measure) in seen:
            continue
        seen.add((f.group, f.measure))
        o = f.label
        X = p["P"].set_index(["label", "scene"])
        sc = sorted(p["G"][f.group])
        mine = X.loc[o][f"{f.measure}|d"].reindex(sc)
        refm = pd.concat([X.loc[r][f"{f.measure}|d"].reindex(sc) for r in REFS], axis=1).max(axis=1)
        if not (mine - refm).max() > 0:          # a degenerate cell (all zeros): no scene shows it, take the next one
            continue
        s = (mine - refm).idxmax()
        lab = OURS[o][0] if len(OURS[o]) == 1 else OURS[o][1]
        Dd = Dpub[(Dpub.label == lab) & (Dpub.scene == s) & Dpub.kept]
        col = {"lat_log": "lat_eval", "gap_min": "gap", "margin": "margin", "head_path": "head_path", "lane_ctr": "lane_off"}.get(f.measure)
        if f.measure in ("onc", "nolane") and len(Dd) and (Dd.cls == {"onc": 2, "nolane": 0}[f.measure]).any():
            k = int(Dd.k[Dd.cls == {"onc": 2, "nolane": 0}[f.measure]].iloc[0])
        elif col is not None and len(Dd):
            v = pd.to_numeric(Dd[col], errors="coerce").abs()
            k = int(Dd.k.iloc[int(v.argmin() if f.measure in ("gap_min", "margin") else v.argmax())])
        else:
            k = int(Dd.k.max()) if len(Dd) else 5
        picks.append(dict(cell=f"{f.group}:{f.measure}", label=lab, scene=s, k=k, row=o, flagged=bool(f.flagged), std_lo=float(f.std_lo),
                          excess=float((mine - refm).max())))
        if len(picks) == 5:
            break
    (out / "bev_picks.json").write_text(json.dumps(picks, indent=1))
    summ = dict(flags=int(len(F)), strong=int(F.strong.sum()) if len(F) else 0,
                by_row={o: int((F.label == o).sum()) for o in OURS} if len(F) else {},
                testable={o: int(C[(C.row == "mstar") & (C.label == o)].testable.sum()) for o in OURS})
    (out / "ca.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def bevfig(a):
    import pickle
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    B_ = pickle.load(open(a.bev, "rb"))
    _ = None
    col = {"YS2G-L18-s1": ps.PALETTE["green"], "YR10m10-F-s0": ps.PALETTE["blue"], "WA-JEPA": ps.PALETTE["purple"], "LTF": ps.PALETTE["orange"],
           "DD": ps.PALETTE["vermillion"], "G0014": ps.PALETTE["sky_blue"]}
    n = len(B_)
    fig, axs = plt.subplots(1, n, figsize=(ps.DOUBLE_COLUMN_IN, 2.6))
    for ax, (key, b) in zip(np.atleast_1d(axs), B_.items()):
        for ln in b["lanes"]:
            c = "#D55E00" if abs(ln["dpsi"]) > 120 else "#BBBBBB"
            for e in ln["ext"]:
                ax.plot(-e[:, 1], e[:, 0], color=c, lw=0.4, alpha=0.8)
        o = b["objs"]
        for x, y, d, Lo, Wo in zip(o["x"], o["y"], o["dpsi"], o["L"], o["W"]):
            cc, ss = np.cos(d), np.sin(d)
            u = np.array([[Lo / 2, Wo / 2], [Lo / 2, -Wo / 2], [-Lo / 2, -Wo / 2], [-Lo / 2, Wo / 2], [Lo / 2, Wo / 2]])
            P_ = np.c_[x + cc * u[:, 0] - ss * u[:, 1], y + ss * u[:, 0] + cc * u[:, 1]]
            ax.fill(-P_[:, 1], P_[:, 0], color="#444444", alpha=0.35, lw=0)
        ax.plot(-b["log"][:, 1], b["log"][:, 0], color="k", lw=1.0, ls="--", label="log")
        for lab, pth in b["paths"].items():
            if lab in col:
                ax.plot(-pth[:, 1], pth[:, 0], color=col[lab], lw=0.9 if lab != b["pick"]["label"] else 1.4, label=lab)
                ax.plot(-pth[-1, 1], pth[-1, 0], "o", color=col[lab], ms=3)          # position at the last decision
        ax.plot(-b["log_end"][1], b["log_end"][0], "x", color="k", ms=4) if b.get("log_end") is not None else None
        if b["plan"] is not None:
            ax.plot(-b["plan"][:, 1], b["plan"][:, 0], color=ps.PALETTE["green"], lw=0.8, ls=":", marker=".", ms=2, label="served plan")
        r, f, h = b["ego"]
        ax.fill([-h, h, h, -h], [r, r, f, f], color=ps.PALETTE["green"], alpha=0.5, lw=0)
        ax.set_xlim(-22, 22)
        ax.set_ylim(-15, 40)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(b["pick"]["cell"].replace("_", " ") + f"\nk={b['pick']['k']}", fontsize=7)
    h_, l_ = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=8, fontsize=6.5)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.88, wspace=0.05)
    print(ps.save(fig, a.out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--pub", required=True)
    p.add_argument("--held", required=True)
    p.add_argument("--out", default=str(R / "experiments/cap_audit/results"))
    p.add_argument("--figs", default=str(R / "experiments/cap_audit/figs"))
    p = sub.add_parser("bevfig")
    p.add_argument("--bev", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(tables=tables, bevfig=bevfig)[a.cmd](a)
