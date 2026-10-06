"""op_probe joint diagnosis of op_parity P2 and WA-JEPA: statistics, paper figures (vector PDF + PNG) and the results page.

  .venv/bin/python experiments/op_probe/scripts/opj_figs.py [--cases DIR]

Inputs: results/joint/data/ (gitignored; copy of the box's $DATA_DIR/runs/op_probe/joint/ plus runs/op_parity/hugsim/routes.json as
hugsim_routes.json; opj_build.py table on the box: navtest / navhard per-token tables, decoder scores; devkit replays of both
models' plans on the op_probe eval tokens, opb_score.py; HUGSIM routes), the op_parity HUGSIM CSVs in the repo, and for the case figure
cases.json + frames_small.npz (opj_build.py export / frames; not committed: --cases points at a local copy of joint/, the figure is committed).
Out: results/joint/figs/*.pdf|png, results/joint/stats.json, results/joint/index.html.

Conventions: P2 = op_parity P2-F (Cinque 382M, frozen vision, ego / pose / command inputs, protocol W frames); rates are the mean of its two
seeds unless a set is defined on seed 0 (the op_probe sets F / R / FF of dac-localize.md); WA = the released WA-JEPA checkpoint (one run).
CIs: 95% percentile cluster bootstrap, B 2000, clusters = navtest logs, navhard scene groups, HUGSIM scenarios.
"""
import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
J = REPO / "experiments/op_probe/results/joint"
DATA, FIGS = J / "data", J / "figs"
B, SEED = 2000, 0
TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
# Okabe-Ito, checked with the dataviz validator (P2 / WA / both: CVD dE >= 9.6, normal-vision dE >= 16.4); tints for secondary marks
C = {"P2": "#D55E00", "WA": "#0072B2", "both": "#CC79A7", "P2l": "#E69F00", "WAl": "#56B4E9", "ink": "#222222", "mute": "#6b6b6b",
     "grid": "#e6e6e6", "GT": "#111111", "band": "#f3f3f3"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.2, "axes.labelsize": 6.8, "xtick.labelsize": 6.3,
                     "ytick.labelsize": 6.3, "legend.fontsize": 6.2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.color": "#333333", "ytick.color": "#333333",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlepad": 4, "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2})


# ---------------------------------------------------------------- statistics
def cboot(V, g, B=B, seed=SEED):
    """Cluster bootstrap of column means. V (n, k) or (n,) (NaN = missing), g (n,) cluster labels -> (k, 3) [mean, lo, hi]."""
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V[:, None]
    ok = np.isfinite(V)
    codes, uniq = pd.factorize(pd.Series(np.asarray(g)))
    nu = len(uniq)
    S = np.stack([np.bincount(codes, np.where(ok[:, j], V[:, j], 0), nu) for j in range(V.shape[1])], 1)
    N = np.stack([np.bincount(codes, ok[:, j].astype(float), nu) for j in range(V.shape[1])], 1)
    idx = np.random.default_rng(seed).integers(nu, size=(B, nu))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S[idx].sum(1) / N[idx].sum(1)
        full = S.sum(0) / N.sum(0)
    return np.stack([full, np.nanquantile(r, 0.025, 0), np.nanquantile(r, 0.975, 0)], 1)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)
    print("wrote", name, flush=True)


def tag(ax, s, x=-0.08, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- data
def load():
    d = pd.read_parquet(DATA / "navtest_tokens.parquet")
    for m in ("P2s0", "P2s1", "WA"):
        for t in TERMS:
            d[f"{m}_f{t}"] = (d[f"{m}_{t}"] < 1 - 1e-9).astype(float).where(d[f"{m}_{t}"].notna())
    for t in TERMS:
        d[f"P2_f{t}"] = (d[f"P2s0_f{t}"] + d[f"P2s1_f{t}"]) / 2
    for t in TERMS + ["score"]:
        d[f"P2_{t}"] = (d[f"P2s0_{t}"] + d[f"P2s1_{t}"]) / 2
    d["vru"] = d.n_ped + d.n_bike
    d["turn"] = np.abs(d.dyaw)
    return d


def strata(d):
    """(group, stratum label, mask) in display order."""
    out = []
    for k, lab in (("straight", "straight"), ("curve", "curve 8-20°"), ("left turn", "left turn > 20°"), ("right turn", "right turn > 20°"),
                   ("lane change", "lane change"), ("launch", "launch from stop"), ("stop", "coming to a stop")):
        out.append(("manoeuvre (logged 4 s)", lab, d.maneuver == k))
    for lo, hi, lab in ((0, 2, "< 2 m/s"), (2, 5, "2-5 m/s"), (5, 10, "5-10 m/s"), (10, 99, "≥ 10 m/s")):
        out.append(("speed at t0", lab, (d.v0 >= lo) & (d.v0 < hi)))
    out.append(("junction", "path through intersection", d.junction_path))
    out.append(("junction", "no intersection", ~d.junction_path))
    for k in ("left", "straight", "right"):
        out.append(("route command", k, d.cmd == k))
    for lo, hi, lab in ((0, 4, "0-3 vehicles"), (4, 10, "4-9 vehicles"), (10, 999, "≥ 10 vehicles")):
        out.append(("traffic within 30 m", lab, (d.n_veh >= lo) & (d.n_veh < hi)))
    out.append(("traffic within 30 m", "pedestrian / cyclist", d.vru > 0))
    out.append(("traffic light", "red light listed", d.red_light))
    out.append(("traffic light", "no red light", ~d.red_light))
    for c in ("Boston", "Pittsburgh", "Las Vegas", "Singapore"):
        out.append(("city", c + (" (left-hand)" if c == "Singapore" else ""), d.city == c))
    for lo, hi, lab in ((0, 30, "sun < 30°"), (30, 50, "sun 30-50°"), (50, 91, "sun ≥ 50°")):
        out.append(("lighting (sun elevation)", lab, (d.sun_el >= lo) & (d.sun_el < hi)))
    return out


SUBS = ("DAC", "NC", "TTC", "LK", "EC", "DDC")


def navtest_metric_stats(d):
    g = d.log.values
    res = {}
    for t in TERMS:
        if t == "EP":
            V = np.stack([d.P2_EP, d.WA_EP, d.P2_EP - d.WA_EP], 1)
        else:
            wa = d[f"WA_f{t}"]
            V = np.stack([d[f"P2_f{t}"], wa, d[f"P2_f{t}"] - wa,
                          ((d[f"P2s0_f{t}"] == 1) & (wa == 0)).astype(float).where(wa.notna()),
                          ((d[f"P2s0_f{t}"] == 0) & (wa == 1)).astype(float).where(wa.notna()),
                          ((d[f"P2s0_f{t}"] == 1) & (wa == 1)).astype(float).where(wa.notna())], 1)
        res[t] = cboot(V, g).tolist()
    res["score"] = cboot(np.stack([d.P2_score, d.WA_score, d.P2_score - d.WA_score], 1), g).tolist()
    res["n"], res["logs"] = len(d), int(d.log.nunique())
    return res


def stratum_stats(d):
    rows = []
    keys = (["score_d", "EP_d"] + [f"{t}_d" for t in SUBS] + [f"P2_{t}" for t in SUBS] + [f"WA_{t}" for t in SUBS])
    for grp, lab, m in strata(d):
        s = d[m]
        V = np.stack([s.P2_score - s.WA_score, s.P2_EP - s.WA_EP] + [s[f"P2_f{t}"] - s[f"WA_f{t}"] for t in SUBS]
                     + [s[f"P2_f{t}"] for t in SUBS] + [s[f"WA_f{t}"] for t in SUBS], 1)
        r = cboot(V, s.log.values)
        rows.append(dict(group=grp, label=lab, n=int(m.sum()), **{k: r[i].tolist() for i, k in enumerate(keys)}))
    return rows


def navhard_stats(nh):
    """Per stage: sub-metric failure rates (unweighted over tokens; the official stage-2 weights are arm-specific), per-token score
    difference by manoeuvre (stage 2: the manoeuvre of the stage-1 scene it was rendered from, via its scene group) and by speed."""
    s1 = nh[nh.stage == 1]
    gman = s1.groupby("group").maneuver.agg(lambda x: x.mode().iat[0])
    nh = nh.copy()
    nh["man2"] = np.where(nh.stage == 1, nh.maneuver, nh.group.map(gman))
    out = {}
    for s in (1, 2):
        h = nh[nh.stage == s]
        g = h.group.values
        r = {}
        for t in TERMS:
            if t == "EP":
                p2, wa = (h.P2s0_EP + h.P2s1_EP) / 2, h.WA_EP
            else:
                p2 = ((h[f"P2s0_{t}"] < 1 - 1e-9).astype(float) + (h[f"P2s1_{t}"] < 1 - 1e-9).astype(float)) / 2
                wa = (h[f"WA_{t}"] < 1 - 1e-9).astype(float)
            r[t] = cboot(np.stack([p2, wa, p2 - wa], 1), g).tolist()
        p2s = (h.P2s0_score + h.P2s1_score) / 2
        r["score"] = cboot(np.stack([p2s, h.WA_score, p2s - h.WA_score], 1), g).tolist()
        r["n"], r["groups"] = len(h), int(h.group.nunique())
        r["by_man"], r["by_speed"] = {}, {}
        for k in ("straight", "curve", "left turn", "right turn", "lane change", "launch"):
            hm = h[h.man2 == k]
            if len(hm) >= 20:
                r["by_man"][k] = dict(n=len(hm), d=cboot((hm.P2s0_score + hm.P2s1_score) / 2 - hm.WA_score, hm.group.values)[0].tolist())
        for lo, hi, lab in ((0, 2, "< 2 m/s"), (2, 5, "2-5 m/s"), (5, 10, "5-10 m/s"), (10, 99, "≥ 10 m/s")):
            hm = h[(h.v0 >= lo) & (h.v0 < hi)]
            if len(hm) >= 20:
                r["by_speed"][lab] = dict(n=len(hm), d=cboot((hm.P2s0_score + hm.P2s1_score) / 2 - hm.WA_score, hm.group.values)[0].tolist())
        out[f"stage{s}"] = r
    return out


DSTAGES = ["E", "P2-V", "P2-M", "P2-T", "P2-H", "WA-Cf", "WA-Ca", "WA-H"]
PSTAGES = ["P2-V", "P2-M", "P2-T", "P2-H", "WA-Cf", "WA-Ca", "WA-T", "WA-H"]
SHORT = {"E": "ego", "P2-V": "V", "P2-M": "M", "P2-T": "T", "P2-H": "H", "WA-Cf": "Cf", "WA-Ca": "Ca", "WA-T": "T", "WA-H": "H"}


def localisation_stats(d, dec):
    sets = {"F": d.set_F, "R": d.set_R, "FF": d.set_FF, "PPturn": d.set_PP & d.pp1500 & (d.turn > 20)}
    lg = dict(zip(d.token, d.log))
    out = {"decoder": {}, "probe": {}, "margin": {}}
    for obj in ("imit", "hinge", "hinge10"):
        q = dec[dec.obj == obj]
        e_all = q[q.stage == "E"].set_index("token")
        for st in DSTAGES:
            qs = q[q.stage == st].set_index("token")
            for sn, msk in sets.items():
                toks = d.token[msk]
                toks = toks[toks.isin(qs.index)]
                ps = (qs.loc[toks, "DAC"] >= 1).astype(float).values
                e = (e_all.loc[toks, "DAC"] >= 1).astype(float).values
                out["decoder"][f"{obj}|{st}|{sn}"] = cboot(np.stack([ps, ps - e], 1), toks.map(lg).values).tolist() + [len(toks)]
    sets_p = {"F": d.set_F, "R": d.set_R, "FF": d.set_FF, "PPturn": d.set_PP & (d.turn > 20)}
    for st in PSTAGES:
        own = "P2" if st.startswith("P2") else "WA"
        for sn, msk in sets_p.items():
            s = d[msk]
            codes, uniq = pd.factorize(s.log)
            S = np.stack([np.bincount(codes, np.nan_to_num(s[f"corr_err|{k}"].values), len(uniq)) for k in (st, "E")], 1)
            idx = np.random.default_rng(SEED).integers(len(uniq), size=(B, len(uniq)))
            r = 1 - S[idx, 0].sum(1) / S[idx, 1].sum(1)
            out["probe"][f"{st}|{sn}"] = [float(1 - S[:, 0].sum() / S[:, 1].sum()), *np.quantile(r, [0.025, 0.975]).tolist()]
            true = s.P2s0_margin_true_probe if own == "P2" else s.WA_margin
            out["margin"][f"{st}|{sn}"] = cboot(s[f"pm_{own}|{st}"] - true, s.log.values)[0].tolist()
    for sn, msk in sets_p.items():
        s = d[msk]
        out["margin"][f"E-P2|{sn}"] = cboot(s["pm_P2|E"] - s.P2s0_margin_true_probe, s.log.values)[0].tolist()
        out["margin"][f"E-WA|{sn}"] = cboot(s["pm_WA|E"] - s.WA_margin, s.log.values)[0].tolist()
    out["n"] = {k: int(v.sum()) for k, v in sets_p.items()}
    out["n_dec"] = {k: int(v.sum()) for k, v in sets.items()}
    return out


TURN_BINS = [0, 5, 20, 45, 400]


def geometry_stats(d, sc):
    out = {}
    g = d.log.values
    for m, mm in (("P2s0", "P2"), ("WA", "WA")):
        out[f"{mm}_raw_out"] = cboot((d[f"{m}_margin"] < 0).astype(float).where(d[f"{m}_margin"].notna()), g)[0].tolist()
        f = d[f"{m}_fDAC"] == 1
        out[f"{mm}_raw_out_given_fail"] = cboot((d[f"{m}_margin"][f] < 0).astype(float), g[f])[0].tolist()
    for mm in ("P2", "WA"):
        s = sc[(sc.model == mm) & (sc.DAC < 1)]
        gg = s.token.map(dict(zip(d.token, d.log))).values
        out[f"{mm}_depth_n"] = len(s)
        out[f"{mm}_depth_med"] = float(s.out_depth.median())
        out[f"{mm}_graze03"] = cboot((s.out_depth < 0.3).astype(float), gg)[0].tolist()
        out[f"{mm}_lqr_only"] = cboot((~s.raw_out.astype(bool)).astype(float), gg)[0].tolist()
    out["turn_bins"] = []
    for lo, hi in zip(TURN_BINS[:-1], TURN_BINS[1:]):
        s = d[(d.turn >= lo) & (d.turn < hi)]
        out["turn_bins"].append(dict(lo=lo, hi=hi, n=len(s), r=cboot(np.stack([s.P2_fDAC, s.WA_fDAC, s.P2_fDAC - s.WA_fDAC], 1), s.log.values).tolist()))
    tr = d[d.turn > 20].copy()
    sg = np.sign(tr.dyaw)
    for m, mm in (("P2s0", "P2"), ("WA", "WA")):
        tr[f"{mm}_in"] = tr[f"{m}_e_lat"] * sg
        tr[f"{mm}_yin"] = tr[f"{m}_e_yaw"] * sg
        tr[f"{mm}_inside_corner"] = (tr[f"{m}_side"] * sg > 0).astype(float)
    out["turn_inside"] = {}
    for sn, msk in (("F", tr.set_F), ("R", tr.set_R), ("FF", tr.set_FF), ("PP", tr.set_PP)):
        s = tr[msk]
        out["turn_inside"][sn] = {k: cboot(s[k], s.log.values)[0].tolist() for k in ("P2_in", "WA_in", "P2_yin", "WA_yin", "P2_inside_corner", "WA_inside_corner")}
        out["turn_inside"][sn]["n"] = len(s)
    return out, tr


CLS = ["complete", "fg_coll", "bg_coll", "off_route", "stuck", "spin"]
CLAB = {"complete": "completed", "fg_coll": "fg collision", "bg_coll": "bg collision", "off_route": "off route", "stuck": "stuck", "spin": "spin"}


def hugsim_load(preset="exam"):
    sys.path.insert(0, str(REPO / "experiments/op_parity/scripts"))
    import pp_gap_hugsim as GH
    W, A = GH.load()
    df = GH.per_scenario(W, A[preset])
    W = W.reindex(df.index)
    for k in ("nc", "dac", "ttc", "rc"):
        df[f"wa_{k}"] = W[k].values
        df[f"p2_{k}"] = ((A[preset]["P2-F-s0"][k] + A[preset]["P2-F-s1"][k]) / 2).reindex(df.index).values
    routes = json.load(open(DATA / "hugsim_routes.json"))
    turn = {k: float(np.degrees(np.ptp(np.unwrap(np.asarray(v["yaw"], float))))) for k, v in routes.items()}
    df["route_turn"] = df.scene.map(turn)
    return df


def hugsim_stats(df):
    g = df.index.values
    d = df.p2_hd - df.wa_hd
    out = {"n": len(df), "hd": cboot(np.stack([df.p2_hd, df.wa_hd], 1), g).tolist(), "hd_d": cboot(d, g)[0].tolist(),
           "route_turn_missing": int(df.route_turn.isna().sum())}
    out["counts"] = {"WA": {c: int((df.wa_cls == c).sum()) for c in CLS}, "P2": {c: ((df.s0_cls == c).sum() + (df.s1_cls == c).sum()) / 2 for c in CLS}}
    M = np.zeros((len(CLS), len(CLS)))
    for s in ("s0", "s1"):
        for a_, b_ in zip(df.wa_cls, df[f"{s}_cls"]):
            M[CLS.index(a_), CLS.index(b_)] += 0.5
    out["matrix"] = M.tolist()
    out["sub_d"] = {k: cboot(df[f"p2_{k}"] - df[f"wa_{k}"], g)[0].tolist() for k in ("nc", "dac", "ttc", "rc")}
    sub = lambda m: dict(n=int(m.sum()), d=cboot(d[m], g[m])[0].tolist(), p2=float(df.p2_hd[m].mean()), wa=float(df.wa_hd[m].mean()))  # noqa: E731
    out["by_diff"] = {k: sub(df.difficulty == k) for k in ("easy", "medium", "hard", "extreme")}
    out["by_dataset"] = {k: sub(df.dataset == k) for k in ("nuscenes", "pandaset", "waymo", "kitti360")}
    out["by_route"] = {"straight route (< 30°)": sub(df.route_turn < 30), "turning route (≥ 30°)": sub(df.route_turn >= 30)}
    out["by_route_dac"] = {k: cboot((df.p2_dac - df.wa_dac)[m], g[m])[0].tolist() for k, m in
                           (("straight", df.route_turn < 30), ("turning", df.route_turn >= 30))}
    out["wins"] = dict(p2=int((d > 0.2).sum()), wa=int((d < -0.2).sum()), tie=int((d.abs() <= 0.2).sum()))
    return out


# ---------------------------------------------------------------- figure helpers
def forest(ax, rows, xlab="", colors=None, ms=3.0, dy=0.24, labels=True, band=True):
    """rows: (group, label, [(key, m, lo, hi), ...]). Group headers become bold rows; one CI per key, offset vertically. Returns the row y."""
    y, yt, yl, bold, ys = 0, [], [], [], []
    last = None
    for g, lab, vals in rows:
        if g and g != last:
            yt.append(y), yl.append(g), bold.append(True)
            y += 1
        last = g
        for j, (k, m, lo, hi) in enumerate(vals):
            off = (j - (len(vals) - 1) / 2) * dy
            col = (colors or C).get(k, C["ink"])
            ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.1, solid_capstyle="round", zorder=3)
            ax.plot([m], [y + off], "o", color=col, ms=ms, mec="white", mew=0.5, zorder=4)
        yt.append(y), yl.append(lab), bold.append(False), ys.append(y)
        y += 1
    if band:
        for i, yy in enumerate(ys):
            if i % 2 == 0:
                ax.axhspan(yy - 0.5, yy + 0.5, color=C["band"], zorder=0, lw=0)
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_yticks(yt)
    if labels:
        ax.set_yticklabels(yl)
        for t, b in zip(ax.get_yticklabels(), bold):
            if b:
                t.set_fontweight("bold")
                t.set_color(C["ink"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab)
    return ys


def zero_sides(ax, left="← WA-JEPA better", right="P2 better →", sym=True):
    ax.axvline(0, color="#8a8a8a", lw=0.7, zorder=1)
    if sym:
        lo, hi = ax.get_xlim()
        m = max(abs(lo), abs(hi))
        ax.set_xlim(-m, m)
    ax.text(0.0, 1.005, left, transform=ax.transAxes, color=C["WA"], fontsize=5.6, va="bottom", ha="left")
    ax.text(1.0, 1.005, right, transform=ax.transAxes, color=C["P2"], fontsize=5.6, va="bottom", ha="right")


def legend_models(ax, loc="upper center", anchor=None, ncol=2, p2="P2 (2-seed mean)", wa="WA-JEPA", **kw):
    h = [Line2D([], [], color=C["P2"], marker="o", lw=1.1, ms=3, label=p2), Line2D([], [], color=C["WA"], marker="o", lw=1.1, ms=3, label=wa)]
    ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=ncol, **kw)


def stage_axis(ax, stg):
    """Short stage codes under the ticks and model brackets below them."""
    ax.set_xticks(np.arange(len(stg)))
    ax.set_xticklabels([SHORT[s] for s in stg])
    n2 = sum(s.startswith("P2") for s in stg)
    ne = sum(s == "E" for s in stg)
    for a_, b_, lab, col in ((ne, ne + n2 - 1, "P2 stages", C["P2"]), (ne + n2, len(stg) - 1, "WA-JEPA stages", C["WA"])):
        ax.annotate("", xy=(a_ - 0.35, -0.13), xytext=(b_ + 0.35, -0.13), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.2))
        ax.text((a_ + b_) / 2, -0.16, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=col)
    ax.axvspan(ne + n2 - 0.5, len(stg) - 0.5, color=C["band"], zorder=0, lw=0)


def ci_err(ax, x, m, **kw):
    m = np.asarray(m)
    ax.errorbar(x, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], **kw)


# ---------------------------------------------------------------- Figure 1: overview
def fig_overview(st):
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.0], height_ratios=[1, 1.25], wspace=0.75, hspace=0.5)
    # (a) benchmark level
    ax = fig.add_subplot(gs[0, 0])
    nhs = st["navhard"]
    rows = [("", "navtest EPDMS", [("d", *np.array(st["navtest"]["score"][2]) * 100)]),
            ("", "navhard stage 1", [("d", *np.array(nhs["stage1"]["score"][2]) * 100)]),
            ("", "navhard stage 2", [("d", *np.array(nhs["stage2"]["score"][2]) * 100)]),
            ("", "HUGSIM-64 HD", [("d", *np.array(st["hugsim"]["hd_d"]) * 100)])]
    forest(ax, rows, "P2 − WA-JEPA (score × 100)", colors={"d": C["ink"]}, ms=3.4)
    zero_sides(ax)
    ax.set_title(pad=11, label="Benchmark level")
    tag(ax, "a", x=-0.75)
    # (b) navtest gates: P2 fails | WA fails, shared part next to the axis
    ax = fig.add_subplot(gs[0, 1])
    terms = ["DAC", "LK", "TTC", "NC", "DDC"]
    m = st["navtest"]
    y = np.arange(len(terms))
    p2o, wao, bo = (np.array([m[t][k][0] for t in terms]) * 100 for k in (3, 4, 5))
    ax.barh(y, -bo, color=C["both"], height=0.64, label="both fail")
    ax.barh(y, -p2o, left=-bo, color=C["P2"], height=0.64, label="only P2 fails")
    ax.barh(y, bo, color=C["both"], height=0.64)
    ax.barh(y, wao, left=bo, color=C["WA"], height=0.64, label="only WA-JEPA fails")
    for yy, a_, b_, c_ in zip(y, p2o, wao, bo):
        ax.text(-a_ - c_ - 0.12, yy, f"{a_:.1f}", ha="right", va="center", fontsize=5.8, color=C["P2"])
        ax.text(b_ + c_ + 0.12, yy, f"{b_:.1f}", ha="left", va="center", fontsize=5.8, color=C["WA"])
    ax.axvline(0, color="#777777", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    lim = max((p2o + bo).max(), (wao + bo).max()) + 1.2
    ax.set_xlim(-lim, lim)
    tk = np.arange(-np.floor(lim), np.floor(lim) + 1, 2 if lim > 6 else 1)
    ax.set_xticks(tk)
    ax.set_xticklabels([f"{abs(v):g}" for v in tk])
    ax.set_xlabel("navtest tokens failing (%)")
    ax.text(0.25, 1.0, "P2 fails", transform=ax.transAxes, color=C["P2"], fontsize=6.3, ha="center", va="bottom", fontweight="bold")
    ax.text(0.75, 1.0, "WA-JEPA fails", transform=ax.transAxes, color=C["WA"], fontsize=6.3, ha="center", va="bottom", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=5.8, handlelength=0.9, columnspacing=0.8)
    ax.tick_params(axis="y", length=0)
    tag(ax, "b", x=-0.12, y=1.05)
    # (c) stage mirror
    ax = fig.add_subplot(gs[0, 2])
    mirror(ax, st["loc"], ["P2-V", "P2-H", "WA-Cf", "WA-H"], sets=("F", "R"))
    ax.set_title("Encoder decides which fail")
    tag(ax, "c", x=-0.42)
    # (d) navtest by scene type
    ax = fig.add_subplot(gs[1, 0:2])
    keep = ["straight", "curve 8-20°", "left turn > 20°", "right turn > 20°", "lane change", "launch from stop", "coming to a stop", "< 2 m/s",
            "2-5 m/s", "≥ 10 m/s", "path through intersection", "no intersection", "0-3 vehicles", "≥ 10 vehicles", "Las Vegas", "Singapore (left-hand)"]
    sr = [r for r in st["strata"] if r["label"] in keep]
    rows = [(r["group"], f"{r['label']}  ({r['n']})", [("d", *(np.array(r["score_d"]) * 100))]) for r in sr]
    forest(ax, rows, "per-token EPDMS, P2 − WA-JEPA (× 100)", colors={"d": C["ink"]}, ms=3)
    zero_sides(ax)
    ax.set_title(pad=11, label="navtest by scene type")
    tag(ax, "d", x=-0.38)
    # (e) HUGSIM by route / dataset
    ax = fig.add_subplot(gs[1, 2])
    hs = st["hugsim"]
    rows = [("", f"all 64", [("d", *np.array(hs["hd_d"]) * 100)])]
    rows += [("route", k.split(" (")[0] + f" ({v['n']})", [("d", *np.array(v["d"]) * 100)]) for k, v in hs["by_route"].items()]
    rows += [("source dataset", f"{k} ({v['n']})", [("d", *np.array(v["d"]) * 100)]) for k, v in hs["by_dataset"].items()]
    rows += [("difficulty", f"{k} ({v['n']})", [("d", *np.array(v["d"]) * 100)]) for k, v in hs["by_diff"].items()]
    forest(ax, rows, "HD, P2 − WA-JEPA (× 100)", colors={"d": C["ink"]}, ms=3)
    zero_sides(ax)
    ax.set_title(pad=11, label="Closed loop (HUGSIM)")
    tag(ax, "e", x=-0.62)
    save(fig, "fig1_overview")


def mirror(ax, loc, stg, sets=("F", "R", "FF"), obj="hinge10"):
    """Decoder DAC pass-rate lift over the ego-only decoder on F (P2 fails, WA passes), R (WA fails, P2 passes), FF (both fail)."""
    x = np.arange(len(stg))
    col = {"F": C["P2"], "R": C["WA"], "FF": C["both"]}
    lab = {"F": "F: P2 fails, WA passes", "R": "R: WA fails, P2 passes", "FF": "FF: both fail"}
    w = 0.8 / len(sets)
    for i, sn in enumerate(sets):
        m = np.array([loc["decoder"][f"{obj}|{s}|{sn}"][1] for s in stg]) * 100
        off = (i - (len(sets) - 1) / 2) * w
        ax.bar(x + off, m[:, 0], w * 0.9, color=col[sn], label=f"{lab[sn]} ({loc['n_dec'][sn]})")
        ci_err(ax, x + off, m, fmt="none", ecolor=C["ink"], lw=0.7, capsize=0)
    ax.axhline(0, color="#777777", lw=0.6)
    stage_axis(ax, stg)
    ax.set_ylabel("DAC pass − ego-only decoder (pp)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -25), hi + 22)
    ax.legend(loc="upper left", fontsize=5.6, handlelength=0.9)


# ---------------------------------------------------------------- Figure 2: taxonomy by scene type
def fig_taxonomy(st):
    terms = ["DAC", "NC", "TTC", "LK", "EC"]
    fig, axs = plt.subplots(1, len(terms) + 1, figsize=(7.2, 6.6), sharey=True,
                            gridspec_kw=dict(wspace=0.14, width_ratios=[1.15] + [1] * len(terms)))
    rows = [(r["group"], f"{r['label']} ({r['n']})", [("d", *(np.array(r["score_d"]) * 100))]) for r in st["strata"]]
    forest(axs[0], rows, "P2 − WA (× 100)", colors={"d": C["ink"]}, ms=2.8)
    axs[0].axvline(0, color="#8a8a8a", lw=0.7)
    axs[0].set_xlim(right=max(1.5, axs[0].get_xlim()[1]))
    axs[0].set_title("score difference")
    for ax, t in zip(axs[1:], terms):
        rows = [(r["group"], r["label"], [("P2", *(np.array(r[f"P2_{t}"]) * 100)), ("WA", *(np.array(r[f"WA_{t}"]) * 100))]) for r in st["strata"]]
        forest(ax, rows, "% failing", ms=2.5, dy=0.3, labels=False)
        ax.set_title(f"{t}")
        ax.set_xlim(left=0)
        ax.grid(axis="x", color=C["grid"], lw=0.5, zorder=0)
    legend_models(axs[3], loc="lower center", anchor=(0.5, 1.03), ncol=2)
    save(fig, "fig2_taxonomy")


# ---------------------------------------------------------------- Figure 3: navhard stages
def fig_navhard(ns):
    terms = ["NC", "DAC", "DDC", "TTC", "LK", "HC"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.5, width_ratios=[1.1, 1.1, 1.1]))
    for ax, s in zip(axs[:2], (1, 2)):
        r = ns[f"stage{s}"]
        x = np.arange(len(terms))
        for k, off, key in ((0, -0.14, "P2"), (1, 0.14, "WA")):
            ci_err(ax, x + off, np.array([r[t][k] for t in terms]) * 100, fmt="o", ms=3.2, color=C[key], lw=1.1, capsize=0, mec="white", mew=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(terms)
        ax.set_ylabel("tokens failing (%)")
        ax.set_title(f"stage {s}: {'logged' if s == 1 else 'rendered'} ({r['n']})")
        ax.grid(axis="y", color=C["grid"], lw=0.5)
        ax.set_ylim(bottom=0)
        tag(ax, "ab"[s - 1], x=-0.13)
    legend_models(axs[0], loc="upper right", ncol=1, p2="P2 (2 seeds, G frames)")
    ax = axs[2]
    rows = []
    for s in (1, 2):
        rows += [(f"stage {s}", f"{k} ({v['n']})", [("d", *(np.array(v["d"]) * 100))]) for k, v in ns[f"stage{s}"]["by_man"].items()]
    forest(ax, rows, "score, P2 − WA-JEPA (× 100)", colors={"d": C["ink"]}, ms=2.8)
    zero_sides(ax)
    ax.set_title(pad=11, label="by manoeuvre (stage-1 scene)")
    tag(ax, "c", x=-0.5)
    save(fig, "fig3_navhard")


# ---------------------------------------------------------------- Figure 4: stage localisation, both directions
def fig_localisation(loc):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw=dict(wspace=0.5, width_ratios=[1.4, 1.0, 1.0]))
    mirror(axs[0], loc, ["P2-V", "P2-M", "P2-T", "P2-H", "WA-Cf", "WA-Ca", "WA-H"])
    axs[0].set_title("decoders (stage + ego, hinge λ=10)")
    tag(axs[0], "a", x=-0.13)
    stg = PSTAGES
    x = np.arange(len(stg))
    spec = (("F", "o", C["P2"], "F (P2 fails)"), ("R", "s", C["WA"], "R (WA fails)"), ("PPturn", "^", C["mute"], "both pass, turn > 20°"))
    ax = axs[1]
    for i, (sn, mk, col, lab) in enumerate(spec):
        ci_err(ax, x + (i - 1) * 0.22, [loc["probe"][f"{s}|{sn}"] for s in stg], fmt=mk, ms=3, color=col, lw=0.9, label=lab, mec="white", mew=0.4)
    stage_axis(ax, stg)
    ax.set_ylabel("corridor skill over ego-only probe")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="lower left", fontsize=5.6, handlelength=0.8)
    ax.set_title("probes: road edges")
    tag(ax, "b", x=-0.2)
    ax = axs[2]
    for i, (sn, mk, col, lab) in enumerate(spec):
        ci_err(ax, x + (i - 1) * 0.22, [loc["margin"][f"{s}|{sn}"] for s in stg], fmt=mk, ms=3, color=col, lw=0.9, label=lab, mec="white", mew=0.4)
    for sn, col in (("F", C["P2"]), ("R", C["WA"]), ("PPturn", C["mute"])):
        for x0, x1, key in ((-0.4, 3.4, "E-P2"), (3.6, 7.4, "E-WA")):
            ax.hlines(loc["margin"][f"{key}|{sn}"][0], x0, x1, color=col, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color="#777777", lw=0.6)
    stage_axis(ax, stg)
    ax.text(0.98, 0.98, "dashed: ego-only probe", transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=C["mute"])
    ax.set_ylabel("probe − true margin along own plan (m)")
    ax.set_title("margin bias")
    tag(ax, "c", x=-0.2)
    save(fig, "fig4_localisation")


# ---------------------------------------------------------------- Figure 5: geometry
def fig_geometry(d, sc, geo, tr):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1, 1.05, 1.3]))
    ax = axs[0]
    for m, key, lab in (("P2s0", "P2", "P2 (seed 0)"), ("WA", "WA", "WA-JEPA")):
        v = np.sort(d[f"{m}_margin"].dropna().values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, color=C[key], lw=1.4, label=lab)
    ax.axvline(0, color="#8a8a8a", lw=0.6)
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(0, 15)
    ax.set_xlabel("plan footprint margin (m)")
    ax.set_ylabel("navtest tokens at or below (%)")
    ax.legend(loc="upper left", fontsize=5.8)
    ax.text(-0.08, 0.5, "off road ←", transform=ax.get_xaxis_transform(), ha="right", fontsize=5.6, color=C["mute"])
    ax.set_title("margin to edge")
    tag(ax, "a", x=-0.25)
    ax = axs[1]
    for key in ("P2", "WA"):
        v = np.sort(sc[(sc.model == key) & (sc.DAC < 1)].out_depth.clip(upper=3).values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, color=C[key], lw=1.4, label=f"{'P2 (seed 0)' if key == 'P2' else 'WA-JEPA'}, n={len(v)}")
    ax.axvline(0.3, color="#8a8a8a", lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("depth off road (m)")
    ax.set_ylabel("own DAC failures (%)")
    ax.legend(loc="lower right", fontsize=5.6)
    ax.set_xlim(0, 2.5)
    ax.set_title("excursion depth")
    tag(ax, "b", x=-0.25)
    ax = axs[2]
    tb = geo["turn_bins"]
    x = np.arange(len(tb))
    for k, key, off in ((0, "P2", -0.1), (1, "WA", 0.1)):
        ci_err(ax, x + off, np.array([b["r"][k] for b in tb]) * 100, fmt="o-", ms=3.2, color=C[key], lw=1.0, mec="white", mew=0.5, capsize=0)
    ax.set_xticks(x)
    ax.set_xticklabels([(f"{b['lo']}-{b['hi']}°" if b["hi"] < 400 else f"> {b['lo']}°") + f"\n{b['n']}" for b in tb], fontsize=5.8)
    ax.set_xlabel("logged |Δ heading| in 4 s (n)")
    ax.set_ylabel("DAC failures (%)")
    legend_models(ax, loc="upper left", ncol=1)
    ax.grid(axis="y", color=C["grid"], lw=0.5)
    ax.set_title("curvature")
    tag(ax, "c", x=-0.25)
    ax = axs[3]
    groups = [("P2\non F", tr[tr.set_F].P2_in, C["P2"], tr[tr.set_F].log), ("P2\non PP", tr[tr.set_PP].P2_in, C["P2l"], tr[tr.set_PP].log),
              ("WA\non R", tr[tr.set_R].WA_in, C["WA"], tr[tr.set_R].log), ("WA\non PP", tr[tr.set_PP].WA_in, C["WAl"], tr[tr.set_PP].log)]
    for i, (lab, v, col, lg) in enumerate(groups):
        ok = v.notna()
        vv = v[ok].clip(-5, 5).values
        parts = ax.violinplot([vv], positions=[i], widths=0.85, showextrema=False)
        for p in parts["bodies"]:
            p.set_facecolor(col), p.set_alpha(0.45), p.set_edgecolor("none")
        q1, q2, q3 = np.percentile(vv, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=col, lw=2.2, solid_capstyle="butt")
        mm = cboot(v[ok].values, lg[ok].values)[0]
        ax.errorbar([i + 0.28], [mm[0]], yerr=[[mm[0] - mm[1]], [mm[2] - mm[0]]], fmt="D", ms=2.8, color=C["ink"], lw=0.9, capsize=0)
        ax.text(i, -4.9, f"{ok.sum()}", ha="center", fontsize=5.4, color=C["mute"])
    ax.axhline(0, color="#8a8a8a", lw=0.6)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g_[0] for g_ in groups], fontsize=6)
    ax.set_ylim(-5.2, 5.2)
    ax.set_ylabel("4 s offset to turn inside (m)")
    ax.set_title("turns > 20°: end offset")
    ax.text(0.99, 0.99, "violin + IQR; ◆ mean, 95% CI", transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=C["mute"])
    tag(ax, "d", x=-0.2)
    save(fig, "fig5_geometry")


# ---------------------------------------------------------------- Figure 6: HUGSIM
def fig_hugsim(df, hs, st):
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.6, 0.85], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    M = np.array(hs["matrix"])
    ax.imshow(np.sqrt(M), cmap="Greys", vmin=0, vmax=np.sqrt(M.max()) * 1.5)
    for i in range(len(CLS)):
        for j in range(len(CLS)):
            if M[i, j]:
                ax.text(j, i, f"{M[i, j]:g}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 10 else C["ink"],
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(CLS)))
    ax.set_xticklabels([CLAB[c] for c in CLS], rotation=40, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(CLS)))
    ax.set_yticklabels([CLAB[c] for c in CLS], fontsize=5.8)
    ax.set_xlabel("P2 end (mean of 2 seeds)", color=C["P2"])
    ax.set_ylabel("WA-JEPA end", color=C["WA"])
    ax.set_title("outcome matrix (64)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    tag(ax, "a", x=-0.45, y=1.02)
    # (b) per scenario
    ax = fig.add_subplot(gs[1])
    dd = df.sort_values(["d", "wa_hd"])
    x = np.arange(len(dd))
    mk = {"complete": "o", "fg_coll": "X", "bg_coll": "P", "off_route": "D", "stuck": "s", "spin": "*"}
    ax.vlines(x, np.minimum(dd.wa_hd, np.minimum(dd.s0_hd, dd.s1_hd)), np.maximum(dd.wa_hd, np.maximum(dd.s0_hd, dd.s1_hd)), color="#cfcfcf", lw=0.6)
    for c in CLS:
        for col, ycol, ccol, sz in ((C["WA"], "wa_hd", "wa_cls", 15), (C["P2"], "s0_hd", "s0_cls", 11), (C["P2l"], "s1_hd", "s1_cls", 11)):
            s = (dd[ccol] == c).values
            ax.scatter(x[s], dd[ycol].values[s], marker=mk[c], s=sz, color=col, edgecolor="white", linewidth=0.3, zorder=3)
    tr = dd.route_turn.values >= 30
    ax.scatter(x[tr], np.full(tr.sum(), -0.07), marker="|", s=18, color=C["ink"], lw=0.8, clip_on=False)
    ax.text(-0.5, -0.07, "turning route", fontsize=5.4, ha="right", va="center", color=C["ink"])
    ax.set_xlim(-1, len(dd))
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("64 scenarios, sorted by P2 (seed mean) − WA-JEPA HD")
    ax.set_ylabel("HD score")
    hl = ([Line2D([], [], marker="o", color=C["WA"], ls="", ms=3.5, label="WA-JEPA"), Line2D([], [], marker="o", color=C["P2"], ls="", ms=3.5, label="P2 s0"),
           Line2D([], [], marker="o", color=C["P2l"], ls="", ms=3.5, label="P2 s1")]
          + [Line2D([], [], marker=mk[c], color=C["mute"], ls="", ms=3.5, label=CLAB[c]) for c in CLS])
    ax.legend(handles=hl, loc="upper center", ncol=5, fontsize=5.4, bbox_to_anchor=(0.5, -0.1), handletextpad=0.1, columnspacing=0.7)
    ax.set_title(f"per scenario (HD gap > 0.2: WA-JEPA {hs['wins']['wa']}, P2 {hs['wins']['p2']})")
    tag(ax, "b", x=-0.1)
    # (c) open vs closed loop on the shared sub-metrics, and the route split
    ax = fig.add_subplot(gs[2])
    nt = st["navtest"]
    rows = [("navtest pass", k, [("d", *(-np.array(nt[k][2])[[0, 2, 1]] * 100))]) for k in ("NC", "DAC", "TTC")]
    rows += [("HUGSIM score", lab, [("d", *(np.array(hs["sub_d"][k]) * 100))]) for k, lab in (("nc", "NC"), ("dac", "DAC"), ("ttc", "TTC"), ("rc", "route compl."))]
    rows += [("HUGSIM DAC", f"{k} route", [("d", *(np.array(v) * 100))]) for k, v in hs["by_route_dac"].items()]
    forest(ax, rows, "P2 − WA-JEPA (pp)", colors={"d": C["ink"]}, ms=2.8)
    zero_sides(ax, "← WA-JEPA", "P2 →")
    ax.set_title(pad=11, label="open vs closed loop")
    tag(ax, "c", x=-0.7, y=1.02)
    save(fig, "fig6_hugsim")


# ---------------------------------------------------------------- Figure 7: cases
QUADS = ["P2 fails, WA passes", "WA fails, P2 passes", "both fail"]


def fig_cases(cdir, d):
    cases = json.load(open(cdir / "cases.json"))["cases"]
    fr = np.load(cdir / "frames_small.npz")
    per = max(sum(c["quadrant"] == q for c in cases) for q in QUADS)
    fig = plt.figure(figsize=(7.2, 2.5 * len(QUADS)))
    outer = fig.add_gridspec(len(QUADS), per, wspace=0.05, hspace=0.12, left=0.035, right=0.995, top=0.96, bottom=0.01)
    row = d.set_index("token")
    for qi, q in enumerate(QUADS):
        for ci, c in enumerate([c for c in cases if c["quadrant"] == q]):
            g = outer[qi, ci].subgridspec(2, 2, height_ratios=[1, 2.1], hspace=0.05, wspace=0.03)
            t = c["token"]
            for k, (key, lab) in enumerate((("p2_road", "P2 sees: road cam (W)"), ("wa_CAM_F0", "WA sees: F0 (+ L0 R0 B0)"))):
                a_ = fig.add_subplot(g[0, k])
                im = fr[f"{t}/{key}"]
                a_.imshow(im, aspect="auto")
                a_.set_axis_off()
                a_.text(0.03, 0.95, lab, transform=a_.transAxes, fontsize=4.8, color="white", va="top", bbox=dict(fc="black", alpha=0.5, lw=0, pad=0.8))
            ab = fig.add_subplot(g[1, :])
            bev(ab, c)
            r = row.loc[t]
            fails = lambda m: ", ".join(k for k in ("NC", "DAC", "DDC", "TLC", "TTC") if r[f"{m}_{k}"] < 1) or "passes"  # noqa: E731
            ab.text(0.01, 0.99, f"{r.maneuver}, {r.v0:.1f} m/s, cmd {r.cmd}", transform=ab.transAxes, fontsize=5.4, va="top", color=C["ink"])
            ab.text(0.01, 0.1, f"P2: {fails('P2s0')} ({100 * r.P2s0_score:.0f})", transform=ab.transAxes, fontsize=5.4, color=C["P2"], fontweight="bold")
            ab.text(0.01, 0.03, f"WA: {fails('WA')} ({100 * r.WA_score:.0f})", transform=ab.transAxes, fontsize=5.4, color=C["WA"], fontweight="bold")
            ab.text(0.99, 0.99, t[:8], transform=ab.transAxes, fontsize=4.8, va="top", ha="right", color=C["mute"])
            if ci == 0:
                fig.text(0.012, 0.96 - (qi + 0.5) * 0.95 / len(QUADS), q, rotation=90, fontsize=7.5, fontweight="bold", va="center", ha="center")
    hl = [Line2D([], [], color=C["P2"], lw=1.4, label="P2 plan (seed 0)"), Line2D([], [], color=C["WA"], lw=1.4, label="WA-JEPA plan"),
          Line2D([], [], color=C["GT"], lw=1.0, ls=(0, (2, 1.5)), label="logged future"), Patch(fc="#e9e9e9", ec="#bdbdbd", label="drivable area"),
          Patch(fc="#9fd3d3", label="vehicle (t0; dashed 4 s)"), Patch(fc="#d7a8e0", label="pedestrian / bike")]
    fig.legend(handles=hl, loc="upper center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig7_cases")


def bev(ax, c):
    to = lambda xy: np.stack([-np.asarray(xy)[..., 1], np.asarray(xy)[..., 0]], -1)  # noqa: E731  (forward = up, left = left)
    for p in c["polygons"]:
        if p["kind"] == "area":
            ax.add_patch(Polygon(to(p["exterior"]), closed=True, fc="#e9e9e9", ec="#c4c4c4", lw=0.4, zorder=0))
            for h in p["holes"]:
                ax.add_patch(Polygon(to(h), closed=True, fc="white", ec="#c4c4c4", lw=0.4, zorder=0))
    for p in c["polygons"]:
        if p["kind"] in ("lane", "route"):
            xy = to(p["exterior"])
            ax.plot(xy[:, 0], xy[:, 1], color="#d6d6d6", lw=0.3, zorder=1)
    for a in c["agents"].values():
        col = {"vehicle": "#9fd3d3", "pedestrian": "#d7a8e0", "bicycle": "#d7a8e0"}.get(a["kind"], "#ffb27a")
        if a["polys"][0] is not None:
            ax.add_patch(Polygon(to(a["polys"][0]), closed=True, fc=col, ec="none", zorder=2))
        if a["polys"][-1] is not None and a["kind"] != "red_light":
            ax.add_patch(Polygon(to(a["polys"][-1]), closed=True, fc="none", ec=col, lw=0.5, ls=(0, (1.5, 1)), zorder=2))
    dims = c["dims"]
    ego = np.array([[dims["front"], dims["width"] / 2], [dims["front"], -dims["width"] / 2], [-dims["rear"], -dims["width"] / 2],
                    [-dims["rear"], dims["width"] / 2]])
    ax.add_patch(Polygon(to(ego), closed=True, fc="#444444", ec="none", zorder=5))
    gt = np.vstack([[0, 0, 0], c["gt"]])
    ax.plot(*to(gt[:, :2]).T, color=C["GT"], lw=1.0, ls=(0, (2, 1.5)), zorder=6)
    for k, col in (("WA", C["WA"]), ("P2s0", C["P2"])):
        P = np.vstack([[0, 0, 0], c["plans"][k]])
        ax.plot(*to(P[:, :2]).T, color=col, lw=1.3, zorder=7)
        e, h = P[-1], P[-1, 2]
        box = np.array([[e[0] + np.cos(h) * fx - np.sin(h) * fy, e[1] + np.sin(h) * fx + np.cos(h) * fy] for fx, fy in ego])
        ax.add_patch(Polygon(to(box), closed=True, fc="none", ec=col, lw=0.8, zorder=7))
    pts = np.vstack([gt[:, :2]] + [np.asarray(c["plans"][k])[:, :2] for k in ("WA", "P2s0")] + [[[-2, 0], [6, 0]]])
    cx, cy = pts[:, 0].mean() * 0 + (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2
    hx = max(pts[:, 0].ptp(), pts[:, 1].ptp() / 1.45, 16) / 2 + 5
    ax.set_ylim(cx - hx, cx + hx)
    ax.set_xlim(-cy - hx * 1.45, -cy + hx * 1.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color("#cccccc"), s.set_linewidth(0.5)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    d = load()
    nh = pd.read_parquet(DATA / "navhard_tokens.parquet")
    dec = pd.read_parquet(DATA / "decoders.parquet")
    wsc = pd.read_csv(DATA / "wa_eval_score.csv")
    sc = pd.concat([pd.read_csv(DATA / "p2_eval_score.csv").assign(model="P2"), wsc.assign(model="WA")]).rename(
        columns={"drivable_area_compliance": "DAC"})
    hdf = hugsim_load()
    st = {"navtest": navtest_metric_stats(d), "strata": stratum_stats(d), "navhard": navhard_stats(nh), "hugsim": hugsim_stats(hdf),
          "loc": localisation_stats(d, dec)}
    st["geo"], tr = geometry_stats(d, sc)
    m = d.set_index("token").loc[wsc.token]
    st["wa_replay_dac_agree"] = float(((wsc.drivable_area_compliance.values >= 1) == (m.WA_DAC.values >= 1)).mean())
    p2r = pd.read_csv(DATA / "p2_eval_score.csv")
    m2 = d.set_index("token").loc[p2r.token]
    st["p2_replay_dac_agree"] = float(((p2r.drivable_area_compliance.values >= 1) == (m2.P2s0_DAC.values >= 1)).mean())
    st["cases"] = json.load(open(DATA / "cases_pick.json"))
    (J / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    figs = {"1": lambda: fig_overview(st), "2": lambda: fig_taxonomy(st), "3": lambda: fig_navhard(st["navhard"]),
            "4": lambda: fig_localisation(st["loc"]), "5": lambda: fig_geometry(d, sc, st["geo"], tr), "6": lambda: fig_hugsim(hdf, st["hugsim"], st)}
    for k, f in figs.items():
        if not a.only or k in a.only:
            f()
    if a.cases and (not a.only or "7" in a.only):
        fig_cases(Path(a.cases), d)
    page(st)


def ci(v, k=100, f=".1f", sign=False):
    v = np.asarray(v, float) * k
    sp = "+" if sign else ""
    return f"{v[0]:{sp}{f}} [{v[1]:{sp}{f}}, {v[2]:{sp}{f}}]"


def fig_html(name, cap, look):
    return (f'<figure id="{name}"><a href="figs/{name}.pdf"><img src="figs/{name}.png" alt="{html.escape(cap[:80])}"></a>'
            f'<figcaption><b>{cap}</b> <span class="look">What to look at: {look}</span> '
            f'<a href="figs/{name}.pdf">PDF</a></figcaption></figure>')


def page(st):
    nt, nh, hs, L, g = st["navtest"], st["navhard"], st["hugsim"], st["loc"], st["geo"]
    sr = {r["label"]: r for r in st["strata"]}
    dec = lambda o, s_, n: ci(L["decoder"][f"{o}|{s_}|{n}"][1], sign=True)  # noqa: E731
    mb = lambda s_, n: ci(L["margin"][f"{s_}|{n}"], k=1, f=".2f", sign=True)  # noqa: E731
    tb = g["turn_bins"]
    ti = g["turn_inside"]
    rows_metric = "".join(
        f"<tr><td>{t}</td><td>{ci(nt[t][0])}</td><td>{ci(nt[t][1])}</td><td>{ci(nt[t][2], sign=True)}</td>"
        f"<td>{100 * nt[t][3][0]:.2f}</td><td>{100 * nt[t][4][0]:.2f}</td><td>{100 * nt[t][5][0]:.2f}</td></tr>"
        for t in ("DAC", "NC", "TTC", "LK", "DDC", "TLC", "HC", "EC"))
    rows_nh = "".join(
        f"<tr><td>{t}</td>" + "".join(f"<td>{ci(nh[f'stage{s_}'][t][0])}</td><td>{ci(nh[f'stage{s_}'][t][1])}</td>" for s_ in (1, 2)) + "</tr>"
        for t in ("NC", "DAC", "DDC", "TTC", "LK", "EP"))
    picks = st["cases"]["picks"]
    rule = html.escape(st["cases"]["rule"].replace("Selection rule (shown on the page): ", "").replace("`per`", "3"))
    FIG1 = fig_html('fig1_overview', 'Figure 1. Where each model works and fails.', '(a) WA-JEPA leads on every benchmark, by less in closed loop. (b) The shared failures (purple) are a small part of each gate; P2-only DAC failures are 4x WA-only ones, every gate has WA-only failures too. (c) Bars near 0 on a model\'s own failure set (orange F for P2 stages, blue R for WA stages): no encoder adds information over ego state where its model fails. (d) The navtest deficit concentrates in turns, left-hand traffic and open junctions. (e) In closed loop the deficit is confined to turning routes and the KITTI-360 / medium scenarios; P2 leads on nuScenes.')
    FIG2 = fig_html('fig2_taxonomy', 'Figure 2. Score difference and per-gate failure rates by scene type (navtest).', 'turns and lane changes carry DAC and LK failures for both models, but P2\'s grow much faster; NC / TTC failures of both concentrate in coming-to-a-stop scenes (P2 more); EC failures are turn-driven and equal for both; in Singapore P2 has 3x WA\'s DAC failures but no NC / TTC excess.')
    FIG3 = fig_html('fig3_navhard', 'Figure 3. navhard two-stage (Protocol G frames for P2).', 'stage 1 differences sit in DAC and DDC; in stage 2 DAC remains and the large shared LK / DDC / TTC failure rates are the same for both models: the synthetic states are hard for both, not specifically for P2. Stage 2 is scored unweighted here (official stage-2 weights are arm-specific).')
    FIG4 = fig_html('fig4_localisation', 'Figure 4. Which stage decides each model\'s failures.', '(a) the mirror: orange bars (F) are at 0 for all P2 stages and high for all WA stages; blue bars (R) are high for all P2 stages and at 0 for all WA stages. (b) probes find more road-edge information in every WA stage than in every P2 stage, on all sets alike, so the failure sets are not where either encoder is blind in general. (c) on its own failures P2\'s vision stage is as optimistic about the remaining margin as an ego-only probe (dashed), i.e. it adds no road edge there; WA\'s front encoder roughly halves the ego-only bias on R and its head hidden returns to it.')
    FIG5 = fig_html('fig5_geometry', 'Figure 5. Margins, excursion depth, curvature and corner cutting.', f'(a) P2\'s raw plans leave the drivable area on {ci(g["P2_raw_out"])}% of navtest tokens vs {ci(g["WA_raw_out"])}% for WA. (b) WA\'s DAC failures are mostly sub-0.3 m tracker grazes; P2\'s go deeper. (c) the P2 − WA DAC gap grows from {ci(tb[0]["r"][2], sign=True)} pp below 5° to {ci(tb[-1]["r"][2], sign=True)} pp above 45°. (d) on its specific failures P2 ends inside the turn; on shared turns both are slightly wide.')
    FIG6 = fig_html('fig6_hugsim', 'Figure 6. HUGSIM outcome matrix, per-scenario scores, and the link to open loop.', f'(a) the diagonal (same end) holds {sum(np.diag(np.array(hs["matrix"]))):g} of 64; where WA-JEPA completes and P2 does not, P2 mostly spins or collides (background more than foreground); where P2 completes and WA-JEPA does not, WA-JEPA hit a foreground agent. (b) tick marks under the strip: routes that turn ≥ 30°; WA-JEPA\'s wins cluster on them. (c) DAC is the one sub-metric where the open-loop deficit reappears in closed loop, and only on turning routes.')
    FIG7 = fig_html('fig7_cases', 'Figure 7. Three cases per quadrant: what each model sees and both plans in BEV.', 'top of each case: P2\'s current road-camera input (protocol W) and WA-JEPA\'s front view (it also sees L0 / R0 / B0); bottom: BEV at t0 with drivable area, lanes, agents (dashed: 4 s later), the logged future and both plans with their 4 s ego boxes. Numbers in brackets: token EPDMS × 100.')
    H = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>P2 vs WA-JEPA joint diagnosis</title>
<style>
:root {{ --ink:#1d1d1f; --mute:#5f6368; --line:#e3e3e3; --bg:#ffffff; --p2:#D55E00; --wa:#0072B2; --both:#CC79A7; --card:#fafafa; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --ink:#e8e8e8; --mute:#a0a0a0; --line:#3a3a3a; --bg:#161616; --card:#1f1f1f; }}
  :root:not([data-theme="light"]) figure img {{ background:#fff; }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; margin:0; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; line-height:1.25; margin:8px 0 4px; }} h2 {{ font-size:20px; margin:40px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
h3 {{ font-size:16px; margin:22px 0 6px; }} .sub {{ color:var(--mute); margin:0 0 18px; }}
.p2 {{ color:var(--p2); font-weight:600; }} .wa {{ color:var(--wa); font-weight:600; }}
.key {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 18px; }}
.key li {{ margin:6px 0; }}
figure {{ margin:22px 0; }} figure img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:4px; }}
figcaption {{ font-size:13.5px; color:var(--ink); margin-top:6px; }} .look {{ color:var(--mute); }}
table {{ border-collapse:collapse; font-size:13px; margin:10px 0; width:100%; display:block; overflow-x:auto; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:right; white-space:nowrap; }} th:first-child, td:first-child {{ text-align:left; }}
th {{ font-weight:600; }} code {{ font-size:12.5px; }} .note {{ font-size:13.5px; color:var(--mute); }}
nav a {{ margin-right:12px; font-size:14px; }}
</style></head><body><main>
<h1>Where <span class="p2">P2</span> and <span class="wa">WA-JEPA</span> work, where each fails, and at which stage</h1>
<p class="sub">Joint diagnosis, 2026-10-06. <span class="p2">P2</span> = op_parity P2-F (Cinque 382M, comma vision frozen, ego / pose / command
inputs, protocol W frames; 2 seeds). <span class="wa">WA-JEPA</span> = released checkpoint (V-JEPA 2.1 ViT-L + MMDiT, one run). Builds on
<a href="../dac-localize.md">dac-localize.md</a> (decision 147) and the op_parity gap page. No model was re-run; all numbers come from stored
plans, devkit sub-scores, op_probe features / probes / decoders, and HUGSIM run logs.</p>
<nav><a href="#summary">Summary</a><a href="#overview">Overview</a><a href="#taxonomy">Taxonomy</a><a href="#stages">Stages</a><a href="#geometry">Geometry</a><a href="#closed">Closed loop</a><a href="#cases">Cases</a><a href="#method">Method</a></nav>

<h2 id="summary">Summary</h2>
<div class="key"><ul>
<li><b>WA-JEPA is ahead on every navtest scene type; the gap is a turning problem.</b> Per-token EPDMS P2 − WA is {ci(sr['straight']['score_d'], sign=True)} on
straight driving and {ci(sr['left turn > 20°']['score_d'], sign=True)} / {ci(sr['right turn > 20°']['score_d'], sign=True)} on left / right turns
(× 100). DAC failures rise with curvature for P2 ({100 * tb[0]['r'][0][0]:.1f}% below 5° → {100 * tb[-1]['r'][0][0]:.1f}% above 45°) and barely for WA
({100 * tb[0]['r'][1][0]:.1f}% → {100 * tb[-1]['r'][1][0]:.1f}%). Near parity: straight, ≥ 10 m/s ({ci(sr['≥ 10 m/s']['score_d'], sign=True)}), Las Vegas
({ci(sr['Las Vegas']['score_d'], sign=True)}). Largest non-manoeuvre gaps: Singapore, left-hand traffic ({ci(sr['Singapore (left-hand)']['score_d'], sign=True)}),
near-empty roads with 0-3 vehicles ({ci(sr['0-3 vehicles']['score_d'], sign=True)}, i.e. turns at open junctions).</li>
<li><b>Where P2 is not worse.</b> Closed loop on straight routes: HUGSIM HD P2 − WA {ci(hs['by_route']['straight route (< 30°)']['d'], sign=True)} (× 100, 41
scenarios); on the nuScenes scenes P2 is ahead, {ci(hs['by_dataset']['nuscenes']['d'], sign=True)}. Open loop: extended comfort (EC fail rate P2 − WA
{ci(nt['EC'][2], sign=True)} pp), launch scenes on navhard stage 1, lane changes on navhard stage 2 (both CIs include 0). No navtest scene type has P2 ahead.</li>
<li><b>Each model's failures are fixed at its own encoder (mirror of decision 147, point 4).</b> A fresh decoder (stage + ego, drivable hinge) on
P2's vision tokens passes DAC on P2's failure set F no more often than an ego-only decoder ({dec('hinge10', 'P2-V', 'F')} pp), while on WA-JEPA's
failure set R it gains {dec('hinge10', 'P2-V', 'R')} pp. The same holds the other way: WA's front-encoder decoder gains {dec('hinge10', 'WA-Cf', 'F')} pp on F and
{dec('hinge10', 'WA-Cf', 'R')} pp on R; WA's head hidden {dec('hinge10', 'WA-H', 'F')} vs {dec('hinge10', 'WA-H', 'R')}. On the tokens both fail (FF), no stage of either model
adds anything over ego state. Caveat: F and R are each selected by one model failing and the other passing, which inflates the cross-set gains (regression to the mean); the
readable part is that every stage of the failing model sits at the ego-only level on its own failure set.</li>
<li><b>P2's DAC failures are in the plan; most of WA-JEPA's arise only in the tracker replay.</b> When P2 fails DAC its raw plan footprint already leaves the road in
{ci(g['P2_raw_out_given_fail'])}% of cases; for WA only {ci(g['WA_raw_out_given_fail'])}%. In the devkit's LQR replay {ci(g['WA_lqr_only'])}% of WA's DAC failures are
tracker-only and {ci(g['WA_graze03'])}% are shallower than 0.3 m (median {100 * g['WA_depth_med']:.0f} cm); P2: {ci(g['P2_lqr_only'])}% and {ci(g['P2_graze03'])}% (median
{100 * g['P2_depth_med']:.0f} cm).</li>
<li><b>P2's specific turn failures cut the corner.</b> On turns > 20° where P2 fails and WA passes, P2's 4 s end point lies
{ci(ti['F']['P2_in'], k=1, f='.2f', sign=True)} m toward the inside of the logged turn; on turns both pass it lies {ci(ti['PP']['P2_in'], k=1, f='.2f', sign=True)} m (slightly wide). WA on its own
failures: {ci(ti['R']['WA_in'], k=1, f='.2f', sign=True)} m. Tokens both fail are the opposite: both run wide ({ci(ti['FF']['P2_in'], k=1, f='.2f', sign=True)} / {ci(ti['FF']['WA_in'], k=1, f='.2f', sign=True)} m).</li>
<li><b>The open-loop deficit carries into closed loop where the route turns.</b> HUGSIM HD gap is {ci(hs['hd_d'], sign=True)} overall (n.s., as in
decision 144), {ci(hs['by_route']['turning route (≥ 30°)']['d'], sign=True)} on the 23 scenarios whose route turns ≥ 30°, with DAC {ci(hs['by_route_dac']['turning'], sign=True)} there and
{ci(hs['by_route_dac']['straight'], sign=True)} on straight routes. How runs end: P2 completes {hs['counts']['P2']['complete']:g} vs {hs['counts']['WA']['complete']}, spins {hs['counts']['P2']['spin']:g}
vs {hs['counts']['WA']['spin']}, bg-collides {hs['counts']['P2']['bg_coll']:g} vs {hs['counts']['WA']['bg_coll']}; WA leaves the route {hs['counts']['WA']['off_route']} vs {hs['counts']['P2']['off_route']:g}.
Foreground collisions are shared ({hs['counts']['P2']['fg_coll']:g} vs {hs['counts']['WA']['fg_coll']}; {hs['matrix'][1][1]:g} scenarios end that way for both).</li>
<li><b>navhard.</b> Stage 1 (logged) P2 − WA {ci(nh['stage1']['score'][2], sign=True)}, from DAC (+{100 * nh['stage1']['DAC'][2][0]:.1f} pp failures), DDC
(+{100 * nh['stage1']['DDC'][2][0]:.1f}) and EP; stage 2 (rendered) {ci(nh['stage2']['score'][2], sign=True)}, from DAC (+{100 * nh['stage2']['DAC'][2][0]:.1f}) and EP
({ci(nh['stage2']['EP'][2], sign=True)}): in synthetic follow-up states P2 also drives too slowly. Turns are the worst manoeuvre in both stages.</li>
</ul></div>

<h2 id="overview">Overview</h2>
{FIG1}

<h2 id="taxonomy">1. Failure taxonomy by sub-metric and scene type</h2>
<p>navtest, 12 146 tokens, {nt['logs']} logs. Rates are the share of tokens with the sub-score below 1 (P2: mean of its two seeds); the three
right-hand columns split tokens by P2 seed 0 vs WA. Scene attributes: logged 4 s motion (manoeuvre classes in <a href="#method">Method</a>),
ego speed, a path through an INTERSECTION polygon, the route command, agents within 30 m, a red light listed for the frame, city, and the sun
elevation at the log's time (navtest has no night frames; weather is not labelled in nuPlan).</p>
<table><tr><th>gate</th><th>P2 fail %</th><th>WA fail %</th><th>P2 − WA (pp)</th><th>only P2 %</th><th>only WA %</th><th>both %</th></tr>{rows_metric}</table>
{FIG2}
{FIG3}
<table><tr><th>navhard</th><th>stage 1 P2</th><th>stage 1 WA</th><th>stage 2 P2</th><th>stage 2 WA</th></tr>{rows_nh}</table>
<p class="note">EP row: mean ego-progress (× 100), not a failure rate. Score: stage 1 P2 {ci(nh['stage1']['score'][0])} vs WA {ci(nh['stage1']['score'][1])};
stage 2 {ci(nh['stage2']['score'][0])} vs {ci(nh['stage2']['score'][1])}. Clusters: {nh['stage1']['groups']} scene groups.</p>

<h2 id="stages">2. Stage localisation in both directions</h2>
<p>Sets as in dac-localize.md, on DAC: F = P2 seed 0 fails, WA passes ({L['n_dec']['F']}); R = WA fails, P2 passes ({L['n_dec']['R']}); FF = both fail
({L['n_dec']['FF']}); PP-turn = both pass with a logged turn > 20°. Stages: P2 V (current vision tokens), M (after adapter bias and token MLPs), T (temporal
summary), H (plan-head hidden); WA Cf (front-view encoder, newest tubelet, pooled like V), Ca (all views), T (trajectory-head input), H (its last hidden).
Decoders: 2-layer MLP on [stage, ego], imitation + drivable hinge (λ = 10), trained on navtrain, scored with the devkit.</p>
{FIG4}
<table><tr><th>decoder lift over ego-only (pp), hinge λ = 10</th><th>F</th><th>R</th><th>FF</th><th>PP-turn</th></tr>
{''.join(f"<tr><td>{s_}</td>" + "".join(f"<td>{dec('hinge10', s_, n)}</td>" for n in ('F', 'R', 'FF', 'PPturn')) + "</tr>" for s_ in ('P2-V', 'P2-M', 'P2-T', 'P2-H', 'WA-Cf', 'WA-Ca', 'WA-H'))}
</table>
<p class="note">Imitation-only decoders give the same pattern (P2-V on F {dec('imit', 'P2-V', 'F')}, on R {dec('imit', 'P2-V', 'R')}; WA-Cf on F {dec('imit', 'WA-Cf', 'F')}, on R
{dec('imit', 'WA-Cf', 'R')}). Margin bias (probe − true footprint margin along the model's own plan, m): P2-V on F {mb('P2-V', 'F')} vs ego-only {mb('E-P2', 'F')};
WA-Cf on R {mb('WA-Cf', 'R')}, WA-H on R {mb('WA-H', 'R')} vs ego-only {mb('E-WA', 'R')}. Reading: P2's failures are encoder-level by both readouts; WA's are
encoder-level by the decoder readout, while its front encoder still carries some road-edge information on R that the head hidden no longer
shows (n = 115, wide CIs).</p>

<h2 id="geometry">3. Geometry of the failures</h2>
{FIG5}
<p class="note">Margins: smallest signed distance of the ego footprint (4 corners, plan linearly interpolated to 0.1 s) to the scorer's drivable polygons
(ROADBLOCK, INTERSECTION, CARPARK_AREA), 0.5 m SDF labels of opb_labels.py. Depth: largest corner distance outside those polygons along the devkit's LQR
replay (opb_score.py), on the op_probe eval tokens (F ∪ R ∪ FF ∪ 1 500 random PP); both models' replays reproduce their devkit DAC on
{100 * st['p2_replay_dac_agree']:.0f}% / {100 * st['wa_replay_dac_agree']:.0f}% of tokens. End offset: lateral offset of the plan's 4 s pose from the logged 4 s pose,
along the logged pose's normal, signed + toward the inside of the logged turn.</p>

<h2 id="closed">4. Closed loop: HUGSIM 64</h2>
{FIG6}
<p class="note">exam preset; P2 = mean of seeds s0 / s1 (one run each), WA-JEPA one run (decision 138), single runs vary by about 0.2 HD per scenario
(decision 144), so per-scenario differences below 0.2 are not read. Route turn = range of the recorded route heading. spec-preset counts in
<a href="../../../op_parity/results/hugsim_full.md">hugsim_full.md</a>.</p>

<h2 id="cases">5. Paired cases per quadrant</h2>
{FIG7}
<p class="note">Selection rule (fixed before looking at plans): {rule} Drawn: {', '.join(f"{p['quadrant']} / {p['gate']} {p['token'][:8]}" for p in picks)}.</p>

<h2 id="method">Method and caveats</h2>
<ul class="note">
<li>Manoeuvre classes from the logged 4 s future (priority order): stationary (v0 &lt; 0.5 m/s, path &lt; 1 m), launch (v0 &lt; 1.5, end speed &gt; 2.5),
stop (v0 &gt; 2.5, end &lt; 0.5), left / right turn (|Δ heading| &gt; 20°), lane change (|Δ heading| &lt; 8°, |lateral| &gt; 2 m, path &gt; 10 m),
curve (8-20°), straight. navhard stage-2 tokens have no logged future and take the class of their scene group's stage-1 token.</li>
<li>CIs: 95% percentile cluster bootstrap, B 2 000 (navtest logs, navhard scene groups, HUGSIM scenarios). Strata overlap across groups; no
multiplicity correction; differences whose CI excludes 0 in one of ~30 strata are descriptive.</li>
<li>F / R / FF are defined on P2 seed 0 as in decision 147; the taxonomy uses the 2-seed mean. Decoder and probe readouts are DAC-only.</li>
<li>Inputs differ (WA: 4 cameras, 2 Hz history; P2: front cameras, protocol W). WA-JEPA is a single run on every benchmark.</li>
<li>The case figure shows P2's current road frame only (it also sees the wide camera and 8 history slots); GIF-style context is on the
<a href="../../../op_parity/results/gap/index.html">op_parity gap page</a>.</li>
</ul>
<p class="note">Reproduce: <code>experiments/op_probe/scripts/opj_build.py</code> (box: attrs / table / cases / export / frames; devkit replays via
<code>opb_score.py</code>), then <code>.venv/bin/python experiments/op_probe/scripts/opj_figs.py --cases DIR</code> (Mac). Per-token tables (gitignored <code>results/joint/data/</code>, from
<code>$DATA_DIR/runs/op_probe/joint/</code> on the box), every number on this page in <a href="stats.json">stats.json</a>.</p>
</main></body></html>
"""
    (J / "index.html").write_text(H)
    print("wrote index.html")


if __name__ == "__main__":
    main()
