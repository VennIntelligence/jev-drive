"""Pedestrian dose-response readout (todos/2026-09-28-ped-dose-response.md; renders by scripts/p3/dose.py).

Set nq4_p3_dose: one scene dir per cell, real = the log, plus = the cell (a donor pedestrian inserted), minus = the target
re-render without insertion. After the registered chain (nq4_p3 index -> p5_openpilot --arrays temporal plan lead
lead_prob -> finalize -> nq4_p3 exam), this reads every cell at t* and t* + 0.4 s:

  native plan   v2 (2 s speed), v_min over the first 4 s, stop = v_min < 0.5 m/s
  lead          P(lead) of selection 0 and its x (openpilot calibrated frame, from the camera)
  temporal      the I3 ridge_late v2 (the decision-42 probe)

per cell and model: reaction = v2(x+) - v2(x-) <= -tau (tau = the model's I3 ridge_late value, borrowed for the native
plan as in P3), stop = stop(x+) and not stop(x-), lead = P(x+) - P(x-) >= 0.3 with |x_lead(x+) - (distance from the
camera)| <= 30 % of it; appearance null = v2(x-) vs v2(log) on the same frames.

  P3_SET=nq4_p3_dose P5_SET=nq4_p3_dose python -m jevdrive.ped_dose --exam-dir <runs/nq4/nq4_p3_dose-exam/...> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger
from .nq4_k import lead_decode
from .nq4_p3 import I3_EXAM, root
from .p5_pairs import v2

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
DISTS = (3, 5, 8, 12, 20, 30)
CAM_X = 1.519            # front camera ahead of the rear axle (m): the WOD rig in jevdrive.p5_openpilot.RIG
FRONT = 4.0


def vmin4(fut: np.ndarray) -> np.ndarray:
    pts = np.concatenate([np.zeros_like(fut[:, :1]), fut[:, :16]], 1)
    return (np.linalg.norm(np.diff(pts, axis=1), axis=-1) / 0.25).min(1)


EGO_HALF, PED_R, EGO_LEN, HORIZON = 1.0, 0.3, 5.0, 4.0     # m, m, m (bumper to rear end), s


def threat_labels(w: pd.DataFrame, arc: dict) -> pd.DataFrame:
    """Readout clarification (user 2026-09-28): not reacting is correct unless the pedestrian is on a collision course.
    Geometry only, x- path = the logged ego path from t*: the ego (half width EGO_HALF) moves along it as logged; the
    pedestrian stands at its lateral offset, or keeps crossing towards the far side at its donor speed. Conflict = within
    HORIZON s, the pedestrian (radius PED_R) is inside the ego corridor while the ego body spans its arc position.
    threat = must_react (conflict, ttc = first conflict time) or no_threat."""
    tt = np.arange(0.0, HORIZON + 1e-9, 0.05)
    ttc, reach = [], []
    for b, dist, lat, ps, vp in zip(w.base_id, w.dist, w.lat, w.ped_state, w.donor_speed):
        s = np.interp(tt, np.r_[0.0, 0.25 * np.arange(1, 21)], np.r_[0.0, arc[b]])
        lt = np.full_like(tt, lat) if ps == "stand" else lat - vp * tt
        hit = (s >= dist - PED_R) & (s <= dist + EGO_LEN + PED_R) & (np.abs(lt) <= EGO_HALF + PED_R)
        ttc.append(tt[hit.argmax()] if hit.any() else np.nan)
        reach.append(tt[(s >= dist - PED_R).argmax()] if (s >= dist - PED_R).any() else np.nan)
    w["ttc"], w["t_reach"] = ttc, reach                    # t_reach: when the logged ego bumper reaches the pedestrian's arc
    w["threat"] = np.where(np.isfinite(w.ttc), "must_react", "no_threat")
    return w


def _boot_ci(x: np.ndarray, g: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """95 % CI of a rate, bootstrapping scene x state clusters."""
    if len(x) == 0:
        return np.nan, np.nan
    u = np.unique(g)
    idx = [np.flatnonzero(g == k) for k in u]
    rng = np.random.default_rng(0)
    m = [np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))]) for _ in range(n)]
    v = np.array([x[i].mean() for i in m])
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def threat_summary(w: pd.DataFrame, out: Path) -> pd.DataFrame:
    """React / stop / lead rates with cluster-bootstrap CIs on must-react cells (ability) and no-threat cells (restraint:
    a reaction there is a false alarm)."""
    w0 = w[w.t_rel_f0 == 0.0].copy()
    w0["cl"] = w0.scene.astype(str) + "_" + w0.state
    rows = []
    for m in MODELS:
        for st in [s for s in ("pull", "creep", "cruise") if (w0.state == s).any()] + ["all"]:
            for th in ("must_react", "no_threat"):
                g = w0[(w0.threat == th) & ((w0.state == st) | (st == "all"))]
                row = {"model": m, "state": st, "threat": th, "cells": len(g), "scenes": g.scene.nunique()}
                for k in ("react", "stop", "lead"):
                    x = g[f"{m}|{k}"].astype(float).to_numpy()
                    lo, hi = _boot_ci(x, g.cl.to_numpy())
                    row |= {k: x.mean() if len(x) else np.nan, f"{k}_lo": lo, f"{k}_hi": hi}
                row["dv2_median"] = g[f"{m}|dv2"].median()
                rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(out / "threat.csv", index=False)
    return T


def cells_table(exam_dir: Path) -> pd.DataFrame:
    import jevdrive.p5_openpilot as P5
    t = pd.read_parquet(root() / "index.parquet")
    t["sfx"] = t.frame_name.str.rsplit("-", n=1).str[-1]
    op = P5.load(t, MODELS, arrays=("plan", "lead", "lead_prob"), sub="op_streams")
    for m in MODELS:
        plan = np.asarray(op[f"op-{m} plan"], np.float64).reshape(-1, 20, 2)
        t[f"{m}|v2"], t[f"{m}|vmin"] = v2(plan), vmin4(plan)
        x, _, p = lead_decode(op[f"op-{m} lead"], op[f"op-{m} lead_prob"])
        t[f"{m}|lead_x"], t[f"{m}|lead_p"] = x, p
    fut = np.load(root() / "future.npy").astype(np.float64)
    arc = np.cumsum(np.linalg.norm(np.diff(np.concatenate([np.zeros_like(fut[:, :1]), fut], 1), axis=1), axis=-1), 1)
    r0 = ((t.world == "real") & (t.t_rel_f0.round(1) == 0.0)).to_numpy()
    arc = dict(zip(t.base_id[r0], arc[r0]))                 # logged ego path length travelled, 0.25 .. 5 s after t*
    t = t[t.t_rel_f0.round(1).isin([0.0, 0.4])]
    vals = [c for c in t.columns if "|" in c]
    w = t.pivot_table(index=["base_id", "t_rel_f0"], columns="world", values=vals).reset_index()
    w.columns = [f"{a}_{b}" if b else a for a, b in w.columns]
    fs = pd.read_parquet(exam_dir / "frames_scored.parquet")
    fs = fs.rename(columns={c: c.replace("ridge_late op-", "rl_").replace(" temporal", "") for c in fs.columns})
    fs["t_rel_f0"] = fs.t_rel_f0.round(1)
    w["t_rel_f0"] = w.t_rel_f0.round(1)
    w = w.merge(fs[["base_id", "t_rel_f0"] + [c for c in fs.columns if c.startswith("rl_")]], on=["base_id", "t_rel_f0"], how="left")
    meta = {}
    for sd in (root() / "scenes").glob("p3_*"):
        m = json.loads((sd / "meta.json").read_text())
        meta[sd.name] = {q: m[q] for q in ("scene", "state", "ego_v", "cell", "dist", "lat", "ped_state", "view_gap", "shadow_on_frac",
                                           "donor_speed")}
    w = w.join(pd.DataFrame(w.base_id.map(meta).tolist()))
    w = threat_labels(w, arc)
    taus = pd.read_csv(data_dir() / I3_EXAM / "flip_rates.csv").query("scope == 'pooled'").set_index("examinee").tau_model
    for m in MODELS:
        tau = float(taus[f"ridge_late op-{m} temporal"])
        d = w[f"{m}|v2_plus"] - w[f"{m}|v2_minus"]
        w[f"{m}|dv2"] = d
        w[f"{m}|react"] = (d <= -tau)
        # stop: the plan comes to a halt with the pedestrian and not without it. From standstill (pull-away) v_min is
        # below 0.5 m/s in both worlds by construction, so there it is "hold": v2 < 0.5 m/s in x+ while x- moves off
        hold = (w[f"{m}|v2_plus"] < 0.5) & ~(w[f"{m}|v2_minus"] < 0.5)
        stop = (w[f"{m}|vmin_plus"] < 0.5) & ~(w[f"{m}|vmin_minus"] < 0.5)
        w[f"{m}|stop"] = np.where(w.state == "pull", hold, stop)
        dist_cam = w.dist + FRONT - CAM_X
        w[f"{m}|lead"] = ((w[f"{m}|lead_p_plus"] - w[f"{m}|lead_p_minus"]) >= 0.3) & \
                         ((w[f"{m}|lead_x_plus"] - dist_cam).abs() <= 0.3 * dist_cam)
        w[f"{m}|null_flip"] = (w[f"{m}|v2_minus"] - w[f"{m}|v2_real"]).abs() >= tau
        rl = w[f"rl_{m}|v2_plus"] - w[f"rl_{m}|v2_minus"]
        w[f"{m}|rl_dv2"], w[f"{m}|rl_react"] = rl, rl <= -tau
    return w


def curves(w: pd.DataFrame, out: Path):
    from . import plots
    import matplotlib.pyplot as plt
    plt.rcParams.update(plots.STYLE)
    w0 = w[w.t_rel_f0 == 0.0]
    states = [s for s in ("pull", "creep", "cruise") if (w0.state == s).any()]
    lats = sorted(w0.lat.unique())
    rows = []
    for (m, st, lat, ps, dist), g in [((m,) + k, g) for m in MODELS for k, g in w0.groupby(["state", "lat", "ped_state", "dist"])]:
        rows.append({"model": m, "state": st, "lat": lat, "ped_state": ps, "dist": dist, "n": len(g),
                     "react": g[f"{m}|react"].mean(), "stop": g[f"{m}|stop"].mean(), "lead": g[f"{m}|lead"].mean(),
                     "rl_react": g[f"{m}|rl_react"].mean(), "dv2_median": g[f"{m}|dv2"].median(),
                     "null_flip": g[f"{m}|null_flip"].mean()})
    R = pd.DataFrame(rows)
    R.to_csv(out / "curves.csv", index=False)
    fig, axes = plt.subplots(len(states), len(lats), figsize=(plots.PAGE, 1.35 * len(states) + 0.3), sharex=True, sharey=True, squeeze=False)
    col = {"cinque": plots.OKABE_ITO[5], "lebowski": plots.OKABE_ITO[6]}
    for i, st in enumerate(states):
        for j, lat in enumerate(lats):
            ax = axes[i, j]
            for m in MODELS:
                for ps, ls in (("stand", "-"), ("cross", "--")):
                    g = R[(R.model == m) & (R.state == st) & (R.lat == lat) & (R.ped_state == ps)].sort_values("dist")
                    ax.plot(g.dist, g.react, ls, color=col[m], marker="o", label=f"{m.capitalize()} react, {ps}")
                    ax.plot(g.dist, g.stop, ls, color=col[m], marker="x", alpha=0.6, label=f"{m.capitalize()} stop, {ps}")
                nf = R[(R.model == m) & (R.state == st)].null_flip.mean()
                ax.axhline(nf, color=col[m], lw=0.5, ls=":")
            ax.set_xscale("log")
            ax.set_xticks([3, 5, 8, 12, 20, 30])
            ax.set_xticklabels(["3", "5", "8", "12", "20", "30"])
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title({0.0: "lane centre", 1.5: "lane edge", 3.0: "kerb", 5.0: "sidewalk"}.get(lat, f"{lat} m"))
            if j == 0:
                ax.set_ylabel(f"{st}\nrate")
            if i == len(states) - 1:
                ax.set_xlabel("distance ahead of bumper (m)")
    plots.legend_below(fig, axes[0, 0], ncol=4)
    plots.save(fig, out, "ped-dose-response")
    return R


LAT_NAME = {0.0: "lane centre (0 m)", 1.5: "lane edge (1.5 m)", 3.0: "kerb (3 m)", 5.0: "sidewalk (5 m)"}


def delta_fig(w: pd.DataFrame, out: Path):
    """Paired differences x+ - x- at t*, per ego state: native-plan speed at 2 s (top) and P(lead) (bottom) against
    distance, one column per lateral position; median over scenes, IQR band when a cell has several scenes.
    The dashed grey line is -tau (Cinque's; Lebowski's is drawn only when it differs by > 0.05 m/s)."""
    from . import plots
    import matplotlib.pyplot as plt
    plt.rcParams.update(plots.STYLE)
    w0 = w[w.t_rel_f0 == 0.0]
    taus = pd.read_csv(data_dir() / I3_EXAM / "flip_rates.csv").query("scope == 'pooled'").set_index("examinee").tau_model
    col = {"cinque": plots.OKABE_ITO[5], "lebowski": plots.OKABE_ITO[6]}
    lats = sorted(w0.lat.unique())
    for st, ws in w0.groupby("state"):
        fig, axes = plt.subplots(2, len(lats), figsize=(plots.PAGE, 3.0), sharex=True, sharey="row", squeeze=False)
        for j, lat in enumerate(lats):
            for m in MODELS:
                ws_ = ws.assign(dlead=ws[f"{m}|lead_p_plus"] - ws[f"{m}|lead_p_minus"])
                for ps, ls, mk in (("stand", "-", "o"), ("cross", "--", "s")):
                    g = ws_[(ws_.lat == lat) & (ws_.ped_state == ps)].groupby("dist")
                    for i, v in enumerate((f"{m}|dv2", "dlead")):
                        q = g[v].quantile([0.25, 0.5, 0.75]).unstack()
                        if q.empty:
                            continue
                        ax = axes[i, j]
                        ax.plot(q.index, q[0.5], ls, color=col[m], marker=mk, mfc="white" if ps == "cross" else col[m],
                                label=f"{m.capitalize()}, {'standing' if ps == 'stand' else 'crossing'}")
                        if (g.size() > 1).any():
                            ax.fill_between(q.index, q[0.25], q[0.75], color=col[m], alpha=0.12, lw=0)
            for i in range(2):
                ax = axes[i, j]
                ax.axhline(0, color="0.6", lw=0.5)
                ax.set_xscale("log")
                ax.set_xticks(list(DISTS))
                ax.set_xticklabels([str(d) for d in DISTS])
                ax.minorticks_off()
            tc = float(taus["ridge_late op-cinque temporal"])
            axes[0, j].axhline(-tc, color="0.4", lw=0.6, ls=":")
            axes[1, j].axhline(0.3, color="0.4", lw=0.6, ls=":")
            axes[0, j].set_title(LAT_NAME.get(lat, f"{lat} m"))
        fig.supxlabel("distance ahead of the bumper at $t^*$ (m)", y=-0.03, fontsize=8)
        axes[0, 0].set_ylabel("$\\Delta$ plan speed at 2 s (m/s)")
        axes[1, 0].set_ylabel("$\\Delta$ P(lead)")
        hl = {lb: h for ax in axes.ravel() for h, lb in zip(*ax.get_legend_handles_labels())}
        fig.legend(hl.values(), hl.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, columnspacing=1.4, handlelength=2.2)
        plots.save(fig, out, f"ped-dose-delta-{st}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replot", action="store_true", help="figures only, from <out>/cells.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.replot:
        w = pd.read_csv(a.out / "cells.csv")
    else:
        w = cells_table(a.exam_dir)
        w.to_csv(a.out / "cells.csv", index=False)
    delta_fig(w, a.out)
    T = threat_summary(w, a.out)
    log.info("threat:\n%s", T.round(3).to_string())
    R = curves(w, a.out)
    log.info("curves:\n%s", R.round(3).to_string())


if __name__ == "__main__":
    main()
