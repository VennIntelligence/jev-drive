#!/usr/bin/env python
"""Tables and figures of the HUGSIM zero-shot exam (todos/2026-09-25-hugsim-exam). Runs on the Mac (.venv) on files
pulled from the box into research/results/hugsim-exam/.

  rate     4 Hz handicap: rate_op.jsonl (comma1M) and rate_alp.jsonl (nuScenes) -> rate_*.csv, fig hugsim-exam-rate
  inputs   model inputs vs the simulator's cameras, from zs_dump npz files -> fig hugsim-exam-inputs
  check    checklist runs: driven path vs route, speed, plan origin -> checklist.csv, fig hugsim-exam-checklist
  results  scored runs: HD-Score and sub-scores with CIs -> scores*.csv, figs hugsim-exam-scores / -ends
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jevdrive import plots as P  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "research" / "results" / "hugsim-exam"
FIG = REPO / "research" / "figs"
MODEL_COLOR = {"alpamayo": "#0072B2", "cinque": "#D55E00", "lebowski": "#E69F00", "ltf": "#7F7F7F",
               "cv": "#BBBBBB", "route": "#009E73"}
MODEL_NAME = {"alpamayo": "Alpamayo 1.5", "cinque": "openpilot Cinque", "lebowski": "openpilot Lebowski",
              "ltf": "LTF (official client)", "cv": "constant velocity", "route": "route (privileged)"}


def boot_ci(x, groups, B=5000, seed=0, stat=np.mean):
    """Percentile CI of stat(x) resampling whole groups (clusters)."""
    x, groups = np.asarray(x, float), np.asarray(groups)
    u, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(inv == k) for k in range(len(u))]
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        vals.append(stat(x[np.concatenate([idx[k] for k in pick])]))
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- rate
def cmd_rate(a):
    op = pd.read_json(RES / "rate_op.jsonl", lines=True)
    op["block"] = op.seg.str[:8] + "_" + (op.i // 100).astype(str)
    rows = []
    for m, g in op.groupby("model"):
        base = g[g.variant == "native"].set_index(["seg", "i"])
        for v, q in g.groupby("variant"):
            q = q.set_index(["seg", "i"])
            common = q.index.intersection(base.index)
            q, b = q.loc[common], base.loc[common]
            r = {"model": m, "variant": v, "n": len(common)}
            for h in ("1", "2", "4"):
                for ax in ("lat", "lon"):
                    e, e0 = q[f"d{ax}@{h}"].abs(), b[f"d{ax}@{h}"].abs()
                    ok = e.notna() & e0.notna()
                    r[f"{ax}@{h}s"] = e[ok].mean()
                    r[f"{ax}@{h}s_rel"] = e[ok].mean() / e0[ok].mean() - 1
                    if h == "2":
                        lo, hi = boot_ci((e - e0)[ok].values, q.loc[ok, "block"].values)
                        r[f"{ax}@2s_diff_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
            fast = (q.speed > 3) & q["dlon@2"].notna()
            r["lon_ratio@2s"] = float(np.median((q.loc[fast, "gtlon@2"] + q.loc[fast, "dlon@2"]) / q.loc[fast, "gtlon@2"]))
            rows.append(r)
    op_t = pd.DataFrame(rows)
    op_t.to_csv(RES / "rate_op.csv", index=False, float_format="%.4f")
    print(op_t.to_string())

    alp = pd.read_json(RES / "rate_alp.jsonl", lines=True)
    rows = []
    base = alp[alp.variant == "native"].set_index("token")
    for v, q in alp.groupby("variant", sort=False):
        q = q.set_index("token").loc[base.index]
        r = {"model": "alpamayo", "variant": v, "n": len(q)}
        for h in ("1s", "2s", "3s"):
            r[f"l2_{h}"] = q[f"l2_{h}"].mean()
            d = q[f"l2_{h}"] - base[f"l2_{h}"]
            lo, hi = boot_ci(d.values, q.scene.values)
            r[f"l2_{h}_diff"], r[f"l2_{h}_diff_ci"] = d.mean(), f"[{lo:+.3f}, {hi:+.3f}]"
        rows.append(r)
    alp_t = pd.DataFrame(rows)
    alp_t.to_csv(RES / "rate_alp.csv", index=False, float_format="%.4f")
    print(alp_t.to_string())

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    with mpl.rc_context(P.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(P.PAGE, 1.9))
        names = {"native": "native", "ctx5-hold": "5 Hz hold", "h4-dilate": "4 Hz, dilated\n(adapter)",
                 "h4-hold": "4 Hz hold,\nreal time", "h4-spread": "4 Hz spread"}
        for ax, col, lab in ((axes[0], "lat@2s", "lateral error at 2 s (m)"),
                             (axes[1], "lon@2s", "longitudinal error at 2 s (m)")):
            vs = ["native", "ctx5-hold", "h4-dilate", "h4-hold"]
            x = np.arange(len(vs))
            for k, m in enumerate(["cinque", "lebowski"]):
                t = op_t[op_t.model == m].set_index("variant").reindex(vs)
                ax.bar(x + (k - .5) * .38, t[col], .36, color=MODEL_COLOR[m], label=MODEL_NAME[m])
            ax.set_xticks(x, [names[v] for v in vs])
            ax.set_ylabel(lab)
            ax.grid(True, axis="y")
        vs = ["native", "h4-hold", "h4-spread"]
        t = alp_t.set_index("variant").reindex(vs)
        x = np.arange(len(vs))
        axes[2].bar(x, t["l2_3s"], .5, color=MODEL_COLOR["alpamayo"], label=MODEL_NAME["alpamayo"])
        axes[2].set_xticks(x, ["native\n10 Hz", "4 Hz nearest\n(adapter)", "4 Hz spread"])
        axes[2].set_ylabel("L2 up to 3 s (m)")
        axes[2].grid(True, axis="y")
        h1, l1 = axes[0].get_legend_handles_labels()
        h2, l2 = axes[2].get_legend_handles_labels()
        fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
        fig.tight_layout()
        P.save(fig, FIG, "hugsim-exam-rate")


# ---------------------------------------------------------------- checklist
def ego_frame(pos, th, k, pts):
    """World ground points -> (right, forward) in the ego frame of step k (theta + right, forward = (sin, cos))."""
    d = np.asarray(pts, float) - pos[k]
    return np.stack([d @ np.array([np.cos(th[k]), -np.sin(th[k])]), d @ np.array([np.sin(th[k]), np.cos(th[k])])], -1)


def shadow_errors(tr):
    """Model plan (logged in shadow mode) vs the driven future: per step and horizon 1 / 2 / 3 s, (lat, lon) error
    and the true displacement, in the ego frame at the planning step."""
    pos, th = np.asarray(tr["pos"]), np.asarray(tr["theta"])
    out = []
    for st in tr["steps"]:
        k, mp = st.get("step", None), st.get("model_plan")
        k = tr["steps"].index(st) if k is None else k
        if mp is None:
            continue
        mp = np.asarray(mp)
        for h, j in ((1, 1), (2, 3), (3, 5)):
            if k + 2 * (j + 1) >= len(pos):
                continue
            g = ego_frame(pos, th, k, pos[k + 2 * (j + 1)][None])[0]
            out.append({"step": k, "h": h, "lat": mp[j, 0] - g[0], "lon": mp[j, 1] - g[1], "g_lat": g[0],
                        "g_lon": g[1], "p_lat": mp[j, 0], "v": st["v"]})
    return pd.DataFrame(out)


def run_row(tr):
    st = tr["steps"]
    v = np.array([x["v"] for x in st]) if st else np.zeros(0)
    t = np.array([x["t"] for x in st]) if st else np.zeros(0)
    lat = np.abs(np.asarray(tr["lat"]))
    th = np.asarray(tr["theta"])
    moving = np.flatnonzero(v > 2.0)
    r = {"tag": tr["tag"], "scenario": tr["scenario"], "end": tr["end"], "hd": tr["hdscore"], "rc": tr["rc"],
         "steps": len(st), "v10": float(v[min(40, len(v) - 1)]) if len(v) else np.nan,
         "t_move": float(t[moving[0]]) if len(moving) else np.nan, "v_min": float(v.min()) if len(v) else np.nan,
         "clamped": float(np.mean(["raw_plan" in x for x in st])) if st else np.nan,
         "lat_med": float(np.median(lat)) if len(lat) else np.nan, "lat_max": float(lat.max()) if len(lat) else np.nan,
         "dheading_deg": float(np.degrees(th[-1] - th[0])) if len(th) else np.nan,
         "route_frac": float(max(tr["s"]) / tr["route_len"]) if tr["s"] else np.nan}
    plans = [np.asarray(x["plan"]) for x in st if "plan" in x and not x.get("oracle")]
    if plans:
        r["plan0_len"] = float(plans[0][-1, 1])
        r["plan_x3s_early"] = float(np.mean([p[-1, 0] for p in plans[1:9]])) if len(plans) > 1 else np.nan
    return r


def cmd_check(a):
    trs = json.load(open(a.traces))
    rows = pd.DataFrame([run_row(t) for t in trs])
    rows.to_csv(RES / "checklist_runs.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(rows.to_string())
    sh = []
    for t in trs:
        if "shadow" in t["tag"]:
            e = shadow_errors(t)
            if len(e):
                e["tag"], e["scenario"] = t["tag"], t["scenario"]
                sh.append(e)
    if sh:
        e = pd.concat(sh)
        e = e[e.v > 1.0]
        g = e.groupby(["tag", "h"])
        tab = pd.DataFrame({"n": g.size(), "abs_lat": g.lat.apply(lambda x: x.abs().mean()),
                            "bias_lat": g.lat.mean(), "abs_lon": g.lon.apply(lambda x: x.abs().mean()),
                            "bias_lon": g.lon.mean(),
                            "cv_abs_lat": g.g_lat.apply(lambda x: x.abs().mean())})
        turn = e[(e.h == 3) & (e.g_lat.abs() > 1.0)]
        sign = turn.groupby("tag").apply(lambda q: float(np.mean(np.sign(q.p_lat) == np.sign(q.g_lat))))
        tab = tab.reset_index()
        tab["sign_agree@3s"] = tab.tag.map(sign)
        tab.to_csv(RES / "checklist_shadow.csv", index=False, float_format="%.3f")
        print(tab.to_string())
    fig_check(trs)


def fig_check(trs):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    want = [s for s in ("scene-0071-easy-00", "scene-0383-easy-00", "scene-0920-easy-00", "scene-0062-medium-00",
                        "scene-0071-offset-00", "scene-0071-standstill-00")
            if any(t["scenario"] == s for t in trs)]
    tags = [(m, c) for m in ("cinque", "lebowski", "alpamayo") for c in ("official", "fixed")
            if any(t["tag"] == f"{m}-{c}" for t in trs)]
    if not want or not tags:
        return
    by = {(t["tag"], t["scenario"]): t for t in trs}
    with mpl.rc_context(P.STYLE):
        fig, axes = plt.subplots(1, len(want), figsize=(P.PAGE, 2.3))
        axes = np.atleast_1d(axes)
        for ax, sc in zip(axes, want):
            ref = next(t for t in trs if t["scenario"] == sc)
            r = np.asarray(ref["route"])
            ax.plot(r[:, 0], r[:, 1], color="#BBBBBB", lw=3, label="recorded route")
            for m, c in tags:
                t = by.get((f"{m}-{c}", sc))
                if t is None or not t["pos"]:
                    continue
                p = np.asarray(t["pos"])
                ax.plot(p[:, 0], p[:, 1], color=MODEL_COLOR[m], ls="-" if c == "fixed" else ":", lw=1,
                        label=f"{MODEL_NAME[m]}, {c}")
                ax.plot(p[-1, 0], p[-1, 1], "x", color=MODEL_COLOR[m], ms=4)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(sc.replace("scene-", "").replace("-00", ""), fontsize=7)
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("z (m)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
        fig.tight_layout()
        P.save(fig, FIG, "hugsim-exam-checklist")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rate")
    c = sub.add_parser("check")
    c.add_argument("traces")
    a = ap.parse_args()
    {"rate": cmd_rate, "check": cmd_check}[a.cmd](a)
