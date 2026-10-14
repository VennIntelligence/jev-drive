#!/usr/bin/env python3
"""SLOW-NOVELTY (plans/2026-10-14-slow-novelty-prereg.md): does the plan's residual shortness track the novelty of the state (off the log path,
heading off the log, inside a junction) once the speed-history ratchet is held (JEV_VHIST=hold) or controlled for? CPU only, read-only on stored
AlpaSim reads. Box, AlpaSim's venv.

  rf_slownov.py extract --runs LABEL=<run dir>[,<run dir>..] ... --out DIR [--workers N] [--limit N]
      -> DIR/decisions.parquet: one row per label x scene x decision (every decision of the last driver session; `kept` = rf_deficit's rule)
  rf_slownov.py analyse --in DIR --out DIR --figs DIR [--primary L] [--plain L] [--extra L ..]
      -> DIR/slownov.json, slownov.md, figs slow_novelty_bins.png, slow_novelty_onset.png

Per decision: Y1 = rf_deficit's plan 2 s deficit (rf_follow.plan_arc2(poses) - 2 x measured speed, kept when speed >= 1 m/s and k <= k_z);
log path = the scene's rig trajectory (the anchors' frame); s = the anchor projected on it (local window as rf_follow); lat / head = |offset| /
|yaw - path tangent| there; v_log(s), logdef(s) = the log's speed and own 2 s deficit at its first arrival at s; Y2 = speed / v_log(s);
junc = rf_follow's route-turn stretch (rf_follow.one, cover none; NaN when the route rebuild fails); fed row = vhist.ego_fed when logged, else ego:
vfed, dvfed = vfed - its 1.5 s history mean speed, held = vfed > measured; tstop / stopped from the measured speeds of the session's decisions.
Statistics: OLS on winsorised (1 / 99 %), z-scored variables, delta R^2 of the novelty block over the base block, log-clustered bootstrap (2000).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_follow as RF  # noqa: E402  (sets the repo paths)

T0, DT = RF.T0, RF.DT
STOP_V, N_BOOT = 0.5, 2000
NOV = ["lat", "head", "junc"]


# ---------------------------------------------------------------- extract
def log_path(ds):
    """Rig trajectory -> (polyline (n, 2) without repeats, its stations, first arrival time (s) at each, S(t) on the raw samples, timestamps s)."""
    from ao_e1 import stations
    tr = ds.rig.trajectory
    P, T = np.asarray(tr.positions, float)[:, :2], np.asarray(tr.timestamps_us, float) / 1e6
    St = stations(P)
    keep = np.r_[True, np.diff(St) > 1e-6]
    return P[keep], St[keep], T[keep], St, T


def one(job):
    """(label, run dir, scene, drive records, cfg) -> decision rows."""
    label, D, scene, recs, cfg = job
    import long_read as LR
    from ao_e1 import loader, project
    ev = LR.one((label, D, scene))
    kz = min([r["k"] for r in ev if r["atf"] or (r["dist_gt"] or 0) >= 4.0], default=10 ** 6)
    try:
        junc = {d["k"]: float(d["junction"]) for d in RF.one((label, D, scene, recs, {}, cfg, "none"))[1]}
    except Exception:  # noqa: BLE001  (no route rebuild: AlpaSim fails the same scenes; junc NaN)
        junc = {}
    L, SL, Tm, St, T = log_path(loader(str(cfg)).get_data_source(scene))
    S_of_t = lambda t: np.interp(t, T, St)                                          # noqa: E731

    def vlog_t(t):                                                                  # central difference, one-sided at the trajectory ends
        a, b = max(t - 0.25, T[0]), min(t + 0.25, T[-1])
        return (S_of_t(b) - S_of_t(a)) / max(b - a, 1e-3)
    at = lambda q: np.c_[np.interp(q, SL, L[:, 0]), np.interp(q, SL, L[:, 1])]     # noqa: E731
    rows, hint, last_stop = [], None, None
    for r in recs:
        k = int((r["t0"] - T0) // DT)
        e = np.asarray(r["ego"], float)
        v = float(np.hypot(e[4], e[5]) * 10.0)
        a = np.asarray(r["anchor"], float)
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        s, lat = project(L, SL, a[:2], lo, hi)
        hint = s
        d = at(min(s + 2.0, SL[-1])) - at(max(s - 2.0, 0.0))
        head = float(np.degrees(abs((a[2] - np.arctan2(d[0, 1], d[0, 0]) + np.pi) % (2 * np.pi) - np.pi)))
        tl = float(np.interp(s, SL, Tm))
        vl = float(vlog_t(tl))
        logdef = float(S_of_t(tl + 2.0) - S_of_t(tl) - 2.0 * vl) if tl + 2.0 <= T[-1] and s < SL[-1] - 0.5 else np.nan
        vh = r.get("vhist") or {}
        f = np.asarray(vh.get("ego_fed", r["ego"]), float)
        vfed = float(np.hypot(f[4], f[5]) * 10.0)
        H = f[8:20].reshape(4, 3)[:, :2] * 10.0
        vhm = float(np.hypot(*np.diff(H, axis=0).T).sum() / 1.5)
        rows.append(dict(label=label, scene=scene, log=scene.rsplit("-", 1)[0], k=k, v=v, kept=bool(v >= 1.0 and k <= kz),
                         y1=RF.plan_arc2(r["poses"]) - 2.0 * v, vlog=vl, y2=v / vl if vl >= 1.0 else np.nan, logdef=logdef,
                         lat=abs(float(lat)), lat_signed=float(lat), head=head, junc=junc.get(k, np.nan), s=s, s_end=float(SL[-1]),
                         vfed=vfed, dvfed=vfed - vhm, held=float(vfed > v + 1e-3), release=vh.get("release", ""),
                         stopped=float(last_stop is not None), tstop=0.5 * (k - (last_stop if last_stop is not None else 0)), kz=kz))
        if v < STOP_V:
            last_stop = k
    return rows


def extract(a):
    import ev as EV
    import pandas as pd
    from ao_e1 import drive_records
    from jevdrive import cache, par
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "slownov", config=vars(a)) as run:
        jobs = []
        for spec in a.runs:
            label, dirs = spec.split("=", 1)
            seen = set()
            for D in dirs.split(","):
                for sc, rr in sorted(drive_records(D).items()):
                    if sc in seen or (a.limit and len(seen) >= a.limit):
                        continue
                    seen.add(sc)
                    jobs.append((label, D, sc, [{k: r[k] for k in ("k", "t0", "ego", "anchor", "poses", "route0", "vhist") if k in r} for r in rr], EV.CFG))
            run.info(f"{label}: {len(seen)} rollouts")
        key = cache.key(params=dict(runs=a.runs, limit=a.limit), code=[one, log_path, RF.one, RF.plan_arc2])

        def go():
            res = par.pmap(one, jobs, run=run, workers=min(a.workers, len(jobs)), desc="rollouts")
            if res.errors:
                run.info(f"{len(res.errors)} rollouts failed: {[(jobs[i][0], jobs[i][2], str(e)[-300:]) for i, e in list(res.errors.items())[:3]]}")
            return pd.DataFrame([r for v in res.values if v for r in v])
        X = cache.cached(out / "decisions.parquet", key, go, force=a.force)
        g = X[X.kept].groupby("label")
        run.info(g.agg(rollouts=("scene", "nunique"), decisions=("k", "size"), y1_median=("y1", "median"), junc_na=("junc", lambda x: x.isna().mean()),
                       logdef_na=("logdef", lambda x: x.isna().mean())).to_string())
        run.summary.update(rows=int(len(X)))


# ---------------------------------------------------------------- analyse
def zprep(df, cols):
    """Winsorise (1 / 99 %) and z-score the columns (constant column -> 0)."""
    Z = {}
    for c in cols:
        x = df[c].to_numpy(float)
        lo, hi = np.nanquantile(x, [0.01, 0.99])
        x = np.clip(x, lo, hi)
        sd = x.std()
        Z[c] = (x - x.mean()) / sd if sd > 0 else x * 0.0
    return Z


def ols_boot(df, y, base, nov, fe=False):
    """OLS y ~ base (+ nov) on z-scored variables; log-clustered bootstrap of the nov coefficients and delta R^2 via per-log sufficient stats."""
    cols = [y] + base + nov
    d = df.dropna(subset=cols).reset_index(drop=True)
    Z = zprep(d, cols)
    if fe:                                                    # scene fixed effects: demean within the scene, then re-standardise
        g = d.scene.to_numpy()
        for c in cols:
            s = pd_series(Z[c], g)
            Z[c] = Z[c] - s.groupby(g).transform("mean").to_numpy()
            sd = Z[c].std()
            Z[c] = Z[c] / sd if sd > 0 else Z[c]
    X = np.c_[np.ones(len(d)), np.column_stack([Z[c] for c in base + nov])]
    Y = Z[y]
    logs, inv = np.unique(d.log.to_numpy(), return_inverse=True)
    nL, p, nb = len(logs), X.shape[1], 1 + len(base)
    G = np.zeros((nL, p, p))
    h, q, sy, n = np.zeros((nL, p)), np.zeros(nL), np.zeros(nL), np.zeros(nL)
    np.add.at(G, inv, X[:, :, None] * X[:, None, :])
    np.add.at(h, inv, X * Y[:, None])
    np.add.at(q, inv, Y * Y)
    np.add.at(sy, inv, Y)
    np.add.at(n, inv, 1.0)

    def fit(w):
        Gw, hw, qw, syw, nw = np.tensordot(w, G, 1), w @ h, w @ q, w @ sy, w @ n
        tss = qw - syw ** 2 / nw
        bf = np.linalg.lstsq(Gw, hw, rcond=None)[0]
        bb = np.linalg.lstsq(Gw[:nb, :nb], hw[:nb], rcond=None)[0]
        r2f, r2b = 1 - (qw - bf @ hw) / tss, 1 - (qw - bb @ hw[:nb]) / tss
        return bf, r2f, r2b
    bf, r2f, r2b = fit(np.ones(nL))
    rng = np.random.default_rng(0)
    B = np.stack([np.bincount(rng.integers(0, nL, nL), minlength=nL).astype(float) for _ in range(N_BOOT)])
    bs = [fit(w) for w in B]
    coefs = np.stack([b[0] for b in bs])
    dr2 = np.array([b[1] - b[2] for b in bs])
    names = base + nov
    out = dict(n=int(len(d)), scenes=int(d.scene.nunique()), logs=int(nL), r2_base=float(r2b), r2_full=float(r2f), dr2=float(r2f - r2b),
               dr2_lo=float(np.quantile(dr2, 0.025)), dr2_hi=float(np.quantile(dr2, 0.975)), coef={})
    for i, c in enumerate(names):
        out["coef"][c] = dict(b=float(bf[1 + i]), lo=float(np.quantile(coefs[:, 1 + i], 0.025)), hi=float(np.quantile(coefs[:, 1 + i], 0.975)))
    return out


def pd_series(x, idx):
    import pandas as pd
    return pd.Series(x, index=idx)


def onset(R):
    """Analysis 3 on one label's kept decisions: threshold, onset contrasts (i) within-rollout, (ii) vs never-slow at the same k."""
    from jevdrive import stats
    R = R.sort_values(["scene", "k"]).copy()
    R["x"] = R.y1 - R.logdef
    thr = float(np.nanquantile(R.x, 0.10))
    slow = R[R.x <= thr].groupby("scene").k.min()
    never = R[~R.scene.isin(slow.index)]
    VARS = NOV + ["dvfed", "vfed", "x"]
    by_k = never.groupby("k")[VARS].mean()
    rows, aligned = [], []
    for sc, k0 in slow.items():
        x = R[R.scene == sc]
        pre, at = x[x.k < k0], x[x.k == k0].iloc[0]
        for _, r in x.iterrows():
            if -6 <= r.k - k0 <= 3:
                aligned.append(dict(rel=int(r.k - k0), log=r.log, **{c: r[c] for c in VARS}))
        if len(pre) < 2:
            continue
        rows.append(dict(scene=sc, log=at.log, k0=int(k0), **{f"w_{c}": at[c] - pre[c].mean() for c in VARS},
                         **{f"m_{c}": (at[c] - by_k.loc[k0, c]) if k0 in by_k.index else np.nan for c in VARS}))
    import pandas as pd
    O = pd.DataFrame(rows)
    res = dict(thr=thr, rollouts=int(R.scene.nunique()), onset=int(len(slow)), onset_used=int(len(O)), never=int(never.scene.nunique()),
               onset_k_median=float(slow.median()) if len(slow) else np.nan, contrasts={})
    for c in VARS:
        res["contrasts"][c] = dict(within=stats.bootstrap(O[f"w_{c}"], O.log, n_boot=N_BOOT), vs_never=stats.bootstrap(O[f"m_{c}"], O.log, n_boot=N_BOOT))
    A = pd.DataFrame(aligned)
    never_mean = {c: stats.bootstrap(never[c], never.log, n_boot=N_BOOT) for c in VARS}
    return res, A, never_mean


def analyse(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from jevdrive import stats
    from jevdrive.run import Run
    out, figs = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    X = pd.read_parquet(Path(a.inp) / "decisions.parquet")
    K = X[X.kept].copy()
    with Run("route_follow", "slownov-analyse", config=vars(a)) as run:
        RAT = ["vfed", "dvfed"]
        B0 = ["logdef", "vlog", "tstop", "stopped", "k"]
        R = dict(labels={}, models={}, onset=None)
        for lab, x in K.groupby("label"):
            R["labels"][lab] = dict(rollouts=int(x.scene.nunique()), decisions=int(len(x)), y1_median=float(x.y1.median()),
                                    y2_median=float(x.y2.median()), junc_na=float(x.junc.isna().mean()), logdef_na=float(x.logdef.isna().mean()),
                                    held=float(x.held.mean()), release_turn=float((x.release == "turn").mean()),
                                    lat_median=float(x.lat.median()), lat_p90=float(x.lat.quantile(0.9)), head_median=float(x["head"].median()),
                                    head_p90=float(x["head"].quantile(0.9)), junc=float(x.junc.mean()))
        hold = lambda lab: ["held"] if K[K.label == lab].held.std() > 0 else []  # noqa: E731
        specs = [("1 primary", a.primary, "y1", B0 + RAT + hold(a.primary), False),
                 ("1b no ratchet controls (brief)", a.primary, "y1", B0, False),
                 ("1c within-scene", a.primary, "y1", B0 + RAT + hold(a.primary), True),
                 ("1d speed ratio Y2", a.primary, "y2", B0 + hold(a.primary), False),
                 ("2 plain read, ratchet in", a.plain, "y1", B0 + RAT, False),
                 ("2b plain, no ratchet controls", a.plain, "y1", B0, False),
                 ("2c plain within-scene", a.plain, "y1", B0 + RAT, True),
                 ("2d plain speed ratio Y2", a.plain, "y2", B0, False)]
        specs += [(f"1r replication {e}", e, "y1", B0 + RAT + hold(e), False) for e in a.extra or []]
        K["x"] = K.y1 - K.logdef                              # post-hoc (not registered): the log's own deficit as a fixed offset, slope 1
        specs += [("1e post-hoc: excess deficit Y1 - logdef", a.primary, "x", B0[1:] + RAT + hold(a.primary), False),
                  ("2e post-hoc: excess deficit, plain", a.plain, "x", B0[1:] + RAT, False)]
        for name, lab, y, base, fe in specs:
            m = ols_boot(K[K.label == lab], y, base, NOV, fe=fe)
            R["models"][name] = dict(label=lab, y=y, base=base, fe=fe, **m)
            run.info(f"{name}: dR2 {m['dr2']:.4f} [{m['dr2_lo']:.4f}, {m['dr2_hi']:.4f}] " + " ".join(f"{c} {m['coef'][c]['b']:+.3f}" for c in NOV))
        P = R["models"]["1 primary"]
        clear = [c for c in NOV if P["coef"][c]["hi"] < 0]
        opposite = [c for c in NOV if P["coef"][c]["lo"] > 0]
        verdict = ("novelty carries" if P["dr2"] >= 0.05 and clear else
                   "pure input loop (novelty a minor term)" if clear else
                   "pure input loop (refuted outright)" if P["dr2"] < 0.02 else "pure input loop")
        R["verdict"] = dict(verdict=verdict, clear_slower=clear, clear_opposite=opposite)
        on, A, never_mean = onset(K[K.label == a.plain])
        R["onset"] = on
        sup = [c for c in NOV if (on["contrasts"][c]["within"]["lo"] > 0 and on["contrasts"][c]["vs_never"]["lo"] > 0)]
        R["onset"]["supported_by"] = sup
        (out / "slownov.json").write_text(json.dumps(R, indent=1, default=float) + "\n")

        # tables
        f = lambda r: f"{r['b']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"  # noqa: E731
        md = ["# SLOW-NOVELTY tables (rf_slownov.py analyse)", "",
              "| label | rollouts | kept decisions | Y1 median (m) | Y2 median | lat median / p90 (m) | head median / p90 (deg) | junction share | held | released for a turn | junc / logdef missing |",
              "|:--|--:|--:|--:|--:|:--|:--|--:|--:|--:|:--|"]
        for lab, r in R["labels"].items():
            md.append(f"| {lab} | {r['rollouts']} | {r['decisions']} | {r['y1_median']:+.3f} | {r['y2_median']:.3f} | {r['lat_median']:.2f} / {r['lat_p90']:.2f} | "
                      f"{r['head_median']:.1f} / {r['head_p90']:.1f} | {r['junc']:.3f} | {r['held']:.3f} | {r['release_turn']:.3f} | {r['junc_na']:.3f} / {r['logdef_na']:.3f} |")
        md += ["", "Standardised OLS coefficients (outcome and covariates winsorised 1 / 99 %, z-scored within the model's rows), 95 % log-clustered "
               f"bootstrap ({N_BOOT} draws). Negative = more of it -> shorter plan (Y1) / slower than the log (Y2).", "",
               "| model | label | Y | n (scenes / logs) | R^2 base | delta R^2 novelty [CI] | lat | head | junc |", "|:--|:--|:--|:--|--:|:--|:--|:--|:--|"]
        for name, m in R["models"].items():
            md.append(f"| {name} | {m['label']} | {m['y']} | {m['n']} ({m['scenes']} / {m['logs']}) | {m['r2_base']:.3f} | {m['dr2']:+.4f} [{m['dr2_lo']:+.4f}, {m['dr2_hi']:+.4f}] | "
                      + " | ".join(f(m["coef"][c]) for c in NOV) + " |")
        md += ["", "Base-block coefficients:", "", "| model | " + " | ".join(B0 + RAT + ["held"]) + " |", "|:--|" + "--:|" * len(B0 + RAT + ["held"])]
        for name, m in R["models"].items():
            md.append(f"| {name} | " + " | ".join(f"{m['coef'][c]['b']:+.3f}" + ("*" if m["coef"][c]["lo"] > 0 or m["coef"][c]["hi"] < 0 else "")
                                                  if c in m["coef"] else "-" for c in B0 + RAT + ["held"]) + " |")
        md += ["", "(* = 95 % CI clear of 0)", "", f"**Verdict (registered rule, primary model): {verdict}**; novelty coefficients clear of 0 in the slower direction: "
               f"{clear or 'none'}; in the opposite direction: {opposite or 'none'}.", "",
               f"Onset ({a.plain}): excess deficit x = Y1 - logdef, threshold = 10th percentile = {on['thr']:+.3f} m; {on['onset']} of {on['rollouts']} rollouts "
               f"reach it (median onset k {on['onset_k_median']:.0f}), {on['onset_used']} with >= 2 kept decisions before it; {on['never']} never.", "",
               "| variable | onset - preceding (same rollout) [CI] | onset - never-slow at the same k [CI] | never-slow mean |", "|:--|:--|:--|:--|"]
        for c in NOV + ["dvfed", "vfed", "x"]:
            w, m = on["contrasts"][c]["within"], on["contrasts"][c]["vs_never"]
            md.append(f"| {c} | {stats.fmt(w, '+.3f')} (n {w['n']}) | {stats.fmt(m, '+.3f')} (n {m['n']}) | {never_mean[c]['mean']:.3f} |")
        md += ["", f"Onset supported (registered: (i) and (ii) both clear of 0 in the more-novel direction) by: {sup or 'none'}."]
        (out / "slownov.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))

        # figure 1: binned outcome vs novelty
        labs = [a.plain, a.primary]
        cols = {a.plain: "#c0392b", a.primary: "#2471a3"}
        fig, ax = plt.subplots(2, 3, figsize=(10, 5.6))
        for i, y in enumerate(["y1", "y2"]):
            for j, c in enumerate(NOV):
                A_ = ax[i, j]
                if c == "junc":
                    edges = None
                else:
                    edges = np.unique(np.nanquantile(K[K.label.isin(labs)][c], np.linspace(0, 1, 11)))
                for li, lab in enumerate(labs):
                    x = K[(K.label == lab)].dropna(subset=[c, y])
                    b = x[c].to_numpy() if edges is None else np.clip(np.searchsorted(edges, x[c].to_numpy(), side="right") - 1, 0, len(edges) - 2)
                    pts = []
                    for u in np.unique(b):
                        sel = b == u
                        r = stats.bootstrap(x[y].to_numpy()[sel], x.log.to_numpy()[sel], n_boot=1000)
                        xc = float(u) if edges is None else float(np.median(x[c].to_numpy()[sel]))
                        pts.append((xc + (0.04 * (li - 0.5) if edges is None else 0), r["mean"], r["lo"], r["hi"]))
                    pts = np.array(pts)
                    A_.errorbar(pts[:, 0], pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]], color=cols[lab], marker="o", ms=3,
                                lw=1, capsize=2, label=lab)
                if c in ("lat", "head"):
                    A_.set_xscale("symlog", linthresh=0.1 if c == "lat" else 1.0)
                A_.set_xlabel({"lat": "|offset from log path| (m)", "head": "|heading - log| (deg)", "junc": "in junction (0 / 1)"}[c])
                A_.set_ylabel("plan 2 s deficit Y1 (m)" if y == "y1" else "speed / log speed Y2")
                A_.grid(alpha=0.3)
        ax[0, 0].legend(fontsize=8)
        fig.suptitle("Slowing vs state novelty (held 460, RSG-L18-s0): decile means, log-clustered 95 % CI", fontsize=10)
        fig.tight_layout()
        fig.savefig(figs / "slow_novelty_bins.png", dpi=110)
        plt.close(fig)

        # figure 2: onset-aligned means
        fig, ax = plt.subplots(1, 5, figsize=(13, 2.8))
        for j, c in enumerate(["x", "dvfed", "lat", "head", "junc"]):
            pts = []
            for rel, g in A.groupby("rel"):
                r = stats.bootstrap(g[c].to_numpy(float), g.log.to_numpy(), n_boot=1000)
                pts.append((rel, r["mean"], r["lo"], r["hi"]))
            pts = np.array(pts)
            ax[j].fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color="#c0392b", alpha=0.2)
            ax[j].plot(pts[:, 0], pts[:, 1], color="#c0392b", marker="o", ms=3, lw=1, label="onset rollouts")
            ax[j].axhline(never_mean[c]["mean"], color="#555", ls="--", lw=1, label="never-slow mean")
            ax[j].axvline(0, color="k", lw=0.6)
            ax[j].set_title({"x": "excess deficit (m)", "dvfed": "fed speed trend (m/s)", "lat": "|offset| (m)", "head": "|heading err| (deg)",
                             "junc": "junction share"}[c], fontsize=9)
            ax[j].set_xlabel("decision relative to onset")
            ax[j].grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / "slow_novelty_onset.png", dpi=110)
        plt.close(fig)
        run.summary.update(verdict=verdict, dr2=P["dr2"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=12)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--force", action="store_true")
    n = sub.add_parser("analyse")
    n.add_argument("--in", dest="inp", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--figs", required=True)
    n.add_argument("--primary", default="RSG+hold")
    n.add_argument("--plain", default="RSG")
    n.add_argument("--extra", nargs="*", default=["RSG+holdlead"])
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else analyse(a)
