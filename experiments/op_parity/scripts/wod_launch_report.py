"""Read-outs of the wod-launch lane (plans/2026-10-08-wod-launch-prereg.md, results/wod_launch.md). jevdrive env, CPU.

  diag             Step 1 tables -> results/wod_launch/: place (e), horizon (f), bias (b), protocol + anticipation (c), targets + probe (d),
                   context (a), verdict.json
  gate --arm --ref pilot gate G-L (stopped d RFS >= thr, moving >= -0.05) -> gate_<ARM>.json; exit 3 when it fails
  full --arm ARM   Step 2 read-outs of ARM-full-s0/s1 against WP2-full, shipped, log -> fix_<ARM>_*.{csv,md}

Conventions (wod_gap.py): 479 rater frames, cluster-mean RFS, paired bootstrap over sequences (B 4 000), WP2 = per-frame mean of the two seeds' scores.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

B = 4000
OUT = _R / "experiments/op_parity/results/wod_launch"
VARS = ("main", "zero", "biasmean", "biasresid", "cmd0", "acc0", "vx1", "vx3", "cv1", "cv3")
V_STOP = 0.5
CTX_ORDER = ("red", "green", "stop_sign", "lead_moving", "lead_stopped", "cross", "open")
CUE_VISIBLE = ("green", "lead_moving", "open")


def ldir():
    return data_dir() / "runs/op_parity/wod/launch"


class Ctx:
    """The 479 rater frames with everything the tables need."""

    def __init__(self):
        from jevdrive import waymo as W
        from jevdrive import wod_zeroshot as Z
        from wod_gap import parts
        self.W, self.Z, self.parts = W, Z, parts
        S = Z.load_sets()
        r, x = S["rater"], S["extra"]
        self.n = n = len(r["name"])
        self.names = np.concatenate([r["name"], x["name"]]).astype(str)
        self.seq = np.concatenate([r["sequence"], x["sequence"]]).astype(str)
        self.fut = np.concatenate([r["future"], x["future"]])[..., :2].astype(np.float64)
        self.traj, self.sc = r["traj"].astype(np.float64), r["scores"].astype(np.float64)
        self.v0 = W.init_speed(r["past"]).astype(np.float64)
        self.intent = r["intent"]
        cl = r["cluster"].astype(str)
        ccode, cu = pd.factorize(pd.Series(cl))
        self.M = np.eye(len(cu))[ccode]
        self.w = 1.0 / (np.bincount(ccode)[ccode] * len(cu))
        scode, su = pd.factorize(pd.Series(self.seq[:n]))
        self.K = np.stack([np.bincount(d, minlength=len(su)) for d in np.random.default_rng(0).integers(len(su), size=(B, len(su)))]).astype(float)[:, scode]
        acode, au = pd.factorize(pd.Series(self.seq))
        self.Ka = np.stack([np.bincount(d, minlength=len(au)) for d in np.random.default_rng(0).integers(len(au), size=(B, len(au)))]).astype(float)[:, acode]
        ii = np.arange(n)
        self.top = self.traj[ii, self.sc.argmax(1)]
        logd = np.linalg.norm(self.fut[:n, -1], axis=-1)
        self.logd = logd
        lum = pd.read_csv(_R / "experiments/leaderboard_audit/results/night_gap/seq_lum.csv").set_index("sequence").l.reindex(self.seq[:n]).to_numpy()
        v0 = self.v0
        self.st = {"all": np.ones(n, bool), "stopped": v0 < V_STOP, "SL (stopped, log moves)": (v0 < V_STOP) & (logd >= 1),
                   "SS (stopped, log stays)": (v0 < V_STOP) & (logd < 1), "launch (v<2, log>5m)": (v0 < 2) & (logd > 5), "moving (v>=0.5)": v0 >= V_STOP,
                   "slow 0.5-5": (v0 >= V_STOP) & (v0 < 5), "mid 5-12": (v0 >= 5) & (v0 < 12), "fast >=12": v0 >= 12, "night": lum < 50, "day": lum >= 120}

    def preds(self, tag, all_frames=False):
        from pp_wod import load_preds
        return load_preds(tag, self.names if all_frames else self.names[: self.n])

    def rfs(self, p):
        return np.asarray(self.W.rater_feedback_score(p[: self.n], self.traj, self.sc, self.v0), float)

    def cm(self, f, rows=None):
        m = self.M if rows is None else self.M * np.asarray(rows, float)[:, None]
        cnt = m.sum(0)
        return float(((m * f[:, None]).sum(0)[cnt > 0] / cnt[cnt > 0]).mean()) if cnt.sum() else np.nan

    def ci(self, d, rows=None):
        """Cluster-mean of the per-frame difference d on rows: point, lo, hi (paired bootstrap over sequences)."""
        m = self.M if rows is None else self.M * np.asarray(rows, float)[:, None]
        cnt, sm = self.K @ m, self.K @ (m * d[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.nanmean(np.where(cnt > 0, sm / cnt, np.nan), 1)
        return (self.cm(d, rows), *np.nanpercentile(b, [2.5, 97.5]))

    def ci_mean(self, d, rows=None):
        """Plain mean of d over rows with a sequence bootstrap (geometry: metres)."""
        m = np.ones(self.n) if rows is None else np.asarray(rows, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (self.K @ (m * d)) / (self.K @ m)
        return (float((m * d).sum() / m.sum()), *np.nanpercentile(b, [2.5, 97.5]))


def f3(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def lab(r, c):
    return "carries" if r >= 0.5 and c[1] > 0 else "part" if r >= 0.25 and c[1] > 0 else "not"


def dist(p, k):
    return np.linalg.norm(p[:, k], axis=-1)


def xcv(p):
    """Waypoints after 4 s replaced by the constant-velocity continuation of the plan's own 3.5 -> 4.0 s motion."""
    q = p.copy()
    v = (p[:, 15] - p[:, 13]) / 0.5
    for k in range(4):
        q[:, 16 + k] = p[:, 15] + v * 0.25 * (k + 1)
    return q


def xca(p):
    """... by the constant-acceleration continuation (velocity over 3.5-4.0 s, acceleration from the 3.0-3.5 s velocity)."""
    q = p.copy()
    v1, v0 = (p[:, 15] - p[:, 13]) / 0.5, (p[:, 13] - p[:, 11]) / 0.5
    a = (v1 - v0) / 0.5
    v4 = v1 + a * 0.25
    for k in range(4):
        t = 0.25 * (k + 1)
        q[:, 16 + k] = p[:, 15] + v4 * t + 0.5 * a * t * t
    return q


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def auc_ci(y, s, g, nb=1000):
    codes, u = pd.factorize(pd.Series(g))
    idx = [np.flatnonzero(codes == k) for k in range(len(u))]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(nb):
        i = np.concatenate([idx[k] for k in rng.integers(len(u), size=len(u))])
        out.append(auc(y[i], s[i]))
    return (auc(y, s), *np.nanpercentile(out, [2.5, 97.5]))


def tau_of(fut, thr=0.5):
    """First future time (s) at which the logged step speed exceeds thr; inf when never within 5 s. fut (n, 20, 2+)."""
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[..., :2].astype(np.float64)], 1)
    sp = np.linalg.norm(np.diff(P, axis=1), axis=-1) * 4
    mv = sp > thr
    return np.where(mv.any(1), (mv.argmax(1) + 1) * 0.25, np.inf)


# ---------------------------------------------------------------- Step 1
def cmd_diag(a):
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "wod-launch-diag", seed=0, config=dict(B=B)) as run:
        run.use_split(splits.load("wod/val"))
        C = Ctx()
        n, st = C.n, C.st
        P = {"shipped": [C.preds("shipped")], "WP2": [C.preds("WP2-full-s0"), C.preds("WP2-full-s1")], "WP1": [C.preds("WP1-full-s0"), C.preds("WP1-full-s1")],
             "WP2-pilot": [C.preds("WP2-pilot-s0")], "log": [C.fut[:n]], "top": [C.top]}
        sco = {k: np.mean([C.rfs(p) for p in v], 0) for k, v in P.items()}
        sh = {k: np.mean([C.parts(p, C.traj, C.sc, C.v0)["sh"] for p in v], 0) for k, v in P.items()}
        d_ = {k: {t: np.mean([dist(p, j) for p in v], 0) for t, j in ((3, 11), (4, 15), (5, 19))} for k, v in P.items()}
        verdict = {}
        stopped = st["stopped"]
        R = C.cm(sco["log"] - sco["WP2"], stopped)
        verdict["R_stopped_log_minus_WP2"] = R
        run.info("reproduce: shipped %.3f WP2 %.3f WP1 %.3f log %.3f top %.3f; R (stopped, log - WP2) %.3f", *(C.cm(sco[k]) for k in ("shipped", "WP2", "WP1", "log", "top")), R)

        # ---- (e) placement
        rows = []
        for nm, m in st.items():
            row = {"stratum": nm, "n": int(m.sum())}
            for k in ("top", "log", "shipped", "WP1", "WP2", "WP2-pilot"):
                row[f"RFS {k}"] = C.cm(sco[k], m)
            for k in ("top", "log", "shipped", "WP1", "WP2"):
                row[f"d3 med {k}"], row[f"d5 med {k}"] = float(np.median(d_[k][3][m])), float(np.median(d_[k][5][m]))
            for k in ("log", "shipped", "WP1", "WP2"):
                row[f"RFS3 {k}"], row[f"RFS5 {k}"] = C.cm(sh[k][:, 0], m), C.cm(sh[k][:, 1], m)
            for x, y in (("log", "WP2"), ("log", "shipped"), ("shipped", "WP1"), ("WP1", "WP2"), ("shipped", "WP2"), ("WP2-pilot", "shipped")):
                c = C.ci(sco[x] - sco[y], m)
                row[f"{x} - {y}"], row[f"{x} - {y} lo"], row[f"{x} - {y} hi"] = c
            c = C.ci_mean(d_["WP2"][5] - d_["log"][5], m)
            row["d5 WP2 - log (mean, m)"], row["d5 WP2 - log lo"], row["d5 WP2 - log hi"] = c
            c = C.ci_mean(d_["WP2"][5] - d_["shipped"][5], m)
            row["d5 WP2 - shipped (mean, m)"], row["d5 WP2 - shipped lo"], row["d5 WP2 - shipped hi"] = c
            rows.append(row)
        pl = stats.write_table(rows, OUT / "place")
        s0 = pl.set_index("stratum").loc["stopped"]
        verdict["e"] = {k: {"d": float(s0[k]), "lo": float(s0[k + " lo"]), "hi": float(s0[k + " hi"]), "share_of_R": float(s0[k] / R)}
                        for k in ("log - shipped", "shipped - WP1", "WP1 - WP2")}
        verdict["pilot_ref"] = {"WP2-pilot - shipped stopped": [float(s0["WP2-pilot - shipped"]), float(s0["WP2-pilot - shipped lo"]), float(s0["WP2-pilot - shipped hi"])]}
        run.info("(e) stopped: %s", json.dumps(verdict["e"]))

        # ---- (f) supervision horizon
        rows, prof = [], []
        for k in ("WP2", "shipped", "WP1", "log", "top"):
            for op, fn in (("xcv", xcv), ("xca", xca)):
                s2 = np.mean([C.rfs(fn(p)) for p in P[k]], 0)
                for nm in ("stopped", "SL (stopped, log moves)", "SS (stopped, log stays)", "launch (v<2, log>5m)", "moving (v>=0.5)", "all"):
                    c = C.ci(s2 - sco[k], st[nm])
                    d5n = np.mean([dist(fn(p), 19) for p in P[k]], 0)
                    rows.append({"arm": k, "op": op, "stratum": nm, "n": int(st[nm].sum()), "RFS": C.cm(sco[k], st[nm]), "RFS op": C.cm(s2, st[nm]), "d": c[0], "lo": c[1],
                                 "hi": c[2], "r (share of R)": c[0] / R if k == "WP2" and nm == "stopped" else np.nan,
                                 "d5 med": float(np.median(d_[k][5][st[nm]])), "d5 med op": float(np.median(d5n[st[nm]]))})
                    if k == "WP2" and nm == "stopped":
                        verdict[f"f_{op}"] = {"d": c[0], "lo": c[1], "hi": c[2], "r": c[0] / R, "label": lab(c[0] / R, c)}
            for nm in ("SL (stopped, log moves)", "launch (v<2, log>5m)", "moving (v>=0.5)"):
                m = st[nm]
                sp = np.mean([np.linalg.norm(np.diff(np.concatenate([np.zeros((n, 1, 2)), p], 1), axis=1), axis=-1) * 4 for p in P[k]], 0)
                for j in range(20):
                    prof.append({"arm": k, "stratum": nm, "t": 0.25 * (j + 1), "speed med": float(np.median(sp[m, j])), "speed mean": float(sp[m, j].mean())})
                mvg = m & (d_[k][4] > 1.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = sp[:, 16:20].mean(1) / sp[:, 12:16].mean(1)
                rows.append({"arm": k, "op": "kink: mean speed 4-5 s / 3-4 s (frames with d4 > 1 m)", "stratum": nm, "n": int(mvg.sum()), "d": float(np.median(ratio[mvg])),
                             "lo": float(np.percentile(ratio[mvg], 25)), "hi": float(np.percentile(ratio[mvg], 75))})
        stats.write_table(rows, OUT / "horizon")
        pd.DataFrame(prof).to_csv(OUT / "speed_profile.csv", index=False)
        run.info("(f) %s", json.dumps({k: v for k, v in verdict.items() if k.startswith("f_")}))

        # ---- (b) ego channel
        rows, bias = [], {}
        have = [v for v in VARS if all(len(list(C.Z.root("preds", f"op_cinque_lx-WP2-full-s{s}_{v}").glob("*.npz"))) >= n for s in (0, 1))]
        for v in have:
            ps = [C.preds(f"lx-WP2-full-s{s}_{v}") for s in (0, 1)]
            bias[v] = (ps, np.mean([C.rfs(p) for p in ps], 0), np.mean([dist(p, 19) for p in ps], 0), [C.rfs(p) for p in ps])
        if "main" in bias:
            eq = [np.linalg.norm(bias["main"][0][s] - P["WP2"][s], axis=-1) for s in (0, 1)]
            verdict["b_equivalence_main_vs_stored_m"] = {"mean": float(np.mean(eq)), "p99": float(np.percentile(eq, 99)), "max": float(np.max(eq))}
            for v in have:
                ps, s2, d5, per = bias[v]
                for nm in ("stopped", "SL (stopped, log moves)", "SS (stopped, log stays)", "launch (v<2, log>5m)", "moving (v>=0.5)", "all"):
                    m = st[nm]
                    c = C.ci(s2 - bias["main"][1], m)
                    cg = C.ci_mean(d5 - bias["main"][2], m)
                    rows.append({"var": v, "stratum": nm, "n": int(m.sum()), "RFS": C.cm(s2, m), "d vs main": c[0], "lo": c[1], "hi": c[2],
                                 "r (share of R)": c[0] / R if nm == "stopped" else np.nan, "d per seed": "/".join(f"{C.cm(per[s] - bias['main'][3][s], m):+.3f}" for s in (0, 1)),
                                 "d vs shipped": C.cm(s2 - sco["shipped"], m), "d vs log": C.cm(s2 - sco["log"], m),
                                 "d5 med": float(np.median(d5[m])), "d5 mean - main (m)": cg[0], "d5 lo": cg[1], "d5 hi": cg[2]})
                    if nm == "stopped":
                        verdict[f"b_{v}"] = {"d": c[0], "lo": c[1], "hi": c[2], "r": c[0] / R, "label": lab(c[0] / R, c)}
            stats.write_table(rows, OUT / "bias")
            for t in ("WP2-full-s0", "WP2-full-s1"):
                f = ldir() / f"bias_stats-{t}.json"
                if f.exists():
                    verdict[f"b_stats_{t}"] = json.loads(f.read_text())["standstill"]
            run.info("(b) %s", json.dumps({k: v for k, v in verdict.items() if k.startswith("b_") and "stats" not in k}))

        # ---- (c) protocol: token path vs harness; anticipation
        fv = ldir() / "tok_val.npz"
        if fv.exists():
            z = np.load(fv, allow_pickle=True)
            pos = {nm: i for i, nm in enumerate(z["names"].astype(str))}
            cov = np.array([nm in pos for nm in C.names[:n]])
            ti = np.array([pos[nm] for nm in C.names[:n][cov]])
            tk = {k: [np.zeros((n, 20, 2))] for k in ("shipped",)} | {"WP2": [np.zeros((n, 20, 2)), np.zeros((n, 20, 2))], "WP1": [np.zeros((n, 20, 2))]}
            for k, tags in (("shipped", ["P0"]), ("WP2", ["WP2-full-s0", "WP2-full-s1"]), ("WP1", ["WP1-full-s0"])):
                for s, t in enumerate(tags):
                    tk[k][s][cov] = z[f"plan_{t}"][ti]
            g0 = np.linalg.norm(tk["shipped"][0][cov] - P["shipped"][0][cov], axis=-1)
            verdict["c_token_vs_harness_shipped_m"] = {"mean": float(g0.mean()), "p99": float(np.percentile(g0, 99)), "n": int(cov.sum())}
            rows = []
            for k in ("shipped", "WP1", "WP2"):
                hs = P[k] if k != "WP1" else P[k][:1]
                s_h, s_t = np.mean([C.rfs(p) for p in hs], 0), np.mean([C.rfs(p) for p in tk[k]], 0)
                d5h, d5t = np.mean([dist(p, 19) for p in hs], 0), np.mean([dist(p, 19) for p in tk[k]], 0)
                for nm in ("stopped", "SL (stopped, log moves)", "SS (stopped, log stays)", "moving (v>=0.5)", "all"):
                    m = st[nm] & cov
                    c = C.ci(s_t - s_h, m)
                    Rc = C.cm(sco["log"] - s_h, m)
                    rows.append({"arm": k, "stratum (covered)": nm, "n": int(m.sum()), "RFS harness": C.cm(s_h, m), "RFS token path": C.cm(s_t, m), "d": c[0], "lo": c[1],
                                 "hi": c[2], "log - harness": Rc, "r": c[0] / Rc if Rc else np.nan, "d5 med harness": float(np.median(d5h[m])),
                                 "d5 med token": float(np.median(d5t[m])), "d5 med log": float(np.median(d_["log"][5][m])),
                                 "plan xy diff mean (m)": float(np.mean([np.linalg.norm(a_ - b_, axis=-1)[m].mean() for a_, b_ in zip(tk[k], hs)]))})
                    if k == "WP2" and nm == "stopped":
                        verdict["c_protocol"] = {"d": c[0], "lo": c[1], "hi": c[2], "r": c[0] / Rc, "label": lab(c[0] / Rc, c), "n": int(m.sum())}
            stats.write_table(rows, OUT / "protocol")
            run.info("(c) %s | G0 %s", json.dumps(verdict.get("c_protocol")), json.dumps(verdict["c_token_vs_harness_shipped_m"]))
            anticipation(z, "val", run)
            fd = ldir() / "tok_dev.npz"
            if fd.exists():
                anticipation(np.load(fd, allow_pickle=True), "dev", run)
                verdict["d"] = targets(C, run)

        # ---- (a) context
        fl = OUT / "context_vlm.csv"
        if fl.exists():
            verdict["a"] = context(C, sco, d_, run)
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
        run.info("verdict written")


def anticipation(z, name, run):
    """Standstill rows by the logged launch time tau: plan distance of every arm against the log (token path)."""
    from jevdrive import stats
    from jevdrive import waymo as W
    fut, v0, seq = z["fut"][..., :2].astype(np.float64), W.init_speed(z["past"]), z["seq"].astype(str)
    tau = tau_of(fut)
    arms = {"shipped": ["P0"], "WP1": ["WP1-full-s0"], "WP2": ["WP2-full-s0", "WP2-full-s1"]}
    bins = [("stopped, tau 0-0.5 s", (v0 < V_STOP) & (tau <= 0.5)), ("stopped, tau 0.5-1 s", (v0 < V_STOP) & (tau > 0.5) & (tau <= 1.0)),
            ("stopped, tau 1-2 s", (v0 < V_STOP) & (tau > 1) & (tau <= 2)), ("stopped, tau 2-3 s", (v0 < V_STOP) & (tau > 2) & (tau <= 3)),
            ("stopped, tau 3-5 s", (v0 < V_STOP) & (tau > 3) & np.isfinite(tau)), ("stopped, no launch in 5 s", (v0 < V_STOP) & ~np.isfinite(tau)),
            ("just launched, v 0.5-2", (v0 >= V_STOP) & (v0 < 2)), ("v 2-5", (v0 >= 2) & (v0 < 5)), ("v >= 5", v0 >= 5)]
    codes, u = pd.factorize(pd.Series(seq))
    K = np.stack([np.bincount(d, minlength=len(u)) for d in np.random.default_rng(0).integers(len(u), size=(1000, len(u)))]).astype(float)[:, codes]
    rows = []
    for nm, m in bins:
        if m.sum() < 5:
            continue
        row = {"bin": nm, "rows": int(m.sum()), "sequences": int(len(set(seq[m])))}
        for t, j in ((3, 11), (5, 19)):
            dl = np.linalg.norm(fut[:, j], axis=-1)
            row[f"log d{t} mean (m)"] = float(dl[m].mean())
            for k, tags in arms.items():
                dp = np.mean([np.linalg.norm(z[f"plan_{x}"][:, j].astype(np.float64), axis=-1) for x in tags], 0)
                row[f"{k} d{t} mean"] = float(dp[m].mean())
                if t == 5:
                    row[f"{k} d5 / log"] = float(dp[m].sum() / max(dl[m].sum(), 1e-9))
                    if k == "WP2":
                        ds = np.linalg.norm(z["plan_P0"][:, j].astype(np.float64), axis=-1)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            b = (K @ (m * (dp - ds))) / (K @ m.astype(float))
                        row["WP2 - shipped d5 (m)"], row["lo"], row["hi"] = float((dp - ds)[m].mean()), *np.nanpercentile(b, [2.5, 97.5])
                        b = (K @ (m * (dp - dl))) / (K @ m.astype(float))
                        row["WP2 - log d5 (m)"], row["log lo"], row["log hi"] = float((dp - dl)[m].mean()), *np.nanpercentile(b, [2.5, 97.5])
        rows.append(row)
    stats.write_table(rows, OUT / f"anticipation_{name}")
    run.info("anticipation %s:\n%s", name, pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.2f}"))


def targets(C, run):
    """(d): base rates, in-distribution launch calibration on r2-dev standstill rows, frozen-token probe of 'launch within 4 s'."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from jevdrive import stats
    from jevdrive import waymo as W
    zd, zt, zv = (np.load(ldir() / f"tok_{k}.npz", allow_pickle=True) for k in ("dev", "train_stop", "val"))
    out = {}

    def prep(z):
        v0 = W.init_speed(z["past"])
        d4 = np.linalg.norm(z["fut"][:, 15, :2].astype(np.float64), axis=-1)
        return v0 < V_STOP, d4, z["seq"].astype(str)
    (sd, d4d, qd), (s_t, d4t, qt), (sv, d4v, qv) = prep(zd), prep(zt), prep(zv)
    n = C.n
    d4r = np.linalg.norm(C.fut[:n, 15], axis=-1)
    base = [{"set": "r2-train rows (even frames)", "standstill rows": int(s_t.sum()), "sequences": int(len(set(qt[s_t]))), "p_go (d4 >= 2 m)": float((d4t[s_t] >= 2).mean()),
             "p_stay (d4 < 0.5 m)": float((d4t[s_t] < 0.5).mean()), "mean d4 (m)": float(d4t[s_t].mean()), "median tau of go rows (s)": float(np.median(tau_of(zt["fut"][s_t & (d4t >= 2)])))},
            {"set": "r2-dev rows", "standstill rows": int(sd.sum()), "sequences": int(len(set(qd[sd]))), "p_go (d4 >= 2 m)": float((d4d[sd] >= 2).mean()),
             "p_stay (d4 < 0.5 m)": float((d4d[sd] < 0.5).mean()), "mean d4 (m)": float(d4d[sd].mean()), "median tau of go rows (s)": float(np.median(tau_of(zd["fut"][sd & (d4d >= 2)])))},
            {"set": "val, every even frame", "standstill rows": int(sv.sum()), "sequences": int(len(set(qv[sv]))), "p_go (d4 >= 2 m)": float((d4v[sv] >= 2).mean()),
             "p_stay (d4 < 0.5 m)": float((d4v[sv] < 0.5).mean()), "mean d4 (m)": float(d4v[sv].mean()), "median tau of go rows (s)": float(np.median(tau_of(zv["fut"][sv & (d4v >= 2)])))},
            {"set": "val rater frames", "standstill rows": int(C.st["stopped"].sum()), "sequences": int(C.st["stopped"].sum()),
             "p_go (d4 >= 2 m)": float((d4r[C.st["stopped"]] >= 2).mean()), "p_stay (d4 < 0.5 m)": float((d4r[C.st["stopped"]] < 0.5).mean()),
             "mean d4 (m)": float(d4r[C.st["stopped"]].mean()), "median tau of go rows (s)": float(np.median(tau_of(C.fut[:n][C.st["stopped"] & (d4r >= 2)])))}]
    stats.write_table(base, OUT / "targets_base")
    out["base"] = {b["set"]: b["p_go (d4 >= 2 m)"] for b in base}
    # in-distribution calibration (dev standstill rows) and the same statistics on val (every even frame, standstill)
    rows, hist = [], []
    arms = {"shipped": ["P0"], "WP1": ["WP1-full-s0"], "WP2": ["WP2-full-s0", "WP2-full-s1"]}
    for setn, z, s, d4l, q in (("r2-dev", zd, sd, d4d, qd), ("val even frames", zv, sv, d4v, qv)):
        go, stay = s & (d4l >= 2), s & (d4l < 0.5)
        for k, tags in arms.items():
            dp = np.mean([np.linalg.norm(z[f"plan_{t}"][:, 15].astype(np.float64), axis=-1) for t in tags], 0)
            a_go = auc_ci((d4l[s] >= 2).astype(int), dp[s], q[s], 300)
            rows.append({"set": setn, "arm": k, "standstill rows": int(s.sum()), "go rows": int(go.sum()), "M = mean d4 plan / log": float(dp[s].mean() / d4l[s].mean()),
                         "R4 = median d4 plan / log (go rows)": float(np.median(dp[go] / d4l[go])), "launch recall (d4 plan >= 0.5 log, go rows)": float((dp[go] >= 0.5 * d4l[go]).mean()),
                         "false go (d4 plan >= 2 m, stay rows)": float((dp[stay] >= 2).mean()), "mean d4 plan on stay rows (m)": float(dp[stay].mean()),
                         "mean d4 plan / log on go rows": float(dp[go].mean() / d4l[go].mean()), "AUC of d4 plan for go": a_go[0], "AUC lo": a_go[1], "AUC hi": a_go[2]})
            if setn == "r2-dev":
                h, e = np.histogram(dp[s], bins=[0, 0.5, 1, 2, 4, 8, 12, 16, 24, 100])
                hist += [{"arm": k, "bin_lo": e[i], "bin_hi": e[i + 1], "share": h[i] / s.sum()} for i in range(len(h))]
                if k == "WP2":
                    out.update(M=rows[-1]["M = mean d4 plan / log"], R4=rows[-1]["R4 = median d4 plan / log (go rows)"], recall=rows[-1]["launch recall (d4 plan >= 0.5 log, go rows)"])
        if setn == "r2-dev":
            h, e = np.histogram(d4l[s], bins=[0, 0.5, 1, 2, 4, 8, 12, 16, 24, 100])
            hist += [{"arm": "log", "bin_lo": e[i], "bin_hi": e[i + 1], "share": h[i] / s.sum()} for i in range(len(h))]
            sg = np.exp(z["spread_x_shipped"][:, 20])
            out["shipped_sigma_x_4s"] = {"go": float(np.median(sg[go])), "stay": float(np.median(sg[stay])), "auc_for_go": auc((d4l[s] >= 2).astype(int), sg[s])}
    out["in_distribution_under_launch"] = "yes" if (out["M"] < 0.8 or out["R4"] < 0.75) else "no" if (out["M"] >= 0.9 and out["R4"] >= 0.9) else "partial"
    # probe: frozen pooled tokens (last slot and the slot 1 s earlier) -> launch within 4 s
    Xt, yt = zt["pooled"][s_t].reshape(int(s_t.sum()), -1).astype(np.float32), (d4t[s_t] >= 2).astype(int)
    sc_ = StandardScaler().fit(Xt)
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc_.transform(Xt), yt)
    pr = {}
    for setn, z, s, d4l, q in (("r2-dev", zd, sd, d4d, qd), ("val even frames", zv, sv, d4v, qv)):
        p = clf.predict_proba(sc_.transform(z["pooled"][s].reshape(int(s.sum()), -1).astype(np.float32)))[:, 1]
        y = (d4l[s] >= 2).astype(int)
        a_ = auc_ci(y, p, q[s], 300)
        pr[setn] = {"AUC": a_[0], "lo": a_[1], "hi": a_[2], "rows": int(s.sum()), "p_go": float(y.mean()), "mean predicted p": float(p.mean())}
        for r in rows:
            if r["set"] == setn:
                r["probe AUC (frozen tokens)"], r["probe lo"], r["probe hi"] = a_
        if setn == "val even frames":                                 # rater standstill frames inside the token cache: probe probability by logged outcome
            pos = {nm: i for i, nm in enumerate(z["names"].astype(str)[s])}
            cov = np.array([nm in pos for nm in C.names[:n]]) & C.st["stopped"]
            pp = np.array([p[pos[nm]] for nm in C.names[:n][cov]])
            yy = (d4r[cov] >= 2).astype(int)
            pr["val rater standstill (covered)"] = {"n": int(cov.sum()), "p_go": float(yy.mean()), "AUC": auc(yy, pp), "mean p on go": float(pp[yy == 1].mean()),
                                                     "mean p on stay": float(pp[yy == 0].mean()) if (yy == 0).any() else np.nan, "share p > 0.5": float((pp > 0.5).mean())}
    out["probe"] = pr
    stats.write_table(rows, OUT / "targets_calibration")
    pd.DataFrame(hist).to_csv(OUT / "targets_hist.csv", index=False)
    run.info("(d) %s", json.dumps(out, default=float))
    return out


def context(C, sco, d_, run):
    """(a): VLM labels, agreement with the hand labels, standstill context classes, traffic-light table."""
    from jevdrive import stats
    n = C.n
    v = pd.read_csv(OUT / "context_vlm.csv").set_index("name").reindex(C.names[:n])
    out = {}
    fh = OUT / "context_hand.csv"
    use = v[["light", "stop_sign", "lead", "cross"]].copy()
    fx = OUT / "context_extra.csv"                                    # post hoc: stop-controlled junction from the right camera (see results)
    if fx.exists():
        use["stop_ctrl"] = pd.read_csv(fx).set_index("name").reindex(C.names[:n])["stop_ctrl"].to_numpy()
    if fh.exists():
        h = pd.read_csv(fh).set_index("name")
        ag = []
        for q in ("light", "stop_sign", "lead", "cross"):
            a_, b_ = h[q], v.loc[h.index, q]
            ag.append({"question": q, "n": len(h), "agreement": float((a_ == b_).mean()),
                       "confusion (hand -> VLM: count)": "; ".join(f"{x} -> {y}: {c}" for (x, y), c in pd.crosstab(a_, b_).stack().items() if c)})
        if "stop_ctrl" in use and "stop_ctrl_inferred" in h:
            hh = h[h.stop_ctrl_inferred != "unk"]
            b_ = use.loc[hh.index, "stop_ctrl"]
            ag.append({"question": "stop_ctrl (post hoc, right camera; hand = inferred from the front image, 'unk' dropped)", "n": len(hh),
                       "agreement": float(((hh.stop_ctrl_inferred == "yes") == (b_ == "stop_controlled")).mean()),
                       "confusion (hand -> VLM: count)": "; ".join(f"{x} -> {y}: {c}" for (x, y), c in pd.crosstab(hh.stop_ctrl_inferred, b_).stack().items() if c)})
        stats.write_table(ag, OUT / "context_agreement")
        out["agreement"] = {r["question"].split(" ")[0]: r["agreement"] for r in ag}
        fa = OUT / "context_hand_all.csv"                             # hand labels of every standstill frame, when a question was unreliable
        if fa.exists():
            ha = pd.read_csv(fa).set_index("name")
            for q in ha.columns.intersection(use.columns):
                use.loc[ha.index, q] = ha[q]
    stop_sign = (use.stop_sign == "stop_sign_for_ego") | (use.get("stop_ctrl", pd.Series("", index=use.index)) == "stop_controlled")
    ctx = np.where(use.light == "red_or_yellow_for_ego", "red", np.where(use.light == "green_for_ego", "green", np.where(stop_sign, "stop_sign", np.where(
        use.lead == "moving_lead", "lead_moving", np.where(use.lead == "stopped_lead", "lead_stopped", np.where(use.cross == "crossing", "cross", "open"))))))
    lost = C.w * (sco["top"] - sco["WP2"])
    stp = C.st["stopped"]
    rows = []
    for scope, base in (("stopped", stp), ("moving", ~stp)):
        for c in CTX_ORDER:
            m = base & (ctx == c)
            if m.sum() == 0:
                continue
            row = {"scope": scope, "context": c, "n": int(m.sum()), "log launches (d5 >= 1 m)": float((C.logd[m] >= 1).mean())}
            for k in ("top", "log", "shipped", "WP1", "WP2"):
                row[f"RFS {k}"] = float(sco[k][m].mean())
            for k in ("top", "log", "shipped", "WP2"):
                row[f"d5 med {k}"] = float(np.median(d_[k][5][m]))
            row["WP2 points lost vs top"] = float(lost[m].sum())
            row["share of the scope's points"] = float(lost[m].sum() / lost[base].sum())
            for x, y in (("WP2", "log"), ("WP2", "shipped")):
                cc = C.ci_mean(sco[x] - sco[y], m)
                row[f"{x} - {y} (frame mean)"], row[f"{x} - {y} lo"], row[f"{x} - {y} hi"] = cc
            rows.append(row)
    stats.write_table(rows, OUT / "context_table")
    vis = float(sum(r["share of the scope's points"] for r in rows if r["scope"] == "stopped" and r["context"] in CUE_VISIBLE))
    out["stopped_points_share_cue_visible"] = vis
    out["reading"] = "visible cue, no launch" if vis >= 0.6 else "anticipation" if vis <= 0.4 else "mixed"
    # traffic lights on all frames
    rows = []
    for scope, base in (("all", np.ones(n, bool)), ("stopped", stp), ("moving", ~stp)):
        for lt in ("red_or_yellow_for_ego", "green_for_ego", "no_light_for_ego"):
            m = base & (use.light == lt).to_numpy()
            if m.sum() == 0:
                continue
            row = {"scope": scope, "light": lt, "n": int(m.sum()), "log launches or moves (d5 >= 1 m)": float((C.logd[m] >= 1).mean())}
            for k in ("top", "log", "shipped", "WP1", "WP2"):
                row[f"RFS {k}"] = float(sco[k][m].mean())
            for k in ("log", "shipped", "WP2"):
                row[f"top - {k}"] = float((sco["top"] - sco[k])[m].mean())
                row[f"d5 med {k}"] = float(np.median(d_[k][5][m]))
            cc = C.ci_mean(sco["WP2"] - sco["shipped"], m)
            row["WP2 - shipped"], row["lo"], row["hi"] = cc
            cc = C.ci_mean(sco["WP2"] - sco["log"], m)
            row["WP2 - log"], row["log lo"], row["log hi"] = cc
            rows.append(row)
    stats.write_table(rows, OUT / "context_lights")
    pd.DataFrame({"name": C.names[:n], "v0": C.v0, "log_d5": C.logd, "stopped": stp, "context": ctx, **{q: use[q].to_numpy() for q in use.columns},
                  **{f"vlm_{q}_p": v[[c for c in v.columns if c.startswith(q + "_p_")]].max(1).to_numpy() for q in ("light", "stop_sign", "lead", "cross")},
                  "rfs_top": sco["top"], "rfs_log": sco["log"], "rfs_shipped": sco["shipped"], "rfs_WP1": sco["WP1"], "rfs_WP2": sco["WP2"],
                  "d5_top": d_["top"][5], "d5_log": d_["log"][5], "d5_shipped": d_["shipped"][5], "d5_WP2": d_["WP2"][5]}).to_csv(OUT / "frames.csv", index=False, float_format="%.4f")
    run.info("(a) %s", json.dumps(out, default=float))
    return out


# ---------------------------------------------------------------- Step 2
def cmd_gate(a):
    C = Ctx()
    s_a, s_r, s_s = C.rfs(C.preds(a.arm)), C.rfs(C.preds(a.ref)), C.rfs(C.preds("shipped"))
    arm = a.arm.split("-pilot")[0]
    ref_gap = C.cm(s_r - s_s, C.st["stopped"])
    thr = 0.10 if ref_gap < -0.10 else 0.0                              # prereg: weak gate when the pilot-scale WP2 shows no standstill shortfall
    g = {"arm": a.arm, "ref": a.ref, "stopped": C.ci(s_a - s_r, C.st["stopped"]), "moving": C.ci(s_a - s_r, C.st["moving (v>=0.5)"]), "all": C.ci(s_a - s_r),
         "SL": C.ci(s_a - s_r, C.st["SL (stopped, log moves)"]), "SS": C.ci(s_a - s_r, C.st["SS (stopped, log stays)"]),
         "ref - shipped stopped": ref_gap, "threshold_stopped": thr, "RFS arm": C.cm(s_a), "RFS ref": C.cm(s_r),
         "d5 med SL arm / ref / log": [float(np.median(dist(C.preds(t), 19)[C.st["SL (stopped, log moves)"]])) for t in (a.arm, a.ref)]
         + [float(np.median(C.logd[C.st["SL (stopped, log moves)"]]))]}
    g["pass"] = bool(g["stopped"][0] >= thr and g["moving"][0] >= -0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gate_{arm}.json").write_text(json.dumps(g, default=float))
    print(json.dumps(g, indent=1, default=float))
    _sys.exit(0 if g["pass"] else 3)


def cmd_full(a):
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", f"wod-launch-full-{a.arm}", seed=0, config=dict(B=B, arm=a.arm)) as run:
        run.use_split(splits.load("wod/val"))
        C = Ctx()
        n, st = C.n, C.st
        tags = {a.arm: [f"{a.arm}-full-s0", f"{a.arm}-full-s1"], "WP2": ["WP2-full-s0", "WP2-full-s1"], "shipped": ["shipped"]}
        PA = {k: [C.preds(t, True) for t in v] for k, v in tags.items()}
        PA["log"] = [C.fut]
        sco = {k: np.mean([C.rfs(p) for p in v], 0) for k, v in PA.items()}
        per = {k: [C.rfs(p) for p in v] for k, v in PA.items()}
        d5 = {k: np.mean([dist(p[:n], 19) for p in v], 0) for k, v in PA.items()}
        d3 = {k: np.mean([dist(p[:n], 11) for p in v], 0) for k, v in PA.items()}
        rows = []
        for nm, m in st.items():
            row = {"stratum": nm, "n": int(m.sum())}
            for k in (a.arm, "WP2", "shipped", "log"):
                row[f"RFS {k}"] = C.cm(sco[k], m)
            for y in ("WP2", "shipped", "log"):
                c = C.ci(sco[a.arm] - sco[y], m)
                row[f"{a.arm} - {y}"], row[f"vs {y} lo"], row[f"vs {y} hi"] = c
            row["per seed vs WP2"] = "/".join(f"{C.cm(per[a.arm][s] - per['WP2'][s], m):+.3f}" for s in (0, 1))
            for k in (a.arm, "WP2", "shipped", "log"):
                row[f"d5 med {k}"] = float(np.median(d5[k][m]))
            c = C.ci_mean(d5[a.arm] - d5["WP2"], m)
            row["d5 mean vs WP2 (m)"], row["d5 lo"], row["d5 hi"] = c
            c = C.ci_mean(d3[a.arm] - d3["WP2"], m)
            row["d3 mean vs WP2 (m)"], row["d3 lo"], row["d3 hi"] = c
            rows.append(row)
        stats.write_table(rows, OUT / f"fix_{a.arm}_strata")
        # ADE on the 1 437 frames with futures
        err = {k: np.mean([np.linalg.norm(p - C.fut, axis=-1) for p in v], 0) for k, v in PA.items() if k != "log"}
        rows2 = []
        for nm, sl in (("ADE@3s", slice(0, 12)), ("ADE@5s", slice(0, 20))):
            e = {k: x[:, sl].mean(1) for k, x in err.items()}
            for y in ("WP2", "shipped"):
                dd = e[a.arm] - e[y]
                b = (C.Ka @ dd) / C.Ka.sum(1)
                rows2.append({"metric": nm, "arm": float(e[a.arm].mean()), y: float(e[y].mean()), "vs": y, "d": float(dd.mean()), "lo": float(np.percentile(b, 2.5)),
                              "hi": float(np.percentile(b, 97.5))})
        stats.write_table(rows2, OUT / f"fix_{a.arm}_ade")
        s_ = pd.DataFrame(rows).set_index("stratum")
        stp, mov = s_.loc["stopped"], s_.loc["moving (v>=0.5)"]
        v = {"arm": a.arm, "stopped vs WP2": [stp[f"{a.arm} - WP2"], stp["vs WP2 lo"], stp["vs WP2 hi"]], "moving vs WP2": [mov[f"{a.arm} - WP2"], mov["vs WP2 lo"], mov["vs WP2 hi"]],
             "all vs WP2": [s_.loc["all"][f"{a.arm} - WP2"], s_.loc["all"]["vs WP2 lo"], s_.loc["all"]["vs WP2 hi"]],
             "all vs shipped": [s_.loc["all"][f"{a.arm} - shipped"], s_.loc["all"]["vs shipped lo"], s_.loc["all"]["vs shipped hi"]]}
        v["moving_no_loss"] = bool(mov["vs WP2 hi"] >= 0 and mov[f"{a.arm} - WP2"] >= -0.05)
        v["label"] = "fixed" if (stp["vs WP2 lo"] > 0 and v["moving_no_loss"]) else "weak / undecided" if stp[f"{a.arm} - WP2"] > 0 else "not fixed"
        (OUT / f"fix_{a.arm}_verdict.json").write_text(json.dumps(v, indent=1, default=float))
        pd.DataFrame({"name": C.names[:n], **{f"rfs_{k}": sco[k] for k in sco}, **{f"d5_{k}": d5[k] for k in d5}}).to_csv(OUT / f"fix_{a.arm}_frames.csv", index=False,
                                                                                                                           float_format="%.4f")
        run.info("%s\n%s", json.dumps(v, default=float), pd.DataFrame(rows).iloc[:, :12].to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    sp_.add_parser("diag")
    p = sp_.add_parser("gate")
    p.add_argument("--arm", required=True)
    p.add_argument("--ref", default="WP2-pilot-s0")
    p = sp_.add_parser("full")
    p.add_argument("--arm", required=True)
    a = ap.parse_args()
    {"diag": cmd_diag, "gate": cmd_gate, "full": cmd_full}[a.cmd](a)
