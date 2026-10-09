"""op_parity path-timing-swap (plans/2026-10-09-path-timing-swap-prereg.md, lane SW1, decision 207): is the turn-token DAC gap a wrong
path shape or a coupling of path shape and timing? A 2 x 2 swap of {path shape, speed profile} between each stored plan and the log,
scored by the official scorer, plus a scorer-free signed-offset check. CPU only, no model runs. The log path and the log timing are
privileged inputs used as analysis swaps only: nothing here is a method or a reportable inference path.

  build   (.venv)          curves of every plan and of the log; cells PP (stored plan), PL (plan path, log timing), LP (log path, plan timing),
                           LL (log) -> $OUT/poses.npz (keys <member>_<cell>, log_ll; SH30 also plx / lpx = straight extension), geom.npz
                           (timed / equal-arc / cross-track offsets, the chord bound E1), tokens_{s0,all}.txt, keys.txt
  gate    (.venv)          identity gate of a score-poses CSV: every <member>_pp row == the stored bench sub-scores
  select  (.venv)          DAC-failing (key, token) rows on > 20 deg tokens -> $OUT/keys_navtest.pkl (input of fd_navsim's instrumented replay)
  replay  (envs/navsim2)   decision 153's scorer hook on those rows: side of the first footprint corner outside -> $OUT/replay.parquet
  report  (.venv)          tables, verdict against the registered lines, figures -> $OUT/report/ (committed under results/pt_swap/, figs/pt_swap/)

Scoring is `python -m jevdrive.bench score-poses --traffic non_reactive` (pt_swap_chain.sh); its parallel loop is the pool's, the replay
uses fd_navsim's worker pool, and the curve construction is ~2 min on one core, so jevdrive.par is not used. Rules (fixed in the pre-registration):
  curve     vertices = origin + 8 poses; a vertex closer than EPS_V to the last kept one is dropped. >= 3 kept vertices: cubic spline of
            (x, y) over cumulative chord length (start tangent (1, 0), natural end), else a polyline; arc length on DS-dense samples; yaw
            linear in arc length between the kept vertices. Amendment A: a spline that leaves its polyline by > GUARD m falls back to the polyline.
  timing    a trajectory's own arc length at 0.5 .. 4 s.
  extend    ext-arc (main): beyond the end, a constant-curvature arc from the last pose (curvature of the last segment, seg >= 0.5 m,
            |kappa| <= 0.3; turn_ceiling.speed's rule). ext-line (sensitivity): straight along the last heading.
  failure   inside-cut = DAC fail and the first LQR-state corner outside is on the turn side; cannot-make-turn = DAC fail, not inside,
            and the cell trajectory's 4 s heading / logged 4 s heading < 0.9 (decision 153).
  offsets   inside positive. T_k = (P(t_k) - L(t_k)) . n_L(t_k); C_k = (plan curve - log curve) . n_L at s* = min(s_plan, s_log)(t_k);
            X_k = signed distance of P(t_k) to the (extended) log curve.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent), str(REPO / "research")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/op_parity/pt_swap"
RES = OUT / "report"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
# family -> (members: key prefix -> bench model spec or innov member)
FAM = {"SH30": {"sh0": "SH30-F-s0", "sh1": "SH30-F-s1"}, "RMH10": {"rm0": "RMH10-F-s0", "rm1": "RMH10-F-s1"},
       "P2": {"p20": "P2-F-s0"}, "OT30": {"ot0": "OT30-F-s0", "ot1": "OT30-F-s1"}, "WA-JEPA": {"wa": "WA-JEPA"}}
MEM = {k: v for f in FAM.values() for k, v in f.items()}
CELLS = ("pp", "pl", "lp")
XCELLS = ("plx", "lpx")                                  # ext-line sensitivity, SH30 only
SUB8 = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC"]
SHOW = ["NC", "DAC", "TTC", "EP", "LK", "DDC"]
EP_TOL = 1e-6
EPS_V, DS, GUARD, KAPPA_MAX = 0.05, 0.05, 0.3, 0.3
TURNS = ["<5", "5-20", "20-45", ">45"]
NQ = 200                                                 # quantile points of the residual-ratio distribution (E1)


# ---------------------------------------------------------------- curves
class Curve:
    """Geometric curve of one (8, 3) trajectory (rules in the module docstring)."""

    def __init__(self, q8, line=False):
        from scipy.interpolate import CubicSpline
        Q = np.vstack([np.zeros(3), np.asarray(q8, np.float64)])
        yaw = np.unwrap(Q[:, 2])
        keep = [0]
        for k in range(1, 9):
            if np.hypot(*(Q[k, :2] - Q[keep[-1], :2])) >= EPS_V:
                keep.append(k)
        V = Q[keep, :2]
        u = np.r_[0, np.cumsum(np.hypot(*np.diff(V, axis=0).T))] if len(keep) > 1 else np.zeros(1)
        self.fallback = False
        if len(keep) >= 3:
            ud = np.unique(np.r_[np.linspace(0, u[-1], int(np.ceil(u[-1] / DS)) + 1), u])
            XY = CubicSpline(u, V, bc_type=((1, (1.0, 0.0)), (2, (0.0, 0.0))))(ud)
            poly = np.stack([np.interp(ud, u, V[:, 0]), np.interp(ud, u, V[:, 1])], 1)
            if np.hypot(*(XY - poly).T).max() > GUARD:
                XY, self.fallback = poly, True
        else:
            ud, XY = u, V
        S = np.r_[0, np.cumsum(np.hypot(*np.diff(XY, axis=0).T))] if len(XY) > 1 else np.zeros(1)
        self.S, self.XY, self.L = S, XY, float(S[-1])
        self.sk = np.interp(u, ud, S) if len(keep) > 1 else np.zeros(1)        # arc length of the kept vertices
        self.yk = yaw[keep]
        sv = np.zeros(9)                                                        # arc length of every vertex (the timing)
        j = 0
        for k in range(1, 9):
            if j + 1 < len(keep) and keep[j + 1] == k:
                j += 1
                sv[k] = self.sk[j]
            else:
                nxt = self.sk[j + 1] if j + 1 < len(keep) else np.inf
                sv[k] = min(self.sk[j] + np.hypot(*(Q[k, :2] - Q[keep[j], :2])), nxt)
        self.sv = np.maximum.accumulate(sv)
        seg = self.sk[-1] - self.sk[-2] if len(keep) > 1 else 0.0
        self.kap = 0.0 if (line or len(keep) < 2) else float(np.clip((self.yk[-1] - self.yk[-2]) / max(seg, 0.5), -KAPPA_MAX, KAPPA_MAX))

    def at(self, s):
        """(x, y, yaw) at arc lengths s (m,), extended beyond the end by the curve's rule."""
        s = np.asarray(s, np.float64)
        si = np.minimum(s, self.L)
        out = np.stack([np.interp(si, self.S, self.XY[:, 0]), np.interp(si, self.S, self.XY[:, 1]), np.interp(si, self.sk, self.yk)], 1)
        over = s > self.L
        if over.any():
            w, th0, (x0, y0) = s[over] - self.L, self.yk[-1], self.XY[-1]
            th = th0 + self.kap * w
            if abs(self.kap) < 1e-6:
                x, y = x0 + w * np.cos(th0), y0 + w * np.sin(th0)
            else:
                x, y = x0 + (np.sin(th) - np.sin(th0)) / self.kap, y0 - (np.cos(th) - np.cos(th0)) / self.kap
            out[over] = np.stack([x, y, th], 1)
        return out

    def dense(self, upto):
        """Dense (m, 2) samples of the curve extended to arc length `upto`."""
        if upto <= self.L + DS:
            return self.XY if len(self.XY) > 1 else self.at(np.array([0.0, DS]))[:, :2]
        return np.vstack([self.XY, self.at(np.arange(self.L + DS, upto + DS, DS))[:, :2]])


def signed_dist(pts, xy):
    """Signed distance (left of the curve positive) of points (m, 2) to a dense polyline xy (n, 2): nearest point on its segments."""
    a, b = xy[:-1], xy[1:]
    ab = b - a
    ap = pts[:, None] - a[None]
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    d = ap - t[..., None] * ab
    dist = np.hypot(d[..., 0], d[..., 1])
    j = dist.argmin(1)
    r = np.arange(len(pts))
    cr = ab[j, 0] * ap[r, j, 1] - ab[j, 1] * ap[r, j, 0]
    return np.sign(cr) * dist[r, j]


def member_poses(spec, names):
    """Stored (n, 8, 3) poses in tab order; rows of NaN where a token has no stored plan (WA-JEPA: 119 tokens)."""
    if spec == "WA-JEPA":
        import innov
        return innov.member_poses("wa:navtest", names, "lb_navtest")
    from jevdrive.bench import compat
    z = np.load(compat.pred_file(spec, "navtest"))
    pos = {t: i for i, t in enumerate(z["tokens"].astype(str))}
    return z["poses"][[pos[t] for t in names]]


def archive(spec):
    from jevdrive.bench import tables as T
    u, src = T.load("navtest", spec)
    assert u is not None, f"no navtest units for {spec}"
    return u, src


# ---------------------------------------------------------------- build
def cmd_build(a):
    from jevdrive import cache
    from jevdrive.bench import tables as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "pt_swap/build", config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        OUT.mkdir(parents=True, exist_ok=True)
        tab = np.load(TAB)
        names, fut = tab["names"].astype(str), tab["fut"].astype(np.float64)
        assert nt.mask(names).all() and np.isfinite(fut).all()
        n = len(names)
        st = T.navtest_strata().reindex(names)
        sgn = np.sign(np.unwrap(np.concatenate([np.zeros((n, 1)), fut[:, :, 2]], 1), axis=1)[:, -1])
        LC = [Curve(f) for f in run.tqdm(fut, desc="log curves")]
        s_log = np.stack([c.sv for c in LC])
        arrs = {"log_ll": fut.astype(np.float32)}
        G = dict(names=names, sgn=sgn, s_log=s_log, L_log=np.array([c.L for c in LC]), log_fallback=np.array([c.fallback for c in LC]))
        chk = {"log": dict(fallback=int(G["log_fallback"].sum()), max_identity_err=float(max(np.abs(c.at(c.sv[1:])[:, :2] - f[:, :2]).max() for c, f in zip(LC, fut))))}
        for m, spec in MEM.items():
            P = member_poses(spec, names)
            ok = np.isfinite(P).all((1, 2))
            arrs[f"{m}_pp"] = np.where(ok[:, None, None], P, fut).astype(P.dtype if ok.all() else np.float32)
            if ok.all():
                arrs[f"{m}_pp"] = P                                             # identity = the stored array itself
            xl = m in FAM["SH30"]
            pl, lp, plx, lpx = (np.zeros((n, 8, 3)) for _ in range(4))
            g = {k: np.full((n, 8), np.nan) for k in ("T", "C", "X")}
            s_plan, Lp, fb, ierr = np.zeros((n, 9)), np.zeros(n), np.zeros(n, bool), 0.0
            for i in run.tqdm(range(n), desc=f"{m} cells"):
                if not ok[i]:
                    pl[i] = lp[i] = fut[i]
                    continue
                pc, lc, p = Curve(P[i]), LC[i], P[i].astype(np.float64)
                s_plan[i], Lp[i], fb[i] = pc.sv, pc.L, pc.fallback
                ierr = max(ierr, float(np.abs(pc.at(pc.sv[1:])[:, :2] - p[:, :2]).max()))
                pl[i], lp[i] = pc.at(lc.sv[1:]), lc.at(pc.sv[1:])
                if xl:
                    plx[i], lpx[i] = Curve(P[i], line=True).at(lc.sv[1:]), Curve(fut[i], line=True).at(pc.sv[1:])
                # scorer-free offsets, inside positive
                nl = np.stack([-np.sin(fut[i, :, 2]), np.cos(fut[i, :, 2])], 1)
                g["T"][i] = sgn[i] * ((p[:, :2] - fut[i, :, :2]) * nl).sum(1)
                ss = np.minimum(pc.sv[1:], lc.sv[1:])
                a_, b_ = pc.at(ss), lc.at(ss)
                g["C"][i] = sgn[i] * ((a_[:, :2] - b_[:, :2]) * np.stack([-np.sin(b_[:, 2]), np.cos(b_[:, 2])], 1)).sum(1)
                g["X"][i] = sgn[i] * signed_dist(p[:, :2], lc.dense(max(pc.L, lc.L) + 5.0))
            arrs[f"{m}_pl"], arrs[f"{m}_lp"] = pl.astype(np.float32), lp.astype(np.float32)
            if xl:
                arrs[f"{m}_plx"], arrs[f"{m}_lpx"] = plx.astype(np.float32), lpx.astype(np.float32)
            G |= {f"{m}_{k}": v for k, v in g.items()} | {f"{m}_s": s_plan, f"{m}_L": Lp, f"{m}_ok": ok, f"{m}_fallback": fb}
            chk[m] = dict(tokens=int(ok.sum()), fallback=int(fb.sum()), max_identity_err=ierr,
                          pl_ext_gt0=int((s_log[ok, -1] > Lp[ok] + 1e-6).sum()), lp_ext_gt0=int((s_plan[ok, -1] > G["L_log"][ok] + 1e-6).sum()))
            # E1: chord bound from the empirical spread of the arc-length ratio (SH30 only)
            if xl:
                turn, spd = st.turn.to_numpy(str), st.speed.to_numpy(str)
                Xh = np.full((n, 8), np.nan)
                base = (s_plan[:, -1] >= 2.0) & (s_log[:, -1] >= 2.0)        # amendment A: moving tokens, ratio clipped to [1/3, 3]
                q = (np.arange(NQ) + 0.5) / NQ
                for tb in TURNS[2:]:
                    for sb in np.unique(spd):
                        cell = (turn == tb) & (spd == sb)
                        for k in range(8):
                            use = cell & base & (s_plan[:, k + 1] >= 0.5)
                            if use.sum() < 30:
                                continue
                            r = np.quantile(np.clip(s_log[use, k + 1] / s_plan[use, k + 1], 1 / 3, 3.0), q)
                            r = r / r.mean()
                            for i in np.flatnonzero(cell & base):
                                lc = LC[i]
                                pts = lc.at(s_log[i, k + 1] * r)[:, :2].mean(0, keepdims=True)
                                Xh[i, k] = sgn[i] * signed_dist(pts, lc.dense(s_log[i, k + 1] * r.max() + 1.0))[0]
                G[f"{m}_Xhat"] = Xh
        np.savez_compressed(OUT / "geom.npz", **G)
        np.savez(OUT / "poses.npz", tokens=names, **arrs)
        keys = [k for k in arrs]
        (OUT / "keys.txt").write_text(" ".join(keys) + "\n")
        rng = np.random.default_rng(0)
        turn = st.turn.to_numpy(str)
        s0 = np.concatenate([rng.choice(np.flatnonzero(turn == b), 12, replace=False) for b in TURNS])
        (OUT / "tokens_s0.txt").write_text("\n".join(sorted(names[s0])) + "\n")
        (OUT / "tokens_all.txt").write_text("\n".join(sorted(names)) + "\n")
        (OUT / "build_check.json").write_text(json.dumps(chk, indent=1))
        run.summary.update(keys=len(keys), tokens=n, check=chk, poses_key=cache.key(inputs=[OUT / "poses.npz"]))
        run.info(json.dumps(chk, indent=1))


# ---------------------------------------------------------------- gate / select / replay
def cmd_gate(a):
    import pandas as pd
    from jevdrive.bench import poses as BP, tables as T
    from jevdrive.bench.navsim import SUBS
    from jevdrive.run import Run
    with Run("op_parity", f"pt_swap/gate-{a.stage}", config=vars(a)) as run:
        tok = np.array(BP.read_tokens(OUT / f"tokens_{a.stage}.txt"))
        df = pd.read_csv(a.score)
        ids = df[df.key.str.endswith("_pp")]                                   # only the identity rows are read here
        del df
        G = np.load(OUT / "geom.npz")
        pos = {t: i for i, t in enumerate(G["names"].astype(str))}
        res, ok = dict(stage=a.stage, n_tokens=len(tok), members={}), True
        for m, spec in MEM.items():
            has = G[f"{m}_ok"][[pos[t] for t in tok]]
            tk = tok[has]
            g = ids[ids.key == f"{m}_pp"].set_index("token").loc[tk]
            u, src = archive(spec)
            r = dict(source=src, tokens=len(tk))
            for k in SUB8:
                dlt = np.abs(g[SUBS[k]].to_numpy(float) - u.loc[tk, k].to_numpy(float))
                r[k] = dict(max_abs=float(dlt.max()), n_diff=int((dlt > (EP_TOL if k == "EP" else 0)).sum()))
            X = u.loc[tk, T.TERMS].to_numpy(float)
            X[:, 8] = np.nan
            dlt = np.abs(g.score.to_numpy(float) - T.score_of(X))
            r["score_noec"] = dict(max_abs=float(dlt.max()), n_diff=int((dlt > EP_TOL).sum()))
            r["pass"] = all(v["n_diff"] == 0 for v in r.values() if isinstance(v, dict))
            ok &= r["pass"]
            res["members"][m] = r
        res["pass"] = bool(ok)
        (OUT / f"gate_{a.stage}.json").write_text(json.dumps(res, indent=1))
        run.summary.update(passed=bool(ok), members={m: r["pass"] for m, r in res["members"].items()})
        run.info(json.dumps({m: {k: v for k, v in r.items() if isinstance(v, dict) and v["n_diff"]} for m, r in res["members"].items()}))
        if not ok:
            raise SystemExit("identity gate failed")


def cmd_select(a):
    import pandas as pd
    from jevdrive.bench import tables as T
    from jevdrive.run import Run
    with Run("op_parity", "pt_swap/select", config=vars(a)) as run:
        sc = pd.read_csv(a.score, usecols=["key", "token", "drivable_area_compliance"])
        st = T.navtest_strata()
        bad = sc[(sc.drivable_area_compliance < 1) & sc.token.map(st.turn).isin(TURNS[2:])]
        z = np.load(OUT / "poses.npz")
        pos = {t: i for i, t in enumerate(z["tokens"].astype(str))}
        plans, want = {}, {}
        for k, g in bad.groupby("key"):
            A = z[k]
            plans[k] = {t: np.asarray(A[pos[t]], np.float64) for t in g.token}
            for t in g.token:
                want.setdefault(t, []).append(k)
        pickle.dump({"plans": plans, "want": want}, open(OUT / "keys_navtest.pkl", "wb"))
        run.summary.update(rows=len(bad), tokens=len(want), keys=len(plans))


def cmd_replay(a):
    import multiprocessing as mp
    import pandas as pd
    import fd_navsim as FD
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    kf = OUT / "keys_navtest.pkl"
    todo = sorted(pickle.load(open(kf, "rb"))["want"])
    procs = a.procs or max(4, n_cpus() // 6)
    with Run("op_parity", "pt_swap/replay", config=vars(a) | {"n_tokens": len(todo), "procs": procs}) as run:
        t0, rows = time.time(), []
        with mp.get_context("fork").Pool(procs, initializer=FD._init, initargs=("navtest", kf)) as pool:
            for i, r in enumerate(pool.imap_unordered(FD.work, todo, chunksize=2)):
                rows.extend({k: v for k, v in x.items() if k != "_states"} for x in r["rows"])
                if (i + 1) % 200 == 0:
                    run.status(f"{i + 1}/{len(todo)} tokens, {time.time() - t0:.0f} s")
        df = pd.DataFrame(rows)
        df[["key", "token", "drivable_area_compliance", "raw_out", "lqr_t", "lqr_side", "lqr_depth"]].to_parquet(OUT / "replay.parquet")
        run.summary.update(tokens=len(todo), rows=len(df), replay_s=time.time() - t0)


# ---------------------------------------------------------------- report
def md(df, digits=2):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    cols = [str(c) for c in df.columns]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


class CB:
    """Log-cluster bootstrap (B 10 000, default_rng(0), percentile) of means and of ratios of sums."""

    def __init__(self, logs, B=10000):
        import pandas as pd
        self.codes, u = pd.factorize(np.asarray(logs))
        self.n = len(u)
        self.idx = np.random.default_rng(0).integers(self.n, size=(B, self.n))

    def _s(self, x, m):
        x = np.where(m, x, 0.0)
        return np.bincount(self.codes, x, self.n)[self.idx].sum(1), float(x.sum())

    def ratio(self, num, den, m=None, f=lambda r: r):
        """f(sum num / sum den) over the masked rows: (point, lo, hi)."""
        m = np.ones(len(self.codes), bool) if m is None else m
        m = m & np.isfinite(num) & np.isfinite(den)
        (bn, pn), (bd, pd_) = self._s(num, m), self._s(den, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = f(bn / bd)
        lo, hi = np.nanquantile(b, [0.025, 0.975])
        return float(f(pn / pd_)) if pd_ else np.nan, float(lo), float(hi)

    def mean(self, x, m=None):
        return self.ratio(np.asarray(x, float), np.isfinite(x).astype(float), m)


def pc(t, f="{:.1f}", scale=1.0):
    return f.format(scale * t[0]) + " [" + f.format(scale * t[1]) + ", " + f.format(scale * t[2]) + "]"


def band(gross, lo, rho):
    """Registered lines: (a) share of DAC failures removed by PL, (b) rho = curve / timed inside offset."""
    a = "S" if (gross >= 0.40 and lo > 0.25) else "F" if gross < 0.15 else "P"
    b = "n/a" if rho is None else "S" if rho < 0.5 else "F" if rho >= 0.8 else "P"
    if a == "F" or b == "F":
        return "FALSIFIED", a, b
    return ("SUPPORTED" if a == "S" and b == "S" else "PARTIAL"), a, b


def cmd_report(a):
    import pandas as pd
    import fd_navsim as FD
    from jevdrive.bench import tables as T
    from jevdrive.bench.navsim import SUBS
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "pt_swap/report", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        RES.mkdir(parents=True, exist_ok=True)
        G = np.load(OUT / "geom.npz")
        names = G["names"].astype(str)
        n = len(names)
        tab = np.load(TAB)
        fut, logs = tab["fut"].astype(np.float64), tab["log"].astype(str)
        st = T.navtest_strata().reindex(names)
        turn, sgn = st.turn.to_numpy(str), G["sgn"]
        dpsi = FD.path_geom(fut)["dpsi"]
        BK = {"all": np.ones(n, bool)} | {b: turn == b for b in TURNS} | {">20": np.isin(turn, TURNS[2:])}
        cb = CB(logs)
        sc = pd.read_csv(a.score)
        rp = pd.read_parquet(OUT / "replay.parquet").set_index(["key", "token"])
        Z = np.load(OUT / "poses.npz")
        S, side = {}, {}
        for k, g in sc.groupby("key"):
            g = g.set_index("token").reindex(names)
            S[k] = {s: g[SUBS[s]].to_numpy(float) for s in SUB8} | {"score": g.score.to_numpy(float)}
            sd = np.full(n, np.nan)
            if k in rp.index.get_level_values(0):
                q = rp.loc[k].reindex(names)
                sd = q.lqr_side.to_numpy(float)
            side[k] = sd
        del sc
        L, summ = [], {}
        P = L.append

        def kinds(key):
            """DAC fail, inside-cut, cannot-make-turn, other (bool (n,)) of one scored key."""
            dac = S[key]["DAC"] < 1
            inside = dac & (side[key] == sgn)
            with np.errstate(invalid="ignore"):
                under = FD.plan_kin(Z[key].astype(np.float64), fut)["gain"] < 0.9
            cannot = dac & ~inside & under
            return dac, inside, cannot, dac & ~inside & ~cannot

        def fam_mean(fam, cell, f):
            """Seed mean of f(key) (n,) with NaN on tokens a member has no plan for."""
            v = []
            for m in FAM[fam]:
                x = np.asarray(f(f"{m}_{cell}"), float).copy()
                x[~G[f"{m}_ok"]] = np.nan
                v.append(x)
            return np.mean(v, 0)

        # ---- 1. the 2 x 2 per family, cell, bucket
        P("## 1. The 2 x 2 by turn bucket (no-EC EPDMS and sub-scores x 100; seed means; log path / log timing are privileged analysis swaps, not a method)\n")
        cell_names = {"pp": "PP plan path, plan timing (stored)", "pl": "PL plan path, log timing", "lp": "LP log path, plan timing", "ll": "LL log (ceiling)"}
        rows, D2 = [], {}
        ll = {s: S["log_ll"][s] for s in SUB8} | {"score": S["log_ll"]["score"]}
        for fam in FAM:
            for cell in ("pp", "pl", "lp", "ll"):
                val = {s: (ll[s] if cell == "ll" else fam_mean(fam, cell, lambda k, s=s: S[k][s])) for s in SUB8 + ["score"]}
                okm = np.isfinite(fam_mean(fam, "pp", lambda k: S[k]["score"]))
                base = fam_mean(fam, "pp", lambda k: S[k]["score"])
                for b, m in BK.items():
                    mm = m & okm
                    r = dict(model=fam, cell=cell.upper(), bucket=b, n=int(mm.sum()), EPDMS=100 * np.nanmean(val["score"][mm]))
                    r |= {s: 100 * np.nanmean(val[s][mm]) for s in SHOW}
                    r["dEPDMS vs PP [95% CI]"] = "" if cell == "pp" else pc(cb.mean(val["score"] - base, mm), "{:+.2f}", 100)
                    rows.append(r)
                    D2[(fam, cell, b)] = r
        t = pd.DataFrame(rows)
        t.to_csv(RES / "two_by_two.csv", index=False)
        for fam in FAM:
            P(f"\n### {fam}\n\n" + md(t[t.model == fam].drop(columns="model")))
        P("\nWA-JEPA rows cover the 12 027 tokens with a stored trajectory. Cells: " + "; ".join(f"{k.upper()} = {v[3:]}" for k, v in cell_names.items()) + ".")

        # ---- 2. failures removed (token-seed rows of the family pooled)
        def pooled(fam, cell, f, ref="pp"):
            """Stacked over members: (fail in ref, fail in cell, logs codes rows mask of valid)"""
            A, B_, ok = [], [], []
            for m in FAM[fam]:
                A.append(f(f"{m}_{ref}"))
                B_.append(f(f"{m}_{cell}"))
                ok.append(G[f"{m}_ok"])
            return np.concatenate(A), np.concatenate(B_), np.concatenate(ok)

        def removal(fam, cell, f, mask):
            k = len(FAM[fam])
            A, B_, ok = pooled(fam, cell, f)
            m = np.tile(mask, k) & ok
            c2 = CB(np.tile(logs, k))
            gross = c2.ratio((A & ~B_).astype(float), A.astype(float), m)
            net = c2.ratio(B_.astype(float), A.astype(float), m, f=lambda r: 1 - r)
            return dict(n_pp=float((A & m).sum() / k), n_cell=float((B_ & m).sum() / k), new=float((B_ & ~A & m).sum() / k), gross=gross, net=net)

        fDAC = lambda k: S[k]["DAC"] < 1                                      # noqa: E731
        fCOL = lambda k: (S[k]["NC"] < 1) | (S[k]["TTC"] < 1)                 # noqa: E731
        Lp = {m: G[f"{m}_L"] for m in MEM}
        P("\n## 2. Failures of the stored plan removed by each swap (token-seed rows pooled; gross = share of PP failures that pass in the cell; "
          "net = 1 - cell failures / PP failures; log-cluster bootstrap)\n")
        rows = []
        for fam in FAM:
            cells = ("pl", "lp") + (XCELLS if fam == "SH30" else ())
            for nm, f in (("DAC", fDAC), ("NC+TTC", fCOL)):
                for b in (">20", ">45", "20-45", "all"):
                    for cell in cells:
                        r = removal(fam, cell, f, BK[b])
                        rows.append(dict(model=fam, set=nm, bucket=b, cell=cell.upper(), **{"PP fails (seed mean)": r["n_pp"], "cell fails": r["n_cell"], "new fails": r["new"],
                                         "gross removed % [95% CI]": pc(r["gross"], "{:.1f}", 100), "net removed % [95% CI]": pc(r["net"], "{:.1f}", 100)}))
                        summ[f"{fam}|{nm}|{b}|{cell}"] = r
        t = pd.DataFrame(rows)
        t.to_csv(RES / "removal.csv", index=False)
        for fam in FAM:
            P(f"\n### {fam}\n\n" + md(t[t.model == fam].drop(columns="model"), 1))

        # extension need and the <= 0.5 m subset; E3 horizon control (SH30)
        P("\n## 3. Extrapolation and horizon controls (SH30)\n")
        rows = []
        k = len(FAM["SH30"])
        ext_pl = np.concatenate([G["s_log"][:, -1] - Lp[m] for m in FAM["SH30"]])
        ext_lp = np.concatenate([G[f"{m}_s"][:, -1] - G["L_log"] for m in FAM["SH30"]])
        tl = np.tile
        for b in ("all", ">20", ">45"):
            for nm, e in (("PL (plan path extended)", ext_pl), ("LP (log path extended)", ext_lp)):
                mm = tl(BK[b], k)
                rows.append(dict(bucket=b, cell=nm, **{"token-seeds": int(mm.sum()), "need ext > 0 %": 100 * (e[mm] > 1e-6).mean(), "> 0.5 m %": 100 * (e[mm] > 0.5).mean(),
                                 "> 2 m %": 100 * (e[mm] > 2).mean(), "median ext of those > 0 (m)": float(np.median(e[mm][e[mm] > 1e-6])) if (e[mm] > 1e-6).any() else 0.0}))
        P(md(pd.DataFrame(rows), 1))
        rows = []
        for nm, f in (("DAC", fDAC), ("NC+TTC", fCOL)):
            for b in (">20", ">45"):
                for cell, e in (("pl", ext_pl), ("lp", ext_lp)):
                    A, B_, ok = pooled("SH30", cell, f)
                    c2 = CB(tl(logs, k))
                    longer = (np.concatenate([G["s_log"][:, -1] - G[f"{m}_s"][:, -1] for m in FAM["SH30"]]) >= 0) == (cell == "pl")
                    for sub, m_ in (("all", np.ones(len(A), bool)), ("ext <= 0.5 m", e <= 0.5), ("cell at least as long as PP", longer), ("cell shorter than PP", ~longer)):
                        m = tl(BK[b], k) & m_
                        g_ = c2.ratio((A & ~B_).astype(float), A.astype(float), m)
                        n_ = c2.ratio(B_.astype(float), A.astype(float), m, f=lambda r: 1 - r)
                        rows.append(dict(set=nm, bucket=b, cell=cell.upper(), subset=sub, **{"PP fails": float((A & m).sum() / k), "cell fails": float((B_ & m).sum() / k),
                                         "gross removed % [95% CI]": pc(g_, "{:.1f}", 100), "net removed % [95% CI]": pc(n_, "{:.1f}", 100)}))
                        summ[f"SH30|{nm}|{b}|{cell}|{sub}"] = dict(gross=g_, net=n_)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "removal_controls.csv", index=False)
        P("\n" + md(t, 1))
        P("\n'cell at least as long as PP': the swapped trajectory covers at least the 4 s arc length of the stored plan (PL: log arc >= plan arc; LP: always equal by "
          "construction, so the split there is by log arc <= plan arc), i.e. a removed failure cannot come from stopping short of where the plan left the road.")

        # ---- 4. failure kinds on > 45 and > 20
        P("\n## 4. DAC failure kinds (decision 153 definitions), % of the bucket's tokens, seed means\n")
        rows = []
        for fam in FAM:
            for cell in ("pp", "pl", "lp") + (("ll",) if fam == "SH30" else ()):
                for b in (">45", ">20"):
                    r = dict(model="log" if cell == "ll" else fam, cell=cell.upper(), bucket=b)
                    for j, nm in enumerate(("DAC fail %", "inside-cut %", "cannot-make-turn %", "other %")):
                        v = kinds("log_ll")[j].astype(float) if cell == "ll" else fam_mean(fam, cell, lambda k_, j=j: kinds(k_)[j])
                        r[nm] = 100 * np.nanmean(v[BK[b]])
                    for j, nm in ((1, "inside-cut"), (2, "cannot-make-turn")):
                        if cell in ("pl", "lp"):
                            base = fam_mean(fam, "pp", lambda k_, j=j: kinds(k_)[j])
                            v = fam_mean(fam, cell, lambda k_, j=j: kinds(k_)[j])
                            r[f"d {nm} pp [95% CI]"] = pc(cb.mean(v - base, BK[b]), "{:+.2f}", 100)
                    rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "failure_kinds.csv", index=False)
        P(md(t))
        nos = {k: int((np.isnan(side[k]) & (S[k]["DAC"] < 1) & BK[">20"]).sum()) for k in S}
        summ["dac_fail_rows_without_side_gt20"] = sum(nos.values())

        # ---- 5. gap to WA-JEPA closed by each swap
        P("\n## 5. Share of the SH30 - WA-JEPA gap closed by each swap (no-EC EPDMS x 100, WA-JEPA's stored per-token scores, 12 146 tokens)\n")
        uw, _ = archive("WA-JEPA")
        Xw = uw.reindex(names)[T.TERMS].to_numpy(float)
        wa_ec = T.score_of(Xw)
        Xw[:, 8] = np.nan
        wa = T.score_of(Xw)
        sh = {c: fam_mean("SH30", c, lambda k_: S[k_]["score"]) for c in CELLS + XCELLS}
        sh_ec = np.mean([T.score_of(archive(sp)[0].reindex(names)[T.TERMS].to_numpy(float)) for sp in FAM["SH30"].values()], 0)
        rows = []
        for b in ("all", "<5", "5-20", "20-45", ">45", ">20"):
            m = BK[b]
            gap = cb.mean(wa - sh["pp"], m)
            r = dict(bucket=b, n=int(m.sum()), **{"gap with EC (official)": 100 * float((wa_ec - sh_ec)[m].mean()), "gap no-EC [95% CI]": pc(gap, "{:.2f}", 100)})
            for c in ("pl", "lp") + XCELLS:
                d_ = cb.mean(sh[c] - sh["pp"], m)
                r[f"{c.upper()} - PP [95% CI]"] = pc(d_, "{:+.2f}", 100)
                r[f"{c.upper()} closes % [95% CI]"] = pc(cb.ratio(sh[c] - sh["pp"], wa - sh["pp"], m), "{:.0f}", 100)
            r["contribution of the bucket to the board gap"] = 100 * float((wa - sh["pp"])[m].sum() / n)
            rows.append(r)
            summ[f"gap|{b}"] = dict(gap_noec=gap, pl=cb.mean(sh["pl"] - sh["pp"], m), lp=cb.mean(sh["lp"] - sh["pp"], m),
                                    pl_closes=cb.ratio(sh["pl"] - sh["pp"], wa - sh["pp"], m), lp_closes=cb.ratio(sh["lp"] - sh["pp"], wa - sh["pp"], m))
        t = pd.DataFrame(rows)
        t.to_csv(RES / "gap_closed.csv", index=False)
        P(md(t))

        # ---- 6. scorer-free offsets
        P("\n## 6. Signed offsets without the scorer (m, inside of the logged turn positive; T = timed points, time-aligned; C = curves at equal arc length; "
          "X = cross-track of the timed points to the log curve; log-cluster bootstrap)\n")
        left = sgn > 0
        GB = {">20": BK[">20"], "20-45": BK["20-45"], ">45": BK[">45"], ">20 left": BK[">20"] & left, ">20 right": BK[">20"] & ~left,
              ">45 left": BK[">45"] & left, ">45 right": BK[">45"] & ~left, "5-20": BK["5-20"], "<5": BK["<5"]}
        rows, GEO = [], {}
        for fam in FAM:
            k = len(FAM[fam])
            c2 = CB(tl(logs, k))
            ok = np.concatenate([G[f"{m}_ok"] for m in FAM[fam]])
            V = {q: np.concatenate([G[f"{m}_{q}"] for m in FAM[fam]]) for q in ("T", "C", "X")}
            for b, m in GB.items():
                mm = tl(m, k) & ok
                r = dict(model=fam, bucket=b, n=int(m.sum()))
                for agg, f in (("mean k", lambda v: v.mean(1)), ("4 s", lambda v: v[:, -1])):
                    for q in ("T", "C", "X"):
                        r[f"{q} {agg}"] = pc(c2.mean(f(V[q]), mm), "{:+.3f}")
                    rho = c2.ratio(f(V["C"]), f(V["T"]), mm)
                    r[f"rho = C / T, {agg}"] = pc(rho, "{:.2f}")
                    GEO[(fam, b, agg)] = dict(T=c2.mean(f(V["T"]), mm), C=c2.mean(f(V["C"]), mm), X=c2.mean(f(V["X"]), mm), rho=rho)
                r["share of tokens with C(4 s) inside > 0.3 m %"] = 100 * float((V["C"][mm, -1] > 0.3).mean())
                r["... outside < -0.3 m %"] = 100 * float((V["C"][mm, -1] < -0.3).mean())
                rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "offsets.csv", index=False)
        for fam in FAM:
            P(f"\n### {fam}\n\n" + md(t[t.model == fam].drop(columns="model")))
        summ["geo"] = {"|".join(k_): v for k_, v in GEO.items()}

        # ---- 7. E1 chord bound and E2 relation to the arc shortfall (SH30)
        P("\n## 7. Chord bound (E1) and the relation to the along-track shortfall (E2), SH30\n")
        k = len(FAM["SH30"])
        c2 = CB(tl(logs, k))
        V = {q: np.concatenate([G[f"{m}_{q}"] for m in FAM["SH30"]]) for q in ("T", "C", "X", "Xhat")}
        rows = []
        for b in (">20", "20-45", ">45", ">45 left", ">45 right"):
            mm = tl(GB[b], k) & np.isfinite(V["Xhat"]).all(1)
            r = dict(bucket=b, **{"token-seeds": int(mm.sum())})
            for agg, f in (("mean k", lambda v: v.mean(1)), ("4 s", lambda v: v[:, -1])):
                r[f"chord bound Xhat {agg}"] = pc(c2.mean(f(V["Xhat"]), mm), "{:+.3f}")
                r[f"observed C {agg}"] = pc(c2.mean(f(V["C"]), mm), "{:+.3f}")
                r[f"observed X {agg}"] = pc(c2.mean(f(V["X"]), mm), "{:+.3f}")
                e1 = c2.ratio(f(V["Xhat"]), f(V["C"]), mm)
                r[f"Xhat / C {agg}"] = pc(e1, "{:.2f}")
                summ[f"E1|{b}|{agg}"] = dict(xhat=c2.mean(f(V["Xhat"]), mm), C=c2.mean(f(V["C"]), mm), ratio=e1)
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "chord_bound.csv", index=False)
        P(md(t))
        ds = np.concatenate([G["s_log"][:, -1] - G[f"{m}_s"][:, -1] for m in FAM["SH30"]])
        kap = tl(np.radians(np.abs(dpsi)) / np.maximum(G["L_log"], 1.0), k)
        z = kap * ds ** 2 / 2

        def slope(y, x, m):
            """OLS slope of y on x (with intercept) and its log-cluster bootstrap CI, on the masked rows."""
            m = m & np.isfinite(y) & np.isfinite(x)
            st_ = [np.bincount(c2.codes, np.where(m, v, 0.0), c2.n) for v in (np.ones_like(x), x, y, x * x, x * y)]
            def sl(s):  # noqa: E306
                n_, sx, sy, sxx, sxy = s
                return (n_ * sxy - sx * sy) / (n_ * sxx - sx * sx)
            b_ = sl([s[c2.idx].sum(1) for s in st_])
            return float(sl([s.sum() for s in st_])), *map(float, np.nanquantile(b_, [0.025, 0.975]))

        from scipy.stats import spearmanr
        rows = []
        for b in (">20", ">45"):
            base = tl(GB[b], k)
            for sub, m_ in (("all", np.ones_like(base)), ("plan shorter (ds > 0)", ds > 0), ("plan longer (ds < 0)", ds < 0)):
                mm = base & m_
                r = dict(bucket=b, subset=sub, **{"token-seeds": int(mm.sum()), "mean ds (m)": float(ds[mm].mean()), "mean z (m)": float(z[mm].mean())})
                r["(T - C)(4 s) on z: slope"] = pc(slope(V["T"][:, -1] - V["C"][:, -1], z, mm), "{:.2f}")
                r["C(4 s) on z: slope"] = pc(slope(V["C"][:, -1], z, mm), "{:.2f}")
                r["X(4 s) on z: slope"] = pc(slope(V["X"][:, -1], z, mm), "{:.2f}")
                r["C(4 s) on signed ds: slope (m / m)"] = pc(slope(V["C"][:, -1], ds, mm), "{:+.3f}")
                r["Spearman C(4 s), z"] = float(spearmanr(V["C"][mm, -1], z[mm]).statistic)
                r["Spearman C(4 s), |ds|"] = float(spearmanr(V["C"][mm, -1], np.abs(ds[mm])).statistic)
                rows.append(r)
                summ[f"E2|{b}|{sub}"] = dict(c_on_z=slope(V["C"][:, -1], z, mm), tc_on_z=slope(V["T"][:, -1] - V["C"][:, -1], z, mm), c_on_ds=slope(V["C"][:, -1], ds, mm))
        t = pd.DataFrame(rows)
        t.to_csv(RES / "shortfall_relation.csv", index=False)
        P("\n" + md(t))
        P("\nz = (logged |heading change| / logged arc) x ds^2 / 2 with ds = logged minus planned 4 s arc length: the inside offset a time-aligned comparison shows "
          "when two trajectories share a path and differ only in how far they got. (T - C) on z is the check of that identity.")

        # ---- post hoc (not registered): lateral dispersion, failure kind x curve offset, direction split of the 2 x 2
        P("\n## 8. Post hoc (not registered)\n")
        rows = []
        for fam in FAM:
            k = len(FAM[fam])
            c2 = CB(tl(logs, k))
            ok = np.concatenate([G[f"{m}_ok"] for m in FAM[fam]])
            C4 = np.concatenate([G[f"{m}_C"][:, -1] for m in FAM[fam]])
            for b in ("<5", "5-20", "20-45", ">45", ">20"):
                mm = tl(BK[b], k) & ok
                rows.append(dict(model=fam, bucket=b, **{"mean |C(4 s)| (m) [95% CI]": pc(c2.mean(np.abs(C4), mm), "{:.3f}"), "median |C(4 s)|": float(np.median(np.abs(C4[mm]))),
                                 "P90 |C(4 s)|": float(np.quantile(np.abs(C4[mm]), 0.9)), "sd C(4 s)": float(C4[mm].std())}))
                summ[f"absC|{fam}|{b}"] = c2.mean(np.abs(C4), mm)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "dispersion.csv", index=False)
        P("### Size of the curve offset at equal arc length, |C(4 s)| (m)\n\n" + md(t, 3))
        k = len(FAM["SH30"])
        KD = [np.concatenate([kinds(f"{m}_pp")[j] for m in FAM["SH30"]]) for j in range(4)]
        Cm = np.concatenate([G[f"{m}_C"] for m in FAM["SH30"]])
        rows = []
        for b in (">20", ">45"):
            for nm, m_ in (("DAC pass", ~KD[0]), ("inside-cut", KD[1]), ("cannot-make-turn", KD[2]), ("other DAC fail", KD[3])):
                mm = tl(BK[b], k) & m_
                rows.append(dict(bucket=b, **{"stored plan": nm, "token-seeds": int(mm.sum()), "mean C(4 s)": float(Cm[mm, -1].mean()), "mean peak inside (max_k C)": float(Cm[mm].max(1).mean()),
                                 "mean peak outside (min_k C)": float(Cm[mm].min(1).mean()), "C(4 s) > +0.3 m %": 100 * float((Cm[mm, -1] > 0.3).mean()),
                                 "C(4 s) < -0.3 m %": 100 * float((Cm[mm, -1] < -0.3).mean()), "mean ds (m)": float(ds[mm].mean())}))
        t = pd.DataFrame(rows)
        t.to_csv(RES / "kind_offset.csv", index=False)
        P("\n### SH30: curve offset by the stored plan's DAC outcome (inside positive)\n\n" + md(t))
        rows = []
        for fam in ("SH30", "WA-JEPA"):
            for b in (">20 left", ">20 right", ">45 left", ">45 right"):
                r = dict(model=fam, bucket=b, n=int(GB[b].sum()))
                for cell in CELLS:
                    r[f"{cell.upper()} EPDMS"] = 100 * np.nanmean(fam_mean(fam, cell, lambda k_: S[k_]["score"])[GB[b]])
                    r[f"{cell.upper()} DAC fail %"] = 100 * np.nanmean(fam_mean(fam, cell, lambda k_: S[k_]["DAC"] < 1)[GB[b]])
                for cell in ("pl", "lp"):
                    r[f"{cell.upper()} DAC gross removed % [95% CI]"] = pc(removal(fam, cell, fDAC, GB[b])["gross"], "{:.1f}", 100)
                rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "direction.csv", index=False)
        P("\n### The 2 x 2 by turn direction\n\n" + md(t))

        # ---- verdict against the registered lines
        key = summ["SH30|DAC|>20|pl"]
        rho = GEO[("SH30", ">20", "mean k")]
        tmean = rho["T"][0]
        v_all = band(key["gross"][0], key["gross"][1], rho["rho"][0] if tmean > 0 else None)
        v_net = band(key["net"][0], key["net"][1], rho["rho"][0] if tmean > 0 else None)
        e05 = summ["SH30|DAC|>20|pl|ext <= 0.5 m"]
        v_e05 = band(e05["gross"][0], e05["gross"][1], rho["rho"][0] if tmean > 0 else None)
        kx = summ["SH30|DAC|>20|plx"]
        v_x = band(kx["gross"][0], kx["gross"][1], rho["rho"][0] if tmean > 0 else None)
        summ["verdict"] = dict(gross=key["gross"], net=key["net"], rho=rho["rho"], T_mean=rho["T"], C_mean=rho["C"], by_gross=v_all, by_net=v_net,
                               ext_le_0p5=dict(gross=e05["gross"], verdict=v_e05), ext_line=dict(gross=kx["gross"], verdict=v_x))
        P("\n## 9. Registered lines\n")
        P(f"- (a) PL removes {pc(key['gross'], '{:.1f}', 100)} % of SH30's DAC failures on > 20 deg tokens (gross; net {pc(key['net'], '{:.1f}', 100)} %): band {v_all[1]} (net: {v_net[1]}).")
        P(f"- (b) rho = mean C / mean T on > 20 deg (k mean) = {pc(rho['rho'], '{:.2f}')}; mean T {pc(rho['T'], '{:+.3f}')} m, mean C {pc(rho['C'], '{:+.3f}')} m: band {v_all[2]}.")
        P(f"- verdict by the registered lines: **{v_all[0]}** (by net: {v_net[0]}; extension <= 0.5 m subset: {v_e05[0]}, gross {pc(e05['gross'], '{:.1f}', 100)} %; "
          f"straight extension: {v_x[0]}, gross {pc(kx['gross'], '{:.1f}', 100)} %).")
        (RES / "tables.md").write_text("\n".join(L) + "\n")
        (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        figures(D2, summ, GEO, RES)
        run.summary.update(verdict=summ["verdict"])
        run.info(json.dumps(summ["verdict"], indent=1, default=float))


def figures(D2, summ, GEO, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    col = {"pp": ps.BASELINE, "pl": ps.PALETTE["orange"], "lp": ps.PALETTE["blue"], "ll": ps.PALETTE["black"]}
    lab = {"pp": "plan path, plan timing", "pl": "plan path, log timing", "lp": "log path, plan timing", "ll": "log"}
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.3))
    for ax, (q, yl) in zip(axs, (("EPDMS", "no-EC EPDMS"), ("DAC", "DAC"), ("NC", "NC"))):
        for j, c in enumerate(("pp", "pl", "lp", "ll")):
            ax.bar(np.arange(4) + (j - 1.5) * 0.2, [D2[("SH30", c, b)][q] for b in TURNS], 0.2, color=col[c], label=lab[c])
        ax.scatter(np.arange(4), [D2[("WA-JEPA", "pp", b)][q] for b in TURNS], marker="_", s=260, color=ps.PALETTE["vermillion"], zorder=3, label="WA-JEPA (stored plan)")
        ax.set_xticks(np.arange(4), [r"$<5$", "5-20", "20-45", r"$>45$"])
        ax.set_xlabel("logged turn (deg)")
        ax.set_ylabel(yl)
        lo = min(min(D2[(f, c, b)][q] for b in TURNS) for f, c in (("SH30", "pp"), ("SH30", "pl"), ("SH30", "lp")))
        ax.set_ylim(np.floor(lo - 1), 100.3)
        ps.bars(ax)
    axs[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, out / "two_by_two")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.3), sharey=True)
    bs = ["20-45", ">45", ">45 left", ">45 right"]
    qc = {"T": ps.BASELINE, "C": ps.PALETTE["blue"], "X": ps.PALETTE["sky_blue"]}
    ql = {"T": "timed points, time-aligned (T)", "C": "curves at equal arc length (C)", "X": "timed points, cross-track (X)"}
    for ax, fam in zip(axs, ("SH30", "WA-JEPA")):
        for j, q in enumerate(("T", "C", "X")):
            v = [GEO[(fam, b, "4 s")][q] for b in bs]
            y = np.array([x[0] for x in v])
            ax.bar(np.arange(len(bs)) + (j - 1) * 0.22, y, 0.22, color=qc[q], label=ql[q],
                   yerr=[y - [x[1] for x in v], [x[2] for x in v] - y], error_kw=dict(lw=0.6, capsize=1.5))
        if fam == "SH30":
            v = [summ[f"E1|{b}|4 s"]["xhat"] for b in bs]
            ax.scatter(np.arange(len(bs)) + 0.0, [x[0] for x in v], marker="D", s=14, color=ps.PALETTE["vermillion"], zorder=3, label="chord bound from timing spread")
        ax.set_xticks(np.arange(len(bs)), ["20-45", r"$>45$", r"$>45$ left", r"$>45$ right"])
        ax.set_xlabel(f"logged turn (deg), {fam}")
        ps.zero_line(ax)
        ps.bars(ax)
    axs[0].set_ylabel("offset at 4 s toward the inside (m)")
    axs[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, out / "offsets")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    p = sp.add_parser("gate")
    p.add_argument("--stage", required=True, choices=["s0", "all"])
    p.add_argument("--score", required=True)
    p = sp.add_parser("select")
    p.add_argument("--score", required=True)
    p = sp.add_parser("replay")
    p.add_argument("--procs", type=int, default=0, help="worker processes (default: a sixth of the cgroup quota)")
    p = sp.add_parser("report")
    p.add_argument("--score", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "gate": cmd_gate, "select": cmd_select, "replay": cmd_replay, "report": cmd_report}[a.cmd](a)
