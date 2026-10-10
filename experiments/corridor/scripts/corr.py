"""CORR0 (experiments/corridor, plans/2026-10-10-corr0-prereg.md, decision 240): is the lane corridor of the commanded exit the right
supervision label for turn failures? Numbers 1-3 on navtest from the cached map labels (corr_geom.py build), the stored SH30 plans and
the stored scores of decision 207; number 4 from corr_geom.py supply. CPU only, no model runs. The map, the logged future and the
driven lane sequence are privileged: every arm below is an analysis swap, not a method and not a reportable inference path.

  poses   (.venv)  arms per SH30 seed -> $OUT/poses.npz: pp stored plan; cp corridor-centred path at plan timing; ce = cp + the plan's own
                   equal-arc curve error; kp / ke the same on the logged path clamped into the corridor interior. feat.npz: the
                   along-track / cross-track fits (number 1) and the map heading errors (number 3). tokens_{s10,turn}.txt, keys.txt
  gate    (.venv)  identity gate: <seed>_pp rows of a score-poses CSV == the stored bench sub-scores, token by token
  report  (.venv)  tables for the four numbers, read lines, figures -> $OUT/report/
Scoring is `python -m jevdrive.bench score-poses --traffic non_reactive` (corr_chain.sh). Curves, extension rule, failure kinds and the
log-cluster bootstrap are decision 207's (experiments/op_parity/scripts/pt_swap.py), imported, not re-implemented.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts"), str(REPO / "research")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/corridor"
RES = OUT / "report"
PT = D / "runs/op_parity/pt_swap"                       # decision 207: stored PP / LP / LL scores, replay sides, plan poses
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
SEEDS = {"sh0": "SH30-F-s0", "sh1": "SH30-F-s1"}
ARMS = ("pp", "cp", "ce", "kp", "ke")
SUB8 = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC"]
HALF_W, MARGIN, DS = 1.1485, 0.2, 0.05
DELTAS, OFFS = np.round(np.arange(-6, 6.001, 0.1), 2), np.round(np.arange(-3, 3.001, 0.05), 2)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Line:
    """Dense polyline: arc length, tangent heading (or a given yaw), straight beyond the end."""

    def __init__(self, xy, yaw=None):
        xy = np.asarray(xy, np.float64)
        keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-6]
        self.xy = xy[keep]
        d = np.diff(self.xy, axis=0)
        self.S = np.r_[0, np.cumsum(np.hypot(*d.T))]
        h = np.unwrap(np.arctan2(d[:, 1], d[:, 0])) if len(d) else np.zeros(0)
        self.h = np.r_[h, h[-1:]] if len(h) else np.zeros(len(self.xy))
        self.yaw = self.h if yaw is None else np.asarray(yaw, np.float64)[keep]
        self.L = float(self.S[-1])

    def at(self, s):
        s = np.atleast_1d(np.asarray(s, np.float64))
        si = np.clip(s, 0, self.L)
        o = np.stack([np.interp(si, self.S, self.xy[:, 0]), np.interp(si, self.S, self.xy[:, 1]), np.interp(si, self.S, self.yaw)], 1)
        m = s > self.L
        o[m, 0] += (s[m] - self.L) * np.cos(self.h[-1])
        o[m, 1] += (s[m] - self.L) * np.sin(self.h[-1])
        return o

    def frenet(self, pts):
        a, ab = self.xy[:-1], np.diff(self.xy, axis=0)
        ap = pts[:, None] - a[None]
        t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        d = ap - t[..., None] * ab
        j = np.hypot(d[..., 0], d[..., 1]).argmin(1)
        r = np.arange(len(pts))
        cr = ab[j, 0] * ap[r, j, 1] - ab[j, 1] * ap[r, j, 0]
        return self.S[j] + t[r, j] * np.hypot(*ab[j].T), np.sign(cr) * np.hypot(d[r, j, 0], d[r, j, 1])


def add_error(base, p, l):
    """base, p, l (8, 3): base pose + (p - l) expressed in l's local frame, applied in base's local frame."""
    d = p[:, :2] - l[:, :2]
    et = d[:, 0] * np.cos(l[:, 2]) + d[:, 1] * np.sin(l[:, 2])
    en = -d[:, 0] * np.sin(l[:, 2]) + d[:, 1] * np.cos(l[:, 2])
    c, s = np.cos(base[:, 2]), np.sin(base[:, 2])
    return np.stack([base[:, 0] + et * c - en * s, base[:, 1] + et * s + en * c, base[:, 2] + (p[:, 2] - l[:, 2])], 1)


def corridor_path(g):
    """CP: the centreline R with the ego's lateral offset d0 blended out over the first `blend` m (smoothstep)."""
    R = Line(g["R"])
    n = np.stack([-np.sin(R.h), np.cos(R.h)], 1)
    return R, Line(R.xy + n * (g["d0"] * (1 - smooth(R.S / g["blend"])))[:, None])


def clamped_log(g, R, lc, upto):
    """KP: the logged curve with its lateral offset from R clamped to the corridor interior (half lane width - half ego width - margin);
    the shift is blended in over the first `blend` m so the path still starts at the ego."""
    u = np.arange(0, upto + DS, DS)
    q = lc.at(u)
    sig, d = R.frenet(q[:, :2])
    lim = np.maximum(np.interp(sig, R.S, g["hw"]) - HALF_W - MARGIN, 0.0)
    sh = (np.clip(d, -lim, lim) - d) * smooth(u / g["blend"])
    hR = np.interp(sig, R.S, R.h)
    xy = q[:, :2] + np.stack([-np.sin(hR), np.cos(hR)], 1) * sh[:, None]
    t0 = np.unwrap(np.arctan2(*np.diff(q[:, :2], axis=0).T[::-1]))
    t1 = np.unwrap(np.arctan2(*np.diff(xy, axis=0).T[::-1]))
    dy = np.r_[t1 - t0, (t1 - t0)[-1:]]
    return Line(xy, q[:, 2] + (dy + np.pi) % (2 * np.pi) - np.pi), float(np.abs(sh).max()), float(np.abs(d).max())


def fits(pc, lc):
    """Number 1: residual RMS (m) of the plan curve against the log curve (r0), against the log curve advanced / delayed along its own
    track by delta (model A: min over DELTAS), and against the log curve offset in parallel by c (model C: min over OFFS)."""
    Lc = min(pc.sv[-1], lc.sv[-1])
    if Lc < 2.0:
        return dict(r0=np.nan, rA=np.nan, rC=np.nan, delta=np.nan, c=np.nan, Lc=Lc)
    s = np.arange(0, Lc + 1e-9, DS)
    P, L = pc.at(s)[:, :2], lc.at(s)
    r0 = float(np.sqrt(((P - L[:, :2]) ** 2).sum(1).mean()))
    n = np.stack([-np.sin(L[:, 2]), np.cos(L[:, 2])], 1)
    rC = np.sqrt((((P - L[:, :2])[None] - OFFS[:, None, None] * n[None]) ** 2).sum(-1).mean(1))
    rA = np.empty(len(DELTAS))
    for k, dl in enumerate(DELTAS):
        if dl >= 0:                                   # late: straight for delta, then the log curve
            A = lc.at(np.maximum(s - dl, 0))[:, :2]
            A[:, 0] += np.minimum(s, dl)
        else:                                         # early: the log curve from |delta| on, re-anchored at the origin
            x0, y0, h0 = lc.at([-dl])[0]
            q = lc.at(s - dl)[:, :2] - [x0, y0]
            A = np.stack([q[:, 0] * np.cos(h0) + q[:, 1] * np.sin(h0), -q[:, 0] * np.sin(h0) + q[:, 1] * np.cos(h0)], 1)
        rA[k] = np.sqrt(((P - A) ** 2).sum(1).mean())
    return dict(r0=r0, rA=float(rA.min()), rC=float(rC.min()), delta=float(DELTAS[rA.argmin()]), c=float(OFFS[rC.argmin()]), Lc=Lc)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


_P = {}


def _pose_chunk(idx):
    """Arms, fits and heading errors of a chunk of token indices (worker of cmd_poses; inputs in _P, inherited by fork)."""
    import pt_swap as PS
    G, names, fut, Z = _P["G"], _P["names"], _P["fut"], _P["Z"]
    out = []
    for i in idx:
        g = G[names[i]]
        lc = PS.Curve(fut[i])
        R, CP = corridor_path(g)
        F = dict(lane_change=g["lane_change"], d0=g["d0"], gap=g.get("gap", 0), n_conn=g["n_conn"], exit_h=g.get("exit_h", np.nan),
                 hR_log=wrap(R.at([lc.sv[-1]])[0, 2] - fut[i, -1, 2]), dR_log4=R.frenet(fut[i, -1:, :2])[1][0])
        A = {}
        for m in SEEDS:
            p = Z[m][i].astype(np.float64)
            pc = PS.Curve(p)
            sk = pc.sv[1:]
            pk, lk = pc.at(sk), lc.at(sk)
            cp = CP.at(sk)
            K, F["kp_shift"], F["log_dmax"] = clamped_log(g, R, lc, max(pc.sv[-1], lc.L) + 1.0)
            kp = K.at(sk)
            A |= {f"{m}_cp": cp, f"{m}_ce": add_error(cp, pk, lk), f"{m}_kp": kp, f"{m}_ke": add_error(kp, pk, lk)}
            F |= {f"{m}_{k}": v for k, v in fits(pc, lc).items()}
            F[f"{m}_hR_plan"] = wrap(R.at([pc.sv[-1]])[0, 2] - fut[i, -1, 2])
            F[f"{m}_h_plan"] = wrap(np.unwrap(np.r_[0, p[:, 2]])[-1] - fut[i, -1, 2])
        out.append((int(i), F, A))
    return out


# ---------------------------------------------------------------- poses
def cmd_poses(a):
    import pt_swap as PS
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", "poses", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        tab, Z = np.load(TAB), np.load(PT / "poses.npz")
        names, fut = tab["names"].astype(str), tab["fut"].astype(np.float64)
        n = len(names)
        G = {r["token"]: r for r in pickle.load(open(OUT / a.geom, "rb"))}
        ok = np.array([t in G and G[t]["status"] == "ok" for t in names])
        arrs = {f"{m}_pp": Z[f"{m}_pp"] for m in SEEDS}
        F = dict(names=names, ok=ok, has=np.array([t in G for t in names]), status=np.array([G[t]["status"] if t in G else "" for t in names]), lane_change=np.zeros(n, bool), d0=np.full(n, np.nan),
                 gap=np.zeros(n, int), exit_h=np.full(n, np.nan), n_conn=np.zeros(n, int), kp_shift=np.full(n, np.nan), log_dmax=np.full(n, np.nan),
                 hR_log=np.full(n, np.nan), dR_log4=np.full(n, np.nan))
        for m in SEEDS:
            for k in ARMS[1:]:
                arrs[f"{m}_{k}"] = Z[f"{m}_pp"].astype(np.float32).copy()
            for k in ("r0", "rA", "rC", "delta", "c", "Lc", "hR_plan", "h_plan"):
                F[f"{m}_{k}"] = np.full(n, np.nan)
            F[f"{m}_cls"] = np.array([G[t].get(f"{m}_cls", "") if t in G else "" for t in names])
            F[f"{m}_cls2"] = np.array([G[t].get(f"{m}_cls2", "") if t in G else "" for t in names])
            F[f"{m}_endgap"] = np.array([G[t].get(f"{m}_endgap", 0) if t in G else 0 for t in names])
        from jevdrive import par
        from scipy.interpolate import CubicSpline  # noqa: F401  (imported before the fork: workers must not import scipy concurrently)
        _P.update(G=G, names=names, fut=fut, Z={m: Z[f"{m}_pp"] for m in SEEDS})
        idx = np.flatnonzero(ok)
        res = par.pmap(_pose_chunk, [idx[k::256] for k in range(min(256, len(idx)))], run=run, desc="token chunks")
        res.raise_if_failed()
        for ch in res.values:
            for i, fa, ar in ch:
                for k, v in fa.items():
                    F[k][i] = v
                for k, v in ar.items():
                    arrs[k][i] = v
        tag = a.tag
        np.savez(OUT / f"poses{tag}.npz", tokens=names, **arrs)
        np.savez_compressed(OUT / f"feat{tag}.npz", **F)
        turn = np.degrees(np.abs(fut[:, -1, 2])) > 20
        (OUT / f"tokens{tag}.txt").write_text("\n".join(sorted(names[ok & turn])) + "\n")
        (OUT / "keys.txt").write_text(" ".join(arrs) + "\n")
        run.summary.update(tokens=int(ok.sum()), turn=int(turn.sum()), turn_ok=int((ok & turn).sum()), keys=len(arrs))
        run.info(json.dumps(run.summary))


# ---------------------------------------------------------------- gate
def cmd_gate(a):
    import pandas as pd
    import pt_swap as PS
    from jevdrive.bench import tables as T
    from jevdrive.bench.navsim import SUBS
    from jevdrive.run import Run
    with Run("corridor", "gate", config=vars(a)) as run:
        df = pd.read_csv(a.score)
        res, ok = {}, True
        for m, spec in SEEDS.items():
            g = df[df.key == f"{m}_pp"].set_index("token")
            u, src = PS.archive(spec)
            r = dict(source=str(src), tokens=len(g))
            for k in SUB8:
                dl = np.abs(g[SUBS[k]].to_numpy(float) - u.loc[g.index, k].to_numpy(float))
                r[k] = dict(max_abs=float(dl.max()), n_diff=int((dl > (PS.EP_TOL if k == "EP" else 0)).sum()))
            X = u.loc[g.index, T.TERMS].to_numpy(float)
            X[:, 8] = np.nan
            dl = np.abs(g.score.to_numpy(float) - T.score_of(X))
            r["score_noec"] = dict(max_abs=float(dl.max()), n_diff=int((dl > PS.EP_TOL).sum()))
            r["pass"] = all(v["n_diff"] == 0 for v in r.values() if isinstance(v, dict))
            ok &= r["pass"]
            res[m] = r
        (OUT / f"gate{a.tag}.json").write_text(json.dumps(res, indent=1))
        run.summary.update(passed=bool(ok), tokens={m: r["tokens"] for m, r in res.items()})
        run.info(json.dumps(run.summary))
        if not ok:
            raise SystemExit("identity gate failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poses"); p.add_argument("--geom", default="geom.pkl"); p.add_argument("--tag", default="")
    p = sub.add_parser("gate"); p.add_argument("--score", required=True); p.add_argument("--tag", default="")
    p = sub.add_parser("report"); p.add_argument("--score", required=True)
    a = ap.parse_args()
    if a.cmd == "report":
        import corr_report
        corr_report.main(a)
    else:
        {"poses": cmd_poses, "gate": cmd_gate}[a.cmd](a)
