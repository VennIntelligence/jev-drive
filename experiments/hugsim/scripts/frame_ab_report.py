#!/usr/bin/env python
"""Frame A/B, stage 3: tables from the replayed plans (frame_ab_onnx.py, frame_ab_torch.py). CPU only.
experiments/hugsim/plans/2026-10-11-frame-ab-prereg.md, results/frame_ab.md.

Every arm's plan is put on the simulator clock in HUGSIM plan axes (x right, y forward, origin at the camera) at 0.5 .. 4 s with the
formula of jevdrive.hugsim_zs.openpilot_to_plan (dilation 1.25 for the served arms and `wd`, 1 for `b`). Per pair of arms and stratum:
displacement at 1 / 2 / 3 / 4 s, arc-length ratio (ratio of sums), lateral offset at 3 s, action curvature. CIs: jevdrive.stats, cluster
bootstrap over scenarios; ratios use the same resampling rule (clusters drawn with stats' seed and count, ratio of the drawn sums).

  .venv/bin/python experiments/hugsim/scripts/frame_ab_report.py --run-dir <bench run dir> --out <dir>
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive import stats  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
TQ = 0.5 * np.arange(1, 9)
DIL = {"log": 1.25, "off": 1.25, "nod": 1.25, "zb": 1.25, "a_t": 1.25, "s8": 1.25, "s8z": 1.25, "wd": 1.25, "b": 1.0}
PAIRS = [("log", "nod", "desire pulse removed"), ("nod", "a_t", "torch port of the served queue (residual)"), ("a_t", "s8", "9th slot dropped"),
         ("s8", "s8z", "oldest pair from a zero image"), ("a_t", "s8z", "9 -> 8 slots (both)"), ("s8z", "wd", "pixel source: raw pairs -> warp lattice (M1, model clock)"),
         ("wd", "b", "clock and key grid: M1 -> M2 (real clock)"), ("log", "wd", "served -> warp lattice on the model clock (M1)"),
         ("log", "b", "served -> training protocol (M2), total")]


def to_plan(pos, dil):
    """(N, 33, 3) model plan (x fwd, y right, model time) -> (N, 8, 2) x right, y forward at TQ simulator seconds."""
    tau = TQ / dil
    return np.stack([np.stack([np.interp(tau, T_IDXS, p[:, 1]), np.interp(tau, T_IDXS, p[:, 0])], -1) for p in pos])


def arc(p, k):
    q = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :k]], 1)
    return np.linalg.norm(np.diff(q, axis=1), axis=-1).sum(1)


def smooth_kappa(pos, yaw, t0=0.5, t1=1.5):
    """jevdrive.openpilot.model.curvature_window on the arm's own model clock, sign as the action curvature (+ = right)."""
    s = np.concatenate([np.zeros((len(pos), 1)), np.cumsum(np.linalg.norm(np.diff(pos[:, :, :2], axis=1), axis=-1), 1)], 1)
    f = lambda A, t: np.array([np.interp(t, T_IDXS, r) for r in A])  # noqa: E731
    return (f(yaw, t1) - f(yaw, t0)) / np.maximum(f(s, t1) - f(s, t0), 1.0 * (t1 - t0))


def ratio_ci(num, den, groups):
    """sum(num) / sum(den) with the cluster bootstrap of jevdrive.stats.bootstrap (same seed, count and percentile rule)."""
    codes, uniq = pd.factorize(np.asarray(groups))
    a, b = np.bincount(codes, num, len(uniq)), np.bincount(codes, den, len(uniq))
    idx = np.random.default_rng(stats.SEED).integers(len(uniq), size=(stats.N_BOOT, len(uniq)))
    lo, hi = np.quantile(a[idx].sum(1) / b[idx].sum(1), [stats.ALPHA / 2, 1 - stats.ALPHA / 2])
    return dict(mean=float(num.sum() / den.sum()), lo=float(lo), hi=float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    cli_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    with Run("hugsim_frame_ab", "report", seed=a.seed, config=vars(a), resume=a.resume) as run:
        units = pd.read_csv(Path(a.run_dir) / "units.csv")
        units["dir"] = units.run_dir.map(lambda s: Path(s).name)
        U = units.set_index("dir")
        D = []                                                     # one row per decision
        P, K, KS = {}, {}, {}                                      # arm -> plans (n, 8, 2), action curvature, smooth curvature
        for f in sorted((out / "onnx").glob("*.npz")):
            zo, zt = np.load(f), np.load(out / "torch" / f.name)
            st, N = zo["state"], len(zo["state"])
            assert np.array_equal(st, zt["state"])
            v = st[:, 4]
            hy = np.degrees(st[:, 3] - st[np.maximum(np.arange(N) - 6, 0), 3])
            for arm in DIL:
                z, key = (zo, arm) if arm in ("log", "off", "nod", "zb") else (zt, arm)
                pos, yaw, act = z[f"pos_{key}"], z[f"yaw_{key}"], z[f"act_{key}"]
                kap = act[:, 0] if z is zo else act[:, 0] / np.maximum(1.0, v * DIL[arm]) ** 2
                P.setdefault(arm, []).append(to_plan(pos, DIL[arm]))
                K.setdefault(arm, []).append(kap)
                KS.setdefault(arm, []).append(smooth_kappa(pos, yaw))
            u = U.loc[f.stem]
            cls = "launch_stall" if bool(u.launch_stall) else str(u.cls)
            D.append(pd.DataFrame(dict(scenario=f.stem, dataset=u.dataset, cls=cls, step=np.arange(N), t=st[:, 0], v=v, hyaw15=hy, cmd=st[:, 6].astype(int),
                                       nkeys_b=zt["nkeys_b"], nkeys_wd=zt["nkeys_wd"])))
        D = pd.concat(D, ignore_index=True)
        P, K, KS = ({k: np.concatenate(x) for k, x in d.items()} for d in (P, K, KS))
        D["warm"] = D.step >= 9
        D["turn"] = (np.abs(K["log"]) > 0.01) | (D.hyaw15.abs() > 5)
        D["vbin"] = pd.cut(D.v, [-1, 2, 5, 99], labels=["< 2 m/s", "2-5 m/s", "> 5 m/s"]).astype(str)
        D["tbin"] = pd.cut(D.step, [-1, 8, 20, 10 ** 6], labels=["0-2 s (warm-up)", "2.25-5 s", "> 5 s"]).astype(str)
        g = D.scenario.values

        def row(x, y, m, **lab):
            """Arm y against arm x on the decisions of mask m."""
            m = np.asarray(m)
            d = np.linalg.norm(P[y][m] - P[x][m], axis=-1)                           # (n, 8)
            gg = g[m]
            r = dict(lab, n=int(m.sum()), scenarios=int(len(set(gg))))
            if r["scenarios"] < 2:
                ci = lambda z: dict(mean=float(np.mean(z)), lo=np.nan, hi=np.nan)  # noqa: E731
                rc = lambda nn, dd: dict(mean=float(nn.sum() / dd.sum()), lo=np.nan, hi=np.nan)  # noqa: E731
            else:
                ci = lambda z: stats.bootstrap(z, gg)  # noqa: E731
                rc = lambda nn, dd: ratio_ci(nn, dd, gg)  # noqa: E731
            for name, z in (("d1", d[:, 1]), ("d2", d[:, 3]), ("d3", d[:, 5]), ("d4", d[:, 7]), ("ade3", d[:, :6].mean(1)),
                            ("lat3", P[y][m][:, 5, 0] - P[x][m][:, 5, 0]), ("abs_lat3", np.abs(P[y][m][:, 5, 0] - P[x][m][:, 5, 0])),
                            ("lon3", P[y][m][:, 5, 1] - P[x][m][:, 5, 1]),
                            ("dkappa", K[y][m] - K[x][m]), ("abs_dkappa", np.abs(K[y][m] - K[x][m])), ("abs_dkappa_smooth", np.abs(KS[y][m] - KS[x][m]))):
                c = ci(z)
                r.update({name: c["mean"], name + "_lo": c["lo"], name + "_hi": c["hi"]})
            r.update(d3_median=float(np.median(d[:, 5])), d3_p90=float(np.quantile(d[:, 5], 0.9)))
            for name, k in (("arc3", 6), ("arc4", 8)):
                c = rc(arc(P[y][m], k), arc(P[x][m], k))
                r.update({name: c["mean"], name + "_lo": c["lo"], name + "_hi": c["hi"]})
            return r

        warm, wu = D.warm.values, ~D.warm.values
        chain = [row(x, y, m, pair=f"{x} -> {y}", what=w, decisions=s) for s, m in (("warm", warm), ("warm-up", wu)) for x, y, w in PAIRS]
        stats.write_table(chain, rep / "chain")
        per = [row("log", y, m & (g == s), scenario=s, dataset=U.loc[s].dataset, cls=D.cls[g == s].iloc[0], arm=y, decisions=lab)
               for s in sorted(set(g)) for y in ("b", "wd") for lab, m in (("warm", warm), ("warm-up", wu)) if (m & (g == s)).sum()]
        stats.write_table(per, rep / "per_scenario")
        where = []
        for x, y in (("log", "b"), ("log", "wd"), ("s8z", "wd"), ("a_t", "s8z")):
            for col in ("vbin", "turn", "tbin", "cls", "dataset"):
                for val in sorted(D[col].astype(str).unique()):
                    m = (D[col].astype(str) == val).values & (warm if col != "tbin" else True)
                    if m.sum():
                        where.append(row(x, y, m, pair=f"{x} -> {y}", by=col, value=val))
            for lab, m in (("launch: v < 2 m/s, warm", warm & (D.v < 2).values), ("turning, v > 2 m/s, warm", warm & D.turn.values & (D.v >= 2).values),
                           ("straight, v > 5 m/s, warm", warm & ~D.turn.values & (D.v > 5).values)):
                if m.sum():
                    where.append(row(x, y, m, pair=f"{x} -> {y}", by="named", value=lab))
        stats.write_table(where, rep / "where")
        prof = [dict(step=n, n=int((D.step == n).sum()), **{f"d3_{x}_{y}": float(np.linalg.norm(P[y][(D.step == n).values][:, 5] - P[x][(D.step == n).values][:, 5], axis=-1).mean())
                                                             for x, y in (("log", "b"), ("log", "wd"), ("log", "a_t"), ("a_t", "s8z"), ("s8z", "wd"))},
                     **{f"arc3_{y}_over_log": float(arc(P[y][(D.step == n).values], 6).sum() / max(arc(P["log"][(D.step == n).values], 6).sum(), 1e-9)) for y in ("b", "wd", "s8z")},
                     nkeys_b=float(D.nkeys_b[D.step == n].mean()), v=float(D.v[D.step == n].mean())) for n in range(0, 16)]
        stats.write_table(prof, rep / "warmup_profile")
        # plan length against the fed speed (context): arc over 3 simulated s / (3 s x simulator speed), warm, v > 3 m/s
        mv = warm & (D.v > 3).values
        length = [dict(arm=y, n=int(mv.sum()), **ratio_ci(arc(P[y][mv], 6), 3.0 * D.v.values[mv], g[mv])) for y in ("log", "nod", "a_t", "s8", "s8z", "wd", "b")]
        stats.write_table(length, rep / "length_vs_speed")
        gates = {k: json.loads((out / f"gate_{k}.json").read_text()) for k in ("i", "ii", "iii")}
        for v in gates.values():
            v.pop("rows", None)
        key = {r["pair"] + " | " + r["decisions"]: {k: r[k] for k in ("n", "scenarios", "d3", "d3_lo", "d3_hi", "arc3", "arc3_lo", "arc3_hi", "abs_lat3", "abs_dkappa")} for r in chain}
        (rep / "summary.json").write_text(json.dumps(dict(decisions=int(len(D)), warm=int(warm.sum()), warmup=int(wu.sum()), scenarios=int(D.scenario.nunique()),
                                                          gates=gates, chain=key), indent=1))
        D.to_csv(rep / "decisions.csv", index=False)
        run.summary.update(decisions=int(len(D)), out=str(rep))
        run.info("%s", json.dumps(key["log -> b | warm"]))


if __name__ == "__main__":
    main()
