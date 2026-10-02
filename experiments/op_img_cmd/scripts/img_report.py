"""Image-command zero-shot: metrics, cluster-bootstrap CIs and the pre-registered verdicts (any env with numpy / pandas).

Per run (img_run.py raw npz) and sample (geom pkl): plan point x(tau) at tau = 2, 3, 4 s; d_c(x) = distance to the
commanded-class path Q_c (approach + branch c); m_ab(x) = d_a(x) - d_b(x). Delta (per family, sample) = mean over
ordered command pairs (a, b) of m_ab(P_b) - m_ab(P_a): positive = the plan moves toward the commanded branch. Heading
version: (yaw_b - yaw_a) * sign(psi_b - psi_a), psi_c the tangent of Q_c at the plan's arc length. Correct = the plan
point is nearer to the commanded path than to every other class, counted where the commanded path is >= 1.5 m from the
others at that point. Straight frames: lateral error at 3 s against the logged future, overlay - none.

  python experiments/op_img_cmd/scripts/img_report.py --domain nav [--glob 'nav-all-*of2.npz']
Writes <out>/<domain>_{per_sample.csv, straight_per_sample.csv, effects.csv, effects.md}.
"""
import argparse, glob, os, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import img_overlay as O  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
TAUS = (2.0, 3.0, 4.0, 6.0)
SEP_M = 1.5
B = 2000


def interp(P, tau):
    return np.array([np.interp(tau, T_IDXS, P[:, k]) for k in range(P.shape[1])])


def near(Q, x):
    """distance from x to polyline Q (dense, 0.5 m) and the tangent heading at the nearest vertex."""
    d = np.linalg.norm(Q - x, axis=1)
    k = int(np.argmin(d))
    t = Q[min(k + 1, len(Q) - 1)] - Q[max(k - 1, 0)]
    return d[k], np.arctan2(t[1], t[0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def sample_rows(s, runs):
    """runs: {(fam, cmd): (plan_pos (33, 2), plan_yaw (33))} -> rows (fam, tau, delta, dpsi, n_correct, n_decisive, dx)."""
    cls = sorted({c for f, c in runs if c})
    Q = {c: O.full_centre(s, O.cmd_path(s, c)) for c in cls}
    none = runs[("none", "")]
    rows = []
    fams = sorted({f for f, c in runs})
    for tau in TAUS:
        x0 = interp(none[0], tau)
        for fam in fams:
            pl = {c: runs[(fam, c)] for c in cls if (fam, c) in runs}
            if fam == "none":
                pl = {c: none for c in cls}
            if fam in O.ROUTE_FREE:
                P, Y = runs[(fam, "")]
                xf = interp(P, tau)
                rows.append(dict(fam=fam, tau=tau, dx=xf[0] - x0[0], dy=xf[1] - x0[1]))
                continue
            if len(pl) < 2:
                continue
            X = {c: interp(pl[c][0], tau) for c in pl}
            Yw = {c: np.interp(tau, T_IDXS, pl[c][1]) for c in pl}
            dl, dp = [], []
            for a in pl:
                for b in pl:
                    if a == b:
                        continue
                    m = lambda x: near(Q[a], x)[0] - near(Q[b], x)[0]  # noqa: E731
                    dl.append(m(X[b]) - m(X[a]))
                    # heading: branch tangents at the arc length reached by the commanded plans
                    pa, pb = near(Q[a], X[a])[1], near(Q[b], X[b])[1]
                    dp.append(np.degrees(wrap(Yw[b] - Yw[a]) * np.sign(wrap(pb - pa))) if abs(wrap(pb - pa)) > np.radians(3) else np.nan)
            nc = nd = 0
            for c in pl:
                dc = near(Q[c], X[c])[0]
                do = [near(Q[o], X[c])[0] for o in pl if o != c]
                # decisive: the commanded path is >= SEP_M from the others near this point
                sep = min(near(Q[o], near_pt(Q[c], X[c]))[0] for o in pl if o != c)
                if sep >= SEP_M:
                    nd += 1
                    nc += int(dc < min(do))
            # fixed set (decided on the `none` plan point x0, the same for every family): correct rate and uptake
            nfc = nfd = 0
            up_num = up_den = 0.0
            tw = {}
            for c in pl:
                p0 = near_pt(Q[c], x0)
                sep = min(near(Q[o], p0)[0] for o in pl if o != c)
                if sep >= SEP_M:
                    nfd += 1
                    nfc += int(near(Q[c], X[c])[0] < min(near(Q[o], X[c])[0] for o in pl if o != c))
                    for o in pl:
                        if o != c:
                            m = lambda x: near(Q[o], x)[0] - near(Q[c], x)[0]  # noqa: E731
                            up_num += m(X[c]) - m(x0)
                            up_den += 2 * near(Q[o], p0)[0]
                tw[c] = np.mean([(near(Q[o], X[c])[0] - near(Q[c], X[c])[0]) - (near(Q[o], x0)[0] - near(Q[c], x0)[0]) for o in pl if o != c])
            rows.append(dict(fam=fam, tau=tau, delta=np.mean(dl), dpsi=np.nanmean(dp) if np.isfinite(dp).any() else np.nan,
                             n_correct=nc, n_decisive=nd, n_fcorrect=nfc, n_fdecisive=nfd, up_num=up_num, up_den=up_den,
                             dx=np.mean([X[c][0] for c in X]) - x0[0], **{f"tw_{c}": v for c, v in tw.items()}))
    return rows


def near_pt(Q, x):
    return Q[int(np.argmin(np.linalg.norm(Q - x, axis=1)))]


def straight_rows(s, runs):
    fut = np.asarray(s["future"])
    if len(fut) < 6:
        return []
    g = fut[5, :2]                                    # logged position at 3 s
    rows = []
    x0 = interp(runs[("none", "")][0], 3.0)
    e0 = abs(x0[1] - g[1])
    for (fam, c), (P, Y) in runs.items():
        x = interp(P, 3.0)
        rows.append(dict(fam=fam, tau=3.0, dlat_err=abs(x[1] - g[1]) - e0, dx=x[0] - x0[0], dy=x[1] - x0[1]))
    return rows


def boot(df, col, cl="log"):
    """mean and 95% cluster-bootstrap CI over clusters."""
    d = df[[cl, col]].dropna()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    g = d.groupby(cl)[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(0)
    k = rng.integers(0, len(g), (B, len(g)))
    bs = s[k].sum(1) / n[k].sum(1)
    return s.sum() / n.sum(), *np.percentile(bs, [2.5, 97.5]), len(d)


def ratio_boot(df, num, den, cl="log"):
    g = df.groupby(cl)[[num, den]].sum()
    a, b = g[num].to_numpy(), g[den].to_numpy()
    if b.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(0)
    k = rng.integers(0, len(g), (B, len(g)))
    bs = a[k].sum(1) / np.maximum(b[k].sum(1), 1)
    return a.sum() / b.sum(), *np.percentile(bs, [2.5, 97.5]), int(b.sum())


def vbin(v):
    return "stop" if v < 0.5 else "low" if v < 3 else "moving"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="nav")
    ap.add_argument("--glob", nargs="+", default=["nav-all-*of2.npz", "nav-junction-*of1-combo.npz"])
    ap.add_argument("--out", default="", help="default $DATA_DIR/runs/op_img_cmd/report (copied into results/ on the Mac)")
    a = ap.parse_args()
    root = data_dir() / "runs" / "op_img_cmd"
    G = {s["token"]: s for s in pickle.load(open(root / "geom" / f"{a.domain}.pkl", "rb"))}
    raw = {}
    for f in sorted(x for g in a.glob for x in glob.glob(str(root / "raw" / a.domain / g))):
        R = np.load(f)
        for i in range(len(R["token"])):
            raw.setdefault(str(R["token"][i]), {})[(str(R["fam"][i]), str(R["cmd"][i]))] = (R["plan_pos"][i], R["plan_yaw"][i])
    rows, srows = [], []
    for tok, runs in raw.items():
        s = G[tok]
        base = dict(token=tok, log=s["log"], v=s["v"], vbin=vbin(s["v"]), classes=s["classes"], taken=s["taken"], dist=s["dist"])
        if s["kind"] == "junction":
            rows += [dict(base, **r) for r in sample_rows(s, runs)]
        else:
            srows += [dict(base, **r) for r in straight_rows(s, runs)]
    J, S = pd.DataFrame(rows), pd.DataFrame(srows)
    out = Path(a.out) if a.out else root / "report"
    out.mkdir(parents=True, exist_ok=True)
    J.to_csv(out / f"{a.domain}_per_sample.csv", index=False)
    S.to_csv(out / f"{a.domain}_straight_per_sample.csv", index=False)
    E = []
    fams = ["none"] + [f for f in O.FAMILIES if f not in O.ROUTE_FREE]
    for tau in TAUS:
        for fam in fams + list(O.ROUTE_FREE):
            for vb in ("all", "stop", "low", "moving"):
                d = J[(J.fam == fam) & (J.tau == tau) & ((J.vbin == vb) if vb != "all" else True)]
                if not len(d):
                    continue
                r = dict(tau=tau, fam=fam, vbin=vb, n=d.token.nunique(), logs=d.log.nunique())
                if fam not in O.ROUTE_FREE:
                    r.update(zip(("delta", "delta_lo", "delta_hi", "_n"), boot(d, "delta")))
                    r.update(zip(("dpsi", "dpsi_lo", "dpsi_hi", "_n2"), boot(d, "dpsi")))
                    r.update(zip(("correct", "correct_lo", "correct_hi", "decisive"), ratio_boot(d, "n_correct", "n_decisive")))
                    r.update(zip(("fcorrect", "fcorrect_lo", "fcorrect_hi", "fdecisive"), ratio_boot(d, "n_fcorrect", "n_fdecisive")))
                    r.update(zip(("uptake", "uptake_lo", "uptake_hi", "_n4"), ratio_boot(d, "up_num", "up_den")))
                    for c in ("left", "straight", "right"):
                        if f"tw_{c}" in d:
                            r.update(zip((f"tw_{c}", f"tw_{c}_lo", f"tw_{c}_hi", f"n_{c}"), boot(d, f"tw_{c}")))
                r.update(zip(("dx", "dx_lo", "dx_hi", "_n3"), boot(d, "dx")))
                E.append(r)
    E = pd.DataFrame(E).drop(columns=[c for c in ("_n", "_n2", "_n3", "_n4") if c in E])
    for fam in ["band", "lines", "arrow_road", "sign"]:
        d = S[S.fam == fam]
        if len(d):
            m, lo, hi, n = boot(d, "dlat_err")
            E = pd.concat([E, pd.DataFrame([dict(tau=3.0, fam=f"straight:{fam}", vbin="all", n=n, logs=d.log.nunique(),
                                                  dlat=m, dlat_lo=lo, dlat_hi=hi, dx=d.dx.mean())])])
    E.to_csv(out / f"{a.domain}_effects.csv", index=False)

    def verdict(r):
        if r.fam == "none" or not np.isfinite(r.get("delta", np.nan)):
            return ""
        if r.delta_lo > 0:
            return "works" if r.fcorrect >= 0.75 and r.delta >= 0.5 else "partial"
        return "reverse" if r.delta_hi < 0 else "none"

    f = lambda x: "" if not np.isfinite(x) else f"{x:+.2f}"  # noqa: E731
    lines = [f"# {a.domain}: image-command zero-shot effects", "",
             f"n samples: junction {J.token.nunique()} ({J.log.nunique()} logs), straight {S.token.nunique()}. "
             "Delta = paired move toward the commanded branch (m); dpsi in deg; correct = fraction of (sample, command) plans nearer the "
             "commanded branch, on the fixed set where the branches are >= 1.5 m apart at the `none` plan point (none = chance); "
             "uptake = sum of the move toward the command / sum of 2 x branch separation on that set (1 = full switch); toward = "
             "move toward class c vs none when c is commanded; dx = plan x change vs none (m). 95% cluster bootstrap by log.", ""]
    for tau in TAUS:
        lines += [f"## tau = {tau:g} s", "", "| family | vbin | n | Delta (m) | dpsi (deg) | correct (fixed set) | n | uptake | toward L / S / R (m) | dx (m) | verdict |",
                  "|:--|:--|--:|:--|:--|:--|--:|:--|:--|:--|:--|"]
        for _, r in E[(E.tau == tau) & ~E.fam.str.startswith("straight")].iterrows():
            g = lambda k: r.get(k, np.nan)  # noqa: E731
            lines.append(f"| {r.fam} | {r.vbin} | {r.n} | {f(g('delta'))} [{f(g('delta_lo'))}, {f(g('delta_hi'))}] | "
                         f"{f(g('dpsi'))} [{f(g('dpsi_lo'))}, {f(g('dpsi_hi'))}] | "
                         f"{g('fcorrect'):.2f} [{g('fcorrect_lo'):.2f}, {g('fcorrect_hi'):.2f}] | {g('fdecisive'):.0f} | "
                         f"{g('uptake'):.2f} [{g('uptake_lo'):.2f}, {g('uptake_hi'):.2f}] | "
                         f"{f(g('tw_left'))} / {f(g('tw_straight'))} / {f(g('tw_right'))} | "
                         f"{f(r.dx)} | {verdict(r) if tau == 4.0 and r.vbin == 'all' else ''} |")
        lines.append("")
    lines += ["## straight frames (lane keeping, 3 s)", "", "| family | n | d lateral error (m) | dx (m) |", "|:--|--:|:--|:--|"]
    for _, r in E[E.fam.str.startswith("straight")].iterrows():
        lines.append(f"| {r.fam} | {r.n} | {f(r.dlat)} [{f(r.dlat_lo)}, {f(r.dlat_hi)}] | {f(r.dx)} |")
    (out / f"{a.domain}_effects.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
