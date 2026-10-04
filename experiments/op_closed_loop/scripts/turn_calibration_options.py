"""Junction-turn option comparison: BEV panels for 4 representative B2D turns + a summary bar chart over all turns >= 25 deg. CPU only.

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/turn_calibration_options.py [--out DIR]

Every option is driven by the same kinematic bicycle (turn_calibration_sparse.pursue constants) from the same entry pose (25 m before the turn, on the dense centreline, aligned) at 5 m/s (3 m/s also):
  A dense route pure pursuit (reference)         B pure pursuit on the leaderboard 50 m sparse route
  C pure pursuit on a 10 m road polyline + 1 m noise (the draw with the median peak error is drawn)
  D action-head curvature as logged             E action-head curvature x 1.95
  F plan-derived curvature (openpilot's own plan path, 1 s chord)
D / E / F: inside the junction zone (route turn +-3 m) the curvature is the feed-forward replay of the logged head output along the dense path (median over the runs of the route per 1 m of route
arc length, from b2d_ticks.npz); outside the zone the car pure-pursues the dense route (lane keeping stand-in). They are open-loop counterfactuals of the head, not a closed loop.
Outputs: panels.png (4 turns), panels_3ms.png, summary.png, numbers.md, per_turn.csv.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import turn_calibration_lib as L  # noqa: E402
import turn_calibration_sparse as S  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
import json  # noqa: E402

T = data_dir() / "runs/op_closed_loop/turn_calibration"
PRIMARY = ("v2-drive-s2-q", "v2-drive-s3-q", "v2-opc-s2-q", "v2-opc-s3-q", "v2-lsc-s2-q", "v2-lsc-s3-q")
OPTS = ["A", "B", "C", "D", "E", "F"]
NAMES = {"A": "A dense route (reference)", "B": "B leaderboard 50 m sparse route", "C": "C 10 m road polyline + 1 m noise", "D": "D action head as logged (x1.0)",
         "E": "E action head x1.95 in junction", "F": "F plan-derived curvature"}
COL = {"A": "#000000", "B": "#D55E00", "C": "#E69F00", "D": "#0072B2", "E": "#56B4E9", "F": "#009E73"}   # Okabe-Ito
HALF = 1.75
GAIN = 1.95
ZM = 3.0


def ff_profile(rt, key, sign, s0, s1):
    """Median over the route's ticks of sign*b[key] per 1 m route arc length on [s0, s1]; None if < 60% of the bins have data."""
    m = (rt["s_route"] >= s0) & (rt["s_route"] < s1)
    if m.sum() < 5:
        return None
    sb = np.arange(np.floor(s0), np.ceil(s1) + 1.0, 1.0)
    idx = np.clip(((rt["s_route"][m] - sb[0]) // 1.0).astype(int), 0, len(sb) - 1)
    v = sign * rt[key][m]
    med = np.full(len(sb), np.nan)
    for i in np.unique(idx):
        med[i] = np.median(v[idx == i])
    ok = np.isfinite(med)
    if ok.sum() < 0.6 * (s1 - s0):
        return None
    return sb + 0.5, np.interp(sb + 0.5, (sb + 0.5)[ok], med[ok])


def run(D, gd, i0, psi0, v, ld, path, ff, zs, ze, i1):
    """Kinematic run; pure pursuit on `path` (polyline) except inside [zs, ze] (route arc length) when ff = (s, k) feeds the curvature. Cut at the dense exit index i1."""
    Pp, _ = S.poly_resample(path, 0.25)
    x, y, psi = D[i0][0], D[i0][1], psi0
    prog = int(np.argmin(np.linalg.norm(Pp - D[i0], axis=1)))
    s = gd[i0]
    out = []
    for _ in range(int(120.0 / S.DT)):
        if ff is not None and zs <= s <= ze:
            k = float(np.interp(s, ff[0], ff[1]))
        else:
            lo, hi = max(prog - 4, 0), min(prog + 80, len(Pp))
            prog = lo + int(np.argmin(np.linalg.norm(Pp[lo:hi] - [x, y], axis=1)))
            tgt = min(prog + int(ld / 0.25), len(Pp) - 1)
            d = Pp[tgt] - [x, y]
            alpha = (np.arctan2(d[1], d[0]) - psi + np.pi) % (2 * np.pi) - np.pi
            k = 2 * np.sin(alpha) / max(float(np.hypot(*d)), 1e-3)
        delta = np.clip(np.arctan(S.WB * k), -S.DELTA_MAX, S.DELTA_MAX)
        k = np.tan(delta) / S.WB
        x += v * S.DT * np.cos(psi)
        y += v * S.DT * np.sin(psi)
        psi += v * S.DT * k
        s += v * S.DT
        out.append((x, y, psi, k))
        if prog >= len(Pp) - 3 or (s > ze + 40 and ff is not None):
            break
        if ff is None and prog >= len(Pp) - 3:
            break
        if np.hypot(x - D[i1][0], y - D[i1][1]) < 0.5 and s > zs:
            break
    tr = np.array(out)
    j = int(np.argmin(np.linalg.norm(tr[:, :2] - D[i1], axis=1)))
    return tr[:j + 1]


def evaluate(tr, D, psiD, it0, i1, iz):
    xy = tr[:, :2]
    d = np.linalg.norm(xy[:, None, :] - D[None, :, :], axis=2)
    near, dev = d.argmin(1), d.min(1)
    w = (near >= it0 - 20) & (near <= i1)
    if not w.any():
        w[:] = True
    k = int(np.argmin(np.abs(near - iz)))     # sample closest to the zone end along the dense path
    he = np.degrees((tr[k, 2] - psiD[near[k]] + np.pi) % (2 * np.pi) - np.pi)
    out = np.where(dev > HALF)[0]
    return dict(peak=float(dev[w].max()), head_err=float(he), leave=int(out[0]) if len(out) else -1, dev=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(T / "options"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = {k: v for k, v in np.load(T / "b2d_ticks.npz", allow_pickle=True).items()}
    prim_routes = set(b["route"][np.array([any(x.startswith(p) for p in PRIMARY) for x in b["arm"]])])
    rng = np.random.default_rng(1)
    routes = {}
    for f in sorted(S.ARMS.glob("*/attempts/*/1/route.json")):
        rid = f.parts[-3]
        if rid not in routes:
            R = json.load(open(f))
            if len(R["xy"]) > 10:
                routes[rid] = (np.array(R["xy"]), np.array(R["cmd"]))
    rows, traj = [], {}
    for rid, (xy, cmd) in routes.items():
        P, g = L.resample(xy)
        mans = [m for m in L.maneuvers(P) if abs(m["angle"]) >= 25]
        if not mans:
            continue
        D, gd = S.poly_resample(xy, 0.25)
        psiD = L.heading(D)
        ds = L.leaderboard_downsample(cmd, 50.0, xy)
        lb = xy[ds]
        rm = b["route"] == rid
        rt = {k: b[k][rm] for k in ("s_route", "act_k", "plan_k1")}
        for mi, m in enumerate(mans):
            s0, s1 = max(g[m["i0"]] - 25.0, 0.0), g[m["i1"]] + 15.0
            zs, ze = g[m["i0"]] - ZM, g[m["i1"]] + ZM
            i0 = int(np.searchsorted(gd, s0))
            i1 = min(int(np.searchsorted(gd, s1)), len(D) - 1)
            iz = min(int(np.searchsorted(gd, ze)), len(D) - 1)
            it0 = int(np.searchsorted(gd, g[m["i0"]]))
            psi0 = np.arctan2(*(D[min(i0 + 4, len(D) - 1)] - D[i0])[::-1])
            fa = ff_profile(rt, "act_k", 1.0, zs, ze)
            fp = ff_profile(rt, "plan_k1", -1.0, zs, ze) if fa is not None else None
            for v in (5.0, 3.0):
                ld = max(3.0, v + 1.5)
                paths = {"A": (xy, None), "B": (lb, None)}
                cs = [S.noisy_road(xy, 1.0, rng, decim=10.0) for _ in range(20)]
                if fa is not None:
                    paths["D"] = (xy, fa)
                    paths["E"] = (xy, (fa[0], GAIN * fa[1]))
                    paths["F"] = (xy, fp)
                res = {}
                for o, (pth, ff) in paths.items():
                    tr = run(D, gd, i0, psi0, v, ld, pth, ff, zs, ze, i1)
                    res[o] = (tr, evaluate(tr, D, psiD, it0, i1, iz))
                cr = [(run(D, gd, i0, psi0, v, ld, c, None, zs, ze, i1)) for c in cs]
                ce = [evaluate(t, D, psiD, it0, i1, iz) for t in cr]
                mid = int(np.argsort([e["peak"] for e in ce])[len(ce) // 2])
                res["C"] = (cr[mid], ce[mid])
                for o, (tr, e) in res.items():
                    rows.append(dict(route=rid, man=mi, v=v, opt=o, peak=e["peak"], head_err=e["head_err"], leave=int(e["leave"] >= 0), angle=m["angle"], rmin=m["rmin"],
                                     held=rid not in prim_routes))
                # all 20 C draws for the summary statistics
                for t, e in zip(cr, ce):
                    rows.append(dict(route=rid, man=mi, v=v, opt="Call", peak=e["peak"], head_err=e["head_err"], leave=int(e["leave"] >= 0), angle=m["angle"], rmin=m["rmin"],
                                     held=rid not in prim_routes))
                if v == 5.0 or True:
                    traj[(rid, mi, v)] = dict(res=res, D=D, gd=gd, i0=i0, i1=i1, ze=ze, zs=zs, angle=m["angle"], rmin=m["rmin"], held=rid not in prim_routes, full="D" in res)
    # ------------------------------------------------------------------ table + summary
    import csv
    with open(out / "per_turn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    def boot(sel_vals, cl, stat, B=2000):
        u, inv = np.unique(cl, return_inverse=True)
        groups = [sel_vals[inv == i] for i in range(len(u))]
        bs = []
        for _ in range(B):
            ix = rng.integers(0, len(u), len(u))
            bs.append(stat(np.concatenate([groups[i] for i in ix])))
        return np.percentile(bs, [2.5, 97.5])

    # common turn set for fairness: turns where D/E/F exist; also report A-C on all turns
    full_keys = {(k[0], k[1]) for k, t in traj.items() if t["full"]}
    summ = {}
    L_ = ["# Junction-turn options: summary over the route turns >= 25 deg\n",
          "Kinematic bicycle (wheelbase 2.86 m), same entry pose (25 m before the turn, on the dense centreline); peak cross-track error vs the dense centreline over the turn window; "
          "leaves the lane = cross-track > 1.75 m at any time. CIs: cluster bootstrap over routes (2000). 'C' uses all 20 noise draws per turn (one cluster = route).\n"]
    for v in (5.0, 3.0):
        L_.append("\n## %.0f m/s\n" % v)
        L_.append("| option | n turns (routes) | median peak [95% CI] m | p90 peak m | leaves lane % [95% CI] | median final heading error deg (signed) | median abs heading err deg |")
        L_.append("|---|---|---|---|---|---|---|")
        for scope, keyset in (("all turns", None), ("turns with logged head data", full_keys)):
            L_.append("| *%s* | | | | | | |" % scope)
            for o in OPTS:
                ob = "Call" if o == "C" else o
                m = (R["opt"] == ob) & (R["v"] == v)
                if keyset is not None:
                    m &= np.array([(r, mi) in keyset for r, mi in zip(R["route"], R["man"])])
                if m.sum() == 0:
                    continue
                pk, lv, he, cl = R["peak"][m], R["leave"][m].astype(float), R["head_err"][m], R["route"][m]
                nt = len({(r, mi) for r, mi in zip(R["route"][m], R["man"][m])})
                ci = boot(pk, cl, np.median)
                cl2 = boot(lv, cl, lambda x: 100 * x.mean())
                summ[(scope, v, o)] = (np.median(pk), ci, 100 * lv.mean(), cl2, nt)
                L_.append("| %s | %d (%d) | %.2f [%.2f, %.2f] | %.2f | %.0f [%.0f, %.0f] | %+.1f | %.1f |" % (
                    NAMES[o], nt, len(set(cl)), np.median(pk), ci[0], ci[1], np.percentile(pk, 90), 100 * lv.mean(), cl2[0], cl2[1], np.median(he), np.median(np.abs(he))))
    # held-out-only 5 m/s for the head options
    L_.append("\n## 5 m/s, held-out routes only (not in the primary 19), turns with head data\n")
    L_.append("| option | n turns | median peak m | leaves lane % |\n|---|---|---|---|")
    for o in OPTS:
        ob = "Call" if o == "C" else o
        m = (R["opt"] == ob) & (R["v"] == 5.0) & R["held"] & np.array([(r, mi) in full_keys for r, mi in zip(R["route"], R["man"])])
        if m.sum():
            L_.append("| %s | %d | %.2f | %.0f |" % (NAMES[o], len({(r, mi) for r, mi in zip(R['route'][m], R['man'][m])}), np.median(R["peak"][m]), 100 * R["leave"][m].mean()))
    # ------------------------------------------------------------------ pick the four turns (held-out, head data, 5 m/s)
    cand = [(k, t) for k, t in traj.items() if k[2] == 5.0 and t["held"] and t["full"]]
    if len(cand) < 4:
        cand = [(k, t) for k, t in traj.items() if k[2] == 5.0 and t["full"]]
    sel, used = {}, set()

    def take(name, key):
        for k, t in sorted(cand, key=key):
            if (k[0], k[1]) not in used:
                sel[name] = k
                used.add((k[0], k[1]))
                return

    take("tight left", lambda c: (c[1]["angle"] > -45, c[1]["rmin"]))                       # left = negative angle (right-positive)
    take("tight right", lambda c: (c[1]["angle"] < 45, c[1]["rmin"]))
    take("wide turn", lambda c: (abs(c[1]["angle"]) < 60, -c[1]["rmin"]))
    take("largest disagreement", lambda c: -(max(c[1]["res"][o][1]["peak"] for o in OPTS) - min(c[1]["res"][o][1]["peak"] for o in OPTS)))
    L_.append("\n## Panel turns (5 m/s)\n")
    for n, k in sel.items():
        t = traj[k]
        L_.append("- %s: route %s turn %d, angle %+.0f deg (negative = left), R_min %.1f m, %s" % (n, k[0], k[1], t["angle"], t["rmin"], "held-out" if t["held"] else "primary"))
    (out / "numbers.md").write_text("\n".join(L_) + "\n")
    # ------------------------------------------------------------------ figures
    for v, fn in ((5.0, "panels.png"), (3.0, "panels_3ms.png")):
        fig, axs = plt.subplots(2, 2, figsize=(13, 13.5))
        for ax, (n, k) in zip(axs.ravel(), sel.items()):
            t = traj[(k[0], k[1], v)]
            D, gd = t["D"], t["gd"]
            a0 = max(t["i0"] - 5, 0)
            Dp = D[a0:min(t["i1"] + 40, len(D))]
            tg = np.gradient(Dp, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            nr = np.stack([-tg[:, 1], tg[:, 0]], -1)
            poly = np.vstack([Dp + HALF * nr, (Dp - HALF * nr)[::-1]])
            ax.fill(poly[:, 0], poly[:, 1], color="#dddddd", zorder=0, label="lane 3.5 m")
            txt = []
            for o in OPTS:
                if o not in t["res"]:
                    continue
                tr, e = t["res"][o]
                ax.plot(tr[:, 0], tr[:, 1], color=COL[o], lw=2.2 if o != "A" else 1.4, ls="--" if o == "A" else "-", zorder=3, label=NAMES[o])
                if e["leave"] >= 0:
                    ax.plot(tr[e["leave"], 0], tr[e["leave"], 1], "X", color=COL[o], ms=11, mec="white", mew=1.2, zorder=5)
                txt.append("%s  peak %.2f m  head %+.1f deg" % (o, e["peak"], e["head_err"]))
            if ax is axs.ravel()[0]:
                pass
            ax.plot(*D[t["i0"]], "k^", ms=8, zorder=6)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title("%s: route %s, %+.0f deg, R_min %.1f m, %.0f m/s" % (n, k[0], t["angle"], t["rmin"], v), fontsize=10)
            ax.set_xlabel("x (m, CARLA)")
            ax.set_ylabel("y (m, CARLA, down)")
            ax.grid(alpha=.25)
            ax.text(0.0, -0.13, "\n".join(txt), transform=ax.transAxes, fontsize=8, va="top", family="monospace")
        h, lab = axs.ravel()[0].get_legend_handles_labels()
        fig.legend(h + [plt.Line2D([], [], marker="X", color="gray", ls="")], lab + ["leaves the lane (1.75 m)"], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, fontsize=9)
        fig.suptitle("Junction turn options, kinematic runs from the same entry pose (triangle); peak = max cross-track vs the dense centreline, head = heading error at the zone exit", y=0.995, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=7)
        fig.savefig(out / fn, dpi=130)
        plt.close(fig)
    # summary bars (turns with head data so all six options share one turn set)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(OPTS))
    for j, (v, al) in enumerate(((5.0, 1.0), (3.0, 0.45))):
        sc = "turns with logged head data"
        med = [summ[(sc, v, o)][0] for o in OPTS]
        lo = [summ[(sc, v, o)][0] - summ[(sc, v, o)][1][0] for o in OPTS]
        hi = [summ[(sc, v, o)][1][1] - summ[(sc, v, o)][0] for o in OPTS]
        pc = [summ[(sc, v, o)][2] for o in OPTS]
        plo = [summ[(sc, v, o)][2] - summ[(sc, v, o)][3][0] for o in OPTS]
        phi = [summ[(sc, v, o)][3][1] - summ[(sc, v, o)][2] for o in OPTS]
        a1.bar(x + (j - .5) * .38, med, .38, color=[COL[o] for o in OPTS], alpha=al, yerr=[lo, hi], capsize=3, label="%.0f m/s" % v)
        a2.bar(x + (j - .5) * .38, pc, .38, color=[COL[o] for o in OPTS], alpha=al, yerr=[plo, phi], capsize=3)
    for ax, t_ in ((a1, "median peak cross-track error (m)"), (a2, "% of turns leaving the lane (> 1.75 m)")):
        ax.set_xticks(x)
        ax.set_xticklabels(OPTS)
        ax.set_title(t_, fontsize=10)
        ax.grid(axis="y", alpha=.3)
    a1.axhline(HALF, color="gray", ls=":", lw=1)
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="gray", alpha=1.0), plt.Rectangle((0, 0), 1, 1, color="gray", alpha=.45)], labels=["5 m/s", "3 m/s (light)"], fontsize=8)
    fig.suptitle("Junction options over the route turns >= 25 deg with logged head data (n = %d; 95%% CI over routes)  " % len(full_keys) + "  ".join("%s=%s" % (o, NAMES[o][2:].split(' (')[0]) for o in OPTS), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "summary.png", dpi=130)
    print("done ->", out, "panels:", sel)


if __name__ == "__main__":
    main()
