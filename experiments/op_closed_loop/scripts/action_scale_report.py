#!/usr/bin/env python
"""Is the shipped Cinque action head's ~0.45 curvature gain real? Decoded desired curvature vs measured vehicle curvature.

  extract (box)  -> <out>/bank.npz (op_adapt_H nav / wod teachers: action, plan, v0, logged future) and <out>/comma.npz
                    (action_scale_replay.py windows, scored frames only, with the localizer motion of the whole segment)
  report         -> tables on stdout + figs/action_scale.png

Sign: openpilot curvature, + = right (calibrated frame z down; lagd correlates desiredCurvature v^2 with yaw_rate v).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
VMIN = 3.0
D_GRID = np.round(np.arange(0.0, 1.55, 0.05), 2)


# ---------------------------------------------------------------- extract (box)
def cmd_extract(a):
    import base64, pickle  # noqa: E401
    import onnx
    from jevdrive import op_adapt as A
    from jevdrive.common import data_dir
    from jevdrive.openpilot.model import prepare_onnx
    m = onnx.load(str(prepare_onnx("cinque")), load_external_data=False)
    sl = pickle.loads(base64.b64decode({p.key: p.value for p in m.metadata_props}["output_slices"]))
    didx = A.distill_index(sl)
    pos = [int(np.flatnonzero(didx == sl["action"].start + k)[0]) for k in range(2)]
    out = Path(a.out)
    d = data_dir() / "runs/op_adapt_H"
    bank = {}
    for dom in ("nav", "wod"):
        z, t = np.load(d / "teacher" / f"{dom}.npz"), np.load(d / "samples" / dom / "tab.npz", allow_pickle=True)
        bank |= {f"{dom}_act": z["out"][:, pos].astype(np.float32), f"{dom}_plan": z["mu"][:, :8].astype(np.float32),
                 f"{dom}_v0": t["v0"], f"{dom}_fut": t["fut20"], f"{dom}_cam": t["cam"], f"{dom}_cluster": t["cluster"]}
    np.savez_compressed(out / "bank.npz", **bank)
    C = {k: [] for k in ("act", "plan", "v", "win", "kin", "kpp", "vplan")}
    for k, f in enumerate(sorted(out.glob("*_*.npz"))):
        if f.name in ("bank.npz", "comma.npz"):
            continue
        z = np.load(f)
        kin, kpp = comma_targets(z)
        s = slice(int(z["warm"]), None)
        i = z["idx"][s]
        C["act"].append(z["act"][s]), C["plan"].append(z["plan"][s][:, :8]), C["v"].append(z["v"][i]), C["win"].append(np.full(len(i), k))
        C["kin"].append(kin[s]), C["kpp"].append(kpp[s]), C["vplan"].append(z["plan"][s][:, 0, 3])
    if C["v"]:
        np.savez_compressed(out / "comma.npz", **{k: np.concatenate(v) for k, v in C.items()}, d_grid=D_GRID)
    print("bank rows", {dom: len(bank[f"{dom}_v0"]) for dom in ("nav", "wod")}, "comma windows", len(C["v"]), "frames", sum(len(x) for x in C["v"]))


def comma_targets(z):
    """Per replayed frame: measured curvature yaw_rate / v at t + d for every d in D_GRID (yaw rate smoothed over 0.25 s), and the
    1 s pure-pursuit curvature 2 y / |p|^2 of the logged position 1 s ahead in the calibrated frame of the frame (y right)."""
    from jevdrive.openpilot.frames import rot_from_euler
    t, v, yr = z["t"], z["v"], z["yr"]
    yr_s = np.convolve(yr, np.ones(5) / 5, mode="same")
    k_all = yr_s / np.maximum(v, 1.0)
    idx = z["idx"]
    kin = np.stack([np.interp(t[idx] + d, t, k_all, right=np.nan) for d in D_GRID], 1)
    cfd = rot_from_euler(z["rpy_calib"]).T
    kpp = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        tq = t[i] + 1.0
        if tq > t[-1]:
            continue
        p = np.array([np.interp(tq, t, z["pos"][:, c]) for c in range(3)]) - z["pos"][i]
        pc = cfd @ z["R"][i].T @ p
        d2 = pc[0] ** 2 + pc[1] ** 2
        kpp[j] = 2 * pc[1] / d2 if d2 > 4 else np.nan
    return kin, kpp


# ---------------------------------------------------------------- report
def fit(x, y, g, nb=2000, seed=0):
    """Slope of y on x through the origin, the reverse slope (1 / slope of x on y), Pearson r; cluster bootstrap CI over g."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    ug, inv = np.unique(g, return_inverse=True)
    sxy, sxx, syy = (np.bincount(inv, w, len(ug)) for w in (x * y, x * x, y * y))
    est = lambda w: (w @ sxy / (w @ sxx), w @ syy / (w @ sxy))  # noqa: E731
    b, rb = est(np.ones(len(ug)))
    rng = np.random.default_rng(seed)
    bs = np.array([est(np.bincount(rng.integers(len(ug), size=len(ug)), minlength=len(ug)).astype(float)) for _ in range(nb)])
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    return dict(n=int(ok.sum()), groups=len(ug), slope=b, ci=(lo[0], hi[0]), rev=rb, rev_ci=(lo[1], hi[1]), r=float(np.corrcoef(x, y)[0, 1]))


def fut_kappa(fut, tau):
    """Instantaneous curvature (right +) at time tau from a logged future (n, 20, 2) at 0.25 s, rear axle, y left: cubic fit
    through the origin and the first six points (0-1.5 s)."""
    T = np.r_[0.0, 0.25 * np.arange(1, 7)]
    out = np.full(len(fut), np.nan)
    for i, f in enumerate(fut):
        P = np.vstack([[0, 0], f[:6]])
        if not np.isfinite(P).all():
            continue
        cx, cy = np.polyfit(T, P[:, 0], 3), np.polyfit(T, P[:, 1], 3)
        dx, dy = np.polyval(np.polyder(cx), tau), np.polyval(np.polyder(cy), tau)
        ddx, ddy = np.polyval(np.polyder(cx, 2), tau), np.polyval(np.polyder(cy, 2), tau)
        sp = np.hypot(dx, dy)
        out[i] = -(dx * ddy - dy * ddx) / sp ** 3 if sp > 1.0 else np.nan
    return out


def plan_kappa(plan, v, t=0.275):
    """openpilot's get_curvature_from_plan (the path for models without an action head), on the plan mean."""
    from jevdrive.openpilot.model import T_IDXS, MIN_STABLE_DELAY
    ti = T_IDXS[:plan.shape[1]]                                     # the stored plan keeps its first 8 times (0-0.48 s)
    psi = np.array([np.interp(max(t, MIN_STABLE_DELAY), ti, p[:, 11]) for p in plan]) * (t / MIN_STABLE_DELAY if t < MIN_STABLE_DELAY else 1.0)
    vv = np.maximum(v, 1.0)
    return 2 * psi / (vv * t) - plan[:, 0, 14] / vv


def line(name, f):
    return (f"| {name} | {f['n']} ({f['groups']}) | **{f['slope']:.2f}** [{f['ci'][0]:.2f}, {f['ci'][1]:.2f}] | "
            f"{f['rev']:.2f} [{f['rev_ci'][0]:.2f}, {f['rev_ci'][1]:.2f}] | {f['r']:.2f} |")


def cmd_report(a):
    out = Path(a.out)
    B, C = np.load(out / "bank.npz", allow_pickle=True), np.load(out / "comma.npz")
    hdr = "| target | n (clusters) | g_reg: slope decoded ~ measured [95% CI] | g_cal: 1 / slope measured ~ decoded [95% CI] | r |\n|---|---|---|---|---|"
    res = {}
    # ---- comma1M (native rig)
    v, win = C["v"], C["win"]
    kd = C["act"][:, 0] / np.maximum(1.0, v) ** 2
    mv = v > VMIN
    print("## comma1M (native comma rig, real 20 Hz), v >", VMIN, "\n" + hdr)
    curve = []
    for j, d in enumerate(C["d_grid"]):
        f = fit(C["kin"][mv, j], kd[mv], win[mv], nb=300)
        curve.append((d, f["slope"], f["r"]))
    curve = np.array(curve)
    jbest = int(np.nanargmax(curve[:, 2]))
    for d in (0.0, 0.2, 0.275, float(curve[jbest, 0]), 0.5, 1.0):
        j = int(np.argmin(np.abs(C["d_grid"] - d)))
        f = fit(C["kin"][mv, j], kd[mv], win[mv])
        res[f"comma_inst_{C['d_grid'][j]:.2f}"] = f
        print(line(f"yaw rate / v at t + {C['d_grid'][j]:.2f} s" + (" (max r)" if j == jbest else ""), f))
    j2 = int(np.argmin(np.abs(C["d_grid"] - 0.2)))
    kpl = plan_kappa(C["plan"], v)
    kdp = C["act"][:, 0] / np.maximum(1.0, C["vplan"]) ** 2
    for name, x, y in (("1 s pure pursuit (rft.py's target)", C["kpp"], kd), ("plan-derived curvature (get_curvature_from_plan) vs yaw rate / v at t + 0.2", C["kin"][:, j2], kpl),
                       ("action / max(1, v_plan)^2 vs yaw rate / v at t + 0.2", C["kin"][:, j2], kdp)):
        f = fit(x[mv], y[mv], win[mv])
        res["comma_" + name.split(" ")[0]] = f
        print(line(name, f))
    kt = np.abs(C["kin"][:, j2])
    print("\nby |curvature| at t + 0.2 (straight < 0.003, curve 0.003-0.02, tight > 0.02 1/m):")
    for lo, hi in ((0, 0.003), (0.003, 0.02), (0.02, 1)):
        s = mv & (kt >= lo) & (kt < hi)
        f = fit(C["kin"][s, j2], kd[s], win[s])
        res[f"comma_bin_{lo}"] = f
        print(line(f"|k| {lo}-{hi}", f))
    for lo, hi in ((3, 8), (8, 15), (15, 40)):
        s = mv & (v >= lo) & (v < hi)
        f = fit(C["kin"][s, j2], kd[s], win[s])
        res[f"comma_v_{lo}"] = f
        print(line(f"v {lo}-{hi} m/s", f))
    print(f"\nplan speed / localizer speed (v > {VMIN}): median {np.median(C['vplan'][mv] / v[mv]):.3f}")
    # ---- WOD / navtrain teachers (wrong camera height)
    bank_curves = {}
    for dom in ("wod", "nav"):
        act, v0, fut, plan = B[f"{dom}_act"][:, 0], B[f"{dom}_v0"], B[f"{dom}_fut"], B[f"{dom}_plan"]
        g = np.unique(B[f"{dom}_cluster"], return_inverse=True)[1]
        m = v0 > VMIN
        kd = act / np.maximum(1.0, v0) ** 2
        x1, y1 = fut[:, 3, 0], fut[:, 3, 1]
        d2 = x1 ** 2 + y1 ** 2
        kpp = np.where(d2 > 4, -2 * y1 / np.maximum(d2, 1e-6), np.nan)
        vpl = plan[:, 0, 3]
        print(f"\n## {dom} (op_adapt_H teachers, camera z median {np.median(B[f'{dom}_cam'][:, 2]):.2f}), v > {VMIN}\n" + hdr)
        f = fit(kpp[m], kd[m], g[m])
        res[f"{dom}_pp"] = f
        print(line("1 s pure pursuit (rft.py's target)", f))
        bank_curves[dom] = []
        for tau in (0.0, 0.275, 0.5, 1.0):
            ki = fut_kappa(fut, tau)
            f = fit(ki[m], kd[m], g[m])
            res[f"{dom}_inst_{tau}"] = f
            bank_curves[dom].append((tau, f["slope"], f["r"]))
            print(line(f"instantaneous curvature of the logged path at t + {tau}", f))
            if tau == 0.275:
                k275 = ki
        f = fit(k275[m], (act / np.maximum(1.0, vpl) ** 2)[m], g[m])
        res[f"{dom}_vplan"] = f
        print(line("action / max(1, v_plan)^2 vs inst. at t + 0.275", f))
        f = fit(k275[m], plan_kappa(plan, v0)[m], g[m])
        res[f"{dom}_plan"] = f
        print(line("plan-derived curvature vs inst. at t + 0.275", f))
        print(f"plan speed / logged speed: median {np.median(vpl[m] / v0[m]):.3f}")
    json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in f.items()} for k, f in res.items()},
              open(out / "action_scale_summary.json", "w"), indent=1)
    if a.fig:
        figure(a.fig, C, curve, j2, VMIN)


def figure(path, C, curve, j2, vmin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = C["v"]
    m = v > vmin
    kd = C["act"][:, 0] / np.maximum(1.0, v) ** 2
    x = C["kin"][:, j2]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ok = m & np.isfinite(x)
    ax[0].scatter(x[ok], kd[ok], s=2, alpha=0.25, c="#3b6ea8", rasterized=True)
    lim = np.nanpercentile(np.abs(x[ok]), 99.5)
    xx = np.array([-lim, lim])
    ax[0].plot(xx, xx, "k--", lw=1, label="gain 1")
    b = (x[ok] @ kd[ok]) / (x[ok] @ x[ok])
    ax[0].plot(xx, b * xx, color="#c0504d", lw=1.5, label=f"fit {b:.2f}")
    ax[0].set(xlim=xx, ylim=xx, xlabel="measured curvature yaw_rate / v at t + 0.2 s (1/m, + right)",
              ylabel="decoded action[0] / max(1, v)^2 at t (1/m)", title=f"comma1M, native rig, v > {vmin:g} m/s")
    ax[0].legend(frameon=False, loc="upper left")
    ax[1].plot(curve[:, 0], curve[:, 1], color="#c0504d", label="slope")
    ax[1].plot(curve[:, 0], curve[:, 2], color="#3b6ea8", label="r")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].axhline(0.45, color="grey", lw=0.8, ls=":")
    ax[1].text(1.5, 0.47, "0.45 (rft.py ACT_ALPHA)", ha="right", fontsize=8, color="grey")
    ax[1].set(xlabel="target time offset d (s)", ylabel="slope / r", title="decoded vs measured at t + d", ylim=(0, 1.2))
    ax[1].legend(frameon=False)
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False), ax[1].spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("fig", path)


def cmd_b2d(a):
    """B2D (CARLA, `drive` arms, camera 1.433 m): logged action curvature act_k vs the curvature the car drove (truth yaw rate / v at t + 0.2 s),
    on ticks where the route geometry steers (lat == route: in a command zone or after divergence), so the car's motion does not come from
    the action head. Also the model's own speed vplan[0] / v."""
    import glob
    from jevdrive.common import data_dir
    X, Y, G, Z, VR = [], [], [], [], []
    files = sorted(glob.glob(str(data_dir() / "runs/vlm_arb/arms/eval-drive-s0*/attempts/*/1/plans.jsonl")))
    for k, f in enumerate(files):
        try:
            P = [json.loads(x) for x in open(f)]
            T = {t["frame"]: t for t in map(json.loads, open(Path(f).with_name("ticks.jsonl")))}
        except (OSError, ValueError):
            continue
        fr = np.array([t for t in sorted(T) if "truth" in T[t]])
        if len(fr) < 50:
            continue
        tt = np.array([T[t]["t"] for t in fr])
        tr = np.array([T[t]["truth"] for t in fr])
        yaw = np.unwrap(tr[:, 2] if np.abs(tr[:, 2]).max() <= 2 * np.pi + 0.1 else np.radians(tr[:, 2]))
        ds = np.r_[0, np.hypot(*np.diff(tr[:, :2], axis=0).T)]
        s = np.cumsum(ds)
        for p in P:
            if p.get("warm") or p["v"] < VMIN or p.get("lat") != "route" or "act_k" not in p:
                continue
            t0 = p["t"] + 0.2
            if t0 + 0.15 > tt[-1]:
                continue
            a_, b_ = np.interp([t0 - 0.15, t0 + 0.15], tt, s), np.interp([t0 - 0.15, t0 + 0.15], tt, yaw)
            if a_[1] - a_[0] < 0.5:
                continue
            X.append((b_[1] - b_[0]) / (a_[1] - a_[0])), Y.append(p["act_k"]), G.append(k), Z.append(p.get("lat_why") == "zone")
            VR.append(p["vplan"][0] / p["v"])
    X, Y, G, Z, VR = map(np.array, (X, Y, G, Z, VR))
    sign = np.sign(np.corrcoef(X, Y)[0, 1])
    print(f"## B2D eval-drive-s0* ({len(files)} runs), route-steered ticks, v > {VMIN}; CARLA yaw sign flip {sign:+.0f}\n"
          "| subset | n (runs) | g_reg [95% CI] | g_cal [95% CI] | r |\n|---|---|---|---|---|")
    kt = np.abs(X)
    for name, m in (("all", np.ones(len(X), bool)), ("command zone", Z), ("divergence fallback", ~Z), ("|k| > 0.02 (R < 50 m)", kt > 0.02)):
        print(line(name, fit(sign * X[m], Y[m], G[m])))
    print(f"vplan[0] / v median {np.median(VR):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "report", "b2d"])
    ap.add_argument("out")
    ap.add_argument("--fig")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)
