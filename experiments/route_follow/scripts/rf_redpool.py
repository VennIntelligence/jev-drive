#!/usr/bin/env python3
"""RED school: pool `on1_RED` = nuPlan navtrain LOG states 0-40 m before a traffic-light stop line whose ego-lane light is known at t0, red
rows with a stop-at-the-line target plus green rows matched on (distance, speed), and the held-out twin `on1_REDheld` (body1 hold logs, every
eligible row, log targets) for the offline read (prereg experiments/route_follow/plans/2026-10-13-red-prereg.md; background
research/next-round/traffic-light-probe.md). CPU only; $DATA_DIR/envs/op-train/bin/python (numpy + matplotlib).

  build [--name RED] [--gates-only]
Rows (LONG-PROBE context runs/long_contact/ctx/ctx.npz, the stopline_probe nuplan matching: the first traffic-light line (map type 2) the
hindsight route crosses, its lane connector matched, the light of that connector at the 9 keyframes t0 + 0.5 k):
  candidate  front bumper 0 < d <= 40 m before the line, connector matched, state known at t0, logged future finite
  red        red at t0, never non-red at a known keyframe of the 4 s window (red -> green rows are dropped and counted), ln_elig (the logged
             driver does not cross on red; drops the ~11 % red crossings), and a stop needs <= 3 m/s^2 (constant deceleration to the stop point)
  green      non-red at t0 and ln_elig (the logged driver does not cross on red later in the window)
Target (rear-axle x, y, yaw at t0 + 0.5 .. 4 s, the token's own frame; "log" = the navtrain cache's own `fut`, the target S imitates):
  green      the log
  red        the log's PATH re-timed (lateral = log, longitudinal only): stop point s* = d_line - 1.0 - s_f(0) - 0.05 (rear-axle travel until
             the front bumper is 1.05 m before the line; s_f(0) = the front bumper's route arc at t0, on_lon B's theta is d_line - 1.0), a = v0^2 / (2 s*), s_stop(t) = v0 t - a t^2 / 2 until v0 / a, then s*;
             target arc T_k = min(S_k, s_stop(t_k)) (never further than the log: a queue / lead vehicle the log stops behind is kept), pose = the
             log polyline (origin + fut) at arc T_k, yaw interpolated along it. Exclusion e_dec: the target's peak deceleration (0.5 s
             differences, v0 at t0) > 3 m/s^2 and > the log's own + 0.5.
Pool (logs not in navsim/body1-hold-logs, which contain op-parity-full-dev): per cell (d edges 0/5/10/20/30/40 m x v0 edges 0/0.5/2/5/10/inf
m/s) min(n_red, n_green) rows of EACH class (seed 0): exactly 1:1 within every cell, so (distance, speed) does not tell red from green.
Held (hold logs): every red / green row (log targets; the read matches by cell itself).
Writes (rf_cfpool.write_pool's layout): cache/on1_<name>/{tab (+ cls, d_line, v0), agent (ok False), road (body1 road raster), lon (on_lon
pack, items B only: the stop-line hinge on the own plan, timing_only), red (row meta)}.npz, cache/on1_<name>@warp/{front.npy, teacher.npz},
runs/route_mem/feat/on1_<name>.npz (the token's own route memory); the same for on1_<name>held. Gates -> runs/route_follow/red/gates.json and
experiments/route_follow/results/red/pool_gates.json; figures experiments/route_follow/figs/red_sheet.png (3 red + 3 green: BEV + speed
profile), red_labels.png (label sanity).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/body1/lib"), str(R / "experiments/long_contact/lib"), str(R / "experiments/onpol_nuplan/lib")]
import b1 as B  # noqa: E402
import on_lon as OL  # noqa: E402

D = B.data_dir()
CR, FEAT, OUT = D / "runs/op_parity/cache", D / "runs/route_mem/feat", D / "runs/route_follow/red"
ROAD = D / "runs/body1/labels_road/navtrain_s01234567891011.npz"
FRONT, D_MAX, M_B, A_MAX, DEC_SLACK, STOP_SLACK = 4.049, 40.0, 1.0, 3.0, 0.5, 0.05
DE, VE = np.array([0, 5, 10, 20, 30, 40.0]), np.array([0, 0.5, 2, 5, 10, np.inf])
TK = 0.5 * np.arange(1, 9)
HOLD = "navsim/body1-hold-logs"


# ---------------------------------------------------------------- rows and targets
def candidates(z):
    """-> dict of per-candidate arrays (global row g, line index q, d (front bumper -> line), states (9,), elig, v0)."""
    ty = z["ln_type"]
    tl = ty == 2
    q = np.where(tl.any(1), tl.argmax(1), -1)
    g = np.flatnonzero(q >= 0)
    q = q[g]
    d = z["ln_d"][g, q] - FRONT
    st = z["ln_st"][g, q]
    keep = (d > 0) & (d <= D_MAX) & z["ln_conn"][g, q] & (st[:, 0] >= 0) & z["ok"][g] & np.isfinite(z["fut"][g]).all((1, 2))
    g, q, d, st = g[keep], q[keep], d[keep], st[keep]
    import lc as LC
    LC.NR = z["route"].shape[1]
    sf0 = np.concatenate([LC.project(z["route"][g[i:i + 4000]], np.tile([[[FRONT, 0.0]]], (len(g[i:i + 4000]), 1, 1)).astype(np.float32))[0][:, 0]
                          for i in range(0, len(g), 4000)])            # the front bumper's route arc at t0 (B's s_f(0); ~FRONT, not exactly)
    return dict(g=g, q=q, d=d.astype(np.float32), st=st, elig=z["ln_elig"][g, q], cross=z["ln_stcross"][g, q], v0=z["v0"][g].astype(np.float32),
                sf0=sf0.astype(np.float32), s_star=(z["ln_d"][g, q] - M_B - sf0 - STOP_SLACK).astype(np.float32))


def log_arcs(fut):
    """fut (n, 8, 3) -> the log polyline's arc at t0 .. 4 s (n, 9)."""
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[:, :, :2]], 1)
    return np.concatenate([np.zeros((len(fut), 1)), np.cumsum(np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)), 1)], 1)


def peak_decel(S, v0):
    """S (n, 9) arcs at 0 .. 4 s, v0 (n,) -> peak deceleration (n,) m/s^2 from the mean speeds of the 0.5 s steps (v0 at t0, dt 0.25 for the first)."""
    v = np.diff(S, axis=1) / 0.5
    a = np.concatenate([(v[:, :1] - v0[:, None]) / 0.25, np.diff(v, axis=1) / 0.5], 1)
    return np.maximum(-a.min(1), 0.0)


def stop_target(fut, s_star, v0):
    """Red target: the log path re-timed by min(log arc, constant-deceleration stop 1 m before the line). -> (fut_t (n, 8, 3), T (n, 9), S (n, 9), a_req)."""
    S = log_arcs(fut)
    s_star = np.maximum(s_star, 0.05)                                  # rear-axle travel to the stop point (front bumper 1.05 m before the line)
    a = v0 ** 2 / (2 * s_star)
    ts = np.where(a > 0, v0 / np.maximum(a, 1e-9), 0.0)
    t = TK[None]
    ss = np.where(t < ts[:, None], v0[:, None] * t - 0.5 * a[:, None] * t ** 2, s_star[:, None])
    T = np.concatenate([np.zeros((len(fut), 1)), np.minimum(S[:, 1:], ss)], 1)
    out = np.zeros_like(fut)
    for i in range(len(fut)):
        x = np.r_[0.0, fut[i, :, 0]]
        y = np.r_[0.0, fut[i, :, 1]]
        h = np.unwrap(np.r_[0.0, fut[i, :, 2]])
        s = S[i] + 1e-6 * np.arange(9)                                  # strictly increasing for interp (a stopped log repeats arcs)
        out[i] = np.stack([np.interp(T[i, 1:], s, x), np.interp(T[i, 1:], s, y), np.interp(T[i, 1:], s, h)], 1)
    return out.astype(np.float32), T, S, a


def front_arcs(z, g, fut):
    """Front-bumper route arcs (n, 8) of rear-axle poses fut (n, 8, 3) (lc.project on the rows' routes)."""
    import lc as LC
    LC.NR = z["route"].shape[1]
    p = fut[..., :2] + FRONT * np.stack([np.cos(fut[..., 2]), np.sin(fut[..., 2])], -1)
    return np.concatenate([LC.project(z["route"][g[i:i + 4000]], p[i:i + 4000].astype(np.float32))[0] for i in range(0, len(g), 4000)])


def fit_stop(z, C, iters=4):
    """stop_target with s* tightened per row until the target's front bumper stays behind B's threshold max(d_line - 1, s_f(0)) on the route
    (the log polyline's chord arcs and the route's arcs differ by ~0.1 m on curves): s* -= overshoot + 0.03 m, at most `iters` times."""
    s_star = C["s_star"].astype(np.float64).copy()
    th = np.maximum(z["ln_d"][C["g"], C["q"]] - M_B, C["sf0"])
    fut = z["fut"][C["g"]]
    for _ in range(iters):
        out = stop_target(fut, s_star, C["v0"])
        ex = (front_arcs(z, C["g"], out[0]) - th[:, None]).max(1)
        fix = ex > -0.01
        if not fix.any():
            break
        s_star[fix] -= ex[fix] + 0.03
    return out


def cell(d, v):
    return np.clip(np.searchsorted(DE, d, "left") - 1, 0, len(DE) - 2) * 10 + np.clip(np.searchsorted(VE, v, "right") - 1, 0, len(VE) - 2)


def match(red, grn, d, v, seed=0):
    """Indices of red and green rows kept: per (d, v) cell min(n_red, n_green) of each, random without replacement."""
    rng = np.random.default_rng(seed)
    c = cell(d, v)
    kr, kg = [], []
    for k in np.unique(c[red | grn]):
        r, g = np.flatnonzero(red & (c == k)), np.flatnonzero(grn & (c == k))
        n = min(len(r), len(g))
        kr += rng.permutation(r)[:n].tolist()
        kg += rng.permutation(g)[:n].tolist()
    return np.sort(np.array(kr, int)), np.sort(np.array(kg, int))


# ---------------------------------------------------------------- lon pack (B only)
def lon_pack(z, g, names):
    """on_lon.build_rows on the rows' context (route, lines, light states at the dense steps); no actors, no turn, no progress reference: only B
    is kept in `items` (A / C / P are inactive by construction)."""
    import lc as LC
    LC.NR = z["route"].shape[1]
    parts = []
    kf = np.clip(np.round(np.arange(OL.NS) * OL.DT / 0.5).astype(int), 0, 8)  # nearest keyframe of every 0.1 s step
    for i in range(0, len(g), 2000):
        gg = g[i:i + 2000]
        n = len(gg)
        red = z["ln_st"][gg][:, :, kf] == 1
        P = OL.build_rows(z["route"][gg], z["rhead"][gg], z["rlen"][gg], np.zeros((n, 9, 1, 5), np.float32), np.zeros((n, 9, 1), bool), np.full((n, 1), -1),
                          z["ln_type"][gg], z["ln_d"][gg], z["ln_elig"][gg], red, np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan),
                          z["vlog"][gg].astype(np.float32), np.full((n, 8), np.nan, np.float32))
        P["p_open"][:] = False
        parts.append(P)
    P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    P["names"], P["items"] = names, np.array(["B"])
    return P


# ---------------------------------------------------------------- pool writer
def write_pool(name, z, g, names, fut, extra, road, logs, run):
    """Rows g (global body1 order) -> cache/on1_<name>{,@warp}, feat/on1_<name>.npz; tab fut = `fut`; lon.npz (B); red.npz (meta)."""
    n = len(g)
    off = B.shard_offsets()
    sh = np.searchsorted(off, g, "right") - 1
    pdir, wdir = CR / f"on1_{name}", CR / f"on1_{name}@warp"
    pdir.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    front = np.lib.format.open_memmap(wdir / "front.npy", mode="w+", dtype=np.float16, shape=(n, 8, 32, 512))
    tb, ft, t_out, t_plan, di, pi = None, None, np.zeros((n, 1086), np.float32), np.zeros((n, 33, 15), np.float32), None, None
    for k in np.unique(sh):
        m = np.flatnonzero(sh == k)
        src = g[m] - off[k]
        d = B.cdir("log", k)
        t = dict(np.load(CR / d / "tab.npz"))
        assert (t["names"][src] == z["tokens"][g[m]]).all() and (t["log"][src] == logs[g[m]]).all(), f"{d}: token / log order"
        if tb is None:
            tb = {q: np.zeros((n,) + v.shape[1:], v.dtype) for q, v in t.items() if q not in ("names", "off", "log")}
        for q in tb:
            tb[q][m] = t[q][src]
        o = np.argsort(src)
        F = np.load(CR / f"{d}@warp" / "front.npy", mmap_mode="r")
        for c in range(0, len(m), 512):
            front[m[o[c:c + 512]]] = F[src[o[c:c + 512]]]
        T = np.load(CR / f"{d}@warp" / "teacher.npz")
        t_out[m], t_plan[m] = T["out"][src], T["plan"][src]
        di, pi = (T["di"], T["pi"]) if di is None else (di, pi)
        assert (T["di"] == di).all() and (T["pi"] == pi).all()
        f = np.load(FEAT / f"{d}.npz")
        assert (f["names"][src] == z["tokens"][g[m]]).all()
        if ft is None:
            ft = dict(f=np.zeros((n,) + f["f"].shape[1:], f["f"].dtype), k=np.zeros(n, f["k"].dtype), ok=np.zeros(n, bool),
                      jdist=np.full(n, np.nan, np.float32), turn40=np.full(n, np.nan, np.float32))
        for q in ft:
            ft[q][m] = f[q][src]
    front.flush()
    del front
    tb["fut"], tb["log"] = np.asarray(fut, np.float32), logs[g]
    assert np.isfinite(tb["fut"]).all() and len(set(names.tolist())) == n
    np.savez(pdir / "tab.npz", names=names, **tb, **extra)
    np.savez(wdir / "teacher.npz", out=t_out, plan=t_plan, di=di, pi=pi)
    np.savez(pdir / "agent.npz", tokens=names, ok=np.zeros(n, bool), box=np.zeros((n, 9, 32, 5), np.float32), valid=np.zeros((n, 9, 32), bool))
    np.savez(pdir / "road.npz", tokens=names, ok=road["ok"][g], sdf=road["sdf"][g])
    np.savez(pdir / "lon.npz", **lon_pack(z, g, names))
    np.savez(pdir / "red.npz", names=names, g=g, token=z["tokens"][g], **extra)
    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez(FEAT / f"on1_{name}.npz", names=names, **ft)
    run.info(f"on1_{name}: {n} rows, {len(set(tb['log'].tolist()))} logs, road ok {road['ok'][g].mean():.3f}, route memory ok {ft['ok'].mean():.3f}, "
             f"classes { {c: int((extra['cls'] == c).sum()) for c in ('red', 'green')} }")


# ---------------------------------------------------------------- figures
def bev(ax, sdf, route, rh, dl, flog, ftgt, title):
    yy, xx = np.meshgrid(-24 + 0.5 * (np.arange(96) + 0.5), -8 + 0.5 * (np.arange(128) + 0.5))
    ax.contourf(-yy, xx, sdf.astype(np.float32), levels=[0, 1e3], colors=["#e4e4e4"])
    ax.contour(-yy, xx, sdf.astype(np.float32), levels=[0], colors=["#999999"], linewidths=0.5)
    ax.plot(-route[:60, 1], route[:60, 0], "-", color="#9ecae1", lw=1.2, label="route (hindsight)")
    j = int(np.clip(round(dl), 0, len(route) - 1))
    nx, ny = -np.sin(rh[j]), np.cos(rh[j])
    ax.plot([-(route[j, 1] - 2.5 * ny), -(route[j, 1] + 2.5 * ny)], [route[j, 0] - 2.5 * nx, route[j, 0] + 2.5 * nx], "-", color="k", lw=2.5, label="stop line")
    ax.plot(np.r_[0, -flog[:, 1]], np.r_[0, flog[:, 0]], "o-", color="#2ca02c", ms=3, lw=1.0, label="log 4 s")
    ax.plot(np.r_[0, -ftgt[:, 1]], np.r_[0, ftgt[:, 0]], "s--", color="#d62728", ms=3, lw=1.0, label="target 4 s")
    ax.plot([0], [0], "k^", ms=7)
    ax.set_xlim(-14, 14), ax.set_ylim(-6, 50), ax.set_aspect("equal")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)


def sheet(path, z, road, rows, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 6, figsize=(15, 7.2), gridspec_kw=dict(height_ratios=[2.2, 1]))
    for c, i in enumerate(rows):
        g, q = P["g"][i], P["q"][i]
        flog, ftgt = z["fut"][g], P["fut"][i]
        kind = P["cls"][i]
        st = "".join({1: "R", 0: "G", -1: "?"}[int(s)] for s in z["ln_st"][g, q])
        bev(axs[0, c], road["sdf"][g], z["route"][g], z["rhead"][g], z["ln_d"][g, q], flog, ftgt,
            f"{kind} {z['tokens'][g][:10]}\nline {P['d'][i]:.1f} m ahead (front), v0 {P['v0'][i]:.1f} m/s\nlight t0..4 s {st}")
        if c == 0:
            axs[0, c].legend(fontsize=6, loc="lower left")
        Sl, St = log_arcs(flog[None])[0], log_arcs(ftgt[None])[0]
        tt = 0.5 * np.arange(9)
        ax = axs[1, c]
        ax.step(np.r_[0, tt[1:] - 0.25], np.r_[P["v0"][i], np.diff(Sl) / 0.5], where="post", color="#2ca02c", lw=1.2, label="log speed")
        ax.step(np.r_[0, tt[1:] - 0.25], np.r_[P["v0"][i], np.diff(St) / 0.5], where="post", color="#d62728", ls="--", lw=1.2, label="target speed")
        ax2 = ax.twinx()
        ax2.plot(tt, Sl, "-", color="#2ca02c", alpha=0.35), ax2.plot(tt, St, "--", color="#d62728", alpha=0.35)
        ax2.axhline(max(P["s_star"][i], 0.05), color="k", lw=0.8, ls=":")
        ax2.set_ylabel("arc (m); dotted = stop point", fontsize=6), ax2.tick_params(labelsize=6)
        ax.set_xlabel("t (s)", fontsize=7), ax.set_ylabel("m/s", fontsize=7), ax.tick_params(labelsize=6)
        if c == 0:
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("on1_RED sample rows (seed 0): 3 red (stop target = log path re-timed to stop 1 m before the line) and 3 matched green (log target)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def labels_fig(path, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    ax[0].hist(L["arc_diff"], bins=np.linspace(-15, 0.5, 63), color="#d62728")
    ax[0].set_title(f"red pool rows: target - log arc at 4 s (m)\nchanged > 0.5 m: {L['changed']:.1%}", fontsize=8)
    for k, c in (("red", "#d62728"), ("green", "#2ca02c")):
        ax[1].hist(L[f"d_{k}"], bins=np.arange(0, 41, 2.5), histtype="step", color=c, lw=1.5, label=k)
        ax[2].hist(L[f"v_{k}"], bins=np.arange(0, 16, 0.5), histtype="step", color=c, lw=1.5, label=k)
    ax[1].set_title("pool: distance front -> line (m), red vs matched green", fontsize=8), ax[1].legend(fontsize=7)
    ax[2].set_title("pool: speed at t0 (m/s), red vs matched green", fontsize=8)
    ax[3].hist(L["front_gap_log"], bins=np.linspace(-5, 40, 91), histtype="step", color="#2ca02c", lw=1.2, label="log")
    ax[3].hist(L["front_gap_tgt"], bins=np.linspace(-5, 40, 91), histtype="step", color="#d62728", lw=1.2, label="target")
    ax[3].axvline(0, color="k", lw=0.8)
    ax[3].set_title("red pool rows: line - front bumper at 4 s (m, route arc)\n< 0 = past the line", fontsize=8), ax[3].legend(fontsize=7)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def front_arc(z, g, fut):
    """Front-bumper route arc at 4 s of rear-axle poses fut (n, 8, 3) (lc.project on the rows' routes)."""
    import lc as LC
    LC.NR = z["route"].shape[1]
    p = fut[:, 7, :2] + FRONT * np.stack([np.cos(fut[:, 7, 2]), np.sin(fut[:, 7, 2])], -1)
    out = np.zeros(len(g), np.float32)
    for i in range(0, len(g), 4000):
        out[i:i + 4000] = LC.project(z["route"][g[i:i + 4000]], p[i:i + 4000, None, :])[0][:, 0]
    return out


# ---------------------------------------------------------------- main
def cmd_build(a):
    import torch
    from jevdrive.data import splits
    from jevdrive.run import Run
    hs = splits.load(HOLD)
    with Run("route_follow", f"redpool-{a.name}", config=vars(a)) as run:
        run.use_split(hs)
        z = dict(np.load(D / "runs/long_contact/ctx/ctx.npz"))
        logs = np.load(B.root() / "labels/log.npy")
        assert (np.load(B.root() / "labels/tokens.npy") == z["tokens"]).all()
        fz = z["fut"]                                                   # the context's future (rebuilt from the log pkl, <= 0.17 m off the cache)
        z["fut"] = np.concatenate([B.tab("log", k)["fut"] for k in range(B.NSH)]).astype(np.float32)   # the cache's own target (what S imitates)
        run.info(f"log future: cache vs context max {np.nanmax(np.abs(z['fut'] - fz)[..., :2]):.3f} m")
        C = candidates(z)
        st, el = C["st"], C["elig"]
        red0, grn0 = st[:, 0] == 1, st[:, 0] == 0
        thru = ~(st == 0).any(1)
        fut_t, T, S, areq = fit_stop(z, C)
        dec_t, dec_l = peak_decel(T, C["v0"]), peak_decel(S, C["v0"])
        e_dec = (dec_t > A_MAX) & (dec_t > dec_l + DEC_SLACK)
        red = red0 & thru & el & (areq <= A_MAX) & ~e_dec
        grn = grn0 & el
        hold = hs.mask(logs[C["g"]])
        kr, kg = match(red & ~hold, grn & ~hold, C["d"], C["v0"])
        hr, hg_ = np.flatnonzero(red & hold), np.flatnonzero(grn & hold)
        cnt = {"candidates": int(len(C["g"])), "red_t0": int(red0.sum()), "green_t0": int(grn0.sum()),
               "red_dropped_cross_on_red": int((red0 & (C["cross"] == 1)).sum()), "red_dropped_not_elig_other": int((red0 & ~el & (C["cross"] != 1)).sum()),
               "red_dropped_turns_green_in_window": int((red0 & el & ~thru).sum()), "red_dropped_decel_gt_3": int((red0 & el & thru & (areq > A_MAX)).sum()),
               "red_dropped_e_dec": int((red0 & el & thru & (areq <= A_MAX) & e_dec).sum()), "red_eligible": int(red.sum()), "green_eligible": int(grn.sum()),
               "red_eligible_train_logs": int((red & ~hold).sum()), "green_eligible_train_logs": int((grn & ~hold).sum()),
               "pool_red": int(len(kr)), "pool_green": int(len(kg)), "pool_rows": int(len(kr) + len(kg)),
               "pool_logs": int(len(set(logs[C["g"][np.r_[kr, kg]]].tolist()))), "held_red": int(len(hr)), "held_green": int(len(hg_)),
               "held_logs": int(len(set(logs[C["g"][np.r_[hr, hg_]]].tolist())))}
        cnt["red_share_of_pool"] = cnt["pool_red"] / max(cnt["pool_rows"], 1)
        run.info("counts " + json.dumps(cnt))
        cr, cg = cell(C["d"][kr], C["v0"][kr]), cell(C["d"][kg], C["v0"][kg])
        cells_equal = bool(all((cr == k).sum() == (cg == k).sum() for k in np.unique(np.r_[cr, cg])))
        # label sanity (red pool rows)
        g_r = C["g"][kr]
        lat = []                                                        # lateral offset of the target from the log polyline (should be 0)
        for i, j in enumerate(kr):
            Pl = np.r_[[[0.0, 0.0]], z["fut"][C["g"][j], :, :2]]
            q = fut_t[j, :, :2]
            A_, B_ = Pl[:-1][None], Pl[1:][None]
            u = B_ - A_
            tt = np.clip(((q[:, None] - A_) * u).sum(-1) / np.maximum((u ** 2).sum(-1), 1e-9), 0, 1)
            lat.append(np.hypot(*(q[:, None] - (A_ + tt[..., None] * u)).transpose(2, 0, 1)).min(1).max())
        lat = np.array(lat)
        pack = OL.to_pack(lon_pack(z, g_r, np.array([f"on:red:{t}" for t in z["tokens"][g_r]])), "cpu", ("B",))
        rr = torch.arange(len(g_r))
        b_t = OL.terms(torch.as_tensor(fut_t[kr]), rr, pack)["lonB"].numpy()
        b_l = OL.terms(torch.as_tensor(z["fut"][g_r]), rr, pack)["lonB"].numpy()
        fa_t, fa_l = front_arc(z, g_r, fut_t[kr]), front_arc(z, g_r, z["fut"][g_r])
        dl = z["ln_d"][g_r, C["q"][kr]]
        arc_diff = T[kr, 8] - S[kr, 8]
        G = {"name": a.name, "counts": cnt, "cells_1to1": cells_equal,
             "batch_share": {"on_k": a.on_k, "batch": 128, "red_rows_per_batch": a.on_k * cnt["red_share_of_pool"],
                             "red_share_of_batch_rows": a.on_k * cnt["red_share_of_pool"] / 128},
             "labels": {"red_target_B_zero_share": float((b_t <= 0.01).mean()), "red_log_B_zero_share": float((b_l <= 0.01).mean()),
                        "red_target_front_past_line_share": float((fa_t > dl + 1e-3).mean()), "red_log_front_past_line_share": float((fa_l > dl + 1e-3).mean()),
                        "red_target_lateral_off_log_path_max_m": float(lat.max()), "red_target_arc_le_log_share": float((arc_diff <= 1e-4).mean()),
                        "red_target_changed_gt_0p5m_share": float((arc_diff < -0.5).mean()), "red_arc_diff_median_m": float(np.median(arc_diff)),
                        "red_arc_diff_p10_m": float(np.quantile(arc_diff, 0.1)), "red_target_peak_decel_max": float(dec_t[kr].max()),
                        "red_target_peak_decel_p99": float(np.quantile(dec_t[kr], 0.99)), "red_log_peak_decel_p99": float(np.quantile(dec_l[kr], 0.99)),
                        "red_moving_share_v_ge_2": float((C["v0"][kr] >= 2).mean()), "green_moving_share_v_ge_2": float((C["v0"][kg] >= 2).mean()),
                        "green_log_front_past_line_4s_share": float((front_arc(z, C["g"][kg], z["fut"][C["g"][kg]]) > z["ln_d"][C["g"][kg], C["q"][kg]]).mean()),
                        "d_median_red_green": [float(np.median(C["d"][kr])), float(np.median(C["d"][kg]))],
                        "v_median_red_green": [float(np.median(C["v0"][kr])), float(np.median(C["v0"][kg]))]}}
        Lb = G["labels"]
        G["gates"] = {"G-red1 rows (pool red >= 2000, 1:1 per cell, held red >= 300)": bool(cnt["pool_red"] >= 2000 and cells_equal and cnt["held_red"] >= 300),
                      "G-red2 labels (target B = 0 on >= 99 % of red rows, lateral off log path <= 0.05 m, target arc <= log arc on all, peak decel <= 3 or <= log + 0.5)":
                      bool(Lb["red_target_B_zero_share"] >= 0.99 and Lb["red_target_lateral_off_log_path_max_m"] <= 0.05 and Lb["red_target_arc_le_log_share"] == 1.0
                           and not e_dec[kr].any())}
        G["pass"] = all(G["gates"].values())
        figs = R / "experiments/route_follow/figs"
        figs.mkdir(parents=True, exist_ok=True)
        road = np.load(ROAD)
        road = dict(ok=road["ok"], sdf=road["sdf"])
        Pq = dict(g=C["g"], q=C["q"], d=C["d"], v0=C["v0"], s_star=C["s_star"], fut=np.where(red[:, None, None], fut_t, z["fut"][C["g"]]),
                  cls=np.where(red, "red", np.where(grn, "green", "-")))
        rng = np.random.default_rng(0)
        mv = lambda ix: ix[C["v0"][ix] >= 2]                            # noqa: E731  sample moving rows (the informative ones)
        pick = rng.permutation(mv(kr))[:3].tolist() + rng.permutation(mv(kg))[:3].tolist()
        sheet(figs / "red_sheet.png", z, road, pick, Pq)
        labels_fig(figs / "red_labels.png", dict(arc_diff=arc_diff, changed=float((arc_diff < -0.5).mean()), d_red=C["d"][kr], d_green=C["d"][kg],
                                                 v_red=C["v0"][kr], v_green=C["v0"][kg], front_gap_log=dl - fa_l, front_gap_tgt=dl - fa_t))
        G["G-red3 sheet"] = dict(fig="experiments/route_follow/figs/red_sheet.png", tokens=[str(z["tokens"][C["g"][i]]) for i in pick])
        G["label_fig"] = "experiments/route_follow/figs/red_labels.png"
        if not a.gates_only:
            for nm, ir, ig, tgt in ((a.name, kr, kg, True), (f"{a.name}held", hr, hg_, False)):
                ix = np.r_[ir, ig]
                g = C["g"][ix]
                isr = np.r_[np.ones(len(ir), bool), np.zeros(len(ig), bool)]
                names = np.array([f"on:{'red' if r else 'grn'}:{t}" for r, t in zip(isr, z["tokens"][g])])
                fut = np.where((isr & tgt)[:, None, None], fut_t[ix], z["fut"][g])
                extra = dict(cls=np.where(isr, "red", "green"), d_line=C["d"][ix], v0=C["v0"][ix], q=C["q"][ix].astype(np.int8),
                             cellid=cell(C["d"][ix], C["v0"][ix]).astype(np.int16), fut_log=z["fut"][g])
                write_pool(nm, z, g, names, fut, extra, road, logs, run)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "gates.json").write_text(json.dumps(G, indent=1))
        rd = R / "experiments/route_follow/results/red"
        rd.mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT / "gates.json", rd / "pool_gates.json")
        run.summary.update(G)
        print(json.dumps(G, indent=1))
        if not G["pass"]:
            raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--name", default="RED"), p.add_argument("--gates-only", action="store_true")
    p.add_argument("--on-k", type=int, default=14, help="per-batch pool rows of the registered arm (for the stated batch share only)")
    a = ap.parse_args()
    dict(build=cmd_build)[a.cmd](a)
