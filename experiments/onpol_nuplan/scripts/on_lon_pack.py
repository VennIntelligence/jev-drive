#!/usr/bin/env python3
"""ONPOL-NUPLAN longitudinal label pack: joins the scene context (scripts/on_ctx.py) with the on-policy pool rows into the per-row tensors
lib/on_lon.py reads, then gate (iv): activation of every item on the pool's stored served plans. CPU; $DATA_DIR/envs/op-train/bin/python.

  on_lon_pack.py pack --name N [--items A B C P]
      N = the rows name of scripts/on_rows.py: reads $DATA_DIR/runs/onpol_nuplan/rows/N/{meta,agent}.npz and
      $DATA_DIR/runs/op_parity/cache/on1_N/tab.npz (row names, imitation target)
      -> $DATA_DIR/runs/onpol_nuplan/rows/N/lon.npz (on_lon.load finds it from the pool dir on1_N),
         $DATA_DIR/runs/onpol_nuplan/lon/gates/lon_pool-N.json, .../lon/figs/lon_preview-N.png
  Without --items an item whose share of active rows on the served plans is below 0.2 % is left out of the pack's `items` and named in the
  gate (the prereg's L inclusion rule (iii)); with --items exactly those are kept.

Join, per row (meta.npz of scripts/on_rows.py: pose_map = the row's rear-axle pose in the nuPlan map frame, t_us = simulation time of the
decision, simulation time 0 = the log frame 3 before the scene token; scene, k, the served plan `poses`):
  frame   the scene's route (origin frame, global origin pose in the context) is re-expressed in the row's own rear-axle frame;
  time    the row's log time (meta `t_log_us`) against the context's keyframe timestamps: the light state of a dense step is the state at the nearest log frame
          (unknown when none is within 0.3 s), the log's progress D_k is its arc at t + 0.5 k minus its arc at t (linear between frames);
  actors  the rows' agent.npz (own frame, t + 0 .. 4 s; cls 0 = vehicle as on_rows.py wrote it).
A row gets ok = False (all four terms 0) when its scene has no context, its time is outside the context's keyframes, it has no actor label,
or its pose is not finite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
EXP = HERE.parents[2]
sys.path[:0] = [str(HERE.parents[3]), str(HERE.parents[1] / "lib"), str(EXP / "long_contact/lib"), str(EXP / "body1/lib")]
import on_lon as OL  # noqa: E402
import lc as LC  # noqa: E402
import b1 as B  # noqa: E402

ARC_ERR = 0.4
T_NEAR, MIN_SHARE, SIM0 = 0.3e6, 0.002, -3                # SIM0: simulation time 0 = log frame index(token) - 3 (on_rows.FRAME_OFF)
_C = {}


def rows_dir(name) -> Path:
    return B.data_dir() / "runs/onpol_nuplan/rows" / name


def tab_path(name) -> Path:
    return B.data_dir() / "runs/op_parity/cache" / f"on1_{name}" / "tab.npz"


def ctx_path() -> Path:
    return B.data_dir() / "runs/onpol_nuplan/ctx/ctx.npz"


def load_all(name: str, ctx: Path) -> dict:
    key = (name, str(ctx))
    if key not in _C:
        tab = np.load(tab_path(name), allow_pickle=False)
        m = np.load(rows_dir(name) / "meta.npz", allow_pickle=False)
        ag = np.load(rows_dir(name) / "agent.npz", allow_pickle=False)
        names = tab["names"]
        assert (ag["tokens"] == names).all() and (m["names"] == names).all(), "agent.npz / meta.npz rows are not the tab's rows"
        X = dict(np.load(ctx, allow_pickle=False))
        M = dict(t_log=m["t_log_us"].astype(np.int64), scene=m["scene"].astype(str), k=m["k"].astype(int), seed=m["seed"].astype(int), t_sim=m["t_us"].astype(np.int64), pose=m["pose_map"].astype(np.float64),
                 served=m["poses"].astype(np.float32), D=m["D"].astype(np.float32),
                 D_reg=m["D_reg"].astype(np.float32) if "D_reg" in m.files else m["D"].astype(np.float32), lead=m["lead"].astype(np.float32),
                 logfut=m["logfut"].astype(np.float32))
        _C.clear()
        _C[key] = dict(names=names, fut=tab["fut"].astype(np.float32), meta=M, box=ag["box"], valid=ag["valid"], cls=ag["cls"].astype(np.int64), aok=ag["ok"], X=X,
                       sidx={s_: i for i, s_ in enumerate(X["scene"].tolist())})
    return _C[key]


def work(u):
    """(rows name, ctx, lo, hi) -> pack arrays of the rows lo .. hi."""
    name, ctx, lo, hi = u
    G = load_all(name, Path(ctx))
    X, M = G["X"], G["meta"]
    LC.NR = X["route"].shape[1]
    S = hi - lo
    si = np.array([G["sidx"].get(s, -1) for s in M["scene"][lo:hi].tolist()])
    have = si >= 0
    sj = np.where(have, si, 0)
    pose = M["pose"][lo:hi]
    k0 = X["tok_k"][sj].astype(int) + SIM0                                                       # context keyframe of simulation time 0
    t_idx = np.take_along_axis(X["kf_ts"][sj], np.clip(k0, 0, None)[:, None], 1)[:, 0] + M["t_sim"][lo:hi]
    t = M["t_log"][lo:hi]                                                                        # the row's log time (us), on_rows.py's own (track-matched)
    have &= ~(X["kf_arc_err"][sj] > ARC_ERR) if "kf_arc_err" in X else True                      # a folded route (standstill jitter)
    ok = have & X["ok"][sj] & np.isfinite(pose).all(1) & G["aok"][lo:hi]
    pose = np.where(np.isfinite(pose), pose, 0.0)
    # frame: route (origin frame) -> the row's own frame
    o = X["origin"][sj]
    c0, s0 = np.cos(o[:, 2]), np.sin(o[:, 2])
    dx, dy = pose[:, 0] - o[:, 0], pose[:, 1] - o[:, 1]
    px, py, pyaw = c0 * dx + s0 * dy, -s0 * dx + c0 * dy, pose[:, 2] - o[:, 2]
    R0 = X["route"][sj].astype(np.float64)
    c, s = np.cos(pyaw)[:, None], np.sin(pyaw)[:, None]
    rx, ry = R0[..., 0] - px[:, None], R0[..., 1] - py[:, None]
    R = np.stack([c * rx + s * ry, -s * rx + c * ry], -1).astype(np.float32)
    H = (X["rhead"][sj] - pyaw[:, None]).astype(np.float32)
    # time: log frames by timestamp
    kt, ks = X["kf_ts"][sj], X["kf_s"][sj].astype(np.float64)
    kv = kt > 0
    last = np.where(kv, kt, 0).max(1)
    ok &= (t >= kt[:, 0] - 5e4) & (t <= last + 5e4)
    Td = t[:, None] + (np.arange(OL.NS) * OL.DT * 1e6).astype(np.int64)[None]                    # (S, 41) log time of every dense step
    dist = np.where(kv[:, None, :], np.abs(kt[:, None, :] - Td[:, :, None]), np.int64(1) << 60)
    near = dist.argmin(-1)                                                                       # (S, 41) nearest log frame
    known = np.take_along_axis(dist, near[..., None], -1)[..., 0] <= T_NEAR
    st = np.take_along_axis(X["ln_st"][sj], np.broadcast_to(near[:, None, :], (S, X["ln_st"].shape[1], OL.NS)), 2)
    ln_red = (st == 1) & known[:, None, :]
    D = np.full((S, 8), np.nan, np.float32)
    slog, dlog = np.full(S, np.nan, np.float32), np.full(S, np.nan, np.float32)
    kp = X["kf_pose"][sj]
    for r in range(S):
        if ok[r]:
            tt, ss = kt[r][kv[r]].astype(np.float64), ks[r][kv[r]]
            a = np.interp(float(t[r]), tt, ss)
            Tk = t[r] + 0.5e6 * np.arange(1, 9)
            D[r] = np.where(Tk <= tt[-1] + 5e4, np.interp(Tk, tt, ss) - a, np.nan)
            slog[r] = a
            dlog[r] = np.hypot(np.interp(float(t[r]), tt, kp[r][kv[r], 0]) - pose[r, 0], np.interp(float(t[r]), tt, kp[r][kv[r], 1]) - pose[r, 1])
    box, valid = G["box"][lo:hi].astype(np.float32), G["valid"][lo:hi]
    cls = G["cls"][lo:hi]
    Z = OL.build_rows(R, H, np.nan_to_num(X["rlen"][sj]), box, valid, cls, X["ln_type"][sj], X["ln_d"][sj], X["ln_elig"][sj], ln_red, X["tn_ok"][sj], X["tn_spk"][sj],
                      X["tn_vpk"][sj], X["vlog"][sj].astype(np.float32), D, ok=ok)
    Z |= dict(slog=slog, dlog=dlog, t_idx_diff=np.abs(t_idx - t).astype(np.float32) / 1e6, n_actor=valid[:, 0].sum(1).astype(np.int16), n_veh=((cls == 0) & valid[:, 0]).sum(1).astype(np.int16),
              red_now=ln_red[:, :, 0] & Z["b_use"], light_known=known.mean(1).astype(np.float32))
    return Z


def evaluate(Z, plans, chunk=2048):
    """Pack arrays + plans (N, 8, 3) -> per-row hinges {A, B, C, P} (N,) and the per-step maxima (N, 41 or 8)."""
    import torch
    pack = OL.to_pack(Z)
    out = {}
    for c in range(0, len(plans), chunk):
        with torch.no_grad():
            T = OL.terms(torch.tensor(plans[c:c + chunk], dtype=torch.float32), torch.arange(c, min(c + chunk, len(plans))), pack)
        for it in OL.ITEMS:
            out.setdefault(it, []).append(T[f"lon{it}"].numpy())
            out.setdefault(f"{it}_steps", []).append(T["steps"][it].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def preview(Z, M, E, served, fut, box, valid, picks, path):
    """BEV panels in the rows' own frames: route (grey), actors at t (boxes) with their 4 s tracks, served plan (blue; red where the item's hinge
    is active), imitation target (green dashed), stop-line arcs (ticks; red = must stop at t), the turn peak (orange cross)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    SW = LC.SW
    n = len(picks)
    cols = 4
    fig, axs = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.6 * cols, 4.6 * ((n + cols - 1) // cols)), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (it, r) in zip(axs.ravel(), picks):
        ax.axis("on")
        R = Z["route"][r]
        i0, i1 = int(max(Z["s0"][r] - 10, 0)), int(min(Z["s0"][r] + 80, len(R) - 1))
        ax.plot(R[i0:i1, 0], R[i0:i1, 1], color="0.6", lw=6, alpha=0.5, solid_capstyle="butt", zorder=1)
        d = OL.dense(*torch.tensor(served[r:r + 1], dtype=torch.float32).unbind(-1))[0].numpy()
        ax.plot(d[:, 0], d[:, 1], color="tab:blue", lw=1.5, zorder=5)
        ax.scatter(served[r, :, 0], served[r, :, 1], s=10, color="tab:blue", zorder=6)
        st = E[f"{it}_steps"][r]
        hot = st > 0
        if it == "P":
            ax.scatter(served[r, hot, 0], served[r, hot, 1], s=40, facecolors="none", edgecolors="red", zorder=7)
        else:
            ax.scatter(d[hot, 0], d[hot, 1], s=14, color="red", zorder=7)
        ax.plot(np.r_[0, fut[r, :, 0]], np.r_[0, fut[r, :, 1]], color="tab:green", ls="--", lw=1.2, zorder=4)
        ex, ey = SW.corners(np.array([SW.C_OFF]), np.array([0.0]), np.array([0.0]), np.array([SW.HALF_L]), np.array([SW.HALF_W]))
        ax.fill(ex[0], ey[0], color="tab:blue", alpha=0.35, zorder=3)
        slots = set(Z["a_slot"][r].tolist())
        for j in np.where(valid[r, 0])[0]:
            b = box[r, 0, j]
            qx, qy = SW.corners(b[None, 0], b[None, 1], b[None, 2], b[None, 3] / 2, b[None, 4] / 2)
            ax.fill(qx[0], qy[0], color="tab:red" if j in slots else "0.35", alpha=0.6, zorder=2)
            tr = box[r, valid[r, :, j], j]
            ax.plot(tr[:, 0], tr[:, 1], color="tab:red" if j in slots else "0.35", lw=0.8, zorder=2)
        for l in range(Z["b_d"].shape[1]):
            if Z["b_d"][r, l] < OL.BIG / 2 and Z["b_d"][r, l] < len(R) - 1:
                q = int(Z["b_d"][r, l])
                h = Z["rhead"][r, q]
                nx, ny = -np.sin(h), np.cos(h)
                col = "red" if Z["b_mu"][r, l, 0] else ("darkorange" if Z["b_mu"][r, l].any() else ("green" if Z["b_use"][r, l] else "0.2"))
                ax.plot([R[q, 0] - 3 * nx, R[q, 0] + 3 * nx], [R[q, 1] - 3 * ny, R[q, 1] + 3 * ny], color=col, lw=2.5, zorder=4)
        if Z["c_ok"][r] and Z["c_spk"][r] < len(R) - 1:
            q = int(Z["c_spk"][r])
            ax.scatter([R[q, 0]], [R[q, 1]], marker="x", s=70, color="darkorange", zorder=8)
        ax.set_aspect("equal")
        xm = max(25.0, float(np.abs(served[r, :, 0]).max()) + 12)
        ax.set_xlim(-8, xm)
        ax.set_ylim(-(xm + 8) / 2, (xm + 8) / 2)
        unit = "m/s" if it == "C" else "m"
        ax.set_title(f"{it} = {E[it][r]:.2f} {unit}  s{M['seed'][r]} k {M['k'][r]}  v {np.hypot(*served[r, 0, :2]) / 0.5:.1f} m/s\n{M['scene'][r][:19]}..{M['scene'][r][-16:]}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_pack(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    ctx = Path(a.ctx) if a.ctx else ctx_path()
    tag = f"-{a.name}"
    with Run("onpol_nuplan", f"lon-pack{tag}", config=vars(a)) as run:
        G = load_all(a.name, ctx)
        M = G["meta"]
        N = len(G["names"])
        units = [(a.name, str(ctx), c, min(c + 256, N)) for c in range(0, N, 256)]
        res = par.pmap(work, units, run=run, workers=min(n_cpus(), len(units), 48), desc="row chunks")
        res.raise_if_failed()
        Z = {k: np.concatenate([r[k] for r in res.values]) for k in res.values[0]}
        Z["names"] = G["names"]
        ok = Z["ok"]
        if a.p_open == "design":
            Z["p_open"] = Z["p_open_design"]
        served, fut = M["served"], G["fut"]
        Dm = np.where(Z["p_D"] > -OL.BIG, Z["p_D"], np.nan)                      # the log's own travel, recomputed here from the context
        Z["p_D_log"] = Z["p_D"]
        if a.p_ref == "amended":                                                # the prereg's D_k after Amendment 2 (on_rows.py meta `D`)
            Z["p_D"] = np.where(np.isfinite(M["D"]) & (Z["p_D"] > -OL.BIG), M["D"], -2 * OL.BIG).astype(np.float32)
        E, F = evaluate(Z, served), evaluate(Z, fut)
        rep_k = ok & (M["k"] <= 1) & np.isfinite(M["logfut"]).all((1, 2))       # log-replay decisions: the row's pose is the log's, so the log's own
        LG = evaluate(Z, np.nan_to_num(M["logfut"][:, 1:]))                     # future is a plan from this row ("does the log violate the item")
        far = np.abs(M["lead"]) > 5.0
        Pd = evaluate(Z | dict(p_open=Z["p_open_design"]), served)["P"]         # P under design.md's "open", for the record
        Pl = evaluate(Z | dict(p_D=Z["p_D_log"]), served)["P"]                  # P against the log's own travel (the prereg before Amendment 2)
        sh = lambda m: float(m.mean()) if len(m) else None  # noqa: E731
        q = lambda v, p: [float(x) for x in np.quantile(v, p)] if len(v) else None  # noqa: E731
        ahead = (Z["s0"] - Z["slog"])[ok]
        gate = dict(rows_name=a.name, ctx=str(ctx), rows=int(N), rows_ok=int(ok.sum()), scenes=len(set(M["scene"].tolist())),
                    not_ok=dict(no_context=int((~np.isin(M["scene"], G["X"]["scene"])).sum()), no_actor_label=int((~G["aok"]).sum()), total=int((~ok).sum())),
                    p_open=a.p_open, p_ref=a.p_ref, folded_route_rows=int((G["X"]["kf_arc_err"][[G["sidx"].get(s_, 0) for s_ in M["scene"].tolist()]] > ARC_ERR).sum()),
                    cross_check_with_on_rows=dict(
                        log_time_by_frame_index_minus_t_log_us_abs_s_max=float(Z["t_idx_diff"].max()), rows_where_they_differ_by_more_than_50ms=int((Z["t_idx_diff"] > 0.05).sum()),
                        D_abs_diff_q50_q99_max=q(np.abs(Dm - M["D_reg"])[np.isfinite(Dm) & ok[:, None]], [0.5, 0.99, 1.0]),
                        lead_abs_diff_q50_q99_max=q(np.abs((Z["s0"] - Z["slog"]) - M["lead"])[ok], [0.5, 0.99, 1.0]),
                        note="D = the log's travel over t .. t + 0.5 k (on_rows.py `D_reg`), lead = arc ahead of the log at the same time; on_rows.py computes both on its own path G"),
                    actors_per_row_mean=float(Z["n_actor"].mean()), vehicles_per_row_mean=float(Z["n_veh"].mean()),
                    lateral_offset_abs_q50_q90_q99_max=q(np.abs(Z["l0"][ok]), [0.5, 0.9, 0.99, 1.0]),
                    arc_ahead_of_log_q01_q10_q50_q90_q99=q(ahead, [0.01, 0.1, 0.5, 0.9, 0.99]),
                    rows_more_than_5m_ahead=sh(ahead > 5), rows_more_than_5m_behind=sh(ahead < -5),
                    distance_to_logged_pose_at_same_time_by_k={int(k): q(Z["dlog"][ok & (M["k"] == k)], [0.5, 0.9, 1.0]) for k in np.unique(M["k"])},
                    light_state_known_share_of_steps=float(Z["light_known"][ok].mean()),
                    context=dict(rows_with_A_eligible_actor=sh(Z["has_a"][ok]), rows_with_light_line_ahead=sh(Z["b_use"][ok].any(1)),
                                 rows_with_red_in_window=sh(Z["b_mu"][ok].any((1, 2))), rows_red_now=sh(Z["red_now"][ok].any(1)),
                                 rows_with_eligible_turn=sh(Z["c_ok"][ok]), rows_open=sh(Z["p_open"][ok]), rows_open_design=sh(Z["p_open_design"][ok]),
                                 rows_design_open_closed_by_a_line=sh((Z["p_open_design"] & Z["any_line"])[ok]),
                                 rows_design_open_closed_by_an_actor_near_the_route=sh((Z["p_open_design"] & Z["any_actor"])[ok]),
                                 rows_open_with_full_D=sh((Z["p_open"] & (Z["p_D"] > -OL.BIG).all(1))[ok]),
                                 A_margin_q10_q50_q90=q(Z["a_m"][Z["a_g0"] > 0], [0.1, 0.5, 0.9]), A_margin_zero_share=sh(Z["a_m"][Z["a_g0"] > 0] <= 0)),
                    replay_rows=int(rep_k.sum()), items={})
        kept = []
        for it in OL.ITEMS:
            h, hf = E[it], F[it]
            act = h > 0
            top = np.argsort(-h)[:20]
            share = float(act.mean())
            gate["items"][it] = dict(active_share_served=share, active_rows=int(act.sum()), active_scenes=len(set(M["scene"][act].tolist())),
                                     mean_when_active=float(h[act].mean()) if act.any() else None, q50_q90_q99_max_when_active=q(h[act], [0.5, 0.9, 0.99, 1.0]),
                                     active_share_by_k={int(k): sh(act[M["k"] == k]) for k in np.unique(M["k"])},
                                     active_share_on_imitation_target=float((hf > 0).mean()), mean_on_imitation_target_when_active=float(hf[hf > 0].mean()) if (hf > 0).any() else None,
                                     active_share_of_the_logs_own_future_at_replay_decisions=sh(LG[it][rep_k] > 0),
                                     mean_of_the_logs_own_future_when_active=float(LG[it][rep_k][LG[it][rep_k] > 0].mean()) if (LG[it][rep_k] > 0).any() else None,
                                     active_share_rows_within_5m_of_the_log=sh(act[ok & ~far]), active_share_rows_beyond_5m_of_the_log=sh(act[ok & far]),
                                     passes_0p2_percent=bool(share >= MIN_SHARE),
                                     top20=[dict(name=str(Z["names"][r]), scene=str(M["scene"][r]), seed=int(M["seed"][r]), k=int(M["k"][r]), value=float(h[r]), lead_m=float(M["lead"][r]),
                                                lateral_m=float(Z["l0"][r])) for r in top if h[r] > 0])
            if share >= MIN_SHARE:
                kept.append(it)
        items = tuple(a.items) if a.items else tuple(kept)
        gate["items"]["P"] |= dict(active_share_served_with_log_travel=float((Pl > 0).mean()), mean_when_active_with_log_travel=float(Pl[Pl > 0].mean()) if (Pl > 0).any() else None,
                                   active_share_served_with_design_open=float((Pd > 0).mean()), mean_when_active_with_design_open=float(Pd[Pd > 0].mean()) if (Pd > 0).any() else None)
        gate |= dict(items_in_pack=list(items), items_dropped_by_the_0p2_percent_rule=[i for i in OL.ITEMS if i not in kept], items_forced=bool(a.items))
        Z["items"] = np.array(items)
        out = rows_dir(a.name) / OL.PACK_NAME if not a.out else Path(a.out)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **Z)
        tmp.rename(out)
        gate["pack"] = str(out)
        # preview: the 3 most active rows of each item from different scenes
        picks = []
        for it in OL.ITEMS:
            seen = set()
            for r in np.argsort(-E[it]):
                if E[it][r] <= 0 or len(seen) == 3:
                    break
                if M["scene"][r] not in seen:
                    seen.add(M["scene"][r])
                    picks.append((it, int(r)))
        gd = B.data_dir() / "runs/onpol_nuplan/lon"
        (gd / "gates").mkdir(parents=True, exist_ok=True)
        (gd / "figs").mkdir(parents=True, exist_ok=True)
        if picks:
            preview(Z, M, E, served, fut, G["box"], G["valid"], picks, gd / "figs" / f"lon_preview{tag}.png")
            gate["preview"] = dict(path=str(gd / "figs" / f"lon_preview{tag}.png"), panels=[dict(item=i, name=str(Z["names"][r])) for i, r in picks])
        p = gd / "gates" / f"lon_pool{tag}.json"
        p.write_text(json.dumps(gate, indent=1))
        run.info("%s", json.dumps({k: v for k, v in gate.items() if k != "items"}))
        for it in OL.ITEMS:
            run.info("%s: %s", it, json.dumps({k: v for k, v in gate["items"][it].items() if k != "top20"}))
        run.summary.update(pack=str(out), gate=str(p), items=list(items), rows=int(N), rows_ok=int(ok.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pack"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--items", nargs="*", default=None)
    ap.add_argument("--p-ref", choices=["amended", "log"], default="amended",
                    help="P's reference travel: the prereg's D_k after Amendment 2 (speed blends from the ego's into the log's over 2 s) or the log's own travel")
    ap.add_argument("--p-open", choices=["strict", "design"], default="strict", help="which rows carry P (lib/on_lon.py docstring)")
    cmd_pack(ap.parse_args())
