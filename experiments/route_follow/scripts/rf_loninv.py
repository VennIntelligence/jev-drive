#!/usr/bin/env python3
"""LON-INVENTORY (plans/2026-10-16-lon-inventory.md): how much material exists for longitudinal envelope rows on the policy's own states.
CPU only, read-only on stores; box, op-train python (torch for on_lon's plan geometry). No training, no simulation.

  rf_loninv.py count --out DIR [--pool L18RSG] [--packs DIR ..] [--no-human]
      -> DIR/inventory.json, DIR/tables.md, DIR/<class>.csv (one line per decision of each class), DIR/pack_check.json

Policy store: the on-policy pool on1_<pool> (rows dir runs/onpol_nuplan/rows/<pool>: meta / all / lon.npz; tab + @warp frames in
runs/op_parity/cache/on1_<pool>) joined with the scene context runs/onpol_nuplan/ctx/ctx.npz (LONG-PROBE lc_ctx for the 1491 public scenes:
route on a 1 m grid in the origin frame, stop lines with type / arc / light states at 24 keyframes / ln_tstop / ln_elig, the human's speed by arc).
A row is "on route" when its pack row is ok and |l0| <= 3.5 m (on_lon's LAT_ROUTE). Distances to lines are from the front bumper (ln_d - sf0).
Light state of a row = the state at the context keyframe nearest the row's log time (<= 0.3 s, on_lon_pack's rule), else unknown.

Classes (policy speed = the fed speed, tab `speed`):
  red     the first traffic-light line ahead (d >= 0) is 0 .. 40 m, red now; speed >= 6
  stop    the first stop-sign line ahead is 0 .. 40 m; speed >= 6
  qual    the first LONG-PROBE-eligible line ahead (ln_elig, either type) is 0 .. 40 m and must stop now (stop sign, or light red now); speed >= 6
          per line: plan_le1 = the served plan reaches <= 1 m/s with its front bumper before the line (dense 0.1 s, on_lon.terms geometry);
          cross4 = the plan's front bumper passes the line within 4 s; nostop_kin = not plan_le1 and the 4 s end state cannot stop before
          the line at <= 3 m/s^2; human_stop = ln_tstop finite (the logged driver < 1 m/s with the rear axle in [d - 15, d - 4.049 + 1.5]
          before crossing); red_4s = every known keyframe in [t, t + 4 s] red
  free    rf_lonol's free cell: no A-eligible actor with gap <= 40 m, no stop line of any type -0.5 .. 60 m, route heading change over
          the next 60 m < 30 deg, speed >= 2, the human's speed at the row's station known; slow_h = speed < 0.85 x human (ctx vlog at s0),
          slow_x = speed < 0.85 x expert v_ref at the station (real pack), slow_p = the same against the proxy v_ref
  turn    the first strict turn ahead (rf_follow.find_turn: 30 m window >= 60 deg net, change over [a - 10, a + 40] >= 45 deg; entry at
          10 %) on the route's logged stretch has its entry 0 .. 40 m ahead of the rear axle; fast_x / fast_p / fast_h = speed > expert
          (pack / proxy) / human speed at the station + 1 m/s
Expert v_ref: rule_expert.Scene.profile(V_E 4.6) of the scene's pack (rf_expert.py pack), at the station of the row's rear-axle anchor
(rollout frame = the pack's frame, checked on the decision-0 anchors against the pack's log path). Proxy v_ref: the same profile function
on the context route (rf_expert.turns on it, ctx vlog), for every scene; its agreement with the packs is reported.
Human comparison: navtrain (12 shards, long_contact ctx + body1 agent labels), the same classes on the human's own state (speed, ax).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "experiments/onpol_nuplan/lib"), str(_R / "experiments/body1/lib"), str(_R / "experiments/long_contact/lib"),
                 str(_R / "experiments/alpasim/lib"), str(_pl.Path(__file__).resolve().parent)]
from jevdrive.tokens import open_front  # noqa: E402
import argparse, glob, json, os  # noqa: E401,E402

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

FRONT, LAT, BIG = 4.049, 3.5, 1e3
V_FAST, D_APP, D_FREE_LINE, D_FREE_LEAD, TURN_FREE = 6.0, 40.0, 60.0, 40.0, 30.0
SLOW_R, FAST_D, V_STOP, A_STOP = 0.85, 1.0, 1.0, 3.0
BINS, BIN_NAMES = (8.0, 12.0, 16.0), ("< 8", "8-12", "12-16", ">= 16")
LINE_CLASSES = ("red", "stop", "qual")
PACK_DIRS = ("runs/route_follow/rewind_train64/pack", "runs/route_follow/rule_expert/pack_held", "runs/route_follow/rule_expert/pack_test")


def wrapd(a):
    return np.degrees(np.abs(np.angle(np.exp(1j * a))))


# ---------------------------------------------------------------- geometry shared by both stores
def strict_turns(R, hi):
    """All strict turns (rf_follow.find_turn, applied again after each exit) on a route polyline R (n, 2) at 1 m within [0, hi] -> [(entry, exit)]."""
    import rf_follow as RF
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]
    s, h = RF.heading(R, S)
    w = int(round(RF.WIN / (s[1] - s[0])))
    i1 = np.searchsorted(s, hi, side="right")
    dh = np.zeros(len(s))
    dh[:len(s) - w] = np.degrees(np.abs(h[w:] - h[:-w]))
    out, lo = [], 0.0
    while True:
        i0 = np.searchsorted(s, lo)
        cand = [a for a in np.flatnonzero(dh >= RF.TURN_DEG) if i0 <= a < i1 - w]
        t = RF.find_turn(s, h, s[cand[0]], hi) if cand else None    # find_turn from the first candidate on = find_turn from lo (exact)
        if t is None:
            return out
        out.append((t[0], t[1]))
        lo = t[1] + 1.0


def proxy_profile(R, vlog):
    """rule_expert.Scene.profile(V_E) on a 1 m route grid R (n, 2) with the human's speed by arc vlog (n,) -> v_ref (n,)."""
    import rule_expert as RE
    import rf_expert as RX
    sc = RE.Scene.__new__(RE.Scene)
    sc.P, sc.S = R.astype(float), np.arange(len(R), dtype=float)
    sc.ds = 1.0
    sc.turns = RX.turns(sc.P, sc.S)
    sc.vlog = vlog.astype(float)
    sc.kap = sc.curvature()
    return sc.profile(RE.V_E)


def first_ahead(mask, d):
    """mask (S, L) candidate lines, d (S, L) distances -> index of the nearest line with mask & d >= 0 (or -1)."""
    dd = np.where(mask & np.isfinite(d) & (d >= 0), d, np.inf)
    q = dd.argmin(1)
    return np.where(np.isfinite(dd[np.arange(len(d)), q]), q, -1)


def take(a, q):
    return a[np.arange(len(q)), np.maximum(q, 0)]


def bins_of(v):
    return np.digitize(v, BINS)


# ---------------------------------------------------------------- policy store
def plan_geometry(Z, served):
    """Served plans (N, 8, 3) -> front-bumper arc sf (N, 41), path speed v (N, 41) at 0.1 s (on_lon.terms geometry)."""
    import torch
    import on_lon as OL
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    pack = OL.to_pack(Z)
    sf, v = [], []
    for c in range(0, len(served), 1024):
        with torch.no_grad():
            T = OL.terms(torch.tensor(served[c:c + 1024], dtype=torch.float32), torch.arange(c, min(c + 1024, len(served))), pack)
        sf.append(T["steps"]["sf"].numpy())
        v.append(T["steps"]["v"].numpy())
    return np.concatenate(sf), np.concatenate(v)


def policy(a, run):
    import b1 as B
    D = B.data_dir()
    rd = D / "runs/onpol_nuplan/rows" / a.pool
    cache = D / "runs/op_parity/cache" / f"on1_{a.pool}"
    tab = np.load(cache / "tab.npz")
    m = np.load(rd / "meta.npz")
    Z = dict(np.load(rd / "lon.npz"))
    al = np.load(rd / "all.npz")
    X = dict(np.load(D / "runs/onpol_nuplan/ctx/ctx.npz"))
    names = tab["names"]
    assert (m["names"] == names).all() and (Z["names"] == names).all(), "tab / meta / lon rows differ"
    front = open_front(D / "runs/op_parity/cache" / f"on1_{a.pool}@warp" / "front.npy")
    assert front.shape[0] == len(names), "frames do not cover the tab"
    N = len(names)
    scene, k, logf = m["scene"].astype(str), m["k"].astype(int), m["log"].astype(str)
    speed = tab["speed"].astype(np.float64)
    sidx = {s: i for i, s in enumerate(X["scene"].tolist())}
    sj = np.array([sidx[s] for s in scene])
    ok = Z["ok"].astype(bool)
    onr = ok & (np.abs(Z["l0"]) <= LAT)
    s0, sf0 = Z["s0"].astype(np.float64), Z["sf0"].astype(np.float64)
    run.info("rows %d (scenes %d), ok %d, on route %d", N, len(set(scene)), ok.sum(), onr.sum())

    # lines
    ty, ld, st, tstop, elig = X["ln_type"][sj], X["ln_d"][sj].astype(np.float64), X["ln_st"][sj], X["ln_tstop"][sj], X["ln_elig"][sj]
    dF = ld - sf0[:, None]
    kt = X["kf_ts"][sj]
    t = m["t_log_us"].astype(np.int64)
    dist = np.where(kt > 0, np.abs(kt - t[:, None]), np.int64(1) << 60)
    near = dist.argmin(1)
    known = dist[np.arange(N), near] <= 3e5
    st_now = np.where(known[:, None], st[np.arange(N), :, near], -1)
    win = (kt >= t[:, None] - 3e5) & (kt <= t[:, None] + 4e6) & (kt > 0)                       # (N, 24) keyframes of the 4 s window
    stw = np.where(win[:, None, :], st, -1)
    red4 = ((stw == 1) | (stw < 0)).all(2) & (stw == 1).any(2)                                  # (N, L)
    sf, v = plan_geometry(Z, m["poses"].astype(np.float32))

    def line_class(mask):
        q = first_ahead(mask, dF)
        d = take(dF, q)
        base = onr & (q >= 0) & (d <= D_APP)
        sel = base & (speed >= V_FAST)
        dl = take(ld, q)[:, None]
        le1 = ((sf <= dl) & (v <= V_STOP)).any(1)
        cross4 = sf[:, -1] > dl[:, 0]
        rem = np.maximum(dl[:, 0] - sf[:, -1], 0.05)
        can = ~cross4 & (v[:, -1] ** 2 / (2 * rem) <= A_STOP)
        icr = np.where(cross4, (sf > dl).argmax(1), 40)
        req = speed ** 2 / (2 * np.maximum(d, 0.5))                                             # deceleration that stops the car at the line
        got = (speed - v[:, -1]) / 4.0                                                             # the plan's mean deceleration over 4 s
        rec = dict(sel=sel, base=base, q=q, d=d, plan_le1=le1, no_intent=v[:, -1] / np.maximum(speed, 0.1) >= 0.7, under_brake=got < 0.5 * req,
                   req_decel=req, plan_decel=got, nostop=~le1, cross4=cross4, nostop_kin=~le1 & ~can, v_cross=np.where(cross4, v[np.arange(N), icr], np.nan),
                   human_stop=np.isfinite(take(tstop, q)), red_4s=take(red4, q), ltype=take(ty, q), elig=take(elig, q), plan_end_ratio=v[:, -1] / np.maximum(speed, 0.1))
        return rec

    C = dict(red=line_class(ty == 2), stop=line_class(ty == 1), qual=line_class(elig & (ty > 0)))
    # 'red' must be the first light line and red now; 'qual' the first eligible line and must-stop now
    for key in ("sel", "base"):
        C["red"][key] &= take(st_now, C["red"]["q"]) == 1
        C["qual"][key] &= (C["qual"]["ltype"] == 1) | ((C["qual"]["ltype"] == 2) & (take(st_now, C["qual"]["q"]) == 1))

    # free road
    lead40 = ((Z["a_g0"] > -BIG / 2) & (Z["a_g0"] <= D_FREE_LEAD)).any(1)
    line60 = ((ty > 0) & np.isfinite(dF) & (dF > -0.5) & (dF <= D_FREE_LINE)).any(1)
    i0 = np.clip(np.round(s0).astype(int), 0, Z["rhead"].shape[1] - 1)
    rh = Z["rhead"].astype(np.float64)
    idx = np.clip(i0[:, None] + np.arange(61)[None], 0, rh.shape[1] - 1)
    turn60 = wrapd(np.take_along_axis(rh, idx, 1) - rh[np.arange(N), i0][:, None]).max(1)
    grid = np.arange(X["vlog"].shape[1], dtype=np.float64)
    vlog = X["vlog"][sj].astype(np.float64)
    vh = np.array([np.interp(s0[r], grid, vlog[r]) if 0 <= s0[r] <= grid[-1] else np.nan for r in range(N)])
    free = onr & ~lead40 & ~line60 & (turn60 < TURN_FREE) & (speed >= 2.0) & np.isfinite(vh)

    # strict turns on the context route of each scene, and the proxy v_ref
    uniq = sorted(set(sj.tolist()))
    turns, vprox = {}, {}
    for g in uniq:
        Rg = X["route"][g].astype(np.float64)
        turns[g] = strict_turns(Rg, float(X["hlen"][g]))
        vprox[g] = proxy_profile(Rg, X["vlog"][g].astype(np.float64))
    vp = np.array([np.interp(s0[r], grid, vprox[sj[r]]) for r in range(N)])
    ent = np.full(N, np.nan)
    for r in range(N):
        e = [en for en, ex in turns[sj[r]] if en >= s0[r]]
        if e:
            ent[r] = e[0] - s0[r]
    turn = onr & np.isfinite(ent) & (ent <= D_APP)

    # expert packs: v_ref at the row's anchor station
    have = {}
    for pdir in a.packs or [str(D / p) for p in PACK_DIRS]:
        for f in glob.glob(os.path.join(pdir, "*.npz")):
            have.setdefault(_pl.Path(f).stem, f)
    import rule_expert as RE
    vx = np.full(N, np.nan)
    chk = {}
    anchor = m["anchor"].astype(np.float64)
    for sc_ in sorted(set(scene) & set(have)):
        sc = RE.Scene(_pl.Path(have[sc_]))
        z = np.load(have[sc_])
        LP = z["log_xy"].astype(float)
        rr = np.flatnonzero(scene == sc_)
        dlog = []
        for r in rr:
            s_e, _ = sc.project(anchor[r, :2])
            vx[r] = np.interp(s_e, sc.S, sc.vbase)
            if k[r] == 0:
                dd = LP - anchor[r, :2]
                dlog.append(float(np.sqrt((dd ** 2).sum(1)).min()))
        chk[sc_] = dict(rows=len(rr), k0_anchor_to_pack_log_m=dlog[0] if dlog else None)
    xs = np.isfinite(vx)
    pc = np.abs(vx - vp)[xs & onr]
    pack_check = dict(pack_scenes_found=len(have), pool_scenes_with_pack=len(chk), rows_with_pack=int(xs.sum()),
                      k0_anchor_to_log_m_q50_max=[float(np.nanmedian([c["k0_anchor_to_pack_log_m"] for c in chk.values() if c["k0_anchor_to_pack_log_m"] is not None])),
                                                  float(np.nanmax([c["k0_anchor_to_pack_log_m"] for c in chk.values() if c["k0_anchor_to_pack_log_m"] is not None]))] if chk else None,
                      proxy_vs_pack_abs_q50_q90=[float(np.quantile(pc, 0.5)), float(np.quantile(pc, 0.9))] if len(pc) else None,
                      proxy_vs_pack_within_1ms=float((pc <= 1.0).mean()) if len(pc) else None,
                      proxy_vs_pack_free_rows_abs_q50=float(np.median(np.abs(vx - vp)[xs & free])) if (xs & free).any() else None,
                      proxy_vs_pack_turn_rows_abs_q50=float(np.median(np.abs(vx - vp)[xs & turn])) if (xs & turn).any() else None)

    base = dict(name=names, scene=scene, log=logf, k=k, speed=np.round(speed, 3), s0=np.round(s0, 2), l0=np.round(Z["l0"], 3))
    out = {}
    for c in LINE_CLASSES:
        R = C[c]
        out[c] = dict(sel=R["sel"], cols=dict(base, d_line=np.round(R["d"], 2), line_type=R["ltype"], elig=R["elig"], plan_le1=R["plan_le1"], cross4=R["cross4"],
                                                nostop_kin=R["nostop_kin"], no_intent=R["no_intent"], under_brake=R["under_brake"],
                                                req_decel=np.round(R["req_decel"], 2), plan_decel=np.round(R["plan_decel"], 2), v_cross=np.round(R["v_cross"], 2), plan_end_ratio=np.round(R["plan_end_ratio"], 3),
                                                human_stop=R["human_stop"], red_4s=R["red_4s"]),
                      flags=dict(nostop=R["nostop"], cross4=R["cross4"], nostop_kin=R["nostop_kin"], human_stop=R["human_stop"],
                                 nostop_and_human_stop=R["nostop"] & R["human_stop"], nostop_kin_and_human_stop=R["nostop_kin"] & R["human_stop"],
                                 no_intent=R["no_intent"], no_intent_and_human_stop=R["no_intent"] & R["human_stop"],
                                 under_brake=R["under_brake"], under_brake_and_human_stop=R["under_brake"] & R["human_stop"], red_4s=R["red_4s"]))
    slow_h, slow_x, slow_p = speed < SLOW_R * vh, speed < SLOW_R * vx, speed < SLOW_R * vp
    out["free"] = dict(sel=free, cols=dict(base, v_human=np.round(vh, 2), v_expert=np.round(vx, 2), v_proxy=np.round(vp, 2), turn60=np.round(turn60, 1),
                                           slow_h=slow_h, slow_x=slow_x & xs, slow_p=slow_p),
                       flags=dict(slow_h=slow_h, slow_x=slow_x & xs, has_pack=xs, slow_p=slow_p),
                       binv=vh)
    fast_x, fast_p, fast_h = speed > vx + FAST_D, speed > vp + FAST_D, speed > vh + FAST_D
    out["turn"] = dict(sel=turn, cols=dict(base, d_entry=np.round(ent, 2), v_human=np.round(vh, 2), v_expert=np.round(vx, 2), v_proxy=np.round(vp, 2),
                                           fast_x=fast_x & xs, fast_p=fast_p, fast_h=fast_h),
                       flags=dict(fast_x=fast_x & xs, has_pack=xs, fast_p=fast_p, fast_h=fast_h, fast_p_and_h=fast_p & fast_h))

    def stats(sel):
        return dict(decisions=int(sel.sum()), scenes=len(set(scene[sel].tolist())), logs=len(set(logf[sel].tolist())), with_frames=int(sel.sum()),
                    replay_k_le1=int((sel & (k <= 1)).sum()))
    res = {}
    for c, O in out.items():
        r = dict(all=stats(O["sel"]), sub={f: stats(O["sel"] & fl) for f, fl in O["flags"].items()})
        if "binv" in O:
            b = bins_of(O["binv"])
            r["by_human_speed"] = {BIN_NAMES[i]: dict(all=stats(O["sel"] & (b == i)), **{f: stats(O["sel"] & (b == i) & fl) for f, fl in O["flags"].items()}) for i in range(4)}
            bx = bins_of(vx)
            r["by_expert_speed"] = {BIN_NAMES[i]: dict(all=stats(O["sel"] & xs & (bx == i)), slow_x=stats(O["sel"] & xs & (bx == i) & slow_x)) for i in range(4)}
        if c in LINE_CLASSES:
            r["speed_q50_q90"] = [float(np.quantile(speed[O["sel"]], q)) for q in (0.5, 0.9)] if O["sel"].any() else None
            B0, hs0, ni = C[c]["base"], C[c]["human_stop"], C[c]["no_intent"]
            r["by_speed_floor"] = {f"speed >= {f:g}": dict(all=stats(B0 & (speed >= f)), human_stop=stats(B0 & (speed >= f) & hs0),
                                                            no_intent_and_human_stop=stats(B0 & (speed >= f) & hs0 & ni)) for f in (0.0, 3.0, 6.0, 9.0, 12.0)}
            hs = O["sel"] & C[c]["human_stop"] & (k >= 2)
            r["own_state_human_stop_plan_decel_vs_required_q50"] = [float(np.median(C[c]["plan_decel"][hs])), float(np.median(C[c]["req_decel"][hs]))] if hs.any() else None
            r["d_line_bins"] = {lab: stats(O["sel"] & (C[c]["d"] >= lo) & (C[c]["d"] < hi)) for lab, lo, hi in (("0-10", 0, 10), ("10-20", 10, 20), ("20-40", 20, 40.001))}
        res[c] = r
        cols = O["cols"]
        sel = O["sel"]
        with open(_pl.Path(a.out) / f"policy_{c}.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for i in np.flatnonzero(sel):
                fh.write(",".join(str(int(x)) if isinstance(x, (bool, np.bool_)) else str(x) for x in (cols[q][i] for q in cols)) + "\n")
    import collections
    excl = collections.Counter(al["excl"].astype(str).tolist())
    meta = dict(pool=a.pool, rows=int(N), scenes_rows=len(set(scene)), logs_rows=len(set(logf)), rows_ok=int(ok.sum()), rows_on_route=int(onr.sum()),
                decisions_all=int(len(al["excl"])), scenes_all=len(set(al["scene"].astype(str).tolist())), excluded_by_reason={kk: int(vv) for kk, vv in excl.items() if kk},
                frames=str(_pl.Path(str(cache) + "@warp") / "front.npy"), frames_shape=list(front.shape), light_known_rows=float(known[onr].mean()),
                policy_speed_q50_q90_q99_max=[float(np.quantile(speed, q)) for q in (0.5, 0.9, 0.99, 1.0)],
                dumps_dir="runs/onpol_long/roll/RSG-L18-s0/c*/run/driver-logs/dump (one npz per decision, excluded decisions included)")
    return dict(meta=meta, classes=res, pack_check=pack_check, pack_scenes=sorted(chk))


# ---------------------------------------------------------------- human navtrain
def human_shard(k):
    import b1 as B
    import lc as LC
    import on_lon as OL
    LC.NR = 131
    L, off = B.labels(), B.shard_offsets()
    z = np.load(B.data_dir() / "runs/long_contact/ctx/ctx.npz")
    g = np.arange(off[k], off[k + 1])
    keys = ("route", "rhead", "rlen", "hlen", "vlog", "ln_type", "ln_d", "ln_st", "ln_tstop", "ln_elig", "tn_ok", "tn_spk", "tn_vpk", "tokens", "ok")
    Z = {q: z[q][g] for q in keys}
    t = B.tab("log", k)
    assert np.array_equal(t["names"], Z["tokens"]) and np.array_equal(t["names"], L["tokens"][g]), f"shard {k}: rows"
    n = len(g)
    v0, ax = t["speed"].astype(float), t["acc"][:, -1, 0].astype(float)
    a_g0 = []
    kf = np.clip(np.round(np.arange(41) * 0.1 / 0.5).astype(int), 0, 8)
    for i in range(0, n, 512):
        s = slice(i, min(i + 512, n))
        gs = g[s]
        r = OL.build_rows(Z["route"][s], Z["rhead"][s], Z["rlen"][s], np.asarray(L["box"][gs]), np.asarray(L["valid"][gs]), np.asarray(L["cls"][gs]),
                          Z["ln_type"][s], Z["ln_d"][s], Z["ln_elig"][s], (Z["ln_st"][s][:, :, kf] == 1), Z["tn_ok"][s], Z["tn_spk"][s], Z["tn_vpk"][s],
                          Z["vlog"][s].astype(np.float32), np.zeros((len(gs), 8), np.float32))
        a_g0.append(r["a_g0"])
    a_g0 = np.concatenate(a_g0)
    ok = Z["ok"].astype(bool)
    ty, dF, st0 = Z["ln_type"], Z["ln_d"].astype(float) - FRONT, Z["ln_st"][:, :, 0]
    out = dict(log=t["log"].astype(str), v0=v0, ax=ax)
    for c, mask in (("red", ty == 2), ("stop", ty == 1), ("qual", Z["ln_elig"] & (ty > 0))):
        q = first_ahead(mask, dF)
        d = take(dF, q)
        sel = ok & (q >= 0) & (d <= D_APP) & (v0 >= V_FAST)
        if c == "red":
            sel &= take(st0, q) == 1
        if c == "qual":
            lt = take(ty, q)
            sel &= (lt == 1) | ((lt == 2) & (take(st0, q) == 1))
        out[c] = sel
        out[c + "_human_stop"] = sel & np.isfinite(take(Z["ln_tstop"], q))
        out[c + "_decel"] = sel & (ax < -0.5)
    lead40 = ((a_g0 > -BIG / 2) & (a_g0 <= D_FREE_LEAD)).any(1)
    line60 = ((ty > 0) & np.isfinite(dF) & (dF <= D_FREE_LINE)).any(1)
    rh = Z["rhead"][:, :61].astype(float)
    turn60 = wrapd(rh - rh[:, :1]).max(1)
    out["free"] = ok & ~lead40 & ~line60 & (turn60 < TURN_FREE) & (v0 >= 2.0)
    ent = np.full(n, np.nan)
    vp0 = np.full(n, np.nan)
    for i in range(n):
        if not ok[i]:
            continue
        R = Z["route"][i].astype(float)
        tt = strict_turns(R, float(Z["hlen"][i]))
        if tt and tt[0][0] <= D_APP:
            ent[i] = tt[0][0]
            vp0[i] = proxy_profile(R, Z["vlog"][i].astype(float))[0]
    out["turn"] = ok & np.isfinite(ent)
    out["turn_fast_p"] = out["turn"] & (v0 > vp0 + FAST_D)
    return out


def human(run):
    from jevdrive import par
    res = par.pmap(human_shard, list(range(12)), run=run, workers=12, desc="navtrain shards")
    res.raise_if_failed()
    H = {q: np.concatenate([r[q] for r in res.values]) for q in res.values[0]}
    log = H["log"]
    st = lambda m: dict(rows=int(m.sum()), logs=len(set(log[m].tolist())))  # noqa: E731
    o = dict(tokens=int(len(log)), logs=len(set(log.tolist())))
    for c in LINE_CLASSES:
        o[c] = dict(all=st(H[c]), human_stop=st(H[c + "_human_stop"]), decelerating_ax_lt_m0p5=st(H[c + "_decel"]))
    b = bins_of(H["v0"])
    o["free"] = dict(all=st(H["free"]), by_speed={BIN_NAMES[i]: st(H["free"] & (b == i)) for i in range(4)})
    o["turn"] = dict(all=st(H["turn"]), faster_than_proxy_plus1=st(H["turn_fast_p"]))
    o["speed_q50_q90_q99_max_all"] = [float(np.quantile(H["v0"], q)) for q in (0.5, 0.9, 0.99, 1.0)]
    o["rows_ge12"], o["rows_ge16"] = int((H["v0"] >= 12).sum()), int((H["v0"] >= 16).sum())
    return o


def paitv():
    import b1 as B
    D = B.data_dir()
    t = np.load(D / "runs/op_parity/cache/on1_PAITV/tab.npz")
    v = t["speed"].astype(float)
    sc = t["log"].astype(str)
    b = bins_of(v)
    return dict(rows=int(len(v)), scenes=len(set(sc.tolist())), speed_q10_q50_q90_max=[float(np.quantile(v, q)) for q in (0.1, 0.5, 0.9, 1.0)],
                by_speed={BIN_NAMES[i]: int((b == i).sum()) for i in range(4)}, frames=(D / "runs/op_parity/cache/on1_PAITV@warp/front.npy").exists())


# ---------------------------------------------------------------- tables
def tables(J):
    P, H = J["policy"], J.get("human")
    f = lambda s: f"{s['decisions']} / {s['scenes']} / {s['logs']} / {s['with_frames']}"  # noqa: E731
    L = ["## Policy store (decisions / scenes / logs / with frames)", ""]
    names = dict(nostop="plan not <= 1 m/s before the line", cross4="plan crosses within 4 s", nostop_kin="plan cannot stop (<= 3 m/s^2)",
                 human_stop="human stopped at the line", no_intent="plan end speed >= 0.7 x speed (no intent)",
                 no_intent_and_human_stop="no intent AND human stopped", under_brake="plan decel < 0.5 x needed to stop at the line",
                 under_brake_and_human_stop="under-braking AND human stopped", nostop_and_human_stop="not <= 1 m/s AND human stopped",
                 nostop_kin_and_human_stop="cannot stop AND human stopped", red_4s="red through the 4 s window")
    for c in LINE_CLASSES:
        R = P["classes"][c]
        L += [f"### {c}", "", "| subset | decisions / scenes / logs / frames | replay k <= 1 |", "|:--|--:|--:|", f"| all | {f(R['all'])} | {R['all']['replay_k_le1']} |"]
        L += [f"| {names[q]} | {f(s)} | {s['replay_k_le1']} |" for q, s in R["sub"].items()]
        L += ["", "| line distance | all |", "|:--|--:|"] + [f"| {b} m | {f(s)} |" for b, s in R["d_line_bins"].items()] + [""]
    for c in ("free", "turn"):
        R = P["classes"][c]
        L += [f"### {c}", "", "| subset | decisions / scenes / logs / frames | replay k <= 1 |", "|:--|--:|--:|", f"| all | {f(R['all'])} | {R['all']['replay_k_le1']} |"]
        L += [f"| {q} | {f(s)} | {s['replay_k_le1']} |" for q, s in R["sub"].items()] + [""]
        if "by_human_speed" in R:
            L += ["| human speed | all | slow_h (< 0.85 x human) | slow_p (< 0.85 x proxy) | with pack | slow_x (< 0.85 x expert) |", "|:--|--:|--:|--:|--:|--:|"]
            for b, s in R["by_human_speed"].items():
                L.append(f"| {b} | {f(s['all'])} | {f(s['slow_h'])} | {f(s['slow_p'])} | {f(s['has_pack'])} | {f(s['slow_x'])} |")
            L += ["", "| expert speed | rows with pack | slow_x |", "|:--|--:|--:|"]
            L += [f"| {b} | {f(s['all'])} | {f(s['slow_x'])} |" for b, s in R["by_expert_speed"].items()] + [""]
    if H:
        L += ["## Human navtrain (rows / logs)", "", f"tokens {H['tokens']}, logs {H['logs']}; speed q50 / q90 / q99 / max {H['speed_q50_q90_q99_max_all']}", ""]
        for c in LINE_CLASSES:
            L.append(f"- {c}: all {H[c]['all']}, human stopped {H[c]['human_stop']}, decelerating {H[c]['decelerating_ax_lt_m0p5']}")
        L.append(f"- free: {H['free']['all']}; by own speed {H['free']['by_speed']}")
        L.append(f"- turn approach: {H['turn']['all']}; faster than proxy + 1 {H['turn']['faster_than_proxy_plus1']}")
    return "\n".join(L) + "\n"


def cmd_count(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "lon-inventory", config=vars(a)) as run:
        run.use_split(splits.load("alpasim_nuplan/onpol-train"))
        if not a.no_human:
            run.use_split(splits.load("navsim/navtrain"))
        J = dict(policy=policy(a, run), paitv=paitv())
        if not a.no_human:
            J["human"] = human(run)
        (out / "inventory.json").write_text(json.dumps(J, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
        (out / "pack_check.json").write_text(json.dumps(J["policy"]["pack_check"], indent=1))
        (out / "tables.md").write_text(tables(J))
        run.summary.update(out=str(out))
        print(tables(J), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["count"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="L18RSG")
    ap.add_argument("--packs", nargs="*", default=None)
    ap.add_argument("--no-human", action="store_true")
    cmd_count(ap.parse_args())
