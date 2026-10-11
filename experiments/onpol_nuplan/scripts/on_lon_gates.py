#!/usr/bin/env python3
"""Gates of the self-consequence L module (lib/on_lon.py). CPU; $DATA_DIR/envs/op-train/bin/python.

  grad      (i)   timing_only: values bit for bit, zero cross-heading and yaw gradient of every item, the kept gradient equals the along-heading
                  part of the unprojected one (autograd, float64) and a central finite difference along the heading
  kat       (ii)  torch operators (project, dense, front point, speed, plan arc) against lc.py / sweep.py numpy on random routes and poses
  navtrain  (iii) label agreement with LONG-PROBE on navtrain states (shards 2, 3; log / ot1 / yr1) with the stored `P2H10-F` own plans
                  (body1 rows, queries 0 and 1): among a_lab = 1 the A hinge at margin 0 is > 0, among red-light b_lab = 1 the B hinge at the
                  line itself, among c_lab = 1 the C hinge as implemented (its cap is below the label's threshold by construction)
  pool      (iv)  see scripts/on_lon_pack.py (activation on the pool's stored served plans)
  -> $DATA_DIR/runs/onpol_nuplan/lon/gates/lon_<name>.json (copied to experiments/onpol_nuplan/results/gates/)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
EXP = HERE.parents[2]
sys.path[:0] = [str(HERE.parents[3]), str(HERE.parents[1] / "lib"), str(EXP / "long_contact/lib"), str(EXP / "body1/lib")]
import on_lon as OL  # noqa: E402
import lc as LC  # noqa: E402
import b1 as B  # noqa: E402

SW = LC.SW


def gdir() -> Path:
    p = B.data_dir() / "runs/onpol_nuplan/lon/gates"
    p.mkdir(parents=True, exist_ok=True)
    return p


def rand_routes(rng, S, NR):
    """Smooth random 1 m grids starting near the origin heading +x."""
    k = np.cumsum(rng.normal(0, 0.004, (S, NR)), 1) + rng.normal(0, 0.01, (S, 1))
    h = np.cumsum(k, 1)
    h -= h[:, :1]
    xy = np.concatenate([np.zeros((S, 1, 2)), np.cumsum(np.stack([np.cos(h[:, :-1]), np.sin(h[:, :-1])], -1), 1)], 1)
    return xy.astype(np.float32), h.astype(np.float32)


def rand_plans(rng, S, s0=8.0):
    """Plans (S, 8, 3) in a frame whose origin sits near arc s0 of a route: speed 0 .. 15 m/s, mild curvature, lateral / heading noise."""
    v = rng.uniform(0, 15, (S, 1)) * np.clip(1 + np.cumsum(rng.normal(0, 0.08, (S, 8)), 1), 0, None)
    yaw = np.cumsum(rng.normal(0, 0.03, (S, 8)), 1)
    x, y = np.cumsum(0.5 * v * np.cos(yaw), 1), np.cumsum(0.5 * v * np.sin(yaw), 1)
    return np.stack([x, y, yaw], -1).astype(np.float32)


def synth_pack(rng, S, NR=131):
    """A synthetic pack in which every item is active on many rows (frames: the route passes through the row's origin)."""
    R, H = rand_routes(rng, S, NR)
    K = 32
    box = np.zeros((S, 9, K, 5), np.float32)
    valid = np.zeros((S, 9, K), bool)
    cls = np.full((S, K), -1, np.int64)
    for j in range(3):                                                          # leads on the route at 8 .. 60 m, slower than the plan
        a0 = rng.uniform(8, 60, S)
        va = rng.uniform(0, 6, S)
        for t in range(9):
            a = np.clip(a0 + 0.5 * t * va, 0, NR - 2)
            i = a.astype(int)
            box[:, t, j, :2] = R[np.arange(S), i] + (a - i)[:, None] * (R[np.arange(S), i + 1] - R[np.arange(S), i])
            box[:, t, j, 2] = H[np.arange(S), i]
        box[:, :, j, 3], box[:, :, j, 4] = 4.6, 1.9
        valid[:, :, j], cls[:, j] = True, 0
    ln_type = np.tile(np.array([2, 2, 1, 0], np.int8), (S, 1))
    ln_d = np.stack([rng.uniform(5, 40, S), rng.uniform(40, 90, S), rng.uniform(5, 60, S), np.full(S, np.nan)], 1).astype(np.float32)
    ln_elig = ln_type > 0
    ln_red = rng.random((S, 4, 1)) < 0.7
    ln_red = np.broadcast_to(ln_red, (S, 4, OL.NS)).copy()
    tn_ok = rng.random(S) < 0.8
    tn_spk = rng.uniform(5, 50, S).astype(np.float32)
    vlog = np.broadcast_to(rng.uniform(1, 8, (S, 1)), (S, NR)).astype(np.float32).copy()
    D = np.cumsum(rng.uniform(0, 8, (S, 8)), 1).astype(np.float32)
    LC.NR = NR
    Z = OL.build_rows(R, H, np.full(S, NR - 1.0, np.float32), box, valid, cls, ln_type, ln_d, ln_elig, ln_red, tn_ok, tn_spk, vlog[:, 0], vlog, D)
    Z["p_open"] = rng.random(S) < 0.5                                           # exercise P regardless of the context
    return Z


def cmd_grad(a, run):
    rng = np.random.default_rng(0)
    S = 4000
    Z = synth_pack(rng, S)
    pack = OL.to_pack(Z)
    pack = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in pack.items()}
    q = torch.tensor(rand_plans(rng, S), dtype=torch.float64, requires_grad=True)
    rows = torch.arange(S)
    x, y, psi = q.detach().unbind(-1)
    xt, yt = OL.timing_only(q[..., 0], q[..., 1], q[..., 2])
    out = dict(rows=S, values_bit_for_bit=bool(torch.equal(xt.detach(), x) and torch.equal(yt.detach(), y)),
               values_bit_for_bit_f32=bool(all(torch.equal(u, w) for u, w in zip(OL.timing_only(x.float(), y.float(), psi.float()), (x.float(), y.float())))))
    ux, uy = torch.cos(psi), torch.sin(psi)
    T = OL.terms(q, rows, pack)
    # the unprojected gradient: the same operators without timing_only
    saved = OL.timing_only
    OL.timing_only = lambda x_, y_, p_: (x_, y_)
    q2 = q.detach().clone().requires_grad_(True)
    T2 = OL.terms(q2, rows, pack)
    items = {}
    for it in OL.ITEMS:
        g, = torch.autograd.grad(T[f"lon{it}"].sum(), q, retain_graph=True)
        g2, = torch.autograd.grad(T2[f"lon{it}"].sum(), q2, retain_graph=True)
        along, cross = g[..., 0] * ux + g[..., 1] * uy, -g[..., 0] * uy + g[..., 1] * ux
        along2 = g2[..., 0] * ux + g2[..., 1] * uy
        cross2 = -g2[..., 0] * uy + g2[..., 1] * ux
        # central finite difference of the unprojected loss along the heading of one pose per active row
        act = (T[f"lon{it}"] > 0).nonzero()[:, 0][:300]
        fd_err, fd_in = 0.0, 1.0
        if len(act):
            kk = torch.as_tensor(rng.integers(0, 8, len(act)))
            e = 1e-5
            dq = torch.zeros(len(act), 8, 3, dtype=torch.float64)
            dq[torch.arange(len(act)), kk, 0], dq[torch.arange(len(act)), kk, 1] = ux[act, kk], uy[act, kk]
            with torch.no_grad():
                fp = OL.terms(q.detach()[act] + e * dq, act, pack)[f"lon{it}"]
                fm = OL.terms(q.detach()[act] - e * dq, act, pack)[f"lon{it}"]
            fd = (fp - fm) / (2 * e)
            e_ = (fd - along[act, kk]).abs()                                    # a hinge has kinks and mask edges: the share within tolerance is the gate
            fd_err, fd_in = float(e_.max()), float((e_ <= 1e-4).double().mean())
        items[it] = dict(active_rows=int((T[f"lon{it}"] > 0).sum()), value_equal_unprojected=bool(torch.equal(T[f"lon{it}"], T2[f"lon{it}"])),
                         grad_along_abs_max=float(along.abs().max()), grad_cross_abs_max=float(cross.abs().max()), grad_yaw_abs_max=float(g[..., 2].abs().max()),
                         unprojected_cross_abs_max=float(cross2.abs().max()), unprojected_yaw_abs_max=float(g2[..., 2].abs().max()),
                         along_minus_unprojected_along_abs_max=float((along - along2).abs().max()), finite_difference_along_abs_err_max=fd_err,
                         finite_difference_share_within_1e_4=fd_in,
                         finite=bool(torch.isfinite(g).all()))
    OL.timing_only = saved
    # float32 pass, as the trainer runs it
    p32 = OL.to_pack(Z)
    q32 = torch.tensor(rand_plans(np.random.default_rng(1), S), requires_grad=True)
    L = OL.loss(q32, rows, p32)
    g, = torch.autograd.grad(sum(v.sum() for v in L.values()), q32)
    c32 = (-g[..., 0] * torch.sin(q32[..., 2]) + g[..., 1] * torch.cos(q32[..., 2])).detach()
    a32 = (g[..., 0] * torch.cos(q32[..., 2]) + g[..., 1] * torch.sin(q32[..., 2])).detach()
    out |= dict(items=items, f32=dict(grad_cross_abs_max=float(c32.abs().max()), grad_along_abs_max=float(a32.abs().max()),
                                      grad_yaw_abs_max=float(g[..., 2].abs().max()), finite=bool(torch.isfinite(g).all()),
                                      keys=sorted(L), zero_plan_finite=None))
    qz = torch.zeros(8, 8, 3, requires_grad=True)                               # a standing plan: no nan from zero-length segments
    gz, = torch.autograd.grad(sum(v.sum() for v in OL.loss(qz, rows[:8], p32).values()), qz)
    out["f32"]["zero_plan_finite"] = bool(torch.isfinite(gz).all())
    out["pass"] = bool(out["values_bit_for_bit"] and out["values_bit_for_bit_f32"] and out["f32"]["finite"] and out["f32"]["zero_plan_finite"]
                       and out["f32"]["grad_cross_abs_max"] <= 1e-6 * max(out["f32"]["grad_along_abs_max"], 1) and out["f32"]["grad_yaw_abs_max"] == 0
                       and all(v["active_rows"] > 50 and v["grad_yaw_abs_max"] == 0 and v["grad_cross_abs_max"] <= 1e-12 * max(v["grad_along_abs_max"], 1)
                               and v["along_minus_unprojected_along_abs_max"] <= 1e-9 and v["finite_difference_share_within_1e_4"] >= 0.99
                               and v["value_equal_unprojected"] and v["finite"] for v in items.values()))
    return out


def cmd_kat(a, run):
    rng = np.random.default_rng(0)
    out, chain = {}, {}
    for NR in (131, 261):
        LC.NR = NR
        S = 2000
        R, H = rand_routes(rng, S, NR)
        q = rand_plans(rng, S)
        q[..., 0] += rng.uniform(-5, NR * 0.6, (S, 1))                           # anywhere along the route, also before its origin
        q[..., 1] += rng.uniform(-6, 6, (S, 1))
        d = SW.dense(q)
        dt = OL.dense(*torch.tensor(q).unbind(-1))
        fx, fy = LC.front_point(d)
        ft = OL.front_point(dt)
        sn, ln = LC.project(R, np.stack([fx, fy], -1))
        st, lt = OL.project(torch.tensor(R), ft)
        srn, lrn = LC.project(R, d[..., :2])
        srt, lrt = OL.project(torch.tensor(R), dt[..., :2])
        pts = rng.uniform(-10, NR * 0.7, (S, 300, 2)).astype(np.float32) * np.array([1, 0.2], np.float32)
        spn, lpn = LC.project(R, pts)
        spt, lpt = OL.project(torch.tensor(R), torch.tensor(pts))
        mx = lambda u, w: float(np.abs(np.asarray(u) - w.numpy()).max())  # noqa: E731
        # operators on IDENTICAL inputs (the numpy dense points fed to the torch projection)
        st_i, lt_i = OL.project(torch.tensor(R), torch.tensor(np.stack([fx, fy], -1)))
        srt_i, lrt_i = OL.project(torch.tensor(R), torch.tensor(d[..., :2]))
        sig = rng.uniform(0, NR, (S, 1)).astype(np.float32)                     # the gap operator of A: an object's rear arc minus the front arc
        out[f"NR{NR}"] = dict(dense=mx(d, dt), front_point=max(mx(fx, ft[..., 0]), mx(fy, ft[..., 1])), project_front_arc=mx(sn, st_i), project_front_lat=mx(ln, lt_i),
                              project_rear_arc=mx(srn, srt_i), project_rear_lat=mx(lrn, lrt_i), project_random_arc=mx(spn, spt), project_random_lat=mx(lpn, lpt),
                              gap=mx(sig - sn, torch.tensor(sig) - st_i), path_speed=mx(LC.path_speed(d), OL.path_speed(dt)),
                              plan_arc_vs_sweep_arc=mx(SW.arc(d)[:, 5::5], OL.plan_arc(torch.tensor(q[..., 0]), torch.tensor(q[..., 1]))),
                              points=int(sn.size + srn.size + spn.size))
        # end to end (torch dense -> torch projection against numpy dense -> numpy projection): inputs differ by float rounding (1e-5 m), and the
        # projection is discontinuous where a point is equidistant from two route segments, so a few points jump; reported, not gated
        e = np.abs(sn - st.numpy())
        chain[f"NR{NR}"] = dict(front_arc_max=float(e.max()), front_arc_share_above_1e_3=float((e > 1e-3).mean()), front_lat_max=mx(ln, lt),
                                rear_arc_max=mx(srn, srt), rear_lat_max=mx(lrn, lrt),
                                front_arc_max_within_3p5m_of_route=float(e[np.abs(ln) <= 3.5].max()))
    worst = max(v for o in out.values() for k, v in o.items() if k != "points")
    out |= dict(end_to_end=chain, max_abs_diff=worst, tol=2e-3, note="float32 on both sides; arcs reach 260 m, so 1e-4 m is rounding", **{"pass": bool(worst <= 2e-3)})
    return out


def to_state(xy, off):
    """Logged-frame points (S, ..., 2) -> the state's own frame (inverse of sweep.to_log); off (S, 2) = (dy, dpsi)."""
    sh = (len(off),) + (1,) * (xy.ndim - 2)
    dy, c, s = off[:, 0].reshape(sh), np.cos(off[:, 1]).reshape(sh), np.sin(off[:, 1]).reshape(sh)
    return np.stack([c * xy[..., 0] + s * (xy[..., 1] - dy), -s * xy[..., 0] + c * (xy[..., 1] - dy)], -1).astype(np.float32)


def nav_chunk(u):
    """(family, shard, lo, hi) -> per (state, own query j in 0, 1): LONG-PROBE labels and the three hinges, computed in the STATE's own frame."""
    f, k, lo, hi = u
    LC.NR = 131
    torch.set_num_threads(1)
    z = np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz")
    y = np.load(LC.root() / "rows" / f"{B.cdir(f, k)}.npz")
    X = np.load(LC.ctx_path())
    L = B.labels()
    sl = slice(lo, hi)
    gi, off = z["gi"][sl], z["off"][sl].astype(np.float32)
    box, valid, cls = np.asarray(L["box"][gi]).astype(np.float32), np.asarray(L["valid"][gi]), np.asarray(L["cls"][gi])
    g = lambda key: X[key][gi]  # noqa: E731
    R, H = g("route").astype(np.float32), g("rhead").astype(np.float32)
    S = len(gi)
    # re-express route and boxes in the state's own frame (the path the pool rows take: the row frame is not the route's origin frame)
    Rs, Hs = to_state(R, off), H - off[:, 1:2]
    bs = box.copy()
    bs[..., :2] = to_state(box[..., :2], off)
    bs[..., 2] -= off[:, 1, None, None]
    kf = np.clip(np.round(np.arange(OL.NS) * OL.DT / 0.5).astype(int), 0, LC.NK - 1)
    Z = OL.build_rows(Rs, Hs, g("rlen"), bs, valid, cls, g("ln_type"), g("ln_d"), g("ln_elig"), g("ln_st")[:, :, kf] == 1, g("tn_ok"), g("tn_spk"),
                      g("tn_vpk"), g("vlog").astype(np.float32), np.full((S, 8), np.nan, np.float32), ok=g("ok"))
    pack = OL.to_pack(Z)
    rows = torch.arange(S)
    out = dict(fam=np.full((S, 2), f), a_lab=y["a_lab"][sl][:, :2], b_lab=y["b_lab"][sl][:, :2], b_kind=y["b_kind"][sl][:, :2], c_lab=y["c_lab"][sl][:, :2],
               b_passed=np.repeat((Z["b_d"] < Z["sf0"][:, None]).any(1)[:, None], 2, 1))
    for j in (0, 1):
        with torch.no_grad():
            q = torch.tensor(z["q"][sl][:, j].astype(np.float32))
            T0 = OL.terms(q, rows, pack, margin_a=0.0, theta_b=0.0)
            T1 = OL.terms(q, rows, pack)
        for it in "ABC":
            out.setdefault(f"h{it}0", []).append(T0[f"lon{it}"].numpy())
            out.setdefault(f"h{it}", []).append(T1[f"lon{it}"].numpy())
    return {k_: (np.stack(v, 1) if isinstance(v, list) else v) for k_, v in out.items()}


def cmd_navtrain(a, run):
    from jevdrive import par
    from jevdrive.data import splits
    run.use_split(splits.load("navsim/navtrain"))
    units = []
    for f in a.fams:
        for k in a.shards:
            n = len(np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz")["gi"])
            n = min(n, a.limit) if a.limit else n
            units += [(f, k, c, min(c + 256, n)) for c in range(0, n, 256)]
    res = par.pmap(nav_chunk, units, run=run, desc="navtrain chunks")
    res.raise_if_failed()
    D = {k: np.concatenate([r[k] for r in res.values]) for k in res.values[0]}
    out = dict(states=int(len(D["a_lab"])), rows=int(D["a_lab"].size), fams=a.fams, shards=a.shards, queries="0, 1 (own plans of P2H10-F s0 / s1)")

    def agree(pos, h, name, extra=None):
        n = int(pos.sum())
        r = dict(label_positive=n, hinge_positive=int((h[pos] > 0).sum()), share=float((h[pos] > 0).mean()) if n else None)
        r["by_family"] = {f: dict(n=int((pos & (D["fam"] == f)).sum()), share=float((h[pos & (D["fam"] == f)] > 0).mean()) if (pos & (D["fam"] == f)).any() else None)
                          for f in a.fams}
        out[name] = r | (extra or {})

    agree(D["a_lab"] == 1, D["hA0"], "A_margin0")
    red = (D["b_lab"] == 1) & (D["b_kind"] == 2)
    agree(red, D["hB0"], "B_red_light_at_line", dict(missed_with_a_line_already_passed_at_t0=int((red & ~(D["hB0"] > 0) & D["b_passed"]).sum())))
    agree(D["c_lab"] == 1, D["hC"], "C_as_implemented")
    for it, lab in (("A", "a_lab"), ("B", "b_lab"), ("C", "c_lab")):                # descriptive: the trained hinge against the label classes
        h = D[f"h{it}"]
        out[f"{it}_trained_hinge"] = dict(active_share_all=float((h > 0).mean()), active_share_label_pos=float((h[D[lab] == 1] > 0).mean()),
                                          active_share_label_neg=float((h[D[lab] == 0] > 0).mean()), mean_when_active=float(h[h > 0].mean()) if (h > 0).any() else None,
                                          p99_when_active=float(np.quantile(h[h > 0], 0.99)) if (h > 0).any() else None)
    out["pass"] = bool(all(out[k]["share"] is not None and out[k]["share"] >= 0.95 for k in ("A_margin0", "B_red_light_at_line", "C_as_implemented")))
    return out


if __name__ == "__main__":
    from jevdrive.run import Run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grad", "kat", "navtrain"])
    ap.add_argument("--fams", nargs="+", default=["log", "ot1", "yr1"])
    ap.add_argument("--shards", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with Run("onpol_nuplan", f"lon-gate-{a.cmd}", config=vars(a)) as run:
        out = {"grad": cmd_grad, "kat": cmd_kat, "navtrain": cmd_navtrain}[a.cmd](a, run)
        p = gdir() / f"lon_{a.cmd}.json"
        p.write_text(json.dumps(out, indent=1))
        run.info("%s -> %s", json.dumps(out), p)
        run.summary.update({"pass": out["pass"], "out": str(p)})
