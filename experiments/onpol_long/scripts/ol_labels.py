#!/usr/bin/env python3
"""ONPOL-LONG labels K and C on an on-policy pool of onpol_nuplan's on_rows.py (plans/2026-10-12-ol1-prereg.md, "Labels"), and the pool gates.
Nothing here changes on_rows.py's files; it adds two label files to the pool dir P = $DATA_DIR/runs/op_parity/cache/on1_<name>:

  P/corr.npz   tokens (n,) = the tab's names; gpath (n, 86, 2) float32: the logged rear-axle path from s0 - 15 m to s0 + 70 m at 1 m spacing in the
               row's own frame (on_rows.py meta `gpath`, copied) -> the corridor hinge K of lib/ol_kc.py
  P/lane.npz   tokens; ok (n,) bool (a lane or lane connector within the raster); m (n, 192, 96) float16: lane-centre margin (m) at the 0.5 m cell
               centres of x in [-8, 88), y in [-24, 24) of the row's own frame: m = max over the lanes and lane connectors of the nuPlan map near
               the row of (half lane width - distance to the lane's baseline) - half ego width (1.1485 m) - 0.2 m, clipped to [-4, 2];
               half lane width = the mean of the distances from the nearest baseline point to the lane's left and right boundaries -> the
               lane-centre hinge C of lib/ol_kc.py (charged where m < 0)

  build --name N [--workers W]    CPU, envs/navsim2 (nuPlan map API); after on_rows.py geo
  gate  --name N --out DIR         envs/op-train (torch): after on_rows.py tok. Gate R1 as amended (onpol_nuplan prereg Amendment 4: (a) first 15
                                   plan points within one fp16 ulp or 0.03 m on >= 0.99 of the rows, (b) the interpreted control of r1.json, (c)
                                   wrong-seed control < 0.20), gate R2 as Amendment 4 read it (ego <= 0.3 m; actor samples within 0.5 m on >= 0.9999),
                                   and gate K0: the terms are finite and their gradient has no along-heading component (shape-only); an item is
                                   trained only if it is active on >= 0.2 % of the rows on the served plans (closed-loop.md, 10-13 row) and
                                   on <= 2 % of the rows on the log's own future (a label the logged driver violates is not a consequence
                                   label); the passing items go to P/kc_items.json, which lib/ol_kc.py reads; no item passing = fail. Writes DIR/gates.json,
                                   gates.md, bev_kc.png; exit 2 if a gate fails.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "experiments/onpol_nuplan/scripts"), str(_R / "experiments/op_probe/scripts"), str(_R / "experiments/body1/lib")]
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402

DATA = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, ROWS = DATA / "runs/op_parity/cache", DATA / "runs/onpol_nuplan/rows"
X0, Y0, RES, NH, NW = -8.0, -24.0, 0.5, 192, 96
HW_EGO, PAD, CLIP, RADIUS, STEP = 2.297 / 2, 0.2, (-4.0, 2.0), 110.0, 1.0
_C = {}


def cells():
    xs, ys = X0 + RES * (np.arange(NH) + 0.5), Y0 + RES * (np.arange(NW) + 0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.c_[X.ravel(), Y.ravel()]


def lane_field(m, pose):
    """nuPlan map m, rear-axle pose (x, y, yaw) in the map frame -> (margin (NH, NW) float32, number of lanes used)."""
    import shapely
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    objs = m.get_proximal_map_objects(Point2D(float(pose[0]), float(pose[1])), RADIUS, [L.LANE, L.LANE_CONNECTOR])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    P = _C.setdefault("cells", cells())
    best, n = np.full(len(P), -np.inf), 0
    for o in (o for v in objs.values() for o in v):
        bl = o.baseline_path.linestring
        u = np.linspace(0, bl.length, max(int(np.ceil(bl.length / STEP)), 1) + 1)
        pts = shapely.line_interpolate_point(bl, u)
        hw = 0.5 * (shapely.distance(pts, o.left_boundary.linestring) + shapely.distance(pts, o.right_boundary.linestring))
        g = shapely.get_coordinates(pts) - pose[:2]
        q = np.c_[c * g[:, 0] + s * g[:, 1], -s * g[:, 0] + c * g[:, 1]]                  # row frame
        if q[:, 0].max() < X0 - 6 or q[:, 0].min() > X0 + NH * RES + 6 or q[:, 1].max() < Y0 - 6 or q[:, 1].min() > Y0 + NW * RES + 6:
            continue
        n += 1
        A, AB = q[:-1], q[1:] - q[:-1]
        t = np.clip(((P[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-9)[None], 0, 1)     # (cells, segments)
        d2 = ((P[:, None] - A[None] - t[..., None] * AB[None]) ** 2).sum(-1)
        j = d2.argmin(1)
        tj = t[np.arange(len(P)), j]
        v = hw[j] + tj * (hw[j + 1] - hw[j]) - np.sqrt(d2[np.arange(len(P)), j])
        best = np.maximum(best, v)
    return np.clip(best - HW_EGO - PAD, *CLIP).reshape(NH, NW).astype(np.float32), n


def build_one(job):
    import on_rows as OR
    import opb_labels as OL
    log, poses = job
    m = OL._map(OR._log(log)["loc"])
    out = [lane_field(m, p) for p in poses]
    return np.stack([f for f, _ in out]).astype(np.float16), np.array([k for _, k in out])


def cmd_build(a):
    from jevdrive import par
    from jevdrive.run import Run
    pool, rows = CR / f"on1_{a.name}", ROWS / a.name
    with Run("onpol_long", f"labels-{a.name}", config=vars(a)) as run:
        M, tab = dict(np.load(rows / "meta.npz")), np.load(pool / "tab.npz")
        names = tab["names"]
        assert (M["names"] == names).all()
        np.savez(pool / "corr.npz", tokens=names, gpath=M["gpath"].astype(np.float32))
        idx = np.arange(len(names))
        jobs, where = [], []
        for lg in sorted(set(M["log"].tolist())):                                     # chunks of one log: the map and the log are loaded once per process
            r = idx[M["log"] == lg]
            for i in range(0, len(r), 64):
                jobs.append((lg, M["pose_map"][r[i:i + 64]])), where.append(r[i:i + 64])
        res = par.pmap(build_one, jobs, run=run, workers=a.workers or None, desc="lane fields")
        res.raise_if_failed()
        F, nl = np.zeros((len(names), NH, NW), np.float16), np.zeros(len(names), int)
        for r, (f, k) in zip(where, res.values):
            F[r], nl[r] = f, k
        ok = nl > 0
        np.savez(pool / "lane.npz", tokens=names, ok=ok, m=F, n_lanes=nl, x0=X0, y0=Y0, res=RES, hw_ego=HW_EGO, pad=PAD)
        run.summary.update(rows=len(names), lane_ok=int(ok.sum()), lanes_median=float(np.median(nl)), m_lt0_share=float((F[ok] < 0).mean()))
        run.info(f"{pool}: corr.npz, lane.npz; rows {len(names)}, lane ok {int(ok.sum())}, lanes per row median {np.median(nl):.0f}")


def cmd_gate(a):
    import torch
    sys_path = str(_R / "experiments/onpol_long/lib")
    _sys.path.insert(0, sys_path)
    import ol_kc
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool, rows = _pl.Path(a.pool or CR / f"on1_{a.name}"), _pl.Path(a.rows or ROWS / a.name)
    res, fail = dict(name=a.name), []
    r1, r2 = (json.loads((rows / f).read_text()) for f in ("r1.json", "r2.json"))
    ra = {t: q["mu_share_one_ulp"] for t, q in r1["tags"].items()}
    rc = {t: q["control_pose_share"] for t, q in r1["tags"].items()}
    res["R1"] = dict(a_one_ulp=ra, b_control=r1["control_single"], c_wrong_seed=rc, written_pass=r1["pass"])
    if not (all(v >= 0.99 for v in ra.values()) and r1["b"] and all(v < 0.2 for v in rc.values())):
        fail.append(f"R1 as amended: (a) {ra}, (b) {r1['b']} {r1['control_single']}, (c) {rc}")
    res["R2"] = {k: r2.get(k) for k in ("ego_max_m", "act_share_within", "act_max_m", "act_n", "rollouts", "with_sim", "frame_excluded")}
    if not (r2["ego_max_m"] <= 0.3 and r2["act_share_within"] >= 0.9999):
        fail.append(f"R2 as amended: ego {r2['ego_max_m']:.3f} m, actors within 0.5 m {r2['act_share_within']:.5f}")
    M, tab = dict(np.load(rows / "meta.npz")), np.load(pool / "tab.npz")
    pack = ol_kc.load(pool, tab["names"], "cpu", items=["kcorr", "klane"])
    P = torch.tensor(M["poses"], dtype=torch.float32, requires_grad=True)
    idx = torch.arange(len(P))
    L = ol_kc.loss(P, idx, pack)
    lf = M["logfut"][:, 1:].astype(np.float32)                                     # the log's own poses at t + 0.5 .. 4 s (NaN past the log's end)
    lok = np.isfinite(lf).all((1, 2))
    LL = ol_kc.loss(torch.from_numpy(np.nan_to_num(lf[lok])), idx[torch.from_numpy(lok)], pack)
    k = M["k"]
    res["K0"], items = {}, []
    for q, v in L.items():
        lg = float((LL[q] > 0).float().mean())
        g = torch.autograd.grad(v.sum(), P, retain_graph=True)[0]
        along = (g[..., 0] * torch.cos(P[..., 2]) + g[..., 1] * torch.sin(P[..., 2])).abs().max().item()
        act = (v > 0).numpy()
        res["K0"][q] = dict(active=float(act.mean()), active_rows=int(act.sum()), scenes=int(len(set(M["scene"][act].tolist()))), mean=float(v.mean()),
                            mean_when_active=float(v[v > 0].mean()) if act.any() else 0.0, finite=bool(torch.isfinite(v).all()), grad_along_max=along,
                            grad_cross_max=float(g[..., :2].abs().max()), by_k={int(j): float(act[k == j].mean()) for j in sorted(set(k.tolist()))})
        res["K0"][q] |= dict(log_active=lg, log_rows=int(lok.sum()))
        if not (res["K0"][q]["finite"] and along <= 1e-5 * max(1.0, res["K0"][q]["grad_cross_max"])):
            fail.append(f"K0 {q}: finite {res['K0'][q]['finite']}, along-heading gradient {along:.2e}")
        elif act.mean() >= 0.002 and lg <= 0.02:
            items.append(q)
    if not items:
        fail.append("K0: no item passes (active on >= 0.2 % of the served plans and on <= 2 % of the logged futures)")
    res["items"] = items
    (pool / "kc_items.json").write_text(json.dumps(dict(items=items, K0=res["K0"]), indent=1) + "\n")
    res["lane_ok"] = float(pack["lane_ok"].float().mean())
    res |= dict(fail=fail, **{"pass": not fail})
    (out / "gates.json").write_text(json.dumps(res, indent=1) + "\n")
    md = [f"# ONPOL-LONG pool gates, rows `{a.name}`: {'PASS' if not fail else 'FAIL'}", "",
          f"R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {ra}; interpreted control {r1['control_single']}; wrong-seed control "
          f"(8 poses within 0.03 m, < 0.20): {rc}. R2: ego at the log-replay decisions max {r2['ego_max_m']:.3f} m (<= 0.3); actor samples within 0.5 m "
          f"{r2['act_share_within']:.5f} of {r2['act_n']} (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > "
          f"{r2.get('frame_tol_m', 0.1)} m; Amendment 2), no rows: {r2.get('frame_excluded', [])}.", "", f"Lane raster present on {res['lane_ok']:.4f} of the rows.", "",
          "| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    md += [f"| {q} | {v['active_rows']} | {v['active']:.4f} | {v['scenes']} | {v['mean']:.4f} | {v['mean_when_active']:.3f} | {v['grad_along_max']:.1e} | {v['log_active']:.4f} |" for q, v in res["K0"].items()]
    md += ["", "Active share by decision index k: " + "; ".join(f"{q}: " + ", ".join(f"{j} {s:.3f}" for j, s in v["by_k"].items()) for q, v in res["K0"].items()), ""]
    md += [f"Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): {items}", "",
           f"Failures: {fail}" if fail else "All gates pass.", ""]
    (out / "gates.md").write_text("\n".join(md))
    bev(M, tab, pack, L, out / "bev_kc.png")
    print("\n".join(md))
    raise SystemExit(0 if not fail else 2)


def bev(M, tab, pack, L, fn, n=12):
    """The rows where each term is largest on the served plan, plus random rows: lane margin (grey = m >= 0, the cells where C does not charge),
    the logged path with its 1.5 m dead band, the served plan with its box centres at 1 .. 4 s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    pick = [(q, int(i)) for q in L for i in np.argsort(-L[q].detach().numpy())[:n // 3]] + [("random", int(i)) for i in rng.choice(len(M["k"]), n // 3, replace=False)]
    xs, ys = X0 + RES * (np.arange(NH) + .5), Y0 + RES * (np.arange(NW) + .5)
    fig, axs = plt.subplots((len(pick) + 3) // 4, 4, figsize=(20, 7.2 * ((len(pick) + 3) // 4)), squeeze=False)
    for A, (q, i) in zip(axs.ravel(), pick):
        m = pack["m"][i].float().numpy()
        A.contourf(ys, xs, m, levels=[0, 99], colors=["#e3e3e3"])
        A.contour(ys, xs, m, levels=[-1, 0], colors=["#ff9896", "#1f77b4"], linewidths=0.7)
        g = M["gpath"][i]
        nx, ny = np.gradient(g[:, 0]), np.gradient(g[:, 1])
        h = np.hypot(nx, ny) + 1e-9
        for sgn in (-1, 1):
            A.plot(g[:, 1] + sgn * 1.5 * nx / h, g[:, 0] - sgn * 1.5 * ny / h, color="#2ca02c", lw=0.6, ls=":")
        A.plot(g[:, 1], g[:, 0], color="#7f7f7f", lw=1, ls="--")
        p = np.r_[np.zeros((1, 3)), M["poses"][i]]
        A.plot(p[:, 1], p[:, 0], "o-", color="k", ms=3, lw=1.2)
        cx, cy = p[2:, 0] + 1.461 * np.cos(p[2:, 2]), p[2:, 1] + 1.461 * np.sin(p[2:, 2])
        A.plot(cy, cx, "s", color="#d62728", ms=3)
        A.set_xlim(24, -24), A.set_ylim(-8, 88), A.set_aspect("equal")
        A.set_title(f"[{q}] {M['names'][i]}\nk {M['k'][i]}, e0 {M['e0'][i]:+.2f} m; " + ", ".join(f"{t} {L[t][i].item():.3f}" for t in L), fontsize=7.5)
    for A in axs.ravel()[len(pick):]:
        A.axis("off")
    fig.suptitle("on-policy rows, own frame (x forward up, y left): grey = lane-centre margin >= 0 (blue edge; red = -1 m), dashed = logged path, "
                 "green dotted = its 1.5 m dead band, black = served plan, red squares = box centres at 1 .. 4 s", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(fn, dpi=70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--name", required=True), p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("gate")
    p.add_argument("--name", required=True), p.add_argument("--out", required=True)
    p.add_argument("--pool", default="", help="pool dir (test; default the cache dir of --name)"), p.add_argument("--rows", default="", help="rows dir (test)")
    a = ap.parse_args()
    {"build": cmd_build, "gate": cmd_gate}[a.cmd](a)
