"""BODY1 stage 2, P0-D1 / P0-D2 (plans/2026-10-10-stage2-prereg.md): the frozen student and the frozen S0 contact head read on CARLA rows.
Nothing is trained or selected here; the b2d_v2 cache is a test set only.

  forward  GPU: P2H10-F-s0 / -s1 plans on the 2 Hz subsample of cache b2d_v2 (tick % 10 == 0), 8 queries per row (own0, own1, lateral ramps of
           own0 at +-0.5 / 1.0 / 1.5 m), S0 head seeds 0 / 1 and the blind head on them           -> $DATA_DIR/runs/body1/s2p0/pred.npz
  labels   CPU, one unit per clip: agent boxes (n, 9, 32, 5) from the collector's actors.npz at matching ticks, sweep labels of the 8 queries
           against them and against the clip's drivable SDF, for the Pacifica and the MKZ footprint -> s2p0/labels.npz
  report   D1 (own-plan likeness against navtrain hold logs, same reader) and D2 (AUC, cluster bootstrap by route / town)
                                                                                                  -> experiments/body1/results/s2p0/*.json, *.csv

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/s2p0_read.py forward|labels|report [--limit-clips N]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402
import contact_head as CH  # noqa: E402
import sweep as SW  # noqa: E402

CACHE = "b2d_v2"
MODELS = ("P2H10-F-s0", "P2H10-F-s1")
HEADS = {"s0": "full-step/20261010-024906", "s1": "full-step/20261010-024909", "blind": "full-blind/20261010-024909"}
RAMPS = (0.5, -0.5, 1.0, -1.0, 1.5, -1.5)
QN = ("own0", "own1") + tuple(f"lat{a:+.1f}" for a in RAMPS)
K, HZ = 32, 10                                    # objects per row; ticks per 0.5 s (the collector logs at 20 Hz)
MKZ = dict(front=3.8286, rear=-1.0638, half_w=0.9184)
PAC = dict(front=SW.FRONT, rear=SW.REAR, half_w=SW.HALF_W)
CLS = {"vehicle": 0, "generic_object": 1, "pedestrian": 2, "bicycle": 3}
BIKES = ("crossbike", "century", "omafiets")
RES = B.REPO / "experiments/body1/results/s2p0"
NB = 2000


def out():
    d = B.root() / "s2p0"
    d.mkdir(parents=True, exist_ok=True)
    return d


def subsample(limit_clips=0):
    """-> (row indices into the cache, tab dict of those rows, extra dict of those rows)."""
    c = B.cache_root() / CACHE
    ex, tb = np.load(c / "extra.npz"), np.load(c / "tab.npz")
    m = ex["tick"] % 10 == 0
    if limit_clips:
        m &= np.isin(ex["route"], np.unique(ex["route"])[:: max(1, len(np.unique(ex["route"])) // limit_clips)][:limit_clips])
    r = np.flatnonzero(m)
    return r, {k: tb[k][r] for k in ("names", "ego", "fut", "speed", "cam")}, {k: ex[k][r] for k in ("route", "tick", "town", "type")}


@contextlib.contextmanager
def footprint(front, rear, half_w):
    """sweep.py with another ego box about the rear axle (its constants are module globals; `clearance` binds two of them as defaults)."""
    old = (SW.C_OFF, SW.HALF_L, SW.HALF_W, SW.clearance.__defaults__)
    SW.C_OFF, SW.HALF_L, SW.HALF_W = (front + rear) / 2, (front - rear) / 2, half_w
    SW.clearance.__defaults__ = (SW.HALF_L, SW.HALF_W)
    try:
        yield
    finally:
        SW.C_OFF, SW.HALF_L, SW.HALF_W, SW.clearance.__defaults__ = old


def ramps(own):
    """own (n, 8, 3) -> (n, 6, 8, 3): rigid lateral ramps, offset a * t / 4 s along the pose normal, heading unchanged."""
    f = (np.arange(1, 9) / 8.0)[None, None, :]
    a = np.array(RAMPS, np.float32)[None, :, None] * f
    o = own[:, None]
    return np.stack([o[..., 0] - a * np.sin(o[..., 2]), o[..., 1] + a * np.cos(o[..., 2]), o[..., 2] + 0 * a], -1).astype(np.float32)


# ---------------------------------------------------------------- forward
def forward(a, run):
    import torch
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    dev = torch.device("cuda")
    rows, tb, _ = subsample(a.limit_clips)
    n = len(rows)
    mm = T.IndexedMM(B.cache_root() / CACHE)
    pi = torch.as_tensor(np.load(B.cache_root() / CACHE / "teacher.npz")["pi"], device=dev)
    W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
    models = [T.load_pmodel(t, dev) for t in MODELS]
    heads = {k: CH.load_ckpt(CH.sroot() / v / "ckpt.pt", dev) for k, v in HEADS.items()}
    ego, cam = torch.as_tensor(tb["ego"], device=dev), torch.as_tensor(tb["cam"][:, 0].astype(np.float32), device=dev)
    own = np.zeros((2, n, 8, 3), np.float32)
    q = np.zeros((n, len(QN), 8, 3), np.float32)
    P = {k: np.zeros((n, len(QN), len(CH.OUT)), np.float32) for k in heads}
    for i in run.tqdm(range(0, n, a.batch), desc="forward"):
        j = min(i + a.batch, n)
        V = torch.from_numpy(np.ascontiguousarray(mm[rows[i:j]])).to(dev)
        tc = torch.tensor([[1.0, 0.0]], device=dev).expand(j - i, 2)
        with torch.no_grad():
            for m, model in enumerate(models):
                for c in range(0, j - i, 128):
                    s = slice(c, min(c + 128, j - i))
                    p = model(V[s], ego[i:j][s], tc[s], None, None).float()[:, pi].view(-1, 33, 15)
                    own[m, i:j][s] = torch.stack(T.rear(p, cam[i:j][s], W), -1).cpu().numpy()
        q[i:j] = np.concatenate([own[:, i:j].transpose(1, 0, 2, 3), ramps(own[0, i:j])], 1)
        qt = torch.as_tensor(q[i:j], device=dev)
        for k, (net, ck) in heads.items():
            E = torch.as_tensor(((tb["ego"][i:j] - ck["emu"]) / ck["esd"]).astype(np.float32), device=dev)
            P[k][i:j] = CH.predict(net, None if k == "blind" else V, E, qt)[0]
    np.savez(out() / ("pred.npz" if not a.limit_clips else f"pred-first{a.limit_clips}.npz"), rows=rows, names=tb["names"], own=own, q=q, qn=np.array(QN),
             out_names=np.array(CH.OUT), **{f"p_{k}": v for k, v in P.items()})
    arc = np.hypot(*np.diff(np.concatenate([np.zeros((n, 1, 2), np.float32), own[0][..., :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)
    run.info(f"{n} rows; own0 4 s arc mean {arc.mean():.2f} m; logged {np.hypot(*np.diff(np.concatenate([np.zeros((n, 1, 2)), tb['fut'][..., :2]], 1), axis=1).transpose(2, 0, 1)).sum(1).mean():.2f} m")
    run.summary.update(rows=n, vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


# ---------------------------------------------------------------- labels (one clip)
def clip_dir(route):
    rd = B.data_dir() / "runs/b2d_collect/data/all/attempts" / str(route)
    ds = sorted((d for d in rd.iterdir() if (d / "clip/DONE").exists() and (d / "clip/labels.npz").exists()), key=lambda d: int(d.name))
    return ds[-1] / "clip"


def clip_boxes(cd, ticks):
    """Agent boxes of the rows at `ticks` -> box (n, 9, K, 5) x, y, yaw, L, W in each row's rear-axle frame, valid (n, 9, K), cls (n, K), dropped kinds."""
    L, A, kinds = np.load(cd / "labels.npz"), np.load(cd / "actors.npz"), json.loads((cd / "kinds.json").read_text())
    xyw, hd, nt = L["xy_world"], L["heading"], len(L["heading"])
    ids = np.unique(A["id"])
    cl, ext, off, static, drop = np.full(len(ids), -1, np.int8), np.zeros((len(ids), 2), np.float32), np.zeros((len(ids), 2), np.float32), np.zeros(len(ids), bool), {}
    for i, u in enumerate(ids):
        kd = kinds[str(u)]
        t = kd["type_id"]
        e = kd["extent"][:2]
        c = (CLS["bicycle"] if any(b in t for b in BIKES) else CLS["vehicle"]) if t.startswith("vehicle.") else CLS["pedestrian"] if t.startswith("walker.") \
            else CLS["generic_object"] if t.startswith("static.prop.") else -1
        if c < 0 or not (0.05 <= e[0] <= 10 and 0.05 <= e[1] <= 10):
            drop[t] = drop.get(t, 0) + 1
            continue
        cl[i], ext[i], off[i], static[i] = c, e, kd["offset"][:2], t.startswith("static.")
    ii = np.searchsorted(ids, A["id"])
    yaw = np.radians(A["yaw"].astype(float))
    cx = A["xyz"][:, 0] + np.cos(yaw) * off[ii, 0] - np.sin(yaw) * off[ii, 1]
    cy = A["xyz"][:, 1] + np.sin(yaw) * off[ii, 0] + np.cos(yaw) * off[ii, 1]
    pos = np.full((len(ids), nt, 3), np.nan, np.float32)                      # right-handed world x, y, heading per (actor, tick)
    ok = A["tick"] < nt
    pos[ii[ok], A["tick"][ok]] = np.stack([cx, -cy, -yaw], -1)[ok]
    for i in np.flatnonzero(static):                                            # static props are logged every prop_every ticks: one pose for the clip
        pos[i] = pos[i][np.isfinite(pos[i, :, 0])][0]
    pos[cl < 0] = np.nan
    n = len(ticks)
    box, valid, cls = np.zeros((n, 9, K, 5), np.float32), np.zeros((n, 9, K), bool), np.full((n, K), -1, np.int8)
    for r, t0 in enumerate(ticks):
        tt = t0 + HZ * np.arange(9)
        p = pos[:, np.minimum(tt, nt - 1)]                                      # (ids, 9, 3)
        p[:, tt >= nt] = np.nan
        d = np.hypot(p[:, 0, 0] - xyw[t0, 0], p[:, 0, 1] - xyw[t0, 1])
        k = np.argsort(np.where(np.isfinite(d), d, np.inf))[:K]
        k = k[np.isfinite(d[k])]
        c, s = np.cos(hd[t0]), np.sin(hd[t0])
        dx, dy = p[k, :, 0] - xyw[t0, 0], p[k, :, 1] - xyw[t0, 1]
        m = len(k)
        box[r, :, :m, 0], box[r, :, :m, 1], box[r, :, :m, 2] = (c * dx + s * dy).T, (-s * dx + c * dy).T, (p[k, :, 2] - hd[t0]).T
        box[r, :, :m, 3], box[r, :, :m, 4] = 2 * ext[k, 0], 2 * ext[k, 1]
        valid[r, :, :m] = np.isfinite(p[k, :, 0]).T
        cls[r, :m] = cl[k]
    box[~valid] = 0
    return box, valid, cls, drop


def clip_labels(job):
    route, ticks, q, fut = job
    cd = clip_dir(route)
    L = np.load(cd / "labels.npz")
    assert np.abs(L["fut"][ticks] - fut).max() < 1e-3, f"{route}: the cache rows are not this attempt's ticks"
    S = np.load(cd / "sdf.npz")
    assert float(S["x0"]) == SW.X0 and float(S["y0"]) == SW.Y0 and float(S["res"]) == SW.RES
    si = np.searchsorted(S["ticks"], ticks)
    assert (S["ticks"][si] == ticks).all()
    sdf = S["sdf"][si].astype(np.float32)
    box, valid, cls, drop = clip_boxes(cd, ticks)
    o = dict(box=box.astype(np.float16), valid=valid, cls=cls)
    for fp, kw in (("pac", PAC), ("mkz", MKZ)):
        with footprint(**kw):
            for c in range(0, len(ticks), 128):
                s = slice(c, c + 128)
                for k, v in SW.labels(q[s], np.zeros((len(ticks[s]), 2), np.float32), box[s], valid[s], cls[s], sdf[s]).items():
                    if k in ("a_hit", "a_rear", "a_clr", "a_t", "a_cls", "b_margin", "b_t0", "b_t", "b_cov"):
                        o.setdefault(f"{fp}_{k}", []).append(v)
    o = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in o.items()}
    w = json.loads((cd / "meta.json").read_text())["weather"]
    return o, drop, [w["sun_altitude_angle"], w["precipitation"], w["wetness"], w["fog_density"]]


def labels(a, run):
    from jevdrive import par
    rows, tb, ex = subsample(a.limit_clips)
    z = np.load(out() / ("pred.npz" if not a.limit_clips else f"pred-first{a.limit_clips}.npz"))
    assert (z["rows"] == rows).all()
    order = np.argsort(ex["route"], kind="stable")
    routes, start = np.unique(ex["route"][order], return_index=True)
    grp = np.split(order, start[1:])
    res = par.pmap(clip_labels, [(r, ex["tick"][g], z["q"][g], tb["fut"][g]) for r, g in zip(routes, grp)], run=run, desc="clips")
    res.raise_if_failed()
    keys = list(res.values[0][0])
    O = {}
    for k in keys:
        x = np.concatenate([v[0][k] for v in res.values])
        O[k] = np.empty_like(x)
        O[k][np.concatenate(grp)] = x
    drop = {}
    for v in res.values:
        for k, c in v[1].items():
            drop[k] = drop.get(k, 0) + c
    wx = np.zeros((len(rows), 4), np.float32)
    for g, v in zip(grp, res.values):
        wx[g] = v[2]
    np.savez_compressed(out() / ("labels.npz" if not a.limit_clips else f"labels-first{a.limit_clips}.npz"), rows=rows, weather=wx, dropped=json.dumps(drop), **O)
    run.info(f"{len(rows)} rows of {len(routes)} clips; dropped kinds {drop}; rows with >= 1 valid object {float((O['cls'] >= 0).any(1).mean()):.3f}; "
             f"own0 agent hit pac / mkz {O['pac_a_hit'][:, 0].mean():.4f} / {O['mkz_a_hit'][:, 0].mean():.4f}; "
             f"own0 boundary margin < -0.2 pac / mkz {(O['pac_b_margin'][:, 0] < -0.2).mean():.4f} / {(O['mkz_b_margin'][:, 0] < -0.2).mean():.4f}")
    run.summary.update(rows=len(rows), clips=len(routes))


# ---------------------------------------------------------------- statistics
def draws(groups, n_boot=NB, seed=0):
    import pandas as pd
    codes, uniq = pd.factorize(np.asarray(groups))
    G = len(uniq)
    idx = np.random.default_rng(seed).integers(G, size=(n_boot, G))
    c = np.zeros((n_boot, G))
    np.add.at(c, (np.arange(n_boot)[:, None], idx), 1.0)
    return codes, c


def ci(b):
    b = b[np.isfinite(b)]
    return [float(x) for x in np.quantile(b, [0.025, 0.975])] if len(b) else [np.nan, np.nan]


def share(num, den, groups):
    """Cluster-bootstrapped ratio of sums -> dict(v, lo, hi, n, k)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    codes, c = draws(groups)
    G = c.shape[1]
    a, b = np.bincount(codes, num, G), np.bincount(codes, den, G)
    lo, hi = ci((c @ a) / np.maximum(c @ b, 1e-9))
    return dict(v=float(num.sum() / max(den.sum(), 1e-9)), lo=lo, hi=hi, n=int(den.sum()), k=int(num.sum()), clusters=int(G))


def median(x, groups, n_boot=500):
    """Cluster-bootstrapped median (weighted by the cluster draw counts)."""
    x = np.asarray(x, float)
    if not len(x):
        return dict(v=np.nan, lo=np.nan, hi=np.nan, n=0)
    codes, c = draws(groups, n_boot)
    o = np.argsort(x)
    w = np.cumsum(c[:, codes[o]], 1)
    b = x[o][np.minimum((w >= w[:, -1:] / 2).argmax(1), len(x) - 1)]
    lo, hi = ci(b)
    return dict(v=float(np.median(x)), lo=lo, hi=hi, n=len(x), clusters=int(c.shape[1]))


def auc_wins(y, s, codes, G):
    sp, gp, sn, gn = s[y], codes[y], s[~y], codes[~y]
    W = np.zeros((len(sp), G))
    for h in np.unique(gn):
        x = np.sort(sn[gn == h])
        lo, hi = np.searchsorted(x, sp, "left"), np.searchsorted(x, sp, "right")
        W[:, h] = lo + 0.5 * (hi - lo)
    U = np.zeros((G, G))
    np.add.at(U, gp, W)
    return U


def auc_pair(y, s1, s2, groups):
    """Paired cluster bootstrap of AUC(s1) - AUC(s2) (the draws of contact_head.auc_boot) -> dict(d, lo, hi)."""
    y = np.asarray(y, bool)
    if not y.any() or y.all():
        return dict(d=np.nan, lo=np.nan, hi=np.nan)
    codes, c = draws(groups)
    G = c.shape[1]
    n1, n0 = np.bincount(codes[y], minlength=G).astype(float), np.bincount(codes[~y], minlength=G).astype(float)
    den = (c @ n1) * (c @ n0)
    U = auc_wins(y, s1, codes, G) - auc_wins(y, s2, codes, G)
    lo, hi = ci(np.einsum("bg,gh,bh->b", c, U, c)[den > 0] / den[den > 0])
    return dict(d=float(U.sum() / (n1.sum() * n0.sum())), lo=lo, hi=hi)


def arcs(p):
    """Poses (..., 8, 3) -> 4 s arc length, displacement in the first 1 s, 4 s heading change (deg)."""
    xy = np.concatenate([np.zeros(p.shape[:-2] + (1, 2), np.float32), p[..., :2]], -2)
    return np.hypot(np.diff(xy[..., 0], axis=-1), np.diff(xy[..., 1], axis=-1)).sum(-1), np.hypot(p[..., 1, 0], p[..., 1, 1]), np.degrees(p[..., -1, 2])


def d1_stats(own, fut, speed, grp, b_margin, b_t0, a_hit, masks=None):
    """own (2, n, 8, 3), fut (n, 8, 3); b_margin / b_t0 / a_hit (n, 2) of the own plans -> the four pre-registered statistics (+ extra subsets)."""
    oa, o1, oh = arcs(own)                                                   # (2, n)
    ea, _, eh = arcs(fut)
    g2 = np.tile(grp, 2)
    two = lambda m: np.tile(m, 2)  # noqa: E731
    mov, stand = speed > 3, speed < 0.5
    R = {}
    m = mov & (ea >= 4)
    ratio = (oa / np.maximum(ea, 1e-6)).reshape(-1)
    R["a_arc_ratio_median"] = median(ratio[two(m)], g2[two(m)])
    R["a_collapse_share"] = share((ratio < 0.5) & two(m), two(m), g2)
    m = stand & (ea >= 8)
    R["b_launch_share"] = share((ratio >= 0.5) & two(m), two(m), g2)
    r1 = (o1 / np.maximum(speed, 1e-6)).reshape(-1)
    R["b_1s_ratio_median"] = median(r1[two(mov)], g2[two(mov)])
    for name, m in (("turn", mov & (np.abs(eh) > 20)), ("over45", mov & (np.abs(eh) > 45))):
        agree = (np.sign(oh) == np.sign(eh)[None]).reshape(-1)
        R[f"c_dir_{name}"] = share(agree & two(m), two(m), g2)
        R[f"c_head_ratio_{name}_median"] = median((oh / np.where(np.abs(eh) > 1, eh, np.nan))[:, m].reshape(-1), g2[two(m)])
    m = mov & (np.abs(eh) < 5)
    bm, bt0 = b_margin.T.reshape(-1), b_t0.T.reshape(-1)
    lab = (bm < 90) & ~bt0
    R["d_boundary_rate_straight"] = share((bm < -0.20) & lab & two(m), lab & two(m), g2)
    R["d_agent_rate_straight"] = share(a_hit.T.reshape(-1) & two(m), two(m), g2)
    R["boundary_rate_moving"] = share((bm < -0.20) & lab & two(mov), lab & two(mov), g2)
    R["agent_rate_moving"] = share(a_hit.T.reshape(-1) & two(mov), two(mov), g2)
    R["rows"] = dict(n=int(len(speed)), moving=int(mov.sum()), standing=int(stand.sum()), turn=int((mov & (np.abs(eh) > 20)).sum()), over45=int((mov & (np.abs(eh) > 45)).sum()),
                     launch=int((stand & (ea >= 8)).sum()), clusters=int(len(np.unique(grp))))
    if masks:
        for k, mk in masks.items():
            mm = mov & mk & (ea >= 4)
            R[f"by/{k}"] = dict(n=int(mm.sum()), arc_ratio=float(np.median(ratio[two(mm)])) if mm.any() else np.nan,
                                dir_turn=float(agree[two(mov & mk & (np.abs(eh) > 20))].mean()) if (mov & mk & (np.abs(eh) > 20)).any() else np.nan,
                                standing_share=float((stand & mk).sum() / max(mk.sum(), 1)))
    return R


def navtrain_ref():
    """On-log hold-log states of navtrain through the same reader -> d1_stats dict."""
    own, fut, sp, lg, bm, bt, ah = [], [], [], [], [], [], []
    for k in range(B.NSH):
        z, t = np.load(B.root() / "rows" / f"{B.cdir('log', k)}.npz"), B.tab("log", k)
        m = z["hold"]
        own.append(z["q"][m][:, :2]), fut.append(t["fut"][m]), sp.append(t["speed"][m] if "speed" in t else t["ego"][m, 4] * 10), lg.append(z["log"][m])
        bm.append(z["b_margin"][m][:, :2]), bt.append(z["b_t0"][m][:, :2]), ah.append(z["a_hit"][m][:, :2])
    c = np.concatenate
    return d1_stats(c(own).transpose(1, 0, 2, 3), c(fut), c(sp), c(lg), c(bm), c(bt), c(ah))


def report(a, run):
    sfx = "" if not a.limit_clips else f"-first{a.limit_clips}"
    rows, tb, ex = subsample(a.limit_clips)
    z, Lb = np.load(out() / f"pred{sfx}.npz"), np.load(out() / f"labels{sfx}.npz")
    assert (z["rows"] == rows).all() and (Lb["rows"] == rows).all()
    RES.mkdir(parents=True, exist_ok=True)
    speed, route, town = tb["speed"], ex["route"], ex["town"]
    _, _, eh = arcs(tb["fut"])
    wx = Lb["weather"]
    # ---- D1
    typ = np.char.lower(ex["type"].astype(str))
    obst = np.array([any(s in t for s in ("obstacle", "accident", "construction", "opensdoor", "hazard", "cutin", "parking")) for t in typ])
    masks = {f"town/{t}": town == t for t in np.unique(town)} | {"scenario/obstacle": obst, "scenario/other": ~obst, "night": wx[:, 0] < 0, "day": wx[:, 0] >= 0}
    D1 = dict(carla=d1_stats(z["own"], tb["fut"], speed, route, Lb["mkz_b_margin"][:, :2], Lb["mkz_b_t0"][:, :2], Lb["mkz_a_hit"][:, :2], masks),
              carla_pacifica_rates=None, navtrain=navtrain_ref())
    cp = d1_stats(z["own"], tb["fut"], speed, route, Lb["pac_b_margin"][:, :2], Lb["pac_b_t0"][:, :2], Lb["pac_a_hit"][:, :2])
    D1["carla_pacifica_rates"] = {k: cp[k] for k in ("d_boundary_rate_straight", "d_agent_rate_straight", "boundary_rate_moving", "agent_rate_moving")}
    c, n = D1["carla"], D1["navtrain"]
    L = {"a_median>=0.80": c["a_arc_ratio_median"]["v"] >= 0.80, "a_median>=0.85xnav": c["a_arc_ratio_median"]["v"] >= 0.85 * n["a_arc_ratio_median"]["v"],
         "a_collapse<=nav+0.10": c["a_collapse_share"]["v"] <= n["a_collapse_share"]["v"] + 0.10,
         "b_launch>=0.6xnav": c["b_launch_share"]["v"] >= 0.6 * n["b_launch_share"]["v"],
         "b_1s>=0.85": c["b_1s_ratio_median"]["v"] >= 0.85, "b_1s_within_0.10": abs(c["b_1s_ratio_median"]["v"] - n["b_1s_ratio_median"]["v"]) <= 0.10,
         "c_turn>=0.80": c["c_dir_turn"]["v"] >= 0.80, "c_over45>=0.75": c["c_dir_over45"]["v"] >= 0.75,
         "d_boundary<=3xnav": c["d_boundary_rate_straight"]["v"] <= 3 * n["d_boundary_rate_straight"]["v"]}
    D1["lines"] = {k: bool(v) for k, v in L.items()}
    (RES / f"d1{sfx}.json").write_text(json.dumps(D1, indent=1))
    run.info("D1 lines " + json.dumps(D1["lines"]))
    for k in ("a_arc_ratio_median", "a_collapse_share", "b_launch_share", "b_1s_ratio_median", "c_dir_turn", "c_dir_over45", "d_boundary_rate_straight", "d_agent_rate_straight"):
        run.info(f"D1 {k}: carla {c[k]['v']:.4f} [{c[k]['lo']:.4f}, {c[k]['hi']:.4f}] n {c[k]['n']}; navtrain {n[k]['v']:.4f} [{n[k]['lo']:.4f}, {n[k]['hi']:.4f}] n {n[k]['n']}")
    # ---- D2
    S = {k: z[f"p_{k}"] for k in ("s0", "s1", "blind")}
    sc = {"S0": (S["s0"] + S["s1"]) / 2, "S0-s0": S["s0"], "S0-s1": S["s1"], "blind": S["blind"]}
    subs = {"primary (speed >= 0.5)": speed >= 0.5, "all": np.ones(len(speed), bool), "moving (> 3)": speed > 3, "standing (< 0.5)": speed < 0.5,
            "turn (> 20 deg)": (speed > 3) & (np.abs(eh) > 20), "over 45 deg": (speed > 3) & (np.abs(eh) > 45), "straight (< 5 deg)": (speed > 3) & (np.abs(eh) < 5),
            "scenario obstacle": (speed >= 0.5) & obst, "night": (speed >= 0.5) & (wx[:, 0] < 0), "day": (speed >= 0.5) & (wx[:, 0] >= 0),
            "wet": (speed >= 0.5) & (wx[:, 2] > 50), "fog": (speed >= 0.5) & (wx[:, 3] > 20)} | {f"town {t}": (speed >= 0.5) & (town == t) for t in np.unique(town)}
    rowsD2 = []
    for fp in ("pac", "mkz"):
        mg, t0, ah, rear = Lb[f"{fp}_b_margin"], Lb[f"{fp}_b_t0"], Lb[f"{fp}_a_hit"], Lb[f"{fp}_a_rear"]
        Y = dict(agent=(ah, ~(rear & ~ah)), boundary=((mg < 90) & (mg < CH.B_DEPTH) & ~t0, ((mg < 90) & (mg < CH.B_DEPTH) & ~t0) | ((mg < 90) & (mg >= 0))))
        for qs, qi in (("own", [0, 1]), ("lat", [2, 3, 4, 5, 6, 7]), ("own+lat", list(range(8)))):
            for sn, m in subs.items():
                if qs != "own" and not sn.startswith(("primary", "all", "moving", "turn", "over 45")):
                    continue
                for ct, col in (("agent", 0), ("boundary", 1)):
                    y, okm = Y[ct][0][m][:, qi].reshape(-1), Y[ct][1][m][:, qi].reshape(-1)
                    g, tw = np.repeat(route[m], len(qi))[okm], np.repeat(town[m], len(qi))[okm]
                    y = y[okm]
                    for name, s in sc.items():
                        if name != "S0" and not (sn.startswith("primary") and qs == "own"):
                            continue
                        sv = s[m][:, qi, col].reshape(-1)[okm]
                        r = CH.auc_boot(y, sv, g, n_boot=NB)
                        r |= dict(footprint=fp, queries=qs, subset=sn, contact=ct, score=name)
                        if sn.startswith("primary"):
                            rt = CH.auc_boot(y, sv, tw, n_boot=NB)
                            r |= dict(town_lo=rt["lo"], town_hi=rt["hi"])
                            if name == "S0":
                                bl = sc["blind"][m][:, qi, col].reshape(-1)[okm]
                                pr = auc_pair(y, sv, bl, g)
                                r |= dict(minus_blind=pr["d"], minus_blind_lo=pr["lo"], minus_blind_hi=pr["hi"])
                        rowsD2.append(r)
    import pandas as pd
    df = pd.DataFrame(rowsD2)
    df.to_csv(RES / f"d2{sfx}.csv", index=False)
    pm = df[(df.queries == "own") & (df.subset.str.startswith("primary")) & (df.score == "S0")]
    V = {}
    for ct, (hi_, lb, fail) in dict(boundary=(0.80, 0.75, 0.70), agent=(0.75, 0.70, 0.65)).items():
        x = pm[pm.contact == ct].sort_values("auc").iloc[0]                    # the lower of the two footprints decides
        few = bool(x.pos < 100 or x.logs_pos < 20)
        if few:
            x = df[(df.queries == "own+lat") & (df.subset.str.startswith("primary")) & (df.score == "S0") & (df.contact == ct)].sort_values("auc").iloc[0]
        ok = x.auc >= hi_ and x.lo >= lb and (ct == "agent" or few or (x.minus_blind >= 0.05 and x.minus_blind_lo > 0))
        V[ct] = dict(verdict="pass" if ok else "fail" if x.auc < fail else "weak", auc=float(x.auc), lo=float(x.lo), hi=float(x.hi), footprint=x.footprint, queries=x.queries,
                     pos=int(x.pos), routes_pos=int(x.logs_pos), n=int(x.n), minus_blind=float(x.get("minus_blind", np.nan)), minus_blind_lo=float(x.get("minus_blind_lo", np.nan)), few_positives=few)
    (RES / f"d2{sfx}.json").write_text(json.dumps(dict(verdict=V, dropped_kinds=json.loads(str(Lb["dropped"])), rows=int(len(speed)), clips=int(len(np.unique(route)))), indent=1))
    run.info("D2 " + json.dumps(V))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        run.info("\n" + df[(df.score == "S0")][["footprint", "queries", "subset", "contact", "auc", "lo", "hi", "n", "pos", "logs_pos"]].round(3).to_string())
        run.info("\n" + df[(df.subset.str.startswith("primary")) & (df.queries == "own")][["footprint", "contact", "score", "auc", "lo", "hi", "town_lo", "town_hi", "minus_blind", "minus_blind_lo"]].round(3).to_string())
    run.summary.update(d1=D1["lines"], d2={k: v["verdict"] for k, v in V.items()})


if __name__ == "__main__":
    from jevdrive.data import splits
    from jevdrive.run import Run, cli_args
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["forward", "labels", "report"])
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    cli_args(ap)
    a = ap.parse_args()
    with Run("body1", f"s2p0-{a.cmd}" + (f"-first{a.limit_clips}" if a.limit_clips else ""), config=vars(a)) as run:
        for s in ("b2d/b2dc-v2-train", "b2d/b2dc-v2-val") + (("navsim/body1-hold-logs",) if a.cmd == "report" else ()):
            run.use_split(splits.load(s))                                   # read as a test set only: nothing is fitted on any of them
        dict(forward=forward, labels=labels, report=report)[a.cmd](a, run)
