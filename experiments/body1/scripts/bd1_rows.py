"""BODY1 row generator (plans/2026-10-10-body1-prereg.md, 2.2): (state, query trajectory, labels) rows on navtrain. CPU.

A state = a row of a (state family, shard) cache dir (b1.py). Per state Q = 24 queries of 8 poses at 0.5 s in the state's own frame:
  0 own0, 1 own1   the student's plan (P2H10-F-s0 / -s1, bd1_plans.py)          2 ship   the shipped plan        3 log   the logged future
  4-7 lat    rigid lateral ramp: offset a * s(t) / s(4 s) along the path normal, |a| ~ U(0.3, 2.5) m, random sign, capped at 0.25 x the
             4 s arc (linear in arc length, so a stopping ego does not slide sideways; equal to linear in time at constant speed)
  8-11 head  heading drift: the plan's motion re-integrated with a heading error growing linearly to +-U(1, 6) deg at 4 s
  12-15 gain turn gain: every segment direction and yaw x U(0.5, 1.5)
  16-19 arc  arc-length scale: the pose at time t is the pose at c x s(t) along the path, c ~ U(0.5, 1.5) (straight continuation past the end)
  20-23 stop the plan followed to U(0.05, 0.95) x its 4 s arc and held there
Perturbations 2 per family on own0 and 2 on own1 (`qbase`); parameters in `par`. Labels: lib/sweep.py (`a_*` agents, `b_*` boundary).

  gen      [--fams log ot1 yr1] [--shards 0] [--limit 200]   -> $DATA_DIR/runs/body1/rows/<cache dir>[-first<N>].npz
  check    sweep.py against swv1_lib.Rollout.sweep (shapely), AgentHinge.distances and drivable_hinge.Hinge.margins -> check.json
  summary  composition, contact rates, first-contact times, balance weights, hold positives -> $DATA_DIR/runs/body1/rows/summary/*.csv, balance.json
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402
import sweep as SW  # noqa: E402

QFAM = ("own0", "own1", "ship", "log", "lat", "head", "gain", "arc", "stop")
NP_ = 4                                                     # perturbed queries per family and state
QF = np.r_[0, 1, 2, 3, np.repeat(np.arange(4, 9), NP_)].astype(np.int8)          # (Q,) query family of every slot
QBASE = np.r_[-1, -1, -1, -1, np.tile([0, 0, 1, 1], 5)].astype(np.int8)          # (Q,) student seed a perturbation is built on
Q = len(QF)
T8 = 0.5 * np.arange(1, 9)
TYPES = ("agent", "boundary", "none")


# ---------------------------------------------------------------- perturbations (P (n, 8, 3), origin = the state's t0 pose)
def _poly(P):
    P0 = np.concatenate([np.zeros_like(P[:, :1]), P], 1)
    seg = np.diff(P0[..., :2], axis=1)
    ln = np.hypot(seg[..., 0], seg[..., 1])
    return P0, seg, ln, np.concatenate([np.zeros_like(ln[:, :1]), np.cumsum(ln, 1)], 1)


def _integrate(ln, phi, yaw):
    return np.stack([np.cumsum(ln * np.cos(phi), 1), np.cumsum(ln * np.sin(phi), 1), yaw], -1)


def at_arc(P, target):
    """Poses at arc lengths target (n, 8) along the polyline origin -> P; straight continuation along the last heading beyond its end."""
    P0, _, ln, s = _poly(P)
    P0[..., 2] = np.unwrap(P0[..., 2], axis=1)
    j = np.clip((s[:, None, 1:] <= target[:, :, None]).sum(-1), 0, 7)                     # segment index (n, 8)
    g = lambda x: np.take_along_axis(x, j, 1)  # noqa: E731
    s0, l0 = g(s[:, :-1]), g(ln)
    f = np.clip(np.where(l0 > 1e-6, (target - s0) / np.maximum(l0, 1e-6), 0.0), 0, 1)[..., None]
    a, b = np.take_along_axis(P0[:, :-1], j[..., None], 1), np.take_along_axis(P0[:, 1:], j[..., None], 1)
    out = a + f * (b - a)
    ext = np.maximum(target - s[:, -1:], 0)
    out[..., 0] += ext * np.cos(P0[:, -1:, 2])
    out[..., 1] += ext * np.sin(P0[:, -1:, 2])
    return out


def perturb(P, fam, u, sgn):
    """One perturbed copy of P per row; u, sgn (n,) uniform draws in [0, 1) / signs -> (query (n, 8, 3), parameter (n,))."""
    P0, seg, ln, s = _poly(P)
    if fam == "lat":
        a = sgn * np.minimum(0.3 + 2.2 * u, 0.25 * s[:, -1])
        off = a[:, None] * s[:, 1:] / np.maximum(s[:, -1:], 1e-3)
        yaw = P[..., 2] + np.arctan2(a, np.maximum(s[:, -1], 1e-3))[:, None] * (s[:, -1:] > 1e-3)
        return np.stack([P[..., 0] - off * np.sin(P[..., 2]), P[..., 1] + off * np.cos(P[..., 2]), yaw], -1), a
    phi = np.where(ln > 1e-4, np.arctan2(seg[..., 1], seg[..., 0]), 0.5 * (P0[:, :-1, 2] + P0[:, 1:, 2]))
    if fam == "head":
        d = sgn * np.radians(1 + 5 * u)
        return _integrate(ln, phi + d[:, None] * (T8 - 0.25) / 4, P[..., 2] + d[:, None] * T8 / 4), d
    if fam == "gain":
        g = 0.5 + u
        return _integrate(ln, g[:, None] * phi, g[:, None] * P[..., 2]), g
    if fam == "arc":
        c = 0.5 + u
        return at_arc(P, c[:, None] * s[:, 1:]), c
    if fam == "stop":
        f = 0.05 + 0.9 * u
        return at_arc(P, np.minimum(s[:, 1:], f[:, None] * s[:, -1:])), f
    raise ValueError(fam)


def queries(own, ship, fut, rng):
    """own (2, n, 8, 3), ship, fut (n, 8, 3) -> q (n, Q, 8, 3) float32, par (n, Q) float32 (nan for the unperturbed slots)."""
    n = len(ship)
    q, par = np.zeros((n, Q, 8, 3), np.float32), np.full((n, Q), np.nan, np.float32)
    q[:, 0], q[:, 1], q[:, 2], q[:, 3] = own[0], own[1], ship, fut
    for j in range(4, Q):
        q[:, j], par[:, j] = perturb(own[QBASE[j]].astype(np.float64), QFAM[QF[j]], rng.random(n), rng.choice([-1.0, 1.0], n))
    return q, par


# ---------------------------------------------------------------- gen
def unit_file(fam, k, limit):
    return B.root() / "rows" / (B.cdir(fam, k) + (f"-first{limit}" if limit else "") + ".npz")


def unit_key(fam, k, limit, seed):
    from jevdrive import cache
    d = B.cdir(fam, k) + (f"-first{limit}" if limit else "")
    return cache.key(params=dict(fam=fam, k=k, limit=limit, seed=seed, q=Q), inputs=[B.root() / "plans" / f"{d}.npz", B.cache_root() / B.cdir(fam, k) / "tab.npz"],
                     code=[perturb, at_arc, queries, SW.labels, SW.agent_labels, SW.boundary_labels, SW.clearance, SW.dense, SW.dense_boxes])


def make_unit(fam, k, limit, seed, chunk=48):
    L, t = B.labels(), B.tab(fam, k)
    d = B.cdir(fam, k) + (f"-first{limit}" if limit else "")
    z = np.load(B.root() / "plans" / f"{d}.npz")
    n = len(z["names"])
    assert (t["names"][:n] == z["names"]).all()
    gi, off = (x[:n] for x in B.state_index(fam, k, t))
    assert (L["tokens"][gi] == z["names"]).all(), "state rows do not map onto the label rows"
    q, par = queries(z["own"], z["ship"], t["fut"][:n], np.random.default_rng([seed, list(B.FAMS).index(fam), k]))
    qok = np.isfinite(q).all((2, 3))
    q = np.nan_to_num(q)
    out = []
    for i in range(0, n, chunk):
        g = gi[i:i + chunk]
        out.append(SW.labels(q[i:i + chunk], off[i:i + chunk], np.asarray(L["box"][g]), np.asarray(L["valid"][g]), np.asarray(L["cls"][g]), np.asarray(L["sdf"][g])))
    lab = {x: np.concatenate([c[x] for c in out]) for x in out[0]}
    return dict(names=z["names"], gi=gi, off=off, log=t["log"][:n], hold=np.array([B.is_hold(g) for g in t["log"][:n]]), q=q, qok=qok, par=par, qfam=QF, qbase=QBASE,
                qfam_names=np.array(QFAM), **lab)


def work(u):
    from jevdrive import cache
    fam, k, limit, seed, force = u
    t0 = time.time()
    z = cache.cached(unit_file(fam, k, limit), unit_key(fam, k, limit, seed), lambda: make_unit(fam, k, limit, seed), force=force)
    return fam, k, len(z["names"]), time.time() - t0


def cmd_gen(a):
    from jevdrive import cache, par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    units = [(f, k, a.limit, a.seed, a.force) for f in a.fams for k in (a.shards if a.shards is not None else range(B.NSH))]
    with Run("body1", "rows" + (f"-first{a.limit}" if a.limit else ""), seed=a.seed, config=vars(a)) as run:
        for x in ("navsim/navtrain", B.HOLD, B.TRAIN):
            run.use_split(splits.load(x))
        (B.root() / "rows").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = par.pmap(work, units, run=run, workers=min(a.workers or n_cpus(), 40, len(units)),
                       skip=None if a.force else (lambda u: cache.done(unit_file(*u[:3]), unit_key(*u[:4]))))
        res.raise_if_failed()
        done = [r for r in res.values if r is not None]
        run.summary.update(units=len(units), computed=len(done), states=int(sum(r[2] for r in done)), rows=int(sum(r[2] for r in done)) * Q,
                           compute_s=time.time() - t0, core_s=float(sum(r[3] for r in done)))
        run.info(str(run.summary))


# ---------------------------------------------------------------- summary
def load_units(fams, shards, limit):
    out = []
    for f in fams:
        for k in shards:
            p = unit_file(f, k, limit)
            if p.exists():
                out.append((f, k, dict(np.load(p))))
    return out


def ctype(z):
    """(n, Q) contact type of a row: 0 agent, 1 boundary (and no agent), 2 none; -1 = not a row (no query) or ambiguous (rear-ended only)."""
    t = np.where(z["a_hit"], 0, np.where(z["b_hit"], 1, 2)).astype(np.int8)
    t[~z["qok"] | (z["a_rear"] & ~z["a_hit"] & ~z["b_hit"])] = -1
    return t


def cmd_summary(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("body1", "rows-summary" + (f"-first{a.limit}" if a.limit else ""), config=vars(a)) as run:
        for x in ("navsim/navtrain", B.HOLD, B.TRAIN):
            run.use_split(splits.load(x))
        T = dict(np.load(B.root() / "taxonomy" / f"{a.tax}.npz"))
        pos = np.full(B.shard_offsets()[-1], -1)
        pos[T["gi"]] = np.arange(len(T["gi"]))
        U = load_units(a.fams, a.shards if a.shards is not None else range(B.NSH), a.limit)
        fr = []
        for f, k, z in U:
            n = len(z["names"])
            r = pos[z["gi"]]
            assert (r >= 0).all()
            fr.append(pd.DataFrame(dict(
                sfam=f, shard=k, state=np.repeat(np.arange(n), Q), qfam=np.tile(np.array(QFAM)[QF], n), cls=np.repeat(T["pc"][r], Q), gt45=np.repeat(np.abs(T["dyaw"][r]) > 45, Q),
                hold=np.repeat(z["hold"], Q), log=np.repeat(z["log"], Q), type=ctype(z).ravel(), a_hit=z["a_hit"].ravel(), b_hit=z["b_hit"].ravel(),
                a_rear=z["a_rear"].ravel(), a_t0=z["a_t0"].ravel(), b_t0=z["b_t0"].ravel(), a_t=z["a_t"].ravel(), b_t=z["b_t"].ravel(), a_cls=z["a_cls"].ravel(),
                b_cov=z["b_cov"].ravel(), qok=z["qok"].ravel(), haz=np.repeat(T["haz"][r], Q), still=np.repeat(T["still"][r], Q))))
        D = pd.concat(fr, ignore_index=True)
        D = D[D.qok]
        out = B.root() / "rows" / ("summary" + (f"-first{a.limit}" if a.limit else ""))
        out.mkdir(parents=True, exist_ok=True)
        D["cls_s"], D["type_s"], D["split"] = np.array(B.CLS)[D.cls], np.array(TYPES + ("excluded",))[D.type], np.where(D.hold, "hold", "train")
        wr = lambda rows, name: stats.write_table(rows if isinstance(rows, list) else rows.reset_index().to_dict("records"), out / name)  # noqa: E731
        # composition before weighting
        wr(D.groupby(["cls_s", "type_s", "qfam", "sfam", "split"]).size().rename("n"), "composition")
        wr(D.groupby(["cls_s", "type_s", "split"]).size().rename("n"), "composition_class_type")
        # balance: equal mass per (user class, contact type) on the training logs; `w4` also gives class `other` a quarter
        tr = D[~D.hold & (D.type >= 0)]
        cnt = tr.groupby(["cls", "type"]).size()
        Wb = np.zeros((2, 4, 3))                                                 # [w3, w4][class, type]
        for (c, ty), m in cnt.items():
            Wb[1, c, ty] = 1.0 / (4 * 3 * m)
            Wb[0, c, ty] = 1.0 / (3 * 3 * m) if c > 0 else 0.0
        (out / "balance.json").write_text(json.dumps(dict(
            classes=B.CLS, types=TYPES, n=[[int(cnt.get((c, ty), 0)) for ty in range(3)] for c in range(4)], w3=Wb[0].tolist(), w4=Wb[1].tolist(),
            note="row weight = w3[class][type] (class other = 0) or w4[class][type] (other is a fourth class); rows of hold logs, rows without a query "
                 "(qok false) and rows whose only contact is a rear-end by a faster object get 0; every (class, type) cell then has equal mass"), indent=1))
        tw = tr.assign(w=Wb[0][tr.cls.to_numpy(), tr.type.to_numpy()])
        before = tr.groupby(["cls_s", "type_s"]).size() / len(tr)
        wr(pd.DataFrame({"n": tr.groupby(["cls_s", "type_s"]).size(), "share_before": before, "share_after_w3": tw.groupby(["cls_s", "type_s"]).w.sum()}), "balance")
        wr(pd.DataFrame({"share_before": tr.groupby("qfam").size() / len(tr), "share_after_w3": tw.groupby("qfam").w.sum()}), "balance_qfam")
        wr(pd.DataFrame({"share_before": tr.groupby("sfam").size() / len(tr), "share_after_w3": tw.groupby("sfam").w.sum()}), "balance_sfam")
        # contact rates per query family and state family
        g = D.groupby(["sfam", "qfam"])
        rates = pd.DataFrame({"n": g.size(), "agent": g.a_hit.mean(), "boundary": g.b_hit.mean(), "rear_flag": g.a_rear.mean(), "agent_t0": g.a_t0.mean(),
                              "boundary_t0": g.b_t0.mean(), "agent_after_t0": g.apply(lambda x: (x.a_hit & ~x.a_t0).mean()), "boundary_after_t0": g.apply(lambda x: (x.b_hit & ~x.b_t0).mean()),
                              "cov_full": g.apply(lambda x: (x.b_cov == SW.NS).mean()), "agent_t_median": g.apply(lambda x: x.a_t[x.a_hit].median()),
                              "boundary_t_median": g.apply(lambda x: x.b_t[x.b_hit].median())})
        wr(rates, "contact_rates")
        gc = D.groupby(["cls_s", "qfam"])
        wr(pd.DataFrame({"n": gc.size(), "agent": gc.a_hit.mean(), "boundary": gc.b_hit.mean()}), "contact_rates_class")
        # first-contact time distribution
        bins = np.r_[-0.05, 0.05, np.arange(0.5, 4.01, 0.5) + 0.05]
        hist = []
        for (qf,), x in D.groupby(["qfam"]):
            for kind in ("a", "b"):
                h = np.histogram(x[f"{kind}_t"][x[f"{kind}_hit"]], bins)[0]
                hist.append(dict(qfam=qf, kind="agent" if kind == "a" else "boundary", n=int(h.sum()), **{f"t{b:.1f}": int(v) for b, v in zip(np.r_[0, np.arange(0.5, 4.01, 0.5)], h)}))
        wr(hist, "first_contact_time")
        obj = D[D.a_hit].groupby(["qfam", "a_cls"]).size().rename("n")
        wr(obj, "struck_class")
        # the student's own plan on hold logs: positives per user class (the G1 gate needs >= 30 per class)
        own = D[D.qfam.isin(["own0", "own1"])]
        rows = []
        for (qf, split), x in own.groupby(["qfam", "split"]):
            for name, m in [(c, x.cls == i) for i, c in enumerate(B.CLS)] + [("all", x.cls >= 0), ("> 45 deg", x.gt45)]:
                y = x[m]
                for pool, yy in (("log", y[y.sfam == "log"]), ("off-track", y[y.sfam != "log"]), ("pooled", y)):
                    rows.append(dict(model=qf, split=split, cls=name, states=pool, n=len(yy), logs=yy.log.nunique(), agent_pos=int(yy.a_hit.sum()), agent_pos_after_t0=int((yy.a_hit & ~yy.a_t0).sum()),
                                     agent_pos_logs=yy.log[yy.a_hit].nunique(), boundary_pos=int(yy.b_hit.sum()), boundary_pos_after_t0=int((yy.b_hit & ~yy.b_t0).sum()),
                                     boundary_pos_logs=yy.log[yy.b_hit].nunique(), agent_rate=float(yy.a_hit.mean()) if len(yy) else np.nan,
                                     boundary_rate=float(yy.b_hit.mean()) if len(yy) else np.nan))
        wr(rows, "own_plan_positives")
        g45 = own[own.gt45].groupby(["qfam", "sfam", np.array(B.HAZ)[own[own.gt45].haz]])
        wr(pd.DataFrame({"n": g45.size(), "agent": g45.a_hit.mean(), "boundary": g45.b_hit.mean(), "agent_after_t0": g45.apply(lambda x: (x.a_hit & ~x.a_t0).mean()),
                         "boundary_after_t0": g45.apply(lambda x: (x.b_hit & ~x.b_t0).mean())}).rename_axis(["qfam", "sfam", "hazard"]), "gt45_own_plan")
        gb = sum(p.stat().st_size for p in (B.root() / "rows").glob("*.npz")) / 2 ** 30
        run.summary.update(rows=len(D), states=int(sum(len(z["names"]) for _, _, z in U)), units=len(U), gb=gb, out=str(out),
                           hold_pos={f"{r['model']}|{r['cls']}": [r["agent_pos"], r["boundary_pos"]] for r in rows if r["split"] == "hold" and r["states"] == "pooled"})
        print(rates.to_string())
        print(pd.DataFrame(rows).query("split == 'hold'").to_string())
        print(pd.DataFrame(hist).to_string())
        run.info(str(run.summary))


# ---------------------------------------------------------------- agreement checks
def cmd_check(a):
    import shapely
    import torch
    from jevdrive.run import Run
    _sys.path[:0] = [str(B.REPO / "experiments/alpasim/scripts"), str(B.REPO / "experiments/alpasim/lib")]
    import swv1_lib as SV
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    with Run("body1", "sweep-check", config=vars(a)) as run:
        L = B.labels()
        z = dict(np.load(unit_file("log", a.shard, a.limit)))
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(np.flatnonzero(z["qok"].all(1)), min(a.n, int(z["qok"].all(1).sum())), replace=False))
        gi, q = z["gi"][pick], z["q"][pick]
        box, valid, cls, sdf = (np.asarray(L[x][gi]) for x in ("box", "valid", "cls", "sdf"))
        S = len(pick)
        d = SW.dense(q)
        b, v, _ = SW.dense_boxes(box, valid)
        v &= (cls >= 0)[:, None]
        ex, ey = SW.ego_centre(d)
        clr, hit = SW.clearance(ex[..., None], ey[..., None], d[..., 2:3], b[:, None, ..., 0], b[:, None, ..., 1], b[:, None, ..., 2], b[:, None, ..., 3] / 2, b[:, None, ..., 4] / 2)
        vq = np.broadcast_to(v[:, None], hit.shape)
        res = {"states": S, "queries": S * Q, "pairs_valid": int(vq.sum())}
        # 1. AgentHinge.distances (torch, margin 0): the same boxes, the same interpolation
        H = AgentHinge(B.data_dir() / "runs/op_parity/agent_labels/navtrain_all-k32.npz", z["names"][pick], torch.device("cpu"), margin=0.0)
        dist = np.stack([H.distances(*(torch.as_tensor(q[:, j, :, c]) for c in range(3)), torch.arange(S))[0].numpy() for j in range(Q)], 1)
        sep = vq & (clr > 0) & (dist > 0)
        res["agent_hinge"] = {"max_abs_diff_separated_m": float(np.abs(clr - dist)[sep].max()), "mean_abs_diff_separated_m": float(np.abs(clr - dist)[sep].mean()),
                              "contact_agree": float(((dist < 0) == hit)[vq].mean()), "contact_ours": int((hit & vq).sum()), "contact_hinge": int(((dist < 0) & vq).sum()),
                              "ours_only": int((hit & ~(dist < 0) & vq).sum()), "hinge_only": int((~hit & (dist < 0) & vq).sum())}
        # 2. swv1_lib.Rollout.sweep (shapely polygons; objects annotated at all 9 times, tracks interpolated by col1_lib.interp)
        full = valid.all(1) & (cls >= 0)
        n_obj = n_first = n_hit = n_hit_agree = 0
        dmax = 0.0
        for i in range(min(S, a.n_shapely)):
            ks = np.flatnonzero(full[i])
            r = object.__new__(SV.Rollout)
            r.o = {"actors": {int(k): np.c_[np.arange(9) * 0.5e6, box[i, :, k, :3]] for k in ks}, "size": {int(k): tuple(box[i, 0, k, 3:5]) for k in ks}}
            r.off, r.Le, r.We, r.ids, r._obj = SW.C_OFF, 2 * SW.HALF_L, 2 * SW.HALF_W, [int(k) for k in ks], {}
            r.now, r.plan = np.zeros(Q), d[i].astype(np.float64)
            for j in range(Q):
                sw = r.sweep(j)
                for k, (dm, f) in sw.items():
                    mine_first = int(hit[i, j, :, k].argmax()) if hit[i, j, :, k].any() else -1
                    n_obj += 1
                    n_first += mine_first == f
                    n_hit += f >= 0
                    n_hit_agree += (f >= 0) and mine_first >= 0
                    if 0 < dm < 5 and mine_first < 0:
                        dmax = max(dmax, abs(dm - float(clr[i, j, :, k].min())))
        res["swv1_sweep"] = {"states": min(S, a.n_shapely), "object_sweeps": n_obj, "first_contact_index_agree": n_first / max(n_obj, 1), "contacts_swv1": n_hit,
                             "contacts_both": n_hit_agree, "max_abs_min_clearance_diff_m": dmax, "shapely": shapely.__version__}
        # 3. drivable_hinge.Hinge.margins (torch grid_sample on the same raster)
        Hd = Hinge(B.data_dir() / "runs/op_probe/labels/navtrain_all.npz", z["names"][pick], torch.device("cpu"))
        m, ins = SW.corner_margins(d, sdf)
        mt = np.stack([Hd.margins(*(torch.as_tensor(q[:, j, :, c]) for c in range(3)), torch.arange(S)).numpy() for j in range(Q)], 1).reshape(m.shape)
        m, mt = np.sort(m, -1), np.sort(mt, -1)                                  # the two modules order the 4 corners differently
        res["drivable_hinge"] = {"corner_samples": int(m.size), "max_abs_diff_m": float(np.abs(m - mt).max()), "mean_abs_diff_m": float(np.abs(m - mt).mean()),
                                 "sign_agree": float(((m < 0) == (mt < 0)).mean()), "inside_raster": float(ins.mean())}
        (B.root() / "rows" / "check.json").write_text(json.dumps(res, indent=1))
        run.summary.update(res)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "summary"):
        p = sp.add_parser(name)
        p.add_argument("--fams", nargs="+", default=["log", "ot1", "yr1"])
        p.add_argument("--shards", type=int, nargs="+", default=None)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--tax", default="tax")
        cli_args(p)
    p = sp.add_parser("check")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--n-shapely", type=int, default=100)
    cli_args(p)
    a = ap.parse_args()
    {"gen": cmd_gen, "summary": cmd_summary, "check": cmd_check}[a.cmd](a)
