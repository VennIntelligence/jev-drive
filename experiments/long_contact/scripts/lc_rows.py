#!/usr/bin/env python3
"""LONG-PROBE row labels (CPU): the longitudinal labels A / B / C (lib/lc.py) for every (state, query) row of BODY1's row units.

  gen     [--fams log ot1 yr1] [--shards ...]  -> $DATA_DIR/runs/long_contact/rows/<cache dir>.npz (a_lab, a_sub, a_t, b_lab, b_kind, b_t,
          c_lab, c_dv, c_t: (n, 24); a_elig_n, b_ahead, c_ahead: (n,)), aligned with body1/rows/<cache dir>.npz
  survey  label counts (no model output): own-plan positives per class, split (hold / val / fit), state family, speed band, sub-class
          -> $DATA_DIR/runs/long_contact/rows/survey/*.csv (copied to experiments/long_contact/results/survey/)
  priv    rung-4 (privileged, oracle-only) inputs per state in the STATE's frame -> $DATA_DIR/runs/long_contact/priv/<cache dir>.npz:
          obj (n, 32, 69) the 32 logged agent boxes at 0 .. 4 s (x, y, sin / cos yaw, L, W, valid) + class + A-eligible + speed;
          rte (n, 157) the hindsight route every 4 m to 120 m + the logged speed / arrival time there + its known length;
          lin (n, 4, 40) the stop lines (type, arc, eligibility, light state at the 9 keyframes, the logged stop / crossing);
          trn (n, 10) the first >= 45 deg turn (start / peak / end arc, curvature, angle, logged speed at peak and start, eligibility)
The trainer never reads a hold-log row; this step labels all rows (labels only, no model).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib")]
import lc as LC  # noqa: E402
import b1 as B  # noqa: E402
import contact_head as C  # noqa: E402

FAMS = ("log", "ot1", "yr1")
KEYS = ("a_lab", "a_sub", "a_t", "b_lab", "b_kind", "b_t", "c_lab", "c_dv", "c_t", "a_elig_n", "b_ahead", "c_ahead")
CTX_KEYS = ("ok", "route", "rhead", "rlen", "ln_type", "ln_d", "ln_st", "ln_elig", "tn_ok", "tn_spk", "tn_vpk")
PRIV_KEYS = CTX_KEYS + ("hlen", "vlog", "tlog", "ln_conn", "ln_tstop", "ln_tcross", "ln_stcross", "tn_sa", "tn_sb", "tn_kpk", "tn_dth", "tn_va", "tn_tpk")
_X = {}


def ctx(keys=CTX_KEYS):
    if not all(k in _X for k in keys):
        z = np.load(LC.ctx_path())
        _X.update({k: z[k] for k in keys})
    return _X


def to_state(x, y, off):
    """Logged-frame points (S, ...) -> the state's own frame (inverse of sweep.to_log); off (S, 2) = (dy, dpsi)."""
    sh = (len(off),) + (1,) * (np.ndim(x) - 1)
    dy, c, s = off[:, 0].reshape(sh), np.cos(off[:, 1]).reshape(sh), np.sin(off[:, 1]).reshape(sh)
    return c * x + s * (y - dy), -s * x + c * (y - dy)


def priv_feats(gi, off, L, X):
    """Rung-4 inputs of states (global label rows gi, offsets off) -> dict obj / rte / lin / trn (float16)."""
    S = len(gi)
    box, valid, cls = np.asarray(L["box"][gi]).astype(np.float32), np.asarray(L["valid"][gi]), np.asarray(L["cls"][gi])
    R, H, rlen = X["route"][gi], X["rhead"][gi], X["rlen"][gi]
    _, _, vel = LC.SW.dense_boxes(box, valid)
    elig, _ = LC.a_eligible(box[:, 0], valid[:, 0] & (cls >= 0), cls, vel[:, 0], R, H, rlen)
    v = valid & (cls >= 0)[:, None]
    bx, by = to_state(box[..., 0], box[..., 1], off)
    yaw = box[..., 2] - off[:, 1, None, None]
    o = np.stack([bx / 32, by / 24, np.sin(yaw), np.cos(yaw), box[..., 3] / 5, box[..., 4] / 2.5, np.ones_like(bx)], -1) * v[..., None]   # (S, 9, K, 7)
    o = o.transpose(0, 2, 1, 3).reshape(S, -1, 63)
    oh = np.eye(4, dtype=np.float32)[np.clip(cls, 0, 3)] * (cls >= 0)[..., None]
    obj = np.concatenate([o, oh, elig[..., None], (np.hypot(vel[:, 0, :, 0], vel[:, 0, :, 1]) / 10)[..., None]], -1)
    idx = np.arange(0, 121, 4)
    rx, ry = to_state(R[:, idx, 0], R[:, idx, 1], off)
    vl, tl = X["vlog"][gi][:, idx].astype(np.float32), X["tlog"][gi][:, idx].astype(np.float32)
    rte = np.concatenate([rx / 32, ry / 24, np.nan_to_num(vl / 10), np.isfinite(vl), np.nan_to_num(tl / 10), (rlen / 100)[:, None], (X["hlen"][gi] / 100)[:, None]], -1)
    ty, st = X["ln_type"][gi], X["ln_st"][gi]
    f = lambda a: np.nan_to_num(a / 10)[..., None]  # noqa: E731
    has = lambda a: np.isfinite(a)[..., None]  # noqa: E731
    lin = np.concatenate([(ty > 0)[..., None], X["ln_elig"][gi][..., None], (ty == 1)[..., None], (ty == 2)[..., None], np.nan_to_num(X["ln_d"][gi] / 50)[..., None],
                          X["ln_conn"][gi][..., None], np.eye(3, dtype=np.float32)[st + 1].reshape(S, LC.NL, -1), f(X["ln_tstop"][gi]), has(X["ln_tstop"][gi]),
                          f(X["ln_tcross"][gi]), has(X["ln_tcross"][gi]), np.eye(3, dtype=np.float32)[X["ln_stcross"][gi] + 1]], -1)
    t = lambda k, sc: np.nan_to_num(X[f"tn_{k}"][gi] / sc)  # noqa: E731
    trn = np.stack([X["tn_ok"][gi], t("sa", 50), t("sb", 50), t("spk", 50), t("kpk", 0.1), t("dth", 90), t("vpk", 10), t("va", 10), t("tpk", 10),
                    X["tn_ok"][gi] & (X["tn_vpk"][gi] >= LC.C_VLOG_MIN)], -1)
    return {k: x.astype(np.float16) for k, x in dict(obj=obj, rte=rte, lin=lin, trn=trn).items()}


def priv_file(f, k):
    return LC.root() / "priv" / f"{B.cdir(f, k)}.npz"


def work_priv(u):
    f, k, chunk = u
    z = np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz")
    L, X = B.labels(), ctx(PRIV_KEYS)
    gi, off = z["gi"], z["off"].astype(np.float32)
    parts = [priv_feats(gi[c:c + chunk], off[c:c + chunk], L, X) for c in range(0, len(gi), chunk)]
    out = {x: np.concatenate([p[x] for p in parts]) for x in parts[0]}
    p = priv_file(f, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".tmp.npz"), gi=gi, **out)
    p.with_suffix(".tmp.npz").rename(p)
    return f, k, len(gi)


def cmd_priv(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("long_contact", "priv", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        units = [(f, k, 512) for f in a.fams for k in (a.shards if a.shards is not None else range(B.NSH))]
        res = par.pmap(work_priv, units, run=run, workers=min(a.workers or n_cpus(), len(units)), desc="priv units",
                       skip=None if a.force else (lambda u: priv_file(u[0], u[1]).exists()))
        res.raise_if_failed()
        run.summary.update(units=len(units))


def unit_file(f, k):
    return LC.root() / "rows" / f"{B.cdir(f, k)}.npz"


def work(u):
    f, k, chunk = u
    t0 = time.time()
    z = np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz")
    L, X = B.labels(), ctx()
    n = len(z["gi"])
    parts = []
    for c in range(0, n, chunk):
        sl = slice(c, min(c + chunk, n))
        gi = z["gi"][sl]
        parts.append(LC.labels(z["q"][sl], z["off"][sl], np.asarray(L["box"][gi]), np.asarray(L["valid"][gi]), np.asarray(L["cls"][gi]),
                               {x: X[x][gi] for x in CTX_KEYS}, z["a_hit"][sl], z["a_t0"][sl], z["qok"][sl]))
    out = {x: np.concatenate([p[x] for p in parts]) for x in KEYS}
    p = unit_file(f, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".tmp.npz"), gi=z["gi"], **out)
    p.with_suffix(".tmp.npz").rename(p)
    return f, k, n, time.time() - t0


def cmd_gen(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("long_contact", "rows", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        units = [(f, k, a.chunk) for f in a.fams for k in (a.shards if a.shards is not None else range(B.NSH))]
        res = par.pmap(work, units, run=run, workers=min(a.workers or n_cpus(), len(units)), desc="row units",
                       skip=None if a.force else (lambda u: unit_file(u[0], u[1]).exists()))
        res.raise_if_failed()
        done = [r for r in res.values if r is not None]
        run.summary.update(units=len(units), computed=len(done), states=int(sum(r[2] for r in done)), core_s=float(sum(r[3] for r in done)))


def load(fams=FAMS, shards=range(B.NSH)):
    """All row units -> dict of concatenated arrays + body1 fields (log, hold, fam, speed, qok, a_hit) and the user class."""
    out = {}
    for fi, f in enumerate(fams):
        for k in shards:
            z, y = np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz"), np.load(unit_file(f, k))
            assert (z["gi"] == y["gi"]).all()
            t = B.tab(f, k)
            cols = {x: y[x] for x in KEYS} | {x: z[x] for x in ("gi", "log", "hold", "qok", "a_hit", "a_t0", "a_rear")}
            cols |= dict(fam=np.full(len(z["gi"]), fi, np.int8), speed=t["speed"].astype(np.float32))
            for x, v in cols.items():
                out.setdefault(x, []).append(v)
    return {x: np.concatenate(v) for x, v in out.items()} | dict(fams=list(fams))


def cmd_survey(a):
    import pandas as pd
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("long_contact", "rows-survey", config=vars(a)) as run:
        for s in (B.HOLD, B.TRAIN):
            run.use_split(splits.load(s))
        D = load()
        T = dict(np.load(B.root() / "taxonomy" / "tax.npz"))
        cls, gt45 = C.classes(T)
        split = np.where(D["hold"], "hold", np.where([C.is_val(l) for l in D["log"].tolist()], "val", "fit"))
        rows = []
        for qs, qname in (((0, 1), "own"), (tuple(range(16, 20)), "arc (own x c)"), (tuple(range(24)), "all queries")):
            for lab in LC.LONG:
                y = D[f"{lab.lower()}_lab"][:, list(qs)]
                for sp in ("hold", "val", "fit"):
                    for fam in ("all",) + tuple(D["fams"]):
                        m = (split == sp) & ((D["fam"] == D["fams"].index(fam)) if fam != "all" else True)
                        yy = y[m]
                        lg = np.broadcast_to(D["log"][m][:, None], yy.shape)
                        rows.append(dict(query=qname, label=lab, split=sp, states=fam, rows=int(yy.size), pos=int((yy == 1).sum()), neg=int((yy == 0).sum()),
                                         excluded=int((yy < 0).sum()), logs=len(set(D["log"][m].tolist())), logs_pos=len(set(lg[yy == 1].tolist())),
                                         rate=float((yy == 1).sum() / max((yy >= 0).sum(), 1))))
        out = LC.root() / "rows" / "survey"
        out.mkdir(parents=True, exist_ok=True)
        S = pd.DataFrame(rows)
        S.to_csv(out / "counts.csv", index=False)
        # own-plan positives on hold, by stratum
        own = list((0, 1))
        rr = []
        sb = LC.speed_band(D["speed"])
        for lab in LC.LONG:
            y = D[f"{lab.lower()}_lab"][:, own]
            for sp in ("hold", "fit"):
                m = split == sp
                for nm, g in (("speed band", sb), ("user class", cls[D["gi"]]), ("> 45 deg", gt45[D["gi"]].astype(int))):
                    for v in np.unique(g[m]):
                        mm = m & (g == v)
                        yy = y[mm]
                        rr.append(dict(label=lab, split=sp, stratum=nm, value=(LC.SPEED_NAMES[v] if nm == "speed band" else int(v)), pos=int((yy == 1).sum()),
                                       neg=int((yy == 0).sum()), excluded=int((yy < 0).sum())))
                if lab == "A":
                    for v, nmv in ((1, "A1 stopped"), (2, "A2 moving")):
                        rr.append(dict(label=lab, split=sp, stratum="sub-class", value=nmv, pos=int(((D["a_sub"][:, own] == v) & (y == 1))[m].sum())))
                if lab == "B":
                    for v, nmv in ((1, "stop sign"), (2, "red light")):
                        rr.append(dict(label=lab, split=sp, stratum="sub-class", value=nmv, pos=int(((D["b_kind"][:, own] == v) & (y == 1))[m].sum())))
                    rr.append(dict(label=lab, split=sp, stratum="state has an eligible line <= 60 m", value=1, pos=int((y == 1)[m & D["b_ahead"]].sum()),
                                   neg=int((y == 0)[m & D["b_ahead"]].sum()), excluded=int((y < 0)[m & D["b_ahead"]].sum())))
                if lab == "C":
                    rr.append(dict(label=lab, split=sp, stratum="state has an eligible turn peak <= 60 m", value=1, pos=int((y == 1)[m & D["c_ahead"]].sum()),
                                   neg=int((y == 0)[m & D["c_ahead"]].sum()), excluded=int((y < 0)[m & D["c_ahead"]].sum())))
        pd.DataFrame(rr).to_csv(out / "strata.csv", index=False)
        # A-type vs other agent contacts of the own plan (the lateral-by-class split of S0's positives)
        ah = D["a_hit"][:, own]
        al = D["a_lab"][:, own]
        rr = []
        for sp in ("hold", "fit"):
            m = (split == sp)[:, None] & D["qok"][:, own]
            rr.append(dict(split=sp, agent_contacts=int((ah & m).sum()), a_type=int((ah & (al == 1) & m).sum()), other=int((ah & (al != 1) & m).sum()),
                           a_pos_without_counted_contact=int((~ah & (al == 1) & m).sum())))
        pd.DataFrame(rr).to_csv(out / "a_split.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_rows", 500):
            print(S[(S.states == "all")].to_string(index=False))
            print(pd.read_csv(out / "strata.csv").to_string(index=False))
            print(pd.read_csv(out / "a_split.csv").to_string(index=False))
        run.summary.update(out=str(out))


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gen")
    p.add_argument("--fams", nargs="+", default=list(FAMS))
    p.add_argument("--shards", type=int, nargs="+", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--chunk", type=int, default=48)
    cli_args(p)
    p = sp.add_parser("survey")
    cli_args(p)
    p = sp.add_parser("priv")
    p.add_argument("--fams", nargs="+", default=list(FAMS))
    p.add_argument("--shards", type=int, nargs="+", default=None)
    p.add_argument("--workers", type=int, default=0)
    cli_args(p)
    a = ap.parse_args()
    {"gen": cmd_gen, "survey": cmd_survey, "priv": cmd_priv}[a.cmd](a)
