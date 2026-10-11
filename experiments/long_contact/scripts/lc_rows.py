#!/usr/bin/env python3
"""LONG-PROBE row labels (CPU): the longitudinal labels A / B / C (lib/lc.py) for every (state, query) row of BODY1's row units.

  gen     [--fams log ot1 yr1] [--shards ...]  -> $DATA_DIR/runs/long_contact/rows/<cache dir>.npz (a_lab, a_sub, a_t, b_lab, b_kind, b_t,
          c_lab, c_dv, c_t: (n, 24); a_elig_n, b_ahead, c_ahead: (n,)), aligned with body1/rows/<cache dir>.npz
  survey  label counts (no model output): own-plan positives per class, split (hold / val / fit), state family, speed band, sub-class
          -> $DATA_DIR/runs/long_contact/rows/survey/*.csv (copied to experiments/long_contact/results/survey/)
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
_X = {}


def ctx():
    if not _X:
        z = np.load(LC.ctx_path())
        _X.update({k: z[k] for k in CTX_KEYS})
    return _X


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
    a = ap.parse_args()
    {"gen": cmd_gen, "survey": cmd_survey}[a.cmd](a)
