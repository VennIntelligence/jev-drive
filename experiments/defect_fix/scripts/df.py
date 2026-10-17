#!/usr/bin/env python3
"""NR1, the next-round joint arm against the visible defects (plans/2026-10-17-nr1-prereg.md). Box, repo root; chain in df.sh.
Outputs: box runs/defect_fix/, repo experiments/defect_fix/results/.

Base stage data (the trainval rows already on disk: SELF-ITER batches A-C, runs/self_iter/batches/*, token caches idx_selfiter_tr*):
  reasons [--workers N]   (navsim2, CPU) every new trainval row: led.log_cells per log (cap_ledger's per-frame row, the hindsight route's stop
                          lines) -> the first stop line ahead: type (1 stop sign, 2 traffic light), front-bumper distance dF, light state at t0,
                          route connector flag -> runs/defect_fix/reasons.npz
  pack                    (op-train, CPU) pool on1_NR1TV = the train rows (hold 0) of the batches as imitation rows (rm_train --on-mode imit):
                          tab (si.py pack's columns) + per-row tags stand / launch / rlead / rsign / rred / cls / cell; teacher, front, route
                          features concatenated (no merge; no label files: imitation rows open none)
  weights --wait W --launch L [--rule reasoned|all] [--line-max 10] --out NAME   (op-train, CPU) per-row draw weights of on1_NR1TV for
                          rm_train --on-wfile -> runs/defect_fix/w/NAME.npz + results/w_NAME.json (realised shares):
      rule reasoned (doses W / L = 0.05 / 0.05, 0.10 / 0.10): standing rows (v0 < 0.3, human 4 s displacement < 1 m) only with a visible
          reason (a stopped lead <= 15 m: wait_cap's W_L; or the first stop line ahead on the route within -2 .. line-max m of the front
          bumper: a stop sign, or a traffic light red at t0 on the route's connector) get the mass W; standing rows without one weight 0;
          launch rows (wait_cap's G_L / G_N) the mass L; every other row the rest, split by navtrain's class shares (lib/self_iter.classes on
          the navtrain rows that are neither standing nor launching), uniform inside a class
      rule all (the coordinator's third dose = decision 338's SIU mix in the base stage): every train row at its natural share (uniform)
Rows per batch (prereg 2.3): k_tv = round(115 x n_tv / (n_nav + n_tv)) = si.py pack's rule on the base stage's 115 normal slots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/wait_cap/scripts"), str(REPO / "experiments/self_iter/scripts")]
import self_iter as SI  # noqa: E402
import wt  # noqa: E402

DATA, NAV, CR, FEAT = wt.DATA, wt.NAV, wt.CR, wt.FEAT
OUT = DATA / "runs/defect_fix"
RES = REPO / "experiments/defect_fix/results"
SIB = DATA / "runs/self_iter/batches"
POOL = "on1_NR1TV"
BATCHES = ("A", "B", "C")
V_STAND, D_STAND = 0.3, 1.0                     # standing: wait_cap's W cells' rule (v0 < 0.3, 4 s displacement < 1 m)
LEAD_CELLS, LAUNCH_CELLS = ("W_L",), ("G_L", "G_N")
NN0_BASE, NAV_TRAIN = 115, 101499              # the base stage's normal slots (128 - 13 yaw-rate rows) and navtrain train rows


def write(name, res, md=None):
    RES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (OUT / f"{name}.json", RES / f"{name}.json"):
        p.write_text(json.dumps(res, indent=1, default=float))
    if md:
        for p in (OUT / f"{name}.md", RES / f"{name}.md"):
            p.write_text("\n".join(md) + "\n")
        print("\n".join(md))
    else:
        print(json.dumps(res, indent=1, default=float))


def batch_rows():
    """The SELF-ITER batch rows (every new trainval row), concatenated, with the derived standing / launch flags."""
    Z = [np.load(SIB / b / "rows.npz") for b in BATCHES]
    R = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}
    f = R["fut"]
    R["d4"], R["d2"] = np.hypot(f[:, 7, 0], f[:, 7, 1]), np.hypot(f[:, 3, 0], f[:, 3, 1])
    R["stand"] = (R["v0"] < V_STAND) & (R["d4"] < D_STAND)
    R["launch"] = np.isin(R["cell"].astype(str), LAUNCH_CELLS)
    return R


# ---------------------------------------------------------------- reasons (navsim2, CPU)
_LED = None


def _reason_job(job):
    global _LED
    path, toks = job
    if _LED is None:
        _LED = wt._led()
    out = []
    for r in _LED.log_cells(path):
        if r["token"] not in toks:
            continue
        ty, dF, st0, conn = (np.asarray(r[k]) for k in ("ln_type", "ln_d", "ln_st0", "ln_conn"))
        dF = dF - _LED.FRONT
        lines = ty > 0
        q = int(np.argmax(lines)) if lines.any() else -1
        out.append((r["token"], int(ty[q]) if q >= 0 else 0, float(dF[q]) if q >= 0 else np.inf, int(st0[q]) if q >= 0 else -1,
                    bool(conn[q]) if q >= 0 else False))
    return out


def cmd_reasons(a):
    from jevdrive import par
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    R = batch_rows()
    tok, log = R["token"].astype(str), R["log"].astype(str)
    by = {}
    for t, l in zip(tok.tolist(), log.tolist()):
        by.setdefault(l, set()).add(t)
    jobs = [(NAV / "navsim_logs/trainval" / f"{l}.pkl", s) for l, s in sorted(by.items())]
    with Run("defect_fix", "reasons", config=vars(a)) as run:
        t0 = time.time()
        res = par.pmap(_reason_job, jobs, run=run, workers=a.workers, desc="logs")
        res.raise_if_failed()
        got = {x[0]: x[1:] for v in res.values for x in v}
        miss = [t for t in tok.tolist() if t not in got]
        g = lambda j, d: np.array([got[t][j] if t in got else d for t in tok.tolist()])  # noqa: E731
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "reasons.npz", token=tok, ln_type=g(0, 0).astype(np.int8), ln_dF=g(1, np.inf).astype(np.float32),
                 ln_st0=g(2, -1).astype(np.int8), ln_conn=g(3, False).astype(bool), labelled=np.isin(tok, list(got)))
        rec = dict(rows=len(tok), logs=len(jobs), labelled=len(got), missing=len(miss), missing_sample=miss[:5], seconds=time.time() - t0)
        run.summary |= rec
        print(json.dumps(rec, indent=1))


def reason_flags(R, line_max=10.0, line_min=-2.0):
    """Per batch row: rlead (wait_cap's W_L), rsign (stop sign line ahead within line_min .. line_max m of the front bumper), rred (traffic
    light line there, red at t0, on the route's connector); rows reasons.npz could not label count as no line."""
    z = np.load(OUT / "reasons.npz")
    assert np.array_equal(z["token"].astype(str), R["token"].astype(str)), "reasons.npz rows != batch rows"
    near = (z["ln_dF"] >= line_min) & (z["ln_dF"] <= line_max) & z["labelled"]
    return dict(rlead=np.isin(R["cell"].astype(str), LEAD_CELLS), rsign=near & (z["ln_type"] == 1),
                rred=near & (z["ln_type"] == 2) & (z["ln_st0"] == 1) & z["ln_conn"])


# ---------------------------------------------------------------- pack (op-train, CPU)
def cmd_pack(a):
    from jevdrive.data import splits
    src, dw = CR / POOL, CR / f"{POOL}@warp"
    assert not (src / "tab.npz").exists(), f"{src} exists"
    R = batch_rows()
    fl = reason_flags(R)
    pos = {t: i for i, t in enumerate(R["token"].astype(str).tolist())}
    parts = []
    for b in BATCHES:
        d = f"idx_{SI_split(b)}"
        tb = dict(np.load(CR / d / "tab.npz"))
        j = np.array([pos[t] for t in tb["names"].astype(str).tolist()])
        keep = ~R["hold"][j].astype(bool)                               # train rows only (hold = crc32(log) % 10 == 0 stays unseen)
        parts.append((d, tb, j, keep))
    cols = ("names", "log", "ego", "pose", "vel", "acc", "cmd", "fut", "cam", "lht", "speed")
    tb = {k: np.concatenate([p[1][k][p[3]] for p in parts]) for k in cols}
    J = np.concatenate([p[2][p[3]] for p in parts])
    n = len(tb["names"])
    assert np.isfinite(tb["fut"]).all(), "rows without a full future"
    assert np.array_equal(tb["names"].astype(str), R["token"].astype(str)[J])
    pn = np.array([f"on:{t}" for t in tb["names"].astype(str).tolist()])
    s01 = lambda m: np.where(m, "1", "0").astype("U1")  # noqa: E731
    cls = SI.classes(tb["fut"], tb["speed"])
    T = {k: v for k, v in tb.items() if k != "names"} | dict(names=pn, src=np.full(n, "nr1tv", "U16"), cls=cls.astype("U8"),
                                                          cell=R["cell"][J].astype("U4"), stand=s01(R["stand"][J]), launch=s01(R["launch"][J]),
                                                          rlead=s01(fl["rlead"][J]), rsign=s01(fl["rsign"][J]), rred=s01(fl["rred"][J]))
    src.mkdir(exist_ok=True), dw.mkdir(exist_ok=True)
    np.savez(src / "tab.npz", **T)
    FZ = [np.load(FEAT / f"{d}.npz") for d, _, _, _ in parts]
    for (d, t_, _, _), z in zip(parts, FZ):
        assert np.array_equal(z["names"], t_["names"]), f"{d}: route feature rows != cache rows"
    np.savez(FEAT / f"{POOL}.npz", names=pn, f=np.concatenate([z["f"][p[3]] for z, p in zip(FZ, parts)]).astype(np.float16),
             k=np.concatenate([z["k"][p[3]] for z, p in zip(FZ, parts)]).astype(np.int64), ok=np.concatenate([z["ok"][p[3]] for z, p in zip(FZ, parts)]))
    TH = [np.load(CR / f"{d}@warp/teacher.npz") for d, _, _, _ in parts]
    assert all(np.array_equal(t["di"], TH[0]["di"]) and np.array_equal(t["pi"], TH[0]["pi"]) for t in TH)
    np.savez(dw / "teacher.npz", out=np.concatenate([t["out"][p[3]] for t, p in zip(TH, parts)]),
             plan=np.concatenate([t["plan"][p[3]] for t, p in zip(TH, parts)]), di=TH[0]["di"], pi=TH[0]["pi"])
    FR = [np.load(CR / f"{d}@warp/front.npy", mmap_mode="r") for d, _, _, _ in parts]
    o = np.lib.format.open_memmap(dw / "front.npy", "w+", np.float16, (n,) + FR[0].shape[1:])
    i0 = 0
    for f_, p in zip(FR, parts):
        idx = np.flatnonzero(p[3])
        for i in range(0, len(idx), 2048):
            o[i0 + i:i0 + min(i + 2048, len(idx))] = f_[idx[i:i + 2048]]
        i0 += len(idx)
    o.flush()
    del o
    rs = np.random.default_rng(0).choice(n, 32, replace=False)
    Fo = np.load(dw / "front.npy", mmap_mode="r")
    starts = np.cumsum([0] + [int(p[3].sum()) for p in parts])
    chk = []
    for i in rs:
        b = int(np.searchsorted(starts, i, side="right") - 1)
        chk.append(bool(np.array_equal(Fo[i], FR[b][np.flatnonzero(parts[b][3])[i - starts[b]]])))
    assert all(chk), "front copy"
    tv = splits.load("navsim/selfiter_tv")
    same = set(tv.members) == set(tb["names"].astype(str).tolist())
    rec = dict(pool=POOL, rows=n, logs=len(set(tb["log"].tolist())), route_feat_ok=float(np.concatenate([z["ok"][p[3]] for z, p in zip(FZ, parts)]).mean()),
               classes=dict(zip(*[x.tolist() for x in np.unique(cls, return_counts=True)])), standing=int(R["stand"][J].sum()), launch=int(R["launch"][J].sum()),
               standing_reasoned=int((R["stand"][J] & (fl["rlead"] | fl["rsign"] | fl["rred"])[J]).sum()),
               reasons_on_standing={k: int((R["stand"][J] & fl[k][J]).sum()) for k in fl}, split_selfiter_tv_equal=same, front_check_32=all(chk),
               k_tv=k_tv(n))
    write("pack_NR1TV", rec)


def SI_split(b):
    return f"selfiter_tr{b}"


def k_tv(n_tv):
    return min(int(round(NN0_BASE * n_tv / (NAV_TRAIN + n_tv))), NN0_BASE // 2)


# ---------------------------------------------------------------- weights (op-train, CPU)
def nav_class_shares():
    """navtrain rows (every navtrain_full shard row with a full future) that are neither standing nor launching (v0 < 0.3 and 2 s / 4 s
    displacement > 0.5 / 2 m): lib/self_iter class shares."""
    G, V = [], []
    for i in range(12):
        z = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        ok = np.isfinite(z["fut"]).all((1, 2))
        G.append(z["fut"][ok].astype(np.float64))
        V.append(z["speed"][ok].astype(np.float64))
    G, V = np.concatenate(G), np.concatenate(V)
    d4, d2 = np.hypot(G[:, 7, 0], G[:, 7, 1]), np.hypot(G[:, 3, 0], G[:, 3, 1])
    stand, launch = (V < V_STAND) & (d4 < D_STAND), (V < V_STAND) & (d2 > 0.5) & (d4 > 2.0)
    c = SI.classes(G, V)[~stand & ~launch]
    u, k = np.unique(c, return_counts=True)
    return {str(x): float(y / k.sum()) for x, y in zip(u, k)}, dict(rows=int(len(V)), standing=int(stand.sum()), launching=int(launch.sum()))


def cmd_weights(a):
    tb = np.load(CR / POOL / "tab.npz")
    n = len(tb["names"])
    one = lambda k: tb[k].astype(str) == "1"  # noqa: E731
    stand, launch = one("stand"), one("launch")
    if a.line_max != 10.0:                                            # a different line distance: recompute the line reasons from reasons.npz
        R = batch_rows()
        fl = reason_flags(R, a.line_max)
        pos = {t: i for i, t in enumerate(R["token"].astype(str).tolist())}
        j = np.array([pos[t[3:]] for t in tb["names"].astype(str).tolist()])
        reason = fl["rlead"][j] | fl["rsign"][j] | fl["rred"][j]
    else:
        reason = one("rlead") | one("rsign") | one("rred")
    cls = tb["cls"].astype(str)
    w = np.zeros(n)
    rec = dict(pool=POOL, rows=n, rule=a.rule, wait=a.wait, launch=a.launch, line_max=a.line_max)
    if a.rule == "all":
        w[:] = 1.0 / n
        grp = np.where(stand, "stand", np.where(launch, "launch", "c_" + cls))
    else:
        sr, su = stand & reason, stand & ~reason
        lr = launch & ~stand
        other = ~stand & ~lr
        assert sr.any() and lr.any(), "no reasoned standing / launch row"
        w[sr] = a.wait / sr.sum()
        w[lr] = a.launch / lr.sum()
        nav, navrec = nav_class_shares()
        present = sorted(set(cls[other].tolist()) & set(nav))
        tot = sum(nav[c] for c in present)
        rest = 1.0 - a.wait - a.launch
        for c in present:
            m = other & (cls == c)
            w[m] = rest * nav[c] / tot / m.sum()
        grp = np.where(sr, "stand_r", np.where(su, "stand_u", np.where(lr, "launch", "c_" + cls)))
        rec |= dict(nav_class_shares=nav, nav_rows=navrec, classes_used=present, rows_by_group={g: int((grp == g).sum()) for g in np.unique(grp)})
    w = w / w.sum()
    rec["share"] = {g: float(w[grp == g].sum()) for g in np.unique(grp)}
    rec["per_row_vs_uniform"] = {g: float(w[grp == g].mean() * n) for g in np.unique(grp)}
    kt = k_tv(n)
    rec |= dict(k_tv=kt, draws_10k={g: float(w[grp == g].sum() * kt * 10000) for g in np.unique(grp)},
                repeats_10k={g: float(w[grp == g].sum() * kt * 10000 / max((grp == g).sum(), 1)) for g in np.unique(grp)})
    (OUT / "w").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "w" / f"{a.out}.npz", names=tb["names"], w=w, grp=grp.astype("U12"))
    write(f"w_{a.out}", rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("reasons").add_argument("--workers", type=int, default=48)
    sp.add_parser("pack")
    p = sp.add_parser("weights")
    p.add_argument("--wait", type=float, default=0.05), p.add_argument("--launch", type=float, default=0.05)
    p.add_argument("--rule", default="reasoned", choices=("reasoned", "all")), p.add_argument("--line-max", type=float, default=10.0)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(reasons=cmd_reasons, pack=cmd_pack, weights=cmd_weights)[a.cmd](a)
