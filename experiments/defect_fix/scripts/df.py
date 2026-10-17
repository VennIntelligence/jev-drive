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


# ---------------------------------------------------------------- identity, gates and summaries (CPU)
V2B, V2, RSG = "RTB1AR-F-s0", "RTB1ARO-L18-s0", "RSG-L18-s0"
ADE_TOL, G_MIN, W_LINE, ARR_MAX, LAUNCH_TOL = 0.01, 0.9, 0.6, 2.0, 0.10
CONT_PUB_LO, CONT_HELD_ZEROS = -0.01, 3                     # a base dose's plain continuation against V2's (prereg 4.1)


def cmd_bitcmp(a):
    """nx.py bitcmp's comparison (every scalar, every weight, dev ADE, on_hash_60 [= --hash]) -> results/<out>.json; exit 2 unless identical.
    --extra-ok: run b may log scalar keys a does not (monitoring-only logging, rm_train --diag); every key of a is compared."""
    (sa, da, ma), (sb, db, mb) = wt.run_of(a.a), wt.run_of(a.b)
    keys = [k for k in sorted(set(sa) & set(sb), key=str) if not k[0].startswith("throughput/")]
    dl = max(abs(sa[k] - sb[k]) for k in keys)
    same = all(set(ma[g] or {}) == set(mb[g] or {}) for g in ("net", "parity"))
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g]) if same else float("inf")
    ka, kb = set(k[0] for k in sa), set(k[0] for k in sb)
    r = dict(a=a.a, b=a.b, note=a.note, scalars=len(keys), scalar_keys_equal=ka <= kb if a.extra_ok else ka == kb, scalar_max_abs_diff=dl,
             extra_keys_b=sorted(kb - ka) if a.extra_ok else None,
             weight_max_abs_diff=dw, weight_keys_equal=same, dev_ade=[da.get("dev_ade"), db.get("dev_ade")], on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")],
             want_hash=a.hash or None)
    r["identical"] = bool(dl == 0 and dw == 0 and same and r["scalar_keys_equal"] and r["dev_ade"][0] == r["dev_ade"][1]
                          and r["on_hash_60"][0] == r["on_hash_60"][1] and (not a.hash or r["on_hash_60"][0] == a.hash))
    write(a.out, r)
    raise SystemExit(0 if r["identical"] else 2)


def launch_bias(tag):
    """4 s arc bias (plan - human, m) on the gate-4 dev rows of the v0 0-0.5 / free-road cell (wt_devdecomp's context rule; nx.py hist s = 1)."""
    NX = DATA / "runs/wait_cap/next"
    z, h = np.load(NX / "devdecomp/devall.npz"), np.load(NX / "hist" / f"{tag}.npz")
    assert np.array_equal(h["rows"], z["rows"]), f"{tag}: hist rows != devall rows"
    arc = lambda P: np.cumsum(np.hypot(*np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[..., :2]], 1), axis=1).transpose(2, 0, 1)), 1)  # noqa: E731
    v0 = z["speed"].astype(np.float64)
    near = z["lead_gap"] < 20
    closing = v0 + z["lead_close"]
    lowttc = near & (closing > 0.1) & (z["lead_gap"] / np.maximum(closing, 1e-3) < 4)
    free = z["ctx_ok"] & ~lowttc & ~near & ~z["ped"].astype(bool)
    m = (v0 >= 0) & (v0 < 0.5) & free
    b = arc(h["plan_1.0"].astype(np.float64))[:, 7] - arc(z["fut"].astype(np.float64))[:, 7]
    return dict(rows=int(m.sum()), bias=float(b[m].mean()))


def base_reads(tag):
    """Open-loop reads of one base: dev ADE (DONE), W / G (d326 instrument, wc.py plans), arrival replay 4 s arc at k = 1, launch-cell bias."""
    import nx
    cap, _ = wt.cap_read(tag)
    hist = DATA / "runs/wait_cap/next/hist" / f"{tag}.npz"                 # amendment 2: the bases' hist read was cut (selection does not use it)
    return dict(dev_ade=wt.dev_ade(tag), wait_W=cap["wait_W"], go_G=cap["go_G"], D=cap["D"], into_WL=cap["into_WL"], stop_SL=cap["stop_SL"],
                arrival_k1=nx.arrival_of(tag)[1]["arc4"], launch=launch_bias(tag) if hist.exists() else dict(rows=0, bias=float("nan")))


def cand_lines(r, ref_ade):
    return dict(dev_ade=r["dev_ade"] <= ref_ade + ADE_TOL, go_G=r["go_G"] >= G_MIN, arrival=r["arrival_k1"] <= ARR_MAX)


def cmd_candgate(a):
    """One base: the open-loop candidate lines (prereg 4.1); exit 3 if it is not a candidate (its continuation hangs on this job)."""
    r = base_reads(a.j)
    ref = wt.dev_ade(V2B)
    g = cand_lines(r, ref)
    r |= dict(lines=g, candidate=bool(all(g.values())), ref_dev_ade=ref, W_line=r["wait_W"] >= W_LINE, launch_line=abs(r["launch"]["bias"]) <= LAUNCH_TOL)
    write(f"candgate_{a.j}", r)
    raise SystemExit(0 if r["candidate"] else 3)


_NR_READ_DIRS = None


def read_dirs(tag):
    """held 460 @18 and public 1491 @10 run dirs of a tag: this topic's public runs first, else nx_regress's locations (its function captured
    before the summaries point nx_regress at this one)."""
    import glob
    import nx_regress as NR
    global _NR_READ_DIRS
    if _NR_READ_DIRS is None:
        _NR_READ_DIRS = NR.read_dirs
    pub = glob.glob(str(OUT / "pub" / tag / "b*")) or _NR_READ_DIRS(tag)["pub"]
    return dict(held=glob.glob(wt.xdirs("held", tag)), pub=pub)


def lat_zeros(dirs):
    """Zero scenes whose failure is lateral (offroad, corridor or an at-fault side contact), nx_regress.failure_type."""
    import nx_regress as NR
    return int(sum(1 for r in NR.rollouts(dirs).values() if float(r.get("score") or 0) == 0
                   and any(k in NR.failure_type(r) for k in ("offroad", "corridor", "AF-side"))))


WS = DATA / "runs/alpasim_proxy/wait_set/reads"


WS_SET = REPO / "experiments/alpasim_proxy/results/wait_set/set.txt"     # the current set (v2: 312 windows; route-fold windows dropped)


def ws_rows(tag):
    """One read's rows, restricted to the windows of the current set file (reads made on the 316-window v1 list keep 4 route-fold zeros otherwise)."""
    import csv
    f = WS / tag / "table" / f"{tag}.csv"
    if not f.exists():
        return None
    keep = set(WS_SET.read_text().split())
    return [x for x in csv.DictReader(open(f)) if x["id"] in keep]


def ws_read(tag, ref=None):
    """The closed-loop waiting set (alpasim_nuplan/waitset@v1; experiments/alpasim_proxy wait_set.py collect): variant a (the human never
    moves) zeros / wait success, variant b (waits, launches at 3.0 s) launch delay (never = censored at the run's end), on all windows and on
    the held-split subset; against `ref` the paired a-zero difference (session-clustered CI) and the median-delay difference (prereg 5)."""
    from jevdrive import stats
    R = ws_rows(tag)
    if R is None:
        return None
    f = lambda x, k: float(x[k]) if x.get(k) not in (None, "", "None") else None  # noqa: E731

    def summ(rows):
        a = [x for x in rows if x["variant"] == "a"]
        b = [x for x in rows if x["variant"] == "b"]
        d = [(f(x, "delay_s") if f(x, "delay_s") is not None else (f(x, "t_end") or 0.0) - (f(x, "h_launch_s") or float("nan"))) for x in b]
        d = [v for v in d if v is not None and np.isfinite(v)]                # windows without a logged human launch carry NaN (df_wscollect.py)
        return dict(a_n=len(a), a_zeros=sum(x["zero"] not in ("", "None") for x in a), a_wait_ok=float(np.mean([x["wait_ok"] == "True" for x in a])) if a else None,
                    a_score=float(np.mean([float(x["score"]) for x in a])) if a else None, b_n=len(b),
                    b_never=sum(f(x, "launch_s") is None for x in b), b_delay_med=float(np.median(d)) if d else None,
                    b_score=float(np.mean([float(x["score"]) for x in b])) if b else None)
    out = dict(all=summ(R), held=summ([x for x in R if x["split"] == "held"]))
    if ref is not None and ws_rows(ref) is not None:
        Q = {x["id"]: x for x in ws_rows(ref)}
        for part, keep in (("all", lambda x: True), ("held", lambda x: x["split"] == "held")):
            a = [(x, Q[x["id"]]) for x in R if x["variant"] == "a" and x["id"] in Q and keep(x)]
            if a:
                zj = np.array([x["zero"] not in ("", "None") for x, _ in a], float)
                zr = np.array([y["zero"] not in ("", "None") for _, y in a], float)
                p = stats.paired(zj, zr, groups=np.array(["_".join(x["src"].split("_")[:2]) for x, _ in a]))
                out[part]["vs_ref"] = dict(ref=ref, a_zero_diff=float(zj.sum() - zr.sum()), a_zero_diff_lo=float(p["lo"] * len(a)), a_zero_diff_hi=float(p["hi"] * len(a)),
                                           b_delay_med_diff=None if out[part]["b_delay_med"] is None else out[part]["b_delay_med"] - summ([y for y in Q.values()
                                                                                                                                     if keep(y)])["b_delay_med"])
    return out


def fmt_ws(tag, w):
    if w is None:
        return [f"| {tag} | - | - | - | - | - |"]
    s = lambda p: (f"{w[p]['a_zeros']} / {w[p]['a_n']} (wait ok {w[p]['a_wait_ok']:.2f}) | {w[p]['b_delay_med']:+.2f} s, never {w[p]['b_never']} / {w[p]['b_n']}"  # noqa: E731
                   if w[p]["a_n"] and w[p]["b_n"] else "-")
    v = lambda p: ("" if "vs_ref" not in w[p] else f"{w[p]['vs_ref']['a_zero_diff']:+.0f} [{w[p]['vs_ref']['a_zero_diff_lo']:+.1f}, {w[p]['vs_ref']['a_zero_diff_hi']:+.1f}]"  # noqa: E731
                   f", delay {w[p]['vs_ref']['b_delay_med_diff']:+.2f} s")
    return [f"| {tag} | {s('all')} | {s('held')} | {v('all')} | {v('held')} |"]


WS_HEAD = ["| tag | a zeros, b median delay (set v2, 312) | the same, held-split | vs RSG: a-zero diff [CI], delay diff (all) | (held-split) |", "|---|---|---|---|---|"]


def cl_read(tag, refs):
    """Closed-loop summary of one tag on held 460 @18 and public 1491 @10 against reference tags (paired, session-clustered)."""
    import nx
    import nx_regress as NR
    D = read_dirs(tag)
    out = {}
    for rd in ("held", "pub"):
        if not D[rd]:
            out[rd] = None
            continue
        M = wt.metrics(D[rd])
        r = dict(**NR.own(M), coll=nx.coll_counts(D[rd]), **NR.creep(NR.rollouts(D[rd])), lat_zeros=lat_zeros(D[rd]), pair={})
        for c in refs:
            Dc = read_dirs(c)[rd]
            if Dc:
                Mc = wt.metrics(Dc)
                x = wt.xpair(M, Mc)
                r["pair"][c] = dict(**nx.pair_ci(M, Mc), zeros=x["J"]["zeros"], zeros_ref=x["C"]["zeros"], scenes=x["scenes"])
        out[rd] = r
    st = OUT / "static" / tag / "run"
    out["static"] = nx.static_of(st) if st.exists() else None
    sp = OUT / "paisub" / tag
    out["stop17"] = nx.stop17_of(sp) if sp.exists() else None
    try:
        cap, _ = wt.cap_read(tag)
        out["W"], out["G"] = cap["wait_W"], cap["go_G"]
    except FileNotFoundError:
        out["W"] = out["G"] = None
    return out


def fmt_cl(tag, r, refs):
    L = []
    for rd in ("held", "pub"):
        x = r[rd]
        if x is None:
            continue
        pr = "; ".join(f"vs {c} {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] (zeros {p['zeros']} / {p['zeros_ref']})" for c, p in x["pair"].items())
        L.append(f"| {tag} | {rd} | {x['scenes']} | {x['zeros']} | {x['mean']:.4f} | {x['at_fault']} | {x['offroad']} | {x['lat_zeros']} | "
                 f"{x['coll']['front']} / {x['coll']['lateral']} / {x['coll']['rear']} | {x['progress_rel']:.3f} | {x['creep']} / {x['no_launch']} | {pr} |")
    return L


CL_HEAD = ["| tag | read | scenes | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired |",
           "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]


def cmd_basesum(a):
    """The three bases: open-loop lines, their plain continuations' closed loop against V2, the selection rule (prereg 4.1)."""
    import nx_regress as NR  # noqa: F401
    ref = wt.dev_ade(V2B)
    res = dict(ref=V2B, ref_dev_ade=ref, bases={}, conts={})
    rb = base_reads(V2B) | dict(lines=None)
    L = ["# NR1 base pilots: open-loop lines (C = %s, dev ADE %.4f) and the plain L18 continuations against %s" % (V2B, ref, V2), "",
         "| base | dose | dev ADE (<= %.4f) | W (pilot line >= %.1f) | G (>= %.1f) | D | into W_L | S_L stop | arrival 4 s arc k=1 (<= %.1f) | launch-cell 4 s bias (+-%.2f) | candidate |"
         % (ref + ADE_TOL, W_LINE, G_MIN, ARR_MAX, LAUNCH_TOL), "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|",
         f"| {V2B} (C) | - | {rb['dev_ade']:.4f} | {rb['wait_W']:.3f} | {rb['go_G']:.3f} | {rb['D']:.3f} | {rb['into_WL']:.3f} | {rb['stop_SL']:.3f} | "
         f"{rb['arrival_k1']:.2f} | {rb['launch']['bias']:+.2f} | - |"]
    for spec in a.bases:
        dose, b, c = spec.split(":")
        try:
            r = base_reads(b)
        except (FileNotFoundError, IndexError) as e:
            res["bases"][b] = dict(error=str(e))
            continue
        g = cand_lines(r, ref)
        r |= dict(dose=dose, lines=g, candidate=bool(all(g.values())), cont=c)
        res["bases"][b] = r
        f = lambda k: "" if g.get(k, True) else " x"  # noqa: E731
        L.append(f"| {b} | {dose} | {r['dev_ade']:.4f}{f('dev_ade')} | {r['wait_W']:.3f} | {r['go_G']:.3f}{f('go_G')} | {r['D']:.3f} | {r['into_WL']:.3f} | "
                 f"{r['stop_SL']:.3f} | {r['arrival_k1']:.2f}{f('arrival')} | {r['launch']['bias']:+.2f} | {'yes' if r['candidate'] else 'no'} |")
    L += ["", "## Plain continuations (V2's L18 command from each candidate base), closed loop", ""] + CL_HEAD
    rv = cl_read(V2, [RSG])
    L += fmt_cl(V2 + " (V2)", rv, [RSG])
    adm = []
    for b, r in res["bases"].items():
        if not r.get("candidate"):
            continue
        c = r["cont"]
        x = cl_read(c, [V2, RSG])
        res["conts"][c] = x
        L += fmt_cl(c, x, [V2, RSG])
        ok = bool(x["held"] and x["held"]["zeros"] <= rv["held"]["zeros"] + CONT_HELD_ZEROS)     # amendment 2: held only (no public at the pilot stage)
        r["closed_loop_admissible"] = ok
        if ok:
            adm.append(b)
    L += ["", "## Waiting set (closed loop, alpasim_nuplan/waitset@v1; base checkpoints and their continuations)", ""] + WS_HEAD
    W = {t: ws_read(t, RSG) for t in [V2B, V2, RSG] + list(res["bases"]) + [r["cont"] for r in res["bases"].values() if r.get("cont")]}
    for t, w in W.items():
        L += fmt_ws(t, w)
    res["waitset"] = W
    wv = W.get(V2)
    sel = []
    for b in adm:                                                     # prereg 4.1: b-delay within 0.5 s of V2's continuation; fewest a zeros
        w = W.get(res["bases"][b]["cont"])
        if w is not None and wv is not None and w["all"]["b_delay_med"] <= wv["all"]["b_delay_med"] + 0.5:
            sel.append(b)
    if sel:
        pick = min(sel, key=lambda b: (W[res["bases"][b]["cont"]]["all"]["a_zeros"], -res["bases"][b]["wait_W"]))
        rule = "fewest variant-a zeros of the continuation on the waiting set among the admissible bases with variant-b median delay <= V2's + 0.5 s"
    else:
        pick = max(adm, key=lambda b: (res["bases"][b]["wait_W"], -res["bases"][b]["dev_ade"])) if adm else None
        rule = "no waiting-set read usable: highest W among the closed-loop-admissible candidates"
    res |= dict(admissible=adm, waitset_eligible=sel, pick=pick, rule=rule)
    L += ["", f"closed-loop admissible (amendment 2: held zeros <= V2 continuation + {CONT_HELD_ZEROS}): {adm or 'none'}",
          f"waiting-set eligible (b median delay <= V2 + 0.5 s): {sel or 'none'}", f"rule: {rule}",
          f"pick: {pick or 'none -> the full chain starts from ' + V2B}"]
    write("base_pilot", res, L)


def cmd_pilotsum(a):
    """The L18 joint-term pilots against their 1 000-step control: held / public, lateral zeros, the hinge rule (prereg 4.2), W / G."""
    import nx
    import nx_regress as NR
    C = a.control
    res = dict(control=C, arms={})
    tags = [C] + a.j
    R = {t: cl_read(t, [C, V2, RSG] if t != C else [V2, RSG]) for t in tags}
    L = [f"# NR1 L18 pilots (1 000 steps from {V2B}, merged pool on1_L18NR1, history-speed noise on) against the control {C}", ""] + CL_HEAD
    for t in tags:
        L += fmt_cl(t, R[t], [C, V2, RSG] if t != C else [V2, RSG])
    L += ["", "| tag | W | G | static scene | held lateral zeros |", "|---|--:|--:|---|--:|"]
    for t in tags:
        s = R[t]["static"]
        L.append(f"| {t} | {R[t]['W'] if R[t]['W'] is None else round(R[t]['W'], 3)} | {R[t]['G'] if R[t]['G'] is None else round(R[t]['G'], 3)} | "
                 f"{'-' if s is None else '%.2f m, %s' % (s['dist'], 'contact' if s['collision'] else 'no contact')} | {R[t]['held']['lat_zeros']} |")
    L += ["", "## Waiting set", ""] + WS_HEAD
    W = {t: ws_read(t, RSG) for t in [RSG, V2] + tags}
    for t, w in W.items():
        L += fmt_ws(t, w)
    res["waitset"] = W
    passing = []
    for t in a.j:
        r, c = R[t], R[C]
        g = dict(held_zeros=r["held"]["zeros"] <= c["held"]["zeros"], held_lateral=r["held"]["lat_zeros"] < c["held"]["lat_zeros"])   # amendment 2: held only
        res["arms"][t] = dict(read=r, lines=g, passed=bool(all(g.values())))
        if all(g.values()):
            passing.append(t)
    pick = min(passing, key=lambda t: (R[t]["held"]["lat_zeros"], R[t]["held"]["zeros"], -R[t]["held"]["mean"])) if passing else None
    res |= dict(control_read=R[C], passing=passing, pick=pick)
    L += ["", "hinge rule (amendment 2, vs the control): held zeros <= control and held lateral zeros < control (fewest held lateral zeros among the passing)",
          "; ".join(f"{t}: {res['arms'][t]['lines']}" for t in a.j), f"pick: {pick or 'none -> the full chain runs without the lateral terms'}"]
    nx.RES, nx.NX = RES, OUT                                          # nx_regress writes through nx.write: into this topic
    NR.read_dirs = read_dirs                                          # and reads this topic's public runs
    write(a.out, res, L)
    NR.cmd_table(argparse.Namespace(tags=[f"RSG={RSG}", f"V2={V2}"] + [f"{t.split('-')[0]}={t}" for t in tags], ref=C.split("-")[0], out=f"{a.out}_regress", dry=False))


def cmd_fullsum(a):
    """The full chain against the registered bar (prereg 5): both seeds held / public / static / stop17 / W / G, seed 0 PAI 281."""
    import nx
    import nx_regress as NR
    res, L = dict(tags=a.j, lines={}), [f"# NR1 full chain: {', '.join(a.j)} against {RSG}", ""] + CL_HEAD
    R = {t: cl_read(t, [RSG, V2]) for t in a.j}
    for t in a.j:
        L += fmt_cl(t, R[t], [RSG, V2])
    ok = {}
    for t in a.j:
        h, p, s = R[t]["held"]["pair"][RSG], R[t]["pub"], R[t]["static"]
        rsg_rear = cl_read(RSG, [])["pub"]["coll"]["rear"]
        ok[t] = dict(held_zeros=h["zeros"] <= h["zeros_ref"] - 5, held_ci=h["lo"] > 0, pub_ci=p["pair"][RSG]["lo"] >= -0.005, pub_rear=p["coll"]["rear"] <= rsg_rear,
                     static=bool(s is not None and not s["collision"] and s["dist"] <= 1.0), W=bool(R[t]["W"] is not None and R[t]["W"] >= W_LINE),
                     G=bool(R[t]["G"] is not None and R[t]["G"] >= G_MIN))
    L += ["", "## Waiting set", ""] + WS_HEAD
    W = {t: ws_read(t, RSG) for t in [RSG, V2] + a.j}
    for t, w in W.items():
        L += fmt_ws(t, w)
    res["waitset"] = W
    for t in a.j:                                                     # the proposed line: a zeros clearly below RSG on all and held-split, b delay
        w = W.get(t)
        ok[t]["waitset (proposed)"] = bool(w is not None and all("vs_ref" in w[p] and w[p]["vs_ref"]["a_zero_diff_hi"] < 0 for p in ("all", "held"))
                                           and w["all"]["vs_ref"]["b_delay_med_diff"] <= 0.5)
    if a.pai:
        import glob
        assert nx.pai_complete(OUT / "pai" / a.pai), f"PAI 281 read of {a.pai} incomplete (runs/defect_fix/pai/{a.pai})"
        mine, ref = wt.metrics(glob.glob(str(OUT / "pai" / a.pai / "runs/*/sim"))), wt.metrics(glob.glob(str(wt.PAI_REF / "runs/*/sim")))
        x = wt.xpair(mine, ref)
        pc = nx.pair_ci(mine, ref)
        res["pai"] = dict(tag=a.pai, pair=pc, zeros=x["J"]["zeros"], zeros_ref=x["C"]["zeros"], at_fault=x["J"]["at_fault"], at_fault_ref=x["C"]["at_fault"])
        ok["PAI " + a.pai] = dict(paired=pc["mean"] >= 0, zeros=x["J"]["zeros"] <= x["C"]["zeros"] + 2)
    res |= dict(reads=R, lines=ok, bar_met=bool(all(all(v.values()) for v in ok.values())))
    L += ["", "| tag | line | met |", "|---|---|:-:|"] + [f"| {t} | {k} | {'yes' if v else 'no'} |" for t, d in ok.items() for k, v in d.items()]
    if "pai" in res:
        p = res["pai"]
        L.append(f"\nPAI 281 ({p['tag']}, submission config) vs RSG-L18-s0: {p['pair']['mean']:+.4f} [{p['pair']['lo']:+.4f}, {p['pair']['hi']:+.4f}], zeros {p['zeros']} / {p['zeros_ref']}, at-fault {p['at_fault']} / {p['at_fault_ref']}")
    L.append(f"\nbar 'clearly better than RSG-L18-s0' met: {'yes' if res['bar_met'] else 'no'}")
    nx.RES, nx.NX = RES, OUT
    NR.read_dirs = read_dirs
    write(a.out, res, L)                                                # a second chain (TAGP) names its own files
    NR.cmd_table(argparse.Namespace(tags=[f"RSG={RSG}", f"V2={V2}"] + [f"N{i}={t}" for i, t in enumerate(a.j)], ref="V2", out=f"{a.out}_regress", dry=False))


# ---------------------------------------------------------------- V2's plans on the L18 pool rows (GPU), for the label gate
def cmd_v2plans(a):
    """A checkpoint's plans on every row of a pool (route memory fed from feat/<pool>.npz, as served) -> runs/defect_fix/v2plans/<tag>_<pool>.npz."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from types import SimpleNamespace
    sys.path[:0] = [str(REPO / "experiments/route_mem/scripts"), str(REPO / "experiments/route_follow/scripts")]
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    dev = torch.device("cuda")
    ns, _, _, _, _ = RT.build(SimpleNamespace(**(REC | dict(on=a.pool, on_mode="self")), road_labels=RT.OT.JT.ROAD, init="", rm_from=a.tag, tag=a.tag,
                                              exp="defect_fix", lr_scale=1.0, prefetch=1, dev_card_gb=4.0), dev)
    S, model, W = ns.S, ns.model, ns.losses(ns.mode).W
    model.eval()
    pi = torch.as_tensor(S.pi, device=dev)
    rows = np.arange(ns.on0, S.n)
    P = np.zeros((len(rows), 8, 3), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            r = rows[i:i + 512]
            rt = torch.as_tensor(r, device=dev)
            p = model(S.front[r], S.ego[rt], S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()[:, pi].view(-1, 33, 15)
            P[i:i + len(r)] = torch.stack(T.rear(p, S.cam_x[rt], W), -1).cpu().numpy()
    (OUT / "v2plans").mkdir(parents=True, exist_ok=True)
    f = OUT / "v2plans" / f"{a.tag}_{a.pool}.npz"
    np.savez(f, names=S.tab["names"][rows], poses=P, tag=np.array(a.tag), route_fed=np.array(True))
    print(f, len(rows), "rows; memory columns fed:", int((S.ego[torch.as_tensor(rows[:4096], device=dev), 20:].abs().sum(1) > 0).sum()), "/ 4096 nonzero")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("v2plans")
    p.add_argument("--tag", default="RTB1ARO-L18-s0"), p.add_argument("--pool", default="on1_L18NR1")
    p = sp.add_parser("bitcmp")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--out", required=True)
    p.add_argument("--hash", default=""), p.add_argument("--note", default=""), p.add_argument("--extra-ok", action="store_true")
    sp.add_parser("candgate").add_argument("--j", required=True)
    sp.add_parser("basesum").add_argument("--bases", nargs="+", required=True, help="dose:base:continuation")
    p = sp.add_parser("pilotsum")
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--control", required=True), p.add_argument("--out", default="l18_pilot")
    p = sp.add_parser("fullsum")
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--pai", default=""), p.add_argument("--out", default="full_chain")
    sp.add_parser("reasons").add_argument("--workers", type=int, default=48)
    sp.add_parser("pack")
    p = sp.add_parser("weights")
    p.add_argument("--wait", type=float, default=0.05), p.add_argument("--launch", type=float, default=0.05)
    p.add_argument("--rule", default="reasoned", choices=("reasoned", "all")), p.add_argument("--line-max", type=float, default=10.0)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(reasons=cmd_reasons, pack=cmd_pack, weights=cmd_weights, v2plans=cmd_v2plans, bitcmp=cmd_bitcmp, candgate=cmd_candgate, basesum=cmd_basesum,
         pilotsum=cmd_pilotsum, fullsum=cmd_fullsum)[a.cmd](a)
