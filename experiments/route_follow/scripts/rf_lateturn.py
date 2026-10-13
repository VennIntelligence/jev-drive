#!/usr/bin/env python3
"""LATE-TURNS (research/route-follow/late-turns.md): the held 24 turns all begin at the clip start, so they do not test whether a policy decides
to turn by itself. This script builds a policy-free set of LATE turns, FORKS (the route leaves the ego lane's straight successor) and EARLY turns
on nuPlan (onpol-train 1031 + held 460) and on PAI (the 36 RSG strict turns), then reads completion by class from @18 closed-loop reads.

  scan    (box, AlpaSim venv)  rf_lateturn.py scan --held FILE --out CSV [--workers N] [--limit N]
          every onpol-train and held scene: the route is the generator's polyline after decisions anchored on the LOGGED ego poses
          (rf_trainread.screen), the window = the log's travel to decision 17 [s0, s_end].
  pick    (Mac)  rf_lateturn.py pick --scan CSV --done FILE --cap 80 --seed 0 --out FILE   late / delayed / fork train scenes not yet read
  scores  (box)  rf_lateturn.py scores --runs MODEL=<dirs> ... --out JSON      score + zero kind per model x scene
  shape   (box, AlpaSim venv)  rf_lateturn.py shape --set LABEL=<rf_tax pkl> ... --scan CSV --out CSV    rf_ledger turn shape per class rollout
  report  (box or Mac)  rf_lateturn.py report --scan CSV --follow DIR [DIR ..] --scores JSON --shape CSV --out DIR
  page-pick / page-subset (box) / page-build (Mac)   the review folder late-turns/ (tmp/held_turns's page: BEV clips + the camera rerun)
  pai     (Mac)  rf_lateturn.py pai --pkl <rf_pai_follow pkl> --out DIR    class of the 36 PAI strict turns, RSG / RBSG2 completion by class

Classes (one scene can be late and fork):
  turn     rf_follow's turn rule on the log-anchored route (30 m window with net heading change >= 60 deg, change over [a - 10, a + 40] m
           >= 45 deg, entry / exit at 10 / 90 %), the first such turn starting at or after s0 with entry <= s_end. strict = exit + 10 m <=
           s_end (rf_follow's own set); half = the log is >= 5 m past the turn's middle at decision 17 ((entry + exit) / 2 + 5 <= s_end: the
           human is half way round). Turn = strict or half. Target station = exit + 10 m (strict) or s_end - 5 m (half): an @18 read ends
           at decision 17, so a turn starting 40 m in is only half inside the read for most speeds. entry_rel = entry - s0. A first turn
           whose middle the log does not pass is recorded as beyond_entry_rel (untestable @18, no class).
  late     turn with entry_rel >= 40 m (the ego drives straight >= 40 m first);  early = turn with entry_rel < 40 m; delayed = the early
           turns with entry_rel >= 20 m (reported as their own bin: @18 windows are ~47 m long, so late turns barely exist on nuPlan).
  fork     the ego lane = the route's lane at s0 (the route is the log on its lanes); its straight chain = from the ego's point, at every lane end
           the successor with the smallest heading change (over its first 15 m), out to s_end - s0 + 30 m. Route separation d(q) = distance of
           the route point at station q to that chain; s_sep = the first q in [s0, s_end] with d >= 3 m (going straight there leaves the
           2.5 m completion band), s_dep = the last q before it with d < 1 m. fork = s_sep exists, s_dep - s0 >= 5 m, the chain reaches s_sep,
           and the route really leaves the chain's road: max d over [s_sep, s_end] >= 5 m or the route heading at s_end differs from the
           chain's by >= 15 deg (else lane_offset: a parallel one-lane offset, a plain lane change or a lane choice inside a turn).
           kind (label only) = branch (the route's lane at s_sep descends within 3 successor steps from a chain lane: a fork or junction
           connector), lc (a left / right neighbour of a chain lane or its successor: lane change before the split), other (no lane relation).
  none     neither turn nor fork.
Completion per rollout (rf_follow decisions: anchor stations on the rollout's route, the same stations as the log-anchored route inside the
log's travel): the ego crosses the target station (turns above; fork-only: min(s_sep + 10, s_end - 5) m, with s_sep <= s_end - 5 m) before
the first zero-score event, <= 2.5 m from the route there; else event / off / short exactly as rf_follow. For strict turns this is rf_follow's
own outcome (rf_outcome kept beside it as a check).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R_ = HERE.parents[2]
sys.path[:0] = [str(R_), str(R_ / "lib"), str(HERE), str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/route_mem/scripts"),
                str(R_ / "experiments/alpasim/scripts"), str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/anchor_offset/scripts"),
                str(R_ / "experiments/eyes_vs_head/scripts"), str(R_ / "experiments/alpasim_long/scripts")]
import rf_pai_follow as PF  # noqa: E402  (numpy-only geometry: lane_at, project, stations, heading, find_turn)

LATE, DELAYED, SEP, DEP, MIN_DEP, NEAR_LANE, OTHER_DEG, BIG_SEP, MARGIN = 40.0, 20.0, 3.0, 1.0, 5.0, 60.0, 15.0, 5.0, 5.0
LAT_OK, PAST = 2.5, 10.0
N_BOOT = 10_000
MODELS = ("RSG-L18-s0", "RSGboth-L18-s0", "RB0-F-s0")


# ---------------------------------------------------------------- geometry (shared by nuPlan and PAI)
def near_lanes(lanes, R, r=NEAR_LANE):
    """Lanes with a centre point within r m of the polyline R."""
    Rq = R[:: max(1, len(R) // 200)]
    out = {}
    for k, l in lanes.items():
        c = l["c"][:, :2].astype(float)
        if len(c) < 2:
            continue
        c_ = c[:: max(1, len(c) // 20)]
        if (((c_[:, None, :] - Rq[None]) ** 2).sum(-1) <= r * r).any():
            out[k] = l
    return out


def lane_head(l, d=15.0):
    """Heading of lane l at its start and at min(d, length) m in."""
    c = l["c"][:, :2].astype(float)
    S = PF.stations(c)
    s, h = PF.heading(c, S) if S[-1] > 2 * PF.HALF else (np.array([0.0]), np.array([np.arctan2(*(c[-1] - c[0])[::-1])]))
    return float(h[0]), float(np.interp(min(d, S[-1]), s, h))


def straight_chain(lanes, lid, p, length):
    """From lane lid's point nearest p: at every lane end take the successor with the smallest heading change -> (polyline, lane ids)."""
    c = lanes[lid]["c"][:, :2].astype(float)
    S = PF.stations(c)
    st, _ = PF.project(c, S, p)
    k = int(np.searchsorted(S, st))
    P, ids = np.r_[[[np.interp(st, S, c[:, 0]), np.interp(st, S, c[:, 1])]], c[k:]], [lid]
    while PF.stations(PF.dedup(P))[-1] < length:
        nx = [n for n in lanes[ids[-1]]["nx"] if n in lanes and n not in ids]
        if not nx:
            break
        Pd = PF.dedup(P)
        h_end = np.arctan2(*(Pd[-1] - Pd[max(0, len(Pd) - 4)])[::-1]) if len(Pd) > 1 else 0.0
        n = min(nx, key=lambda q: abs((lane_head(lanes[q])[1] - h_end + np.pi) % (2 * np.pi) - np.pi))
        P, ids = np.r_[P, lanes[n]["c"][:, :2].astype(float)], ids + [n]
    return PF.dedup(P), ids


def descends(lanes, lid, chain, steps=3):
    """Is lane lid reachable from a chain lane within `steps` successor steps (walking predecessors back)?"""
    front = {lid}
    for _ in range(steps + 1):
        if front & set(chain):
            return True
        front = {q for f in front if f in lanes for q in lanes[f]["pv"]}
    return False


def classify(lanes, R, s0, s_end, p0, psi0):
    """Route polyline R (local frame), window [s0, s_end], ego start point / heading -> class dict (turn + fork fields)."""
    R = PF.dedup(R)
    S = PF.stations(R)
    s, h = PF.heading(R, S)
    out = dict(s0=float(s0), s_end=float(s_end), win_m=float(s_end - s0), turn=False, fork=False, fork_kind="", entry=np.nan, exit=np.nan,
               entry_rel=np.nan, deg=0.0, side="", s_sep=np.nan, s_dep=np.nan, sep_rel=np.nan, dep_rel=np.nan, fork_side="", fork_deg=np.nan)
    t = next(iter(PF.all_turns(s, h, s0, min(s_end + PF.WIN + 10.0, S[-1]))), None)     # the first route turn reaching into the window
    out.update(turn_strict=False, target=np.nan, beyond_entry_rel=np.nan)
    if t is not None and t[0] <= s_end:
        strict, half = t[1] + PAST <= s_end, (t[0] + t[1]) / 2 + MARGIN <= s_end
        if strict or half:
            out.update(turn=True, turn_strict=strict, entry=t[0], exit=t[1], entry_rel=t[0] - s0, deg=float(np.degrees(t[2])),
                       side="L" if t[2] > 0 else "R", target=t[1] + PAST if strict else s_end - MARGIN,
                       entry_x=float(np.interp(t[0], S, R[:, 0])), entry_y=float(np.interp(t[0], S, R[:, 1])))
        else:
            out["beyond_entry_rel"] = t[0] - s0                                 # the log ends before half of this turn: untestable @18
    L = near_lanes(lanes, R[(S >= s0 - 20) & (S <= s_end + 40)] if ((S >= s0 - 20) & (S <= s_end + 40)).sum() > 1 else R)
    at = lambda q: np.array([np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])])  # noqa: E731
    hq = lambda q: float(np.interp(q, s, h))  # noqa: E731
    ego = PF.lane_at(L, at(s0 + 1.0), hq(s0 + 1.0)) or PF.lane_at(L, p0, psi0)
    out["ego_lane"] = ego or ""
    if ego is None:
        out["fork_kind"] = "no_lane"
        return out
    C, ids = straight_chain(L, ego, at(s0), s_end - s0 + 30.0)
    CS = PF.stations(C)
    out["chain_m"] = float(CS[-1])
    q = np.arange(s0, s_end + 1e-9, 1.0)
    d, cst, hint = np.empty(len(q)), np.empty(len(q)), None
    for i, x in enumerate(q):
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        cst[i], d[i] = PF.project(C, CS, at(x), lo, hi)
        hint = cst[i]
    ad = np.abs(d)
    j = np.flatnonzero(ad >= SEP)
    if not len(j):
        return out
    j = int(j[0])
    k = np.flatnonzero(ad[:j] < DEP)
    s_sep, s_dep = float(q[j]), float(q[int(k[-1])] + 1.0) if len(k) else float(q[0])
    out.update(s_sep=s_sep, s_dep=s_dep, sep_rel=s_sep - s0, dep_rel=s_dep - s0, fork_side="L" if d[j] > 0 else "R")
    if cst[j] >= CS[-1] - 1.0:
        out["fork_kind"] = "chain_end"
        return out
    rl = PF.lane_at(L, at(s_sep), hq(s_sep), prefer=())
    ch = set(ids)
    adj = {a for c_ in ids for a in L[c_]["al"] + L[c_]["ar"]}
    # heading of the route vs the chain at s_end (chain heading at the station the route's end projects to)
    Ch, Chh = PF.heading(C, CS)
    out["fork_deg"] = float(np.degrees((hq(s_end) - np.interp(cst[-1], Ch, Chh) + np.pi) % (2 * np.pi) - np.pi))
    out["max_sep"] = float(ad[j:].max())
    kind = ("branch" if rl is not None and rl not in ch and descends(L, rl, ids) else
            "lc" if rl is not None and (rl in adj or any(descends(L, rl, [a], 1) for a in adj)) else "other")
    real = out["max_sep"] >= BIG_SEP or abs(out["fork_deg"]) >= OTHER_DEG       # not a one-lane offset that runs parallel (lane choice, plain lc)
    out["fork_kind"] = kind if real else "lane_offset"
    out["fork"] = bool(real and out["dep_rel"] >= MIN_DEP and s_sep <= s_end - MARGIN)
    if out["fork"] and not out["turn"]:
        out["target"] = min(s_sep + PAST, s_end - MARGIN)
    return out


def cls_name(r):
    t = ("late" if r["entry_rel"] >= LATE else "early") if r["turn"] else ""
    return "+".join(x for x in (t, "fork" if r["fork"] else "") if x) or "none"


# ---------------------------------------------------------------- scan (box)
def scan_one(job):
    scene, split = job
    import ev as EV
    import rf_follow as F
    from ao_e1 import loader, project, stations
    try:
        ds = loader(str(EV.CFG)).get_data_source(scene)
        tr = ds.rig.trajectory
        P, Tt, Y = np.asarray(tr.positions, float), np.asarray(tr.timestamps_us, float), np.unwrap(np.asarray(tr.yaws, float))
        recs = [dict(k=k, t0=F.T0 + F.DT * k, anchor=(np.interp(F.T0 + F.DT * k, Tt, P[:, 0]), np.interp(F.T0 + F.DT * k, Tt, P[:, 1]),
                                                       np.interp(F.T0 + F.DT * k, Tt, Y))) for k in range(F.k_end(Tt) + 1)]
        _, A, _, Rl, log_end = F.regen_full(scene, recs, EV.CFG)
        Rl = Rl[np.r_[True, np.hypot(*np.diff(Rl, axis=0).T) > 1e-6]]
        S = stations(Rl)
        s0 = project(Rl, S, np.asarray(A[0][:2]))[0]
        s_end = project(Rl, S, log_end, s0 - 10.0, s0 + 150.0)[0]
        lanes = {str(l.id): dict(c=np.asarray(l.center.points, float)[:, [0, 1, 3]], nx=sorted(map(str, l.next_lanes)), pv=sorted(map(str, l.prev_lanes)),
                                 al=sorted(map(str, l.adj_lanes_left)), ar=sorted(map(str, l.adj_lanes_right))) for l in ds.map.lanes}
        r = classify(lanes, Rl, s0, s_end, np.asarray(A[0][:2]), float(A[0][2]))
        v0 = float(np.hypot(*np.diff(P[:2, :2], axis=0)[0]) / max((Tt[1] - Tt[0]) / 1e6, 1e-3))
        return dict(scene=scene, split=split, ok=True, v0=v0, **r)
    except Exception as e:  # noqa: BLE001
        return dict(scene=scene, split=split, ok=False, err=repr(e)[-200:])


def scan(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    sp = splits.load("alpasim_nuplan/onpol-train")
    held = Path(a.held).read_text().split()
    jobs = [(s, "train") for s in sorted(sp.members)] + [(s, "held") for s in held]
    jobs = jobs[: a.limit or None] if not a.limit else jobs[: a.limit // 2] + [(s, "held") for s in held[: a.limit // 2]]
    if a.only:   # long reads (late-turns-long): only these scenes; RF_K_END / EV_CFG select the window and the cache
        keep = set(Path(a.only).read_text().split())
        jobs = [j for j in jobs if j[0] in keep]
    with Run("route_follow", "lateturn-scan", config=vars(a)) as run:
        run.use_split(sp)
        res = par.pmap(scan_one, jobs, run=run, workers=a.workers, desc="scan")
        D = pd.DataFrame([v for v in res.values if v is not None])
        D["cls"] = [cls_name(r) if r["ok"] else "error" for r in D.to_dict("records")]
        D.to_csv(a.out, index=False, float_format="%.4g")
        msg = D.groupby(["split", "cls"]).size().unstack(fill_value=0)
        run.info(f"scan: {len(D)} scenes\n{msg}")
        print(msg)
        print(D.groupby(["split", "fork_kind"]).size().unstack(fill_value=0))


def pick(a):
    import pandas as pd
    D = pd.read_csv(a.scan)
    done = set(Path(a.done).read_text().split())
    D = D[D.ok.astype(bool)]
    m = D.fork.astype(bool) | (D.turn.astype(bool) & (D.entry_rel >= DELAYED))       # late, delayed (20-40 m) or fork
    C = D[(D.split == "train") & m & ~D.scene.isin(done)].scene.sort_values().to_numpy()
    rng = np.random.default_rng(a.seed)
    pk = np.sort(rng.choice(C, a.cap, replace=False)) if len(C) > a.cap else C
    Path(a.out).write_text("\n".join(pk) + "\n")
    print(f"candidates {len(C)} (train late / delayed / fork not yet read) -> picked {len(pk)}")


# ---------------------------------------------------------------- shape (box): rf_ledger turn rows for the class scenes
def shape(a):
    import pickle
    import pandas as pd
    import rf_ledger as LG
    D = pd.read_csv(a.scan)
    keep = set(D[D.cls != "none"].scene)
    rows = []
    for spec in a.set:
        lab, f = spec.split("=", 1)
        for o in pickle.load(open(f, "rb"))["rolls"]:
            if o["scene"] not in keep:
                continue
            try:
                r, _ = LG.turn_rows(lab, o, set())
            except Exception as e:  # noqa: BLE001
                print("turn row failed", lab, o["scene"], repr(e)[:200], flush=True)
                continue
            if r is not None:
                rows.append(r)
    pd.DataFrame(rows).to_csv(a.out, index=False, float_format="%.4g")
    print(len(rows), "shape rows ->", a.out)


# ---------------------------------------------------------------- report (Mac)
def logof(x):
    return np.array([str(s).rsplit("-", 1)[0] for s in x])


def boot(x, logs, seed=0):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    x, logs = x[m], np.asarray(logs)[m]
    if not len(x):
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan)
    L, inv = np.unique(logs, return_inverse=True)
    s, c = np.bincount(inv, x, len(L)), np.bincount(inv, minlength=len(L)).astype(float)
    i = np.random.default_rng(seed).integers(len(L), size=(N_BOOT, len(L)))
    b = s[i].sum(1) / c[i].sum(1)
    return dict(n=int(len(x)), mean=float(x.mean()), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)))


def fork_outcome(dec, kz, target):
    """rf_follow's completion rule at an arbitrary target station on one rollout's decisions (st, lat, k sorted by k)."""
    st, lat, ks = dec.st.to_numpy(), dec.lat.to_numpy(), dec.k.to_numpy()
    kz = kz if kz >= 0 else 10 ** 6
    k_x, lat_x = -1, np.nan
    for i in range(len(st)):
        if st[i] >= target:
            w = 0.0 if i == 0 else (target - st[i - 1]) / max(st[i] - st[i - 1], 1e-9)
            lat_x = abs(lat[i] if i == 0 else (1 - w) * lat[i - 1] + w * lat[i])
            k_x = int(ks[i])
            break
    reached = k_x >= 0 and k_x < kz
    if reached and lat_x <= LAT_OK:
        return "completed"
    last = np.flatnonzero(ks < kz)
    lat_last = abs(lat[int(last[-1])]) if len(last) else np.nan
    return "event" if kz < 10 ** 6 and (k_x < 0 or kz <= k_x) else "off" if k_x >= 0 or lat_last > LAT_OK else "short"


def scores(a):
    """MODEL=<dir>,<dir> run dirs (several specs per model merge, the first read of a scene wins) -> json {model: {scene: [score, zero kind]}}."""
    import rf_trainread as TR
    out = {}
    for spec in a.runs:
        m, d = spec.split("=", 1)
        for k, v in TR.scores(d).items():
            out.setdefault(m, {}).setdefault(k, v)
    Path(a.out).write_text(json.dumps(out))
    print({m: len(v) for m, v in out.items()})


def report(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C = pd.read_csv(a.scan)
    C = C[C.ok.astype(bool)].set_index("scene")
    Fs = pd.concat([pd.read_csv(Path(d) / "scenes.csv") for d in a.follow], ignore_index=True)
    Dx = pd.concat([pd.read_parquet(Path(d) / "decisions.parquet") for d in a.follow], ignore_index=True)
    Fs["model"] = Fs.label.str.split("@").str[0]
    Fs = Fs[Fs.model.isin(MODELS) & Fs.scene.isin(C.index)]
    Fs = Fs.drop_duplicates(["model", "scene"])                          # a scene read twice (train pick + new) keeps the first
    Dx = Dx.merge(Fs[["label", "scene"]], on=["label", "scene"])
    sc = json.loads(Path(a.scores).read_text())                          # {model: {scene: [score, kind]}}
    rows = []
    for r in Fs.itertuples():
        c = C.loc[r.scene]
        cl = c.cls
        if cl == "none":
            continue
        oc = fork_outcome(Dx[(Dx.label == r.label) & (Dx.scene == r.scene)].sort_values("k"), int(r.kz), c.target)
        s, kind = sc.get(r.model, {}).get(r.scene, [np.nan, ""])
        rows.append(dict(model=r.model, scene=r.scene, log=r.scene.rsplit("-", 1)[0], split=c.split, cls=cl, late=cl.startswith("late"),
                         fork="fork" in cl, early=cl.startswith("early"), entry_rel=c.entry_rel, deg=c.deg, side=c.side, fork_kind=c.fork_kind,
                         sep_rel=c.sep_rel, score=s, zero=bool(s == 0), zkind=kind, outcome=oc, rf_outcome=r.outcome if isinstance(r.outcome, str) else "",
                         turn_strict=bool(c.turn_strict), completed=oc == "completed", short=oc == "short"))
    T = pd.DataFrame(rows)
    # inside cut (rf_trainread): turn scenes, junction decisions, max inward lateral > 1 m
    J = Dx[Dx.junction.astype(bool)].merge(Fs[Fs.turn.astype(bool)][["label", "scene", "side", "model"]], on=["label", "scene"])
    J["inw"] = J.lat * np.where(J.side == "L", 1.0, -1.0)
    cut = J.groupby(["model", "scene"]).inw.max().gt(1.0).rename("cut").reset_index()
    T = T.merge(cut, on=["model", "scene"], how="left")
    if a.shape and Path(a.shape).exists():
        Sh = pd.read_csv(a.shape).rename(columns=dict(label="model"))
        keep = ["v_ego_40", "v_log_40", "v_ego_20", "v_log_20", "v_ego_0", "v_log_0", "onset_d", "peak_in", "peak_out", "shape_1.0", "entry", "lag_x10"]
        Sh = Sh[["model", "scene"] + keep].rename(columns=dict(entry="lp_entry"))
        T = T.merge(Sh, on=["model", "scene"], how="left")
        for d in (40, 20, 0):
            T[f"dv_{d}"] = T[f"v_ego_{d}"] - T[f"v_log_{d}"]
    # one common scene set per class: scenes every model has
    n_mod = T.groupby("scene").model.nunique()
    T = T[T.scene.isin(n_mod[n_mod == len(MODELS)].index)]
    T.to_csv(out / "rollouts.csv", index=False, float_format="%.4g")
    er = T.entry_rel
    classes = {"late (all, entry >= 40 m)": T.late, "delayed (entry 20-40 m)": T.early & (er >= DELAYED), "late or delayed (entry >= 20 m)": er >= DELAYED,
               "fork (all)": T.fork, "fork, no turn": T.cls == "fork", "turn + fork": T.fork & (T.late | T.early),
               "early (all, entry < 40 m)": T.early, "early, entry < 20 m": T.early & (er < DELAYED), "early < 20 m, no fork": (T.cls == "early") & (er < DELAYED),
               "turn, no fork (all)": T.cls.isin(["early", "late"]), "turn + fork, entry < 20 m": T.fork & T.early & (er < DELAYED)}
    R = dict(counts={}, table={}, pairs={})
    pc = lambda d: "-" if not d or not d["n"] else f"{100 * d['mean']:.0f} % [{100 * d['lo']:.0f}, {100 * d['hi']:.0f}] ({d['n']})"  # noqa: E731
    pp = lambda d: "-" if not d or not d["n"] else f"{100 * d['mean']:+.0f} [{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]"  # noqa: E731
    cf = lambda d, f=".2f": "-" if not d or not d["n"] else f"{d['mean']:{f}} [{d['lo']:{f}}, {d['hi']:{f}}]"  # noqa: E731
    md = ["# LATE-TURNS: completion by class (rf_lateturn.py report)", "",
          "Class from the policy-free scan (scan.csv; rules in the script docstring): late = turn (strict or half-way) with entry >= 40 m after the clip start, "
          "early = entry < 40 m (delayed = 20-40 m), fork = the route leaves the ego lane's straight successor (branch or lane change then turn / branch) inside the "
          "log's travel. Reads @18 decisions, rm_chain readlist serving; train (onpol-train) and held 460 pooled unless split. Completion: the ego crosses the target station (strict turn: exit + 10 m; half-way "
          "turn: s_end - 5 m; fork only: min(s_sep + 10, s_end - 5) m) <= 2.5 m from the route before any zero event (= rf_follow on strict turns). "
          "CIs: log-clustered bootstrap 10 000; model pairs paired on the same scenes.", ""]
    md += ["## Completion by class x model", "", "| class | scenes (train / held) | " + " | ".join(MODELS) + " |", "|:--|:--|" + ":--|" * len(MODELS)]
    for cn, m in classes.items():
        x = T[m]
        ns = x[x.model == MODELS[0]]
        R["counts"][cn] = dict(train=int((ns.split == "train").sum()), held=int((ns.split == "held").sum()))
        cells = []
        for M in MODELS:
            g = x[x.model == M]
            R["table"].setdefault(cn, {})[M] = q = dict(completion=boot(g.completed, g.log), zero=boot(g.zero, g.log), short=boot(g.short, g.log),
                                                        cut=boot(g.cut.astype(float) if "cut" in g else [], g.log),
                                                        outcomes={o: int((g.outcome == o).sum()) for o in ("completed", "event", "off", "short", "noturn")},
                                                        zkinds={k: int((g.zkind == k).sum()) for k in ("collision_at_fault", "offroad", "left_corridor_laterally", "other")},
                                                        mean=boot(g.score, g.log))
            cells.append(pc(q["completion"]))
        md.append(f"| {cn} | {len(ns)} ({R['counts'][cn]['train']} / {R['counts'][cn]['held']}) | " + " | ".join(cells) + " |")
    md += ["", "## Paired completion differences (pp, same scenes)", "", "| class | RSGboth - RSG | RSG - RB0 | RSGboth - RB0 |", "|:--|:--|:--|:--|"]
    for cn, m in classes.items():
        x = T[m]
        cells = []
        for A, B in ((MODELS[1], MODELS[0]), (MODELS[0], MODELS[2]), (MODELS[1], MODELS[2])):
            xa, xb = x[x.model == A].set_index("scene"), x[x.model == B].set_index("scene")
            j = xa.index.intersection(xb.index)
            d = boot(xa.loc[j].completed.astype(float) - xb.loc[j].completed.astype(float), logof(j))
            R["pairs"].setdefault(cn, {})[f"{A} - {B}"] = d
            cells.append(pp(d))
        md.append(f"| {cn} | " + " | ".join(cells) + " |")
    md += ["", "## Contrasts within model (unpaired, different scenes): completion difference (pp)", "", "| contrast | " + " | ".join(MODELS) + " |",
           "|:--|" + ":--|" * len(MODELS)]
    for nm, (ma, mb) in {"entry >= 20 m - entry < 20 m": (er >= DELAYED, er < DELAYED), "turn + fork - turn, no fork": (T.fork & T.turn_strict.notna() & (T.late | T.early), T.cls.isin(["early", "late"])),
                         "fork, no turn - early < 20 m, no fork": (T.cls == "fork", (T.cls == "early") & (er < DELAYED))}.items():
        cells = []
        for M in MODELS:
            xa, xb = T[ma & (T.model == M)], T[mb & (T.model == M)]
            la, lb = np.unique(xa.log, return_inverse=True), np.unique(xb.log, return_inverse=True)
            if not len(xa) or not len(xb):
                cells.append("-")
                continue
            rng = np.random.default_rng(0)
            sa, ca = np.bincount(la[1], xa.completed.astype(float)), np.bincount(la[1]).astype(float)
            sb, cb = np.bincount(lb[1], xb.completed.astype(float)), np.bincount(lb[1]).astype(float)
            ia, ib = rng.integers(len(sa), size=(N_BOOT, len(sa))), rng.integers(len(sb), size=(N_BOOT, len(sb)))
            b = sa[ia].sum(1) / ca[ia].sum(1) - sb[ib].sum(1) / cb[ib].sum(1)
            d = dict(n=len(xa), mean=float(xa.completed.mean() - xb.completed.mean()), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)))
            R.setdefault("contrast", {}).setdefault(nm, {})[M] = d
            cells.append(pp(d))
        md.append(f"| {nm} | " + " | ".join(cells) + " |")
    md += ["", "## Outcomes, zeros, too slow, inside cut", "",
           "| class | model | mean score | completed / event / off / short | zeros (col / off / cor / other) | zero % | short % | inside-cut % (turns) |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for cn in classes:
        for M in MODELS:
            q = R["table"][cn][M]
            o, z = q["outcomes"], q["zkinds"]
            md.append(f"| {cn} | {M} | {cf(q['mean'])} | {o['completed']} / {o['event']} / {o['off']} / {o['short']}" + (f" (+{o['noturn']} no turn)" if o["noturn"] else "") +
                      f" | {sum(z.values())} ({z['collision_at_fault']} / {z['offroad']} / {z['left_corridor_laterally']} / {z['other']}) | {pc(q['zero'])} | {pc(q['short'])} | {pc(q['cut'])} |")
    if "dv_40" in T:
        md += ["", "## Turn shape vs the human (rf_ledger.turn_rows on the human path; turn classes only)", "",
               "Speed ego - human (m/s) when the ego first crosses entry - 40 / - 20 / 0 m of the human-path turn; onset - human (m, - = early); "
               "peak inside / outside offset to the human path (m). Means [log-clustered CI] (n).", "",
               "| class | model | ego v @-40 | ego - human @-40 | ego - human @-20 | ego - human @0 | onset - human | peak in | peak out |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
        for cn in ("late (all, entry >= 40 m)", "late or delayed (entry >= 20 m)", "early, entry < 20 m", "fork (all)", "turn, no fork (all)"):
            for M in MODELS:
                g = T[classes[cn] & (T.model == M)]
                f = lambda c: cf(boot(g[c], g.log)) + f" ({int(g[c].notna().sum())})"  # noqa: E731
                md.append(f"| {cn} | {M} | {f('v_ego_40')} | {f('dv_40')} | {f('dv_20')} | {f('dv_0')} | {f('onset_d')} | {f('peak_in')} | {f('peak_out')} |")
            g = T[classes[cn] & (T.model == MODELS[0])]
            md.append(f"| {cn} | human | {cf(boot(g.v_log_40, g.log))} | | | | | | |")
    (out / "summary.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ---------------------------------------------------------------- PAI (Mac)
def pai(a):
    import pickle
    import pandas as pd
    D = pickle.load(open(a.pkl, "rb"))
    rows = []
    for sc in D["scenes"]:
        for lab in ("rsg", "rbsg2"):
            o = sc["rolls"].get(lab)
            if o is None or "ctrl" not in o:
                continue
            row, _, _ = PF.one(sc, lab, o, zeros=False)
            if not row["turn"]:
                continue
            R = PF.dedup(o["route"])
            S = PF.stations(R)
            LG = sc["log"]
            lt, _ = PF.track(R, S, LG["P"])
            p0 = LG["P"][0]
            psi0 = float(np.arctan2(*(LG["P"][min(5, len(LG["P"]) - 1)] - p0)[::-1]))
            s0 = PF.project(R, S, p0)[0]
            c = classify(sc["lanes"], R, s0, float(lt.max()), p0, psi0)
            c["entry_rel"], c["turn"] = row["entry"] - s0, True             # the rollout's strict turn (rf_pai_follow) defines entry
            rows.append(dict(label=lab, scene=sc["scene"], side=row["side"], turn_deg=row["turn_deg"], entry_rel=c["entry_rel"], outcome=row["outcome"],
                             completed=row["completed"], score=row["score"], fork=c["fork"], fork_kind=c["fork_kind"], sep_rel=c["sep_rel"],
                             cls=cls_name(c)))
    T = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "pai_classes.csv", index=False, float_format="%.4g")
    md = ["# PAI 36 strict turns by class (rf_lateturn.py pai; rf_pai_follow.py's turns and outcomes, no new rollouts)", "",
          "PAI clips are 20 s: window = the log's whole travel. late = entry >= 40 m after the ego's start station; fork rule as nuPlan (route vs the "
          "ego lane's straight chain on the PAI map). Scene bootstrap 95 % CI.", "", "| class | label | turns | completion | completed / event / off / short |",
          "|:--|:--|--:|:--|:--|"]
    for lab in ("rsg", "rbsg2"):
        x = T[T.label == lab]
        for cn, m in {"late (all)": x.cls.str.startswith("late"), "early (all)": x.cls.str.startswith("early"), "fork (all)": x.fork,
                      "late, no fork": x.cls == "late", "early, no fork": x.cls == "early", "all": np.ones(len(x), bool)}.items():
            g = x[m]
            b = PF.boot(g.completed.astype(float))
            oc = g.outcome.str.replace("short_.*", "short", regex=True).value_counts()
            md.append(f"| {cn} | {lab} | {len(g)} | " + (f"{100 * b['rate']:.0f} % [{100 * b['lo']:.0f}, {100 * b['hi']:.0f}]" if len(g) else "-") +
                      f" | {' / '.join(str(int(oc.get(o, 0))) for o in ('completed', 'event', 'off', 'short'))} |")
    md += ["", "Fork kinds (rsg turns): " + ", ".join(f"{k} {v}" for k, v in T[T.label == 'rsg'].fork_kind.replace('', 'none').value_counts().items()),
           "", "Entry station after the start (rsg turns, m): " + ", ".join(f"{x:.0f}" for x in sorted(T[T.label == 'rsg'].entry_rel))]
    (out / "pai.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ---------------------------------------------------------------- review page (late-turns/: BEV + front camera, RSG vs RSGboth)
PAGE = {"rsg": "RSG-L18-s0", "rsgboth": "RSGboth-L18-s0"}
RANK = {"off": 0, "event": 1, "short": 2, "completed": 3}


def page_pick(a):
    """rollouts.csv (report) -> page.csv: every late / delayed turn, then the worst RSGboth fork scenes up to --n (outcome rank, score)."""
    import pandas as pd
    T = pd.read_csv(a.rollouts)
    W = T.pivot_table(index="scene", columns="model", values=["outcome", "score"], aggfunc="first")
    base = T[T.model == PAGE["rsgboth"]].set_index("scene")
    base["rank"] = base.outcome.map(RANK)
    lat = base[base.entry_rel >= DELAYED].sort_values(["rank", "score"])
    frk = base[base.fork.astype(bool) & ~base.index.isin(lat.index)].sort_values(["rank", "score"])
    keep = list(lat.index) + list(frk.index[: max(0, a.n - len(lat))])
    P = base.loc[keep].reset_index()
    P["group"] = np.where(P.entry_rel >= DELAYED, "late", "fork")
    P = P[["scene", "group", "split", "cls", "entry_rel", "deg", "side", "fork_kind", "sep_rel"]]
    for k, m in PAGE.items():
        x = T[T.model == m].set_index("scene").loc[P.scene]
        for c in ("outcome", "score", "zkind", "dv_40", "dv_20", "dv_0", "v_ego_0", "v_log_0", "onset_d", "peak_in", "peak_out"):
            if c in x:
                P[f"{c}_{k}"] = x[c].to_numpy()
    P.to_csv(a.out, index=False, float_format="%.4g")
    Path(a.out).with_suffix(".txt").write_text("\n".join(P.scene) + "\n")
    print(f"page: {int((P.group == 'late').sum())} late / delayed + {int((P.group == 'fork').sum())} fork scenes -> {a.out}")


def page_subset(a):
    """(box) rf_tax pkls -> one pkl {(model, scene): rollout} with the page scenes (RSG, RSGboth)."""
    import pickle
    S = set(Path(a.scenes).read_text().split())
    out = {}
    for spec in a.set:
        lab, f = spec.split("=", 1)
        for r in pickle.load(open(f, "rb"))["rolls"]:
            if r["scene"] in S and (lab, r["scene"]) not in out and "ctrl" in r:
                out[(lab, r["scene"])] = {k: r.get(k) for k in ("score", "ctrl", "log", "route", "traffic", "tg", "metrics")}
    pickle.dump(out, open(a.out, "wb"))
    print(len(out), "rollouts ->", a.out)


def _box(x, y, yaw, L, W):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]]) @ np.array([[c, s], [-s, c]]) + [x, y]


def _clip(job):
    """tmp/held_turns/render.py's BEV clip: ego controller trace vs human log, route lane, traffic; 800 px, 10 fps, 1 s hold."""
    import pickle
    import subprocess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    pkl, scene, key, dst, png, note = job
    D = _clip.D if getattr(_clip, "f", None) == pkl else None
    if D is None:
        _clip.D, _clip.f = pickle.load(open(pkl, "rb")), pkl
        D = _clip.D
    col = {"rsg": "#ffb347", "rsgboth": "#5fd7ff"}[key]
    r, ref = D[(PAGE[key], scene)], D.get((PAGE["rsg"], scene)) or D[(PAGE[key], scene)]
    pts = [np.c_[D[(m, scene)]["ctrl"]["x"], D[(m, scene)]["ctrl"]["y"]] for m in PAGE.values() if (m, scene) in D]
    Pp = np.vstack(pts + [ref["log"]["P"], np.asarray(ref["route"])[:60]])
    c, h = (Pp.min(0) + Pp.max(0)) / 2, max(max(Pp.max(0) - Pp.min(0)) / 2 + 14, 30)
    ct, cx, cy, cyaw = r["ctrl"]["t"], r["ctrl"]["x"], r["ctrl"]["y"], np.unwrap(r["ctrl"]["yaw"])
    lt, lp, ly = ref["log"]["T"], ref["log"]["P"], np.unwrap(ref["log"]["Y"])
    S_ = 8.0
    fig = plt.figure(figsize=(S_, S_), dpi=100, facecolor="#111")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#161616")
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    rt = np.asarray(ref["route"])
    ax.plot(rt[:, 0], rt[:, 1], color="#555", lw=6, alpha=.5, solid_capstyle="round", zorder=1)
    ax.plot(lp[:, 0], lp[:, 1], color="#ddd", lw=1.2, ls="--", zorder=2)
    tr = ax.plot([], [], color=col, lw=2, zorder=3)[0]
    ego = Polygon(_box(0, 0, 0, 4.9, 2), closed=True, fc=col, ec="w", lw=1, zorder=5)
    hum = Polygon(_box(0, 0, 0, 4.9, 2), closed=True, fc="none", ec="#ddd", lw=1.2, ls="--", zorder=4)
    ax.add_patch(ego)
    ax.add_patch(hum)
    tg, tf, tp = np.asarray(ref["tg"], float), ref["traffic"], []
    for _ in range(40):
        q = Polygon(_box(0, 0, 0, 4, 2), closed=True, fc="#556", ec="#889", lw=.6, zorder=3)
        ax.add_patch(q)
        tp.append(q)
    txt = ax.text(.02, .975, "", transform=ax.transAxes, color="#eee", va="top", family="monospace", fontsize=13, zorder=9)
    ax.text(.02, .02, f"dashed = human log, grey = route lane, solid = {key.upper()} ego", transform=ax.transAxes, color="#999", family="monospace", fontsize=10)
    ts = np.arange(ct[0], ct[-1] + 1, 1e5)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "800x800", "-r", "10", "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "medium", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    for i, t in enumerate(list(ts) + [ts[-1]] * 10):
        ego.set_xy(_box(np.interp(t, ct, cx), np.interp(t, ct, cy), np.interp(t, ct, cyaw), 4.9, 2))
        k = ts.searchsorted(t, side="right")
        tr.set_data(np.interp(ts[:k], ct, cx), np.interp(ts[:k], ct, cy))
        tl = min(t, lt[-1])
        hum.set_xy(_box(np.interp(tl, lt, lp[:, 0]), np.interp(tl, lt, lp[:, 1]), np.interp(tl, lt, ly), 4.9, 2))
        j = int(np.abs(tg - t).argmin())
        A, dt = tf[j], (t - tg[j]) / 1e6
        for q, pp in enumerate(tp):
            if q < len(A["x"]):
                mv = 0.0 if A["static"][q] else A["v"][q] * dt
                pp.set_xy(_box(A["x"][q] + mv * np.cos(A["yaw"][q]), A["y"][q] + mv * np.sin(A["yaw"][q]), A["yaw"][q], A["L"][q], A["W"][q]))
                pp.set_visible(True)
            else:
                pp.set_visible(False)
        txt.set_text(f"{PAGE[key]}  t={min(t, ts[-1]) / 1e6:4.1f}s  score {float(r['score'] or 0):.2f}" + (f"\nended: {note}" if t >= ts[-1] else ""))
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if png and i == len(ts) - 1:
            fig.savefig(png, dpi=60, facecolor="#111")
    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    return dst


def page_build(a):
    """(Mac) page.csv + page pkl (+ cam/<model>/<scene>.mp4) -> OUT/{late,fork}/NN_hash_{rsg,rsgboth,cam_rsg,cam_rsgboth}.mp4, index.html."""
    import html
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    P = pd.read_csv(a.page)
    out = Path(a.out)
    P["rank"] = P[f"outcome_rsgboth"].map(RANK)
    rows, jobs = [], []
    for g in ("late", "fork"):
        (out / g).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(P[P.group == g].sort_values(["rank", "score_rsgboth"]).itertuples(), 1):
            base = f"{g}/{i:02d}_{r.scene.split('-')[-1][:8]}"
            rows.append((g, i, base, r))
            for k in PAGE:
                if not (out / f"{base}_{k}.mp4").exists():
                    jobs.append((a.pkl, r.scene, k, str(out / f"{base}_{k}.mp4"), str(out / f"{base}_bev.png") if k == "rsgboth" else None,
                                 getattr(r, f"outcome_{k}")))
                cam = Path(a.cam) / PAGE[k] / f"{r.scene}.mp4"
                if cam.exists():
                    shutil.copy(cam, out / f"{base}_cam_{k}.mp4")
    with ProcessPoolExecutor(a.workers) as ex:
        for _ in ex.map(_clip, jobs):
            pass
    f = lambda x, n=1: "" if pd.isna(x) else f"{float(x):.{n}f}"  # noqa: E731
    o = [Path(a.head).read_text()]
    for g, title in (("late", "late / delayed：入口在片头 20 m 以后的转弯（按 RSGboth 结果由差到好）"), ("fork", "fork：路线离开本车道直行后继（分叉 / 先变道），RSGboth 最差的若干个")):
        rr = [x for x in rows if x[0] == g]
        o.append(f"<h2>{g}：{title}，{len(rr)} 个</h2><table><thead><tr><th>#</th><th>视频（BEV RSG、BEV RSGboth、前视相机 RSG、前视相机 RSGboth）</th>"
                 "<th>scene / 类别</th><th>入口站位（片头后 m）</th><th>结果 RSG → RSGboth</th><th>分 RSG / RSGboth</th><th>形状（RSG | RSGboth）</th><th>备注</th></tr></thead><tbody>")
        for _, i, bs, r in rr:
            def v(k, cam=False):
                src = f"{bs}_{'cam_' if cam else ''}{k}.mp4"
                if cam and not (out / src).exists():
                    return f"<div class=cell><div class=lab>camera {PAGE[k]}: 无重渲染</div></div>"
                poster = "" if cam else f' poster="{bs}_bev.png"'
                return (f'<div class=cell><video autoplay muted loop playsinline preload="auto" controls{poster} src="{src}"></video>'
                        f'<div class=lab>{"camera" if cam else "BEV"} {PAGE[k]} {src.split("/")[1]}</div></div>')
            shp = lambda k: (f"速度差 -40/-20/0 m：{f(getattr(r, f'dv_40_{k}', np.nan))} / {f(getattr(r, f'dv_20_{k}', np.nan))} / {f(getattr(r, f'dv_0_{k}', np.nan))} m/s；"  # noqa: E731
                             f"起转 {f(getattr(r, f'onset_d_{k}', np.nan), 0)} m；内 {f(getattr(r, f'peak_in_{k}', np.nan))} 外 {f(getattr(r, f'peak_out_{k}', np.nan))}")
            ent = f"{f(r.entry_rel, 0)} m（{r.side} {f(r.deg, 0)}°）" if not pd.isna(r.entry_rel) else f"分离点 {f(r.sep_rel, 0)} m（无 60° 转弯）"
            o.append(f'<tr><td>{i}</td><td><div class=vids>{v("rsg")}{v("rsgboth")}{v("rsg", True)}{v("rsgboth", True)}</div></td>'
                     f'<td><code>{html.escape(r.scene)}</code><br>{r.split}，{r.cls}，fork 类型 {r.fork_kind if isinstance(r.fork_kind, str) else "-"}</td><td>{ent}</td>'
                     f'<td>{r.outcome_rsg} → <b>{r.outcome_rsgboth}</b></td><td>{f(r.score_rsg, 2)} / {f(r.score_rsgboth, 2)}</td>'
                     f'<td>RSG：{shp("rsg")}<br>RSGboth：{shp("rsgboth")}</td><td class=ans contenteditable data-k=ans-{g}{i}></td></tr>')
        o.append("</tbody></table>")
    o.append(Path(a.tail).read_text())
    (out / "index.html").write_text("\n".join(o))
    print(f"page -> {out}/index.html: {len(rows)} scenes, {len(jobs)} BEV clips rendered")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("scan")
    p.add_argument("--held", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", default="", help="scan: keep only the scene ids in this file")
    p = sp.add_parser("pick")
    p.add_argument("--scan", required=True)
    p.add_argument("--done", required=True)
    p.add_argument("--cap", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sp.add_parser("shape")
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--scan", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("scores")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--scan", required=True)
    p.add_argument("--follow", nargs="+", required=True)
    p.add_argument("--scores", required=True, help="json {model: {scene: [score, zero kind]}}")
    p.add_argument("--shape", default="")
    p.add_argument("--out", required=True)
    p = sp.add_parser("pai")
    p.add_argument("--pkl", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("page-pick")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", required=True)
    p = sp.add_parser("page-subset")
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("page-build")
    p.add_argument("--page", required=True)
    p.add_argument("--pkl", required=True)
    p.add_argument("--cam", required=True, help="dir with <model>/<scene>.mp4 front-camera reruns")
    p.add_argument("--head", required=True)
    p.add_argument("--tail", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"page-pick": page_pick, "page-subset": page_subset, "page-build": page_build, "scan": scan, "pick": pick, "shape": shape, "scores": scores, "report": report, "pai": pai}[a.cmd](a)
