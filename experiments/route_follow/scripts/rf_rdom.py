#!/usr/bin/env python3
"""ROUTE-DOMINANCE (plans/2026-10-16-route-dominance-prereg.md): open-loop served / zeroed / swapped route memory at RSG-L18-s0's own states
before its turn zeros (held pool on1_L18RSGallheld, train pool on1_L18RSG) and at matched states of completed turns. Frames fixed, only the
route input changes; the plan's contact with the road raster and the agent boxes (the hinges' own geometry) is read for every input.

  rf_rdom.py states --zeros DIR --out O          any python with numpy: DIR/<set>/rollouts.pkl (rf_appzeros.py extract on the pool's run
                                                 dirs) -> O/states.npz (failing / control decisions mapped to pool rows), O/states.json
  rf_rdom.py swap   --states O --out O [--workers N]   envs/navsim2 (nuPlan lane graph): the other-exit route per selected state -> O/swap_<set>.npz
  rf_rdom.py run    --set S --states O --tags T1 T2 --out O   GPU pool job, envs/op-train: rm_train.build's store with the pool, plans of every
                                                 pool row under served / zero / zeroS and of the swap rows under swap -> O/run_<set>/plans_<tag>.npz, gate.json
  rf_rdom.py report --states O --out RESULTS --fig PNG   envs/op-train CPU: contacts, registered lines, per-state CSV, read.json, md, figure

Sets: held = on1_L18RSGallheld (runs/onpol_long/roll/RSG-L18-s0-held), train = on1_L18RSG (runs/onpol_long/roll/RSG-L18-s0).
Inputs: served = the stored memory + command; zero = memory 200 numbers 0 (rm_train's --rm-drop / G-rely masking), command kept;
zeroS = zero + command S (descriptive); swap = the other exit's centreline through rmem.window, valid bits AND the served ones, command
recomputed from the centreline 42.1 m ahead (navigation.command_from_route's +-2 m rule).
Contact over the 40 dense steps 0.1 .. 4.0 s: road = a Pacifica corner with road-raster sdf < 0 (road.npz, the road hinge's label); agent =
AgentHinge distance < 0 to a counted object (margin 0). States already in contact at t = 0 leave the line's denominator.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(R_), str(R_ / "lib"), str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/corridor/lib"),
                str(R_ / "experiments/corridor/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = D / "runs/op_parity/cache"
POOL = dict(held="on1_L18RSGallheld", train="on1_L18RSG")
ROWS = dict(held="L18RSGallheld", train="L18RSG")
CLS = {"offroad": "offroad", "collision_at_fault": "collision", "left_corridor_laterally": "corridor"}
PRE, CTRL_WIN, MIN_DH, MAX_TURN, WINDOW, AHEAD, L0 = 6, 25.0, 45.0, 120.0, 40.0, 110.0, 80.0 / 19.0 * 10
CONDS = ("served", "zero", "zeroS", "swap")
NB, SEED = 10_000, 0


def ext(P, e=0.05):
    """Polyline (m, 2) extended by e m along its first and last segments: rmem.window's coverage test at a slot that sits exactly on an end
    point is a float coin toss (G-feat lost one end slot on 93 % of the rows without it)."""
    P = np.asarray(P, np.float64)
    if len(P) < 2:
        return P
    u0, u1 = P[1] - P[0], P[-1] - P[-2]
    return np.vstack([P[0] - e * u0 / max(np.hypot(*u0), 1e-9), P, P[-1] + e * u1 / max(np.hypot(*u1), 1e-9)])


def feats_pts(f):
    """(200,) route features -> valid (40,), xy (40, 2) rear-axle frame (m), heading (40,)."""
    F = np.asarray(f, np.float64).reshape(40, 5)
    return F[:, 0] > 0.5, F[:, 1:3] * 20.0, np.arctan2(F[:, 4], F[:, 3])


# ---------------------------------------------------------------- states (CPU, numpy)
def cmd_states(a):
    import rf_appzeros as AZ
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = {k: [] for k in ("set", "row", "scene", "log", "k", "kind", "cls", "rel", "kz", "turn_deg", "zid")}
    summ = {}
    for st_ in ("held", "train"):
        RR = pickle.load(open(Path(a.zeros) / st_ / "rollouts.pkl", "rb"))
        M = np.load(D / "runs/onpol_nuplan/rows" / ROWS[st_] / "meta.npz", allow_pickle=False)
        tab = np.load(CR / POOL[st_] / "tab.npz")["names"]
        assert np.array_equal(M["names"], tab), "rows meta != pool tab"
        at = {(s, int(k)): i for i, (s, k) in enumerate(zip(M["scene"].tolist(), M["k"].tolist()))}
        n_z, n_tz, n_ct, bad_anchor, cnt = 0, 0, 0, 0, {}
        for r in RR:
            R = r["route"]
            S = AZ.stations(R)
            s_h, h_h = AZ.heading(R, S)
            T = AZ.turns(s_h, h_h, 45.0)
            t = AZ.track(r, R, S, s_h, h_h)
            st, ks = t["st"], r["k"]
            sc, lg = r["scene"], r["scene"].rsplit("-", 1)[0]
            z = r["score"] == 0
            cls = CLS.get(r["fail"]) if z else None
            if z and cls is None:
                cnt["zero_other"] = cnt.get("zero_other", 0) + 1
                continue
            sel = []
            if z:
                n_z += 1
                kz = AZ.zero_k(r)
                iz = int(np.flatnonzero(ks <= kz)[-1]) if kz is not None and (ks <= kz).any() else len(ks) - 1
                kz = int(ks[iz]) if kz is None else int(kz)
                sz, s0 = st[iz], st[0]
                tw = [q for q in T if s0 - 10 <= q[0] <= sz + 60]
                cnt[f"zero_{cls}"] = cnt.get(f"zero_{cls}", 0) + 1
                if not tw:
                    cnt["zero_noturn"] = cnt.get("zero_noturn", 0) + 1
                    continue
                n_tz += 1
                ent = min(tw, key=lambda q: 0.0 if q[0] <= sz <= q[1] + 10 else abs(sz - q[0]) if sz < q[0] else abs(sz - q[1]))
                for i, k in enumerate(ks):
                    if kz - PRE <= k <= kz - 1:
                        sel.append((i, "fail", cls, st[i] - ent[0], kz, np.degrees(ent[2])))
            elif r["score"] is not None and r["score"] > 0:
                smax = np.nanmax(st)
                done = [q for q in T if q[0] >= st[0] - 10 and smax >= q[1] + 10]
                n_ct += len(done)
                for i, k in enumerate(ks):
                    for q in done:
                        if -CTRL_WIN <= st[i] - q[0] <= CTRL_WIN:
                            sel.append((i, "ctrl", "ctrl", st[i] - q[0], -1, np.degrees(q[2])))
                            break
            for i, kind, c, rel, kz, td in sel:
                row = at.get((sc, int(ks[i])))
                if row is None:
                    cnt[f"{kind}_not_in_pool"] = cnt.get(f"{kind}_not_in_pool", 0) + 1
                    continue
                if np.hypot(*(M["anchor"][row][:2] - r["anchor"][i][:2])) > 1e-2:
                    bad_anchor += 1
                    continue
                for key, v in zip(cols, (st_, row, sc, lg, int(ks[i]), kind, c, rel, kz, td, f"{st_}:{sc}")):
                    cols[key].append(v)
        summ[st_] = dict(rollouts=len(RR), zeros_behaviour=n_z, turn_zeros=n_tz, completed_turns=n_ct, anchor_mismatch=bad_anchor, **cnt)
        print(st_, summ[st_], flush=True)
    O = {k: np.asarray(v) for k, v in cols.items()}
    np.savez(out / "states.npz", **O)
    for st_ in ("held", "train"):
        m = O["set"] == st_
        summ[st_].update(fail_states=int((m & (O["kind"] == "fail")).sum()), ctrl_states=int((m & (O["kind"] == "ctrl")).sum()),
                         fail_by_cls={c: int((m & (O["cls"] == c)).sum()) for c in ("offroad", "collision", "corridor")},
                         fail_logs=len(set(O["log"][m & (O["kind"] == "fail")].tolist())))
    (out / "states.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, default=float))


# ---------------------------------------------------------------- swap (navsim2, lane graph)
_LOC = {}


def map_loc(log):
    if log not in _LOC:
        for sp in ("test", "trainval"):
            f = D / "datasets/navsim/navsim_logs" / sp / f"{log}.pkl"
            if f.exists():
                _LOC[log] = pickle.load(open(f, "rb"))[0]["map_location"]
                break
        else:
            _LOC[log] = None
    return _LOC[log]


def swap_one(job):
    """One pool row -> dict(ok, why, f (200,), cmd, gap_deg, inside, feat_gate (valid equal, max coord diff in m))."""
    import lanegraph as LG
    import rmem as RM
    row, log, pose, f = job
    v, P, H = feats_pts(f)
    out = dict(row=row, ok=False, why="", f=np.zeros(200, np.float32), cmd=-1, gap=np.nan, inside=False, gate_valid=False, gate_end=False, gate_dxy=np.nan, d_ego=np.nan)
    # G-feat: the served valid points as one polyline through rmem.window reproduce the stored features
    if v.sum() >= 2:
        Fr = RM.window(ext(P[v])).reshape(-1)
        vr, Pr, _ = feats_pts(Fr)
        out["gate_valid"] = bool(np.array_equal(vr, v))
        dd = np.flatnonzero(vr != v)
        ends = np.flatnonzero(v)[[0, -1]]
        out["gate_end"] = bool(len(dd) == 0 or (len(dd) == 1 and dd[0] in ends))
        both = vr & v
        out["gate_dxy"] = float(np.hypot(*(Pr[both] - P[both]).T).max()) if both.any() else np.nan
    loc = map_loc(log)
    if loc is None:
        out["why"] = "no_log"
        return out
    g = LG.get(loc)
    gp = RM.to_local(pose, P)                                         # rear-axle frame -> map frame (pose = rear axle in the map frame)
    gh = H + pose[2]
    idx = [i for i in np.flatnonzero(v) if RM.OFFS[i] >= -2.0][::2]
    cand = [(i, g.candidates(gp[i], gh[i])) for i in idx]
    cand = [(i, c) for i, c in cand if c]
    if len(cand) < 3:
        out["why"] = "no_match"
        return out
    nodes = g.viterbi([c for _, c in cand])
    if nodes is None:
        out["why"] = "no_sequence"
        return out
    seq, _ = g.sequence(nodes)
    i0 = next((j for j, (i, _) in enumerate(cand) if RM.OFFS[i] >= 0.0), 0)
    start = nodes[i0]
    exy = np.asarray(pose[:2], np.float64)
    if g.kind[start] == 1:                                            # inside a junction connector: branch at its predecessor lane
        k = seq.index(start)
        prv = seq[k - 1] if k > 0 and g.kind[seq[k - 1]] == 0 else (min(g.pred[start], key=lambda q: float(np.hypot(*(g.line[q].xy - exy).T).min())) if g.pred[start] else None)
        if prv is None:
            out["why"] = "no_pred"
            return out
        out["inside"] = True
        rs = seq[seq.index(start):] if start in seq else [start]
        rseq, s0 = [prv] + rs, max(0.0, g.line[prv].L - 1.0)
    else:
        rseq, s0 = seq[seq.index(start):], float(g.line[start].frenet(exy[None])[0][0])
    paths = {}
    for p, _ in g.paths_ahead(rseq[0], s0, WINDOW):
        paths.setdefault(p[-1], (p, g.heading_at(p, s0, WINDOW)))
    paths = list(paths.values())
    drv = [(p, h) for p, h in paths if all(rseq[j] == p[j] for j in range(min(len(rseq), len(p))))]
    if not drv:
        out["why"] = "route_not_in_paths"
        return out
    dh = np.array([h for _, h in drv])
    alts = []
    for p, h in paths:
        if any(p is q for q, _ in drv):
            continue
        gap = float(np.degrees(np.abs(LG.wrap(h - dh)).min()))
        if gap >= MIN_DH and abs(float(np.degrees(LG.wrap(h - pose[2])))) <= MAX_TURN:
            alts.append((gap, p))
    if not alts:
        out["why"] = "no_other_exit"
        return out
    gap, p = max(alts, key=lambda x: x[0])
    Rxy, _, _ = g.centreline(p, exy, AHEAD)
    if len(Rxy) < 4 * 40:
        out["why"] = "short"
        return out
    loc_alt = RM.in_ego(pose, Rxy[::4])                                # 1 m samples, rear-axle frame
    back = P[v & (RM.OFFS <= -2.0)]                                   # the shared past (behind the ego) from the served memory
    poly = np.vstack([back, loc_alt]) if len(back) else loc_alt
    Fs = RM.window(ext(poly)).reshape(40, 5)
    vs = (Fs[:, 0] > 0.5) & v
    Fs[~vs] = 0.0
    if vs.sum() < 2:
        out["why"] = "no_overlap"
        return out
    L = np.r_[0.0, np.cumsum(np.hypot(*np.diff(loc_alt, axis=0).T))]
    y42 = float(np.interp(L0, L, loc_alt[:, 1])) if L[-1] >= L0 else float(loc_alt[-1, 1])
    out.update(ok=True, f=Fs.reshape(-1).astype(np.float32), cmd=0 if y42 > 2.0 else 2 if y42 < -2.0 else 1, gap=gap,
               d_ego=float(np.hypot(*loc_alt[0])))
    return out


def swap_log(jobs):
    return [swap_one(j) for j in jobs]


def cmd_swap(a):
    from jevdrive import par
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
    Z = np.load(Path(a.states) / "states.npz")
    with Run("route_follow", "rdom-swap", config=vars(a)) as run:
        for st_ in ("held", "train"):
            m = Z["set"] == st_
            rows = np.unique(Z["row"][m])
            M = np.load(D / "runs/onpol_nuplan/rows" / ROWS[st_] / "meta.npz", allow_pickle=False)
            F = np.load(D / "runs/route_mem/feat" / f"{POOL[st_]}.npz")
            assert np.array_equal(F["names"], M["names"])
            by = {}
            for r in rows:
                by.setdefault(str(M["log"][r]), []).append((int(r), str(M["log"][r]), M["pose_map"][r].astype(np.float64), F["f"][r]))
            byloc = {}
            for lg, js in by.items():                                  # one pmap per map location: a worker builds one lane graph
                byloc.setdefault(map_loc(lg) or "", []).append(js)
            V = []
            for loc, jobs in sorted(byloc.items()):
                res = par.pmap(swap_log, jobs, run=run, workers=min(a.workers, len(jobs)), desc=f"swap {st_} {loc}")
                if res.errors:
                    run.info(f"{st_} {loc}: {len(res.errors)} errors, first {list(res.errors.items())[:2]}")
                V += [x for v in res.values if v for x in v]
            O = {k: np.array([x[k] for x in V]) for k in V[0]}
            np.savez(Path(a.out) / f"swap_{st_}.npz", **O)
            why = {w: int((O["why"] == w).sum()) for w in set(O["why"].tolist())}
            run.info(f"{st_}: rows {len(rows)}, swap ok {int(O['ok'].sum())}, why {why}, G-feat valid equal {O['gate_valid'].mean():.3f}, "
                     f"up to one end slot {O['gate_end'].mean():.3f}, "
                     f"dxy <= 1 m {(np.nan_to_num(O['gate_dxy'], nan=9) <= 1.0).mean():.3f}")


# ---------------------------------------------------------------- run (GPU)
def cmd_run(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from types import SimpleNamespace
    sys.path[:0] = [str(R_ / "experiments/route_mem/scripts"), str(R_ / "experiments/onpol_nuplan/lib")]
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    dev = torch.device("cuda")
    pool = POOL[a.set]
    ns_a = SimpleNamespace(**REC, road_labels=RT.OT.JT.ROAD, on=pool, init="", rm_from=a.tags[0], tag=a.tags[0], exp="route_follow", lr_scale=1.0,
                           prefetch=1, dev_card_gb=4.0)
    ns, _, _, _, _ = RT.build(ns_a, dev)
    S, model, on0 = ns.S, ns.model, ns.on0
    W = ns.losses(ns.mode).W
    tab = np.load(CR / pool / "tab.npz")
    M = np.load(D / "runs/onpol_nuplan/rows" / ROWS[a.set] / "meta.npz", allow_pickle=False)
    F = np.load(D / "runs/route_mem/feat" / f"{pool}.npz")
    n = len(tab["names"])
    assert S.ego.shape[1] == 220, S.ego.shape
    E = S.ego[on0:on0 + n].cpu().numpy()
    gate = dict(ego_equal=float(np.abs(E[:, :20] - tab["ego"]).max()), feat_equal=float(np.abs(E[:, 20:] - F["f"]).max()))
    SW = np.load(Path(a.states) / f"swap_{a.set}.npz")
    sw_rows = SW["row"][SW["ok"]].astype(np.int64)
    sw_f = torch.as_tensor(SW["f"][SW["ok"]], device=dev)
    sw_c = torch.as_tensor(SW["cmd"][SW["ok"]].astype(np.int64), device=dev)
    pi = torch.as_tensor(S.pi, device=dev)

    def fwd(rows, cond, bs=256):
        res = []
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                r = on0 + np.asarray(rows[i:i + bs])
                rt = torch.as_tensor(r, device=dev)
                e = S.ego[rt].clone()
                if cond in ("zero", "zeroS"):
                    e[:, 20:] = 0
                if cond == "zeroS":
                    e[:, 1:4] = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=e.dtype)
                if cond == "swap":
                    e[:, 20:] = sw_f[i:i + bs].to(e.dtype)
                    e[:, 1:4] = torch.nn.functional.one_hot(sw_c[i:i + bs], 3).to(e.dtype)
                p = model(S.front[r], e, S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()[:, pi].view(-1, 33, 15)
                x, y, psi = T.rear(p, S.cam_x[rt], W)
                res.append(torch.stack([x, y, psi], -1).cpu())
        return torch.cat(res).numpy()

    out = Path(a.out) / f"run_{a.set}"
    out.mkdir(parents=True, exist_ok=True)
    Z = np.load(Path(a.states) / "states.npz")
    rd = np.unique(Z["row"][Z["set"] == a.set])
    allr = np.arange(n)
    for j, tag in enumerate(a.tags):
        if j:
            ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
            model.load_state(ck["model"])
        model.eval()
        P = {c: fwd(allr, c) for c in ("served", "zero", "zeroS")}
        P["swap"] = np.full((n, 8, 3), np.nan, np.float32)
        P["swap"][sw_rows] = fwd(sw_rows, "swap")
        if j == 0:
            d = np.hypot(*(P["served"][rd, :, :2] - M["poses"][rd, :, :2]).transpose(2, 0, 1)).mean(1)
            dall = np.hypot(*(P["served"][..., :2] - M["poses"][..., :2]).transpose(2, 0, 1)).mean(1)
            again = fwd(rd[:512], "served")
            gate.update(tag=tag, read_rows=int(len(rd)), rep_median=float(np.median(d)), rep_p95=float(np.percentile(d, 95)), rep_all_median=float(np.median(dall)),
                        rep_all_p95=float(np.percentile(dall, 95)), det_equal=bool(np.array_equal(again, P["served"][rd[:512]])),
                        swap_rows=int(len(sw_rows)))
            gate["pass"] = bool(gate["rep_median"] <= 0.05 and gate["rep_p95"] <= 0.3 and gate["det_equal"] and gate["ego_equal"] < 1e-5 and gate["feat_equal"] < 1e-5)
            print("gate", json.dumps(gate), flush=True)
            (out / "gate.json").write_text(json.dumps(gate, indent=1))
        np.savez(out / f"plans_{tag}.npz", **{c: v.astype(np.float32) for c, v in P.items()})
        print(tag, "done", flush=True)


# ---------------------------------------------------------------- report (CPU)
def contacts(st_, plans, rows):
    """plans {cond: (n_pool, 8, 3)} -> {cond: dict(road, agent, any (rows,) over 0.1-4 s; road2, agent2, any2 over 0.1-2 s)}, t0 flags."""
    import torch
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    pool = CR / POOL[st_]
    names = np.load(pool / "tab.npz")["names"]
    dev = torch.device("cpu")
    ah, rh = AgentHinge(pool / "agent.npz", names, dev, 0.0, 0.0), Hinge(pool / "road.npz", names, dev, 0.0)
    rr = torch.as_tensor(rows)
    okA, okR = ah.ok[rr].numpy(), rh.ok[rr].numpy()

    def one(Pn):
        P = torch.as_tensor(np.nan_to_num(Pn[rows]), dtype=torch.float32)
        with torch.no_grad():
            dist, cnt, _ = ah.distances(P[..., 0], P[..., 1], P[..., 2], rr)
            ag = ((dist < 0) & cnt).any(-1).numpy() & okA[:, None]                     # (B, 41)
            ro = (rh.margins(P[..., 0], P[..., 1], P[..., 2], rr).view(len(rows), 41, 4) < 0).any(-1).numpy() & okR[:, None]
        return ag, ro

    out = {}
    for c, Pn in plans.items():
        ag, ro = one(Pn)
        out[c] = dict(agent=ag[:, 1:].any(1), road=ro[:, 1:].any(1), agent2=ag[:, 1:21].any(1), road2=ro[:, 1:21].any(1), t0=ag[:, 0] | ro[:, 0])
        out[c]["any"], out[c]["any2"] = out[c]["agent"] | out[c]["road"], out[c]["agent2"] | out[c]["road2"]
    return out, okA, okR


def disp(Pc, Ps):
    """Counterfactual plan vs served, lateral in the served pose's heading frame: mean |dlat| 1.0-2.0 s, |dlat| 4 s, dheading 2 s / 4 s (deg)."""
    d = Pc[..., :2] - Ps[..., :2]
    c, s = np.cos(Ps[..., 2]), np.sin(Ps[..., 2])
    lat = -s * d[..., 0] + c * d[..., 1]
    dh = np.degrees((Pc[..., 2] - Ps[..., 2] + np.pi) % (2 * np.pi) - np.pi)
    return np.abs(lat[:, 1:4]).mean(1), np.abs(lat[:, 7]), dh[:, 3], dh[:, 7], lat[:, 7]


def boot_ratio(num, den, logs, nb=NB, seed=SEED):
    """Log-clustered percentile bootstrap of sum(num) / sum(den) -> (point, lo, hi, n, logs)."""
    u, inv = np.unique(logs, return_inverse=True)
    a, b = np.bincount(inv, num, len(u)), np.bincount(inv, den, len(u))
    pt = a.sum() / b.sum() if b.sum() else np.nan
    if b.sum() == 0:
        return dict(p=np.nan, lo=np.nan, hi=np.nan, n=0, logs=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (nb, len(u)))
    A, B = a[idx].sum(1), b[idx].sum(1)
    r = np.where(B > 0, A / np.maximum(B, 1e-12), np.nan)
    return dict(p=float(pt), lo=float(np.nanpercentile(r, 2.5)), hi=float(np.nanpercentile(r, 97.5)), n=int(b.sum()), logs=int((b > 0).sum()))


def line(b):
    if b["n"] == 0:
        return "n/a"
    if b["p"] >= 0.5 and b["lo"] > 0.3:
        return "ROUTE-OVERRIDE"
    if b["p"] <= 0.2 and b["hi"] < 0.35:
        return "BLIND"
    return "UNDECIDED"


def turn60(f):
    v, P, H = feats_pts(f)
    m = v & (np.arange(40) >= 3) & (np.arange(40) <= 33)
    if m.sum() < 2:
        return np.nan
    h = H[m]
    return float(np.degrees(np.abs((h - h[0] + np.pi) % (2 * np.pi) - np.pi).max()))


def cmd_report(a):
    import pandas as pd
    O, res = Path(a.states), Path(a.out)
    res.mkdir(parents=True, exist_ok=True)
    Z = {k: v for k, v in np.load(O / "states.npz").items()}
    tags = a.tags
    rows_out, S2 = [], {}
    for st_ in ("held", "train"):
        m = Z["set"] == st_
        if not m.any():
            continue
        rows = Z["row"][m].astype(np.int64)
        PL = {t: dict(np.load(O / f"run_{st_}" / f"plans_{t}.npz")) for t in tags}
        SW = np.load(O / f"swap_{st_}.npz")
        swi = {int(r): i for i, r in enumerate(SW["row"])}
        F = np.load(D / "runs/route_mem/feat" / f"{POOL[st_]}.npz")["f"]
        M = np.load(D / "runs/onpol_nuplan/rows" / ROWS[st_] / "meta.npz", allow_pickle=False)
        base = {k: Z[k][m] for k in Z}
        tu = np.array([turn60(f) for f in F])
        for t in tags:
            C, okA, okR = contacts(st_, {c: PL[t][c] for c in CONDS}, rows)
            Ps = PL[t]["served"][rows]
            rep = np.hypot(*(Ps[..., :2] - M["poses"][rows][..., :2]).transpose(2, 0, 1)).mean(1)
            rec = dict(base, tag=np.full(len(rows), t), rep=rep, label_ok=okA & okR, turn60=np.array([turn60(F[r]) for r in rows]),
                       swap_ok=np.array([bool(SW["ok"][swi[r]]) if r in swi else False for r in rows]),
                       swap_why=np.array([str(SW["why"][swi[r]]) if r in swi else "absent" for r in rows]),
                       swap_gap=np.array([float(SW["gap"][swi[r]]) if r in swi else np.nan for r in rows]),
                       swap_inside=np.array([bool(SW["inside"][swi[r]]) if r in swi else False for r in rows]),
                       cmd_served=np.argmax(np.load(CR / POOL[st_] / "tab.npz")["ego"][rows, 1:4], 1),
                       cmd_swap=np.array([int(SW["cmd"][swi[r]]) if r in swi else -1 for r in rows]))
            for c in CONDS:
                for q in ("road", "agent", "any", "any2", "t0"):
                    rec[f"{c}_{q}"] = C[c][q]
                if c != "served":
                    l12, l4, h2, h4, s4 = disp(PL[t][c][rows], Ps)
                    rec.update({f"{c}_lat12": l12, f"{c}_lat4": l4, f"{c}_dh2": h2, f"{c}_dh4": h4, f"{c}_slat4": s4})
            rows_out.append(pd.DataFrame(rec))
            # S2 over every pool row: zeroing displacement by route bucket
            allP = PL[t]
            l12, l4, _, h4, _ = disp(allP["zero"], allP["served"])
            for b, mm in (("turn60>=30", tu >= 30), ("straight<10", tu < 10), ("no memory", ~np.isfinite(tu))):
                S2[f"{st_}:{t}:{b}"] = dict(rows=int(mm.sum()), lat12_med=float(np.median(l12[mm])) if mm.any() else np.nan,
                                            lat12_p90=float(np.percentile(l12[mm], 90)) if mm.any() else np.nan,
                                            lat4_med=float(np.median(l4[mm])) if mm.any() else np.nan, lat4_p90=float(np.percentile(l4[mm], 90)) if mm.any() else np.nan,
                                            dh4_absmed=float(np.median(np.abs(h4[mm]))) if mm.any() else np.nan)
    X = pd.concat(rows_out, ignore_index=True)
    X.to_csv(res / "route_dominance_states.csv", index=False, float_format="%.4f")
    R = dict(states=json.loads((O / "states.json").read_text()), s2_all_rows=S2, gates={st_: json.loads((O / f"run_{st_}" / "gate.json").read_text())
                                                                                          for st_ in ("held", "train") if (O / f"run_{st_}" / "gate.json").exists()})
    R["swap"] = {st_: dict(rows=int(len(z["row"])), ok=int(z["ok"].sum()), why={w: int((z["why"] == w).sum()) for w in set(z["why"].tolist())},
                           g_feat_valid=float(z["gate_valid"].mean()), g_feat_end=float(z["gate_end"].mean()), g_feat_dxy_le1=float((np.nan_to_num(z["gate_dxy"], nan=9.0) <= 1.0).mean()))
                 for st_ in ("held", "train") for z in [np.load(O / f"swap_{st_}.npz")]}
    gf = all(v["g_feat_end"] >= 0.95 and v["g_feat_dxy_le1"] >= 0.95 for v in R["swap"].values())       # Amendment 1: one end slot may differ
    R["g_feat_pass"] = bool(gf)

    def block(Y):
        """Registered quantities on a frame of states (one tag)."""
        out = {}
        f = Y[Y.kind == "fail"]
        den = (f.served_any & ~f.served_t0 & f.label_ok)
        out["fail_states"], out["fail_t0_contact"], out["fail_no_label"] = int(len(f)), int((f.served_t0 & f.label_ok).sum()), int((~f.label_ok).sum())
        out["F"] = boot_ratio((den & ~f.zero_any).to_numpy(float), den.to_numpy(float), f.log.to_numpy())
        out["F"]["line"] = line(out["F"])
        out["F_zeroS"] = boot_ratio((den & ~f.zeroS_any).to_numpy(float), den.to_numpy(float), f.log.to_numpy())
        d2 = (f.served_any2 & ~f.served_t0 & f.label_ok)
        out["F_2s"] = boot_ratio((d2 & ~f.zero_any2).to_numpy(float), d2.to_numpy(float), f.log.to_numpy())
        for kind in ("fail", "ctrl"):
            g = Y[(Y.kind == kind) & Y.label_ok & ~Y.served_t0]
            dd = g.served_any
            out[f"{kind}_freed"] = boot_ratio((dd & ~g.zero_any).to_numpy(float), dd.to_numpy(float), g.log.to_numpy())
            cl = ~g.served_any
            out[f"{kind}_induced_zero"] = boot_ratio((cl & g.zero_any).to_numpy(float), cl.to_numpy(float), g.log.to_numpy())
            sw = cl & g.swap_ok
            out[f"{kind}_swap_into"] = boot_ratio((sw & g.swap_any).to_numpy(float), sw.to_numpy(float), g.log.to_numpy())
            out[f"{kind}_served_contact"] = boot_ratio(dd.to_numpy(float), np.ones(len(g)), g.log.to_numpy())
            for c in ("zero", "swap", "zeroS"):
                v = g[c + "_lat12"].dropna()
                out[f"{kind}_{c}_disp"] = dict(n=int(len(v)), lat12_med=float(v.median()) if len(v) else np.nan, lat12_p90=float(v.quantile(0.9)) if len(v) else np.nan,
                                               lat4_med=float(g[c + "_lat4"].dropna().median()) if len(v) else np.nan,
                                               lat4_p90=float(g[c + "_lat4"].dropna().quantile(0.9)) if len(v) else np.nan,
                                               dh4_absmed=float(g[c + "_dh4"].dropna().abs().median()) if len(v) else np.nan)
        for c in ("offroad", "collision", "corridor"):
            fc = f[f.cls == c]
            dc = (fc.served_any & ~fc.served_t0 & fc.label_ok)
            out[f"F_{c}"] = boot_ratio((dc & ~fc.zero_any).to_numpy(float), dc.to_numpy(float), fc.log.to_numpy())
            out[f"F_{c}"]["line"] = line(out[f"F_{c}"])
        # G-swap-sanity: the swap moves the 4 s end toward the other exit's side (command sign) on control states
        g = Y[(Y.kind == "ctrl") & Y.swap_ok & (Y.cmd_swap != 1)]
        side = np.where(g.cmd_swap == 0, 1.0, -1.0)
        out["gswap_ctrl_toward_med"] = float(np.median(side * g.swap_slat4)) if len(g) else np.nan
        out["gswap_ctrl_n"] = int(len(g))
        return out

    R["read"] = {}
    for t in tags:
        for nm, sel in (("pooled", X.tag == t), ("held", (X.tag == t) & (X.set == "held")), ("train", (X.tag == t) & (X.set == "train"))):
            R["read"][f"{t}:{nm}"] = block(X[sel])
    p, h, tr = (R["read"][f"{tags[0]}:{k}"]["F"]["line"] for k in ("pooled", "held", "train"))
    R["verdict"] = "UNDECIDED" if {h, tr} == {"ROUTE-OVERRIDE", "BLIND"} else p
    (res / "read.json").write_text(json.dumps(R, indent=1, default=lambda o: None if o is None else float(o) if np.isscalar(o) else str(o)))
    print(json.dumps({k: (v["F"], v["fail_freed"], v["ctrl_freed"]) for k, v in R["read"].items()}, default=str), flush=True)
    print("verdict", R["verdict"])
    if a.fig:
        figure(X, O, Path(a.fig), tags[0])


def figure(X, O, fig, tag):
    """4 example states: front frame (dump jpeg), BEV with road raster, agent boxes at t0 / 2 s, served / zeroed / swapped plans and routes."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    Y = X[(X.tag == tag) & X.label_ok & ~X.served_t0]
    f, g = Y[Y.kind == "fail"], Y[Y.kind == "ctrl"]
    picks = []
    for cand, key in ((f[f.served_any & ~f.zero_any], "zero_lat4"), (f[f.served_any & f.zero_any], "zero_lat4"),
                      (f[f.swap_ok & ~f.served_any & f.swap_any], "swap_lat4"), (g[g.swap_ok & ~g.served_any], "swap_lat4")):
        cand = cand[~cand.scene.isin([q.scene for q in picks])]
        if len(cand):
            picks.append(cand.sort_values(key, ascending=False).iloc[0])
    fg, ax = plt.subplots(2, len(picks), figsize=(4.6 * len(picks), 9.2), gridspec_kw=dict(height_ratios=[1, 1.6]), squeeze=False)
    for j, p in enumerate(picks):
        st_, row = p.set, int(p.row)
        M = np.load(D / "runs/onpol_nuplan/rows" / ROWS[st_] / "meta.npz", allow_pickle=False)
        PL = np.load(O / f"run_{st_}" / f"plans_{tag}.npz")
        road = np.load(CR / POOL[st_] / "road.npz")
        ag = np.load(CR / POOL[st_] / "agent.npz")
        F = np.load(D / "runs/route_mem/feat" / f"{POOL[st_]}.npz")["f"][row]
        SW = np.load(O / f"swap_{st_}.npz")
        try:
            dz = np.load(str(M["dump"][row]))
            ax[0, j].imshow(Image.open(io.BytesIO(dz["jpeg"].tobytes())))
        except Exception as e:                                                       # noqa: BLE001
            ax[0, j].text(0.5, 0.5, f"no frame ({type(e).__name__})", ha="center")
        ax[0, j].axis("off")
        ax[0, j].set_title(f"{p.set} {p.kind} {p.cls}  k {int(p.k)}\n{p.scene[-16:]}", fontsize=9)
        b = ax[1, j]
        sdf = road["sdf"][row].astype(np.float32)
        b.imshow(sdf > 0, origin="lower", extent=(24, -24, -8, 56), cmap="Greys", alpha=0.25, vmin=0, vmax=1.6)
        b.contour(np.linspace(24, -24, 96)[::-1], np.linspace(-8, 56, 128), sdf[:, ::-1], levels=[0], colors="0.4", linewidths=0.8)
        for kk in range(ag["box"].shape[2]):
            for ti, al in ((0, 0.9), (4, 0.35)):
                if not ag["valid"][row, ti, kk]:
                    continue
                x, y, yw, L, Wd = ag["box"][row, ti, kk]
                c, s = np.cos(yw), np.sin(yw)
                cr = np.array([[L, Wd], [L, -Wd], [-L, -Wd], [-L, Wd], [L, Wd]]) / 2
                pts = np.c_[x + c * cr[:, 0] - s * cr[:, 1], y + s * cr[:, 0] + c * cr[:, 1]]
                b.plot(-pts[:, 1], pts[:, 0], color="tab:red", alpha=al, lw=1)
        v, P, _ = feats_pts(F)
        b.plot(-P[v, 1], P[v, 0], ".", color="tab:blue", ms=3, alpha=0.5)
        i = np.flatnonzero(SW["row"] == row)
        if len(i) and SW["ok"][i[0]]:
            vs, Ps, _ = feats_pts(SW["f"][i[0]])
            b.plot(-Ps[vs, 1], Ps[vs, 0], ".", color="tab:green", ms=3, alpha=0.5)
        for c, col in (("served", "tab:blue"), ("zero", "tab:orange"), ("swap", "tab:green")):
            q = PL[c][row]
            if np.isfinite(q).all():
                b.plot(-np.r_[0, q[:, 1]], np.r_[0, q[:, 0]], "-o", color=col, ms=3, lw=1.8, label=f"{c}{' contact' if p[c + '_any'] else ''}")
        b.add_patch(plt.Rectangle((-1.1485, -1.127), 2.297, 5.176, fc="none", ec="k"))
        b.set_xlim(20, -20)
        b.set_ylim(-8, 48)
        b.set_aspect("equal")
        b.legend(fontsize=7, loc="lower left")
        b.set_title(f"zeroed moves 4 s end {p.zero_lat4:.2f} m; swap {p.swap_lat4:.2f} m", fontsize=9)
    fg.suptitle(f"ROUTE-DOMINANCE: {tag}, plans at fixed frames under served (blue), zeroed (orange), swapped (green) route memory; dots = route memory; "
                "red = agent boxes at t0 (solid) / 2 s (faint); grey line = road raster edge", fontsize=10)
    fg.tight_layout()
    fig.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(fig, dpi=110)
    print("wrote", fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("states")
    s.add_argument("--zeros", required=True), s.add_argument("--out", required=True)
    w = sp.add_parser("swap")
    w.add_argument("--states", required=True), w.add_argument("--out", required=True), w.add_argument("--workers", type=int, default=16)
    r = sp.add_parser("run")
    r.add_argument("--set", choices=("held", "train"), required=True), r.add_argument("--states", required=True)
    r.add_argument("--tags", nargs="+", default=["RSG-L18-s0", "APP1X-L18-s0"]), r.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--states", required=True), p.add_argument("--out", required=True), p.add_argument("--fig", default="")
    p.add_argument("--tags", nargs="+", default=["RSG-L18-s0", "APP1X-L18-s0"])
    a = ap.parse_args()
    dict(states=cmd_states, swap=cmd_swap, run=cmd_run, report=cmd_report)[a.cmd](a)
