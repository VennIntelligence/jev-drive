#!/usr/bin/env python3
"""DAGGER-CHECK (2026-10-16): two offline reads, no training, behind the RECOVERY-ENVELOPE round. Box; envs/op-train.

  rf_dagger.py a      CPU, after the GPU forward job (rf_recov.py fwd on RCVall dev-rec / train-rec and L18RSGallheld held01):
        Read A. Did the trained R / RL / RLF checkpoints move at the recovery rows of pool on1_RCVall? Per split (dev / train logs) and row kind
        (`h` = the policy's own state at the handover decision k = h; `post` = k > h, states the rule expert drove to), the imitation ADE to the
        expert target (tab `fut`, the expert's served plan) of RSG-L18-s0 and each checkpoint; the 2 s / 4 s point error; the plan's movement
        against RSG-L18-s0 (mean pose distance); log-clustered CI of the ADE change. Plus rf_recov's turning instrument (equal-arc 4 s heading
        toward the turn on the 276 decisions 0 / 1 turning rows) -> results/dagger_check/read_a.json
  rf_dagger.py b [--workers N]   CPU: Read B. The rule expert (experiments/alpasim/lib/rule_expert.py, XP-L12-E46: lookahead 1.2 s, entry
        4.6 m/s) planned from every row of RSG-L18-s0's own pool on1_L18RSG (17 474 rows): the row's rear-axle anchor, its measured speed
        (meta `speed`) and its simulation time (meta `t_us`, the driver's t0), on the scene's pack from decision 319's stages
        (rewind_onpol_all/{rest,s50,smoke}/pack, built from the 30-decision identity configs: same scene frame and time base, checked on the
        replay rows k <= 1). Rows of one scene in k order share one expert session (the projection hint), as in the driver.
        Validity: the expert plan's contact over 0.1-4 s (and 0.1-2 s) with the pool's agent boxes (AgentHinge distance < 0 to a counted box,
        margin 0) and road raster (a Pacifica corner with sdf < 0): rf_rdom's contact geometry, rows already in contact at t = 0 kept apart.
        Must-stop legality: the expert ignores lights; on rows with a must-stop line (rf_recov's envelope stop class) its plan against the stop
        bound alone (on_envelope.terms; excess > 0.05 m/s = runs the line).
        Disagreement: served plan - expert plan at 1 / 2 / 4 s, split along / across the expert pose's heading (lon / lat), heading at 4 s.
        Segments (exclusive, first wins; ctx route stations, strict turns = rf_loninv.strict_turns, >= 60 deg): junction (entry <= s0 <= exit),
        turn approach (entry 0-40 m ahead of the rear axle), stop approach (must-stop line 0-40 m ahead of the front bumper), straight
        (route heading change over the next 60 m < 30 deg), other (curves; off route |l0| > 3.5 m or no route). k <= 1 rows are log replay.
        Failure windows: decision 319's units (results/rewind_onpol_all/units.csv; 30-decision identity runs of the same policy and serving):
        failed (event / off) rows k_ref - 6 .. k_ref - 1, and entry-aligned windows (the 6 decisions before the rear axle crosses the strict
        turn's entry) of failed and completed units; route_dominance's states (same stored rollouts: fail = k_z - 6 .. k_z - 1, ctrl = completed
        turns within +-25 m of the entry) as a second cut.
        Yield: rows whose expert plan is contact-free over 4 s (agent and road, no t = 0 contact) and stop-legal AND disagrees (> 0.5 m at
        2 s or > 5 deg at 4 s) -> $O/rows.parquet (per row), $O/expert_L18RSG.npz (plans + info), results/dagger_check/read_b.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R_), str(HERE), str(R_ / "lib"), str(R_ / "experiments/onpol_nuplan/lib"), str(R_ / "experiments/onpol_nuplan/scripts"),
                str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/long_contact/lib"), str(R_ / "experiments/body1/lib"),
                str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/corridor/lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, ROWS = D / "runs/op_parity/cache", D / "runs/onpol_nuplan/rows"
O = D / "runs/route_follow/dagger_check"
RES = R_ / "experiments/route_follow/results/dagger_check"
RW = D / "runs/route_follow/rewind_onpol_all"
PACKS = [RW / s / "pack" for s in ("rest", "s50", "smoke")] + [D / "runs/route_follow/rewind_train64/pack"]
BASE = "RSG-L18-s0"
TAGS = ["RCVR-L18-s0", "RCVR-L18-s1", "RCVRL-L18-s0", "RCVRL-L18-s1", "RCVRLF-L18-s0", "RCVRLF-L18-s1"]
D2_M, H4_DEG, APPR_M, STOP_M_SEG, LAT_ROUTE, WIN = 0.5, 5.0, 40.0, 40.0, 3.5, 6
SEGS = ("junction", "turn_appr", "stop_appr", "straight", "other")


def ci(x, g):
    from jevdrive import stats
    r = stats.bootstrap(np.asarray(x, float), None if g is None else np.asarray(g))
    return [round(float(r["mean"]), 4), round(float(r["lo"]), 4), round(float(r["hi"]), 4)]


def pdist(A, B):
    """(n, 8, 3) x 2 -> per-pose xy distance (n, 8)."""
    return np.hypot(*(A[..., :2] - B[..., :2]).transpose(2, 0, 1))


# ---------------------------------------------------------------- Read A
def cmd_a(a):
    import rf_recov as RC
    from jevdrive import stats
    T = dict(np.load(CR / "on1_RCVall" / "tab.npz"))
    dev, kind, fut, logs = RC.dev_of(T), T["kind"].astype(str), T["fut"], T["log"].astype(str)
    out = dict(pool="on1_RCVall", rows=int(len(kind)), kind={k: int((kind == k).sum()) for k in np.unique(kind)},
               dev_rows=int(dev.sum()), base=BASE, tags=[], splits={})
    for split, rset in (("dev", "dev-rec"), ("train", "train-rec")):
        rr, Pb = RC.plans("RCVall", rset, BASE)
        f, kk, lg = fut[rr], kind[rr], logs[rr]
        eb = pdist(Pb, f)
        sm = pdist(Pb, T["served_model"][rr])
        S = dict(rows=int(len(rr)), check_base_vs_served_model={g: dict(median=float(np.median(sm[kk == g].mean(1))), p95=float(np.percentile(sm[kk == g].mean(1), 95)))
                                                                  for g in ("h", "post")}, groups={})
        for g in ("h", "post", "all"):
            m = (kk == g) if g != "all" else np.ones(len(rr), bool)
            G = dict(n=int(m.sum()), logs=len(set(lg[m])), base=dict(ade=float(eb[m].mean()), e2=float(eb[m, 3].mean()), e4=float(eb[m, 7].mean())), tags={})
            for tag in TAGS:
                try:
                    r2, Pt = RC.plans("RCVall", rset, tag)
                except FileNotFoundError:
                    continue
                assert np.array_equal(r2, rr)
                et = pdist(Pt, f)
                mv = pdist(Pt, Pb)
                r = stats.paired(et[m].mean(1), eb[m].mean(1), groups=lg[m])
                G["tags"][tag] = dict(ade=float(et[m].mean()), e2=float(et[m, 3].mean()), e4=float(et[m, 7].mean()),
                                      drop=float(1 - et[m].mean() / eb[m].mean()), d_ade=[r["mean"], r["lo"], r["hi"]],
                                      move_mean=float(mv[m].mean()), move_e4=float(mv[m, 7].mean()),
                                      toward=float((eb[m].mean(1) - et[m].mean(1)).mean() / max(mv[m].mean(), 1e-9)))
            S["groups"][g] = G
        out["splits"][split] = S
    out["tags"] = [t for t in TAGS if t in out["splits"]["dev"]["groups"]["all"]["tags"]]
    out["turn"] = {}
    for tag in out["tags"]:
        try:
            tr, _ = RC.gate_turn(tag, BASE)
            out["turn"][tag] = tr
        except FileNotFoundError as e:
            out["turn"][tag] = dict(error=repr(e)[:200])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "read_a.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------- Read B: the expert on every row
def expert_scene(job):
    """One scene's rows (k order, one session) -> plans (n, 8, 3), info (n, 6): station, lateral, stop distance, hits, plan end speed, err."""
    import rule_expert as XP
    name, f, idx, anc, v, t = job
    XP._scenes[name] = XP.Scene(Path(f))
    P = np.full((len(idx), 8, 3), np.nan, np.float32)
    Q = np.full((len(idx), 6), np.nan, np.float32)
    for j in range(len(idx)):
        try:
            p, info = XP.plan(name, name, int(t[j]), anc[j], float(v[j]))
            P[j] = p
            Q[j] = (info["s"], info["lat"], np.nan if info["stop"] is None else info["stop"], len(info["hits"]), info["v_plan"][-1], 0)
        except Exception:  # noqa: BLE001
            Q[j, 5] = 1
    XP._scenes.pop(name)
    XP.close(name)
    z = np.load(f)
    lt, lxy = z["log_t"].astype(float), z["log_xy"].astype(float)
    dlog = np.hypot(anc[:, 0] - np.interp(t, lt, lxy[:, 0]), anc[:, 1] - np.interp(t, lt, lxy[:, 1]))
    return idx, P, Q, dlog.astype(np.float32)


def contacts(pool, plans, rows, chunk=1024):
    """rf_rdom.contacts' geometry in chunks: {name: (n_pool, 8, 3)} -> {name: dict(agent, road, agent2, road2, t0) (len(rows),)}, okA, okR."""
    import torch
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    names = np.load(CR / pool / "tab.npz")["names"]
    dev = torch.device("cpu")
    ah, rh = AgentHinge(CR / pool / "agent.npz", names, dev, 0.0, 0.0), Hinge(CR / pool / "road.npz", names, dev, 0.0)
    okA, okR = ah.ok[torch.as_tensor(rows)].numpy(), rh.ok[torch.as_tensor(rows)].numpy()
    out = {c: {k: np.zeros(len(rows), bool) for k in ("agent", "road", "agent2", "road2", "t0")} for c in plans}
    for i in range(0, len(rows), chunk):
        rr = torch.as_tensor(rows[i:i + chunk])
        sl = slice(i, i + len(rr))
        for c, Pn in plans.items():
            P = torch.as_tensor(np.nan_to_num(Pn[rows[sl]]), dtype=torch.float32)
            with torch.no_grad():
                dist, cnt, _ = ah.distances(P[..., 0], P[..., 1], P[..., 2], rr)
                ag = ((dist < 0) & cnt).any(-1).numpy() & okA[sl, None]
                ro = (rh.margins(P[..., 0], P[..., 1], P[..., 2], rr).view(len(rr), 41, 4) < 0).any(-1).numpy() & okR[sl, None]
            o = out[c]
            o["agent"][sl], o["road"][sl], o["agent2"][sl], o["road2"][sl] = ag[:, 1:].any(1), ro[:, 1:].any(1), ag[:, 1:21].any(1), ro[:, 1:21].any(1)
            o["t0"][sl] = ag[:, 0] | ro[:, 0]
    return out, okA, okR


def disagree(Ps, Pe):
    """Served minus expert, in the expert pose's heading frame -> dict of (n,) arrays at 1 / 2 / 4 s."""
    d = Ps[..., :2] - Pe[..., :2]
    c, s = np.cos(Pe[..., 2]), np.sin(Pe[..., 2])
    lon, lat = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
    dh = np.degrees((Ps[..., 2] - Pe[..., 2] + np.pi) % (2 * np.pi) - np.pi)
    out = {}
    for t, i in (("1", 1), ("2", 3), ("4", 7)):
        out[f"d{t}"], out[f"lat{t}"], out[f"lon{t}"] = np.hypot(*d[:, i].T), lat[:, i], lon[:, i]
    out["dh4"] = dh[:, 7]
    return out


def segments(M, Z, X, Lb):
    """Per row: segment, strict-turn entry / exit on the ctx route (nan), rear-axle station s0."""
    import rf_loninv as LI
    N = len(M["names"])
    s0, l0, ok = Z["s0"].astype(float), Z["l0"].astype(float), Z["ok"].astype(bool)
    onr = ok & (np.abs(l0) <= LAT_ROUTE)
    sidx = {s: i for i, s in enumerate(X["scene"].tolist())}
    scene = M["scene"].astype(str)
    ent, ext = np.full(N, np.nan), np.full(N, np.nan)
    seg = np.full(N, "other", "U9")
    junc, appr = np.zeros(N, bool), np.zeros(N, bool)
    turns_of = {}
    for sc in np.unique(scene):
        i = np.flatnonzero(scene == sc)
        j = sidx.get(sc)
        if j is None:
            continue
        tn = LI.strict_turns(X["route"][j].astype(float), float(X["hlen"][j]))
        turns_of[sc] = tn
        for r in i:
            for en, ex in tn:
                if en <= s0[r] <= ex:
                    junc[r], ent[r], ext[r] = True, en, ex
                    break
                if 0 < en - s0[r] <= APPR_M:
                    appr[r], ent[r], ext[r] = True, en, ex
                    break
    rh = Z["rhead"].astype(float)
    NR = rh.shape[1]
    i0 = np.clip(np.round(s0).astype(int), 0, NR - 1)
    ii = np.clip(i0[:, None] + np.arange(61)[None], 0, NR - 1)
    turn60 = LI.wrapd(np.take_along_axis(rh, ii, 1) - rh[np.arange(N), i0][:, None]).max(1)
    stop = Lb["f_stop"].astype(bool) & (Lb["d_stop"] <= STOP_M_SEG)
    seg[onr & (turn60 < 30.0)] = "straight"
    seg[onr & stop] = "stop_appr"
    seg[onr & appr] = "turn_appr"
    seg[onr & junc] = "junction"
    return seg, ent, ext, s0, turns_of, turn60


def stop_check(Lb, plans, rows):
    """Excess over the must-stop bound alone (vuf; no turn / lead bound) on rows -> {name: (len(rows),) mean excess m/s}."""
    import torch
    import on_envelope as OE
    BIG = 1e3
    N = len(Lb["names"])
    pk = dict(route=torch.as_tensor(Lb["route"]), vuf=torch.as_tensor(Lb["vuf"]), vur=torch.full(Lb["vur"].shape, BIG),
              vl=torch.zeros(Lb["vl"].shape), ok=torch.as_tensor(Lb["ok"]), lsig=torch.full((N, 41), BIG), sf0=torch.as_tensor(Lb["sf0"]),
              v0=torch.as_tensor(Lb["v0"]))
    out = {}
    for c, P in plans.items():
        e = np.full(len(rows), np.nan, np.float32)
        good = np.isfinite(P[rows]).all((1, 2))
        r = rows[good]
        if len(r):
            e[good] = OE.terms(torch.as_tensor(P[r].astype(np.float32)), torch.as_tensor(r), pk)["envU"].numpy()
        out[c] = e
    return out


def cmd_b(a):
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    from jevdrive.run import Run
    with Run("route_follow", "dagger-check-b", config=vars(a)) as run:
        t_start = time.time()
        M = dict(np.load(ROWS / "L18RSG" / "meta.npz"))
        Z = dict(np.load(ROWS / "L18RSG" / "lon.npz"))
        X = dict(np.load(D / "runs/onpol_nuplan/ctx/ctx.npz"))
        Lb = dict(np.load(D / "runs/route_follow/recovery_envelope/lab_L18RSG.npz"))
        tab = np.load(CR / "on1_L18RSG" / "tab.npz")
        names = M["names"]
        assert (Z["names"] == names).all() and (Lb["names"] == names).all() and (tab["names"] == names).all()
        N = len(names)
        scene, k = M["scene"].astype(str), M["k"].astype(int)
        logu = M["logu"].astype(str)
        served = M["poses"].astype(np.float32)
        chk_served = float(np.abs(served - tab["served"]).max())
        pack = {}
        for d in PACKS:
            for sc in np.unique(scene):
                f = d / f"{sc}.npz"
                if sc not in pack and f.exists():
                    pack[sc] = str(f)
        miss = sorted(set(scene) - set(pack))
        jobs = []
        for sc in sorted(pack):
            i = np.flatnonzero(scene == sc)
            i = i[np.argsort(k[i])]
            jobs.append((sc, pack[sc], i, M["anchor"][i].astype(float), M["speed"][i].astype(float), M["t_us"][i].astype(np.int64)))
        E = np.full((N, 8, 3), np.nan, np.float32)
        info = np.full((N, 6), np.nan, np.float32)
        dlog = np.full(N, np.nan, np.float32)
        with ProcessPoolExecutor(a.workers) as ex:
            for idx, P, Q, dl in run.tqdm(ex.map(expert_scene, jobs, chunksize=4), total=len(jobs), desc="scenes"):
                E[idx], info[idx], dlog[idx] = P, Q, dl
        t_exp = time.time() - t_start
        has = np.isfinite(E).all((1, 2))
        O.mkdir(parents=True, exist_ok=True)
        np.savez(O / "expert_L18RSG.npz", names=names, poses=E, info=info, info_cols=np.array(["s", "lat", "stop", "hits", "v_end", "err"]),
                 dlog=dlog, has=has)
        rows = np.arange(N)
        C, okA, okR = contacts("on1_L18RSG", {"served": served, "expert": E}, rows)
        seg, ent, ext, s0, turns_of, turn60 = segments(M, Z, X, Lb)
        stopr = np.flatnonzero(Lb["f_stop"].astype(bool) & (Lb["d_stop"] <= 60))
        SC = stop_check(Lb, {"served": served, "expert": E}, stopr)
        stop_ex_e, stop_ex_s = np.full(N, np.nan, np.float32), np.full(N, np.nan, np.float32)
        stop_ex_e[stopr], stop_ex_s[stopr] = SC["expert"], SC["served"]
        Dg = disagree(served, E)
        df = pd.DataFrame(dict(name=names, scene=scene, log=logu, k=k, dev=Lb["dev"].astype(bool), replay=k <= 1, seg=seg,
                               speed=M["speed"].astype(float), s0=s0, entry=ent, exit=ext, has=has, xp_s=info[:, 0], xp_lat=info[:, 1],
                               xp_stop=info[:, 2], xp_hits=info[:, 3], xp_vend=info[:, 4], xp_err=info[:, 5], dlog=dlog, okA=okA, okR=okR,
                               **{f"{c}_{q}": C[c][q] for c in ("served", "expert") for q in ("agent", "road", "agent2", "road2", "t0")},
                               stop_ex_xp=stop_ex_e, stop_ex_sv=stop_ex_s, f_stop=Lb["f_stop"].astype(bool), d_stop=Lb["d_stop"], **Dg))
        df["xp_free"] = df.has & ~df.expert_agent & ~df.expert_road & ~df.expert_t0
        df["xp_free2"] = df.has & ~df.expert_agent2 & ~df.expert_road2 & ~df.expert_t0
        df["sv_free"] = ~df.served_agent & ~df.served_road & ~df.served_t0
        df["stop_legal"] = ~(df.stop_ex_xp > 0.05)
        df["disagree"] = (df.d2 > D2_M) | (df.dh4.abs() > H4_DEG)
        df["cand"] = df.xp_free & df.stop_legal & df.disagree
        # failure windows: decision 319's units and route_dominance's states
        U = list(csv.DictReader(open(R_ / "experiments/route_follow/results/rewind_onpol_all/units.csv")))
        at = {(s, int(kk)): i for i, (s, kk) in enumerate(zip(scene.tolist(), k.tolist()))}
        df["win"] = ""
        wl = {}
        for u in U:
            sc, oc = u["scene"], u["outcome"]
            if oc not in ("event", "off", "completed"):
                continue
            ri = np.flatnonzero(scene == sc)
            if not len(ri):
                continue
            ri = ri[np.argsort(k[ri])]
            if oc in ("event", "off") and u["k_ref"] != "":
                kr = int(u["k_ref"])
                for kk in range(kr - WIN, kr):
                    if (sc, kk) in at:
                        wl.setdefault("fail_kref", []).append(at[(sc, kk)])
            tn = turns_of.get(sc, [])
            cross = [(en, ex) for en, ex in tn if (s0[ri] >= en).any() and s0[ri[0]] < en]
            if cross:
                en = cross[0][0]
                kc = int(k[ri][np.argmax(s0[ri] >= en)])
                key = "fail_entry" if oc in ("event", "off") else "done_entry"
                for kk in range(kc - WIN, kc):
                    if (sc, kk) in at:
                        wl.setdefault(key, []).append(at[(sc, kk)])
        S_ = np.load(D / "runs/route_follow/route_dominance/states.npz")
        tr = S_["set"] == "train"
        wl["rdom_fail"] = S_["row"][tr & (S_["kind"] == "fail")].astype(int).tolist()
        wl["rdom_ctrl"] = S_["row"][tr & (S_["kind"] == "ctrl")].astype(int).tolist()
        try:
            df.to_parquet(O / "rows.parquet")
            rows_file = str(O / "rows.parquet")
        except Exception:  # noqa: BLE001
            df.to_csv(O / "rows.csv.gz", index=False)
            rows_file = str(O / "rows.csv.gz")

        def block(d):
            g = d.log.to_numpy()
            e = d[d.has]
            ge = e.log.to_numpy()
            return dict(rows=int(len(d)), scenes=int(d.scene.nunique()), logs=int(d.log.nunique()), expert=int(len(e)),
                        xp_free4=ci(e.xp_free, ge), xp_free2=ci(e.xp_free2, ge), sv_free4=ci(e.sv_free, ge),
                        xp_agent4=float(e.expert_agent.mean()), xp_road4=float(e.expert_road.mean()), xp_t0=float(e.expert_t0.mean()),
                        sv_agent4=float(e.served_agent.mean()), sv_road4=float(e.served_road.mean()), sv_t0=float(e.served_t0.mean()),
                        stop_rows=int(e.stop_ex_xp.notna().sum()), xp_runs_stop=float((e.stop_ex_xp > 0.05).mean()) if e.stop_ex_xp.notna().any() else None,
                        **{f"{q}_med": float(e[q].abs().median()) for q in ("d1", "d2", "d4", "lat4", "lon4", "dh4")},
                        **{f"{q}_mean": float(e[q].mean()) for q in ("d1", "d2", "d4", "lat4", "lon4", "dh4")},
                        d2_ci=ci(e.d2, ge), dh4abs_ci=ci(e.dh4.abs(), ge), lon4_ci=ci(e.lon4, ge),
                        d2_p90=float(e.d2.quantile(0.9)), disagree=float(e.disagree.mean()),
                        yield_rows=int(e.cand.sum()), yield_scenes=int(e[e.cand].scene.nunique()), yield_logs=int(e[e.cand].log.nunique()),
                        yield_train_rows=int((e.cand & ~e.dev).sum()), yield_train_scenes=int(e[e.cand & ~e.dev].scene.nunique()),
                        yield_train_logs=int(e[e.cand & ~e.dev].log.nunique()), yield_own=int((e.cand & ~e.replay).sum()),
                        yield_share=float(e.cand.mean()), n_logs_ci=len(set(g)))
        rec = dict(pool="on1_L18RSG", rows=N, scenes=int(len(set(scene))), packs=len(pack), scenes_no_pack=len(miss), no_pack=miss[:50],
                   expert_rows=int(has.sum()), expert_errors=int((info[:, 5] == 1).sum()), served_vs_tab=chk_served,
                   frame_check_replay_k01=dict(median=float(np.nanmedian(dlog[k <= 1])), p95=float(np.nanpercentile(dlog[k <= 1], 95)),
                                               max=float(np.nanmax(dlog[k <= 1]))),
                   xp_lat_abs=dict(median=float(np.nanmedian(np.abs(info[:, 1]))), p95=float(np.nanpercentile(np.abs(info[:, 1]), 95)),
                                   over_3_5=float(np.nanmean(np.abs(info[:, 1]) > 3.5))),
                   thresholds=dict(d2_m=D2_M, dh4_deg=H4_DEG, appr_m=APPR_M, stop_m=STOP_M_SEG, window=WIN),
                   all=block(df), own=block(df[~df.replay]), replay=block(df[df.replay]),
                   seg={s: block(df[(df.seg == s) & ~df.replay]) for s in SEGS},
                   seg_all={s: block(df[df.seg == s]) for s in SEGS},
                   windows={w: block(df.iloc[sorted(set(v))]) | dict(units=int(df.iloc[sorted(set(v))].scene.nunique())) for w, v in wl.items()},
                   rows_file=rows_file, expert_s=round(t_exp, 1), wall_s=round(time.time() - t_start, 1))
        for w, v in wl.items():
            d = df.iloc[sorted(set(v))]
            rec["windows"][w]["seg"] = d.seg.value_counts().to_dict()
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "read_b.json").write_text(json.dumps(rec, indent=1, default=float))
        run.summary.update(dict(rows=N, expert_rows=int(has.sum()), yield_rows=rec["all"]["yield_rows"]))
        print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("a")
    p = sp.add_parser("b")
    p.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    dict(a=cmd_a, b=cmd_b)[a.cmd](a)
