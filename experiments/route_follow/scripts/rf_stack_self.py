#!/usr/bin/env python3
"""STACK reflex module (experiments/route_follow/plans/2026-10-16-stack-prereg.md 4.5): the offline self-state rows and the serving checks.
Box, repo root. Results: experiments/route_follow/results/stack/checks.md, "Reflex: self-state data and serving".

  feat     CPU (.venv): $D/runs/route_follow/stack/self/<P>.npz, names (= the pool tab's, same order) and f (N, 14) float32 = reflex.selfstate,
           for on1_L18RB0, on1_L18RSG, on1_L18RSGallheld (rows dirs runs/onpol_nuplan/rows/<name>: meta run / dump / k) and the merged
           on1_RCVL (rows mapped back by name prefix: s -> on1_L18RB0, e0 -> on1_L18RSG; checked on served / ego / speed).
           Rule = the served driver's: each session's drive.jsonl records replayed through lib/reflex_mem.Memory (the previous decision is
           the newest one >= 0.45 s back = k - 1 at 2 Hz; its anchor and t0 from drive.jsonl, its served poses from its SH30_DUMP npz,
           unrounded, drive.jsonl's 1e-4 m poses without one); the current pose = the row's anchor, v = the tab's `speed`.
           Validation: every session recomputed from drive.jsonl alone (rounded poses, the logged ego's speed: what a driver log gives)
           against the rows; the tab's `pose` history against the replayed rel. -> self/summary.json
  ckpt     GPU (op-train): STKRX0-test = RSG-L18-s0 + the module at its zero init; STKRX1-test = the same with gate 1 and out weights
           N(0, 1e-3) (the context bias and the residual both live). New tags only; sidecar = RSG-L18-s0's + `reflex`
  replay   GPU (op-train): sh30_core.Core.plan (of the tree --code: this one, or a worktree before the change) on stored decisions: the
           4 dumped CAM_F0 JPEGs of decisions k-3 .. k, the logged history / ego / command / route memory; with --self the self-state
           from reflex_mem over the stored rollout -> npz of mu, poses, fed row length, rx
  cmp      CPU: the replay checks (1) old vs new code, no module; (2) zero module = base, fed row 20 + route + 14, valid; (3) live module
           moves the near plan by a bounded amount, JEV_REFLEX_OFF=1 = base -> self/replay_checks.json
  smoke    CPU: closed-loop reads (rm_chain.sh readlist, SH30_COMPILE=0) compared on drive.jsonl; the logged rx against reflex_mem replayed
           over the same drive.jsonl -> self/smoke_checks.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_R))
import numpy as np  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, ROWS = D / "runs/op_parity/cache", D / "runs/onpol_nuplan/rows"
O = D / "runs/route_follow/stack/self"
POOLS = ("L18RB0", "L18RSG", "L18RSGallheld")
MERGED = {"RCVL": {"s": "L18RB0", "e0": "L18RSG"}}
NAMES = ["valid"] + [f"{q}{t}" for t in ("0", "05", "10", "15") for q in ("x", "y", "yaw")] + ["dv"]


def _lib():
    if str(_R / "lib") not in sys.path:
        sys.path.insert(0, str(_R / "lib"))
    import reflex_mem as M
    return M


def sessions(run) -> dict:
    """drive.jsonl of a run -> {(scene, n): [(uuid, start record, drive records sorted by k)]}."""
    S, rec = {}, collections.defaultdict(list)
    run = Path(run)
    log = run / "driver-logs" if (run / "driver-logs/drive.jsonl").exists() else run       # PAI: drive.jsonl + start.jsonl in the run dir
    if (log / "start.jsonl").exists():
        for line in open(log / "start.jsonl"):
            r = json.loads(line)
            S[r["session"]] = r
    for line in open(log / "drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "start":
            S[r["session"]] = r
        elif r["kind"] == "drive" and r["session"] in S:
            rec[r["session"]].append(r)
    out = collections.defaultdict(list)
    for u, st in S.items():
        out[st["scene"], st["n"]].append((u, st, sorted(rec[u], key=lambda r: r["k"])))
    return out


def replay(recs, served_of, v_of):
    """Records of one session -> {k: selfstate} by reflex_mem.Memory (the driver's rule). served_of(r) -> the served poses kept; v_of(r) the speed."""
    M = _lib()
    m, out = M.Memory(), {}
    for r in recs:
        v = v_of(r)
        if v is not None:
            out[r["k"]] = m.features(r["t0"], r["anchor"], v)
        m.add(r["t0"], r["anchor"], served_of(r))
    return out


def feat_run(job):
    """(pool, run dir, row indices, scenes, ks, dumps, served (n, 8, 3), speeds) -> features of those rows + diagnostics."""
    P, run, idx, scene, ks, dumps, served, speed = job
    S = sessions(run)
    f, fd = np.zeros((len(idx), 14), np.float32), np.full((len(idx), 14), np.nan, np.float32)
    dt, kp, prel = np.full(len(idx), np.nan), np.full(len(idx), -1), np.full((len(idx), 3), np.nan)
    by = collections.defaultdict(list)
    for j in range(len(idx)):
        n = int(re.search(r"s(\d+)_k(\d+)\.npz$", dumps[j]).group(1))
        by[scene[j], n].append(j)
    miss = 0
    for (sc, n), js in by.items():
        cand = S.get((sc, n), [])
        def ok(c):
            R = {r["k"]: r for r in c[2]}
            return all(ks[j] in R and np.abs(np.asarray(R[ks[j]]["poses"]) - served[j]).max() < 2e-3 for j in js)
        cand = [c for c in cand if ok(c)]
        if len(cand) != 1:
            miss += len(js)
            continue
        _, st, recs = cand[0]
        ddir = Path(dumps[js[0]]).parent
        sp = {ks[j]: float(speed[j]) for j in js}

        def served_dump(r):
            p = ddir / f"s{n:02d}_k{r['k']}.npz"
            if p.exists():
                with np.load(p) as z:
                    return z["poses"].astype(np.float64)
            return np.asarray(r["poses"], np.float64)
        A = replay(recs, served_dump, lambda r: sp.get(r["k"]))
        B = replay(recs, lambda r: np.asarray(r["poses"], np.float64), lambda r: float(np.hypot(*np.asarray(r["ego"][4:6], np.float64))) * 10.0)
        M = _lib()
        R = {r["k"]: r for r in recs}
        for j in js:
            fa, ia = A[ks[j]]
            f[j], fd[j] = fa, B[ks[j]][0]
            if ia:
                dt[j], kp[j] = ia["dt"], next(r["k"] for r in recs if r["t0"] == ia["t_prev"])
                prel[j] = M.rel_pose(R[kp[j]]["anchor"], R[ks[j]]["anchor"])
    return dict(P=P, idx=idx, f=f, fd=fd, dt=dt, kp=kp, prel=prel, miss=miss)


def inv(p):
    """(n, 3) poses -> the poses of the origin in their frames."""
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    return np.c_[-(c * p[:, 0] + s * p[:, 1]), s * p[:, 0] - c * p[:, 1], -p[:, 2]]


def stats(f):
    v = f[:, 0] > 0.5
    q = lambda c: dict(median=round(float(np.median(np.abs(f[v, c]))), 4), p90=round(float(np.percentile(np.abs(f[v, c]), 90)), 4)) if v.any() else None  # noqa: E731
    return dict(rows=int(len(f)), valid=int(v.sum()), coverage=round(float(v.mean()), 4), **{f"abs_{NAMES[c]}": q(c) for c in (1, 2, 3, 8, 9, 13)},
                clipped_share=round(float((np.abs(f[v][:, [1, 2, 4, 5, 7, 8, 10, 11]]) >= 5 - 1e-6).any(1).mean()), 5) if v.any() else None)


def cmd_feat(a):
    from jevdrive import par
    from jevdrive.run import Run
    O.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "stack-self-feat", config=vars(a)) as run:
        jobs, T = [], {}
        for P in POOLS:
            T[P] = dict(np.load(CR / f"on1_{P}" / "tab.npz"))
            Mt = np.load(ROWS / P / "meta.npz", allow_pickle=True)
            assert (Mt["names"] == T[P]["names"]).all(), P
            for rd in sorted(set(Mt["run"].tolist())):
                i = np.flatnonzero(Mt["run"] == rd)
                jobs.append((P, rd, i, Mt["scene"][i], Mt["k"][i], Mt["dump"][i], T[P]["served"][i].astype(np.float64), T[P]["speed"][i]))
        res = par.pmap(feat_run, jobs, run=run, workers=a.workers or None, desc="runs")
        res.raise_if_failed()
        F, summ = {}, dict(pools={}, merged={})
        for P in POOLS:
            n = len(T[P]["names"])
            f, fd, dt, kp, prel = np.zeros((n, 14), np.float32), np.full((n, 14), np.nan, np.float32), np.full(n, np.nan), np.full(n, -1), np.full((n, 3), np.nan)
            miss = 0
            for v in res.values:
                if v["P"] == P:
                    f[v["idx"]], fd[v["idx"]], dt[v["idx"]], kp[v["idx"]], prel[v["idx"]] = v["f"], v["fd"], v["dt"], v["kp"], v["prel"]
                    miss += v["miss"]
            Mt = np.load(ROWS / P / "meta.npz", allow_pickle=True)
            k = Mt["k"]
            np.savez(O / f"on1_{P}.npz", names=T[P]["names"], f=f, k=k, dt=dt.astype(np.float32), k_prev=kp)
            F[P] = f
            val = f[:, 0] > 0.5
            d = np.abs(f.astype(np.float64) - fd.astype(np.float64))
            ph = T[P]["pose"][:, -2].astype(np.float64)                         # the tab's history pose at -0.5 s, current frame
            hd = np.abs(ph[val] - inv(prel[val]))
            hd[:, 2] = np.abs((hd[:, 2] + np.pi) % (2 * np.pi) - np.pi)
            dts = collections.Counter(np.round(dt[val] * 1e3).astype(int).tolist())
            summ["pools"][P] = stats(f) | dict(
                unmatched_rows=int(miss), invalid_with_k_ge_1=int(((k >= 1) & ~val).sum()), k0_rows=int((k == 0).sum()),
                prev_not_k_minus_1=int((val & (kp != k - 1)).sum()), dt_ms_counts={str(q): c for q, c in sorted(dts.items())},
                validation=dict(rows=int(np.isfinite(fd[:, 0]).sum()), max_abs_diff=float(np.nanmax(d)), per_feature_max={NAMES[c]: float(np.nanmax(d[:, c])) for c in range(14)},
                                valid_flag_agree=bool((fd[:, 0] == f[:, 0]).all())),
                pose_hist_vs_prev_anchor=dict(max_xy_m=float(hd[:, :2].max()), max_yaw_rad=float(hd[:, 2].max()), p99_xy_m=float(np.percentile(hd[:, :2].max(1), 99))) if val.any() else None)
            run.info(f"{P}: {json.dumps(summ['pools'][P])}")
        for P, parts in MERGED.items():
            Tm = dict(np.load(CR / f"on1_{P}" / "tab.npz"))
            names, pool = Tm["names"], Tm["pool"]
            f = np.zeros((len(names), 14), np.float32)
            chk = {}
            for pre, src in parts.items():
                i = np.flatnonzero(pool == pre)
                pos = {q: j for j, q in enumerate(T[src]["names"].tolist())}
                strip = [f"on:{q[len('on:' + pre + ':'):]}" for q in names[i].tolist()]
                assert all(q.startswith(f"on:{pre}:") for q in names[i].tolist())
                j = np.array([pos[q] for q in strip])
                chk[pre] = dict(src=src, rows=int(len(i)), served_max_diff=float(np.abs(Tm["served"][i] - T[src]["served"][j]).max()),
                                ego_max_diff=float(np.abs(Tm["ego"][i] - T[src]["ego"][j]).max()), speed_max_diff=float(np.abs(Tm["speed"][i] - T[src]["speed"][j]).max()))
                assert chk[pre]["served_max_diff"] == 0 and chk[pre]["ego_max_diff"] == 0, chk
                f[i] = F[src][j]
            assert set(np.unique(pool)) == set(parts), np.unique(pool)
            np.savez(O / f"on1_{P}.npz", names=names, f=f)
            summ["merged"][P] = dict(all=stats(f), map=chk, by_pool={q: stats(f[pool == q]) for q in parts},
                                     by_src={q: stats(f[Tm["src"] == q]) for q in np.unique(Tm["src"]).tolist()} if "src" in Tm else None)
            run.info(f"{P}: {json.dumps(summ['merged'][P])}")
        (O / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        run.summary.update(summ)


# ---------------------------------------------------------------- checkpoints and the offline replay
def cmd_ckpt(a):
    sys.path[:0] = [str(_R / "experiments/route_mem/lib"), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_parity/scripts"),
                    str(_R / "experiments/op_adapt_r2/lib")]
    import torch
    import pp_train as T
    import reflex as RX
    import rmem as RM
    side = json.loads((T.proot("runs", a.init) / RM.SIDECAR).read_text())
    assert "reflex" not in side, f"{a.init} already has the module"
    for tag, live in ((a.tag0, False), (a.tag1, True)):
        d = T.proot("runs", tag)
        assert not (d / "ckpt-final.pt").exists(), f"{d} exists: never overwrite"
        m = RM.load_pmodel(a.init, torch.device("cuda"))
        RX.attach(m)
        if live:
            g = torch.Generator().manual_seed(7)
            with torch.no_grad():
                m.adapter.rx.gate.fill_(1.0)
                w = m.adapter.rx.out.weight
                w.copy_(1e-3 * torch.randn(w.shape, generator=g).to(w.device))
        torch.save({"model": m.state(), "route_mem": side.get("head", {}), "reflex_test_of": a.init, "live": live}, d / "ckpt-final.pt")
        (d / RM.SIDECAR).write_text(json.dumps(side | {"reflex": {"sf": RX.SF, "d": 128}, "reflex_test_of": a.init,
                                                       "reflex_test": "gate 1, out ~ N(0, 1e-3), seed 7" if live else "zero init"}, indent=1))
        print("wrote", d)


def cmd_replay(a):
    sys.path.insert(0, str(Path(a.code) / "experiments/alpasim/lib"))
    import sh30_core as C
    assert Path(C.__file__).resolve().parents[3] == Path(a.code).resolve(), C.__file__
    core = C.Core(a.tag, "cuda", synth="cpu")
    rx = bool(getattr(core, "rx", False))
    M = _lib() if rx else None
    out = collections.defaultdict(list)
    for spec in a.sessions:
        run, scene = spec.split("::")
        u, st, recs = max((c for (sc, _), cs in sessions(run).items() if sc == scene for c in cs), key=lambda c: c[1]["n"])   # the last session
        dd = Path(run) / "driver-logs/dump"
        cam = C.camera(**{k: np.asarray(v) for k, v in st["cam"].items() if k in ("K", "R", "t", "D")})
        mem = M.Memory() if rx else None
        R = {r["k"]: r for r in recs}
        for r in recs:
            k = r["k"]
            e = np.asarray(r["ego"], np.float64)
            v = float(np.hypot(e[4], e[5])) * 10.0
            sf = mem.features(r["t0"], r["anchor"], v)[0] if rx and a.self else None
            fs = [dd / f"s{st['n']:02d}_k{q}.npz" for q in range(k - 3, k + 1)]
            if k >= 3 and r["n_keys"] == 4 and all(p.exists() for p in fs) and all(q in R for q in range(k - 3, k + 1)) \
                    and all(R[q + 1]["t0"] - R[q]["t0"] == 500_000 for q in range(k - 3, k)):
                keys = []
                for p in fs:
                    with np.load(p) as z:
                        keys.append(C.pack(z["jpeg"].tobytes(), cam))
                kw = dict(route=np.asarray(r["rmem"], np.float32)) if "rmem" in r else {}
                if rx:
                    kw["selfstate"] = sf
                o = core.plan(keys, np.asarray(r["hist"], np.float64), np.tile(e[4:6] * 10.0, (4, 1)), e[6:8] * 3.0, np.eye(4, dtype=np.float32)[r["cmd"]], cam["t"], **kw)
                out["scene"].append(scene), out["k"].append(k), out["mu"].append(o["mu"]), out["poses"].append(o["poses"])
                out["served"].append(np.asarray(r["poses"])), out["n_fed"].append(len(o["ego"]) + len(o.get("rmem", ())) + len(o.get("rx", ())))
                out["rx"].append(o["rx"] if "rx" in o else np.full(14, np.nan, np.float32))
            if rx:
                mem.add(r["t0"], r["anchor"], np.asarray(r["poses"], np.float64))       # the stored rollout's served plan, as the driver keeps it
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **{k: np.array(v) for k, v in out.items()}, tag=a.tag, code=a.code, off=os.environ.get("JEV_REFLEX_OFF", ""))
    print(f"{a.tag} ({a.code}): {len(out['k'])} decisions -> {a.out}; plan vs stored served max {np.abs(np.array(out['poses']) - np.array(out['served'])).max():.4f} m")


def cmd_cmp(a):
    L = {Path(p).stem: dict(np.load(p, allow_pickle=True)) for p in a.files}
    b, old = L[a.base], L[a.old]
    same = lambda x, y: bool(np.array_equal(x["mu"], y["mu"]) and np.array_equal(x["poses"], y["poses"]))  # noqa: E731
    near = lambda x: np.abs(x["poses"][:, :3, :2] - b["poses"][:, :3, :2])  # noqa: E731
    z, l, off = L[a.zero], L[a.live], L[a.off]
    assert all((x["k"] == b["k"]).all() and (x["scene"] == b["scene"]).all() for x in L.values())
    rz = z["rx"]
    chk = {
        "decisions": int(len(b["k"])), "scenes": sorted(set(b["scene"].tolist())),
        "1_old_vs_new_no_module_bit_identical": same(old, b), "1_max_abs_mu": float(np.abs(old["mu"] - b["mu"]).max()),
        "1_new_repeat_bit_identical": same(L["new_base2"], b) if "new_base2" in L else None,
        "replay_plan_vs_stored_served_max_m": float(np.abs(b["poses"] - b["served"]).max()),
        "2_zero_module_bit_identical": same(z, b), "2_max_abs_mu": float(np.abs(z["mu"] - b["mu"]).max()),
        "2_fed_row_lengths": sorted(set(z["n_fed"].tolist())), "2_base_fed_row_lengths": sorted(set(b["n_fed"].tolist())),
        "2_valid_share": float((rz[:, 0] == 1).mean()),
        "3_live_near_plan_shift_m": dict(max=float(near(l).max()), median=float(np.median(near(l).max((1, 2)))), at_s={t: float(near(l)[:, i].max()) for i, t in enumerate(("0.5", "1.0", "1.5"))}),
        "3_live_far_plan_shift_max_m": float(np.abs(l["poses"][:, 3:, :2] - b["poses"][:, 3:, :2]).max()),
        "3_live_bounded_le_0.5": bool(near(l).max() <= 0.5), "3_live_moves": bool(near(l).max() > 0),
        "3_off_bit_identical": same(off, b), "3_off_env": str(off["off"]), "3_off_valid_share": float((off["rx"][:, 0] == 1).mean()),
    }
    chk["pass"] = bool(chk["1_old_vs_new_no_module_bit_identical"] and chk["2_zero_module_bit_identical"] and chk["3_live_bounded_le_0.5"]
                       and chk["3_live_moves"] and chk["3_off_bit_identical"] and chk["2_valid_share"] == 1.0)
    O.mkdir(parents=True, exist_ok=True)
    (O / "replay_checks.json").write_text(json.dumps(chk, indent=1) + "\n")
    print(json.dumps(chk, indent=1))


def drive_records(run) -> dict:
    """{scene: drive records sorted by k} of the last session of each scene (a retried rollout: ao_e1.drive_records' rule)."""
    best = {}
    for (sc, n), cs in sessions(run).items():
        for _, st, recs in cs:
            recs = [r for r in recs if "poses" in r and "anchor" in r]          # PAI: the inference records only
            if recs and (sc not in best or n > best[sc][0]):
                best[sc] = (n, recs)
    return {sc: rr for sc, (n, rr) in best.items()}


def cmd_smoke(a):
    R = {Path(p).parent.name if Path(p).name in ("run", "sim") else Path(p).name: drive_records(p) for p in a.runs}
    base = R[a.base]
    res = {}
    for name, X in R.items():
        d, first = [], {}
        for sc in sorted(set(X) & set(base)):
            pb = {r["k"]: np.asarray(r["poses"]) for r in base[sc]}
            for r in X[sc]:
                if r["k"] in pb:
                    d.append(float(np.abs(np.asarray(r["poses"]) - pb[r["k"]]).max()))
                    if d[-1] > 0 and sc not in first:
                        first[sc] = r["k"]
        d = np.array(d)
        e = dict(decisions=len(d), scenes=len(set(X) & set(base)), equal_share=float((d == 0).mean()) if len(d) else None,
                 max_abs_m=float(d.max()) if len(d) else None, first_diff_k=first)
        if any("rx" in r for sc in X for r in X[sc]):              # logged rx against reflex_mem replayed over the same drive.jsonl
            dm, vk, nf, dts = 0.0, collections.defaultdict(list), collections.Counter(), set()
            for sc in X:
                tl = [dict(r, anchor=[*r["anchor"][:2], r["anchor"][2] + (r.get("track") or {}).get("tilt", 0.0)]) for r in X[sc]]   # tilted: the controller's yaw
                A = replay(tl, lambda r: np.asarray(r.get("poses_track", r["poses"]), np.float64),                # served: the JEV_TRACK reference when on
                           lambda r: float(np.hypot(*np.asarray(r["ego"][4:6], np.float64))) * 10.0)
                for r in X[sc]:
                    lg = np.asarray(r["rx"])
                    vk[min(r["k"], 1)].append(float(lg[0]))
                    nf[len(r["ego"]) + len(r.get("rmem", ())) + len(lg)] += 1
                    if lg[0] == 1:
                        dm = max(dm, float(np.abs(lg - A[r["k"]][0]).max()))
                    if r.get("rx_dt") is not None:
                        dts.add(r["rx_dt"])
            e["rx"] = dict(valid_k0=float(np.mean(vk[0])), valid_k_ge_1=float(np.mean(vk[1])), logged_vs_replay_max=dm,
                           fed_row_lengths={str(q): c for q, c in nf.items()}, dt_s=sorted(dts))
        res[name] = e
    O.mkdir(parents=True, exist_ok=True)
    (O / f"smoke_checks{a.suffix}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("feat")
    p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("ckpt")
    p.add_argument("--init", default="RSG-L18-s0"), p.add_argument("--tag0", default="STKRX0-test"), p.add_argument("--tag1", default="STKRX1-test")
    p = sub.add_parser("replay")
    p.add_argument("--code", required=True), p.add_argument("--tag", required=True), p.add_argument("--out", required=True)
    p.add_argument("--sessions", nargs="+", required=True, help="<run dir>::<scene>"), p.add_argument("--self", action="store_true")
    p = sub.add_parser("cmp")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--base", default="new_base"), p.add_argument("--old", default="old_base"), p.add_argument("--zero", default="zero")
    p.add_argument("--live", default="live"), p.add_argument("--off", default="off")
    p = sub.add_parser("smoke")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--base", required=True), p.add_argument("--suffix", default="")
    a = ap.parse_args()
    dict(feat=cmd_feat, ckpt=cmd_ckpt, replay=cmd_replay, cmp=cmd_cmp, smoke=cmd_smoke)[a.cmd](a)
