#!/usr/bin/env python3
"""Equivalence checks of the PAI driver's slot switch (lib/pai_core.py `slots(keywarp=...)`, PAI_KEYWARP), before any closed loop.
Plan: plans/2026-10-11-pai-keywarp-prereg.md; results: results/pai_keywarp.md.

  stream   a recorded PAI stream (the driver-side messages col1_pai_extract.py took out of a finished run, <dir>/<scene>.pkl) replayed
           through the real pai_driver.Driver with the switch off. At every decision, on the driver's own inputs:
             (i)  slots with the switch off against the file before the switch (--ref: `git show ef2f829d:.../pai_core.py`): same bytes
             (ii) slots with the switch on against sh30_core.lattice (the CPU reference path of training and of Core.plan) on the same
                  keyframes, pixel by pixel; on warm decisions the plan from them against sh30_core.Core.plan (the nuPlan driver's
                  entry, synth cpu and gpu) on the same four keyframes, with on-against-off as the scale of a real difference
           Open loop (the ego does not react). Runs inside the driver image on the Tokyo box:
             docker run --rm --gpus device=0 -v <lib>/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro (same for
               pai_driver.py, serve_fix.py) -v <ref pai_core.py>:/ref/pai_core.py:ro -v <msgs>:/msgs:ro -v <out>:/out -v <this>:/c.py:ro \\
               <image> python /c.py stream --msgs /msgs --ref /ref/pai_core.py --out /out [--limit 3]
  navtest  (iii) the offline frame-source table of experiments/body1/results/served_plan_length.md (600 tokens of lb_hq_navtestX, the
           same draw) through the driver's functions: each token's real 10 Hz frames as a stream -> pai_core.slots off / on ->
           pai_core.plan, next to the plans from the cached real / warp tokens and the bench's warp predictions. Ego features are the
           table's (the benchmark's own), so only the frame path differs. On the GPU box, inside jevdrive.run.Run, as a pool job:
             python -m jevdrive.cl submit --name pai-kw-check --vram 8 --cpu 8 -- $DATA_DIR/envs/op-train/bin/python \\
               experiments/alpasim/scripts/pai_keywarp_check.py navtest [--n 600] [--tag P2H10-F-s0]
"""
import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
LIB = Path("/app/jev-drive/experiments/alpasim/lib") if Path("/app/jev-drive/experiments/alpasim/lib/sh30_core.py").exists() else HERE.parents[1] / "lib"
sys.path.insert(0, str(LIB))
TOL = 0.03                                               # m: two fp16 ulps at 32 m (docs/long-runs.md, equality gates on plan points)


def dist(a, b):
    """Largest xy distance over the 8 poses (all within 4 s) of plans (..., 8, 3)."""
    return np.linalg.norm(np.asarray(a)[..., :2] - np.asarray(b)[..., :2], axis=-1).max(-1)


def q(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "median": float(np.median(x)), "p99": float(np.percentile(x, 99)), "max": float(x.max()),
            "share_over_tol": float((x > TOL).mean())} if len(x) else {"n": 0}


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def stream(a):
    import pai_driver as D
    PC, C, I = D.PC, D.C, D.I
    spec = importlib.util.spec_from_file_location("pai_core_ref", a.ref)
    REF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(REF)
    core = C.Core(a.tag, "cuda")
    out = Path(a.out)
    drv, pb, ctx = D.Driver(core, out / "driver", "backwarp", 1, 0, False), D.egodriver_pb2, Ctx()
    new_slots, new_plan, R, last = PC.slots, PC.plan, [], {}
    sync = lambda: (core.torch.cuda.synchronize(), time.perf_counter())[1]  # noqa: E731

    def slots(frames, t0, P, V, cam_t, cold, **kw):
        t = sync()
        off = new_slots(frames, t0, P, V, cam_t, cold)
        t_off = sync() - t
        ref = REF.slots(frames, t0, P, V, cam_t, cold)
        t = sync()
        cur, valid, real = new_slots(frames, t0, P, V, cam_t, cold, keywarp=True, dev=core.dev)
        t_on = sync() - t
        ts, keys, e = np.array(sorted(frames), np.int64), [], 4          # the keyframes, picked here independently of key_slots
        for i in (3, 2, 1, 0):
            n = int(ts[np.abs(ts - (t0 + PC.T_KEY_US[i])).argmin()])
            if abs(n - (t0 + int(PC.T_KEY_US[i]))) > 30_000:
                break
            keys.insert(0, frames[n])
            e = i
        K = np.zeros((4,) + C.FRAME, np.uint8)
        K[e:] = keys
        lat, lv = C.lattice(K, e, I.track_navsim(P, V), np.asarray(cam_t, np.float64), cold)
        d = cur.cpu().numpy()[valid].astype(np.int16) - lat[lv] if np.array_equal(valid, lv) else np.full(1, 255)
        last.update(cur=cur, valid=valid, keys=keys, e=e, off=off[0], row=dict(
            t0=int(t0), n_keys=4 - e, n_real_off=int(off[2].sum()), n_slots_on=int(real.sum()),
            off_same=bool(all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(off, ref))),
            on_valid_same=bool(np.array_equal(valid, lv)), px=int(d.size), px_diff=int((d != 0).sum()), px_maxabs=int(abs(d).max()),
            off_on_px_mean_abs=float(np.abs(cur.cpu().numpy().astype(np.int16) - off[0]).mean()), ms_off=1e3 * t_off, ms_on=1e3 * t_on))
        return off

    def plan(core_, cur, valid, P, V, acc, cmd, cam_t, lht=False):
        o = new_plan(core_, cur, valid, P, V, acc, cmd, cam_t, lht)
        row = last["row"]
        on = new_plan(core_, last["cur"], last["valid"], P, V, acc, cmd, cam_t, lht)["poses"]
        row.update(on_vs_off_m=float(dist(on, o["poses"])), speed=float(np.hypot(*V[-1])))
        if last["e"] == 0:                                               # warm: Core.plan's history fill is the identity
            for s in C.SYNTH:
                core_.synth = s
                row[f"on_vs_core_{s}_m"] = float(dist(on, core_.plan(last["keys"], P, V, acc, cmd, cam_t, lht=lht)["poses"]))
        R.append(dict(row, scene=last.get("scene", "")))
        return o

    PC.slots, PC.plan = slots, plan
    for f in sorted(Path(a.msgs).glob("*.pkl"))[:a.limit]:
        last["scene"], n, fail = f.stem, 0, 0
        for kind, raw in pickle.load(open(f, "rb")):
            if kind == "driver_session_request":
                drv.start_session(pb.DriveSessionRequest.FromString(raw), ctx)
            elif kind == "driver_camera_image":
                drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
            elif kind == "driver_ego_trajectory":
                drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
            elif kind == "route_request":
                drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
            elif kind == "driver_request":
                n += 1
                try:
                    drv.drive(pb.DriveRequest.FromString(raw), ctx)
                except RuntimeError as e:
                    fail += 1
                    print(f.stem, n, "drive failed:", e, flush=True)
        print(f.stem, n, "drive calls,", fail, "failed", flush=True)
    warm = [r for r in R if r["n_keys"] == 4]
    S = {"tag": a.tag, "scenes": sorted({r["scene"] for r in R}), "decisions": len(R), "warm_decisions": len(warm),
         "decisions_by_keys": {str(k): sum(r["n_keys"] == k for r in R) for k in (1, 2, 3, 4)},
         "i_off_identical_decisions": sum(r["off_same"] for r in R),
         "ii_on_validity_equal_decisions": sum(r["on_valid_same"] for r in R),
         "ii_on_pixels": sum(r["px"] for r in R), "ii_on_pixels_differing": sum(r["px_diff"] for r in R),
         "ii_on_pixel_maxabs": max(r["px_maxabs"] for r in R), "ii_decisions_with_a_differing_pixel": sum(r["px_diff"] > 0 for r in R),
         "ii_plan_on_vs_core_cpu_m": q([r["on_vs_core_cpu_m"] for r in warm]), "ii_plan_on_vs_core_gpu_m": q([r["on_vs_core_gpu_m"] for r in warm]),
         "control_plan_on_vs_off_m": q([r["on_vs_off_m"] for r in warm]),
         "control_plan_on_vs_off_m_above_3mps": q([r["on_vs_off_m"] for r in warm if r["speed"] > 3]),
         "off_on_pixel_mean_abs_warm": float(np.mean([r["off_on_px_mean_abs"] for r in warm])) if warm else None,
         "slots_ms_median_off_on_warm": [float(np.median([r[k] for r in warm])) for k in ("ms_off", "ms_on")] if warm else None, "tol_m": TOL}
    out.mkdir(parents=True, exist_ok=True)
    (out / "stream.json").write_text(json.dumps(S, indent=1))
    (out / "stream_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in R))
    print(json.dumps(S, indent=1))
    ok = S["i_off_identical_decisions"] == len(R) and S["ii_on_validity_equal_decisions"] == len(R) and \
        S["ii_plan_on_vs_core_cpu_m"].get("share_over_tol", 1) == 0 and S["ii_plan_on_vs_core_gpu_m"].get("share_over_tol", 1) == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def navtest(a):
    import pai_core as PC
    from pai_core import C, I
    import op_lb as OL
    from jevdrive import navsim_zs as Z
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("alpasim", "pai-kw-check", seed=0, config=vars(a)) as run:
        sp = splits.load("navsim/navtest")
        run.use_split(sp)
        c = data_dir() / "runs/op_parity/cache"
        tx, tn, mt = np.load(c / "lb_hq_navtestX/tab.npz"), np.load(c / "lb_navtest/tab.npz"), OL.meta("lb_hq_navtestX")
        assert (np.array(mt["names"]) == tx["names"]).all() and np.allclose(mt["syn_t"], [-1.4, -1.2, -0.8, -0.6, -0.4, -0.2])
        real = np.load(OL.root("lb_hq_navtestX") / "real.npy", mmap_mode="r")
        idx = Z.load_index("navtest", slim=True)                         # the keyframes: the tokens' CAM_F0 JPEGs (sh30_check.py parity)
        by = {e["token"]: e for e in idx}
        keys = lambda i: np.stack([C.pack_fast(by[tx["names"][i]]["cams"][f]["CAM_F0"]["path"], by[tx["names"][i]]["cams"][-1]["CAM_F0"]) for f in range(4)])  # noqa: E731
        rown = {t: i for i, t in enumerate(tn["names"].tolist())}
        ix = np.array([k for k, t in enumerate(tx["names"].tolist()) if t in rown])
        sel = np.sort(ix[np.random.default_rng(0).permutation(len(ix))[:a.n]])       # the draw of served_plan_length.md
        rn = np.array([rown[t] for t in tx["names"][sel].tolist()])
        assert sp.mask(tx["names"][sel]).all(), "tokens outside navsim/navtest"
        run.info("%d tokens; ego tables equal to %.2g", len(sel), float(np.abs(tx["ego"][sel] - tn["ego"][rn]).max()))
        core = C.Core(a.tag, "cuda")
        torch, T0 = core.torch, 2_000_000
        t_us = lambda t: T0 + int(round(float(t) * 1e6))  # noqa: E731
        P_, ego_row, ms = {k: [] for k in ("slots_off", "slots_on")}, [None], {"off": [], "on": []}
        PC.PA.ego_features = lambda *_: ego_row[0]                       # the benchmark's ego features (its 4-step acceleration history)
        px = []
        for i in run.tqdm(sel, desc="tokens"):
            kf = keys(int(i))
            fr = {t_us(t): kf[j] for j, t in enumerate(I.T_KEY)} | {t_us(t): np.asarray(real[i, j]) for j, t in enumerate(mt["syn_t"])}
            P, V, cam, ego_row[0] = tx["pose"][i].astype(np.float64), tx["vel"][i].astype(np.float64), tx["cam"][i].astype(np.float64), tx["ego"][i]
            for k, kw in (("off", {}), ("on", dict(keywarp=True, dev=core.dev))):
                t = time.perf_counter()
                cur, valid, rl = PC.slots(fr, T0, P, V, cam, "backwarp", **kw)
                assert rl.all() and valid.all()
                o = PC.plan(core, cur, valid, P, V, tx["acc"][i, -1], tx["cmd"][i, -1], cam, bool(tx["lht"][i]))
                ms[k].append(1e3 * (time.perf_counter() - t))
                P_[f"slots_{k}"].append(o["poses"])
                if k == "on":                                            # against the CPU reference lattice of the same four keys
                    d = cur.cpu().numpy().astype(np.int16) - C.lattice(kf, 0, I.track_navsim(P, V), cam, "backwarp")[0]
                    px.append(((d != 0).sum(), d.size, abs(d).max()))

        def cached(front, tab, rows):                                    # the same policy on cached tokens, fp16 on the card
            out, s = [], core.pi
            with torch.no_grad():
                for b in range(0, len(rows), 64):
                    r = rows[b:b + 64]
                    tc = torch.tensor(np.where(tab["lht"][r][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32), device=core.dev)
                    H = torch.from_numpy(np.asarray(front[r])).to(core.dev)
                    mu = core.model(H, torch.from_numpy(tab["ego"][r]).to(core.dev), tc).float()[:, s].reshape(-1, 33, 15).cpu().numpy()
                    out += [I.to_rear(mu[j, :, 0:3], mu[j, :, 11], I.T_IDXS, tab["cam"][k, :2].astype(np.float64), Z.T_OUT, "lever") for j, k in enumerate(r)]
            return np.array(out)
        P_ = {k: np.array(v) for k, v in P_.items()}
        P_["cache_real"] = cached(np.load(c / "lb_hq_navtestX@real/front.npy", mmap_mode="r"), tx, sel)
        P_["cache_warp"] = cached(np.load(c / "lb_navtest@warp/front.npy", mmap_mode="r"), tn, rn)
        bp = data_dir() / "runs/bench/ol/lb_navtest/preds"
        for s in (0, 1):
            f = bp / f"P2H10-F-s{s}-warp__base.npz"
            if f.exists():
                P_[f"bench_warp_s{s}"] = np.load(f)["poses"][rn]
        fut = tn["fut"][rn]
        ok = ~np.isnan(fut).any((1, 2))
        arc = lambda p: np.linalg.norm(np.diff(np.concatenate([np.zeros(p.shape[:-2] + (1, 2)), p[..., :2]], -2), axis=-2), axis=-1).sum(-1)  # noqa: E731
        la = arc(fut[ok]).sum()
        rows = [dict(frames=k, n=int(ok.sum()), arc_over_log=float(arc(v[ok]).sum() / la),
                     ade_m=float(np.linalg.norm(v[ok][..., :2] - fut[ok][..., :2], axis=-1).mean())) for k, v in P_.items()]
        pairs = [("slots_on", "cache_warp"), ("slots_on", f"bench_warp_s{a.tag[-1]}"), ("slots_off", "cache_real"), ("slots_on", "slots_off"),
                 ("slots_on", f"bench_warp_s{1 - int(a.tag[-1])}")]
        eq = [dict(pair=f"{x} vs {y}", **q(dist(P_[x], P_[y]))) for x, y in pairs if x in P_ and y in P_]
        from jevdrive import stats
        stats.write_table(rows, run.path("frame_source"), floatfmt=".4f")
        stats.write_table(eq, run.path("plan_equality"), floatfmt=".4f", note=f"largest xy distance over the 8 poses per token, m; tolerance {TOL} m")
        on = next(r for r in rows if r["frames"] == "slots_on")
        npx, tot, mx = (int(sum(x[0] for x in px)), int(sum(x[1] for x in px)), int(max(x[2] for x in px)))
        res = dict(tag=a.tag, n=int(ok.sum()), rows=rows, equality=eq, pixels_on_vs_cpu_lattice=dict(differing=npx, total=tot, maxabs=mx),
                   ms_median=dict(off=float(np.median(ms["off"])), on=float(np.median(ms["on"]))),
                   iii_pass=bool(abs(on["arc_over_log"] - 0.994) <= 0.01 and abs(on["ade_m"] - 0.61) <= 0.03))
        run.path("navtest.json").write_text(json.dumps(res, indent=1))
        np.savez_compressed(run.path("poses.npz"), tokens=tx["names"][sel], fut=fut, **P_)
        for r in rows:
            run.info("%-14s arc / log %.4f  ADE %.3f m", r["frames"], r["arc_over_log"], r["ade_m"])
        for r in eq:
            run.info("%-34s median %.4f  max %.4f  share > %.2f m: %.4f", r["pair"], r["median"], r["max"], TOL, r["share_over_tol"])
        run.info("pixels on vs CPU lattice: %d of %d differ (max %d); (iii) %s", npx, tot, mx, "PASS" if res["iii_pass"] else "FAIL")
        run.summary.update(iii_pass=res["iii_pass"], on_arc=on["arc_over_log"], on_ade=on["ade_m"])
        if not res["iii_pass"]:
            raise SystemExit("check (iii) failed: the switch does not recover the warp-frame plan length / ADE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stream")
    s.add_argument("--msgs", required=True), s.add_argument("--ref", required=True), s.add_argument("--out", required=True)
    s.add_argument("--tag", default="P2H10-F-s0"), s.add_argument("--limit", type=int, default=3)
    n = sub.add_parser("navtest")
    n.add_argument("--n", type=int, default=600), n.add_argument("--tag", default="P2H10-F-s0")
    a = ap.parse_args()
    {"stream": stream, "navtest": navtest}[a.cmd](a)
