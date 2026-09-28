#!/usr/bin/env python
"""openpilot on a 2 Hz benchmark history: frame synthesis and contract fixes (research/openpilot-openloop-integration.md).
Run dir: $DATA_DIR/runs/op_interp/{wod,nav}/.

  wod-cache   (envs/openpilot) the 479 WOD-E2E val rater frames: the 16 real 10 Hz frames t = -1.5 ... 0 rendered as the
              exam (scripts/wod_openpilot_timeline.py, `ctx1.5`) -> wod/real.npy (479, 16, 2, 6, 128, 256) + meta.json
  nav-cache   (envs/openpilot) a seeded navtest subset: its four exam keyframes (navsim_zs frames cache) -> nav/keys.npy,
              the same keyframes rendered with a nominal (uncalibrated, axis-aligned) CAM_F0 -> nav/keys_nominal.npy
  synth       history synthesis from the four keyframes, any method of jevdrive.op_interp (rife / gimm in envs/vfi on a
              GPU); WOD also gets image metrics against the real 10 Hz frames -> <data>/<frames>.npy + .json
  run         (envs/openpilot) one model over one frame file, zero state, plan at t0 -> <data>/plans/<frames>@<model>.npz
  score-wod   (envs/jevdrive) RFS / ADE / longitudinal bias / plan drift per plan file and output adapter
  nav-export  (envs/jevdrive) NAVSIM pose files per plan file x adapter -> nav/preds/<name>.npz, tokens in nav/tokens.txt
  nav-report  (envs/jevdrive) PDMS from the devkit's per-token CSVs (scripts/op_interp_score.sh), paired deltas

    PY=$DATA_DIR/envs/openpilot/bin/python; VF=$DATA_DIR/envs/vfi/bin/python
    $PY scripts/op_interp.py wod-cache && $VF scripts/op_interp.py synth --data wod --method rife
    CUDA_VISIBLE_DEVICES=2 $PY scripts/op_interp.py run --data wod --frames rife --model cinque
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

VFI_ROOT = data_dir() / "third_party" / "vfi"
BACKENDS = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}      # the exams' backends
ACTION_T = (0.275, 0.525)
LHT_MAPS = {"sg-one-north"}


def root(*p) -> Path:
    d = data_dir() / "runs" / "op_interp" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def meta(data):
    return json.loads((root(data) / "meta.json").read_text())


# ---------------------------------------------------------------- caches

def cmd_wod_cache(a):
    from concurrent.futures import ProcessPoolExecutor
    import wod_openpilot_rigs as R
    import wod_openpilot_timeline as T
    from jevdrive import wod_zeroshot as Z
    r = Z.load_sets()["rater"]
    names = [str(x) for x in r["name"]]
    spans, _ = Z.load_spans()
    cal = json.loads((Z.root() / "op_calib.json").read_text())
    out = np.lib.format.open_memmap(root("wod") / "real.npy", "w+", np.uint8, (len(names), 16, 2, 6, 128, 256))
    t0 = time.time()
    shard = data_dir() / "datasets" / "waymo_e2e" / "front3"
    with ProcessPoolExecutor(a.workers, initializer=R._init, initargs=(spans, cal, str(shard), ["base"])) as ex:
        for i, (n, fr) in enumerate(ex.map(T.render, names, chunksize=2)):
            assert n == names[i] and fr["ctx1.5"].shape[0] == 16
            out[i] = fr["ctx1.5"]
    out.flush()
    cams = [np.array(cal[n.rsplit("-", 1)[0]]["1"]["extrinsic"]).reshape(4, 4)[:3, 3].tolist() for n in names]
    (root("wod") / "meta.json").write_text(json.dumps({"names": names, "cam": cams, "speed": np.linalg.norm(
        r["past"][:, -1, 2:4], axis=-1).tolist()}))
    np.save(root("wod") / "past.npy", r["past"])
    print(f"{len(names)} frames in {time.time() - t0:.0f} s")


def _nominal_render(e):
    from jevdrive import navsim_zs as Z
    cam = dict(e["cams"][-1]["CAM_F0"])
    cam["R"] = I.OPENCV_TO_VEHICLE.astype(np.float32)
    m = Z.OpenpilotMaps(cam)
    return np.stack([m(m.decode(e["cams"][f]["CAM_F0"]["path"])) for f in range(4)])


def cmd_nav_cache(a):
    from concurrent.futures import ProcessPoolExecutor
    from jevdrive import navsim_zs as Z
    idx = Z.load_index("navtest", slim=True)
    rng = np.random.default_rng(a.seed)
    sel = np.sort(rng.choice(len(idx), a.n, replace=False))
    frames = np.load(Z.root("openpilot", "navtest") / "frames.npy", mmap_mode="r")
    assert json.loads((Z.root("openpilot", "navtest") / "frames.json").read_text())["tokens"] == [e["token"] for e in idx]
    np.save(root("nav") / "keys.npy", np.ascontiguousarray(frames[sel]))
    sub = [idx[k] for k in sel]
    with ProcessPoolExecutor(a.workers) as ex:
        nom = np.stack(list(ex.map(_nominal_render, sub, chunksize=8)))
    np.save(root("nav") / "keys_nominal.npy", nom)
    toks = [e["token"] for e in sub]
    (root("nav") / "tokens.txt").write_text("\n".join(toks) + "\n")
    (root("nav") / "meta.json").write_text(json.dumps({
        "names": toks, "index": sel.tolist(), "cam": [np.asarray(e["cams"][-1]["CAM_F0"]["t"], float).tolist() for e in sub],
        "pose": [np.asarray(e["pose"], float).tolist() for e in sub], "vel": [np.asarray(e["vel"], float).tolist() for e in sub],
        "speed": [float(np.linalg.norm(e["vel"][-1])) for e in sub], "lht": [e["map"] in LHT_MAPS for e in sub],
        "cmd": [int(np.argmax(e["cmd"][-1])) for e in sub]}))
    print(f"{len(sel)} tokens cached")


# ---------------------------------------------------------------- synthesis

def keys_of(data, variant=""):
    if data == "wod":
        return np.load(root("wod") / "real.npy", mmap_mode="r")[:, [0, 5, 10, 15]]
    return np.load(root("nav") / f"keys{'_' + variant if variant else ''}.npy", mmap_mode="r")


def tracks(data):
    mt = meta(data)
    if data == "wod":
        past = np.load(root("wod") / "past.npy")
        return [I.track_wod(p) for p in past]
    return [I.track_navsim(p, v) for p, v in zip(mt["pose"], mt["vel"])]


def _cpu_job(args):
    keys, method, times, tr, cam = args
    return I.synth_cpu(keys, method, times, tr, cam)


def cmd_synth(a):
    from concurrent.futures import ProcessPoolExecutor
    keys = keys_of(a.data, a.keys)
    mt = meta(a.data)
    times = I.grid(-1.5 - a.preroll)
    tag = a.method + (f"_pre{a.preroll:g}" if a.preroll else "") + (f"_{a.keys}" if a.keys else "")
    out = np.lib.format.open_memmap(root(a.data) / f"{tag}.npy", "w+", np.uint8, (len(keys), len(times), 2, 6, 128, 256))
    t0 = time.time()
    if a.method in ("hold", "blend", "warp"):
        tr = tracks(a.data)
        with ProcessPoolExecutor(a.workers) as ex:
            jobs = ((np.asarray(keys[i]), a.method, times, tr[i], mt["cam"][i]) for i in range(len(keys)))
            for i, fr in enumerate(ex.map(_cpu_job, jobs, chunksize=4)):
                out[i] = fr
        dev = "cpu"
    else:
        import torch
        model = {"rife": I.RIFE, "gimm": I.GIMM}[a.method](VFI_ROOT)
        pre = times < -1.5 - 1e-6                    # preroll frames (before the first keyframe) come from warp
        for q in range(0, len(keys), a.chunk):
            out[q:q + a.chunk, ~pre] = I.synth_vfi(np.asarray(keys[q:q + a.chunk]), model, times[~pre],
                                                  batch=a.batch or {"rife": 64, "gimm": 8}[a.method])
            torch.cuda.synchronize()
        if pre.any():
            tr = tracks(a.data)
            with ProcessPoolExecutor(a.workers) as ex:
                jobs = ((np.asarray(keys[i]), "warp", times[pre], tr[i], mt["cam"][i]) for i in range(len(keys)))
                for i, fr in enumerate(ex.map(_cpu_job, jobs, chunksize=4)):
                    out[i, pre] = fr
        dev = torch.cuda.get_device_name()
    out.flush()
    el = time.time() - t0
    info = {"times": times.tolist(), "method": a.method, "preroll": a.preroll, "keys": a.keys, "n": len(keys),
            "wall_s": el, "s_per_scene": el / len(keys), "device": dev,
            "workers": a.workers if a.method in ("hold", "blend", "warp") else 1}
    (root(a.data) / f"{tag}.json").write_text(json.dumps(info))
    print(json.dumps(info | {"times": len(times)}))
    if a.data == "wod" and not a.preroll:
        image_metrics(tag)


def image_metrics(tag):
    """PSNR of the synthesized luma against the real 10 Hz frame, per non-key slot and view (road / wide)."""
    import pandas as pd
    real = np.load(root("wod") / "real.npy", mmap_mode="r")
    syn = np.load(root("wod") / f"{tag}.npy", mmap_mode="r")
    rows = []
    for j, t in enumerate(I.T10):
        if np.isclose(I.T_KEY, t).any():
            continue
        for v, view in enumerate(("road", "wide")):
            ya = I.unpack(np.asarray(real[:, j, v]))[0].astype(np.float32)
            yb = I.unpack(np.asarray(syn[:, j, v]))[0].astype(np.float32)
            mse = ((ya - yb) ** 2).mean((1, 2))
            psnr = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))
            rows.append({"method": tag, "t": t, "view": view, "psnr_mean": psnr.mean(), "psnr_p10": np.percentile(psnr, 10)})
    df = pd.DataFrame(rows)
    df.to_csv(root("wod", "img") / f"{tag}.csv", index=False)
    print(df.groupby("view")[["psnr_mean", "psnr_p10"]].mean().round(2).to_string())


# ---------------------------------------------------------------- openpilot

def _run_tag(a):
    return f"{a.frames}@{a.model}" + (f"_start{a.start:g}" if a.start is not None else "") + (f"_{a.backend}" if a.backend else "")


def cmd_run(a):
    """One process per shard: a single ORT/TensorRT session is bound by one CPU core's launch overhead on this box, so
    `--procs` sessions share the card and the shards are merged in order."""
    if a.procs > 1 and not a.shard:
        import subprocess
        argv = [sys.executable, __file__, "run", "--data", a.data, "--frames", a.frames, "--model", a.model] + \
               (["--backend", a.backend] if a.backend else []) + (["--start", str(a.start)] if a.start is not None else [])
        ps = [subprocess.Popen(argv + ["--shard", f"{k}/{a.procs}"]) for k in range(a.procs)]
        assert all(p.wait() == 0 for p in ps), "a shard failed"
        tag = _run_tag(a)
        parts = [root(a.data, "plans") / f"{tag}.part{k}of{a.procs}.npz" for k in range(a.procs)]
        zs = [dict(np.load(f)) for f in parts]
        out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if k not in ("names", "steps", "ms_per_scene")}
        np.savez(root(a.data, "plans") / f"{tag}.npz", names=np.array(meta(a.data)["names"]), steps=zs[0]["steps"],
                 ms_per_scene=np.mean([z["ms_per_scene"] for z in zs]), **out)
        for f in parts:
            f.unlink()
        return
    from jevdrive.openpilot.model import OPModel, decode
    fr = np.load(root(a.data) / f"{a.frames}.npy", mmap_mode="r")
    times = np.array(json.loads((root(a.data) / f"{a.frames}.json").read_text())["times"]) if a.frames != "real" else I.T10
    mt = meta(a.data)
    cr = a.model == "lebowski"
    backend = a.backend or BACKENDS[a.model]
    m = OPModel(a.model, backend, cache=root("trt_cache") / f"{a.model}-{backend}", context_rate=cr)
    sched = I.schedule(times, cr)
    if a.start is not None:                        # warm-up ablation: drop the frames before `start`
        sched = [s for s, t in zip(sched, _step_times(times, cr)) if t >= a.start - 1e-6]
    n = len(fr)
    k, K = map(int, (a.shard or "0/1").split("/"))
    rows = np.array_split(np.arange(n), K)[k]
    P = {"plan_pos": np.zeros((n, 33, 3), np.float32), "plan_vel": np.zeros((n, 33, 3), np.float32),
         "plan_yaw": np.zeros((n, 33), np.float32), "lead_prob": np.zeros((n, 3), np.float32)}
    lht = mt.get("lht", [False] * n)
    t0, tg = time.time(), 0.0
    for i in rows:
        f = np.ascontiguousarray(fr[i])
        tc = (0, 1) if lht[i] else (1, 0)
        t = time.perf_counter()
        m.reset()
        for s in sched:
            raw = m.step(f[s], traffic=tc, action_t=ACTION_T)
        tg += time.perf_counter() - t
        d = decode(raw, m.slices, float(mt["speed"][i]), ACTION_T)
        for q in ("plan_pos", "plan_vel", "plan_yaw"):
            P[q][i] = d[q]
        P["lead_prob"][i] = d["lead_prob"]
    tag = _run_tag(a) + (f".part{k}of{K}" if a.shard else "")
    P = {q: v[rows] for q, v in P.items()}
    np.savez(root(a.data, "plans") / f"{tag}.npz", names=np.array(mt["names"])[rows], steps=len(sched),
             ms_per_scene=1e3 * tg / len(rows), **P)
    print(f"{tag}: {len(rows)} scenes, {len(sched)} steps, {1e3 * tg / len(rows):.1f} ms/scene, {time.time() - t0:.0f} s wall")


def _step_times(times, cr):
    dt = 0.2 if cr else 0.05
    k = len(I.schedule(times, cr))
    return np.round(np.arange(-k + 1, 1) * dt, 3)


# ---------------------------------------------------------------- readouts

ADAPTERS = {"base": dict(mode="lever"), "retime": dict(mode="lever", retime=True), "nolever": dict(mode="none"),
            "shift": dict(mode="shift"), "cubic": dict(mode="lever", interp="cubic")}


def adapt(z, i, mt, t_out, mode="lever", retime=False, interp="linear"):
    T_IDXS = I.T_IDXS
    pos, yaw = z["plan_pos"][i], z["plan_yaw"][i]
    r = 1.0
    if retime:
        pos, yaw, r = I.retime(pos, yaw, z["plan_vel"][i], T_IDXS, float(mt["speed"][i]))
    return I.to_rear(pos, yaw, T_IDXS, mt["cam"][i][:2], t_out, mode, interp), r


def _cluster_ci(v, codes, rng, B=5000):
    """95% CI of the cluster mean, frames resampled within each cluster (jevdrive.openloop_standing._cluster_ci)."""
    reps = np.zeros(B)
    for c in range(codes.max() + 1):
        g = np.flatnonzero(codes == c)
        reps += v[g][rng.integers(0, len(g), (B, len(g)))].mean(1)
    return tuple(np.percentile(reps / (codes.max() + 1), [2.5, 97.5]))


def cmd_score_wod(a):
    import pandas as pd
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    mt = meta("wod")
    r = Z.load_sets()["rater"]
    assert [str(x) for x in r["name"]] == mt["names"]
    traj, sc, cl = r["traj"], r["scores"], r["cluster"].astype(str)
    speed, log_xy = W.init_speed(r["past"]), r["future"][..., :2]
    best = traj[np.arange(len(traj)), sc.argmax(1)]
    rng = np.random.default_rng(0)
    fidx = rng.integers(0, len(traj), (a.boot, len(traj)))
    per, preds = {}, {}
    for f in sorted(root("wod", "plans").glob("*.npz")):
        z = np.load(f)
        for ad in a.adapters:
            out = [adapt(z, i, mt, Z.T_FUT, **ADAPTERS[ad]) for i in range(len(traj))]
            p = np.stack([o[0][:, :2] for o in out])
            key = f"{f.stem}|{ad}"
            preds[key] = p
            per[key] = dict(rfs=W.rater_feedback_score(p, traj, sc, speed), ade5=np.linalg.norm(p - best, axis=-1).mean(-1),
                            lon5=p[:, -1, 0] - log_xy[:, -1, 0], r=np.array([o[1] for o in out]))
    for m in ("small", "cinque", "lebowski"):            # the earlier exam runs of the same feeds, as a reproduction check
        for v in ("ctx1.5", "nav2hz"):
            d = Z.root("preds", f"op_{m}@{v}")
            if all((d / f"{n}.npz").exists() for n in mt["names"]):
                p = np.stack([np.load(d / f"{n}.npz")["wod"] for n in mt["names"]])
                preds[f"exam {v}@{m}|base"] = p
                per[f"exam {v}@{m}|base"] = dict(rfs=W.rater_feedback_score(p, traj, sc, speed),
                                                 ade5=np.linalg.norm(p - best, axis=-1).mean(-1),
                                                 lon5=p[:, -1, 0] - log_xy[:, -1, 0], r=np.ones(len(p)))
    codes = pd.factorize(cl)[0]
    rows = []
    for k, q in per.items():
        frames, rest = k.split("@")
        model = rest.split("|")[0].split("_")[0]
        lo, hi = _cluster_ci(q["rfs"], codes, np.random.default_rng(0))
        row = dict(frames=frames, model=rest.split("|")[0], adapter=k.split("|")[1], n=len(q["rfs"]),
                   rfs=float(pd.Series(q["rfs"]).groupby(cl).mean().mean()), rfs_lo=lo, rfs_hi=hi, rfs_frame=float(q["rfs"].mean()),
                   ade5=float(q["ade5"].mean()), lon5=float(q["lon5"].mean()), lon5_moving=float(q["lon5"][speed > 5].mean()),
                   retime_r_med=float(np.median(q["r"])))
        for ref in ("real", "hold"):
            rk = f"{ref}@{model}|base"
            if rk in per and rk != k:
                dd = q["rfs"] - per[rk]["rfs"]
                bs = dd[fidx].mean(1)
                row |= {f"d_{ref}": dd.mean(), f"d_{ref}_lo": np.percentile(bs, 2.5), f"d_{ref}_hi": np.percentile(bs, 97.5),
                        f"drift_{ref}": float(np.linalg.norm(preds[k] - preds[rk], axis=-1).mean())}
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["model", "adapter", "rfs"], ascending=[True, True, False])
    ref = res[res.adapter == "base"].set_index(["frames", "model"])["rfs"]
    res["recovered"] = [(r.rfs - ref.get(("hold", r.model), np.nan)) / (ref.get(("real", r.model), np.nan) - ref.get(("hold", r.model), np.nan))
                        for r in res.itertuples()]   # share of the real-1.5 s minus hold gap won back
    res.to_csv(root("wod") / "results.csv", index=False)
    np.savez_compressed(root("wod") / "rfs_per_frame.npz", **{k.replace("|", "__"): v["rfs"] for k, v in per.items()})
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(res.round(3).to_string(index=False))


def cmd_nav_export(a):
    from jevdrive import navsim_zs as Z
    mt = meta("nav")
    for f in sorted(root("nav", "plans").glob("*.npz")):
        if a.plans and f.stem not in a.plans:
            continue
        z = np.load(f)
        assert z["names"].tolist() == mt["names"]
        for ad in a.adapters:
            out = root("nav", "preds") / f"{f.stem}__{ad}.npz".replace("@", "-")
            if out.exists() and not a.force:
                continue
            poses = np.stack([adapt(z, i, mt, Z.T_OUT, **ADAPTERS[ad])[0] for i in range(len(mt["names"]))])
            np.savez(out, tokens=np.array(mt["names"]), poses=poses)
            print(out.name)


def cmd_nav_report(a):
    import glob
    import pandas as pd
    subs = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
    toks = set(meta("nav")["names"])
    res = {}
    for d in sorted(glob.glob(str(data_dir() / "runs/navsim/eval" / "v1_navtest_opi_*"))):
        fs = sorted(glob.glob(d + "/*/*.csv"))
        if not fs:
            continue
        df = pd.read_csv(fs[-1])
        df = df[df["token"].isin(toks) & df["valid"].astype(bool)]
        res[Path(d).name[len("v1_navtest_opi_"):]] = df.set_index("token")
    # the exam's full-navtest runs on the same tokens (reproduction check and reference rows)
    for name in ("cinque_none", "lebowski_none", "small_none", "cv", "human", "heads_cls_late_cinque_temporal",
                 "heads_cls_ego_K1024"):
        fs = sorted(glob.glob(str(data_dir() / "runs/navsim/eval" / f"v1_navtest_{name}" / "*" / "*.csv")))
        if fs:
            df = pd.read_csv(fs[-1])
            res[f"exam {name}"] = df[df["token"].isin(toks) & df["valid"].astype(bool)].set_index("token")
    rows = []
    rng = np.random.default_rng(0)
    for k, df in res.items():
        v = df["score"].to_numpy(float)
        bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
        row = dict(name=k, n=len(v), pdms=100 * v.mean(), lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                   **{s: 100 * df[s].mean() for s in subs})
        for ref in a.refs:
            if ref in res and ref != k:
                x, y = df["score"].align(res[ref]["score"], join="inner")
                dd = (x - y).to_numpy(float)
                bb = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
                row |= {f"d[{ref}]": 100 * dd.mean(), f"d[{ref}]_lo": 100 * np.percentile(bb, 2.5),
                        f"d[{ref}]_hi": 100 * np.percentile(bb, 97.5)}
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("pdms", ascending=False)
    out.to_csv(root("nav") / "results.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("wod-cache")
    p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("nav-cache")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("synth")
    p.add_argument("--data", choices=("wod", "nav"), required=True)
    p.add_argument("--method", choices=I.METHODS, required=True)
    p.add_argument("--preroll", type=float, default=0.0)
    p.add_argument("--keys", default="", help="nav keyframe variant, e.g. nominal")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--batch", type=int, default=0, help="VFI pairs per forward (0: 64 RIFE, 8 GIMM, <= ~12 GB)")
    p = sp.add_parser("run")
    p.add_argument("--data", choices=("wod", "nav"), required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--model", choices=list(BACKENDS), required=True)
    p.add_argument("--backend", default="")
    p.add_argument("--start", type=float, default=None, help="first step time (warm-up ablation)")
    p.add_argument("--procs", type=int, default=3, help="ORT sessions in parallel (shards)")
    p.add_argument("--shard", default="", help="k/K: run only shard k (set by --procs)")
    p = sp.add_parser("score-wod")
    p.add_argument("--adapters", nargs="+", default=["base", "retime"])
    p.add_argument("--boot", type=int, default=5000)
    p = sp.add_parser("nav-export")
    p.add_argument("--adapters", nargs="+", default=["base"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--plans", nargs="*", default=[], help="plan file stems (default: all)")
    p = sp.add_parser("nav-report")
    p.add_argument("--refs", nargs="+", default=["hold-cinque__base"])
    a = ap.parse_args()
    {"wod-cache": cmd_wod_cache, "nav-cache": cmd_nav_cache, "synth": cmd_synth, "run": cmd_run, "score-wod": cmd_score_wod,
     "nav-export": cmd_nav_export, "nav-report": cmd_nav_report}[a.cmd](a)
