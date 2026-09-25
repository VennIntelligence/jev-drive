"""Elicitation E5: a 20 Hz student of the M-C reaction head on the fast channel (todos/2026-09-26-elicitation-program.md,
E5 and deviation-log entry [E5] 01:40, written before any E5 number).

  imglist   every P5 v1 BA index row x 3 cameras (current frame) -> processed/elicit_e5/images.parquet; the detector
            runs on it with `jevdrive.fastperc detect --full` (envs/ultralytics, YOLO26x-seg 640 fp16)
  embed     detections -> flat-ground BEV (fusion_q4.lift, vehicle origin) -> route-centreline corridor (|d| <= 4 m,
            0 < s <= 40 m, from route.json + pose.jsonl only) -> k = 8 nearest by s, 7 features + mask bit = 64 dims
  fit       per model and route fold: prior and teacher from reactivity_mc.fit_fold (checked against the stored run),
            MLP student [z_op, e] -> Delta (arms A: paired difference, B: + teacher Delta), seeds 0-2; the exam is
            p5_exam.exam + reactivity_mc.criteria unchanged, with the stored teacher and `pair op` control alongside
  latency   end-to-end on the GPU, batch 1: YOLO 3 cameras fp16 + lift / corridor / embedding + MLP

Run on the box: P5_SET=carla_p5v1_ba python -m jevdrive.elicit_e5 <step>
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
K_DET, HALF_W, REACH, ROUTE_LEN = 8, 4.0, 40.0, 60.0
CLASSES = ("pedestrian", "cyclist", "vehicle")
YOLO = "yolo:yolo26x-seg.pt:640:half"
SCORE = 0.25
MC_RUN = "runs/reactivity/mc-carla_p5v1_ba/20260925-233126"
HIDDEN, LR, WD, MAX_STEPS, EVAL_EVERY, PATIENCE, TEACHER_W = 256, 1e-3, 1e-4, 3000, 25, 300, 1.0
SEEDS = (0, 1, 2)


def root(*p) -> Path:
    d = data_dir() / "processed" / "elicit_e5"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def _index() -> pd.DataFrame:
    assert os.environ.get("P5_SET") == "carla_p5v1_ba", "E5 is registered on the P5 v1 BA set: set P5_SET"
    from . import p5_pairs as P
    return pd.read_parquet(P.processed() / "index.parquet")


def imglist() -> pd.DataFrame:
    from .fusion_q4 import CAMS
    t = _index()
    rows = []
    for r in t.itertuples():
        f = list(r.files)
        for i, cam in enumerate(CAMS):
            assert f"/{cam}/" in f[4 * i + 3], f[4 * i + 3]
            rows.append({"key": f"{r.frame_name}|{cam}", "path": f[4 * i + 3]})
    out = pd.DataFrame(rows)
    out.to_parquet(root("images.parquet"), index=False)
    for i, sl in enumerate(np.array_split(np.arange(len(out)), 3)):      # three detector processes, disjoint slices
        out.iloc[sl].to_parquet(root(f"images_{i}.parquet"), index=False)
    log.info("%d rows x 3 cameras = %d images", len(t), len(out))
    return out


# ---------------------------------------------------------------- embedding

def _route_path(adir: Path, frame: int, cache: dict):
    """Ego-frame route centreline ahead (vehicle origin, +x forward, +y left), fusion_diag.gt_run's construction."""
    if adir not in cache:
        pose = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame")
        route = pd.read_json(adir / "route.json")
        cache[adir] = (pose, np.c_[route.x.to_numpy(), -route.y.to_numpy()], -route.yaw.to_numpy())
    pose, rxy, ryaw = cache[adir]
    if frame not in pose.index:
        return None
    e = pose.loc[frame]
    th = -np.radians(e.yaw)
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, s_], [-s_, c]])
    exy = np.array([e.x, -e.y])
    dist = np.hypot(*(rxy - exy).T)
    dist[np.abs((ryaw - np.degrees(th) + 180) % 360 - 180) > 90] = np.inf
    j = int(dist.argmin())
    pts = (rxy[j:] - exy) @ R.T
    fwd = pts[:, 0] > 0
    pts = pts[int(fwd.argmax()):] if fwd.any() else pts[-1:]
    path = np.r_[[[0.0, 0.0]], pts]
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path[: max(2, int(np.searchsorted(cum, ROUTE_LEN)) + 1)]


def _embed_group(args):
    from .fusion_diag import project
    adir, rows = args
    cache, out = {}, []
    for fn, frame, d in rows:
        e = np.zeros((K_DET, 8), np.float32)
        path = _route_path(Path(adir), frame, cache)
        if path is not None and len(d):
            s, dd, _, _ = project(path, d[:, 3:5])
            keep = (np.abs(dd) <= HALF_W) & (s > 0) & (s <= REACH)
            sel = np.flatnonzero(keep)[np.argsort(s[keep], kind="stable")][:K_DET]
            for j, i in enumerate(sel):
                e[j, int(d[i, 0])] = 1.0
                e[j, 3:7] = d[i, 3], d[i, 4], d[i, 5], d[i, 6]
                e[j, 7] = 1.0
        out.append((fn, e.reshape(-1)))
    return out


def embed(det_dir: Path, workers: int = 12) -> pd.DataFrame:
    """processed/elicit_e5/embed.npy (n_index, 64) aligned to the index, and a small summary."""
    from multiprocessing import Pool
    from . import fusion_q4 as Q
    t = _index()
    subs = sorted(p for p in Path(det_dir).iterdir() if p.is_dir()) or [Path(det_dir)]   # one dir per list slice
    d = pd.concat([Q.load_dets(s, SCORE) for s in subs], ignore_index=True)
    d = d[d.prompt.isin(CLASSES)]
    d = Q.lift_dets(d, d.key.str.split("|").str[1].to_numpy(), Q.p5_calib())
    d = d[d.lift_ok].copy()
    H = pd.read_parquet(sorted(subs[0].glob("part-*.parquet"))[0], columns=["H"]).H.iloc[0]
    d["fn"] = d.key.str.split("|").str[0]
    arr = np.c_[d.prompt.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(), np.zeros((len(d), 2)),
                d.gx.to_numpy() + Q.REAR_AXLE_X, d.gy.to_numpy(), ((d.y1 - d.y0) / H).to_numpy(), d.score.to_numpy()]
    by = {fn: arr[idx] for fn, idx in d.groupby("fn").indices.items()}
    empty = np.zeros((0, 7))
    t["adir"] = t.files.map(lambda f: f[3].rsplit("/cams/", 1)[0])
    jobs = [(a, [(r.frame_name, int(r.frame), by.get(r.frame_name, empty)) for r in g.itertuples()])
            for a, g in t.groupby("adir")]
    with Pool(workers) as p:
        res = dict(x for part in p.imap_unordered(_embed_group, jobs, chunksize=4) for x in part)
    E = np.stack([res[f] for f in t.frame_name]).astype(np.float32)
    np.save(root("embed.npy"), E)
    m = E.reshape(len(E), K_DET, 8)
    summ = {"rows": len(E), "dets_lifted": len(d), "rows_with_any": float((m[:, :, 7].sum(1) > 0).mean()),
            "mean_in_corridor": float(m[:, :, 7].sum(1).mean()), "rows_with_ped": float((m[:, :, 0].sum(1) > 0).mean())}
    (root("embed_summary.json")).write_text(json.dumps(summ, indent=1))
    log.info("embedding: %s", summ)
    return summ


# ---------------------------------------------------------------- student

def _mlp(d_in: int, seed: int):
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, HIDDEN),
                              torch.nn.GELU(), torch.nn.Linear(HIDDEN, 40))
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    return net.cuda()


def train_student(X, ip, im, R, tr, grp, teacher, seed: int, rl, tag: str):
    """Arm A (teacher None) or B. X (n, d) standardised input; pair rows ip / im with target R; train rows tr."""
    import copy
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(ip, groups=grp))
    net = _mlp(X.shape[1], seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    rows_t = np.unique(np.r_[ip[a], im[a], tr]) if teacher is not None else None
    best, best_step, best_state = np.inf, 0, copy.deepcopy(net.state_dict())
    for step in range(1, MAX_STEPS + 1):
        net.train()
        loss = ((net(X[ip[a]]) - net(X[im[a]]) - R[a]) ** 2).sum(1).mean() + (net(X[tr]) ** 2).sum(1).mean()
        if teacher is not None:
            loss = loss + TEACHER_W * ((net(X[rows_t]) - teacher[rows_t]) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                v = float(((net(X[ip[b]]) - net(X[im[b]]) - R[b]) ** 2).sum(1).mean())
            if v < best:
                best, best_step, best_state = v, step, copy.deepcopy(net.state_dict())
            elif step - best_step >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    rl.event("e5_fit", tag=tag, seed=seed, best_step=best_step, stop_step=step, holdout_mse=best)
    with torch.no_grad():
        return net(X)


def fit(rl, models=("cinque", "lebowski")):
    import torch
    from . import p5_exam as E, p5_openpilot, p5_pairs as P, reactivity_mc as MC
    t, past, fut, obs, null, pairs = E.load()
    n = len(t)
    Q = torch.as_tensor(P.load_features(t, ("L18_last",))["L18_last"], device="cuda")
    Emb = torch.as_tensor(np.load(root("embed.npy")), device="cuda")
    fold = E.folds(t, pairs)
    F = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    Ego = torch.as_tensor(E.ego_input(t, past), device="cuda")
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    role = t.role.to_numpy()
    ref = np.load(data_dir() / MC_RUN / "preds_obs.npz")
    at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
    preds = {}
    mask_col = np.arange(64) % 8 == 7
    for m in models:
        Xop = torch.as_tensor(p5_openpilot.load(t, (m,), sub="op_streams_vis")[f"op-{m} temporal"], device="cuda")
        for key in (f"prior [{m}]", f"M-C pair [{m}]", f"M-C pair op [{m}]"):
            preds[key] = np.full((n, 20, 2), np.nan, np.float32)
            preds[key][obs_rows] = ref[key]
        for f in range(E.K_FOLDS):
            ev = obs_rows[fold[obs_rows] == f]
            if not len(ev):
                continue
            tr = np.flatnonzero((role == "train") & (fold != f))
            o = MC.fit_fold(f, fold, t, F, Ego, Xop, Q, pr_ip, pr_im, pr_group, rl, m)
            prior, teach = o["prior"], o["M-C pair"] - o["prior"]
            diff = float(np.abs(o["prior"][ev].reshape(-1, 20, 2).cpu().numpy() - ref[f"prior [{m}]"][at[ev].to_numpy()]).max())
            rl.event("e5_prior_check", model=m, fold=f, max_abs_diff=diff)
            assert diff < 1e-3, f"prior of fold {f} does not reproduce the stored run ({diff})"
            zo = (Xop - Xop[tr].mean(0)) / Xop[tr].std(0, correction=0).clamp_min(1e-6) / np.sqrt(Xop.shape[1])
            mu, sd = Emb[tr].mean(0), Emb[tr].std(0, correction=0).clamp_min(1e-6)
            mc = torch.as_tensor(mask_col, device="cuda")
            ze = torch.where(mc, Emb, (Emb - mu) / sd) / np.sqrt(Emb.shape[1])
            X = torch.cat([zo, ze], 1).float()
            keep = fold[pr_ip] != f
            ip, im, grp = pr_ip[keep], pr_im[keep], pr_group[keep]
            R = (F[ip] - F[im]) - (prior[ip] - prior[im])
            for arm, tch in (("A", None), ("B", teach)):
                for seed in SEEDS:
                    d = train_student(X, ip, im, R, tr, grp, tch, seed, rl, f"{m} f{f} {arm}")
                    key = f"E5 {arm} s{seed} [{m}]"
                    preds.setdefault(key, np.full((n, 20, 2), np.nan, np.float32))[ev] = \
                        (prior[ev] + d[ev]).reshape(-1, 20, 2).cpu().numpy()
            rl.log.info("%s fold %d done", m, f)
        del Xop
        torch.cuda.empty_cache()
    oo, nn = E.deltas(obs, null, t, preds)
    res = E.exam(oo, nn, pairs, list(preds))
    crit = pd.concat([MC.criteria(res, [k for k in preds if k.endswith(f"[{m}]")], f"prior [{m}]") for m in models])
    o = res["obs"]
    nr = []
    for ex in preds:
        tau = res["taus"][ex]
        for scope, sub in (("all", o[~o.reactive]), ("DynamicObjectCrossing", o[~o.reactive & (o.family == "DynamicObjectCrossing")])):
            nr.append({"arm": ex, "scope": scope, "n": len(sub), "nonreactive_flip": float(E._moved(sub[ex], tau).mean())})
    d = rl.dir
    res["flips"].to_csv(d / "flip_rates.csv", index=False)
    crit.to_csv(d / "criteria.csv", index=False)
    pd.DataFrame(nr).to_csv(d / "nonreactive.csv", index=False)
    np.savez_compressed(d / "preds_obs.npz", rows=obs_rows, **{k: v[obs_rows] for k, v in preds.items()})
    rl.log.info("criteria\n%s", crit.to_markdown(index=False, floatfmt=".3f"))


# ---------------------------------------------------------------- latency

def latency(rl, n: int = 200, warm: int = 20) -> dict:
    """Head-side cost per 3-camera frame, batch 1: lift + corridor + embedding (CPU numpy) and the MLP (GPU). The
    detector's own 3-camera fp16 latency is measured with jevdrive.fastperc latency in envs/ultralytics."""
    import torch
    from . import fusion_q4 as Q
    from .fusion_diag import project
    cal = Q.p5_calib()
    t = _index()
    t = t[t.role == "obs"].iloc[:n + warm]
    rng = np.random.default_rng(0)
    net = _mlp(576, 0)
    net.eval()
    cache, ts_cpu, ts_gpu = {}, [], []
    for r in t.itertuples():
        adir = Path(r.files[3].rsplit("/cams/", 1)[0])
        a = time.perf_counter()
        dets = []
        for cam in Q.CAMS:          # 6 synthetic detections per camera through the same lift / corridor code
            u, v = rng.uniform(0, 972, 6), rng.uniform(540, 1079, 6)
            g, ok = Q.lift(u, v, cal[cam])
            dets.append(g[ok])
        g = np.concatenate(dets)
        path = _route_path(adir, int(r.frame), cache)
        if path is not None and len(g):
            s, dd, _, _ = project(path, g)
            np.argsort(s[(np.abs(dd) <= HALF_W) & (s > 0) & (s <= REACH)])
        x = torch.zeros(1, 576, device="cuda")
        b = time.perf_counter()
        with torch.no_grad():
            net(x)
        torch.cuda.synchronize()
        c = time.perf_counter()
        ts_cpu.append(1000 * (b - a))
        ts_gpu.append(1000 * (c - b))
    cpu_, gpu_ = np.array(ts_cpu[warm:]), np.array(ts_gpu[warm:])
    out = {"embed_cpu_p50_ms": float(np.percentile(cpu_, 50)), "embed_cpu_p95_ms": float(np.percentile(cpu_, 95)),
           "mlp_gpu_p50_ms": float(np.percentile(gpu_, 50)), "mlp_gpu_p95_ms": float(np.percentile(gpu_, 95)), "n": len(cpu_)}
    (rl.dir / "head_latency.json").write_text(json.dumps(out, indent=1))
    rl.log.info("head latency: %s", out)
    return out


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("imglist", "embed", "fit", "latency"))
    ap.add_argument("--det", default=str(data_dir() / "processed/elicit_e5/dets"))
    a = ap.parse_args()
    if a.step == "imglist":
        imglist()
    elif a.step == "embed":
        embed(Path(a.det))
    else:
        rl = RunLog("elicitation", f"e5-{a.step}")
        (fit if a.step == "fit" else latency)(rl)
        rl.close()


if __name__ == "__main__":
    main()
