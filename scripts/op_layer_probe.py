"""E0 (todos/2026-09-29-e0-layer-probe.md): at which openpilot (Cinque) layer does CARLA pedestrian information disappear.

  extract   per dataset (nusc = nuScenes val keyframes, p5 = P5 v1 BA readout frames): render the frames, run the port
            (fp16) from pixels to the four stage outputs, mean + max pool each (2 x channels), and pool the cached stage-3
            trunk (processed/op_adapt/<ds>/) the same way. -> runs/op_layer/feats/<ds>.npz
  probe     the decision 42 D0 probe (CARLA, p5_exam) and the decision 55 (a) probe (nuScenes val) per layer,
            plus descriptive MLP probes and domain classifiers -> runs/op_layer/probe/*.csv

  taskset -c 48-67 python scripts/op_layer_probe.py extract nusc --workers 20
  python scripts/op_layer_probe.py probe
"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_data as D  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

TAPS = {"stage1": "permute_9", "stage2": "permute_17", "stage3": "permute_73", "stage4": "permute_81"}
LAYERS = ["stage1", "stage2", "stage3", "stage4", "vision", "temporal"]
OUT = data_dir() / "runs" / "op_layer"


def pool(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, 2C) [spatial mean | spatial max], fp32."""
    x = x.float()
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


@torch.no_grad()
def forward(net, prev, cur, batch=32):
    out = {k: [] for k in TAPS}
    for i in range(0, len(cur), batch):
        p = torch.as_tensor(prev[i:i + batch]).cuda(non_blocking=True)
        c = torch.as_tensor(cur[i:i + batch]).cuda(non_blocking=True)
        o = net.run_batched(A.vision_feeds(p, c), list(TAPS.values()))
        for k, n in TAPS.items():
            out[k].append(pool(o[n][:, 0]).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def extract(a):
    import op_adapt_cache as C
    log = RunLog("op_layer", f"extract-{a.dataset}")
    a.limit = 0
    its, job, init, initargs = C.items(a)
    if a.dataset == "nusc":
        lab = pd.read_parquet(D.root() / "nusc_labels.parquet")
        val = set(lab[lab.split == "val"].scene.unique())
        its = [s for s in its if s in val]
        slots_of = lambda meta: np.asarray(meta["key_slot"])            # noqa: E731
        keys_of = lambda meta, s: np.asarray(meta["tokens"])             # noqa: E731  (one token per key slot)
    else:
        slots_of = lambda meta: np.asarray(meta["targets"])             # noqa: E731
        keys_of = lambda meta, s: np.asarray(meta["names"])[s]          # noqa: E731
    if a.limit_n:
        its = its[:a.limit_n]
    log.info(f"{len(its)} streams")
    from drive_backbones_openpilot import bounded_map
    t0, acc, n, worst = time.time(), {}, 0, 0.0
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=initargs) if init else ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(int, range(a.workers)))       # fork the render workers before CUDA exists in this process
        net = A.load("cinque", torch.float16).cuda()
        for k, prev, cur, meta in bounded_map(ex, job, its, 2 * a.workers):
            s = slots_of(meta)
            if not len(s):
                continue
            f = forward(net, prev[s], cur[s])
            cache = np.load(D.root(a.dataset) / f"{k}.npz")["trunk"][s]     # (n, 1024, 8, 16) fp16, the P5 / nuScenes trunk cache
            c3 = pool(torch.as_tensor(cache)).numpy()
            worst = max(worst, float(np.abs(c3 - f["stage3"]).max()))
            f["stage3_cache"] = c3
            del f["stage3"]
            f["key"] = keys_of(meta, s).astype(str)
            for kk, v in f.items():
                acc.setdefault(kk, []).append(v)
            n += 1
            if n % 20 == 0 or n == len(its):
                el = time.time() - t0
                log.info(f"[{n}/{len(its)}] {el / 60:.1f} min, ETA {(len(its) - n) * el / n / 60:.0f} min, "
                         f"max |forward - cache| pooled stage 3 so far {worst:.4f}")
    res = {k: np.concatenate(v) for k, v in acc.items()}
    (OUT / "feats").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "feats" / f"{a.dataset}.npz", **res)
    log.info(f"{a.dataset}: {len(res['key'])} rows, forward-vs-cache max pooled diff {worst:.4f}, {(time.time() - t0) / 60:.1f} min")
    log.event("end", rows=len(res["key"]), stage3_check_max=worst, wall_s=time.time() - t0)


# ---------------------------------------------------------------- probes
def load_eval(ds):
    import glob
    f = sorted(glob.glob(str(data_dir() / f"runs/op_adapt/train-lam10/*/eval/{ds}.npz")))[0]
    z = np.load(f)
    return z["key"], {"vision": z["orig_vision"], "temporal": z["orig_temporal"]}


def features(ds):
    z = np.load(OUT / "feats" / f"{ds}.npz")
    key = z["key"]
    ek, ev = load_eval(ds)
    pos = pd.Series(np.arange(len(ek)), index=ek)
    at = pos.reindex(key).to_numpy()
    assert not np.isnan(at).any(), f"{ds}: extracted frames missing from the eval features"
    at = at.astype(int)
    X = {"stage1": z["stage1"], "stage2": z["stage2"], "stage3": z["stage3_cache"], "stage4": z["stage4"],
         "vision": ev["vision"][at], "temporal": ev["temporal"][at]}
    return key, X


def mlp_scores(Z, y, tr, ev, seed=0):
    """Descriptive 2-layer MLP probe: hidden 256, ReLU, AdamW lr 1e-3 wd 1e-2, 200 full-batch steps, class-balanced loss."""
    torch.manual_seed(seed)
    yt = torch.as_tensor(y[tr].astype(np.int64), device=Z.device)
    m = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 2)).to(Z.device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    cnt = torch.bincount(yt, minlength=2).float().clamp(min=1)
    w = len(yt) / (2 * cnt)
    Zt = Z[tr]
    for _ in range(200):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(Zt), yt, weight=w).backward()
        opt.step()
    with torch.no_grad():
        return m(Z[ev]).softmax(1)[:, 1].cpu().numpy()


def carla(X, key, rl):
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from jevdrive import p5_exam as E
    from jevdrive.p4_carla import _std
    t, past, fut, obs, null, pairs = E.load()
    pos = pd.Series(np.arange(len(key)), index=key)
    at = pos.reindex(t.frame_name).to_numpy()
    assert not np.isnan(at).any(), "P5 index frames without extracted features"
    at = at.astype(int)
    XX = {k: X[k][at] for k in LAYERS}
    fold = E.folds(t, pairs)
    scores, _ = E.probes(t, XX, fold, rl)
    # descriptive MLP probe on the same rows / folds as the linear one
    role, n = t.role.to_numpy(), len(t)
    y_all = t["hazard"].to_numpy(dtype=float)
    for k, Xa in XX.items():
        Xt = torch.as_tensor(Xa, device="cuda")
        s = np.full(n, np.nan)
        for f in range(E.K_FOLDS):
            tr = np.flatnonzero((role == "train") & (fold != f) & ~np.isnan(y_all))
            ev = np.flatnonzero((role == "obs") & (fold == f))
            if len(np.unique(y_all[tr])) < 2 or not len(ev):
                continue
            mu, sd = _std(Xt, tr)
            s[ev] = mlp_scores((Xt - mu) / sd, np.nan_to_num(y_all).astype(int), tr, ev)
        scores[("hazard", f"mlp_{k}")] = s
        del Xt
        torch.cuda.empty_cache()
    comps = [(k, "vision") for k in LAYERS if k != "vision"] + [("vision", "temporal")] + \
            [(f"mlp_{k}", k) for k in LAYERS]
    ps = E.probe_auc_paired_scopes(obs, t, scores, comps)
    return ps, t


def real(X, key, rl):
    import op_adapt_readout as R
    from jevdrive.p4_carla import _std
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet").set_index("token")
    L = lab.loc[key]
    g = L.scene.to_numpy()
    tasks = {"ped_corr": L.ped_corr.to_numpy(bool), "ped_wide": L.ped_wide.to_numpy(bool)}
    near = L.ped_corr.to_numpy(bool) & (L.ped_dist.to_numpy() <= 10)
    neg = ~L.ped_wide.to_numpy(bool)
    rows = []
    for task, y in list(tasks.items()) + [("ped_corr<=10m (descriptive)", near)]:
        m = np.ones(len(y), bool) if "<=10m" not in task else (near | neg)
        yy = y[m].astype(int)
        sc = {k: R.oof_probe(X[k][m], yy, g[m]) for k in LAYERS}
        for k in LAYERS:
            r = R.paired_boot(yy, sc[k], sc["vision"], g[m])
            rows.append({"task": task, "layer": k, "probe": "linear", **r})
        # descriptive MLP, same scene folds as oof_probe
        ug = np.array(sorted(set(g[m])))
        fold = pd.Series(np.arange(len(ug)) % 5, index=np.random.default_rng(0).permutation(ug))[g[m]].to_numpy()
        for k in LAYERS:
            Xt = torch.as_tensor(X[k][m], device="cuda")
            s = np.full(m.sum(), np.nan)
            for f in range(5):
                tr, ev = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                mu, sd = _std(Xt, tr)
                s[ev] = mlp_scores((Xt - mu) / sd, yy, tr, ev)
            r = R.paired_boot(yy, s, sc[k], g[m])
            rows.append({"task": task, "layer": k, "probe": "mlp (delta = mlp - linear)", **r})
            del Xt
        rl.info(f"{task}: " + " ".join(f"{k} {[r for r in rows if r['task'] == task and r['layer'] == k and r['probe'] == 'linear'][0]['auc']:.3f}" for k in LAYERS))
    return pd.DataFrame(rows)


def domain(Xc, tc, Xr, gr, rl):
    """Domain classifier AUC per layer: CARLA P5 frames (<= 6000, route-grouped) vs nuScenes val keyframes (<= 6000, scene-grouped)."""
    import op_adapt_readout as R
    rng = np.random.default_rng(0)
    ic = np.sort(rng.choice(len(Xc["vision"]), min(6000, len(Xc["vision"])), replace=False))
    ir = np.sort(rng.choice(len(Xr["vision"]), min(6000, len(Xr["vision"])), replace=False))
    g = np.r_[np.array(["c" + str(v) for v in tc[ic]]), np.array(["r" + str(v) for v in gr[ir]])]
    y = np.r_[np.ones(len(ic), int), np.zeros(len(ir), int)]
    rows = []
    for k in LAYERS:
        s = R.oof_probe(np.concatenate([Xc[k][ic], Xr[k][ir]]), y, g)
        from jevdrive.p4_carla import auc
        rows.append({"layer": k, "domain_auc": float(auc(y, s)), "n_carla": len(ic), "n_real": len(ir)})
        rl.info(f"domain {k}: {rows[-1]['domain_auc']:.4f}")
    return pd.DataFrame(rows)


def probe(a):
    rl = RunLog("op_layer", "probe")
    out = OUT / "probe"
    out.mkdir(parents=True, exist_ok=True)
    kr, Xr = features("nusc")
    kc, Xc = features("p5")
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet").set_index("token")
    gr = lab.loc[kr].scene.to_numpy()
    parts = a.parts
    if "real" in parts:
        df = real(Xr, kr, rl)
        df.to_csv(out / "real.csv", index=False)
        rl.info("real\n" + df[df.probe == "linear"][["task", "layer", "auc", "auc_ci", "delta", "delta_ci"]].to_string())
    if "carla" in parts:
        ps, t = carla(Xc, kc, rl)
        ps.to_csv(out / "carla.csv", index=False)
        rl.info("carla\n" + ps[ps.scope == "pedestrian"].to_string())
    if "domain" in parts:
        pos = pd.Series(np.arange(len(kc)), index=kc)
        os.environ["P5_SET"] = "carla_p5v1_ba"
        from jevdrive import p5_exam as E
        t = E.load()[0]
        at = pos.reindex(t.frame_name).to_numpy().astype(int)
        df = domain({k: v[at] for k, v in Xc.items()}, t.base_id.to_numpy(), Xr, gr, rl)
        df.to_csv(out / "domain.csv", index=False)
    rl.event("end")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("dataset", choices=("nusc", "p5"))
    e.add_argument("--workers", type=int, default=20)
    e.add_argument("--limit-n", type=int, default=0, help="first n streams only (smoke test)")
    p = sp.add_parser("probe")
    p.add_argument("--parts", nargs="+", default=["real", "carla", "domain"])
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else probe(a)


if __name__ == "__main__":
    main()
