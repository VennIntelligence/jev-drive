"""E1 (todos/2026-09-29-e1-cosmos-probe.md): 2 x 2 {CARLA, Cosmos} x {x+, x-} openpilot layer probe on the Cosmos G4 full-run
pairs, plus a spatially resolved probe (GT-box cell pooling S1, small conv head on the stage-3 map S2).

  extract   the frozen pair list -> runs/op_cosmos_probe/feats.npz (pooled taps, box / outside cell pooled taps, stage-3 maps)
  probe     pooled 2 x 2, S1, S2 on the Cosmos pairs, S2 on the P5 CARLA pairs and nuScenes val -> runs/op_cosmos_probe/probe/*.csv

  taskset -c 48-63 python scripts/op_cosmos_probe.py extract --workers 16
  python scripts/op_cosmos_probe.py probe
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

OUT = data_dir() / "runs" / "op_cosmos_probe"
PAIRS = data_dir() / "runs" / "cosmos_full" / "pairs"
STREAMS = [("carla", "plus"), ("carla", "minus"), ("cosmos", "plus"), ("cosmos", "minus")]
TAPS = {"stage1": "permute_9", "stage2": "permute_17", "stage3": "permute_73", "stage4": "permute_81"}
LAYERS = ["stage1", "stage2", "stage3", "stage4", "vision", "temporal"]
H, W, T, STEP, NSLOT = 704, 1280, 93, 4, 24
CTX, MIN_SLOT, MIN_PX, EXPAND = A.CONTEXT, 9, 100, 8
AT = (0.275, 0.525)
CELL = {"stage1": 4, "stage2": 8, "stage3": 16}         # cell size on the packed 128 x 256 model frame


# ---------------------------------------------------------------- extract
_IDX = None


def _init(idx):
    global _IDX
    _IDX = idx


def box_cells(box: np.ndarray, idx: dict) -> dict:
    """box (x0, y0, x1, y1) on the 1280 x 704 camera -> {stage: bool (h, w)} feature cells hit by the box (+ EXPAND px),
    union over the road and wide model frames. A model pixel is inside when its nearest-neighbour source pixel is; a box
    too small to be sampled falls back to the model pixel nearest to its centre (dropped when farther than 40 px)."""
    out = {s: np.zeros((128 // c, 256 // c), bool) for s, c in CELL.items()}
    if box[0] < 0:
        return out
    x0, y0, x1, y1 = box[0] - EXPAND, box[1] - EXPAND, box[2] + EXPAND, box[3] + EXPAND
    for k in ("road", "wide"):
        sy, sx = np.divmod(idx[k], W)
        inb = (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
        if not inb.any():
            d = np.hypot(sx - (box[0] + box[2]) / 2, sy - (box[1] + box[3]) / 2)
            j = int(d.argmin())
            if d[j] > 40:
                continue
            inb[j] = True
        inb = inb.reshape(256, 512).reshape(128, 2, 256, 2).any((1, 3))
        for s, c in CELL.items():
            out[s] |= inb.reshape(128 // c, c, 256 // c, c).any((1, 3))
    return out


def pair_job(pair):
    import cosmos_openpilot as CO
    d = PAIRS / pair
    gt = np.load(d / "gt.npz")
    sup = np.unpackbits(gt["support"])[: T * H * W].reshape(T, H, W, 1)[::STEP].astype(bool)
    clip = {f"{c}_{s}": CO.read_mp4(d / f"{c}_{s}.mp4") for c in ("carla", "cosmos") for s in ("plus", "minus")}
    assert all(len(v) == T for v in clip.values()), pair
    sl = {k: v[::STEP] for k, v in clip.items()}
    sl["cosmos_plus"] = np.where(sup, sl["cosmos_plus"], sl["cosmos_minus"])
    frames = np.stack([CO.model_frames(sl[f"{c}_{s}"], _IDX) for c, s in STREAMS])       # (4, 24, 2, 6, 128, 256)
    box = gt["box"][::STEP].astype(int)
    cells = [box_cells(b, _IDX) for b in box]
    spec = json.load(open(d / "spec.json"))
    return pair, frames, gt["px"][::STEP].astype(int), box, {s: np.stack([c[s] for c in cells]) for s in CELL}, \
        {"inst": spec["inst"], "family": spec["family"]}


def pool(x):
    x = x.float()
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


def masked_pool(x, m):
    """x (B, C, h, w), m (B, h, w) bool -> (B, 2C) mean | max over the True cells (zeros where m is empty)."""
    x, m = x.float(), m[:, None]
    n = m.sum((2, 3)).clamp(min=1)
    mean = (x * m).sum((2, 3)) / n
    mx = x.masked_fill(~m, float("-inf")).amax((2, 3))
    return torch.cat([mean, torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))], 1)


@torch.no_grad()
def forward(net, frames, cells):
    """frames (24, 2, 6, 128, 256) one stream -> dict of per-slot arrays."""
    fr = torch.as_tensor(frames).cuda()
    prev = torch.cat([torch.zeros_like(fr[:1]), fr[:-1]])
    o = net.run_batched(A.vision_feeds(prev, fr), list(TAPS.values()))
    res = {}
    for k, n in TAPS.items():
        x = o[n][:, 0]
        res[k] = pool(x).cpu().numpy()
        if k in CELL:
            m = torch.as_tensor(cells[k]).cuda()
            res[f"{k}_box"] = masked_pool(x, m).cpu().numpy()
            res[f"{k}_out"] = masked_pool(x, ~m).cpu().numpy()
    tr = o[TAPS["stage3"]][:, 0]
    res["map3"] = tr.cpu().numpy()                                   # fp16 (24, 1024, 8, 16)
    j = torch.arange(NSLOT, device="cuda")[:, None] + torch.arange(-(CTX - 1), 1, device="cuda")[None]
    valid = j >= 0
    p = A.stage4_policy(net, tr[j.clamp(min=0)], AT, torch.tensor([[1.0, 0.0]], device="cuda").expand(NSLOT, 2), valid)
    res["vision"], res["temporal"] = p["mean"].float().cpu().numpy(), p["select_4"].float().cpu().numpy()
    return res


def extract(a):
    import cosmos_openpilot as CO
    log = RunLog("op_cosmos_probe", "extract")
    names = (OUT / "pairs_frozen.txt").read_text().split()
    if a.limit_n:
        names = names[:a.limit_n]
    idx = CO.maps()
    log.info(f"{len(names)} pairs, {a.workers} workers")
    t0, acc, meta = time.time(), {}, []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(idx,)) as ex:
        list(ex.map(int, range(a.workers)))                          # fork before CUDA exists in this process
        net = A.load("cinque", torch.float16).cuda()
        for i, (pair, frames, px, box, cells, sp) in enumerate(ex.map(pair_job, names)):
            for si, (c, s) in enumerate(STREAMS):
                r = forward(net, frames[si], {k: v for k, v in cells.items()})
                for kk, v in r.items():
                    acc.setdefault(kk, {}).setdefault(si, []).append(v)
            acc.setdefault("px", {}).setdefault(0, []).append(px)
            acc.setdefault("box", {}).setdefault(0, []).append(box)
            for s in CELL:
                acc.setdefault(f"cells_{s}", {}).setdefault(0, []).append(cells[s].reshape(NSLOT, -1).any(1))
            meta.append({"pair": pair, **sp})
            if (i + 1) % 5 == 0 or i + 1 == len(names):
                el = time.time() - t0
                log.info(f"[{i + 1}/{len(names)}] {el / 60:.1f} min, ETA {(len(names) - i - 1) * el / (i + 1) / 60:.1f} min")
    res = {}
    for k, d in acc.items():
        res[k] = np.stack([np.stack(d[si]) for si in sorted(d)], 1) if len(d) > 1 else np.stack(d[0])   # (pairs, 4, 24, ...)
    np.savez(OUT / "feats.npz", **res)
    pd.DataFrame(meta).to_csv(OUT / "meta.csv", index=False)
    log.info(f"done {(time.time() - t0) / 60:.1f} min, {sorted(res)}")
    log.event("end", pairs=len(names), wall_s=time.time() - t0)


# ---------------------------------------------------------------- probes
def conv_scores(Xget, y, tr, ev, n_ch=1024, seed=0, steps=300, bs=512, hidden=32):
    """Small stage-3-map probe: 1x1 conv (C -> 32), ReLU, spatial max, linear. Xget(idx) -> fp16 tensor (b, C, 8, 16) on the
    CPU or GPU. Channels z-scored on the training rows; AdamW lr 1e-3 wd 1e-2; class-balanced loss; hyper-parameters not tuned."""
    torch.manual_seed(seed)
    tri = np.asarray(tr)
    sub = tri[np.random.default_rng(seed).permutation(len(tri))[:4096]]
    xs = Xget(sub).cuda().float()
    mu, sd = xs.mean((0, 2, 3), keepdim=True), xs.std((0, 2, 3), keepdim=True).clamp(min=1e-3)
    del xs
    m = torch.nn.Sequential(torch.nn.Conv2d(n_ch, hidden, 1), torch.nn.ReLU(), torch.nn.AdaptiveMaxPool2d(1),
                            torch.nn.Flatten(), torch.nn.Linear(hidden, 2)).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    yt = torch.as_tensor(np.asarray(y)[tri].astype(np.int64)).cuda()
    w = len(yt) / (2 * torch.bincount(yt, minlength=2).float().clamp(min=1))
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randperm(len(tri), generator=g)[:min(bs, len(tri))].numpy()
        x = (Xget(tri[b]).cuda().float() - mu) / sd
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x), yt[b], weight=w).backward()
        opt.step()
    out = []
    with torch.no_grad():
        for i in range(0, len(ev), 1024):
            x = (Xget(np.asarray(ev)[i:i + 1024]).cuda().float() - mu) / sd
            out.append(m(x).softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def cosmos_rows(z, meta):
    """Readout rows: slot >= MIN_SLOT with >= MIN_PX GT pixels on the x+ stream; returns (pair, slot) index arrays."""
    px = z["px"]
    ok = (px >= MIN_PX) & (np.arange(NSLOT)[None] >= MIN_SLOT)
    return np.nonzero(ok)


def cosmos_part(rl):
    import op_adapt_readout as R
    z = dict(np.load(OUT / "feats.npz"))
    meta = pd.read_csv(OUT / "meta.csv")
    P, S = cosmos_rows(z, meta)
    inst = meta.inst.to_numpy()[P].astype(str)
    fam = meta.family.to_numpy()[P]
    px = z["px"][P, S]
    rl.info(f"readout slots {len(P)} over {len(set(P))} pairs; px median {np.median(px):.0f} (p10 {np.percentile(px, 10):.0f}, p90 {np.percentile(px, 90):.0f})")
    cell = {"carla": (0, 1), "cosmos": (2, 3)}
    y = np.r_[np.ones(len(P)), np.zeros(len(P))].astype(int)
    g = np.r_[inst, inst]
    rows, sc = [], {}

    def rowsx(arr, c):                       # (pairs, 4, 24, D) -> plus rows then minus rows of cell c
        return np.concatenate([arr[P, cell[c][0], S], arr[P, cell[c][1], S]])

    def run(tag, feat, keep=None, subset="all"):
        for c in cell:
            X = rowsx(z[feat] if feat in z else None, c)
            m = np.ones(len(y), bool) if keep is None else np.r_[keep, keep]
            s = np.full(len(y), np.nan)
            s[m] = R.oof_probe(X[m], y[m], g[m])
            sc[(tag, c, subset)] = (s, m)
        m = sc[(tag, "carla", subset)][1]
        for c in cell:
            other = "carla" if c == "cosmos" else "cosmos"
            r = R.paired_boot(y[m], sc[(tag, c, subset)][0][m], sc[(tag, other, subset)][0][m], g[m])
            rows.append({"probe": tag, "cell": c, "subset": subset, "n_pairs": len(set(g[m])), **r})
            rl.info(f"{tag:14s} {c:6s} {subset:5s} auc {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] (n {r['n']}, groups {r['groups']})")

    big = px >= 1500
    for L in LAYERS:
        run(f"pooled_{L}", L)
    for L in ("stage1", "stage2", "stage3"):
        has = z[f"cells_{L}"][P, S]                    # any box cell at that stage (M non-empty)
        run(f"S1_box_{L}", f"{L}_box", has)
        run(f"S1_out_{L}", f"{L}_out", has)
    for L in ("stage3", "temporal"):
        run(f"pooled_{L}", L, big, "big")
    run("S1_box_stage3", "stage3_box", z["cells_stage3"][P, S] & big, "big")
    run("S1_box_stage1", "stage1_box", z["cells_stage1"][P, S] & big, "big")
    # S2 conv head on the stage-3 map, same rows / folds
    ug = np.array(sorted(set(g)))
    fold = pd.Series(np.arange(len(ug)) % 5, index=np.random.default_rng(0).permutation(ug))[g].to_numpy()
    for c in cell:
        Xm = torch.as_tensor(np.concatenate([z["map3"][P, cell[c][0], S], z["map3"][P, cell[c][1], S]]))
        s = np.full(len(y), np.nan)
        for f in range(5):
            tr, ev = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            s[ev] = conv_scores(lambda i: Xm[torch.as_tensor(i)], y, tr, ev)
        sc[("S2_conv_stage3", c, "all")] = (s, np.ones(len(y), bool))
    for c in cell:
        other = "carla" if c == "cosmos" else "cosmos"
        ref = sc[("pooled_stage3", c, "all")][0]
        r = R.paired_boot(y, sc[("S2_conv_stage3", c, "all")][0], ref, g)
        rows.append({"probe": "S2_conv_stage3 (delta vs pooled stage3, same cell)", "cell": c, "subset": "all", "n_pairs": len(set(g)), **r})
        rl.info(f"S2 conv {c}: auc {r['auc']:.3f} {r['auc_ci']}  pooled stage3 {r['auc_ref']:.3f}")
    r = R.paired_boot(y, sc[("S2_conv_stage3", "cosmos", "all")][0], sc[("S2_conv_stage3", "carla", "all")][0], g)
    rows.append({"probe": "S2_conv_stage3 (delta cosmos vs carla)", "cell": "cosmos", "subset": "all", "n_pairs": len(set(g)), **r})
    # descriptive: cross-cell transfer of the pooled probe (train on one cell, test on the other), same folds
    from jevdrive.p4_carla import _std, auc, logreg
    for L in ("stage3", "temporal"):
        for a, b in (("carla", "cosmos"), ("cosmos", "carla")):
            Xa, Xb = rowsx(z[L], a), rowsx(z[L], b)
            s = np.full(len(y), np.nan)
            for f in range(5):
                tr, ev = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                Xt = torch.as_tensor(Xa, device="cuda", dtype=torch.float32)
                mu, sd = _std(Xt, tr)
                Wb = logreg(((Xt - mu) / sd)[tr], torch.as_tensor(y[tr], device="cuda"), 1e-2)
                Zb = (torch.as_tensor(Xb, device="cuda", dtype=torch.float32) - mu) / sd
                s[ev] = (Zb[ev] @ Wb[0] + Wb[1]).softmax(1)[:, 1].cpu().numpy()
            rows.append({"probe": f"transfer_{L} (lam 1e-2)", "cell": f"train {a} -> test {b}", "subset": "all", "n_pairs": len(set(g)),
                         "auc": float(auc(y, s)), "n": len(y)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe" / "cosmos_cells.csv", index=False)
    per = []                                             # per family, pooled stage 3 / temporal, in-domain scores already computed
    for tag in ("pooled_stage3", "pooled_temporal", "S1_box_stage3"):
        for c in cell:
            s, m = sc[(tag, c, "all")]
            for f in sorted(set(fam)):
                mm = m & np.r_[fam == f, fam == f]
                if len(set(y[mm])) == 2:
                    per.append({"probe": tag, "cell": c, "family": f, "n_pairs": len(set(g[mm])), "n": int(mm.sum()), "auc": float(auc(y[mm], s[mm]))})
    pd.DataFrame(per).to_csv(OUT / "probe" / "cosmos_family.csv", index=False)
    pd.DataFrame({"px": px, "family": fam, "inst": inst}).groupby("family").px.describe().to_csv(OUT / "probe" / "px_by_family.csv")
    return df


def size_bins(rl):
    """Descriptive, added after the first Cosmos-cell numbers: AUC of the all-rows pooled probe within pedestrian-size bins
    (x+ GT mask pixels at 1280 x 704; the x- row of the same slot goes with its x+ row)."""
    import op_adapt_readout as R
    from jevdrive.p4_carla import auc
    z = dict(np.load(OUT / "feats.npz"))
    meta = pd.read_csv(OUT / "meta.csv")
    P, S = cosmos_rows(z, meta)
    inst = meta.inst.to_numpy()[P].astype(str)
    px = z["px"][P, S]
    y = np.r_[np.ones(len(P)), np.zeros(len(P))].astype(int)
    g = np.r_[inst, inst]
    bins = [(100, 500), (500, 1500), (1500, 6000), (6000, 10 ** 9)]
    rows = []
    for L in ("stage3", "temporal"):
        for c, (ip, im) in {"carla": (0, 1), "cosmos": (2, 3)}.items():
            X = np.concatenate([z[L][P, ip, S], z[L][P, im, S]])
            sc = R.oof_probe(X, y, g)
            for lo, hi in bins:
                m = np.r_[(px >= lo) & (px < hi), (px >= lo) & (px < hi)]
                r = R.paired_boot(y[m], sc[m], sc[m], g[m]) if len(set(y[m])) == 2 else None
                rows.append({"layer": L, "cell": c, "px_lo": lo, "px_hi": hi, "pairs_rows": int(m.sum() // 2),
                             "auc": r["auc"] if r else np.nan, "auc_ci": r["auc_ci"] if r else None})
                rl.info(f"{L} {c} px [{lo},{hi}): n {int(m.sum() // 2)} auc {rows[-1]['auc']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "probe" / "cosmos_sizebins.csv", index=False)


_MAPS = {}


def p5_maps(kc, rl):
    """Cached stage-3 maps (n, 1024, 8, 16) fp16 of the P5 readout rows, in the order of the E0 feature keys `kc`."""
    if "p5" in _MAPS:
        return _MAPS["p5"]
    from concurrent.futures import ThreadPoolExecutor
    from jevdrive import op_adapt_data as D
    files = [f for f in sorted(D.root("p5").glob("*.npz")) if not f.stem.endswith(".tmp")]
    want = pd.Series(np.arange(len(kc)), index=kc)

    def load(f):
        z = np.load(f)
        s = z["targets"]
        return z["names"][s], z["trunk"][s]
    maps = torch.empty(len(kc), 1024, 8, 16, dtype=torch.float16)
    n = 0
    with ThreadPoolExecutor(16) as ex:
        for names, tr in ex.map(load, files):
            j = want.reindex(names).to_numpy()
            ok = ~np.isnan(j)
            maps[torch.as_tensor(j[ok].astype(int))] = torch.as_tensor(tr[ok])
            n += int(ok.sum())
    rl.info(f"P5 stage-3 maps loaded: {n} / {len(kc)} rows")
    _MAPS["p5"] = maps
    return maps


def p5_pairtrain(rl):
    """Descriptive bridge (added after the Cosmos-pair numbers, see the todo): the P5 CARLA pairs probed like the Cosmos cells,
    i.e. trained on the pair frames themselves (x+ vs x-, groups = base route, 5 folds), linear pooled layers and the conv head."""
    import op_adapt_readout as R
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from jevdrive import p5_exam as E
    import op_layer_probe as LP
    kc, Xc = LP.features("p5")
    t, past, fut, obs, null, pairs = E.load()
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    at = pos.reindex(t.frame_name).to_numpy()
    kpos = pd.Series(np.arange(len(kc)), index=kc)
    o = obs[obs.family.isin(E.PED_FAMILIES)]
    ip = kpos.reindex(o.fn_plus).to_numpy().astype(int)
    im = kpos.reindex(o.fn_minus).to_numpy().astype(int)
    rows_i = np.r_[ip, im]
    y = np.r_[np.ones(len(ip)), np.zeros(len(im))].astype(int)
    g = np.r_[o.base_id.to_numpy(), o.base_id.to_numpy()].astype(str)
    rows = []
    for L in LP.LAYERS:
        s = R.oof_probe(Xc[L][rows_i], y, g)
        r = R.paired_boot(y, s, s, g)
        rows.append({"probe": f"pair-trained linear {L}", "auc": r["auc"], "auc_ci": r["auc_ci"], "n": r["n"], "groups": r["groups"]})
        rl.info(f"P5 pair-trained {L}: {r['auc']:.3f} {r['auc_ci']}")
    maps = p5_maps(kc, rl)
    ug = np.array(sorted(set(g)))
    fold = pd.Series(np.arange(len(ug)) % 5, index=np.random.default_rng(0).permutation(ug))[g].to_numpy()
    s = np.full(len(y), np.nan)
    for f in range(5):
        tr, ev = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        s[ev] = conv_scores(lambda i: maps[torch.as_tensor(rows_i[np.asarray(i)])], y, tr, ev)
    lin = R.oof_probe(Xc["stage3"][rows_i], y, g)
    r = R.paired_boot(y, s, lin, g)
    rows.append({"probe": "pair-trained conv stage3", "auc": r["auc"], "auc_ci": r["auc_ci"], "n": r["n"], "groups": r["groups"]})
    rl.info(f"P5 pair-trained conv stage 3: {r['auc']:.3f} {r['auc_ci']}")
    pd.DataFrame(rows).to_csv(OUT / "probe" / "p5_pairtrain.csv", index=False)


def s2_p5(rl):
    """S2 on the P5 CARLA pairs: E0's data, folds and labels; conv head on the cached stage-3 maps."""
    from concurrent.futures import ThreadPoolExecutor
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from jevdrive import p5_exam as E
    from jevdrive import op_adapt_data as D
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import op_layer_probe as LP
    kc, Xc = LP.features("p5")
    t, past, fut, obs, null, pairs = E.load()
    pos = pd.Series(np.arange(len(kc)), index=kc)
    at = pos.reindex(t.frame_name).to_numpy().astype(int)
    maps = p5_maps(kc, rl)
    fold = E.folds(t, pairs)
    role, y_all = t.role.to_numpy(), t["hazard"].to_numpy(dtype=float)
    s = np.full(len(t), np.nan)
    for f in range(E.K_FOLDS):
        tr = np.flatnonzero((role == "train") & (fold != f) & ~np.isnan(y_all))
        ev = np.flatnonzero((role == "obs") & (fold == f))
        rl.info(f"P5 fold {f}: train {len(tr)} (pos {int(y_all[tr].sum())}), eval {len(ev)}")
        yy = np.nan_to_num(y_all).astype(int)
        s[ev] = conv_scores(lambda i: maps[torch.as_tensor(at[np.asarray(i)])], yy, tr, ev)
    scores, _ = E.probes(t, {"stage3": Xc["stage3"][at]}, fold, rl)
    scores[("hazard", "conv_stage3")] = s
    ps = E.probe_auc_paired_scopes(obs, t, scores, [("conv_stage3", "stage3")])
    ps.to_csv(OUT / "probe" / "p5_s2.csv", index=False)
    rl.info("P5 S2\n" + ps[ps.scope.isin(["pedestrian", "hazard pooled"])].to_string())
    return ps


def s2_nusc(rl):
    from concurrent.futures import ThreadPoolExecutor
    import op_adapt_readout as R
    import op_layer_probe as LP
    from jevdrive import op_adapt_data as D
    key, X = LP.features("nusc")
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet")
    val = sorted(lab[lab.split == "val"].scene.unique())
    L = lab.set_index("token").loc[key]
    pos = pd.Series(np.arange(len(key)), index=key)

    def load(sc):
        z = np.load(D.root("nusc") / f"{sc}.npz")
        return z["tokens"], z["trunk"][z["key_slot"]]
    maps = torch.empty(len(key), 1024, 8, 16, dtype=torch.float16)
    n = 0
    with ThreadPoolExecutor(16) as ex:
        for tok, tr in ex.map(load, val):
            j = pos.reindex(tok).to_numpy()
            ok = ~np.isnan(j)
            maps[torch.as_tensor(j[ok].astype(int))] = torch.as_tensor(tr[ok])
            n += int(ok.sum())
    rl.info(f"nuScenes stage-3 maps loaded: {n} / {len(key)} rows")
    g = L.scene.to_numpy()
    y = L.ped_corr.to_numpy(bool).astype(int)
    ug = np.array(sorted(set(g)))
    fold = pd.Series(np.arange(len(ug)) % 5, index=np.random.default_rng(0).permutation(ug))[g].to_numpy()
    s = np.full(len(y), np.nan)
    for f in range(5):
        tr, ev = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        s[ev] = conv_scores(lambda i: maps[torch.as_tensor(np.asarray(i))], y, tr, ev)
    lin = R.oof_probe(X["stage3"], y, g)
    r = R.paired_boot(y, s, lin, g)
    rl.info(f"nuScenes ped_corr: conv {r['auc']:.3f} {r['auc_ci']}, pooled linear stage 3 {r['auc_ref']:.3f}")
    df = pd.DataFrame([{"task": "ped_corr", "probe": "S2_conv_stage3 (delta vs pooled linear stage3)", **r}])
    df.to_csv(OUT / "probe" / "nusc_s2.csv", index=False)
    return df


def probe(a):
    rl = RunLog("op_cosmos_probe", "probe")
    (OUT / "probe").mkdir(parents=True, exist_ok=True)
    for part in a.parts:
        {"cosmos": cosmos_part, "p5": s2_p5, "nusc": s2_nusc, "p5pair": p5_pairtrain, "sizebins": size_bins}[part](rl)
    rl.event("end")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--limit-n", type=int, default=0, help="first n pairs only (smoke test)")
    p = sp.add_parser("probe")
    p.add_argument("--parts", nargs="+", default=["cosmos", "p5", "nusc", "p5pair"])
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else probe(a)


if __name__ == "__main__":
    main()
