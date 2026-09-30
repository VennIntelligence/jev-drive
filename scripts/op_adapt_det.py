"""op-adapt r2, package D: YOLO26x-seg detections and detection tokens for every trunk cache row (jevdrive.op_adapt_det;
formats in tmp/2026-09-30-op-adapt-r2-build.md, "D"). Root R2/det = $DATA_DIR/runs/op_adapt_r2/det.

  list <ds>      (envs/op-train) the front-camera image of every trunk row: det/<ds>/images.parquet (img_id + source),
                 det/<ds>/rows.parquet (stem, row, img_id; -1 = the zero image), det/<ds>/cams.npz (homography native ->
                 road frame per calibration, fit RMS, focal)
  detect <ds>    (envs/ultralytics) Yolo on the images, 4096-image chunks, part i/n of them per process:
                 det/<ds>/raw/part-<k>.npz (img_id, cls, score, box, feat), resumable per chunk
  pca            PCA-16 of the detection appearance on WOD train (det/wodtrain/raw) -> det/pca.npz
  tokens <ds>    det/tok/<ds>/<stem>.npz: tok (n_rows, 8, 25) f16, mask (n_rows, 8), img_id, aligned with the trunk rows
  check          (envs/op-train, GPU) adapter unit checks on real cached inputs -> det/checks/adapter.json

ds: nusc wod wodtrain p5 navtrain (round-1 caches, processed/op_adapt/<ds>), sim (runs/cosmos_full pairs, rows =
stream * 24 + slot as C's cache-sim), navtest / navhard (the 4 keyframes of every leaderboard token, rows = 4 * i + k),
off (C's offset frames; list waits for C's table).

  PY=$DATA_DIR/envs/op-train/bin/python; $PY scripts/op_adapt_det.py list nusc
  CUDA_VISIBLE_DEVICES=1 $DATA_DIR/envs/ultralytics/bin/python scripts/op_adapt_det.py detect nusc --part 0/2
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt_det as DT  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

CHUNK, SIM_SLOTS, SIM_STEP = 4096, 24, 4
SIM_STREAMS = (("carla", 0), ("carla", 1), ("cosmos", 0), ("cosmos", 1))     # C+, C-, K+, K- (0 = plus)
ROUND1 = data_dir() / "processed" / "op_adapt"


# ================================================================ lists

def _save(ds, images: pd.DataFrame, rows: pd.DataFrame, cams: dict, rl, wh=None):
    """images / rows parquet and cams.npz (key, H ideal pixel -> road, fit RMS, intr (9,)), each written atomically;
    also the undistortion round trip over each calibration's image (wh = (W, H), default from the principal point)."""
    d = DT.root(ds)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in (("images", images), ("rows", rows)):
        df.to_parquet(d / f"{name}.tmp.parquet", index=False)
        (d / f"{name}.tmp.parquet").replace(d / f"{name}.parquet")
    keys = sorted(cams)
    H, rms, intr = (np.stack([cams[k][i] for k in keys]) for i in range(3))
    trip = []
    for k, it in zip(keys, intr):
        W, Hh = wh or (2 * it[2], 2 * it[3])
        g = np.stack(np.meshgrid(np.linspace(0, W, 33), np.linspace(0, Hh, 33)), -1)
        trip.append(np.abs(DT.distort(DT.undistort(g, it), it) - g).max())
    np.savez(d / "cams.tmp.npz", key=np.array(keys), H=H, rms=rms, intr=intr, roundtrip=np.array(trip))
    (d / "cams.tmp.npz").replace(d / "cams.npz")
    rl.info(f"{ds}: {len(rows)} trunk rows over {rows.stem.nunique()} files, {len(images)} unique images, {len(keys)} "
            f"calibrations (road -> native -> road error max {rms.max():.4f} road px; undistortion round trip max {max(trip):.4f} px)")
    rl.event("list", ds=ds, rows=len(rows), images=len(images), cams=len(keys), rms_max=float(rms.max()), roundtrip_max=float(max(trip)))


def _frame(rows_img: list):
    """rows_img: per trunk row an image key (None = zero image) -> (img_id per row, unique keys in first-use order)."""
    ids, uniq = {}, []
    out = np.full(len(rows_img), -1, np.int64)
    for j, k in enumerate(rows_img):
        if k is None:
            continue
        if k not in ids:
            ids[k] = len(uniq)
            uniq.append(k)
        out[j] = ids[k]
    return out, uniq


def list_nusc(rl):
    from jevdrive import nuscenes_zs as Z
    from jevdrive import op_adapt_data as D
    from jevdrive.common import dataroot
    idx = Z.load_index("trainval")
    stems, rws, keys, cams = [], [], [], {}
    for f in sorted(ROUND1.joinpath("nusc").glob("*.npz")):
        sc = idx["scenes"][f.stem]
        steps = np.load(f)["steps"]
        fr = D.scene_plan(idx, f.stem)["frame"][steps]
        paths = [str(dataroot() / sc["cams"]["CAM_FRONT"]["path"][i]) for i in fr]
        stems += [f.stem] * len(paths)
        rws += list(range(len(paths)))
        keys += [(p, f.stem) for p in paths]
        cams[f.stem] = DT.h_wod(Z.cam_calib(sc, "CAM_FRONT"))
    img, uniq = _frame(keys)
    images = pd.DataFrame({"img_id": np.arange(len(uniq)), "path": [u[0] for u in uniq], "cam": [u[1] for u in uniq]})
    _save("nusc", images, pd.DataFrame({"stem": stems, "row": rws, "img_id": img}), cams, rl)


def list_wod(ds, rl):
    from jevdrive import drive_backbones as DB
    from jevdrive import wod_zeroshot as Z
    plan = json.loads((DB.root() / DB.plan_name("subset" if ds == "wod" else "trainval")).read_text())
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    if ds == "wodtrain":
        calib |= json.loads((DB.root() / "op_calib_trainval.json").read_text())
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    spans = plan["spans"]
    stems, rws, keys, cams = [], [], [], {}
    for f in sorted(ROUND1.joinpath(ds).glob("*.npz")):
        names = [str(n) for n in np.load(f)["names"]]
        stems += [f.stem] * len(names)
        rws += list(range(len(names)))
        keys += names
        for s in {n.rsplit("-", 1)[0] for n in names} - set(cams):
            cams[s] = DT.h_wod(calib[s]["1"])
    img, uniq = _frame(keys)
    sp = [spans[n] for n in uniq]
    images = pd.DataFrame({"img_id": np.arange(len(uniq)), "name": uniq, "shard": [str(shard_dir / s[0]) for s in sp],
                           "off": [int(s[1]) for s in sp], "len": [int(s[2]) for s in sp],
                           "cam": [n.rsplit("-", 1)[0] for n in uniq]})
    _save(ds, images, pd.DataFrame({"stem": stems, "row": rws, "img_id": img}), cams, rl)


def list_p5(rl):
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from jevdrive import p5_openpilot as PP
    plan = json.loads((PP.root() / "op_plan.json").read_text())
    calibs = plan.get("calibs") or {"carla": plan.get("calib")}
    by = {s["key"]: s for s in plan["streams"]}
    stems, rws, keys = [], [], []
    for f in sorted(ROUND1.joinpath("p5").glob("*.npz")):
        st = by[f.stem]
        names = [str(n) for n in np.load(f)["names"]]
        assert names == [str(n) for n in st["names"]] and len(st["files"]) == len(names), f.stem
        stems += [f.stem] * len(names)
        rws += list(range(len(names)))
        keys += [(fl[0], st.get("seq", "carla")) for fl in st["files"]]
    img, uniq = _frame(keys)
    cams = {s: DT.h_wod(c["1"]) for s, c in calibs.items()}
    images = pd.DataFrame({"img_id": np.arange(len(uniq)), "path": [u[0] for u in uniq], "cam": [u[1] for u in uniq]})
    _save("p5", images, pd.DataFrame({"stem": stems, "row": rws, "img_id": img}), cams, rl)


def _nav_cams(entries, cams):
    from jevdrive import navsim_zs as Z
    for e in entries:
        for c in e["cams"]:
            k = str(Z.calib_key({"CAM_F0": c["CAM_F0"]}))
            if k not in cams:
                cams[k] = DT.h_nuplan(c["CAM_F0"])
            c["_key"] = k


def list_navtrain(rl):
    """navtrain_job of scripts/op_adapt_cache.py without decoding: the unique (prev, cur) image pairs of every log in the
    same first-use order; the row's image is the pair's current frame. Checked against the stored ctx."""
    from jevdrive import navsim_zs as Z
    from jevdrive import op_adapt_data as D
    keep = set(pd.read_parquet(D.root() / "navtrain_labels.parquet").token)
    by = defaultdict(list)
    for e in Z.load_index("navtrain", slim=True):
        if e["token"] in keep:
            by[e["log_name"]].append(e)
    T = np.round(np.arange(-8, 1) * 0.2, 3)
    slot = lambda t: int(np.searchsorted(Z.T_HIST2, t + 1e-6) - 1) if t >= -1.5 - 1e-6 else -1  # noqa: E731
    stems, rws, keys, cams = [], [], [], {}
    for log in sorted(by):
        f = ROUND1 / "navtrain" / f"{log}.npz"
        if not f.exists():
            continue
        ents = by[log]
        _nav_cams(ents, cams)
        paths, pairs, ctx = {}, {}, []
        for e in ents:
            fidx = lambda k: paths.setdefault(e["cams"][k]["CAM_F0"]["path"], (len(paths), e["cams"][k]["_key"]))[0]  # noqa: E731
            row = []
            for t in T:
                c = slot(t)
                if c < 0:
                    row.append(-1)
                    continue
                pr = slot(round(t - 0.2, 3))
                row.append(pairs.setdefault((fidx(pr) if pr >= 0 else -1, fidx(c)), len(pairs)))
            ctx.append(row)
        z = np.load(f)
        assert np.array_equal(np.array(ctx, np.int32), z["ctx"]), log
        inv = {v[0]: (p, v[1]) for p, v in paths.items()}
        cur = [inv[c] for _, c in pairs]
        stems += [log] * len(cur)
        rws += list(range(len(cur)))
        keys += cur
    img, uniq = _frame(keys)
    images = pd.DataFrame({"img_id": np.arange(len(uniq)), "path": [u[0] for u in uniq], "cam": [u[1] for u in uniq]})
    _save("navtrain", images, pd.DataFrame({"stem": stems, "row": rws, "img_id": img}), cams, rl)


def list_navtest(ds, rl):
    """navtest / navhard (two-stage): the 4 CAM_F0 keyframes (t = -1.5, -1.0, -0.5, 0) of every leaderboard token, in the
    order of runs/op_lb/lb_<ds>/tokens.txt; one token file det/tok/<ds>/<ds>.npz, rows = 4 * i + k."""
    from jevdrive import navsim_zs as Z
    split = {"navtest": "navtest", "navhard": "navhard_two_stage"}[ds]
    toks = (data_dir() / "runs" / "op_lb" / f"lb_{ds}" / "tokens.txt").read_text().split()
    by = {e["token"]: e for e in Z.load_index(split, slim=True)}
    ents = [by[t] for t in toks]
    cams = {}
    _nav_cams(ents, cams)
    keys = [(e["cams"][k]["CAM_F0"]["path"], e["cams"][k]["_key"]) for e in ents for k in range(4)]
    img, uniq = _frame(keys)
    images = pd.DataFrame({"img_id": np.arange(len(uniq)), "path": [u[0] for u in uniq], "cam": [u[1] for u in uniq]})
    rows = pd.DataFrame({"stem": ds, "row": np.arange(len(keys)), "img_id": img})
    _save(ds, images, rows, cams, rl)
    (DT.root(ds) / "tokens.txt").write_text("\n".join(toks))


def list_sim(rl):
    """Every finished Cosmos G4 pair; rows = stream * 24 + slot (C's cache-sim), slot s = 20 Hz frame 4 s of the clip."""
    import cosmos_openpilot as CO
    pairs = sorted(p.parent.name for p in (data_dir() / "runs" / "cosmos_full" / "pairs").glob("*/done.json"))
    n = len(pairs) * len(SIM_STREAMS) * SIM_SLOTS
    pi = np.repeat(np.arange(len(pairs)), len(SIM_STREAMS) * SIM_SLOTS)
    r = np.tile(np.arange(len(SIM_STREAMS) * SIM_SLOTS), len(pairs))
    images = pd.DataFrame({"img_id": np.arange(n), "pair": np.array(pairs)[pi], "stream": r // SIM_SLOTS,
                           "slot": r % SIM_SLOTS, "cam": "cosmos"})
    rows = pd.DataFrame({"stem": images.pair, "row": r, "img_id": images.img_id})
    _save("sim", images, rows, {"cosmos": DT.h_wod(CO.calib())}, rl)


# ================================================================ detection

class _Chunks:
    """Map-style dataset over batches of one chunk list: JPEG bytes (decoded on the GPU) or, for sim, a pair's 96 frames."""

    def __init__(self, images: pd.DataFrame, batches: list, sim: bool):
        self.im, self.b, self.sim = images, batches, sim

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        import torch
        rows = self.im.iloc[self.b[i]]
        if self.sim:
            from jevdrive.cosmos_full import load_pair
            pair = rows.pair.iloc[0]
            fr = {}
            for kind in ("carla", "cosmos"):
                p, m = load_pair(pair, kind)
                fr[(kind, 0)], fr[(kind, 1)] = p[::SIM_STEP][:SIM_SLOTS], m[::SIM_STEP][:SIM_SLOTS]
            x = np.stack([fr[SIM_STREAMS[s]][t] for s, t in zip(rows.stream, rows.slot)])
            return rows.img_id.to_numpy(), torch.from_numpy(x)
        blobs = []
        for r in rows.itertuples():
            if getattr(r, "shard", None):
                with open(r.shard, "rb") as f:
                    f.seek(r.off)
                    b = f.read(r.len)
            else:
                b = Path(r.path).read_bytes()
            blobs.append(torch.frombuffer(bytearray(b), dtype=torch.uint8))
        return rows.img_id.to_numpy(), blobs


def detect(a, rl):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.io import ImageReadMode, decode_jpeg
    ds = a.ds
    images = pd.read_parquet(DT.root(ds, "images.parquet"))
    sim = ds == "sim"
    out = DT.root(ds, "raw")
    out.mkdir(parents=True, exist_ok=True)
    i, n = map(int, a.part.split("/"))
    if sim:                                          # one batch = one pair (96 frames), 40 pairs per chunk
        grp = images.groupby("pair", sort=True).indices
        pb = [grp[p] for p in sorted(grp)]
        chunks = [pb[k:k + 40] for k in range(0, len(pb), 40)]
    else:
        ids = np.arange(len(images))
        chunks = [[ids[s:s + a.batch] for s in range(c, min(c + CHUNK, len(ids)), a.batch)] for c in range(0, len(ids), CHUNK)]
    todo = [k for k in range(len(chunks)) if k % n == i and not (out / f"part-{k:05d}.npz").exists()]
    rl.info(f"{ds} part {i}/{n}: {len(todo)} chunks to do of {len(chunks)}")
    if not todo:
        return
    det = DT.Yolo()
    batches, owner = [], []
    for k in todo:
        batches += chunks[k]
        owner += [k] * len(chunks[k])
    dl = DataLoader(_Chunks(images, batches, sim), batch_size=None, num_workers=a.workers, prefetch_factor=4,
                    persistent_workers=False)
    acc, t0, t_last, done_img = defaultdict(list), time.time(), time.time(), 0
    left = {k: len(chunks[k]) for k in todo}
    for bi, (idx, x) in enumerate(dl):
        if sim:
            parts = [x[j:j + a.batch].cuda(non_blocking=True).permute(0, 3, 1, 2) for j in range(0, len(x), a.batch)]
        else:
            dec = decode_jpeg(x, mode=ImageReadMode.RGB, device="cuda")
            shapes = defaultdict(list)
            for j, d in enumerate(dec):
                shapes[tuple(d.shape)].append(j)
            assert len(shapes) == 1, f"mixed image sizes in one batch: {list(shapes)}"
            parts = [torch.stack(dec)]
        res = [det(p) for p in parts]
        r = {f: np.concatenate([q[f] for q in res]) for f in res[0]}
        k = owner[bi]
        acc[k].append((idx, r, x[0].shape if sim else tuple(dec[0].shape)))
        left[k] -= 1
        done_img += len(idx)
        if left[k] == 0:
            items = acc.pop(k)
            hw = np.concatenate([np.tile(np.array(s[-2:] if not sim else s[:2]), (len(ii), 1)) for ii, _, s in items])
            np.savez(out / f"part-{k:05d}.tmp.npz", img_id=np.concatenate([ii for ii, _, _ in items]), hw=hw.astype(np.int16),
                     **{f: np.concatenate([q[f] for _, q, _ in items]) for f in items[0][1]})
            (out / f"part-{k:05d}.tmp.npz").replace(out / f"part-{k:05d}.npz")
            el = time.time() - t0
            rl.info(f"chunk {k}: {done_img} images, {done_img / el:.0f} img/s")
            rl.event("chunk", chunk=k, images=done_img, img_s=done_img / el)
    rl.event("end", images=done_img, wall_s=time.time() - t0)


def load_raw(ds) -> dict:
    parts = sorted(DT.root(ds, "raw").glob("part-*.npz"))
    parts = [p for p in parts if ".tmp" not in p.name]
    zs = [dict(np.load(p)) for p in parts]
    r = {f: np.concatenate([z[f] for z in zs]) for f in zs[0]}
    o = np.argsort(r["img_id"])
    return {f: v[o] for f, v in r.items()}


# ================================================================ PCA and tokens

def pca(a, rl):
    rng = np.random.default_rng(0)
    parts = sorted(p for p in DT.root("wodtrain", "raw").glob("part-*.npz") if ".tmp" not in p.name)
    feats = []
    for p in parts:
        z = np.load(p)
        m = z["cls"] >= 0
        feats.append(z["feat"][m])
    F = np.concatenate(feats)
    sel = rng.choice(len(F), min(len(F), a.max_rows), replace=False)
    res = DT.fit_pca(F[np.sort(sel)])
    np.savez(DT.root("pca.npz"), **res)
    rl.info(f"PCA-{DT.PCA_D} on {len(sel)} of {len(F)} WOD-train detections: explained {res['explained'].sum():.3f} "
            f"({np.round(res['explained'], 3).tolist()})")
    rl.event("pca", rows=len(sel), total=len(F), explained=float(res["explained"].sum()))


def tokens(a, rl):
    ds = a.ds
    pc = dict(np.load(DT.root("pca.npz")))
    raw = load_raw(ds)
    images = pd.read_parquet(DT.root(ds, "images.parquet"))
    rows = pd.read_parquet(DT.root(ds, "rows.parquet"))
    cams = np.load(DT.root(ds, "cams.npz"))
    assert np.array_equal(raw["img_id"], images.img_id.to_numpy()), f"{ds}: detections missing for some images"
    ci = pd.Series(np.arange(len(cams["key"])), index=cams["key"]).reindex(images.cam.astype(str)).to_numpy()
    tok, mask = DT.build_tokens(raw["cls"], raw["score"].astype(np.float32), raw["box"], raw["feat"], cams["H"][ci],
                                cams["intr"][ci], pc)
    tok = np.concatenate([tok, np.zeros((1,) + tok.shape[1:], tok.dtype)])          # row -1 = the zero image: no tokens
    mask = np.concatenate([mask, np.zeros((1,) + mask.shape[1:], bool)])
    od = DT.root("tok", ds)
    od.mkdir(parents=True, exist_ok=True)
    for stem, g in rows.groupby("stem", sort=False):
        g = g.sort_values("row")
        assert np.array_equal(g.row.to_numpy(), np.arange(len(g)))
        ii = g.img_id.to_numpy()
        np.savez(od / f"{stem}.npz", tok=tok[ii], mask=mask[ii], img_id=ii)
    m = mask[:-1]
    per = m.sum(1)
    cl = raw["cls"][m[:, :]]
    rl.info(f"{ds}: tokens for {rows.stem.nunique()} files / {len(rows)} rows; per image detections mean {per.mean():.2f}, "
            f"= 8 (truncated) {np.mean(per == 8):.3f}, none {np.mean(per == 0):.3f}; classes ped / cyc / veh "
            f"{[int((cl == c).sum()) for c in range(3)]}")
    rl.event("tokens", ds=ds, files=int(rows.stem.nunique()), rows=len(rows), det_mean=float(per.mean()),
             trunc=float(np.mean(per == 8)), none=float(np.mean(per == 0)))


# ================================================================ adapter unit checks

def check(a, rl):
    import torch
    from jevdrive import op_adapt as A
    dev = torch.device("cuda")
    net = A.load("cinque", torch.float16, trainable=A.stage4_weights()).to(dev).eval()
    files = sorted(ROUND1.joinpath("nusc").glob("*.npz"))[:4]
    xs, ts, ms = [], [], []
    rng = np.random.default_rng(0)
    for f in files:
        z = np.load(f)
        T = z["trunk"]
        have = DT.tok_path(f).exists()
        tk, mk = DT.load_tok(f) if have else (None, None)
        for j in rng.choice(np.arange(16, len(T)), 4, replace=False):
            ctx = j - 2 * np.arange(A.CONTEXT - 1, -1, -1)
            xs.append(T[ctx])
            if have:
                ts.append(tk[ctx]), ms.append(mk[ctx])
            else:
                ts.append(rng.normal(size=(A.CONTEXT, DT.K_TOK, DT.TOK_D)).astype(np.float16))
                ms.append(rng.random((A.CONTEXT, DT.K_TOK)) < 0.5)
    x = torch.as_tensor(np.stack(xs)).to(dev)
    tok, msk = torch.as_tensor(np.stack(ts)).to(dev), torch.as_tensor(np.stack(ms)).to(dev)
    msk[0, :, :] = False                                           # one sample with no detection at all
    tc = torch.tensor([[1.0, 0.0]], device=dev).expand(len(x), 2)
    valid = torch.ones(len(x), A.CONTEXT, dtype=torch.bool, device=dev)
    valid[1, :3] = False
    AT = (0.275, 0.525)
    torch.manual_seed(0)
    ad = DT.DetAdapter().to(dev)
    n_par = sum(p.numel() for p in ad.parameters())
    with torch.no_grad():
        o0 = A.stage4_policy(net, x, AT, tc, valid)
        o1 = DT.stage4_policy_det(net, ad, x, tok, msk, AT, tc, valid)
    same = {k: bool(torch.equal(o0[k], o1[k])) for k in o0}
    o = DT.stage4_policy_det(net, ad, x, tok, msk, AT, tc, valid)
    o["outputs"].float().pow(2).mean().backward()
    g = {n: (None if p.grad is None else float(p.grad.abs().max())) for n, p in ad.named_parameters()}
    gate_grad = g["gate"]
    with torch.no_grad():
        ad.gate.fill_(0.5)
        o2 = DT.stage4_policy_det(net, ad, x, tok, msk, AT, tc, valid)
        o3 = DT.stage4_policy_det(net, ad, x, tok, torch.zeros_like(msk), AT, tc, valid)
    moved = float((o2["outputs"].float() - o0["outputs"].float()).abs().max())
    nodet = bool(torch.equal(o3["outputs"], o0["outputs"]))
    ad.zero_grad()
    DT.stage4_policy_det(net, ad, x, tok, msk, AT, tc, valid)["outputs"].float().pow(2).mean().backward()
    reach = {n: bool(p.grad is not None and p.grad.abs().max() > 0) for n, p in ad.named_parameters()}
    res = {"params": n_par, "real_tokens": bool(ts and DT.tok_path(files[0]).exists()), "step0_bit_identical": same,
           "gate_grad_step0": gate_grad, "other_grads_step0_max": max(v for k, v in g.items() if k != "gate" and v is not None),
           "gate0.5_output_change": moved, "gate0.5_all_masked_bit_identical": nodet, "grads_reach_all_at_gate0.5": reach}
    ok = all(same.values()) and gate_grad and gate_grad > 0 and np.isfinite(gate_grad) and moved > 0 and nodet and all(reach.values())
    res["pass"] = bool(ok)
    DT.root("checks").mkdir(parents=True, exist_ok=True)
    (DT.root("checks") / "adapter.json").write_text(json.dumps(res, indent=1))
    rl.info(json.dumps(res))
    rl.event("check", **{k: v for k, v in res.items() if not isinstance(v, dict)})
    assert ok, res


def main():
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "detect", "pca", "tokens", "check"))
    ap.add_argument("ds", nargs="?", default="")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=400_000, help="pca: detections sampled for the fit")
    a = ap.parse_args()
    rl = RunLog("op_adapt_r2", "det", f"{a.cmd}-{a.ds}" if a.ds else a.cmd)
    rl.event("start", args=vars(a))
    if a.cmd == "list":
        L = {"nusc": list_nusc, "p5": list_p5, "navtrain": list_navtrain, "sim": list_sim,
             "wod": lambda r: list_wod("wod", r), "wodtrain": lambda r: list_wod("wodtrain", r),
             "navtest": lambda r: list_navtest("navtest", r), "navhard": lambda r: list_navtest("navhard", r)}
        L[a.ds](rl)
    else:
        {"detect": detect, "pca": pca, "tokens": tokens, "check": check}[a.cmd](a, rl)


if __name__ == "__main__":
    main()
