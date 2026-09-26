"""Night queue 2, N4: the fast channel without the lift -- image-plane detection tokens (todos/2026-09-26-night-queue-2.md,
N4 and the [B] 10:12 entry, written before any N4 number).

  detect   (envs/ultralytics) YOLO26x-seg 640 fp16 on E5's image list with a forward pre-hook on the Segment head:
           per kept detection the RoIAlign (1 x 1) of the three neck maps, concatenated (384 + 768 + 768 = 1920)
  tokens   PCA-16 of the detection features (fitted on role == train frames), then per camera the first k = 8
           detections by box bottom (y1, lowest first): camera one-hot, (u, v, w, h) / (W, H), class one-hot, score,
           PCA-16, mask bit -> 3 x 8 x 28 = 672 per row
  fit      E5's student (elicit_e5.train_student, pair-difference loss, no teacher) on op (+) B and op (+) B (+) A,
           seeds 0-2, both models; A = E5's stored student A; one p5_exam.exam over A / B / C
  latency  B's head side (RoIAlign + PCA on the GPU, token build on the CPU, MLP) at batch 1, three cameras

    envs/ultralytics/bin/python -m jevdrive.night2_n4 detect --part 0/6
    .venv/bin/python -m jevdrive.night2_n4 tokens | fit | latency
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
K_TOK, PCA_D, SCORE = 8, 16, 0.25
CAMS = ("front", "front_left", "front_right")
CLASSES = ("pedestrian", "cyclist", "vehicle")
TOK_D = 3 + 4 + 3 + 1 + PCA_D + 1
E5_FIT = "runs/elicitation/e5-fit/20260926-021421"
WEIGHTS = "yolo26x-seg.pt"


def out(*p) -> Path:
    d = data_dir() / "processed" / "night2" / "n4"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


# ---------------------------------------------------------------- detection with per-detection features

class Detector:
    """Ultralytics YOLO (COCO -> three classes, fastperc.COCO_MAP) plus the Segment head's input maps; __call__ returns
    per image (labels, scores, xyxy in original pixels, RoI features (m, 1920) fp16)."""

    def __init__(self, weights: str = WEIGHTS, imgsz: int = 640):
        import torch
        from ultralytics import YOLO
        from .fastperc import COCO_MAP, models_dir
        self.m = YOLO(str(models_dir() / "ultralytics" / weights))
        self.imgsz = imgsz
        head = self.m.model.model[-1]
        self.strides = [float(s) for s in head.stride]
        self._x, self._in, self.checked = None, None, False
        head.register_forward_pre_hook(lambda mod, a: setattr(self, "_x", [t.detach() for t in a[0]]))
        self.m.model.register_forward_pre_hook(lambda mod, a: setattr(self, "_in", a[0].shape))
        self.cmap = {i: COCO_MAP.get(n) for i, n in self.m.names.items()}
        self.torch = torch

    def __call__(self, imgs):
        from torchvision.ops import roi_align
        torch = self.torch
        rs = self.m.predict(imgs, imgsz=self.imgsz, conf=SCORE, half=True, retina_masks=True, verbose=False)
        _, _, hi, wi = self._in
        out_ = []
        for b, r in enumerate(rs):
            lab = np.array([self.cmap.get(c) or "" for c in r.boxes.cls.int().tolist()])
            keep = torch.as_tensor(np.isin(lab, CLASSES), device=r.boxes.data.device)
            xyxy, sc = r.boxes.xyxy[keep].float(), r.boxes.conf[keep].float()
            H, W = r.orig_shape
            g = min(self.imgsz / H, self.imgsz / W)
            nw, nh = int(round(W * g)), int(round(H * g))
            left, top = round((wi - nw) / 2 - 0.1), round((hi - nh) / 2 - 0.1)
            bi = xyxy * g + torch.tensor([left, top, left, top], device=xyxy.device, dtype=xyxy.dtype)
            if not self.checked and len(bi):         # the inverse of Ultralytics' own input -> original mapping
                from ultralytics.utils.ops import scale_boxes
                back = scale_boxes((hi, wi), bi.clone(), (H, W))
                assert float((back - xyxy).abs().max()) < 1.0, "letterbox mapping does not invert"
                self.checked = True
            feats = []
            for x, s in zip(self._x, self.strides):
                if not len(bi):
                    feats.append(torch.zeros((0, x.shape[1]), device=x.device))
                    continue
                f = roi_align(x[b:b + 1].float(), [bi], output_size=1, spatial_scale=1.0 / s, sampling_ratio=2, aligned=True)
                feats.append(f.flatten(1))
            out_.append((lab[np.isin(lab, CLASSES)], sc.cpu().numpy(), xyxy.cpu().numpy(),
                         torch.cat(feats, 1).half().cpu().numpy(), (H, W)))
        return out_


def detect(part: str, batch: int = 12, shard: int = 3000, workers: int = 3):
    """Slice `part` = i/n of E5's image list (interleaved by shard) -> out/dets/part-<k>.parquet + feat-<k>.npy."""
    import torch
    from torch.utils.data import DataLoader
    from .sam_detect import _collate, _Images
    i, n = map(int, part.split("/"))
    t = pd.read_parquet(data_dir() / "processed/elicit_e5/images.parquet")
    det = Detector()
    d = out("dets")
    d.mkdir(exist_ok=True)
    for k, s0 in enumerate(range(0, len(t), shard)):
        if k % n != i or (d / f"part-{k:04d}.parquet").exists():
            continue
        rows = t.iloc[s0:s0 + shard].to_dict("records")
        dl = DataLoader(_Images(rows), batch_size=batch, num_workers=workers, collate_fn=_collate, prefetch_factor=4)
        recs, feats, ts = [], [], time.time()
        for idx, ims in dl:
            res = det([np.ascontiguousarray(x.permute(1, 2, 0).numpy()[:, :, ::-1]) for x in ims])
            for j, (lab, sc, bx, f, (H, W)) in zip(idx, res):
                if len(lab):
                    recs.append(pd.DataFrame({"key": rows[j]["key"], "prompt": lab, "score": sc, "x0": bx[:, 0], "y0": bx[:, 1],
                                              "x1": bx[:, 2], "y1": bx[:, 3], "H": H, "W": W}))
                    feats.append(f)
        df = pd.concat(recs, ignore_index=True)
        np.save(d / f"feat-{k:04d}.npy", np.concatenate(feats))
        df.to_parquet(d / f"part-{k:04d}.tmp", index=False)
        (d / f"part-{k:04d}.tmp").rename(d / f"part-{k:04d}.parquet")
        log.info(f"part {k}: {len(rows)} images, {len(df)} detections, {1000 * (time.time() - ts) / len(rows):.1f} ms/image")
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- tokens

def _index() -> pd.DataFrame:
    return pd.read_parquet(data_dir() / "processed/carla_p5v1_ba/index.parquet")


def tokens(rl):
    """PCA-16 on the train-role detections, then the 672-d image-plane rows in P5 v1 BA index order ([B] 10:12 (3)-(4))."""
    import glob
    t = _index()
    parts = sorted(glob.glob(str(out("dets", "part-*.parquet"))))
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    F = np.concatenate([np.load(p.replace("part-", "feat-").replace(".parquet", ".npy")) for p in parts]).astype(np.float32)
    assert len(F) == len(d)
    fn, cam = d.key.str.split("|").str[0].to_numpy(), d.key.str.split("|").str[1].to_numpy()
    role = t.set_index("frame_name").role.reindex(fn).to_numpy()
    tr = role == "train"
    mu = F[tr].mean(0)
    C = np.cov((F[tr] - mu).T)
    ev, V = np.linalg.eigh(C)
    V = V[:, ::-1][:, :PCA_D].astype(np.float32)
    Z = (F - mu) @ V
    rl.log.info(f"{len(d)} detections over {d.key.nunique()} images; PCA-{PCA_D} fitted on {tr.sum()} train-role detections, "
                f"explained variance {ev[::-1][:PCA_D].sum() / ev.sum():.3f}")
    np.savez(out("pca.npz"), mu=mu, V=V, explained=ev[::-1][:PCA_D] / ev.sum())
    # E5's stored detections: per-image count agreement (description)
    e5 = pd.concat([pd.read_parquet(p, columns=["key", "prompt", "score"]) for p in glob.glob(str(data_dir() / "processed/elicit_e5/dets/*/part-*.parquet"))])
    e5 = e5[(e5.score > SCORE) & e5.prompt.isin(CLASSES)].groupby("key").size()
    mine = d.groupby("key").size()
    keys = pd.read_parquet(data_dir() / "processed/elicit_e5/images.parquet").key
    agree = float((mine.reindex(keys).fillna(0).to_numpy() == e5.reindex(keys).fillna(0).to_numpy()).mean())
    rl.log.info(f"per-image detection count equal to E5's stored detections on {agree:.4f} of {len(keys)} images")
    rl.event("n4_tokens", detections=len(d), images=int(d.key.nunique()), explained=float(ev[::-1][:PCA_D].sum() / ev.sum()), count_agree=agree)
    # first k per camera by box bottom, lowest first
    d = d.assign(_z=list(range(len(d))), fn=fn, cam=cam).sort_values(["key", "y1"], ascending=[True, False], kind="stable")
    d["rank"] = d.groupby("key").cumcount()
    d = d[d["rank"] < K_TOK]
    rows = pd.Series(np.arange(len(t)), index=t.frame_name).reindex(d.fn).to_numpy()
    ci = d.cam.map({c: i for i, c in enumerate(CAMS)}).to_numpy()
    tok = np.zeros((len(t), len(CAMS), K_TOK, TOK_D), np.float32)
    W, H = d.W.to_numpy(np.float32), d.H.to_numpy(np.float32)
    v = np.zeros((len(d), TOK_D), np.float32)
    v[np.arange(len(d)), ci] = 1
    v[:, 3], v[:, 4] = (d.x0 + d.x1).to_numpy() / 2 / W, (d.y0 + d.y1).to_numpy() / 2 / H
    v[:, 5], v[:, 6] = (d.x1 - d.x0).to_numpy() / W, (d.y1 - d.y0).to_numpy() / H
    v[np.arange(len(d)), 7 + d.prompt.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()] = 1
    v[:, 10] = d.score.to_numpy()
    v[:, 11:11 + PCA_D] = Z[d._z.to_numpy()]
    v[:, -1] = 1
    tok[rows, ci, d["rank"].to_numpy()] = v
    np.save(out("tokB.npy"), tok.reshape(len(t), -1))
    rl.log.info(f"tokens: {len(t)} rows x {tok.shape[1] * K_TOK * TOK_D}; rows with any detection {(tok[..., -1].sum((1, 2)) > 0).mean():.3f}, "
                f"camera-images truncated at k = {K_TOK}: {(d.groupby('key').size() >= K_TOK).mean():.3f}")


# ---------------------------------------------------------------- students

def _prior(tr, ev, t, F, Ego, Xop):
    """reactivity_mc.fit_fold's prior only (`ridge ego` then `ridge_late` on op `temporal`, the same calls): fit_fold
    also solves the five M-C arms on the 3072-d dual stream with float64 eighs on the CPU, which N4 never reads."""
    from types import SimpleNamespace
    from . import planner, waymo_stage_a as sa
    n = len(t)
    sp = SimpleNamespace(train=tr, val=ev, seq=t.base_id.to_numpy())
    Xe = planner.standardize(Ego, tr)
    _, _, We = sa.ridge_cv(Xe, F, sp, F.reshape(n, 20, 2).cpu().numpy())
    base = planner.linear_apply(We, Xe, np.arange(n))[0]
    Xi = planner.standardize(Xop, tr)
    R0 = F - base
    _, _, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(n, 20, 2).cpu().numpy())
    return base + planner.linear_apply(Wp, Xi, np.arange(n))[0]



def fit(rl, models=("cinque", "lebowski")):
    """Arms B (op + image-plane tokens) and C (op + tokens + E5's lifted embedding): E5's student and fold set-up
    unchanged, pair-difference loss only; arm A = E5's stored student A; one exam over A / B / C."""
    import torch
    from . import elicit_e5 as E5, elicit_i3 as I, p5_exam as E, p5_openpilot, reactivity_mc as MC
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        op = p5_openpilot.load(t, models, sub="op_streams_vis")
    n = len(t)
    B = torch.as_tensor(np.load(out("tokB.npy")), device="cuda")
    A = torch.as_tensor(np.load(data_dir() / "processed/elicit_e5/embed.npy"), device="cuda")
    fold = E.folds(t, pairs)
    F = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    Ego = torch.as_tensor(E.ego_input(t, past), device="cuda")
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    role = t.role.to_numpy()
    ref = np.load(data_dir() / E5_FIT / "preds_obs.npz")
    assert (ref["rows"] == obs_rows).all()
    at = pd.Series(np.arange(len(obs_rows)), index=obs_rows)
    preds = {}
    mB = torch.as_tensor(np.arange(B.shape[1]) % TOK_D == TOK_D - 1, device="cuda")
    mA = torch.as_tensor(np.arange(64) % 8 == 7, device="cuda")

    def z(X, tr, mask=None):
        mu, sd = X[tr].mean(0), X[tr].std(0, correction=0).clamp_min(1e-6)
        s_ = (X - mu) / sd
        return (s_ if mask is None else torch.where(mask, X, s_)) / np.sqrt(X.shape[1])
    for m in models:
        Xop = torch.as_tensor(op[f"op-{m} temporal"], device="cuda")
        for key in [f"prior [{m}]", f"M-C pair [{m}]"] + [f"E5 A s{sd} [{m}]" for sd in E5.SEEDS]:
            preds[key] = np.full((n, 20, 2), np.nan, np.float32)
            preds[key][obs_rows] = ref[key]
        for f in range(E.K_FOLDS):
            ev = obs_rows[fold[obs_rows] == f]
            tr = np.flatnonzero((role == "train") & (fold != f))
            prior = _prior(tr, ev, t, F, Ego, Xop)
            diff = float(np.abs(prior[ev].reshape(-1, 20, 2).cpu().numpy() - ref[f"prior [{m}]"][at[ev].to_numpy()]).max())
            rl.event("n4_prior_check", model=m, fold=f, max_abs_diff=diff)
            assert diff < 1e-2, f"prior of fold {f} does not reproduce E5's ({diff})"
            zo, zb, za = z(Xop, tr), z(B, tr, mB), z(A, tr, mA)
            keep = fold[pr_ip] != f
            ip, im, grp = pr_ip[keep], pr_im[keep], pr_group[keep]
            R = (F[ip] - F[im]) - (prior[ip] - prior[im])
            for arm, X in (("B", torch.cat([zo, zb], 1).float()), ("C", torch.cat([zo, zb, za], 1).float())):
                for sd in E5.SEEDS:
                    d = E5.train_student(X, ip, im, R, tr, grp, None, sd, rl, f"{m} f{f} {arm}")
                    preds.setdefault(f"N4 {arm} s{sd} [{m}]", np.full((n, 20, 2), np.nan, np.float32))[ev] = \
                        (prior[ev] + d[ev]).reshape(-1, 20, 2).cpu().numpy()
            rl.log.info("%s fold %d done", m, f)
            del X, zo, zb, za
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

def build_tokens(res, pca_mu, pca_V) -> np.ndarray:
    """Detector output for the three cameras of one frame -> the 672-d row (the same rule as `tokens`)."""
    tok = np.zeros((len(CAMS), K_TOK, TOK_D), np.float32)
    for c, (lab, sc, bx, f, (H, W)) in enumerate(res):
        if not len(lab):
            continue
        o = np.argsort(-bx[:, 3], kind="stable")[:K_TOK]
        z = (f[o].astype(np.float32) - pca_mu) @ pca_V
        v = tok[c, :len(o)]
        v[:, c] = 1
        v[:, 3], v[:, 4] = (bx[o, 0] + bx[o, 2]) / 2 / W, (bx[o, 1] + bx[o, 3]) / 2 / H
        v[:, 5], v[:, 6] = (bx[o, 2] - bx[o, 0]) / W, (bx[o, 3] - bx[o, 1]) / H
        v[np.arange(len(o)), 7 + np.array([CLASSES.index(x) for x in lab[o]])] = 1
        v[:, 10], v[:, 11:11 + PCA_D], v[:, -1] = sc[o], z, 1
    return tok.reshape(-1)


def latency(rl, n: int = 200, warm: int = 20):
    """E5's protocol at batch 1, three cameras in one call (envs/ultralytics, an idle card): YOLO alone; YOLO + the
    RoIAlign hook; token build (CPU); the B / C MLP forward (GPU). p50 / p95 per component."""
    import torch
    from .elicit_e5 import _mlp
    from .sam_detect import _reader, decode
    t = _index()
    fr = t[t.role == "obs"].frame_name.iloc[:: max(1, len(t[t.role == "obs"]) // (n + warm))].iloc[:n + warm]
    lst = pd.read_parquet(data_dir() / "processed/elicit_e5/images.parquet").set_index("key")
    frames = [[np.ascontiguousarray(decode(_reader({"path": lst.loc[f"{f}|{c}", "path"]})).permute(1, 2, 0).numpy()[:, :, ::-1])
               for c in CAMS] for f in fr]
    det = Detector()
    pca = np.load(out("pca.npz"))
    mu, V = pca["mu"], pca["V"]
    nets = {"B": _mlp(512 + len(CAMS) * K_TOK * TOK_D, 0).eval(), "C": _mlp(512 + len(CAMS) * K_TOK * TOK_D + 64, 0).eval()}
    sync = torch.cuda.synchronize
    ts = {k: [] for k in ("yolo_only", "yolo_roi", "tokens_cpu", "mlp_B", "mlp_C")}
    for ims in frames:
        sync(); a = time.perf_counter()
        det.m.predict(ims, imgsz=det.imgsz, conf=SCORE, half=True, retina_masks=True, verbose=False)
        sync(); ts["yolo_only"].append(1000 * (time.perf_counter() - a))
    for ims in frames:
        sync(); a = time.perf_counter()
        res = det(ims)
        sync(); b = time.perf_counter()
        x = build_tokens(res, mu, V)
        c = time.perf_counter()
        for arm, net in nets.items():
            xx = torch.zeros(1, net[0].in_features, device="cuda")
            xx[0, 512:512 + len(x)] = torch.as_tensor(x, device="cuda")
            sync(); d0 = time.perf_counter()
            with torch.no_grad():
                net(xx)
            sync(); ts[f"mlp_{arm}"].append(1000 * (time.perf_counter() - d0))
        ts["yolo_roi"].append(1000 * (b - a)), ts["tokens_cpu"].append(1000 * (c - b))
    res_ = {k: {"p50_ms": float(np.percentile(v[warm:], 50)), "p95_ms": float(np.percentile(v[warm:], 95)), "n": len(v) - warm}
            for k, v in ts.items()}
    res_["gpu"] = torch.cuda.get_device_name(0)
    (rl.dir / "latency.json").write_text(json.dumps(res_, indent=1))
    rl.log.info(json.dumps(res_, indent=1))


# ---------------------------------------------------------------- figure

def figs(res_dir="research/results/night2/N4", out_dir="research/figs"):
    """Pedestrian flip rate, cut-in delta vs the prior and DynamicObjectCrossing non-reactive flips for A (E5, lifted),
    B (image plane) and C (both): seed 0 with the route-bootstrap CI, seeds 1-2 as crosses; both models."""
    import matplotlib.pyplot as plt
    from . import plots
    c = pd.read_csv(Path(res_dir) / "criteria.csv")
    nr = pd.read_csv(Path(res_dir) / "nonreactive.csv")
    nr = nr[nr.scope == "DynamicObjectCrossing"].set_index("arm").nonreactive_flip
    arms = [("prior", "Prior"), ("M-C pair", "M-C"), ("E5 A", "A: lifted"), ("N4 B", "B: image plane"), ("N4 C", "C: both")]
    cols = {"cinque": plots.OKABE_ITO[5], "lebowski": plots.OKABE_ITO[6]}
    key = lambda a, m, sd: f"{a} s{sd} [{m}]" if a.startswith(("E5", "N4")) else f"{a} [{m}]"  # noqa: E731
    with plots.mpl.rc_context(plots.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(plots.PAGE, 1.9))
        for ax, spec in zip(axes, (("ped_flip", "ped_lo", "ped_hi", "Pedestrian flip rate (%)"),
                                   ("cutin_delta_vs_prior", "cutin_lo", "cutin_hi", r"Cut-in $\Delta$ vs prior (pp)"),
                                   (None, None, None, "DOC non-reactive flips (%)"))):
            val, lo, hi, lab = spec
            for k, (m, col) in enumerate(cols.items()):
                for i, (a, _) in enumerate(arms):
                    x = i + (k - 0.5) * 0.3
                    if val is None:
                        ax.plot(x, 100 * nr[key(a, m, 0)], "o", color=col, ms=3, label=m.capitalize() if i == 0 else None)
                        seeds = [100 * nr[key(a, m, sd)] for sd in (1, 2)] if a.startswith(("E5", "N4")) else []
                    else:
                        r0 = c[c.arm == key(a, m, 0)].iloc[0]
                        ax.errorbar(x, 100 * r0[val], yerr=[[100 * (r0[val] - r0[lo])], [100 * (r0[hi] - r0[val])]], fmt="o",
                                    color=col, ms=3, lw=0.8, capsize=1.5, label=m.capitalize() if i == 0 else None)
                        seeds = [100 * c[c.arm == key(a, m, sd)].iloc[0][val] for sd in (1, 2)] if a.startswith(("E5", "N4")) else []
                    for v in seeds:
                        ax.plot(x + 0.07, v, "x", color=col, ms=3, mew=0.7)
            ax.set_xticks(range(len(arms)), [n for _, n in arms], rotation=25, ha="right")
            ax.set_ylabel(lab)
            ax.axhline(0, color="0.5", lw=0.5)
        axes[0].legend(loc="upper left")
        fig.tight_layout(w_pad=0.6)
        plots.save(fig, Path(out_dir), "night2-n4-image-plane")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("detect", "tokens", "fit", "latency", "figs"))
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    if a.step == "detect":
        return detect(a.part)
    if a.step == "figs":
        return figs()
    from .runlog import RunLog
    rl = RunLog("night2", f"n4-{a.step}")
    {"tokens": tokens, "fit": fit, "latency": latency}[a.step](rl)
    rl.close()


if __name__ == "__main__":
    main()
