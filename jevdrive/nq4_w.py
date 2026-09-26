"""Night queue 4, W: an action-conditioned latent world model and the paired test
(todos/2026-09-26-night-queue-4.md, section W and its [E] entries, written before any number).

z = openpilot Cinque `temporal` (512) | V-JEPA 2 ViT-L `mean` (3 cameras x 1024), 5 Hz. The predictor sees 8 steps of
history (z, previous action, speed) and 10 future actions, and predicts the 10 future z in one pass. Frozen probes
trained on real z of the training routes read the predicted z.

  vjepa    V-JEPA 2 `mean` for universe rows without a stored feature (n6_backbones' model and transform, 4-frame
           clip per camera) -> processed/<set>/w_vjepa2/c<NNN>.npz (resumable chunks)
  check    the extractor against the stored BA features on a few rows (bf16 batch noise only)
  prep     universe table, z, actions and the ego-frame labels -> processed/nq4_w/{meta.parquet, z.npy}
  bench    training-step throughput: CPU DataLoader + fp32 against GPU-resident data + bf16 (the profile)
  run      one seed: 5 folds of training, probes on real z, the paired and action readouts -> runs/nq4/w/seed<s>/
  report   3-seed tables and verdicts -> research/results/nq4/w/
"""
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "nq4" / "w"
SETS = ("carla_p5v1_ba", "carla_p5v1_pdm", "carla_p6")
CAMS = ("front", "front_left", "front_right")
D_OP, D_VJ = 512, 3072
HIST, FUT, TICKS, DT = 8, 10, 4, 0.2
WIN = HIST + FUT
PED_FAM = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian", "ParkingCrossingPedestrian")
CUTIN_FAM = ("StaticCutIn", "ParkingCutIn", "HighwayCutIn")
CLS_PROBE = {"ped": "ped", "cutin": "occ", "obstacle": "a"}
PROBES_CLS = ("a", "b", "c", "ped", "occ")
VIS_DZ, PED_X, PED_Y, OCC_X, LANE_Y, D_MAX = 5.0, 30.0, 4.0, 30.0, 1.75, 40.0
EGO_R = 2.0                                      # actors this close to the ego position are the ego itself
BRAKE, V_HAZ, SHIFT_M, OMEGA_MAX = -5.0, 3.0, 3.5, 0.6
CFG = dict(d=512, layers=6, heads=8, ff=2048, drop=0.1, lr=3e-4, wd=0.05, batch=256, warmup=500, steps=8000,
           eval_every=1000, clip=1.0)
RIDGE_LAMS = (1e1, 1e2, 1e3, 1e4, 1e5)


def proc(*p) -> Path:
    return data_dir() / "processed" / Path(*p)


def wdir(*p) -> Path:
    d = proc("nq4_w")
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def adir_of(files) -> str:
    return list(files)[3].rsplit("/cams/", 1)[0]


# ================================================================ universe and V-JEPA features

def universe(set_name: str) -> pd.DataFrame:
    t = pd.read_parquet(proc(set_name, "index.parquet"))
    if set_name != "carla_p6":
        t = t[t.role == "obs"]
    return t.reset_index(drop=True)


def _stored_vjepa(set_name: str) -> pd.Series:
    """frame_name -> (source file, row) of every stored V-JEPA 2 `mean` row of this set."""
    out = {}
    d = proc(set_name, "bb_vjepa2")
    if (d / "index.parquet").exists():
        for i, f in enumerate(pd.read_parquet(d / "index.parquet").frame_name):
            out[f] = (str(d / "mean.npy"), i)
    for c in sorted(proc(set_name, "w_vjepa2").glob("c*.npz")) if proc(set_name, "w_vjepa2").exists() else []:
        with np.load(c) as z:
            for i, f in enumerate(z["frame_name"]):
                out[str(f)] = (str(c), i)
    return pd.Series(out, dtype=object)


class _Clips:
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        return i, self.tf([Image.open(f).convert("RGB") for f in self.files[i]])


def _extract_rows(t: pd.DataFrame, fx, batch: int, workers: int, rl=None) -> np.ndarray:
    """(len(t), 3072) float16: per row the three cameras' 4-frame clips through V-JEPA 2, `mean` tap concatenated."""
    import torch
    from torch.utils.data import DataLoader
    files = [list(fs)[4 * c: 4 * c + 4] for fs in t.files for c in range(len(CAMS))]
    dl = DataLoader(_Clips(files, fx.transform), batch_size=batch, num_workers=workers, prefetch_factor=4,
                    pin_memory=True, collate_fn=lambda b: torch.stack([x[1] for x in b]))
    out, pending, t0, n = [], None, time.time(), 0
    for v in dl:
        o = fx(v)["mean"].half().to("cpu", non_blocking=True)
        if pending is not None:
            out.append(pending.numpy())
        pending, n = o, n + len(v)
        if rl and (n // batch) % 40 == 0:
            el = time.time() - t0
            rl.info(f"{n}/{len(files)} clips, {n / el:.1f} clips/s, ETA {(len(files) - n) / (n / el) / 60:.1f} min")
    torch.cuda.synchronize()
    out.append(pending.numpy())
    return np.concatenate(out).reshape(len(t), -1)


def vjepa(batch: int = 64, workers: int = 7, chunk: int = 3000, rl=None) -> dict:
    from . import features as F
    fx, info = None, {}
    for s in SETS:
        t = universe(s)
        have = _stored_vjepa(s)
        miss = t[~t.frame_name.isin(have.index)].reset_index(drop=True)
        info[s] = {"universe": len(t), "missing": len(miss)}
        if not len(miss):
            continue
        d = proc(s, "w_vjepa2")
        d.mkdir(exist_ok=True)
        start = len(list(d.glob("c*.npz")))
        for ci, lo in enumerate(range(0, len(miss), chunk)):
            fx = fx or F.VJepaFeatures(frames=4)
            part = miss.iloc[lo: lo + chunk]
            rl and rl.info(f"{s}: chunk {start + ci}, rows {lo}-{lo + len(part)} of {len(miss)}")
            z = _extract_rows(part, fx, batch, workers, rl)
            tmp = d / f"c{start + ci:03d}.tmp.npz"
            np.savez(tmp, frame_name=part.frame_name.to_numpy().astype(str), mean=z)
            tmp.replace(d / f"c{start + ci:03d}.npz")
    return info


def check(n: int = 48, batch: int = 16, workers: int = 4) -> dict:
    """Re-extract n stored BA rows through this path; max |diff| relative to the row's max |value|."""
    from . import features as F
    t = universe("carla_p5v1_ba")
    t = t.iloc[np.linspace(0, len(t) - 1, n).astype(int)].reset_index(drop=True)
    have = _stored_vjepa("carla_p5v1_ba")
    ref = np.load(have[t.frame_name.iloc[0]][0], mmap_mode="r")[[have[f][1] for f in t.frame_name]].astype(np.float32)
    z = _extract_rows(t, F.VJepaFeatures(frames=4), batch, workers).astype(np.float32)
    rel = np.abs(z - ref).max(1) / np.abs(ref).max(1)
    cos = (z * ref).sum(1) / np.linalg.norm(z, axis=1) / np.linalg.norm(ref, axis=1)
    return {"rows": n, "max_rel_diff": float(rel.max()), "median_rel_diff": float(np.median(rel)), "min_cos": float(cos.min())}


# ================================================================ labels and actions (CPU)

def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _run_rows(args):
    """Per indexed frame of one run: speed, the actions into / out of the frame, and the ego-frame labels."""
    adir, frames = args
    d = Path(adir)
    pose = pd.read_json(d / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame")
    a = np.load(d / "actors.npz")
    kinds = json.loads((d / "actor_kinds.json").read_text())
    walker = {int(k): v[0].startswith("walker.") for k, v in kinds.items()}
    road = {int(k): v[0].startswith(("walker.", "vehicle.")) for k, v in kinds.items()}
    order = np.argsort(a["frame"], kind="stable")
    fr, ids, xyz = a["frame"][order], a["id"][order], a["xyz"][order]
    v = np.hypot(pose.vx, pose.vy)
    psi = np.radians(pose.yaw)

    def at(s, f):
        return float(s.get(f, np.nan))
    out = []
    for f in frames:
        rec = {"frame": f, "v": at(v, f),
               "a_next": (at(v, f + TICKS) - at(v, f)) / DT, "w_next": _wrap(at(psi, f + TICKS) - at(psi, f)) / DT,
               "a_prev": (at(v, f) - at(v, f - TICKS)) / DT, "w_prev": _wrap(at(psi, f) - at(psi, f - TICKS)) / DT,
               "ped": False, "occ": False, "d_front": D_MAX}
        if f in pose.index:
            e = pose.loc[f]
            i0, i1 = np.searchsorted(fr, f), np.searchsorted(fr, f, side="right")
            if i1 > i0:
                q = xyz[i0:i1].astype(np.float64)
                dx, dy = q[:, 0] - e.x, q[:, 1] - e.y
                c, s = np.cos(np.radians(e.yaw)), np.sin(np.radians(e.yaw))
                x, y = dx * c + dy * s, -dx * s + dy * c          # CARLA: x forward, y right
                ok = (np.abs(q[:, 2] - e.z) <= VIS_DZ) & (np.hypot(dx, dy) > EGO_R)
                ok &= np.array([road.get(int(k), False) for k in ids[i0:i1]])
                wk = np.array([walker.get(int(k), False) for k in ids[i0:i1]])
                rec["ped"] = bool((ok & wk & (x > 0) & (x <= PED_X) & (np.abs(y) <= PED_Y)).any())
                lane = ok & (x > 0) & (np.abs(y) <= LANE_Y)
                rec["occ"] = bool((lane & (x <= OCC_X)).any())
                if (lane & (x <= D_MAX)).any():
                    rec["d_front"] = float(x[lane & (x <= D_MAX)].min())
        out.append(rec)
    return adir, out


def prep(workers: int = 8) -> dict:
    """meta.parquet (one row per universe frame, sorted by set / run / frame, with window starts) and z.npy (float16)."""
    metas, zs = [], []
    for s in SETS:
        t = universe(s)
        t = t.assign(set=s, adir=t.files.map(adir_of), base_id=t.base_id.astype(str))
        have = _stored_vjepa(s)
        src = have[t.frame_name]
        vj = np.empty((len(t), D_VJ), np.float16)
        for f, g in pd.Series(np.arange(len(t))).groupby(src.map(lambda x: x[0]).to_numpy()):
            arr = np.load(f)["mean"] if f.endswith(".npz") else np.load(f, mmap_mode="r")
            vj[g.to_numpy()] = arr[src.iloc[g.to_numpy()].map(lambda x: x[1]).to_numpy()]
        op = proc(s, "op_cinque_vis")
        pos = pd.Series(np.arange(len(names := pd.read_parquet(op / "index.parquet").frame_name)), index=names)
        opz = np.load(op / "temporal.npy", mmap_mode="r")[pos[t.frame_name].to_numpy()].astype(np.float16)
        n2 = pd.read_parquet(proc(s, "night2_labels.parquet"))[["a", "b", "c"]]
        t = t.join(n2.astype(float), on="frame_name")
        if s == "carla_p6":
            c = pd.read_parquet(proc(s, "nq3_cases.parquet"))
            c = c.assign(base_id=c.base_id.astype(str))[["base_id", "seed", "t_vis", "t_div_lat", "mode_x10", "side", "main",
                                                         "bypass_pair", "reason"]]
            t = t.merge(c, on=["base_id", "seed"], how="left")
            t["family"] = t.scenario
            t["cls"] = np.where(t.main.fillna(False).astype(bool) & (t.reason == "ok"), "obstacle", "p6_other")
        else:
            fam = pd.read_csv(proc(s, "pairs.csv"), usecols=["base_id", "family"]).drop_duplicates("base_id")
            t = t.merge(fam.assign(base_id=fam.base_id.astype(str)), on="base_id", how="left")
            t["cls"] = np.select([t.family.isin(PED_FAM), t.family.isin(CUTIN_FAM)], ["ped", "cutin"], "p5_other")
        jobs = [(a, sorted(g.frame.astype(int))) for a, g in t.groupby("adir")]
        with Pool(workers) as p:
            lab = {a: pd.DataFrame(r) for a, r in p.imap_unordered(_run_rows, jobs, chunksize=4)}
        lab = pd.concat([x.assign(adir=a) for a, x in lab.items()], ignore_index=True)
        t = t.merge(lab, on=["adir", "frame"], how="left")
        keep = ["set", "frame_name", "adir", "frame", "k", "base_id", "world", "seed", "family", "cls", "v", "a_next",
                "w_next", "a_prev", "w_prev", "ped", "occ", "d_front", "a", "b", "c"]
        keep += [c for c in ("t_vis", "t_div_lat", "mode_x10", "side", "bypass_pair") if c in t]
        t["op_row"] = np.arange(len(t))
        metas.append(t[keep + ["op_row"]])
        zs.append(np.concatenate([opz, vj], 1))
        log.info("%s: %d rows, labels ped %.3f occ %.3f, a %.3f", s, len(t), t.ped.mean(), t.occ.mean(), t.a.mean())
    m = pd.concat(metas, ignore_index=True)
    z = np.concatenate(zs)
    o = m.sort_values(["set", "adir", "frame"]).index.to_numpy()
    m, z = m.loc[o].drop(columns="op_row").reset_index(drop=True), z[o]
    for c in ("a_next", "w_next", "a_prev", "w_prev"):
        m[c] = m[c].fillna(0.0).clip(-15, 15)
    m["v"] = m.v.fillna(0.0)
    seg = (m.adir != m.adir.shift()) | (m.frame.diff() != TICKS)
    m["seg"] = seg.cumsum() - 1
    pos_in_seg = m.groupby("seg").cumcount()
    seg_len = m.groupby("seg").seg.transform("size")
    m["win_start"] = (seg_len - pos_in_seg) >= WIN            # a full window starts here
    m.to_parquet(wdir("meta.parquet"), index=False)
    np.save(wdir("z.npy"), z)
    return {"rows": len(m), "windows": int(m.win_start.sum()), "by_set": m.groupby("set").win_start.sum().to_dict(),
            "routes": int(m.base_id.nunique())}


# ================================================================ model

def build_model(cfg=CFG, dz: int = D_OP + D_VJ):
    import torch
    from torch import nn

    class WM(nn.Module):
        def __init__(s):
            super().__init__()
            d = cfg["d"]
            s.zin = nn.Linear(dz, d)
            s.hin = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            s.qin = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
            s.pos = nn.Parameter(torch.randn(WIN, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, cfg["heads"], cfg["ff"], cfg["drop"], activation="gelu",
                                               batch_first=True, norm_first=True)
            s.enc = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
            s.norm = nn.LayerNorm(d)
            s.out = nn.Linear(d, dz)
            nn.init.zeros_(s.out.weight)
            nn.init.zeros_(s.out.bias)

        def forward(s, zh, hact, fact):
            """zh (B, 8, dz) standardised; hact (B, 8, 3) = (a_prev, w_prev, v) scaled; fact (B, 10, 2) scaled.
            Returns (B, 10, dz): z_{t0} + Delta z_h."""
            x = torch.cat([s.zin(zh) + s.hin(hact), s.qin(fact)], 1) + s.pos
            return zh[:, -1:] + s.out(s.norm(s.enc(x)[:, HIST:]))
    return WM()


ACT_SCALE = np.array([3.0, 0.5], np.float32)       # a (m/s^2), omega (rad/s)
V_SCALE = 10.0


class Data:
    """Everything on the GPU: standardised z (bf16), per-row action / speed arrays, window gathers."""

    def __init__(self, meta: pd.DataFrame, z: np.ndarray, train_rows: np.ndarray, dev: str = "cuda"):
        import torch
        zt = torch.as_tensor(z[train_rows], device=dev, dtype=torch.float32)
        self.mu, self.sd = zt.mean(0), zt.std(0).clamp_min(1e-4)
        del zt
        self.Z = torch.empty((len(z), z.shape[1]), device=dev, dtype=torch.bfloat16)
        for lo in range(0, len(z), 8192):
            self.Z[lo: lo + 8192] = ((torch.as_tensor(z[lo: lo + 8192], device=dev).float() - self.mu) / self.sd).bfloat16()
        self.hact = torch.as_tensor(np.c_[meta[["a_prev", "w_prev"]].to_numpy(np.float32) / ACT_SCALE,
                                          meta.v.to_numpy(np.float32)[:, None] / V_SCALE], device=dev)
        self.fact = torch.as_tensor(meta[["a_next", "w_next"]].to_numpy(np.float32) / ACT_SCALE, device=dev)
        self.dev = dev
        self.off = torch.arange(WIN, device=dev)

    def batch(self, starts, fact=None, act_starts=None):
        """starts: (B,) window starts (row of the first history frame). fact overrides the future actions (B, 10, 2)
        in physical units; act_starts takes the history actions and speeds from other windows (the x+ side of a pair)."""
        import torch
        r = starts[:, None] + self.off
        zh, zf = self.Z[r[:, :HIST]].float(), self.Z[r[:, HIST:]].float()
        ha = self.hact[(r if act_starts is None else act_starts[:, None] + self.off)[:, :HIST]]
        fa = self.fact[r[:, HIST - 1: WIN - 1]] if fact is None else fact / torch.as_tensor(ACT_SCALE, device=self.dev)
        return zh, ha, fa, zf


def block_mse(pred, tgt):
    e = (pred - tgt) ** 2
    return 0.5 * (e[..., :D_OP].mean() + e[..., D_OP:].mean())


def train(model, data: Data, tr_starts: np.ndarray, va_starts: np.ndarray, seed: int, rl=None, tag: str = "",
          cfg=CFG, steps: int | None = None) -> dict:
    import torch
    steps = steps or cfg["steps"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    tr = torch.as_tensor(tr_starts, device="cuda")
    va = torch.as_tensor(va_starts, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]) *
                                              0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    best, best_state, hist, t0 = np.inf, None, [], time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = tr[torch.randint(len(tr), (cfg["batch"],), device="cuda", generator=g)]
        zh, ha, fa, zf = data.batch(idx)
        with torch.autocast("cuda", torch.bfloat16):
            pred = model(zh, ha, fa)
        loss = block_mse(pred.float(), zf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == steps:
            vl = evaluate(model, data, va)
            tl = float(loss)
            hist.append({"step": step, "train": tl, "val": vl, "wall_s": time.time() - t0})
            if rl:
                rl.scalar(f"{tag}/train_loss", tl, step)
                rl.scalar(f"{tag}/val_loss", vl, step)
                rl.info(f"{tag} step {step}: train {tl:.4f}, inner-val {vl:.4f}, {step / (time.time() - t0):.1f} steps/s")
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"best_val": best, "curve": hist, "steps_per_s": steps / (time.time() - t0)}


def evaluate(model, data: Data, starts, bs: int = 1024) -> float:
    """Inner-val loss (block MSE over all 10 horizons), and the persistence baseline for reference is z_t0 itself."""
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for lo in range(0, len(starts), bs):
            zh, ha, fa, zf = data.batch(starts[lo: lo + bs])
            tot += float(block_mse(model(zh, ha, fa).float(), zf)) * len(zh)
            n += len(zh)
    return tot / max(n, 1)


def predict(model, data: Data, starts, fact=None, act_starts=None, bs: int = 1024):
    """(n, 10, dz) predicted standardised z (float32, on the GPU)."""
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for lo in range(0, len(starts), bs):
            zh, ha, fa, _ = data.batch(starts[lo: lo + bs], None if fact is None else fact[lo: lo + bs],
                                       None if act_starts is None else act_starts[lo: lo + bs])
            out.append(model(zh, ha, fa).float())
    return torch.cat(out)


# ================================================================ probes on real z

def fit_logreg(X, y, C: float = 1.0):
    """N2's probe objective (sum BCE + ||w||^2 / 2C, L-BFGS) on already-standardised X (GPU float32)."""
    import torch
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, tolerance_grad=1e-6, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, reduction="sum") + (w * w).sum() / (2 * C)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def fit_ridge(X, y, Xv, yv):
    """Ridge on standardised X with lambda from RIDGE_LAMS picked on (Xv, yv)."""
    import torch
    mu = y.mean()
    A = X.double()
    e, V = torch.linalg.eigh(A.T @ A)
    r = V.T @ (A.T @ (y.double() - mu))
    best = None
    for lam in RIDGE_LAMS:
        w = (V @ (r / (e + lam))).float()
        err = float(((Xv @ w + mu - yv) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, lam, w)
    return best[2], mu, best[1]


def probes(data: Data, meta: pd.DataFrame, tr_rows: np.ndarray, va_rows: np.ndarray) -> dict:
    import torch
    X = data.Z[torch.as_tensor(tr_rows, device="cuda")].float()
    out = {}
    for p in PROBES_CLS:
        y = meta[p].to_numpy(np.float32)[tr_rows]
        ok = ~np.isnan(y)
        if ok.sum() < 100 or y[ok].min() == y[ok].max():
            continue
        okt = torch.as_tensor(ok, device="cuda")
        out[p] = ("logit", *fit_logreg(X[okt], torch.as_tensor(y[ok], device="cuda")))
    Xv = data.Z[torch.as_tensor(va_rows, device="cuda")].float()
    w, mu, lam = fit_ridge(X, torch.as_tensor(meta.d_front.to_numpy(np.float32)[tr_rows], device="cuda"), Xv,
                           torch.as_tensor(meta.d_front.to_numpy(np.float32)[va_rows], device="cuda"))
    out["d_front"] = ("ridge", w, mu, lam)
    return out


def apply_probes(P: dict, X) -> dict:
    """X (..., dz) standardised -> {probe: (...) numpy} (logits for the classifiers, metres for d_front)."""
    return {k: (X @ v[1] + v[2]).cpu().numpy() for k, v in P.items()}


# ================================================================ folds and readouts

def folds(meta: pd.DataFrame, seed: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Route folds (every set together) and the inner-val flag: seed-permuted routes, round-robin into k folds; 10 %
    of each fold's training routes are its inner-val routes."""
    routes = np.array(sorted(meta.base_id.unique()))
    rng = np.random.RandomState(seed)
    perm = routes[rng.permutation(len(routes))]
    fold_of = {r: i % k for i, r in enumerate(perm)}
    inner = {r: rng.rand() < 0.10 for r in perm}
    return meta.base_id.map(fold_of).to_numpy(), meta.base_id.map(inner).to_numpy()


def pair_table(meta: pd.DataFrame) -> pd.DataFrame:
    """x+ / x- and x+ / weather-null anchor pairs (rows of the anchor frames, each with a full window) of every class."""
    anchor_ok = pd.Series(False, index=meta.index)
    st = np.flatnonzero(meta.win_start.to_numpy())
    anchor_ok.iloc[st + HIST - 1] = True
    fn_row = pd.Series(meta.index.to_numpy(), index=meta.set + "|" + meta.frame_name)
    rows = []
    for s in SETS[:2]:
        for f, kind, col in (("obs", "pair", "fn_minus"), ("null", "null", "fn_null")):
            o = pd.read_parquet(proc(s, f"{f}.parquet"))
            a = fn_row.reindex(s + "|" + o.fn_plus).to_numpy()
            b = fn_row.reindex(s + "|" + o[col]).to_numpy()
            rows.append(pd.DataFrame({"set": s, "kind": kind, "family": o.family.to_numpy(), "base_id": o.base_id.astype(str).to_numpy(),
                                      "seed": o.seed.to_numpy(), "k": o.k.to_numpy(), "r_plus": a, "r_other": b}))
    p6 = meta[meta.set == "carla_p6"]
    key = p6.reset_index().set_index(["base_id", "seed", "world", "k"])["index"]
    x10 = p6[(p6.world == "x10") & (p6.cls == "obstacle") & (p6.k >= p6.t_vis)]
    for kind, w in (("pair", "x00"), ("null", "wnull")):
        other = key.reindex(pd.MultiIndex.from_arrays([x10.base_id, x10.seed, np.full(len(x10), w), x10.k])).to_numpy()
        rows.append(pd.DataFrame({"set": "carla_p6", "kind": kind, "family": x10.family.to_numpy(), "base_id": x10.base_id.to_numpy(),
                                  "seed": x10.seed.to_numpy(), "k": x10.k.to_numpy(), "r_plus": x10.index.to_numpy(), "r_other": other}))
    p = pd.concat(rows, ignore_index=True).dropna(subset=["r_plus", "r_other"])
    p = p.astype({"r_plus": int, "r_other": int})
    p = p[anchor_ok.to_numpy()[p.r_plus] & anchor_ok.to_numpy()[p.r_other]]
    p["cls"] = np.select([p.family.isin(PED_FAM), p.family.isin(CUTIN_FAM), p.set == "carla_p6"], ["ped", "cutin", "obstacle"],
                         "other")
    return p.reset_index(drop=True)


def _brake(v0: np.ndarray) -> np.ndarray:
    a, v = np.zeros((len(v0), FUT), np.float32), v0.astype(np.float64).copy()
    for j in range(FUT):
        a[:, j] = np.maximum(BRAKE, -v / DT)
        v = np.maximum(0.0, v + a[:, j] * DT)
    return a


def lateral_actions(meta: pd.DataFrame, anchors: np.ndarray, sign: np.ndarray) -> dict:
    fut = anchors[:, None] + np.arange(FUT)
    a_exp = meta.a_next.to_numpy(np.float32)[fut]
    v0 = meta.v.to_numpy(np.float32)[anchors]
    om = np.minimum(SHIFT_M / np.maximum(v0, 0.1), OMEGA_MAX)[:, None]
    prof = np.r_[np.ones(FUT // 2), -np.ones(FUT - FUT // 2)][None] * om * sign[:, None]
    return {"keep_lane": np.stack([a_exp, np.zeros_like(a_exp)], -1), "shift": np.stack([a_exp, prof.astype(np.float32)], -1)}


def bypass_sign(meta: pd.DataFrame) -> dict:
    """Yaw-rate sign of the expert's own bypass (P6 x10 bypass cases, frames from t_div_lat on for 2 s), per side."""
    m = meta[(meta.set == "carla_p6") & (meta.world == "x10") & meta.bypass_pair.fillna(False).astype(bool)]
    m = m[(m.k >= m.t_div_lat - 40) & (m.k < m.t_div_lat + 20)]
    side = np.where(m.mode_x10.str.contains("_L"), "L", "R")
    return m.groupby(side).w_next.agg(["mean", "median", "size"]).to_dict("index")


def run(seed: int, rl, steps: int | None = None, folds_only=None) -> dict:
    import torch
    torch.manual_seed(seed)
    meta = pd.read_parquet(wdir("meta.parquet"))
    z = np.load(wdir("z.npy"), mmap_mode="r")
    fold, inner = folds(meta, seed)
    starts_all = np.flatnonzero(meta.win_start.to_numpy())
    anchors_all = starts_all + HIST - 1
    pairs = pair_table(meta)
    sgn = bypass_sign(meta)
    s_left = float(np.sign(sgn["L"]["mean"]))          # CARLA left-handed: a left bypass is omega < 0
    rl.info(f"seed {seed}: {len(meta)} rows, {len(starts_all)} windows, {len(pairs)} anchor pairs; expert bypass yaw rate {sgn}")
    rl.event("bypass_sign", **{k: v for k, v in sgn.items()})
    anc_fold = fold[anchors_all]
    pair_rows, act_rows, curves = [], [], []
    for f in (folds_only or range(5)):
        t0 = time.time()
        tr_rows = np.flatnonzero((fold != f) & ~inner)
        va_rows = np.flatnonzero((fold != f) & inner)
        data = Data(meta, z, tr_rows)
        st_tr = starts_all[(anc_fold != f) & ~inner[anchors_all]]
        st_va = starts_all[(anc_fold != f) & inner[anchors_all]]
        torch.manual_seed(seed * 100 + f)
        model = build_model().cuda()
        info = train(model, data, st_tr, st_va, seed * 100 + f, rl, f"seed{seed}/fold{f}", steps=steps)
        curves.append({"fold": f, **{k: v for k, v in info.items() if k != "curve"}, "curve": info["curve"]})
        P = probes(data, meta, tr_rows, va_rows)
        rl.info(f"fold {f}: trained ({info['steps_per_s']:.1f} steps/s, best inner-val {info['best_val']:.4f}); "
                f"d_front ridge lambda {P['d_front'][3]:g}")
        # ---- paired separation
        pf = pairs[fold[pairs.r_plus.to_numpy()] == f]
        fut = pf.r_plus.to_numpy()[:, None] + np.arange(FUT)
        fact = torch.as_tensor(np.stack([meta.a_next.to_numpy(np.float32)[fut], meta.w_next.to_numpy(np.float32)[fut]], -1),
                               device="cuda")
        ast = torch.as_tensor(pf.r_plus.to_numpy() - (HIST - 1), device="cuda")
        for side, col in (("plus", "r_plus"), ("other", "r_other")):
            st = pf[col].to_numpy() - (HIST - 1)
            pred = predict(model, data, torch.as_tensor(st, device="cuda"), fact, ast)
            real = data.Z[torch.as_tensor(st, device="cuda")[:, None] + data.off].float()
            for h in (5, 10):
                for src, X in (("pred", pred[:, h - 1]), ("persist", real[:, HIST - 1]), ("oracle", real[:, HIST - 1 + h])):
                    for p, val in apply_probes(P, X).items():
                        pair_rows.append(pd.DataFrame({"pair": pf.index.to_numpy(), "side": side, "h": h, "src": src,
                                                       "probe": p, "score": val}))
        # ---- action sensitivity
        an = anchors_all[anc_fold == f]
        mm = meta.iloc[an]
        hz = an[(mm.world.isin(["plus", "x10"]) & mm.occ.fillna(False).astype(bool) & (mm.v >= V_HAZ)).to_numpy()]
        lat = an[((mm.set == "carla_p6") & (mm.world == "x10") & mm.bypass_pair.fillna(False).astype(bool) &
                  (mm.cls == "obstacle") & (mm.k >= mm.t_vis) & (mm.k < mm.t_div_lat) & (mm.v >= V_HAZ) &
                  mm.occ.fillna(False).astype(bool)).to_numpy()]
        for test, rows_ in (("long", hz), ("lat", lat)):
            if not len(rows_):
                continue
            if test == "long":
                fut = rows_[:, None] + np.arange(FUT)
                w_exp = meta.w_next.to_numpy(np.float32)[fut]
                v0 = meta.v.to_numpy(np.float32)[rows_]
                acts = {"base": np.stack([np.zeros_like(w_exp), w_exp], -1), "alt": np.stack([_brake(v0), w_exp], -1)}
            else:
                ls = np.array([s_left if "_L" in m else -s_left for m in meta.mode_x10.astype(str).to_numpy()[rows_]])
                la = lateral_actions(meta, rows_, ls)
                acts = {"base": la["keep_lane"], "alt": la["shift"]}
            st = torch.as_tensor(rows_ - (HIST - 1), device="cuda")
            outs = {k: predict(model, data, st, torch.as_tensor(a, device="cuda")) for k, a in acts.items()}
            for h in (5, 10):
                sc = {k: apply_probes(P, o[:, h - 1]) for k, o in outs.items()}
                act_rows.append(pd.DataFrame({"row": rows_, "test": test, "h": h, "fold": f,
                                              "d_base": sc["base"]["d_front"], "d_alt": sc["alt"]["d_front"],
                                              "occ_base": sc["base"]["occ"], "occ_alt": sc["alt"]["occ"]}))
        rl.info(f"fold {f}: {len(pf)} pairs, {len(hz)} hazard anchors, {len(lat)} lateral anchors, {time.time() - t0:.0f} s")
        del data, model
        torch.cuda.empty_cache()
    d = rl.dir
    pr = pd.concat(pair_rows, ignore_index=True)
    pr.to_parquet(d / "pair_scores.parquet", index=False)
    pairs.to_parquet(d / "pairs.parquet")
    pd.concat(act_rows, ignore_index=True).assign(seed=seed).to_parquet(d / "action_scores.parquet", index=False)
    (d / "train_curves.json").write_text(json.dumps(curves, default=float))
    return {"seed": seed, "pairs": len(pairs), "dir": str(d)}


# ================================================================ profile

def bench(steps: int = 300, workers: int = 7) -> dict:
    """Steps/s of the same model and batch: (a) windows gathered from the float16 memmap in DataLoader workers, fp32;
    (b) GPU-resident standardised z, index gathers on the card, bf16 autocast, fused AdamW (what `run` uses)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    meta = pd.read_parquet(wdir("meta.parquet"))
    z = np.load(wdir("z.npy"), mmap_mode="r")
    starts = np.flatnonzero(meta.win_start.to_numpy())
    rows = np.arange(len(meta))
    res = {}
    torch.manual_seed(0)

    class W(Dataset):
        def __init__(s):
            s.h = np.c_[meta[["a_prev", "w_prev"]].to_numpy(np.float32) / ACT_SCALE, meta.v.to_numpy(np.float32)[:, None] / V_SCALE]
            s.f = meta[["a_next", "w_next"]].to_numpy(np.float32) / ACT_SCALE

        def __len__(s):
            return len(starts)

        def __getitem__(s, i):
            r = starts[i] + np.arange(WIN)
            x = np.asarray(z[r], np.float32)
            return x[:HIST], s.h[r[:HIST]], s.f[r[HIST - 1: WIN - 1]], x[HIST:]
    mu = torch.as_tensor(np.asarray(z[::7], np.float32).mean(0), device="cuda")
    sd = torch.as_tensor(np.asarray(z[::7], np.float32).std(0), device="cuda").clamp_min(1e-4)
    model = build_model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dl = DataLoader(W(), batch_size=CFG["batch"], shuffle=True, num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    it, t0 = iter(dl), None
    for s in range(steps + 20):
        if s == 20:
            torch.cuda.synchronize()
            t0 = time.time()
        zh, ha, fa, zf = (x.cuda(non_blocking=True) for x in next(it))
        loss = block_mse(model((zh - mu) / sd, ha, fa), (zf - mu) / sd)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    res["a_dataloader_fp32_steps_per_s"] = steps / (time.time() - t0)
    del it, dl
    data = Data(meta, np.asarray(z), rows[::2])
    model = build_model().cuda()
    t0 = time.time()
    train(model, data, starts, starts[:4096], 0, None, "", steps=steps + 20)
    res["b_gpu_bf16_steps_per_s"] = (steps + 20) / (time.time() - t0)
    res["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    return res


# ================================================================ report

def _auc(pos, neg) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def _seed_dirs() -> dict:
    import glob
    out = {}
    for s in (0, 1, 2):
        fs = sorted(glob.glob(str(data_dir() / "runs" / "nq4" / "w" / f"seed{s}" / "*" / "action_scores.parquet")))
        if fs:
            out[s] = Path(fs[-1]).parent
    return out


def report(n_boot: int = 1000) -> dict:
    meta = pd.read_parquet(wdir("meta.parquet"))
    dirs = _seed_dirs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, wide = [], {}
    for s, d in dirs.items():
        pairs = pd.read_parquet(d / "pairs.parquet")
        sc = pd.read_parquet(d / "pair_scores.parquet")
        w = sc.pivot_table(index=["pair", "h", "src", "probe"], columns="side", values="score").reset_index()
        w = w.join(pairs[["cls", "kind", "set", "family", "base_id"]], on="pair")
        wide[s] = w
        for (cls, kind, h, src, probe), g in w.groupby(["cls", "kind", "h", "src", "probe"]):
            a = _auc(g.plus.to_numpy(), g.other.to_numpy())
            rows.append({"seed": s, "cls": cls, "kind": kind, "h": h, "src": src, "probe": probe, "n_pairs": len(g),
                         "n_routes": g.base_id.nunique(), "auc": a, "auc_sym": max(a, 1 - a),
                         "paired_win": float((g.plus > g.other).mean())})
    long = pd.DataFrame(rows)
    long.to_csv(RESULTS / "pair_auc_seeds.csv", index=False, float_format="%.4f")
    # 3-seed means, primary probe per class, pair vs weather null
    m = long.groupby(["cls", "kind", "h", "src", "probe"]).agg(auc=("auc", "mean"), auc_sym=("auc_sym", "mean"),
                                                               auc_min=("auc", "min"), auc_max=("auc", "max"),
                                                               paired_win=("paired_win", "mean"), n_pairs=("n_pairs", "first"),
                                                               n_routes=("n_routes", "first"), seeds=("seed", "nunique")).reset_index()
    m.to_csv(RESULTS / "pair_auc.csv", index=False, float_format="%.4f")
    prim = []
    rng = np.random.RandomState(0)
    for cls, probe in CLS_PROBE.items():
        for h in (5, 10):
            r = {"cls": cls, "probe": probe, "horizon_s": h * DT}
            for src in ("pred", "persist", "oracle"):
                q = m[(m.cls == cls) & (m.h == h) & (m.src == src) & (m.probe == probe)].set_index("kind")
                if "pair" not in q.index:
                    continue
                r[f"{src}_auc"] = q.auc["pair"]
                r[f"{src}_null_auc_sym"] = q.auc_sym.get("null", np.nan)
                if src == "pred":
                    r.update(pred_auc_min=q.auc_min["pair"], pred_auc_max=q.auc_max["pair"], n_pairs=q.n_pairs["pair"],
                             n_routes=q.n_routes["pair"], n_null=q.n_pairs.get("null", 0), paired_win=q.paired_win["pair"])
                    # route-cluster bootstrap of the seed-averaged scores
                    g = pd.concat([wide[s][(wide[s].cls == cls) & (wide[s].h == h) & (wide[s].src == src) &
                                           (wide[s].probe == probe) & (wide[s].kind == "pair")] for s in wide])
                    g = g.groupby("pair").agg(plus=("plus", "mean"), other=("other", "mean"), base_id=("base_id", "first"))
                    rts = g.base_id.unique()
                    by = {b: x for b, x in g.groupby("base_id")}
                    bs = []
                    for _ in range(n_boot):
                        x = pd.concat([by[b] for b in rng.choice(rts, len(rts))])
                        bs.append(_auc(x.plus.to_numpy(), x.other.to_numpy()))
                    r["pred_auc_ci"] = f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"
            prim.append(r)
    prim = pd.DataFrame(prim)
    ok = (prim.pred_auc >= 0.70) & (prim.pred_auc >= prim.pred_null_auc_sym + 0.10)
    prim["pass"] = ok
    cls_pass = prim.groupby("cls")["pass"].all()
    prim.to_csv(RESULTS / "criterion1_pairs.csv", index=False, float_format="%.4f")
    # action sensitivity
    acts = []
    for s, d in dirs.items():
        a = pd.read_parquet(d / "action_scores.parquet")
        a = a.join(meta[["set", "family", "cls", "base_id"]], on="row")
        for (test, h), g in a.groupby(["test", "h"]):
            acts.append({"seed": s, "test": test, "h": h, "n": len(g), "n_routes": g.base_id.nunique(),
                         "correct_d_front": float((g.d_alt > g.d_base).mean()),
                         "median_delta_d_m": float(np.median(g.d_alt - g.d_base)),
                         "occ_lower": float((g.occ_alt < g.occ_base).mean())})
    acts = pd.DataFrame(acts)
    acts.to_csv(RESULTS / "criterion2_actions_seeds.csv", index=False, float_format="%.4f")
    am = acts.groupby(["test", "h"]).agg(correct=("correct_d_front", "mean"), correct_min=("correct_d_front", "min"),
                                         correct_max=("correct_d_front", "max"), median_delta_d_m=("median_delta_d_m", "mean"),
                                         occ_lower=("occ_lower", "mean"), n=("n", "first"), n_routes=("n_routes", "first")).reset_index()
    am.to_csv(RESULTS / "criterion2_actions.csv", index=False, float_format="%.4f")
    c2 = {t: bool(am[(am.test == t) & (am.h == 10)].correct.iloc[0] >= 0.70) for t in am.test.unique()}
    verdict = {"criterion1_by_class": cls_pass.to_dict(), "criterion1": bool(cls_pass.all()),
               "criterion2_by_test": c2, "criterion2": bool(len(c2) == 2 and all(c2.values())), "seeds": list(dirs)}
    (RESULTS / "verdict.json").write_text(json.dumps(verdict, indent=1))
    return {"criterion1": prim, "criterion2": am, "verdict": verdict}


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("vjepa", "check", "prep", "bench", "run", "report"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--folds", default="", help="run: comma list of folds (smoke)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    tag = {"run": f"seed{a.seed}" + ("-smoke" if a.folds else "")}.get(a.step, a.step)
    rl = RunLog("nq4", "w", tag)
    rl.event("start", args=vars(a), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    rl.info(f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}, args {vars(a)}")
    if a.step == "vjepa":
        r = vjepa(a.batch, a.workers, rl=rl)
    elif a.step == "check":
        r = check()
    elif a.step == "prep":
        r = prep(a.workers)
    elif a.step == "bench":
        r = bench(workers=a.workers)
    elif a.step == "run":
        r = run(a.seed, rl, a.steps or None, [int(x) for x in a.folds.split(",")] if a.folds else None)
    else:
        o = report()
        rl.info("criterion 1\n" + o["criterion1"].to_markdown(index=False, floatfmt=".3f"))
        rl.info("criterion 2\n" + o["criterion2"].to_markdown(index=False, floatfmt=".3f"))
        r = o["verdict"]
    rl.info(json.dumps(r, default=float))
    rl.event("end", r=r)
    rl.close()


if __name__ == "__main__":
    main()
