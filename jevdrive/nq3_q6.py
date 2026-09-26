"""Night queue 3, Q6 (todos/2026-09-26-night-queue-3.md, Q6 and the [D] 16:50 entry, written before any Q6 number):
the main-table protocol and the two follow-ups of decision 48.

(b) V-JEPA 2 single-frame control on the P5 v1 BA set
  sf-extract   N6's V-JEPA 2 recipe (features.VJepaFeatures(frames=4), 256^2, bf16, front | front_left | front_right)
               with the clip = the current frame repeated 4 times -> processed/carla_p5v1_ba/bb_vjepa2_1f/
  sf-check     the same driver on real 4-frame clips of 256 units against N6's stored features (bf16 batch noise)
  sf-fit       n6_backbones.fit unchanged on the new backbone (route-fold seed s, GPU eigh); --backbone vjepa2 with
               seed 0 is the refit check against N6's stored run
  sf-report    per seed: single-frame vs 4-frame pedestrian flips, the registered verdict
(c) the V-JEPA 2 stream on real data (M-C dual stream with Qwen -> V-JEPA 2 `mean`, E1 / G0 readouts unchanged)
  rt-nav-extract  V-JEPA 2 on navtest (CAM_F0 / L0 / R0, the NAVSIM 2 Hz 4-frame history) -> processed/navsim_vjepa2/navtest
  rt-heads     the 5 fold heads per model x seed (E1.fold_heads with V-JEPA 2 in place of Qwen), checked against N6
  rt-wod       E1.readouts on the 19 663 WOD frames (CARLA statistics)
  rt-nav       prior + Delta predictions on navtest for the devkit and the activation table
  rt-verdict   paired navtest PDMS table (after the devkit), the registered cells
(a) table      the main table from stored results with the unified protocol -> research/results/nq3/q6/

Run on the box: python -m jevdrive.nq3_q6 <step> [--seed s] (scripts/nq3_d/q6.sh runs them in order).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
SET = "carla_p5v1_ba"
SF = "vjepa2_1f"
MODELS = ("cinque", "lebowski")
SEEDS = (0, 1, 2)
CAMS = ("front", "front_left", "front_right")
NAV_CAMS = ("CAM_F0", "CAM_L0", "CAM_R0")
RESULTS = Path(__file__).resolve().parents[1] / "research/results/nq3/q6"
ACT_HARM = 0.07


def out(*p) -> Path:
    d = data_dir() / "runs" / "nq3" / "q6"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def n6_run(seed: int) -> Path:
    """N6's stored GPU-eigh fit of a seed (the one with results)."""
    fs = sorted((data_dir() / "runs/night2/n6" / f"fit-seed{seed}-cuda").glob("*/criteria.csv"))
    assert fs, f"no N6 fit for seed {seed}"
    return fs[-1].parent


def _gpu_eigh(dev: str = "cuda"):
    """n6_backbones.fit's switch to float64 eigh on the card (pair-Delta and ridge grams)."""
    import torch
    from . import planner, reactivity_mc as M
    M.EIGH_DEVICE = dev

    def _gram_eigh(A):
        e, V = torch.linalg.eigh((A.T @ A).double().to(dev))
        return e.float().to(A.device), V.float().to(A.device)
    planner.gram_eigh = _gram_eigh


# ================================================================ V-JEPA 2 extraction (P5 single frame, navtest)

class _Clips:
    """(i, clip (4, 3, 256, 256)) from lists of 4 JPEG paths; `single` decodes only the last and repeats it."""

    def __init__(self, files, fx, single: bool):
        self.files, self.fx, self.single = files, fx, single

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        if self.single:
            x = self.fx.tf(Image.open(self.files[i][-1]).convert("RGB"))
            return i, torch.stack([x] * self.fx.frames)
        return i, self.fx.transform([Image.open(f).convert("RGB") for f in self.files[i]])


def _run_vjepa(files, single: bool, dst: Path, batch: int, workers: int, rl=None) -> dict:
    """float16 `mean` / `last_mean` per clip, in `files` order, written once at the end (atomic)."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from . import features as F
    if (dst / "done.json").exists():
        return json.loads((dst / "done.json").read_text())
    dst.mkdir(parents=True, exist_ok=True)
    fx = F.VJepaFeatures(frames=4)
    dl = DataLoader(_Clips(files, fx, single), batch_size=batch, num_workers=workers, prefetch_factor=4,
                    pin_memory=True, collate_fn=lambda b: ([x[0] for x in b], torch.stack([x[1] for x in b])))
    res = {"mean": np.empty((len(files), 1024), np.float16), "last_mean": np.empty((len(files), 1024), np.float16)}
    t0, n = time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    for idx, x in tqdm(dl, desc=dst.name, mininterval=30):
        o = fx(x)
        for k in res:
            res[k][idx] = o[k].half().cpu().numpy()
        n += len(idx)
        if rl and (n // batch) % 100 == 0:
            el = time.time() - t0
            rl.event("progress", clips=n, of=len(files), clips_per_s=n / el)
            rl.info(f"{dst.name}: {n}/{len(files)} clips, {n / el:.0f} clips/s, ETA {(len(files) - n) / (n / el) / 60:.1f} min")
    for k, v in res.items():
        np.save(dst / f"{k}.tmp.npy", v)
        os.replace(dst / f"{k}.tmp.npy", dst / f"{k}.npy")
    info = {"clips": len(files), "wall_s": time.time() - t0, "clips_per_s": len(files) / (time.time() - t0),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, "single": single, "batch": batch, "workers": workers}
    (dst / "done.json").write_text(json.dumps(info))
    return info


def _p5_units() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    os.environ.setdefault("P5_SET", SET)
    from . import n6_backbones as N6
    d = N6._root("bb")
    return pd.read_parquet(d / "units.parquet"), np.load(d / "row_units.npy"), pd.read_parquet(d / "rows.parquet")


def sf_extract(rl, batch: int, workers: int, limit: int = 0) -> dict:
    """Single-frame clips for every P5 unit -> per-row features (N6's finalize layout) in bb_vjepa2_1f/."""
    from . import n6_backbones as N6
    u, ru, rows = _p5_units()
    files = u.files.tolist()[: limit or None]
    tmp = out("sf_units" + ("_limit" if limit else ""))
    info = _run_vjepa(files, True, tmp, batch, workers, rl)
    if limit:
        return info
    o = N6._root(f"bb_{SF}")
    rows.to_parquet(o / "index.parquet", index=False)
    for tap in ("mean", "last_mean"):
        a = np.load(tmp / f"{tap}.npy")
        np.save(o / f"{tap}.npy", a[ru].reshape(len(ru), -1))
    return {**info, "rows": len(ru)}


def sf_check(rl, n: int = 256, batch: int = 64, workers: int = 8) -> dict:
    """This driver on real 4-frame clips of n units against N6's stored per-row features (same recipe, other batches);
    also the single-frame path on the same units (descriptive: how far one repeated frame moves the feature)."""
    from . import n6_backbones as N6
    u, ru, rows = _p5_units()
    sel = np.linspace(0, len(u) - 1, n).astype(int)
    files = [u.files.iloc[i] for i in sel]
    four = _run_vjepa(files, False, out("sf_check_4f"), batch, workers)
    one = _run_vjepa(files, True, out("sf_check_1f"), batch, workers)
    # the stored feature of unit k: any (row, cam) that maps to it
    where = {}
    for r, c in zip(*np.nonzero(np.isin(ru, sel))):
        where.setdefault(int(ru[r, c]), (r, c))
    res = {}
    for tap in ("mean", "last_mean"):
        st = np.load(N6._root("bb_vjepa2") / f"{tap}.npy", mmap_mode="r")
        ref = np.stack([np.asarray(st[where[k][0], 1024 * where[k][1]: 1024 * (where[k][1] + 1)], np.float32) for k in sel])
        a4 = np.load(out("sf_check_4f") / f"{tap}.npy").astype(np.float32)
        a1 = np.load(out("sf_check_1f") / f"{tap}.npy").astype(np.float32)
        rel = np.abs(a4 - ref).max(1) / np.abs(ref).max(1)
        cos = lambda a, b: float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).mean())  # noqa: E731
        res[tap] = {"four_vs_stored_max_rel": float(rel.max()), "four_vs_stored_median_rel": float(np.median(rel)),
                    "four_vs_stored_cos": cos(a4, ref), "single_vs_four_cos": cos(a1, a4),
                    "single_vs_four_median_rel": float(np.median(np.abs(a1 - a4).max(1) / np.abs(a4).max(1)))}
    res["throughput_4f_clips_per_s"], res["throughput_1f_clips_per_s"] = four["clips_per_s"], one["clips_per_s"]
    rl.info(json.dumps(res, indent=1))
    (out("sf_check.json")).write_text(json.dumps(res, indent=1))
    assert res["mean"]["four_vs_stored_max_rel"] <= 0.05, "extractor does not reproduce N6's stored features"
    return res


def _register_sf():
    from . import n6_backbones as N6
    N6.TAPS[SF] = ("mean", "last_mean")
    N6.PRIMARY[SF] = "mean"


def sf_fit(rl, seed: int, backbone: str) -> dict:
    from . import n6_backbones as N6
    _register_sf()
    return N6.fit(seed, [backbone], rl, MODELS, "cuda")


def sf_refit_check(fit_dir: Path) -> dict:
    """The driver's 4-frame seed-0 refit against N6's stored seed-0 run: same pedestrian flips (+-1 frame of 406),
    predictions within 1e-3 m."""
    ref = n6_run(0)
    a, b = pd.read_csv(fit_dir / "criteria.csv"), pd.read_csv(ref / "criteria.csv")
    za, zb = np.load(fit_dir / "preds_obs.npz"), np.load(ref / "preds_obs.npz")
    rows = []
    for arm in a.arm:
        if arm not in set(b.arm) or "vjepa2" not in arm and not arm.startswith("prior"):
            continue
        ra, rb = a.set_index("arm").loc[arm], b.set_index("arm").loc[arm]
        if isinstance(ra, pd.DataFrame):
            ra, rb = ra.iloc[0], rb.iloc[0]
        key = arm if arm in za.files else None
        rows.append({"arm": arm, "ped_flip_refit": ra.ped_flip, "ped_flip_n6": rb.ped_flip,
                     "frames_diff": abs(ra.ped_flip - rb.ped_flip) * ra.ped_reactive,
                     "max_abs_pred_m": float(np.nanmax(np.abs(za[key] - zb[key]))) if key and key in zb.files else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out("sf_refit_check.csv"), index=False)
    ok = (df.frames_diff <= 1.01).all() and (df.max_abs_pred_m.fillna(0) <= 1e-3).all()
    return {"ok": bool(ok), "table": df.to_dict("records")}


def sf_report(rl) -> pd.DataFrame:
    """Per seed and prior: single-frame vs N6's 4-frame V-JEPA 2 pedestrian flips, CI overlap, the registered verdict."""
    rows = []
    for s in SEEDS:
        fs = sorted(out("fits").glob(f"sf-seed{s}/*/criteria.csv"))
        assert fs, f"single-frame fit of seed {s} missing"
        a = pd.read_csv(fs[-1]).set_index(["prior", "arm"])
        b = pd.read_csv(n6_run(s) / "criteria.csv").set_index(["prior", "arm"])
        for m in MODELS:
            for arm4, arm1 in ((f"pair-Δ vjepa2 mean [{m}]", f"pair-Δ {SF} mean [{m}]"),
                               (f"pair-Δ dual vjepa2 mean [{m}]", f"pair-Δ dual {SF} mean [{m}]"),
                               ("ridge_late vjepa2 mean", f"ridge_late {SF} mean"),
                               (f"pair-Δ vjepa2 last_mean [{m}]", f"pair-Δ {SF} last_mean [{m}]")):
                if (m, arm1) not in a.index or (m, arm4) not in b.index:
                    continue
                x, y = a.loc[(m, arm1)], b.loc[(m, arm4)]
                rows.append({"seed": s, "prior": m, "arm_1f": arm1, "arm_4f": arm4, "ped_1f": x.ped_flip, "lo_1f": x.ped_lo,
                             "hi_1f": x.ped_hi, "ped_4f": y.ped_flip, "lo_4f": y.ped_lo, "hi_4f": y.ped_hi,
                             "null_1f": x.null_ff_oos, "cutin_1f": x.cutin_flip, "cutin_4f": y.cutin_flip,
                             "ci_overlap": bool(x.ped_lo <= y.ped_hi and y.ped_lo <= x.ped_hi),
                             "e_layer_1f": bool(x.ped_lo > x.null_ff_oos + 0.10)})
    df = pd.DataFrame(rows)
    main = df[(df.prior == "cinque") & (df.arm_1f == f"pair-Δ {SF} mean [cinque]")]
    below10 = main.ped_1f.mean() < 0.10
    overlap = main.ci_overlap.all()
    if below10:
        v = "time, not video pre-training (single-frame ped flip < 10%)"
    elif overlap:
        v = "video pre-training itself (single-frame CI overlaps the 4-frame one in every seed)"
    elif not main.ci_overlap.any():
        v = "partly time (single-frame >= 10% but below the 4-frame CI in every seed)"
    else:
        v = "varies with seed"
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "vjepa_single_frame.csv", index=False, float_format="%.4f")
    (RESULTS / "vjepa_single_frame_verdict.json").write_text(json.dumps(
        {"verdict": v, "ped_1f_mean": float(main.ped_1f.mean()), "ped_4f_mean": float(main.ped_4f.mean()),
         "ci_overlap_seeds": int(main.ci_overlap.sum())}, indent=1))
    rl.info(f"single-frame verdict: {v}\n" + df.to_markdown(index=False, floatfmt=".3f"))
    return df


# ================================================================ (c) V-JEPA 2 stream on real data

def rt_nav_extract(rl, batch: int, workers: int) -> dict:
    """V-JEPA 2 (N6 recipe) on navtest: per token and camera the 4 history frames (-1.5 ... 0 s) -> front | L | R."""
    import pickle
    with open(data_dir() / "runs/navsim_zs/index/navtest.pkl", "rb") as f:
        idx = pickle.load(f)
    tok = np.array([e["token"] for e in idx])
    files = [[fr[c]["path"] for fr in e["cams"]] for e in idx for c in NAV_CAMS]
    assert all(len(x) == 4 for x in files)
    dst = data_dir() / "processed/navsim_vjepa2/navtest"
    info = _run_vjepa(files, False, dst / "units", batch, workers, rl)
    for tap in ("mean", "last_mean"):
        if not (dst / f"{tap}.npy").exists():
            a = np.load(dst / "units" / f"{tap}.npy")
            np.save(dst / f"{tap}.npy", a.reshape(len(tok), -1))
    np.save(dst / "tokens.npy", tok)
    return info


def _load_p5(model: str, dev: str):
    import torch
    from . import n6_backbones as N6, p5_exam as E, p5_openpilot
    os.environ["P5_SET"] = SET
    t, past, fut, obs, null, pairs = E.load()
    V = torch.as_tensor(N6.load_backbones(t, ["vjepa2"])["vjepa2 mean"], device=dev)
    Xop = torch.as_tensor(p5_openpilot.load(t, (model,), sub="op_streams_vis")[f"op-{model} temporal"], device=dev)
    return t, past, fut, obs, null, pairs, V, Xop


def fold_heads_vjepa(model: str, seed: int, rl) -> list[dict]:
    """E1.fold_heads with the V-JEPA 2 `mean` stream in place of Qwen, route folds of seed s, GPU eigh (N6's fit);
    each fold's dual-stream prediction must equal N6's stored `pair-Δ dual vjepa2 mean` within 1e-3 m, same lambda."""
    import torch
    from . import elicit_e1 as E1, p5_exam as E, reactivity_mc as MC
    _gpu_eigh()
    dev = "cuda"
    t, past, fut, obs, null, pairs, V, Xop = _load_p5(model, dev)
    n = len(t)
    fold = E.folds(t, pairs, seed)
    F = torch.as_tensor(fut.reshape(n, -1), device=dev)
    Ego = torch.as_tensor(E.ego_input(t, past), device=dev)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    run = n6_run(seed)
    ref = np.load(run / "preds_obs.npz")
    at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
    ref_pred = ref[f"pair-Δ dual vjepa2 mean [{model}]"]
    tag = f"{model}/vjepa2 mean"
    ref_lam = {e["fold"]: e["lam"] for e in map(json.loads, open(run / "events.jsonl"))
               if e.get("kind") == "mc_fold" and e.get("arm") == "pair" and e.get("model") == tag}
    heads = []
    for f in range(E.K_FOLDS):
        ev = obs_rows[fold[obs_rows] == f]
        keep = {}
        o = MC.fit_fold(f, fold, t, F, Ego, Xop, V, pr_ip, pr_im, pr_group, rl, tag, None, keep, arms=("pair",))
        diff = float(np.abs(o["M-C pair"][ev].reshape(-1, 20, 2).cpu().numpy() - ref_pred[at[ev].to_numpy()]).max())
        rl.event("q6_head_check", model=model, seed=seed, fold=f, lam=keep["lam"], lam_ref=ref_lam.get(f), max_abs_diff=diff)
        rl.info(f"{model} s{seed} fold {f}: lam {keep['lam']:g} (N6 {ref_lam.get(f)}), max |pred - N6| {diff:.2e} m")
        assert np.isclose(keep["lam"], ref_lam[f]) and diff < 1e-3, "fold head does not reproduce N6's stored run"
        tr = keep["tr"]
        heads.append({"W": keep["W"].double().cpu(), "zbar": keep["zbar"].double().cpu(), "lam": keep["lam"],
                      "q": tuple(x.cpu() for x in E1._stats(V, tr)), "op": tuple(x.cpu() for x in E1._stats(Xop, tr)),
                      "diff": diff})
    del V, Xop, F, Ego
    torch.cuda.empty_cache()
    return heads


def rt_heads(rl) -> dict:
    import torch
    for m in MODELS:
        for s in SEEDS:
            p = out("heads", f"vjmc_{m}_s{s}.pt")
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(fold_heads_vjepa(m, s, rl), p)
    return {"heads": len(list(out("heads").glob("*.pt")))}


def _heads(m, s):
    import torch
    return torch.load(out("heads", f"vjmc_{m}_s{s}.pt"), weights_only=False)


def _tau(m: str, s: int) -> tuple[float, float]:
    """The arm's own P5 null threshold and out-of-sample null false flip (N6's stored run of that seed)."""
    run = n6_run(s)
    fl, cr = pd.read_csv(run / "flip_rates.csv"), pd.read_csv(run / "criteria.csv")
    k = f"pair-Δ dual vjepa2 mean [{m}]"
    return (float(fl[(fl.examinee == k) & (fl.scope == "pooled")].tau_model.iloc[0]),
            float(cr[(cr.arm == k) & (cr.prior == m)].null_ff_oos.iloc[0]))


def _wod_vjepa(names: np.ndarray) -> np.ndarray:
    root = data_dir() / "processed/waymo_e2e/features"
    parts = []
    for st in ("vjepa2_p3", "vjepa2_p3_fl", "vjepa2_p3_fr"):
        fn = pd.read_parquet(root / st / "index.parquet").frame_name
        at = pd.Series(np.arange(len(fn)), index=fn).reindex(names)
        assert at.notna().all(), f"{st}: {int(at.isna().sum())} frames missing"
        parts.append(np.load(root / st / "mean.npy", mmap_mode="r")[at.astype(int).to_numpy()].astype(np.float32))
    return np.concatenate(parts, 1)


def rt_wod(rl) -> dict:
    from . import elicit_e1 as E1
    d = E1.wod_frames()
    Vw = _wod_vjepa(d["frame_name"])
    tabs, acts = [], []
    for m in MODELS:
        for s in SEEDS:
            tau, null_ff = _tau(m, s)
            delta = E1.correction(_heads(m, s), Vw, d[f"op {m}"])
            tab, act = E1.readouts(d, d[f"prior {m}"], delta, tau)
            tabs.append(tab.assign(model=m, seed=s))
            acts.append(act.assign(model=m, seed=s, tau=tau, null_ff_oos=null_ff))
            # RFS cluster mean of prior and prior + Delta (the table's absolute protocol), descriptive
            tabs.append(_rfs_cluster(d, d[f"prior {m}"], d[f"prior {m}"] + delta).assign(model=m, seed=s))
            np.savez_compressed(out(f"wod_delta_{m}_s{s}.npz"), frame_name=d["frame_name"], delta=delta)
            rl.info(f"{m} s{s}\n" + tab[tab.judge == "RFS (rater)"].to_markdown(index=False, floatfmt=".3f"))
    pd.concat(tabs).to_csv(out("wod_deltas.csv"), index=False)
    pd.concat(acts).to_csv(out("wod_activation.csv"), index=False)
    return {"rows": len(tabs)}


def _rfs_cluster(d: dict, prior: np.ndarray, arm: np.ndarray, b: int = 10000) -> pd.DataFrame:
    """RFS cluster mean (mean over the rater clusters of the per-cluster frame mean) of prior and arm, and the paired
    difference with a cluster-stratified frame bootstrap."""
    from . import waymo
    pos = d["rater_pos"]
    rfs = lambda p: waymo.rater_feedback_score(p[pos], d["rater_traj"], d["rater_score"], d["speed"][pos])  # noqa: E731
    ra, rp = rfs(arm), rfs(prior)
    cl = d["cluster"][pos]
    groups = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    cm = lambda v: float(np.mean([v[g].mean() for g in groups]))  # noqa: E731
    rng = np.random.default_rng(0)
    diff = ra - rp
    bs = np.array([np.mean([diff[g][rng.integers(0, len(g), len(g))].mean() for g in groups]) for _ in range(b)])
    return pd.DataFrame([{"scope": "all", "judge": "RFS cluster mean (rater)", "n": len(pos), "prior_abs": cm(rp),
                          "arm_abs": cm(ra), "delta": cm(diff), "lo": float(np.percentile(bs, 2.5)),
                          "hi": float(np.percentile(bs, 97.5))}])


def rt_nav(rl) -> dict:
    """prior + Delta on navtest for the devkit (E1's mapping onto 0.5 ... 4.0 s, heading kept) + activation table."""
    from . import elicit_e1 as E1, p5_pairs as P
    dst = data_dir() / "processed/navsim_vjepa2/navtest"
    vtok = np.load(dst / "tokens.npy")
    Vn = np.load(dst / "mean.npy").astype(np.float32)
    sc, acts, jobs = None, [], []
    for m in MODELS:
        z = np.load(data_dir() / "runs/navsim_zs/openpilot/navtest" / f"{m}_temporal.npz")
        tok = z["tokens"]
        at = pd.Series(np.arange(len(vtok)), index=vtok).reindex(tok)
        assert at.notna().all()
        V = Vn[at.astype(int).to_numpy()]
        p = np.load(data_dir() / E1.NAV_HEADS / f"navtest_ridge_late_{m}_temporal.npz")
        assert (p["tokens"] == tok).all()
        if sc is None:
            sc = E1.nav_scopes(tok)
            sc.to_csv(out("navtest_scopes.csv"), index=False)
        for s in SEEDS:
            tau, _ = _tau(m, s)
            delta = E1.correction(_heads(m, s), V, z["temporal"].astype(np.float32))
            arm = p["poses"].copy()
            arm[..., :2] += delta[:, 1:16:2]
            f = out("nav", f"navtest_q6vj_{m}_s{s}.npz")
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, tokens=tok, poses=arm.astype(np.float32))
            jobs.append(f"v1 navtest q6vj_{m}_s{s} {f}")
            act = (np.abs(P.v2(E1._grid20(arm)) - P.v2(E1._grid20(p["poses"]))) >= tau).astype(float)
            mag = np.linalg.norm(delta[:, 1:16:2], axis=-1).mean(-1)
            for name, msk in (("all", np.ones(len(tok), bool)), ("straight", sc.straight.to_numpy()),
                              ("ped_cyc_corridor", sc.ped_cyc_corridor.to_numpy())):
                acts.append({"model": m, "seed": s, "scope": name, "n": int(msk.sum()), "tau": tau,
                             "activation": float(act[msk].mean()), "delta_mag_median_m": float(np.median(mag[msk]))})
    pd.DataFrame(acts).to_csv(out("navsim_activation.csv"), index=False)
    out("nav", "jobs.txt").write_text("\n".join(jobs) + "\n")
    return {"jobs": len(jobs)}


def rt_verdict(rl) -> pd.DataFrame:
    """Paired navtest PDMS (token bootstrap, E1 / G0's) and the registered cells per model x seed."""
    from .elicit_e1 import _devkit
    sc = pd.read_csv(out("navtest_scopes.csv")).set_index("token")
    f = lambda df: df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)].set_index("token")["score"].astype(float)  # noqa: E731
    rng = np.random.default_rng(0)
    nav = []
    for m in MODELS:
        b = _devkit("v1", "navtest", f"heads_ridge_late_{m}_temporal")
        for s in SEEDS:
            a = _devkit("v1", "navtest", f"q6vj_{m}_s{s}")
            assert a is not None and b is not None, f"devkit scores missing for {m} s{s}"
            x, y = f(a).align(f(b), join="inner")
            for g, msk in (("all", None), ("ped_cyc_corridor", sc.ped_cyc_corridor), ("straight", sc.straight)):
                keep = np.ones(len(x), bool) if msk is None else msk.reindex(x.index).fillna(False).to_numpy(bool)
                dd = (x - y).to_numpy()[keep]
                bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
                nav.append({"model": m, "seed": s, "group": g, "n": len(dd), "prior": 100 * y.to_numpy()[keep].mean(),
                            "arm": 100 * x.to_numpy()[keep].mean(), "delta": 100 * dd.mean(),
                            "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)})
    nav = pd.DataFrame(nav)
    nav.to_csv(out("navsim_paired.csv"), index=False)
    wt, wa, na = pd.read_csv(out("wod_deltas.csv")), pd.read_csv(out("wod_activation.csv")), pd.read_csv(out("navsim_activation.csv"))
    rows = []
    for m in MODELS:
        for s in SEEDS:
            w = wt[(wt.model == m) & (wt.seed == s) & (wt.judge == "RFS (rater)")].set_index("scope")
            wc = wt[(wt.model == m) & (wt.seed == s) & (wt.judge == "RFS cluster mean (rater)")].iloc[0]
            n = nav[(nav.model == m) & (nav.seed == s)].set_index("group")
            a_w = float(wa[(wa.model == m) & (wa.seed == s)].set_index("scope").loc["straight_yaw", "activation"])
            a_n = float(na[(na.model == m) & (na.seed == s)].set_index("scope").loc["straight", "activation"])
            covers = lambda r: r.hi >= 0  # noqa: E731   CI covers 0 or lies above it
            cand = bool(covers(w.loc["all"]) and covers(n.loc["all"]))
            harm = w.loc["all", "hi"] < 0 or n.loc["all", "hi"] < 0 or a_w > ACT_HARM or a_n > ACT_HARM
            useful = (not harm) and (w.loc["Pedestrians", "lo"] > 0 or n.loc["ped_cyc_corridor", "lo"] > 0)
            cross = ((w.lo <= 0) & (w.hi >= 0)).all() and ((n.lo <= 0) & (n.hi >= 0)).all()
            g0 = "harmful" if harm else "useful" if useful else "harmless, not useful" if cross else "none of the three cells"
            rows.append({"model": m, "seed": s, "candidate": cand, "g0_cell": g0,
                         "wod_rfs_all": w.loc["all", "delta"], "wod_lo": w.loc["all", "lo"], "wod_hi": w.loc["all", "hi"],
                         "wod_rfs_ped": w.loc["Pedestrians", "delta"], "wod_rfs_cluster_delta": wc.delta,
                         "wod_rfs_cluster_lo": wc.lo, "wod_rfs_cluster_hi": wc.hi, "wod_act_straight": a_w,
                         "nav_pdms_all": n.loc["all", "delta"], "nav_lo": n.loc["all", "lo"], "nav_hi": n.loc["all", "hi"],
                         "nav_pdms_ped": n.loc["ped_cyc_corridor", "delta"], "nav_act_straight": a_n})
    v = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    v.to_csv(RESULTS / "vjepa_real_verdict.csv", index=False, float_format="%.4f")
    nav.to_csv(RESULTS / "vjepa_real_navsim_paired.csv", index=False, float_format="%.4f")
    wt.to_csv(RESULTS / "vjepa_real_wod_deltas.csv", index=False, float_format="%.4f")
    wa.to_csv(RESULTS / "vjepa_real_wod_activation.csv", index=False, float_format="%.4f")
    na.to_csv(RESULTS / "vjepa_real_navsim_activation.csv", index=False, float_format="%.4f")
    summ = {m: {"candidate_seeds": int(v[v.model == m].candidate.sum()),
                "verdict": "V-JEPA 2 enters the fast-channel candidates" if v[v.model == m].candidate.all()
                else ("not a candidate" if not v[v.model == m].candidate.any() else "varies with seed")} for m in MODELS}
    (RESULTS / "vjepa_real_summary.json").write_text(json.dumps(summ, indent=1))
    rl.info(json.dumps(summ) + "\n" + v.to_markdown(index=False, floatfmt=".3f"))
    return v


# ================================================================ main

def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("sf-extract", "sf-check", "sf-fit", "sf-refit-check", "sf-report", "rt-nav-extract",
                                     "rt-heads", "rt-wod", "rt-nav", "rt-verdict", "table"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default=SF)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--fit-dir", default="")
    a = ap.parse_args()
    os.environ.setdefault("P5_SET", SET)
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    if a.step == "sf-fit":
        rl = RunLog("nq3", "q6", "fits", f"{'sf' if a.backbone == SF else 'refit-' + a.backbone}-seed{a.seed}")
    else:
        rl = RunLog("nq3", "q6", "steps", a.step)
    rl.event("start", args=vars(a), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    t0 = time.time()
    if a.step == "sf-extract":
        r = sf_extract(rl, a.batch, a.workers, a.limit)
    elif a.step == "sf-check":
        r = sf_check(rl, a.n, a.batch, a.workers)
    elif a.step == "sf-fit":
        r = sf_fit(rl, a.seed, a.backbone)
        r["dir"] = str(rl.dir)
    elif a.step == "sf-refit-check":
        r = sf_refit_check(Path(a.fit_dir))
        rl.info(json.dumps(r, indent=1, default=float))
        assert r["ok"], "driver refit does not reproduce N6"
    elif a.step == "sf-report":
        r = {"rows": len(sf_report(rl))}
    elif a.step == "rt-nav-extract":
        r = rt_nav_extract(rl, a.batch, a.workers)
    elif a.step == "rt-heads":
        r = rt_heads(rl)
    elif a.step == "rt-wod":
        r = rt_wod(rl)
    elif a.step == "rt-nav":
        r = rt_nav(rl)
    elif a.step == "rt-verdict":
        r = {"rows": len(rt_verdict(rl))}
    else:
        from . import nq3_q6_table as T
        r = T.run(rl)
    rl.info(f"{a.step} done in {time.time() - t0:.0f} s: {json.dumps(r, default=float)[:2000]}")
    rl.event("end", step=a.step, wall_s=time.time() - t0)
    rl.close()


if __name__ == "__main__":
    main()
