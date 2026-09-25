"""Elicitation E1: zero-shot transfer of the P5 v1 M-C dual-stream reaction head to real data
(todos/2026-09-26-elicitation-program.md, E1 and deviation-log entry [E1] 00:25, written before any number).

  head    the five route-fold heads of the `M-C pair` arm on the P5 v1 BehaviorAgent set, recomputed with
          reactivity_mc.fit_fold unchanged and checked against the stored predictions and lambdas of that run;
          the transferred correction is the mean of the five folds' Delta, each fold standardising with its own
          CARLA training rows (main) -- or, descriptive only, with WOD train rows
  prior   entry 40 (iii): WOD-train `A ridge_late op-<model> temporal`, per-frame predictions on val
  eval    p2p3_v1 frames that have Qwen `L18_last` (19 663, 478 rater frames); RFS delta on rater frames, ADE delta
          on s_ego deciles 1-9 (edges on the whole of val), per cluster; activation |v2(prior+Delta) - v2(prior)| >= tau
          with tau the arm's own P5 null threshold

Run on the box: P5_SET=carla_p5v1_ba python -m jevdrive.elicit_e1 wod
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from . import p5_exam as E, p5_openpilot, p5_pairs as P, reactivity_mc as MC, traj, waymo
from .common import data_dir, get_logger

log = get_logger(__name__)
MC_RUN = "runs/reactivity/mc-carla_p5v1_ba/20260925-233126"
PRIOR_NPZ = "runs/drive_backbones/heads_train/20260925-110819/p3drive_heads_preds_dir0.npz"
WOD_FEAT = "processed/waymo_e2e/features"
MODELS = ("cinque", "lebowski")
CLUSTERS = {"Pedestrians": "Pedestrian", "Cyclists": "Cyclist", "Cut_ins": "Cut_ins",
            "FOD": "Foreign Object Debris", "Intersections": "Interections"}
ACT_HARM = 0.07


# ---------------------------------------------------------------- the head

def _stats(X: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """planner.standardize's statistics, kept so the same map can be applied to foreign rows."""
    mu, sd = X[rows].double().mean(0), X[rows].double().std(0, correction=0)
    return mu, torch.where(sd > 1e-6, sd, torch.ones_like(sd))


def fold_heads(model: str, rl) -> list[dict]:
    """Recompute the five `M-C pair [<model>]` fold heads of MC_RUN and verify them against its stored output."""
    assert os.environ.get("P5_SET") == "carla_p5v1_ba", "E1 transfers the v1 BehaviorAgent head: set P5_SET"
    run = data_dir() / MC_RUN
    t, past, fut, obs, null, pairs = E.load()
    n = len(t)
    dev = "cuda" if torch.cuda.is_available() else "cpu"     # the stored run is a GPU fit; see deviation [E1] (7) 00:31
    Q = torch.as_tensor(P.load_features(t, ("L18_last",))["L18_last"], device=dev)
    Xop = torch.as_tensor(p5_openpilot.load(t, (model,), sub="op_streams_vis")[f"op-{model} temporal"], device=dev)
    fold = E.folds(t, pairs)
    F = torch.as_tensor(fut.reshape(n, -1), device=dev)
    Ego = torch.as_tensor(E.ego_input(t, past), device=dev)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    ref = np.load(run / "preds_obs.npz")
    at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
    ref_pred = ref[f"M-C pair [{model}]"]
    ref_lam = {e["fold"]: e["lam"] for e in map(json.loads, open(run / "events.jsonl"))
               if e.get("kind") == "mc_fold" and e.get("arm") == "pair" and e.get("model") == model}
    heads = []
    for f in range(E.K_FOLDS):
        ev = obs_rows[fold[obs_rows] == f]
        keep = {}
        o = MC.fit_fold(f, fold, t, F, Ego, Xop, Q, pr_ip, pr_im, pr_group, rl, model, None, keep)
        diff = float(np.abs(o["M-C pair"][ev].reshape(-1, 20, 2).cpu().numpy() - ref_pred[at[ev].to_numpy()]).max())
        rl.event("e1_head_check", model=model, fold=f, lam=keep["lam"], lam_ref=ref_lam.get(f), max_abs_diff=diff)
        log.info("%s fold %d: lam %g (stored %s), max |pred - stored| %.2e m", model, f, keep["lam"], ref_lam.get(f), diff)
        assert np.isclose(keep["lam"], ref_lam[f]) and diff < 1e-3, "fold head does not reproduce the stored run"
        tr = keep["tr"]
        heads.append({"W": keep["W"].double(), "zbar": keep["zbar"].double(), "lam": keep["lam"],
                      "q": tuple(x.cpu() for x in _stats(Q, tr)), "op": tuple(x.cpu() for x in _stats(Xop, tr)),
                      "diff": diff})
    del Q, Xop, F, Ego
    torch.cuda.empty_cache()
    return heads


def correction(heads: list[dict], Qx: np.ndarray, Ox: np.ndarray, own: tuple | None = None) -> np.ndarray:
    """Mean over the fold heads of Delta(x) = (z(x) - zbar) W, (n, 20, 2). `own` = ((mu, sd) Qwen, (mu, sd) op)
    replaces each head's CARLA statistics by another dataset's (and zbar by that dataset's mean, i.e. 0)."""
    Qx, Ox = torch.as_tensor(Qx).double(), torch.as_tensor(Ox).double()
    out = 0
    for h in heads:
        (mq, sq), (mo, so) = own if own else (h["q"], h["op"])
        Z = torch.cat([(Qx - mq) / sq / np.sqrt(Qx.shape[1]), (Ox - mo) / so / np.sqrt(Ox.shape[1])], 1)
        out = out + (Z - (0 if own else h["zbar"])) @ h["W"]
    return (out / len(heads)).float().numpy().reshape(-1, 20, 2)


# ---------------------------------------------------------------- WOD

def _rows(names: pd.Series, index_names) -> np.ndarray:
    at = pd.Series(np.arange(len(index_names)), index=index_names).reindex(names.to_numpy())
    assert at.notna().all(), f"{int(at.isna().sum())} frames missing"
    return at.astype(int).to_numpy()


def wod_train_stats(model: str) -> tuple:
    """Per-column (mu, sd) of Qwen `L18_last` and op `temporal` over the WOD train rows that have both."""
    root = data_dir() / WOD_FEAT
    qs, names = [], []
    for sh in sorted((root / "qwenvid_train_t4").iterdir()):
        if sh.name.startswith("training_") and (sh / "index.parquet").exists():
            names.append(pd.read_parquet(sh / "index.parquet").frame_name)
            qs.append(np.load(sh / "L18_last.npy"))
    names = pd.concat(names, ignore_index=True)
    Q = torch.as_tensor(np.concatenate(qs).astype(np.float32))
    oi = pd.read_parquet(root / f"op_{model}_p3_trainval/index.parquet").frame_name
    O = np.load(root / f"op_{model}_p3_trainval/temporal.npy", mmap_mode="r")[_rows(names, oi)].astype(np.float32)
    log.info("WOD train statistics: %d rows", len(names))
    allr = np.arange(len(names))
    return _stats(Q, allr), _stats(torch.as_tensor(O), allr)


def wod_frames() -> dict:
    """The evaluation frames, their features, prior predictions, labels and scopes."""
    root = data_dir()
    z = np.load(root / PRIOR_NPZ)
    qi = pd.read_parquet(root / WOD_FEAT / "qwenvid_p3/index.parquet").frame_name
    npz_at = _rows(qi, z["frame_name"])
    s_all = z["s_ego"]
    edges = np.quantile(s_all, np.linspace(0, 1, 11))[1:-1]
    d = {"frame_name": qi.to_numpy(), "sequence": z["sequence"][npz_at], "fut": z["fut"][npz_at],
         "speed": z["speed"][npz_at], "dec": np.clip(np.searchsorted(edges, s_all[npz_at], "right"), 0, 9),
         "Q": np.load(root / WOD_FEAT / "qwenvid_p3/L18_last.npy").astype(np.float32)}
    for m in MODELS:
        d[f"prior {m}"] = z[f"pred_A ridge_late op-{m} temporal"][npz_at]
        oi = pd.read_parquet(root / WOD_FEAT / f"op_{m}_p3_trainval/index.parquet").frame_name
        d[f"op {m}"] = np.load(root / WOD_FEAT / f"op_{m}_p3_trainval/temporal.npy", mmap_mode="r")[_rows(qi, oi)].astype(np.float32)
    sub = pd.read_parquet(root / "processed/waymo_e2e/subsets/p2p3_v1.parquet").set_index("frame_name").loc[qi]
    d["pre_onset"], d["straight_yaw"] = sub.pre_onset.to_numpy(bool), sub.straight_yaw.to_numpy(bool)
    df = waymo.load_index()
    names = waymo.frame_names(df)
    d["cluster"] = df.cluster.to_numpy()[_rows(qi, names)]
    r, rtraj, rscore = waymo.load_rater(df)
    rpos = pd.Series(np.arange(len(qi)), index=qi).reindex(names[r]).to_numpy()
    ok = ~np.isnan(rpos)
    d["rater_pos"], d["rater_traj"], d["rater_score"] = rpos[ok].astype(int), rtraj[ok], rscore[ok]
    log.info("WOD eval: %d frames, %d rater, %d pre_onset, %d straight_yaw", len(qi), ok.sum(), d["pre_onset"].sum(),
             d["straight_yaw"].sum())
    return d


def _ci_row(v: np.ndarray, seq: np.ndarray) -> dict:
    if not len(v):
        return {"n": 0, "delta": np.nan, "lo": np.nan, "hi": np.nan}
    lo, hi = traj.boot_ci(v, seq, b=10000)
    return {"n": len(v), "delta": float(v.mean()), "lo": lo, "hi": hi}


def readouts(d: dict, prior: np.ndarray, delta: np.ndarray, tau: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cluster RFS / ADE deltas and activation rates of prior + Delta against the prior."""
    arm = prior + delta
    seq, pos = d["sequence"], d["rater_pos"]
    rfs = lambda p: waymo.rater_feedback_score(p[pos], d["rater_traj"], d["rater_score"], d["speed"][pos])  # noqa: E731
    drfs = np.full(len(seq), np.nan)
    drfs[pos] = rfs(arm) - rfs(prior)
    ade = lambda p: np.linalg.norm(p - d["fut"], axis=-1).mean(-1)  # noqa: E731
    dade = ade(arm) - ade(prior)
    act = (np.abs(P.v2(arm) - P.v2(prior)) >= tau).astype(float)
    mag = np.linalg.norm(delta, axis=-1).mean(-1)
    scopes = {"all": np.ones(len(seq), bool), **{k: d["cluster"] == v for k, v in CLUSTERS.items()}}
    rows, acts = [], []
    for name, m in scopes.items():
        r = m & ~np.isnan(drfs)
        a = m & (d["dec"] < 9)
        b = a & d["pre_onset"]
        rows += [{"scope": name, "judge": "RFS (rater)", **_ci_row(drfs[r], seq[r])},
                 {"scope": name, "judge": "ADE dec1-9", **_ci_row(dade[a], seq[a])},
                 {"scope": name, "judge": "ADE pre_onset dec1-9", **_ci_row(dade[b], seq[b])}]
    act_scopes = {"straight_yaw": d["straight_yaw"], "pre_onset": d["pre_onset"], **scopes}
    for name, m in act_scopes.items():
        acts.append({"scope": name, "n": int(m.sum()), "activation": float(act[m].mean()),
                     **dict(zip(("lo", "hi"), traj.boot_ci(act[m], seq[m], b=10000))),
                     "delta_mag_median_m": float(np.median(mag[m]))})
    return pd.DataFrame(rows), pd.DataFrame(acts)


def verdict(tab: pd.DataFrame, act: pd.DataFrame, null_ff: float) -> str:
    """The three pre-registered cells (E1 criteria; deviation [E1] (6) for the straight-frame threshold)."""
    g = tab.set_index(["scope", "judge"])
    rfs = lambda s: g.loc[(s, "RFS (rater)")]  # noqa: E731
    a_st = float(act.set_index("scope").loc["straight_yaw", "activation"])
    if rfs("all").hi < 0 or a_st > ACT_HARM:
        return "harmful"
    if (rfs("Pedestrians").lo > 0 or rfs("Cyclists").lo > 0) and a_st <= null_ff and not rfs("all").hi < 0:
        return "transferable"
    crossing = ((tab.lo <= 0) & (tab.hi >= 0)) | tab.n.eq(0)
    if crossing.all() and a_st <= ACT_HARM:
        return "not transferable, harmless"
    return "none of the three cells (some delta excludes zero without a pedestrian/cyclist RFS gain)"


def run_wod(rl):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 12)))
    fl = pd.read_csv(data_dir() / MC_RUN / "flip_rates.csv")
    crit = pd.read_csv(data_dir() / MC_RUN / "criteria.csv").set_index("arm")
    d = wod_frames()
    tabs, acts, verdicts = [], [], []
    for m in MODELS:
        heads = fold_heads(m, rl)
        tau = float(fl[(fl.examinee == f"M-C pair [{m}]") & (fl.scope == "pooled")].tau_model.iloc[0])
        null_ff = float(crit.loc[f"M-C pair [{m}]", "null_ff_oos"])
        for stats in ("carla", "wod-train"):
            own = wod_train_stats(m) if stats == "wod-train" else None
            delta = correction(heads, d["Q"], d[f"op {m}"], own)
            tab, act = readouts(d, d[f"prior {m}"], delta, tau)
            tag = {"model": m, "stats": stats}
            tabs.append(tab.assign(**tag))
            acts.append(act.assign(**tag, tau=tau))
            v = verdict(tab, act, null_ff) if stats == "carla" else "descriptive"
            verdicts.append({**tag, "verdict": v, "tau": tau, "null_ff_oos": null_ff,
                             "fold_lams": [h["lam"] for h in heads], "max_head_diff": max(h["diff"] for h in heads)})
            rl.event("e1_verdict", **verdicts[-1])
            log.info("%s / %s stats: %s\n%s\n%s", m, stats, v, tab.to_markdown(index=False, floatfmt=".3f"),
                     act.to_markdown(index=False, floatfmt=".3f"))
            if stats == "carla":
                np.savez_compressed(rl.dir / f"wod_delta_{m}.npz", frame_name=d["frame_name"], delta=delta)
    pd.concat(tabs).to_csv(rl.dir / "wod_deltas.csv", index=False)
    pd.concat(acts).to_csv(rl.dir / "wod_activation.csv", index=False)
    pd.DataFrame(verdicts).to_csv(rl.dir / "wod_verdict.csv", index=False)


# ---------------------------------------------------------------- NAVSIM (deviation [E1] (5); scored by the devkit)

NAV_HEADS = "runs/navsim_zs/heads/20260925-232810"
NAV_SPLITS = ("navtest", "navhard_two_stage")
NAV_PRIORS = ("ridge_late", "cls_late")          # ridge_late + Delta is the registered row, cls_late + Delta descriptive


def _grid20(p8: np.ndarray) -> np.ndarray:
    """(n, 20, 2) on the 0.25 s grid with only the points P.v2 reads (1.75 s, 2.0 s) filled from 0.5 s poses."""
    g = np.zeros((len(p8), 20, 2), np.float32)
    g[:, 6], g[:, 7] = (p8[:, 2, :2] + p8[:, 3, :2]) / 2, p8[:, 3, :2]
    return g


def nav_scopes(tokens: np.ndarray) -> pd.DataFrame:
    """navtest groups: a pedestrian / cyclist GT agent in the logged-path corridor (<= 30 m, +-1.5 m; E3's function),
    and straight frames (waymo.subsets' straight_yaw on NAVSIM's 2 Hz poses: yaw rate over the last 0.5 s < 1 deg/s,
    moved >= 1 m over the last 1 s, chord bearing at 3 s <= 5 deg with a chord >= 3 m)."""
    from . import elicit_e3 as E3, navsim_zs as Z
    fz = np.load(data_dir() / "runs/navsim_zs/index/navtest_future.npz")
    assert (fz["tokens"] == tokens).all()
    fut = fz["poses"]
    fl = E3.cause_flags_nav(tokens, fut[:, :, :2], E3.extract("navtest"))
    idx = {e["token"]: e for e in Z.load_index("navtest", slim=True)}
    pose = np.stack([idx[t]["pose"] for t in tokens])
    dyaw = np.arctan2(np.sin(pose[:, -1, 2] - pose[:, -2, 2]), np.cos(pose[:, -1, 2] - pose[:, -2, 2]))
    w = np.abs(np.degrees(dyaw)) / 0.5
    moved = np.linalg.norm(pose[:, -1, :2] - pose[:, -3, :2], axis=1) >= waymo.MIN_PAST_DISP
    b = np.degrees(np.arctan2(fut[:, 5, 1], fut[:, 5, 0]))
    chord = np.linalg.norm(fut[:, 5, :2], axis=1) >= waymo.MIN_CHORD
    return pd.DataFrame({"token": tokens, "ped_cyc_corridor": (fl.in_pedestrian | fl.in_bicycle).to_numpy(),
                         "straight": moved & (w < waymo.ONSET_YAW_RATE) & chord & (np.abs(b) <= waymo.ONSET_BEARING)})


def run_navsim(rl):
    """Write prior + Delta predictions for the devkit (navtest, navhard two-stage) and the activation table."""
    from . import navsim_qwen as NQ
    fl = pd.read_csv(data_dir() / MC_RUN / "flip_rates.csv")
    acts = []
    for m in MODELS:
        heads = fold_heads(m, rl)
        tau = float(fl[(fl.examinee == f"M-C pair [{m}]") & (fl.scope == "pooled")].tau_model.iloc[0])
        for split in NAV_SPLITS:
            z = np.load(data_dir() / "runs/navsim_zs/openpilot" / split / f"{m}_temporal.npz")
            tok = z["tokens"]
            Q = NQ.load(split, tok)["L18_last"]
            delta = correction(heads, Q, z["temporal"].astype(np.float32))
            np.savez_compressed(rl.dir / f"{split}_delta_{m}.npz", tokens=tok, delta=delta)
            for pr in NAV_PRIORS:
                p = np.load(data_dir() / NAV_HEADS / f"{split}_{pr}_{m}_temporal.npz")
                assert (p["tokens"] == tok).all()
                arm = p["poses"].copy()
                arm[..., :2] += delta[:, 1:16:2]                   # 0.5 ... 4.0 s of the 0.25 s grid; heading kept
                np.savez(rl.dir / f"{split}_{pr}_{m}_plus_mc.npz", tokens=tok, poses=arm.astype(np.float32))
                if split != "navtest":
                    continue
                act = (np.abs(P.v2(_grid20(arm)) - P.v2(_grid20(p["poses"]))) >= tau).astype(float)
                sc = nav_scopes(tok)
                mag = np.linalg.norm(delta[:, 1:16:2], axis=-1).mean(-1)
                for name, msk in (("all", np.ones(len(tok), bool)), ("straight", sc.straight.to_numpy()),
                                  ("ped_cyc_corridor", sc.ped_cyc_corridor.to_numpy()),
                                  ("no ped_cyc", ~sc.ped_cyc_corridor.to_numpy())):
                    acts.append({"model": m, "prior": pr, "scope": name, "n": int(msk.sum()), "tau": tau,
                                 "activation": float(act[msk].mean()), "delta_mag_median_m": float(np.median(mag[msk]))})
                sc.to_csv(rl.dir / "navtest_scopes.csv", index=False)
    a = pd.DataFrame(acts)
    a.to_csv(rl.dir / "navsim_activation.csv", index=False)
    log.info("activation\n%s", a.to_markdown(index=False, floatfmt=".3f"))


def _devkit(ver: str, split: str, name: str) -> pd.DataFrame | None:
    from .openloop_standing import _latest
    df = _latest(ver, split, name)
    if df is None:
        return None
    return df


def navsim_table(rl, run_dir):
    """PDMS / EPDMS of prior + Delta against the stored prior scores (paired, token bootstrap), per navtest group, and
    navhard two-stage EPDMS (aggregate: its two-stage weighting is not per token)."""
    from pathlib import Path
    run_dir = Path(run_dir)
    sc = pd.read_csv(run_dir / "navtest_scopes.csv").set_index("token")
    groups = {"all": None, "ped_cyc_corridor": sc.ped_cyc_corridor, "no ped_cyc": ~sc.ped_cyc_corridor,
              "straight": sc.straight}
    rows, hard = [], []
    rng = np.random.default_rng(0)
    for m in MODELS:
        for pr in NAV_PRIORS:
            for ver, metric in (("v1", "PDMS"), ("v2", "EPDMS")):
                a = _devkit(ver, "navtest", f"e1_{pr}_{m}_plus_mc")
                b = _devkit(ver, "navtest", f"heads_{pr}_{m}_temporal")
                if a is None or b is None:
                    continue
                f = lambda df: df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)].set_index("token")["score"].astype(float)  # noqa: E731
                x, y = f(a).align(f(b), join="inner")
                for g, msk in groups.items():
                    keep = np.ones(len(x), bool) if msk is None else msk.reindex(x.index).fillna(False).to_numpy(bool)
                    d = (x - y).to_numpy()[keep]
                    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                    rows.append({"model": m, "prior": pr, "metric": metric, "group": g, "n": len(d),
                                 "prior_score": 100 * y.to_numpy()[keep].mean(), "plus_mc": 100 * x.to_numpy()[keep].mean(),
                                 "delta": 100 * d.mean(), "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)})
        for name in (f"e1_ridge_late_{m}_plus_mc", f"heads_ridge_late_{m}_temporal"):
            df = _devkit("v2", "navhard_two_stage", name)
            if df is not None:
                summ = df[df["token"].str.startswith("extended_pdm_score")].set_index("token")["score"]
                hard.append({"model": m, "name": name, "EPDMS": 100 * summ.get("extended_pdm_score_combined", np.nan)})
    t, h = pd.DataFrame(rows), pd.DataFrame(hard)
    t.to_csv(rl.dir / "navsim_paired.csv", index=False)
    h.to_csv(rl.dir / "navhard.csv", index=False)
    log.info("navtest\n%s\nnavhard\n%s", t.to_markdown(index=False, floatfmt=".2f"), h.to_markdown(index=False, floatfmt=".2f"))


def figs(res_dir, out_dir):
    """RFS delta per cluster and activation rate per scope (research/results/elicitation/e1 -> research/figs)."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from . import plots
    res_dir, out_dir = Path(res_dir), Path(out_dir)
    tab, act = pd.read_csv(res_dir / "wod_deltas.csv"), pd.read_csv(res_dir / "wod_activation.csv")
    scopes = ["all", *CLUSTERS]
    series = [(m, s, c, mk) for m, c in (("cinque", plots.OKABE_ITO[5]), ("lebowski", plots.OKABE_ITO[6]))
              for s, mk in (("carla", "o"), ("wod-train", "^"))]
    with plots.mpl.rc_context(plots.STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(plots.PAGE, 2.0), gridspec_kw={"width_ratios": [1.1, 1]})
        for k, (m, s, c, mk) in enumerate(series):
            r = tab[(tab.model == m) & (tab.stats == s) & (tab.judge == "RFS (rater)")].set_index("scope").loc[scopes]
            x = np.arange(len(scopes)) + (k - 1.5) * 0.17
            a.errorbar(x, r.delta, yerr=[r.delta - r.lo, r.hi - r.delta], fmt=mk, color=c, ms=3, lw=0.8, capsize=1.5,
                       mfc=c if s == "carla" else "white", label=f"{m.capitalize()}, {'CARLA' if s == 'carla' else 'WOD-train'} stats")
        a.axhline(0, color="0.5", lw=0.6)
        a.set_xticks(np.arange(len(scopes)), ["All", "Ped.", "Cyc.", "Cut-in", "FOD", "Inters."])
        a.set_ylabel(r"$\Delta$RFS (prior + $\Delta$ $-$ prior)")
        sc2 = ["straight_yaw", "pre_onset", "Pedestrians", "all"]
        for k, (m, s, c, mk) in enumerate(series):
            r = act[(act.model == m) & (act.stats == s)].set_index("scope").loc[sc2]
            x = np.arange(len(sc2)) + (k - 1.5) * 0.17
            b.errorbar(x, 100 * r.activation, yerr=[100 * (r.activation - r.lo), 100 * (r.hi - r.activation)], fmt=mk,
                       color=c, ms=3, lw=0.8, capsize=1.5, mfc=c if s == "carla" else "white")
        b.axhline(100 * ACT_HARM, color="0.3", ls="--", lw=0.7)
        b.set_xlim(-0.5, 3.75)
        b.text(3.72, 100 * ACT_HARM + 0.5, "7%", ha="right", va="bottom", fontsize=6.5, color="0.3")
        b.set_xticks(np.arange(len(sc2)), ["Straight", "Pre-onset", "Ped.", "All"])
        b.set_ylabel("Activation rate (%)")
        plots.legend_below(fig, a, ncol=4)
        plots.save(fig, out_dir, "elicit-e1-wod-transfer")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("wod", "navsim", "navsim-table", "figs"))
    ap.add_argument("--run", default="", help="navsim-table: the e1-navsim run dir")
    ap.add_argument("--res", default="research/results/elicitation/e1")
    ap.add_argument("--out", default="research/figs")
    a = ap.parse_args()
    if a.what == "figs":
        figs(a.res, a.out)
        return
    rl = RunLog("elicitation", f"e1-{a.what}")
    if a.what == "navsim-table":
        navsim_table(rl, a.run)
    else:
        {"wod": run_wod, "navsim": run_navsim}[a.what](rl)
    rl.close()


if __name__ == "__main__":
    main()
