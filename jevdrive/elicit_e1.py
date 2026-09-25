"""Elicitation E1: zero-shot transfer of the P5 v1 M-C dual-stream reaction head to real data
(todos/2026-09-26-elicitation-program.md, E1 and deviation-log entry [E1] 00:35, written before any number).

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
    dev = "cuda" if torch.cuda.is_available() else "cpu"     # the stored run is a GPU fit; see deviation [E1] 00:45
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


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("wod",))
    a = ap.parse_args()
    rl = RunLog("elicitation", f"e1-{a.what}")
    run_wod(rl)
    rl.close()


if __name__ == "__main__":
    main()
