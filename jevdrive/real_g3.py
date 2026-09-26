"""Real-data transfer round 2, G3: diagnostics of the E2 edit pairs (todos/2026-09-26-real-data-transfer.md, G3 and
deviation-log entries [G3], written before any G3 number).

  g3a   CARLA reference of E2's gate-(d) statistic: per-stream feature shift |f(x+) - f(x-)| on the P5 v1 BA pairs
        over the same shift on the weather null pairs (elicit_e2_train.feature_floor's statistic, unchanged, route
        bootstrap), for Qwen `L18_last`, openpilot `temporal` and the E5 YOLO embedding; plus an R1-style row on the
        stored M-C Delta
  pdm-prep / pdm-read   label (c) of E2 on the 911 navtrain edit pairs: the official v1.1 PDMS of E2's own "continue"
        (elicit_e2_train.cv_path) and "brake" (ctra, 3 m/s^2) proposals on x+ (scripts/real_g3_pdm.sh runs the devkit),
        written to processed/elicit_e2/navtrain/main/pdm_scores.csv, where elicit_e2_train.load_data reads it
  g3c   cost table of extending the WOD edit pairs from 216 to every in-corridor pedestrian frame of WOD train (estimate
        only: E2's measured rates, the 0.8 s SAM scan's corridor frames and their event structure)
  mc    E2's training and readouts (elicit_e2_train.run, unchanged) with label (c) present; seed s >= 1 only replaces
        the inner lambda split (reactivity_mc._inner_splits: logs permuted with default_rng(s), then 3 folds)

Run on the box: P5_SET=carla_p5v1_ba python -m jevdrive.real_g3 g3a
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
PED = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian", "ParkingCrossingPedestrian")
CUTIN = ("HighwayCutIn", "StaticCutIn", "ParkingCutIn")
MODELS = ("cinque", "lebowski")
MC_RUN = "runs/reactivity/mc-carla_p5v1_ba/20260925-233126"
E2_RUN = "runs/elicitation/e2-train/20260926-020936"
# E2's gate (d) and R1 on the navtrain edit pairs (elicitation todo, results "E2: navtrain ..."), the comparators
E2_FEATURE = {"Qwen L18_last": (0.322, 0.267, 1.21, 1.08, 1.28), "openpilot cinque temporal": (0.070, 0.029, 2.40, 2.03, 2.76),
              "openpilot lebowski temporal": (0.076, 0.040, 1.92, 1.67, 2.31)}


def _shift(Xp: np.ndarray, Xq: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """feature_floor's per-row shift: RMS over dims of the difference in units of the x+ rows' per-dim std."""
    return np.sqrt((((Xp - Xq) / sd) ** 2).mean(1))


def g3a(rl):
    assert os.environ.get("P5_SET") == "carla_p5v1_ba", "G3a is registered on the P5 v1 BA set: set P5_SET"
    from . import p5_exam as E, p5_openpilot, p5_pairs as P
    from .elicit_e2_train import _mag, _ratio_ci
    t, past, fut, obs, null, pairs = E.load()
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    ip, im = pos[obs.fn_plus].to_numpy(), pos[obs.fn_minus].to_numpy()
    np_, nn = pos[null.fn_plus].to_numpy(), pos[null.fn_null].to_numpy()
    tau_exp = max(float(np.quantile(np.abs(null.d_expert), 0.95)), E.TAU_EXP_MIN)
    fam, nfam = obs.family.to_numpy(), null.family.to_numpy()
    px = obs.factor_px.to_numpy().astype(float)
    ped = np.isin(fam, PED)
    q = np.nanquantile(px[ped], [1 / 3, 2 / 3])
    scopes = {"pedestrian (judged)": (ped, np.ones(len(null), bool)),
              "all families": (np.ones(len(obs), bool), np.ones(len(null), bool)),
              "cut-in": (np.isin(fam, CUTIN), np.ones(len(null), bool)),
              "pedestrian, reactive frames": (ped & (np.abs(obs.d_expert.to_numpy()) > tau_exp), np.ones(len(null), bool)),
              "pedestrian, null of pedestrian cases": (ped, np.isin(nfam, PED)),
              "pedestrian, factor_px tercile 1": (ped & (px <= q[0]), np.ones(len(null), bool)),
              "pedestrian, factor_px tercile 2": (ped & (px > q[0]) & (px <= q[1]), np.ones(len(null), bool)),
              "pedestrian, factor_px tercile 3": (ped & (px > q[1]), np.ones(len(null), bool))}
    rl.info(f"obs {len(obs)} ({int(ped.sum())} pedestrian, {obs[ped].base_id.nunique()} routes), null {len(null)} "
            f"({null.base_id.nunique()} routes), tau_exp {tau_exp:.3f}, factor_px terciles {q}")

    feats = {"Qwen L18_last": P.load_features(t, ("L18_last",))["L18_last"]}
    for m in MODELS:
        feats[f"openpilot {m} temporal"] = p5_openpilot.load(t, (m,), sub="op_streams_vis")[f"op-{m} temporal"]
    feats["YOLO embedding (E5)"] = np.load(data_dir() / "processed" / "elicit_e5" / "embed.npy").astype(np.float32)
    rows = []
    g_obs, g_null = obs.base_id.astype(str).to_numpy(), null.base_id.astype(str).to_numpy()
    for name, X in feats.items():
        X = np.asarray(X, np.float32)
        s = X[ip].std(0)                                   # x+ rows of the pairs, as feature_floor
        sd = np.where(s > 1e-6, s, 1.0)
        e, p = _shift(X[ip], X[im], sd), _shift(X[np_], X[nn], sd)
        for sc, (mo, mn) in scopes.items():
            r, lo, hi = _ratio_ci(e[mo], p[mn], g_obs[mo], g_null[mn])
            rows.append({"feature": name, "scope": sc, "n_edit": int(mo.sum()), "routes_edit": int(len(np.unique(g_obs[mo]))),
                         "n_null": int(mn.sum()), "median_edit": float(np.median(e[mo])), "median_null": float(np.median(p[mn])),
                         "ratio": r, "lo": lo, "hi": hi})
        rl.event("g3a_feature", feature=name, dim=int(X.shape[1]))
    ff = pd.DataFrame(rows)
    # R1-style description on the stored out-of-fold M-C Delta (pred - prior) of the obs rows
    ref = np.load(data_dir() / MC_RUN / "preds_obs.npz")
    at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
    r1 = []
    for m in MODELS:
        D = ref[f"M-C pair [{m}]"] - ref[f"prior [{m}]"]
        a = lambda ix: D[at[ix].to_numpy()]  # noqa: E731
        e, p = _mag(a(ip), a(im)), _mag(a(np_), a(nn))
        for sc, (mo, mn) in scopes.items():
            r, lo, hi = _ratio_ci(e[mo], p[mn], g_obs[mo], g_null[mn])
            r1.append({"head": f"M-C pair [{m}] (stored, out of fold)", "scope": sc, "n_edit": int(mo.sum()), "n_null": int(mn.sum()),
                       "median_edit_m": float(np.median(e[mo])), "median_null_m": float(np.median(p[mn])), "ratio": r, "lo": lo, "hi": hi})
    r1 = pd.DataFrame(r1)
    ff.to_csv(rl.dir / "feature_shift.csv", index=False)
    r1.to_csv(rl.dir / "r1_mc_delta.csv", index=False)
    rl.info("feature shift, CARLA pairs vs weather null\n" + ff.to_markdown(index=False, floatfmt=".3f"))
    rl.info("R1-style, stored M-C Delta\n" + r1.to_markdown(index=False, floatfmt=".3f"))
    # side by side with E2 (judged scope only)
    j = ff[ff.scope == "pedestrian (judged)"].set_index("feature")
    side = []
    for name in feats:
        e2 = E2_FEATURE.get(name, (np.nan,) * 5)
        c = j.loc[name]
        side.append({"feature": name, "carla_median_edit": c.median_edit, "carla_median_null": c.median_null,
                     "carla_ratio": c.ratio, "carla_lo": c.lo, "carla_hi": c.hi, "e2_median_edit": e2[0],
                     "e2_median_placebo": e2[1], "e2_ratio": e2[2], "e2_lo": e2[3], "e2_hi": e2[4]})
    side = pd.DataFrame(side)
    side.to_csv(rl.dir / "side_by_side.csv", index=False)
    qr = float(j.loc["Qwen L18_last"].ratio)
    verdict = "< 2: edit quality is not E2's bottleneck, run G3b" if qr < 2 else \
        "[2, 3): gray zone, run G3b" if qr < 3 else ">= 3: stop at G3a"
    rl.info("side by side\n" + side.to_markdown(index=False, floatfmt=".3f") + f"\nverdict (Qwen, pedestrian scope, {qr:.3f}): {verdict}")
    rl.event("g3a_verdict", qwen_ratio=qr, verdict=verdict)


# ---------------------------------------------------------------- G3b (1): PDM scorer label (c)

T8 = np.arange(1, 9) * 0.5


def proposal8(e: dict, decel: float | None = None) -> np.ndarray:
    """(8, 3) x, y, yaw at 0.5 ... 4 s of elicit_e2_train.ctra (decel=None with zero acceleration: cv_path), the same
    integration, sampled for the devkit; yaw = yaw rate x t."""
    v0 = float(np.linalg.norm(e["vel"][-1]))
    a0 = -decel if decel is not None else 0.0
    dyaw = e["pose"][-1, 2] - e["pose"][-2, 2]
    w = float(np.arctan2(np.sin(dyaw), np.cos(dyaw)) / 0.5)
    dt = 0.01
    ts = np.arange(0, 4.0 + dt / 2, dt)
    v = np.maximum(v0 + a0 * ts, 0.0)
    th = w * ts
    x = np.r_[0, np.cumsum(v[:-1] * np.cos(th[:-1]) * dt)]
    y = np.r_[0, np.cumsum(v[:-1] * np.sin(th[:-1]) * dt)]
    return np.stack([np.interp(T8, ts, x), np.interp(T8, ts, y), w * T8], -1).astype(np.float32)


def pdm_prep(rl):
    from . import elicit_e2 as E2, navsim_zs as Z
    from .elicit_e2_train import ctra, cv_path
    toks = pd.read_parquet(E2.out_root("navtrain", "main") / "pairs.parquet").token.tolist()
    slim = {e["token"]: e for e in Z.load_index("navtrain", slim=True)}
    cont = np.stack([proposal8(slim[t]) for t in toks])
    brake = np.stack([proposal8(slim[t], 3.0) for t in toks])
    # the devkit poses are the label's own trajectories: (x, y) at 0.5 ... 4 s = every second point of the 16-point grid
    for P8, P16 in ((cont, np.stack([cv_path(slim[t]) for t in toks])), (brake, np.stack([ctra(slim[t], 3.0) for t in toks]))):
        err = float(np.abs(P8[:, :, :2] - P16[:, 1::2]).max())
        assert err < 1e-4, err
    (rl.dir / "tokens.txt").write_text("\n".join(toks) + "\n")
    np.savez(rl.dir / "continue.npz", tokens=np.array(toks), poses=cont)
    np.savez(rl.dir / "brake.npz", tokens=np.array(toks), poses=brake)
    rl.info(f"{len(toks)} tokens; continue / brake proposals match the label trajectories")


def pdm_read(rl, run_dir: str, cont_csv: str, brake_csv: str):
    from . import elicit_e2 as E2, navsim_zs as Z
    toks = np.loadtxt(Path(run_dir) / "tokens.txt", dtype=str).tolist()
    sc = {k: pd.read_csv(p).set_index("token") for k, p in (("continue", cont_csv), ("brake", brake_csv))}
    S = pd.DataFrame({"token": toks, **{k: sc[k].score.reindex(toks).to_numpy() for k in sc}})
    dst = E2.out_root("navtrain", "main") / "pdm_scores.csv"
    S.to_csv(dst, index=False)
    S.to_csv(rl.dir / "pdm_scores.csv", index=False)
    d = S.brake - S["continue"]
    g = pd.read_csv(data_dir() / E2_RUN / "rule_gates.csv").set_index("token").reindex(toks)
    rule = (g.gate_plus & ~g.gate_minus).to_numpy()
    slim = {e["token"]: e for e in Z.load_index("navtrain", slim=True)}
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        fut = dict(zip(f["tokens"].tolist(), f["poses"]))
    slow = np.array([4.0 * float(np.linalg.norm(slim[t]["vel"][-1])) -
                     float(np.linalg.norm(np.diff(np.r_[[[0.0, 0.0]], fut[t][:, :2]], axis=0), axis=1).sum()) for t in toks])
    human = np.where(slow > 2, 1, np.where(slow < -2, -1, 0))
    fire = (d > 0).to_numpy()
    dec = (human != 0) & (np.abs(d.to_numpy()) >= 0.05)
    summ = {"tokens": len(toks), "scored": int(S[["continue", "brake"]].notna().all(1).sum()),
            "label_c_fires": int(fire.sum()), "share_fires": float(fire.mean()),
            "diff_median": float(d.median()), "diff_p10": float(d.quantile(.1)), "diff_p90": float(d.quantile(.9)),
            "continue_pdms_mean": float(S["continue"].mean()), "brake_pdms_mean": float(S.brake.mean()),
            "rule_b_fires": int(rule.sum()), "both_fire": int((fire & rule).sum()), "c_only": int((fire & ~rule).sum()),
            "b_only": int((~fire & rule).sum()),
            "human_decisive": int(dec.sum()), "agree_with_human_slowing": float((np.sign(d.to_numpy()[dec]) == human[dec]).mean())}
    pd.DataFrame([summ]).to_csv(rl.dir / "pdm_summary.csv", index=False)
    rl.info("label (c) summary\n" + pd.DataFrame([summ]).T.to_markdown(floatfmt=".3f"))
    rl.event("g3b_pdm", **summ)


# ---------------------------------------------------------------- G3c: cost table (no GPU)

# E2's measured rates (elicitation todo, results "E2: WOD ..." and run logs): SAM 3.1 exact scan 1.3 GPU h for 51 970
# frames; pair build (SAM on 12 clip images + LaMa) 1.6-1.7 s / pair on a free card (6.7 s shared); Qwen 0.254 s / clip
SAM_S_PER_FRAME = 1.3 * 3600 / 51970
BUILD_S_PER_PAIR = 1.7
QWEN_S_PER_CLIP = 0.254
LAMA_SAM_S_PER_IMG = 0.45            # navtrain build: 0.23-0.74 s / edited image incl. SAM, shared cards
YOLO_S_PER_FRAME = SAM_S_PER_FRAME / 25   # decision 45: YOLO26x-seg at 1/25 of SAM 3.1's latency
OP_HISTORY_S = 10.0                  # E2 [E2] 01:22 (3): openpilot's effective memory to edit for a streamed x-


def g3c(rl):
    from . import elicit_e2 as E2, fusion_q4 as Q, waymo as W
    from .fusion_q2b import calibs, extend
    df = W.load_index()
    cal = calibs()
    full = (df.split == "train").to_numpy() & df.has_future.to_numpy() & df.sequence.isin(cal).to_numpy()
    n_full, n_seq = int(full.sum()), int(df.sequence[full].nunique())
    lst = pd.read_parquet(E2.out_root("wod") / "scan_list.parquet")
    d = Q.load_dets(E2.out_root("wod", "sam"), score=E2.SAM_SCORE)
    d = d[d.prompt.isin(E2.SAM_PROMPTS) & ((d.y1 - d.y0) >= E2.MIN_PX)]
    d = Q.lift_dets(d, d.key.map(lst.set_index("key").sequence).to_numpy(), cal)
    d = d[d.lift_ok & (d.gx > 0) & (d.gdist <= E2.RANGE)]
    at = pd.Series(np.arange(len(df)), index=W.frame_names(df))
    _, future = W.load_ego()
    keep = []
    for key, g in d.groupby("key"):                      # wod_candidates' corridor test, unchanged
        path = extend(np.r_[[[0.0, 0.0]], future[at[key], :, :2]], E2.RANGE)
        if any(E2.seg_dist(q, path) <= E2.CORRIDOR for q in g[["gx", "gy"]].to_numpy(float)):
            keep.append(int(at[key]))
    c = pd.DataFrame({"sequence": df.sequence.to_numpy()[keep], "frame": df.frame.to_numpy()[keep]}).sort_values(["sequence", "frame"])
    # events: runs of consecutive scanned frames (0.8 s apart) in a sequence
    new = (c.sequence != c.sequence.shift()) | (c.frame.diff() != E2.WOD_THIN)
    c["event"] = new.cumsum()
    ev = c.groupby("event").agg(sequence=("sequence", "first"), n=("frame", "size"))
    dur = ev.n * E2.WOD_THIN / 10.0                       # s, lower bound at 0.8 s sampling
    n_corr, n_ev = len(c), len(ev)
    share = n_corr / len(lst)
    corr_full = share * n_full                            # every in-corridor frame at 10 Hz
    cand_gap = int(sum(np.ceil(x / E2.WOD_GAP_S) for x in dur))   # E2's one-per-5-s rule at full rate, upper-ish
    stats = {"train_frames_full": n_full, "train_sequences": n_seq, "scan_frames": len(lst), "scan_corridor_frames": n_corr,
             "corridor_share": share, "scan_events": n_ev, "event_sequences": int(ev.sequence.nunique()),
             "event_duration_median_s": float(dur.median()), "event_duration_p90_s": float(dur.quantile(.9)),
             "full_rate_corridor_frames_est": corr_full, "full_rate_candidates_5s_rule_est": cand_gap, "e2_pairs": 216}
    rl.info("scan structure\n" + pd.Series(stats).to_markdown(floatfmt=".3f"))
    gh = lambda sec: sec / 3600  # noqa: E731
    rows = []
    def row(option, scan, n_pairs, indep, extra=0.0, note=""):
        build, qwen = gh(n_pairs * BUILD_S_PER_PAIR), gh(n_pairs * 3 * QWEN_S_PER_CLIP)
        tot = scan + build + qwen + extra
        rows.append({"option": option, "pairs": int(round(n_pairs)), "independent_events": int(round(indep)),
                     "scan_gpu_h": scan, "build_gpu_h": build, "qwen_gpu_h": qwen, "op_stream_edit_gpu_h": extra,
                     "total_gpu_h": tot, "wall_h_5_cards": tot / 5, "note": note})
    sam_full, yolo_full = gh(n_full * SAM_S_PER_FRAME), gh(n_full * YOLO_S_PER_FRAME)
    row("E2 as run (0.8 s SAM scan, one per sequence per 5 s)", gh(len(lst) * SAM_S_PER_FRAME), 216, 216, note="measured")
    row("full 10 Hz SAM scan, one per sequence per 5 s", sam_full, cand_gap, cand_gap)
    row("full 10 Hz SAM scan, every in-corridor frame", sam_full, corr_full, n_ev, note="frames of one event are near-duplicates")
    row("full 10 Hz YOLO scan + SAM masks on hits, every in-corridor frame", yolo_full, corr_full, n_ev,
        note="YOLO recall not below SAM (decision 45); SAM only inside the build")
    row("existing 0.8 s scan, every in-corridor frame (no new scan)", 0.0, n_corr, n_ev)
    op_imgs = n_ev * OP_HISTORY_S * 10 + corr_full    # front history per event + the event's own frames
    row("full YOLO scan, every in-corridor frame, + openpilot stream edited", yolo_full, corr_full, n_ev,
        extra=gh(op_imgs * LAMA_SAM_S_PER_IMG), note="per-frame LaMa; video-consistent inpainting would add to this")
    T = pd.DataFrame(rows)
    T.to_csv(rl.dir / "g3c_cost.csv", index=False)
    pd.DataFrame([stats]).to_csv(rl.dir / "g3c_scan_structure.csv", index=False)
    rl.info("G3c cost table\n" + T.to_markdown(index=False, floatfmt=".2f"))


# ---------------------------------------------------------------- G3b (2): M-C retrain, E2's code

def _seeded_inner_splits(seed: int):
    def splits(groups: np.ndarray, k: int = 3):
        u = np.random.default_rng(seed).permutation(np.unique(groups))
        f = pd.Series(np.arange(len(u)) % k, index=u)[groups].to_numpy()
        return [(np.flatnonzero(f != j), np.flatnonzero(f == j)) for j in range(k)]
    return splits


def mc(rl, seed: str = "0"):
    from . import elicit_e2_train as E2T, reactivity_mc as MC
    assert os.environ.get("P5_SET") == "carla_p5v1_ba", "R2 of E2 reads the P5 v1 BA set: set P5_SET"
    s = int(seed)
    if s:
        MC._inner_splits = _seeded_inner_splits(s)
    rl.event("g3b_mc_seed", seed=s)
    E2T.run(rl)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("g3a", "g3c", "pdm-prep", "pdm-read", "mc"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    name = a.step if a.step.startswith("g3") else "g3" + a.step.replace("-", "") + (f"-s{a.args[0]}" if a.step == "mc" else "")
    rl = RunLog("real-data-transfer", name)
    {"g3a": g3a, "g3c": g3c, "pdm-prep": pdm_prep, "pdm-read": pdm_read, "mc": mc}[a.step](rl, *a.args)
    rl.close()


if __name__ == "__main__":
    main()
