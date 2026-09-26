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
  edit-list / edit-embed   t0 images (3 cameras) of every side of the edit pairs -> YOLO (scripts/real_g3_detect.sh,
        E5 / G0 config) -> G0's real-data embedding (real_g0: ego-history arc corridor, per-token calibration)
  p5-embed   the same real-data embedding on the P5 v1 BA rows from E5's stored detections (for R2)
  student    E5's student recipe on the navtrain edit pairs (labels (c) judged, (a) described; arms A / B; seeds 0-2),
        R1 / R2 / R3 exactly as E2
  figs  (on the Mac) research/figs/real-g3a-shift.{pdf,png} from the pulled small tables

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


def g3a_yolo_e2(rl):
    """The E2 side of G3a's YOLO row (description): the same statistic on the navtrain edit pairs' real-data
    embedding (G0's definition), edit vs placebo, log bootstrap."""
    from . import elicit_e2 as E2
    from .elicit_e2_train import _ratio_ci
    pairs = pd.read_parquet(E2.out_root("navtrain", "main") / "pairs.parquet")
    emb = {}
    for sd in SIDES:
        z = np.load(g3_root(f"edit_embed_{sd}.npz"), allow_pickle=True)
        emb[sd] = pd.DataFrame(z["embed"], index=z["tokens"])
    toks, pl = pairs.token.to_numpy(), pairs.token[pairs.n_placebo_imgs > 0].to_numpy()
    X = emb["plus"].loc[toks].to_numpy()
    s = X.std(0)
    sd = np.where(s > 1e-6, s, 1.0)
    e, p = _shift(X, emb["minus"].loc[toks].to_numpy(), sd), _shift(emb["plus"].loc[pl].to_numpy(), emb["placebo"].loc[pl].to_numpy(), sd)
    lg = pairs.set_index("token").log
    r, lo, hi = _ratio_ci(e, p, lg[toks].to_numpy(), lg[pl].to_numpy())
    out = pd.DataFrame([{"feature": "YOLO embedding (G0 real-data)", "n_edit": len(e), "n_placebo": len(p),
                         "median_edit": float(np.median(e)), "median_placebo": float(np.median(p)), "ratio": r, "lo": lo, "hi": hi,
                         "share_edit_zero": float((e == 0).mean()), "share_placebo_zero": float((p == 0).mean())}])
    out.to_csv(rl.dir / "e2_yolo_shift.csv", index=False)
    rl.info("E2 side, YOLO embedding\n" + out.to_markdown(index=False, floatfmt=".3f"))


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


# ---------------------------------------------------------------- G3b (3): the student on the edit pairs

SIDES = ("plus", "minus", "placebo")
MC_G3_RUN = "runs/real-data-transfer/g3mc-s0"          # the seed-0 M-C retrain (latest run dir inside), teacher of arm B


def g3_root(*p) -> Path:
    d = data_dir() / "processed" / "real_transfer" / "g3"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def edit_list(rl):
    """Detector list: the current frame of CAM_F0 / L0 / R0 on every side (x+ original, x- / placebo edited where
    elicit_e2 edited them), keys '<side>:<token>|<cam>' with G0's camera names."""
    from . import navsim_qwen as NQ
    from .real_g0 import CAMS
    rows = []
    for sd in SIDES:
        q = pd.read_parquet(NQ.root(f"e2nav_{sd}", "index.parquet"))
        for tok, files in zip(q.token, q.files):
            for i, cam in enumerate(CAMS):                  # files: CAM_F0 / L0 / R0 x 4 frames, oldest first
                assert ("CAM_F0", "CAM_L0", "CAM_R0")[i] in files[4 * i + 3], files[4 * i + 3]
                rows.append({"key": f"{sd}:{tok}|{cam}", "path": files[4 * i + 3], "shard": "", "off": 0, "len": 0})
    li = pd.DataFrame(rows)
    li.to_parquet(g3_root("edit_images.parquet"), index=False)
    rl.info(f"{len(li)} images: " + str(li.key.str.split(":").str[0].value_counts().to_dict()))


def _embed(d: pd.DataFrame, n: int, cal: dict, keys: pd.DataFrame, p1: np.ndarray, x_off: np.ndarray) -> np.ndarray:
    """real_g0.embed_set's computation for detections d (columns of fusion_q4.load_dets + fi, cam) of n frames."""
    from multiprocessing import Pool
    from . import fusion_q4 as Q
    from .common import n_cpus
    from .real_g0 import CAMS, CLS3, K_DET, _embed_frames, arc_path, p5_f_over_h
    ck = np.empty(len(d), object)
    for cam in CAMS:
        m = (d.cam == cam).to_numpy()
        ck[m] = keys[cam].to_numpy()[d.fi.to_numpy()[m]]
    d = Q.lift_dets(d, ck, cal)
    fv = np.array([cal[k]["intrinsic"][1] for k in ck])
    arr = np.c_[d.prompt.map({c: i for i, c in enumerate(CLS3)}).to_numpy(), np.zeros((len(d), 2)),
                d.gx.to_numpy() + x_off[d.fi.to_numpy()], d.gy.to_numpy(), (d.y1 - d.y0).to_numpy() / fv * p5_f_over_h(),
                d.score.to_numpy()]
    rows_ok = np.flatnonzero(d.lift_ok.to_numpy())
    by = pd.Series(rows_ok).groupby(d.fi.to_numpy()[rows_ok]).indices
    jobs = [[(i, arc_path(p1[i], x_off[i]), arr[rows_ok[by[i]]] if i in by else np.zeros((0, 7)),
              rows_ok[by[i]] if i in by else np.zeros(0, np.int64)) for i in range(a, min(a + 256, n))]
            for a in range(0, n, 256)]
    E = np.zeros((n, K_DET * 8), np.float32)
    with Pool(min(48, n_cpus())) as p:
        for part in p.imap_unordered(_embed_frames, jobs):
            for i, e, *_ in part:
                E[i] = e
    return E


def edit_embed(rl):
    from . import fusion_q4 as Q
    from .real_g0 import CLS3, SCORE, geometry
    d = Q.load_dets(g3_root("dets"), SCORE)
    d = d[d.prompt.isin(CLS3)].reset_index(drop=True)
    li = pd.read_parquet(g3_root("edit_images.parquet"))
    k = li.key.str.split(":", n=1)
    toks = pd.unique(k.str[1].str.split("|").str[0])
    fr = pd.DataFrame({"frame_id": toks})
    cal, keys, p1, x_off = geometry("navtrain", fr)
    at = pd.Series(np.arange(len(fr)), index=toks)
    kd = d.key.str.split(":", n=1)
    d["side"], d["frame_id"], d["cam"] = kd.str[0], kd.str[1].str.split("|").str[0], kd.str[1].str.split("|").str[1]
    for sd in SIDES:
        ds = d[d.side == sd].reset_index(drop=True)
        ds["fi"] = at[ds.frame_id].to_numpy()
        E = _embed(ds, len(fr), cal, keys, p1, x_off)
        mine = pd.unique(k.str[1][k.str[0] == sd].str.split("|").str[0])
        E = E[at[mine].to_numpy()]
        np.savez(g3_root(f"edit_embed_{sd}.npz"), tokens=mine, embed=E)
        m = E.reshape(len(E), 8, 8)
        rl.info(f"{sd}: {len(E)} tokens, rows with any {float((m[:, :, 7].sum(1) > 0).mean()):.3f}, "
                f"mean in corridor {float(m[:, :, 7].sum(1).mean()):.2f}, rows with ped {float((m[:, :, 0].sum(1) > 0).mean()):.3f}")


def p5_embed(rl):
    """G0's real-data embedding on the P5 v1 BA index rows: E5's detections, the P5 rig, the ego-history arc."""
    from . import fusion_q4 as Q, p5_exam as E, waymo as W
    from .real_g0 import CAMS, CLS3, SCORE
    assert os.environ.get("P5_SET") == "carla_p5v1_ba"
    t, past, *_ = E.load()
    wp, _ = W.load_ego()
    assert past.shape[1:] == wp.shape[1:], (past.shape, wp.shape)      # the WOD past layout, 0.25 s steps
    subs = sorted(p for p in (data_dir() / "processed/elicit_e5/dets").iterdir() if p.is_dir())
    d = pd.concat([Q.load_dets(s_, SCORE) for s_ in subs], ignore_index=True)
    d = d[d.prompt.isin(CLS3)].reset_index(drop=True)
    kd = d.key.str.split("|")
    d["cam"] = kd.str[1]
    d["fi"] = pd.Series(np.arange(len(t)), index=t.frame_name)[kd.str[0]].to_numpy()
    cal = Q.p5_calib()
    keys = pd.DataFrame({cam: [cam] * len(t) for cam in CAMS})
    p1 = past[:, -5, :2].astype(np.float64)                             # 1 s ago (real_g0.ARC_T), like WOD
    Em = _embed(d, len(t), cal, keys, p1, np.full(len(t), Q.REAR_AXLE_X))
    np.save(g3_root("p5_embed.npy"), Em)
    m = Em.reshape(len(Em), 8, 8)
    e5 = np.load(data_dir() / "processed/elicit_e5/embed.npy").reshape(len(Em), 8, 8)
    rl.info(f"P5 rows {len(Em)}: rows with any {float((m[:, :, 7].sum(1) > 0).mean()):.3f} (E5 route corridor "
            f"{float((e5[:, :, 7].sum(1) > 0).mean()):.3f}), with ped {float((m[:, :, 0].sum(1) > 0).mean()):.3f} "
            f"(E5 {float((e5[:, :, 0].sum(1) > 0).mean()):.3f})")


def _mlp32(d_in: int, seed: int):
    """elicit_e5._mlp with the navtrain 16-point output (32)."""
    import torch
    from .elicit_e5 import HIDDEN
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, HIDDEN),
                              torch.nn.GELU(), torch.nn.Linear(HIDDEN, 32))
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    return net.cuda()


def _train(X, ip, im, R, tr, grp, teacher, rows_t, seed: int, rl, tag: str):
    """elicit_e5.train_student with the 32-d output and the teacher term on the given rows (pair sides only)."""
    import copy
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    from .elicit_e5 import EVAL_EVERY, LR, MAX_STEPS, PATIENCE, TEACHER_W, WD
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(ip, groups=grp))
    net = _mlp32(X.shape[1], seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    if teacher is not None:
        rows_t = np.intersect1d(rows_t, np.r_[ip[a], im[a]])
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
    rl.event("g3_student_fit", tag=tag, seed=seed, best_step=best_step, stop_step=step, holdout_mse=best)
    rl.tb.add_scalar(f"holdout_mse/{tag}", best, seed) if rl.tb else None
    return net


def student(rl):
    import glob
    import pickle
    import torch
    from . import elicit_e1 as E1, elicit_e2_train as E2T, navsim_heads as NH, p5_exam as E, p5_openpilot
    from . import reactivity_mc as MC
    from .real_g0 import load_embed
    assert os.environ.get("P5_SET") == "carla_p5v1_ba"
    d = E2T.load_data(rl)
    assert "c" in d["dy"], "label (c) missing: run scripts/real_g3_pdm.sh first"
    toks, pl, lg = d["tokens"], d["pl_tokens"], d["log"]
    ipl = np.array([list(toks).index(x) for x in pl])
    emb = {}
    for sd, tk in (("plus", toks), ("minus", toks), ("placebo", pl)):
        z = np.load(g3_root(f"edit_embed_{sd}.npz"), allow_pickle=True)
        emb[sd] = z["embed"][pd.Series(np.arange(len(z["tokens"])), index=z["tokens"])[tk].to_numpy()]
    mc_run = sorted(glob.glob(str(data_dir() / MC_G3_RUN / "*/heads.pkl")))[-1]
    teach_heads = pickle.load(open(mc_run, "rb"))
    rl.info(f"teacher heads: {mc_run}")
    # mu rows: E2's prior training rows of navtrain (stage one, complete future) with G0's embedding
    tr = NH.load("navtrain", True)
    keep = (tr["stage"] == "one") & ~np.isnan(tr["fut"]).any((1, 2))
    fr_nav, emb_nav = load_embed("navtrain")
    at = pd.Series(np.arange(len(fr_nav)), index=fr_nav.frame_id)
    has = keep & np.isin(tr["tokens"], fr_nav.frame_id.to_numpy())
    rl.info(f"mu rows: {int(has.sum())} of {int(keep.sum())} navtrain training rows have G0's embedding")
    Emu = emb_nav[at[tr["tokens"][has]].to_numpy()]
    # foreign rows: P5 v1 BA (R2) and WOD val (R3)
    t, past, fut, obs, null, pairs = E.load()
    ref = np.load(data_dir() / E1.MC_RUN / "preds_obs.npz")
    rows = ref["rows"]
    Ep5 = np.load(g3_root("p5_embed.npy"))[rows]
    w = E1.wod_frames()
    fr_w, emb_w = load_embed("wod_val")
    aw = pd.Series(np.arange(len(fr_w)), index=fr_w.frame_id)
    Ew = emb_w[aw[w["frame_name"]].to_numpy()]
    mask_col = np.arange(64) % 8 == 7
    n = len(t)
    preds, r1_rows, nets_meta = {}, {}, []
    for m in MODELS:
        Opmu = tr[m][has]
        mo, so = Opmu.mean(0), np.where(Opmu.std(0) > 1e-6, Opmu.std(0), 1.0)
        me, se = Emu.mean(0), np.where(Emu.std(0) > 1e-6, Emu.std(0), 1.0)
        me[mask_col], se[mask_col] = 0.0, 1.0

        def z(op, e):
            return np.concatenate([(op - mo) / so / np.sqrt(op.shape[1]), (e - me) / se / np.sqrt(e.shape[1])], 1).astype(np.float32)
        Xs = [z(Opmu, Emu)] + [z(d[f"op {m} {sd}"], emb[sd]) for sd in SIDES]
        off = np.cumsum([0] + [len(x) for x in Xs])
        X = torch.as_tensor(np.concatenate(Xs), device="cuda")
        i_mu, i_p, i_m, i_pl = (np.arange(off[k], off[k + 1]) for k in range(4))
        ip, im = np.r_[i_p, i_p[ipl]], np.r_[i_m, i_pl]
        grp = np.r_[lg, lg[ipl]]
        Xp5 = torch.as_tensor(z(p5_openpilot.load(t, (m,), sub="op_streams_vis")[f"op-{m} temporal"][rows], Ep5), device="cuda")
        Xw = torch.as_tensor(z(w[f"op {m}"], Ew), device="cuda")
        pp, pm, ppl = d[f"prior {m} plus"], d[f"prior {m} minus"], d[f"prior {m} placebo"]
        R0 = (-(pp[ipl] - ppl)).reshape(len(ipl), -1)
        preds[f"prior [{m}]"] = np.full((n, 20, 2), np.nan, np.float32)
        preds[f"prior [{m}]"][rows] = ref[f"prior [{m}]"]
        for lab in ("c", "a"):
            R = torch.as_tensor(np.concatenate([(d["dy"][lab] - (pp - pm)).reshape(len(toks), -1), R0]), dtype=torch.float32, device="cuda")
            h = teach_heads[m][f"E2 pair ({lab})"]
            tch = np.zeros((len(X), 32), np.float32)
            for sd, ix in (("plus", i_p), ("minus", i_m), ("placebo", i_pl)):
                tch[ix] = E2T.apply(h, {s_: d[f"{s_} {sd}"] for s_ in h["streams"]})[:, :16].reshape(len(ix), -1)
            tch = torch.as_tensor(tch, device="cuda")
            for arm in ("A", "B"):
                for seed in (0, 1, 2):
                    tag = f"G3 student {arm} ({lab}) s{seed}"
                    net = _train(X, ip, im, R, i_mu, grp, tch if arm == "B" else None, np.r_[i_p, i_m, i_pl], seed, rl, f"{m} {tag}")
                    with torch.no_grad():
                        f16 = lambda Z_: E2T.extend20(net(Z_).reshape(-1, 16, 2).cpu().numpy())  # noqa: E731
                        r1_rows[f"{tag} [{m}]"] = [f16(X[ix]) for ix in (i_p, i_m, i_pl)]
                        v = np.full((n, 20, 2), np.nan, np.float32)
                        v[rows] = ref[f"prior [{m}]"] + f16(Xp5)
                        preds[f"{tag} [{m}]"] = v
                        preds_w = f16(Xw)
                    nets_meta.append((m, tag, preds_w))
        del X, Xp5, Xw
        torch.cuda.empty_cache()
    t1 = E2T.r1(d, r1_rows)
    t1.to_csv(rl.dir / "r1_magnitude.csv", index=False)
    rl.info("R1\n" + t1[t1.pairs == "all"].to_markdown(index=False, floatfmt=".3f"))
    oo, nn = E.deltas(obs, null, t, preds)
    res = E.exam(oo, nn, pairs, list(preds))
    crit = pd.concat([MC.criteria(res, [k for k in preds if k.endswith(f"[{m}]")], f"prior [{m}]") for m in MODELS])
    res["flips"].to_csv(rl.dir / "p5_flip_rates.csv", index=False)
    crit.to_csv(rl.dir / "p5_criteria.csv", index=False)
    rl.info("R2 P5 v1 BA\n" + crit.to_markdown(index=False, floatfmt=".3f"))
    tabs, acts = [], []
    for m, tag, delta in nets_meta:
        tau = res["taus"][f"{tag} [{m}]"]
        tab, act = E1.readouts(w, w[f"prior {m}"], delta, tau)
        tabs.append(tab.assign(model=m, arm=tag))
        acts.append(act.assign(model=m, arm=tag, tau=tau))
    pd.concat(tabs).to_csv(rl.dir / "wod_deltas.csv", index=False)
    pd.concat(acts).to_csv(rl.dir / "wod_activation.csv", index=False)
    rl.info("R3 WOD done")


# ---------------------------------------------------------------- figures (pulled tables -> research/figs)

def figs(rl=None, res_dir="research/results/real-data-transfer/g3", out_dir="research/figs"):
    """G3a: median feature shift (edit vs null / placebo) and their ratio, CARLA pedestrian pairs vs E2 navtrain."""
    import matplotlib.pyplot as plt
    from . import plots
    s = pd.read_csv(Path(res_dir) / "g3a_side_by_side.csv")
    names = {"Qwen L18_last": "Qwen $L18$", "openpilot cinque temporal": "op Cinque", "openpilot lebowski temporal": "op Lebowski",
             "YOLO embedding (E5)": "YOLO emb."}
    col = {"carla": plots.OKABE_ITO[5], "e2": plots.OKABE_ITO[6]}
    x = np.arange(len(s))
    with plots.mpl.rc_context(plots.STYLE):
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(plots.PAGE, 2.0), gridspec_kw={"width_ratios": [1.25, 1]})
        w = 0.19
        for j, (c, dom, lab) in enumerate((("carla_median_edit", "carla", "CARLA: remove hazard"),
                                           ("carla_median_null", "carla", "CARLA: weather null"),
                                           ("e2_median_edit", "e2", "E2 navtrain: erase pedestrian"),
                                           ("e2_median_placebo", "e2", "E2 navtrain: placebo patch"))):
            filled = "edit" in c
            a0.bar(x + (j - 1.5) * w, s[c], w, color=col[dom] if filled else "white", edgecolor=col[dom], lw=0.7,
                   hatch=None if filled else "////", label=lab)
        a0.set_yscale("log")
        a0.set_xticks(x, [names[f] for f in s.feature])
        a0.set_ylabel("Median shift (RMS, $x^+$ std units)")
        a0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6)
        for k, (dom, lab) in enumerate((("carla", "CARLA pedestrian pairs / weather null"), ("e2", "E2 navtrain edits / placebo"))):
            v, lo, hi = s[f"{dom}_ratio"], s[f"{dom}_lo"], s[f"{dom}_hi"]
            ok = v.notna().to_numpy()
            a1.errorbar(x[ok] + (k - 0.5) * 0.25, v[ok], yerr=[(v - lo)[ok], (hi - v)[ok]], fmt="o", color=col[dom], ms=3,
                        lw=0.8, capsize=1.5, label=lab)
        for yv, ls in ((2, "--"), (3, ":")):
            a1.axhline(yv, color="0.4", ls=ls, lw=0.6)
        a1.set_yscale("log")
        a1.set_xticks(x, [names[f] for f in s.feature])
        a1.set_ylabel("Shift ratio, edit / null")
        a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=6)
        plots.save(fig, Path(out_dir), "real-g3a-shift")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("g3a", "g3c", "pdm-prep", "pdm-read", "mc", "edit-list", "edit-embed", "p5-embed", "student", "figs", "g3a-yolo-e2"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.step == "figs":
        return figs()
    name = a.step.replace("-", "") if a.step.startswith("g3") else "g3" + a.step.replace("-", "") + (f"-s{a.args[0]}" if a.step == "mc" else "")
    rl = RunLog("real-data-transfer", name)
    {"g3a": g3a, "g3c": g3c, "pdm-prep": pdm_prep, "pdm-read": pdm_read, "mc": mc, "edit-list": edit_list,
     "edit-embed": edit_embed, "p5-embed": p5_embed, "student": student, "g3a-yolo-e2": g3a_yolo_e2}[a.step](rl, *a.args)
    rl.close()


if __name__ == "__main__":
    main()
