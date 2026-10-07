"""op_parity on WOD-E2E val: P2 / P2H checkpoints against shipped Cinque under the unified WOD interface (results/wod_p2h.md).

  bias    (op-train env, CPU) WOD past states + intent -> lib/parity_adapter.ego_features -> the arm's adapter -> (32, 512) fp16 `intent_bias` per
          target frame of the rater + extra sets -> $DATA_DIR/runs/op_parity/wod/bias-<tag>.npz (scripts/wod_zeroshot_openpilot.py --bias).
          `--tags P0` writes zeros (shipped weights through the bias-input ONNX: the equivalence check against the stored shipped run).
  report  (jevdrive env, CPU) preds/op_cinque_<tag> vs preds/op_cinque (shipped): RFS (479 rater frames), ADE@3s / @5s (1 437 frames with futures),
          night / day split by the night_gap luma labels, paired bootstraps over sequences -> results/wod_p2h.{md,csv}

Input mapping (WOD -> NAVSIM AgentInput as pp_prep builds it; WOD ego frame = rear axle at t0, +x forward, +y left, as NAVSIM):
  command  intent GO_LEFT / GO_STRAIGHT / GO_RIGHT -> one-hot [left, straight, right]; UNKNOWN -> all zero (never occurs on val)
  poses    the 4 history keys -1.5 / -1.0 / -0.5 / 0 s are past_states steps 9 / 11 / 13 / 15 (4 Hz lattice, exact): x, y as given; yaw of
           key k = chord heading of the positions k-1 -> k+1 (WOD stores no yaw), held from the next key when the chord is < 0.1 m; the
           t0 yaw is 0 by construction
  vx       speed at t0 from the positions (jevdrive.waymo.past_kinematics "v"); vy = 0
  ax, ay   the given accel_x / accel_y of the last past state
  clock    1.0: WOD frames are real time (no dilation), the model's 0.2 s context steps are real 0.2 s
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

KEYS = (9, 11, 13, 15)
DT = 0.25
B = 4000


def wod_ego(past, intent):
    """(n, 16, 6) past states, (n,) intent index (0 UNKNOWN, 1 STRAIGHT, 2 LEFT, 3 RIGHT) -> (n, 20) parity ego features, raw pose (n, 4, 3)."""
    import parity_adapter as PA
    from jevdrive import waymo as W
    p = past.astype(np.float64)
    n = len(p)
    pose = np.zeros((n, 4, 3))
    pose[:, :, :2] = p[:, list(KEYS), :2]
    nxt = np.zeros(n)                                              # yaw of the later key, held when the chord is too short
    for j in (2, 1, 0):
        k = KEYS[j]
        ch = p[:, k + 1, :2] - p[:, k - 1, :2]
        ok = np.linalg.norm(ch, axis=-1) >= 0.1
        nxt = np.where(ok, np.arctan2(ch[:, 1], ch[:, 0]), nxt)
        pose[:, j, 2] = nxt
    kin = W.past_kinematics(past)
    vel = np.zeros((n, 4, 2))
    vel[:, :, 0] = kin["v"][:, None]
    acc = np.tile(p[:, -1, 4:6][:, None], (1, 4, 1))
    cmd = np.zeros((n, 4))
    for i, c in ((1, 1), (2, 0), (3, 2)):                          # WOD intent -> NAVSIM [left, straight, right, unknown] index
        cmd[intent == i, c] = 1.0
    return PA.ego_features(pose, vel, acc, cmd), pose


def cmd_bias(a):
    import torch
    from jevdrive import wod_zeroshot as Z
    from jevdrive.common import data_dir
    S = Z.load_sets()
    names = np.concatenate([S[k]["name"].astype(str) for k in ("rater", "extra")])
    past = np.concatenate([S[k]["past"] for k in ("rater", "extra")])
    intent = np.concatenate([S[k]["intent"] for k in ("rater", "extra")])
    ego, pose = wod_ego(past, intent)
    out = data_dir() / "runs/op_parity/wod"
    out.mkdir(parents=True, exist_ok=True)
    import pp_hugsim as H
    for tag0 in a.tags:
        tag, _, var = tag0.partition(":")                          # variants (diagnostics): ":nobias" zero bias, ":nocmd" command one-hot zeroed
        e = ego.copy()
        if var == "nocmd":
            e[:, 1:4] = 0
        if H.is_shipped(tag) or var == "nobias":
            bias = np.zeros((len(names), 32, 512), np.float16)
        else:
            m = H.pmodel(tag, torch.device("cpu"))
            assert m.adapter is not None and not m.adapter.use_side, tag
            with torch.no_grad():
                bias = np.concatenate([m.adapter(torch.from_numpy(e[i:i + 256]), None, None).to(torch.float16).numpy()
                                       for i in range(0, len(ego), 256)])
        rms = float(np.sqrt(np.mean(bias.astype(np.float32) ** 2)))
        np.savez(out / f"bias-{tag0.replace(':', '_')}.npz", names=names, bias=bias, ego=e)
        print(f"{tag0}: {len(names)} frames, bias rms {rms:.4f}, |max| {float(np.abs(bias).max()):.3f}", flush=True)
    st = {"n": len(names), "intent_counts": np.bincount(intent, minlength=4).tolist(), "ego_mean": ego.mean(0).round(3).tolist(),
          "ego_std": ego.std(0).round(3).tolist(), "yaw_abs_deg_p50_p99": [float(np.degrees(np.percentile(np.abs(pose[:, :3, 2]), q))) for q in (50, 99)]}
    (out / "ego_stats.json").write_text(json.dumps(st))
    print(st)


# ---------------------------------------------------------------- report
def frames():
    """Per-frame table of the two sets: names, sequence, cluster, intent, speed, future, past (rater first)."""
    from jevdrive import wod_zeroshot as Z
    S = Z.load_sets()
    return S


def load_preds(tag, names):
    from jevdrive import wod_zeroshot as Z
    d = Z.root("preds", "op_cinque" if tag == "shipped" else f"op_cinque_{tag}")
    return np.stack([np.load(d / f"{n}.npz")["wod"] for n in names]).astype(np.float64)[..., :2]


def cmd_report(a):
    import pandas as pd
    from jevdrive import waymo as W
    S = frames()
    r, x = S["rater"], S["extra"]
    seq_all = np.concatenate([r["sequence"], x["sequence"]]).astype(str)
    names_all = np.concatenate([r["name"], x["name"]]).astype(str)
    fut = np.concatenate([r["future"], x["future"]])[..., :2].astype(np.float64)
    cl = r["cluster"].astype(str)
    lum = pd.read_csv(_R / "experiments/leaderboard_audit/results/night_gap/seq_lum.csv").set_index("sequence").l
    lab_all = np.where(lum.reindex(seq_all).to_numpy() < 50, "night", np.where(lum.reindex(seq_all).to_numpy() < 120, "dusk", "day"))
    nr = len(r["name"])
    lab_r = lab_all[:nr]
    tags = ["shipped"] + a.tags
    P = {t: load_preds(t, names_all) for t in tags}
    rfs = {t: np.asarray(W.rater_feedback_score(P[t][:nr], r["traj"].astype(np.float64), r["scores"].astype(np.float64), W.init_speed(r["past"])), float)
           for t in tags}
    seeds = a.tags
    P["seedmean"], rfs["seedmean"] = None, np.mean([rfs[t] for t in seeds], 0)
    err = {t: np.linalg.norm(P[t] - fut, axis=-1) for t in tags}
    ade3 = {t: err[t][:, :12].mean(1) for t in tags}
    ade5 = {t: err[t].mean(1) for t in tags}
    ade3["seedmean"], ade5["seedmean"] = (np.mean([d[t] for t in seeds], 0) for d in (ade3, ade5))
    seq_r = seq_all[:nr]
    codes_r, ur = pd.factorize(pd.Series(seq_r))
    codes_a, ua = pd.factorize(pd.Series(seq_all))
    idx_r = [np.flatnonzero(codes_r == k) for k in range(len(ur))]
    idx_a = [np.flatnonzero(codes_a == k) for k in range(len(ua))]
    rng = np.random.default_rng(0)
    draws = [rng.integers(len(ur), size=len(ur)) for _ in range(B)]                 # RFS: the rater sequences
    draws_a = [rng.integers(len(ua), size=len(ua)) for _ in range(B)]               # ADE: every sequence of rater + extra frames
    rfs_f = lambda f, i: W.rfs_by_cluster(f[i], cl[i])[0]                          # noqa: E731

    def boot_rfs(fa, fb, mask=None):
        out = []
        for d in draws:
            i = np.concatenate([idx_r[k] for k in d])
            if mask is not None:
                i = i[mask[i]]
            out.append(rfs_f(fa, i) - rfs_f(fb, i))
        return np.percentile(out, [2.5, 97.5])

    def boot_mean(da, db, mask=None):
        out = []
        for d in draws_a:
            i = np.concatenate([idx_a[k] for k in d])
            if mask is not None:
                i = i[mask[i]]
            out.append(da[i].mean() - db[i].mean())
        return np.percentile(out, [2.5, 97.5])

    allr = np.arange(nr)
    rows = []
    for t in tags + ["seedmean"]:
        if t == "shipped":
            continue
        row = {"arm": t}
        row["RFS"], row["RFS shipped"] = rfs_f(rfs[t], allr), rfs_f(rfs["shipped"], allr)
        d = row["RFS"] - row["RFS shipped"]
        lo, hi = boot_rfs(rfs[t], rfs["shipped"])
        row["d_RFS"], row["d_lo"], row["d_hi"] = d, lo, hi
        for nm, dd in (("ADE@3s", ade3), ("ADE@5s", ade5)):
            row[nm], row[nm + " shipped"] = dd[t].mean(), dd["shipped"].mean()
            lo, hi = boot_mean(dd[t], dd["shipped"])
            row["d_" + nm], row["d_" + nm + "_lo"], row["d_" + nm + "_hi"] = dd[t].mean() - dd["shipped"].mean(), lo, hi
        for lab in ("night", "day"):
            mr = lab_r == lab
            ma = lab_all == lab
            row[f"RFS {lab}"] = rfs_f(rfs[t], np.flatnonzero(mr))
            row[f"RFS {lab} shipped"] = rfs_f(rfs["shipped"], np.flatnonzero(mr))
            lo, hi = boot_rfs(rfs[t], rfs["shipped"], mr)
            row[f"d_RFS {lab}"], row[f"d_RFS {lab}_lo"], row[f"d_RFS {lab}_hi"] = row[f"RFS {lab}"] - row[f"RFS {lab} shipped"], lo, hi
            for nm, dd in (("ADE@3s", ade3), ("ADE@5s", ade5)):
                lo, hi = boot_mean(dd[t], dd["shipped"], ma)
                row[f"{nm} {lab}"], row[f"{nm} {lab} shipped"] = dd[t][ma].mean(), dd["shipped"][ma].mean()
                row[f"d_{nm} {lab}"], row[f"d_{nm} {lab}_lo"], row[f"d_{nm} {lab}_hi"] = dd[t][ma].mean() - dd["shipped"][ma].mean(), lo, hi
        rows.append(row)
    # seed vs seed noise reference: the two seeds against each other
    if len(seeds) == 2:
        lo, hi = boot_rfs(rfs[seeds[0]], rfs[seeds[1]])
        rows.append({"arm": f"{seeds[0]} - {seeds[1]}", "d_RFS": rfs_f(rfs[seeds[0]], allr) - rfs_f(rfs[seeds[1]], allr), "d_lo": lo, "d_hi": hi})
    # plan speed: 5 s displacement of the plan vs the log, moving frames
    sp = {}
    lg = np.linalg.norm(fut[:, -1], axis=-1)
    mv = lg > 2.0
    for t in tags:
        sp[t] = float(np.median(np.linalg.norm(P[t][:, -1], axis=-1)[mv] / lg[mv]))
    meta = {"n_rater": int(nr), "n_futures": int(len(names_all)), "n_night_rater": int((lab_r == "night").sum()), "n_day_rater": int((lab_r == "day").sum()),
            "n_dusk_rater": int((lab_r == "dusk").sum()), "n_night_all": int((lab_all == "night").sum()), "n_day_all": int((lab_all == "day").sum()),
            "plan_5s_displacement_over_log_median": sp, "B": B}
    out = _R / "experiments/op_parity/results"
    pd.DataFrame(rows).to_csv(out / "wod_p2h.csv", index=False)
    (out / "wod_p2h_meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 99)
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(meta, indent=1))
    # per-frame table for later reading
    pd.DataFrame({"name": names_all, "seq": seq_all, "lab": lab_all, **{f"ade3_{t}": ade3[t] for t in tags}, **{f"ade5_{t}": ade5[t] for t in tags},
                  **{f"rfs_{t}": np.concatenate([rfs[t], np.full(len(names_all) - nr, np.nan)]) for t in tags}}).to_csv(
        out / "wod_p2h_frames.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    p = sp_.add_parser("bias")
    p.add_argument("--tags", nargs="+", required=True)
    p = sp_.add_parser("report")
    p.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    {"bias": cmd_bias, "report": cmd_report}[a.cmd](a)
