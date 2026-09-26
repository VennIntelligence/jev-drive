"""nq4 P3: real-appearance pedestrian exam, feasibility (todos/2026-09-26-night-queue-4.md, P section, [P3] entries).

  scenes   runs/nq4/p3/scenes.json and one target json per scene from the registered selection (nq4_p3_select)
  index    processed/nq4_p3: index.parquet, past.npy, future.npy and op_plan.json over the rendered worlds
           (real / plus / minus, one 5 Hz stream each), shaped like processed/hugsim_pairs (elicit_i3.op_prepare)
  exam     openpilot `ridge_late` (Cinque, Lebowski) fitted on the P5 v1 BehaviorAgent set exactly as for the I3 exam
           (reactivity_mc.fit_fold prior, 5 route folds, verified against the stored run, mean of the folds), P3 rows
           riding along; null false flips at I3's tau (fixed, not refitted): |v2(plus) - v2(real)| >= tau
  report   per-scene table, pooled gate readout, comparison figures -> research/results/nq4/p3/, research/figs/
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
SET = "nq4_p3"
WORLDS = ("real", "plus", "minus")
CAMS = ("front", "front_left", "front_right")
I3_EXAM = "runs/elicitation/i3-exam/20260926-012841"
GATE = 0.07
HZ = 10


def run_dir(*p) -> Path:
    d = data_dir() / "runs/nq4/p3" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def root(*p) -> Path:
    d = data_dir() / "processed" / SET / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def scenes():
    s = pd.read_csv(run_dir("select") / "selection_wod.csv")
    s = s[s.picked].reset_index(drop=True)
    tg = []
    for k, r in s.iterrows():
        d = {"scene": k, "key": f"p3_{k:03d}", "segment": r["name"], "split": r.split, "f0": int(r.f0), "t0": r.t0,
             "v0": r.v0, "target_track": r.target_track, "delete_tracks": json.loads(r.delete_tracks)}
        (run_dir("targets") / f"{k:03d}.json").write_text(json.dumps(d, indent=1))
        tg.append(d)
    (run_dir() / "scenes.json").write_text(json.dumps({"segments": s.name.tolist(), "targets": tg}, indent=1))
    log.info("scenes: %d", len(tg))


# ---------------------------------------------------------------- ego rows
def _poses(sd: Path) -> np.ndarray:
    fs = sorted((sd / "ego_pose").glob("*.txt"))
    return np.stack([np.loadtxt(f) for f in fs])


def _interp(P: np.ndarray, t: np.ndarray):
    """Rear-axle xy and yaw at times t (s) from 10 Hz ego->world poses; clamped to the log."""
    tt = np.arange(len(P)) / HZ
    tc = np.clip(t, 0, tt[-1])
    x, y = np.interp(tc, tt, P[:, 0, 3]), np.interp(tc, tt, P[:, 1, 3])
    yaw = np.unwrap(np.arctan2(P[:, 1, 0], P[:, 0, 0]))
    return np.stack([x, y], -1), np.interp(tc, tt, yaw)


def ego_rows(P: np.ndarray, t: float):
    """P4 / WOD-E2E-shaped rows at t (rear axle origin, x forward, y left): past (16, 6) at 0.25 s
    [x, y, vx, vy, dvx, dvy] and future (20, 2); the same construction as hugsim_pairs.past_future."""
    dur = (len(P) - 1) / HZ
    tp = t + 0.25 * np.arange(-15, 1)
    xy0, th0 = _interp(P, np.array([t]))
    R = np.array([[np.cos(th0[0]), np.sin(th0[0])], [-np.sin(th0[0]), np.cos(th0[0])]])
    loc = lambda q: (q - xy0) @ R.T  # noqa: E731
    p = loc(_interp(P, tp)[0])
    h = 0.1
    a, b = np.minimum(np.maximum(tp, 0) + h, dur), np.maximum(np.maximum(tp, 0) - h, 0)
    vel = (_interp(P, a)[0] - _interp(P, b)[0]) / (a - b)[:, None]
    vv = vel @ R.T
    aa = vv - np.r_[vv[:1], vv[:-1]]
    pad = tp < 0
    vv[pad], aa[pad] = 0.0, 0.0
    vv[-2], aa[-2] = vv[-1], aa[-1]
    tf = t + 0.25 * np.arange(1, 21)
    fut = loc(_interp(P, tf)[0])
    fut[tf > dur + 1e-6] = np.nan
    return np.concatenate([p, vv, aa], -1).astype(np.float32), fut.astype(np.float32)


def intent(P: np.ndarray, t: float) -> int:
    """WOD intent from the logged heading change over the next 5 s: 2 left, 3 right (|dyaw| > 25 deg), else 1 straight."""
    _, th = _interp(P, np.array([t, t + 5.0]))
    d = np.degrees(th[1] - th[0])
    return 2 if d > 25 else 3 if d < -25 else 1


def calib_record(K, ext, wh) -> dict:
    return {"intrinsic": [K[0][0], K[1][1], K[0][2], K[1][2], 0.0, 0.0, 0.0, 0.0, 0.0],
            "extrinsic": np.asarray(ext, np.float64).ravel().tolist(), "width": wh[0], "height": wh[1]}


def index(processed_root: str):
    rows, past, fut, calibs, streams = [], [], [], {}, []
    for sd in sorted(root("scenes").glob("p3_*")):
        if not (sd / "meta.json").exists():
            continue
        meta = json.loads((sd / "meta.json").read_text())
        key = sd.name
        P = _poses(Path(processed_root) / f"{meta['scene']:03d}")
        calibs[key] = {str(i + 1): calib_record(meta["calib"][c]["K"], meta["extrinsics_cam_to_ego"][c], meta["calib"][c]["wh"])
                       for i, c in enumerate(CAMS)}
        for w in WORLDS:
            fr = [json.loads(x) for x in open(sd / w / "frames.jsonl")]
            names = [f"{key}-{w}-{r['frame']:07d}" for r in fr]
            streams.append({"key": f"{key}-{w}", "seq": key, "names": names, "targets": list(range(3, len(fr))),
                            "files": [[str(sd / w / r["files"][c]) for c in CAMS] for r in fr],
                            "gaps": int((np.diff([r["frame"] for r in fr]) != 4).sum())})
            for j, r in enumerate(fr):
                if j < 3:
                    continue
                pa, fu = ego_rows(P, r["t"])
                past.append(pa), fut.append(fu)
                rows.append({"frame_name": names[j], "route_id": f"{key}-{w}", "base_id": key, "world": w, "t": r["t"],
                             "frame": r["frame"], "t_rel_f0": round(r["t"] - meta["f0"] / HZ, 2), "intent": intent(P, r["t"]),
                             "role": "p3", "family": "ped"})
    t = pd.DataFrame(rows)
    t.to_parquet(root() / "index.parquet", index=False)
    np.save(root() / "past.npy", np.stack(past)), np.save(root() / "future.npy", np.stack(fut))
    (root() / "op_plan.json").write_text(json.dumps({"calibs": calibs, "streams": streams}))
    info = {"scenes": len(calibs), "streams": len(streams), "rows": len(t), "gaps": sum(s["gaps"] for s in streams)}
    log.info("index: %s", info)
    return info


# ---------------------------------------------------------------- exam
def exam(rl):
    import torch
    from . import elicit_i3 as I, p5_exam as E, p5_openpilot, p5_pairs as P, reactivity_mc as MC
    dev = "cuda"
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        op = p5_openpilot.load(t, I.MODELS, sub="op_streams_vis")
        fold = E.folds(t, pairs)
    with I.p5_set(SET):
        t3 = pd.read_parquet(root() / "index.parquet")
        past3 = np.load(root() / "past.npy")
        op3 = p5_openpilot.load(t3, I.MODELS, sub="op_streams")
    n, n3 = len(t), len(t3)
    ta = pd.concat([t[["frame_name", "role", "base_id", "intent"]],
                    t3[["frame_name", "base_id", "intent"]].assign(role="p3", base_id="p3:" + t3.base_id)], ignore_index=True)
    fold_a = np.r_[fold, np.full(n3, -2)]
    F = torch.as_tensor(np.r_[fut.reshape(n, -1), np.zeros((n3, 40), np.float32)], device=dev)
    Ego = torch.as_tensor(np.r_[E.ego_input(t, past), E.ego_input(t3, past3)], device=dev)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    ref = np.load(data_dir() / I.MC_RUN / "preds_obs.npz")
    at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
    I3 = np.arange(n, n + n3)
    preds, checks = {}, []
    for m in I.MODELS:
        Xop = torch.as_tensor(np.r_[op[f"op-{m} temporal"], op3[f"op-{m} temporal"]], device=dev)
        name = f"ridge_late op-{m} temporal"
        preds[name] = np.zeros((n3, 20, 2))
        for f in range(E.K_FOLDS):
            o = MC.fit_fold(f, fold_a, ta, F, Ego, Xop, None, pr_ip, pr_im, pr_group, rl, m, arms=())
            ev = obs_rows[fold[obs_rows] == f]
            d = float(np.abs(o["prior"][ev].reshape(-1, 20, 2).cpu().numpy() - ref[f"prior [{m}]"][at[ev].to_numpy()]).max())
            checks.append({"model": m, "fold": f, "max_abs_diff": d})
            assert d < 1e-3, f"{m} fold {f} prior does not reproduce the stored run ({d})"
            preds[name] += o["prior"][I3].reshape(-1, 20, 2).cpu().double().numpy() / E.K_FOLDS
        del Xop
        torch.cuda.empty_cache()
    pd.DataFrame(checks).to_csv(rl.dir / "head_checks.csv", index=False)
    np.savez_compressed(rl.dir / "preds_p3.npz", frame_name=t3.frame_name.to_numpy(), **{k: v.astype(np.float32) for k, v in preds.items()})
    taus = pd.read_csv(data_dir() / I3_EXAM / "flip_rates.csv").query("scope == 'pooled'").set_index("examinee").tau_model
    t3 = t3.assign(sfx=t3.frame_name.str.rsplit("-", n=1).str[-1], **{f"{k}|v2": P.v2(v) for k, v in preds.items()})
    out = t3.pivot_table(index=["base_id", "sfx"], columns="world", values=[f"{k}|v2" for k in preds] + ["t_rel_f0"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.rename(columns={"t_rel_f0_real": "t_rel_f0"}).reset_index()
    rows = []
    for name in preds:
        tau = float(taus[name])
        dn = out[f"{name}|v2_plus"] - out[f"{name}|v2_real"]
        dp = out[f"{name}|v2_plus"] - out[f"{name}|v2_minus"]
        out[f"{name}|null_d"], out[f"{name}|pair_d"] = dn, dp
        ffn = ((dn.abs() >= tau) & (dn.abs() > 0)).astype(float).to_numpy()
        fr, lo, hi = E.boot_ratio(ffn, np.ones(len(ffn)), out.base_id.to_numpy())
        mov = ((dp.abs() >= tau) & (dp.abs() > 0))
        rows.append({"examinee": name, "tau_i3": tau, "n_null_frames": len(ffn), "scenes": out.base_id.nunique(),
                     "null_false_flip": fr, "lo": lo, "hi": hi, "gate": GATE, "passes": bool(fr <= GATE),
                     "pair_moved": float(mov.mean()), "pair_slower": float((mov & (dp < 0)).mean()),
                     "pair_moved_after_f0": float(mov[out.t_rel_f0 >= 0].mean())})
        for s, g in out.groupby("base_id"):
            rows.append({"examinee": name, "scope": s, "tau_i3": tau, "n_null_frames": len(g),
                         "null_false_flip": float(((g[f"{name}|null_d"].abs() >= tau)).mean()),
                         "pair_moved": float((g[f"{name}|pair_d"].abs() >= tau).mean())})
    res = pd.DataFrame(rows)
    res["scope"] = res.get("scope", pd.Series(dtype=object)).fillna("pooled")
    res.to_csv(rl.dir / "null_gate.csv", index=False)
    out.to_parquet(rl.dir / "frames_scored.parquet", index=False)
    log.info("null gate:\n%s", res[res.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    return res


# ---------------------------------------------------------------- report
def report(exam_dir: str | None, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    outd, figd = Path(out), Path(out).parents[2] / "figs"
    outd.mkdir(parents=True, exist_ok=True)
    per = []
    for sd in sorted(root("scenes").glob("p3_*")):
        if not (sd / "meta.json").exists():
            continue
        meta = json.loads((sd / "meta.json").read_text())
        st = pd.read_csv(sd / "render_stats.csv")
        per.append({"scene": sd.name, "segment": meta["segment"], "f0_s": meta["f0"] / HZ, "v0": meta["v0"],
                    "n_deleted": len(meta["deleted_node_instances"]), "deleted_missing": len(meta["deleted_missing"]),
                    "psnr": st.psnr.mean(), "psnr_ped": st.psnr_ped.mean(), "determinism_max_abs": meta["determinism_max_abs"],
                    "del_diff_px_out": int(st.diff_px_out.sum()), "del_diff_px_in": int(st.diff_px_in.sum()),
                    "train_min": meta.get("train_min"), "render_s": meta["render_s"]})
        # comparison figure: front camera at f0 - 1 s, f0, f0 + 1 s; real / plus / minus / |plus - minus|
        ts = [meta["f0"] - 10, meta["f0"], meta["f0"] + 10]
        ts = [min(meta["frames"], key=lambda x: abs(x - q)) for q in ts]
        fig, ax = plt.subplots(3, 4, figsize=(16, 7.6))
        for i, tt in enumerate(ts):
            im = {w: np.asarray(Image.open(sd / w / "cams/front" / f"{2 * tt:07d}.jpg")) for w in WORLDS}
            d = np.abs(im["plus"].astype(int) - im["minus"].astype(int)).max(-1)
            for j, (lab, x) in enumerate([("real (log)", im["real"]), ("x+ (re-render)", im["plus"]),
                                          ("x- (pedestrians removed)", im["minus"]), ("|x+ - x-|", d)]):
                ax[i, j].imshow(x, cmap="magma" if j == 3 else None, vmin=0, vmax=255 if j < 3 else 64)
                ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(lab, fontsize=11)
            ax[i, 0].set_ylabel(f"t - f0 = {(tt - meta['f0']) / HZ:+.1f} s", fontsize=10)
        fig.suptitle(f"{sd.name}  (WOD {meta['segment']}), front camera", fontsize=11)
        fig.tight_layout()
        fig.savefig(figd / f"nq4-p3-{sd.name}.png", dpi=110)
        plt.close(fig)
    per = pd.DataFrame(per)
    if exam_dir:
        g = pd.read_csv(Path(exam_dir) / "null_gate.csv")
        for m in ("cinque", "lebowski"):
            s = g[(g.examinee == f"ridge_late op-{m} temporal") & (g.scope != "pooled")].set_index("scope")
            per[f"null_ff_{m}"] = per.scene.map(s.null_false_flip)
            per[f"pair_moved_{m}"] = per.scene.map(s.pair_moved)
        g[g.scope == "pooled"].to_csv(outd / "null_gate_pooled.csv", index=False)
    per.to_csv(outd / "scenes.csv", index=False)
    (outd / "scenes.md").write_text(per.to_markdown(index=False, floatfmt=".3f"))
    log.info("report:\n%s", per.to_markdown(index=False, floatfmt=".3f"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("scenes", "index", "exam", "report"))
    ap.add_argument("--processed-root", default=str(data_dir() / "processed/waymo_ds/training"))
    ap.add_argument("--exam-dir")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "research/results/nq4/p3"))
    a = ap.parse_args()
    if a.cmd == "scenes":
        scenes()
    elif a.cmd == "index":
        print(json.dumps(index(a.processed_root)))
    elif a.cmd == "exam":
        from .runlog import RunLog
        rl = RunLog("nq4", "p3-exam")
        exam(rl)
        print(rl.dir)
        rl.close()
    else:
        report(a.exam_dir, a.out)


if __name__ == "__main__":
    main()
