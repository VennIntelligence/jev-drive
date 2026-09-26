"""Top-10 intersection, executor T2: DrivoR and WA-JEPA on our exams (todos/2026-09-26-top10-intersection.md, 5.1-5.5
and the [T2] entries under "执行分派", written before the numbers they affect).

  req-i3   I3 requests from the 10 Hz + CAM_BACK re-render (processed/hugsim_pairs_10hz), one file per model
  req-p5   P5 v1 BehaviorAgent requests (three front cameras, black rear camera), one file per model
  exam-i3  p5_exam.exam unchanged (no TFv6 columns), tau from I3's null scenes, scene bootstrap; plus WA-JEPA's
           complete-history sensitivity
  exam-p5  per frame (p5_exam.exam, reproduced through elicit_e4.score) and per pair (E4 (b)), with the stored
           reference examinees of the official BA runs as context rows

Requests (npz): keys, img (n, 16) paths time-major t-1.5 ... t x [l0, f0, r0, b0] ('' = black), hist (n, 4, 3) poses of
the four history frames relative to the current one (x fwd, y left, yaw ccw), ego (n, 4) = vx, vy, ax, ay, cmd (n,)
NAVSIM order. Runners: scripts/top10_t2/{drivor,wajepa}_run.py in the models' own envs. Predictions come back as
(n, 8, 3) rear-axle poses at 0.5 s and go onto the exams' 0.25 s grid by a cubic spline through (0, 0) + 8 points.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("drivor", "wajepa")
NAME = {"drivor": "DrivoR", "wajepa": "WA-JEPA"}
NAV_CMD = {0: 3, 1: 1, 2: 0, 3: 2}      # WOD intent (UNKNOWN, STRAIGHT, LEFT, RIGHT) -> NAVSIM [left, straight, right, unknown]
F0_REAR = np.array([1.655, -0.012])      # nuPlan CAM_F0 in the rear-axle frame (median of 500 navtest tokens' sensor2lidar t)
I3_SRC, I3_10 = "hugsim_pairs", "hugsim_pairs_10hz"
HIST_S = (-1.5, -1.0, -0.5, 0.0)


def root(*p) -> Path:
    d = data_dir() / "runs" / "top10_t2"
    (d / Path(*p)).parent.mkdir(parents=True, exist_ok=True)
    return d / Path(*p)


def save_req(name: str, keys, img, hist, ego, cmd) -> Path:
    p = root("requests", f"{name}.npz")
    np.savez(p, keys=np.asarray(keys).astype(str), img=np.asarray(img).astype(str), hist=np.asarray(hist, np.float32),
             ego=np.asarray(ego, np.float32), cmd=np.asarray(cmd, np.int64))
    log.info("%s: %d requests -> %s", name, len(keys), p)
    return p


def ego_rows(past_t0: np.ndarray, model: str) -> np.ndarray:
    """[vx, vy, ax, ay] from a WOD-shaped past row at t0 ([x, y, vx, vy, dvx, dvy], dv per 0.25 s). WA-JEPA on I3 / P5
    follows its HUGSIM adapter: speed as vx, longitudinal acceleration as ax, vy = ay = 0 ([T2] choice 3)."""
    v, a = past_t0[:, 2:4], past_t0[:, 4:6] / 0.25
    if model == "wajepa":
        sp = np.linalg.norm(v, axis=-1)
        return np.stack([sp, np.zeros_like(sp), a[:, 0], np.zeros_like(sp)], -1)
    return np.concatenate([v, a], -1)


# ---------------------------------------------------------------- output -> exam grid

def grid(traj: np.ndarray, shift: np.ndarray | None = None, extrap: bool = False) -> np.ndarray:
    """(n, 8, 3) poses at 0.5 ... 4.0 s -> (n, 20, 2) at 0.25 ... 5.0 s. shift: rigid move of the origin from the rear
    axle to a point d of the body, p'(t) = p(t) + R(psi_t) d - d. Cubic spline in time through (0, 0) + 8 points
    (x and y separately); 4.25 ... 5.0 s NaN, or with extrap the last two points' constant velocity (WOD's 5 s point)."""
    from scipy.interpolate import CubicSpline
    p = traj[..., :2].astype(np.float64).copy()
    if shift is not None:
        c, s = np.cos(traj[..., 2]), np.sin(traj[..., 2])
        p += np.stack([c * shift[0] - s * shift[1], s * shift[0] + c * shift[1]], -1) - shift
    t8 = np.r_[0.0, 0.5 * np.arange(1, 9)]
    pts = np.concatenate([np.zeros((len(p), 1, 2)), p], 1)
    tg = 0.25 * np.arange(1, 21)
    out = np.full((len(p), 20, 2), np.nan)
    out[:, :16] = CubicSpline(t8, pts, axis=1)(tg[:16])
    if extrap:
        v = (p[:, 7] - p[:, 6]) / 0.5
        out[:, 16:] = p[:, 7:8] + v[:, None] * (tg[16:, None] - 4.0)
    return out.astype(np.float32)


def load_preds(set_: str) -> dict:
    """{model: (keys, traj)} of finished prediction files."""
    out = {}
    for m in MODELS:
        f = root("preds", f"{set_}_{m}.npz")
        if f.exists():
            z = np.load(f)
            out[m] = (z["keys"], z["traj"])
    return out


# ---------------------------------------------------------------- I3

def i3_frames() -> pd.DataFrame:
    """The I3 index rows the exam reads (elicit_i3.needed), in index order."""
    from . import elicit_i3 as I
    t = pd.read_parquet(data_dir() / "processed" / I3_SRC / "index.parquet")
    return t[t.frame_name.isin(set(I.needed()))].reset_index(drop=True)


def req_i3():
    from . import hugsim_pairs as H
    t = i3_frames()
    past = np.load(data_dir() / "processed" / I3_SRC / "past.npy")[
        pd.read_parquet(data_dir() / "processed" / I3_SRC / "index.parquet").frame_name.isin(set(t.frame_name)).to_numpy()]
    new = data_dir() / "processed" / I3_10 / "scenes"
    img, hist, padded, logs = [], [], [], {}
    for r in t.itertuples():
        meta = json.loads((new / r.base_id / "meta.json").read_text())
        t10 = np.array(meta["t_render_10hz"])
        k = int(np.argmin(np.abs(t10 - r.t)))
        assert abs(t10[k] - r.t) < 1e-3 and k == 2 * r.cam_index
        ks = [max(0, k + int(round(s / 0.1))) for s in HIST_S]        # warmup clamp to the window start
        padded.append(k + int(round(HIST_S[0] / 0.1)) < 0)
        w = r.route_id[len(r.base_id) + 1:]
        img.append([str(new / r.base_id / w / "cams" / c / f"{2 * j:07d}.jpg")
                    for j in ks for c in ("front_left", "front", "front_right", "back")])
        if r.base_id not in logs:
            logs[r.base_id] = H.Logged(H.unpack(meta["dataset"], meta["scene"]), meta["dataset"])
        L = logs[r.base_id]
        th = L.pose_at(t10[ks])[2]
        xy = H.ego_frame(L, float(t10[k]), np.stack(L.pose_at(t10[ks])[:2], -1))
        hist.append(np.c_[xy, -(th - th[-1])])
    assert all(Path(p).exists() for row in img[:50] for p in row)
    cmd = t.intent.map(NAV_CMD).to_numpy()
    for m in MODELS:
        save_req(f"i3_{m}", t.frame_name, img, hist, ego_rows(past[:, -1], m), cmd)
    pd.DataFrame({"frame_name": t.frame_name, "hist_padded": padded}).to_parquet(root("requests", "i3_meta.parquet"))
    log.info("I3: %d frames, %d with incomplete WA-JEPA history (clamped to the window start)", len(t), sum(padded))


def exam_i3(rl):
    from . import elicit_i3 as I, p5_exam as E
    with I.p5_set(I3_SRC):
        _, _, _, obs, null, pairs = E.load()
    t = i3_frames()
    preds = {}
    for m, (keys, traj) in load_preds("i3").items():
        assert (keys == t.frame_name.to_numpy()).all()
        preds[NAME[m]] = grid(traj, shift=F0_REAR)
    np.savez_compressed(rl.dir / "preds_i3_grid.npz", frame_name=t.frame_name.to_numpy(), **preds)
    tfv6, E.TFV6 = E.TFV6, {}
    try:
        oo, nn = E.deltas(obs, null, t, preds)
        res = E.exam(oo, nn, pairs, list(preds))
        # WA-JEPA sensitivity: only pair / null frames whose 1.5 s history lies inside the render window (both sides)
        pad = pd.read_parquet(root("requests", "i3_meta.parquet")).set_index("frame_name").hist_padded
        full_o = ~(pad.reindex(oo.fn_plus).to_numpy() | pad.reindex(oo.fn_minus).to_numpy())
        full_n = ~(pad.reindex(nn.fn_plus).to_numpy() | pad.reindex(nn.fn_null).to_numpy())
        rs = E.exam(oo[full_o], nn[full_n], pairs, list(preds))
    finally:
        E.TFV6 = tfv6
    fl = pd.concat([res["flips"].assign(subset="all"), rs["flips"].assign(subset="complete_history")])
    fl.to_csv(rl.dir / "flip_rates.csv", index=False)
    res["validity"].to_csv(rl.dir / "validity.csv", index=False)
    res["obs"].to_parquet(rl.dir / "obs_scored.parquet", index=False)
    nn.to_parquet(rl.dir / "null_scored.parquet", index=False)
    info = {"tau_exp": res["tau_exp"], "pooled_families": res["pooled_families"], "taus": res["taus"],
            "reactive_frames": int(res["obs"].reactive.sum()), "padded_frames": int(pad.sum()),
            "pair_frames_complete_history": int(full_o.sum()), "null_frames_complete_history": int(full_n.sum())}
    (rl.dir / "summary.json").write_text(json.dumps(info, indent=1))
    rl.log.info("%s\n%s", info, fl[fl.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    return fl


# ---------------------------------------------------------------- P5 v1 BA

def p5_rows():
    from . import elicit_i3 as I, p5_exam as E
    with I.p5_set(I.BA):
        t, past, _, obs, null, pairs = E.load()
    rows = np.flatnonzero(t.role.to_numpy() == "obs")
    return t, past, rows


def _frame_path(p: str, dticks: int) -> str:
    d, f = p.rsplit("/", 1)
    return f"{d}/{int(f[:-4]) + dticks:07d}.jpg"


def req_p5():
    t, past, rows = p5_rows()
    tr = t.iloc[rows]
    img, hist, clamped = [], [], 0
    first = {}
    for r, pr in zip(tr.itertuples(), past[rows]):
        files = list(r.files)                         # 3 cameras x 4 frames (t-0.6 ... t), camera-major
        cur = {c: files[4 * i + 3] for i, c in enumerate(("front", "front_left", "front_right"))}
        d0 = cur["front"].rsplit("/", 1)[0]
        if d0 not in first:
            first[d0] = min(int(Path(f).stem) for f in Path(d0).iterdir())
        row, dts = [], []
        for s in (-1.4, -1.0, -0.4, 0.0):             # [T2] choice 3: nearest 5 Hz frames to -1.5 / -1.0 / -0.5 / 0 s
            dt = int(round(s / 0.05))
            fr = int(Path(cur["front"]).stem) + dt
            if fr < first[d0]:
                dt += first[d0] - fr
                clamped += 1
            dts.append(dt * 0.05)
            row += [_frame_path(cur["front_left"], dt), _frame_path(cur["front"], dt), _frame_path(cur["front_right"], dt), ""]
        img.append(row)
        hist.append(p5_hist(pr, np.array(dts)))
    for p in (img[0][1], img[-1][1], img[len(img) // 2][9]):
        assert Path(p).exists(), p
    cmd = tr.intent.map(NAV_CMD).to_numpy()
    for m in MODELS:
        save_req(f"p5_{m}", tr.frame_name, img, hist, ego_rows(past[rows, -1], m), cmd)
    log.info("P5: %d frames, %d history slots clamped to the stream start", len(rows), clamped)


def p5_hist(pr: np.ndarray, dts: np.ndarray) -> np.ndarray:
    """(4, 3) poses at times dts (s, <= 0) from a WOD-shaped past (0.25 s grid, t0 last): x, y interpolated, yaw the
    direction of the interpolated velocity (held from the next later sample below 0.5 m/s, 0 at t0)."""
    tp = 0.25 * np.arange(-15, 1)
    x, y = np.interp(dts, tp, pr[:, 0]), np.interp(dts, tp, pr[:, 1])
    vx, vy = np.interp(dts, tp, pr[:, 2]), np.interp(dts, tp, pr[:, 3])
    yaw = np.zeros(len(dts))
    for k in range(len(dts) - 2, -1, -1):
        yaw[k] = np.arctan2(vy[k], vx[k]) if np.hypot(vx[k], vy[k]) >= 0.5 else yaw[k + 1]
    return np.stack([x, y, yaw], -1)


def exam_p5(rl):
    from . import elicit_e4 as E4, p5_exam as E, p5_pairs as P
    obs, null, taus, pooled, ref = E4.load("ba")
    t, _, rows = p5_rows()
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    mine = []
    for m, (keys, traj) in load_preds("p5").items():
        assert (keys == t.frame_name.to_numpy()[rows]).all()
        v = np.full(len(t), np.nan)
        v[rows] = P.v2(grid(traj))
        name = NAME[m]
        obs[name] = v[pos[obs.fn_plus].to_numpy()] - v[pos[obs.fn_minus].to_numpy()]
        null[name] = v[pos[null.fn_plus].to_numpy()] - v[pos[null.fn_null].to_numpy()]
        taus[name] = float(np.quantile(np.abs(null[name].dropna()), 0.95))
        mine.append(name)
    keep = ["TFv6 target speed", "TFv6 waypoint speed 2 s", "ridge_late op-cinque temporal", "prior [cinque]",
            "M-C pair [cinque]"]
    ex = mine + [k for k in keep if k in ref or k in obs.columns]
    lat = E4.latency
    E4.latency = lambda e: 0.0 if e in mine else lat(e)
    try:
        sc = E4.score(obs, null, taus, pooled, ex, null)
    finally:
        E4.latency = lat
    sc = sc[sc.window != "(a) gated"]
    sc.to_csv(rl.dir / "p5_scores.csv", index=False)
    # per family (p5_exam.exam on the same deltas; its pooled row equals score's per-frame pooled row)
    tfv6, E.TFV6 = E.TFV6, {}
    try:
        pairs = pd.read_csv(data_dir() / "runs" / E4.RUNS["ba"][0] / "pairs.csv", dtype={"base_id": str})
        fam = E.exam(obs.drop(columns=["reactive", "dstop"], errors="ignore"), null, pairs, mine)
    finally:
        E.TFV6 = tfv6
    fam["flips"].to_csv(rl.dir / "p5_flip_rates_family.csv", index=False)
    chk = fam["flips"][fam["flips"].scope == "pooled"].set_index("examinee").flip_rate
    s2 = sc[(sc.window == "per frame") & (sc.scope == "pooled")].set_index("examinee").flip
    d = float((chk - s2.reindex(chk.index)).abs().max())
    rl.log.info("per-frame pooled: p5_exam.exam vs elicit_e4.score max |diff| %.2e", d)
    assert d < 1e-9
    rl.log.info("P5 BA\n%s", sc[sc.scope != "cut-in"].to_markdown(index=False, floatfmt=".3f"))
    return sc


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("req-i3", "req-p5", "exam-i3", "exam-p5"))
    a = ap.parse_args()
    if a.cmd == "req-i3":
        req_i3()
    elif a.cmd == "req-p5":
        req_p5()
    else:
        rl = RunLog("top10_t2", a.cmd)
        (exam_i3 if a.cmd == "exam-i3" else exam_p5)(rl)
        rl.close()


if __name__ == "__main__":
    main()
