#!/usr/bin/env python
"""Alpamayo 1.5 on real sharp turns (PhysicalAI-AV test split): does it need its cross cameras? Plan: ../plans/.

Alpamayo runs as shipped (NVlabs/alpamayo1.5 loader, message builder and sampler, unchanged); cameras are dropped through
the loader's own `camera_features` argument (variable camera count is a supported 1.5 feature), never blanked.

  scan     (alpamayo venv, CPU, network)  egomotion of 12 test chunks -> turn candidates -> pilot split + cases.json
  fetch    (alpamayo venv, CPU, network)  4 cameras + egomotion of the selected clips into the sparse cache
  infer    (alpamayo venv, GPU pool)       arms A / B / B1 / An / Bn x t0 (primary, +1.5 s) x 6 samples -> cache/alp/*.npz
  opframes (alpamayo venv, CPU)            front-wide f-theta -> openpilot road / wide frames, 101 steps at 20 Hz -> cache/op/
  oprun    (project venv, GPU pool)        shipped Cinque, zero state, no desire -> cache/op/preds.npz
  score    (any venv)                      per-sample / per-case metrics, paired effects -> results/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir  # noqa: E402

TOPIC = REPO / "experiments" / "alpamayo_turns"
CACHE = data_dir() / "runs" / "alpamayo_turns" / "cache"
SPLIT = ("physical_ai_av", "alpamayo-turns-pilot")
T = np.arange(1, 65) * 0.1                      # Alpamayo future: 0.1 ... 6.4 s
CAMS = {"cross_left": "CAMERA_CROSS_LEFT_120FOV", "front_wide": "CAMERA_FRONT_WIDE_120FOV",
        "cross_right": "CAMERA_CROSS_RIGHT_120FOV", "front_tele": "CAMERA_FRONT_TELE_30FOV"}
ARMS = {"A": (("cross_left", "front_wide", "cross_right", "front_tele"), False),
        "B": (("front_wide", "front_tele"), False),
        "B1": (("front_wide",), False),
        "An": (("cross_left", "front_wide", "cross_right", "front_tele"), True),
        "Bn": (("front_wide", "front_tele"), True)}
T0_OFFSETS = {"p": 0, "s": 1_500_000}          # primary t0, secondary t0 + 1.5 s
N_SAMPLES = 6
# scan rule (plan): heading change over the 6.4 s horizon, low speed, small radius
H_MIN, H_MAX, V_LO, V_HI, K_MIN = np.radians(60), np.radians(135), 2.0, 10.0, 1 / 20
PRE_S, HOR_S = 5.1, 6.4
FEATS = ("egomotion", "camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov",
         "camera_front_tele_30fov")


def snap_dir() -> Path:
    from jevdrive.alpamayo import data as D
    rev = (D.cache_dir() / "revision.txt").read_text().strip()
    return D.cache_dir() / f"datasets--{D.REPO.replace('/', '--')}" / "snapshots" / rev


def cache(*p) -> Path:
    d = CACHE.joinpath(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def yaw_of(R: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(R[..., 1, 0], R[..., 0, 0]), axis=-1)


# ---------------------------------------------------------------- scan
def ego_series(avdi, clip: str, dt_us: int = 100_000):
    ego = avdi.get_clip_feature(clip, avdi.features.LABELS.EGOMOTION, maybe_stream=False)
    ts = np.arange(ego.timestamps[0], ego.timestamps[-1], dt_us, dtype=np.int64)
    st = ego(ts)
    yaw = np.unwrap(st.pose.rotation.as_euler("zyx")[:, 0])
    v = np.linalg.norm(st.velocity[:, :2], axis=1)
    return ts, yaw, v


def turn_of(ts, yaw, v):
    """The clip's sharpest turn under the plan's rule, or None. All arrays at 10 Hz."""
    h, pre = int(HOR_S * 10), int(PRE_S * 10)
    if len(ts) < pre + h + 1:
        return None
    i0 = np.arange(pre, len(ts) - h)
    H = yaw[i0 + h] - yaw[i0]
    k = int(np.argmax(np.abs(H)))
    Hm = abs(H[k])
    if not H_MIN <= Hm <= H_MAX:
        return None
    first = int(i0[np.argmax(np.abs(H) >= 0.9 * Hm)])               # earliest t0 with >= 90 % of the turn ahead
    w = slice(first, first + h + 1)
    vmed = float(np.median(v[w]))
    om = np.convolve(np.gradient(yaw, 0.1), np.ones(10) / 10, "same")  # 1 s smoothing
    kap = np.abs(om[w]) / np.maximum(v[w], 1.0)
    if not (V_LO <= vmed <= V_HI and kap.max() >= K_MIN):
        return None
    return dict(t0=int(ts[first]), H_deg=float(np.degrees(H[first - pre])), side="L" if H[k] > 0 else "R",
                v_med=float(vmed), k_peak=float(kap.max()), r_min=float(1 / kap.max()),
                hist_dpsi_deg=float(np.degrees(yaw[first] - yaw[first - 15])), t0_rel_s=float((ts[first] - ts[0]) * 1e-6))


def cmd_scan(a, run):
    import pandas as pd
    from jevdrive.alpamayo import data as D
    from jevdrive.data import splits
    avdi = D.interface()
    ci = avdi.clip_index
    test = ci[(ci.split == "test") & ci.clip_is_valid]
    chunks = np.sort(np.random.default_rng(0).choice(np.sort(test.chunk.unique()), a.chunks, replace=False))
    clips = sorted(test[test.chunk.isin(chunks)].index)
    run.info(f"{len(clips)} test clips in chunks {chunks.tolist()}")
    D.fetch_clips(clips, features=("egomotion",), log=run.info)
    rows = []
    for c in run.tqdm(clips, desc="scan"):
        try:
            r = turn_of(*ego_series(avdi, c))
        except Exception as e:  # noqa: BLE001  (a clip without egomotion in the zip)
            run.info(f"{c}: {e!r}")
            continue
        if r:
            rows.append(dict(clip=c, chunk=int(ci.chunk[c]), **r))
    df = pd.DataFrame(rows)
    df = df[avdi.feature_presence.reindex(df.clip)[list(FEATS[1:])].all(1).to_numpy()]
    df.to_csv(TOPIC / "results" / "candidates.csv", index=False)
    run.info(f"{len(df)} candidate turns ({(df.side == 'L').sum()} L / {(df.side == 'R').sum()} R)")
    rng = np.random.default_rng(0)
    pick = pd.concat([g.iloc[rng.permutation(len(g))[:a.per_side]] for _, g in df.sort_values("clip").groupby("side")])
    cases = [dict(case=f"{r['clip']}@{r['t0']}", **r) for r in pick.to_dict("records")]
    sp = splits.define(*SPLIT, [c["case"] for c in cases], unit="clip@t0_us",
                       origin=f"experiments/alpamayo_turns/scripts/turns.py scan: PhysicalAI-AV @ {snap_dir().name[:7]} "
                              f"test split, rng(0) {a.chunks} chunks {chunks.tolist()}, heading change 60-135 deg over 6.4 s, median 2-10 m/s, "
                              f"peak curvature >= 1/20, t0 = earliest with >= 90 % of the turn ahead; rng(0) {a.per_side} per side",
                       used_by=["alpamayo_turns"], status="frozen", notes="alpamayo_turns pilot: sharp real turns")
    run.use_split(sp)
    (TOPIC / "results" / "cases.json").write_text(json.dumps(cases, indent=1, default=lambda o: o.item()))
    run.summary.update(n_clips=len(clips), n_candidates=len(df), split=sp.id)


def cmd_fetch(a, run):
    from jevdrive.alpamayo import data as D
    from jevdrive.hfdl import snapshot
    cases = json.loads((TOPIC / "results" / "cases.json").read_text())
    run.summary.update(D.fetch_clips(sorted({c["clip"] for c in cases}), features=FEATS, log=run.info))
    snapshot(D.REPO, tuple(f"calibration/{k}/{k}.chunk_{ch:04d}.parquet" for ch in sorted({c["chunk"] for c in cases})
                           for k in ("camera_intrinsics", "sensor_extrinsics")), dataset=True, cache_dir=D.cache_dir(), log=run.info)


# ---------------------------------------------------------------- Alpamayo
def nav_text(c: dict, gt_xy: np.ndarray, gt_yaw: np.ndarray) -> str:
    half = 0.5 * gt_yaw[-1]
    j = int(np.argmax(np.sign(half) * gt_yaw >= abs(half)))
    seg = np.linalg.norm(np.diff(np.vstack([[0, 0], gt_xy[:j + 1]]), axis=0), axis=1).sum()
    return f"Turn {'left' if gt_yaw[-1] > 0 else 'right'} in {int(round(seg))}m"


def cmd_infer(a, run):
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from jevdrive.alpamayo import data as D
    from jevdrive.alpamayo.infer import load
    from jevdrive.data import splits
    run.use_split(splits.load("/".join(SPLIT)))
    cases = json.loads((TOPIC / "results" / "cases.json").read_text())
    if a.limit:
        cases = cases[:a.limit]
    avdi = D.interface()
    model, proc = load(attn="sdpa")
    arms = a.arms.split(",")
    out = cache("alp")
    for ci, c in enumerate(run.tqdm(cases, desc="cases")):
        for tk, off in T0_OFFSETS.items():
            t0 = c["t0"] + off
            f = out / f"{c['clip']}_{tk}.npz"
            have = dict(np.load(f, allow_pickle=True)) if f.exists() else {}
            seed = 1000 * ci + (0 if tk == "p" else 1)
            for arm in arms:
                if f"{arm}_xyz" in have:
                    continue
                cams, nav = ARMS[arm]
                d = load_physical_aiavdataset(c["clip"], t0_us=t0, avdi=avdi, maybe_stream=False,
                                              camera_features=[getattr(avdi.features.CAMERA, CAMS[k]) for k in cams])
                gxy = d["ego_future_xyz"][0, 0, :, :2].numpy()
                gyaw = yaw_of(d["ego_future_rot"][0, 0].numpy())
                nt = nav_text(c, gxy, gyaw) if nav else None
                msgs = helper.create_message(d["image_frames"].flatten(0, 1), camera_indices=d["camera_indices"], nav_text=nt)
                tok = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                               return_dict=True, return_tensors="pt")
                inp = helper.to_device({"tokenized_data": tok, "ego_history_xyz": d["ego_history_xyz"],
                                        "ego_history_rot": d["ego_history_rot"]}, "cuda")
                torch.cuda.manual_seed_all(seed)
                torch.manual_seed(seed)
                with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                    xyz, rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=inp, top_p=0.98, temperature=0.6, num_traj_samples=N_SAMPLES, max_generation_length=256,
                        return_extra=True)
                have[f"{arm}_xyz"] = xyz[0].reshape(-1, 64, 3).float().cpu().numpy()
                have[f"{arm}_yaw"] = yaw_of(rot[0].reshape(-1, 64, 3, 3).float().cpu().numpy())
                have[f"{arm}_cot"] = np.array([str(s) for s in np.asarray(extra["cot"][0]).ravel()])
                have[f"{arm}_nav"] = np.array(nt or "")
                have[f"{arm}_cams"] = d["camera_indices"].numpy()
                im = d["image_frames"][:, -1]                                          # last frame per camera
                have[f"{arm}_img"] = torch.nn.functional.interpolate(im.float(), size=(180, 320), mode="area").byte().numpy()
                have.update(gt_xy=gxy, gt_yaw=gyaw, hist_xyz=d["ego_history_xyz"][0, 0].numpy(), t0=t0,
                            n_prompt=tok["input_ids"].shape[1])
                np.savez(f, **have)
                run.info(f"{c['case']} {tk} {arm}: nav={nt!r} yaw_end gt {np.degrees(gyaw[-1]):+.0f} pred "
                         f"{np.degrees(have[f'{arm}_yaw'][:, -1]).round(0).tolist()} | {have[f'{arm}_cot'][0][:90]}")
    run.summary["cases"] = len(cases)


# ---------------------------------------------------------------- openpilot (front-wide -> road / wide frames)
OP_CAM = "camera_front_wide_120fov"
DT, N_STEPS = 50_000, 101


def calib(avdi, clip: str, G) -> dict:
    """pai_openpilot.calib, reading the calibration parquets of the current revision's snapshot."""
    import pandas as pd
    ch = avdi.get_clip_chunk(clip)
    i = pd.read_parquet(snap_dir() / f"calibration/camera_intrinsics/camera_intrinsics.chunk_{ch:04d}.parquet").loc[(clip, OP_CAM)]
    e = pd.read_parquet(snap_dir() / f"calibration/sensor_extrinsics/sensor_extrinsics.chunk_{ch:04d}.parquet").loc[(clip, OP_CAM)]
    return {"cx": i.cx, "cy": i.cy, "fw": [i[f"fw_poly_{k}"] for k in range(5)], "w": int(i.width), "h": int(i.height),
            "R": G.quat_to_matrix((e.qx, e.qy, e.qz, e.qw)), "xyz": np.array([e.x, e.y, e.z])}


def cmd_opframes(a, run):
    from PIL import Image
    from jevdrive.alpamayo import data as D
    sys.path.insert(0, str(REPO / "experiments" / "zeroshot_openloop" / "archive"))
    import pai_openpilot as PO   # the frame renderer of the earlier PAI openpilot run (calib, f-theta gather, YUV pack)
    avdi = D.interface()
    cases = json.loads((TOPIC / "results" / "cases.json").read_text())
    out = cache("op")
    for c in run.tqdm(cases, desc="frames"):
        cal = calib(avdi, c["clip"], PO.G)
        ix, cov = PO.ftheta_index(cal)
        cam = avdi.get_clip_feature(c["clip"], avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV, maybe_stream=False)
        for tk, off in T0_OFFSETS.items():
            t0 = c["t0"] + off
            ts = t0 - DT * np.arange(N_STEPS - 1, -1, -1, dtype=np.int64)
            imgs, fts = cam.decode_images_from_timestamps(ts)
            ycc = [np.asarray(Image.fromarray(im).convert("YCbCr")) for im in imgs]
            frames = np.stack([np.stack([PO.pack(y, ix[0]), PO.pack(y, ix[1])]) for y in ycc])
            np.savez(out / f"{c['clip']}_{tk}.npz", frames=frames, t=ts, fts=np.asarray(fts, np.int64), cam_xyz=cal["xyz"],
                     coverage=cov, t0=t0)
            run.info(f"{c['case']} {tk}: coverage {cov:.3f}, |frame - step| max {np.abs(np.asarray(fts) - ts).max() / 1e3:.1f} ms, "
                     f"cam height {cal['xyz'][2]:.2f} m")


def cmd_oprun(a, run):
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    ACTION_T = (0.275, 0.525)
    mod = OPModel("cinque", "trt")
    files = sorted(cache("op").glob("*_[ps].npz"))
    res = {}
    for f in run.tqdm(files, desc="op"):
        z = np.load(f)
        alp = np.load(cache("alp") / f.name)
        mod.reset()
        curv = []
        for fr in z["frames"]:
            raw = mod.step(fr, desire=np.zeros(8, np.float32), traffic=(1, 0), action_t=ACTION_T)
        hist = alp["hist_xyz"]
        v = float(np.linalg.norm(hist[-1, :2] - hist[-2, :2]) / 0.1)
        o = decode(raw, mod.slices, v, ACTION_T)
        yaw = -np.asarray(o["plan_yaw"], np.float64)                     # openpilot: + = right; here + = left
        p = np.stack([o["plan_pos"][:, 0], -o["plan_pos"][:, 1]], -1)
        dv = z["cam_xyz"][:2]                                              # device -> rear axle (rig origin)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rear = dv + p - np.stack([cy * dv[0] - sy * dv[1], sy * dv[0] + cy * dv[1]], -1)
        res[f.stem] = dict(yaw=np.interp(T, T_IDXS, yaw), xy=np.stack([np.interp(T, T_IDXS, rear[:, k]) for k in range(2)], -1),
                           curv=-o["curvature"], v=v)
    np.savez(cache("op") / "preds.npz", keys=np.array(list(res)), yaw=np.stack([r["yaw"] for r in res.values()]),
             xy=np.stack([r["xy"] for r in res.values()]), curv=np.array([r["curv"] for r in res.values()]))
    run.summary["n"] = len(res)


# ---------------------------------------------------------------- metrics
def t_half(yaw: np.ndarray, target: float) -> np.ndarray:
    """First time the signed heading reaches 0.5 * target; 6.5 s when it never does within 6.4 s (censored)."""
    s = np.sign(target) * yaw >= 0.5 * abs(target)
    return np.where(s.any(-1), T[np.argmax(s, -1)], 6.5)


def sample_metrics(yaw, xy, gyaw, gxy) -> dict:
    """yaw (n, 64), xy (n, 64, 2) -> per-sample metrics."""
    tg = t_half(gyaw[None], gyaw[-1])[0]
    tp = t_half(yaw, gyaw[-1])
    ade = np.linalg.norm(xy - gxy[None], axis=-1).mean(-1)
    return dict(A_H=(yaw * gyaw).sum(-1) / (gyaw ** 2).sum(), R_end=yaw[:, -1] / gyaw[-1], lag50=tp - tg,
                censored=(tp > 6.4).astype(float), ade64=ade)


def cmd_score(a, run):
    import pandas as pd
    from jevdrive import stats
    cases = json.loads((TOPIC / "results" / "cases.json").read_text())
    op = np.load(cache("op") / "preds.npz") if (cache("op") / "preds.npz").exists() else None
    opk = {k: i for i, k in enumerate(op["keys"])} if op is not None else {}
    rows = []
    for c in cases:
        for tk in T0_OFFSETS:
            f = cache("alp") / f"{c['clip']}_{tk}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            gyaw, gxy = z["gt_yaw"], z["gt_xy"]
            arms = {k[:-4]: (z[f"{k[:-4]}_yaw"], z[f"{k[:-4]}_xyz"][..., :2]) for k in z.files if k.endswith("_xyz") and k != "hist_xyz"}
            if f"{c['clip']}_{tk}" in opk:
                i = opk[f"{c['clip']}_{tk}"]
                arms["OP"] = (op["yaw"][i][None], op["xy"][i][None])
            for arm, (yaw, xy) in arms.items():
                m = sample_metrics(yaw, xy, gyaw, gxy)
                r = dict(case=c["case"], side=c["side"], t0=tk, arm=arm, gt_dpsi_deg=float(np.degrees(gyaw[-1])),
                         **{k: float(np.mean(v)) for k, v in m.items()}, minade=float(m["ade64"].min()),
                         turn_rate=float(np.mean(m["R_end"] >= 0.5)), n=len(yaw))
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(TOPIC / "results" / "per_case.csv", index=False)
    eff = []
    pairs = [("An", "Bn"), ("A", "B"), ("A", "B1"), ("An", "A"), ("A", "OP"), ("An", "OP"), ("B", "OP")]
    for tk in T0_OFFSETS:
        d = df[df.t0 == tk]
        piv = {k: d.pivot(index="case", columns="arm", values=k) for k in ("A_H", "lag50", "R_end", "ade64", "turn_rate")}
        for x, y in pairs:
            if x not in piv["A_H"] or y not in piv["A_H"]:
                continue
            for k, p in piv.items():
                eff.append(dict(t0=tk, contrast=f"{x}-{y}", metric=k, **stats.paired(p[x].to_numpy(), p[y].to_numpy())))
        for arm in sorted(d.arm.unique()):
            for k in ("A_H", "lag50", "R_end", "ade64", "minade", "turn_rate", "censored"):
                eff.append(dict(t0=tk, contrast=arm, metric=k, **stats.bootstrap(d[d.arm == arm][k].to_numpy())))
    stats.write_table(eff, TOPIC / "results" / "effects")
    run.info("\n" + df.groupby(["t0", "arm"])[["A_H", "R_end", "lag50", "ade64", "turn_rate"]].mean().round(3).to_string())


if __name__ == "__main__":
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("scan", "fetch", "infer", "opframes", "oprun", "score"))
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--per-side", type=int, default=5)
    ap.add_argument("--arms", default="A,B,B1,An,Bn")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with Run("alpamayo_turns", a.cmd, config=vars(a)) as run:
        globals()[f"cmd_{a.cmd}"](a, run)
