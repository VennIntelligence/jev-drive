#!/usr/bin/env python
"""Validation of collected B2D clips (plans/2026-10-06-b2d-collect-prereg.md, "Checks"): replay clips through Cinque / op_parity P2's input
pipeline and check that frames, ego, command and labels line up. GPU (envs/op-train), one pool job.

  $DATA_DIR/envs/op-train/bin/python experiments/b2d_collect/scripts/b2dc_check.py --data $DATA_DIR/runs/b2d_collect/data/<tag> \
      [--models P0 P2-F-s0] [--stride 10] [--gif 4]
  -> <data>/check/{check.json, samples.csv, panels/<route>.npz}

Per clip (labels from scripts/b2dc_labels.py):
  integrity   ticks == video pictures, picture index = tick, sim clock exactly 0.05 s per tick, consecutive frame numbers, camera data
              frame = world frame of the logged state (sensor_frame), frozen pictures (identical to the previous one) < 1 %, luma range
  time        from the pictures alone, at every launch from standstill: the first big change of the ground just ahead (road frame rows
              200-255) must fall between pictures s - 1 and s, s = the first logged tick above 0.2 m/s (lag 0). Gated on the launch from the spawn
              (s < 40 ticks), where the ego moves first; later launches are reported only (a waiting ego starts after the traffic ahead
              moves, so the ground region changes before it does: smoke clip lags -2 / -1 at the blocked-intersection restarts)
  horizon     body pitch (deg) -> horizon row shift in the road frame (910 tan(pitch)); nominal rows 47.6 (road) / 151.8 (wide)
  model       every `stride` ticks with a full 4 s future, the 8 context slots (pairs (t0 - 4k - 4, t0 - 4k), k = 7..0: op_parity's 0.2 s
              protocol, native frames, no synthesis) -> Cinque's frozen encoder -> P0 (shipped) and P2 (+ ego / pose / command from the labels)
              -> plan -> 8 rear-axle poses (pp_train.rear, camera 1.59 m ahead of the rear axle). ADE to the logged future, 4 s path-length
              ratio, heading-sign agreement on lane following (route command straight, no junction turn within 45 m, |logged yaw at 4 s| > 10 deg:
              a mirrored or misaligned frame shows here; junction turns are reported apart, they are the model's open problem, decision 121),
              and P0's ADE with the frame stack shifted by -8 / -4 / +4 / +8 ticks (report).
  command     route command vs the heading change actually driven over the next 60 m of path, moving ticks: turn-command precision
              (commanded side, > 30 deg) and the straight-but-turned share (> 45 deg; also catches curved roads, report)
  sdf         logged future footprint (hero corners) at the 8 poses on the t0 SDF raster: share of poses with every corner >= -0.3 m
GIF panels (`--gif` clips, turn routes first): every 4 ticks around the turn (or the whole clip), chase view | road | wide model frames as Cinque
gets them (YUV -> RGB), with the nominal horizon rows (dashed), the logged future (green) and P2's plan (red) projected on the ground. Captions
are added on the Mac (scripts/b2dc_gif.py, CJK font).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_pl.Path(__file__).resolve().parents[1] / "lib")]
import argparse, csv, json, math  # noqa: E401,E402

import numpy as np  # noqa: E402

import b2dc_frames as F  # noqa: E402
from jevdrive.run import Run  # noqa: E402

MOUNT = (1.59, 0.0, 1.86)
LAGS = (-8, -4, 0, 4, 8)


def yuv_rgb(p):
    """packed (6, 128, 256) -> (256, 512, 3) uint8 RGB (BT.601 limited range, chroma upsampled)."""
    from jevdrive.openpilot import frames as opf
    Y = opf.unpack_luma(p).astype(np.float32) - 16
    U = np.repeat(np.repeat(p[4].astype(np.float32) - 128, 2, 0), 2, 1)
    V = np.repeat(np.repeat(p[5].astype(np.float32) - 128, 2, 0), 2, 1)
    r, g, b = 1.164 * Y + 1.596 * V, 1.164 * Y - 0.392 * U - 0.813 * V, 1.164 * Y + 2.017 * U
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def project(xy, name):
    """Ground points (n, 2) rear-axle frame (x fwd, y left) -> model-frame pixels (n, 2) and validity."""
    K = F.MODEL_K[name]
    x, y = xy[:, 0] - MOUNT[0], xy[:, 1] - MOUNT[1]
    dev = np.stack([x, -y, np.full_like(x, MOUNT[2])], -1)        # device frame: x fwd, y right, z down (ground at +height)
    view = dev[:, [1, 2, 0]]
    uvw = view @ K.T
    ok = uvw[:, 2] > 1.0
    return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6), ok


def densify(P):
    """(k, 2) poses with the origin prepended -> 0.25 m polyline."""
    P = np.concatenate([np.zeros((1, 2)), P[np.isfinite(P[:, 0])]], 0)
    if len(P) < 2:
        return P
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    q = np.arange(0, d[-1], 0.25)
    return np.stack([np.interp(q, d, P[:, 0]), np.interp(q, d, P[:, 1])], -1) if len(q) > 1 else P


def draw(rgb, name, paths):
    import cv2
    img = rgb.copy()
    h = int(round(F.MODEL_K[name][1, 2]))
    for x0 in range(0, img.shape[1], 16):
        cv2.line(img, (x0, h), (x0 + 8, h), (255, 220, 0), 1)
    for P, col in paths:
        uv, ok = project(densify(P), name)
        pts = np.round(uv[ok]).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
    return img


def main(a):
    import torch
    import pp_train as T
    from jevdrive import op_adapt as A
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    data = _pl.Path(a.data)
    out = data / "check"
    (out / "panels").mkdir(parents=True, exist_ok=True)
    with open(data / "index.csv") as fh:
        idx = list(csv.DictReader(fh))
    if a.clips:
        idx = idx[: a.clips]
    order = sorted(idx, key=lambda r: (not r.get("turn"), r["route_id"]))
    gif_ids = {r["route_id"] for r in order[: a.gif]}
    dev = torch.device("cuda")
    with Run("b2d_collect", f"check-{data.name}", config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("b2d/b2dc-train"))
        net = A.load("cinque", torch.float16).to(dev).eval()
        models = {m: T.load_pmodel(m, dev) for m in a.models}
        pi = torch.as_tensor(A.plan_index(models[a.models[0]].net.slices), device=dev)
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)

        @torch.no_grad()
        def encode(prev, cur):
            o = []
            for i in range(0, len(cur), 128):
                p, c = (torch.from_numpy(np.ascontiguousarray(x[i:i + 128])).to(dev) for x in (prev, cur))
                o.append(net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), *A.H_SHAPE))
            return torch.cat(o)

        @torch.no_grad()
        def plans(model, front, ego):
            B = len(front)
            tc = torch.tensor([[1.0, 0.0]], device=dev).expand(B, 2)
            o = model(front, torch.from_numpy(ego).to(dev), tc).float()[:, pi].view(B, 33, 15)
            x, y, psi = T.rear(o, torch.full((B,), MOUNT[0], device=dev), W)
            return torch.stack([x, y, psi], -1).cpu().numpy()

        rows, clips = [], []
        for r in run.tqdm(idx, desc="clips"):
            clip = _pl.Path(r["clip"])
            ego = dict(np.load(clip / "ego.npz"))
            lab = dict(np.load(clip / "labels.npz"))
            sdz = np.load(clip / "sdf.npz")
            pairs = F.read_pairs(clip / "frames.mp4")
            n = len(ego["t"])
            c = {"route_id": r["route_id"], "type": r.get("type", ""), "town": r["town"], "ticks": n, "pictures": len(pairs)}
            c["vid_is_tick"] = bool(np.array_equal(ego["vid"], np.arange(n)))
            dt = np.diff(ego["t"])
            c["dt_max_err"] = float(np.abs(dt - 0.05).max()) if n > 1 else 0.0
            c["frames_consecutive"] = bool((np.diff(ego["frame"]) == 1).all())
            same = (pairs[1:] == pairs[:-1]).reshape(len(pairs) - 1, -1).all(1) if len(pairs) > 1 else np.zeros(0, bool)
            c["frozen_share"] = float(same.mean()) if len(same) else 0.0
            Yr = pairs[:, 0, :4].astype(np.float32)
            c["luma_mean"], c["luma_p1"], c["luma_p99"] = float(Yr.mean()), float(np.percentile(Yr[::10], 1)), float(np.percentile(Yr[::10], 99))
            pitch = ego["rot"][:, 0]
            c["pitch_p95_abs_deg"] = float(np.percentile(np.abs(pitch), 95))
            c["horizon_shift_p95_rows"] = float(910.0 * math.tan(math.radians(c["pitch_p95_abs_deg"])))
            c["sensor_frame_eq"] = bool((ego["sensor_frame"] == ego["frame"][:, None]).all()) if "sensor_frame" in ego else None
            # time alignment from the pictures alone, at launches (5 ticks below 0.1 m/s, then above 0.2 m/s at tick s; spawn drop has vz): the first big
            # change of the ground just ahead (road frame rows 200-255, centre half) must be between pictures s - 1 and s (lag 0)
            from jevdrive.openpilot import frames as opf
            Yg = np.stack([opf.unpack_luma(x)[200:256, 128:384] for x in pairs[:, 0]]).astype(np.int16)
            g = np.abs(np.diff(Yg, axis=0)).mean((1, 2))                 # g[k] = change from picture k to k + 1
            sp = lab["speed"]
            lags = []
            for s_ in range(11, n - 3):
                if sp[s_] > 0.2 and sp[s_ - 1] <= 0.2 and (sp[s_ - 6:s_ - 1] < 0.1).all():
                    base = np.median(g[s_ - 9:s_ - 2])
                    w = np.where(g[s_ - 5:s_ + 3] > max(2 * base, base + 1.0))[0]
                    if len(w):
                        lags.append(int(w[0] + s_ - 5 - (s_ - 1)))
                        if s_ < 40 and "first_launch_lag" not in c:
                            c["first_launch_lag"] = lags[-1]
            c["launch_lags"] = " ".join(map(str, lags))
            c["moving_ticks"] = int((lab["speed"] > 0.5).sum())
            # command vs the heading change actually driven over the next 60 m of path (> 30 deg), moving ticks: precision of the turn
            # commands (gated) and the share of straight-commanded ticks before a > 45 deg heading change (report: also curved roads)
            h = lab["heading"]
            P = lab["xy_world"]
            sdist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            j = np.searchsorted(sdist, sdist + 60.0)
            ok = (j < n) & (lab["speed"] > 1.0)
            dh = np.degrees(h[np.minimum(j, n - 1)] - h)
            cm = lab["cmd"][:, :3].argmax(1)
            tc = ok & (cm != 1)
            c["cmd_turn_precision"] = float(((cm == 0) & (dh > 30) | (cm == 2) & (dh < -30))[tc].mean()) if tc.any() else float("nan")
            sc = ok & (cm == 1)
            c["cmd_straight_but_turned"] = float((np.abs(dh) > 45)[sc].mean()) if sc.any() else float("nan")
            c["cmd_turn_ticks"] = int((cm != 1).sum())
            # samples
            t0s = np.arange(max(36, 0), n, a.stride)
            t0s = t0s[lab["fut_ok"][t0s]] if len(t0s) else t0s
            t0s = t0s[(t0s + max(LAGS) < n) & (t0s + min(LAGS) - 32 >= 0)] if len(t0s) else t0s
            if len(t0s):
                if len(t0s) > a.max_samples:                              # long clips: evenly spaced subset (host / GPU memory)
                    t0s = t0s[np.linspace(0, len(t0s) - 1, a.max_samples).astype(int)]
                res = {}
                for lag in LAGS if a.lag else (0,):
                    parts = {}
                    for i0 in range(0, len(t0s), 32):
                        tt = t0s[i0:i0 + 32]
                        cur = np.stack([[pairs[t + lag - 4 * k] for k in range(7, -1, -1)] for t in tt])        # (s, 8, 2, 6, 128, 256)
                        prev = np.stack([[pairs[t + lag - 4 * k - 4] for k in range(7, -1, -1)] for t in tt])
                        fr = encode(prev.reshape(-1, 2, 6, 128, 256), cur.reshape(-1, 2, 6, 128, 256)).view(len(tt), 8, *A.H_SHAPE)
                        for m, model in models.items():
                            if lag != 0 and m != "P0":
                                continue
                            parts.setdefault(m, []).append(plans(model, fr, lab["ego"][tt]))
                        del fr
                    for m, v in parts.items():
                        res[(m, lag)] = np.concatenate(v)
                fut = lab["fut"][t0s]
                L = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(t0s), 1, 2)), fut[:, :, :2]], 1), axis=1), axis=-1).sum(1)
                mv = L > 2.0
                for k, t in enumerate(t0s):
                    row = {"route_id": r["route_id"], "t0": int(t), "speed": float(lab["speed"][t]), "cmd": int(cm[t]),
                           "junction_turn": bool(lab["turn_dist"][t] < 45.0 and lab["turn_next"][t] > 0),
                           "fut_yaw4": float(np.degrees(fut[k, -1, 2])), "log_len": float(L[k])}
                    for (m, lag), P in res.items():
                        ade = float(np.linalg.norm(P[k, :, :2] - fut[k, :, :2], axis=-1).mean())
                        row[f"ade_{m}" + (f"_lag{lag}" if lag else "")] = ade
                        if lag == 0:
                            pl = float(np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), P[k, :, :2]]), axis=0), axis=-1).sum())
                            row[f"len_{m}"] = pl
                            row[f"yaw4_{m}"] = float(np.degrees(P[k, -1, 2]))
                    rows.append(row)
                # sdf of the logged footprint
                st = list(sdz["ticks"])
                fp = sdz["footprint"]
                inside = []
                for k, t in enumerate(t0s):
                    if t not in st:
                        continue
                    S = sdz["sdf"][st.index(t)].astype(np.float32)
                    for q in fut[k]:
                        cth, sth = math.cos(q[2]), math.sin(q[2])
                        cx = q[0] + cth * fp[:, 0] - sth * fp[:, 1]
                        cy = q[1] + sth * fp[:, 0] + cth * fp[:, 1]
                        ii, jj = (cx - (-8.0)) / 0.5 - 0.5, (cy - (-24.0)) / 0.5 - 0.5
                        from scipy.ndimage import map_coordinates
                        v = map_coordinates(S, [ii, jj], order=1, mode="nearest")
                        inside.append(bool((v >= -0.3).all()))
                c["fut_footprint_drivable"] = float(np.mean(inside)) if inside else float("nan")
                c["samples"] = int(len(t0s))
                c["moving"] = int(mv.sum())
                if r["route_id"] in gif_ids:
                    c["gif"] = True
                    _panels(out / "panels" / f"{r['route_id']}.npz", clip, ego, lab, pairs, t0s, res.get(("P2-F-s0", 0), res.get((a.models[-1], 0))), r)
            clips.append(c)
        # ---- aggregate
        import pandas as pd
        S = pd.DataFrame(rows)
        S.to_csv(out / "samples.csv", index=False)
        C = pd.DataFrame(clips)
        C.to_csv(out / "clips.csv", index=False)
        agg = {"clips": len(C), "samples": len(S)}
        if len(S):
            mv = S.log_len > 2.0
            for m in a.models:
                agg[f"ade_{m}"] = float(S.loc[mv, f"ade_{m}"].mean())
                agg[f"len_ratio_med_{m}"] = float((S.loc[mv, f"len_{m}"] / S.loc[mv, "log_len"]).median())
                big = mv & (S.fut_yaw4.abs() > 20)
                agg[f"turn_sign_agree_{m}"] = float((np.sign(S.loc[big, f"yaw4_{m}"]) == np.sign(S.loc[big, "fut_yaw4"])).mean()) if big.any() else None
                agg["turn_samples"] = int(big.sum())
                lane = mv & ~S.junction_turn & (S.fut_yaw4.abs() > 10)            # lane following on curved roads: what Cinque can do
                agg[f"lane_sign_agree_{m}"] = float((np.sign(S.loc[lane, f"yaw4_{m}"]) == np.sign(S.loc[lane, "fut_yaw4"])).mean()) if lane.any() else None
                agg["lane_samples"] = int(lane.sum())
            if a.lag:
                agg["lag_ade_P0"] = {int(l): float(S.loc[mv, "ade_P0" + (f"_lag{l}" if l else "")].mean()) for l in LAGS}
                agg["lag_min_at_0"] = min(agg["lag_ade_P0"], key=agg["lag_ade_P0"].get) == 0
        for k in ("vid_is_tick", "frames_consecutive"):
            agg[k] = bool(C[k].all())
        agg["sensor_frame_eq_all"] = bool(C.sensor_frame_eq.dropna().all()) if C.sensor_frame_eq.notna().any() else None
        ll = [int(x) for v in C.launch_lags.fillna("") for x in str(v).split()]
        agg["launch_lags"] = {str(k): ll.count(k) for k in sorted(set(ll))}
        agg["launch_lag0_share"] = float(np.mean([x == 0 for x in ll])) if ll else None
        fl = C.first_launch_lag.dropna() if "first_launch_lag" in C else []
        agg["first_launch_lag0_share"] = float((fl == 0).mean()) if len(fl) else None
        agg["first_launch_lag_le1_share"] = float((fl.abs() <= 1).mean()) if len(fl) else None
        agg["first_launches"] = int(len(fl))
        for k in ("dt_max_err", "frozen_share", "horizon_shift_p95_rows", "pitch_p95_abs_deg"):
            agg[k + "_max"] = float(C[k].max())
        for k in ("cmd_turn_precision", "cmd_straight_but_turned", "fut_footprint_drivable"):
            agg[k + "_mean"] = float(C[k].mean())
        (out / "check.json").write_text(json.dumps(agg, indent=1))
        run.summary.update(agg)
        run.info(json.dumps(agg))


def _panels(path, clip, ego, lab, pairs, t0s, plan, r):
    """GIF material: every 4 ticks in a window around the first turn (else the whole clip), up to 90 frames."""
    import cv2
    n = len(ego["t"])
    turn = np.where((lab["turn_dist"] <= 0) & (lab["turn_next"] > 0))[0]          # ticks inside the junction turn
    if len(turn):
        lo, hi = max(0, turn[0] - 200), min(n, turn[-1] + 100)
    else:
        lo, hi = 0, n
    ticks = np.arange(lo, hi, 4)[-90:]
    chase_ok = (clip / "chase.mp4").exists()
    ch = F.decode(clip / "chase.mp4", w=480, h=270, fmt="rgb24") if chase_ok else None
    pmap = {int(t): plan[k] for k, t in enumerate(t0s)} if plan is not None else {}
    frames, caps = [], []
    last_plan = None
    for t in ticks:
        road, wide = yuv_rgb(pairs[t, 0]), yuv_rgb(pairs[t, 1])
        fut = lab["fut"][t, :, :2]
        near = [k for k in pmap if abs(k - t) <= 5]
        if near:
            last_plan = pmap[min(near, key=lambda k: abs(k - t))][:, :2]
        paths = [(fut, (40, 220, 40))] + ([(last_plan, (230, 40, 40))] if last_plan is not None else [])
        road, wide = draw(road, "road", paths), draw(wide, "wide", paths)
        ci = ego["chase"][t] if ch is not None else -1
        if ci < 0 and ch is not None:
            prev = ego["chase"][: t + 1]
            ci = prev[prev >= 0][-1] if (prev >= 0).any() else -1
        cview = cv2.resize(ch[ci], (384, 216)) if ci >= 0 else np.zeros((216, 384, 3), np.uint8)
        row = np.concatenate([cview, cv2.resize(road, (432, 216)), cv2.resize(wide, (432, 216))], 1)
        frames.append(row)
        caps.append({"t": round(float(ego["t"][t] - ego["t"][0]), 1), "speed": round(float(lab["speed"][t]), 1),
                     "cmd": int(lab["cmd"][t, :3].argmax()), "turn_dist": float(lab["turn_dist"][t]),
                     "ctl": [round(float(x), 2) for x in lab["ctl"][t]], "plan": last_plan is not None})
    jpg = [cv2.imencode(".jpg", f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes() for f in frames]   # small: the link to the Mac is slow
    np.savez(path, jpg=np.array(jpg, dtype=object), caps=json.dumps(caps), route=json.dumps(r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=["P0", "P2-F-s0"])
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--clips", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=150)
    ap.add_argument("--gif", type=int, default=4)
    ap.add_argument("--no-lag", dest="lag", action="store_false")
    main(ap.parse_args())
