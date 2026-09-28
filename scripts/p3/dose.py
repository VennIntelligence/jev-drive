"""Pedestrian dose-response exam for openpilot (todos/2026-09-28-ped-dose-response.md, registered before any number).

  anchors  (CPU) per reconstructed P3 scene: one anchor frame t* per ego state (crc32-minimal among the qualifying
           frames in [3.0, 17.5] s): pull-away (< 0.3 m/s for the 1 s before, >= 1 m/s within 2 s after, < 0.5 m/s at t*),
           creep (0.5-2.0 m/s), cruise (>= 5 m/s).  -> runs/nq4/p3/dose/anchors.json
  render   (GPU, one scene x state) the 48 cells: longitudinal d in {3, 5, 8, 12, 20, 30} m ahead of the bumper at t*
           (along the logged path), lateral offset to the right of the path centre in {0, 1.5, 3.0, 5.0} m (lane
           centre / lane edge / kerb / sidewalk), pedestrian standing (still for the whole clip, facing the lane) or
           crossing (walking in from the sidewalk at its own gait, reaching the point at t*), under the walk-in rules of
           xinsert.walk_plan (no spawning, visible outside the lane >= 1.5 s before entering it, no overlap with any
           reconstructed actor; a cell that cannot satisfy them is skipped with the reason); plus x- (the target
           re-render) and the log, 5 Hz over [t* - 4.0, t* + 0.4] s, three front cameras.
           Donor = one per scene, crc32 order over the donor bank, the first with a 1.0-1.8 m/s window, placed with the approved grounded
           standard: feet on the LiDAR ground, per-channel gain at t* (clip 1.33), contact shadow only where the ground
           under the feet is lit. View gap per cell is recorded.  -> runs/nq4/p3/dose/items/<scene>_<state>/<cell>/
  stage    (CPU) the rendered cells as the P3 set nq4_p3_dose (scene dirs <scene>_<state>_<cell> with real / plus = the
           cell / minus = x-), for the registered readout chain (index -> openpilot temporal + plan + lead -> exam).

  $DATA_DIR/envs/drivestudio/bin/python scripts/p3/dose.py anchors
  CUDA_VISIBLE_DEVICES=6 $DATA_DIR/envs/drivestudio/bin/python scripts/p3/dose.py render --scene 0 --state cruise
  $DATA_DIR/envs/drivestudio/bin/python scripts/p3/dose.py stage
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import insert as INS  # noqa: E402
import xinsert as XI  # noqa: E402

DSM = INS.DSM
DATA = Path(os.environ["DATA_DIR"])
DO = DATA / "runs/nq4/p3/dose"
HZ = 10
DISTS, LATS, STATES_PED = (3, 5, 8, 12, 20, 30), (0.0, 1.5, 3.0, 5.0), ("stand", "cross")
PRE_R, POST_R = 40, 4                     # rendered window in 10 Hz frames, 5 Hz stride (4 s lead-in: walk-in rules)
SHADE, GAIN_MAX = 0.6, 1.33
NEAR_HOLD_D = 20                          # m: cells nearer than this wait while runs/nq4/p3/dose/NEAR_HOLD exists


def anchors(a):
    out = {}
    for k in range(66):
        if not (DATA / "ckpt/nq4_p3/p3" / f"{k:03d}/checkpoint_final.pth").exists():
            continue
        g = XI.scene_geom(k)
        path = INS.Path2D(g["E"])
        v = path.v
        seg = json.loads((DATA / "runs/nq4/p3/targets" / f"{k:03d}.json").read_text())["segment"]
        rng = range(30, min(176, g["n"] - POST_R - 21))
        cands = {"pull": [t for t in rng if (v[t - 10:t] < 0.3).all() and v[t:t + 20].max() >= 1.0 and v[t] < 0.5],
                 "creep": [t for t in rng if 0.5 <= v[t] <= 2.0],
                 "cruise": [t for t in rng if v[t] >= 5.0]}
        out[k] = {s: min(c, key=lambda t: INS.crc(f"{seg}/dose/{s}/{t}")) for s, c in cands.items() if c}
        out[k] = {s: {"t_star": t, "v": round(float(v[t]), 2)} for s, t in out[k].items()}
    DO.mkdir(parents=True, exist_ok=True)
    (DO / "anchors.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"scenes": len(out), **{s: sum(s in v for v in out.values()) for s in ("pull", "creep", "cruise")}}))


COLL_D = (5, 10, 15)      # (d) collision-course cells (user 2026-09-28): entry into the ego path this far ahead at t*


def cells():
    return [(d, lat, st) for st in STATES_PED for lat in LATS for d in DISTS] + [(d, 0.0, "collide") for d in COLL_D]


def cell_id(d, lat, st):
    return f"{st}_d{d:02d}_l{int(lat * 10):02d}"


def render(a):
    import torch
    from PIL import Image, ImageDraw, ImageFilter
    from scipy.ndimage import binary_dilation
    from pytorch3d.transforms import quaternion_to_matrix
    t0 = time.time()
    k, state = a.scene, a.state
    kk = f"{k:03d}"
    anc = json.loads((DO / "anchors.json").read_text())[str(k)][state]
    ts = anc["t_star"]
    seg = json.loads((DATA / "runs/nq4/p3/targets" / f"{kk}.json").read_text())["segment"]
    bank_ = XI.load_bank()
    # one donor per scene (registered): crc32(segment / dose / donor) order, first with a 1.0-1.8 m/s walking window
    cand = sorted(bank_, key=lambda q: INS.crc(f"{seg}/dose/{q['scene']}/{q['node']}"))
    d, win = next((q, w) for q in cand for w in q["windows"] if XI.V_WALK[0] <= w["speed"] <= XI.V_WALK[1])
    pl = {"donor": {kk_: d[kk_] for kk_ in ("scene", "node", "waymo_id", "height", "span", "psnr", "ratio", "ckpt")}, "window": win}
    obs_all = np.radians(np.asarray(d["az"]))
    run = DATA / "ckpt/nq4_p3/p3" / kk
    cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt = DSM._load(run, None)
    from models.trainers.base import GSModelType
    g = XI.scene_geom(k)
    path = INS.Path2D(g["E"])
    frames = list(range(ts - PRE_R, ts + POST_R + 1, 2))
    gr = XI.Ground(g, range(ts - PRE_R, min(ts + 50, g["n"] - 1)))     # ground past t* too: collide cells plan to t_c
    dev = node.instances_trans.device
    xds = {}

    def donor_obj(dd):
        """XDonor, its per-frame rotations and local means, cached per donor (a crossing cell may use its own donor)."""
        key = (dd["scene"], dd["node"])
        if key not in xds:
            x_ = XI.XDonor(dd, dev)
            xds[key] = (x_, quaternion_to_matrix(x_.iq / x_.iq.norm(dim=-1, keepdim=True)).double().cpu().numpy(),
                        x_.means.double().cpu().numpy())
        return xds[key]

    xd, Rq, Mloc = donor_obj(d)
    tr.models["XDonor"] = xd
    tr.gaussian_classes["XDonor"] = GSModelType.DeformableNodes
    out = DO / "items" / f"p3_{kk}_{state}"
    camK = {}
    to8 = lambda x: (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()  # noqa: E731

    def view(t, ci=0):
        ii, cc = ds.full_image_set.get_image(t * ps.num_cams + ci, 1)
        ii = {q: v.cuda() if torch.is_tensor(v) else v for q, v in ii.items()}
        cc = {q: v.cuda() if torch.is_tensor(v) else v for q, v in cc.items()}
        camK[(t, ci)] = (cc["intrinsics"].cpu().numpy().astype(np.float64), cc["camera_to_world"].cpu().numpy().astype(np.float64))
        return ii, cc

    actors = XI.scene_actors(g)
    frames10 = list(range(ts - PRE_R, ts + POST_R + 1))

    def place(dist, lat, st):
        """Walk-in rules (xinsert.walk_plan, right-hand kerb side): standing = still for the whole clip at the point;
        crossing = from the sidewalk at the donor's gait into the point, reached at t*. {t: (R, T, Rd, s)}, feet, gap,
        or (None, None, reason)."""
        S = path.Se[ts] + XI.FRONT + dist
        t_conf, lead_, post_, lat_ts = ts, PRE_R, POST_R, lat
        if st == "collide":
            # collision course (user 2026-09-28, (d)): the pedestrian crosses from the kerb and reaches the path centre
            # at the moment the logged ego bumper reaches arc S (dist ahead of the bumper at t*); planned (and checked:
            # visibility, overlap, resolution) up to t_c + 0.4 s, rendered over the usual window. Skipped when the logged
            # ego does not get there within 4 s.
            tc = [t for t in range(ts + 1, min(ts + 41, g["n"] - 2)) if path.Se[t] >= path.Se[ts] + dist]
            if not tc:
                return None, None, f"the logged ego does not reach {dist} m within 4 s"
            t_conf = tc[0]
            lead_, post_ = PRE_R + (t_conf - ts), POST_R
        # resolution gate (user 2026-09-28): only donors seen in their own log at <= MAG_MAX x this cell's farthest
        # possible camera distance can pass xinsert.walk_plan's exact per-frame check, so the rest are not tried
        reach = XI.MAG_MAX * (dist + XI.FRONT + abs(lat) + 8.0)
        cand = [q for q in bank_ if q["d_close"] <= reach]
        if st == "stand":
            # over the donor bank (resolution gate: a scene donor is rarely seen close enough), any facing in 15 deg
            # steps: the feasible placement with the smallest view gap
            best, why = None, {}
            for dd in cand:
                ob = np.radians(np.asarray(dd["az"]))
                for f in np.radians(np.arange(0, 360, 15)):
                    o = XI.walk_plan(g, path, gr, actors, ob, dd, dd["windows"][0], 1.0, ts, S, lat, PRE_R, POST_R, "stand", face=f)
                    if "fail" in o:
                        k_ = o["fail"].split(" at frame")[0].split(" (")[0].split(":")[0]
                        why[k_] = why.get(k_, 0) + 1
                    elif best is None or o["gap"] < best["gap"]:
                        best = {**o, "window": dd["windows"][0], "donor": dd}
            if best is None:
                return None, None, "no feasible standing placement: " + (", ".join(f"{k_} {v_}" for k_, v_ in sorted(why.items(), key=lambda q: -q[1])[:3])
                                                                          or f"no donor seen closer than {reach:.0f} m")
            if best["gap"] > XI.AZ_MAX:
                return None, None, f"best view gap {best['gap']:.0f} deg > {XI.AZ_MAX:.0f}"
            b = best
        else:
            # crossing: over the whole donor bank and every 1.0-1.8 m/s window, the feasible walk-in with the smallest
            # view gap (a walking donor is seen from where the log saw it walk; one scene donor rarely fits every cell)
            best, why = None, {}
            for dd in cand:
                ob = np.radians(np.asarray(dd["az"]))
                for w_ in dd["windows"]:
                    if not XI.V_WALK[0] <= w_["speed"] <= XI.V_WALK[1]:
                        continue
                    o = XI.walk_plan(g, path, gr, actors, ob, dd, w_, 1.0, t_conf, S, lat, lead_, post_, "cross", gate_frames=frames10)
                    if "fail" in o:
                        k_ = o["fail"].split(" at frame")[0].split(" from ")[0].split(" only")[0].split(" (")[0]
                        why[k_] = why.get(k_, 0) + 1
                    elif best is None or o["gap"] < best["gap"]:
                        best = {**o, "window": w_, "donor": dd, "t_conf": int(t_conf)}
            if best is None:
                return None, None, "no feasible walk-in: " + (", ".join(f"{k_} {v_}" for k_, v_ in sorted(why.items(), key=lambda q: -q[1])[:3])
                                                              or f"no donor seen closer than {reach:.0f} m")
            if best["gap"] > XI.AZ_MAX:
                return None, None, f"best view gap {best['gap']:.0f} deg > {XI.AZ_MAX:.0f}"
            b = best
        if "fail" in b:
            return None, None, b["fail"]
        xd_, Rq, Mloc = donor_obj(b["donor"])
        phi = b["phi"]
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        zs = []
        for t in frames:
            s_, x, y = b["traj"][t]
            xy = np.array([x, y])
            Rw = Rz @ Rq[s_]
            f = np.percentile((Mloc @ Rw.T)[:, 2], 1)
            zs.append((t, s_, xy, Rw, gr(xy) - f, f))
        z = np.array([q[4] for q in zs])
        z = np.where(np.isfinite(z), z, np.nanmedian(z) if np.isfinite(z).any() else 0.0)
        res, feet = {}, {}
        for (t, s_, xy, Rw, _, f), zz in zip(zs, z):
            res[t] = (torch.tensor(Rw, dtype=torch.float32, device=dev), torch.tensor(np.r_[xy, zz], dtype=torch.float32, device=dev),
                      torch.tensor(Rq[s_], dtype=torch.float32, device=dev), int(s_))
            feet[t] = (xy, zz + f)
        return res, feet, b

    def lum(im):
        return im[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])

    def disc(t, ci, xy, zg, r, shape):
        K, c2w = camK[(t, ci)]
        ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        Pd = np.c_[xy[0] + r * np.cos(ang), xy[1] + r * np.sin(ang), np.full(32, zg), np.ones(32)]
        pc = (np.linalg.inv(c2w) @ Pd.T).T[:, :3]
        if not np.isfinite(zg) or (pc[:, 2] <= 0.5).any():
            return None, 0.0
        uv = (K @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:]
        mk = Image.new("L", (shape[1], shape[0]), 0)
        ImageDraw.Draw(mk).polygon([tuple(p) for p in uv], fill=255)
        return np.asarray(mk) > 0, float(uv[:, 0].max() - uv[:, 0].min())

    rat0 = np.asarray(d["ratio"])
    meta_cells = {}
    with torch.no_grad():
        for w in ("real", "minus"):                                     # the log and x-, shared by all cells
            if (out / w / "frames.jsonl").exists():
                for t in frames:
                    for ci in range(3):
                        view(t, ci)
                continue
            rows = []
            for t in frames:
                rec = {}
                for ci, c in enumerate(INS.CAMS):
                    ii, cc = view(t, ci)
                    xd.pose = {}
                    im = to8(ii["pixels"]) if w == "real" else to8(tr(ii, cc)["rgb"])
                    (out / w / "cams" / c).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(im).save(out / w / "cams" / c / f"{2 * t:07d}.jpg", quality=95)
                    rec[c] = f"cams/{c}/{2 * t:07d}.jpg"
                rows.append({"frame": 2 * t, "t": t / HZ, "files": rec})
            (out / w / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        # lit-lane reference per frame (as xinsert.render)
        ref = {}
        for t in frames:
            mi = lum(np.asarray(Image.open(out / "minus/cams/front" / f"{2 * t:07d}.jpg")))
            K, c2w = camK[(t, 0)]
            w2c = np.linalg.inv(c2w)
            vals = []
            for sd_ in np.arange(8, 40, 1.0):
                q, th, _ = path.at(path.Se[t] + sd_)
                for lt in (-1.0, 0.0, 1.0):
                    p = q + lt * np.array([-np.sin(th), np.cos(th)])
                    zz = gr(p)
                    if not np.isfinite(zz):
                        continue
                    pc = w2c[:3, :3] @ np.r_[p, zz] + w2c[:3, 3]
                    if pc[2] <= 1:
                        continue
                    uu, vv = (K @ pc)[:2] / pc[2]
                    uu, vv = int(uu), int(vv)
                    if 1 <= uu < mi.shape[1] - 1 and 1 <= vv < mi.shape[0] - 1:
                        vals.append(np.median(mi[vv - 1:vv + 2, uu - 1:uu + 2]))
            ref[t] = np.percentile(vals, 90) if len(vals) >= 5 else None
        for dist, lat, st in cells():
            cid = cell_id(dist, lat, st)
            cd = out / cid
            if (cd / "meta.json").exists():
                continue
            if dist < NEAR_HOLD_D and (DO / "NEAR_HOLD").exists():
                # user 2026-09-28: near cells wait for the resolution gate and the expanded donor bank (no file written,
                # so a later pass renders them)
                print(json.dumps({"cell": cid, "held": "near-range hold"}), flush=True)
                continue
            pose, feet, b = place(dist, lat, st)
            if pose is not None:
                xd = donor_obj(b["donor"])[0]
                tr.models["XDonor"] = xd
                rat0 = np.asarray(b["donor"]["ratio"])
            if pose is None:
                (cd).mkdir(parents=True, exist_ok=True)
                (cd / "skipped.json").write_text(json.dumps({"cell": cid, "reason": b}))
                print(json.dumps({"cell": cid, "skipped": b}), flush=True)
                continue
            gap = b["gap"]
            # per-channel gain at t* (one value per cell)
            ii, cc = view(ts)
            xd.gain_map, xd.pose = {}, pose
            wi = to8(tr(ii, cc)["rgb"])
            wo = np.asarray(Image.open(out / "minus/cams/front" / f"{2 * ts:07d}.jpg"))
            m = np.abs(wi.astype(int) - wo.astype(int)).max(-1) > INS.DONOR_THR
            gain = np.ones(3)
            if m.sum() >= 80:
                ring = INS._ring(m)
                gain = np.clip([rat0[c] * max(np.median(wo[..., c][ring]), 1.0) / max(np.median(wi[..., c][m]), 1.0) for c in range(3)],
                               1 / GAIN_MAX, GAIN_MAX)
            gt = torch.tensor(gain, dtype=torch.float32, device=dev)
            xd.gain_map = {t: gt for t in frames}
            # shade test per frame at the feet
            shade = {}
            for t in frames:
                mi = np.asarray(Image.open(out / "minus/cams/front" / f"{2 * t:07d}.jpg"))
                mk, _ = disc(t, 0, feet[t][0], feet[t][1], 0.6, mi.shape)
                shade[t] = None if (mk is None or not mk.any() or ref[t] is None) else bool(np.median(lum(mi)[mk]) >= SHADE * ref[t])
            vals = [v for v in shade.values() if v is not None]
            default = (sum(vals) >= len(vals) / 2) if vals else True
            seq = [shade[t] if shade[t] is not None else default for t in frames]
            on = {t: bool(np.mean(seq[max(i - 1, 0):i + 2]) >= 0.5) for i, t in enumerate(frames)}
            rows = []
            for t in frames:
                rec = {}
                for ci, c in enumerate(INS.CAMS):
                    ii, cc = view(t, ci)
                    im = to8(tr(ii, cc)["rgb"]).astype(np.float64)
                    if on[t]:
                        mi = np.asarray(Image.open(out / "minus/cams" / c / f"{2 * t:07d}.jpg"))
                        donor = np.abs(im.astype(int) - mi.astype(int)).max(-1) > INS.DONOR_THR
                        mk, wpx = disc(t, ci, feet[t][0], feet[t][1], INS.R_SHADOW, im.shape)
                        if mk is not None and mk.any():
                            blur = Image.fromarray((mk * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(float(max(1.5, 0.25 * wpx))))
                            im = im * (1 - INS.ALPHA_SHADOW * (np.asarray(blur) / 255.0 * ~binary_dilation(donor, iterations=2))[..., None])
                    (cd / "plus/cams" / c).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.clip(im + 0.5, 0, 255).astype(np.uint8)).save(cd / "plus/cams" / c / f"{2 * t:07d}.jpg", quality=95)
                    rec[c] = f"cams/{c}/{2 * t:07d}.jpg"
                rows.append({"frame": 2 * t, "t": t / HZ, "files": rec})
            (cd / "plus/frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            xd.pose = {}
            _, lat_ts = path.locate(np.asarray(b["traj"][ts][1:]))           # lateral offset at t* (collide: still walking in)
            mc = {"cell": cid, "dist": dist, "lat": lat, "ped_state": st, "lat_tstar": round(-float(lat_ts), 2), "t_conf": b.get("t_conf"),
                  "view_gap": gap, "gain": [float(x) for x in gain],
                  "shadow_on_frac": float(np.mean(list(on.values()))), "kerb": b["kerb"], "kerb_src": b["kerb_src"], "walk_s": b["walk_s"],
                  "vis_out_s": b["vis_out_s"], "donor_speed": b["speed"], "mag": b.get("mag"), "donor_d_close": b["donor"]["d_close"],
                  "donor": f"p3_{b['donor']['scene']:03d}/{b['donor']['node']}"}
            (cd / "meta.json").write_text(json.dumps(mc))
            meta_cells[cid] = mc
            print(json.dumps(mc), flush=True)
    meta = {"scene": k, "state": state, "t_star": ts, "ego_v": anc["v"], "frames": frames, "donor": d, "window": pl["window"],
            "render_s": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps({"done": f"p3_{kk}_{state}", "cells": len(meta_cells), "render_s": meta["render_s"]}))


def gatecheck(a):
    """Resolution gate on cells rendered before it (user 2026-09-28): the pedestrian's closest front-camera distance over
    the clip from the cell geometry (standing: the fixed point; crossing: perpendicular to the path at the donor's speed
    from its start point, at the point at t*, walking on after it), against the donor's closest logged distance. A cell
    over the magnification cap moves to <item>/_gate_dropped/<cell> with the reason; the next render pass re-places it."""
    import shutil
    dc = {f"p3_{q['scene']:03d}/{q['node']}": q["d_close"] for q in XI.load_bank()}
    n = {"kept": 0, "dropped": 0, "placed_under_gate": 0}
    for sd in sorted((DO / "items").glob("p3_*_*")):
        if not (sd / "meta.json").exists() or sd.name.count("_") != 2:
            continue
        m = json.loads((sd / "meta.json").read_text())
        g = XI.scene_geom(m["scene"])
        path = INS.Path2D(g["E"])
        ts = m["t_star"]
        for cd in sorted(p for p in sd.iterdir() if (p / "meta.json").exists() and p.name not in ("real", "minus")):
            mc = json.loads((cd / "meta.json").read_text())
            if mc.get("mag") is not None:
                n["placed_under_gate"] += 1
                continue
            donor = mc.get("donor") or f"p3_{m['donor']['scene']:03d}/{m['donor']['node']}"
            Q, th, _ = path.at(path.Se[ts] + XI.FRONT + mc["dist"])
            right = np.array([np.sin(th), -np.cos(th)])
            stand = mc["ped_state"] == "stand"
            start = mc["lat"] if stand else max(mc["kerb"] + XI.KERB_OFF, mc["lat"] + 1.0)
            ds = []
            for t in range(ts - PRE_R, ts + POST_R + 1):
                xy = Q + (mc["lat"] if stand else min(start, mc["lat"] + mc["donor_speed"] * (ts - t) / HZ)) * right
                if XI.in_fov(g, t, np.r_[xy, 0.0]):
                    ds.append(np.linalg.norm(xy - XI.cam_axis(g, t)[0][:2]))
            mag = dc[donor] / min(ds) if ds else 0.0
            if mag <= XI.MAG_MAX:
                n["kept"] += 1
                continue
            dst = sd / "_gate_dropped" / cd.name
            dst.parent.mkdir(exist_ok=True)
            shutil.move(str(cd), str(dst))
            (dst / "gate.json").write_text(json.dumps({"mag": round(mag, 2), "donor": donor, "d_close": dc[donor],
                                                       "closest_cam_m": round(min(ds), 2), "rule": XI.GATE_RULE}))
            n["dropped"] += 1
            print(json.dumps({"item": sd.name, "cell": cd.name, "mag": round(mag, 2)}), flush=True)
    print(json.dumps(n))


def stage(a):
    """processed/nq4_p3_dose/scenes/<key>: real / plus / minus linked; meta with the P3 calibration of the scene."""
    OUT = DATA / "processed/nq4_p3_dose/scenes"
    n = 0
    for sd in sorted((DO / "items").glob("p3_*_*")):
        if not (sd / "meta.json").exists():
            continue
        m = json.loads((sd / "meta.json").read_text())
        ref = json.loads((DATA / "processed/nq4_p3/scenes" / f"p3_{m['scene']:03d}" / "meta.json").read_text())
        for cd in sorted(p for p in sd.iterdir() if (p / "meta.json").exists() and p.name not in ("real", "minus")):
            key = f"{sd.name}_{cd.name}"                                   # p3_<k>_<state>_<cell>: matches the index glob p3_*
            dst = OUT / key
            for w, src in (("real", sd / "real"), ("plus", cd / "plus"), ("minus", sd / "minus")):
                (dst / w).mkdir(parents=True, exist_ok=True)
                if not (dst / w / "cams").exists():
                    (dst / w / "cams").symlink_to(src / "cams")
                (dst / w / "frames.jsonl").write_text((src / "frames.jsonl").read_text())
            mc = json.loads((cd / "meta.json").read_text())
            (dst / "meta.json").write_text(json.dumps({"scene": m["scene"], "key": key, "f0": m["t_star"], "state": m["state"],
                                                       "ego_v": m["ego_v"], "calib": ref["calib"],
                                                       "extrinsics_cam_to_ego": ref["extrinsics_cam_to_ego"], **mc}))
            n += 1
    print(json.dumps({"staged": n}))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("anchors")
    p = sp.add_parser("render")
    p.add_argument("--scene", type=int, required=True)
    p.add_argument("--state", choices=("pull", "creep", "cruise"), required=True)
    sp.add_parser("stage")
    sp.add_parser("gatecheck")
    a = ap.parse_args()
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    {"anchors": anchors, "render": render, "stage": stage, "gatecheck": gatecheck}[a.cmd](a)


if __name__ == "__main__":
    main()
