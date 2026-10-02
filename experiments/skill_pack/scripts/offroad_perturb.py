"""Input perturbations of the native openpilot Cinque on navhard tokens (plan: experiments/skill_pack/plans/2026-10-03-...).

openpilot env + one GPU (CUDA_VISIBLE_DEVICES=2). The model, the GIMM frame cache, the step schedule and the output decoding
are exactly those of scripts/op_lb.py (`run`, schedule `none`); the unperturbed rerun is checked against the cached plan.
Variants (per token): see VARIANTS. Output: <out>/perturb_<tag>.pkl = {token: {variant: dict(pose8, plan_pos, plan_yaw,
road_edges (2, 33, 2), lane_lines (4, 33, 2))}} and a printed table.

  CUDA_VISIBLE_DEVICES=2 $DATA_DIR/envs/openpilot/bin/python experiments/skill_pack/scripts/offroad_perturb.py \
      --tokens 04d2f35c7f6db137f --tag case01
  ... --token-file tokens.txt --tag class
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import op_lb as B  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.openpilot.model import OPModel, mdn_mu, decode  # noqa: E402

DATA = "lb_navhard"
T_OUT = np.arange(1, 9) * 0.5
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
TURN = {"left": 1, "right": 2}     # log.Desire turnLeft / turnRight
LHT = {"sg-one-north"}


def mirror(f):
    """Horizontally mirror a packed frame (2, 6, 128, 256) about the principal point column (cx = 256 of 512)."""
    Y, U, V = I.unpack(f)
    return I.pack(np.ascontiguousarray(Y[..., ::-1]), np.ascontiguousarray(U[..., ::-1]), np.ascontiguousarray(V[..., ::-1]))


GRAY = np.array([128, 128, 128, 128, 128, 128], np.uint8)[:, None, None]


def edit_frame(f, how):
    """f (2, 6, 128, 256) packed [road, wide]; gray = Y 128, U = V = 128."""
    g = np.array(f)
    if how == "wide_blank":
        g[1] = GRAY
    elif how == "road_blank":
        g[0] = GRAY
    elif how == "wide_from_road":
        g[1] = g[0]
    elif how == "road_from_wide":
        g[0] = g[1]
    elif how in ("crop50", "crop25"):
        keep = 0.5 if how == "crop50" else 0.25
        c0, c1 = int(128 * (1 - keep) / 2 * 1), int(256 - 128 * (1 - keep) / 2 * 1)       # packed width is 256 (= 512 / 2)
        c0, c1 = int(256 * (1 - keep) / 2), int(256 * (1 + keep) / 2)
        for k in range(2):
            m = np.broadcast_to(GRAY, g[k].shape).copy()
            m[:, :, c0:c1] = g[k][:, :, c0:c1]
            g[k] = m
    return np.ascontiguousarray(g)


def variants():
    """name -> dict(desire=(kind, onset_s, sustained) | None, frames='gimm'|'current'|'mirror', tc='map'|'rhd'|'lhd'|'swap')"""
    V = {"base": {}}
    for k in ("right", "left"):
        for T in (-1.5, -0.5, 0.0):
            V[f"turn_{k}@{T:g}"] = dict(desire=(k, T, False))
            V[f"turn_{k}@{T:g}_sustained"] = dict(desire=(k, T, True))
    for k in ("right", "left"):
        for T, L in ((-1.5, 0.5), (-1.5, 1.0), (-0.5, 0.5)):
            V[f"turn_{k}@{T:g}_len{L:g}"] = dict(desire=(k, T, L))
    for T in (-1.5, -0.5, 0.0):                  # the t0 command's own desire (none for straight / unknown)
        V[f"cmd@{T:g}"] = dict(desire=("cmd", T, False))
    V["cmd@-1.5_len0.5"] = dict(desire=("cmd", -1.5, 0.5))
    V["history_current"] = dict(frames="current")
    # same speed history, current frame only, warped along: the true track / the track without yaw / the mirrored track
    # camera-input variants (plan question: wide / road inputs and the virtual-camera yaw)
    V["wide_blank"] = dict(edit="wide_blank")
    V["road_blank"] = dict(edit="road_blank")
    V["wide_copy_of_road"] = dict(edit="wide_from_road")
    V["road_copy_of_wide"] = dict(edit="road_from_wide")
    V["crop_center50"] = dict(edit="crop50")
    V["crop_center25"] = dict(edit="crop25")
    for d in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        V[f"yaw_warp_{d:+g}"] = dict(frames="camyaw", yaw=d, pitch=0.0)
    for d in (-1.0, 1.0):
        V[f"pitch_warp_{d:+g}"] = dict(frames="camyaw", yaw=0.0, pitch=d)
    V["warp_true_track"] = dict(frames="warp", track="true")
    V["warp_no_yaw_track"] = dict(frames="warp", track="straight")
    V["warp_mirrored_track"] = dict(frames="warp", track="mirror")
    V["mirror"] = dict(frames="mirror")
    V["mirror_tc_swap"] = dict(frames="mirror", tc="swap")
    V["tc_rhd_traffic"] = dict(tc="rhd")
    V["tc_lhd_traffic"] = dict(tc="lhd")
    V["current_turn_right@-1.5_sustained"] = dict(frames="current", desire=("right", -1.5, True))
    V["current_turn_left@-1.5_sustained"] = dict(frames="current", desire=("left", -1.5, True))
    return V


class Runner:
    def __init__(self):
        self.mt = B.meta(DATA)
        self.row = {t: i for i, t in enumerate(self.mt["names"])}
        self.keys = B.Keys(DATA)
        self.syn = np.load(B.root(DATA) / "gimm.npy", mmap_mode="r")
        backend = B.BACKENDS["cinque"]
        self.m = OPModel("cinque", backend, cache=data_dir() / "runs" / "op_interp" / "trt_cache" / f"cinque-{backend}", context_rate=False)
        self.ts, self.src = B._steps(0.0, False)
        from jevdrive import navsim_zs as Z
        self.idx = {e["token"]: e for e in Z.load_index("navhard_two_stage", slim=True)}

    def frames(self, i):
        kf, sf = self.keys[i], np.asarray(self.syn[i])
        return [np.ascontiguousarray(kf[j] if s == "k" else sf[j]) for s, j in self.src]

    def warp_frames(self, i, track):
        """The t0 key frame re-projected along the ego track (road-plane warp of jevdrive.op_interp) at every step time."""
        pose, vel = np.asarray(self.mt["pose"][i], float), np.asarray(self.mt["vel"][i], float)
        if track != "true":
            tr = I.track_navsim(pose, vel)
            T = np.linspace(-1.5, 0, 151)
            xy = np.array([tr(t)[:2] for t in T])
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            s_k = -(arc[-1] - np.interp(I.T_KEY, T, arc))               # signed distance behind t0 at the key times
            if track == "straight":
                pose = np.stack([s_k, np.zeros(4), np.zeros(4)], 1)
                vel = np.stack([np.linalg.norm(vel, axis=1), np.zeros(4)], 1)
            else:                                                           # mirrored: y -> -y, yaw -> -yaw, v_y -> -v_y
                pose = pose * np.array([1, -1, -1])
                vel = vel * np.array([1, -1])
        cur = self.keys[i][3]
        out = I.synth_cpu(np.stack([cur] * 4), "warp", self.ts, I.track_navsim(pose, vel), self.mt["cam"][i])
        return [np.ascontiguousarray(f) for f in out]

    def cam_frames(self, i, yaw_deg, pitch_deg):
        """Keys re-rendered with the virtual camera yawed / pitched by a small angle (left / up positive) through the dataset calibration, then
        the same road-plane warp history synthesis as the warp arm (the 0 deg row is the reference for the warp synthesis itself)."""
        from jevdrive import navsim_zs as Z
        from jevdrive.openpilot.frames import MEDMODEL_K, SBIGMODEL_K, VIEW_FROM_DEVICE, MODEL_W, MODEL_H
        e = self.idx[self.mt["names"][i]]
        cam = e["cams"][-1]["CAM_F0"]
        a, b = np.radians(yaw_deg), np.radians(pitch_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])      # up positive: x forward tilts toward +z
        uu, vv = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
        w, h = Z.NUPLAN_WH
        idxs = []
        for Km in (MEDMODEL_K, SBIGMODEL_K):
            ray_dev = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(Km @ VIEW_FROM_DEVICE).T
            ray_ego = (ray_dev * np.array([1., -1., -1.])) @ (Rz @ Ry).T
            uv, ok, _ = Z.project_nuplan(ray_ego, cam, 1)
            xi = np.clip(np.rint(uv[..., 0]), 0, w - 1).astype(np.int64)
            yi = np.clip(np.rint(uv[..., 1]), 0, h - 1).astype(np.int64)
            idxs.append((yi * w + xi).ravel())
        keys = []
        for f in range(4):
            cat = Z.OpenpilotMaps.decode(e["cams"][f]["CAM_F0"]["path"]).reshape(-1, 3)
            out = np.empty((2, 6, 128, 256), np.uint8)
            for k, ix in enumerate(idxs):
                p = cat[ix].reshape(256, 512, 3)
                Y = p[..., 0]
                uv_ = np.rint(p[..., 1:].reshape(128, 2, 256, 2, 2).astype(np.float32).mean((1, 3))).astype(np.uint8)
                out[k] = np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv_[..., 0], uv_[..., 1]])
            keys.append(out)
        res = I.synth_cpu(np.stack(keys), "warp", self.ts, I.track_navsim(self.mt["pose"][i], self.mt["vel"][i]), self.mt["cam"][i])
        return [np.ascontiguousarray(f) for f in res]

    def run(self, i, fr, desire_steps, sustained, tc):
        m = self.m
        m.reset()
        des = B.desire_arr(desire_steps, len(fr))
        for f, d in zip(fr, des):
            if sustained:
                m.prev_desire[:] = 0          # every step a rising edge: the pulse is repeated while the desire is held
            raw = m.step(f, desire=d, traffic=tc, action_t=B.ACTION_T)
        return raw

    def decode_row(self, raw, i, flip):
        d = decode(raw, self.m.slices, float(self.mt["speed"][i]), B.ACTION_T)
        pos, yaw = d["plan_pos"].copy(), d["plan_yaw"].copy()
        re = mdn_mu(raw[self.m.slices["road_edges"]], (2, 33, 2)).copy()
        ll = d["lane_lines"].copy()
        if flip:                                   # mirrored input: mirror the output back (openpilot y right, yaw clockwise)
            pos[:, 1] *= -1
            yaw *= -1
            re = re[::-1].copy()
            re[..., 0] *= -1
            ll = ll[::-1].copy()
            ll[..., 0] *= -1
        pose8 = I.to_rear(pos, yaw, I.T_IDXS, self.mt["cam"][i][:2], T_OUT, "lever", "linear")
        return dict(pose8=pose8, plan_pos=pos, plan_yaw=yaw, road_edges=re, lane_lines=ll, plan_vel=d["plan_vel"])

    def battery(self, token, only=None):
        i = self.row[token]
        fr0 = self.frames(i)
        lht = self.mt["lht"][i]
        tc0 = (0, 1) if lht else (1, 0)
        out = {}
        for name, v in variants().items():
            if only and name not in only:
                continue
            fr = fr0
            if v.get("frames") == "current":
                fr = [fr0[-1]] * len(fr0)
            elif v.get("frames") == "mirror":
                fr = [mirror(f) for f in fr0]
            elif v.get("frames") == "warp":
                fr = self.warp_frames(i, v["track"])
            elif v.get("frames") == "camyaw":
                fr = self.cam_frames(i, v["yaw"], v["pitch"])
            if v.get("edit"):
                fr = [edit_frame(f, v["edit"]) for f in fr]
            tc = {"map": tc0, None: tc0, "rhd": (1, 0), "lhd": (0, 1), "swap": tc0[::-1]}[v.get("tc")]
            ds = np.zeros(len(fr0), int)
            sustained = False
            if v.get("desire"):
                k, T, sustained = v["desire"]
                if k == "cmd":
                    k = {0: "left", 2: "right"}.get(self.mt["cmd"][i])
                on = (self.ts >= T - 1e-9) & ((self.ts < T + sustained - 1e-9) if (sustained and sustained is not True) else True)
                ds = np.where(on & (k is not None), TURN.get(k, 0), 0)
                # a held state: OPModel emits a single rising-edge pulse at the first step >= T; `sustained` (True or a window
                # length in s) repeats the pulse on every step of the window
                sustained = bool(sustained)
            raw = self.run(i, fr, ds, sustained, tc)
            out[name] = self.decode_row(raw, i, v.get("frames") == "mirror")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="*", default=[])
    ap.add_argument("--token-file", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(data_dir() / "runs/skill_pack/offroad_diag"))
    a = ap.parse_args()
    toks = list(a.tokens) + (Path(a.token_file).read_text().split() if a.token_file else [])
    R = Runner()
    cached = np.load(data_dir() / "runs/op_lb/lb_navhard/preds/gimm-cinque__base.npz")
    cp = dict(zip(cached["tokens"].tolist(), cached["poses"]))
    res = {}
    for n, t in enumerate(toks):
        res[t] = R.battery(t, a.only)
        if "base" in res[t]:
            err = np.abs(res[t]["base"]["pose8"] - cp[t]).max()
            res[t]["base_vs_cached_max_abs"] = float(err)
            print(f"[{n + 1}/{len(toks)}] {t[:8]} base vs cached max |dpose| = {err:.2e}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    plain = lambda o: {k: plain(v) for k, v in o.items()} if isinstance(o, dict) else o.tolist() if isinstance(o, np.ndarray) else o  # noqa: E731
    pickle.dump(plain(res), open(Path(a.out) / f"perturb_{a.tag}.pkl", "wb"))   # numpy-version independent (read by the navsim2 env)
    for t, r in res.items():
        print(t)
        for k, v in r.items():
            if isinstance(v, dict):
                p = v["pose8"]
                print(f"  {k:36s} end(4s) x={p[-1, 0]:6.2f} y={p[-1, 1]:6.2f} yaw={np.degrees(p[-1, 2]):6.1f} deg | y(1s)={p[1, 1]:5.2f} y(2s)={p[3, 1]:5.2f}")
    sys.stdout.flush()
    os._exit(0)          # ORT / TensorRT teardown can hang


if __name__ == "__main__":
    main()
