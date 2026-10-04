#!/usr/bin/env python
"""openpilot policy server for the closed-loop integration study (research/openpilot-closedloop-integration.md,
fc65452:todos/2026-09-28-op-closedloop.md). envs/openpilot. Agent side: lib/op_arb_agent.py.

Same model path as scripts/zeroshot_policy_server.py (Cinque at 20 Hz, road + wide warped as modeld, desire as a
rising-edge pulse; the class is imported, not copied), plus two things the arbitration and the diagnosis need:

  extras  every head the network already outputs, decoded as openpilot does: plan speed / accel along the plan, the
          lead head (3 leads x 6 times x (x, y, v, a), lead_prob), meta (engaged, disengage / hard-brake / gas-press /
          brake-press / blinker probabilities), desire_state, desire_pred (4 x 8), pose (vision ego motion), lane-line
          probabilities. None of these is an extra model; they are outputs the plan is computed alongside.
  twin    (meta "twin": true) a second session of the same model stepped on the same frames with desire 0, so each
          plan has a desire-free counterpart: the turn diagnosis compares the two plans frame by frame.
  sel     (env OP_SEL=<ratio>, off when unset; decision 94 / 101 selector, the B2D port of HUGSIM sel3,
          experiments/op_adapt_h/plans/2026-10-04-od2-prereg.md section 3) a second session per connection that sees the
          last OP_SEL_N (132 = 6.6 s: Cinque's ONNX state holds 132 desire / 128 feature steps; with 132 the
          unrotated replay reproduces the native plan exactly in CARLA, with 100 it did not) camera frames rotated in place to a reference heading. It runs only while speed <
          OP_SEL_VMAX (3 m/s) and the buffered history holds >= 0.05 deg of yaw against now (meta "yaw", the agent's pose,
          rad, CARLA right-positive). It is rebuilt (reset + replay of the buffer) when it switches on or when the heading
          moved >= OP_SEL_REBUILD_DEG (0.5 deg) from its reference and >= OP_SEL_MIN_GAP (4) steps passed since the last
          rebuild (HUGSIM's 0.2 s replay cadence), else stepped once with the new frame rotated to the reference. Its plan replaces the native one (pos, vel, yaw, acc, curvature, accel; the lead / meta heads stay
          native) iff its summed 0-4 s lateral plan std < ratio x the native plan's.
  lstab   (env OP_LSTAB=<json of lib/launch_stab.LaunchGate keywords>, e.g. '{}'; off when unset; experiments/hugsim/plans/
          2026-10-04-launch-stab-prereg.md) launch stabilisation: a second session per connection, forked (exact state copy)
          from the native one when the car moves off after a standstill, is stepped on the frames rotated to the launch heading
          (meta "yaw"); for the launch phase its lateral plan (pos y, yaw, curvature), rotated into the car frame, replaces the
          native one; longitudinal heads stay native.

    CUDA_VISIBLE_DEVICES=6 $DATA_DIR/envs/openpilot/bin/python experiments/op_closed_loop/archive/op_arb_server.py cinque --pool 4 \
        --backend cuda-iob --socket S
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("scripts",)]
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3] / "scripts"
sys.path[:0] = [str(HERE), str(HERE.parent)]
import zeroshot_policy_server as ZP  # noqa: E402
sys.path.insert(0, str(HERE.parent / "lib"))
import launch_stab as LS  # noqa: E402

# openpilot selfdrive/modeld/constants.py, Meta: indices into the 55 sigmoid outputs (times 2, 4, 6, 8, 10 s)
META = {"engaged": slice(0, 1), "gas_disengage": slice(1, 31, 6), "brake_disengage": slice(2, 31, 6),
        "steer_override": slice(3, 31, 6), "hard_brake_3": slice(4, 31, 6), "hard_brake_4": slice(5, 31, 6),
        "hard_brake_5": slice(6, 31, 6), "gas_press": slice(31, 55, 4), "brake_press": slice(32, 55, 4),
        "left_blinker": slice(33, 55, 4), "right_blinker": slice(34, 55, 4)}


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


class ArbModel(ZP.OpenpilotModel):
    def __init__(self, a):
        self.twin = not getattr(a, "no_twin", False)
        e = os.environ.get
        self.sel = float(e("OP_SEL", "0") or 0)
        self.sel_n, self.sel_vmax = int(e("OP_SEL_N", "132")), float(e("OP_SEL_VMAX", "3.0"))
        self.sel_rebuild, self.sel_warm = np.radians(float(e("OP_SEL_REBUILD_DEG", "0.5"))), int(e("OP_SEL_WARM", "0"))
        self.sel_gap = int(e("OP_SEL_MIN_GAP", "4"))       # steps between rebuilds at least (HUGSIM replays once per 0.2 s step)
        self.sel_check = int(e("OP_SEL_CHECK", "0") or 0)  # control: unrotated, its plan must match the native one (1: rebuild every
        #                                                    step, 2: the normal rebuild / incremental policy; 3: as 2 but every frame
        #                                                    rotated by the constant OP_SEL_CHECK_DEG, a camera mounted that much off)
        self.sel_check_deg = float(e("OP_SEL_CHECK_DEG", "0.01"))
        self.rot_cache, self.rot_lock = OrderedDict(), threading.Lock()
        ls = e("OP_LSTAB", "")
        self.lstab = None if not ls else __import__("json").loads(ls)
        a.pool = (1 + self.twin + (self.sel > 0) + (self.lstab is not None)) * a.pool   # a twin (unless --no-twin) and a selector session per connection
        self._warm = []
        self.E = None                             # adapted model: intent adapter table (4, 32, 512), next to the ONNX
        if getattr(a, "onnx", ""):
            e = Path(a.onnx).with_suffix(".E.npy")
            self.E = np.load(e).astype(np.float16) if e.exists() else None
        # vmerge trigger heads (experiments/vlm_arb/scripts/vmerge_det_fit.py): logistic regression on the native output vector,
        # enabled by env OP_DET_HEAD=<det_head.npz>; adds out["det"] = P(light, stop sign within 40 m). No effect when unset.
        dp = os.environ.get("OP_DET_HEAD", "")
        self.det = dict(np.load(dp)) if dp else None
        self.lanes = os.environ.get("OP_LANES", "") == "1"        # vmerge3: lane lines + road edges in out (no effect when unset)
        super().__init__(a)                       # its warm-up takes a state from new_state(None) and drops it
        self.free += self._warm

    def new_state(self, state=None):
        keys = ("model",) + (("twin",) if self.twin else ()) + (("sel",) if self.sel > 0 else ()) + \
            (("stab",) if self.lstab is not None else ())
        gate = (lambda: LS.LaunchGate(**self.lstab)) if self.lstab is not None else (lambda: None)  # noqa: E731
        if state and "model" in state:
            for k in keys:
                state[k].reset()
            state.update(buf=deque(maxlen=self.sel_n + 1), sel_ref=None, gate=gate())
            return state
        take = lambda: self.free.pop() if self.free else self.make()  # noqa: E731
        ss = {k: take() for k in keys}
        for m in ss.values():
            m.reset()
        if state is None:
            self._warm = list(ss.values())
        ss.update(buf=deque(maxlen=self.sel_n + 1), sel_ref=None, gate=gate())
        return ss

    def release(self, state):
        if state:
            self.free += [state[k] for k in ("model", "twin", "sel", "stab") if k in state]

    # ------------------------------------------------------------------------ selector
    def prepare(self, meta, arrays):
        prep = super().prepare(meta, arrays)
        if (self.sel > 0 or self.lstab is not None) and "OP_ROAD" in arrays:
            prep["raw"] = (arrays["OP_ROAD"], arrays["OP_WIDE"])
        return prep

    def _rot_idx(self, deg):
        """Gather indices of the road / wide model frames for a virtual camera yawed by deg (device frame, right-positive)
        against the frame's own camera, rounded to 0.01 deg (0.16 px in the road model frame; 0.1 deg = 1.6 px made the replayed
        history jitter between rounded and unrounded frames in the CARLA smoke), LRU-cached; source pixels outside the image are
        flagged."""
        key = round(float(deg), 2)
        with self.rot_lock:
            if key in self.rot_cache:
                self.rot_cache.move_to_end(key)
                return self.rot_cache[key]
        opf = self.opf
        w, h = ZP.rigs.OP_CAMERA_WH

        def nn(M, dst_wh, src_wh):
            x, y = np.meshgrid(np.arange(dst_wh[0], dtype=np.float32), np.arange(dst_wh[1], dtype=np.float32))
            M = M.astype(np.float32)
            sx, sy, sw = (M[i, 0] * x + M[i, 1] * y + M[i, 2] for i in range(3))
            u, v = np.rint(sx / sw), np.rint(sy / sw)
            bad = (sw <= 0) | (u < 0) | (u > src_wh[0] - 1) | (v < 0) | (v > src_wh[1] - 1)
            u = np.clip(u, 0, src_wh[0] - 1).astype(np.int64)
            v = np.clip(v, 0, src_wh[1] - 1).astype(np.int64)
            return (v * src_wh[0] + u).ravel(), bad.ravel()
        out = {}
        for name, f in ZP.rigs.OP_FOCAL.items():
            M = opf.get_warp_matrix(np.array([0.0, 0.0, np.radians(key)]), opf.intrinsics(w, h, f), name == "wide")
            y, ybad = nn(M, (opf.MODEL_W, opf.MODEL_H), (w, h))
            uv, qbad = nn(M * np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], np.float32), (opf.MODEL_W // 2, opf.MODEL_H // 2), (w // 2, h // 2))
            r, c = np.divmod(uv, w // 2)
            quad = np.stack([(2 * r + i) * w + 2 * c + j for i in (0, 1) for j in (0, 1)])
            out[name] = (y, quad, ybad, qbad)
        with self.rot_lock:
            self.rot_cache[key] = out
            if len(self.rot_cache) > 1024:
                self.rot_cache.popitem(last=False)
        return out

    def _pack_rot(self, raw, deg):
        """img2 (2, 6, 128, 256) of one buffered frame pair seen from a camera yawed by deg: pack() with the rotated gather
        indices. Source pixels outside the image are clipped to the border as in the native warp (_nn_index): blacking them
        (HUGSIM) also blacked a bottom row the native road frame clips, and the replay drifted from the native rollout."""
        if round(float(deg), 2) == 0.0:
            return np.stack([self.pack(raw[0], "road"), self.pack(raw[1], "wide")])
        idx, H, W = self._rot_idx(deg), self.opf.MODEL_H, self.opf.MODEL_W
        res = []
        for bgra, name in ((raw[0], "road"), (raw[1], "wide")):
            y_idx, quad = idx[name][:2]
            px = bgra.reshape(-1, 4)
            b, g, r = (px[y_idx, k].astype(np.float32) for k in range(3))
            Y = (16 + 0.257 * r + 0.504 * g + 0.098 * b).reshape(H, W)
            q = px[quad].astype(np.float32).mean(0)
            b, g, r = q[:, 0], q[:, 1], q[:, 2]
            U = 128 - 0.148 * r - 0.291 * g + 0.439 * b
            V = 128 + 0.439 * r - 0.368 * g - 0.071 * b
            Y, U, V = (np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in (Y, U, V))
            o = np.empty((6, H // 2, W // 2), np.uint8)
            o[0], o[1], o[2], o[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
            o[4], o[5] = U.reshape(H // 2, W // 2), V.reshape(H // 2, W // 2)
            res.append(o)
        return np.stack(res)

    def _lat_std4(self, m, raw):
        pl = raw[m.slices["plan"]]                         # MDN mu | log-std, (33, 15) each; column 1 = lateral position
        lat = np.exp(np.minimum(pl[pl.size // 2:], 11)).reshape(33, 15)[:, 1]
        return float(lat[self.t_idxs <= 4.0 + 1e-6].sum())

    def _selector(self, state, meta, prep, info, out):
        """Second rollout on the heading-aligned history; swaps the plan outputs when its lateral std is low enough."""
        yaw, speed = meta.get("yaw"), float(meta.get("speed", 0.0))
        if yaw is None or "raw" not in prep:
            info["sel"] = "noyaw"
            return
        buf = state["buf"]
        buf.append((prep["raw"], float(yaw), int(meta.get("desire", 0))))
        hist = max(abs(wrap(y - yaw)) for _, y, _ in list(buf)[-self.sel_n:])
        on = speed < self.sel_vmax and len(buf) > 1 and np.degrees(hist) >= 0.05
        if self.sel_check:
            on = len(buf) == buf.maxlen
        if not on:
            state["sel_ref"] = None
            info["sel"] = 0
            return
        m = state["sel"]
        m.extra = state["model"].extra                  # adapted model with an intent adapter: the same intent input
        traffic = (1, 0)
        t0 = time.perf_counter()
        rot = (lambda y: np.degrees(wrap(state["sel_ref"] - y))) if not self.sel_check else \
            (lambda y: self.sel_check_deg) if self.sel_check == 3 else (lambda y: 0.0)   # noqa: E731
        state["sel_age"] = state.get("sel_age", 0) + 1
        rebuilt = state["sel_ref"] is None or self.sel_check == 1 or \
            (abs(wrap(yaw - state["sel_ref"])) >= self.sel_rebuild and state["sel_age"] >= self.sel_gap)
        if rebuilt:
            state["sel_ref"], state["sel_age"] = float(yaw), 0
            m.reset()
            win = list(buf)[-self.sel_n:]
            if len(buf) > self.sel_n:            # the desire before the window: no spurious rising-edge pulse at the replay start
                m.prev_desire = np.zeros(8, np.float32)
                m.prev_desire[buf[0][2]] = 1
                m.prev_desire[0] = 0
            for i, (raw, y, d) in enumerate(win):
                img2 = self._pack_rot(raw, rot(y))
                dv = np.zeros(8, np.float32)
                dv[d] = 1
                for _ in range(1 + (self.sel_warm if i == 0 else 0)):
                    sraw = m.step(img2, desire=dv, traffic=traffic)
        else:
            raw, y, d = buf[-1]
            dv = np.zeros(8, np.float32)
            dv[d] = 1
            sraw = m.step(self._pack_rot(raw, rot(y)), desire=dv, traffic=traffic)
        s_std, n_std = self._lat_std4(m, sraw), self._lat_std4(state["model"], state["model"].last_raw)
        use = s_std < self.sel * n_std
        info.update(sel=1, sel_used=bool(use), sel_rebuilt=bool(rebuilt), sel_std=round(s_std, 3), sel_std_base=round(n_std, 3),
                    sel_ms=round(1e3 * (time.perf_counter() - t0), 1), sel_hist_deg=round(float(np.degrees(hist)), 2))
        if self.sel_check:
            sp = ZP_decode_plan(sraw[m.slices["plan"]])
            info["sel_check_dpos"] = float(np.abs(sp[:, 0:3] - ZP_decode_plan(state["model"].last_raw[m.slices["plan"]])[:, 0:3]).max())
        if use:
            d = self.decode(sraw, m.slices, speed)
            out.update(pos=d["plan_pos"].astype(np.float32), vel=d["plan_vel"][:, 0].astype(np.float32),
                       yaw=d["plan_yaw"].astype(np.float32), acc=ZP_decode_plan(sraw[m.slices["plan"]])[:, 6].astype(np.float32))
            info.update(curvature=d["curvature"], accel=d["accel"])

    def _lstab(self, state, meta, prep, info, out):
        """Launch stabilisation step: the stab session sees this frame from the launch heading; its lateral plan, rotated into
        the car frame, replaces the native lateral plan (lib/launch_stab.py)."""
        g, m = state["gate"], state["stab"]
        m.extra = state["model"].extra
        delta = g.delta(float(meta["yaw"]))              # car heading past the launch heading, right-positive
        dv = np.zeros(8, np.float32)
        dv[int(meta.get("desire", 0))] = 1
        sraw = m.step(self._pack_rot(prep["raw"], np.degrees(-delta)), desire=dv, traffic=(1, 0))
        d = self.decode(sraw, m.slices, float(meta.get("speed", 0.0)))
        sp, sy = LS.to_car(d["plan_pos"], d["plan_yaw"], delta)
        info.update(ls_delta=round(float(np.degrees(delta)), 2), ls_dlat3=round(float(np.interp(3.0, self.t_idxs, sp[:, 1] - out["pos"][:, 1])), 3),
                    curvature=d["curvature"])
        out.update(pos=LS.merge_lateral(out["pos"], sp), yaw=sy.astype(np.float32))

    def plan(self, state, meta, prep):
        if self.E is not None:                           # intent adapter input of this step: 0 unknown, 1 straight, 2 left, 3 right
            state["model"].extra = {"intent_bias": self.E[int(meta.get("intent", 0))][None]}
        g, mode = state.get("gate"), "off"
        if g is not None and meta.get("yaw") is not None and "raw" in prep:
            mode = g.step(float(meta.get("t", 0.0)), float(meta.get("speed", 0.0)), float(meta["yaw"]), int(meta.get("desire", 0)))
            if mode == "fork":                           # the native state after the last standstill step, before this one
                LS.fork_state(state["stab"], state["model"])
        info, out = super().plan(state, meta, prep)      # steps state["model"] with the route desire
        m = state["model"]
        raw = m.last_raw
        s = lambda k: raw[m.slices[k]]  # noqa: E731
        sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -30, 30)))  # noqa: E731
        plan = ZP_decode_plan(s("plan"))
        mt = sig(s("meta"))
        f32 = np.float32
        out.update(acc=plan[:, 6].astype(f32), lead=s("lead")[:72].reshape(3, 6, 4).astype(f32),
                   lead_prob=sig(s("lead_prob")).astype(f32), meta=mt.astype(f32),
                   desire_state=softmax(s("desire_state")).astype(f32),
                   desire_pred=softmax(s("desire_pred").reshape(4, 8)).astype(f32),
                   pose=s("pose")[:6].astype(f32), lane_prob=sig(s("lane_lines_prob"))[1::2].astype(f32))
        info.update({k: float(mt[v].max()) if k != "engaged" else float(mt[v][0]) for k, v in META.items()})
        if self.det is not None:
            d = self.det
            z = (raw[:d["mu"].shape[0]].astype(np.float32) - d["mu"]) / d["sd"]
            out["det"] = sig(d["W"] @ z + d["b"]).astype(f32)
        if self.lanes:
            # vmerge3 (experiments/vlm_arb/plans/2026-10-04-vmerge3.md): lane lines (4 x 33) and road edges (2 x 33), lateral y (m,
            # openpilot calib frame: right positive) and its std at X_IDXS = 192 (i / 32)^2 m ahead of the camera
            for k, n in (("lane_lines", 4), ("road_edges", 2)):
                if k in m.slices:
                    r = s(k)
                    h = r.shape[-1] // 2
                    out[k] = r[:h].reshape(n, 33, 2)[:, :, 0].astype(f32)
                    out[k + "_sd"] = np.exp(np.clip(r[h:].reshape(n, 33, 2)[:, :, 0], -10, 5)).astype(f32)
        if self.sel > 0:
            self._selector(state, meta, prep, info, out)
        if g is not None:
            info.update(ls=mode, ls_why=g.why)
            if mode != "off":
                self._lstab(state, meta, prep, info, out)
        if meta.get("twin"):
            t = state["twin"]
            traw = t.step(prep["img2"], desire=np.zeros(8, np.float32), traffic=(1, 0))
            tp = ZP_decode_plan(traw[t.slices["plan"]])
            out.update(twin_pos=tp[:, 0:3].astype(f32), twin_yaw=tp[:, 11].astype(f32), twin_vel=tp[:, 3].astype(f32),
                       twin_desire_pred=softmax(traw[t.slices["desire_pred"]].reshape(4, 8)).astype(f32))
        return info, out


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ZP_decode_plan(raw_plan):
    """MDN mean of the plan head: (33, 15) = pos 0:3, vel 3:6, acc 6:9, rot 9:12, rot rate 12:15."""
    return raw_plan[: raw_plan.shape[-1] // 2].reshape(33, 15)


def _patch_step():
    """OPModel.step returns the raw vector; keep the last one on the session so plan() can decode the extras without a
    second forward (the parent's plan() calls step() once)."""
    from jevdrive.openpilot.model import OPModel
    if getattr(OPModel, "_arb_patched", False):
        return
    step = OPModel.step

    def wrapped(self, *a, **k):
        self.last_raw = step(self, *a, **k)
        return self.last_raw
    OPModel.step, OPModel._arb_patched = wrapped, True


if __name__ == "__main__":
    _patch_step()
    ZP.OpenpilotModel = ArbModel      # main() builds OpenpilotModel(a) for every openpilot model name
    sys.exit(ZP.main())
