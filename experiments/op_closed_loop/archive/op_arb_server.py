#!/usr/bin/env python
"""openpilot policy server for the closed-loop integration study (research/openpilot-closedloop-integration.md,
fc65452:todos/2026-09-28-op-closedloop.md). envs/openpilot. Agent side: experiments/op_closed_loop/lib/op_arb_agent.py.

Same model path as scripts/zeroshot_policy_server.py (Cinque at 20 Hz, road + wide warped as modeld, desire as a
rising-edge pulse; the class is imported, not copied), plus two things the arbitration and the diagnosis need:

  extras  every head the network already outputs, decoded as openpilot does: plan speed / accel along the plan, the
          lead head (3 leads x 6 times x (x, y, v, a), lead_prob), meta (engaged, disengage / hard-brake / gas-press /
          brake-press / blinker probabilities), desire_state, desire_pred (4 x 8), pose (vision ego motion), lane-line
          probabilities. None of these is an extra model; they are outputs the plan is computed alongside.
  twin    (meta "twin": true) a second session of the same model stepped on the same frames with desire 0, so each
          plan has a desire-free counterpart: the turn diagnosis compares the two plans frame by frame.

    CUDA_VISIBLE_DEVICES=6 $DATA_DIR/envs/openpilot/bin/python experiments/op_closed_loop/archive/op_arb_server.py cinque --pool 4 \
        --backend cuda-iob --socket S
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("scripts",)]
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3] / "scripts"
sys.path[:0] = [str(HERE), str(HERE.parent)]
import zeroshot_policy_server as ZP  # noqa: E402

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
        a.pool = (2 if self.twin else 1) * a.pool   # a twin session per connection (none with --no-twin)
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
        keys = ("model", "twin") if self.twin else ("model",)
        if state and "model" in state:
            for k in keys:
                state[k].reset()
            return state
        take = lambda: self.free.pop() if self.free else self.make()  # noqa: E731
        ss = {k: take() for k in keys}
        for m in ss.values():
            m.reset()
        if state is None:
            self._warm = list(ss.values())
        return ss

    def release(self, state):
        if state:
            self.free += [state[k] for k in ("model", "twin") if k in state]

    def plan(self, state, meta, prep):
        if self.E is not None:                           # intent adapter input of this step: 0 unknown, 1 straight, 2 left, 3 right
            state["model"].extra = {"intent_bias": self.E[int(meta.get("intent", 0))][None]}
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
        if meta.get("twin"):
            t = state["twin"]
            traw = t.step(prep["img2"], desire=np.zeros(8, np.float32), traffic=(1, 0))
            tp = ZP_decode_plan(traw[t.slices["plan"]])
            out.update(twin_pos=tp[:, 0:3].astype(f32), twin_yaw=tp[:, 11].astype(f32), twin_vel=tp[:, 3].astype(f32),
                       twin_desire_pred=softmax(traw[t.slices["desire_pred"]].reshape(4, 8)).astype(f32))
        return info, out


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
