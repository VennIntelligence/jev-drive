"""BODY1 arm 4.1 (plans/2026-10-10-body1-prereg.md, section 4.1 + Amendment 2): the S0 contact predictor in the AlpaSim driver as a stop.

  JEV_STOP=<m>   unset / 0 = off (the driver does not import this module and returns what it returned before, bit for bit).
                 <m> > 0: at every `drive` call, after serve_fix.apply, the served plan is scored by the contact head (lib/contact_head.py,
                 arch `step`, the two full-scale seeds; score = mean of the two agent-contact logits, the gated score of results/s0_gate.md).
                 Flag = score >= THR (logit 0.408, p = 0.60: 2 % of the clean own-plan decisions of body1-hold-logs). A flagged plan keeps
                 its path and is re-timed so that the rear axle stops <m> metres before the predicted first-contact arc length.

The rule (every constant fixed in Amendment 2 from hold logs / the controller, before any closed-loop score was read):
  contact arc    s_c = min(regressed first-contact arc length, arc length at the start of the first 0.5 s interval of the plan whose
                 predicted clearance is under C_STEP = 0.25 m), both the two-seed mean. The regression alone is unbiased over all true
                 positives but over-estimates near contacts (true arc < 3 m: median +1.4 m on hold logs); the interval outputs pull those in.
  stop point     d = max(s_c - m, 0) along the served path.
  speed profile  v(t) = min(plan's own speed, sqrt(2 a (D - s(t)))): the plan's profile under a constant-deceleration envelope that
                 ends at D. D = max(d, v0^2 / (2 A_CAP)) and a = clip(v0^2 / (2 D), A_MIN, A_CAP). So: when the stop point is reachable at
                 or under A_CAP the reference is a constant deceleration from the ego's own speed to a standstill at d (speed-continuous,
                 so positions 1.0 .. 2.0 s ahead, the only part AlpaSim's MPC tracks, ask for exactly that deceleration); when it is not,
                 the reference brakes at A_CAP = 6 m/s^2 (two thirds of the controller's -9 m/s^2 limit; the organisers report zig-zag
                 steering for braking references beyond what the vehicle can do) and stops past d; when the stop point is far
                 (v0^2 / (2 d) < A_MIN = 1 m/s^2) the plan is followed until the 1 m/s^2 envelope reaches it, so a standing or slow ego
                 may still roll up to the stop point. Speed is only removed, never added; the path is never changed.
  at standstill  no minimum speed: a flagged plan of a standing ego is held at min(plan, envelope), i.e. the ego stays or creeps to d.
  no latch       each decision stands alone. When the flag clears the plan is served unchanged from that decision on.

Every decision is logged by the driver as `body` in drive.jsonl: p, the two logits, flag, v0, ms; when flagged also s_reg, s_step, d, D, a,
cut (metres removed from the first 2 s) and plan (the poses before re-timing).
No simulator state, no boxes, no map: inputs are the vision tokens, the ego vector and the plan of the same forward pass.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parents[1] / "alpasim" / "lib"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
import serve_fix as FX  # noqa: E402

THR = 0.4079687500000091                    # two-seed mean agent logit: 2 % flags on clean own-plan hold decisions (results/s0/stop.json)
C_STEP, A_MIN, A_CAP = 0.25, 1.0, 6.0       # m, m/s^2, m/s^2 (module docstring)
S_SCALE = 10.0                              # contact_head.S_SCALE: the arc-length output is in units of 10 m
CKPTS = ("full-step/20261010-024906/ckpt.pt", "full-step/20261010-024909/ckpt.pt")      # seed 0, seed 1 under $DATA_DIR/runs/body1/s0

M = float(os.environ.get("JEV_STOP", "0") or 0)
ON = M > 0
SUFFIX = f"-stop{M:g}" if ON else ""
BODY = None


def stop_point(out, step, poses, m: float):
    """Head outputs of one plan (out (S, 7), per-interval clearance (S, 9) for S seeds) and the plan poses (8, 3)
    -> (s_reg, s_step (inf when no interval is under C_STEP), d)."""
    xy = np.concatenate([np.zeros((1, 2)), np.asarray(poses, np.float64)[:, :2]])
    node = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    s_reg = max(float(np.mean(out[:, 2])) * S_SCALE, 0.0)
    low = np.flatnonzero(np.mean(step, 0) < C_STEP)
    s_step = float(node[max(low[0] - 1, 0)]) if len(low) else float("inf")
    return s_reg, s_step, max(min(s_reg, s_step) - m, 0.0)


def retime(poses, v0: float, d: float):
    """The plan's path under the stop envelope (module docstring) -> (poses (8, 3), a, D, metres removed from the first 2 s)."""
    p, ds, vp = FX.path(poses)
    v0 = max(float(v0), 0.0)
    D = max(d, v0 * v0 / (2 * A_CAP))
    a = float(np.clip(v0 * v0 / (2 * max(D, 1e-6)), A_MIN, A_CAP))
    v, s = np.empty_like(vp), 0.0
    for i, x in enumerate(vp):
        left = max(D - s, 0.0)
        v[i] = min(x, np.sqrt(2 * a * left), left / FX.DT)
        s += v[i] * FX.DT
    return FX.place(p, ds, v), a, D, float(np.sum((vp - v)[:4 * FX.K5]) * FX.DT)


class Body:
    """The two-seed contact head on the driver's card."""

    def __init__(self, dev, ckpts=None):
        import torch
        import contact_head as CH
        root = Path(os.environ.get("DATA_DIR", "/nonexistent")) / "runs/body1/s0"
        ckpts = ckpts or [x for x in os.environ.get("JEV_STOP_CKPT", "").split(",") if x] or [root / c for c in CKPTS]
        self.torch, self.dev, self.nets = torch, torch.device(dev), []
        for c in ckpts:
            net, ck = CH.load_ckpt(c, self.dev)
            assert ck["config"]["arch"] == "step", ck["config"]
            self.nets.append((net, np.asarray(ck["emu"], np.float32), np.asarray(ck["esd"], np.float32)))

    def score(self, tokens, valid, ego, poses):
        """The gate's input standard (bd1_gate.py predict): fp16 tokens of the valid slots scattered into the 8 slots, the slot mask,
        the ego vector standardised per checkpoint, the plan as the only query. -> out (S, 7), per-interval clearance (S, 9), numpy."""
        torch, dev = self.torch, self.dev
        valid = np.asarray(valid, bool)
        with torch.no_grad():
            V = torch.zeros((1, len(valid), 32, 512), dtype=torch.float16, device=dev)
            V[0, torch.as_tensor(np.flatnonzero(valid), device=dev)] = torch.as_tensor(tokens, device=dev).to(torch.float16)
            vm = torch.as_tensor(valid[None], device=dev)
            q = torch.as_tensor(np.asarray(poses, np.float32)[None, None], device=dev)
            O, S = [], []
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                for net, emu, esd in self.nets:
                    o, st = net(V, vm, torch.as_tensor(((np.asarray(ego, np.float32) - emu) / esd)[None], device=dev), q)
                    O.append(o[0, 0]), S.append(st[0, 0, :, 0])
            return torch.stack(O).float().cpu().numpy(), torch.stack(S).float().cpu().numpy()


def load(dev):
    global BODY
    BODY = Body(dev)
    return BODY


def warm(o):
    """One head pass on a plan output (the driver warms every gRPC worker thread with it)."""
    BODY.score(o["tokens"], o["valid"], o["ego"], o["poses"])
    return o


def apply(lock, o: dict, v0: float):
    """Driver hook, after serve_fix.apply: o["poses"] (the served plan) is scored; when flagged it becomes the stop trajectory and
    o["poses_plan"] keeps what would have been served. `lock` = the driver's card lock. -> the log record."""
    t = time.perf_counter()
    with lock:
        out, step = BODY.score(o["tokens"], o["valid"], o["ego"], o["poses"])
    z = float(out[:, 0].mean())
    info = {"p": round(1 / (1 + np.exp(-z)), 4), "z": [round(float(x), 3) for x in out[:, 0]], "flag": bool(z >= THR), "v0": round(float(v0), 3)}
    if info["flag"]:
        s_reg, s_step, d = stop_point(out, step, o["poses"], M)
        poses, a, D, cut = retime(o["poses"], v0, d)
        info.update(s_reg=round(s_reg, 2), s_step=None if not np.isfinite(s_step) else round(s_step, 2), d=round(d, 3), D=round(D, 3), a=round(a, 3),
                    cut=round(cut, 3), plan=np.asarray(o["poses"]).round(4).tolist())
        o["poses_plan"], o["poses"] = o["poses"], poses
    info["ms"] = round(1e3 * (time.perf_counter() - t), 2)
    return info
