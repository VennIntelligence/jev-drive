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

Arm 4.2 (section 4.2 + Amendment 3): the same head as a lateral re-plan.

  JEV_REPLAN=1   unset / 0 = off (bit-identical as above). On (not together with JEV_STOP): at every `drive` call the served plan and its
                 lateral ramps are scored in one batched head pass (two seeds, mean logit of agent contact `za` and boundary contact `zb`).
  candidates     the row generator's `lat` family (scripts/bd1_rows.py perturb): offset a * s(t) / s(4 s) along the path normal, yaw + atan(a / s(4 s)),
                 a in +-{0.3, 0.6, 0.9, 1.2, 1.5} m (+ = left). A candidate exists only while |a| <= R_CAP x the plan's 4 s arc (R_CAP = 0.10: the
                 ramp's heading offset stays under 5.7 deg; the generator's own cap is 0.25), so a slow or standing ego has few or none.
                 The timing along the path is the plan's: speed is not changed.
  flagged        za(plan) >= FA or zb(plan) >= FB.            clear   a candidate with za < CA and zb < CB (CA <= FA, CB <= FB).
  choice         flagged and some candidate clear -> the clear candidate of the smallest |a|; the two sides of one size are ordered by
                 max(za - CA, zb - CB), lower first. Not flagged -> the plan. Flagged and none clear -> the plan, unchanged (no stop).
  cold start     decision 0 of a session (one real frame, no motion in the slots) is never re-planned and leaves the memory untouched.
  memory         per session: the side of the last served shift. While fewer than IDLE = 2 decisions have been served unchanged since, only
                 candidates of that side are eligible (a shift cannot change side from one decision to the next; the other side opens again
                 after 1 s of unchanged plans). Sizes are not carried: every decision re-plans from the state the ego has reached.
Every decision is logged as `body` in drive.jsonl: rp, k, za / zb of [plan, candidates], ok, flag ("", "a", "b", "ab"), a (served shift, 0 = plan),
reason (cold | clean | replan | none_clear), side (memory before the decision), v0, ms; when re-planned also plan (the poses before the shift).
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
RP = os.environ.get("JEV_REPLAN", "0") not in ("", "0")
assert not (M > 0 and RP), "JEV_STOP and JEV_REPLAN are separate arms: set one"
ON = M > 0 or RP
SUFFIX = f"-stop{M:g}" if M > 0 else ("-rp1" if RP else "")
BODY = None

# arm 4.2 (module docstring; thresholds = two-seed mean logits, fixed on body1-hold-logs in Amendment 3: results/replan/hold_*.csv)
A_RP = np.array([s_ * a_ for a_ in (0.3, 0.6, 0.9, 1.2, 1.5) for s_ in (1.0, -1.0)])      # m at 4 s, + = left
R_CAP, IDLE = 0.10, 2
THR_RP = None                                                                              # (FA, FB, CA, CB); set by Amendment 3


def ramps(P, A=A_RP):
    """Plans P (..., 8, 3) -> lateral-ramp candidates (..., len(A), 8, 3) float64 and their existence mask (..., len(A))."""
    P = np.asarray(P, np.float64)
    xy = np.concatenate([np.zeros_like(P[..., :1, :2]), P[..., :2]], -2)
    s = np.cumsum(np.hypot(np.diff(xy[..., 0], axis=-1), np.diff(xy[..., 1], axis=-1)), -1)        # (..., 8)
    s4 = np.maximum(s[..., -1:], 1e-3)
    off = A[:, None] * (s / s4)[..., None, :]                                                      # (..., m, 8)
    x, y, yaw = (P[..., None, :, i] for i in range(3))
    return np.stack([x - off * np.sin(yaw), y + off * np.cos(yaw), yaw + np.arctan2(A, s4)[..., None]], -1), np.abs(A) <= R_CAP * s[..., -1:]


def pick(za, zb, ok, thr, side=0, A=A_RP):
    """The choice rule on (n, 1 + m) logits of [plan, candidates], ok (n, m), side (n,) or scalar in {-1, 0, +1} (0 = both sides eligible)
    -> (index into A of the served candidate, -1 = the plan (n,); agent flag (n,); boundary flag (n,); any eligible clear candidate (n,))."""
    FA, FB, CA, CB = thr
    za, zb = np.atleast_2d(za), np.atleast_2d(zb)
    fa, fb = za[:, 0] >= FA, zb[:, 0] >= FB
    sg = np.sign(A)
    side = np.broadcast_to(np.asarray(side), fa.shape)[:, None]
    clear = np.atleast_2d(ok) & (za[:, 1:] < CA) & (zb[:, 1:] < CB) & ((side == 0) | (sg == side))
    key = np.where(clear, np.abs(A) + 0.01 / (1 + np.exp(-np.maximum(za[:, 1:] - CA, zb[:, 1:] - CB))), np.inf)
    has = clear.any(1)
    return np.where((fa | fb) & has, key.argmin(1), -1), fa, fb, has


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

    def score_many(self, tokens, valid, ego, Q):
        """As `score` for Q (q, 8, 3) queries of one scene in one batched pass -> two-seed mean (agent logit (q,), boundary logit (q,))."""
        torch, dev = self.torch, self.dev
        valid = np.asarray(valid, bool)
        with torch.no_grad():
            V = torch.zeros((1, len(valid), 32, 512), dtype=torch.float16, device=dev)
            V[0, torch.as_tensor(np.flatnonzero(valid), device=dev)] = torch.as_tensor(tokens, device=dev).to(torch.float16)
            vm = torch.as_tensor(valid[None], device=dev)
            q = torch.as_tensor(np.asarray(Q, np.float32)[None], device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                O = [net(V, vm, torch.as_tensor(((np.asarray(ego, np.float32) - emu) / esd)[None], device=dev), q)[0][0, :, :2] for net, emu, esd in self.nets]
            z = torch.stack(O).float().mean(0).cpu().numpy()
        return z[:, 0], z[:, 1]


def load(dev):
    global BODY
    BODY = Body(dev)
    return BODY


def warm(o):
    """One head pass on a plan output (the driver warms every gRPC worker thread with it)."""
    if RP:
        BODY.score_many(o["tokens"], o["valid"], o["ego"], np.concatenate([np.asarray(o["poses"], np.float64)[None], ramps(o["poses"])[0]]))
    else:
        BODY.score(o["tokens"], o["valid"], o["ego"], o["poses"])
    return o


def replan(lock, o: dict, v0: float, mem: dict, k: int, thr=None):
    """Arm 4.2 (module docstring). mem = the session's memory {"side", "idle"}; k = decision index of the session. -> the log record."""
    t = time.perf_counter()
    P = np.asarray(o["poses"])
    cand, ok = ramps(P)
    with lock:
        za, zb = BODY.score_many(o["tokens"], o["valid"], o["ego"], np.concatenate([P[None].astype(np.float64), cand]))
    side = mem.get("side", 0) if mem.get("idle", IDLE) < IDLE else 0
    j, fa, fb, has = (int(x[0]) for x in pick(za[None], zb[None], ok[None], thr or THR_RP, side))
    flagged = bool(fa or fb)
    reason = "cold" if k == 0 else "clean" if not flagged else "replan" if j >= 0 else "none_clear"
    info = {"rp": 1, "k": int(k), "za": np.round(za, 3).tolist(), "zb": np.round(zb, 3).tolist(), "ok": ok.astype(int).tolist(), "flag": "a" * fa + "b" * fb,
            "a": float(A_RP[j]) if reason == "replan" else 0.0, "reason": reason, "side": int(side), "v0": round(float(v0), 3)}
    if reason == "replan":
        info["plan"] = P.round(4).tolist()
        o["poses_plan"], o["poses"] = P, cand[j].astype(P.dtype)
        mem.update(side=int(np.sign(A_RP[j])), idle=0)
    elif reason != "cold":
        mem["idle"] = mem.get("idle", IDLE) + 1
    info["ms"] = round(1e3 * (time.perf_counter() - t), 2)
    return info


def apply(lock, o: dict, v0: float, sess=None, k: int = 0):
    """Driver hook, after serve_fix.apply: o["poses"] (the served plan) is scored; when flagged it becomes the stop trajectory and
    o["poses_plan"] keeps what would have been served. `lock` = the driver's card lock. -> the log record.
    With JEV_REPLAN the call is `replan` with the session's memory (kept on the driver's session object `sess`)."""
    if RP:
        return replan(lock, o, v0, sess.__dict__.setdefault("body_mem", {}), k)
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
