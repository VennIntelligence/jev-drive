"""TTS: serve-side test-time candidate selection, no training (experiments/route_follow/plans/2026-10-16-tts-prereg.md).

Switch: JEV_TTS unset / 0 = off (the drivers do not import this module; serving is bit-identical to the code before it). JEV_TTS=1: at every
inference the driver scores K candidates of the model's own plan in one batched forward of the STACK consequence head (lib/conshead.py,
frozen, $DATA_DIR/runs/route_follow/stack/cons/head.pt) and serves the best one; the rest of the serving path (serve_fix, track_ref, ...)
then acts on the chosen plan as it acted on the model's.

Candidates (K = 16, CANDS order; candidate 0 = the plan itself). A plan is (8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in the t0 frame.
  lat d     d in {-1.0, -0.5, +0.5, +1.0} m along each pose's normal, ramped 0 -> d over 0 .. 2 s (lib/stack_cand.py "lat")
  gain g    g in {0.8, 1.2, 1.4}: every pose yaw and segment course about the ego x g at the same segment lengths (perturb.branch_poses(P, g, 1))
  arc s     s in {0.9, 1.1}: the same path re-sampled at s x each pose's arc length; s > 1 extends the path past its last pose along the last
            segment (straight, last yaw), so the 4 s pose moves too (branch_poses clamps there)
  g x s     the 6 pairs: gain first, then arc
Score (lower is better), per candidate k:
  S_k = w_road p_road_k + w_agent p_agent_k + w_route d_route_k + w_dev d_dev_k
  p_road / p_agent  the head's contact probability, max over the 8 bins (0 .. 4 s) of the road / agent channel
  d_route           mean over the 8 poses of the distance (m, capped at D_CAP) to the route polyline the model is fed (the valid points of the
                    route-memory features, either layout, in order; the polyline extended straight past both ends); 0 for every candidate
                    when the features have fewer than MIN_PTS valid points (no route term)
  d_dev             mean over the 8 poses of the xy distance to the model's own plan (m; 0 for candidate 0)
Choice: k* = argmin S; served = candidate k* only if S_0 - S_k* > margin (hysteresis towards the own plan), else candidate 0. No memory
across decisions. Head inputs as in training (rf_stack_c2.py): token slots conshead.SLOTS (t - 1.0 s, t) of the fed slots, the first 20
numbers of the fed ego row (raw; the head standardises), the candidate poses. A decision whose slot 2 or 7 is not valid serves the plan
(reason "slots"). Weights: a JSON {"w_road", "w_agent", "w_route", "w_dev", "margin"} (JEV_TTS_W, default the frozen fit committed at
experiments/route_follow/results/tts/weights.json).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import stack_cand as SC

ON = os.environ.get("JEV_TTS", "") not in ("", "0")
SUFFIX = "-tts" if ON else ""
LAT, GAIN, ARC = (-1.0, -0.5, 0.5, 1.0), (0.8, 1.2, 1.4), (0.9, 1.1)
CANDS = (("own", 0.0, 1.0, 1.0),) + tuple(("lat", d, 1.0, 1.0) for d in LAT) + tuple(("gain", 0.0, g, 1.0) for g in GAIN) \
    + tuple(("arc", 0.0, 1.0, s) for s in ARC) + tuple(("gain_arc", 0.0, g, s) for g in GAIN for s in ARC)
K = len(CANDS)
NAMES = tuple("own" if k == "own" else f"lat{d:+g}" if k == "lat" else f"g{g:g}" if k == "gain" else f"s{s:g}" if k == "arc" else f"g{g:g}s{s:g}"
              for k, d, g, s in CANDS)
TYPE = tuple(k for k, *_ in CANDS)
D_CAP, MIN_PTS = 5.0, 3
REPO = Path(__file__).resolve().parents[1]
W_DEFAULT = REPO / "experiments/route_follow/results/tts/weights.json"


def arc_np(P, s):
    """P (8, 3) re-sampled at s x each pose's arc length; past the last pose the path continues straight along the last segment."""
    if s <= 1.0:
        return SC.branch_np(P, 1.0, s)
    q = np.asarray(P, np.float64)
    xy = np.r_[[[0.0, 0.0]], q[:, :2]]
    ds = np.hypot(*np.diff(xy, axis=0).T)
    S = np.r_[0.0, np.cumsum(ds)]
    yaw = np.unwrap(np.r_[0.0, q[:, 2]])
    if S[-1] < 1e-6:
        return q.copy()
    j = np.flatnonzero(ds > 1e-6)[-1]                       # the last segment with length
    u = (xy[j + 1] - xy[j]) / ds[j]
    ext = (s - 1.0) * S[-1] + 1.0
    xy = np.r_[xy, [xy[-1] + ext * u]]
    S = np.r_[S, S[-1] + ext]
    yaw = np.r_[yaw, yaw[-1]]
    S = np.maximum.accumulate(S + 1e-9 * np.arange(len(S)))
    w = s * S[1:-1]
    return np.c_[np.interp(w, S, xy[:, 0]), np.interp(w, S, xy[:, 1]), np.interp(w, S, yaw)]


def cands_np(P):
    """(8, 3) -> (K, 8, 3) float64 in CANDS order."""
    P = np.asarray(P, np.float64)
    out = []
    for kind, d, g, s in CANDS:
        if kind == "own":
            out.append(P)
        elif kind == "lat":
            out.append(SC.one_np(P, "lat", d))
        else:
            Q = SC.branch_np(P, g, 1.0) if g != 1.0 else P
            out.append(arc_np(Q, s) if s != 1.0 else Q)
    return np.stack(out)


def route_pts(rf):
    """Route-memory features (DIM or DIM_O,) -> (n, 2) the valid points in slot order (rear-axle frame, m)."""
    f = np.asarray(rf, np.float32).reshape(-1)
    n = 40
    B = f[:n * 5].reshape(n, 5)
    v = B[:, 0] > 0.5
    return np.c_[B[v, 1] * 20.0, B[v, 2] * 20.0].astype(np.float64)


def route_dist(C, pts):
    """C (..., 8, 3) candidate poses, pts (n, 2) -> (...,) mean over poses of min(distance to the polyline extended past both ends, D_CAP);
    zeros when n < MIN_PTS."""
    C = np.asarray(C, np.float64)
    if pts is None or len(pts) < MIN_PTS:
        return np.zeros(C.shape[:-2])
    a, b = pts[:-1], pts[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(-1), 1e-12)
    X = C[..., None, :2]                                     # (..., 8, 1, 2)
    t = ((X - a) * d).sum(-1) / L2                           # (..., 8, m)
    lo = np.zeros(len(a))
    hi = np.ones(len(a))
    lo[0], hi[-1] = -np.inf, np.inf                          # the end segments extend straight
    t = np.clip(t, lo, hi)
    q = a + t[..., None] * d
    dist = np.sqrt(((X - q) ** 2).sum(-1)).min(-1)           # (..., 8)
    return np.minimum(dist, D_CAP).mean(-1)


def dev_dist(C):
    """C (..., K, 8, 3) -> (..., K) mean xy distance of each candidate to candidate 0."""
    C = np.asarray(C, np.float64)
    return np.sqrt(((C[..., :2] - C[..., :1, :, :2]) ** 2).sum(-1)).mean(-1)


def score(pr, pa, dr, dv, w):
    return w["w_road"] * pr + w["w_agent"] * pa + w["w_route"] * dr + w["w_dev"] * dv


def choose(S, margin):
    """S (..., K) -> index (...,): argmin unless it beats candidate 0 by no more than margin."""
    S = np.asarray(S, np.float64)
    k = S.argmin(-1)
    gain = S[..., 0] - np.take_along_axis(S, k[..., None], -1)[..., 0]
    return np.where(gain > margin, k, 0)


def segment(rf):
    """Route-memory features -> "junction" (the route bends >= 20 deg between -4 m and +8 m of the ego's projection), "approach" (else a turn
    >= 30 deg within 0 .. 40 m, rmem.turn_dist's rule), "straight", or "none" (fewer than MIN_PTS valid points)."""
    f = np.asarray(rf, np.float32).reshape(-1)
    B = f[:200].reshape(40, 5)
    v = B[:, 0] > 0.5
    if v.sum() < MIN_PTS:
        return "none"
    offs = -6.0 + 2.0 * np.arange(40)
    h = np.arctan2(B[:, 4], B[:, 3])
    i0, i1 = int(np.flatnonzero(offs == -4.0)[0]), int(np.flatnonzero(offs == 8.0)[0])
    if v[i0] and v[i1] and abs((h[i1] - h[i0] + np.pi) % (2 * np.pi) - np.pi) >= np.radians(20.0):
        return "junction"
    ref = h[int(np.flatnonzero(offs == 0.0)[0])] if v[int(np.flatnonzero(offs == 0.0)[0])] else 0.0
    dh = np.abs((h - ref + np.pi) % (2 * np.pi) - np.pi)
    c = v & (dh >= np.radians(30.0))
    c[:-2] &= c[2:] | ~v[2:]                                  # persistent over 4 m (rmem.turn_dist)
    c &= (offs >= 0.0) & (offs <= 40.0)
    return "approach" if c.any() else "straight"


class Selector:
    """The head on the driver's card + the frozen weights."""

    def __init__(self, dev, head=None, weights=None):
        import torch
        import conshead as CH
        self.torch, self.dev = torch, torch.device(dev)
        self.head = CH.load(head or os.environ.get("JEV_TTS_HEAD") or None, self.dev)
        wf = Path(weights or os.environ.get("JEV_TTS_W") or W_DEFAULT)
        self.w = {k: float(v) for k, v in json.loads(wf.read_text()).items() if k in ("w_road", "w_agent", "w_route", "w_dev", "margin")}
        assert len(self.w) == 5, f"{wf}: weights incomplete {self.w}"
        self.wfile = str(wf)
        self.slots = CH.SLOTS

    def probs(self, tokens, valid, ego20, C):
        """tokens (n_valid, 32, 512) of the fed slots, valid (8,), ego20 (20,), C (K, 8, 3) -> p_agent (K,), p_road (K,)."""
        torch, dev = self.torch, self.dev
        valid = np.asarray(valid, bool)
        idx = np.cumsum(valid) - 1                           # position of each slot among the fed (valid) ones
        sel = [int(idx[s]) for s in self.slots]
        with torch.no_grad():
            T = torch.as_tensor(tokens, device=dev) if not isinstance(tokens, torch.Tensor) else tokens
            F = T[sel].to(torch.float16)[None]                                   # (1, 2, 32, 512)
            e = torch.as_tensor(np.asarray(ego20, np.float32)[None], device=dev)
            q = torch.as_tensor(np.asarray(C, np.float32)[None], device=dev)
            p = torch.sigmoid(self.head(F, e, q)).amax(-1)[0].float().cpu().numpy()   # (K, 2)
        return p[:, 0], p[:, 1]

    def select(self, o, rf):
        """o: the core's plan output (poses, tokens, valid, ego / ego_fed); rf: the fed route-memory features or None.
        -> (served poses (8, 3), log record)."""
        t = time.perf_counter()
        P = np.asarray(o["poses"], np.float64)
        valid = np.asarray(o["valid"], bool)
        if not all(valid[s] for s in self.slots):
            return o["poses"], dict(k=0, name="own", reason="slots", ms=round(1e3 * (time.perf_counter() - t), 2))
        C = cands_np(P)
        ego = np.asarray(o.get("ego_fed", o["ego"]), np.float32)[:20]
        pa, pr = self.probs(o["tokens"], valid, ego, C)
        pts = route_pts(rf) if rf is not None else None
        dr = route_dist(C, pts)
        dv = dev_dist(C)
        S = score(pr, pa, dr, dv, self.w)
        k = int(choose(S, self.w["margin"]))
        info = dict(k=k, name=NAMES[k], reason="switch" if k else "own", gain=round(float(S[0] - S[k]), 4),
                    seg=segment(rf) if rf is not None else "none", route=int(pts is not None and len(pts) >= MIN_PTS),
                    S=np.round(S, 4).tolist(), pr=np.round(pr, 4).tolist(), pa=np.round(pa, 4).tolist(), dr=np.round(dr, 3).tolist(),
                    ms=round(1e3 * (time.perf_counter() - t), 2))
        served = C[k].astype(np.asarray(o["poses"]).dtype) if k else o["poses"]
        return served, info

    def warm(self, o):
        o = dict(o)
        o.setdefault("valid", np.ones(8, bool))
        self.select(o, None)


SEL = None


def load(dev):
    global SEL
    SEL = Selector(dev)
    return SEL


def selftest(n=200):
    """Candidate checks: own = the plan; lat / gain / arc <= 1 equal stack_cand's; arc 1.1 keeps the path (distance to the plan's polyline
    extension <= 1e-6 m) and its arc lengths are 1.1 x; route_dist of a plan on its own polyline is 0."""
    rng = np.random.default_rng(0)
    worst = dict(own=0.0, sc=0.0, arc_path=0.0, arc_len=0.0, route0=0.0)
    for i in range(n):
        v = rng.uniform(0.5, 15)
        arc = np.cumsum(np.full(8, 0.5 * v) * rng.uniform(0.6, 1.2, 8))
        th = np.linspace(0, rng.uniform(-1.0, 1.0), 8)
        P = np.c_[arc * np.cos(th), arc * np.sin(th), th]
        C = cands_np(P)
        worst["own"] = max(worst["own"], float(np.abs(C[0] - P).max()))
        for j, (kind, d, g, s) in enumerate(CANDS):
            if kind == "lat":
                ref = SC.one_np(P, "lat", d)
            elif kind == "gain":
                ref = SC.branch_np(P, g, 1.0)
            elif kind == "arc" and s < 1:
                ref = SC.branch_np(P, 1.0, s)
            else:
                continue
            worst["sc"] = max(worst["sc"], float(np.abs(C[j] - ref).max()))
        j = NAMES.index("s1.1")
        xy = np.r_[[[0.0, 0.0]], P[:, :2]]
        L = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        Q = C[j]
        ql = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], Q[:, :2]], axis=0).T))]
        worst["arc_len"] = max(worst["arc_len"], float(np.abs(ql[1:] - 1.1 * L[1:]).max()))
        worst["arc_path"] = max(worst["arc_path"], float(route_dist(Q[None], np.r_[[[0.0, 0.0]], P[:, :2]])[0]))
        worst["route0"] = max(worst["route0"], float(route_dist(P[None], np.r_[[[0.0, 0.0]], P[:, :2]])[0]))
    assert worst["own"] == 0 and worst["sc"] <= 1e-9 and worst["route0"] <= 1e-6, worst
    assert worst["arc_path"] <= 0.05, worst                 # chords of a curved plan vs its resampled arc
    return worst


if __name__ == "__main__":
    print("tts selftest", selftest(), "K", K, NAMES)
