"""AP2: openpilot Cinque + the op_parity ego adapter + drivable hinge, trained for the inputs AlpaSim hands a nuPlan-track driver
(experiments/alpasim/lib/ap2_inputs.py; trainer scripts/ap2_train.py). Same network and serving path as SH30 (sh30_core.py); what differs:

  cold start  the rule the checkpoint was trained with, on decisions drawn with a rollout's own mix of m = 1..4 keyframes: `backwarp`
              (arm AB, chosen by the pilot: the missing slots are the first keyframe re-projected to back-extrapolated poses) or `zero`
              (arm A: slots older than the first keyframe are invalid)
  ego state   as AlpaSim's DynamicState defines it per decision (the driver passes it through; training built the same per m)
  command     the shipped samples' route rule, computed in training from AlpaSim's own route generator on navtrain; arm R (`route` in the
              checkpoint) also feeds the 20 route waypoints to the adapter's ego MLP (ap2_inputs.route_feat)

`ego_table` builds the training / offline ego features of logged tokens under that standard; `Core.plan` is one online decision.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ap2_inputs as AI  # noqa: E402
import sh30_core as C  # noqa: E402  (puts the repo, lib/ and op_parity/scripts on sys.path)
import parity_adapter as PA  # noqa: E402
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402


def ego_table(tab: dict, wp: np.ndarray, route: bool = False) -> np.ndarray:
    """Tokens of an op_parity tab (pose, vel, acc, fut) with their rebuilt AlpaSim routes wp (N, 4, 20, 2) (scripts/ap2_route.py) ->
    (N, 4, 20 [+ 60]) adapter ego features for a decision with m = 1..4 keyframes under the AlpaSim input standard."""
    w = AI.yaw_rates(tab["pose"], tab["fut"])
    out = []
    for m in (1, 2, 3, 4):
        P, V = AI.fill_history(tab["pose"], tab["vel"], w, m)
        v, a = AI.sim_state(tab["vel"][:, -1], tab["acc"][:, -1], w[:, 3], m)
        V[:, -1] = v
        e = PA.ego_features(P, V, np.repeat(a[:, None], 4, 1), AI.route_cmd(wp[:, m - 1]))
        out.append(np.concatenate([e, AI.route_feat(wp[:, m - 1])], -1) if route else e)
    return np.stack(out, 1).astype(np.float32)


def widen(model, route: bool):
    """Arm R: the adapter's ego MLP reads 60 more inputs (the route waypoints). The adapter output layer starts at zero either way."""
    if route:
        import torch.nn as nn
        lin = model.adapter.ego[0]
        model.adapter.ego[0] = nn.Linear(lin.in_features + AI.ROUTE_DIM, lin.out_features)
    return model


def load_model(tag: str, dev):
    """An op_parity / AP2 run tag -> (PModel in eval mode, route flag, the cold-start rule it was trained with; op_parity tags, trained
    on full history only: `backwarp`, what the SH30 driver serves them with)."""
    import torch
    import pp_train as T
    f = T.proot("runs", tag) / "ckpt-final.pt"
    ap = (torch.load(f, map_location="cpu", weights_only=False).get("ap2") or {}) if f.exists() else {}
    cold = ap.get("cold", "backwarp") if ap.get("std", "navsim") == "alpasim" else "backwarp"
    if not ap.get("route"):
        return T.load_pmodel(tag, dev), False, cold
    ck = torch.load(f, map_location="cpu", weights_only=False)
    m = widen(T.PModel(ck["model"]["arm"]), True).to(dev).eval()
    m.load_state(ck["model"])
    return m, True, cold


class Core(C.Core):
    def __init__(self, tag: str, dev: str = "cuda", cold: str = "", motion: float = 1.0, synth: str = "cpu"):
        """cold: "" = the rule the checkpoint was trained with; synth as sh30_core.Core."""
        import cv2
        import torch
        from jevdrive import op_adapt as A
        cv2.setNumThreads(1)
        assert synth in C.SYNTH, synth
        self.torch, self.A, self.tag, self.motion, self.synth = torch, A, tag, float(motion), synth
        self.dev = torch.device(dev)
        self.model, self.route, trained = load_model(tag, self.dev)
        self.cold = cold or trained
        assert self.cold in C.COLD, self.cold
        assert self.model.arm == "P2" and self.model.adapter is not None, f"{tag}: expected an ego-only parity arm, got {self.model.arm}"
        s = self.model.net.slices["plan"].start
        self.pi = slice(s, s + 33 * 15)

    def plan(self, keys, pose, vel, acc, cmd, cam_t, yaw_rate: float = 0.0, lht: bool = False, wp=None, motion: float | None = None) -> dict:
        """sh30_core.Core.plan with the route: wp (20, 2) rig-frame waypoints (NaN = padding), read only by a route arm."""
        torch, A = self.torch, self.A
        t0 = time.perf_counter()
        e = 4 - len(keys)
        P, V = C.fill_history(pose, vel, yaw_rate)
        K = C.stack_keys(keys)
        cam_t = np.asarray(cam_t, np.float64)
        track, gpu = I.track_navsim(*C.damp_history(P, V, self.motion if motion is None else motion)), self.synth == "gpu"
        cur, valid = C.lattice_gpu(K, e, track, cam_t, self.cold, self.dev) if gpu else C.lattice(K, e, track, cam_t, self.cold)
        prev = torch.cat([torch.zeros_like(cur[:1]), cur[:-1]]) if gpu else np.concatenate([np.zeros((1,) + C.FRAME, np.uint8), cur[:-1]])
        ego = PA.ego_features(P, V, np.tile(np.asarray(acc, np.float32), (4, 1)), np.asarray(cmd, np.float32))
        if self.route:
            ego = np.concatenate([ego, AI.route_feat(np.full((AI.N_WP, 2), np.nan) if wp is None else wp)])
        t1 = self._sync() if gpu else time.perf_counter()
        with torch.no_grad():
            p, c = (x[np.flatnonzero(valid)] if gpu else torch.from_numpy(x[valid]).to(self.dev) for x in (prev, cur))
            H = self.model.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *A.H_SHAPE)
            t2 = self._sync()
            tc = torch.tensor([[0.0, 1.0] if lht else [1.0, 0.0]], device=self.dev)
            out = self.model(H, torch.from_numpy(ego[None]).to(self.dev), tc).float()
            t3 = self._sync()
            mu = out[0, self.pi].reshape(33, 15).cpu().numpy()
            cur = cur.cpu().numpy() if gpu else cur
        poses = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")
        t4 = time.perf_counter()
        return {"poses": poses, "mu": mu, "ego": ego, "hist": P, "cur": cur, "valid": valid, "tokens": H[0],
                "ms": {"frames": 1e3 * (t1 - t0), "encode": 1e3 * (t2 - t1), "policy": 1e3 * (t3 - t2), "export": 1e3 * (t4 - t3)}}
