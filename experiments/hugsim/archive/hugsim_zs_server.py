#!/usr/bin/env python
"""Resident Alpamayo 1.5 / openpilot policy server for the HUGSIM zero-shot exam (experiments/hugsim/results/hugsim-exam-plan).

The CARLA server (scripts/zeroshot_policy_server.py) with two changes, both subclasses, nothing else touched:
the model images arrive already built by the HUGSIM agent (jevdrive.hugsim_zs: Alpamayo views (4 cams, 4 frames,
3, 320, 576) uint8, openpilot packed model frames (2, 6, 128, 256)), and the replies carry what the HUGSIM adapter
needs (Alpamayo headings; openpilot plan after `reps` steps with the traffic convention of the scene).
Protocol and threading as the CARLA server (scripts/zeroshot_wire.py, one connection per scenario).

    $DATA_DIR/third_party/alpamayo1.5/.venv/bin/python experiments/hugsim/archive/hugsim_zs_server.py alpamayo --socket <path>
    $DATA_DIR/envs/openpilot/bin/python experiments/hugsim/archive/hugsim_zs_server.py cinque --socket <path>
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("scripts",)]
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "scripts")]
import zeroshot_policy_server as S  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
import launch_stab as LS  # noqa: E402


class Alpamayo(S.Alpamayo):
    def prepare(self, meta, arrays):
        if "frames" not in arrays:                        # CARLA-shaped warm-up of the parent constructor
            return super().prepare(meta, arrays)
        torch = self.torch
        t0 = time.perf_counter()
        frames = torch.from_numpy(np.ascontiguousarray(arrays["frames"]))          # (4, 4, 3, 320, 576) uint8
        data = {"image_frames": frames, "camera_indices": self.cam_idx,
                "ego_history_xyz": torch.from_numpy(np.asarray(arrays["hist_xyz"], np.float32))[None, None],
                "ego_history_rot": torch.from_numpy(np.asarray(arrays["hist_rot"], np.float32))[None, None]}
        inputs = self.infer.build_inputs(data, self.processor, nav_text=meta.get("nav_text"))
        return {"inputs": inputs, "frames": frames, "prep_ms": 1e3 * (time.perf_counter() - t0)}

    def plan(self, state, meta, prep):
        torch, cfg = self.torch, self.cfg
        seed = int(meta.get("seed", 0))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            t0 = time.perf_counter()
            xyz, rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=prep["inputs"], top_p=cfg.top_p, temperature=cfg.temperature, num_traj_samples=1,
                num_traj_sets=1, max_generation_length=cfg.max_gen, return_extra=True,
                diffusion_kwargs={"inference_step": cfg.flow_steps})
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        xyz = xyz.reshape(-1, 64, 3)[0].float().cpu().numpy()
        rot = rot.reshape(-1, 64, 3, 3)[0].float().cpu().numpy()
        info = {"cot": str(np.asarray(extra["cot"]).ravel()[0]), "prep_ms": prep["prep_ms"], "infer_ms": 1e3 * wall}
        return info, {"xyz": xyz.astype(np.float32), "yaw": np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)}

    def finish(self, meta, prep, info, out):
        pass


class State(dict):
    """A connection's state; a dict subclass so a finalizer can return its ONNX session to the pool when the
    connection's thread drops it."""


class Openpilot(S.OpenpilotModel):
    def new_state(self, state=None):
        """Queued models (small / Cinque) own one ONNX session (TensorRT engine, ~2.3 GB, ~2 min to deserialize) per
        connection. HUGSIM opens one connection per scenario, so sessions are pooled: a connection takes a free one
        (reset to zero state) and gives it back when it closes."""
        if self.context_rate:
            return State(super().new_state(state))
        import threading
        import weakref
        if not hasattr(self, "pool"):
            self.pool, self.pool_lock = [self.model], threading.Lock()
        if isinstance(state, State) and "model" in state:
            state["model"].reset()
            return state
        with self.pool_lock:
            m = self.pool.pop() if self.pool else self.make()
        m.reset()
        st = State(model=m)
        weakref.finalize(st, self._give_back, m)
        return st

    def _give_back(self, m):
        with self.pool_lock:
            self.pool.append(m)

    def prepare(self, meta, arrays):
        if "img2" not in arrays:
            return super().prepare(meta, arrays)
        return {"img2": np.ascontiguousarray(arrays["img2"]), "img2s": arrays.get("img2s"), "prep_ms": 0.0}

    def stab_session(self, state):
        """Launch stabilisation (lib/launch_stab.py): a second pooled session per connection, taken on the first fork."""
        if "stab" not in state:
            with self.pool_lock:
                m = self.pool.pop() if self.pool else self.make()
            m.reset()
            state["stab"] = m
            import weakref
            weakref.finalize(state, self._give_back, m)
        return state["stab"]

    def plan(self, state, meta, prep):
        stab = meta.get("stab", "off")                 # launch stabilisation: "fork" | "on" | "off" (lib/launch_stab.py)
        if stab != "off" and not self.context_rate:
            sm = self.stab_session(state)
            if stab == "fork":                         # the native state after the last standstill step, before this one
                LS.fork_state(sm, state["model"])
        desire = np.zeros(8, np.float32)
        desire[int(meta.get("desire", 0))] = 1
        traffic = tuple(meta.get("traffic", (1, 0)))
        reps = int(meta.get("reps", 1))
        if "prev_desire" in meta and not self.context_rate:    # replay start: the desire of the step before the window
            pd = np.zeros(8, np.float32)
            pd[int(meta["prev_desire"])] = 1
            pd[0] = 0
            state["model"].prev_desire = pd
        t1 = time.perf_counter()
        if not self.context_rate:
            m = state["model"]
            for _ in range(reps):
                raw = m.step(prep["img2"], desire=desire, traffic=traffic)
        else:
            m = self.model
            for k in self.STATE:
                setattr(m, k, state[k])
            for _ in range(reps):
                raw = m.step(prep["img2"], desire=desire, traffic=traffic)
            for k in self.STATE:
                state[k] = getattr(m, k)
        sraw = None
        if stab != "off" and not self.context_rate:
            img2s = np.ascontiguousarray(prep["img2s"])
            for _ in range(reps):
                sraw = sm.step(img2s, desire=desire, traffic=traffic)
        d = self.decode(raw, m.slices, float(meta.get("speed", 0.0)))
        pl = raw[m.slices["plan"]]                         # MDN mu | log-std, (33, 15) each; column 1 = lateral position
        lat_std = np.exp(np.minimum(pl[pl.size // 2:], 11)).reshape(33, 15)[:, 1]
        info = {"infer_ms": 1e3 * (time.perf_counter() - t1), "curvature": d["curvature"], "accel": d["accel"],
                "lat_std4": float(lat_std[self.t_idxs <= 4.0 + 1e-6].sum()),
                "engaged": d["engaged"], "lead_prob": float(np.ravel(d["lead_prob"])[0])}
        out = {"pos": d["plan_pos"].astype(np.float32), "vel": d["plan_vel"][:, 0].astype(np.float32),
               "yaw": d["plan_yaw"].astype(np.float32), "t": self.t_idxs}
        if sraw is not None:
            ds = self.decode(sraw, sm.slices, float(meta.get("speed", 0.0)))
            out.update(stab_pos=ds["plan_pos"].astype(np.float32), stab_yaw=ds["plan_yaw"].astype(np.float32))
        return info, out

    def finish(self, meta, prep, info, out):
        pass


if __name__ == "__main__":
    S.Alpamayo, S.OpenpilotModel = Alpamayo, Openpilot     # the parent main() instantiates these module names
    sys.exit(S.main())
