"""Bench2Drive agent with VLM Arbitration Table (R1-R5) and openpilot Cinque.

Integrates VLM slow-channel categorical decisions into op-drive arbitration layer.
Protocol: experiments/vlm_arb/plans/2026-10-02-vlm-arb.md.
"""
import sys
import os
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Ensure bare imports from lib and scripts work
_p = Path(__file__).resolve()
REPO = _p.parent if (_p.parent / "jevdrive").exists() else (_p.parents[1] if (_p.parents[1] / "jevdrive").exists() else _p.parents[3])
for _d in (str(REPO / "lib"), str(REPO / "experiments/b2d_privileged/lib"), str(REPO / "experiments/op_closed_loop/lib"), str(REPO / "scripts")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import b2d_zeroshot_agent as Z
import zeroshot_rigs as rigs
from op_arb_agent import OpArbAgent, TIMES, place, idm, REAR_TO_BUMPER, arc
from b2d_privileged_geometry import Privileged, project, ego_boxes, overlap, rectangle_gap, visibility
from vlm_protocol import parse_vlm_response
from vlm_client import VLMClient, DummyVLMClient


def get_entry_point():
    return "VlmArbAgent"


# Fixed pre-registered arbitration parameters
ARB_PARAMS = {
    "N_jslow_m": 25.0,        # R1: Distance before junction entrance to apply speed cap
    "v_jslow": 4.5,           # R1: Junction target speed limit (m/s)
    "K_debounce": 2,          # R2/R3/R4: Consecutive identical answers required to trigger / release
    "T_stop_sign_s": 2.5,     # R3: Stop duration at stop sign (s)
    "T_max_stop_s": 25.0,     # R5: Maximum standstill holding duration before fallback (s)
    "TTL_answer_s": 2.5,      # R5: VLM answer freshness timeout (s)
    "sim_delay_L_s": 0.5,     # Latency simulation: answers take effect after t + L (s)
    "idm_s0": 2.5,
    "idm_T": 1.2,
    "idm_b": 2.0,
    "idm_amax": 1.5,
    "cruise_base": 8.0,
}


class VlmArbAgent(OpArbAgent):
    """OpArbAgent extended with VLM-based slow-channel arbitration table."""

    def __init__(self, path_to_conf_file):
        super().__init__(path_to_conf_file)
        
        # Arm mode: 'drive', 'dslow', 'jslow', 'vred', 'vbyp', 'vall', 'shadow'
        self.vlm_arm = os.environ.get("VLM_ARM", "drive")
        self.vlm_shadow = os.environ.get("VLM_SHADOW", "0") == "1" or self.vlm_arm == "shadow"
        self.save_frames = os.environ.get("VLM_SAVE_FRAMES", "0") == "1"
        self.frame_save_interval = float(os.environ.get("VLM_FRAME_INTERVAL", "0.5")) # simulation seconds
        
        # Privileged geometry helper for obstacle bypass geometry and ground-truth logging
        # Privileged object requires an arm in ("drive", "pjunc", "pbyp", "pbypgap", "pred", "pall")
        priv_arm = "drive"
        if self.vlm_arm == "vbyp":
            priv_arm = "pbyp"
        elif self.vlm_arm == "vall":
            priv_arm = "pall"
        self.priv = Privileged(self, priv_arm)
        
        # VLM client setup
        endpoint = os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:8080/v1/systemone")
        self.vlm_client = VLMClient(endpoint=endpoint) if os.environ.get("VLM_DUMMY", "0") != "1" else DummyVLMClient()
        
        # Delay queue: stores (t_available, vlm_answer_dict)
        self.answer_queue = deque()
        self.current_vlm_answer = None
        self.last_vlm_query_sim_t = -1e9
        self.last_frame_save_sim_t = -1e9
        
        # Debounce counters for decisions
        self.light_history = deque(maxlen=ARB_PARAMS["K_debounce"])
        self.sign_history = deque(maxlen=ARB_PARAMS["K_debounce"])
        self.block_history = deque(maxlen=ARB_PARAMS["K_debounce"])
        
        # State tracking for rules
        self.r2_holding_red = False
        self.r3_stop_sign_completed_junctions = set()
        self.r3_stopped_since = None
        self.r3_holding_stop = False
        self.r4_bypassing = False
        self.r5_fallback_active = False
        self.stop_started_sim_t = None
        
        # Log file for VLM decisions and ground truth labels
        out_root = Path(self.out) if hasattr(self, "out") and self.out else Path("/tmp/vlm_arb_run")
        out_root.mkdir(parents=True, exist_ok=True)
        self.vlm_log_path = out_root / "vlm_decisions.jsonl"
        self.vlm_log = self.vlm_log_path.open("w", buffering=1)
        self.frame_dir = out_root / "vlm_frames"
        if self.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def _query_vlm(self, sim_t, rgb_frames):
        """Send camera frames to VLM and enqueue result with simulated delay L."""
        # Non-blocking / synchronous query
        res = self.vlm_client.query(rgb_frames, state_desc=f"CARLA ego at t={sim_t:.2f}s")
        # Answer only becomes active in simulation at sim_t + L
        t_effective = sim_t + ARB_PARAMS["sim_delay_L_s"]
        res["sim_t_queried"] = sim_t
        res["sim_t_effective"] = t_effective
        self.answer_queue.append((t_effective, res))

    def _update_vlm_answer(self, sim_t):
        """Pop all answers whose effective simulation time has arrived."""
        while self.answer_queue and self.answer_queue[0][0] <= sim_t:
            _, ans = self.answer_queue.popleft()
            self.current_vlm_answer = ans
            # Update debounce buffers
            self.light_history.append(ans.get("Q_light", "no_light"))
            self.sign_history.append(ans.get("Q_sign", "no"))
            self.block_history.append((ans.get("Q_block", "clear"), ans.get("Q_side", "none_free")))

    def _extract_ground_truth(self, sim_t, ego_s, next_junc_dist):
        """Extract ground-truth state for shadow evaluation."""
        ctx = self._ctx() if hasattr(self, "_ctx") else {}
        tl_state = ctx.get("tl") # 0: green, 1: yellow, 2: red
        tl_dist = ctx.get("tl_dist", 999.0)
        
        # Ground truth Q_light label
        if tl_dist < 50.0 and tl_state in (1, 2):
            gt_light = "red_or_yellow_for_ego"
        elif tl_dist < 50.0 and tl_state == 0:
            gt_light = "green_for_ego"
        else:
            gt_light = "no_light"
            
        # Ground truth Q_sign label
        # In CARLA B2D, stop sign is present if non-signalized junction is within 25m
        gt_sign = "yes" if next_junc_dist < 25.0 and tl_dist > 900.0 else "no"
        
        # Ground truth obstacles
        obstacles = self.priv.meta.get("obstacles", []) if hasattr(self.priv, "meta") else []
        if obstacles:
            gt_block = "static_block"
            gt_side = "left_free" if self.priv.meta.get("borrow") else "right_free"
        else:
            gt_block = "clear"
            gt_side = "none_free"
            
        return {
            "gt_light": gt_light,
            "tl_state": tl_state,
            "tl_dist": tl_dist,
            "gt_sign": gt_sign,
            "gt_block": gt_block,
            "gt_side": gt_side,
            "next_junc_dist": next_junc_dist
        }

    def _next_junction_info(self, ego_s):
        """Find next junction entrance along the route using map waypoints."""
        r = self.route
        if not hasattr(self.priv, "flags") or len(self.priv.flags) == 0:
            return 999.0, None, -1
        
        ahead_indices = np.flatnonzero(self.priv.flags & (r.s >= ego_s - 1.0) & (r.s <= ego_s + 100.0))
        if len(ahead_indices) == 0:
            return 999.0, None, -1
        
        junc_entry_idx = ahead_indices[0]
        junc_entry_s = float(r.s[junc_entry_idx])
        junc_dist = junc_entry_s - ego_s
        jid = int(self.priv.jids[junc_entry_idx]) if hasattr(self.priv, "jids") else 0
        return junc_dist, junc_entry_s, jid

    def _apply_arbitration_table(self, speed, sim_t, ego_s, next_junc_dist, next_junc_s, next_jid):
        """Evaluate Arbitration Table rules R1 - R5."""
        out = {}
        release = False
        active_rules = []
        K = ARB_PARAMS["K_debounce"]

        # Helper for IDM stop target
        def idm_stop(dist_to_stop):
            return idm(speed, max(0.1, dist_to_stop + ARB_PARAMS["idm_s0"]), 0.0, 0.0,
                       ARB_PARAMS["cruise_base"], ARB_PARAMS["idm_amax"], ARB_PARAMS["idm_b"],
                       ARB_PARAMS["idm_s0"], ARB_PARAMS["idm_T"])

        # Check VLM answer validity and TTL
        vlm_valid = False
        ans = self.current_vlm_answer
        if ans and (sim_t - ans.get("sim_t_effective", 0.0)) <= ARB_PARAMS["TTL_answer_s"]:
            vlm_valid = True

        # Track standstill duration for R5
        if speed < 0.2:
            if self.stop_started_sim_t is None:
                self.stop_started_sim_t = sim_t
        else:
            self.stop_started_sim_t = None

        # R5: Fallback if stopped for too long or VLM expired while holding stop
        stopped_duration = (sim_t - self.stop_started_sim_t) if self.stop_started_sim_t is not None else 0.0
        if (self.r2_holding_red or self.r3_holding_stop) and (stopped_duration > ARB_PARAMS["T_max_stop_s"] or not vlm_valid):
            self.r5_fallback_active = True
            self.r2_holding_red = False
            self.r3_holding_stop = False
            release = True
            active_rules.append("R5_fallback")

        # R1: `jslow` - Junction speed cap (Map-based, independent of VLM)
        enable_r1 = self.vlm_arm in ("jslow", "vall")
        cruise_cap = ARB_PARAMS["cruise_base"]
        if enable_r1 and next_junc_dist <= ARB_PARAMS["N_jslow_m"] and next_junc_dist >= -10.0:
            cruise_cap = min(cruise_cap, ARB_PARAMS["v_jslow"])
            active_rules.append("R1_jslow")

        # R2: Traffic Light (VLM-based)
        enable_r2 = self.vlm_arm in ("vred", "vall") and not self.r5_fallback_active
        if enable_r2 and vlm_valid:
            # Trigger check: Q_light == red_or_yellow_for_ego for K consecutive queries
            is_red_k = len(self.light_history) == K and all(x == "red_or_yellow_for_ego" for x in self.light_history)
            is_green_k = len(self.light_history) == K and all(x == "green_for_ego" for x in self.light_history)
            
            # Stop position is entrance to intersection
            dist_to_line = next_junc_dist - REAR_TO_BUMPER - 0.5
            
            if is_red_k and dist_to_line < 50.0 and dist_to_line > -2.0:
                self.r2_holding_red = True
            elif self.r2_holding_red and is_green_k:
                self.r2_holding_red = False
                release = True
                
            if self.r2_holding_red and dist_to_line > -2.0:
                out["R2_red"] = idm_stop(dist_to_line)
                active_rules.append("R2_red")

        # R3: Stop Sign (VLM-based, suppressed by R2)
        enable_r3 = self.vlm_arm in ("vred", "vall") and not self.r5_fallback_active and not self.r2_holding_red
        if enable_r3 and vlm_valid:
            is_sign_k = len(self.sign_history) == K and all(x == "yes" for x in self.sign_history)
            dist_to_line = next_junc_dist - REAR_TO_BUMPER - 0.5
            
            # Only trigger once per junction
            if next_jid not in self.r3_stop_sign_completed_junctions and is_sign_k and 0.0 < dist_to_line < 25.0:
                self.r3_holding_stop = True
                
            if self.r3_holding_stop:
                if speed < 0.2:
                    if self.r3_stopped_since is None:
                        self.r3_stopped_since = sim_t
                    elif sim_t - self.r3_stopped_since >= ARB_PARAMS["T_stop_sign_s"]:
                        # Release stop sign
                        self.r3_holding_stop = False
                        self.r3_stopped_since = None
                        self.r3_stop_sign_completed_junctions.add(next_jid)
                        release = True
                        
                if self.r3_holding_stop:
                    out["R3_sign"] = idm_stop(dist_to_line)
                    active_rules.append("R3_sign")

        # Reset R5 fallback once moving again
        if self.r5_fallback_active and speed > 1.0:
            self.r5_fallback_active = False

        return out, cruise_cap, release, active_rules

    def step(self, input_data):
        """Main agent tick: collect frames, execute VLM query, apply arbitration table."""
        # 1. Update simulation time and ego pose
        sim_t = input_data["timestamp"]
        
        # Read cameras: front road and wide
        # In B2D, camera names follow zeroshot rigs
        rgb_frames = []
        for cam_key in ("rgb_road", "rgb_wide", "rgb_left"):
            if cam_key in input_data:
                rgb_frames.append(input_data[cam_key][1][:, :, :3])
        if not rgb_frames and "rgb" in input_data:
            rgb_frames.append(input_data["rgb"][1][:, :, :3])

        # 2. Geometry & Snapshot update
        xy, yaw = self.poses[-1][1], self.poses[-1][2] if self.poses else ([0., 0.], 0.)
        ego_s = float(project([xy], self.route.xy)[0][0])
        next_junc_dist, next_junc_s, next_jid = self._next_junction_info(ego_s)

        # 3. VLM Query cadence: query every 0.5s of simulation time
        if rgb_frames and (sim_t - self.last_vlm_query_sim_t >= self.frame_save_interval):
            self._query_vlm(sim_t, rgb_frames)
            self.last_vlm_query_sim_t = sim_t
            
            # Optional: save frame for offline inspection / multi-model evaluation
            if self.save_frames:
                frame_idx = len(self.cam_sets)
                frame_path = self.frame_dir / f"frame_{sim_t:.2f}s_{frame_idx}.npy"
                np.save(str(frame_path), rgb_frames[0])

        # 4. Consume arrived VLM answers (accounting for delay L)
        self._update_vlm_answer(sim_t)

        # 5. Compute ground-truth context for shadow evaluation
        gt_ctx = self._extract_ground_truth(sim_t, ego_s, next_junc_dist)

        # 6. Apply arbitration rules R1 - R5
        speed = self.speeds[-1] if self.speeds else 0.0
        vlm_constraints, cruise_cap, release, active_rules = self._apply_arbitration_table(
            speed, sim_t, ego_s, next_junc_dist, next_junc_s, next_jid
        )

        # 7. Apply speed cap from R1 to base governor
        if "cruise_by_route" not in self.arb:
            self.arb["cruise_by_route"] = {}
        original_cruise = self.arb.get("cruise", ARB_PARAMS["cruise_base"])
        self.arb["cruise"] = min(original_cruise, cruise_cap)

        # 8. Log step info
        log_entry = {
            "t": sim_t,
            "speed": speed,
            "ego_s": ego_s,
            "vlm_arm": self.vlm_arm,
            "vlm_answer": self.current_vlm_answer,
            "gt": gt_ctx,
            "active_rules": active_rules,
            "release": release,
            "r5_fallback": self.r5_fallback_active
        }
        self.vlm_log.write(json.dumps(log_entry) + "\n")

        # 9. Invoke base OpenPilot OpArbAgent step
        control = super().step(input_data)
        
        # Restore cruise setting
        self.arb["cruise"] = original_cruise
        return control

    def destroy(self):
        """Cleanup resources and close logs."""
        if hasattr(self, "vlm_log") and self.vlm_log and not self.vlm_log.closed:
            self.vlm_log.close()
        if hasattr(self, "priv") and self.priv:
            self.priv.close()
        super().destroy()
