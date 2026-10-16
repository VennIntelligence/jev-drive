#!/usr/bin/env python3
"""ALPASIM_FAST: the one place that turns an AlpaSim read into a fast local read (docs/alpasim.md "Fast local reads", results/pai_fast.md).
Wizard overrides that turn the eval video off (the official ec2 presets run with it off; it is drawn after scoring) and shrink the camera set:
  ALPASIM_FAST=1     every preset camera stays (same ids and shutters), the driver's camera exactly as the preset asks, the others at
                     TINY px height and one frame per rollout (frame interval ONCE_US): the runtime anchors the rollout clock on the first
                     frame end over the requested cameras (alpasim_runtime unbound_rollout.py, render_start_us), so keeping them keeps every
                     decision time; the driver's image is the same request. Scores should not move. (At 64 px but the preset's 10 Hz the
                     five extra PAI cameras cost as much as at 1080 px: Tokyo, bit-identical but no faster.)
  ALPASIM_FAST=solo  the driver's camera alone: cheaper, but on PAI scenes the clock moves by 0-30 ms (the cameras' first frames differ), so
                     the run is a perturbed copy, not the same run (results/pai_fast.md).
On nuPlan presets (+e2e_challenge_nuplan=*) 1 means solo: the clock there is synthetic (scene start + shutter, the same for all 8
cameras) and MTGS renders all of them in one pass at native size whatever is requested, so the unread ones cost only their JPEG encode,
transport and logging: bit-identical to the official set on 304 scenes, 1.00x wall (results/pai_fast.md "nuPlan"). The nuPlan `full` preset already has the eval video off.
The driver and the image scorer read only that camera; official submissions keep the full set. ALPASIM_VIDEO=1 keeps the eval video
(it draws the driver's camera only, so it works with either mode).

The camera comes from the driver itself: its module-level `CAM` (a literal, or os.environ.get(<var>, <default>) honouring <var> in the
environment or in the driver command), e.g. pai_driver.py -> camera_front_wide_120fov (PAI_CAM), sh30_driver.py -> CAM_F0. The entry comes
from the preset named in the wizard arguments (+e2e_challenge=dev, +e2e_challenge_nuplan=full, ...), resolved through its Hydra defaults.

  python3 sim_fast.py <alpasim src> <driver .py> <wizard args...>     prints the overrides (mode from ALPASIM_FAST), one per line
  from sim_fast import overrides                                      overrides(src, wizard_args, driver_py, env, mode) -> [str]
Runners pass the switch only: run_native.py (box: pai_native.sh / pai_box.sh, run.sh nuPlan) and pai_run.sh (Tokyo Docker). The PAI runners
default to ALPASIM_FAST=1 (2026-10-16), run.sh / run_native.py (nuPlan) to 0 (measured: no faster); ALPASIM_FAST=0 is the official set.
"""
import ast, os, re, sys

import yaml

TINY, ONCE_US = 64, 100_000_000     # the cameras nobody reads under ALPASIM_FAST=1: px height (aspect kept), frame interval (> a rollout)
SHUTTER_US = 17_000                 # RuntimeCameraConfig.shutter_duration_us default (alpasim_runtime/config.py)


def driver_camera(driver_py, env=None):
    env = os.environ if env is None else env
    for node in ast.parse(open(driver_py).read()).body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            names, vals = ([tgt], [node.value]) if isinstance(tgt, ast.Name) else (tgt.elts, node.value.elts) if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple) else ([], [])
            for n, v in zip(names, vals):
                if isinstance(n, ast.Name) and n.id == "CAM":
                    if isinstance(v, ast.Constant):
                        return v.value
                    if isinstance(v, ast.Call) and ast.unparse(v.func) == "os.environ.get":
                        var, default = (ast.literal_eval(a) for a in v.args)
                        return env.get(var, default)
    raise SystemExit(f"sim_fast: no module-level CAM in {driver_py}")


def _cameras(cfg_dir, rel):
    """The runtime.simulation_config.cameras a config file ends up with: its own entry, else the last of its defaults that sets one."""
    d = yaml.safe_load(open(os.path.join(cfg_dir, rel + ".yaml"))) or {}
    own = (((d.get("runtime") or {}).get("simulation_config") or {}).get("cameras"))
    found = None
    for e in d.get("defaults") or []:
        if isinstance(e, dict):
            (k, v), = e.items()
            k = k.replace("override ", "").strip().lstrip("/")
            if isinstance(v, str) and os.path.exists(os.path.join(cfg_dir, k, v + ".yaml")):
                found = _cameras(cfg_dir, f"{k}/{v}") or found
    return own or found


def overrides(src, wizard_args, driver_py, env=None, mode="1"):
    cfg_dir = os.path.join(src, "src", "wizard", "configs")
    presets = [m.groups() for a in wizard_args if (m := re.fullmatch(r"\+(e2e_challenge\w*)=(\w+)", a))]
    if len(presets) != 1:
        raise SystemExit(f"sim_fast: expected one +e2e_challenge*=<preset> in the wizard args, got {presets}")
    cams = _cameras(cfg_dir, "/".join(presets[0])) or []
    cam = driver_camera(driver_py, env)
    entry = [c for c in cams if c.get("logical_id") == cam]
    if not entry:
        raise SystemExit(f"sim_fast: the driver's camera {cam} is not in preset {presets[0]}'s cameras {[c.get('logical_id') for c in cams]}")
    if mode not in ("1", "solo"):
        raise SystemExit(f"sim_fast: ALPASIM_FAST={mode}: 1 | solo")
    # nuPlan (MTGS on trajdata scenes): the first frame is synthesised as scene start + shutter (unbound_rollout.py,
    # _synthetic_first_camera_frame_ranges_us) and the renderer draws every asset camera at native size whatever is requested
    # (plugins/mtgs/server/servicer.py, _render_aggregated_batch). Dropping the unread cameras keeps the clock whenever none of
    # them closes its shutter earlier than the driver's, and a tiny request would come back full size anyway: there 1 = solo.
    shutter = lambda c: c.get("shutter_duration_us", SHUTTER_US)
    if presets[0][0].startswith("e2e_challenge_nuplan") and all(shutter(c) >= shutter(entry[0]) for c in cams):
        mode = "solo"
    keep = []
    for c in (entry if mode == "solo" else cams):
        if c["logical_id"] != cam:
            c = dict(c, height=TINY, width=-(-c["width"] * TINY // c["height"]) // 2 * 2, frame_interval_us=ONCE_US)
        keep.append("{" + ",".join(f"{k}:{v}" for k, v in c.items()) + "}")
    video = (os.environ if env is None else env).get("ALPASIM_VIDEO", "0") == "1"      # 1 = keep the preset's eval video (front camera)
    return [f"runtime.simulation_config.cameras=[{','.join(keep)}]"] + ([] if video else ["eval.video.render_video=false"])


if __name__ == "__main__":
    print("\n".join(overrides(sys.argv[1], sys.argv[3:], sys.argv[2], mode=os.environ.get("ALPASIM_FAST", "1"))))
