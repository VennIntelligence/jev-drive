#!/usr/bin/env python3
"""ALPASIM_FAST: the one place that turns an AlpaSim read into a fast local read (docs/alpasim.md "Fast local reads", results/pai_fast.md).
Wizard overrides that (1) render only the camera the driver consumes, with the preset's own entry for it (resolution, frame interval,
shutter: the delivered image is the same request), and (2) turn the eval video off (the official ec2 presets run with it off; it is drawn
after scoring). The driver and the image scorer read only that camera, so scores should not move; official submissions keep the full set.

The camera comes from the driver itself: its module-level `CAM` (a literal, or os.environ.get(<var>, <default>) honouring <var> in the
environment or in the driver command), e.g. pai_driver.py -> camera_front_wide_120fov (PAI_CAM), sh30_driver.py -> CAM_F0. The entry comes
from the preset named in the wizard arguments (+e2e_challenge=dev, +e2e_challenge_nuplan=full, ...), resolved through its Hydra defaults.

  python3 sim_fast.py <alpasim src> <driver .py> <wizard args...>     prints the overrides, one per line
  from sim_fast import overrides                                      overrides(src, wizard_args, driver_py, env) -> [str]
Runners pass the switch only: run_native.py (box: pai_native.sh / pai_box.sh, run.sh nuPlan) and pai_run.sh (Tokyo Docker).
"""
import ast, os, re, sys

import yaml


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


def overrides(src, wizard_args, driver_py, env=None):
    cfg_dir = os.path.join(src, "src", "wizard", "configs")
    presets = [m.groups() for a in wizard_args if (m := re.fullmatch(r"\+(e2e_challenge\w*)=(\w+)", a))]
    if len(presets) != 1:
        raise SystemExit(f"sim_fast: expected one +e2e_challenge*=<preset> in the wizard args, got {presets}")
    cams = _cameras(cfg_dir, "/".join(presets[0])) or []
    cam = driver_camera(driver_py, env)
    entry = [c for c in cams if c.get("logical_id") == cam]
    if not entry:
        raise SystemExit(f"sim_fast: the driver's camera {cam} is not in preset {presets[0]}'s cameras {[c.get('logical_id') for c in cams]}")
    spec = ",".join(f"{k}:{v}" for k, v in entry[0].items())
    return [f"runtime.simulation_config.cameras=[{{{spec}}}]", "eval.video.render_video=false"]


if __name__ == "__main__":
    print("\n".join(overrides(sys.argv[1], sys.argv[3:], sys.argv[2])))
