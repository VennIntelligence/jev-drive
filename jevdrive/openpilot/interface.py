"""openpilot as on the car: the one interface spec every board harness resolves its configuration against.

docs/openpilot-interface.md is the prose version (sources, decisions, per-board tables). Stdlib only and Python 3.8, so
the CARLA route process (envs/carla) imports it too.

SPEC is what openpilot does on a comma device in a car. A harness describes what it actually runs as a flat dict of the
same keys (`resolve_<board>`); every key whose value differs from SPEC is a Deviation, and every deviation must be declared
for that board in DECLARED (privilege level, why, decision). `check` refuses an undeclared deviation and the one forbidden
setting (a ground-truth traffic light releasing a stop latch, `"resume": "nored"`). `record` / `write` put the resolved
configuration and its deviation list into a run dir (`interface.json`), so every reported number carries its interface.

Keys (values in SPEC):
  rig.height_m          1.22      camera height above the road (openpilot's training rig, behind the windshield)
  rig.level             True      calibrated frame = road frame (pitch 0, horizon on rows 47.6 road / 151.8 wide)
  rig.wide              sensor    wide input from one wide camera (120 deg fisheye warped to 58.7 deg); also crop1 / stitched3 / pinhole1
  history.frames        real      frames come from the camera at its own rate; render (simulator), synth-gimm, synth-static
  history.rate_hz       20        rate frames reach the model (real or after the declared synthesis)
  history.clock         real      model clock = wall clock; dilate (HUGSIM 4 Hz -> 5 Hz context), hold, repeat2 (10 Hz fed twice)
  history.warmup        real      what the model saw before the first plan: real frames (the car stands with openpilot on);
                                  static (one frame repeated), cold (no warm-up beyond the first step)
  lateral.source        action    desired curvature = action[0] / max(1, v)^2 (modeld); plan = a tracker follows the plan positions;
                                  plan-curvature = modeld.get_curvature_from_plan (the plan's own yaw / yaw rate at action_t) through the op path
  lateral.exec          op-path   controlsd clip_curvature + latActive + lateralDelay (lib/op_ctrl.py OpLateral);
                                  bicycle (raw curvature, no clip / delay), ilqr, p7, scorer-lqr (the leaderboard replays it)
  lateral.delay_s       0.2       lateralDelay in model seconds (lagd's initial steerActuatorDelay + 0.2 class value)
  lon.source            action    action[1] -> LongControl (selfdrive/controls); plan-ilqr, plan-scorer, plan-idm-latch
  lon.resume            driver    a held standstill is left when the driver presses resume / taps the gas (many cars; the model
                                  never launched from a held standstill alone). none (nobody resumes: only the model's plan),
                                  timer (B2D: resume after latch_max_s whatever is ahead), rule (jevdrive/openpilot/resume.py:
                                  an emulated driver from the ego speed and the model's own lead head), timer+rule
  command.channel       none      nothing tells the model where to go; desire only for a driver-initiated lane change.
                                  desire is NOT a choice signal (decisions 92, 121). desire-route, desire-sim, onehot, intent, sky-arrow
  command.route_geometry none     dense-zones: the route geometry steers in command zones / on divergence (B2D DRIVE_ZONES, semi);
                                  nav-polyline: the route ahead as a navigation polyline input of a route-choice fine-tune (semi)
  light.source          none      the model sees the light like any pixel; vlm (a VLM reads it), gt-stop (privileged ceiling);
                                  gt-latch (nored) is forbidden
  inputs.extra          ()        model inputs besides the road / wide frames, desire and traffic convention (modeld feeds none):
                                  ego-status (speed / acceleration), pose-history, side-cams (experiments/op_parity)
  tricks                ()        harness post-processing outside openpilot (forward_only, coast_v, x1.06, selector, ...)
"""
import json
import os
import time
from dataclasses import asdict, dataclass

SPEC = {
    "rig.height_m": 1.22, "rig.level": True, "rig.wide": "sensor",
    "history.frames": "real", "history.rate_hz": 20.0, "history.clock": "real", "history.warmup": "real",
    "lateral.source": "action", "lateral.exec": "op-path", "lateral.delay_s": 0.2,
    "lon.source": "action", "lon.resume": "driver",
    "command.channel": "none", "command.route_geometry": "none",
    "light.source": "none",
    "inputs.extra": (),
    "tricks": (),
}
DEVICE = {"road_focal_px": 910.0, "wide_focal_px": 455.0, "road_horizon_row": 47.6, "wide_horizon_row": 151.8,
          "model_frame": [512, 256], "wide_fov_deg": 58.7, "op_ctrl": {"max_curvature": 0.2, "max_lateral_jerk": 5.0,
          "max_lateral_accel": 3.0, "v_active": 0.3, "v_hold": 0.3}, "source": "openpilot master ec95db3f, opendbc 35f7e081"}
PRIVILEGE = ("real-car", "LB", "semi", "priv")          # what the deviation needs beyond a real car
FORBIDDEN = {"light.source": ("gt-latch",)}             # never, on any board (ledger 2026-10-05 section 8)
TOL = {"rig.height_m": 0.03, "history.rate_hz": 0.5, "lateral.delay_s": 0.011}


@dataclass(frozen=True)
class Deviation:
    key: str
    spec: object
    actual: object
    privilege: str          # one of PRIVILEGE
    why: str
    ref: str                # decision / doc


# Declared per-board deviations: {board: {key: (privilege, why, ref)}}. A key listed here may differ from SPEC on that board
# with any value except FORBIDDEN ones; the value actually run is recorded with every run.
DECLARED = {
    "navsim": {
        "rig.height_m": ("LB", "CAM_F0 is ~1.87 m above the road; the virtual camera (--vcam) is an offline arm", "d104, d108"),
        "rig.wide": ("LB", "road and wide both cut from the single CAM_F0 image", "d108"),
        "history.frames": ("LB", "the board gives 2 Hz keyframes; the 20 Hz context frames are GIMM-VFI synthesis", "d116"),
        "history.rate_hz": ("LB", "keyframes at 2 Hz, synthesized to the 5 Hz context rate", "d116"),
        "history.clock": ("LB", "context steps at synthesized frame times", "d116"),
        "history.warmup": ("LB", "zero model state, the 6 synthesized context frames only", "d116"),
        "lateral.source": ("LB", "the board scores a trajectory: the plan is submitted, the scorer's LQR drives it", "d112"),
        "lateral.exec": ("LB", "scorer LQR + bicycle, not ours to change", "d112"),
        "lon.source": ("LB", "plan speed profile replayed by the scorer", "d97"),
        "command.channel": ("LB", "driving-command one-hot only in the lc@-1.0 desire arm (not the headline)", "d66, d92"),
        "tricks": ("semi", "lever (rear-axle pose) and retime adapters; selector sel-rot0-r0.6 when named", "d94, d107"),
    },
    "wod": {
        "rig.height_m": ("LB", "front camera z 1.8065 m (true ~1.86 m); virtual heights hX.XX are an offline arm", "d108"),
        "rig.wide": ("LB", "wide stitched from FRONT + FRONT_LEFT + FRONT_RIGHT by rotation-only warps", "d108"),
        "history.rate_hz": ("LB", "real 10 Hz frames", "ledger s2"),
        "history.clock": ("LB", "each 10 Hz frame fed twice (20 Hz model clock)", "ledger s2"),
        "history.warmup": ("LB", "10 s of real history from a zero state", "ledger s2"),
        "lateral.source": ("LB", "open loop: the plan positions are the prediction", "-"),
        "lateral.exec": ("LB", "no controller (open loop)", "-"),
        "lon.source": ("LB", "plan positions", "-"),
        "command.channel": ("LB", "WOD intent only in the op_adapt_L intent adapter", "d78"),
        "tricks": ("semi", "x1.06 longitudinal stretch on the test submission (fitted scalar)", "d107"),
    },
    "hugsim": {
        "rig.height_m": ("LB", "each dataset's own camera height (nuScenes ~1.2, PandaSet / KITTI-360 1.5, Waymo 1.8)", "d108.4"),
        "rig.level": ("LB", "KITTI-360 has a real -1.6 deg pitch", "d108"),
        "rig.wide": ("LB", "wide stitched from CAM_FRONT / FRONT_LEFT / FRONT_RIGHT (Waymo / PandaSet 5% black border)", "d108"),
        "history.frames": ("LB", "frames rendered by the simulator at its 4 Hz step", "ledger s2"),
        "history.rate_hz": ("LB", "4 Hz renders", "ledger s2"),
        "history.clock": ("LB", "dilate: one 0.25 s step = one 0.2 s context step, speed x1.25 in, plan times /1.25 out; "
                                "hold: 20 Hz at face value, each render held 5 model steps", "this note"),
        "history.warmup": ("LB", "no real history exists (the car spawns moving at 1.0 m/s): 5 s static warm-up on the first frame "
                                 "(amplifies the launch lean, d100, but the action path removes the spins, d118; a cold start "
                                 "does not launch, unified_interface.md)", "d100, d118, unified_interface.md"),
        "lateral.source": ("LB", "legacy exam preset: iLQR tracks the plan; spec_plan: the curvature of the model's own plan (modeld.get_curvature_from_plan) "
                                 "through the op path, a diagnostic", "d118, d133"),
        "lateral.exec": ("LB", "legacy exam preset: iLQR (PR#57)", "d118"),
        "lateral.delay_s": ("LB", "pure delay in simulator seconds (0.25 under dilate = 0.2 model s)", "d118"),
        "lon.source": ("LB", "iLQR tracks the plan's speed: action acceleration -> LongControl fails launches (d119)", "d119"),
        "lon.resume": ("real-car", "none: no driver in the simulator, a standstill ends only when the model's plan moves (decision 118's "
                                   "stuck runs); rule: an emulated driver resume (jevdrive/openpilot/resume.py) from the ego speed and "
                                   "the model's own lead head, no light / route / actor state", "d118, d119, op_resume"),
        "command.channel": ("LB", "simulator command (from the recorded route) -> turn desire; d92: no help. op_parity arms "
                                  "(+onehot) also read it as a NAVSIM one-hot [L, S, R] (WA-JEPA's command map [2, 0, 1])",
                            "d90, d92, op_parity"),
        "inputs.extra": ("real-car", "op_parity arms: simulator ego speed / acceleration and a 4-pose 2 Hz history (odometry), "
                                     "side / rear camera renders, through lib/parity_adapter.py's bias (WA-JEPA's inputs)",
                         "op_parity"),
        "tricks": ("semi", "forward_only / straight_stop plan post-processing; simulator initial speed 1.0 m/s; optional "
                           "derot / selector / launch_stab / launch_long arms", "d90, d96"),
    },
    "b2d": {
        "rig.height_m": ("LB", "spec camera aligned with the open-loop boards' viewpoint (mean of NAVSIM CAM_F0 and WOD front, "
                               "B2D_MOUNTS['openloop']: x 1.59 m, z 1.86 m) instead of openpilot's 1.22 m; named presets keep "
                               "1.22 m at the bumper line (x 3.8 m) and the legacy 1.433 m windshield top (hood fills 20% of the "
                               "frame at 1.22 m inside the MKZ)", "d104.8, d125, unified_interface.md"),
        "rig.wide": ("real-car", "sensor-f<focal>: the same wide sensor (1928 x 1208, f 567, 118.9 deg) warped to a wide model frame of "
                                 "focal <focal> instead of 455 (160 = 116 deg HFOV, horizon row unchanged); only for models fine-tuned on "
                                 "that input (experiments/op_wide_ft); a comma device can do the same warp", "op_wide_ft"),
        "history.warmup": ("real-car", "warm-up while braked at the start: real frames from a standing car", "-"),
        "lateral.source": ("LB", "legacy p7 arms track the plan path", "d118.5"),
        "lateral.exec": ("LB", "legacy drive preset: raw action curvature through the bicycle model, no clip / delay", "d118.5"),
        "lateral.delay_s": ("LB", "CARLA's own steering lag stands in for lateralDelay in the legacy preset", "d118.5"),
        "lon.source": ("semi", "our scheduler: min(set speed 8 m/s + curvature cap, lead-head IDM, plan while rolling) + stop "
                               "latch + timer resume; action acceleration not used (d119)", "d74, d82, d119"),
        "lon.resume": ("semi", "timer: the stop latch's driver resume after latch_max_s whatever is ahead (blind to the light and the "
                               "lead; d126); +rule: also the emulated driver resume of jevdrive/openpilot/resume.py (ego speed and "
                               "the model's lead head only); none: modes without a stop latch", "d74, d126, op_resume"),
        "command.channel": ("semi", "route turn desire 20 m before LEFT / RIGHT (not a choice signal, d121); sky arrow arms",
                            "d92, d102, d121"),
        "command.route_geometry": ("semi", "dense-zones = DRIVE_ZONES: the dense route geometry steers in command zones (LEFT / RIGHT "
                                           "15/5 m, STRAIGHT 5/5, lane change 5/10) and on divergence > 1 m at 15 m; a fallback, not an "
                                           "openpilot capability; every report row carries the zones-off reading. nav-polyline = the "
                                           "route-choice fine-tune's input (experiments/op_route_ft, lib/route_adapter.py): the route "
                                           "ahead as a 10 m-vertex polyline to 150 m in the ego frame, taken from the dense route without "
                                           "noise, i.e. what a car's navigation knows about the route (semi: the dense route is lane-exact, "
                                           "a navigation route is road-level); it only conditions the model, nothing steers from it",
                                   "d121, d122, op_route_ft"),
        "light.source": ("priv", "R3a tl_stop is a privileged ceiling (diagnosis only); VLM arms read the light from pixels",
                         "d82, d107"),
        "tricks": ("semi", "coast_v 2.5 (MKZ stops dead on a light brake); camera_at_bumper_x3.80 only in the "
                           "bumper122 preset; Privileged bypass (priv, vmerge2)", "d74, d101, unified_interface.md"),
    },
}


class InterfaceError(ValueError):
    pass


def _differs(key, a, b):
    if key in TOL and isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > TOL[key]
    return tuple(a) != tuple(b) if isinstance(a, (list, tuple)) else a != b


def deviations(board, values):
    """Deviations of `values` (a resolved dict) from SPEC. Raises on a forbidden value or an undeclared deviation."""
    unknown = set(values) - set(SPEC)
    if unknown:
        raise InterfaceError("%s: unknown interface keys %s" % (board, sorted(unknown)))
    out, decl = [], DECLARED.get(board, {})
    for key, spec in SPEC.items():
        actual = values.get(key, spec)
        if actual in FORBIDDEN.get(key, ()):
            raise InterfaceError("%s: %s = %r is forbidden on every board (ground-truth light)" % (board, key, actual))
        if not _differs(key, actual, spec):
            continue
        if key not in decl:
            raise InterfaceError("%s: undeclared deviation %s = %r (spec %r); declare it in DECLARED" % (board, key, actual, spec))
        priv, why, ref = decl[key]
        out.append(Deviation(key, spec, list(actual) if isinstance(actual, tuple) else actual, priv, why, ref))
    return out


def record(board, preset, values, config=None, **extra):
    """The interface.json content: resolved values, their deviations and the harness config they came from."""
    devs = deviations(board, values)
    full = dict(SPEC, **values)
    full["tricks"] = list(full["tricks"])
    return {"interface": "jevdrive.openpilot.interface", "board": board, "preset": preset,
            "resolved": full, "deviations": [asdict(d) for d in devs],
            "privileges": sorted({d.privilege for d in devs}, key=PRIVILEGE.index),
            "device": DEVICE, "config": config, "t": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}


def write(run_dir, rec, name="interface.json"):
    """Write `rec` (from `record`) into run_dir (tmp + rename); returns the path."""
    os.makedirs(str(run_dir), exist_ok=True)
    path = os.path.join(str(run_dir), name)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------------ board resolvers
def resolve_navsim(vcam=None, frames="gimm", schedule="none", adapter="lever", selector=None):
    """scripts/op_lb.py run / op_interp export: vcam None = CAM_F0 at its true height (1.87 m); vcam H = a virtual camera."""
    h = 1.87 if vcam in (None, 0, 0.0) else float(vcam)
    tricks = [adapter] + (["selector:%s" % selector] if selector else [])
    return {"rig.height_m": h, "rig.wide": "crop1", "history.frames": "synth-" + frames, "history.rate_hz": 5.0,
            "history.clock": "synth", "history.warmup": "cold", "lateral.source": "plan", "lateral.exec": "scorer-lqr",
            "lon.source": "plan-scorer", "command.channel": "none" if schedule == "none" else "desire-schedule:" + schedule,
            "tricks": tuple(tricks)}


def resolve_wod(height=1.8065, long_scale=1.0, intent=False):
    """scripts/wod_zeroshot_openpilot.py (+ the test submission's x1.06)."""
    tricks = ("long_scale:%g" % long_scale,) if long_scale != 1.0 else ()
    return {"rig.height_m": float(height), "rig.wide": "stitched3", "history.rate_hz": 10.0, "history.clock": "repeat2",
            "history.warmup": "real-10s", "lateral.source": "plan", "lateral.exec": "none", "lon.source": "plan",
            "command.channel": "intent" if intent else "none", "tricks": tricks}


HUGSIM_HEIGHT = {"nuscenes": 1.2, "pandaset": 1.5, "kitti360": 1.5, "waymo": 1.8}   # approximate, d108.4
HUGSIM_PRESETS = {
    # spec: openpilot's lateral path (action + clip_curvature + lateralDelay), iLQR longitudinal, dilate clock, and the 5 s static
    # warm-up kept (declared): without it the model does not launch (cold start: 6 of 11 stuck, mean HD 0.358 vs 0.544; hold clock: 8 of
    # 11 stuck, HD 0.049; experiments/leaderboard_audit/results/unified_interface.md). = decision 118's arm.
    "spec": dict(controller="opctrl", env={"OP_CTRL": {"delay": 0.25}}, opts={"op_ctrl": True}),
    "spec_cold": dict(controller="opctrl", env={"OP_CTRL": {"delay": 0.25}}, opts={"op_ctrl": True, "warmup_s": 0.0}),
    "spec_hold": dict(controller="opctrl", env={"OP_CTRL": {"delay": 0.2}},
                      opts={"op_ctrl": True, "warmup_s": 0.0, "op_clock": "hold"}),
    # spec_plan: spec with the lateral curvature taken from the model's OWN plan (modeld.get_curvature_from_plan: yaw and yaw rate at
    # action_t 0.275 s) instead of the action head; clip_curvature, lateralDelay and the longitudinal path unchanged (a diagnostic, not openpilot)
    "spec_plan": dict(controller="opctrl", env={"OP_CTRL": {"delay": 0.25}}, opts={"op_ctrl": True, "op_ctrl_src": "plan"}),
    "spec_plan_smooth": dict(controller="opctrl", env={"OP_CTRL": {"delay": 0.25}}, opts={"op_ctrl": True, "op_ctrl_src": "plan_smooth"}),
    "spec_plan_mpc": dict(controller="opctrl", env={"OP_CTRL": {"delay": 0.25}}, opts={"op_ctrl": True, "op_ctrl_src": "plan_mpc"}),
    "opctrl_d118": dict(controller="opctrl", env={"OP_CTRL": {"delay": 0.25}}, opts={"op_ctrl": True}),   # alias of spec
    # legacy: the exam / every result before 2026-10-05; --controller and --opts are taken literally
    "exam": dict(controller=None, env={}, opts={}),
}


def resolve_hugsim(opts, controller, dataset="nuscenes", op_ctrl_env=None):
    """experiments/hugsim/lib/zs_agent.py opts (HUGSIM_ZS_OPTS) + the controller tree of experiments/hugsim/archive/zs_run.py."""
    o = dict(opts or {})
    hold = o.get("op_clock", "dilate") == "hold"
    warm = float(o.get("warmup_s", 5.0))
    opc = bool(o.get("op_ctrl")) and controller in ("opctrl", "opctrl_long")
    if bool(o.get("op_ctrl")) and not opc:
        raise InterfaceError("hugsim: op_ctrl needs the opctrl tree (got controller %r)" % controller)
    delay = float((op_ctrl_env or {}).get("delay", 0.25))
    dil = 1.0 if hold else float(o.get("dilation", 1.25))
    tricks = [k for k in ("forward_only", "straight_stop") if o.get(k, True)] + ["init_speed_1.0"]
    tricks += [k for k in ("derot_below", "derot_sel", "lstab", "launch_long", "engage_s") if o.get(k)]
    vals = {"rig.height_m": HUGSIM_HEIGHT.get(dataset, 1.5), "rig.level": dataset != "kitti360", "rig.wide": "stitched3",
            "history.frames": "render", "history.rate_hz": 4.0, "history.clock": "hold" if hold else "dilate",
            "history.warmup": "static" if warm > 0 else "cold",
            "lateral.source": {"plan": "plan-curvature", "plan_smooth": "plan-smooth", "plan_mpc": "plan-mpc"}.get(o.get("op_ctrl_src"), "action") if opc else "plan", "lateral.exec": "op-path" if opc else "ilqr",
            "lateral.delay_s": round(delay / dil, 4) if opc else "n/a",
            "lon.source": "action" if (opc and o.get("op_long") and controller == "opctrl_long") else "plan-ilqr",
            "lon.resume": "rule" if o.get("resume") is not None else "none",
            "command.channel": "desire-sim" if o.get("desire", True) else "none", "tricks": tuple(tricks)}
    par = o.get("parity")
    if par and par.get("ego", True):              # experiments/op_parity (lib/parity_hugsim.py): what the served arm reads
        vals["command.channel"] += "+onehot"
        vals["inputs.extra"] = ("ego-status", "pose-history") + (("side-cams",) if par.get("side", True) else ())
    return vals


VLM_LIGHT_ARMS = ("jslow", "vred", "vred3", "vall", "vmerge")   # lib/vlm_arb_agent.py ARM_ROWS with a light row (R1 / R2)


def resume_rule(params):
    """The shared resume rule (jevdrive/openpilot/resume.py) for a board's config value: None = off (lon.resume none / timer),
    {} = the pre-registered defaults, a dict = overrides (recorded with the run)."""
    if params is None:
        return None
    from jevdrive.openpilot.resume import ResumeRule
    return ResumeRule(**({} if params is True else dict(params)))


def b2d_values(cfg, env=None):
    """lib/op_arb_agent.py resolved config (top level + "arb") -> interface values. env: the route process environment."""
    env = os.environ if env is None else env
    a = cfg.get("arb", {})
    mount = cfg.get("op_mount") or (1.779, 0.0, 1.433)
    tick = float(cfg.get("op_camera_tick", 0.05))           # scripts/zeroshot_rigs.OP_CAMERA_TICK
    drive = a.get("mode") == "drive"
    lat_op = drive and a.get("lat", "route") == "op"
    curv = lat_op and a.get("lat_exec", "p7") in ("curv", "hyb")   # hyb: action below hyb_v, plan tracking above (op_route_ft/plan_tracking)
    opc = curv and bool(cfg.get("op_ctrl") is not None or env.get("OP_CTRL"))
    opc_rule = cfg.get("op_ctrl") if cfg.get("op_ctrl") is not None else json.loads(env.get("OP_CTRL") or "{}")
    if a.get("resume") == "nored":
        light = "gt-latch"
    elif a.get("tl_stop"):
        light = "gt-stop"
    elif env.get("VLM_ORACLE") and env.get("VLM_ARM", "") in VLM_LIGHT_ARMS:
        light = "gt-stop"
    elif env.get("VLM_ARM", "") in VLM_LIGHT_ARMS:
        light = "vlm"
    else:
        light = "none"
    channel = "sky-arrow" if a.get("img_cmd") else "intent" if a.get("intent") == "route" else \
        "desire-route" if cfg.get("desire", True) else "none"
    zones = drive and a.get("zones", True) and lat_op
    geom = "+".join((["dense-zones"] if zones else []) + (["nav-polyline"] if cfg.get("route_adapter") else [])) or "none"
    tricks = []
    if float(mount[0]) > 3.0:
        tricks.append("camera_at_bumper_x%.2f" % float(mount[0]))
    if float(a.get("coast_v", 0)) > 0:
        tricks.append("coast_v:%g" % float(a["coast_v"]))
    if cfg.get("pc"):
        tricks.append("privileged:%s" % cfg["pc"].get("arm"))
    if lat_op and float(a.get("div_m", 1.0)) < 1e8:
        tricks.append("divergence_fallback")
    if lat_op and a.get("lat_exec") == "hyb":
        tricks.append("hybrid_plan_above_%gmps" % float(a.get("hyb_v", 3.0)))
    if float(a.get("zone_gain", 1.0)) != 1.0:
        tricks.append("zone_gain:%g" % float(a["zone_gain"]))
    wf = env.get("OP_WIDE_FOCAL", "")                    # server-side wide warp focal (zeroshot_policy_server; experiments/op_wide_ft)
    wide = "sensor" if not wf or float(wf) == DEVICE["wide_focal_px"] else "sensor-f%g" % float(wf)
    return {"rig.height_m": float(mount[2]), "rig.wide": wide, "history.rate_hz": round(1.0 / tick, 2),
            "history.warmup": "real" if float(cfg.get("warmup_s", 0)) > 0 else "cold",
            "lateral.source": ("plan-smooth" if a.get("curv_src") == "plan_smooth" else "action") if curv else "plan", "lateral.exec": "op-path" if opc else "bicycle" if curv else "p7",
            "lateral.delay_s": float(opc_rule.get("delay", 0.25)) if opc else "carla",
            "lon.source": "plan-idm-latch" if drive else "plan-p7",
            "lon.resume": ("timer" if drive else "none") + ("+rule" if drive and a.get("resume_rule") is not None else ""),
            "command.channel": channel, "command.route_geometry": geom,
            "light.source": light, "tricks": tuple(tricks)}


# B2D spec preset: the shipped `drive` arbitration with openpilot's lateral path (clip + 0.2 s delay) and no privileged light.
B2D_SPEC_OP_CTRL = {"delay": 0.2}
# Rear-axle frame (x, y, z above the road) of the B2D camera pair. `openloop` is the spec camera: one viewpoint for every board,
# derived from the open-loop boards' real front-camera extrinsics (ego origin = rear axle; both level after openpilot's own calibration,
# real pitch NAVSIM -1.3 deg / WOD -0.2 deg removed):
#   NAVSIM CAM_F0 (40 navtest / navhard frames): x 1.665, y -0.019, z 1.512 above the ego origin (0.35 m above the road) = 1.862 m
#   WOD front (1984 segments): x 1.519, y +0.026, z 1.8065 above the ground origin (~1.86 m true, decision 108)
#   mean: x 1.59, y 0, z 1.86   (the mean, not NAVSIM's alone, so neither board is privileged)
# The camera sits ~0.4 m above the MKZ roof and 2.2 m behind the bumper: no body in the frame (the hood tip is 21 deg below the horizon,
# the 256-row road frame ends at 13 deg). `bumper122` (1.22 m at the front bumper line) and `windshield143` (1.433 m, the legacy `drive`
# preset) are the earlier rigs. lib/b2d_privileged_geometry.py and lib/vm3_perception.py still assume 1.779 m: privileged / vmerge arms keep `drive`.
B2D_MOUNTS = {"openloop": (1.59, 0.0, 1.86), "bumper122": (3.8, 0.0, 1.22), "windshield143": (1.779, 0.0, 1.433)}
B2D_SPEC_MOUNT = B2D_MOUNTS["openloop"]
