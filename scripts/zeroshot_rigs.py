"""Camera rigs that reproduce each zero-shot model's native cameras in CARLA, and the maps from a CARLA
pinhole render to the model's own image. NumPy only, Python 3.8 compatible: the CARLA agent (envs/carla)
imports it for the sensor specs, the policy servers (model venvs) for the image maps.
Pre-registration and the reasoning behind every number: todos/2026-09-24-zeroshot-exam/bench2drive.md.

Frames. Rig: the model's vehicle frame, origin at the rear axle on the ground, x forward, y left, z up.
CARLA sensor specs are relative to the Lincoln MKZ actor origin (x forward, y right, z up, yaw clockwise), whose
rear axle is at x = REAR_AXLE_X (measured, docs/b2d-controller.md). Ground height of the actor origin is 0.
"""
import math

import numpy as np

REAR_AXLE_X = -1.388633220199954

# --- Alpamayo 1.5: NVIDIA Hyperion cameras of PhysicalAI-AV ---------------------------------------------------
# Medians over the 100 clips of PhysicalAI-AV chunk 3119 (calibration/sensor_extrinsics, camera_intrinsics).
# That fleet's vehicle: wheelbase 2.850 m, length 4.872 m, width 2.121 m; the MKZ's wheelbase is 2.860 m.
# f-theta model: pixel radius r = fw(theta), theta = bw(r), on a 1920x1080 image (physical_ai_av.camera_models).
ALPAMAYO_CAMERAS = [  # (name, model camera index, rig xyz, rig yaw/pitch deg, cx, cy, fw_poly, bw_poly)
    ("camera_cross_left_120fov", 0, (2.531, 0.938, 0.897), (66.92, -0.02), 960.86, 547.80,
     (0.0, 929.4999, -1.43563, -21.76419, 2.07662),
     (0.0, 1.0759036e-3, 1.6736639e-9, 2.2076200e-11, 7.878904e-16)),
    ("camera_front_wide_120fov", 1, (1.779, 0.000, 1.433), (-0.36, -0.49), 960.48, 545.03,
     (0.0, 926.6360, -2.64397, -19.69783, 5.23325),
     (0.0, 1.0791624e-3, 1.9519810e-9, 2.7773296e-11, -3.255497e-15)),
    ("camera_cross_right_120fov", 2, (2.535, -0.926, 0.902), (-66.58, 0.71), 961.93, 546.75,
     (0.0, 929.8789, -1.60648, -21.48485, 2.45063),
     (0.0, 1.0752661e-3, 2.2412641e-9, 2.5418904e-11, 4.311999e-16)),
    ("camera_front_tele_30fov", 6, (1.729, 0.093, 1.446), (-0.11, -0.21), 961.76, 546.42,
     (0.0, 3687.1000, -12.53237, 220.93676, -393.49547),
     (0.0, 2.7121662e-4, 2.4360797e-10, -1.1716851e-12, 5.823925e-16)),
]
ALPAMAYO_NATIVE_WH = (1920, 1080)
ALPAMAYO_MODEL_WH = (576, 320)   # what the Qwen3-VL processor makes of 1920x1080 (min/max pixels 163840/196608)
ALPAMAYO_CAMERA_TICK = 0.1       # 10 Hz, the model's frame spacing


def _ftheta_rays(cam, u, v):
    """Unit rays (x right, y down, z forward) of native pixels (u, v)."""
    _, _, _, _, cx, cy, _, bw = cam
    du, dv = np.asarray(u, float) - cx, np.asarray(v, float) - cy
    r = np.hypot(du, dv)
    th = np.polynomial.polynomial.polyval(r, bw)
    s = np.where(r > 1e-9, np.sin(th) / np.maximum(r, 1e-9), 0.0)
    return np.stack([s * du, s * dv, np.cos(th)], -1)


def alpamayo_render_spec(cam, margin=1.02):
    """Pinhole render that covers the whole native f-theta image, sampled at least as finely as the model image
    at its centre. Returns (width, height, horizontal fov deg, focal px)."""
    w, h = ALPAMAYO_NATIVE_WH
    edge = np.r_[np.linspace(0, w - 1, 64)]
    u = np.r_[edge, edge, np.zeros(64), np.full(64, w - 1.)]
    v = np.r_[np.zeros(64), np.full(64, h - 1.), np.linspace(0, h - 1, 64), np.linspace(0, h - 1, 64)]
    ray = _ftheta_rays(cam, u, v)
    tx = margin * np.max(np.abs(ray[:, 0] / ray[:, 2]))
    ty = margin * np.max(np.abs(ray[:, 1] / ray[:, 2]))
    focal = math.ceil(cam[6][1] * ALPAMAYO_MODEL_WH[0] / w)   # f-theta centre scale at model resolution
    width, height = 8 * math.ceil(focal * tx / 4), 8 * math.ceil(focal * ty / 4)
    return width, height, math.degrees(2 * math.atan(width / 2 / focal)), float(focal)


def alpamayo_sensor_specs():
    specs = []
    for cam in ALPAMAYO_CAMERAS:
        name, _, (x, y, z), (yaw, pitch) = cam[:4]
        width, height, fov, _ = alpamayo_render_spec(cam)
        # Roll is below 0.25 deg on every camera and is not reproduced.
        specs.append({"type": "sensor.camera.rgb", "id": name, "x": x + REAR_AXLE_X, "y": -y, "z": z,
                      "roll": 0.0, "pitch": pitch, "yaw": -yaw, "width": width, "height": height, "fov": fov,
                      "sensor_tick": ALPAMAYO_CAMERA_TICK})
    return specs


def alpamayo_sample_grid(cam):
    """For each model-image pixel (H, W), the (x, y) pixel of the CARLA pinhole render it samples.
    The model image is the native f-theta image resized to 576x320, as the processor would resize it."""
    (w, h), (mw, mh) = ALPAMAYO_NATIVE_WH, ALPAMAYO_MODEL_WH
    width, height, _, focal = alpamayo_render_spec(cam)
    u, v = np.meshgrid((np.arange(mw) + .5) * w / mw - .5, (np.arange(mh) + .5) * h / mh - .5)
    ray = _ftheta_rays(cam, u, v)
    return np.stack([focal * ray[..., 0] / ray[..., 2] + width / 2 - .5,
                     focal * ray[..., 1] / ray[..., 2] + height / 2 - .5], -1).astype(np.float32)


def alpamayo_project(cam, xyz_rig):
    """Rig points (N, 3) -> model-image pixels (N, 2) and a visibility mask, for overlays."""
    name, _, t, (yaw, pitch) = cam[:4]
    _, _, _, _, cx, cy, fw, _ = cam
    cy_, sy_, cp, sp = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), \
        math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    fwd = np.array([cp * cy_, cp * sy_, sp])
    left = np.array([-sy_, cy_, 0.])
    up = np.cross(fwd, left)
    d = np.asarray(xyz_rig, float) - np.asarray(t)
    ray = np.stack([-d @ left, -d @ up, d @ fwd], -1)
    th = np.arccos(np.clip(ray[:, 2] / np.linalg.norm(ray, axis=1), -1, 1))
    r = np.polynomial.polynomial.polyval(th, fw)
    rho = np.maximum(np.hypot(ray[:, 0], ray[:, 1]), 1e-9)
    uv = np.stack([cx + r * ray[:, 0] / rho, cy + r * ray[:, 1] / rho], -1)
    (w, h), (mw, mh) = ALPAMAYO_NATIVE_WH, ALPAMAYO_MODEL_WH
    uv = (uv + .5) * [mw / w, mh / h] - .5
    ok = (ray[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < mw) & (uv[:, 1] >= 0) & (uv[:, 1] < mh)
    return uv, ok


# --- openpilot: comma 3X road and wide cameras ------------------------------------------------------------------
# Native sensor 1928x1208; road focal 2648 px (40.0 deg), wide focal 567 px (118.9 deg), both treated as pinhole
# by openpilot's own warp (common/transformations/camera.py). The device sits behind the windshield; the model
# frame is the calibrated (road-aligned) device frame, so the CARLA cameras are mounted level and straight and the
# calibration is rpy = 0.
OP_CAMERA_WH = (1928, 1208)
OP_FOCAL = {"road": 2648.0, "wide": 567.0}
OP_MOUNT_RIG = (1.779, 0.0, 1.22)   # x as the Hyperion windshield camera; z = openpilot's nominal camera height
OP_CAMERA_TICK = 0.2                # 5 Hz, the model's context rate (frames t-0.2 s and t)


def openpilot_sensor_specs():
    x, y, z = OP_MOUNT_RIG
    w, h = OP_CAMERA_WH
    return [{"type": "sensor.camera.rgb", "id": "OP_" + name.upper(), "x": x + REAR_AXLE_X, "y": -y, "z": z,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "width": w, "height": h,
             "fov": math.degrees(2 * math.atan(w / 2 / f)), "sensor_tick": OP_CAMERA_TICK}
            for name, f in OP_FOCAL.items()]


def openpilot_plan_to_rig(plan_pos):
    """openpilot plan positions (calib frame at the camera: x forward, y right, z down) -> rig xy."""
    p = np.asarray(plan_pos, float)
    return np.stack([p[:, 0] + OP_MOUNT_RIG[0], OP_MOUNT_RIG[1] - p[:, 1]], -1)
