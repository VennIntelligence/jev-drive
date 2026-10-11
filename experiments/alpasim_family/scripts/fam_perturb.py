"""FAMILY: what a driver is told about CAM_F0, or the CAM_F0 image it receives, perturbed per FAM_ARM. Shared by the two wrappers
(fam_driver.py: our served sh30 driver; fam_ltf.py: the shipped LTF sample). No model or driver logic is changed; numpy / PIL only.

FAM_ARM = key=value items joined by "+" (a GPU-pool job spec cannot carry commas; empty = control):
  calibration told to the driver, image untouched (a reported calibration that does not match the render)
    pitch=d   optical axis believed d deg further DOWN (rotation about the rig's y / left axis, applied to camera->rig R)
    yaw=d     believed d deg further LEFT (about rig z);  roll=d  about rig x
    f=s       fx, fy x s;  fx=s / fy=s  one of them;  cx=d / cy=d  principal point + d px
    tz=d      camera height + d m (slot warp road plane);  tx=d  camera x + d m (plan lever arm)
    D=nom     the public assets' median CAM_F0 distortion (what MTGS renders with, never reported); D=asset the scene's own (oracle,
              looked up by scene id in FAM_CALIB, results/asset_calib_public.csv); D=raw nuPlan's raw CAM_F0 distortion
    Rnom=p    ignore the told mount rotation: a level camera on the rig's x axis pitched p deg down (public median 1.45; exploratory,
              the serving fix that does not trust the report)
    default   the MTGS adapter's fallback camera (fx 1920, fy 1080, cx 960, cy 560, identity mount at the rig origin)
  image (applied to every CAM_F0 / L0 / R0 JPEG before the driver sees it; re-encoded at the runtime's quality 95)
    zoom=z    centre crop of z x the frame, resized back to 1920 x 1080 (a narrower lens on the same mount); with kfollow=1 the
              told K follows (a consistent rig change), else the told K is the render's (mismatch)
    reenc=1   decode / re-encode only (the image arms' control)
"""
from __future__ import annotations

import csv
import io
import os

import numpy as np

D_NOM = (-0.3377, 0.1129, -0.0011, -0.0003, 0.0)         # public assets, median of colmap_param distortion (asset_calib.py)
D_RAW = (-0.3561, 0.1725, -0.0021, 0.0005, -0.0523)      # nuPlan raw CAM_F0 (cam_intrinsic / distortion of the assets)


def arm() -> dict:
    s = os.environ.get("FAM_ARM", "").strip()
    return dict((kv.split("=", 1) if "=" in kv else (kv, "1")) for kv in s.replace(",", "+").split("+") if kv)


def _rot(axis: int, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


_ASSET = None


def asset_D(scene: str):
    global _ASSET
    if _ASSET is None:
        with open(os.environ["FAM_CALIB"]) as f:
            _ASSET = {r["scene"]: tuple(float(r[k]) for k in ("k1", "k2", "p1", "p2", "k3")) for r in csv.DictReader(f)}
    return _ASSET[scene]


def told(cam: dict, scene: str, a: dict) -> dict:
    """A navsim_zs camera dict (K, R camera->rig, t, D) -> the one the driver is told under arm `a`."""
    K, R, t, D = (np.array(cam[k], np.float64) for k in ("K", "R", "t", "D"))
    if "default" in a:
        K, R, t, D = np.array([[1920., 0, 960], [0, 1080, 560], [0, 0, 1]]), np.eye(3), np.zeros(3), np.zeros(5)
    if "Rnom" in a:
        R = _rot(1, float(a["Rnom"])) @ np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])
    for key, axis, sign in (("roll", 0, 1), ("pitch", 1, 1), ("yaw", 2, 1)):
        if key in a:
            R = _rot(axis, sign * float(a[key])) @ R
    for key, ij in (("fx", [(0, 0)]), ("fy", [(1, 1)]), ("f", [(0, 0), (1, 1)])):
        for i, j in ij if key in a else []:
            K[i, j] *= float(a[key])
    if "cx" in a:
        K[0, 2] += float(a["cx"])
    if "cy" in a:
        K[1, 2] += float(a["cy"])
    if "zoom" in a and a.get("kfollow") == "1":
        z = float(a["zoom"])
        K[0, 0], K[1, 1] = K[0, 0] / z, K[1, 1] / z
        K[0, 2], K[1, 2] = 960 + (K[0, 2] - 960) / z, 540 + (K[1, 2] - 540) / z
    if "tz" in a:
        t[2] += float(a["tz"])
    if "tx" in a:
        t[0] += float(a["tx"])
    if "D" in a:
        D = np.array({"nom": D_NOM, "raw": D_RAW}.get(a["D"]) or asset_D(scene), np.float64)
    return {"K": K.astype(np.float32), "R": R.astype(np.float32), "t": t.astype(np.float32), "D": D.astype(np.float32)}


def image(jpeg: bytes, a: dict) -> bytes:
    """The image arm on one JPEG; the bytes unchanged when the arm has none."""
    if "zoom" not in a and "reenc" not in a:
        return jpeg
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if "zoom" in a:
        z, (W, H) = float(a["zoom"]), im.size
        w, h = W * z, H * z
        box = ((W - w) / 2, (H - h) / 2, (W + w) / 2, (H + h) / 2)
        im = im.resize((W, H), Image.BILINEAR, box=box)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=95)
    return out.getvalue()
