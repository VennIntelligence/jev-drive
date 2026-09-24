"""openpilot camera input pipeline on recorded comma video, ported from openpilot master (2026-09):

  common/transformations/{camera,model}.py   intrinsics, calib/model frames, get_warp_matrix
  tinygrad examples/openpilot/compile_warp.py nearest-neighbour perspective warp of NV12 to a 512x256 YUV420
                                              model frame and the 6x128x256 channel packing
Plus comma1M segment readers (frame_info / localizer safetensors, HEVC decode) and ground-truth future
motion in the calib frame.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# --- camera and model frames (common/transformations) ---------------------------------------------------
# device frame: x forward, y right, z down; view frame: x right, y down, z forward
VIEW_FROM_DEVICE = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
CAMERAS = {  # (width, height): (road focal, wide focal); comma 3/3X AR0231/OX03C10 and comma four OS04C10
    (1928, 1208): (2648.0, 567.0),
    (1344, 760): (1522.0 * 3 / 4, 567.0 / 4 * 3),
}
MODEL_W, MODEL_H, MEDMODEL_CY = 512, 256, 47.6
MEDMODEL_K = np.array([[910.0, 0, MODEL_W / 2], [0, 910.0, MEDMODEL_CY], [0, 0, 1]])
SBIGMODEL_K = np.array([[455.0, 0, MODEL_W / 2], [0, 455.0, 0.5 * (256 + MEDMODEL_CY)], [0, 0, 1]])


def intrinsics(w, h, focal):
    return np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.]])


def rot_from_euler(rpy):
    """openpilot euler2rot: R = Rz(yaw) Ry(pitch) Rx(roll) (extrinsic xyz)."""
    return Rotation.from_euler("xyz", rpy).as_matrix()


def get_warp_matrix(device_from_calib_euler, cam_K, bigmodel_frame=False):
    """camera pixel <- model pixel (the M_inv the warp kernel samples with), as in modeld."""
    model_K = SBIGMODEL_K if bigmodel_frame else MEDMODEL_K
    calib_from_model = np.linalg.inv(model_K @ VIEW_FROM_DEVICE)  # view_frame_from_calib_frame(0,0,0) = VIEW_FROM_DEVICE
    return cam_K @ VIEW_FROM_DEVICE @ rot_from_euler(device_from_calib_euler) @ calib_from_model


# --- warp + pack (compile_warp.py) ---------------------------------------------------------------------
def _nn_index(M, dst_wh, src_wh):
    w, h = dst_wh
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    M = M.astype(np.float32)
    sx, sy, sw = (M[i, 0] * x + M[i, 1] * y + M[i, 2] for i in range(3))
    xi = np.clip(np.rint(sx / sw), 0, src_wh[0] - 1).astype(np.int64)  # tinygrad round = half-to-even
    yi = np.clip(np.rint(sy / sw), 0, src_wh[1] - 1).astype(np.int64)
    return (yi * src_wh[0] + xi).ravel()


class Warper:
    """Precomputed gather indices for one camera and one warp matrix (calibration is fixed per segment)."""

    def __init__(self, M, cam_wh):
        cw, ch = cam_wh
        self.y_idx = _nn_index(M, (MODEL_W, MODEL_H), (cw, ch))
        uv_scale = np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], dtype=np.float32)
        self.uv_idx = _nn_index(M * uv_scale, (MODEL_W // 2, MODEL_H // 2), (cw // 2, ch // 2))

    def __call__(self, y, u, v, out=None):
        """y (H, W), u/v (H/2, W/2) uint8 planes -> (6, 128, 256) uint8, channel order of frames_to_tensor."""
        Y = y.ravel()[self.y_idx].reshape(MODEL_H, MODEL_W)
        out = np.empty((6, MODEL_H // 2, MODEL_W // 2), np.uint8) if out is None else out
        out[0], out[1], out[2], out[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
        out[4] = u.ravel()[self.uv_idx].reshape(MODEL_H // 2, MODEL_W // 2)
        out[5] = v.ravel()[self.uv_idx].reshape(MODEL_H // 2, MODEL_W // 2)
        return out


def unpack_luma(packed):
    """(6,128,256) -> (256,512) Y image, for visual checks."""
    Y = np.empty((MODEL_H, MODEL_W), np.uint8)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = packed[:4]
    return Y


# --- comma1M segments ----------------------------------------------------------------------------------
def load_safetensors(path):
    from safetensors.numpy import load_file
    return load_file(str(path))


def load_segment_meta(seg_dir):
    """Calibration, camera sizes, frame times and per-frame localizer states (ECEF) of a comma1M segment."""
    seg_dir = Path(seg_dir)
    fi, loc = load_safetensors(seg_dir / "frame_info.safetensors"), load_safetensors(seg_dir / "localizer.safetensors")
    s = loc["frame_states"]
    q = s[:, 3:7]  # ecef_from_device quaternion, w first
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    # body angular rate from consecutive orientations (device frame), rad/s
    dt = np.gradient(loc["frame_t"])
    dR = np.einsum("nji,njk->nik", R[:-1], R[1:])
    omega = np.vstack([Rotation.from_matrix(dR).as_rotvec(), np.zeros((1, 3))]) / dt[:, None]
    omega[-1] = omega[-2]
    return dict(
        rpy_calib=loc["rpy"], wide_from_device_euler=loc["wide_from_device_euler"],
        fcam_wh=(int(fi["fcamera/width"][0]), int(fi["fcamera/height"][0])),
        has_ecam="ecamera/width" in fi and int(fi["ecamera/frame_count"][0]) > 0,
        t_f=fi["fcamera/t"], t_e=fi.get("ecamera/t"), t_loc=loc["frame_t"], device_type=int(fi["device_type"]),
        pos=s[:, 0:3], R=R, vel=s[:, 7:10], vel_dev=np.einsum("nji,nj->ni", R, s[:, 7:10]), omega_dev=omega,
    )


def future_in_calib(meta, i, t_idxs):
    """Ground-truth future positions (len(t_idxs), 3) and speeds in the calib frame of frame i."""
    t = meta["t_loc"]
    tq = t[i] + np.asarray(t_idxs)
    ok = tq <= t[-1]
    p = np.stack([np.interp(tq, t, meta["pos"][:, k]) for k in range(3)], 1) - meta["pos"][i]
    calib_from_device = rot_from_euler(meta["rpy_calib"]).T
    p_calib = (calib_from_device @ meta["R"][i].T @ p.T).T
    v = np.interp(tq, t, np.linalg.norm(meta["vel"], axis=1))
    p_calib[~ok], v[~ok] = np.nan, np.nan
    return p_calib, v


def decode_hevc(path):
    """All frames of a raw HEVC stream as (Y, U, V) uint8 planes (no color conversion: NV12 in, NV12 out)."""
    import av
    with av.open(str(path), format="hevc") as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        out = []
        for fr in c.decode(st):
            a = fr.to_ndarray(format="yuv420p")
            h, w = a.shape[0] * 2 // 3, a.shape[1]
            out.append((a[:h], a[h:h + h // 4].reshape(h // 2, w // 2), a[h + h // 4:].reshape(h // 2, w // 2)))
        return out


def segment_model_frames(seg_dir, meta=None):
    """Warped + packed model inputs for a comma1M segment: (N, 2, 6, 128, 256) uint8, [road, wide]."""
    seg_dir = Path(seg_dir)
    meta = meta or load_segment_meta(seg_dir)
    wh = meta["fcam_wh"]
    f_road, f_wide = CAMERAS[wh]
    warp_road = Warper(get_warp_matrix(meta["rpy_calib"], intrinsics(*wh, f_road), False), wh)
    warp_wide = Warper(get_warp_matrix(meta["rpy_calib"], intrinsics(*wh, f_wide), True), wh)
    with ThreadPoolExecutor(2) as ex:
        road, wide = ex.map(decode_hevc, (seg_dir / "fcamera.hevc", seg_dir / "ecamera.hevc"))
    n = min(len(road), len(wide))
    out = np.empty((n, 2, 6, MODEL_H // 2, MODEL_W // 2), np.uint8)
    for i in range(n):
        warp_road(*road[i], out=out[i, 0])
        warp_wide(*wide[i], out=out[i, 1])
    return out, meta
