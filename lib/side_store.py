"""t0 side-camera pixels for lane VT's arm W (experiments/vis_train, prereg amendments 1 and 2): the protocol-W image pair of CAM_L0
and CAM_R0, each rendered as openpilot road + wide model frames along its own mounting yaw.

The cache (built by experiments/vis_train/scripts/vt_side.py, one dir per pp_prep data dir, rows in its tab.npz order, i.e. the rows of
lib/pixel_store.PixelStore over the same `datas`):

  $DATA_DIR/runs/vis_train/px_side/<data>/side_t0.npy  (N, 2, 2, 2, 6, 128, 256) uint8: [CAM_L0, CAM_R0] x [frame at t0 - 0.2 s, frame at t0]
                                                       x [road, wide]. The t0 frame is P3's rendering of the t0 key (pp_prep.render_side:
                                                       jevdrive.navsim_zs.OpenpilotMaps(cam, yaw_deg = mounting yaw)).
  $DATA_DIR/runs/vis_train/px_side/<data>/meta.npz     names, cam_t (N, 2, 3) camera positions (ego frame), yaw (N, 2) mounting yaws (deg),
                                                       pose (N, 2, 3) ego pose at t0 - 0.2 s and at t0 (x, y, yaw in the t0 frame)

The pair is the front camera's protocol-W t0 pair: (the t0 key warped by the ego motion to the pose 0.2 s earlier, the t0 key);
jevdrive.op_interp `warp` between keys takes the nearer key. A camera at position c looking along yaw psi is a forward camera on a
vehicle frame turned by psi about the vertical (`virtual`: camera position R(-psi) c, poses (x, y, yaw + psi)), so op_interp's
road-plane warp (warp_frame) applies to a side camera unchanged.

    SS = SideStore(datas, dev)
    prev, cur = SS.t0(rows)          # uint8 (B, 2, 2, 6, 128, 256) each, on dev: per camera [CAM_L0, CAM_R0] the pair of the t0 slot
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevdrive.common import data_dir, n_cpus

CAMS = ("CAM_L0", "CAM_R0")
FRAME = (2, 6, 128, 256)
FB = int(np.prod(FRAME))                       # bytes per camera frame (393 216)
RB = len(CAMS) * 2 * FB                        # bytes per row


def side_root() -> Path:
    return data_dir() / "runs" / "vis_train" / "px_side"


def virtual(cam_t, yaw_deg, pose):
    """A camera at cam_t (..., 3) looking along yaw_deg (...) on an ego at pose (..., 3) -> the camera position and the pose of the
    vehicle frame turned by the yaw, in which the camera looks forward (same shapes)."""
    cam_t, pose, a = np.asarray(cam_t, np.float64), np.asarray(pose, np.float64), np.radians(np.asarray(yaw_deg, np.float64))
    c, s = np.cos(a), np.sin(a)
    cam = np.stack([c * cam_t[..., 0] + s * cam_t[..., 1], -s * cam_t[..., 0] + c * cam_t[..., 1], cam_t[..., 2]], -1)
    return cam, np.stack([pose[..., 0], pose[..., 1], pose[..., 2] + a], -1)


class SideStore:
    def __init__(self, datas: list[str], dev, root=None, threads: int = 0):
        root = Path(root) if root else side_root()
        self.dev, self.datas = dev, list(datas)
        self.files = [root / d / "side_t0.npy" for d in datas]
        self.fd, self.base, n, names = [], [], [], []
        for f in self.files:
            m = np.load(f, mmap_mode="r")
            assert m.shape[1:] == (len(CAMS), 2) + FRAME and m.dtype == np.uint8, f"{f}: {m.shape} {m.dtype}"
            self.base.append(int(m.offset)), n.append(len(m))
            del m
            self.fd.append(os.open(f, os.O_RDONLY))
            names.append(np.load(f.with_name("meta.npz"))["names"])
            assert len(names[-1]) == n[-1], f
        self.off = np.cumsum([0] + n)
        self.n = int(self.off[-1])
        self.names = np.concatenate(names)
        self.threads = threads or max(2, min(16, n_cpus(), len(os.sched_getaffinity(0))))
        self.pool = ThreadPoolExecutor(self.threads)

    def __len__(self):
        return self.n

    def close(self):
        self.pool.shutdown(wait=False)
        for fd in self.fd:
            os.close(fd)
        self.fd = []

    def host(self, rows):
        """rows -> page-locked uint8 tensor (B, 2, 2, 2, 6, 128, 256), one positional read per row (os.preadv releases the GIL; as
        PixelStore.host, so it may run in pp_train.Prefetch worker threads)."""
        import torch
        r = (rows.cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)).astype(np.int64).reshape(-1)
        buf = torch.empty((len(r), len(CAMS), 2) + FRAME, dtype=torch.uint8, pin_memory=True)
        mv = memoryview(buf.numpy()).cast("B")
        s = np.searchsorted(self.off, r, side="right") - 1
        at = [self.base[k] + int(i - self.off[k]) * RB for i, k in zip(r, s)]

        def rd(js):
            for j in js:
                got, fd = 0, self.fd[s[j]]
                while got < RB:
                    k = os.preadv(fd, [mv[j * RB + got:(j + 1) * RB]], at[j] + got)
                    if k <= 0:
                        raise IOError(f"short read of row {int(r[j])} in {self.files[s[j]]}")
                    got += k
        order = np.argsort(np.asarray(at) + s * (1 << 50), kind="stable")          # file order inside each thread's share
        list(self.pool.map(rd, [order[i::self.threads] for i in range(min(self.threads, len(r)))]))
        return buf

    def t0(self, rows):
        """rows -> (prev, cur) uint8 (B, 2, 2, 6, 128, 256) on the device: per camera the protocol-W pair of the t0 slot."""
        g = self.host(rows).to(self.dev, non_blocking=True)
        return g[:, :, 0], g[:, :, 1]
