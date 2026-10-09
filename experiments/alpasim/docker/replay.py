"""Driver-only parity check: replay decisions a NATIVE run dumped (SH30_DUMP=n: <driver-logs>/dump/sNN_kK.npz + drive.jsonl) through the
model inside the submission image, so a native-vs-containerised score difference can be split into "the driver computes something else"
and "the driver is shown something else" (renderer, hardware). Per dump:
  pack   the dumped CAM_F0 JPEG -> model frames here (PIL / libjpeg decode + sampling maps) against the native frames of the t0 slot: CPU only
  model  the dumped slot frames + ego features -> plan here (encoder + policy on this GPU, fp16) against the native plan: mu and the exported poses
  docker run --rm --gpus device=0 -v <driver-logs>:/n:ro -v $PWD/experiments/alpasim/docker/replay.py:/replay.py:ro <image> \
      python /replay.py /n <tag> [ap2|sh30]
Prints one JSON line per dump and a summary. Needs a tag of the family the dump was made with.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/jev-drive/experiments/alpasim/lib")
import ap2_core as AC  # noqa: E402
import sh30_core as C  # noqa: E402
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402


def main() -> None:
    import torch
    root, tag, fam = Path(sys.argv[1]), sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "ap2")
    core = AC.Core(tag) if fam == "ap2" else C.Core(tag)
    cams = {r["n"]: r["cam"] for r in map(json.loads, open(root / "drive.jsonl")) if r.get("kind") == "start"}
    rows = []
    for f in sorted((root / "dump").glob("s*_k*.npz")):
        d = np.load(f, allow_pickle=True)
        cur, valid, ego, cam_t = d["cur"], d["valid"], d["ego"], np.asarray(d["cam_t"], np.float64)
        cam = cams.get(int(f.name[1:3]))
        fr = C.pack(d["jpeg"].tobytes(), C.camera(cam["K"], cam["R"], cam["t"], cam["D"])) if cam and d["jpeg"].size else None
        prev = np.concatenate([np.zeros((1,) + C.FRAME, np.uint8), cur[:-1]])
        with torch.no_grad():
            p, c = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
            H = core.model.net.run_batched(core.A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *core.A.H_SHAPE)
            out = core.model(H, torch.from_numpy(ego[None]).to(core.dev), torch.tensor([[1.0, 0.0]], device=core.dev)).float()
            mu = out[0, core.pi].reshape(33, 15).cpu().numpy()
        poses = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")
        rows.append({"dump": f.name, "pack_pixels_differ": None if fr is None else int((fr != cur[-1]).sum()),
                     "mu_max_abs": float(np.abs(mu - d["mu"]).max()), "plan_xy_max_m": float(np.abs(mu[:, :2] - d["mu"][:, :2]).max()),
                     "pose_xy_max_m": float(np.abs(poses[:, :2] - d["poses"][:, :2]).max()), "pose_yaw_max_rad": float(np.abs(poses[:, 2] - d["poses"][:, 2]).max())})
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"n": len(rows), "tag": tag, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                      "pack_identical": sum(r["pack_pixels_differ"] == 0 for r in rows),
                      "pose_xy_max_m": max(r["pose_xy_max_m"] for r in rows), "pose_xy_median_m": float(np.median([r["pose_xy_max_m"] for r in rows])),
                      "mu_max_abs": max(r["mu_max_abs"] for r in rows)}))


if __name__ == "__main__":
    main()
