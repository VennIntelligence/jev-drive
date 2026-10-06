"""op_parity gap page, adapter input frames: for the tokens of cases.json, rebuild exactly what P2 received under protocol W (the 4 CAM_F0 keys of
the op_lb navtest cache + the 6 CPU-warped lattice frames, steps 2, 6, .., 30 of the op_lb rollout) and convert the packed YUV frames to RGB.
CPU only, one process (a handful of tokens). Writes gap/frames.npz: per token (8, 2, 256, 512, 3) uint8 [slot, road / wide] and the slot times.

  python experiments/op_parity/scripts/pp_gap_frames.py       (box, .venv; run pp_gap_export.py first)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import json  # noqa: E402

import numpy as np  # noqa: E402

import op_lb as OL  # noqa: E402
import pp_prep as PP  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def rgb(packed):
    """packed (6, 128, 256) -> (256, 512, 3) uint8 RGB from the YUV planes (as op_img_cmd/img3_sheet.rgb)."""
    Y, U, V = (z.astype(np.float32) for z in I.unpack(packed))
    U, V = (np.repeat(np.repeat(z, 2, 0), 2, 1) - 128 for z in (U, V))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344 * U - 0.714 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def main():
    gap = data_dir() / "runs" / "op_parity" / "gap"
    cases = json.loads((gap / "cases.json").read_text())["cases"]
    mt = OL.meta("lb_navtest")
    names = {t: i for i, t in enumerate(mt["names"])}
    keys = OL.Keys("lb_navtest")
    ts, src = OL._steps(0.0, False)
    out, slot_t = {}, ts[PP.STEPS].tolist()
    for c in cases:
        i = names[c["token"]]
        kf = np.asarray(keys[i])
        sf = PP._warp_job((kf, mt["pose"][i], mt["vel"][i], mt["cam"][i], np.asarray(mt["syn_t"])))
        img = lambda s: kf[src[s][1]] if src[s][0] == "k" else sf[src[s][1]]  # noqa: E731
        out[c["token"]] = np.stack([[rgb(img(s)[v]) for v in (0, 1)] for s in PP.STEPS])
        print(c["token"], out[c["token"]].shape, flush=True)
    np.savez_compressed(gap / "frames.npz", slot_t=np.array(slot_t), **out)
    print("slot times", slot_t)


if __name__ == "__main__":
    main()
