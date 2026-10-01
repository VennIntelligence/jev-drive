#!/usr/bin/env python
"""openpilot Cinque on the Cosmos pilot clips (fc65452:todos/2026-09-28-cosmos-pilot.md, checks 2 and 3). envs/openpilot.

  CUDA_VISIBLE_DEVICES=1 python experiments/cosmos/lib/cosmos_openpilot.py --variant edgeA [--pairs 24211-s0,...]

The pilot camera (scripts/cosmos_pair_agent.CAM: pinhole 1280 x 704, 64 deg, axis-aligned with the car) is written
as a one-camera WOD calibration record, so openpilot's road and wide model frames are rendered by the same camgeom
path as every other openpilot run in this repo (rotation-only, nearest neighbour, chroma 2 x 2 mean). Cinque is
stepped once per 20 Hz frame from a zero state (native rate, no hold). Streams per pair:

  raw_plus, raw_minus         CARLA RGB (the lossless rgb.mp4 Cosmos was given)
  raw_comp                    raw x+ with the hazard region (gt.npz `region`) filled from raw x-
  <v>_plus, <v>_minus         Cosmos outputs of variant v (raw npy)
  <v>_comp                    Cosmos x+ with the hazard region filled from Cosmos x-
  <v>_rep, <v>_alt            x- again with the same seed / another seed (noise floors), when they exist
Per stream and frame: temporal (Cinque `select_4`), plan speed and lateral position at 2 s, lead prob / x at t = 0.
Output: $DATA_DIR/runs/cosmos/op/<variant>/<pair>.npz
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("scripts",)]
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import wod_zeroshot_openpilot as WZ  # noqa: E402
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402

DATA = Path(os.environ["DATA_DIR"])
ROOT = Path(os.environ.get("COSMOS_ROOT", DATA / "runs" / "cosmos"))
W, H, FOV = 1280, 704, 64.0
F = W / 2.0 / np.tan(np.radians(FOV) / 2.0)
SEED, SEED_ALT = 2025, 2026


def calib() -> dict:
    """The pilot camera as a WOD calibration record: camera x = optical axis, y left, z up, in the vehicle frame."""
    E = np.eye(4)
    E[:3, 3] = 1.519, 0.026, 1.806          # Waymo rear-axle frame, as P5's front camera
    return {"intrinsic": [F, F, (W - 1) / 2.0, (H - 1) / 2.0, 0, 0, 0, 0, 0], "extrinsic": E.ravel().tolist(),
            "width": W, "height": H}


def maps() -> dict:
    cal = {1: calib()}
    idx = {}
    for k in ("road", "wide"):
        src, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), cal)
        assert (src >= 0).all(), f"{k}: model frame not covered by the pilot camera"
        idx[k] = G.nn_gather_index(src, U, V, [(W, H)]).ravel()
    return idx


def model_frames(rgb: np.ndarray, idx: dict) -> np.ndarray:
    """(T, H, W, 3) RGB uint8 -> (T, 2, 6, 128, 256) packed [road, wide]."""
    from PIL import Image
    out = np.empty((len(rgb), 2, 6, 128, 256), np.uint8)
    for t, f in enumerate(rgb):
        ycc = np.asarray(Image.fromarray(f).convert("YCbCr")).reshape(-1, 3)
        for m, k in enumerate(("road", "wide")):
            out[t, m] = WZ._pack(ycc[idx[k]].reshape(G.OP_H, G.OP_W, 3))
    return out


def read_mp4(path: Path) -> np.ndarray:
    import glob
    import subprocess
    ff = sorted(glob.glob(str(DATA / "envs/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*")))[0]
    out = subprocess.run([ff, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H, W, 3)


def run(m, frames: np.ndarray) -> dict:
    from jevdrive.openpilot.model import T_IDXS, decode
    tap = D.OP_TAPS[m.name]["temporal"]
    m.reset()
    rows = {k: [] for k in ("temporal", "v2", "y2", "lead_prob", "lead_x")}
    for f in frames:
        out = m.step(f, action_t=WZ.ACTION_T)
        d = decode(out, m.slices, 0.0, WZ.ACTION_T)
        rows["temporal"].append(m.tap_values[tap].copy())
        rows["v2"].append(np.interp(2.0, T_IDXS, d["plan_vel"][:, 0]))
        rows["y2"].append(np.interp(2.0, T_IDXS, d["plan_pos"][:, 1]))
        rows["lead_prob"].append(float(np.ravel(d["lead_prob"])[0]))
        rows["lead_x"].append(float(d["lead"][0, 0, 0]))
    return {k: np.asarray(v, np.float32) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    import csv
    from jevdrive.openpilot.model import OPModel
    from jevdrive.runlog import RunLog
    rl = RunLog("cosmos", "openpilot", a.variant)
    p = csv.DictReader(open(Path(__file__).resolve().parents[3] / "experiments/cosmos/results/pairs.csv"))
    names = [x for x in a.pairs.split(",") if x] or [r["pair"] for r in p]
    # CUDA EP, not the TensorRT cache: the cached engines were built on the previous box's card (188 SMs; TensorRT
    # warns "deadlocking is likely" on this one). Every stream here goes through the same backend.
    m = OPModel("cinque", "cuda-iob", taps=list(D.OP_TAPS["cinque"].values()))
    idx = maps()
    od = ROOT / "op" / a.variant
    od.mkdir(parents=True, exist_ok=True)
    v, cos = a.variant, ROOT / "out" / a.variant
    for pair in names:
        gt = np.load(ROOT / "clips" / pair / "gt.npz")
        region = np.unpackbits(gt["region"], axis=1)[:, : H * W].reshape(-1, H, W).astype(bool)[..., None]
        clips = {"raw_plus": read_mp4(ROOT / "clips" / pair / "plus" / "rgb.mp4"),
                 "raw_minus": read_mp4(ROOT / "clips" / pair / "minus" / "rgb.mp4")}
        for key, name in ((f"{v}_plus", f"{pair}_plus_{v}_s{SEED}"), (f"{v}_minus", f"{pair}_minus_{v}_s{SEED}"),
                          (f"{v}_rep", f"{pair}_minus_{v}_s{SEED}_rep"), (f"{v}_alt", f"{pair}_minus_{v}_s{SEED_ALT}")):
            f = cos / f"{name}.npy"
            if f.exists():
                clips[key] = np.load(f)
        b = {"M2": "E2", "P2": "E2"}.get(v)          # borrowed seed floor (experiments.cosmos.lib.cosmos_eval.ALT_FROM)
        if f"{v}_alt" not in clips and b and (ROOT / "out" / b / f"{pair}_minus_{b}_s{SEED_ALT}.npy").exists():
            clips[f"{v}_alt"] = np.load(ROOT / "out" / b / f"{pair}_minus_{b}_s{SEED_ALT}.npy")
            clips[f"{v}_altbase"] = np.load(ROOT / "out" / b / f"{pair}_minus_{b}_s{SEED}.npy")
        clips["raw_comp"] = np.where(region, clips["raw_minus"], clips["raw_plus"])
        if f"{v}_plus" in clips and f"{v}_minus" in clips:
            clips[f"{v}_comp"] = np.where(region, clips[f"{v}_minus"], clips[f"{v}_plus"])
        res = {}
        for key, rgb in clips.items():
            assert rgb.shape == (len(region), H, W, 3), (key, rgb.shape)
            for k, arr in run(m, model_frames(rgb, idx)).items():
                res[f"{key}/{k}"] = arr
        np.savez(od / f"{pair}.npz", **res)
        rl.info(f"{pair}: {sorted(clips)}")
        rl.event("pair", pair=pair, streams=sorted(clips))


if __name__ == "__main__":
    main()
