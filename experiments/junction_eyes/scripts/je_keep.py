"""junction_eyes keep set (comma1M, native 1.22 m camera, no geometry labels): model frames of turn / straight event windows,
the held-out segments, and the factory-Cinque teacher cache. Layout and numbers: experiments/junction_eyes/results/keep_set.md.

  census                 camera type / ecamera / length of every segment dir -> $RUN/census.json
  select                 events (op_fov scan, rescanned for every usable segment) -> held-out segments, windows, rows -> $RUN/{windows,rows}.json
  frames                 decode + warp every kept window (streaming, native focals) -> $RUN/frames.npy (F, 2, 6, 128, 256) uint8
  pairs                  the rows' t0 image pairs in the shared layout -> $RUN/t0.npy (N, 2, 2, 6, 128, 256) uint8
  teacher                factory Cinque stepped at 20 Hz over every window (op_adapt.Stepper) -> $RUN/teacher/*.npy
RUN = $DATA_DIR/runs/junction_eyes/keep.
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "experiments/op_fov/scripts")]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "datasets" / "comma1M"
RUN = data_dir() / "runs" / "junction_eyes" / "keep"
SEED = 0
N_HOLD = 60
ACTION_T = (0.275, 0.525)                       # as navsim_zs_openpilot / fov_replay


def segs():
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and (p / "frame_info.safetensors").exists())


def _census1(s):
    from jevdrive.openpilot.frames import load_safetensors
    fi = load_safetensors(ROOT / s / "frame_info.safetensors")
    return dict(seg=s, wh=[int(fi["fcamera/width"][0]), int(fi["fcamera/height"][0])], frames=int(fi["fcamera/frame_count"][0]),
                ecam=bool("ecamera/width" in fi and int(fi["ecamera/frame_count"][0]) > 0 and (ROOT / s / "ecamera.hevc").exists()),
                device=int(fi["device_type"]))


def cmd_census(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("junction_eyes", "keep-census", config=vars(a)) as run:
        res = par.pmap(_census1, segs(), run=run, desc="segments")
        res.raise_if_failed()
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / "census.json").write_text(json.dumps(res.values))
        from collections import Counter
        run.info("%s", Counter((tuple(v["wh"]), v["ecam"], v["device"]) for v in res.values))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    from jevdrive.run import cli_args
    for c in ("census",):
        cli_args(sp.add_parser(c))
    a = ap.parse_args()
    {"census": cmd_census}[a.cmd](a)
