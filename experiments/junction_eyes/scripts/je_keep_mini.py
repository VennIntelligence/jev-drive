"""A mini keep store for the step-1 smoke and gates (the real one is executor 2's; same layout, je_data.KeepStore):
a few comma1M segments decoded to model frames (jevdrive.openpilot.frames.segment_model_frames, native cameras), the factory
Cinque stepped modeld-style at 20 Hz from the segment start (jevdrive.op_adapt.Stepper, fp32 port), rows every --every 20 Hz
steps from step 36 on (full 1.8 s history), the frames at the rows' 5 Hz phase stored once.

  python je_keep_mini.py --segs 4 [--out $DATA_DIR/runs/junction_eyes/step1/keep_mini] [--held 1]
The last --held segments are marked held (exercises the held mask; nothing here is read).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, os  # noqa: E401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.openpilot.frames import segment_model_frames  # noqa: E402

AT = (0.275, 0.525)


def main(a):
    from jevdrive.run import Run
    out = _pl.Path(a.out)
    with Run("junction_eyes", "step1/keep-mini", config=vars(a)) as run:
        root = data_dir() / "datasets" / "comma1M"
        segs = sorted(p.name for p in root.iterdir() if (p / "fcamera.hevc").exists())[a.skip:a.skip + a.segs]
        net = A.load("cinque", torch.float32).cuda()
        st = A.Stepper(net, "cinque")
        frames, fidx, teach, seg, held = [], [], [], [], []
        for i, s in enumerate(segs):
            fr, _ = segment_model_frames(root / s, max_frames=a.max_frames)
            raw = st.run_stream(fr, AT, repeat=1)[0]                       # (n, n_out) per 20 Hz step
            t0 = np.arange(36, len(fr), a.every)
            t0 = t0[t0 % 4 == 0]
            base = sum(len(f) for f in frames)
            frames.append(fr[0::4])
            fidx.append(base + t0[:, None] // 4 + np.arange(-9, 1)[None])
            teach.append(raw[t0])
            seg += [s] * len(t0)
            held += [i >= a.segs - a.held] * len(t0)
            run.info(f"{s}: {len(fr)} frames, {len(t0)} rows")
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "frames.npy", np.concatenate(frames))
        np.savez(out / "rows.npz", fidx=np.concatenate(fidx), teacher=np.concatenate(teach).astype(np.float32), seg=np.array(seg),
                 held=np.array(held), action_t=np.array(AT))
        run.summary.update(rows=len(seg), segs=segs, out=str(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segs", type=int, default=4)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--held", type=int, default=1)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--out", default=str(data_dir() / "runs" / "junction_eyes" / "step1" / "keep_mini"))
    main(ap.parse_args())
