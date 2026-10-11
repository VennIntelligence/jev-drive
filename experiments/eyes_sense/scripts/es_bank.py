#!/usr/bin/env python3
"""Rebuild of the vtr_W0 token bank (decision 248's RW0 bank: the frozen shipped encoder on the t0 views CAM_F0 / CAM_L0 / CAM_R0, protocol-W
image pairs), after the disk cleanup of 2026-10-11 (about 13:00 CST) removed runs/op_parity/mem/vtr_* and the vis_train pixel caches.

  CAM_F0          the t0 slot token of the W token cache (cache/<data>@warp/front.npy[:, -1]). The deleted bank's F0 view differed from it by
                  0.0003 mean |d| (token RMS 1.691; experiments/vis_train/results/reads.md, bank table), the fp16 batch noise of the same
                  encoder on the same pixels, so no front pixel cache is rebuilt.
  CAM_L0 / R0     the protocol-W pairs of vis_train's side-view cache (scripts/vt_side.py split + build, unchanged; lib/side_store.py layout
                  (N, 2 cams, 2 frames [t0 - 0.2 s, t0], 2, 6, 128, 256)) through the frozen Cinque encoder (experiments/op_parity/scripts/pp_prep.encoder)
Writes $DATA_DIR/runs/op_parity/mem/vtr_W0/<data>.npy (N, 3, 32, 512) fp16 in tab order, views [CAM_F0, CAM_L0, CAM_R0] (vis_train vt.VIEWS), and
bank.json (src cinque-w3-rebuilt; per data dir rows, mean |token| and RMS per view).

  $DATA_DIR/envs/op-train/bin/python experiments/eyes_sense/scripts/es_bank.py --datas navtrain_full.s0of12 ... lb_navtest lb_navhard     (GPU)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

FULL = [f"navtrain_full.s{i}of12" for i in range(12)]
CACHE = data_dir() / "runs/op_parity/cache"
SIDE = data_dir() / "runs/vis_train/px_side"
OUT = data_dir() / "runs/op_parity/mem/vtr_W0"


def main():
    import torch
    import pp_prep as PP
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--datas", nargs="+", default=FULL + ["lb_navtest", "lb_navhard"])
    ap.add_argument("--rows", type=int, default=512)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mf = OUT / "bank.json"
    meta = json.loads(mf.read_text()) if mf.exists() else {"kind": "vtr_W0", "src": "cinque-w3-rebuilt", "step": 0, "rows": {}, "stats": {},
                                                           "note": "rebuilt 2026-10-11 by experiments/eyes_sense/scripts/es_bank.py (module docstring)"}
    with Run("eyes_sense", "bank-vtr_W0", config=vars(a)) as run:
        _, enc = PP.encoder(torch.device("cuda"))
        for d in a.datas:
            names = np.load(CACHE / d / "tab.npz")["names"]
            f = OUT / f"{d}.npy"
            if f.exists() and meta["rows"].get(d) == len(names):
                continue
            front = np.load(CACHE / f"{d}@warp" / "front.npy", mmap_mode="r")
            side = np.load(SIDE / d / "side_t0.npy", mmap_mode="r")
            sn = np.load(SIDE / d / "meta.npz")["names"]
            assert len(front) == len(side) == len(names) and (sn == names).all(), f"{d}: row order of the caches differs"
            tmp = OUT / f".{d}.{os.getpid()}.npy"
            o = np.lib.format.open_memmap(tmp, "w+", np.float16, (len(names), 3, 32, 512))
            for i in run.tqdm(range(0, len(names), a.rows), desc=d):
                r = slice(i, min(i + a.rows, len(names)))
                o[r, 0] = front[r, -1]
                s = np.asarray(side[r])                                          # (B, 2, 2, 2, 6, 128, 256)
                B = len(s)
                o[r, 1:] = enc(s[:, :, 0].reshape(B * 2, *s.shape[3:]), s[:, :, 1].reshape(B * 2, *s.shape[3:])).reshape(B, 2, 32, 512)
            x = np.asarray(o[:4096], np.float32)
            assert np.isfinite(x).all()
            meta["stats"][d] = {v: dict(mean_abs=float(np.abs(x[:, j]).mean()), rms=float(np.sqrt((x[:, j] ** 2).mean()))) for j, v in enumerate(("CAM_F0", "CAM_L0", "CAM_R0"))}
            o.flush()
            del o
            os.replace(tmp, f)
            meta["rows"][d] = len(names)
            mf.write_text(json.dumps(meta, indent=1))
            run.info(f"vtr_W0: {d} {len(names)} rows -> {f}; {meta['stats'][d]}")
        run.summary.update(meta)


if __name__ == "__main__":
    main()
