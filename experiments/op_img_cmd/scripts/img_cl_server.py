#!/usr/bin/env python
"""op_img_cmd closed-loop smoke: the op_arb openpilot server with the sky-arrow command drawn into every frame (envs/openpilot).
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md. Same server as experiments/op_closed_loop/archive/op_arb_server.py (imported,
not copied); when a request's meta carries "img_cmd" = [command, distance to the junction in m or None] (lib/op_arb_agent.py
img_cmd(): the official route only), the packed road + wide frames get img_overlay.draw_sky before the model steps. The model
steps one frame per request, so every frame of its history carries the arrow drawn with that frame's own distance.

  op_arb.sh with SRV_PY=experiments/op_img_cmd/scripts/img_cl_server.py (and SRV_ONNX = the fine-tuned model)
"""
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / "experiments" / "op_closed_loop" / "archive"), str(R / "scripts"), str(R), str(Path(__file__).resolve().parent)]
import img_overlay as O  # noqa: E402
import op_arb_server as S  # noqa: E402


class ImgArbModel(S.ArbModel):
    def plan(self, state, meta, prep):
        c = meta.get("img_cmd")
        if c:
            d = np.nan if c[1] is None else float(c[1])
            prep = dict(prep, img2=O.draw_sky(np.asarray(prep["img2"]), str(c[0]), d, (0.0, 0.0, 0.0)))
            self.plan_dump(prep["img2"], c)
        return super().plan(state, meta, prep)

    _n = 0

    def plan_dump(self, img2, c):
        """IMG_CL_DUMP=<dir>: every 200th overlaid request as a png (both views, RGB from YUV) to check the drawing."""
        import os
        d = os.environ.get("IMG_CL_DUMP", "")
        ImgArbModel._n += 1
        if not d or ImgArbModel._n % 200:
            return
        from PIL import Image
        from jevdrive import op_interp as I

        def rgb(packed):
            Y, U, V = (z.astype(np.float32) for z in I.unpack(packed))
            U, V = (np.repeat(np.repeat(z, 2, 0), 2, 1) - 128 for z in (U, V))
            return np.clip(np.stack([Y + 1.402 * V, Y - 0.344 * U - 0.714 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(np.concatenate([rgb(img2[0]), rgb(img2[1])], 0)).save(f"{d}/{ImgArbModel._n:07d}_{c[0]}_{c[1]}.png")

if __name__ == "__main__":
    S._patch_step()
    S.ZP.OpenpilotModel = ImgArbModel
    sys.exit(S.ZP.main())
