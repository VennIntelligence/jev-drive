#!/usr/bin/env python
"""op_arb openpilot server that also dumps what the model actually received and said (envs/openpilot, loss-budget example reruns).

Same server as experiments/op_closed_loop/archive/op_arb_server.py (imported, not copied; nothing about the model call changes).
LBX_DUMP=<dir>: every LBX_EVERY-th (default 2 = 10 Hz) plan request writes
  <dir>/<seq>.jpg   the packed road + wide model frames exactly as fed to the network (YUV -> RGB), side by side, 1024 x 256
  <dir>/dump.jsonl  {seq, cid, t (agent frame time = scenario clock), v, pos (33 x 3, x fwd / y right), lane_lines (4 x 33, y right),
                     road_edges (2 x 33), lane_prob, lead (first lead, t = 0: x, y, v, a), lp (lead prob)}
Lane lines / road edges need OP_LANES=1 in the server env (op_arb_server). The JPEG encode runs on a writer thread.

  op_arb.sh with SRV_PY=experiments/leaderboard_audit/scripts/lbx_dump_server.py
"""
import itertools
import json
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / "experiments" / "op_closed_loop" / "archive"), str(R / "scripts"), str(R)]
import op_arb_server as S  # noqa: E402

DUMP, EVERY = os.environ.get("LBX_DUMP", ""), int(os.environ.get("LBX_EVERY", "2"))
Q, SEQ, CID = queue.Queue(maxsize=400), itertools.count(), itertools.count()


def rgb(packed):
    from jevdrive import op_interp as I
    Y, U, V = (z.astype(np.float32) for z in I.unpack(packed))
    U, V = (np.repeat(np.repeat(z, 2, 0), 2, 1) - 128 for z in (U, V))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344 * U - 0.714 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def writer():
    from PIL import Image
    f = open(Path(DUMP) / "dump.jsonl", "a")
    while True:
        seq, img2, rec = Q.get()
        Image.fromarray(np.concatenate([rgb(img2[0]), rgb(img2[1])], 1)).save(f"{DUMP}/{seq:07d}.jpg", quality=88)
        f.write(json.dumps(dict(rec, seq=seq)) + "\n")
        f.flush()


def r(x, n=2):
    return np.round(np.asarray(x, float), n).tolist()


class DumpModel(S.ArbModel):
    def plan(self, state, meta, prep):
        info, out = super().plan(state, meta, prep)
        if DUMP:
            state["lbx_cid"] = state.get("lbx_cid", next(CID))
            state["lbx_n"] = n = state.get("lbx_n", -1) + 1
            if n % EVERY == 0:
                try:
                    lead = np.asarray(out["lead"])[0, 0]
                    Q.put_nowait((next(SEQ), np.array(prep["img2"]), dict(
                        cid=state["lbx_cid"], t=meta.get("t"), v=meta.get("speed"), pos=r(out["pos"]),
                        lane_lines=r(out["lane_lines"]) if "lane_lines" in out else None,
                        road_edges=r(out["road_edges"]) if "road_edges" in out else None,
                        lane_prob=r(out["lane_prob"], 3), lead=r(lead), lp=r(np.asarray(out["lead_prob"])[0], 3))))
                except queue.Full:
                    pass
        return info, out


if __name__ == "__main__":
    if DUMP:
        Path(DUMP).mkdir(parents=True, exist_ok=True)
        threading.Thread(target=writer, daemon=True).start()
    S._patch_step()
    S.ZP.OpenpilotModel = DumpModel
    sys.exit(S.ZP.main())
