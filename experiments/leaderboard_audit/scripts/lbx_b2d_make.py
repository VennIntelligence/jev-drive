"""Render every B2D loss-budget example (envs/openpilot python, CPU only): lbx_b2d_make.py [name ...]
Outputs to $DATA_DIR/runs/lbx_b2d/; the fragment is tmp/lbx/b2d_fragment.md, the media go to experiments/leaderboard_audit/figs/loss_budget_examples/.
Segments are scenario seconds t0:t1[:speedup]; attempts are the vlm_arb chase reruns (v2-gif*) or vmerge2 runs (BEV + model view)."""
import contextlib
import io
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import lbx_b2d_clips as C  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
A, OUT = D / "runs/vlm_arb/arms", D / "runs/lbx_b2d"
# name: (attempt, label, segments, bev)
JOBS = {
    "b2d_red_drive_24944_s0": ("v2-gif-drive-s0-a/attempts/24944/1", "drive r24944 s0", ["44:60"], False),
    "b2d_red_vred_24944_s0": ("v2-gif-vred-s0-a/attempts/24944/1", "vred r24944 s0", ["45:52", "52:74:6", "74:79"], False),
    "b2d_red_drive_24944_s0_sheet": ("v2-gif-drive-s0-a/attempts/24944/1", "drive r24944 s0", ["50:50", "55:55", "57:57"], False),
    "b2d_red_vred_24944_s0_sheet": ("v2-gif-vred-s0-a/attempts/24944/1", "vred r24944 s0", ["50:50", "55:55", "76.5:76.5"], False),
    "b2d_red_vred_27297_s0": ("v2-gif-vred-s0-a/attempts/27297/1", "vred r27297 s0", ["7:24"], False),
    "b2d_red_vred_16390_s0": ("v2-gif-vred-s0-a/attempts/16390/1", "vred r16390 s0", ["3:12"], False),
    "b2d_blocked_drive_19324_s0": ("v2-gif-drive-s0-a/attempts/19324/1", "drive r19324 s0", ["30:42", "42:196:20", "196:200"], False),
    "b2d_blocked_drive_9196_s0": ("v2-gifr2-drive-s0-d/attempts/9196/1", "drive r9196 s0", ["26:34", "34:50:3", "50:90:10"], False),
    "b2d_blocked_vmerge2_15612_s1": ("v2-vmerge2-s1-q1/attempts/15612/1", "vmerge2 r15612 s1", ["6:24", "24:81:8"], True),
    "b2d_byp_pbyp2ng_19832_s0": ("v2-gif-pbyp2ng-s0-a/attempts/19832/1", "pbyp2ng r19832 s0", ["14:26"], False),
    "b2d_byp_pbyp2ng_2520_s0": ("v2-gif-pbyp2ng-s0-a/attempts/2520/2", "pbyp2ng r2520 s0", ["14:26"], False),
    "b2d_byp_vmerge2_19832_s1": ("v2-vmerge2-s1-q1/attempts/19832/1", "vmerge2 r19832 s1", ["30:41"], True),
    "b2d_coll_drive_27043_s0": ("v2-gif-drive-s0-a/attempts/27043/1", "drive r27043 s0", ["16:27"], False),
    "b2d_coll_vmerge2_27043_s0": ("v2-vmerge2-s0-q0/attempts/27043/1", "vmerge2 r27043 s0", ["11:22"], True),
    "b2d_stop_vmerge2_17280_s2": ("v2-vmerge2-s2-q0/attempts/17280/1", "vmerge2 r17280 s2", ["5:13:2", "13:20"], True),
}


def one(name):
    att, label, segs, bev = JOBS[name]
    out = OUT / (name + (".png" if name.endswith("_sheet") else ".gif"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run = C.Run(A / att)
        for i, th in run.hits.items():
            print("  first contact id=%d at scenario t=%.2f s, ego v=%.1f m/s" % (i, th, run.at(run.ticks, "t", th)["v"]))
        C.render(run, str(out), label, segs, 480, 8 if bev else 6, bev, 3.0)
    return name + "\n" + buf.getvalue()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(JOBS)
    with Pool(min(len(names), 8)) as p:
        for s in p.imap_unordered(one, names):
            print(s, flush=True)
