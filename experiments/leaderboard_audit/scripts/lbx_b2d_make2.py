"""Render the B2D loss-budget examples that come from the dump reruns (lbx_b2d_rerun.py): GIF + JPG key frame each, third-person | road | wide.
envs/openpilot python, CPU only: lbx_b2d_make2.py [name ...]; outputs to $DATA_DIR/runs/lbx_b2d/out/ (copied to figs/loss_budget_examples/).
Segments are scenario seconds t0:t1[:speedup]; key = scenario second of the JPG."""
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
A, DUMP, OUT = D / "runs/vlm_arb/arms", D / "runs/lbx_b2d/dump", D / "runs/lbx_b2d/out"
VIEW = Path(__file__).resolve().parent / "lbx_b2d_view.py"
# name: (arm unit, route, dump dir, label, key, segments)
JOBS = {
    "b2d_red_drive_24944_s0": ("lbx-drive-s0", "24944", "drive-24944-s0", "drive r24944 s0", 52.0, ["46:60"]),
    "b2d_blocked_drive_19324_s0": ("lbx-drive-s0", "19324", "drive-19324-s0", "drive r19324 s0", 60.0, ["36:42", "42:190:40", "190:196"]),
    "b2d_blocked_drive_9196_s0": ("lbx-drivec-s0", "9196", "drive-9196-s0c", "drive r9196 s0 (attempt c)", 30.0, ["27:33", "33:50:4", "50:100:12"]),
    "b2d_blocked_vmerge2_15612_s1": ("lbx-vmerge2-s1", "15612", "vmerge2-15612-s1", "vmerge2 r15612 s1", 22.0, ["8:22", "22:43:8"]),
    "b2d_byp_pbyp2ng_19832_s0": ("lbx-pbyp2ng-s0", "19832", "pbyp2ng-19832-s0", "pbyp2ng r19832 s0", 21.0, ["15:26"]),
    "b2d_byp_pbyp2ng_2520_s0": ("lbx-pbyp2ng-s0", "2520", "pbyp2ng-2520-s0", "pbyp2ng r2520 s0", 21.0, ["15:26"]),
    "b2d_byp_vmerge2_19832_s1": ("lbx-vmerge2c-s1", "19832", "vmerge2-19832-s1c", "vmerge2 r19832 s1 (attempt c)", 35.0, ["29:40"]),
    "b2d_coll_drive_27043_s0": ("lbx-drive-s0", "27043", "drive-27043-s0", "drive r27043 s0", 22.0, ["16:27"]),
    "b2d_coll_vmerge2_27043_s0": ("lbx-vmerge2-s0", "27043", "vmerge2-27043-s0", "vmerge2 r27043 s0", 18.0, ["11:22"]),
    "b2d_stop_vmerge2_17280_s2": ("lbx-vmerge2-s2", "17280", "vmerge2-17280-s2", "vmerge2 r17280 s2", 17.0, ["5:13:2", "13:20"]),
}
PY = sys.executable


def one(name):
    unit, route, dump, label, key, segs = JOBS[name]
    att = A / ("v2-%s-%s" % (unit, route)) / "attempts" / route / "1"
    out = []
    for ext, extra in (("gif", sum([["--seg", s] for s in segs], [])), ("jpg", ["--key", str(key)])):
        r = subprocess.run([PY, str(VIEW), str(att), str(DUMP / dump), str(OUT / (name + "." + ext)), "--label", label] + extra,
                           capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        out.append(r.stdout + r.stderr)
    return name + "\n" + "".join(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(JOBS)
    with Pool(min(len(names), 6)) as p:
        for s in p.imap_unordered(one, names):
            print(s, flush=True)
