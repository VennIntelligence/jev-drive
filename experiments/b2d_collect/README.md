# b2d_collect: PDM-Lite imitation data for a B2D P2

status: live
decisions: (pending) (inputs: 121, 127, 128, 133, 134, 137, 144, 147, 148)
index: B2D collector: 1000 hold-out routes, 61% junction turns, openpilot frames at 20 Hz
key: experiments/b2d_collect/plans/2026-10-06-b2d-collect-prereg.md, experiments/b2d_collect/scripts/b2dc_agent.py, experiments/b2d_collect/lib/b2dc_frames.py, experiments/b2d_collect/lib/b2dc_labels.py, experiments/b2d_collect/scripts/b2dc_routes.py, experiments/b2d_collect/scripts/b2dc_labels.py, experiments/b2d_collect/scripts/b2dc_check.py, experiments/b2d_collect/scripts/b2dc_lane.py, experiments/b2d_collect/scripts/b2dc_gif.py, tests/test_b2dc_frames.py

**Question.** Can we collect a Bench2Drive imitation dataset (PDM-Lite driving, openpilot road + wide cameras on the open-loop-aligned rig at
native 20 Hz, stored as the model frames Cinque consumes) that op_parity P2 (decision 144, + drivable-SDF hinge, decision 148) can train on
as is, weighted toward the junction turns that stay open (decisions 121-148), with no evaluation route in it?

**Design.** Route set (`b2dc_routes.py`, CPU, offline carla.Map): one scenario per route from the LB2.0 long routes (Town12 / Town13) and a
junction-manoeuvre generator over all twelve towns (the eight Bench2Drive-only types included), with a positional hold-out from
bench2drive220 and bench2drive_0.0.4_val (no shared junction, no trigger within 50 m, no 20 m of shared path); 1000 routes, all 44 B2D
types, 60.6% junction turns, split `b2d/b2dc-train@v1`. Collector (`b2dc_agent.py`): PDM-Lite as shipped plus the recording; frames packed
bit-identically to the closed-loop policy server and stored losslessly (H.264 crf 0); ego, route, expert controls and path, actors, lights.
Labels (`b2dc_labels.py`): op_parity's ego features / 8-pose future / NAVSIM command, action targets at the lateral delay, drivable SDF on
op_probe's grid. Checks (`b2dc_check.py`): replay through Cinque / P2 (frame-lag and turn-sign alignment), SDF and command consistency, GIFs.
Staged lane (`b2dc_lane.py`): 1 -> 10 -> all through the GPU pool. Plan: [plans/2026-10-06-b2d-collect-prereg.md](plans/2026-10-06-b2d-collect-prereg.md).

**Status.** Route set built; smoke (1 route) running.

**Next.** Smoke -> throughput and storage estimate -> 10 routes -> full collection when the pool has room.

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
