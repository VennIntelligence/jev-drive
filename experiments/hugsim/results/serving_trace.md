# What the fine-tuned arm's policy slots receive in HUGSIM, and on which clock

2026-10-11. Diagnosis only: code reading, existing rollout logs, one traced 4-scenario run, one ONNX probe, one CPU reader check. Nothing in
the serving path was changed; no training. Pre-registration: [plans/2026-10-11-serving-trace-prereg.md](../plans/2026-10-11-serving-trace-prereg.md).
Opened by [served_plan_length.md](../../body1/results/served_plan_length.md), which left this path untraced. Code references are to commit
`1624c3ea` (line numbers of that commit). Tables: [serving_trace/](serving_trace/).

Arm: `P2H10-F-s0`, preset `spec`, the `jevdrive.bench` HUGSIM path: `experiments/hugsim/lib/zs_agent.py` (one process per scenario) ->
`experiments/hugsim/archive/hugsim_zs_server.py` running the queued ONNX `pp-P2H10-F-s0.onnx` (TensorRT), plus the bias server of
`experiments/op_parity/scripts/pp_hugsim.py`.

**Answers.**

1. **Frame source: mismatch present, as suspected, and wider than suspected.** Every policy slot is a pair of two consecutive rendered frames,
   put into the model as rendered; no keyframe, no warp. The checkpoint was trained on 2 real keyframes plus 6 ego-motion warps of them
   (decision 142, protocol W). Two further differences from the training rows were not in the suspicion: the queue holds **9** filled slots
   (training: 8, the oldest zeroed, the next built from a zero image), and the first 8 decisions run on repeats of the first frame.
   What the mismatch costs in HUGSIM is **not measured**.
2. **Clock: no bug.** A 1.25x factor exists and enters in one place, by design (one 0.25 s simulator step is fed as one 0.2 s context step);
   it is undone at every consumer we could find, and the logs agree with the code at each of them. What remains is the declared input shift:
   the model sees a world that runs 1.25x faster.

## 1. Frame source

### Path (code)

- Each simulator step the agent packs the three front renders into one openpilot frame pair `img2` (road + wide), a rotation-only
  nearest-neighbour gather, no ego-motion warp: `zs_agent.py:220`, `jevdrive/hugsim_zs.py:220`.
- It sends that one frame with `reps` = 4 model steps (`zs_agent.py:226-228`; 100 at step 0: the 5 s static warm-up of presets `spec` / `exam`)
  and the server steps the ONNX `reps` times on the same image (`hugsim_zs_server.py:163`).
- The temporal queues are states of the ONNX graph: `state_img_q` (2, 5, 6, 128, 256) and `state_feat_q` (128, 1, 16384), one entry per 20 Hz
  step (`jevdrive/openpilot/model.py:227`; `FRAME_SKIP = 4`, `model.py:21`). The bias of the parity adapter is added at the policy read, not
  stored in the queue (probe: zeroing the bias leaves the new feature row unchanged).
- Training rows: 8 slots at -1.4 .. 0 s, pair (slot - 0.2 s, slot), frames = 4 keys at -1.5 / -1.0 / -0.5 / 0 s and 6 lattice frames warped
  from the nearest key (`experiments/op_parity/scripts/pp_prep.py:42`, `:300`; `jevdrive/op_interp.py:265`; the serving twin is
  `experiments/alpasim/lib/sh30_core.py:44`, `:159-177`). The 9th (oldest) slot is zero and masked
  (`experiments/op_parity/scripts/pp_train.py:218-220`).

### Evidence

**Probe** ([serving_trace/probe.json](serving_trace/probe.json), `scripts/serving_trace_probe.py`, the served ONNX, one state entry perturbed
per step): the vision part reads the new frame and the frame **4 steps back**, nothing else in `state_img_q`; the policy reads the feature
rows **4, 8, .., 32 steps back** (8 past slots) plus the current one; both queues shift by one entry per step.

**Trace** (agent opt `trace`, 4 scenarios, 365 decisions, 1 844 model steps; `scripts/serving_trace_report.py`,
[slots_warm.csv](serving_trace/slots_warm.csv), [slots_early.csv](serving_trace/slots_early.csv)): the server's model step count equals the
sum of `reps` in 1 844 / 1 844 steps; the CRC32 of every image-queue entry equals the CRC32 of the frame the agent sent at that model step
in 9 220 / 9 220; every feature-queue row is the row appended at the expected step in 46 272 / 46 272.

Warm decisions (simulator step n >= 9; 329 decisions, identical in all of them), next to the training rows:

| slot | model offset | HUGSIM: frames of the pair (real offset to t0) | source | training row: frames of the pair |
|--:|--:|:--|:--|:--|
| 0 | 0 s | render n (0 s), render n-1 (-0.25 s) | rendered, as is | key 0 s; key 0 s warped to -0.2 s |
| 1 | -0.2 s | render n-1 (-0.25), n-2 (-0.50) | rendered | key 0 s warped to -0.2; key -0.5 s warped to -0.4 |
| 2 | -0.4 s | n-2 (-0.50), n-3 (-0.75) | rendered | key -0.5 warped to -0.4; key -0.5 warped to -0.6 |
| 3 | -0.6 s | n-3 (-0.75), n-4 (-1.00) | rendered | key -0.5 warped to -0.6; key -1.0 warped to -0.8 |
| 4 | -0.8 s | n-4 (-1.00), n-5 (-1.25) | rendered | key -1.0 warped to -0.8; key -1.0 |
| 5 | -1.0 s | n-5 (-1.25), n-6 (-1.50) | rendered | key -1.0; key -1.0 warped to -1.2 |
| 6 | -1.2 s | n-6 (-1.50), n-7 (-1.75) | rendered | key -1.0 warped to -1.2; key -1.5 warped to -1.4 |
| 7 | -1.4 s | n-7 (-1.75), n-8 (-2.00) | rendered | key -1.5 warped to -1.4; **zero image** |
| 8 | -1.6 s | n-8 (-2.00), n-9 (-2.25) | rendered | **slot zeroed and masked** |

Read it as: in HUGSIM 329 / 329 warm decisions have 9 / 9 slots made of two different renders 0.25 s apart; no repeat, no warp, no empty slot.
Each of the 4 model steps of a simulator step sees the same pair, so the three intermediate feature rows are copies and are never read.

Early decisions: at step n < 9 the slots older than the episode are the pair (render 0, render 0), i.e. a standing image, while the ego input
says 1.25 m/s (the car spawns at 1.0 m/s): 9 such slots at n = 0, 8 at n = 1, .., 1 at n = 8. No slot is ever empty or zero. Declared
(`interface.json` `history.warmup = static`, decisions 100, 118, 124); training has no such row.

### Verdict

Present. Three differences between what `P2H10` was trained on and what it gets in HUGSIM, all verified on the serving path:

| | training (W) | HUGSIM serving |
|:--|:--|:--|
| pixels of a slot pair | 2 keys + 6 warps of a key (static scene between keys, plane homography) | 2 renders, real parallax and actor motion, 3DGS render noise |
| filled slots | 8; oldest pair has a zero image; 9th slot zero | 9, all real pairs |
| pair spacing, real time | 0.2 s | 0.25 s (shown to the model as 0.2 s, section 2) |

Size of the effect in HUGSIM: not measured. Offline on navtest the same kind of mismatch (real 10 Hz lattice instead of warp) shortens this
checkpoint's plans by about 10 % (second-reader section). The HUGSIM logs do not show a short plan overall, but the reading is confounded:
over the 2 x 64 existing `spec` runs of `P2H10-F-s0` (warm decisions, v > 3 m/s, model clock) the plan arc to 3.9 s over fed speed x time is
1.03 pooled (median 1.05, n = 3 204; the cars spawn at 1 m/s and accelerate for most of an episode); on the 398 decisions whose speed
changed by less than 0.3 m/s over the last second it is **0.86** pooled (median 0.84; seed 1: 0.86, n = 373). The second number is a
selected subset of a closed loop (a car holds its speed exactly when its plan stops asking for more), not an isolation of the frame source.

## 2. Clock

### Path (code)

| where | what it does | reference |
|:--|:--|:--|
| simulator | one step = 0.25 s | HUGSIM `configs/sim/kinematic.yaml` `dt: 0.25`, `hug_sim.py` `self.timestamp += self.dt` |
| frames in | one render = 4 model steps at 20 Hz = 0.2 model s; dilation `dil` = 1.25 | `zs_agent.py:226-228` |
| speed to the policy server | `ego_velo * dil` | `zs_agent.py:230`, used at `hugsim_zs_server.py:180` |
| ego features to the bias server | `vx * 1.25`, `ax * 1.25^2`, 4 poses at model -1.5 / -1.0 / -0.5 / 0 s = simulator -1.875 / -1.25 / -0.625 / 0 s | `lib/parity_hugsim.py:65`, `:71-72`, `:88`; `zs_agent.py:157-158` |
| plan out | point k (simulator 0.5 k s) = model plan at 0.5 k / 1.25 s; the 3 s plan uses the model plan up to 2.4 s | `zs_agent.py:301`, `jevdrive/hugsim_zs.py:283-289` |
| lateral | action curvature = `action[0] / max(1, v_model)^2`, a geometric quantity, unchanged by the clock; `op_ctrl.hugsim_steer` runs with `dt = 0.25` and simulator speed; delay 0.25 simulator s = 0.2 model s | `jevdrive/openpilot/model.py:358`, `patches/hugsim/optional/op-ctrl.patch:30`, `jevdrive/openpilot/interface.py:246`, `:279` |
| longitudinal | upstream iLQR tracks the plan, which is already in simulator time | `interface.json` `lon.source = plan-ilqr` |

Not undone, and not undoable by rescaling: thresholds in model units (`MIN_SPEED` 1 m/s = 0.8 m/s simulated), the model's own lateral
action time (0.275 model s = 0.34 s simulated), and the speed-dependent priors of the model (it is told 12.5 m/s at 10 m/s).
Arms with side cameras (not `P2H10`) take their 2 Hz keys at 7 / 5 / 2 / 0 steps back, i.e. uneven 0.5 / 0.75 / 0.5 s (`parity_hugsim.py:54-56`).

### Evidence

Existing logs, `P2H10-F-s0_spec-rr1` / `-rr2`, 2 x 64 scenarios, `zs_steps.jsonl` ([clock.csv](serving_trace/clock.csv); seed 1 in
[clock_ref.csv](serving_trace/clock_ref.csv), the traced run in [clock_trace.csv](serving_trace/clock_trace.csv)); p5 / median / p95:

| quantity | expected if compensated | if not | measured | n |
|:--|--:|--:|:--|--:|
| time between decisions (s) | 0.25 | | 0.25 / 0.25 / 0.25 | 7 387 |
| model steps per decision (x 0.05 s = assumed dt) | 4 | | 4 / 4 / 4 | 7 387 |
| fed speed / simulator speed | 1.25 | 1.00 | 1.250 / 1.250 / 1.251 | 6 766 |
| simulator age of the oldest fed pose (s), straight driving | 1.875 | 1.50 | 1.874 / 1.875 / 1.875 | 2 178 |
| model time of plan point k / (0.5 k / 1.25) | 1.00 | 1.25 | 0.978 / 1.000 / 1.022 | 6 260 |
| first plan point distance / (0.5 s x speed), v > 3 m/s | ~1 | 1.25 or 0.80 | 0.72 / 1.03 / 1.19 | 3 204 |
| distance driven in the next 0.5 s / first plan point distance | ~1 | 1.25 or 0.80 | 0.91 / 1.01 / 1.12 | 3 026 |

The first five rows are exact up to logging precision; the last two are behaviour (acceleration, tracking) and sit at 1, not at 1.25 or 0.8.
Seed 1 and the traced run read the same (medians 1.250, 1.875, 0.999 / 0.995, 1.02 / 0.99, 1.02 / 1.05).

### Verdict

Absent as a bug; present as the declared dilation (docs/hugsim.md, `interface.json` `history.clock = dilate`). It enters at the frame
cadence and nowhere else, and the executed plan is the model plan read at t / 1.25. Its cost on the shipped model (offline, comma1M:
+25-38 % lateral error at 2 s) is the earlier measurement and was not repeated; nothing was measured for the fine-tuned arm, whose
training rows are all on the real clock.

## Instrumentation

- Agent opt `trace` (default off): `zs_agent.py` sends the request flag, `hugsim_zs_server.queue_trace` copies the two queue states to the host
  after each model step and returns CRC32s; the agent writes `<scenario>/zs_trace.jsonl`. Nothing is written back.
- Read-only check on the serving backend (TensorRT, probe): the same 48-step stream with and without `queue_trace` after every step gives
  bit-identical outputs (max abs difference 0.0; untraced against untraced 0.0).
- Closed loop: the traced run against the untraced repeats `rr1` / `rr2` on the same 4 scenarios: step-0 model plans identical in 4 / 4;
  every sent plan identical to `rr1` over the whole episode in 3 / 4 (34, 119, 21 steps); in the fourth (Waymo, 191 steps) the plans differ
  from step 1 by 8 mm (median 0.07 m over the episode), the size by which `rr1` and `rr2` differ on the KITTI-360 scenario (10 mm at step 1,
  median 0.07 m). Ends identical in 4 / 4, HD-Score 0.2061 / 0.2481 / 0.9976 / 0.7898 against 0.2061 / 0.2481 / 0.9976 / 0.7913.
- Run: `python -m jevdrive.bench run --model P2H10-F-s0 --bench hugsim --preset spec --opts '{"trace": true}' --jobs 1 --workers 4
  --scenarios scene-0013-medium-00,scene-570_770-easy-00,scene-113792265837-easy-00,scene-040-easy-00` (run dir
  `runs/bench/hugsim/P2H10-F-s0_spec-c85e22e1821d5`, 8 min on one card); probe and report under `runs/hugsim_serving_trace/`.

## Second reader: served_plan_length.md

Checked against the code, against a re-run of the note's own scratch scripts `a`, `e`, `f` (box `/tmp/spl_*.py`, outputs kept in
`runs/hugsim_serving_trace/spl/`), and against an independent re-implementation of its item 7 in the repo
(`scripts/serving_trace_reader_check.py`, [serving_trace/reader_check.json](serving_trace/reader_check.json); CPU fp32, the same 600 tokens).

**Holds (reproduced).**

- Item 2 (keyframes 1, 2, 3, 4 by decision, every scene) and item 3 (both table rows, the open-loop 0.913 / 0.968 / 0.995 / 0.995 / 1.004, the
  decelerating decision-0 profile, the by-speed row): every number reproduces to the printed digit.
- Item 7 table: warp 0.994 / 0.61 m, real 0.895 / 1.35 m, GIMM 0.885 / 1.52 m; real / warp by speed 0.96, 0.98, 0.90, 0.88, 0.89; warp
  forward against the bench predictions 0.066 m max. Same numbers from the independent script.
- The code references (`sh30_driver.drive`, `sh30_core.fill_history` / `lattice` / `lattice_gpu`, `pai_core.slots`, `prog_cl.py` comparing
  k = 0 with the token, sh30_smoke.md's cross-check) say what the note says they say.
- Scope line "HUGSIM: bench path, not this driver": correct, and section 1 above is the trace it asked for.

**Control the note did not run, now run: passes.** The real-frame tokens come from another data dir (`lb_hq_navtestX`) than the warp tokens
(`lb_navtest`), so the table could have mixed the frame source with the image pipeline. The 2 Hz `keys` protocol exists for both dirs and
uses no lattice frame: the plans from `lb_hq_navtestX@keys` and `lb_navtest@keys` on the 600 tokens are identical (mean pose distance
0.000 m, arc ratio 1.0000; ego tables identical). The 10 % is the lattice source.

**Does not hold, or is loose.**

1. *The PAI rollout numbers are not from the base arm.* Script `g` takes the first arm name on the box; `runs/alpasim/fix1/paibox/runs/` has
   no `base_*`, so it read `a_*`, FIX1's arm (a) = `JEV_VCONT=1.0` (speed-continuous serving). The note calls it "base-family arm `a`". It
   read `poses_model` (the model's plan before the switch), so the plan lengths 0.949 / 0.887 are model outputs, but in the closed-loop
   states of the switched arm. Not re-run here.
2. *Two values for one quantity.* Item 5 gives fed speed over logged speed at k = 3 as 0.973; "Not verified" gives 0.950 for the same
   checkpoint and decision. Script `e` takes the logged speed from pose differences, script `f` from the table velocity for k <= 3 and from
   pose differences after it. Over all 12 146 navtest tokens the table velocity is 2.8 % above the pose-difference speed (5.21 against
   5.06 m/s at the token; 3.0 % above 3 m/s), in history and future alike. So the series 0.973 .. 0.877 is on the pose-difference basis
   except its k = 0 point (table velocity), script `f`'s series (0.950 at k = 3, then 0.956) jumps at the switch, and the level of "the
   simulator ego is slower than the log" is uncertain by about 3 %. The downward trend holds on either basis.
3. Small: "seed 1 within 0.004" is 0.0041 at one entry; the decision-3 profile "0.987 .. 0.993" has a minimum of 0.983; "0.005 m mean" for
   `P2H10_nav` against the bench plan is the mean absolute difference of arc lengths, not a pose distance.

**Context the note does not give.** Decision 142 (point 4) found that pilot-scale P2 arms tested on real frames moved by only -0.47 to +0.78
EPDMS against their own protocol. The 10 % shorter plan here is the full-scale `P2H10-F-s0` and a plan-length reading, so the two are not
in contradiction, but nobody has scored `P2H10-F` on real-frame tokens; the note's table is the only reading.

**Not re-checked.** Item 1 beyond the 0.665 m/s at the token; item 4 (the 200-scene direct test; its per-scene table is on the box,
not recomputed); item 6; the scope numbers for `AP2H10-AB` reproduce in script `f`.

**One addition from this trace (estimate, not a measurement of HUGSIM).** A stream server fills the 9th slot and replaces the zero-image pair
(section 1). Stand-in on the same 600 warp rows: the token of slot -1.2 s copied into the zeroed slot, and also over the zero-image slot:

| input (600 navtest tokens, `P2H10-F-s0`) | arc / log | ADE vs log (m) | mean pose distance to the trained input (m) |
|:--|--:|--:|--:|
| warp, 8 slots (trained) | 0.994 | 0.61 | |
| warp, 9th slot filled | 1.013 | 0.72 | 0.36 |
| warp, 9th slot filled and zero-image slot replaced | 1.012 | 0.75 | 0.42 |
| real 10 Hz lattice, 8 slots | 0.895 | 1.35 | 1.20 |
| real 10 Hz lattice, 9th filled and zero-image slot replaced | 0.905 | 1.21 | |

The filled queue moves the plan by about a third of what the frame source does and does not shorten it. The copied token is not the true
older frame pair, so this bounds the sensitivity, not the HUGSIM effect.

## What a fix would touch (not done)

- **Frame source and slot count.** Serve the fine-tuned arms in HUGSIM through the stateless 8-slot protocol they were trained on, as the
  AlpaSim nuPlan driver does: keep the renders and ego poses of the last 1.5 s in the agent, take the keys from them, build the lattice with
  `sh30_core.lattice_gpu` and run `sh30_core.Core.plan` instead of stepping the queued ONNX. Touches `zs_agent.py` (a new branch beside
  `openpilot()`), the bench server set in `jevdrive/bench/hugsim.py` (a `Core` server instead of ONNX + bias server), and the action
  curvature the `spec` preset steers with (the `Core` path returns plans, so the action head would have to be exposed or the plan curvature of
  `spec_plan_smooth` used). Smaller
  alternatives: keep the queued ONNX but feed it warped lattice frames (does not remove the 9th slot), or train the HUGSIM checkpoint on
  render-pair slots.
- **Clock.** Nothing to fix in the current path. The stateless protocol above would remove the dilation as a side effect: keys every 0.5 s
  are every second simulator step, and the 0.2 s lattice is warped from them on the real clock, so speed, ego features and plan times
  would need no factor (`dil` = 1 in `zs_agent.py`, `parity_hugsim.py` clock `sim`, `OP_CTRL` delay 0.2).
- **Warm-up.** With the stateless protocol the first decisions are the cold start `sh30_core` already defines (back-warped key), which
  `served_plan_length.md` shows gives a decelerating plan at decision 0 unless the checkpoint was trained with it (`AP2H10-AB`).
- Before any of it: one offline A / B on logged HUGSIM renders (same decisions through both input paths) sizes the effect without a closed loop.

## Not verified

- The size of any of the three frame-side differences on HUGSIM plans or HD-Score.
- Other parity arms and presets (`SH30`, `P2H10S`, `spec_plan_smooth`): same code path, not traced.
- The cost of the dilation on the fine-tuned arm.
