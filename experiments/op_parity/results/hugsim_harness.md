# op_parity arms in the HUGSIM harness: serving path and equivalence tests (all pass)

Written 2026-10-05. Code: [../scripts/pp_hugsim.py](../scripts/pp_hugsim.py) (ONNX build, bias server, tests), [../scripts/pp_hugsim.sh](../scripts/pp_hugsim.sh)
(one GPU-pool job per mode), agent side [lib/parity_hugsim.py](../../../lib/parity_hugsim.py) (zs_agent.py opt `parity`). Raw test output:
[hugsim_harness/](hugsim_harness/) (`check-*.json`, `signs.json`). Closed-loop results: [hugsim_spin10.md](hugsim_spin10.md).

## Answer

| test | what is compared | result |
|---|---|---|
| 1 | shipped weights + `intent_bias` input fed zeros vs the stock `cinque` ONNX, both TensorRT, 2 WOD reference streams x 134 frames | plan xy max 0.039 m, mean 0.004 m; all output columns max 0.375, p99 0.0625 (fp16 engine level: the port vs stock ONNX is 0.082 / 0.007 m) |
| 2 | P2-init / P3-init bias server on the same inputs | bias exactly 0 on every frame (max \|b\| = 0); served outputs identical to test 1's |
| 3 | synthetic P3 checkpoint (3 plan-pathway tensors x 1.01, adapter output layer N(0, 0.01), bias \|max\| 0.60, rms 0.16) served (ONNX + bias server over the socket) vs the torch port `pp_train.PModel` weights + adapter, side tokens from pp_prep's encoder | bias max diff 4.9e-4 (one fp16 ulp at 0.6); plan xy max 0.109 m, mean 0.007 m on TensorRT (cuda-iob 0.039 / 0.004 m); with the bias dropped the same check reads 1.95 m / 0.45 m, so the bias path is what makes them agree |
| 4 | sign conventions of the ego features the agent computed in 3 057 HUGSIM steps vs the 3 000 NAVSIM training rows (lb_navtrain) | identical: oldest pose x < 0 when moving 100 % / 100 %; vx > 0 100 % / 100 %; turning rows with sign(y_past) = -sign(yaw_past) 100 % / 99.9 %; past yaw opposite an independent turn signal 100 % (HUGSIM heading change since 1.5 s) / 98.2 % (NAVSIM lateral acceleration); HUGSIM command 0 / 1 / 2 -> one-hot right / left / straight in 269 / 143 / 2 645 of 269 / 143 / 2 645 steps |

P0 through this path reproduces the stored Cinque run: HD 0.352 on the 10 spinners vs 0.350 (cinque-fixed, 2026-09-25) and 0.315
(cinque-fixed-base rerun, 2026-10-03); every scenario within 0.015 of the stored run (the 2026-10-03 rerun itself differs from it by
0.46 on 8440_8640-easy and 0.14 on 040-easy); 8 spins / 10 like the 2026-10-03 rerun (the stored run's two extra spins, 570_770 at 63 deg and 5980_6180 at 70 deg, sit
at the threshold and are not reproduced by either rerun). Details in hugsim_spin10.md.

## Serving path

- **Configuration = the wajepa_ref Cinque row.** The stored `cinque-fixed` run (runs/hugsim-exam/scored-op, 2026-09-25) is
  `zs_run.py --preset exam --agent cinque --controller fixed` with opts `{traffic}` only: tree `fixed` (official + PR #57 heading fix),
  TensorRT backend, dilate clock (one 0.25 s step = one 0.2 s context step, 4 model steps per simulator step), 5 s static warm-up (100
  model steps on the first frame), desire from the simulator command, forward_only + straight_stop. Every arm runs exactly this, plus opt
  `parity`. The brief mentioned the `spec` preset (opctrl tree, op-path lateral); the comparison run is `exam` + `fixed`, so that is
  what is used (prereg line 27 says "spec" for the front view preset; the front view packer is the same in both presets).
- **Model.** `pp_hugsim.py onnx`: cinque.ort.onnx with the arm's trained initializers and the `intent_bias` (1, 32, 512) fp16 input
  (op_l_onnx.py build --bias-input: added to the current frame's tokens and to the 8 past policy slots, not to the copy that re-enters the
  queue, so every step adds the current bias to all 9 context frames, as `ParityAdapter.apply` in training). P0 / P*-init = shipped
  weights + the bias input. The HUGSIM policy server serves it with `--onnx` and sets the request's `intent_bias` before its `reps` steps
  (zeros if a request carries none; never a stale bias of a pooled session).
- **Bias.** A resident bias server per arm (`pp_hugsim.py serve`, op-train env) loads the arm via `pp_train.load_pmodel` and per request
  returns `adapter(ego, side)` cast to fp16 (as `apply` casts it to the hidden tokens' dtype). Side tokens: the arm's own frozen Cinque
  vision encoder (port, fp16), pairs (key k-1, key k), k = 1..3, the encoder and pairing of pp_prep. P1 (no adapter) returns zeros.
  P0 is served with the untrained P3 adapter (bias exactly 0), so it exercises the whole input path (ego, side packing, encoder, socket).
- **Agent inputs** (lib/parity_hugsim.py), sources as WA-JEPA's client (third_party/wajepa close_loop/hugsim_planner.py):
  command `info["command"]` through their map [2, 0, 1] to the NAVSIM one-hot [left, straight, right] (HUGSIM 0 right / 1 left / 2
  straight; the existing desire map agrees); vx = `ego_velo`, ax = `accelerate`, vy = ay = 0 (HUGSIM has no lateral components; NAVSIM
  rows do have ay, so this is a train / test difference WA-JEPA shares); 4 poses relative SE(2) to now, the first state repeated before
  the episode start (their buffer clamps); CAM_FRONT_LEFT / CAM_FRONT_RIGHT / CAM_BACK as rendered (black CAM_BACK on KITTI-360 / Waymo
  fed as is).
- **Choices where the openpilot harness or the NAVSIM rows decide** (declared, not WA-JEPA's):
  - poses at the rear axle (NAVSIM AgentInput) = ego pose moved back by `hugsim_zs.rear_offset` (1.73 m), as this harness' Alpamayo
    history; WA-JEPA uses the box centre.
  - **model clock.** The harness already feeds speed x1.25 and stretches the plan by 1.25 (dilate). The ego inputs follow that clock:
    vx x1.25, ax x1.25^2, poses at model times -1.5 / -1.0 / -0.5 / 0 s (= -1.875 / -1.25 / -0.625 / 0 simulated s, linear between
    steps). On the simulator clock a model that holds the speed it is told would slow to 1 / 1.25 of it after the stretch. Side keys are
    the simulator steps nearest those times (7, 5, 2, 0 back; ties to the newer), so the side pairs are 0.5 / 0.75 / 0.5 s apart in
    simulated time, not exactly 0.5. Opt `parity.clock: "sim"` gives WA-JEPA's clock (stride 2, raw values); not run.
  - side views: each camera re-projected rotation-only to an openpilot road + wide pair along its mounting yaw (nuScenes 54.8 / -58.2 /
    178.9 deg; coverage road 1.0, wide 0.91-1.0), packed like the HUGSIM front (BT.601 limited-range YUV from RGB). The NAVSIM training
    frames come from JPEG YCbCr (full range): a colour-range difference the front view also has between the two boards.
- **Interface.** `interface.json` of a parity run carries the bias server (tag, arm, which inputs it reads) and two declared deviations
  added to jevdrive/openpilot/interface.py: `command.channel = desire-sim+onehot` and the new key `inputs.extra = [ego-status,
  pose-history(, side-cams)]` (real-car: odometry, IMU and cameras exist on a car). P1 records neither (its server reads nothing).

## Tests (pp_hugsim.sh equiv, one pool job, 9 min)

Streams: the two WOD reference streams of op_adapt (`runs/op_adapt/ref/frames_{0,1}.npz`, first 150 frames, stepped twice per frame,
read at the second step; frames 16+ compared, 134 per stream), desire zero, right-hand traffic. Per frame a synthetic ego vector
(speed 0-12 m/s, curving history, command cycled) and synthetic side key frames (the stream at f-6 / f-4 / f-2 / f, rolled per camera).
The port reference runs the bias on all 9 context frames and then zeroes invalid frames (pp_train's order); only frames with a full
context are compared, where the two agree on validity.

| run | stream | bias max diff | all columns max / p99 | plan xy max / mean (m) |
|---|--:|--:|--:|--:|
| test 1: zero bias vs stock ONNX (TRT) | 0 / 1 | - | 0.375 / 0.0625; 0.375 / 0.0625 | 0.035 / 0.0039; 0.039 / 0.0041 |
| shipped served vs P0 port (TRT), the fp16 floor | 0 / 1 | 0 | 0.5 / 0.0625; 0.75 / 0.0625 | 0.059 / 0.0061; 0.082 / 0.0067 |
| test 2: P2-init, P3-init served vs port (TRT) | 0 / 1 | 0 (\|b\| = 0) | as the row above | as the row above |
| test 3: synthetic P3, served vs port (TRT) | 0 / 1 | 4.9e-4; 4.9e-4 | 0.5 / 0.0625; 0.5625 / 0.094 | 0.048 / 0.0059; 0.109 / 0.0069 |
| test 3: synthetic P3, served vs port (cuda-iob) | 0 / 1 | 4.9e-4; 4.9e-4 | 0.25 / 0.031; 0.375 / 0.031 | 0.038 / 0.0044; 0.039 / 0.0043 |
| test 3 control: bias dropped vs port (TRT) | 0 / 1 | - | 3.57 / 2.54; 2.63 / 1.13 | 1.95 / 0.395; 1.75 / 0.453 |

Test 4 (`pp_hugsim.py signs`, [hugsim_harness/signs.json](hugsim_harness/signs.json)): HUGSIM = the features the agent logged in the
40 closed-loop runs of hugsim_spin10.md; the oldest pose's yaw over the simulator's heading change since 1.5 s has median 1.16 (1.25
expected for a steady turn: the oldest key is 1.875 simulated s back).

## Caveats

- Test 1 is not bit-exact: the two TensorRT engines (different graphs) pick different fp16 kernels. The difference is below the port vs
  ONNX floor; test 2's exact-zero bias is what makes P*-init the shipped model.
- Training zeroes the oldest of the 9 context slots (NAVSIM rows have 8 GIMM frames); HUGSIM has all 9 real frames, each with the bias.
  Same for every arm and for shipped Cinque, untested as a separate effect.
- HUGSIM feeds desire pulses from the route command (exam preset); the NAVSIM fine-tuning rows had desire zero. Same for every arm.
