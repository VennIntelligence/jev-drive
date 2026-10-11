# How far the plan moves when HUGSIM decisions are fed through the training protocol instead of the served queue

2026-10-11. Offline measurement on recorded HUGSIM renders; nothing in the serving path was changed, no closed loop through the other
protocol, no training. Pre-registration: [plans/2026-10-11-frame-ab-prereg.md](../plans/2026-10-11-frame-ab-prereg.md). Follows
[serving_trace.md](serving_trace.md) (decision 249), which established the mismatch and left its size open. Tables:
[frame_ab/](frame_ab/). Arm `P2H10-F-s0`, preset `spec`.

**Open loop.** Every arm is evaluated on the states the served arm visited. The numbers say how much the plan of one decision changes
with the input protocol; they are not a prediction of HD-Score under the other protocol.

**Answer.** On 845 warm decisions of 16 scenarios, the training protocol (8-slot warp lattice on the real clock, `b`) moves the served
plan by 2.28 m at 3 s [1.55, 3.41] and makes it **12 % longer** (arc ratio 1.124 [1.073, 1.181]), the opposite direction of the PAI
reading (0.895). By size this is PAI-class under the registered rule. By attribution it is mostly not the frame source: with the clock
held fixed, raw pairs -> warp lattice moves the plan 1.52 m [1.12, 2.26] with an arc ratio of 0.971 [0.932, 1.012] (no length change
established), 9 -> 8 slots moves it 0.69 m [0.51, 1.02] without a length change, and the step that removes the 1.25x dilation moves it
2.14 m and carries the whole lengthening (1.143 [1.116, 1.164]). The lateral side is small everywhere (0.27 m at 3 s in total). The
pixel-source step cannot be isolated cleanly on a 0.25 s render grid (see Limits).

## Data

One recording run, `python -m jevdrive.bench run --model P2H10-F-s0 --bench hugsim --preset spec --opts '{"trace_frames": true}'
--jobs 1 --workers 6`, 16 scenarios (4 per dataset; by their outcome in this run: 7 complete, 3 foreground collisions, 1 background
collision, 2 off route, 2 spins, 1 launch stall; the same classes as in `rr1` / `rr2`), HD 0.535, 989 decisions: 845 warm (simulator
step n >= 9) and 144 warm-up (n <= 8). Run dir `runs/bench/hugsim/P2H10-F-s0_spec-c11011ea9047d`; per decision
`<scenario>/zs_frames/<step>.npz` holds the packed frame, the parity bias and ego features at full precision, the ego state, and the
plan the model returned. Agent opt `trace_frames` (default off) only writes these files.

## Arms

All on the same decisions; each differs from the previous one in one thing.

| arm | input | path |
|:--|:--|:--|
| `log` | as served | the plan the rollout logged (queued ONNX, TensorRT) |
| `off` | recorded frame, bias, desire, model steps | the same ONNX stepped offline (`scripts/frame_ab_onnx.py`) |
| `nod` | `off` with the desire input zero | same |
| `a_t` | 9 raw slots (render n-j-1, render n-j), served ego features, no desire | the checkpoint's torch port (`pp_train.PModel` and its vision encoder, as loaded by `sh30_core.Core`) |
| `s8` / `s8z` | `a_t` without the oldest slot / and the oldest pair starting from a zero image | same |
| `wd` | 8-slot warp lattice on the **model clock** (mapping M1) | `sh30_core.Core.plan`, called unchanged |
| `b` | 8-slot warp lattice on the **real clock** (mapping M2), nothing dilated | `sh30_core.Core.plan`, called unchanged, its own cold start |

Mapping (the registered design choice). M2, the headline: keys = renders n, n-2, n-4, n-6 (0 / -0.5 / -1.0 / -1.5 s of simulator time,
each at the pose it was rendered from), lattice warped on the real clock, speed and pose history undilated. This is the training
protocol and what a fix would serve, but it also removes the declared 1.25x dilation. M1, the closest alternative and the bridge: the
served clock is kept, keys = renders n-7, n-5, n-2, n (`lib/parity_hugsim.key_steps`), poses at model -1.5 / -1.0 / -0.5 / 0 s; its ego
features equal the served ones bit for bit on every warm decision (max difference 0.0), but two of its four keys are images taken
0.125 s after the pose they are labelled with.

Plans are compared on the simulator clock in HUGSIM plan axes at 0.5 .. 4 s (`hugsim_zs.openpilot_to_plan`'s formula: dilation 1.25 for
all arms but `b`). CIs: `jevdrive.stats`, cluster bootstrap over scenarios (B = 10 000, seed 0); ratios are ratios of sums.

## Gates

| gate | reading | result |
|:--|:--|:--|
| (i) offline replay = serving | `off` against the logged plan, points within 3.2 model s, tolerance 0.03 m: 0 / 989 decisions over, max difference 0.0 m (bit-identical, action curvature too). Control (bias zeroed): 989 / 989 over, median 2.1-4.3 m per scenario | pass |
| (ii) `Core.plan` = training protocol | 32 navtest tokens (real CAM_F0 JPEGs) against the bench predictions of the checkpoint: 32 / 32 within 0.05 m, max 0.046 m, median 0.009 m. Control (adapter bias off): 32 / 32 over, median 2.37 m | pass |
| (iii) HUGSIM pose convention of the warp | render n-2 warped along the recorded ego motion to the pose of step n-1, luma MAE against the real render n-1, 111 decisions at v > 3 m/s: 10.0 against 13.0 unwarped, 12.1 mirrored, 16.3 reversed; below unwarped in 93 % | pass |

Gate (iii) also says the plane warp removes less than a quarter of the frame-to-frame difference of a render pair: the rest is
parallax, actors and render noise, i.e. the pixel-source difference itself.

## Result: warm decisions (n = 845, 16 scenarios)

[frame_ab/chain.csv](frame_ab/chain.csv). d = displacement of the plan point; arc = path length over 0-3 s, second arm over first;
kappa = the action curvature `spec` steers with.

| step | what changes | d at 1 s (m) | d at 3 s (m) | median / p90 at 3 s | arc ratio 0-3 s | abs. lateral at 3 s (m) | abs. d kappa (1e-3 / m) |
|:--|:--|--:|:--|:--|:--|:--|:--|
| `log -> nod` | desire pulse removed | 0.02 | 0.22 [0.07, 0.44] | 0.00 / 0.68 | 1.003 [1.000, 1.007] | 0.14 [0.05, 0.29] | 1.08 [0.20, 2.42] |
| `nod -> a_t` | torch port (residual) | 0.02 | 0.07 [0.02, 0.16] | 0.01 / 0.33 | 1.004 [1.001, 1.008] | 0.01 [0.00, 0.01] | 0.03 [0.02, 0.05] |
| `a_t -> s8z` | **9 -> 8 slots** | 0.21 | 0.69 [0.51, 1.02] | 0.40 / 1.49 | 1.007 [0.989, 1.022] | 0.06 [0.04, 0.10] | 1.22 [0.93, 1.47] |
| `s8z -> wd` | **pixel source** (raw pairs -> warp lattice, M1) | 0.46 | 1.52 [1.12, 2.26] | 1.12 / 3.20 | 0.971 [0.932, 1.012] | 0.21 [0.12, 0.36] | 1.79 [1.08, 2.89] |
| `wd -> b` | clock and key grid (M1 -> M2) | 0.56 | 2.14 [1.72, 2.83] | 1.70 / 4.08 | 1.143 [1.116, 1.164] | 0.22 [0.13, 0.36] | 1.14 [0.87, 1.59] |
| `log -> wd` | served -> lattice on the served clock | 0.47 | 1.70 [1.28, 2.43] | 1.20 / 3.41 | 0.984 [0.936, 1.036] | 0.25 [0.15, 0.42] | 2.76 [1.80, 4.11] |
| `log -> b` | **served -> training protocol, total** | 0.59 | 2.28 [1.55, 3.41] | 1.67 / 5.22 | 1.124 [1.073, 1.181] | 0.27 [0.16, 0.45] | 3.30 [2.20, 5.01] |

- Of the 9 -> 8 step, dropping the 9th slot is 0.62 m and the zero-image pair 0.18 m. The navtest stand-in of serving_trace.md
  (0.36-0.42 m mean over 4 s, no shortening) had the right sign and order.
- Displacements at 4 s: 1.00 (slots), 2.16 (pixels), 3.17 (clock), 3.55 m (total); arc ratio 0-4 s of the total 1.140.
- The total is longitudinal: +1.87 m [1.05, 3.07] along the road at 3 s, signed lateral +0.04 m [-0.09, 0.14].
- Plan length against the fed speed (warm, v > 3 m/s, 515 decisions, [frame_ab/length_vs_speed.csv](frame_ab/length_vs_speed.csv)):
  arc over 3 s / (3 s x speed) = 1.029 served, 1.042 `s8z`, 1.009 `wd`, 1.127 `b`.
- All 16 scenarios have `b` longer than served on warm decisions (1.013 to 1.325); `wd` is shorter in 9 of 16
  ([frame_ab/per_scenario.csv](frame_ab/per_scenario.csv)).

Against the registered thresholds: `log -> b` is outside [0.95, 1.05] with a CI excluding 1 and above 1.0 m at 3 s, so
"of the PAI size", with the sign reversed (longer, not shorter). The floor is 0.07 m (port residual; replay noise is zero).

## Three-way attribution (plus two terms decision 249 did not list)

| difference | warm decisions | warm-up decisions (n <= 8) |
|:--|:--|:--|
| pixel source (raw render pair vs warp of a key), clock fixed | 1.52 m at 3 s, arc 0.971 [0.932, 1.012] | 1.19 m [1.01, 1.37], arc 0.959 [0.931, 0.991] |
| 9 vs 8 slots | 0.69 m, arc 1.007 | 0.32 m [0.26, 0.37], arc 1.007 |
| static warm-up | not present | see below: one decision (step 0) carries it |
| clock and key grid (removing the dilation; not in the list, needed to reach `b`) | 2.14 m, arc 1.143 | 2.22 m [1.93, 2.49], arc 1.262 [1.241, 1.283] |
| desire pulse (served: `desire-sim+onehot`; training rows: desire zero; not in the list) | 0.22 m, 163 / 989 decisions above 0.03 m, up to 6.2 m, in the scenarios with turn commands | 0.02 m |

The steps are sequential and order dependent (the 9th slot was only removed on raw frames, the lattice only built with 8 slots).

**Warm-up** ([frame_ab/warmup_profile.csv](frame_ab/warmup_profile.csv), 16 decisions per step). At step 0 the served queue holds the
first render 100 times and the car is told 1.25 m/s: the served plan is 4.2 times shorter than the cold-start plan of `b` (one key,
back-warped slots) and 3.4 times shorter than `wd`'s; displacement 4.28 m at 3 s. From step 1 on the two protocols are within the
warm-decision range: 0.6-1.1 m at steps 1-4 (arc ratio `b` / served 1.38, 1.09, 1.07, 1.11), 2.5-2.9 m at steps 5-8 (1.17-1.23), then
2.2-2.7 m. Outside step 0 the warm-up decisions show no excess over warm ones; the static slots make the 9 -> 8 step smaller there
(0.32 against 0.69 m), since the dropped slot is a copy.

## Where in a rollout (served -> `b`, warm unless stated; [frame_ab/where.csv](frame_ab/where.csv))

| stratum | n (scenarios) | d at 3 s (m) | arc ratio 0-3 s | abs. lateral at 3 s (m) | abs. d kappa (1e-3 / m) |
|:--|--:|:--|:--|--:|--:|
| launch, v < 2 m/s | 272 (8) | 1.58 [1.18, 1.83] | **1.435 [1.308, 1.565]** | 0.10 | 5 |
| 2-5 m/s | 229 (15) | 1.70 [1.08, 3.25] | 1.116 [1.058, 1.257] | 0.29 | 4 |
| > 5 m/s | 344 (14) | 3.22 [1.92, 4.35] | 1.094 [1.030, 1.153] | 0.39 | 2 |
| turning, v > 2 m/s | 75 (7) | 2.79 [1.57, 4.11] | 1.123 [1.059, 1.255] | 0.60 | 6 |
| straight, v > 5 m/s | 306 (12) | 3.18 [1.78, 4.47] | 1.089 [1.021, 1.154] | 0.38 | 2 |
| launch-stall scenario (1) | 15 | 0.74 | 1.313 | 0.01 | 9 |
| spin scenarios (2) | 100 | 3.66 | 1.230 | 0.34 | 6 |
| off-route scenarios (2) | 55 | 4.88 | 1.244 | 0.93 | 8 |

- In relative length the change concentrates at **launch**: below 2 m/s the training protocol plans 44 % further, and 4.2 times
  further at step 0. In metres it grows with speed. Laterally it is largest in turns and in the off-route scenarios, still under 1 m.
- The same strata for the pixel-source step alone (`s8z -> wd`): 0.62 m at launch (arc 1.018 [0.898, 1.183]), 1.34 m at 2-5 m/s,
  2.34 m above 5 m/s (arc 0.970 [0.929, 1.007]); straight above 5 m/s 2.38 m, arc 0.962 [0.915, 1.006]. So the launch lengthening
  comes with the clock and the cold start, not with the frame source; in the launch-stall scenario `wd` is 0.87 of served where `b`
  is 1.31.
- By dataset, total: KITTI-360 4.37 m (1.200), nuScenes 2.48 m (1.063), PandaSet 1.86 m (1.147), Waymo 1.14 m (1.067).
- Launch stall (1 scenario), spins (2), off route (2), background collision (1) are single cases; no CI claim.

## Limits

- Open loop on the served arm's states (top of the note).
- **The pixel-source step is not clean.** Raw pairs exist only 0.25 s apart, lattice pairs are 0.2 s apart, so pixel source and clock
  cannot be varied separately on recorded renders. `wd` holds the clock but labels two keys 0.125 s off their pose, an error that
  grows with speed, as the step's displacement does (0.62 -> 2.34 m). The 1.52 m is the frame source plus that artefact; the split
  between them was not measured. `wd -> b` in turn contains the key-grid change besides the dilation.
- `b` removes the declared dilation. Its +12 % is the plan of a model that is told the true speed and shown frames on the true clock;
  whether that helps or hurts in closed loop is not measured here.
- Camera height for the road-plane warp is the approximate per-dataset value (`interface.HUGSIM_HEIGHT`), not a calibrated one.
- One checkpoint (seed 0), one preset, one recording run, 16 of 64 scenarios chosen by outcome, not at random.
- The cold start of `b` at steps 0-5 is `sh30_core`'s back-warp rule, which this checkpoint was not trained with
  ([served_plan_length.md](../../body1/results/served_plan_length.md): a decelerating plan at decision 0 on nuPlan).
- Not done: an arm with the real-clock lattice and dilated ego features (would split the clock term), a 9-slot lattice, seed 1.

## Reproduce

Box, repo root; outputs in `$DATA_DIR/runs/hugsim_frame_ab/out/` (`onnx/`, `torch/`, `report/`, `gate_*.json`), pool jobs
`hugsim-frameab-{onnx,torch,report}` (6 GB / 4 cores 1 min, 8 GB / 12 cores 4 min, CPU).

    RD=$DATA_DIR/runs/bench/hugsim/P2H10-F-s0_spec-c11011ea9047d; O=$DATA_DIR/runs/hugsim_frame_ab/out
    $DATA_DIR/envs/openpilot/bin/python experiments/hugsim/scripts/frame_ab_onnx.py --run-dir $RD --out $O
    $DATA_DIR/envs/op-train/bin/python experiments/hugsim/scripts/frame_ab_torch.py --run-dir $RD --out $O
    .venv/bin/python experiments/hugsim/scripts/frame_ab_report.py --run-dir $RD --out $O
