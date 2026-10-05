# op_dagger pilot: results so far (2026-10-05 / 06)

Status: final for the pilot (2026-10-05). dg1 / st1 / dg2 trained (400 steps, single seed), evaluated on the held-out clips, dg1 run through the
HUGSIM guard subset. No further DAgger rounds: the skill is to be merged into one joint training later; the `dg_*` engine stays ready to run on top of the merged model.
Pre-registration: [../plans/2026-10-06-dagger-prereg.md](../plans/2026-10-06-dagger-prereg.md). Tables: `pilot/levels.md`, `pilot/paired.md`.

## 1. Engine checks (before any training)

| Check | Result |
|---|---|
| Logged heading from the 5 s future vs exact sceneflow poses (max over the 2 s window, per frame) | moving: median 0.20 deg, p90 0.87 (n 220); launch 0.17 / 0.50 (n 19); position < 1 cm |
| Replay (zero offsets) through the engine's incremental trunk cache vs the trainer's batch path (`op_adapt_h.trunks` on 10 images) | max abs dphi1 1e-5 deg, dkappa 4e-8 (sf 42 states, heldout 30): identical |
| Port (5 Hz, 9 pairs, stateless) vs decision 123's ONNX/TRT stream (20 Hz, frames held) on the same sf frames, phi1 | r 0.999999, median abs 0.003 deg (n 154) |
| Decision 123's step gain, its own construction (t0 frame re-projected, yaw +-2 deg, 14 anchors) | step 1 **7.8 [4.3, 15.3]** deg/deg (d123: 7.4 [3.2, 15.3]); step 2 -1.1, step 3 -0.36, step 10 -0.13 (d123 -1.1 / -0.34 / -0.12) |
| Same with the logged frame at each step re-projected (this engine's source) | step 1 8.0 [3.6, 15.0]; launch anchors 15.3 (anchor source 16.5, n 6) |

The engine reproduces the open-loop outputs exactly and decision 123's step-1 gain. Files: `results/baseline/engine.json`.

## 2. Shipped on the held-out clips (WOD val, 120 clips: launch 40, moving 80; cluster bootstrap over sequences)

| Readout | launch | moving |
|---|---|---|
| G1 open-loop yaw gain, step 1 (deg/deg) | 16.5 [15.8, 17.2] | 6.3 [5.1, 7.5] |
| G2, step 2 | -2.1 | -0.83 |
| action-curvature gain step 1 (1e-3 / m per deg) | -3.97 [-4.65, -3.27] | +0.52 [+0.26, +0.80] |
| K10: heading offset at 2 s / 2 deg kick (closed loop) | 0.96 [0.89, 1.03] | 0.31 [0.23, 0.39] |
| S10: lateral offset at 2 s / at step 3 (0.5 m swerve) | - | 0.93 [0.87, 1.00] (n 56) |
| free rollout |dy| / |dpsi| at 2 s | 0.04 m / 2.4 deg | 0.10 m / 0.68 deg |
| rollouts inside the cap at 2 s (kick / free) | 0.81 / 0.85 | 0.95 / 0.99 |

Reading: at launch the plan swings 16 deg per degree of heading jump while the action head moves the other way (restoring, -4e-3/m per deg) and
cannot act (v < 0.3 m/s, latActive off), so a 2 deg kick is simply kept (K10 0.96). Moving, the action path takes back 70% of a heading kick
within 2 s but almost none of a 0.5 m lateral offset (S10 0.93), the decision-123 gap seen closed loop. Free rollouts drift 2.4 deg at launch:
the action is held at the t0 curvature until 0.3 m/s while the log turns (no model error, a controller fact).

Table: `results/baseline/levels.md`.

## 3. Training collection (shipped, train clips)

Shard 0 (160 clips, 800 rollouts): 6 852 of 8 000 states inside the cap (86%). Swerves at low speed need > 5 deg of heading and leave the cap
(the states are dropped); kicks of 0.5-4 deg stay inside on 72-93% of rollouts.

## 4. Pilot arms (held-out WOD val, 120 clips; cluster bootstrap over sequences)

Training: dg1 13 779 visited states from shipped rollouts, 400 steps, about 3.4 min on one card (2 it/s); st1 same states with scripted-drift history; dg2 from
shipped on shipped + dg1 states. No OOM, no change to `dg_train.py` was needed.

| Readout | group | shipped | dg1 | st1 | dg2 |
|---|---|---|---|---|---|
| G1 yaw gain, step 1 (deg/deg) | launch | 16.50 | 10.46 [9.59, 11.33] | 16.25 | 10.02 |
| G1 | moving | 6.28 | 2.62 [2.03, 3.26] | 5.84 | 2.47 |
| K10 (heading kick kept at 2 s) | launch | 0.96 | 0.76 | 0.84 | 0.75 |
| K10 | moving | 0.31 | -0.01 | 0.12 | -0.02 |
| S10 (lateral offset kept at 2 s) | moving | 0.933 | 0.243 [0.154, 0.329] | 0.756 | 0.240 |
| free |dy| at 2 s (m) | all | 0.081 | 0.089 | 0.091 | 0.087 |
| replay |dphi1| median / mean (deg) | all | 0 | 0.054 / 0.54 | 0.083 / 0.47 | 0.060 / 0.55 |
| action-curvature gain gk1 (1e-3/m per deg) | launch / moving | -3.97 / +0.52 | +3.24 / +4.46 | -21.5 / -1.65 | - |

![launch gain, kick and swerve recovery per arm](../figs/dagger-pilot.png)

Figure: left two panels, open-loop yaw gain per step (launch, moving): dg1 / dg2 drop the step-1 gain, st1 does not. Right two panels, closed-loop 2 deg kick and 0.5 m swerve: dg1 / dg2 recover fully by 2 s, shipped and st1 do not.

Pre-registered lines:

- **L1 (launch G1 <= 0.5 x shipped = 8.25, and dg1 - st1 CI upper < 0): not passed.** dg1 is 0.63 x shipped; dg1 - st1 = -5.79 [-6.23, -5.37] holds, the first clause fails. dg2 does not close it (10.02).
- **L2 (moving S10 - shipped <= -0.2, CI upper < 0, and dg1 - st1 CI upper < 0): passed.** -0.690 [-0.754, -0.627]; dg1 - st1 -0.513 [-0.574, -0.455]. K10 moving goes the same way (0.31 to -0.01). st1 (same states and labels, scripted history) recovers only 0.18 of it, so the closed-loop history is what adds the recovery.
- **L3 (free |dy| CI upper <= +0.1 m; replay |dphi1| median <= 1 deg): passed.** free |dy| dg1 - shipped +0.008 [-0.004, +0.021]; per-clip median replay drift 0.054 deg (launch 0.196, moving 0.037); but the mean is 0.54 (launch 1.29) and p90 0.72 (launch 1.9), so a tail of launch clips moves.
- Conclusion rule: L2 and L3 pass, so the closed-loop rollout adds something the labels alone (st1) do not. L1 fails.

Open-loop gains: the step-1 yaw gain halves, but the plan and action path changed more than these lines look at (next section).

### Flag: action-curvature gain sign flip (gk1)

Shipped at launch has gk1 -3.97 (restoring, but the controller is inactive below 0.3 m/s) and +0.52 moving. dg1 has +3.24 at launch and +4.46 moving (8x shipped moving), and its gk2 / gk3 go from +0.8 / +1.4 to +3.2 / +3.6. In other words dg1 steers into a heading offset much harder at every step, which is what produces the 2 s recovery (S10 0.24, K10 0), yet replay on logged states stays within 0.05 deg median. st1 goes the other way at launch (-21.5). The action target was relabelled by decision 130 (`rft.act_target`) before this training, so this is not the old target. A gain this large can over-correct in a loop that includes the controller lag, which the HUGSIM result below hints at.

## 5. HUGSIM guard (dg1, spec preset, decision 124's 11 scenes, one run each)

Guard line `hugsim`: PASS (spins 0 vs 0 for shipped). Reported: stuck (max_steps) 4 vs 6, completes 3 vs 3, HD-Score 0.532 vs 0.544, paired diff -0.012 [-0.200, +0.189]. Files: `experiments/op_guard/results/dg1/subset/`.

Spins: none (max heading error vs the route < 60 deg in every scene; 8440 easy peaks at 58.0 deg vs 53.4 shipped, close to the line for both).

Weaving: yes, a mild signal. Per-scene curvature jitter (mean |d kappa| per step) is above shipped in 8 of 11 scenes (for example 053: 2.6 vs 0.26 e-3; 3000_3200: 5.5 vs 0.59; 0013: 2.1 vs 0.92), and curvature sign flips are 14 vs 8 in 053 (82 steps) and 20 vs 32 in 8440 (fewer). Heading error peaks are higher in 040 (10.2 vs 5.6 deg), 053 (7.4 vs 2.8), 0418 (1.4 vs 0.5).

Outcome mix changed, not just the count: the stuck reduction (6 to 4) comes with more early collisions (4 vs 2). dg1 completes 0930 (shipped: fg collision) and ends 053 / 3000 / 034 in collisions after 82 / 26 / 28 steps where shipped sat stuck for 400, and loses 040 (shipped completes in 186 steps; dg1 stuck, e 10.2 deg). With 11 scenes and CI [-0.20, +0.19] on HD-Score this is within noise of shipped, but the early-collision pattern fits the gk1 over-correction and should be read before dg1-style data is merged.

## 6. Limits

Single seed, 400 steps (1/6 of it_dw3), 11 HUGSIM scenes, one run per scene. dg2 only descriptive (same as dg1 within CI everywhere). Engine validity range |dy| <= 1 m, |dpsi| <= 5 deg, 2 s. Launch reprojection is the weakest part of the engine (d123).

## 7. Reuse

`experiments/op_dagger/scripts/` is unchanged apart from two fixes this round (`dg_chain.py --gpus`, `dg_roll.py` terminates its pool on exit; orphaned workers had kept about 63 GB of the card). To run on a merged model: `dg_roll.py collect --model <ckpt dir tag or path to ckpt-final.pt> --set train`, then `dg_train.py --rolls ...`; the shipped rollouts under `$DATA_DIR/runs/op_dagger/roll/shipped` stay valid.
