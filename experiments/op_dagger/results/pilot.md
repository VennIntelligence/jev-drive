# op_dagger pilot: results so far (2026-10-05 / 06)

Status: the engine checks and the shipped held-out baseline are done (CPU). The GPU stages (training dg1 / st1 / dg2, rollouts with the
fine-tuned models) wait for a card: on 2026-10-06 the p7 B2D plan-tracking lane has first claim on all three. `dg_chain.py` (tmux `dg-chain`)
waits for a 1-card lease and then runs them. Pre-registration: [../plans/2026-10-06-dagger-prereg.md](../plans/2026-10-06-dagger-prereg.md).

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

## 4. Pilot arms (pending GPU)

dg1 (DAgger states), st1 (same states, scripted drift history), dg2 (iteration 2, aggregated) -> section to be filled from
`$DATA_DIR/runs/op_dagger/report/pilot/` when the chain writes DONE.
