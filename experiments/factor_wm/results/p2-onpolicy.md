# P2 + on-policy: the line stops at the seed-0 gate. On-policy training from P2 brings back S3's HUGSIM launch failure (launch stalls 1 -> 9 of 12 scenes)

Written 2026-10-06. Pre-registration: [../plans/2026-10-06-p2-onpolicy-prereg.md](../plans/2026-10-06-p2-onpolicy-prereg.md), written and committed
before any readout (43cd007f, label fix 2nd commit). Code: `scripts/fw_p2.py` (engine with plan execution, collection, trainer),
`scripts/fw_p2_chain.sh` (lane), `scripts/fw_p2report.py` (check / gate / report). Raw files: [p2-onpolicy/](p2-onpolicy/) (`gate.json`, `gate.md`,
`check.json`, `hugsim_extract.csv`); run data on the box under `runs/factor_wm/p2op/`.

## Gate verdict: STOP (pre-registered rules (b) and (c), both presets)

Seed 0 only: X = 3 DAgger rounds from P2-F-s0 (800 / 1 600 / 2 400 steps, 270 k labelled visited states), C = the off-policy control (2 400 steps,
WOD logged states), both with P2's own navtrain rows. P2 = the existing P2-F-s0 runs.

| read | P2 | C | X | X - P2 | X - C | rule | verdict |
|:--|--:|--:|--:|:--|:--|:--|:--|
| navtest EPDMS, protocol W (12 146 tokens, 136 logs) | 88.12 | 88.00 | 87.83 | -0.29 [-0.64, +0.08] | -0.17 [-0.46, +0.13] | (a) X - P2 >= -1.0 | pass |
| navhard two-stage combined, protocol G (225 groups, 76 logs) | 30.26 | 31.89 | 32.83 | +2.58 [-0.62, +5.97] | +0.94 [-1.42, +3.22] | (d) X - P2 >= -3.0 | pass |
| HUGSIM-12 exam: launch stalls / stuck | 1 / 0 | 3 / 0 | **9** / 1 | | | (b) X <= P2 + 2 | **fail** |
| HUGSIM-12 spec: launch stalls / stuck | 1 / 0 | 3 / 0 | **9** / 0 | | | (b) | **fail** |
| HUGSIM-12 exam mean HD | 0.406 | 0.345 | 0.280 | -0.126 | -0.065 | (c) X - P2 >= -0.10 | **fail** |
| HUGSIM-12 spec mean HD | 0.414 | 0.402 | 0.313 | -0.101 | -0.089 | (c) | **fail** (by 0.001) |

As registered, the lane stops: no seed 1, no full HUGSIM 64 / navhard W / navtest G readouts. So there are no full-run tables against WA-JEPA.
The seed-0 point gaps are navtest -3.9 (X 87.83 vs 91.71) and navhard -2.6 (32.83 vs 35.41, our harness). HUGSIM-12 is a rule-picked
subset, so it is not comparable with WA-JEPA's 0.451 on 64 scenes. HUGSIM CIs were not computed at gate scale (12 scenes); the stall counts
decide the gate.

The 12 scenes (`scripts/p2op_hug12.txt`, picked by rule before any X run): 8 at evenly spaced ranks of P2's HD and 4 of shipped's spec
launch-stall scenes. X keeps the peak speed over the first 40 steps at the initial 1.0 m/s in 9 of 12 scenes under both presets
(`hugsim_extract.csv`, `v_max40`). That includes scenes P2 completes (scene-124-extreme-01: P2 1.00, C 0.89, X 0.02) and scenes P2 drives
at 6-13 m/s (scene-095-extreme-01, scene-570_770-easy-00). C stalls in 3 of the same 12 (P2 1).

## Engine readouts (WOD val, plan engine, seed 0)

| model | g0b perturbed-arm failure | stall | heading | lane | launch-clip stall | g1s false go |
|:--|--:|--:|--:|--:|--:|--:|
| P2-F-s0 | 0.768 | 0.014 | 0.707 | 0.047 | 0.013 | 0.808 |
| C (PC-s0) | 0.757 | 0.006 | 0.677 | 0.075 | 0.013 | 0.675 |
| X (PX-s0) | **0.494** | 0.014 | **0.285** | 0.196 | 0.063 | **0.017** |

Inside its own engine X does what DAgger should do: heading failures 0.71 -> 0.28 and false starts on standing clips 0.81 -> 0.02. On
HUGSIM the car no longer launches. This is decision 143's pattern again: it fixes the engine and breaks HUGSIM transfer at launch. This
time it happens with P2's explicit ego inputs and with the plan (the quantity HUGSIM executes) as the trained and executed channel.

## What this says

- **Explicit ego inputs do not protect launch from on-policy training in this engine.** Taking H2 out (the engine now executes plan speed,
  not action[1]) did not prevent the failure. [I] The likely cause is a variant of H1. In the engine, "ego at 0 m/s with a static scene"
  occurs almost only in the stay clips, where P2 creeps forward (false go 0.81) and the on-policy labels pull it back. When the logs
  launch, the scene is always moving (the lead car pulls away, and the time-synchronous source keeps moving while the ego lags). X learns
  "standing + static history -> stay". HUGSIM's 5 s static warm-up is exactly that input. Not tested: a static-history probe on WOD launch
  clips (g1-diag's H1 readout) on PX-s0 would check it in about 10 min.
- **The open-loop side moves the expected way but is not significant at one seed.** navhard G X - C +0.94 [-1.42, +3.22] and X - P2 +2.58
  [-0.62, +5.97]. That is the direction of S3's stage-2 gain, but the CIs include 0. navtest X - C -0.17 (n.s.). C - P2 is -0.11 on
  navtest and +1.63 on navhard (point), so extra WOD data alone does not cost NAVSIM.
- Guards that were read: navtrain dev drift_off 0.049-0.061 m (line 0.10); X's navtrain dev ADE rises 0.56 -> 0.58 m.

## Disclosed deviations and findings

1. **The engine executes the plan** (pure pursuit + plan-speed tracking) instead of S3's action / OpLongitudinal path. Pre-registered (§2.1):
   P2's recipe pins the action heads to shipped, so an action-driven engine would roll out shipped's behaviour.
2. **Recovery-label bug in G1 (affects S3's labels).** `op_adapt_h.recover_target` takes the direction of segments > 1 mm, so the
   centimetre jitter of a standing log flips the normal and puts a 2 x dy jump into the target path. On the 1 069 train clips (states every
   0.8 s, dy 0.5 m) **9.9 % of states** have a lateral jump > 0.3 m relative to the log: launch 10 %, mid-speed 16 %, turn 12 %, stay 9 %,
   cruise 0.4 %. A 5 cm threshold (`fw_p2.recover`) leaves 0.9 %. This lane uses the fix for X and C alike. S3 (decision 143) was trained
   with the bug.
3. The gate read full navtest instead of a subset (pre-registered §5, scoring costs about 5 min per model).
4. Plan speed on WOD: P2 with inputs on plans 0.87 x the logged speed on cruise clips (shipped 0.89). [I] The WOD camera viewpoint shrinks
   speed as in decision 104. Not corrected here. It also explains the cruise "behind" events in the free arm (`check.json`).
5. Wall time: prep to gate about 1 h (11:35-12:30). The seed-0 collection was 3 rounds x 3 shards x about 5 min at partial GPU use (the
   pool placed the shards; the runner is CPU-warp bound with 16-18 workers). Total collection stayed under 1 h, so it was not tuned.

## Cost

About 3 GPU·h: prep 0.1, collection 0.8, training 0.5, HUGSIM-12 0.6, NAVSIM / engine reads 0.4. Disk: about 23 GB of rollout tokens under
`runs/factor_wm/p2op/roll/` (removable if the line stays closed).
