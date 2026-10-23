# NTV0-NOS: is NTV0's PAI gain from the S rows or from the continuation? (round 9, 2026-10-23)

Pre-registration (Chinese, amendment 1 = new-scorer lines): [../../plans/2026-10-23-ntv0-nos.md](../../plans/2026-10-23-ntv0-nos.md). Lane
`scripts/nos.sh`, checks and report `scripts/nos.py`. Box run dir `$DATA_DIR/runs/round9/ntv0_nos/`. Decision 456.

**Question.** NTV0-L18-s0 reads 0.744 on PAI 281 (ACC + GAP, Harmonizer OFF, old scorer) against 0.650 for its base RB0-F-s0. Its L18 stage
adds S rows (on-policy pool on1_JT with agent / road hinges and the lateral terms), a low-lr 9 500-step continuation, and history-speed
noise. Does the gain need the S rows?

**Answer.** Yes, the S rows carry the gain. Without them the same continuation reads 0.624, which is 0.119 below NTV0-s0 and 0.022 below
RB0-F (CI spans 0). The continuation alone adds nothing on PAI. Both scorers land on the same registered line. So a clean NTV1 cannot
be fully offline in this form: it needs S-type rows (on-policy states with hinges).

## Arm

NTV0NS-L18-s0 = NTV0-L18-s0's stored meta.json argv. The token diff (`nos.sh dry`) is exactly:

```
--tag NTV0-L18-s0            -> --tag NTV0NS-L18-s0
--on-k 32                    -> --on-k 0          (the effective --on-k; RSG's --on-k 13 earlier in the argv is overridden)
--on-wfile runs/defect_fix/w/JT00S.npz            deleted (rm_train asserts it needs k_on > 0; it only set the S draw and JT's probes)
```

- **Batch.** Still 128: 115 navtrain + 13 yaw-rate + 0 pool, instead of 83 + 13 + 32.
- **Kept as in NTV0.** Pool, hinge and lateral setup are still built, but no pool row ever enters a batch, and `LossesOn` returns Losses43's
  loss unchanged when a batch has no pool row. Also unchanged: lr-scale 0.1, warm-up 500, lr floor 0.1, 9 500 steps, history-speed noise,
  route-memory drop, init RB0-F-s0, seed 0, compiled.
- **No trainer code changed.** The only code edit is one READS entry in `experiments/alpasim/scripts/scorer_v2.py`.

## Gates and checks

| check | result |
|:--|:--|
| DRY argv diff | only the three tokens above |
| smoke, 60 steps ([smoke.json](smoke.json)) | pass: DONE, all scalars finite, batch 115 / 13 / 0 (mixed), no S / pool loss term present |
| identity, informational: NTV0's own argv, 60 steps on HEAD vs the stored NTV0 run (code 1599227) ([identity.json](identity.json)) | draws identical (on_hash_60 equal); 31 / 86 loss scalars bit-identical, the rest differ by about 1e-4 relative at step 25 and up to about 1e-2 at step 50 |
| identity rerun: a second HEAD run vs the first ([identity_rerun.json](identity_rerun.json)) | 86 / 86 bit-identical |

So HEAD is deterministic run to run. Its small numeric difference from the stored NTV0 is code or build drift since 1599227, not
nondeterminism; the row draws are the same. That drift is a seed-noise-sized perturbation. For scale: NTV0 s1 - s0 is -0.006, the effect
here is -0.119.

## PAI 281, ACC + GAP (g10, FAST, NTV0's chunk lists), Harmonizer OFF

### Old scorer: the read's own, as registered

278 scenes are common to the three reads without a driver RPC error. Full tables: [pai281.md](pai281.md), [pai281_s1.md](pai281_s1.md),
[verdict.md](verdict.md).

| read | mean | zeros | offroad | drift / lane change (zero-other + side) | route-turn | stop overshoot | AF-front | rear-hit partials | slow partials | launched | navtrain dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0NS-L18-s0 | **0.624** | 93 | 21 | 13 + 2 | 20 | 23 | 14 | 24 | 6 | 32 / 37 | 0.5287 |
| NTV0-L18-s0 | 0.743 | 59 | 12 | 6 + 0 | 15 | 23 | 3 | 25 | 11 | 33 / 37 | 0.5362 |
| NTV0-L18-s1 (own common set) | 0.737 | 62 | 14 | 5 + 1 | 17 | 22 | 3 | 24 | 11 | 33 / 37 | 0.5354 |
| RB0-F-s0 | 0.646 | 88 | 19 | 14 + 4 | 21 | 22 | 8 | 22 | 8 | 32 / 37 | 0.5276 |

Paired comparisons (scene bootstrap):

| pair | mean difference (95 % CI) | zeros |
|:--|--:|--:|
| NTV0NS - NTV0-s0 | **-0.119 [-0.160, -0.081]** | +34 (35 new, 1 fixed) |
| NTV0NS - NTV0-s1 | -0.113 [-0.153, -0.075] | |
| NTV0NS - RB0-F-s0 | -0.022 [-0.051, +0.006] | +5 |
| NTV0-s0 - RB0-F-s0 | +0.097 [+0.060, +0.136] | |
| NTV0-s1 - NTV0-s0 | -0.006 [-0.026, +0.014] | |

Against NTV0-s0, the arm loses zeros in:
- AF-front collisions: +11 [+4, +18];
- offroad: +9 [+3, +16];
- drift / lane change (zero-other): +7 [+2, +13];
- route-turn: +5 [-2, +12].

It gains 5 fewer slow partials [-10, -1]. Stop overshoot, rear-hit and launch do not move. Against RB0-F, no class moves beyond its CI
except rear-hit (+2 [+0, +5]).

So the S rows (agent / road hinges and the lateral terms on on-policy states) are what fix collisions, offroad and drift. The continuation
alone leaves the model at its base. Navtrain dev ADE barely moves without the S rows (0.5276 -> 0.5287); NTV0's 0.536 is the imitation cost
of the S rows.

### New scorer (scorer_v2, decision 453), amendment 1

- NTV0NS 0.696 vs NTV0-s0 0.814 on 278 scenes: -0.119 [-0.159, -0.081]; zeros 73 / 39.
- vs RB0-F 0.715: -0.019 [-0.048, +0.010].
- vs NTV0-s1 0.748 (partial: its zero ASLs were mostly deleted): -0.051 [-0.101, -0.003].

## Reading line

| scorer | lines | NTV0NS | gap share | line |
|:--|:--|--:|--:|:--|
| old (registered) | >= 0.72 / <= 0.67 | 0.624 | -23 % | the S rows carry the gain |
| new (amendment 1) | the same 74 % / 21 % shares of the RB0-F -> NTV0-s0 gap: 0.789 / 0.736 | 0.696 | -19 % | the S rows carry the gain |

Evidence: strong for the direction (1 seed of the arm, but the paired CI sits far below both NTV0 seeds and the arm stays at RB0-F level).
Which part of S matters (the hinges, the lateral terms, or simply on-policy states) is not separated here.

## Cost

About 3.6 GPU job-hours, roughly 1.2 card-h at the pool's packing:
- training 21 min;
- smoke and two identity runs, 1.5 min each;
- 8 PAI chunk reads, 3.1 job-h at 4 per card.
