# BODY1 loss arm 4.3: stopped at the pilot gate

2026-10-10. Amendment 4 of [the prereg](../plans/2026-10-10-body1-prereg.md) (in force before any training; implementation notes i to x and the
status paragraph there). Follows decision 227 ([zeros_diagnosis.md](zeros_diagnosis.md)). No full run, no G3 at full scale, no closed loop, no
PAI read, no ablation; variant D not opened. There is no new servable checkpoint: the driver stays P2H10-F.

**Verdict.** The pilot gate has two halves and the registered loss half is missed, so the arm ends by Amendment 4 item 5.
1. **Loss half, missed.** Agent hinge on own-plan positives (mean over rows with a non-zero hinge): 0.1216 at steps 201-400 -> 0.1100 at steps
   2 701-3 000, a fall of 9.5 % against the registered 30 %.
2. **Direction half, held.** On hold-log states the own-plan contact rates of the pilot are below the same-scale switch-off pilot and below
   P2H10-F-s0 (table below): agent -26 %, boundary -17 % relative, intervals excluding 0. At pilot scale neither reaches G3 (a)'s 30 %.
3. **Which term did what** (the per-term log of item 5). The road hinge on the new `bd4` rows is the only term whose depth on positives fell
   clearly (-43 %; yr1 -35 %, ot1 flat). The agent term's depth on positives barely moved (`bd4` -7 %, imitation rows -20 %); what fell is the
   share of rows that touch (`bd4` 0.13 -> 0.09). The registered number is conditional on rows that still touch and does not see rows that
   stopped touching. It was written that way and is not changed after the read.

Pilot: `P2H10B-P-s0` against the switch-off `P2H10-P-s0`, shards s2 + s3, 3 000 steps, seed 0, P2H10 recipe unchanged plus A + B + C
(hinge-only rows 4 / 4 / 5 of 128 from ot1 / yr1 / `bd4`, lambda 10, margin 0.3 m). Switch-off identity: 60 steps against the unedited
`pp_train.py`, 18 logged scalars and every trained weight equal bit for bit (`loss/ident.json`).

## Hold-log rates of the own plan (5 062 states of shards s2 + s3, 118 logs, paired by log)

| Rate | `P2H10B-P-s0` | switch-off pilot | difference [95 %] | relative | P2H10-F-s0 |
|---|--:|--:|---|--:|--:|
| agent contact | 0.0194 | 0.0261 | -0.0067 [-0.0101, -0.0035] | 26 % | 0.0237 |
| boundary, NAVSIM raster (the label with the line) | 0.0227 | 0.0275 | -0.0047 [-0.0074, -0.0022] | 17 % | 0.0259 |
| boundary, scorer-layer raster | 0.0223 | 0.0279 | -0.0055 [-0.0081, -0.0029] | 20 % | 0.0259 |

By state family (agent / boundary relative fall): `bd4` 27 % / 19 % and yr1 31 % / 19 % (intervals exclude 0); ot1 7 % / 11 % and on-log
25 % / 18 % (intervals include 0; on-log has about 8 and 11 positives). **> 45 deg bucket** (686 states): agent 0.0292 against 0.0379,
boundary 0.0452 against 0.0583, intervals touch 0. Full table: `loss/g3_pilot.csv`.

## Per-term loss (mean of the 100 steps ending at the step; `loss/pilot_terms_P2H10B-P-s0.csv`)

| Term | 300 | 1000 | 2000 | 3000 |
|---|--:|--:|--:|--:|
| imitation (switch-off pilot) | 1.209 (1.198) | 0.852 (0.846) | 0.757 (0.731) | 0.715 (0.704) |
| anchor (switch-off pilot) | 0.0220 (0.0200) | 0.0163 (0.0132) | 0.0080 (0.0065) | 0.0061 (0.0046) |
| on-log drivable hinge (switch-off pilot) | 0.0046 (0.0045) | 0.0037 (0.0043) | 0.0035 (0.0038) | 0.0035 (0.0035) |
| A, imitation rows: share positive / mean on positives | 0.0167 / 0.100 | 0.0125 / 0.090 | 0.0128 / 0.088 | 0.0121 / 0.081 |
| A on `bd4` | 0.132 / 0.171 | 0.064 / 0.165 | 0.088 / 0.167 | 0.092 / 0.158 |
| A on yr1 | 0.043 / 0.105 | 0.033 / 0.112 | 0.043 / 0.142 | 0.033 / 0.076 |
| A on ot1 | 0.025 / 0.142 | 0.015 / 0.096 | 0.005 / 0.026 | 0.013 / 0.134 |
| road hinge (B + C) on `bd4` | 0.340 / 0.182 | 0.254 / 0.113 | 0.288 / 0.113 | 0.284 / 0.115 |
| road hinge on yr1 | 0.240 / 0.060 | 0.223 / 0.060 | 0.208 / 0.056 | 0.225 / 0.058 |
| road hinge on ot1 | 0.233 / 0.031 | 0.235 / 0.038 | 0.155 / 0.046 | 0.203 / 0.062 |
| gate number (A on positives, pooled) | 0.120 | 0.110 | 0.113 | 0.113 |

Dev ADE at step 3 000: 0.591 m against 0.590 m; `drift_off` 0.047 against 0.046.

## The `bd4` rows

41 433 of 103 288 navtrain tokens (heading offset 2 to 8 deg, lateral within 0.5 m, no speed cut; all 10 773 tokens over 45 deg; 53 % of the
tokens under 1 m/s). Label rates of P2H10-F-s0's plan on 600 preview states: agent contact 8.8 %, boundary 14.3 % (NAVSIM raster), 14.8 %
(scorer-layer); under 1 m/s 5.5 % / 11.0 %.

![launch](../figs/loss/bd4_preview_launch_vlt1.png)

Launch states under 1 m/s. Look at the model-view frame against the BEV: the frame shifts by the heading offset with standing vehicles ahead
displaced consistently, and an edge-padding band of up to a tenth of the width appears on one side at 7 deg.

![class2](../figs/loss/bd4_preview_class2.png)

Turn states. Look at the plan's sweep against the two road labels: where they differ is the scorer-layer raster ending before the NAVSIM one.
The sheets for slow 1-3 m/s, class 1 and class 3 (`figs/loss/bd4_preview_{slow_1-3,class1,class3}.png`) were generated and not inspected.

## Costs, deviations, limits

Costs: about 0.4 card-h of the 8 (all through the pool; card 2 not used directly), about 35 min of box wall, 4.4 GB on the box (`bd4` for
shards s2 + s3 1.8 GB, preview cache 0.16 GB, scorer-layer raster 2.4 GB); free disk 364 GB. The full `bd4` cache would be 10.1 GB (draft: 18).

Deviations (each written into Amendment 4's notes before the number it affects):
1. The trainer is `scripts/bd4_train.py` with `lib/loss43.py`, a subclass of `pp_train.py`'s loss; `pp_train.py` is not edited. Two default-off
   hooks were added to other topics' scripts: `SELECT` in `op_parity/scripts/ot_rows.py`, `OPB_LAYERS` / `OPB_OUT` in `op_probe/scripts/opb_labels.py`.
2. Choices the amendment left open: 4 / 4 / 5 hinge-only rows per family; a hinge-only row weighs as one imitation row; agent side margin 0
   (decision 158's setting); A on imitation rows of train logs only; static history under 1 m/s and a 1.5 m history cap; the pilot is read
   against a same-scale switch-off run as well as P2H10-F-s0.
3. C is applied only to hinge-only rows that start on the scorer-layer road (note x: car-park launches start outside it); 1 101 of 30 141
   pilot rows keep the NAVSIM raster.
4. The scorer-layer list (ROADBLOCK, ROADBLOCK_CONNECTOR, INTERSECTION, LANE, LANE_CONNECTOR) is a reading of "road areas and lanes" on the
   nuPlan side; it was not checked against AlpaSim's map conversion.

Limits: one seed, two shards, 3 000 steps; the gate windows are steps 201-400 and 2 701-3 000 ("value at step 300" read as that window); with
equal per-row weight the hinge-only terms are small in the total (agent 0.001, road 0.003, times 10, against imitation 0.7), so the missed
line may partly reflect that weight, a choice of note iii and not of the draft; the user classes of the G3 table use the taxonomy file's `pc`
column, not checked for Amendment 1's class-3 change; the on-log subset has too few positives to read; no clips.

Not run, with the commands ready: the other ten `bd4` shards (`bd4_prep.py prep --data navtrain_full.s<k>of12`) and the full runs
(`bd4_train.py train --seed <i> --data navtrain_full.s{0..11}of12 --steps 10000 --ho ot1:4,yr1:4,bd4:5 --agent-lam 10 --tag P2H10B-F-s<i>`).
