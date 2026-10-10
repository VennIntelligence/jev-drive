# STOPLBL: what labels supervise "slow down / stop for what the picture shows", sized by what each board loses to it

Written 2026-10-10. Read-only inventory: no training, no GPU, no closed-loop run. Scripts [`scripts/stoplbl_label.py`](../scripts/stoplbl_label.py) (box, CPU),
[`scripts/stoplbl_part0.py`](../scripts/stoplbl_part0.py) (Mac); tables in [stoplbl/](stoplbl/).

## Part 0: points each board loses to the three situations

Situation labels are mine (recomputed): for every log frame the ego's *hindsight route* is the driven path of the next 30 s (+15 m straight), and a token is in a
situation when the map target is crossed by that route within 50 m: **red** = a traffic-light stop line whose lane connector is red in the log's
`traffic_lights` (the connector the route takes after the line); **stop sign** = a type-1 stop polygon; **yield** = a type-3 turn-stop or type-4 yield polygon;
**ped** = a marked crosswalk on the route with a pedestrian box within 3 m of it; **turn > 45** = the bench strata bin (logged 4 s heading change, 1 441 of 12 146
valid navtest tokens; the map-based route turn >= 45 deg within 50 m gives 3 457, a looser definition). Sets overlap (552 of the 2 191 yield tokens are turn > 45 tokens: turn-stop
polygons sit on turning connectors); the exclusive partition is in `stoplbl/part0_navtest_excl_turnfirst_SH30.csv`. **loss in set** = sum over the set of (100 - EPDMS) / N,
in board points; **excess** = fraction x (EPDMS of the rest - EPDMS of the set), the points the set costs *beyond* what the rest of the board loses per token,
cluster bootstrap by log (B 1000). Navtest = seed mean per token, all 12 146 tokens, bench `units.csv` (SH30 s0 s1, P2H10 s0 s1, P2H10S s0 s1; SH30 shown, others within
0.1 point, `part0_situations.csv`). Navhard = 225 groups, situation of the group's stage-1 tokens (stage-2 frames inherit it; approximation), `combined` score.

| board | situation | n units | loss in set (pts of board) | share of board loss | excess beyond rest [95% CI] | how attributed |
|---|---|--:|--:|--:|--:|---|
| navtest SH30 (EPDMS 89.54, loss 10.46) | traffic-light compliance (TLC) term | 30 tokens fail (0.25%) | 0.25 | 2.4% | n.a. | Shapley split of the per-token EPDMS to perfect, recomputed |
| | red light within 50 m | 1 277 | 0.65 | 6.2% | -0.49 [-0.70, -0.31] | map labels, recomputed; set EPDMS 93.8 against 89.1 for the rest |
| | any traffic-light line within 50 m | 2 575 | 1.41 | 13.5% | -1.02 [-1.38, -0.66] | same |
| | stop sign within 50 m | 821 | 0.89 | 8.5% | +0.20 [+0.06, +0.33] | same; EP 0.807 against 0.877, DAC 0.956 against 0.973 |
| | yield / turn-stop within 50 m | 2 108 | 2.01 | 19.3% | +0.25 [-0.04, +0.53] | same; overlaps turn by 552 tokens |
| | pedestrian crosswalk within 50 m | 1 464 | 1.07 | 10.2% | -0.22 [-0.41, -0.04] | same |
| | turn > 45 deg (logged) | 1 441 | 2.37 | 22.6% | +1.29 [+0.93, +1.61] | strata bin; DAC term carries 0.71 of the 1.29, EP 0.10, EC 0.32 |
| navtest P2H10 / P2H10S | turn > 45 excess | 1 441 | | 21.7% | +1.28 / +1.24 | same; stop sign +0.21 / +0.20, red -0.58 / -0.54 |
| navhard SH30 (33.67, loss 66.33) | TLC | stage 1: 0 tokens, stage 2: 62 of 5 462 (1.1%) | 0.75 of the stage-2 token mean | 1.4% of stage-2 token loss | n.a. | stage-2 token Shapley, recomputed |
| | red light | 5 groups | 1.27 | 1.9% | -0.20 [-0.78, +0.30] | not separable from zero |
| | stop sign / yield / ped | 27 / 43 / 26 groups | 8.0 / 13.5 / 7.4 | 12% / 20% / 11% | +0.04 / +0.96 / -0.32, all CIs contain 0 | |
| | turn > 45 | 56 groups | 19.17 | 28.9% | +3.54 [+1.66, +5.55] | P2H10 +3.65, P2H10S +3.61 |
| AlpaSim nuPlan 700 (P2H10-F s0 s1, mean 0.9483, loss 72.4 scene-seeds) | red | 108 scenes x 2 seeds | 3.25 | 4.5% | scene score 0.970 against 0.946 | navtest tokens of the scenes, recomputed; 2 zeros, 17 partial |
| | stop sign | 96 | 4.02 | 5.5% | | 3 zeros, 12 partial |
| | yield / turn-stop | 178 (non-exclusive) | 18.2 | 25.2% | 0.898 against 0.956 | 15 zeros, 23 partial; most are also turns |
| | ped crosswalk | 156 | 6.03 | 8.3% | | 2 zeros |
| | turn > 45 (logged) | 122 | 19.2 | 26.5% | 0.843 against 0.958 | 17 zeros, 19 partial; zeros = 18 of 49 by decision 227 |
| AlpaSim PAI (60 scenes, 1 seed, lost 38.46) | overrun of the logged stop, red light | 3 zeros | 3.0 | 7.8% | | quoted: decision 237 / pai.md (L5); scene ids 21626256, b0fa4732, 9e3fd12d |
| | overrun at a give-way T-junction | 1 | 1.0 | 2.6% | | quoted (7a824ffa, 12.8 against 2.8 m/s) |
| | stop sign passed at 15.2 against 6.4 m/s | 1 | 1.0 | 2.6% | | quoted (59e085d7, flagged L2 in pai.md) |
| | overrun, reason unknown | 1 | 1.0 | 2.6% | | quoted (1c7e2423) |
| | fast turn entry (flags L2), 1 also command unused | 3 | 3.0 | 7.8% | | quoted: 1d6e30bc, 3a48e906, 96da4f1a (the last two never attempt the turn) |
| | lead vehicle (L1, both lateral in fact) | 2 | 2.0 | 5.2% | | quoted |
| | slow progress (L4), 8 without + 4 behind the lead limit | 12 | 5.46 | 14.2% | | quoted; situation not determinable from stored results (rollout logs pruned) |
| HUGSIM 64 (SH30, 1 seed, lost 35.63 HD) | red light / stop line | | | | | HD has no light term (rc, nc, dac, ttc, comfort); scenario yaml carries no signal state: **not determinable**, loss 0 by construction unless a collision follows |
| | yield / stop sign | | | | | not determinable from stored results |
| | fast turn entry (L2) | 4 | 3.21 | 9.0% | | quoted: decision 237 / hugsim.md |
| | lead vehicle stopped (L1) | 9 | 7.81 | 21.9% | | quoted |
| WOD-E2E val | red light | | | ~0 | | quoted decision 169: red-light stops tie across arms (8.3-8.4 RFS), "red lights are not where WOD loses" |
| | stop-sign junction launch | 45 frames | | 35% of the stopped-frame loss | | quoted decision 169 (hand check of the auto labels: light 92.5%, stop sign 87.5%) |

### Reading

- **Red light / stop line costs almost nothing on navtest and navhard.** TLC is 0.25 points of 10.46 (2.4%) on navtest; 30 of 12 146 tokens fail, 13 of them in the 1 293 red-within-50-m tokens (1.0%).
  The red-light set scores *higher* than the rest (93.8 against 89.1) because the log driver stops there and so does the model (in 912 of 1 277 the log driver is stopped at the line; the model's
  EP, NC, DAC are no worse). The navhard stage-2 TLC (1.1% of tokens, 0.75 points of the stage-2 token mean) is a consequence of the synthetic pose perturbation, not of a stop prior.
  On the closed-loop boards the same situation is not small: PAI 4 to 5 of 60 scenes (3 certain red-light overruns plus an unknown one, 7.8 to 10.4% of the PAI loss).
  nuPlan 700: 108 scenes with a red light on the route lose 3.25 of 72.4 (4.5%), a below-average share.
- **Yield / stop sign** is a modest navtest cost (stop sign +0.20 points beyond the rest, EP and DAC; yield not separable) and a larger share on the closed-loop nuPlan board
  (178 yield scenes lose 25% of the loss, but they are mostly the same scenes as the turns).
- **Turns above 45 degrees are the one situation that costs on every EPDMS board**: +1.29 [0.93, 1.61] points beyond the rest on navtest (22.6% of the loss in 11.9% of the tokens),
  +3.5 [1.7, 5.6] on navhard (28.9% of the loss in 24.9% of the groups), 26.5% of the nuPlan 700 loss, 7.8 to 9.0% of the PAI and HUGSIM loss as fast entries.
  **Speed against path**: on navtest the turn excess is DAC 0.71 (55%), EC 0.32, EP 0.10 (7%): the cost is in the path, not in progress (recomputed). Decision 203 /
  op_parity pt-swap (quoted, not recomputed): putting the log timing on the plan's path removes 12.6% of the > 20 deg DAC failures, the log path with the plan's timing removes 94.6%, the lever is the path shape.
  The speed part is visible on the closed-loop boards (PAI 4 and HUGSIM 4 fast entries at 6.4 to 10.4 m/s against 2.6 to 2.8, decisions 153 / 159).
- What cannot be attributed from stored results: PAI progress loss (12 units, 5.46), HUGSIM stop / yield, any board's "slow" scenes by situation except nuPlan 700 (token labels recomputed by me).

Sub-score split, all situations and recipes: `stoplbl/part0_situations.csv`, `part0_subscore_loss_navtest.csv`, `part0_term_excess_navtest.csv`, `part0_tlc.csv`; nuPlan 700: `alpasim_nuplan_700_*.csv`, `alpasim_nuplan_zeros_labels.csv`.
