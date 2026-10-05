# Desire-off control on the B2D 25 junction turns (decision 128, suspect b)

Question: in closed loop the route turn desire is still fed (training fed desire 0) and the no-command rc-ctl steers to the commanded side on 9 / 11 choice turns. Does the side information come from the desire pulse rather than the route adapter?

Setup: the guard `b2d_turns` unit (20 val routes, 25 turns = 13 choice + 12 forced, zones off, open-loop-aligned camera, tm seed 2) for rc-bear-s0, rc-ctl-s0 and shipped, with the agent's route desire forced to none every step (`DESIRE=false`; plans.jsonl carries desire 0 throughout; adapter input of rc-bear unchanged, so rc-bear-off = adapter only, rc-ctl-off = no side input at all). Lane file `scripts/desire_off_lane.py`, readouts `scripts/desire_off_report.py` (rft_split.split_one + junction_rig122 score_attempt, same definitions as report.md / split.md). Reference "on" = the existing desire-on units (shipped: rig122 olnz cache). Runs: `$DATA_DIR/runs/op_route_ft/desire_off/<arm>`, per-turn data `desire_off.json`.

Missing cell: shipped desire-off route 26365 (turn 0, forced curve) failed in the lane (job rc 0, no done record) and the cards were taken by other lanes before a rerun; shipped-off therefore has 24 turns and 19 routes (26365 was never entered in any of the other five conditions). Paired rows for shipped use the 24 common turns. One seed, closed loop is not bitwise deterministic: a single turn flips between reruns (decision 127), so differences of 1-2 turns are noise.

Note on "steered": the steering criterion here is the peak desired curvature towards the commanded side >= 0.5 / R_min and above the opposite side's peak, over the whole span of an entered turn (the 3 / 11 for shipped-on versus 4 / 11 in split.md is the extra "above the opposite side" clause).

## Reading (numbers below)

1. Desire off does not collapse any arm. rc-ctl, which has neither adapter nor (now) desire, still steers towards the commanded side on 9 / 11 entered choice turns (on: 9 / 11) and 9 / 10 forced; peak steer / needed curvature stays above 1 for it. So the side of the rc-ctl steering does not come from the desire pulse; suspect (b) is not supported. What does give the side to a model with no side input is not tested here (candidates: the lane / road geometry the car sees at the junction, i.e. the route lane the car sits in, and the fine-tune's targets; a no-information control would need randomized or flipped commands, which this run does not have). Chance level for a side bit is 50%, 9 / 11 is only moderately above it (p about 0.03 one-sided), and a "wrong side" count of 1-3 per cell means the wrong direction is rare.
2. Took-exit: desire off - on is +0.08 [0.00, +0.21] shipped (24 turns), -0.04 [-0.17, +0.09] rc-ctl, +0.08 [0.00, +0.20] rc-bear. Nothing is a decrease; the three arms sit at 3 / 24, 5 / 25, 6 / 25 versus 1 / 25, 6 / 25, 4 / 25 on. The only interval touching an effect is a slight increase, in the opposite direction of "desire is the side carrier".
3. rc-bear - rc-ctl under desire off: took +0.04 [-0.17, +0.26], choice -0.15 [-0.46, +0.15], steered (choice) +0.00 [-0.31, +0.31]; under desire on -0.08 [-0.28, +0.15]. The adapter adds nothing measurable in either condition, as in decision 128. Neither "rc-ctl collapses and rc-bear holds" nor "both collapse" occurred: both hold.
4. The gain over shipped comes with the fine-tune in both conditions: steered (all) rc-ctl - shipped +0.38 [+0.20, +0.56] off, +0.44 [+0.23, +0.64] on; rc-bear - shipped +0.38 [+0.15, +0.62] off. Shipped itself steers 3 / 11 on choice turns with desire on or off (desire does not make shipped turn either).
5. Timing and lane: turn-in stays 2.5-4.5 m of arc past the turn start (rc-bear off 4.5 m, rc-ctl off 2.9 m, rc-ctl on 3.5 m), i.e. the late-turn-in failure is unchanged by removing desire. Leaves-lane share is flat (17-18 of 20-22 entered). Window collisions: rc-ctl 7 -> 3, rc-bear 5 -> 11, shipped 5 -> 7; with 25 turns each and one seed these move by a few events either way and are not read as an effect (rc-bear 11 is the largest cell; no test). Route DS (zones off; 20 routes, shipped-off 19): shipped 26.3 -> 30.4, rc-ctl 34.0 -> 38.5, rc-bear 36.2 -> 30.5, not pre-registered.

## Per arm and condition (25 turns: 13 choice, 12 forced)

took = came within 5 m of the dense exit; steered = peak desired curvature towards the commanded side >= 0.5 / R_min and above the opposite side's peak (entered turns only, whole span); turn-in = arc of the car past the turn start at the first plan step with curvature >= 0.5 / R_min towards the commanded side (median over turns that steer; < 0 = before the turn start); leaves lane = peak cross-track > 1.75 m, share of entered turns; collisions = window collisions summed over the 25 turns.

| arm | desire | took choice (13) | took forced (12) | took all | entered | steered, choice (entered) | steered, forced | wrong side | turn-in m, median (n) | leaves lane | window collisions | route DS mean |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped | on | 0 / 13 | 1 / 12 | 1 / 25 | 20 | 3 / 11 | 4 / 9 | 2 | 3.1 (8) | 18 / 20 | 5 | 26.3 (20 routes) |
| shipped | off | 1 / 13 | 2 / 11 | 3 / 24 | 21 | 3 / 11 | 6 / 10 | 3 | 3.0 (10) | 17 / 21 | 7 | 30.4 (19 routes) |
| rc-ctl-s0 | on | 2 / 13 | 4 / 12 | 6 / 25 | 21 | 9 / 11 | 9 / 10 | 2 | 3.5 (19) | 17 / 21 | 7 | 34.0 (20 routes) |
| rc-ctl-s0 | off | 3 / 13 | 2 / 12 | 5 / 25 | 21 | 9 / 11 | 9 / 10 | 1 | 2.9 (20) | 18 / 21 | 3 | 38.5 (20 routes) |
| rc-bear-s0 | on | 1 / 13 | 3 / 12 | 4 / 25 | 21 | 7 / 12 | 8 / 9 | 0 | 2.5 (17) | 18 / 21 | 5 | 36.2 (20 routes) |
| rc-bear-s0 | off | 1 / 13 | 5 / 12 | 6 / 25 | 22 | 9 / 12 | 9 / 10 | 0 | 4.5 (22) | 18 / 22 | 11 | 30.5 (20 routes) |

## Paired differences (turns paired, cluster bootstrap over routes, 95%)

| contrast | took, all | took, choice | steered, choice | steered, all (0 if not entered) |
|---|---|---|---|---|
| shipped: desire off - on | +0.08 [+0.00, +0.21] (n 24) | +0.08 [+0.00, +0.23] (n 13) | +0.00 [-0.31, +0.31] (n 13) | +0.08 [-0.13, +0.27] (n 24) |
| rc-ctl-s0: desire off - on | -0.04 [-0.17, +0.09] (n 25) | +0.08 [+0.00, +0.23] (n 13) | +0.00 [-0.23, +0.23] (n 13) | +0.00 [-0.17, +0.16] (n 25) |
| rc-bear-s0: desire off - on | +0.08 [+0.00, +0.20] (n 25) | +0.00 [+0.00, +0.00] (n 13) | +0.15 [-0.15, +0.46] (n 13) | +0.12 [-0.04, +0.30] (n 25) |
| rc-bear - rc-ctl, desire off | +0.04 [-0.17, +0.26] (n 25) | -0.15 [-0.46, +0.15] (n 13) | +0.00 [-0.31, +0.31] (n 13) | +0.00 [-0.18, +0.21] (n 25) |
| rc-bear - rc-ctl, desire on | -0.08 [-0.28, +0.15] (n 25) | -0.08 [-0.31, +0.15] (n 13) | -0.15 [-0.46, +0.15] (n 13) | -0.12 [-0.29, +0.04] (n 25) |
| rc-ctl-s0 - shipped, desire off | +0.08 [+0.00, +0.22] (n 24) | +0.15 [+0.00, +0.38] (n 13) | +0.46 [+0.23, +0.69] (n 13) | +0.38 [+0.20, +0.56] (n 24) |
| rc-ctl-s0 - shipped, desire on | +0.20 [+0.07, +0.36] (n 25) | +0.15 [+0.00, +0.38] (n 13) | +0.46 [+0.08, +0.77] (n 13) | +0.44 [+0.23, +0.64] (n 25) |
| rc-bear-s0 - shipped, desire off | +0.12 [-0.07, +0.33] (n 24) | +0.00 [-0.23, +0.23] (n 13) | +0.46 [+0.23, +0.69] (n 13) | +0.38 [+0.15, +0.62] (n 24) |
| rc-bear-s0 - shipped, desire on | +0.12 [+0.00, +0.30] (n 25) | +0.08 [+0.00, +0.23] (n 13) | +0.31 [+0.00, +0.62] (n 13) | +0.32 [+0.11, +0.54] (n 25) |

## Per turn (took / peak s over R_min-need ratio / turn-in m)

| route | turn | forced | kind | side | R_min | shipped on | shipped off | rc-ctl-s0 on | rc-ctl-s0 off | rc-bear-s0 on | rc-bear-s0 off |
|---|--:|--:|---|---|--:|---|---|---|---|---|---|
| 10255 | 0 | 0 | junction | R | 6.4 | n 0.1 None | n 0.1 None | n 1.2 5.2 | n 1.0 9.8 | n 1.9 10.5 | n 0.9 3.8 |
| 15102 | 0 | 0 | junction | L | 11.5 | n 0.1 None | n 0.2 None | n 1.9 17.2 | n 1.8 17.2 | n 1.1 8.8 | n 1.2 8.5 |
| 24416 | 0 | 1 | curve | R | 36.6 | n 3.0 -2.5 | n 3.1 -2.5 | Y 3.8 -2.5 | Y 3.8 -2.5 | Y 5.2 -2.5 | Y 6.1 -2.5 |
| 24758 | 0 | 1 | curve | L | 33.3 | n 0.8 31.0 | n 0.1 None | n 0.4 None | n 1.5 30.2 | Y 3.1 32.5 | Y 2.6 31.2 |
| 24758 | 1 | 0 | curve | R | 31.2 | not entered | not entered | not entered | not entered | Y 1.1 3.0 | Y 1.4 6.2 |
| 24944 | 0 | 1 | curve | L | 10.7 | n 0.2 None | n 0.2 None | n 0.5 2.0 | n 0.6 1.5 | n 1.5 1.5 | n 1.8 1.2 |
| 24944 | 1 | 0 | junction | L | 8.7 | not entered | not entered | not entered | not entered | not entered | not entered |
| 25051 | 0 | 1 | junction | R | 6.2 | n 0.3 None | n 0.5 None | n 1.3 3.8 | n 1.3 -0.2 | n 1.8 2.5 | n 4.3 4.8 |
| 26153 | 0 | 1 | curve | R | 41.0 | n 11.1 -1.5 | Y 2.9 -6.5 | Y 3.7 -4.5 | Y 2.3 -3.8 | n 1.8 -4.2 | Y 2.2 -4.2 |
| 26153 | 1 | 1 | curve | L | 15.4 | not entered | n 0.8 -2.2 | n 2.9 2.2 | n 3.0 2.5 | not entered | n 7.3 -4.0 |
| 26365 | 0 | 1 | curve | R | 42.5 | not entered | - | not entered | not entered | not entered | not entered |
| 26723 | 0 | 1 | curve | R | 12.0 | n 0.4 None | n 0.5 3.8 | n 5.6 1.2 | n 1.7 1.2 | n 4.6 3.2 | n 3.5 3.2 |
| 26723 | 1 | 1 | curve | R | 11.3 | not entered | not entered | not entered | not entered | not entered | not entered |
| 26872 | 0 | 0 | junction | L | 12.6 | n 0.1 None | n 0.1 None | n 1.2 14.5 | n 0.5 3.2 | n 0.2 None | n 1.8 10.5 |
| 27297 | 0 | 0 | junction | R | 6.6 | n 0.7 1.2 | n 0.2 None | Y 6.2 -0.2 | Y 0.8 -2.0 | n 1.1 0.2 | n 1.3 2.8 |
| 27994 | 0 | 1 | junction | L | 7.0 | n 0.1 None | n 0.2 None | n 0.9 -3.0 | n 0.0 None | n 1.3 11.0 | Y 8.5 5.8 |
| 28008 | 0 | 1 | curve | L | 33.7 | Y 5.4 -2.5 | Y 3.0 -2.5 | Y 7.8 -2.5 | n 6.8 -2.5 | Y 2.2 -2.5 | Y 3.9 -2.5 |
| 28008 | 1 | 0 | junction | L | 8.1 | n 0.0 None | n 0.0 None | Y 2.1 6.2 | Y 1.8 5.5 | n 0.1 None | n 1.1 4.5 |
| 28147 | 0 | 0 | junction | L | 15.2 | n 0.6 23.2 | n 0.4 None | n 0.3 None | n 0.8 4.5 | n 0.4 None | n 9.3 20.5 |
| 28180 | 0 | 1 | junction | R | 4.7 | n 0.8 5.0 | n 3.8 2.5 | Y 2.4 -0.2 | n 0.6 -1.0 | n 2.6 -0.5 | n 3.8 2.0 |
| 334 | 0 | 0 | junction | L | 7.4 | n 0.0 None | n 1.8 4.2 | n 1.0 8.2 | n 7.3 -2.2 | n 9.1 0.8 | n 2.3 1.0 |
| 34183 | 0 | 0 | junction | L | 13.4 | n 0.3 None | n 0.5 12.5 | n 2.8 9.5 | n 11.5 3.8 | n 0.0 None | n 5.5 4.8 |
| 5423 | 0 | 0 | junction | R | 6.2 | n 1.1 5.5 | Y 5.5 3.5 | n 2.0 3.5 | Y 7.0 3.2 | n 1.0 -0.2 | n 0.8 5.8 |
| 6999 | 0 | 0 | junction | L | 6.1 | n 0.1 None | n 0.1 None | n 1.1 6.8 | n 1.1 7.0 | n 0.7 4.5 | n 1.0 5.2 |
| 9196 | 0 | 0 | junction | L | 8.1 | n 0.0 None | n 3.2 6.8 | n 1.0 15.0 | n 1.3 3.2 | n 1.1 14.5 | n 1.3 4.5 |
