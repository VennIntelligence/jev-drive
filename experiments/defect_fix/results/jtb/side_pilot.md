# Side-buffer mechanism check (NR1 prereg supplement 11): JTN at 1 000 steps with and without `--lat-side`

Question: does JTN-L18-s0's side-vehicle buffer hinge (`--lat-side 0.5`, buffer 0.4 m) push the car toward the road edge and cause JTN's extra offroad zeros
(arena 49 vs RSG 43, PAI 19 vs 14)? Mechanism check, not a candidate: 1 seed, 1 000 steps, held 460 @18 only (6 chunks). Prereg: `plans/2026-10-17-nr1-prereg.md`
supplement 11 (commit 10b4c306, written before any training job).

Arms: both from RB0-F-s0, seed 0, pool `on1_JT`, draw `JT00`, lateral 0.5v2, history noise, `--on-arc 0 --on-arc-rows brake`, 1 000 steps, `--warmup 50`
(JTN's 500 / 9 500 scaled), `--lr-floor 0.1`. `JTNS1K-L18-s0` = control (`--lat-side 0.5`); `JTNN1K-L18-s0` = `--lat-side 0`.
Command check (`DRY=1 jt.sh train`, both with the same env): the two `rm_train` commands differ only in `--tag` and `--lat-side` (0.5 vs 0).
Code: `scripts/jt.sh side`, readout `scripts/side_pilot.py`, data `side_pilot_data.json`. Arm reads: `JTNS1K-L18-s0/held.txt`, `JTNN1K-L18-s0/held.txt`.

## Held 460 @18

Counts are over 460 scenes (zero = score 0). Lateral zeros = zero scores that are offroad, corridor or AF-side. AF = at-fault collision.
Paired = no-side minus control, 95 % CI clustered by log session (a count difference is the CI of the indicator mean x 460).

| | control (side 0.5) | no side (0) | paired no-side - control |
|---|--:|--:|--:|
| zeros | 42 | 36 | **-6** [-9.6, -1.4] |
| offroad zeros | 14 | 11 | **-3** [-6.0, 0.0] |
| lateral zeros | 27 | 23 | -4 [-6.6, 0.0] |
| AF-front (at-fault, front) | 7 | 5 | -2 [-4.5, 0.0] |
| AF-side (at-fault, lateral contact) | 0 | 0 | 0 |
| AF-side zeros | 0 | 0 | 0 |
| mean score | 0.8268 | 0.8359 | +0.009 [-0.000, +0.016] |
| progress_rel | 0.767 | 0.763 | -0.004 [-0.006, -0.002] |

Reference (not part of the rule): RSG-L18-s0 34 zeros, mean 0.8656; the 9 500-step JTN-L18-s0 34 zeros, 15 offroad (RSG 17), mean 0.8645.
Both 1 000-step arms are below RSG (control -0.039 [-0.065, -0.012], no-side -0.030 [-0.055, -0.005]).

## Offroad flips (scene-level)

Removed by dropping the side term (offroad zero in control only; 3): `2021.09.29.15.23.04_veh-28_00601_00802-aec02ec2aec85c06` (0 -> 0.77),
`2021.10.06.08.16.17_veh-52_01949_02501-20728d3e677b593d` (0 -> 0.47), `2021.10.06.08.16.17_veh-52_01949_02501-e072351fbbfd5765` (0 -> 1.00).
Added (offroad zero in no-side only): none. Control offroad zeros 14 = 11 shared + 3 removed.

## Was a car alongside at the offroad event?

Event = first step with per-step `offroad` > 0 (0.5 s steps); departure side = side of the ego's offset from the logged path; distance = centre to centre to the
nearest other actor on the opposite side with |lon| <= 10 m in the ego frame (`side_pilot_data.json`, `offroad_events`).

- 4 of the 14 control offroad zeros are offroad from t = 0 (ego starts offroad; unaffected by training): `veh-35_02943_03220`, `veh-52_00922_01296-da6d`,
  `veh-52_01430_01579`, `veh-52_01949_02501-9a37`. All four recur in the no-side arm.
- Of the 10 later events, an opposite-side car within 10 m existed for only 2 in the control: `veh-40_00743_01071` (6.7 m lateral, unchanged in the no-side arm, still offroad
  there) and `veh-28_00601_00802-aec0` (10 m at the event, 2.0 m lateral within the preceding second; this one is removed by dropping the term). Others: 17-22 m or none.
- The two other removed scenes (`veh-52_01949_02501-20728d3e`, `-e072351f`) have no opposite-side car at all within 10 m at the event (the control's event in
  20728d3e has a vehicle 7 m away, not on the opposite side). So a car alongside is not what the removed offroad events share, apart from one.

## Verdict under the prereg rule

Rule: supported if offroad zeros drop by >= 3 with lateral / at-fault side zeros not rising by more than 2; refuted if offroad zeros differ by <= 1; else inconclusive.
Observed: offroad zeros 14 -> 11 (-3, CI [-6.0, 0.0]); lateral zeros -4, AF-side 0 -> 0. **The rule's "supported" condition is met, at its threshold.**

Reading it honestly (not part of the rule):
- The effect is exactly at the line (-3) with a CI that touches 0, from 1 seed, and only 3 scenes flip. Only one of the three has an opposite-side car near the
  event, which is not the mechanism the hypothesis names (a car on the other side squeezes the ego toward the edge). The other two flip scenes are the same log
  (`veh-52_01949_02501`) as an event that stays offroad in both arms, i.e. closer to chaotic divergence than to a clean effect.
- Dropping the term also removes 6 zeros overall (CI [-9.6, -1.4]) and 2 AF-front, so at 1 000 steps the side term costs more than offroad alone; the paired mean is +0.009.
- It does not reproduce the problem it is meant to explain: the 9 500-step JTN has fewer offroad zeros than RSG on this same held 460 (15 vs 17). The extra offroad
  zeros of JTN are in the arena and PAI reads, which this check did not read.
- Reading: weak support that the side term adds offroad zeros in a short run; not evidence that it explains JTN's arena / PAI offroad gap. A decision needs more seeds
  and the 9 500-step schedule (JTN with `--lat-side 0`) read on the arena.
