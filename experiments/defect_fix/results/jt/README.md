# JT (NR1 prereg supplement 9 + amendments A-C): readout, 10-18 03:45 CST

One joint L18 stage from RB0-F-s0, 9 500 steps, arc W 0.0039, seed 0. Pool on1_JT = RB0 + RTB1 + YS + Y S rows (87 012; 4 logs / 18 073 rows held out)
+ NR1TV trainval imitation rows; per batch 45 navtrain + 13 yaw-rate + 32 S + 38 trainval. JT10: standing / launch 0.10 / 0.10 of the trainval draw; JT00: 0 / 0.
Gate and identity: `smoke_gate.json` (passes under amendment C), `identity_jt.json` (identical). Per-arm reads: `<tag>/arena.md`, `<tag>/held.txt`, `<tag>/diag.json`.

| read | RSG ref | JT10 | JT00 |
|---|--:|--:|--:|
| (a) arena held 2 228 @ leash3, paired vs RSG + leash3 | 0.934 | 0.905, -0.029 [-0.052, -0.012] | 0.904, -0.031 [-0.053, -0.013] |
| (b) arena held 2 228, no hold, paired vs RSG | 0.786 | 0.760, -0.026 [-0.051, -0.003] | 0.752, -0.034 [-0.058, -0.014] |
| (c) held 460 @18 zeros (crash guard RSG + 5 = 39) | 34 | 58 | 57 |
| (a) AF-front / offroad / corridor zeros | 12 / 44 / 33 | 51 / 65 / 52 | 52 / 65 / 54 |
| (a) "slow" partial scores | 175 | 45 | 36 |

Success line (a >= +0.01 with CI > 0, or b >= +0.05, with c <= RSG + 5): **not met by either arm; both are worse than RSG.**

Where it is lost (contributions under leash3): braking -0.013 (32 new zeros, mostly AF-front), static -0.015 (34 new zeros, corridor), turn -0.002;
launch -0.001, cruise / other +0.001 / +0.002. The models drive faster (slow partials 175 -> 45, rear-hits 23 -> 15, held progress_rel 0.90 vs 0.83)
and hit the car ahead when it brakes, roll out of the leash, and leave the road more.

Offline everything moved the "right" way, on the held-out pool logs too: agent hinge -33 %, road -42 %, 4 s arc bias -4.1 -> -0.9 m, slower-than-0.8x
share 0.37 -> 0.12; JT10's standing probe stay1m 0.001 -> 0.88 (JT00 0.02). None of it reached closed loop: JT10 and JT00 are inseparable on every
stratum (static -0.039 vs -0.040), so the open-loop waiting did not show in closed loop (as d335 / d344); dev ADE 0.528 -> 0.573.

Likely mechanism (not tested): the arc term is two-sided per row but its rows are not, since every row where the human slows to < 0.5 m/s is excluded,
so it only ever says "keep up with a moving human" and never "brake with him"; at 0.2 x the imitation gradient it outweighed the
braking evidence in the S rows (where the hinges only fire at contact). The speed gain is what the arc term was for; the cost is in braking and
in the leash. Not separated from the other changes (base RB0 vs V2, pool, schedule), one seed.
