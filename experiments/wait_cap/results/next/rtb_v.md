# RTB1 variant round: V1 (lambda 0.5) and V2 (rolling-arrival rows only), prereg 补记 2 (exploratory, one seed)

Decision: [338](../../../../research/decisions/338.md). Follows [rtb_cont.md](rtb_cont.md) (decision 337: the RTB1 retain base buys the
closed-loop arrival stop but costs held zeros, at-fault and public rear contacts). Prereg: [2026-10-17-retain-anticopy-draft.md](../../plans/2026-10-17-retain-anticopy-draft.md)
§补记 2. Exploratory; no registered verdict; one seed each.

## Arms

All three bases are the RB0-F recipe (10 000 steps) plus the P0 longitudinal-only KD on arrival rows. Each is continued with the RSG-L18 recipe
(2 500 steps, old pool on1_L18RB0, seed 0). Only the retain term differs.

| arm | base | retain term | continuation |
|:--|:--|:--|:--|
| a | RTB1-F-s0 | lambda 1, groups AR:2 + W:1 (rolling-arrival and standing-wait rows) | RTB1O-L18-s0 |
| V1 | RTB05-F-s0 | lambda 0.5, same rows | RTB05O-L18-s0 |
| V2 | RTB1AR-F-s0 | lambda 1, `--kd-groups AR:1`: rolling-arrival rows only, the standing-wait (W) rows dropped | RTB1ARO-L18-s0 |

Hypothesis behind V2: the W rows teach an unconditional hold at standstill, which shows up as the start delay on v0 < 0.5 frames (pilot 4 s arc
-0.44 m) and as late relaunches in closed loop. Both bases passed the registered stop rule (arrival 4 s arc at k = 1 <= 2.0 m, dev ADE <= 0.5556):
V1 0.71 m / 0.5536, V2 0.89 m / 0.5519 ([rtb_v_gate_RTB05-F-s0.md](rtb_v_gate_RTB05-F-s0.md), [rtb_v_gate_RTB1AR-F-s0.md](rtb_v_gate_RTB1AR-F-s0.md)).

## Public nuPlan 1491 @10, paired against RSG-L18-s0 (stored read)

Sources: [rtb_wide_RTB05O-L18-s0.md](rtb_wide_RTB05O-L18-s0.md), [rtb_wide_RTB1ARO-L18-s0.md](rtb_wide_RTB1ARO-L18-s0.md), [rtb_v_sum.md](rtb_v_sum.md).

| arm | zeros | mean | vs RSG [session CI] | vs a [CI] | at-fault | offroad | front / lateral / rear |
|:--|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | 58 | 0.9397 | - | - | 8 | 31 | 6 / 2 / 18 |
| a | 63 | 0.9282 | -0.011 [-0.023, -0.001] | - | 7 | 36 | 5 / 2 / 24 |
| V1 | 62 | 0.9311 | -0.009 [-0.018, 0.001] | +0.003 [-0.003, 0.009] | 8 | 34 | 4 / 4 / 23 |
| **V2** | 65 | 0.9378 | **-0.002 [-0.011, 0.007]** | **+0.010 [0.001, 0.019]** | **5** | 36 | 3 / 2 / **17** |

V2 closes the public gap: the CI against RSG now contains 0, the rear contacts (17) fall to RSG's level (18) from a's 24, and at-fault (5) is
below RSG's 8. V1 barely moves from a (rear 23). V2's zeros (65) are the most of the three and offroad stays at a's 36 (RSG 31): what is left
is lateral. The V2-vs-a CI lower edge is 0.001, so the gain over a is borderline on its own; the rear-contact count (24 -> 17) is the sharper evidence.

## Held 460 @18 (one read per arm; ref RSG-L18-s0)

Sources: [rtb_cont_v1.md](rtb_cont_v1.md), [rtb_cont_v2.md](rtb_cont_v2.md), [rtb_v_sum.md](rtb_v_sum.md). 452 paired scenes, session-clustered CI.

| arm | zeros | paired vs RSG [CI] | paired vs a [CI] | progress_rel | at-fault | offroad | turns /24 | static scene | stop17 zeros | arrival @k=1 | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|--:|
| RSG-L18-s0 | 34 | - | - | 0.788 | 4 | 17 | 16 | 2.61 m, contact | 16 / 17 | 2.90 | 0.5284 |
| a | 38 | 0.000 [-0.019, 0.021] | - | - | 9 | 15 | 16 | 0.53 m, none | 15 / 17 | 1.34 | 0.5472 |
| V1 | 41 | -0.007 [-0.029, 0.015] | -0.007 [-0.021, 0.004] | 0.796 | 8 | 18 | 18 | 0.54 m, none | 16 / 17 | 1.20 | 0.5434 |
| **V2** | 39 | **+0.007 [-0.005, 0.025]** | +0.007 [-0.002, 0.021] | **0.817** | 7 | 16 | 16 | 0.86 m, none | 15 / 17 | 1.49 | 0.5445 |

The arrival stop is kept by both variants: the static scene stops without contact (V2 at 0.86 m, the human 0.50 m, RSG hits the lead) and
stop17 is 15 / 17 for V2. Of the seven held zeros that all four a-line arms share (53d3, de4a, 6d2b, 9d1d, 9a67, ddc1, 1f54), V1 keeps five
and V2 keeps four (9a67, ddc1, 9d1d, 1f54). By the class rule in [explore_f.md](explore_f.md), the two longitudinal ones (de4a and 6d2b, class v)
and 53d3 are the ones V2 drops; the four left are all class (iv), lateral / opposite lane (offroad 3, the 9a67 head-on).

## Open-loop base reads (descriptive)

| base | arrival @k=1 | dev ADE | wait W | go G | D | S_L F | Y_P F | Y_V F |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RTB1-F-s0 | 0.61 | 0.5556 | 0.967 | 0.879 | 0.846 | 0.844 | 0.211 | 0.342 |
| RTB05-F-s0 | 0.71 | 0.5536 | 0.965 | 0.875 | 0.839 | 0.847 | 0.211 | 0.342 |
| RTB1AR-F-s0 | 0.89 | 0.5519 | 0.326 | 0.992 | 0.318 | 0.849 | 0.218 | 0.335 |

V2's open-loop wait W fell from 0.967 to 0.326 and go G rose to 0.992, as designed (the W rows are what taught standing waits). The static scene
and stop17 still stop, so the closed-loop arrival stop survives without the W rows.

Dev-ADE decomposition ([rtb_v_devdecomp.md](rtb_v_devdecomp.md), 1789 dev rows, control RB0-F-s0): all three bases keep a 0.024-0.028 rise, 94-98 %
along-track. The change is where it sits. On v0 < 0.5 free-road rows the 4 s arc bias is -0.64 (V1) and -0.69 (RTB1) against +0.02 (V2); the
number of rows where the plan waits but the human goes is 6 (V1), 7 (RTB1), 0 (V2). V2's rise moves to v0 0.5-8 m/s (share 0.86).

## Against the prereg lines (描述, single seed)

"Cost fixed" = held zeros <= 37 and at-fault <= 6 and the arrival stop still seen.
- Held: V2 39 zeros and 7 at-fault, so the line is **not met** (V1 41 / 8, not met either). The arrival stop is kept in both.
- Public: V2 meets it (CI contains 0, rear 17 <= 20). V1 does not (rear 23).
- Reading rule: V2 fixes and V1 does not -> the cost is mostly from the W rows (supports the hypothesis), but for the public set only; on held V2 is a
  partial fix. V1 (a dose cut) only trades a little cost for a little stop (arrival 0.71 m, rear 23): no free point on the dose axis.

## What the cost diagnosis said ([explore_f.md](explore_f.md))

The public cost of a is a launch failure: in 12 scenes it stays under 1 m in 5 s while the log moves over 5 m (RSG 0), and these carry -9.44 of
the -17.15 paired points. The held shared zeros are mostly lateral (class iv 5 of 7), not stop / relaunch events. V2 targeted the first and, on the
public rear contacts, removes it; the second is not touched by removing W rows.

## Caveats

One seed per arm, one read each. V2 vs RSG is "not distinguishable", not "equal". PAI 281 and a second base seed were not read for V1 / V2. The V2-vs-a public gain has a CI edge of 0.001. The cost that is left is lateral (public offroad 36 vs 31, the four lateral shared
zeros), which fits the user's judgement from the zero videos that turn tracking on the RTB1 line is weaker than RSG's; this page has no turn-tracking
measure of its own (held turns 16 / 24 for V2, equal to RSG).

Files: `rtb_v_sum.{md,json}`, `rtb_cont_v{1,2}.{md,json}`, `rtb_wide_RTB05O-L18-s0.*`, `rtb_wide_RTB1ARO-L18-s0.*`, `rtb_v_gate_*`,
`rtb_v_devdecomp.*`, `explore_f.md` + `explore_f/`. Code: `scripts/nx.sh rtb-v`, `nx.py`, `nx_atfault.py --relaunch`.
