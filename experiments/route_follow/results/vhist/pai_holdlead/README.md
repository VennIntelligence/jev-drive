# VHIST holdlead on PAI 281: RSG-L18-s0 + `JEV_VHIST=holdlead` against the stored gains-20 reference, 2026-10-14

Prereg: [plans/2026-10-13-vhist-prereg.md](../../../plans/2026-10-13-vhist-prereg.md), Amendment 2 (fc3483cb, pushed before the run; the
user overrode the PAI rule on 2026-10-14). No research/decisions entry.

**Verdict: PASS.** Mean score +0.052 [+0.030, +0.075] (scene bootstrap, 281 scenes) with front-collision zeros +1 (limit +2). The gain is
progress, not fewer zeros: zeros 72 vs 70, rear-ended 64 vs 77, progress score +0.057, plan 2 s deficit +0.125 m. Turn completion is
unchanged (5 / 36; the hold releases on a turn command). **On the 72 never-trained exam scenes it is flat**: +0.001 [-0.046, +0.042],
zeros 41 vs 38, GT +0.16 m [+0.04, +0.32]. The 281-set gain comes from the 209 non-exam scenes.

## What ran

- Tokyo, worktree `/data/runs/route_night/vhist_pai/wt` at fc3483cb. `pai_eval.sh` used the reference's own 8 chunk lists, the driver image
  `jev-alpasim:p2h10-f-s0-6f3d05d1`, `alpasim-base:0.89.0`, gains 20, CONC 4, and
  `SERVE=-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e JEV_VHIST=holdlead`. Output: `/data/runs/route_night/r281/rsg_holdlead`.
- Cards and timing: card 1 ran 08:11-11:19 JST. Card 0 joined at 10:19 and took c005 and c007, on a user instruction relayed by the
  coordinator that released card 0 for this read. Chunks took 1741-1840 s each (reference 1859 s); the whole read took 3.1 h against the
  4.2 h estimated for one card.
- Failed rollout: 00fabf60 (c002) failed in the renderer (sensorsim RPC "tensors on different devices": infrastructure, no metrics). It
  was rerun once, alone, on card 1 (`r281/rsg_holdlead_rerun`, 218 s) and merged. Merged root of symlinks: `/data/runs/route_night/vhist_pai/merged`.
- Smoke (card 1, first 2 scenes of c000):
  - Switch off: 398 / 398 plans bit-identical to the stored reference.
  - holdlead: 0 rule mismatches over 398 records.
- Full read, `rf_vhist.py check`: 55 693 records, 0 mismatches. Of these, 2 near-ties and 9 held rows that differ only in the
  re-spaced history's lateral y count as matches: that y is limited by the log's rounding, largest 0.074 m.
- Consistency: on the 99 scenes the coordinator-exception exam read had already driven, the plans are 19 653 / 19 653 identical.

## Tables

Scripts:
- `scripts/rf_vhist_pai.py` wrote `tables.md`, `vhist_pai.json` and `scenes.csv`.
- `rf_pai281.py` wrote `pai281.md`, the same format as `r281/exam_rsg_holdlead_read.md`.
- `rf_pai_follow.py` wrote `follow/turns.md`, `follow/zeros.md` and `follow/pai_follow.json`.

| set | label | mean | zeros (coll / off / corr) | front-collision zeros | rear-ended | progress score | GT (m) | spins | deficit median / rollout mean (m) |
|:--|:--|--:|:--|--:|--:|--:|--:|--:|:--|
| 281 | rsg (stored) | 0.5720 | 70 (9 / 14 / 47) | 3 | 77 | 0.7448 | 2.175 | 26 | -0.191 / -0.597 |
| 281 | holdlead | 0.6239 | 72 (9 / 14 / 49) | 4 | 64 | 0.8018 | 2.239 | 26 | -0.141 / -0.472 |
| exam 72 | rsg (stored) | 0.3348 | 38 (6 / 9 / 23) | 2 | 11 | 0.6916 | 2.494 | 10 | -0.227 / -0.769 |
| exam 72 | holdlead | 0.3361 | 41 (7 / 9 / 25) | 3 | 7 | 0.7258 | 2.655 | 10 | -0.172 / -0.627 |

Paired holdlead - rsg (scene bootstrap 95 %, 10 000 draws):

| set | score | zeros | front-coll zeros | rear-ended | progress score | GT (m) | deficit, rollout means (m) |
|:--|:--|--:|--:|--:|:--|:--|:--|
| 281 | **+0.0519 [+0.0303, +0.0748]** | +2 (4 to zero, 2 from zero) | **+1** | -13 | +0.057 [+0.042, +0.073] | +0.064 [-0.015, +0.148] | +0.125 [+0.097, +0.155] |
| exam 72 | +0.0013 [-0.0461, +0.0418] | +3 (3 to zero, 0 from) | +1 | -4 | +0.034 [+0.011, +0.062] | +0.161 [+0.037, +0.320] | +0.142 [+0.087, +0.204] |

Spins are counted by `sf_detect.py`, all kinds; the handoff spins alone are 2 vs 2 (`pai281.md`).

Turn completion (`rf_pai_follow.py`, the registered rule, 36 turn scenes):
- rsg 5 / 36 = 13.9 % [3, 25]; holdlead 5 / 36 = 13.9 %.
- @bend 19.6 % vs 21.7 %; @open 13.7 % vs 11.8 %.
- Entry speed / log speed: median 1.10 vs 1.07.

Flips and front collisions:
- Scenes that became zero: 7750265d (corridor, from 1.0), 444fca69 (corridor, from 1.0), c4a58f83 (corridor, from 0.68), 8120803d
  (collision, from 0.48). 7750265d, 8120803d and c4a58f83 are exam scenes; 444fca69 is not.
- Scenes that left zero: ba62e10d (was corridor), 229791a5 (was collision).
- The extra front-collision zero, 382a7247, was a zero in the reference too, of a different kind; holdlead adds no new zero there.

## Reading

- The PAI result repeats held 460: front collisions are unchanged (+1 against the plain hold's +35 on nuPlan), rear-end truncations
  fall (-13) and progress rises. The cost is a few corridor zeros (+2) and GT (+0.06 m here, +0.16 m on the exam scenes; +0.10 m on held).
- The pass rests on the non-exam scenes. On the 72 exam scenes the score is unchanged, with 3 new corridor / collision zeros against a
  progress gain. Taken alone, the exam subset would not pass.
- Interim read on c000-c003 (`/data/runs/route_night/r281/rsg_holdlead_interim.md`): +0.048 [+0.012, +0.082], consistent with the full read.
