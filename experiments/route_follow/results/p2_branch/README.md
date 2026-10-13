# Turning school P2: branch ceiling (2026-10-13)

Prereg: [../../plans/2026-10-13-p2-branch-prereg.md](../../plans/2026-10-13-p2-branch-prereg.md). The ego follows the human log up to the
handover decision k and RSG-L18-s0 drives from k on (onp_log_driver.py `ONP_SWITCH`; gains 20, VCONT 1.0 LEAD 1). k is set at the log
point 40 / 20 / 10 / 0 m before the junction entry. Hard-list turn scenes, 6 left and 6 right; 48 rollouts on Tokyo card 1, code 146dc7ad.
The k17 column is the stored 281 rollout (pai_driver from the start). Files: `plan.json` (entry / exit / target, k per arm), `read.json`
(per rollout), `table.md` (rf_branch.py table), `smoke.txt` (the smoke gates).

**Result: ceiling 4 / 12 = 33 %, under the prereg's 40 % line.** Even handed the human's entry state, the policy cannot do most of these
turns, so the school has to teach how to turn (source 1 / 2 rows), not only when.

| arm | completed | zeros | mean score |
|:--|--:|--:|--:|
| k17 (policy from the start) | 1 / 12 | 12 | 0.000 |
| d40 | 1 / 12 | 12 | 0.000 |
| d20 | 2 / 12 | 11 | 0.083 |
| d10 | 3 / 12 | 10 | 0.167 |
| d0 (handover at the entry) | 4 / 12 | 8 | 0.273 |

- The handover state is right. The ego's offset from the log at k is <= 0.42 m (median 0.05 m) and, for every k > 17, its speed is within 0.6 m/s of the
  log's (read.json `ho_off`, `ho_v`; plan.json `v_log`). So the failures are not a bad entry state.
- Completion rises as the handover moves to the entry (1 -> 2 -> 3 -> 4), but only 3 scenes gained a completion that k17 did not have.
  382a7247 already completes at k17. The other three are eaba3f6a (from d20), 2f227f2c (from d10) and fd01e9ab (d0 only).
- 8 scenes fail from every handover, d0 included. At d0 the ego is at the 5-degree departure point with the log's speed and heading, and the
  policy still leaves: corridor 302c5c99, a174faed, e37aa159, f9e361aa; offroad 94c1d722, ce161092, e5e47fd2; collision 3ce13eed. Each fails
  the same way in all four arms. The 302c5c99 target is a 183-degree U-turn.
- By side: right 2 / 6 scenes completed (2f227f2c, eaba3f6a), left 2 / 6 (382a7247, fd01e9ab).
- The 9 d40 branches with k_eff 17 reproduce the k17 rollouts (same zero type and time). The smoke gates were bit-exact (`smoke.txt`).

Caveats: 12 scenes, one seed, one checkpoint. The handover is near the log, not from the policy's own states (route C, not A).

## Control arms at the d0 handover (prereg Amendment 2; 24 rollouts, code 748f8584, card 1, 23:24-23:49 JST)

Question: once handed over at the entry, does the policy turn because of its state or because of its hints? Both controls act only from k
on. The run itself confirms it: before k all 913 decisions per arm are bit-identical to d0's, and after k all 1471 have the control on
(`rmem` all zeros, or the speed-history entries of `vhist.ego_fed` all zero).

| scene | d0 | d0+rmem0 (route memory empty from k) | d0+vhzero (speed history zeroed from k) |
|:--|:--|:--|:--|
| 2f227f2c R | **done** | **done** | corridor (score 0) |
| 382a7247 L | **done** | offroad 7.5 s | **done** |
| eaba3f6a R | **done** | **done** | not done: rear-ended 4 m short (0.77) |
| fd01e9ab L | **done** | corridor | corridor |
| a174faed R | corridor | corridor | **done** (0.96) |
| f9e361aa R | corridor | corridor | **done** (0.82) |
| 302c5c99 L | corridor | corridor | not done: past target, > 2.5 m off the route (0.60) |
| ce161092 L | offroad | offroad | not done: rear-ended before target (0.82) |
| 3ce13eed, 94c1d722, e37aa159, e5e47fd2 | zero | zero (same type) | zero (same type) |

| arm | completed | zeros | mean score | mean speed 0-4 s after k |
|:--|--:|--:|--:|--:|
| k17 | 1 / 12 | 12 | 0.000 | |
| d0 | 4 / 12 | 8 | 0.273 | 6.24 m/s |
| d0+rmem0 | 2 / 12 | 10 | 0.167 | 6.37 m/s |
| d0+vhzero | 3 / 12 | 6 | 0.407 | 4.20 m/s |

- **Route memory: the prereg makes no call (2 of the 4 d0 completions kept).** Masking it after the entry loses 382a7247 and fd01e9ab
  and changes nothing in the 8 failures. So the memory helps in about half of the turns that work, and it isn't what blocks the rest.
- **Speed history: zeroing it is a large intervention, not a clean ablation.** By the letter of the prereg rule, only 1 of the 4 d0
  completions is kept, so that hint "carries" them. But the same arm completes 2 turns d0 never did (a174faed, f9e361aa), takes 6 scenes
  off zero and has the best mean score. The car runs about 2 m/s slower in the 4 s after the handover. The model reads a zeroed history
  as "slow", plans slower, and some turns then work. Two of the new non-zeros are cut short by the log's car behind (rear-ended).
- Across d0, rmem0 and vhzero, 6 of 12 scenes complete in at least one variant. 4 scenes fail in every variant, with the same failure
  type each time (3ce13eed collision; 94c1d722, e5e47fd2 offroad; e37aa159 corridor).
- Reading: at the entry, what decides the turn is mainly the speed the policy plans and holds, not the route memory. When the speed-history
  ratchet (decision 285) lets the car go slower into the junction, half of the failing turns work. The school's "how" rows should include
  the speed into the turn, not only the path. With 12 scenes and one seed this is a direction, not a rate.
