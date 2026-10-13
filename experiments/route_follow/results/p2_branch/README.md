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
