# ARENA-V2: does the arena held split still order the private-board anchors under the fixed scorer? (2026-10-25)

Diagnostic, no training. Decision 481. Pre-registration [plans/2026-10-25-arena-v2-check.md](../plans/2026-10-25-arena-v2-check.md).
Full tables: [arena_v2/summary.md](arena_v2/summary.md) (+ `summary.json`).

**Question.** Decision 480: both boards were rescored with PR #215 (1b024f0). Decision 355's six-anchor tau-b +1.00 was the old scorer
against the old board. On the new board go-straight (0.8165) is now above YR (0.7969). Old board: YR 0.7999, go-straight 0.7995. Do the
anchors, re-read on the arena held split and scored with both scorers, still order like the board?

**Read.** There are 8 subjects: the six d355 anchors (garage G0014, go-straight = AlpaSim starter, YR10m10-F-s0, DD-sample, LTF-sample,
APY10m10-AB-s0) plus JTN-L18-s1 native and + stop3. All are native reads with the arena.sh / jt.sh drivers, on 6 of the 18 arena held
blocks (c0 3 6 9 12 15, the decision-378 subset). That gives 753 valid windows, the same for every subject, with no route-fails. Every
rollout the fix can change kept its ASL (`cb2v2.py prune`) and was rescored with `cb2v2.py score`, old and new variants.
- The old variant reproduces every stored score in all 8 reads (0 changed). The rescore is exact: unresolved risk is 0 in every read.
- Re-read against the stored d355 / jt reads on the same windows: garage, go-straight, YR, APY and both JTN reads are identical or nearly so
  (same-score share 0.97-1.00, mean difference 0). LTF differs by +0.001 and DD by +0.012 [+0.001, +0.023]: the DD sample is not
  deterministic.
- Wave 1 took 41 min on idle cards at priority 0. Wave 2 (the other 12 blocks) was **not** submitted: CB2's P1N training was about 15 min
  from its end, and its reads would have been held up by ours. The command is ready:
  `arena_v2.sh wave w2 "1 2 4 5 7 8 10 11 13 14 16 17"`, then `arena_v2.sh sum`.

## Result

| subject | old scorer | new scorer | new - old | windows up / down | board old | board new | board new - old |
|---|--:|--:|--:|--:|--:|--:|--:|
| JTN-s1 + stop3 | 0.9551 | 0.9577 | +0.0027 | 2 / 0 | 0.9503 | 0.9553 | +0.005 |
| JTN-s1 native | 0.8711 | 0.8684 | -0.0027 | 5 / 7 | 0.9155 | 0.9145 | -0.001 |
| garage | 0.8028 | 0.8055 | +0.0027 | 5 / 3 | 0.8479 | 0.8529 | +0.005 |
| YR | 0.7644 | 0.7657 | +0.0013 | 10 / 9 | 0.7999 | 0.7969 | -0.003 |
| go-straight | 0.7072 | 0.7338 | **+0.0266** | 20 / 0 | 0.7995 | 0.8165 | **+0.017** |
| DD | 0.6882 | 0.7002 | +0.0120 | 19 / 10 | 0.7717 | 0.7777 | +0.006 |
| LTF | 0.6769 | 0.6876 | +0.0106 | 19 / 11 | 0.7377 | 0.7407 | +0.003 |
| APY | 0.6584 | 0.6531 | -0.0053 | 10 / 14 | 0.6401 | 0.6371 | -0.003 |

| scorer vs board | 6 anchors tau-b | 8 subjects tau-b | 8 subjects slope / r |
|---|--:|--:|---|
| old vs old (d355 setting) | +1.000 | +1.000 | 0.99 / 0.93 |
| **new vs new** | **+0.867** | **+0.929** | 0.96 / 0.94 |

- **The arena order does not change under the fix.** Under both scorers it is stop3 > JTN-s1 > garage > YR > go-straight > DD > LTF >
  APY. The one discordant pair against the new board is go-straight / YR, the pair the board swapped.
- **The scorer reproduces the board's move.** On the arena the fix lifts go-straight by +0.027, with all 20 changed windows going up: about 16
  are braking windows, where the human slows to a stop and the stationary go-straight car used to be zeroed by the corridor rule (braking
  0.433 -> 0.546). YR moves by +0.001, so go-straight - YR moves by +0.025; on the board that pair moved by +0.020. Across the 8
  subjects, the arena change (new - old) against the board change has r 0.91. Creeping on static windows costs everyone (static stratum
  -0.02 to -0.04 for YR / DD / LTF / APY / JTN native). Moving-then-stopped humans give back zeros (braking +0.02 to +0.11), as decision
  480's mechanism says.
- **go-straight - YR on the arena is a tie on this subset under both scorers:** old -0.057 [-0.169, +0.062], new -0.032 [-0.144, +0.089]
  (log-clustered CI). Under the pre-registered rule this is a tie, not a confirmed inversion. The point estimate still has YR ahead
  because the arena under-weights what go-straight is good at (static windows 1.000, launch / turn very poor; decision 382: "go-straight is
  underestimated everywhere"). That bias was already in the old setting, where the board had the two tied (0.0005) and the arena had
  -0.057. It is not caused by the scorer.
- Every other adjacent pair keeps the board's sign under both scorers. The CIs that cover 0 on 753 windows are garage - YR, YR - DD, DD -
  LTF and LTF - APY: 6 blocks resolve the families, not neighbours 0.01-0.03 apart.
- The JTN pair (stop3 - native) is +0.089 [+0.030, +0.158] on the arena under the new scorer, against +0.041 on the board. The direction
  is right; the arena exaggerates static-hold gains about 2x, as before (old +0.084 vs board +0.035).

**Reading for decision 355.** The arena held split is still the selection ruler under scorer_v2. It orders all of our own submissions
and garage / YR / DD / LTF / APY as the board does. The only anchor it cannot place is go-straight, and that is a known arena bias
against pure standstill policies, not a scorer effect. Select under the new scorer (cb2v2 rescoring), which is what CB2 already does.

## Frozen PCS pool (decision 382)

The anchors' old and new subset rows were scored on the frozen board-mix items (`pcs_local.py score --subset`, 716 of the 2 106 pool
items):

| subject | frozen PCS, old scores | frozen PCS, new scores | board PCS old -> new |
|---|--:|--:|--:|
| garage | 1519 | **1428** | 1550 -> 1538 |
| JTN-s1 + stop3 | 1358 | 1359 | 1470 -> 1470 |
| JTN-s1 native | 1218 | 1196 | 1321 -> 1317 |
| YR | 1095 | 1086 | 1017 -> 1023 |
| go-straight | 1053 | 1064 | 1312 -> 1345 |
| DD | 1053 | 1049 | 1070 -> 1080 |
| LTF | 1041 | 1036 | 1002 -> 1013 |
| APY | 1016 | 1005 | 755 -> 766 |

**The frozen pool needs a v2 refit before its PCS is used on new-scorer rows.** New-scorer rows on old-scorer items give garage -91
PCS while its mean goes up (+0.003) and its board PCS barely moves (-12). A handful of garage windows flip on items that the old fit made
highly discriminating, and on the board the items were refitted together with every subject's new scores. Our frozen items were not. On
the means the fix is small (+-0.01). On frozen PCS it can be about 100 for one subject, so CB2's planned "new scores on the frozen items
(approximate)" PCS column should be read as unreliable, and not used as a tie-breaker. A proper refit needs new-scorer reads of every
pool subject: 33 subjects x 18 blocks, the cost CB2's pre-registration already declined. This lane did not refit. The arena mean under the
new scorer stays the selection ruler.

## Files and rerun

Box `$DATA_DIR/runs/alpasim_proxy/arena/v2read/`: `reads/<tag>/c<k>/run` (ASLs kept where v2 can change), `v2/<tag>/{old,new}/rollouts.csv`,
`report/` (summary, `rows_{old,new}.csv.gz`).
```bash
bash experiments/alpasim_proxy/scripts/arena_v2.sh wave w1 "0 3 6 9 12 15"     # PRIO 0; reads + per-tag v2 jobs
bash experiments/alpasim_proxy/scripts/arena_v2.sh wait w1                      # one background waiter
.venv/bin/python experiments/alpasim_proxy/scripts/arena_v2.py sum
tmp/pcs_local/.venv/bin/python experiments/alpasim_proxy/scripts/pcs_local.py score --rows rows_new.csv.gz --subset   # Mac
```
