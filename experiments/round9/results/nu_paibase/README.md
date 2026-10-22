# NU-PAIBASE: the PAI-mixed bases through the nuPlan candidate's recipe (2026-10-22)

Plan (Chinese, registered before any training or read; addendum 1 before any training):
[../../plans/2026-10-22-nu-paibase.md](../../plans/2026-10-22-nu-paibase.md). Code: [nupb.sh](../../scripts/nupb.sh) (the whole chain in one
submit), [nupb.py](../../scripts/nupb.py) (smoke check, Mac PCS / easy items / verdict). Box lane dir `runs/round9/nu_paibase/`.

**Question.** Decision 449's PAI-mixed base RB0-P70-s0 had fewer corridor zeros than RB0-F-s0 on the nuPlan arena (353 vs 392) but more offroad
(94 vs 55) and AF-front (37 vs 11) zeros. Does this base, carried through B2's recipe (L18 S stage), keep the corridor gain while S repairs
offroad / AF-front, and so beat B2?

**Answer: no.** Neither share passes the registered line (both seeds, each against B2 of the same seed: arena mean >= B2, corridor zeros < B2,
held 460 zeros <= B2 + 2, frozen PCS >= B2). Every arm's PCS is 11-44 below its B2 seed. Every arm has more held 460 zeros than B2 (+2 to +8).
Only NPB70-s0 has fewer corridor zeros, and the s1 seed reverses that. The base-stage corridor advantage (-39) does not survive the S stage, which
matches decision 346 (L18 washes out what the base learned about waiting). Base-stage reference: [arena_base.md](arena_base.md).

## Arms

B2 = `STANDFMXH-L18-s0 / s1` (decisions 389 / 391). Each arm uses B2's training command for its seed (stored command = meta.json argv,
code 5858f899) with only two values changed: `--tag` and `--init RB0-F-s0` -> `--init RB0-P70-s0` or `RB0-P30-s0`
([cmd_diff.txt](cmd_diff.txt)). Tags: `NPB70-L18-s0/s1`, `NPB30-L18-s0/s1`. 9 500 steps, compiled, HEAD 5c1fa909.

## Gates (passed)

- Identity ([identity.json](identity.json)): B2-s0's command, 60 steps eager, 5858f899 (its own worktree) vs HEAD. Bitwise identical:
  315 scalars max diff 0, weights max diff 0, dev ADE 0.54682 for both runs, on_hash_60 81d78547676d739f for both runs. Addendum 1: since the
  10-21 swap, on1_JTX is a zero-copy pool. The old tree opens its manifest through HEAD's `jevdrive.tokens.open_front`, which is bitwise equal
  to the copied pool (cl_infra/results/zero-copy). Without this change 5858f899 cannot read the pool at all.
- Smoke ([smoke.json](smoke.json)): P70-s0's command, 60 steps eager. Losses are finite and the init is RB0-P70-s0 (init dev ADE 0.583 vs
  0.528 for RB0-F). Wrong-init control: after 60 steps dev ADE is 0.610 vs 0.547 and weights differ by up to 0.37.

## Verdict ([verdict.md](verdict.md))

| arm | ref | arena paired [CI] | corridor zeros arm / B2 | held 460 zeros arm / B2 | PCS arm / B2 | a | b | c | d |
|:--|:--|--:|--:|--:|--:|:-:|:-:|:-:|:-:|
| NPB70-L18-s0 | B2-s0 | +0.0046 [-0.0054, +0.0154] | 34 / 37 | 49 / 41 | 1349 / 1369 | pass | pass | FAIL | FAIL |
| NPB70-L18-s1 | B2-s1 | -0.0050 [-0.0178, +0.0066] | 52 / 45 | 48 / 41 | 1354 / 1398 | FAIL | FAIL | FAIL | FAIL |
| NPB30-L18-s0 | B2-s0 | -0.0021 [-0.0121, +0.0087] | 45 / 37 | 43 / 41 | 1350 / 1369 | FAIL | FAIL | pass | FAIL |
| NPB30-L18-s1 | B2-s1 | -0.0048 [-0.0161, +0.0054] | 55 / 45 | 44 / 41 | 1358 / 1398 | FAIL | FAIL | FAIL | FAIL |

The arena differences are inside the B2 seed spread (|diff| < 0.005, except NPB70-s0 at +0.005). PCS is the frozen board-mix model
(`pcs_local.py score`, 2 106 items, [pcs.txt](pcs.txt)). It reproduces B2's stored 1369 / 1398 and JTN-L18-s1's 1247 exactly.

## Zero layers (arena held, route-fail dropped)

| tag | zeros | corridor, static window | corridor, moving window | offroad | AF-front | PCS |
|:--|--:|--:|--:|--:|--:|--:|
| NPB70-L18-s0 | 80 | 13 | 21 | 47 | 1 | 1349 |
| NPB70-L18-s1 | 110 | 27 | 25 | 56 | 2 | 1354 |
| NPB30-L18-s0 | 96 | 24 | 21 | 49 | 2 | 1350 |
| NPB30-L18-s1 | 111 | 34 | 21 | 51 | 4 | 1358 |
| B2-s0 | 92 | 14 | 23 | 53 | 3 | 1369 |
| B2-s1 | 101 | 21 | 24 | 55 | 2 | 1398 |
| JTN-L18-s1 | 210 | 130 | 21 | 58 | 7 | 1247 |

The S stage fixes offroad and AF-front as expected: they return to B2's level. The static-window corridor zeros do not stay lower. Seed noise
dominates (B2 14 / 21; P70 13 / 27; P30 24 / 34), and P30 is worse than B2 in both seeds. Held 460 offroad zeros are 20-21 in every arm vs
17 / 19 for B2 ([held460.md](held460.md)).

## Easy items (reported only; `easy_items.py subject`, ref JTN-L18-s1, the 2 106 frozen items, easy = garage 0014 scores 1)

| tag | PCS | zeros | partials | easy zeros | easy partials | easy points lost |
|:--|--:|--:|--:|--:|--:|--:|
| NPB70-L18-s0 | 1349 | 75 | 116 | 17 | 32 | 25.5 |
| NPB70-L18-s1 | 1354 | 105 | 94 | 29 | 20 | 35.7 |
| NPB30-L18-s0 | 1350 | 92 | 105 | 29 | 23 | 36.5 |
| NPB30-L18-s1 | 1358 | 104 | 95 | 34 | 19 | 40.3 |
| B2-s0 (stored, easy_items_B2_s0.json) | 1369 | 86 | 102 | 30 | 24 | 37.0 |
| B2-s1 (stored, easy_items_B2_s1.json) | 1398 | 96 | 86 | 31 | 16 | 37.0 |

NPB70-s0 loses fewer easy points than B2-s0 (25.5 vs 37.0; easy zeros 17 vs 30) and still scores 20 PCS lower. It has more partials (116 vs
102), so the loss sits in progress shortfalls on items with higher discrimination. The other three arms lose as much on easy items as B2 or
more. Per-window classes: `easy_<tag>.json`, `misses_<tag>.csv.gz`.

Against JTN-L18-s1 (reported only): arena +0.040 to +0.051 (CIs exclude 0); held 460 -0.024 to -0.039, the same direction as B2
(decision 391). Full tables: [arena.md](arena.md), [held460.md](held460.md); windows: [arena_windows.csv.gz](arena_windows.csv.gz).

## Cost

4 trainings plus 96 read jobs, submitted at 17:58. Reads finished at about 20:09. Estimate: about 7 card-h. Upper bound: 5 cards (shared with R9DL) x 2.2 h = 11 card-h, under the 15 card-h stop line. The summary needed two
fixes (the facts back-fill runs in the AlpaSim env; reference reads included).
