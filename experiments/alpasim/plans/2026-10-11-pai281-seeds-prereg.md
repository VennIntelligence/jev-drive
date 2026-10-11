# PAI-281: more seeds on the 281-scene PAI read (pre-registration, 2026-10-11)

Lane PAI-281. Written and pushed before any score of the new runs is read. A descriptive, mechanical extension of decision 247
([research/decisions/247.md](../../../research/decisions/247.md), [results/pai_seeds281.md](../results/pai_seeds281.md)): same scripts, same
configuration, same scenes, nothing new in the driver. It promotes nothing, changes no serving choice and adds no method; decision 247 keeps its
standing (descriptive, medium) and only gets an addendum. Which checkpoint is servable stays with the user.

## What runs

Three runs on the Tokyo box (two RTX 3090), one PAI stack per card, CONC 4, one rollout per scene:

| Run | Checkpoint | Scenes | Note |
|---|---|--:|---|
| base s1, rest | `P2H10-F-s1` | 142 | `s_b2`, `s_n1`, `s_n2`, `s_n3` of `/data/runs/alpasim/pai_full`: the scenes base s1 has not run (it has the other 139, `pai_base_s1`). Base s0's own `pai_base` run used exactly this list in this order, chunks of 36 / 36 / 36 / 34; so does this run. |
| S s2 | `P2H10S-F-s2` | 281 | `pai_full/s_*.tsv`, all 281 stage-1 scenes of decision 247 |
| S s3 | `P2H10S-F-s3` | 281 | same lists, same chunk cut as S s2 |

Afterwards the base has two seeds (s0, s1) and `P2H10S-F` four (s0 .. s3) on every one of the 281 scenes. S s2 and s3 are the main-box
checkpoints of decision 235's four seeds, pulled with `tokyo_pull.sh ckpt` and sha256-checked against the box (s2 `de577646...3e6b`,
s3 `357acaaf...1b55`). Nothing else is retrained or re-tuned.

Configuration, identical to decision 247 and to the stored rows: served `JEV_VCONT=1.0 JEV_LEAD=1`; `PAI_KEYWARP` unset (off; decision 252
found the off path equal to the stored rollouts on 33 of 33 scenes in each of three reruns); unharmonised renderer; image
`jev-alpasim:p2h10-f-s0-6f3d05d1`; CHUNK 36. Scene scores did not depend on the scene list in decision 252's reruns (33 of 33 scenes
identical against a stored 99-scene launch, 34 / 35-scene chunks and the same list), so the different chunk cuts of the older rows (per
list, 42, 35) do not need to be matched.

Queue: `scripts/pai_queue.sh` (new, thin): every card walks the three runs in order and takes the next unclaimed chunk of the current run
through `pai_eval.sh` (`JOIN=1`, one card per call), so a card that has nothing left in one run starts the next run at once and no card idles
while a chunk is unclaimed anywhere. A chunk that fails twice is retried in one repair pass at the end. Code is run from a frozen clone of
the commit that holds the script, so a later pull on the box cannot change a running queue. Expected: about 133 scenes per hour on two cards,
704 scenes, 5 to 6 hours.

## Read-out (fixed before the runs)

Tool: `scripts/pai_eval_report.py` (existing outputs unchanged; `--detail` adds the per-row and decomposition tables, `--only` restricts the
scenes). Rows are per-scene score vectors over the 281 scenes; a mean row is the per-scene mean of its seeds and is zero on a scene only if
every seed is zero. B2 = mean of base s0 and s1, S4 = mean of S s0 .. s3. Bootstrap: 10 000 resamples of scenes, seed 0, percentile 95 % CI
(as in decision 247); it covers scene sampling only, the seed variation is read from the per-seed rows and the seed-to-seed pairs.

1. Per row (each base seed, each S seed): mean scene score, scenes scored 1, slow scenes (0 < score < 1), zeros, and the zeros by scorer flag
   (at-fault collision, offroad, left corridor, other; a scene can carry two flags), plus the driver's error counters of the run dirs.
2. Paired differences with CIs, all on the 281 scenes: S s0, s1, s2, s3 each against base s0 and against B2; S4 against B2; base s1 against
   base s0 (the base's own spread). Supplementary, same method: each S seed against base s1; S4 and (S s0 + S s1) / 2 against base s0 and B2;
   the six S seed-to-seed pairs; every pair above restricted to the 139 scenes base s1 already had and to the 142 new ones.
3. For every pair: scenes fixed (zero on the second side, scored on the first), scenes broken (the reverse), both scored, both zero, and the
   decomposition of the mean difference into the zero -> scored part, the scored -> zero part and the both-scored part (each part is its
   scenes' summed difference divided by 281, so the three add up to the mean difference), as decision 247 point 1 does.
4. Pipeline check before anything new is added: the existing rows reproduce decision 247 (base s0 0.3746, S s0 0.4629, S s1 0.4837 on 281;
   base s1 0.3121 against base s0 0.3245 on the 139).

Caveat carried over unchanged (decision 247 point 3): `pai_core.slots` feeds rendered 10 Hz frames to all slots of checkpoints trained on
warped frames ([served_plan_length.md](../../body1/results/served_plan_length.md), a diagnosis without a second reader, fix not applied). Every
row is served that way, so absolute scores are low by an unknown amount; the paired differences have the same input on both sides.

## Rules

- No score of the three new runs is read until all three are done. While they run only health is read: STATUS / DONE / ERROR files, the
  number of rollouts per chunk, driver error counters, card clocks and utilisation.
- A scene without a result after the repair pass is named in the results and dropped from every row; the report prints each row's scene count.
  No scene is excluded for its score, a rollout that broke counts as the zero the scorer gave it (as in decisions 235 and 247).
- Whatever is looked at after reading, beyond the lists above, is marked exploratory in the results.

## What this can change in decision 247

Only the reading of its point 2: after this read there is no single-base-seed caveat left on the 281 scenes, and the base's own spread is
measured on all of them. The direction of 247 (S above base, the whole difference coming from zero-score scenes becoming scored) is said
to hold if S4 - B2 is positive with a CI excluding 0, every S seed is above B2, and |base s1 - base s0| is smaller than the smallest of the
four S seed - B2 differences. If not, the addendum states which part failed. Neither outcome promotes a checkpoint.

## As run (appended 2026-10-11 after the read; the text above is unchanged)

- Runs, configuration, chunk cut and order as written. The first launch hung before any stack started: `pai_queue.sh` initialises each run with
  `pai_eval.sh` in `JOIN=1` mode and no card, and a bare `wait` there never returned. Fixed in `pai_eval.sh` (36669eda), relaunched at 13:41:27 JST.
- The queue was paused by the user at 16:51:09 and resumed at 16:57:30 by running the launch script again; two in-flight chunks (`s2` c006, `s3` c000)
  were rerun from scratch, 34 min of card time lost. The rerun of c006 reproduced the killed attempt's scores and flags on 36 of 36 scenes.
- The read-out lists were run as written, including the supplementary pairs and the two halves. Added after the first read and marked exploratory in
  the results: the seed-pair decomposition of the pairs with a mean row, the zero-status flips between seed pairs, the scene counts 44 / 5 / 19, the SD
  of the S seeds and the ratio of the smallest S - base difference to the base's own.
- The statement of the last section was checked and holds (S4 - B2 +0.1100 [+0.0760, +0.1458]; every S seed above B2; 0.0016 against 0.0891).

