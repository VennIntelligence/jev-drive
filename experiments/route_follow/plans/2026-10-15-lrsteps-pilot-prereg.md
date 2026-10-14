# Pre-registration: lr x steps pilot on the RSG-L18 recipe, 2026-10-15

User approval (2026-10-15): "在 RSG-L18 配方上做一个 3 × 3 的 lr × 步数 pilot，只读 held 零分；最优者作为后面训练的基座".
RSG-L18 is the single candidate base (decision 286). Its learning rate and step count were never scanned; decision 264 addendum 6
showed the schedule matters (the full 10 000-step schedule cost the no-row control 0.092 in score, the gentle one 0.011).
Written, committed and pushed before any job of this pilot is submitted. Amendments are appended, never edited in place.

## Recipe (found, not reconstructed)

`$DATA_DIR/runs/route_mem/train-RSG-L18-s0/20261013-023026/meta.json` (argv; git d4864575, dirty):

```
rm_train.py train --tag RSG-L18-s0 --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank
  --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self
  --on on1_L18RB0 --seed 0
```

(= rf_rsgboth.sh's REC with `--on-mode self`.) Its log: 382 s, 6.9 it/s, 18.2 GB; dev ADE 0.5276 (@0) -> 0.5372 (@500) -> 0.5284 (@2500);
on-policy row stream hash at step 100 `f22e42deba570489`.

What `--lr-scale` scales (rm_train.py): the net (3e-5) and the old adapter (3e-4) groups. The route-memory branch keeps `--rm-lr 3e-4`
with its own 100-step warm-up under the same cosine; it is NOT scaled. This pilot scans `--lr-scale` only and keeps the branch lr as
RSG-L18 has it (the user asked for the recipe's lr knob; the branch lr is a separate question).

## Grid

`--lr-scale` {0.03, 0.1, 0.3} x `--steps` {1250, 2500, 5000}, seed 0, everything else the command above (same pool, same row-order
seeds: the draw RNGs depend on the seed only, so every cell sees the same row stream prefix), warm-up 1, cosine to 0 over the cell's
steps. `--eval-every` 250 for 1250 steps, 500 otherwise (dev eval is read-only, consumes no training RNG).
Tags `RSGlr{03,10,30}n{1250,2500,5000}-L18-s0` (lr03 = 0.03, lr10 = 0.1, lr30 = 0.3). The cell (0.1, 2500) IS `RSG-L18-s0`: its stored
checkpoint and stored held read are used, it is not retrained. So 8 new trainings.
Chain: [scripts/rf_lrsteps.sh](../scripts/rf_lrsteps.sh) (`first | rest | train | seed1 | report`).

## Read-out

Held 460 at 18 decisions (rm_chain.sh `read`, 6 chunks, the driver path of every RSG-L18 held read). Report as rf_round2.sh `report`:
`held.py` on the 460 common scenes (`held_raw.txt`), `ol_confirm.py` paired on the 452 scenes with a rollout (39 logs, log-clustered
bootstrap B = 10 000) with base RSG-L18-s0.
- **Primary:** held-452 zeros, paired difference vs RSG-L18-s0 with 95 % CI.
- Secondary (reported, never used for selection): mean score, at-fault zeros, collision / offroad / corridor split, GT distance, dev ADE,
  turn completion (rf_follow.py, 24 held turns).
- No PAI, no public 1491 in this pilot.

## Selection rule

1. Winner = the cell with the fewest held-452 zeros among the 9 (ties: fewer at-fault zeros, then higher mean score, then the cell
   closer to (0.1, 2500)).
2. The winner replaces RSG-L18-s0 as the base only if it is not (0.1, 2500) and its paired zero difference vs RSG-L18-s0 has a 95 % CI
   with upper bound < 0.
3. **Flat** = no such winner: either the fewest-zeros cell is (0.1, 2500), or its CI vs RSG-L18-s0 includes 0. Then RSG-L18 stays the
   base and the pilot concludes "flat". Cells whose CI lies entirely above 0 are reported as "worse" (the sensitivity map), they do not
   change the verdict.
4. **Stage 2** (only if rule 2 names a winner): train seed 1 of the winner (`--seed 1`, otherwise identical) and read it on held 460.
   Report both seeds and the seed mean vs RSG-L18 (s0, s1 mean; `ol_confirm --mean`). Reading: "replicated" if the seed-mean paired zero
   difference vs RSG-L18 also has CI upper < 0; otherwise "s0 only, not replicated". The public 1491 guard and the decision to adopt are
   the main session's.

With 8 comparisons at 95 % one false "winner" by chance is plausible; stage 2 is the guard against it.

## Cost estimate (written before launch)

- Training: measured 382 s per 2500 steps (0.153 s / step incl. dev evals and reliance) + ~1 min start-up per job. 8 cells =
  23 750 steps -> ~61 min + ~8 min = **~1.2 card-h** (one training per card, 18 GB).
- Held read: measured 6 chunks x 10.8-11.8 min (RSGSELF-L18, 2026-10-14) = ~1.1 job-h per checkpoint; packed up to 4 reads per card
  that is ~0.3 card-h, the 0.7 card-h reference is the conservative figure: 8 x 0.7 = **~5.6 card-h**.
- Report: CPU only.
- **Stage 1 ~6.8 card-h** (~3.6 if reads pack 4 per card); stage 2 +~0.8. Total **~7.6 card-h** worst case, under the 12 card-h stop.
  (The user was told "about 2 card-h"; the honest figure is ~7.) Wall: 3 idle cards (cl top 2026-10-15), ~1.5-2 h.

## Staged launch and stop rules

1. `first`: one cell, `RSGlr10n5000-L18-s0` (same lr as RSG-L18, so its start must look like RSG-L18's), with its 6 read chunks
   chained. Checks before `rest`: the job starts; losses finite; the pool line (rows 17 429, agent / road rows) equals RSG-L18-s0's;
   on_hash at step 100 = `f22e42deba570489`; step-100 losses within a few % of RSG-L18-s0's (imit 0.7552); dev ADE @ 500 within 0.02 m of
   RSG-L18-s0's 0.5372 (and the final dev ADE within 0.02 m of 0.528). Any miss: stop, nothing else is submitted, report (the recipe is not
   reproduced).
2. `rest`: the other 7 trainings as independent pool jobs at once, each with its 6 read chunks `--after` it, one report job `--after`
   every read chunk.
3. A failed training (non-finite loss, crash) or read chunk: report at once; a crashed read chunk may be resubmitted once, a training
   is not retried silently.
4. Estimate above 12 card-h, or the pool busy with other lanes' work: report instead of launching. No job of another lane is evicted or
   deprioritised; priority 14 as route_follow.

Results: `results/lrsteps/` (`lrsteps.md` with the 3 x 3 table and the verdict by the rule above, `held18/confirm.md`, `held_raw.txt`,
`follow.md`). The decision entry is written by the main session.

## Amendment 1 (2026-10-15, after the stage-1 read; coordinator request): seed 1 of the fewest-zeros cell

**Seen before this amendment:** the 9-cell held read and its verdict, FLAT (results/lrsteps/lrsteps.md). The fewest-zeros cell is (0.03, 5000)
(`RSGlr03n5000-L18-s0`). Its zeros are 23 vs 26, -3.0 [-7.6, +1.0], so the CI includes 0. Its mean score is +0.0139 [+0.0032, +0.0270],
and that CI excludes 0 (mean score is a secondary metric, not used for selection).

**Added:** `RSGlr03n5000-L18-s1` (same command, `--seed 1`), read on held 460 at 18 decisions, submitted at pool priority 4 so it only fills
gaps behind the route-v2 jobs (`PRIO=4 rf_lrsteps.sh seed1 lr03n5000`). The report adds the seed-1 row, the 2-seed mean
`RSGlr03n5000-L18` vs `RSG-L18` (s0, s1 mean) and the same-seed pair s1 vs RSG-L18-s1. The stage-1 verdict (FLAT, RSG-L18 stays the
base) is not changed by this addendum. The read is descriptive: "the cell's score gain replicates" only if the 2-seed mean score
difference vs RSG-L18 has a CI excluding 0 and the s1 pair has the same sign. Any adoption is the main session's decision.
