# PAI-PCS-V2: the PAI PCS ruler refit under the fixed corridor scorer (2026-10-23)

The decision-392 ruler ([../pai_pcs_local.md](../pai_pcs_local.md)) refit after the organisers fixed the ground-truth corridor scorer
(AlpaSim e2e_challenge 1b024f0, PR #215) and recomputed the PAI board (snapshot [../leaderboard_pai_20261023.json](../leaderboard_pai_20261023.json)).
Code: `scripts/pai_pcs_v2.py` (`collect` on the box, `fit` / `report` on the Mac in `tmp/pcs_local/.venv`), which reuses
`pai_pcs_local.py` / `pcs_local.py`. Per-scene scores: `experiments/alpasim/scripts/scorer_v2.py` (decision 453). Generated
tables: [report.md](report.md). Decision 455.

## Short answer

1. **The validation passes, but only inside our own family.**
   - **p1021a (NTV0-s0 + ACC + GAP): predicted 2265 [2202, 2343] against the board's 2251 (+14).** Over 14 pool variants the error
     ranges from -44 to +34.
   - **NTV0 - RSG: +44 at the point estimate, +39 [-16, +123] paired, against the board's +62.**
   - Both of these subjects are the pool's strong band. Neither is an anchor.
2. **The anchor fit is weaker than decision 392's.** Anchor RMSE is 71 (392: 25).
   - Leave-one-out: garage -153, go-straight +227, RSG +471.
   - Garage is under-fitted at 1739 against 1837. Its local new-scorer mean (0.534) is 1.15x its board mean; go-straight's is 1.29x.
3. **Every R6DAG / R8K read is predicted below NTV0-s0, by 110-330 PCS, even though their local means are 0.02-0.06 higher.**
   - **R8K-L18-s2:** 2059 OFF / 1931 ON, against NTV0 -207 [-338, +19] / -144 [-279, -60].
   - **R6DAG-L18-s1:** 2001 OFF / 1893 ON, against NTV0 -258 [-352, -126] / -179 [-314, -92].
   - The 392 ruler said the same on old scores (DAG-s1 -153, R8K-s0 / s1 -181 / -156; [round8 pcs_r8k.md](../../../round8/results/r8k/pcs_r8k.md)).
     So this is the ruler, not the scorer change.
   - **Why: the strong band is now two reads, RSG-s0 and NTV0-s0.** The other 11 of 392's 13 band reads are partial under the new
     scorer, and 4 were never rescored.
   - All 112 items with discrimination a >= 2 are scenes that both band reads pass and the garage mostly fails. A candidate that
     misses any of them is read as low ability.
   - Example, R6DAG-s1 OFF: its 5 such misses (0.7 points, all partials) are worth +198 PCS; its other 48 losses (32.7 points)
     are worth +8.
   - **The result reverses with the pool.** Pooling all 27 complete ACC + GAP reads (`allag`) puts R6DAG-s1 / R8K-s2 OFF at
     2522 / 2554, 300 above NTV0. Fixing a (`1pl`) puts them at 2334 / 2340, +105 above NTV0. Those two variants fit the anchors
     best (RMSE 17 / 11) but miss p1021a by -44 / -22.
   - **Reading: this ruler cannot rank the DAgger family against NTV0.** Whether R8K / R6DAG beat p1021a on the board is not
     settled by it.
   - The local means (decision 453: +0.047 / +0.045 for 3 seeds, paired) are the only evidence, and they point the other way
     from the default fit.
4. **Harmonizer on costs 70-180 PCS in NTV0 and 5 of the 6 R reads** (NTV0 2265 -> 2083). The exception is R6DAG-s0
   (1909 -> 1913). The local means differ by only 0.005-0.03.
5. **The overshoot items moved from near-zero to mid discrimination.**
   - These are the 36 items the fix changed in some read. Their median a went from 0.16 to 0.66. Under the old scorer every band
     read and the garage failed them together, so they did not separate anyone.
   - Most are now passed by all four reference subjects (a about 0.5).
   - Five, where go-straight still fails and the band and the garage now pass, became top items (a about 6.8: b333cf02, ba005726,
     61d738bb, 11250b18, 5c9c5f59). One more, which the band passes and both third-party drivers fail, sits at 2.47 (92e7e034).
   - The 239 unchanged items barely moved (median a 0.82 -> 0.79; Spearman of a old / new over all 275: 0.91).
   - By decision 381 stratum: straight 2.88 -> 2.42, launch 0.80 -> 0.76, stop / lead 0.77 -> 0.75, turn 0.16 -> 0.16.

## Inputs

- **Anchors** (new board): garage `014-pai-0014-v1` **1837** (mean 0.4649), host `go_straight_with_delay_v2` **1441** (0.1746),
  vennai-org `rsg-l18-s0-acc-gap-v1` **2189** (0.6802).
- **Validation** (not an anchor): ujs-iv `p1021a` = NTV0-L18-s0 + ACC + GAP, **2251** (0.7155).
- **Anchor re-reads:** `$DATA_DIR/runs/alpasim_proxy/pai_pcs_v2/{G0014,GS}`. Same commands and chunk table as 392, zero-score
  ASLs kept.
- **Rescoring:** scorer_v2, CPU, 16 single-thread shards, root `pai_pcs_v2/rescore`.
  - Old-scorer reproduction gate: all 327 re-evaluated parquets are identical, and every rollout with metrics matches its stored
    score and failure reason.
  - New-scorer means: G0014 0.477 -> 0.534 (16 scenes flip), GS 0.160 -> 0.221 (17 flip, 276 scenes).
- **Five GS-v2 rollouts failed in the runtime** (`can't start new thread`, the box pids cap). They have no metrics and a stored 0.
  - The first gate run flagged them. scorer_v2 now lists them with the RPC failures, and `collect` masks them like RPC errors.
  - Four are among the 392 items. The default (`--gs fill`) takes go-straight's scores there from the first GS read
    (`pai_pcs/GS`): 0 / 1 / 0 / 0.
  - **Rerun:** the 5 scenes were re-read alone as one pool job (`pai_pcs_v2/GS5`, same settings, KEEP_ASL=1) and rescored
    (`rescore5`; new = old on all 4 that ran). That gave 0 / 1 / 1 / 0, and 1bfad87c failed again with an RPC error, as it did in
    the 392 reads.
  - The rerun therefore agrees with the fill on every item it scored, and leaves 274 items.
  - **The handling barely matters:**

| go-straight on the 5 | items | anchor RMSE | p1021a | NTV0 - RSG | R6DAG-s1 off / on | R8K-s2 off / on |
|---|--:|--:|--:|--:|---|---|
| fill from the first read (default) | 275 | 71 | 2265 | +44 | 2001 / 1893 | 2059 / 1931 |
| mask, items dropped (option 3) | 271 | 69 | 2255 | +34 | 1994 / 1883 | 2058 / 1922 |
| rerun | 274 | 69 | 2280 | +60 | 2006 / 1903 | 2046 / 1936 |

- **Band cuts** shift with the board: a local linear fit of new on old PCS, over the 104 submissions on both boards within
  +-60 of each 392 cut. 1940 / 1750 / 1440 become **1961 / 1772 / 1441**.
  - On the 120-row board that gives band 37, garage 45, go-straight 14, low 24 (left out, as in 392).
  - The `oldcuts` variant (40 / 45 / 11) changes p1021a by 3.
- **Pool:** 45 garage + 14 go-straight + 37 band rows. The band is spread evenly over RSG-s0 and NTV0-s0 + ACC + GAP, the only
  two of 392's 13 band reads that are complete under the new scorer (392 rounded k / n per read).
  - **The partial reads are dropped from the pool** (default) and scored out of pool as lower bounds. The `lb` variant pools the
    7 rescored partial band reads as they are.
  - NTV0-hl and STANDFMX-s0 / s1 / STANDFMXH-s0 kept only 8 of their zero ASLs and were not rescored.
  - There is no `dedup` variant: with 4 rows it fails the fit's data guard.
- **Fit settings:** same as 392, plus a scene bootstrap of 100 draws.

## Predictions (default pool, new scorer)

Paired differences are on the same bootstrap draws, against NTV0-s0 with the same Harmonizer setting. Full table, with the
variant ranges: [report.md](report.md). Machine-readable: [pred.csv](pred.csv).

| candidate | OFF PCS [95 % CI] | OFF vs NTV0 | ON PCS [95 % CI] | ON vs NTV0 |
|---|---|---|---|---|
| NTV0-s0 (p1021a) | **2265** [2202, 2343] | reference | 2083 [1988, 2212] | reference |
| R6DAG-s0 | 1909 [1862, 2028] | -331 [-438, -227] | 1913 [1858, 2012] | -158 [-270, -85] |
| **R6DAG-s1** | **2001** [1934, 2152] | **-258 [-352, -126]** | **1893** [1848, 1979] | **-179 [-314, -92]** |
| R6DAG-s2 | 1967 [1911, 2166] | -286 [-384, -76] | 1879 [1828, 1957] | -191 [-324, -99] |
| R8K-s0 | 1993 [1935, 2146] | -278 [-369, -133] | 1923 [1869, 2003] | -157 [-274, -76] |
| R8K-s1 | 2032 [1948, 2244] | -244 [-351, -31] | 1962 [1895, 2037] | -114 [-247, -40] |
| **R8K-s2** | **2059** [1967, 2302] | **-207 [-338, +19]** | **1931** [1875, 2004] | **-144 [-279, -60]** |
| R6DAG 3-seed mean | 1959 | -287 [-373, -157] | 1895 | -176 [-308, -95] |
| R8K 3-seed mean | 2028 | -241 [-351, -43] | 1939 | -139 [-267, -66] |

- (OFF + ON) / 2: R8K-s2 1995 (-179 against NTV0) and R6DAG-s1 1947 (-218).
- R8K-s2 - R6DAG-s1 = +37 [-4, +138], P(> 0) 0.96.

The pool variants that fit the anchors best disagree on the sign, in PCS:

| variant | anchor RMSE | p1021a error | NTV0 off | R6DAG-s1 off / on | R8K-s2 off / on |
|---|--:|--:|--:|---|---|
| v2 (default) | 71 | +14 | 2265 | 2001 / 1893 | 2059 / 1931 |
| low band as go-straight | 12 | -15 | 2236 | 2028 / 1931 | 2096 / 1977 |
| 1PL | 11 | -22 | 2229 | 2334 / 2211 | 2340 / 2253 |
| allag (27 complete reads as the band) | 17 | -44 | 2207 | 2522 / 2285 | 2554 / 2322 |

## Commands

```bash
# box, repo root
P=$DATA_DIR/runs/alpasim_proxy/pai_pcs_v2
$P/rescore/run.sh            # scorer_v2 old -> new -> report for G0014-v2, GS-v2 (16 workers); $P/rescore5/run.sh for GS5-v2
CHUNK=5 DRV=starter SERVE= ALPASIM_FAST=0 KEEP_ASL=1 WIZ=eval.video.render_video=false \
  bash experiments/alpasim/scripts/pai_box.sh submit $P/GS5 GS $P/GS5/scenes.txt        # the 5 thread-cap failures, re-read
.venv/bin/python experiments/alpasim_proxy/scripts/pai_pcs_v2.py collect --out $P/rows_rerun.csv.gz
# Mac
X="tmp/pcs_local/.venv/bin/python experiments/alpasim_proxy/scripts/pai_pcs_v2.py"
$X --gs fill fit --rows rows.csv.gz --out out                                     # 14 variants, ~40 s each
$X --gs fill fit --rows rows.csv.gz --out out --variants v2 --old                 # same pool on old-scorer scores (discrimination check)
$X --gs fill fit --rows rows.csv.gz --out out --variants - --boot 25 --boot0 {0,25,50,75}   # 4 parallel shards, ~25 min
$X --gs fill report --rows rows.csv.gz --out out --gap scenes.csv                 # scenes.csv: $DATA_DIR/runs/route_follow/pai_gap/scenes.csv
```

Files: [rows.csv.gz](rows.csv.gz) (new / old score, RPC / infra flag and partial flag per read and scene, GS5 rerun included),
[boot.csv.gz](boot.csv.gz) (100 draws x subjects, default pool), [items.csv](items.csv) (a / b per item, old and new scores,
reads changed), [pred.csv](pred.csv), [pai_pcs_v2.json](pai_pcs_v2.json).
