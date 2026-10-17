# GPU pool booking change b05b8fb4: pack-check validation

Checked 2026-10-17 15:25 (box clock, CST). Change under test: `rm_train` jobs share the history key `rm-train`, and same-key GPU jobs on one
card are booked together at m x 1.1 x the key's measured concurrent p99 VRAM share instead of m own p95 caps
([runbook](../../../docs/closed-loop-runbook.md) "VRAM" row, [speed.md](../../../docs/speed.md) headroom items 2-3). The dispatcher has run it since
13:10:10 (pid 598583). Fail criteria (speed.md item 3): wall per read more than +15 % against the pre-change reference; any OOM / SIGKILL; scores not
equal where they should be equal. Batch: box `runs/cl_infra/pack_check` (pool owner `cl-pack-check`, submitted 13:12, code b05b8fb4).

**Verdict: none of the three fail conditions holds, so this data gives no reason to revert. The validation is partial:** the speed.md check ("one batch at
5 sh30 / 4 PAI per card") was not carried out as specified. The batch never reached 5 sh30 reads on a card, reached 4 PAI reads only because it declared
19 GB instead of the default 22, and the load it did reach (4 jobs per card) is what the pre-change booking already admitted. What is missing is one GPU run
(section 6).

| criterion | result | status |
|---|---|---|
| wall per read <= +15 % | sh30: -3.7 % ... +4.9 % against 5 matched pre-change reads of the same list at about 4 jobs per card (worst c1b, 714.7 s against 681.2 s); PAI: +1.1 % (c000, matched); c001 lost one rollout to a renderer crash, so its wall is not comparable | pass |
| no OOM / SIGKILL | none in pool events, job logs or the kernel counter; peak host working set in the batch window 378 GB (82 %); two host-RAM episodes of explore-G CPU jobs (workers died) at 14:11 and 14:18, unrelated to GPU packing | pass |
| scores equal | the `score` field is equal in all 9 reads against their references (385 + 71 scenes); all other fields are bit-identical except one scene of sh30 c0c (0.24 m of travelled distance) and one PAI c001 rollout that is a renderer crash scoring 0 (the reference scores 0 there too) | pass, two caveats |
| load reached (not a fail criterion) | sh30: card 2 held 3 sh30 + 1 PAI read for 22 min, 4 sh30 for 21 s, never 5; PAI: 4 per card for 12 min, only at a 19 GB declaration | not reached |

## 1. What was run

No design document exists (not in the run dir, `experiments/cl_infra`, or the box repo). The design below is read off the pool specs (`cl show <id>`),
the chunk lists and the run layout. References were matched by list (path or md5), checkpoint tag and, for PAI, `config.txt`; the sh30 serve configuration is the
default of `rm_chain.sh readlist`, and the bit-identical results of section 5 confirm it.

| batch | reads | each read | card | declaration |
|---|---|---|---|---|
| sh30 | c0a, c0b, c0c; c1a, c1b | RSG-L18-s0 on held-460 chunk c0 (3 identical reads) and c1 (2 identical reads), 77 scenes, `rm_chain.sh readlist`, plain serve | 2 (`--gpus 2 --pin-strict`) | `CL_HIST_KEY=alpasim-read4`, `--vram 20 --cpu 6 --ram 40` |
| PAI | c000 ... c003 (17 + 17 + 19 + 18 scenes) | RSG-L18-s0, submission serve (`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`, steering weight 10), `ALPASIM_FAST=1`, `CONC=4`; c000 and c001 are the same list | 3 | `CL_HIST_KEY=pai-fast4`, `--vram 19` (the `pai_box.sh` default is 22), `--ram 30` |

References (same checkpoint, same list):

| batch reads | reference reads | used for |
|---|---|---|
| sh30 c0* / c1* | `runs/route_mem/read/RSG-L18-s0/c0` and `c1` (10-13 02:37, 2 jobs per card); the 10-16 re-read `RSG-L18-s0-re1016` (3.7 jobs per card, cards 0-1) is bit-identical to them | scores |
| sh30 c0* / c1* | five pre-change reads per list at 3.7-3.9 jobs per card (10-13 21:54 to 10-16 18:24; ids in section 3). Some use other serve switches (route gate, spin fix, `acc`, `vl`; `re1016` is the plain serve), so they serve as wall references only | wall |
| PAI c000, c001 | `runs/wait_cap/train/explore_e/paisub/RSG-L18-s0` c000 (job 1017-025556-b820, 10-17 02:57, 3.6 jobs per card) | scores, wall |
| PAI c002, c003 | `runs/route_follow/acc_gap/read/s1acc` c000, c001 (jobs 1016-194626-62c5, -b272, 10-16 19:46, alone on the card) | scores, wall |

## 2. Load that was reached

Both batches queued behind the compile-validation reads (same pool, same priority) and started as those ended: the first three reads of each waited on VRAM
520-587 s, the last ones 945-1 170 s. "Jobs per card" counts every pool GPU job (declared VRAM > 1 GB) on the job's card, time-weighted over its life.

- Card 2 (sh30): from 13:10:33 to 13:33:00 the card held three sh30 reads (compile-validation reads, replaced one by one by c0a-c0c between 13:21:03 and 13:22:10) + one
  compile-validation PAI read (22 GB): 4 jobs, 66.1 GB booked (3 x 14.7 + 22), VRAM used p50 44.9 / max 51.0 GB of 83.6 (13:22:10-13:31:52). A fourth sh30 read adds
  14.6 GB (80.7 GB booked against 79.6 usable = 83.6 - 4 GB headroom), so c1a and c1b waited. Four sh30 reads ran together for 21 s (13:32:38-13:32:59, 58.7 GB
  booked); then 2 sh30 + 2 PAI for 9.7 min (72.9 GB booked, VRAM used max 53.4 GB). A fifth sh30 read never ran: since 13:10 no card has held more than 4 pool GPU jobs.
- Card 3 (PAI): four `pai-fast` jobs for 12.3 min (13:22:10-13:34:28; three of the batch + one compile-validation chunk at a time): 79.0 GB booked of 79.6, VRAM used
  p50 60.8 / max 70.5 GB. It fit because the batch declared 19 GB per chunk: at the default 22 GB the group booking for 4 is 4 x 1.1 x 19.5 = 85.8 GB.
- Both schedules were also admissible under the pre-change arithmetic (3 x 17.7 + 22 = 75.1 GB; 3 x 19 + 22 = 79.0 GB, against 79.6). The batch therefore
  checks that 4 jobs per card are fine; it does not exercise what the new rule adds (a 5th sh30 read, a 4th PAI read at 22 GB).

## 3. Wall time

Wall = pool job start to end (includes driver warm-up and the `rollout.asl` prune). Reference for each sh30 list: median of five pre-change reads of that list at
3.7-3.9 jobs per card (c0: 657.6, 659.7, 673.5, 681.4, 681.4 s = jobs 1013-215254-3ba3, 1016-182327-4492, 1013-223439-b2c7, 1016-182329-ad10, 1016-182331-3d56;
c1: 651.4, 657.3, 681.2, 681.4, 681.5 s = 1013-223439-e79f, 1013-215254-67c8, 1016-182332-1dca, 1016-182329-7448, 1016-182327-a954).

| sh30 read | start - end | wall s | jobs per card mean (max) | same-key mean (max) | own VRAM peak GB | reference median s | delta |
|---|---|--:|--:|--:|--:|--:|--:|
| c0a | 13:21:03 - 13:31:52 | 648.7 | 4.00 (4) | 3.00 (3) | 11.8 | 673.5 | -3.7 % |
| c0b | 13:21:48 - 13:32:59 | 671.3 | 4.00 (4) | 3.03 (4) | 12.5 | 673.5 | -0.3 % |
| c0c | 13:22:10 - 13:33:21 | 671.3 | 4.00 (4) | 3.03 (4) | 11.6 | 673.5 | -0.3 % |
| c1a | 13:31:53 - 13:43:03 | 670.0 | 4.00 (4) | 2.16 (4) | 11.7 | 681.2 | -1.6 % |
| c1b | 13:32:38 - 13:44:32 | 714.7 | 3.87 (4) | 1.97 (4) | 11.9 | 681.2 | +4.9 % |

| PAI read | start - end | wall s | native s per scene | jobs per card mean (max) | own VRAM peak GB | reference wall s | delta |
|---|---|--:|--:|--:|--:|--:|--:|
| c000 | 13:21:25 - 13:34:51 | 805.0 | 46.7 | 3.97 (4) | 17.1 | 796.4 (3.6 jobs per card) | +1.1 % |
| c001 | 13:21:48 - 13:32:59 | 671.4 | 39.3 | 4.00 (4) | 17.0 | 796.4 | -15.7 % (one 198-step rollout crashed after 11 s, section 5) |
| c002 | 13:22:10 - 13:34:28 | 738.3 | 38.7 | 4.00 (4) | 17.7 | 798.5 (alone) | -7.5 % |
| c003 | 13:33:00 - 13:45:39 | 759.5 | 42.0 | 3.11 (4) | 19.1 | 798.6 (alone) | -4.9 % |

Cross-checks over the whole pool history (jobs per card = time-weighted mean, rounded; sh30 = 76-77-scene held chunks, "pre" = the 4 days before the change):

- sh30 wall median by jobs per card, pre: 1: 607.9 s (n 10); 2: 622.0 (73); 3: 643.9 (94); **4: 667.3 (378, p25-p75 646-687)**; 5: 712.7 (18). The batch mean is
  675.2 s (+1.2 % against the 4-per-card median); the worst read is +7.1 %. The pre-change history already holds 18 held reads at 5 jobs per card (median 712.7 s,
  +6.8 % over 4), which is the wall to expect at the untested level.
- Natural experiment: the compile validation's twelve held reads straddle the restart. Eight were placed by the old dispatcher (13:09:24-13:09:46), four by the new
  one (13:10:11), all on cards with about 3 sh30 + 1 other job (3.8-4.0 jobs per card): median 708.9 s against 718.2 s (+1.3 %).
- PAI (`ALPASIM_FAST=1`, native s per scene): pre 3 per card 39.0 (n 51), pre 4 per card 41.0 (n 9); the batch 41.6 over 71 scenes (+1.4 % against 4 per
  card, +6.6 % against 3); post-change reads at 3 per card 38.9 (n 37).
- If a solo read (607.9 s) were the reference, c1b would be +17.6 % and the batch mean +11 %. sh30 reads already ran at 4 per card before the change (since
  10-13, speed.md "Reads"), so that comparison measures the 4-per-card regime, not this change; the criterion is read against the pre-change level of the same packing.

## 4. OOM / SIGKILL

- Pool events 13:10:10-15:20 (106 launches, 103 ends: 94 done, 7 cancelled, 2 failed): no `oom`, `kill`, `vram_stop`, `vram_over`, `retry`, `ws_override`. The 24 h
  before: 2 `vram_over`, none of the others. `cl vram` (48 h, 509 prefixes) flags 3 prefixes, all under-declared with OOM 0 (pai-solo, alpasim-p4c, ra-prow-sm); none is a batch key.
- All 9 batch jobs: rc 0, tries 1. No `out of memory`, `CUDA error` or `Killed` in their pool logs, native logs, txt logs (the only "OOM" text in the renderer logs is
  its config echo "LRU with OOM retry"). Besides the route sanity-check failures every read of these lists has (deterministic, the same scenes in the references),
  the only exception-type rollout failure is the one in section 5.
- Kernel `oom_kill` counter 0 -> 0 (boxwatch). Host working set 13:10-13:50: max 378.2 GB at 13:19 (82 % of the 460 GB limit, 72.6 GB below the 450.8 GB kill line).
- Not caused by the change: the explore-G `geo` CPU jobs (xg2-geo from 14:09, declared `--ram 96`, 316 GB measured; xg3-geo from 14:15, declared 128, 322 GB; 32 workers each)
  drove the working set to 450.6 GB at 14:11:09 and 438.8 GB at 14:18:11, each followed by a drop of about 280 GB within seconds (boxwatch); xg3-geo then failed with
  "a worker died" (rc 1). This matches the platform's memory kill line hitting workers (inferred; no pool `kill` event because the job roots survived). Both are
  CPU-only jobs, 25-35 min after the batch ended, with 3 GPU jobs per card.

## 5. Scores

Exact comparison of every numeric field of `aggregate/results-summary.json` (ids and timestamps aside) against the reference read.

| read | scenes | mean score | reference mean | scenes with a different score | scenes with any different field |
|---|--:|--:|--:|--:|--:|
| sh30 c0a | 77 | 0.840939 | 0.840939 | 0 | 0 |
| sh30 c0b | 77 | 0.840939 | 0.840939 | 0 | 0 |
| sh30 c0c | 77 | 0.840939 | 0.840939 | 0 | 1 |
| sh30 c1a | 77 | 0.825787 | 0.825787 | 0 | 0 |
| sh30 c1b | 77 | 0.825787 | 0.825787 | 0 | 0 |
| PAI c000 | 17 | 0.058824 | 0.058824 | 0 | 0 |
| PAI c001 | 17 | 0.058824 | 0.058824 | 0 | 1 |
| PAI c002 | 19 | 0.879875 | 0.879875 | 0 | 0 |
| PAI c003 | 18 | 0.862971 | 0.862971 | 0 | 0 |

Every read except sh30 c0c and PAI c001 is bit-identical to its reference (c0a and c0b also agree with each other on all 1 368 driver plans); those two differ in one scene each.

- **sh30 c0c, scene `2021.05.25.17.54.41_veh-35_01654_01850-0a0d91d55b0757f8`:** its plans differ from c0a / c0b from drive step 7 on (1e-4 m in one pose; 11 of 1 368 calls),
  and the ego travels 90.46 instead of 90.22 m (`progress_rel` 0.8607 against 0.8585, `min_distance_to_obstacle_m` 4.28 against 4.16); score 1.0 in both.
  Configs are identical apart from ports. Whether packing causes this cannot be told: the pre-change record holds only two clean replicate reads (the 10-16 re-read of
  both lists, 154 / 154 scenes bit-identical), so 1 deviating scene in 385 batch scene-replicates against 0 in 154 is not testable.
- **PAI c001, scene `clipgt-2d10a08d-2f60-49fe-94e8-9d2797a289d2`:** the rollout failed at 13:23:36 with a renderer exception (`ego_hood.py:99 overlay_on_image`: "Expected
  all tensors to be on the same device, cuda:0 and cpu", during a burst of three `EgocarHood: resize cache miss`), recorded as `failure_reason` with score 0.0.
  The reference scene scores 0 as well (`left_corridor_laterally`), so the score field matches by coincidence; the list `paisub` has 16 zero scenes in 17 and
  cannot reveal a crash. The native summary still says `rollouts_failed: 0`; `failure_reason` is the field to read. The skipped 198-step rollout probably explains c001's
  shorter wall. This renderer race (third-party code, not VRAM) predates the change. Exception rate per PAI rollout, 4 days before the change and 2 h after, by jobs per card:

| jobs per card | before the change | after the change | pooled |
|--:|--:|--:|--:|
| 1-2 | 1 / 944 | 0 / 72 | 0.10 % (1 / 1 016) |
| 3 | 6 / 1 548 (0.39 %) | 2 / 1 279 (0.16 %) | 0.28 % (8 / 2 827) |
| 4 | 2 / 172 (1.2 %) | 1 / 125 (0.8 %) | 1.0 % (3 / 297, 95 % CI 0.3-2.9 %) |

  No before / after difference at a fixed density (3 per card: p 0.31; 4 per card: p 1.0, Fisher); the rate rises with density (4 against 3: p 0.08). Each crash is a scored
  zero, so moving PAI reads from 3 to 4 per card costs about 3 instead of 1 zero in a 281-scene read.

## 6. Not covered, and what is missing

- **5 sh30 reads on one card.** Needs a GPU run (not started): five held-chunk reads pinned to a card that holds no other pool job, `CL_HIST_KEY=alpasim-read4`,
  `--vram 20`, about 12-13 min of wall. Pass if wall <= 667.3 s x 1.15 = 767 s per read (about 713 s expected from the pre-change 5-per-card reads), no
  `oom` / `vram_stop` / `kill` event, scores equal to the c0 / c1 references above. The new rule books 5 sh30 reads at 5 x 1.1 x 13.35 = 73.4 GB, so it fits only on a card
  without PAI neighbours; a mixed card still tops out at 3 sh30 + 1 PAI (80.7 GB for the fourth).
- **4 PAI reads on one card at the default 22 GB declaration.** By arithmetic the new rule does not admit it (85.8 GB against 79.6 GB: the share is 19.5 GB per job
  at 3 per card, so 1.1 x share is about the 22 GB own booking), so at the default the change gives PAI nothing; a 4th read needs `VRAM=19` in `pai_box.sh` (measured
  at 4 per card: 17.0-19.1 GB per job, 70.5 GB per card at most). With the renderer race above, 4 PAI per card also costs more zeros.
- `rm-train` key (first half of the change): works. After the change `nx-train-RTB1AR-F-s1` and `nx-train-V2AC-L18-s0` carry `CL_HIST_KEY=rm-train` and are booked 8.8 GB
  of RAM instead of the declared 70 (measured 2.9 GB); the key's history holds 20 runs (RAM 2.9-6.5 GB, VRAM 17.1-24.8 GB).

Computed with throwaway scripts over the pool's `state.json`, `events.jsonl`, `usage.jsonl`, `boxwatch/20261017.tsv` and the runs' `results-summary.json` (not committed).
