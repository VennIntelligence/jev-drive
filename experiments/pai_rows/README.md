# pai_rows: PAI human-state and frame-stretch training rows

status: live
decisions: none yet (rows only)
index: built: PAIH 135 793 human-state rows (851 scenes) + stretch 0.5x / 1.5x 223 148 + S 50 932; R1 and f = 1 identity pass

**Question.** Build training pools from every training-eligible NuRec scene on the box: PAI human-state imitation rows (the ego follows the
human log, label = the human's logged future), their frame-stretched variants (slots re-picked at 0.5x / 1.5x spacing, ego row scaled to
match, label unchanged; against the frame-channel speed copy of decision 404), and on-policy S rows (the d402 pipeline) on the scenes the
d402 pool lacks; plus a class-stratified half split for the data-scaling arms of round 2.

**Conclusion.** Built 2026-10-20 (no training, no decision entry yet). Every gate of the prereg passes: roll checks (every log-following
record returned the log), R1 trainer forward on the full pools (PAIH median 0.0098 m, 4.7 % over, wrong model 99.4 %; CPAIX 0.0096 m, 0.6 %,
99.4 %) and the f = 1 identity (ego, label, 8 tokens, teacher bit-equal on 1 605 probe rows, after supplement 1 fixed the encoder batch
order). Under supplement 2 (label dilated with the clip) at most 0.5 % of stretched rows need more than 6 m/s^2 at 0.5 / 1 / 4 s (plain rows 0.2 %);
the 4 s acceleration is 0.92-0.97 x f^2 the human's own. Under the first, unchanged label 76 % of stretched rows needed > 6 m/s^2 at 0.5 s.

| pool (`$DATA_DIR/runs/op_parity/cache/`) | scenes | rows | outside held | stop_nolead | turn_sharp | launch | bend | rest | half A (outside held) | GB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `on1_PAIH` (human state, fut = log future) | 851 | 135 793 | 114 453 | 34 892 | 34 379 | 12 933 | 21 542 | 32 047 | 68 059 (57 425) | 38 |
| `on1_PAIH05` (f = 0.5, dilated) | 851 | 123 879 | 104 415 | 31 840 | 31 355 | 11 799 | 19 652 | 29 233 | 62 081 (52 385) | 35 |
| `on1_PAIH15` (f = 1.5, dilated) | 851 | 99 269 | 83 671 | 25 527 | 25 131 | 9 450 | 15 757 | 23 404 | 49 747 (41 974) | 28 |
| `on1_CPAIX` (S rows, d402 pipeline, new scenes) | 288 | 50 932 | 42 170 | 0 | 0 | 0 | 14 614 | 36 318 | 25 821 (20 436) | 15 |

- Stretch drops (of the plain rows): f = 0.5 only the cold-start rows (11 914, 8.8 %); f = 1.5 also 16 951 without 6 s of log (12.5 %) and 7 659
  in the first 2.3 s (frames / history). Stretched rows carry `agent.ok` False (boxes on the undilated clock): use them as imitation rows.
- Missing: 2 of 853 H scenes (655dfa49, 8ca50753) lost their rollout to a renderer RPC error (score 0, no rows); a rerun of `pr.sh chain H`
  does not relist them (they sit in a chunk).
- The d402 pool holds 5 scenes now excluded via drives_val160 (results/scenes.md); a grown SIMP pool should drop them.
- Passes per row at k = 16 human PAI draws a step (outside held): J-half (half A, N steps) 16 N / 57 425 plain, 16 N / 151 784 with both
  stretch pools; J (all, 2N steps) 32 N / 114 453 and 32 N / 302 539; N = 1 000: 0.28 / 0.11 for both arms.
- Cost: GPU 12.0 job-h of rolls (H 8.95, S 3.05; a roll books a third of a card, about 4.0 card-h) + 1.45 job-h encode / forward / stretch
  -> about 5.5 card-h. Disk about 250 GB net (rows 58, cache 42, stretch parts 28, pools 116, probe pools 2.3); 741 GB free after.

**Next.** Round 2 (main session): joint arm J on PAIH + PAIH05 + PAIH15 (full, 2N steps), J-half on half A (N steps), same-step control.

**Read more.** [plans/2026-10-20-pai-rows-prereg.md](plans/2026-10-20-pai-rows-prereg.md) (scenes, recipe, gates, half split and passes per row, cost;
supplements 1 and 2); [results/H/rows.md](results/H/rows.md), [results/H/feas.md](results/H/feas.md), [results/S/rows.md](results/S/rows.md),
[results/probe/gate1.json](results/probe/gate1.json) (f = 1 identity), [results/probe/feas.md](results/probe/feas.md) (dilated label, probe),
[results/scenes.md](results/scenes.md) (the 853 scenes by class and half).

Pools (box, `$DATA_DIR/runs/op_parity/cache/`): `on1_PAIH` (names `on:paih:<scene>:<k>`), `on1_PAIH05` / `on1_PAIH15` (`on:paih05:..` /
`on:paih15:..`, tab `f`, `base` = the plain row's name, `speed` = f x v0), `on1_CPAIX` (`on:pai:..`, as on1_CPAI570). Probe pools `on1_PAIHT*`,
`on1_CPAIXT` (11 scenes each) are kept for the gates. Splits: `physical_ai_av/pai-rows@v1` (853), `physical_ai_av/pai-rows-half-a@v1` (428).
Run dir `$DATA_DIR/runs/pai_rows/` (jobs.txt, stages.txt). Append scenes: add them to scenes.tsv, `pr.sh chain H|S`.

## Second batch N (built 2026-10-21, no training, no decision entry)

The 64 scenes of the download queue v2 (all on the same drive as a PAI test281 scene, 144-591 s apart, median 391 s; [results/scenes_n.tsv](results/scenes_n.tsv),
[results/n_t281_partner.tsv](results/n_t281_partner.tsv)), built with H's recipe by `pr.sh chain N` ([plans/2026-10-21-pai-rows-n.md](plans/2026-10-21-pai-rows-n.md)), new
pools plus zero-copy merges (box `$DATA_DIR/runs/op_parity/cache/`). Every row of the new pools carries tab key `t281` (first 8 characters of the partner test281
scene id; "" on the older rows of the merged pools); scene-level partner id and gap are in `n_t281_partner.tsv`. Splits `physical_ai_av/pai-rows-n@v1` (64),
`physical_ai_av/pai-rows-x@v1` (853 + 64). `scenes.tsv` and the 853-scene pools are unchanged.

| pool | scenes | rows | outside held | stop_nolead | turn_sharp | launch | other (bend / rest) |
|---|--:|--:|--:|--:|--:|--:|--:|
| `on1_PAIHN` (f = 1) | 64 | 10 202 | 8 926 | 4 614 | 4 633 | 955 | 0 |
| `on1_PAIHN05` | 64 | 9 306 | 8 142 | 4 208 | 4 227 | 871 | 0 |
| `on1_PAIHN15` | 64 | 7 450 | 6 518 | 3 367 | 3 386 | 697 | 0 |
| `on1_PAIHX` = PAIH + PAIHN | 915 | 145 995 | 123 379 | 39 506 | 39 012 | 13 888 | 53 589 |
| `on1_PAIHX05` | 915 | 133 185 | 112 557 | 36 048 | 35 582 | 12 670 | 48 885 |
| `on1_PAIHX15` | 915 | 106 719 | 90 189 | 28 894 | 28 517 | 10 147 | 39 161 |
| `on1_JTPHX` = JTPH parts + PAIH* + PAIHN* (agent ok False on all PAI human parts) | | 555 505 | | | | | |

(PAIHX scene counts are 915 = 851 + 64.) Gates: roll checks 1 / 1 and 10 / 10 scored, every log-following record returned the log; R1 (trainer forward) PAIHN median 0.0103 m,
7.0 % over (line 10 %), wrong model 99.3 % over, 0 missing rows (probe PAIHNT 0.0083 m, 4.3 %); f = 1 identity bit-equal on the probe (1 597 / 1 751 rows) **and on
the full pool** (9 306 rows: ego, label, 8 tokens, teacher, slots 9 306 / 9 306); zero-copy prefix checks of PAIHX* against PAIH* and of JTPHX against JTPH pass
(every tab key, agent / road / route, feat, teacher, 4 000 sampled token rows). Under the dilated label at most 0.8 % of PAIHN15 rows need > 6 m/s^2 at 1 s
(15 % > 3). Results: [results/N/](results/N/) (rows.md, feas.md, r1_*.json, gate1.json, x/mcheck_*.json, x/xfix_JTPHX.json). Cost about 1 h wall at PRIO 6, under 1 card-h
(estimate; rolls ~0.7 job-h); disk about 72 GB (new pools 12, merged pools PAIHX* 14, JTPHX 34 because the merge rewrites agent / road / drivable, rows, cache and stretch parts 11).
The first merge job stopped at the prefix check because of a wrong pool-name argument in `pr.sh` (fixed, rerun passed; the lane's own half-written PAIHX was overwritten).

## Download queue v2 (downloaded 2026-10-21 02:51, built as batch N above)

[`results/download_queue_v2.csv`](results/download_queue_v2.csv): the NuRec scenes of `experiments/pai_av/results/nurec_select.csv` that are
not on disk and have `cls` in stop_nolead / turn_sharp / launch, rank-ordered, with the select list's columns (so `nurec_fetch.py --list`
reads it) plus `queue_pos` and `queue_cum_gb`. **64 scenes, 111.15 GB** (stop_nolead 29, turn_sharp 29, launch 6; largest 2.04 GB; ranks
1177-1546).

How it was cut: 1 554 listed, 1 301 on disk (the box's `all-usdzs` equals `pai_rows/results/usdz_20261020.txt`, checked 2026-10-21), 253
not on disk, 174 of the three classes. Removed 110 (195.1 GB): scenes in `pai_fix/scenes/train_exclude.csv` (68) or linked to val160 by
`pai_fix/scenes/drives_val160.csv` (89, union 110); nothing of `exam_scenes.txt`, `nurec_train_excluded.csv` or the val160 / test281 scenes of
`pai_av/results/exam_nurec.csv` was in the 174 (those are all on disk). All 64 kept scenes are select-list tranche 3 (same drive as an exam
scene, 145-591 s away, median 392 s; tranches 1 and 2 are fully on disk). Nine `.part` files from the 10-20 fetch sit in `_dl/` (none of them
in this queue).

Command (box repo root, not run; `--weight 0:1` gives the box every rank, `--floor-gb 600` is the box default and needs about 711 GB free to
finish; free space was 558 GB before the 2026-10-21 zero-copy swap, [swap_20261021.md](../cl_infra/results/zero-copy/swap_20261021.md)):

    ssh autodl 'cd ~/data/jev-drive && scripts/nurec_fetch.sh box --list experiments/pai_rows/results/download_queue_v2.csv --weight 0:1 --floor-gb 600'

<!-- files:begin -->
## Files

- `2026-10-20-pai-rows-prereg.md` (plans): PAI-ROWS 预登记：盘上全部可训练 NuRec 场景的 PAI …
- `pr.sh` (scripts): box, repo root
- `pr.py` (scripts): PAI human-state imitation rows on every …
- `scenes.md` (results): PAI-ROWS scene table
- `scenes.tsv` (results): 
- `onp_cache.py` (experiments/onpol_pai/scripts): the PAI row files -> what the trainer …
- `rf_pai_pool.py` (experiments/route_follow/scripts): Route-night section 9 step 5
- `onp_log_driver.py` (experiments/onpol_pai/lib): the simulated ego is steered along the …

[results/](results/) 22 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
