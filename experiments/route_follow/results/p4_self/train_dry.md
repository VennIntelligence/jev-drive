# RSGSELF-L18 training arm: dry run on the dev pool (2026-10-14)

Arm (turning school phase 2): **RSGSELF-L18-s{0,1}** = RSG-L18's recipe (init RB0-F-s0, gentle schedule: 2 500 steps, lr x 0.1, `--on-k 13
--on-split alpasim_nuplan/onpol-train`, RSG-L18's own on-policy pool `on1_L18RB0` as `self` rows) plus the self-branch demonstration rows of
[rf_selfpool.py](../../scripts/rf_selfpool.py) as `both` rows in `--on-mode mixed`. The SELF rows get 30 % of the pool draws, with a per-scene cap
inside the SELF rows. The control is RSG-L18-s{0,1}, using its stored reads. The reads are the same as [rf_tschool.sh](../../scripts/rf_tschool.sh):
held 460 @18 and the public 1491 @10 guard.

Chain: [rf_self_train.sh](../../scripts/rf_self_train.sh) (`merge`, `submit`, `report`, `dry`, `dry-post`, `clean-dry`). Lateral guard and ladder
table: [rf_selfread.py](../../scripts/rf_selfread.py). Draw cap: `rm_train.py --on-cap-scene C --on-cap-in KEY=VALUE`
([rm_train.py](../../../route_mem/scripts/rm_train.py)).

**This was a dry run only.** The dev pool `on1_SELFdev` (666 rows, 6 scenes) holds hard_exam scenes, so the tag was RSGSELF-DRY-s0 and training
ran 60 steps. The checkpoint (`$DATA_DIR/runs/op_parity/runs/RSGSELF-DRY-s0`) and the merged dev pool `on1_RSGSELFdev` (cache, @warp, feat) were
**deleted** by `dry-post`, and the deletion was checked afterwards. Nothing from this run is read on the PAI exam.

## Per-scene draw cap

`--on-weight log=<scene>:w` could express a cap, but only with hand-computed per-scene factors. These would have to be recomputed for every pool,
and they would break the W of the 30 % share. The new flag does the following:

- `--on-cap-scene C --on-cap-in pool=s` computes the weights after `--on-weight`. Among the rows matching `--on-cap-in`, it rescales per scene (tab
  `log`). It uses water-filling: scenes over C are set to C, the others are scaled up in proportion, and this repeats until no scene is over C.
- The subset's total draw mass is unchanged, so W (computed for the 30 % share) stays exact.
- An infeasible cap (scenes x C < 1) is an error.
- Without the flag, `pw` stays `None` and the draw is the old uniform `orng.choice`, bit for bit. RSG-L18's recipe is untouched.
- `rm_train.py capcheck` is the known-answer test. Synthetic pool: 100 `s` rows (3 scenes, 70 / 20 / 10 %) and 30 other rows, capped at 0.4.
  - uniform weights: 0.4 / 0.4 / 0.2 (max error 1.1e-16).
  - weights from `topo=x:3,pool=s:2.5` (unequal within a scene): the result equals an independent bisection solution (3.9e-16).
  - In both cases the subset mass is kept, the other rows are untouched, and each scene is scaled by one factor.
  - An infeasible cap is refused.
  - Result: **pass** on the Mac and on the box; the test exits 2 on any miss.
- `rm_train.py draws --on POOL ...` prints the shares the trainer will use. It is the same code path as training, with no GPU.

## Dry run numbers (box, commit f98892f2)

| item | value |
|:--|:--|
| merge | `rf_pait.py merge --out RSGSELFdev --pools on1_L18RB0:r:self on1_SELFdev:s:both` -> 18 095 rows (17 429 `self` + 666 `both`); merge + W + draws 18 s |
| W | `pool=s:11.2156` = 0.3 / 0.7 x 17 429 / 666. Drawn pool share: s 0.300, r 0.700 (trainer log `draw shares`) |
| cap (dry: 0.25, 6 scenes) | eaba3f6a 0.581 -> **0.250**; 2f227f2c 0.122 -> 0.218; 94c1d722 0.089 -> 0.159; fd01e9ab 0.087 -> 0.156; f9e361aa 0.071 -> 0.126; a174faed 0.051 -> 0.091 |
| mixcheck (`rm_train.py mixcheck` on on1_RSGSELFdev) | **pass**: max abs diff of loss and gradient, mixed vs pure mode: self 0.0, both 0.0 (`self` rows keep RSG-L18's loss exactly) |
| training, 60 steps | 3.24 it/s over all 60 steps; 3.3-3.4 it/s from step 25 (tb `throughput/steps_per_s`); 18.3 GB reserved (pool vram peak 18.9); dev ADE 0.5276 -> 0.5314; job 1.1 min |
| held 5 (`rm_chain.sh readlist`, 5 held scenes, one per log) | 5 / 5 rollouts, 0 failed, stack 153 s, job 2.9 min |
| report path on the 5 scenes | held.py, ol_confirm (paired vs RSG-L18-s0 on the common 5), `rf_selfread.py lat`, `table`: all rc 0. Pipeline check only; the numbers mean nothing on 5 scenes |
| deletion | `clean-dry`: checkpoint dir, on1_RSGSELFdev, on1_RSGSELFdev@warp, feat/on1_RSGSELFdev.npz and rely json gone (checked: 0 left) |

The mixed mode trains at RSGmixPAIT-L18's rate: 3.28 it/s and 18.3 GB, with 2 500-step jobs of 13.3-14.0 min. The `both`-mode arms ran at
6.9 it/s (6.7 min).

## Lateral guard (anti lateral ratchet)

`rf_selfread.py lat` reads the held 460 @18 rollouts.

- **Decisions kept:** each decision up to the first zero event, with fed speed >= 1 m/s, whose anchor class (rf_offroad.rclass on the rebuilt
  AlpaSim route plus the map's intersection test) is straight or curve. Junction, approach (junction within 30 m) and jog decisions are dropped.
- **dlat2:** the signed lateral offset of the served plan's 2.0 s pose from the route, minus the anchor's own offset. This is the plan's
  sideways move relative to the lane over 2 s.
- **Statistic:** per-scene mean |dlat2|, then the mean over scenes. Seed means use the per-scene mean of both seeds. The pairs are vs RSG-L18,
  with log-clustered CIs.
- **Line:** RSGSELF-L18 minus RSG-L18 <= +0.02 m (point estimate).
- **Other figures reported:** signed dlat2 (a one-sided drift) and |lateral| of the 2 s pose.
- **Exam entry speed** is read later on Tokyo and is listed as pending in the table.

## Estimate for the full arm (box: 3 cards; reads pinned to cards 1-2 as in rf_tschool / rm_chain)

| stage | time |
|:--|:--|
| merge + W + draws (CPU, on the box host) | < 1 min |
| mixcheck | ~1 min |
| training, 2 seeds in parallel | ~14 min (2 500 / 3.3 it/s = 12.6 min + setup, 5 dev evals, rely; RSGmixPAIT jobs 13.3-14.0 min) |
| reads | 12 held chunks of ~10 min + 20 guard chunks of ~14 min (RSGmixPAIT and RSGPAIT stored jobs) = ~400 job-min. RSGmixPAIT's 32 reads took 26 min on 5 cards; on cards 1-2 at ~6-7 concurrent: **~60-70 min** |
| report (held.py, 2 x ol_confirm, lat on 4 x 460 rollouts at 48 workers, table) | ~5 min (5 scenes x 2 labels: 8 s) |
| **total** | **~1.5 h** from submit, if the cards are free (another agent's PAI read shares the pool) |

## Commands for the real pool `on1_SELF`

Pool on the box is `$DATA_DIR/runs/op_parity/cache/on1_SELF` plus `@warp` and `runs/route_mem/feat/on1_SELF.npz`. These come from the
rf_selfpool `build ... --tag on1_SELF` of turning-school-rows.md. If the pool is built on Tokyo, rsync it. Tokyo's `autodl` ssh alias still has
the old port (25199) and is refused, so use `-e "ssh -o HostName=116.172.94.204 -p 53804"` as this dry run did.

```bash
ssh autodl
cd ~/data/jev-drive && git pull
bash experiments/route_follow/scripts/rf_self_train.sh merge && bash experiments/route_follow/scripts/rf_self_train.sh submit
```

- **Defaults:** `SELF=on1_SELF`, `OUT=RSGSELF` (merged pool `on1_RSGSELF`), `SHARE=0.3`, `CAP=0.10` (no scene above 10 % of the SELF draws,
  which needs >= 10 scenes in the pool).
  - `merge` prints W, the per-scene shares before and after the cap, and the pool shares (`$O/W_RSGSELF.txt`, `$O/draws_RSGSELF.json`).
  - With fewer than 10 scenes, `merge` stops at `draws` with the infeasibility message. Set `CAP=...` (e.g. 0.25 on 6 scenes) before both
    calls. CAP must be the same for `merge` and `submit`.
- **Refusals:** `merge` refuses a pool that carries `EXAM_SCENES_INSIDE` (only `*dev` pools may). `submit` refuses if the stored RSG-L18 reads
  are missing.
- **`submit` chain:** `sf-mixcheck` -> `sf-tr-RSGSELF-L18-s{0,1}` (`--after` mixcheck) -> `rm_chain.sh submit-read` (6 held chunks + cover) and
  10 guard chunks per seed -> `sf-report`.
  - Job ids: `$DATA_DIR/runs/route_follow/selftrain/jobs.txt`; log dirs: `selftrain/pool/<job>`.
  - Report outputs: `selftrain/summary.md` (ladder table), `held18/`, `guard/`, `lat/`, `held_raw.txt`.
