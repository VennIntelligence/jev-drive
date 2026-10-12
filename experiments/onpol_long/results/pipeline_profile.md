# Pipeline profile: one on-policy round at 18 decisions (2026-10-12)

What one round costs, where the cards sit idle, and what `scripts/ol_fast.sh` changes. Sources: the GPU pool's `state.json` (job start / end,
card, peak VRAM, cores) and `usage.jsonl` (per card util and VRAM once a minute), each AlpaSim run's `native_summary.json` and `usage.jsonl`
(per service CPU, VRAM, RSS once a second), rollout `_complete` times, `top -H` on a live renderer. Scripts that produced the tables:
`tmp/pspeed/` on the Mac (not committed; profile.py, timeline.py, cardrate.py, runsum.py, ramp.py, compare.py, cmpdump.py).

## Round 1, Y family, as run (03:13-06:27)

Stage x time (Y family only: 6 rollout chunks of YR10m10-F-s0, geo / tok / strip, six trainings and six held reads: SKC, NONE and the S arm of
Amendment 3; the geo / tok pair ran twice after a tok failure):

| stage | jobs | first start | last end | stage wall (min) | job min (median / max) | GPU job-h | mean card util while on it | peak VRAM GB | cores (max) |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|
| roll | 6 | 03:13 | 03:42 | 29 | 15.2 / 28.9 | 1.90 | 66 % | 26.7 | 5.7 |
| geo (on_rows geo + ol_labels build) | 2 | 03:43 | 04:28 | 45 | 19.2 / 19.2 | CPU, 48 cores | - | - | 48.0 |
| tok | 1 | 04:28 | 04:31 | 3 | 3.1 | 0.05 | 83 % | 5.5 | 3.8 |
| train | 6 | 03:14 | 05:55 | 161 | 26.6 / 35.3 | 2.70 | 90 % | 19.2 | 1.6 |
| read (460 held, one job per checkpoint) | 6 | 03:42 | 06:27 | 165 | 31.9 / 54.0 | 3.73 | 25 % | 27.9 | 5.9 |
| report | 2 | 05:29 | 06:27 | - | 0.3 | - | - | - | - |

Whole lane (Y and P), 03:13-06:28, 185 min x 6 cards: a lane GPU job on the card 66 % of card-minutes, card util < 10 % in 40 %, mean card util
34 %. Serial gaps: 03:57-04:29 cards 1-3 idle while geo ran on the CPU (and again 04:08-04:28 after the tok failure); 04:40-04:56 cards 0 and 5
idle while four trainings ran; 05:01-05:27 four read jobs alone on cards 0-3 at 8-50 % util, cards 4-5 idle.

Clean serial path of one round, from these stage times: rollouts 15-29 + geo 19 + tok 3 + training 24-27 + read 32 + report 0.3 = **95-110 min**
(round 1 took 3 h 14 min with the tok failure and two amendments).

## Round 2 (Amendment 5, running; measured 11:24-TBD)

TBD

## Where the time goes

**One AlpaSim stack per job, and a stack uses little of a card.** A stack is runtime (2 workers) + MTGS renderer + controller + sh30 driver,
8 concurrent rollouts. Per service over a 460-scene read: renderer 3.2-3.4 cores, runtime 0.6, controller 0.5, driver 0.2; VRAM renderer 18-34 GB
peak (its scene cache, 9 scenes), driver 2.2 GB; host RSS runtime 33 GB, driver 9 GB. `top -H` on a live renderer: its 4 gRPC worker threads
(`max_workers=4` in AlpaSim's mtgs server) at ~79 % each, everything else idle, while the card shows 10-40 % util. The stack is bound by the
renderer's 4 Python threads (CPU / GIL side of each render call), not by the GPU. A stack alone on a card: 3.8 s per scene in steady state
(460-scene read), 4.4 s with the frame dump (rollouts).

**Stacks per card scale, trainers do not mix.** Round 1, 03:15-03:40, per card (rollouts completed per minute): two rollout stacks on a card
22-28 scenes / min at 6-46 % util, the same per stack as one alone (P2H10S chunks 4.7-5.4 s / scene next to a YR chunk, 4.7 alone); one stack
next to a training 5.8-7.2 / min (8.0-9.6 s / scene: a training at 90 % util halves a stack); two stacks next to a training 12.8-13.0. Two
trainings on one card: 35-41 min instead of 24-25. Scaling test on one card (`ol_fast.sh scale`, 1-4 stacks of 48 held scenes at once): TBD.

**Fixed cost per run.** Startup 50-60 s (driver compile warm-up 35 s), first rollout done after ~60 s more, then the tail and aggregation
(~40 s). A 48-scene run: 5.7-5.9 s / scene in steady state, 320-470 s in all; a 172-scene chunk 830-900 s; a 460-scene read 1 840 s.

**Rows: the K / C label build is 18 of geo's 19 minutes.** `on_rows.py geo` over 1 031 rollouts: 31 s on 48 workers; `ol_labels.py build`
(lane-centre raster per row, 319 jobs of 64 rows): 18 min on 48 cores. The S-only recipe (adopted, Amendment 3; round 2) never reads lane.npz /
corr.npz; only `ol_labels.py gate` (K0) does. Tokens 3 min on one card.

**Training is GPU-bound.** 7.2 it/s, 80-97 % card util, 17-22 GB, 1.3 cores; dev evals ~3 s each. 24 min for 10 000 steps alone on a card.

## What ol_fast.sh changes (results unchanged)

| change | where | effect |
|:--|:--|:--|
| rollouts in 12 chunks instead of 6 (`OL_NCH`) | `ol_fast.sh roll` | two stacks per card instead of one; 86 scenes per chunk, ~9 min instead of 15-29 |
| each held read in 6 chunks (`OL_RCH`), each chunk its own job right after its checkpoint's training | `ol_fast.sh read`, `report` joins the chunk dirs | ~8-9 min per checkpoint instead of 32 |
| label build beside the tokens: geo -> {tok, labels}; tok -> arm trainings; {tok, labels} -> gate -> strip; reads + gate -> report | `ol_fast.sh geo / labels / tok / gate` | 18 min off the critical path; gate K0 still decides the report |
| the arm's path (rollouts, rows, arm trainings, arm reads) one priority step above the controls | `submit-round` | the pool starts the critical path first |
| `CL_HIST_KEY=ol-fast-roll / ol-fast-read` | `submit-round` | VRAM booking by measured peaks across rounds (job names change per round) |

`ol_chain.sh` is untouched (round 2 runs on it). Job bodies call the same programs with the same arguments (run.sh sh30, c1_extract,
on_rows geo / tok, ol_labels build / gate, ol_chain.sh strip, on_train.py, ol_report.py); only the scene lists per job and the dependency graph
differ.

Before / after for one Y round (2 arm + 2 control trainings, 4 held reads; after = estimate from the measured per-stack rates and fixed costs,
to be replaced by the first ol_fast round):

| stage | before (ol_chain.sh, round 1) | after (ol_fast.sh, estimate) |
|:--|:--|:--|
| rollouts, 1 031 train scenes | 6 jobs, 15-29 min wall, 1.9 GPU job-h | 12 jobs, 2 per card, ~9 min wall, ~1.8 job-h (~0.9 card-h) |
| rows geo | 19 min (on the critical path) | ~1 min (labels 18 min beside tok) |
| tokens | 3 min | 3 min |
| training, 4 x 10 000 steps | 24-35 min each | unchanged, 24-28 min (arms first) |
| held reads, 4 x 460 | 32-54 min each, one job | 6 jobs each, ~8-9 min each, overlapped with the arm trainings for the controls |
| report | 0.3 min | 0.3 min |
| **critical path** | **95-110 min (clean); 3 h 14 min as run** | **~45-50 min** |
| GPU job-h | ~7.5 | ~8 (more startups; same scenes) |

## Identity checks

| check | runs | per-scene scores | plans (drive.jsonl, every decision) | frame dumps |
|:--|:--|:--|:--|:--|
| held read, 48 scenes in their own job (ol_fast readlist) vs round 1's 460-scene read of YS-L18-s1 | card 2 vs card 1 | 48 / 48 identical | 846 / 846 identical | - |
| rollout with the frame dump, 48 train scenes, rerun of round 1's chunk c2 of YR10m10-F-s0 | card 2 vs card 5 | 48 / 48 identical | 864 / 864 identical | 864 / 864 identical (all arrays but the stripped `cur`) |
| the same rollout smoke on card 0 | card 0 vs card 5 | 36 / 48 identical, max diff 0.012, mean 0.8917 vs 0.8918 | 259 / 864 | 253 / 864: JPEGs differ by 1-2 levels in 12-116 pixels from decision 0 |
| CAM_F0-only render request (rejected) | card 0 vs card 1 | 47 / 48, max diff 0.0036 | 832 / 846 | - |
| scaling test, 1-4 stacks on one card | card TBD | TBD | TBD | - |

So splitting the scene list across jobs and stacks does not change a rollout. **Card 0 does**: it is the one card with ECC enabled
(`nvidia-smi`: ecc.mode Enabled on card 0, Disabled on 1-5), and every run on it renders a few pixels differently from the other cards, which
moves plans from the first decision of some scenes. This is not new with ol_fast.sh: ol_chain.sh jobs land on card 0 as well (round 1: roll c0,
reads YNONE-s0/s1 and YS-L18-s0 ran there).

Rejected: asking the renderer for CAM_F0 only (the camera sh30 reads). MTGS renders every asset camera per request either way; renderer CPU per
scene did not fall (2.7 cores, 24 cpu-s / scene against 17), and its one differing scene was on card 0.

## Left on the table

- A third stack per card: GPU util stays < 50 % with two, but the renderer's VRAM peak (up to 34 GB, its 9-scene cache) and the pool booking
  at the peak keep it at two. Worth ~2 min per phase; would need a smaller renderer cache with fewer concurrent rollouts per stack (untested).
- Training (24 min x 4, now half the round): GPU-bound. DDP over two cards would roughly halve it but changes batches and numerics, so results
  change; not done. Removing the per-step `float()` syncs in on_train.py might give 5-10 %; not done (shared trainer, round 2 queued on it).
- Pipelining rollouts into geo per chunk: geo is now ~1 min, nothing left to gain.
