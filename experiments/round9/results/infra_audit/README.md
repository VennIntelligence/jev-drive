# Round 9 infra audit: can the box run a continuous actor-learner loop?

2026-10-22. Read-only audit of the draft in [plans/2026-10-22-round9-draft.md](../../plans/2026-10-22-round9-draft.md) section 1
(P = 3 learners, actor fan-out under the submission serve, replay buffer, PBT controller). No jobs were submitted for it.
Numbers come from the pool's `events.jsonl` / `usage.jsonl` (`$DATA_DIR/runs/pool/`), the lanes' `jobs.txt` / `stages.txt`
(`$DATA_DIR/runs/round6/dag`, `round6/dag2`, `nuplan_dag`) and the job logs. The scratch analysis scripts are in `tmp/r9audit/`
(gitignored). Box: 5 x RTX 6000D (85.6 GB), 110-core CPU quota (overcommit 1.4), 458 GB memory cgroup, 1.3 TB disk free.

**Short answer.** About 75 % of a DAgger iteration's GPU time goes to AlpaSim rollouts and reads. The trainer is cheap:
1 000 steps take 2.5 min on a card of its own. The loop can be built. Today two things block it: the serial build that runs
after the rollouts (about 47 min in R6DAG), and a trainer that can neither resume its optimizer state nor take a new pool
without a restart. The minimum design runs each learner round as a short "segment" job that restarts from the last
snapshot and its optimizer state. Rollouts of round r run while the learners train round r + 1 on data that is one round old.
Every post-roll stage runs per chunk, and the buffer grows by an append instead of a merge. A round then takes about
20-25 min. The must-have engineering is about 22-31 agent-hours.

## 1. One DAgger round today (R6DAG / DAG2), measured

Lane scripts: [dag.sh](../../../round6/scripts/dag.sh) (iteration 1, one student) and [dag2.sh](../../../round6/scripts/dag2.sh)
(iteration 2: two students run through `dag.sh` with `DAG_IT2=1`, then a join). Every stage is a pool job.

| Stage | Script (evidence) | Job wall (median / max) | Resources (measured peak) | Bound | Card-h, iteration 1 |
|---|---|---|---|---|---|
| roll (30 chunks x 25 scenes, 718 scenes) | `pai_box.sh run` -> native AlpaSim stack, `CONC=4 ALPASIM_FAST=1`, ACC + GAP, `PAI_DUMP` (dag.sh:31-37) | 16.6 / 25.9 min a chunk; 8.05 stack-h; wave 12:01-12:37 (36 min) for 712 scenes | 22 GB booked, 20 GB VRAM, 3.6 cores, 15.7 GB RSS; 3 stacks a card | GPU (renderer); card util only 49-60 % | 2.7 |
| rows (per chunk) | `onp_rows.py rows` + `cold` (dag.sh:128-142) | 1.9 / 2.7 min | 8 cores, 5.2 GB | CPU | - |
| teach (per chunk) | `dag.py teach` map teacher (dag.sh:143) | < 0.5 min | 8 workers | CPU | - |
| clean, link, chk1 | dag.sh:144-148 | < 0.5 min | 1 core | - | - |
| **cache** (one join) | `onp_cache.py build`: frozen Cinque encoder over the rendered frame pairs (dag.sh:149) | **17.5 min** (DAG2: 20.2 each) | 5.2 GB VRAM, 1.8 cores; 354 176 pairs at 355 pairs/s; 10.8 GiB tokens | GPU encoder / I/O | 0.29 job-h (small) |
| pool | `rf_pai_pool.py --fut poses --src dagger` (dag.sh:150-153) | 7.0 min | 1 core | CPU, single-threaded | - |
| label | `dag.py reteach` + `label` (dag.sh:179) | 3.4 min (first try failed the T4 gate, then about 10 min lost to the stop and resubmit) | 16 cores | CPU | - |
| fwd + r1in | `rf_recov.py fwd`, input-path gate (dag.sh:180-181) | 6.1 min | 6.5 GB VRAM | GPU-light | 0.10 job-h |
| **merge** | `rf_pait.py merge`: rewrites tab / agent / road / drivable / route (rf_pait.py:504-561; tokens are zero-copy at :561) | **7.5 min** | 1 core, **67-72 GB RSS**, writes **35 GB** (`on1_JTPHD`: road 16 GB, drivable 16 GB, agent 4 GB) | CPU / disk | - |
| weights | `dag.py weights` (dag.sh:187-189) | < 0.5 min | - | - | - |
| smoke + identity (T3) | 60-step trainings + `df.py bitcmp` (dag.sh:190-203) | 2.6-3.8 min each, in parallel | 30-34 GB VRAM | GPU | 0.06 |
| train x 2 seeds | `rm_train.py`, 9 500 steps, `--save-at 2500,5000,7500` (dag.sh:231) | **41.0 min** (DAG2: 33.7; nuPlan `nd.sh`: 25.8) | 30-35.5 GB VRAM, 1.4-1.6 cores, 12-13 GB RSS | GPU, partly host-bound | 1.37 job-h |
| PAI 281 read x 2 | 8 chunks of about 35 scenes each + `pai_fix.sh tax` | 23 min a chunk; tax 1.9 min (16 cores) | as roll | GPU | 2.1 |
| r281 + PCS ruler | 20 bootstrap jobs | 3 min each, 5 min span | 4 cores each | CPU | - |

Wall clock of a whole iteration:

- **R6DAG:** chain 11:54, built 13:45 (1 h 51 min), training 13:46-14:27, reads and ruler until 15:33. Total 3 h 39 min.
- **DAG2:** rolls 15:40-16:56 (1 436 scenes, two students); build until 17:51; training 17:54-18:28; reads until 19:45. Total 4 h 05 min.
- **nuPlan `nd.sh` (10-22):** rolls start 10:15, training ends 11:23. Total 68 min.

nd.sh runs every post-roll stage per chunk (geo / tok / teach / label / route / strip / anchor, each under 1 min). Its single
join is `append`: 3.3 min, 27 GB RSS (nd.sh:255, nd_teach.py `cmd_append`). That is the pattern to copy.

**The critical path after the last R6DAG roll chunk** (12:37) is 69 min of serial joins. About 47 min of it is compute:
cache 17.5, pool 7, label 3.4, fwd 6.1, merge 7.5, gates 4, trainer setup 1.5. All of it is CPU or a light GPU job. The
GPU-heavy work is only the rolls, the training and the reads.

**Token caching.** New rows are not usable without an encode, but only the new frames are encoded:

- The frozen encoder runs once per new rendered frame pair (1.4 s a scene, 5 GB VRAM).
- Its per-scene parts can be resumed (`onp_cache.py` docstring, `parts/<scene>.{npz,tok.npy}`), so it can run per chunk:
  about 35 s for 25 scenes.
- The merge never re-encodes. Tokens are linked through `front.parts.json` (rf_pait.py:561, `jevdrive/tokens.py`).
- What the merge does rewrite is every per-row label file of the whole pool (about 36 GB) on every merge.
- The pool's `front.npy` stores 8 token slots per row: 29 GB for 111 k rows (`on1_PAID@warp`), on top of 39 MB of row file
  per scene. That comes to about 80-90 MB of disk per rolled scene.

## 2. Trainer (`experiments/route_mem/scripts/rm_train.py`)

| Capability | Today | Evidence |
|---|---|---|
| periodic snapshots | **yes**: `--save-at` steps write a new tag `<tag>x<step>` (weights, adapter, route_mem sidecar) in seconds | rm_train.py:1524-1533; log: snapshot at 13:57:05, dev eval at 13:57:11 |
| start from a snapshot | weights only: `--init <snapshot> --allow-init` (the sidecar makes it load with the branch) | rm_train.py:17-20, 805 |
| resume with optimizer state | **no**: only `model.state()` + `adapter.pt` are saved; no AdamW / GradScaler / RNG / step | rm_train.py:1529, 1544; optimizer built fresh at :842 |
| LR schedule across segments | cosine over `cfg.steps` from step 0, per-group warm-up (route branch 100 steps). `--lr-floor 1.0` gives a constant LR today; pass `--rm-warmup 1` on resume | rm_train.py:1325-1330 |
| change the mixing weights mid-run | **no**: `--on-wfile` is read once and `pw` normalised before the loop; `draw()` closes over it; `Prefetch` draws ahead | rm_train.py:857-866, 993-1002, 1315 |
| read a growing pool | **no**: `Store` is built once from a fixed tuple of data dirs plus exactly one `--on` pool; agent / road / drivable hinges are loaded onto the GPU by row name at setup | on_train.py:217, 291, 303 |
| new tag per run | enforced (`a tag is never overwritten`) | rm_train.py:805 |
| speed (R6DAG s0 / s1, DAG2 s0 / s1, per 100-step window vs pool GPU jobs on the card) | learner alone: **6.7-6.8 it/s** at 56-59 % card util; with 1 more job: 3.1-6.4; with 2 more: **3.5-3.9 it/s** | log.txt + usage.jsonl |
| setup | about 1.5 min (launch 13:46:36, run dir 13:48:08, step 0 13:48:17) | dag-train-s0 log |

**Smallest change: segment learners, not a resident learner.**

- Add `--save-state` and `--resume-state <tag>`. Save the AdamW state, GradScaler, every numpy Generator (`rng`, `hrng`,
  `orng`, `mrng`, `erng`, and the optional streams) and the torch / CUDA RNG, next to each snapshot. Load them after setup.
- Run with a constant LR (`--lr-floor 1.0`, or a `--step0` offset into the cosine) and `--warmup 1 --rm-warmup 1` on resume.
- Each round is then a fresh `rm_train` job of N steps on the newest pool and weight file. Everything (pool, `--on-wfile`,
  theta) is re-read at setup for 1.5 min. That is about 25 % of a 1 000-step segment on a card of its own and about 20 % when
  the card is shared.
- Gate: 2 x 500 resumed steps bit-equal to 1 000 straight (`df.py bitcmp`, the T3 identity already in dag.sh:199-203).
- Reloading the weight file in the process every N steps (a mutable `pw` read by `draw`, the prefetch queue flushed) is about
  2 h, but only pays off for segments well above 1 000 steps.
- Growing the pool inside the process means rebuilding `Store`, the hinges and the route features: in effect the setup. Not
  worth it at 1.5 min.

## 3. Pool (`jevdrive/cl/pool.py`)

| Question | Answer |
|---|---|
| a learner beside actor stacks | Fits. A learner books about 34 GB (history cap; 30-35 GB peak) and 2 PAI stacks book 2 x 22 GB: 78 + 4 GB headroom < 85.6 GB. `train_per_card=2` (pool.py:186). A second learner plus a stack does not fit (90 GB). Placement picks the least-loaded card (pool.py:795), so learners spread out. |
| long-lived learner | Possible but against the rules (section 4, flags). |
| chain rounds with `--after` | Yes. A pool job may itself submit the next stages: dag.sh `rest` (dag.sh:105-127) and `chain` already do this. The controller of round r can therefore be an in-queue CPU job that submits round r + 2's jobs (see the timeline in section 4). |
| keep 5 cards busy | Measured in the R6DAG and DAG2 roll waves: 3.0 GPU jobs on every card for the whole wave, card util 49-60 %, CPU 72-78 measured cores of the 110 quota. |
| PAI rollout throughput (Harmonizer off, ACC + GAP, FAST) | 25 scenes in 16.6 min a stack, about **90 scenes per stack-hour**. With 3 stacks a card: 270 per card-hour in theory; measured over whole waves **237** (R6DAG) and **227** (DAG2), including start and tail. 281 read chunks: 35 scenes in 23 min, the same rate. |
| concurrency | 3 stacks a card (booked 22 GB, peak 20-21 GB). Inside a stack, `CONC=4` rollouts at once. |
| startup per job | AlpaSim native services are ready after 20.6 s of a 991 s chunk (2 %), from `native_summary.json`. The dispatcher starts at most 4 jobs per 20 s round (pool.py:186-187), so 15 stacks take at least 80 s to start. A learner's setup takes 1.5 min. |
| nuPlan rollouts (for the 1 031 onpol-train scenes) | 43 scenes in 7.9 min a chunk; 32 GB booked (2 a card); 1 031 scenes in a 28 min wave, about **440 scenes per card-hour** (`nd-roll`, 10-22). |

Per-scene GPU cost, about 13 card-seconds on PAI, against 150 card-seconds for 1 000 learner steps: one learner segment
costs about as much as 11 PAI scenes. The loop is bound by its actors.

## 4. Minimum design for a 20-30 min round with P = 3

**Per round r** (every job goes through the pool with `--after`; there are no hand-picked cards):

1. **Learner segments `L[p,r]`** (p = 1..3, `--train`, about 34 GB):
   - `rm_train` with `--resume-state L[p,r-1]` (or the PBT donor's state), pool `B[r-1]`, weight file `W[p,r-1]` built
     from theta_p, 1 000 steps.
   - Saves a snapshot and its state.
   - Wall about 6 min: 1.5 min setup + 2.5-4.5 min of training (depending on how many stacks share the card) + saving.
   - Depends on `L[p,r-1]`, `B[r-1]` and `C[r-1]`.
2. **Actors `A[r]`** (fan-out of stack jobs, `--after L[p,r]`):
   - About 18-20 scenes a chunk (12-14 min).
   - Practice slice: 3 snapshots x 80 scenes (Harmonizer off, fixed scenes, registered as a split in `jevdrive.data.splits`).
   - Training scenes: about 60 PLR-picked scenes per snapshot from the 718 (281 / val160 never included).
3. **Per-chunk tail** (CPU, plus a 5 GB encode), started on each chunk as it finishes:
   - `rows` (8 cores, 1.9 min), `teach`, `cache` for the chunk (about 30 s), pool part (`rf_pai_pool` on the chunk's
     scenes), `label` part.
   - Then the frame dumps and row frames are removed.
   - Practice chunks get only their facts (`pai_fix tax`).
4. **Buffer `B[r]`** (CPU):
   - Append the round's labelled parts in the nd_teach `append` style.
   - Best case: the hinge files are passed as name-aligned lists, so the 36 GB rasters are never re-concatenated
     (`Hinge` already accepts a list, drivable_hinge.py:71; `AgentHinge` maps by name, agent_hinge.py:174). Estimated
     1-2 min.
   - Fallback: a full append (3.3-7.5 min, 27-72 GB RAM), keeping only the last 2 pools on disk.
   - Then `weights` per learner (k_dag, S share, PAI human share from theta).
5. **Controller `C[r]`** (CPU, 2 min):
   - Paired fitness on the practice facts (mean minus a penalty for new breaks on NTV0 full-score scenes; LAUNCH guard).
   - PBT copy and perturb: the bottom member resumes from the top member's state.
   - PLR update.
   - Submits round r + 2's jobs.

**Staleness of one round.** `L[p,r+1]` trains on `B[r-1]` and `C[r-1]` while `A[r]` rolls, so learners and actors overlap.
The round period is set by the actors: max(about 6 min of learner, about 14 min of rollout + about 3 min chunk tail + about
3 min buffer and controller), about **20-25 min**. Without the staleness, rounds run serially in about 30-35 min and the
cards empty between waves.

**Card budget per 25 min round** (5 cards x 25 min = 125 card-min, 15 stack slots):

- The 3 learners take 3 of the 15 slots for about 6 min each.
- That leaves about 357 stack-min, about 535 scene-rollouts in theory, about 430 after start-up and tail losses.
- Allocation:
  - Practice: 240 (56 %).
  - Training data: 180. At about 155 drawable rows a scene (110 981 / 718), that is about 28 k new rows a round. The
    learners draw about 40 k pool rows a segment each, about 1-4 draws a row, which stays inside the 4.6-draw ceiling of
    nd.sh (nd.sh:257).
- Reading the practice slice every second round, or 60 scenes instead of 80, frees 25-40 % more training rollouts. The
  draft's risk 1 (practice-slice noise) decides that, not the infrastructure.
- Expected GPU utilisation:
  - Booked occupancy 85-90 %. The gaps come from the start limit of 4 jobs per 20 s and from chunk tails.
  - SM utilisation 50-60 %, the measured ceiling of the AlpaSim read path (cards at 49-60 % when packed;
    research/alpasim-read-throughput.md: persistent stacks +4..+9 %, inside noise).
  - "GPUs never idle" is reachable in booking terms. SM utilisation is bounded by the renderer, not by the loop.
- CPU concurrently: 15 stacks use about 54 cores; chunk tails average about 6 cores (180 training scenes x 8 cores x
  1.9 min / 25 scenes, per 25 min); the buffer and controller are short. That stays under the 110-core quota.

**Bottlenecks, in order:**

1. **Rollout throughput.** About 90 scenes per stack-hour, renderer-bound. Concretely, about 430 rollouts a round, more than
   half of them practice reads.
2. **The serial post-roll build.** About 47 min in R6DAG, mostly the one-shot cache, pool and merge. It must become per
   chunk plus append, otherwise no round gets under 60 min.
3. **Host RAM.** 15 stacks x about 15 GB + 3 learners x 13-23 GB + a 72 GB merge comes to about 370 GB, at the pool's anon
   line (about 85 % of the 458 GB cgroup). Use the append, not `rf_pait merge`.
4. **Disk.** About 80-90 MB a rolled training scene (rows plus 8-slot pool tokens), about 15 GB a round, plus 35 GB per
   merged pool if the merge is kept. 1.3 TB lasts about 25 rounds with merges and about 80 without. Prune row frames after
   the cache and keep 2 pools.
5. **Learner co-tenancy.** 6.8 it/s alone, 3.5-3.9 it/s next to two stacks. That is fine for 1 000-step segments, but a
   learner never gets a card of its own while stacks are queued.

**Engineering cost.**

Must-have, about 22-31 agent-hours (about 1.5 days with M1, M2 + M3 and M5 run in parallel by three agents):

| # | Item | Hours |
|---|---|---|
| M1 | `rm_train` `--save-state` / `--resume-state` (AdamW, scaler, all RNG streams, step), constant-LR resume, identity gate 2 x 500 = 1 000 | 4-6 |
| M2 | Per-chunk PAI tail as jobs: rows -> teach -> cache part -> pool part -> label part -> LABELLED (nd.sh pattern; `rf_pai_pool` and `dag.py label` take a scene subset) | 5-7 |
| M3 | Buffer append for PAI parts (port `nd_teach append`); name-aligned hinge file lists or full append + pruning | 3-5 |
| M4 | Weight file per learner from theta (generalise `dag.py weights`) | 2 |
| M5 | Lane script `r9.sh`: in-queue round submitter, PLR scene sampler, practice split, fitness from pai_fix facts, PBT, jobs.txt / stages.txt, one `wait` stage | 6-8 |
| M6 | Staged pilot (1 learner x 1 round x 2 chunks, then P = 3 x 2 rounds), about 1.5 card-h | 2-3 |

Nice-to-have:

| # | Item | Hours | Note |
|---|---|---|---|
| N1 | In-process `--on-wfile` reload every N steps | 2 | |
| N2 | Activation x learnability x age weights recorded by the learner on its drawn rows | 4-6 | Replaces a separate forward job of about 0.2 card-h a pass |
| N3 | Adaptive S rows per batch from `--diag` grad norms | 3-4 | Keep the batch size fixed: `torch.compile(dynamic=False)`, rm_train.py:1314 |
| N4 | `road_pai` rasters for the new S rows per chunk | 3 | `road_pai.py build` took 3 min on 36 cores for the PAIH build |
| N5 | nuPlan actors on the 1 031 onpol-train scenes | 4-6 | nd.sh pipeline, about 440 scenes per card-hour |
| N6 | Pool tokens as an index into the cache instead of 8-slot copies | 4 | 29 GB -> about 11 GB per 718 scenes |
| N7 | Persistent AlpaSim stacks | - | Measured +4..+9 %, not worth it |

Drop the per-round fwd / r1in / smoke / identity gates once round 0 has passed them: about 10 min saved a round.

**Pool-rule flags:**

- **A long-lived learner that loops over rounds and polls for the buffer violates the rules.** It is "one job that loops"
  and holds about 34 GB while waiting (runbook "Pipelines"), and the polling is banned (global CLAUDE.md). Use segment
  jobs chained with `--after`.
- **The controller must not be a tmux daemon.** It runs as an in-queue CPU job that submits the next rounds (the dag.sh
  `rest` precedent). Agents wait with one `run_in_background` waiter on a DONE / VERDICT marker.
- **The draft's "one learner per card" conflicts with "nobody picks a card".** Spreading comes from the pool's
  least-loaded placement (pool.py:795), not from pinning. A per-lane spread option for learners would be a pool change for
  the user to decide, not part of this lane.
- **Thread or env tuning for learners or stacks goes only into `jevdrive/cl/profiles.py`.**
- **Never put 281 or val160 in any round.** The practice slice comes from `jevdrive.data.splits` (`run.use_split`).
