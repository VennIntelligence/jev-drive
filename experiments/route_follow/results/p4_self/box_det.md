# P4 on the GPU box: determinism against Tokyo, box-vs-box, packing throughput, disk (2026-10-14)

Read this before you set the box-side gates or the per-card packing of the P4 fanout
([prereg](../../plans/2026-10-14-p4-self-branch-prereg.md), Amendments 1-2). Data: [box_det.json](box_det.json). Probe:
`scripts/rf_p4det.sh` (launch) and `scripts/rf_p4det.py` (`det`, `pack`). Box run root: `$DATA_DIR/runs/route_follow/p4/boxdet`
(rollout.asl and dumps deleted after the read; logs, metrics and controller CSVs kept).

**Headline.**
- **Box vs Tokyo: never bit-identical.** 117 of 117 box sessions differ from Tokyo's served plan from **k = 0**. The first plan is
  4.1 mm off on 77fdd75c and 4.75 cm off on 1e8e939e.
- **The difference is the same in every box session.** Score, zero kind and zero time match Tokyo on both scenes.
- **Boundedness depends on the scene.** On 77fdd75c the gap stays small for the whole rollout: track within 0.10 m, plan within
  1.23 m. On 1e8e939e the track stays within 0.1 m up to row 80. That is after the collision zero at 6.5 s, which happens at the same
  time on both hosts. The track then grows to 16.5 m by the end.
- **Box vs box: bit-identical.** 115 of 115 sessions match their scene's box anchor: plans, track and score. This holds within one
  launch, across launches and across cards, at 1 to 7.7 launches per card, and with up to 4 launches of the same scene running at once
  on one card.
- **Tokyo vs Tokyo: bit-identical** (2 of 2).
- **Throughput.** One rollout takes 190 s alone on a card. At about 4.5 launches per card it takes 217.5 s (+14.5 %, just under the
  15 % line), which works out to **0.80 min per rollout per card**. At about 7.5 launches per card it takes 231.5 s (+22 %), or
  0.52 min per rollout per card.
- **Disk.** 1 503 GB free on the data disk. P4's output lands on the same disk.

## Setup

- Scenes: the two P4 scenes that were in the box's scene cache and could be given a Tokyo reference. Both are curve_single R.
  - **1e8e939e**: reference `/data/runs/route_night/r281/rsg` chunk c004, plus one fresh Tokyo session.
  - **77fdd75c**: has no r281 rollout. Its reference is two fresh Tokyo sessions run with the frozen P4 clone
    `/data/runs/route_night/p4/code`, card 1, `pai_run.sh`, `jev-alpasim:p2h10-f-s0-6f3d05d1`, `alpasim-base:0.89.0`. The driver
    is the same file as on the box (`onp_log_driver.py`, sha256 23d5f8e9...).
  - S1 on Tokyo (results/p4_self/smoke.md) already shows that Tokyo reproduces r281/rsg bit for bit, so a fresh Tokyo session is a
    valid reference.
- Serving: exactly as rf_p4.sh's `stack` with Amendment 1's one rollout at a time:
  - RSG-L18-s0, `JEV_VCONT=1.0 JEV_LEAD=1 PAI_DUMP=100000`, `controller.gains.rel_front_steering_angle_weight=20`,
    `send_recording_ground_truth=true`;
  - ONP_CELLS with the identity cell only (g 1, raw: the branch switch off), `CONC=1`, one scene per launch, n sessions run one
    after another;
  - box: `pai_native.sh` through the GPU pool.
- Runs, every one a pool job of 3 identity sessions:
  - **T1**: 3 jobs at `--vram 60`, so one per card. Card 0 also carried the P4 agent's one-scene smoke for most of T1.
  - **T4**: 12 jobs at P4's Amendment 1 declaration, `--vram 12 --cpu 3 --ram 24`.
  - **T8**: 24 jobs at `--vram 9 --cpu 2 --ram 10`, so 8 per card.
  - The scenes alternate across jobs. T4 and T8 were chained with `--after`. History keys `p4det-*` are separate from P4's
    `alpasim-p4c1`.
- Reader: `rf_p4det.py det`, which is rf_p4.py s1's comparison of served plan, controller track and score, plus:
  - the first diverging k and the max |diff| per k band;
  - the track difference per band;
  - the zero kind and time (rf_p4.zero_time).

## 1. Determinism

| comparison | sessions | bit-identical (plans, track, score) | first diverging k | plan max \|diff\| m | track max m | score / zero |
|:--|--:|--:|--:|--:|--:|:--|
| box vs box anchor, 1e8e939e | 59 | 59 | none | 0 | 0 | 0 / collision, same |
| box vs box anchor, 77fdd75c | 56 | 56 | none | 0 | 0 | 1.0 / none, same |
| Tokyo vs Tokyo (fresh vs r281; session 2 vs 1) | 2 | 2 | none | 0 | 0 | same |
| box vs Tokyo, 1e8e939e | 60 | 0 | **0** (all 60) | 38.22 | 16.48 | 0 / collision at 6.5 s, same kind and time |
| box vs Tokyo, 77fdd75c | 57 | 0 | **0** (all 57) | 1.23 | 0.105 | 1.0 / none, same |

Every box session differs from Tokyo by exactly the same numbers. The difference is deterministic per host; it is not noise.

How box and Tokyo diverge over the rollout (identical across sessions). k counts decisions at 10 Hz; track rows are per decision:

| scene | plan diff at k = 0 | plan max per k band 0-50 / 50-100 / 100-150 / 150-199 | track max per band | track first > 0 / 0.1 / 1 m (row) |
|:--|--:|:--|:--|:--|
| 1e8e939e | 0.0475 m | 0.31 / 6.08 / 3.44 / 38.22 | 0.010 / 0.41 / 0.56 / 16.48 | 17 / 80 / 182 |
| 77fdd75c | 0.0041 m | 0.51 / 1.23 / 0.44 / 0.32 | 0.070 / 0.105 / 0.098 / 0.097 | 17 / 86 / never |

**Reading:**
- The plans differ from the first decision. During force-GT (k < 17) the ego follows the log, so the tracks are identical up to row 17.
- After that the two hosts drive slightly different states:
  - On 77fdd75c the gap stays at about 0.1 m to the end, with the same score.
  - On 1e8e939e it stays under 0.1 m until row 80. That is after the collision at 6.5 s, which occurs at the same time and with the
    same kind on both hosts. Only after the score is decided does the gap grow into a different trajectory (16.5 m at the end).
- The cause is not isolated here. The first plan already differs, so either the box's native NRE renderer on Blackwell produces
  different frames, or the driver's numerics differ on the other GPU. docs/alpasim.md reports the same class of cm-level difference
  for nuPlan.
- docs/alpasim.md's 22-scene read on the box (19 of 22 within 0.01) is consistent with this. Most scenes keep their score;
  some flip.

## 2. Throughput and packing (one rollout at a time per launch, as Amendment 1)

| packing | jobs | pool jobs on the job's card (time-averaged) | s per rollout, steady (min / median / max) | vs alone | min per rollout per card | job wall (3 rollouts) |
|:--|--:|:--|:--|--:|--:|--:|
| T1, 1 per card (conc 1.0 jobs) | 2 (+1 at conc 2) | 1.0 | 184 / **190.0** / 192 | 1.00 | **3.17** | 615 s |
| T4, P4's declaration | 12 | 3.3 / 4.5 / 5.0 | 202 / **217.5** / 229 | **1.145** | **0.80** (0.91 at exactly 4) | 703 s |
| T8, 8 per card | 24 | 7.2 / 7.5 / 7.7 | 193 / **231.5** / 261 | 1.22 | **0.52** | 731 s |

- "Steady" is the time from one session's start to the next. The last session of a job (including teardown) took 207 / 234 / 228 s.
  Start-up was 30-32 s per launch at every packing.
- **The 15 % line.** At about 4.5 per card a rollout is 14.5 % slower than alone, so the per-card throughput is 14.5 % below
  proportional. That is just under the line. At about 7.5 per card the per-rollout slowdown is 22 %, but the per-card throughput still
  improves by 1.55x over 4.5 per card. Fewer per card would lower throughput, so the fallback is not warranted.
- **Pool behaviour.** The pool spread T4 as 5 / 4 / 3 per card, not 4 / 4 / 4: the start limit admits 4 jobs per round, and
  another lane's job sat on card 2. The time-averaged column is what each job actually shared its card with.
- **Per-launch peaks**, the same at every packing:
  - VRAM 6.3-6.5 GB, RAM (private) 7.6-8.7 GB, 1.2 cores.
  - Per card: 6.5 GB at 1 per card, 38-44 GB at 3-5 per card (including another lane's 19 GB job on card 2), 51 GB at 8 per card.
  - GPU utilisation was 99 % at 8 per card, yet a rollout was only 22 % slower. The wall per rollout is not GPU-bound.
- **Host memory binds first, not VRAM.** At 24 launches box-wide, anon memory had a plateau of about 160-170 GiB, about 7 GiB per
  launch. The working-set peak reached 243.7 GiB against the kill line of 270.5 GiB. Most of that peak was a non-pool CPU job of
  another lane (`route_follow/slownov`, about 70 GB for 20 s at 12:41) on top of our plateau. No job was killed.
- **Amendment 2's P4 layout** is 3 launches per job and 4 jobs per card, `--vram 22 --ram 32`, which is 12 launches per card and 36
  box-wide.
  - VRAM fits: 4 x 18.8 = 75 GB per card.
  - Host memory at 36 launches would be about 250 GiB, above the pool's admission line (0.85 x 276 = 235). The pool will therefore
    hold P4 at about 9-10 jobs (about 28 launches) by RAM, not 12.
  - 12 launches per card was not measured. The curve 1.00 / 1.145 / 1.22 at 1 / 4.5 / 7.5 suggests about 1.3x at 9-12, below
    Amendment 2's assumed 1.5-2x.

**Implied wall time on 3 cards.** The first number in each pair is the throughput bound. The second is the makespan once one job per
scene runs its rollouts in sequence (Amendment 2: phase B split over 3 launches).

| rollouts | 4 per card, 12 slots (217.5 s, measured at ~4.5) | 7.5 per card, 22.5 slots (231.5 s) | 1 per card (190 s) |
|:--|:--|:--|:--|
| 1 710 (the brief's count; 63 per scene) | 8.6 h; one Amendment-1 job of 63 serial rollouts = 3.8 h (> the old 4 h `--timeout-h`), 27 jobs in 12 slots ~11.5 h | 4.9 h; with P4_PAR 3, ~21 rollouts per launch ~1.4 h per job, ~3 waves ~4.5 h | 30 h |
| ~690 (prereg count; ~26 per scene) | 3.5 h | 2.0 h; with P4_PAR 3, ~9 per launch ~40 min per job, 27 jobs at ~9-10 concurrent ~2.0-2.5 h | 12 h |

## 3. Disk

- **Data disk:** `/dev/md0` on `/root/autodl-tmp` (= `$DATA_DIR`), 6 295 GB. **1 521 GB free at 12:00 CST, 1 503 GB at 13:00 CST**.
  The fetch of the remaining P4 scenes adds about 1.7 GB per scene, about 20 GB in all.
- **P4 output:** `rf_p4.sh fanout`'s root is `$DATA_DIR/runs/route_follow/p4/run` (= `/root/autodl-tmp/ujs/runs/route_follow/p4/run`),
  on the same disk. `fanout` refuses to submit below 500 GB free (`MIN_FREE_GB`).
- **Per rollout, measured here:**
  - rollout.asl: 0.374 GB (1e8e939e), 0.262 GB (77fdd75c);
  - whole rollout directory with the 1 Hz dumps: 0.41 / 0.30 GB;
  - the prereg's range for other scenes: 0.4-0.55 GB per .asl, plus about 0.03 GB of dumps.

| rollouts | all .asl kept | failing .asl deleted after each scene's read |
|:--|:--|:--|
| 1 710 | 0.6-1.0 TB. Margin to 1.48 TB free: 0.5-0.9 TB (0.3-0.7 TB above the 150 GB floor) | **The peak is the same as all kept when every scene job runs at once** (reads come at the end of each job): 0.6-1.0 TB. With jobs in waves of ~10: about 10 x 35 GB in flight plus what is kept. Final footprint about 0.25-0.4 TB if a third complete. |
| ~690 | 0.25-0.4 TB, margin 1.1-1.2 TB | Peak at most 0.25-0.4 TB; final about 0.1-0.17 TB (Amendment 2: peak ~160 GB, final ~115 GB) |

Disk is not a constraint for either count.

## Recommendation for the fanout

- **Per-card count: let the pool pack, do not cap at 4.**
  - The wall per rollout grows only 14.5 % at about 4.5 launches per card and 22 % at about 7.5, while per-card throughput keeps rising.
  - Host memory (about 7-8 GiB per launch, and the pool's 235 GiB admission line) binds before VRAM (6.5 GB per launch) or the GPU.
  - Amendment 2's `--vram 22 --ram 32`, 3 launches per job, will run about 9-10 jobs (27-30 launches) at once. That is fine.
  - The one rule to keep: watch `cl top` for non-pool bursts from other lanes near the kill line.
- **The box-side S1 gate must be "box vs box bit-identical", not "bit-identical to Tokyo".**
  - Every box rollout differs from Tokyo from k = 0 by a fixed, host-specific amount. A Tokyo-identity gate fails by construction.
  - The box is perfectly repeatable: across launches, cards, load, and concurrent launches of the same scene in separate renderer
    processes, which is exactly Amendment 2's P4_PAR layout.
  - The gate that matters is the one `rf_p4.py read` already applies: every self branch's plans before h equal the scene's own box
    phase A. Keep h, reach and all comparisons inside one host.
  - Do not pair box branches with Tokyo or r281 identity rollouts, and do not read an r281 score as the box identity's score. They
    agree in kind (same zero class and time here) but not bit for bit, and docs/alpasim.md records occasional flips.

**Not covered.** Two scenes, both curve_single R. Exactly 4 and 12 launches per card were not measured (the pool placed 3-5 and
7.2-7.7). The cause of the k = 0 difference (renderer frames or driver numerics) was not isolated.

**Side finding.** `python -m jevdrive.cl fanout` without `--no-check` failed here with
`AttributeError: module 'jevdrive.cl.pool' has no attribute 'fanout_fill'`. `__main__.py` calls it and `pool.py` does not define it,
at 0a2c7a94 and still at 652d48b7. These jobs were submitted with `--no-check`.
