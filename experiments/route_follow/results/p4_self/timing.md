# P4 box run: timing and resources (2026-10-14)

Box, 3 x RTX 6000D, GPU pool, one job per scene (`p4-<id8>`, declared `--vram 22 --cpu 4 --ram 32`), code 652d48b7 (code.json).
First submit 12:43 (stage 1), stage 10 at 13:14, `all` at 13:50; the last scene job finished 15:33 (box time), and the collect read wrote read.json at 15:33.
Wall from the first submit to the collect: 2 h 50 min, 693 rollouts (27 phase A + 666 phase B).

| item | value |
|:--|:--|
| stage 1 (0d10d2e3, 25 rollouts; a determinism read shared the cards at the start) | job 27.9 min: phase A 262 s, phase B 24 cells over 3 launches 1379 s (172 s per rollout incl. start-up) |
| stage 10 (9-10 jobs on 3 cards at once) | phase A 264-277 s; phase B 1502-1588 s for 5-6 cells per launch (250-265 s per rollout, about +45 % against stage 1) |
| all 27 jobs, wall per job | min 14.8, median 31.4, max 69.4 min (5-handover scenes 50-70 min); sum 17.6 job-hours |
| phase A (start-up + 1 rollout), 27 jobs | min 241, median 289, max 359 s |
| phase B, slowest launch / its cells | min 172, median 228, max 320 s per rollout |
| per job peaks (cl show) | VRAM 18.3-20.2 GB, RSS 24.0-31.6 GB, cores 1.3-3.4 |
| concurrency | 9 jobs at once (27 launches) in the middle; `cl top` working set 163-207 GiB there, margin 57-194 GiB over the run; one sample of margin 8 GiB at 13:51 (start-up of 17 jobs at once), no SIGKILL |
| failures | 0 jobs with ERROR, 0 retries (tries 1 for all 27) |

Per job: [timing_jobs.tsv](timing_jobs.tsv) (name, wall min, VRAM peak GB, RSS peak GB, cores peak, tries, state). Run log: [p4_run_log.txt](p4_run_log.txt).
