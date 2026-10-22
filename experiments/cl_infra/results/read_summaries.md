# Read summaries: inventory, timings, facts cache (2026-10-22)

**Question.** Summaries after closed-loop reads took minutes to tens of minutes although the scores exist as soon as each
read chunk ends. Where does the time go, and does a per-read facts cache reproduce the old summaries exactly?

**Answer.** The time was the native_gap re-extraction of every window of every tag (including references extracted
before), in one 32-core job after all reads; the 73 min of 2026-10-22 was that job crashing on a missing log dir and
waiting for a resubmit. Each read chunk now writes its own facts at the end of the read job (~9 s on the read's 6
cores), and the nuplan_dag Part 1 summary joins the cache in **23 s on 4 cores instead of 179 s on 32**, with outputs
identical to the stored ones. Code: [jevdrive/facts.py](../../../jevdrive/facts.py); usage:
[docs/closed-loop-runbook.md](../../../docs/closed-loop-runbook.md#read---facts-cache---summary).

## Inventory: summary paths that touch rollouts

| Path | Reads | What it re-extracts | Time (measured) |
|---|---|---|---|
| nuplan_dag `nd.sh p1sum` / `p2sum` (before 6ed22fe3) | arena held + held 460 | `free_m.py ext` = native_gap extract of all tags, refs included; then `free_m.py arena` x 3 (each reloads the extract), `nd.py latsplit` | 179 s on 32 cores (table below) |
| defect_fix `stand*.sh`, `tight_in.sh`, `turn_cap.sh`, `nu_flat.sh`, `stoppt.sh` | arena | the same `$AP $FM ext` | same cost per tag; concluded lanes, not changed (drop-in: `$AP -m jevdrive.facts ext`, same arguments) |
| `pai_fix.sh tax` (PAI 281, LAUNCH-37, PAI-FIX; used by round5..8, r8k) | PAI | rf_tax extract + `pai_fix.py facts`, once per read after all chunks | 111 s on 16 cores per read (8 reads 2026-10-22); the summaries (`r2.py r281`, `r5.py`, `r8k.py`) join `runs/pai_fix/facts/*.csv`: already this pattern |
| `jt.py arena` (arena.py collect) | arena | scorer outputs only | seconds |
| `nd.py quick`, `nd.py held` (df.cl_read) | arena / held 460 | scorer outputs only | 3.9 s / 0.6 s |
| `rm_chain.sh submit-read` -> `rm_build.py cover` | held 460 | drive records per read | ~21 s pool job |
| `rm_report.py report` -> `zm_read.py collect` (ol_zm) | held 460 | scene loader + vector map per scene | heavy; route_mem lane, concluded |
| `c1_extract.py logs` | rolls | full rollout pickles for training rows | not a summary (nd.sh rolljob) |

## Timings, nuplan_dag Part 1 (R8K-L18-s2, R6DAG-L18-s1 vs B2 = STANDFMXH-L18-s1, NTV0-L18-s0)

`experiments/cl_infra/scripts/rs_bench.sh` (pool jobs, box idle otherwise). Old = nd.sh p1sum's commands verbatim.

| Step | Old (32 cores) | New (4 cores, cache) |
|---|--:|--:|
| score-only tables (`facts table` arena + held 460) | - | 2.1 s |
| `nd.py quick` (score-only, scorer outputs) | 3.9 s | 2.8 s |
| extract (4 tags x 18 chunks, 2 228 windows, 89 120 decisions) | 98.9 s | 1.7 s (concatenate the cache) |
| `free_m.py arena` x 3 | 39.7 s (serial) | 15.8 s (parallel; held and latsplit run alongside) |
| `nd.py held` | 0.6 s | 0.8 s |
| lateral-zero split | 35.2 s | 0.4 s (cached per chunk) |
| windows csv, p1md | 1.2 s | 1.0 s |
| **total** | **179 s** | **23.4 s** |

Where the old time went: the extract loads each window's human track and logged traffic through the scene loader (one
call per window, shared by the tags) and projects every decision on it; latsplit filters the 89 k-row extract once per
zero window; each arena call reloads and re-derives the whole extract.

Per-read extraction (written at the end of each read job, `jevdrive.facts chunk`): one arena chunk (127 windows) 9.6 s
on 6 cores, one held 460 chunk (77 scenes) 8.6 s; ~90 % is the scene loader. Backfill of the six reference reads
(STANDFMXH-L18-s1 / -s0, NTV0-L18-s0, RSG-L18-s0, R8K-L18-s2, R6DAG-L18-s1): arena 145 chunks in 105 s, held 460 36
chunks in 40 s, one 32-core job.

## Identity

`rs_check.py` (results `$DATA_DIR/runs/facts/_bench/identity.json`): **identical**.

- Extract from the cache vs the old extract (same run) and vs the lane's stored `runs/nuplan_dag/p1/ext`: dec / win /
  steps / set_census frames equal (rows, order, dtypes, values), human tracks equal.
- `arena_<tag>.json` x 3 (means, paired CIs, zeros, AF-front, k9 speed ratio with its bootstrap CI, strata, zero
  classes), `held.json`, `latsplit.csv` (d395 lateral split): byte-identical to the lane's stored outputs and to the
  old run; `p1.md` byte-identical to the lane's stored one (against the old run line-identical: nd.py orders tied
  zero-class rows by set order, which varies per process).
- `facts table` (score-only) vs `nd.py quick`: same pairs, means, CIs, zeros, lateral zeros and zero classes, arena
  and held 460.
- Concurrency: two writers on one chunk (one forced) serialize on the read's lock; the rebuilt chunk equals the old one.
- nd.sh p1sum as committed (6ed22fe3) run as a pool job (8 cores) into a scratch dir: 22 s wall, outputs identical to the stored Part 1 outputs.

## What changed for live lanes

- `jt.sh aread` and `rm_chain.sh read` end with `jevdrive.facts chunk` (non-fatal; `|| true`), so every new arena / held
  460 read chunk is cached when it ends.
- `nd.sh p1sum` / `p2sum`: `$AP -m jevdrive.facts ext` instead of `$AP $FM ext`, `jevdrive.facts latsplit` instead of
  `nd.py latsplit`, arena pairs in parallel. Tonight's queued p2sum (1022-103351-a985) runs this body; tags whose chunks
  are not cached are extracted by it first (same code, its 32 cores).
- PAI lanes: unchanged (`pai_fix.sh tax` already writes per-read facts).
