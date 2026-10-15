# REWIND-ONPOL-ALL: expert recovery labels on every failed turn of onpol-train (2026-10-16, decision 319)

Prereg [plans/2026-10-16-rewind-onpol-all-prereg.md](../../plans/2026-10-16-rewind-onpol-all-prereg.md) (+ Amendment 1). No training.
Same instrument as decision 316 ([../rewind_train64/rewind_train64.md](../rewind_train64/rewind_train64.md)), extended from 64 scenes to the
whole onpol-train split. Setup: RSG-L18-s0 alone, sh30 held-read serving, 30-decision sessions (after280 cache). The rule expert XP-L12-E46
takes over in the same session at h = k_ref - 2 Delta, for Delta = 2 and 3 s.

**Trainer row list (box): `$DATA_DIR/runs/route_follow/rewind_onpol_all/rows/rows.csv`.** The format is in the README next to it.
- One line per 2 Hz decision k >= h on the junction stretch, before the session's first zero.
- Columns: scene, log, delta, h, k, expert session (run dir + rollout_id), completed, zero_free, label (= completed AND zero_free), gate_pai,
  gate_nuplan, within_tol.
- Filter `label == True` for the registered label set.
- `rows_partial_0105/` beside it is an incomplete first read (see Amendment 1). Do not use it.

## Scope

- **Scenes.** The stored RSG-L18-s0 onpol-train rollouts have 1 031 scenes. The follow table and the inventory list 1 021; the other 10
  are the 18-decision simulator errors there. 17 scenes fail in the simulator at 30 decisions too ("route folds back on itself"). That
  leaves 1 014 scenes read.
- **Turn window.** rf_follow's turn window is the log up to decision 29 (`RF_K_END=29`). With decision 17, the only window d316 had,
  the split holds only d316's 64 turns.
- **Strict turns.** 166 turns in 60 logs.
  - Identity outcomes: completed 71, event 65, off 17, short 13.
  - Failed (A) 95. Of these, 78 event / off units are handed over. The 13 short units are listed in units.csv but not rewound.
  - 4 units at Delta 2 s and 5 at 3 s cannot be rewound, because h < 1.
- **d316's 32 event / off units.** 30 are again event / off here, 1 completes, and 1 has no strict turn within this scene's 6 s log.

## Expert from h (files: table_raw.md, rewind_onpol_all.csv)

| Delta | handovers | within identity tolerance | completed (log CI) | completed / event / off / short | any zero after h (at-fault) | creep v(h+3 s) >= 2 | completed + gate PAI | completed + gate nuPlan |
|--:|--:|--:|:--|:--|:--|:--|--:|--:|
| 2 s | 78 | 69 | 0.65 [0.55, 0.76] | 51 / 20 / 0 / 7 | 25 (20) | 68 / 77 | 24 | 16 |
| 3 s | 77 | 73 | 0.68 [0.57, 0.79] | 52 / 18 / 0 / 7 | 23 (18) | 69 / 77 | 28 | 25 |

- **Completion vs d316.** Completion is lower than d316's 0.80 at 2 s. The new turns sit later in the 15 s window (k_ref > 17 on 17
  units), so the expert has less time to finish them: 7 short at each Delta. The expert fails by collision, 19 / 17 at Delta 2 / 3 s.
- **Tolerance.** 9 / 78 and 4 / 77 handovers are outside the identity tolerance, below the 20 % stop line. Their pre-h plan differs by at most
  1.2 m, the track by at most 0.23 m. They stay in the row list, flagged `within_tol = False`. At most 4 + 1 label units are among them.
- **Gates.**
  - nuPlan-recalibrated thresholds (human p95 of the 166 turns): kmax 0.151 1/m, dk 0.0406 1/m^2, alat 2.83 m/s^2, entry 6.7 m/s.
  - PAI thresholds: 0.184 / 0.0425 / 4.42 / 5.6.
  - Human logs passing: 122 / 166 under PAI, 136 / 166 under nuPlan.
  - The nuPlan gate is looser on entry speed and tighter on lateral acceleration.
- **Exclusions by a later zero.** 5 completed continuations per Delta lose their label only because of a zero after the turn (4 corridor,
  i.e. dist_to_gt >= 4 m late in the session, and 1 collision). Their junction rows are in rows.csv with `zero_free = False`.

## Label rows (2 Hz, k >= h, junction stretch [entry - 20, exit + 10] m, completed and zero-free after h)

| Delta | category | units | rows | unique turns | logs | top-3 turns' share |
|--:|:--|--:|--:|--:|--:|--:|
| 2 s | completed + gate (PAI) | 22 | 234 | 22 | 21 | 0.24 |
| 2 s | completed + gate (nuPlan) | 16 | 192 | 16 | 15 | 0.29 |
| 2 s | completed, zero-free, no gate | 46 | 483 | 46 | 34 | 0.11 |
| 3 s | completed + gate (PAI) | 24 | 308 | 24 | 22 | 0.20 |
| 3 s | completed + gate (nuPlan) | 23 | 301 | 23 | 21 | 0.20 |
| 3 s | completed, zero-free, no gate | 47 | 581 | 47 | 35 | 0.11 |
| both | completed + gate (PAI) | - | 542 | 29 | 25 | 0.21 |
| both | completed + gate (nuPlan) | - | 493 | 26 | 23 | 0.23 |
| both | completed, zero-free, no gate | - | 1 064 | 49 | 36 | 0.11 |

- **Totals.** Against d316's 122 (Delta 2 s, gated) and 230 (Delta 3 s), the whole split gives 234 / 308 gated (PAI), or 1 064 ungated
  label rows over both Delta.
- **Spread.** These come from 49 turns in 36 logs; no turn dominates (top 3 = 11 %).
- **Overlap of the two Delta.** A turn that labels at both Delta contributes two different handover states, so the "both" rows do not repeat
  a state. They do share the end of the turn.
- **Ceiling.** This is all the split holds under this serving. More rows need more rollouts per scene (seeds, other checkpoints' states)
  or other splits.

## Files

- `table_raw.md`, `summary.json`: n-yield output.
- `rewind_onpol_all.csv`: one line per unit and Delta.
- `units.csv`: every strict turn, with outcome, k_ref and h.
- `units.json`: every scene's unit (tfail).
- `from_D2.txt`, `from_D3.txt`: `<scene> <h>`.
- `read.json`: n-read output.

Box: `$DATA_DIR/runs/route_follow/rewind_onpol_all/`. Stages `smoke`, `s50`, `rest`, `restb`, the combined read in `all/`, packs in
`<stage>/pack`, and `jobs.txt`.

Code: `scripts/rf_rewind.sh train64 {scenes-all, cache, ostage, oread}`, `rf_rewind.py n-tfail --kend --deltas --rewind`, `n-read` and
`n-yield`. Commits 412f677c..8fa48dc1.

Cost:
- About 5.8 GPU job-h at 4 jobs per card, i.e. about 1.5 card-h.
- CPU: cache 25 min wall on 32 cores; tfail, pack and read 2-4 min each.
- Wall clock 00:03-02:30 box time, including the repair.
