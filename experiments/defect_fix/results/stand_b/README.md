# STAND-B: arm B's front collisions, a stricter FREE-M free check, half dose, and B at full length (2026-10-19; NR1 prereg supplement 19)

Question: arm B of the JTX STAND pilot (`STANDFMX1K-L18-s0` = STAND W_S 0.0068 + FREE-M W 0.002 / move / 0.95 on on1_JTX, against
`JTX1K-L18-s0`, native reads) was the best result of the day. It failed two driving guards: held 460 zeros 37 -> 45 and front collisions
2 -> 7. Where do they come from? What must FREE-M's free check block? Do B's gains survive at full length?

Code: [scripts/stand_b.py](../../scripts/stand_b.py) (diag / pool / ratchet / smoke / full), [scripts/stand_b.sh](../../scripts/stand_b.sh)
(check, smoke, chain), `nr1_free.configure(clear=)` / `rm_train --on-free-clear CMIN,CTH` (default off). Box outputs:
`$DATA_DIR/runs/defect_fix/stand_b/`. Prereg: [supplement 19](../../plans/2026-10-17-nr1-prereg.md).

## Diagnosis (CPU, stored rollouts; [diag.json](diag.json), [diag_windows.csv](diag_windows.csv), [pool.json](pool.json))

- **Windows.** On held 460, B against JTX1K has 12 new zeros (6 AF-front, 6 offroad) and 4 gone (3 offroad, 1 AF-front). B has 7
  front-collision windows (6 new, 1 shared); the control has 2. Arm A (STAND alone) has 3, 2 of them shared with the control, so the
  collisions come from FREE-M. FREEM1K against JTNS1K (decision 378) shows the same pattern: 13 front windows, 19 new zeros.
- **Lead state before each collision.** These are the 47 decisions in the 3 s before the first front collision, over B's 7 windows.
  The struck object is in the plan's path in 38 of them, and it is the path lead in 34. Medians: ego 5.4 m/s, gap 4.0 m, closing speed
  3.6 m/s. TTC is below 4 s in 45 / 47. So the ego is approaching a slowing or stopped lead.
- **FREE-M never fires on those states.** Only 3 of the 47 decisions are FREE-M rows (the human is braking or stopping there). The term
  fires 0 times, under the current check and under each stricter one. Still, B's plans there are long: the plan's 4 s arc over the
  reference has median 1.5-2.35, against 0.96-1.22 for the control in the same windows. The collisions are generalisation.
- **Where FREE-M fires in training.** Set: on1_JTX moving S rows, 52 610. The plan is approximated as the human's `fut` path cut at
  RB0-F-s0's own 4 s arc. The current check fires on 31 036 rows. 14 387 of them have a lead on the path, at gaps of 11 / 16 / 26 / 42 / 58 m
  (10-90 %), median closing -0.58 m/s. 5 854 have the lead within max(20 m, 3 s x v), but only 473 have TTC < 4 s. The check (2 m + 0.5 s x v,
  same-time sweep) treats following a lead at 11-20 m as a free road. Those rows are the near neighbours of the pre-collision states. In
  the read windows, 25 of B's 39 fires had a lead within max(20, 3v).
- **The 6 new offroad zeros** have nothing to do with a lead. In 4 of the 6 windows B is 0.9 m/s faster on average, and the term fires
  22 times inside them. That is FREE-M's speed effect. A stricter free check cannot reach it; only the dose can.

## The stricter check (chosen before any B1 / B2 output)

| check (safety distance ahead of the bumper) | pool fires removed | ratchet decisions still fired (supplement 16 gate 2, line >= 50 %) |
|:--|--:|--:|
| current: 2 m + 0.5 s x v | 0 | 65.1 % (decision 378) |
| TTC > 4 s | 1.5 % | not run (does not discriminate) |
| max(15 m, 2 s x v) | 17 % | **56.3 %** ([ratchet_clear.json](ratchet_clear.json)) |
| max(20 m, 3 s x v) | 29 % | 49.7 % ([ratchet_d20t3.json](ratchet_d20t3.json)): fails |
| max(30 m, 4 s x v) | 42 % | not run |

Chosen: **`--on-free-clear 15,2`**. A 2 s headway is the usual following rule. 20 m / 3 s misses supplement 16's coverage line.

## Runs (supplement 19; one pool submission)

- B1 `STANDFMXG1K-L18-s0` = B + `--on-free-clear 15,2`. B2 `STANDFMXH1K-L18-s0` = B with FREE-M W 0.001. Both are 1 000 steps against
  JTX1K-L18-s0. Reads: wait set v2, held 460, arena subset (6 chunks), all native. The guards are supplement 17's plus the drift guard;
  among them, held zeros <= control + 2 and held front collisions <= control + 2. Main reads: the frozen-pool subset PCS against the
  control (Mac: `tmp/pcs_local/.venv/bin/python experiments/alpasim_proxy/scripts/pcs_local.py score --rows arena_windows.csv.gz --subset`)
  and the arena subset paired. An arm passes on all guards, paired >= +0.01, and PCS >= control + 30.
- Full-length B `STANDFMX-L18-s0`: B's command at 9 500 steps with warm-up 500 (the on1_JTX analogue of JTN-L18-s0). Reads: the arena's
  18 held chunks (native) against JTN-L18-s1, held 460, and PAI 281 with ACC + GAP against NTV0 + ACC + GAP. This is a read, not an adoption
  gate.
- Command diffs: [cmd_diff.txt](cmd_diff.txt) (B's DRY command = its logged training command; B1 / B2 / full differ only in the stated tokens).

## Status (submitted 2026-10-19 17:55 box time; nothing read yet)

- Smoke passed before the chain: B's command, 60 steps eager, is bit-identical on 6fec7b27 and this code ([identity.json](identity.json)).
  B1's smoke is finite, and its FREE-M fire share is 0.095 -> 0.083 at step 0 and 0.102 -> 0.084 over the run ([smoke.json](smoke.json)).
- Trains: B1 `1019-175426-1cbe`, B2 `1019-175439-9a03`, full B `1019-175451-43de`. Training priority 12, reads 13. Every id is in box
  `runs/defect_fix/stand_b/jobs.txt` / `stages.txt`.
- **Summary jobs** (`runs/defect_fix/stand_b/summary_ids.txt`):
  - `stb-psum` **1019-175450-6192**: B1 / B2 pilot. It writes `pilot_B1.md`, `pilot_B2.md`, `arena_B*.json`, `drift_B*.json`,
    `arena_windows.csv.gz`, `diag_pilot.json` and `diag_pilot_windows.csv` here, and `runs/defect_fix/stand_b/PILOT_DONE`.
  - `stb-fsum` **1019-175506-bf15**: full B. It writes `full.md` / `full.json`, `arena_full.json`, `arena_windows_full.csv.gz` and
    `pai281_accgap.md` here, and `runs/defect_fix/stand_b/FULL_DONE`.
- Expected finish if the pool stays free: psum around 18:40, fsum around 20:00 (box time).
- Still to do in the write-up: the frozen-pool subset PCS on the Mac for both summaries, the pass / fail per supplement 19, and a decision row.
