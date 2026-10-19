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

## Pilot results (1 000 steps against JTX1K-L18-s0; [pilot_B1.md](pilot_B1.md), [pilot_B2.md](pilot_B2.md), [arena_windows.csv.gz](arena_windows.csv.gz))

| | control JTX1K | B (W 0.002) | B1 (gate 15 m / 2 s) | B2 (W 0.001) | line |
|:--|--:|--:|--:|--:|:--|
| frozen-pool PCS, 716-window subset (Mac) | 1173 | 1259 | 1260 | 1246 | arm >= control + 30 |
| held 460 zeros | 37 | 45 | 45: FAIL | 42: FAIL | <= control + 2 (= 39) |
| held 460 front collisions | 2 | 7 | 5: FAIL (line 4) | 3: ok | <= control + 2 |
| arena subset (753) paired mean | | +0.043 | +0.0455 [+0.0046, +0.1103] | +0.0444 [+0.0036, +0.1093] | >= +0.01 |
| k = 9 speed ratio | 0.845 | 0.948 | 0.937 | 0.907 | descriptive |
| turn drift k11-17 (m, descriptive) | | | -0.070 (CI spans 0) | **-0.176** [-0.296, -0.032] | drift guard (all k11-17, > -0.05): ok in both |

- B1 and B2 both keep the static gain (static zeros 74 -> 43, red light at the line, no lead 62 -> 37). Neither passes: B1 leaves the held
  zeros and front collisions where B had them, B2 cuts front collisions 7 -> 3 but still has held zeros 42 (line 39).
- B2 drifts right in turns (-0.18 m), only in turns; the drift guard itself (all k11-17: B1 -0.020, B2 -0.025) passes in both.

## Full-length B (`STANDFMX-L18-s0`, 9 500 steps; [full.md](full.md), [pai281_accgap.md](pai281_accgap.md), [arena_windows_full.csv.gz](arena_windows_full.csv.gz))

| read | STANDFMX-L18-s0 | reference |
|:--|--:|--:|
| frozen-pool PCS, all 2 106 items (Mac) | **1368** (mean 0.9368) | JTN-L18-s1 1247 (0.8933), JTN-L18-s0 1198, RSG 1106 |
| native arena held (18 chunks), mean / paired vs JTN-L18-s1 | 0.9350 / **+0.0441 [+0.0203, +0.0715]** | JTN-L18-s1 0.8909 |
| arena zeros | 100 | 210 (JTN-L18-s1) |
| arena AF-front | 3 | 7 |
| k = 9 speed ratio | **1.129** (overshoot) | 0.912 |
| held 460 zeros / front collisions | 53 / 11 | 36 (JTN-L18-s1) |
| held 460 paired vs JTN-L18-s1 | -0.0436 [-0.068, -0.023] | |
| PAI 281 ACC + GAP paired vs NTV0 + ACC + GAP | -0.0472 [-0.078, -0.018], zeros +12 | NTV0 0.7434, 59 zeros |
| PAI AF-front | 9 | 3 (NTV0) |

Reading (supplement 19.4 rules): the arena gain and PCS hold at full length (best of any model so far, stop3-level, without a hold), but
held 460 and PAI are worse than the references (overspeed and front collisions), so the guard leak persists at full length. 1 seed; a read,
not an adoption gate.

## Status

- Pilots and full-length B were read on 2026-10-19 (summaries `stb-psum`, `stb-fsum`). The nuPlan native arena and PCS are the best of any
  model so far, but held 460 and PAI are worse. Neither B1 nor B2 passes the pilot rule.
- Full-length B2 `STANDFMXH-L18-s0` (supplement 19.7; FREE-M W 0.001, everything else the same command as the full B, diff in
  [cmd_diff_H.txt](cmd_diff_H.txt), smoke in [smoke_H.json](smoke_H.json)) is running to test whether the lower dose keeps the static hold and
  the PCS while removing the overshoot. Outputs when done: `full_H.md`, `arena_windows_full_H.csv.gz`, `pai281_accgap_H.md`, `arena_full_H.json`.
- Full-length B2 submitted 2026-10-19 19:18 box time as one chain (`chainh`): train `1019-191831-0e2f` (priority 12), 18 arena chunks + 6 held 460
  + 8 PAI 281 jobs (priority 13), summary `stb-fsum-H` **1019-191850-d1a3** (writes the H files here and `runs/defect_fix/stand_b/FULL_H_DONE`).
  Expected finish about 21:30-22:00 box time. The smoke's first criterion (fire share not above B, meant for the B1 gate) failed on the dose arm
  (0.0966 vs 0.0949 at the first step), so it was changed to finite-only before the chain; see supplement 19.7.
