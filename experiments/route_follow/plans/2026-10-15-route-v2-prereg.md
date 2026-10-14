# ROUTE v2: ordered route input, near-field lane hinge with negatives, lane-change counterfactual pairs (pre-registration)

2026-10-15 night, written before any pilot read. Task book: research/next-round/turn-night.md section 2 (training-design candidates, to be cut
by readings A and B). Context: decisions 282, 284, 286 (RB0-F / RSG-L18 / RBSG2), 287, 293, 297. Code: `experiments/route_mem/lib/rmem.py`,
`experiments/route_mem/scripts/rm_build.py order`, `rm_train.py` (flags below), `experiments/route_follow/scripts/rf_v2gate.py`, `rf_v2.sh`.

## 1. The three changes (each behind a flag, all off = RSG-L18's recipe operation for operation)

| flag | what | why (symptom in turn-night.md section 0) |
|:--|:--|:--|
| `--rm-order` | memory window cut at +40 m (24 points, -6 .. +40 m at 2 m from the ego's projection; far points dropped, not masked) + two scalars: arc distance to the next route heading change >= 30 deg (cap 60 m, unknown -> mask bit) and arc length travelled since the first route message | near route empty -> the plan heads for the far points (shortcut, 180 instead of 90 deg: 9f393c8a, 2466266e); progress / order missing (d287's 40 m gap) |
| `--lane-lam` | lateral hinge of plan poses 1.0-2.0 s (w 2) and 2.5-4 s (w 1) to the carried route, dead zone `--lane-dz` 0.5 m, shape-only (lateral) gradient, on every non-anchor row whose memory is fed (log, yaw-rate and pool rows); negatives `--lane-neg` 0.2: fake input memory (parallel shift +-3.5 m, or the straight continuation on curved non-junction NAVSIM rows) with the ORIGINAL route as hinge target and the imitation target unchanged; over-turn guard: relu(heading excess into the turn at 2 s - 10 deg) | right lane never penalised (1a69f757); over-turn into kerbs (382a7247, de9db5dc); RBSG2's hinge over-turned and cost public offroad zeros (d286 correction 2) |
| `--lc-cf` | lane-change counterfactual pairs: per batch k tokens, the real row (real route memory, imitation of the log) and a stay-lane row (same camera tokens, stay-lane route memory + its command, lane hinge to the stay-lane label with 0.5 m dead zone, no imitation) | lane-change commands ignored (d284: swing 0.05 m; d286: CF rows taught junction switching but lane changes only 5-7 %) |

Implementation facts settled while building (recorded here as the decisions they are):

- D1 (ordered layout). The existing window (rmem.window) is already ordered by arc from the ego's projection on the merged route and
  resampled every 2 m; the ordered layout is therefore its first 24 points (bit for bit) plus the scalars, a pure function
  `rmem.to_order(old window, trav)` shared by the offline rows, the served Tracker and the trainer's on-the-fly negatives. A progress anchor
  for the near gap (ego arc = -42.1 m + odometry instead of the geometric extrapolation) was considered and rejected: with a 40 m horizon it
  leaves the first decisions with no point at all, whereas the geometric anchor keeps the post-turn points of a route that starts beyond a
  turn in the 0-40 m slots. The d_turn reference heading is the route's at +0 m when that point is valid, else the ego's own (the gap).
- D2 (train / serve gap). Training rows keep the serve-time near gap: NAVSIM rows are decision k = crc32 % 18 of an 18-decision rollout that
  follows the log (rm_build nav), pool rows are rebuilt at the driver's anchors (G-serve of d282: 90 / 90 masks equal). trav uses the same
  rule on both sides (chords between the route-message poses, then to the anchor). Verified by `rm_build.py order`'s near-field summary
  (share of rows with no valid point in 0-20 m) and, for any ordered checkpoint that is read, `rm_build.py served` (now compares d_turn and
  trav too).
- D3 (hinge target). The lane hinge's target is the old-layout memory polyline (-6 .. +72 m) of the row, whatever the input layout: the
  carried route, not the log path, so the hinge is 0 where the memory does not reach the pose (no target in the gap). The pose weights are
  on_route's W_T (0 at 0.5 s).
- D4 (straight negatives). Only on NAVSIM training rows whose route turns >= 15 deg within 40 m and whose token has no alternative-exit
  counterfactual in runs/route_follow/cf/rows (j0 / j1 = a junction choice exists): a straight continuation at a junction can be a legal
  route, and teaching the plan to ignore it would contradict G-swap. Half of the negatives on such rows are straight, the rest shifts.
- D5 (lane-change rows). rf_cfpool already builds stay-lane rows: lane-graph lane changes within 4 s ending >= 2 m off the t0 lane, 3 972
  candidates, 2 615 ok (2 589 train in on1_CF + 26 dev in on1_CFdev); 1 140 fail because the real route memory already equals the stay-lane
  route (the generator kept the lane), 262 on the lane snap. d284's 4 668 is a geometric proxy (lateral shift >= 2 m in 4 s) that also
  counts curves. So `--lc-cf` reuses the 2 589 ok rows (no duplicate pool; the rows are drawn straight from cf/rows/<shard>.npz with the
  token's own Store row), and the extension is the pairing: the real row is drawn in the same batch. No new rows tonight.
- D6 (weights). `--lane-lam 10` (per row-metre, summed / non-anchor rows: a third of RBSG's 30 per pool row, now on every row), over-turn
  weight = `--lane-lam` per radian, `--lc-lam` = `--lane-lam`, `--lc-k` 8 at batch 128 (4 at the pilot's 64). On-k 7 at batch 64.
- D7 (init conversion). An old-layout branch (RSG-L18-s0) initialises the ordered head with its first 24 position embeddings and a zero
  scalar map, so the converted init differs from the base only by the dropped far points (step-0 dev ADE logged).

## 2. Which reading gates which flag

| reading (turn-night.md section 1, other agents) | supports -> | does not support -> |
|:--|:--|:--|
| A (progress: turn zeros by near 0-20 m memory coverage) | `--rm-order` on | off |
| B (lane: lateral offset to the route in the 20 m before the entry, completed vs failed) | `--lane-lam` on | off; the lever is steering amplitude / speed (separate line) |
| B's lane-change share of the failures (route asked a lane change) | `--lc-cf` on | off |

The pilot below runs all three on; the gates are read per flag only through the pilot table (no per-flag pilot unless a gate fails).

## 3. Gates (offline, `rf_v2gate.py`, pilot = 1 000 steps, seed 0, batch 64, gentle schedule, init RSG-L18-s0; base = RSG-L18-s0)

| gate | rows | line |
|:--|:--|:--|
| G-rely | dev junction rows (route enters a junction <= 40 m) | mean ADE(memory masked) - ADE(on) >= 0.01 m |
| G-swap | on1_CFdev junction rows (j0 / j1) | median share of the needed move at 1.5-2 s >= 0.50 and 4 s pose on the counterfactual side >= 70 % |
| G-ST | dev straight rows +-3.5 m shift; dev curve rows (no junction, turn >= 15 deg) straight continuation | median lateral move at 1.5-2 s <= 0.10 m, both |
| G-LC | on1_CFdev lc rows | median |y(2 s) stay - y(2 s) real| >= 0.50 m |
| G-real | dev rows | ADE (memory on) <= base + 0.01 m |
| G-over | dev junction rows | share with plan heading at 2 s > route heading + 10 deg toward the inside <= base's share |

Pass on all six -> the flag set may go to `train` (2 seeds) once the user names the base and flags. G-swap < 0.20 at 1 000 steps -> the route
input is not read: stop the training change (R-CF's stop rule). G-real / G-ST / G-over fail -> no full training with that flag set;
diagnose with single-flag 1 000-step pilots (each <= 0.1 card-h) before proposing.

## 4. Held 460 lines (after `train`, 2 seeds, 18 decisions; `rf_v2.sh report`)

- H1 no harm: zeros and at-fault zeros (452 paired, seed mean) vs the base and vs RSG-L18: log-clustered 95 % CI upper bound <= +5 each.
- H2 turns: completion on the 24 held >= 60 deg turns (rf_follow), seed mean >= base + 3 turns; right-turn-inside offroad zeros not above the
  base + 2 (the RBSG failure).
- H3 public guard later (1491 / 10 decisions, separate submission after H1 and H2): zeros and at-fault vs the base, CI upper bound <= +5
  (RBSG2 failed here with +32.5 zeros); then PAI 281 on Tokyo.

## 5. Cost (training 0.5 card-h / 10 000 steps; held read 0.7 card-h / checkpoint, closed-loop.md)

feature build CPU only (~5 min, 24 cores); bit identity 2 x 50 steps ~0.15 card-h (setup dominated); pilot 1 000 steps x 64 ~0.05 card-h +
gate ~0.1; total tonight <= 0.4 card-h. Full round: 2 seeds x 2 500 steps ~0.25 card-h + gates 0.2 + held 2 x 0.7 = ~1.9 card-h; public
guard ~2 card-h per checkpoint.

## 6. Stop rules

- Bit identity with every flag off fails (any tensor differs after 50 eager steps vs the code before route v2): nothing runs until fixed.
- Pilot gate rules in section 3; full training only on the user's word.
- Held: H1 fails on either seed mean -> not adopted, whatever H2 says.
