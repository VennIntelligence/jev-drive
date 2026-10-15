# APPROACH COURSE v2 (APP2): the approach term scoped to its purpose (pre-registration)

2026-10-15 evening, written before any v2 row, pilot, held or PAI read. Design by the coordinator (research/next-round/turn-night.md
section 8, line C), from decision 303: APP1X (approach rows + three terms at W 1, 2 500 gentle steps on RSG-L18-s0) moved the PAI entry
state to the human's (entry speed 7.38 -> 4.80 m/s, inward offset at entry - 20 m -0.44 -> -0.12 m) and doubled strict-36 completion
(11 vs 5), but harmed everything else: held zeros +44 (offroad 41 vs 17, entering 0.17 m further inside), PAI all-281 -0.062, handoff
spins 29 vs 2, control-75 zeros 6 vs 0. Reading: the term acts outside its purpose: at the handoff (a slow plan at the log's speed, d285),
on straights (slowing, drifting), and its lateral pull enters too far inside. APP2 keeps the mechanism and removes those three.
v1 prereg: `2026-10-15-approach-course-prereg.md` (rows, labels, losses, pool on1_APP, Amendments 1-4). Code:
`experiments/onpol_nuplan/lib/on_approach.py` (v2 branch), `rm_train.py --app-v2`, `rf_apppool.py v2`, `rf_appgate.py --guards-only`,
`rf_app.sh v2rows | ident | pilot2 | train2 | report2`.

## 1. Rows and labels

Same pool `on1_APP` (17 429 RSG-L18 S rows + 3 864 approach rows: 3 091 turn, 773 straight negatives), same labels and the same
`--on-weight` (~7 approach rows per batch of 128). New sidecar `on1_APP/approach_v2.npz` (`rf_apppool.py v2`) per approach row:

- `t_hand`: seconds since the handoff = the row's decision index x the decision period (nuPlan rollouts 0.5 s, decision k = the row name's
  last field, checked against meta `k`; PAI driver 0.1 s, k checked against the cache).
- `side`, `tgt` (section 2), `Hvh` = the human's speed on the label grid for every row (negatives too; `rf_apppool.py label` now keeps it;
  the relabel must reproduce every v1 key bit for bit, else stop), `l3n` = negative rows whose human is >= 2 m/s over the 26 m window of
  the virtual entry (the v1 VH_MIN rule: a human stopped for traffic gives no speed target).

## 2. Losses (`rm_train.py --app-lam W --app-v2`; with `--app-v2` off on_approach is v1 exactly)

1. **Handoff.** Every approach term is 0 on rows with `t_hand < 1.5 s`: the first 3 decisions after the handoff in the nuPlan rollouts.
   PAI decisions are 0.1 s, so the same 1.5 s is k < 15 there (this also covers the PAI cold-start rows, k <= 13). Routine decision: the
   coordinator's "first 3 decisions" is read in time, because the 3 decisions were meant as the d285 handoff transient and 3 PAI
   decisions would be only 0.3 s.
2. **Speed term L3 only inside the turn window.** Turn rows with the speed label (l3) and the ego itself inside the window (d_e <= 26 m),
   plan segments at distance-to-entry in [0, 26] m (as v1). One-sided relu(v - v_h - 0.5 m/s), unchanged.
   **Straight negatives:** a symmetric term relu(|v - v_h| - 0.5 m/s) to the human's speed on the same window of the virtual entry
   (d_e <= 26 m, rows with l3n), weight equal to L3's (`app3n`): the straights teach "drive at the human's speed", never "slow down".
   Longitudinal gradient only (lon_only), as L3.
3. **Lateral terms L1 / L2.** Dead zone 0.5 m (v1 0.3 m). Target = the human's entry lateral position, with its inward part clipped at
   LANE_IN = 0.75 m inside the route centreline (3.5 m lane half-width 1.75 m minus a 1.0 m car half-width): target offset from the human
   path = -side x max(lat_h - 0.75, 0), lat_h = the human entry point's signed offset to the route (+ toward the inside). The target is
   therefore never further inside than the human, and the outward side is not moved (an outward human stays the target). L2 blends the
   ego's own offset e0 into that target over 0 .. d_e. Routine decision: the clip is one-sided (inward only) because moving an outward
   human's target inward would put it beyond the human's own position. The heading part of L1 (5 deg) is unchanged. Negatives: L1 / L2 to
   the lane centre with the 0.5 m dead zone.
4. **Over-turn guard (`app4`, arm b's rule).** relu(side x (psi_2s - h(s_2s)) - 10 deg) in radians: psi_2s = the plan's 2.0 s heading
   output, h = the heading of the row's target path at the 2 s pose's station; side = the side of the human's turn ahead; turn rows only;
   gradient through the heading only; per-row weight equal to L1's (arm b: over-lam = lane-lam). Routine decision: the target path H (the
   human path, which takes the route's turn, route_check <= 2.5 m) stands in for the route, because the PAI route labels carry the 40 m
   near gap (v1 Amendment 1) and would leave the guard silent on most PAI rows.

Scale as v1: each term summed over the pool rows / n_imit x ho_w (3) x 10 W.

## 3. Arms, pilot gate

- **Arms.** APP2W05-L18-s0 (W 0.5) and APP2W1-L18-s0 (W 1). Seed 0 each, init RSG-L18-s0, gentle schedule, batch mix as APP1X.
- **Pilots.** APP2P05 / APP2P1-L18-s0: 1 000 steps, batch 64, on-k 10 (the v1 pilot recipe).
- **The pilot gate is the guards only** (the offline dev rows cannot test the approach, d303). Pass needs all of:
  - G-real: dev ADE <= base + 0.015 m. Relaxed from 0.01 by the coordinator, stated before any read: APP1X's +0.020 was a harm proxy, the
    v1 pilots +0.021 / +0.037.
  - G-ST, G-over, G-rely: rf_v2gate's lines, unchanged.
- **Reported, not gating.**
  - A-lat, A-speed, A-neg.
  - neg_dv: the plan's signed speed minus the human's on the dev straight negatives. It must not drop; it is read, not gated.
  - over10: the share of dev turn rows whose 2 s heading points > 10 deg further into the turn than the human path.
- A weight that fails its pilot gate is not trained. Both fail: report and stop.

## 4. Full training and held 460

`rf_app.sh train2 <W> 0` for each weight that passed: 2 500 gentle steps (batch 128, on-k 20), the guard gates again on the final
checkpoint (reported), the held 460 / 18-decision read (6 chunks), `report2` (held.py, ol_confirm paired vs RSG-L18-s0, rf_follow, rf_appheld).

- **H1 (no harm).** Zeros and at-fault zeros vs RSG-L18-s0, log-clustered 95 % CI upper bound <= +5. Must pass.
- **Descriptive.** Completion on the 24 held turns, and rf_appheld's entry speed and inward offset table (APP1X: +0.17 m inside; read
  whether v2 removes it).
- **Choice for PAI.** Among the weights that pass H1, the one with the lower dev ADE cost (G-real value of the full checkpoint minus the
  base's) goes to PAI 281. Neither passes: report and stop.

## 5. PAI 281 (Tokyo, when the coordinator gives the slot)

Command: `rf_v2pai.sh run <tag>` then `rf_v2pai.sh read <tag>`, gains 20, paired vs r281/rsg; the mechanism read is `rf_appread.py` as in
Amendment 4 of the v1 prereg. Every line must hold:

| line | threshold | APP1X | RSG-L18-s0 |
|:--|:--|--:|--:|
| strict-36 completion | >= 11 | 11 | 5 |
| all-281 mean score, paired | not down: 95 % CI upper bound >= 0 | -0.062 [-0.108, -0.016] | |
| handoff spins | <= 5 | 29 | 2 |
| control-75 zeros | <= 1 | 6 | 0 |

Reported: the runway-22 mechanism (entry speed, speed excess, inward offset at entry - 20 m), completion on the 22 / 14, exam-72 zeros,
the shape gate, zeros by kind, and new non-turn zeros.

## 6. Identity, cost, stop rules

- **Bit identity, before any pilot.** 50 eager steps (`rf_app.sh ident`), compared to the code before this change (BASE_REV = the parent
  commit), with on_train ident (every trained tensor) and both dev ADE logs:
  - `2off`: every flag off.
  - `2app1`: v1 `--app-lam 1` on on1_APP, `--app-v2` off. This shows the v1 path of on_approach is untouched.
  - Either fails: nothing runs until it is fixed.
- **Cost.**
  - Relabel: ~10 min of CPU.
  - Identity: 4 x 50 steps, ~0.2 card-h.
  - Pilots: 2 x (1 000 x 64 + gates), ~0.3 card-h.
  - Trainings: 2 x 2 500 steps, ~0.5 card-h, plus gates ~0.4.
  - Held reads: 2 x 6 chunks, ~1.4 card-h.
  - Box total: ~3 card-h, on the three cards in parallel: ~2 h wall.
  - PAI 281: ~2.3 h on both Tokyo cards.
- No seed 1 and no public guard unless the coordinator asks after PAI.

## Amendment 1 (rows, identity, pilot gate; 2026-10-15 ~13:25, before any full training or held read)

- **Rows** ([results/approach/v2/rows.md](../results/approach/v2/rows.md)). The relabel reproduces all 18 v1 keys of both lab files bit for
  bit.
  - Past the handoff: 3 252 / 3 864 approach rows (turn rows: nuPlan 1 935 / 2 379, PAI 629 / 712).
  - Speed rows: 1 620 -> 958. nuPlan 1 255 -> 719 and PAI 365 -> 239 (ego <= 26 m before the entry and past the handoff).
  - Clipped lateral targets: 210. nuPlan 206 (median clip 0.37 m), PAI 4.
  - Straight negatives with the symmetric speed term: 444 in the window (human median 8.4 / 15.2 m/s nuPlan / PAI); 604 past the handoff
    have l3n.
- **Bit identity** (50 eager steps vs `9c64aaa0`, the parent commit): `2off` (every flag off) and `2app1` (v1 `--app-lam 1` on on1_APP)
  both have 109 net + 97 adapter tensors identical, and their dev ADE logs are equal (0.5284 -> 0.5320 and 0.5284 -> 0.5362). PASS
  (`results/approach/v2/ident2*.json`).
- **Pilot gate** (1 000 steps, batch 64, seed 0, vs RSG-L18-s0; `results/approach/v2/gate/APP2P{05,1}-L18-s0.md`):

  | pilot | G-real (line 0.5435) | G-ST | G-over | G-rely | gate | A-lat drop | A-speed drop | neg_dv change | over10 share change |
  |:--|--:|:--|--:|--:|:--|--:|--:|--:|--:|
  | W 0.5 | 0.5403 PASS | 0.005 / 0.029 | 0.0067 | 0.0197 | **PASS** | +0.005 | -0.32 | +0.08 [-0.02, +0.16] | -0.007 |
  | W 1 | 0.5508 FAIL | 0.005 / 0.027 | 0.0056 | 0.0176 | FAIL | -0.046 | -0.39 | +0.15 [+0.04, +0.23] | -0.007 |

  - The base's speed excess on the dev rows is 0.006 m/s, so the A-speed "drops" are noise on a near-zero base.
  - neg_dv is the plan speed minus the human's on the dev straights (base -1.95 m/s). It rises, i.e. no slow-down was learnt on the
    straights.
  - By section 3, **W 1 is not trained**. W 0.5 trains in full (APP2W05-L18-s0, launched 13:24, held read and report2 chained).

## Amendment 2 (full training and held 460: H1 fails, stop; 2026-10-15 ~14:00)

- **APP2W05-L18-s0** (W 0.5, 2 500 gentle steps; `results/approach/v2/gate/APP2W05-L18-s0.md`). Dev ADE 0.5284 -> 0.5537 (+0.025; at 1 000
  steps 0.5617). The final checkpoint's guard reads are reported, not gating: G-real FAIL at the 0.015 line, G-rely 0.0100 FAIL, G-ST and
  G-over pass. On the dev straights the plan speeds up toward the human: neg_dv -1.96 -> -1.43 m/s, +0.52 [+0.39, +0.72].
- **Held 460 H1 FAILS** (`results/approach/v2/held/confirm.md`, held18, 452 paired, 39 logs, vs RSG-L18-s0):

  | read | RSG-L18-s0 | APP2W05 | paired difference [95 % CI] | APP1X (v1) |
  |:--|--:|--:|:--|:--|
  | zeros | 26 | 70 | **+44.0 [+29.4, +59.5]** | +44.0 [+30.4, +58.3] |
  | at-fault zeros | 21 | 45 | +24.0 [+12.5, +35.4] | +26.0 [+15.5, +36.9] |
  | collision / offroad / corridor zeros | 4 / 17 / 6 | 11 / 34 / 27 | | 6 / 41 / 25 |
  | mean score | | | -0.042 [-0.079, -0.006] | -0.126 [-0.156, -0.098] |
  | GT distance (m) | 1.135 | 1.808 | +0.673 [+0.526, +0.833] | +0.116 |

- **Held turns** (`held/entry.md`, `held/follow.md`). Completion 14 / 24 vs 16 / 24 (APP1X 8). Entry speed +0.10 [+0.04, +0.16] m/s. The
  inward offset at the entry is still +0.16 [+0.08, +0.23] m (median 0.58 vs 0.34 m), so v2 does not remove the inward entry seen in
  APP1X (+0.17).
- **Where the harm is** (descriptive, per_scene.csv of both reports). APP2W05 has 50 new zeros over RSG (21 offroad, 20 corridor, 10
  collision at fault). 35 of them are also new zeros of APP1X (52). The rollouts travel on average 5.7 m further than RSG's over the 18
  decisions.
  - The harm is common to both term designs. It is not specific to the handoff, straight-slowdown or lateral-pull parts that v2 changed.
  - Candidate common causes (not tested): the pool mix (~7 approach rows of 128 that carry the S hinges without imitation; PAI rows have no
    road raster) or any approach-row gradient at all. The diagnostic would be a full 2 500-step W 0 control on on1_APP. It is not
    registered and was not run.
- **Verdict by section 4.** W 0.5 fails H1, and W 1 was not trained (pilot G-real FAIL). No weight passes H1: **stop, no PAI read**.

## Amendment 3 (coordinator, 2026-10-15 ~14:10, after Amendment 2, before any W 0 read): the W 0 control

The question: does the held harm shared by APP1X and APP2W05 come from the pool mix or from the approach gradient?

- **Run.** APP0C-L18-s0 = 2 500 gentle steps on RSG-L18-s0, seed 0 (batch 128, on-k 20), pool on1_APP with the same --on-weight, and
  `--app-lam 0`: the approach rows are present as S-hinge-only rows and get no approach gradient (`rf_app.sh control 0`).
- **Gates.** The guard gates are reported, not gating (`results/approach/w0/gate/`).
- **Held 460 read**, paired vs RSG-L18-s0 and vs APP2W05-L18-s0 (`rf_app.sh report0`; the report now pairs every base line of a tag):
  - zeros, at-fault zeros, the collision / offroad / corridor split
  - completion on the 24 held turns and the rf_appheld entry table
- **Reading rule.**
  - W 0 reproduces the +44 zeros: the harm is the pool mix (approach rows as S-only rows), and the approach terms are exonerated.
  - W 0 is level with RSG: the approach gradient, of either design, is the harm.
- Either way: report and stop. The verdict on the approach line is the coordinator's.

### Amendment 3 read (2026-10-15 ~14:20): W 0 is level with RSG, so the approach gradient is the harm

- **APP0C-L18-s0 training** (`results/approach/w0/gate/APP0C-L18-s0.md`). Dev ADE 0.5288 (+0.0004). Every guard passes: G-ST 0.005 / 0.028,
  G-over 0.0067, G-rely 0.0234.
- **Held 460** (`results/approach/w0/held/confirm.md`, held18, 452 paired, 39 logs):

  | read | RSG-L18-s0 | APP0C (W 0) | APP2W05 | W 0 - RSG [95 % CI] | W 0 - APP2W05 [95 % CI] |
  |:--|--:|--:|--:|:--|:--|
  | zeros | 26 | 24 | 70 | -2.0 [-7.0, +2.7] | -46.0 [-63.7, -29.5] |
  | at-fault zeros | 21 | 20 | 45 | -1.0 [-4.6, +2.2] | -25.0 [-36.8, -13.3] |
  | collision / offroad / corridor zeros | 4 / 17 / 6 | 4 / 16 / 5 | 11 / 34 / 27 | | |
  | mean score | | | | +0.015 [+0.003, +0.027] | +0.057 [+0.019, +0.098] |
  | GT distance (m) | 1.135 | 1.165 | 1.808 | +0.030 [-0.015, +0.073] | -0.643 [-0.798, -0.504] |
  | held turns completed | 16 / 24 | 19 / 24 | 14 / 24 | | |
  | inward offset at the entry (median m) | 0.34 | 0.34 | 0.58 | -0.01 [-0.03, +0.00] | -0.17 [-0.24, -0.09] |

- **Reading, by the rule above.** W 0 is level with RSG: the zeros CI includes 0, and the score and the turns are, if anything, slightly
  better. The pool mix (approach rows as S-only rows, ~7 of 128 per batch) is harmless. The held harm of APP1X and APP2W05 comes from the
  approach gradient itself, in both term designs, together with the inward entry (+0.16 / +0.17 m). Stop; the line's verdict is the
  coordinator's.

## Amendment 4 (coordinator, 2026-10-15 ~14:25, before any seed-1 read): seed 1 of the W 0 control

Seed 0 of the W 0 control read 19 / 24 held turns and a mean score of +0.015 [+0.003, +0.027], with zeros -2. Seed 1 tests whether that
is a cheap gain from "approach-window rows as S-only rows" or seed noise.
- **Run.** APP0C-L18-s1 = the same recipe on RSG-L18-s1 (pool on1_APP, `--app-lam 0`, 2 500 gentle steps, seed 1), PRIO 8.
  `rf_app.sh control 1`, then `rf_app.sh report0 APP0C-L18-s0 APP0C-L18-s1`.
- **Held 460 reads.**
  - Seed 1 paired vs RSG-L18-s1.
  - The 2-seed mean APP0C-L18 vs RSG-L18 (ol_confirm --mean): zeros, at-fault zeros, mean score.
  - 24-turn completion per seed and summed over 48.
  - The entry table.
- **Reading.**
  - Seed 1 repeats seed 0 (score up, zeros not up, turns not down): a candidate for a public guard (the coordinator's call).
  - Otherwise: seed noise.
- Report and stop.
