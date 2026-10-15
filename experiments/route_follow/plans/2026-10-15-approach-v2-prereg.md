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
