# PAI-PARTIALS: how much of NTV0-s0 + ACC + GAP's partial-score PCS loss the serve layer can reach (2026-10-20)

Prereg: [../plans/2026-10-20-pai-partials-sweep-prereg.md](../plans/2026-10-20-pai-partials-sweep-prereg.md), written after part 1 and
before any sweep read. Code: `scripts/pai_partials.py` (`decomp` on the box, `value` / `sweep` on the Mac) and `scripts/pai_partials.sh`
(smoke, submit, sum). New serve knobs: `serve_fix.py` `JEV_ACC_VSET`, `JEV_ACC_REL`, `JEV_ACC_ALAT` (unset = bit-identical).
Generated tables: [pai_partials/value.md](pai_partials/value.md), [pai_partials/sweep.md](pai_partials/sweep.md), per-arm paired reads
in [pai_partials/pai281/](pai_partials/pai281/), per-scene split in [pai_partials/decomp.csv](pai_partials/decomp.csv). Box root
`$DATA_DIR/runs/alpasim_proxy/pai_partials/`.

**Read every PCS here as an ordering, not a level.** The arms were chosen and scored on the same 281 scenes the ruler is fitted on, and
every value lies above the ruler's top anchor (2089). That anchor's leave-one-out error is +108 (decision 392). Only order and paired
differences are evidence.

## Short answer

1. **The partials are not serve-removed speed.** In the 36 partial scenes, 95 % of the speed deficit against the human (1 915 m in all)
   is speed the model's own plan never asked for. Only 88 m (4.6 %) is speed the plan had and the serve took away:
   - real-lead cuts 31 m;
   - curve limit 42 m;
   - phantom-lead cuts 7 m.

   That serve-removed share is worth **about +26 PCS**, against +168 / +52 if every rear-hit / slow partial went to 1.
2. **The rest can only be reached by serving faster than the model asks, and the ruler does not pay for that trade.**
   - Raising the ACC set speed above the plan (V1 / V1b) does fix the partials: rear-hit partials 26 -> 11-13, slow 11 -> 4-5, mean
     +0.005 / +0.007.
   - It also adds 4-5 zeros (corridor, offroad, AF-front 3 -> 5).
   - On the default ruler it lands at -9 / -7 PCS against A0 and -34 / -32 against holdlead.
   - Relaxing the brake guard (V2) and tighter gap + curve limit (V3) are no better. V3 breaks the AF guard.
   - The joint arm (J) is last: -73.
3. **Recommended PAI serve: NTV0-L18-s0 + ACC + GAP + holdlead**, the stored HL read (decision 370):
   - PCS 2221, +25 [-21, +55] over A0;
   - first in 10 of 13 pool variants;
   - passes the guard (AF-front + AF-side 5, the limit is A0's 3 + 2).

   No new arm beats it. Serve env:
   `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_VHIST=holdlead`, controller gains `{"rel_front_steering_angle_weight": 10}`.
   The new knobs stay unset.
4. **PAI reads are deterministic.** R0, a re-read of A0, is bit-identical on every scene both reads scored. Its PCS still differs by
   +9 because A0 had driver RPC errors on a few scenes that R0 scored, and the complete-matrix rule masks different items. So the ruler's
   "noise" between reads of one config is about 9 PCS, all of it infrastructure. The spread between serve configs (tens of PCS, bootstrap
   CIs of +-100) is chaotic closed-loop sensitivity, not read noise.

## 1. Where the partial deficit is (A0 = stored NTV0-L18-s0 + ACC + GAP)

Per inference decision, from the handoff to the rear hit (rear-hit) or the rollout's end (slow):
- **deficit** = the human's speed 1.5 s ahead (time-matched; the following car is log replay) minus the served 1.5 s speed, counted only
  where positive;
- **serve-removed** = the part the plan's own 1.5 s speed had and the serve took away, labelled by the binding ACC / lead term;
- the rest is the plan's. It is split by the ACC guard state: the plan brakes >= 1 m/s^2 (brake guard), the plan's 2 s arc is < 2 m
  (stop guard), a standing launch, or cruise.

Integrated x 0.1 s gives metres.

| term | rear-hit (25 scenes in the ruler, 10.4 points) m | slow (11, 2.2 points) m |
|---|--:|--:|
| deficit | 895 | 1020 |
| serve-removed: real-lead cut | 22 | 9 |
| serve-removed: phantom-lead cut | 3 | 4 |
| serve-removed: curve limit | 7 | 35 |
| serve-removed: set / ramp | 6 | 8 |
| plan: cruises below the human | 581 | 578 |
| plan: brakes where the human does not (brake guard) | 227 | 317 |
| plan: stop guard + standing launch | 78 | 69 |
| (plan metres with the ACC set speed below the human's speed) | 807 | 862 |
| (ACC serving faster than the plan, up to the human) | 145 | 71 |

- **No scene is serve-dominated.** All 37 have more plan metres than serve-removed metres.
- **The binding cap is the ACC set speed.** It is max(handoff speed, the plan's own 2-4 s speed), so it rises only when the plan rises.
  The human speeds up past it while the plan does not.
- This is decision 376's under-planning seen from the partials. It also matches decision 381's split (serve 3.8 of 10.4 points by
  decision share): the lead does win many decisions, but by small margins below the plan.

**PCS worth on the d392 ruler** (paimix, NTV0's row, items fixed, MAP; [value.md](pai_partials/value.md)):

| group | scenes | points | PCS if -> 1 | PCS of the serve-removed share |
|---|--:|--:|--:|--:|
| rear-hit, plan brakes / stops / launches late | 14 | 6.6 | +74 | +21.5 |
| rear-hit, plan cruises below the human | 11 | 3.8 | +35 | +1.8 |
| slow, plan brakes / stops / launches late | 4 | 0.7 | +7 | +0.8 |
| slow, plan cruises below the human | 7 | 1.4 | +41 | +1.5 |
| all rear-hit | 25 | 10.4 | +168 | +23.4 |
| all slow | 11 | 2.2 | +52 | +2.3 |

## 2. Sweep (one seed each, PAI 281, A0's 8-chunk table, g10, ALPASIM_FAST=1)

| arm | serve added to A0 | mean | zeros | AF-front / side | corridor / offroad | rear-hit / slow partials | PCS (paimix) [95 %] | PCS - A0 paired [95 %] | mean - A0 paired [95 %] | pool variants |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| A0 | (ref) | 0.7443 | 59 | 3 / 0 | 44 / 12 | 26 / 11 | 2195 [2138, 2299] | 0 | 0 | 2130-2260 |
| R0 | re-read | 0.7452 | 59 | 3 / 0 | 44 / 12 | 26 / 11 | 2204 [2147, 2320] | +9 [-4, +38] | +0.0000 (bit-identical on shared scenes) | 2127-2272 |
| **HL** | holdlead (stored, d370) | 0.7456 | 61 | 4 / 1 | 42 / 14 | 21 / 8 | **2221** [2155, 2298] | **+25** [-21, +55] | +0.0004 [-0.014, +0.013] | 2151-2332 |
| V1 | HL + VSET 1.1 | 0.7498 | 64 | 5 / 0 | 45 / 14 | 13 / 5 | 2186 [2077, 2347] | -9 [-109, +133] | +0.0046 [-0.020, +0.029] | 2116-2350 |
| V1b | HL + VSET 1.2 | 0.7537 | 63 | 5 / 0 | 45 / 12 | 11 / 4 | 2189 [2084, 2297] | -7 [-116, +81] | +0.0074 [-0.017, +0.032] | 2122-2370 |
| V2 | HL + brake guard 2 m/s^2 | 0.7423 | 62 | 3 / 1 | 45 / 13 | 21 / 7 | 2191 [2093, 2294] | -5 [-116, +68] | -0.0029 [-0.026, +0.019] | 2121-2284 |
| V3 | HL + GAP 0.5,2.0 + ALAT 3.0 | 0.7333 | 65 | 4 / **2** (guard fail: 6 > 5) | 42 / 17 | 20 / 7 | 2202 [2127, 2282] | +6 [-69, +72] | -0.0119 [-0.035, +0.009] | 2137-2291 |
| J | HL + VSET 1.1 + REL 2,2 + GAP 0.5,2 + ALAT 3 | 0.7506 | 65 | 4 / 1 | 48 / 12 | 11 / 4 | 2122 [2021, 2341] | -73 [-208, +140] | +0.0055 [-0.023, +0.034] | 2079-2225 |

How to read it:
- **HL is first** in paimix, its two seeds, garage x0.5, band x0.5 / x2, go-straight x0.5 / x2 and low-as-gs: 10 of 13 variants.
- **V1b / V1 lead only in three variants**: allours, dedup and 1PL. Those are the variants that weigh partials more and our family's shared
  zeros less.
  - So "fewer partials for more zeros" is a bet on how the board weighs partials against zeros, and the default ruler says no.
  - V1b also has the best mean (+0.0074). If the user prefers the mean line, V1b is the alternative. It passes the AF guard (5 = 3 + 2).
- The set-speed knob's new zeros are corridor / offroad on moving scenes, three of them formerly rear-hit partials, plus two AF-front
  (8 new, 4 fixed for V1b). That is the cost of serving faster than the plan where the plan was right to be slow.
- **V3 (the serve-removed terms) is the test of part 1's caveat.** If cuts lowered later plans in closed loop (d376), removing them should
  have helped more than the static split says. It did not: rear-hit 26 -> 20 (HL alone 21), mean -0.012, and two AF-side zeros. The
  static split's small serve share holds.

## 3. Recommendation

- **PAI probe image:** NTV0-L18-s0 with `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_VHIST=holdlead`, steering weight 10.
  `JEV_ACC_VSET`, `JEV_ACC_REL` and `JEV_ACC_ALAT` stay unset.
- This changes decision 370's call, which kept holdlead off on the mean line. Decision 392's ruler and this sweep both put it first on
  PCS. Its paired CI over plain ACC + GAP, [-21, +55], still includes 0.
- **The serve layer is exhausted for the partials.** What remains is the plan's own speed: it cruises below the human, and it brakes or
  launches late where the human does not. That is a model lever: the under-planning of decisions 376 / 381. It is not a serve knob.

GPU: 49 pool jobs (1 smoke and 48 chunks, no retries), 18.1 stack-hours, about 6 card-hours at 3 stacks per card. Wall time 1.4 h. One
RPC error in V1b (scene masked).
