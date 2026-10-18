# LEAD-AWARE HOLD: latch only without a lead, serve natively behind one (offline, 2026-10-18)

CPU only, stored reads, no closed-loop read, no new serve switch. Code: `scripts/lead_aware.py` (one run, about a minute). Full tables:
[lead_aware_tables.md](lead_aware_tables.md), with per-window rows on the box at `$DATA_DIR/runs/alpasim_proxy/arena/lead_aware/`.

**Question.** The wait-set cut showed that P0-retention models wait behind a stationary lead and launch when it moves, while no model waits
without a lead. Does a lead-aware `stop3` pay? The rule: latch only when no lead is present; with a lead, serve the model natively. Variants:
with a lead, latch but release on the plan arc or the lead outputs, or use a shorter leash. Reference: RSG-L18-s0 + stop3 on the 2 228 held
arena windows (0.9438).

**Result: no.** No rule with a serve-time lead flag reaches +0.005 for any model. The plain rule ("lead -> native") loses for every model,
even with the ground-truth flag:

| model | rule A, oracle flag | rule A, serve flag | best variant, oracle | best variant, serve |
|---|--:|--:|--:|--:|
| RSG-L18-s0 | **-0.0253** [-0.0413, -0.0115] | **-0.0499** [-0.0778, -0.0238] | E 2.5 m + C' +0.0086 (UB) | E 2.5 m +0.0038 |
| JTN-L18-s0 (vs its own stop3) | -0.0031 [-0.0088, +0.0019] | -0.0183 [-0.0323, -0.0054] | C' +0.0040 (UB) | C' +0.0038 (UB) |
| JTN-L18-s0 (vs RSG + stop3) | -0.0043 [-0.0135, +0.0072] | -0.0196 [-0.0349, -0.0029] | C' +0.0027 (UB) | C' +0.0025 (UB) |
| NR1F-L18-s0 (no-lead latches credited with RSG + stop3) | -0.0140 [-0.0277, -0.0024] | -0.0430 [-0.0700, -0.0177] | B >= 2 m +0.0003 | B >= 2 m +0.0003 |

Per held window, paired against RSG + stop3 unless stated, 95 % CI from a log-clustered bootstrap (`jevdrive.stats.paired`). "UB" = the
release is credited with the native read's score, i.e. an in-sample upper bound with no release delay. The one number above +0.005 needs the
oracle flag, an in-sample release rule and 11 AF-front windows (9 from one log); see section 4.

## Method and approximations

- **Windows.** The 2 228 held arena windows of hold2 / residual (route-fail dropped). Strata follow the arena's definitions. The latch is
  taken from RSG's stop3 calls: 1 004 static, 142 stop, 1 082 free. JTN's own stop3 latch is identical on every window, because the latch
  reads the force-GT speed at the 0.5 s handoff.
- **Lead flags.**
  - Oracle: the set's `lead` flag. It means a logged vehicle within 1.8 m of the human's heading line, ahead of the front bumper, gap <= 30 m,
    at the window start (`wait_set.py` `window_stats`).
  - Serve: RSG's lead head at the handoff call, filtered probability > 0.5 and gap <= 30 m. It comes from the `JEV_HOLD=lead` read's hold
    log, the only read that logged lead outputs. Before the handoff the ego is on the human's pose, so the value is policy-independent for
    RSG's checkpoint. **JTN and NR1F were given RSG's head, not their own.** The calls parquets of those reads carry no lead outputs.
- **Crediting rule A** (and A', B). A session served natively from the handoff is the native read of that window, so switched windows get
  the native read's score. This is exact up to closed-loop re-read noise. On the 1 082 free windows, RSG native vs stop3 differ in 22
  windows, summed +0.00; JTN in 9, summed +0.02.
- **Interactions.** Offline crediting ignores interaction effects within a window: the native read is exactly what the lead-aware driver
  would have served there. Prior offline estimates matched closed loop twice within about 0.002. Only the release rules (C, C') and the
  shorter leash (E) are model-based estimates.
- **Release rules (C, C').** The released window is credited with the native score, which ignores the late release (an upper bound). The
  release signal pairs each model's own held-plan arc (stop3 calls) with RSG's lead outputs at the same call time. Those outputs come from
  the `lead` read, whose ego is held at 0 m, not leashed at 3 m.
- **Shorter leash (E).** The estimator is diag.md (b) / residual.md section 4's. That estimator was calibrated at 3 m at 91.9 estimated vs
  99.7 read on launches, so it is pessimistic on launch progress.
- **NR1F.** There is no stop3 read. **Its latched windows are credited with RSG + stop3's score** (no-lead latches in every rule, and all
  latches in its "model + stop3" row). Free windows use NR1F's own native read. The literal variant (RSG + stop3 on every no-lead window,
  NR1F native on every lead window) gives -0.0124 (oracle) / -0.0431 (serve).
- **Single reads.** One seed and one read per arm, so the CIs are over windows and logs, not seeds.

## 1. Where the switched windows sit (static latches, oracle flag)

| lead | class | windows | RSG native | RSG stop3 | JTN off | JTN stop3 | NR1F native | native - stop3, sum: RSG / JTN / NR1F |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| no | launch, human >= 5 m | 52 | 0.883 | 0.491 | 0.773 | 0.437 | 0.847 | +20.4 / +17.5 / +18.5 |
| no | still end | 323 | 0.084 | 0.994 | 0.536 | 0.997 | 0.105 | -294.0 / -149.0 / -287.0 |
| yes | launch, human >= 5 m | 30 | 0.925 | 0.508 | 0.771 | 0.433 | 0.877 | **+12.5 / +10.1 / +11.1** |
| yes | still end | 484 | 0.835 (80 zeros) | 0.975 (12) | 0.965 (17) | 0.998 (1) | 0.895 (51) | **-68.0 / -16.0 / -39.0** |
| yes | launch < 5 m / moving end | 64 | | | | | | -1.0 / -1.0 / -1.0 |

(NR1F's stop3 column is RSG + stop3's.) Two facts settle it:

- **The lead-aware rule can only touch 12.5 of the launch cost.** Of the 82 latched launches with the human >= 5 m, 52 have no lead, and they
  carry 20 of the roughly 33 points. Those are the cases the brief expects no model to discriminate.
- **Behind a stationary lead the hold still beats every model by a wide margin.** Native zero rate on the 484 lead-present still-end windows:
  RSG 16.5 %, NR1F 10.5 %, JTN 3.5 %. The hold's rate is 0.2-2.5 %. Every native zero there costs a full point, while a launch gains about
  0.4. Break-even would need a native zero rate around 12.5 / 484 = 2.6 %. JTN (3.5 %) is the only model close.

**Lead-present still-end windows that turn into zeros when served natively** (oracle flag, 484 windows):

| model | native zeros | corridor | AF-front | offroad | stop3 zeros there |
|---|--:|--:|--:|--:|--:|
| RSG-L18-s0 | 80 | 58 | 21 | 1 | 12 (AF-front 11, offroad 1) |
| JTN-L18-s0 | 17 | 13 | 3 | 1 | 1 (offroad) |
| NR1F-L18-s0 | 51 | 35 | 15 | 1 | 12 (RSG + stop3 credit) |

With the serve flag, 471 of the 484 are flagged: 67 / 11 / 44 native zeros.

## 2. The serve-time flag

On the 1 004 static latches the head flags 675 and the oracle 578. Both flag 565, the oracle alone 13, the head alone **110**. At a
standstill the head reads the lead's gap well (median 3.8 m vs 4.2 m logged, p = 1.00). The 110 head-only windows have no logged in-lane
vehicle within 30 m. Head gap there is a median 17.6 m, so these are probably vehicles farther ahead or in the next lane. 76 of the 110 are
still-end windows. RSG served natively there loses 67.7 points, the whole difference between the oracle and serve rows. Tightening the head
gap to <= 20 m helps but stays negative: RSG -0.0310, JTN -0.0096 (vs its stop3), NR1F -0.0232.

## 3. All rules, sum and per window vs RSG + stop3

Columns:
- switched: windows served or credited differently from the model's stop3.
- increment: the rule against the same model's stop3 (the lead-aware effect alone).
- zeros, AF-front: totals over the 2 228 windows. RSG + stop3 has 64 zeros, 12 AF-front, 8 corridor.

| model | rule | switched | sum | per window [CI] | increment (sum, per window [CI]) | new / fixed zeros | zeros | AF-front |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| RSG | A: static latch with a lead -> native [oracle] | 578 | -56.5 | -0.0253 [-0.0413, -0.0115] | same | 69 / 0 | 133 | 23 |
| RSG | A [serve] | 675 | -111.2 | -0.0499 [-0.0778, -0.0238] | same | 130 / 0 | 194 | 19 |
| RSG | A': static or stop latch with a lead -> native [oracle] | 650 | -56.0 | -0.0251 [-0.0411, -0.0113] | same | 69 / 0 | 133 | 23 |
| RSG | B: A only if the handoff plan's 2 s arc >= 1 m [oracle] | 9 | +1.4 | +0.0006 [-0.0010, +0.0022] | same | 2 / 0 | 66 | 12 |
| RSG | B, arc >= 0.5 m [oracle] | 275 | -54.1 | -0.0243 [-0.0406, -0.0104] | same | 65 / 0 | 129 | 23 |
| RSG | C: lead -> release when the held plan's arc >= 2 m (UB) [oracle] | 173 | -56.5 | -0.0253 | same | 69 / 0 | 133 | 23 |
| RSG | C': ... and the head has the lead moving > 1 m/s at that call (UB) [oracle] | 49 | +8.3 | +0.0037 [+0.0000, +0.0069] | same | 4 / 0 | 68 | 12 |
| RSG | C' (UB) [serve] | 71 | -3.2 | -0.0014 [-0.0105, +0.0064] | same | 18 / 0 | 82 | 12 |
| RSG | D: lead -> the real `JEV_HOLD=lead` read (full hold, head release) [oracle] | 578 | -3.9 | -0.0018 [-0.0078, +0.0078] | same | 0 / 11 | 53 | 1 |
| RSG | D [serve] | 675 | -9.2 | -0.0041 [-0.0106, +0.0060] | same | 1 / 13 | 52 | 1 |
| JTN | model + stop3 | 0 | -2.8 | -0.0013 [-0.0087, +0.0096] | 0 | 20 / 24 | 60 | 2 |
| JTN | A [oracle] | 578 | -9.7 | -0.0043 [-0.0135, +0.0072] | -6.9, -0.0031 [-0.0088, +0.0019] | 36 / 23 | 77 | 6 |
| JTN | A [serve] | 675 | -43.6 | -0.0196 [-0.0349, -0.0029] | -40.8, -0.0183 [-0.0323, -0.0054] | 76 / 23 | 117 | 6 |
| JTN | B, arc >= 0.5 m [serve] | 29 | +4.4 | +0.0020 [-0.0065, +0.0136] | +7.2, +0.0032 [+0.0006, +0.0062] | 23 / 24 | 63 | 2 |
| JTN | C' (UB) [oracle] | 36 | +6.0 | +0.0027 [-0.0053, +0.0141] | +8.8, +0.0040 [+0.0020, +0.0059] | 21 / 24 | 61 | 2 |
| JTN | C' (UB) [serve] | 43 | +5.6 | +0.0025 [-0.0058, +0.0142] | +8.4, +0.0038 [+0.0006, +0.0071] | 23 / 24 | 63 | 2 |
| NR1F | model + stop3 (latches credited RSG + stop3) | 0 | -2.3 | -0.0010 [-0.0037, +0.0017] | 0 | 15 / 8 | 71 | 13 |
| NR1F | A [oracle] | 578 | -31.2 | -0.0140 [-0.0277, -0.0024] | -28.9, -0.0130 | 56 / 9 | 111 | 18 |
| NR1F | A [serve] | 675 | -95.8 | -0.0430 [-0.0700, -0.0177] | -93.5, -0.0420 | 127 / 9 | 182 | 17 |
| NR1F | B, arc >= 1 m [oracle] | 11 | -4.9 | -0.0022 [-0.0056, +0.0012] | -2.6 | 20 / 8 | 76 | 13 |

C' is not available for NR1F because there is no held plan. Two more rows from the tables file:
- **The C' grid** (held arc 0-3 m x lead speed 0.5-3 m/s, in-sample best). It tops out at +8.3 (RSG, oracle), +2.6 (RSG, serve) and +8.9
  (JTN, oracle) / +8.4 (JTN, serve) over the model's stop3. All are below +0.005 per window even as upper bounds.
- **The handoff plan carries almost no go signal.** At 0.5 s from a standstill, the native plan's 2 s arc is >= 1 m in only 7-23 of the
  578-675 lead latches.

**By stratum.** All rules change only the static and launch strata; every other stratum equals the model's stop3. For rule A, the summed
difference vs RSG + stop3 splits as follows:
- RSG: static -68.0 / launch +11.5 (oracle), static -127.0 / launch +15.8 (serve).
- JTN: static -3.0 / launch +2.5 (oracle), static -40.0 / launch +5.6 (serve). JTN's own stop3 already carries static +13.0, launch -6.6,
  braking -6.0, other -3.5, turn +0.8, cruise -0.4 against RSG + stop3.
- NR1F: static -39.0 / launch +8.8 (oracle), static -106.0 / launch +11.2 (serve). Its free-window model difference: braking -2.7, other
  +3.8, turn -2.8, cruise +0.7.

Per-stratum CIs and zero counts are in the tables file.

## 4. The one positive piece: a shorter leash behind a lead (E)

The real `lead` read (D) fixes the 11 AF-front zeros that stop3 makes in lead-present still-end windows. Those cars are latched behind a
stopped lead at a logged 2.36-2.88 m and roll into it under the 3 m leash (diag.md (b)). Shortening the static leash only when a lead is
present keeps that fix and spares the 52 no-lead launches:

| model | flag | leash with a lead | AF-front avoided | launch progress | increment vs its stop3 | E + C' (UB) |
|---|---|--:|--:|--:|--:|--:|
| RSG | oracle | 2.5 m | 11 | -1.6 | +9.4 (+0.0042) | +19.2 (+0.0086) |
| RSG | serve | 2.5 m | 11 | -2.6 | +8.4 (+0.0038) | +7.0 (+0.0031) |
| RSG | serve | 2.0 m | 11 | -5.2 | +5.8 (+0.0026) | +6.1 (+0.0027) |
| JTN | serve | 2.5 m | 0 | -2.1 | -2.1 (-0.0009) | +7.4 (+0.0033) |

Notes on these rows:
- **Against a shorter leash for every latch.** diag.md (b) estimated that at +5.7 (2.5 m), so gating by the lead adds about +2.7 points.
  Serve-time E stays below the bar (+0.0038).
- **Fragility.** All of E's gain is the same 11 windows from 3 logs, 9 of them from one log. While held, the head reads that lead at about 21 m
  (diag.md (b)). The 30 m serve gate flags all 11, but a tighter gap gate would lose them.
- **JTN gains nothing from E.** Its stop3 has no AF-front zeros there.
- **The only number above +0.005 is E + C' on the oracle flag.** It stacks an in-sample release rule (an upper bound) on the 11-window AF-front
  fix. With the serve flag it drops to +0.0031, because C' releases wrongly on head-only windows.

## Verdict against the +0.005 bar

- **Lead-aware latching (latch only without a lead, serve natively behind one) does not pay for any model.** Oracle flag: RSG -0.025,
  NR1F -0.014, JTN -0.003 vs its own stop3. Serve flag: -0.050 / -0.043 / -0.018.
  - The models do discriminate behind a lead, but not well enough. Waiting 83-97 % of the time against the hold's 98-100 % costs more on
    484 still-end windows than the 30 lead-present launches return (about 12 points).
  - The serve-time flag makes it worse: the head sees a lead in 110 windows the log does not, and 76 of them end still.
- **Release variants (latch, then release on the plan arc and the lead outputs) stay below the bar even as in-sample upper bounds:** RSG
  +0.0037 (oracle) / +0.0012 (serve grid best), JTN +0.0040 / +0.0038 over its own stop3.
- **The shorter leash behind a lead is the best serve-time variant at +0.0038 (RSG, 2.5 m).** It is below the bar and rests on 11 windows,
  mostly from one log.
- **No prereg supplement, no new `JEV_HOLD` mode, no closed-loop read.** RSG-L18-s0 + stop3 stays the nuPlan configuration.

## Caveats

- **Offline crediting.** The A / A' / B rows are exact up to re-read noise, which is small: 22 of 1 082 free windows differ, summed +0.00.
  The C / C' rows are upper bounds (no release delay, an in-sample grid). The E rows rest on diag.md's estimator, which is pessimistic on
  launch progress. Interaction effects inside a window are ignored. Prior offline estimates matched closed loop twice within about 0.002.
- **Borrowed lead head.** JTN and NR1F were judged with RSG's lead head at the handoff (and RSG's head for C'), not their own. The `lead` read's
  lead outputs come from a fully held ego, while stop3 lets the ego roll up to 3 m.
- **NR1F's credit.** NR1F's latched no-lead windows carry RSG + stop3's score. Against RSG + stop3, its rule-A loss comes from its own native reads
  on the lead windows plus its free-window model difference (-2.3), not from that credit.
- **Oracle flag geometry.** The flag is measured at the window start (about 0.5 s before the handoff) within 30 m and 1.8 m laterally. A
  different gate moves the 13 / 110 disagreements, but not the still-end arithmetic of section 1.
- **Sample.** One read per model, one seed, held windows only. JTN's own stop3 is -0.0013 [-0.0087, +0.0096] against RSG's, which is not
  separable.
