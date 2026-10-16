# STACK dose pilots (prereg `plans/2026-10-16-stack-prereg.md` section 3, Amendment 1)

Arm = RSG-L18 recipe from RB0-F-s0, pool on1_RCVL, speed upper bounds only (`--env-lam 1 --env-lo-scale 0`), Lagrangian multipliers on agent / road / envU
(tau = level x human log-future rate: agent 0.0005, road 0.1265; envU auto), imitation dropout, one-sided minimal-correction preference, reflex module;
consequence hinge OFF (Amendment 1; C2 later failed: pooled own-dev AUC 0.773 < 0.85, `checks.md`). 1 000 steps, seed 0, ~4.1 it/s (rx: one extra
constraint-side backward). Flags `flags.json`; per-pilot gate files `gate_<tag>.json`.

## Registered gates (all four FAIL)

| pilot | level / pref W / lr | dev ADE (<= 0.545) | G-rely (d_junction, CI low) | equal-arc turning vs RSG-L18-s0 (>= -0.3 deg) | 4 s arc d (m) | served-plan contact, dev own rows (drop >= 30 %) |
|:--|:--|--:|:--|:--|--:|:--|
| STKAL01 | A / 1 / 0.1 | 0.5530 FAIL | false (0.0085, -0.0007) | -1.38 [-1.70, -1.00] FAIL | -2.5 | 0.103 -> 0.124 (+20 %, d +0.021 [+0.011, +0.030]) FAIL |
| STKAL03 | A / 1 / 0.3 | 0.5438 pass | false (0.0103, -0.0063) | -1.13 [-1.66, -0.70] FAIL | -3.0 | 0.103 -> 0.133 (+29 %) FAIL |
| STKBL01 | B / 3 / 0.1 | 0.5609 FAIL | true (0.0161, 0.0072) | -0.61 [-0.92, -0.30] FAIL | -3.8 | 0.103 -> 0.159 (+54 %) FAIL |
| STKBL03 | B / 3 / 0.3 | 0.5624 FAIL | false (0.0110, -0.0002) | -0.34 [-0.57, -0.08] FAIL | -3.9 | 0.103 -> 0.157 (+53 %) FAIL |

Contact rows: 3 618 dev own-state rows of on1_L18RSG, 4 held-out logs (CIs are log-clustered over 4 logs: wide in truth). No pilot passed, so no held
460 read was submitted.

## Multipliers (end of training)

| pilot | lam agent (lam0 30) | lam road (30) | lam envU (30) | agent violation EMA (own rows) | road EMA | pref: rows with a term | pref labels (lateral / gain / arc) |
|:--|--:|--:|--:|--:|--:|:--|:--|
| STKAL01 | 300 (cap) | 85 | 9.5 | 0.095 | 0.114 | ~4 % of pool rows (smoke) | 359 / 178 / 81 |
| STKAL03 | 300 (cap) | 58 | 3.2 | 0.099 | 0.111 | | 389 / 189 / 74 |
| STKBL01 | 300 (cap) | 300 (cap) | 300 (cap) | 0.120 | 0.111 | | 363 / 156 / 63 |
| STKBL03 | 300 (cap) | 300 (cap) | 300 (cap) | 0.113 | 0.106 | | 372 / 138 / 53 |

The agent target (human rate 0.0005) is unreachable on own states (~0.07 at the start), so lam_agent hit the 10x cap within tens of steps in every
pilot, and the own-row agent violation rate ROSE during training (0.07 at step 60 -> 0.09-0.12 at step 1 000) instead of falling. distill_ho on the
own rows rose from 0.0006 to 0.04-0.07.

## Where the harm comes from (descriptive, offline, no training)

Same checkpoints read with the reflex module switched off at serve (`JEV_REFLEX_OFF=1`, valid 0; the switch is bit-exact by the serving checks):

| pilot | reflex on: agent / road / any contact, 4 s arc median | reflex off: agent / road / any, arc | equal-arc turning, reflex off |
|:--|:--|:--|:--|
| RSG-L18-s0 (base) | 0.072 / 0.041 / 0.103, 17.6 m | | |
| STKAL01 | 0.097 / 0.038 / 0.130, 12.8 m | 0.072 / 0.046 / 0.115, 17.9 m | -1.55 [-2.01, -1.07] |
| STKAL03 | 0.109 / 0.033 / 0.139, 11.9 m | 0.071 / 0.042 / 0.110, 17.4 m | -0.96 [-1.39, -0.53] |
| STKBL01 | 0.138 / 0.032 / 0.165, 10.0 m | 0.072 / 0.040 / 0.110, 17.4 m | -0.88 [-1.31, -0.44] |
| STKBL03 | 0.135 / 0.032 / 0.163, 10.3 m | 0.073 / 0.038 / 0.108, 17.1 m | -0.62 [-1.01, -0.20] |

- **The reflex module shortens the plan by 27-43 % (4 s arc 17.6 -> 10-12.8 m)** through its context-bias path (the bounded 0.5 m residual alone cannot do
  that). The new agent contacts are overwhelmingly the log-replayed follower running into the slowed ego: of the 134-293 new contacts per pilot,
  83-86 % have the object's centre less than 2 m ahead of the rear axle (median 1.7 m ahead, |lat| 0.4 m, at 3.5 s): rear / overlap contacts of a
  non-reactive replay, the slowness failure (d315 / d320) in another form. Road contact falls 7-22 % with the module on.
  Its parameters take no distillation gradient by design (constraint side only), so nothing anchored the non-plan heads on the own rows it changes.
- **Without the module, the constraint training changes nothing that matters for contact** (any-contact 0.108-0.115 vs 0.103, agent equal,
  road -7 to +12 %) **but still costs turning** (equal-arc -0.6 to -1.5 deg, all CIs below 0) and, at three of four doses, dev ADE / route reliance.
  The lateral preference labels (about 60 % of the chosen candidates are lateral offsets) are a one-sided geometry target on own states; the turning
  cost has the same sign as decisions 286 / 303 / 321.

Cost: 4 pilots ~0.3 job-h each (2.5 card-h with loading and forward reads), the reflex-off reads 0.2 card-h.
