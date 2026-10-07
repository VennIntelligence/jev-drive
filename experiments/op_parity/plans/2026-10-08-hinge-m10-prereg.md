# op_parity hinge-m10 pre-registration (2026-10-08, written before any SH30M10 arm is scored)

## Question

Decision 172: in the pilot-scale lambda x margin scan the only point that could change "SH30 (lambda 30 / margin 0.5) stays" was lambda 30 / margin 1.0
(EPDMS +0.32 [-0.01, +0.68] vs SHP, seed 0, 1 seed; EP -0.10 [-0.14, -0.06]; straight EPDMS -0.31; raw-plan out-of-bounds -0.56 pp). SH30's full-scale gain was
larger than its pilot gain, so pilot rankings may not carry. Settle that one point at full scale.

## Setting (one arm, no other hyperparameter changes)

- Arm `SH30M10-F-s{0,1}`: exactly the SH30 recipe (`strong_hinge_chain.sh`: 12 navtrain shards, `navsim/op-parity-full`, 10 000 steps x 128, warmup 300,
  seeds 0 / 1, `--eval-every 1000`, P2 warp host) with `--hinge-lam 30 --hinge-margin 1.0`.
- Baselines (not retrained): `SH30-F-s{0,1}` (the comparison), `P2H10-F-s{0,1}` (reference).
- Scoring through `python -m jevdrive.bench` as SH30 was: navtest, navhard (`@gimm`), HUGSIM 64 `spec_plan_smooth` (`derot_all64.txt`).
- Extra reads: raw-plan vs replay-only out-of-bounds (`rh.py proxy` + `geomtab`, all tokens), straight (S5) EPDMS, EP, inside-cut and cannot-make-turn rates on
  turns > 20 / > 45 deg (`turn_oracle.py replay` + `report`).
- Statistics: log-cluster paired bootstrap (B 4 000, `jevdrive.stats`), seed means, 95% CI.

## Rule (written before scoring)

SH30M10 replaces SH30 only if ALL of:
1. navtest EPDMS paired CI vs SH30 excludes 0 on the positive side (lower bound > 0);
2. EP and straight (S5) EPDMS show no cost vs SH30: CI upper bound >= 0 for both;
3. HUGSIM 64 HD is not worse than SH30: CI upper bound >= 0.

Otherwise SH30 stays. navhard, the out-of-bounds split and turn rates are reported, not gating.

## Budget and constraints

About 4 card-hours (2 x ~45-60 min training in parallel + scoring). All GPU / CPU jobs through the pool; a job killed with rc 137 is logged (time, job id,
`free -g`, RSS trend if available) and resumed. Does not touch research/decisions.md or HTML.
