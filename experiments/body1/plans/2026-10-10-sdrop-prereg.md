# S-DROP: full-scale drop-one ablation of `P2H10S` (pre-registration, 2026-10-10)

Lane S-DROP, topic body1. Pushed before any job of this lane is submitted. An ablation only: no new loss term, no closed loop
(AlpaSim / HUGSIM), no change to any existing checkpoint. Inputs: decisions 228 to 230, 232 to 234, 236;
[2026-10-10-body1-prereg.md](2026-10-10-body1-prereg.md) Amendments 4 to 7; results/shape_pilot.md, route_pilot.md, other_boards.md.

## 1. Questions

`P2H10S` = `P2H10` + (A) agent hinge on the student's own plan, (B) hinge-only off-track rows `ot1` / `yr1` / `bd4` (13 of 128 per batch),
(C) drivable hinge of those rows on the road-and-lane raster; the new hinges' pose gradient has no along-heading component.

1. Which ingredient brings the navhard gain of `P2H10S` (4 seeds +1.91 [+0.13, +3.74], stage 2 +2.92 [+1.02, +4.85]; seeds 0 + 1:
   +2.22 [+0.25, +4.36], stage 2 +3.33 [+1.27, +5.48]; decision 236)?
2. At full scale, does the outward widening at turns over 45 deg come only from the road hinge on the hinge-only rows (pilot drop-ones of
   the full-gradient recipe: "A on on-log rows only" 55 against 54, "B + C without A" 71 against 54; decision 234)?

Not re-tested: that the shape-only switch restores the speed profile (232), that the widening exists at full scale (234 point 1: W2 58 / 65 /
59 / 65 navtest tokens against 45 / 46 / 47 / 49), that item C's raster choice does not change the gradient (230 point 6, Amendment 6 item 3 (5)).

## 2. Arms

Trainer `experiments/body1/scripts/bd4_train.py train` at the pushed checkout, unchanged; all 12 shards, 10 000 steps, batch 128, seeds 0 and 1.
Reference command (stored `P2H10S-F-s{0,1}`, from its meta.json): `--ho ot1:4,yr1:4,bd4:5 --agent-lam 10 --ho-w 3 --ho-excl navsim/body1-val-logs --shape`.

| Arm (tag) | Dropped | Flags against the reference | What is left of the new terms |
|:--|:--|:--|:--|
| `P2H10S-noA-F-s{0,1}` | A | `--agent-lam 0` | hinge-only rows with the road hinge (w = 3), no agent hinge anywhere |
| `P2H10S-noB-F-s{0,1}` | B (and with it C) | `--ho ""` (hence no `--ho-w`, `--ho-excl`) | agent hinge on the imitation rows of the train logs only; 128 normal rows per batch instead of 115 + 13 |
| `P2H10S-noC-F-s{0,1}` | C | `--road-lam 0` | hinge-only rows carrying the agent hinge only (w = 3), agent hinge on imitation rows |

Two things fixed here:
- **C lives on B's rows, so the design is nested.** Dropping B removes C as well; "minus C" is therefore read as "no road hinge on the
  hinge-only rows" (`--road-lam 0`), not as "the same hinge on the NAVSIM raster" (Amendment 4 item 11's wording). Question 2 names the
  road hinge on hinge-only rows as the term, and the raster swap is settled as gradient-neutral. The raster swap is not run.
- The base (`P2H10-F-s{0,1}`) and the full recipe (`P2H10S-F-s{0,1}`) are the stored checkpoints; nothing of them is retrained.
  Seeds 2 and 3 of both exist and are used only as the seed range of the references.

## 3. Checks before the full runs (stage 1 of the staged launch)

1. **Trainer reproduces the stored setup.** `git log` shows no change to `bd4_train.py`, `lib/loss43.py`, `pp_train.py`, `ot_rows.py` or the hinge
   libs since 3a853121 (Amendment 7 code, itself bit-identical to the `P2H10S` code path on 60 steps: `results/route/ident_a7_shape.json`).
   Re-checked at today's checkout: 60 steps with the reference flags on shard s2 (`SDROP-IDENT-S`) against the stored run `B43A7-IDENT-S1`
   by `bd4_train.py ident` (scalars and every weight bit for bit). If it is not identical the lane stops and reports, unless a second
   run of the same command also differs from the first (run-to-run nondeterminism of the box), in which case config identity alone is
   reported and the size of the difference stated.
2. **Config identity.** For each full run, `meta.json` `config` and `config.body1` against the stored `P2H10S-F-s<k>`: the only differing
   keys may be the tag and the arm's dropped flag(s) of the table above (written to `results/sdrop/config_identity.json`).
3. **Smoke, one per arm** (60 steps, shard s2): finishes, loss finite, and the logged terms are the arm's: noA has no `loss/agent*`
   scalar and has `loss/road_ho`; noB has `loss/agent` and no `*_ho` scalar; noC has `loss/agent_ho` and no `loss/road_ho`.
4. **After the full runs, before any read:** 10 000 steps, `dev_ade` <= 0.56 m (reference 0.541 to 0.546), `dev_drift_off` <= 0.30 (0.041 to
   0.045), hinge-only rows per batch 13 / 0 / 13. A run failing this is rerun unchanged once (infrastructure) or the lane stops.

All six full runs are then submitted together (N independent runs = N pool jobs). Each is about 23 min, under the 1 h staged-launch
threshold; stage 1 above is the "1 unit" look.

## 4. Reads (one read each, after all six runs passed check 4)

Every read is arm against the base and arm against the full recipe of the same seed; the pooled number averages the two seeds per unit
first and resamples logs (B 10 000, `jevdrive.stats.paired(groups=log)`; navhard: stage-1 log of the group, through `jevdrive.bench report`).

| Read | Tool (unchanged) | Units |
|:--|:--|:--|
| navhard two-stage, G frame: combined, stage 1, stage 2 | `jevdrive.bench run / report --bench navhard`, `<tag>@gimm` | 225 groups, 76 logs |
| navtest EPDMS, sub-scores, turn bins | `jevdrive.bench run / report --bench navtest` | 12 146 tokens |
| inside-cut and cannot-make-the-turn at > 45 deg (and EPDMS / DAC by stratum) | `turn_oracle.py replay` + `bd4_g3d.py` | 1 517 tokens > 45 deg |
| W2 (own plan more than 2 m outside the logged path within 4 s), mean signed lateral at 4 s, 4 m tube exits | `prog_ol.py states` dumps + `route_ol.py` / `lib/route.py` | navtest > 45 deg tokens (1 517); hold on-log turn rows (1 173) and all four families (3 954) |
| 4 s arc-length ratio (pooled, open, lead; hold families) | `prog_ol.py` | navtest 12 146 tokens; hold 30 083 states |
| own-plan agent-contact and boundary rate (margin < -0.20 m, NAVSIM raster), pooled and by family `log` / `ot1` / `yr1` / `bd4`, > 45 deg | `bd4_g3.py --set hold / navtest` with both references | hold 30 083 states, 121 logs; navtest 12 146 tokens |

`scripts/sdrop_report.py` (new, CPU, no model) collects these into `results/sdrop/summary.{md,json}`: it recomputes the two-seed paired
differences from the per-state dumps and per-unit bench files and applies the lines below; it must agree with the readers' own per-seed
numbers (asserted for the contact rates).

## 5. How each question is read (fixed now)

**Question 1 (navhard).** Let G2 = stage 2 of `P2H10S-F` minus `P2H10-F`, seeds 0 + 1 (stored: +3.33). For an arm X:
- **the dropped ingredient carries the gain** if X - `P2H10S` on stage 2 is <= -G2 / 2 and its interval excludes 0 (upper bound < 0);
- **the dropped ingredient is not needed for the gain** if X - `P2H10` on stage 2 is >= G2 / 2 and its interval excludes 0 (lower bound > 0);
- otherwise **unresolved** for that arm; the share kept, (X - base) / G2, is reported with both intervals.
Combined and stage 1 are reported with the same two differences and carry no line. A combined two-seed mean is called outside the base's
seed range only beyond 31.25 to 32.28 (four base seeds) and beyond the same-seed differences of base pairs (2-seed against 4-seed mean
-0.29 [-0.74, +0.16]). Reading of the nested design: noB carries and noC carries = the road hinge on hinge-only rows; noB carries, noC and
noA not needed = the hinge-only rows with either hinge; noA carries = the agent hinge; no arm carries and no arm is "not needed" = the gain
is not attributable at two seeds (the expected outcome if the terms are redundant or the power is short: the stored S - base interval on
stage 2 has a half-width of 2.1, G2 / 2 is 1.67).

**Question 2 (widening).** W2 on the 1 517 navtest tokens over 45 deg, per seed, against the base of the same seed (stored: base 45 / 46,
`P2H10S` 58 / 65); tolerance +5 tokens as in Amendment 7 (the four base seeds differ by at most 4).
- an arm is **not wide** if W2 <= base + 5 on both seeds, **wide** if W2 >= base + 6 on both seeds, **mixed** otherwise;
- **"only the road hinge on hinge-only rows" holds** iff noA is wide, noC is not wide and noB is not wide;
- it **fails** if noC is wide (the agent hinge on hinge-only rows widens without the road hinge) or if noA is not wide (the road hinge
  alone does not widen at full scale with the shape-only gradient);
- any mixed arm = unresolved for that arm, reported as such.
Reported next to it, no line: W2 on hold on-log turn rows (stored 40 / 38 against 48 / 52), the paired difference of the mean signed
lateral at 4 s (X - base, X - `P2H10S`), W1 / W3 counts, by turn side.

**Every other read** is descriptive with one common rule, so that the follow-up lane gets an attribution table: for an effect E of
`P2H10S` over the base whose stored interval excludes 0 (navtest EPDMS, > 45 deg EPDMS, DAC failure, inside-cut, hold agent and boundary
rates), the dropped ingredient "carries" E if X - `P2H10S` gives back at least half of E with an interval excluding 0, "is not needed" if
X - base keeps at least half of E with an interval excluding 0, else unresolved. The arc ratio is read against Amendment 6's lines
(navtest pooled and open >= 0.995, lead >= 0.990, hold families >= 0.980) as a guard: an arm that misses one is reported as shortening.

Lines are not changed after a read. No arm is promoted, served or run in a closed loop whatever it reads.

## 6. Cost, before running

| Step | Jobs | Card-hours | Declared per job (measured on the `P2H10S` runs) |
|:--|--:|--:|:--|
| identity + 3 smokes (60 steps, shard s2) | 4 | 0.3 | 20 GB VRAM, 4 cores, 30 GiB (pilot: 15.4 GB, 12 GiB) |
| full runs | 6 | 2.4 (23 min each alone; up to 4.5 if six together slow to 45 min) | 32 GB VRAM, 4 cores, 56 GiB (29.4 GB, 1.3 cores, 51.7 GiB) |
| `bd4_g3.py` hold + navtest | 12 | 0.6 | 8 GB, 2 cores, 16 GiB (5.7 GB, 13 GiB) |
| `prog_ol.py` dumps (10 checkpoints each) | 2 | 0.3 | 16 GB, 2 cores, 20 GiB (8.8 GB with 4 checkpoints) |
| bench navtest (plans + devkit shards) | 6 | 0.6 | sized by `jevdrive.bench` |
| bench navhard `@gimm` | 6 | 1.2 | sized by `jevdrive.bench` |
| turn-oracle replay, reports | 2 | 0 (CPU, 48 cores for minutes) | 0.5 GB, 48 cores, 64 GiB |
| **total** | | **about 5.5, at most 8** | under the lane's stop line of 20 |

Wall about 1.5 h. Disk about 0.5 GB (six checkpoints, two plan dumps). Everything through the GPU pool, owner `body1-sdrop`; the chain
(`scripts/sdrop_chain.sh`, tmux `jev:sdrop`) only submits and waits on DONE / ERROR files. CPU jobs take pool allocations; the CORR0 lane's
files and windows are not touched.

## 7. Limits known before the read

- Two seeds per arm; the base's seed range is taken from four seeds of the base and of the full recipe.
- The hold logs of body1 have been read at pilot scale by four variants of this lane and at full scale by `P2H10B`, `P2H10S` (four seeds)
  and the route look; this read is the seventh look at them. navtest tokens have been read by the same arms. navhard was read once
  (decision 236) with the full recipe only; reading three more arms on it spends that.
- All contact / widening / arc numbers are open loop on the student's own plan. Whether W2 predicts closed-loop corridor zeros is untested
  (decision 234).
- noB changes the batch composition (128 normal rows) as well as removing the rows; noB removes C with B.
- The drop-ones are of the shape-only recipe; the pilot drop-ones (decision 232 point 3) were of the full-gradient recipe, so a
  disagreement with them can be scale or gradient form.
