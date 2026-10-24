# CB2: clean B2 on nuPlan (round 9, 2026-10-25), decision 482

**Question.** Can one training run from the shipped Cinque weights, with on-policy rows only from its own lineage, tie B2
(`STANDFMXH-L18-s1`, decision 391) on nuPlan? Recipe (user, 10-25): P1L's recipe (own rb0 S rows from step 10 000, NTV0's L18 lr
schedule at T) + B2's nuPlan terms STAND + FREE-M (W 0.001) = **P1N**; then one DAgger iteration on P1N's own nuPlan states with the b3
teacher = **D1N**. No trainval rows, no PAI rows, no DAgger iteration 2.

**Answer: no, stopped at the registered early-stop gate E1 after stage 1.** P1N reads 0.784 on the arena held split (fixed scorer)
against 0.941 for the B2-s1 re-read (-0.157 [-0.217, -0.099]); the E1 line was 0.881. D1N was not trained. Cause, measured: STAND
had almost nothing to act on. STAND fires only on S rows where the visited state stands and the human stands. In B2's draw those
rows are 14.5 % of the S draw (4.65 a batch), almost all from the practice-start static rows. Here those rows are dropped by the
decision 377 / 385 static rule (user instruction), so STAND-eligible rows are **0.2 % of the S draw (0.06 a batch)**. P1N therefore
rolls out of static windows like NTV0 / D3 (arena corridor zeros 363 vs 25).

Plan (Chinese, registration): [../../plans/2026-10-25-cb2.md](../../plans/2026-10-25-cb2.md). Lane: [cb2.sh](../../scripts/cb2.sh),
[cb2.py](../../scripts/cb2.py), [cb2v2.py](../../scripts/cb2v2.py). Box: `$DATA_DIR/runs/round9/cb2/`. Verdict: [verdict.md](verdict.md).

**Correction (plan amendment 2, 10-25).** The static-row drop was an error in the coordinator's brief, not a user decision; B2 kept its
practice-start static rows. The P1N above stays as the "no-static" reference. P1N' (`CB2P1NB-F-s0`) uses B2's S-row rule on the same
pool (every live self row equal, static rows kept): STAND-firing rows are 23.3 % of the S draw, 7.45 a batch (B2 14.5 % / 4.65). Results
of that lane are added below when they land.

**Box sync incident.** During the first records sync I ran `git checkout -- .` in the box repo. It reverted an uncommitted local change
to `experiments/round9/scripts/diag.sh` (not mine, owner unknown); that change is lost. The box is synced with `git pull` only from now
on (`git stash` if it ever has local changes; never `checkout -- .` or `reset --hard`).

## Step 0 (offline, before registration): DAgger rows on human-standing states

The b3 teacher's longitudinal is the student's own served arc. Measured on existing nuPlan rows (RB0-F-s0 onpol-train, D3, JTN
practice starts): of the DAgger candidates, 4 % are human-standing / -stopping on onpol-train and 10-25 % on practice starts; on
2 / 3 of them the label ends beyond the STAND cap, by a median of 7 m (onpol) to 10-17 m (practice), mostly with no lead (line
stops). Gate G0 (keep >= 80 % of rows and of the lateral correction, leave <= 10 % of the conflict): R1 (drop standing rows) and
R2 (drop standing + stopping rows) fail; R3 (re-place the label of those rows at the arc lengths of the reachable human profile
`fut`, path unchanged) passes on every pool and was registered. Tables: [creep2.md](creep2.md) (R1-R3), [creep.md](creep.md).
Not used in the end (stage 2 not run).

## Gates passed (stage 1)

- Practice-starts roll by RB0-F-s0: c0 config check vs JTN's c0 passed (105 / 105 scored, mean 0.962) ([c0chk1.json](c0chk1.json));
  all 7 shards pass R1 / R2 and were merged into `on1_JTPR`.
- Draw CB2S ([w_CB2S.json](w_CB2S.json)): 13 793 rb0 rows + 21 937 moving practice rows (13 132 static rows dropped), uniform, so
  practice rows are 61 % of the S draw.
- G1 identity: P1L's argv on on1_JTPR / CB2S0 is bit-identical to the stored PCL1L-F-s0 for 85 / 85 scalars to step 100
  ([g1_ident.json](g1_ident.json)); argv diff = the registered words ([diff.json](diff.json)); smoke ([smoke1.json](smoke1.json));
  S / imitation gradient ratio 0.624 (band 0.21-1.04); post (dev ADE, lr) passed.

## B2 re-reads (the paired baseline under the fixed scorer)

Same serve, ASL kept for every zero, failed or 4 m-tripped rollout. Old-scorer reproduction of the stored reads: arena s1 2 240 /
2 246 windows equal, s0 2 227 / 2 246 (max diff 0.006, mean diff ~0), held 460 s1 460 / 460, s0 451 / 460; means and zero classes
identical ([arena_repro_b2.md](arena_repro_b2.md), [held460_repro_b2.md](held460_repro_b2.md)). Frozen PCS of the re-reads 1398 / 1369 = stored.
Fixed-scorer rescoring (`cb2v2.py`: the offline evaluator reproduces every stored score; only ground-truth corridor metrics change):
B2-s1 arena 0.9358 -> 0.9412 (24 windows change, 18 up / 6 down), s0 0.9373 -> 0.9431; held 460 +0.002.

## Results (P1N = CB2P1N-F-s0, one seed)

| read | scorer | P1N | B2-s1 re-read | P1N - B2-s1 [95 % CI] | zeros P1N / B2-s1 |
|:--|:--|--:|--:|--:|--:|
| arena held 2 228, native | fixed (primary) | 0.7844 | 0.9412 | -0.1567 [-0.2168, -0.0993] | 474 / 89 |
| arena held 2 228, native | old | 0.7835 | 0.9358 | -0.1522 [-0.2108, -0.0966] | 476 / 101 |
| held 460 @18 | fixed (primary) | 0.8976 | 0.8549 | +0.0427 [+0.0179, +0.0730] | 44 / 40 |
| held 460 @18 | old | 0.8910 | 0.8527 | +0.0383 [+0.0150, +0.0681] | 47 / 41 |

Against B2-s0: arena -0.159 [-0.218, -0.103], held 460 +0.052 [+0.029, +0.079] (fixed). References, old scorer, stored reads:
arena vs NTV0 +0.014 [-0.001, +0.029], vs D3 -0.004 [-0.014, +0.005]; held 460 vs NTV0 +0.004, vs D3 +0.026 [+0.000, +0.055].
Frozen board-mix PCS (old scorer, reference only; decision 481): P1N 1131, B2-s1 1398, B2-s0 1369.

Arena zero classes (fixed scorer): corridor 363 vs 25 / 19 (B2 s1 / s0), new corridor zeros under the fix 25 vs 6 / 3, AF-front 15 +
AF-front+corridor 17 vs 2 / 3, offroad 54 vs 53 / 51. Partials: P1N has almost no launch failures (no-launch 1 vs 27 / 30, rear-hit 1 vs
9 / 10). Held 460 zeros: offroad 20 / 19, corridor 10 / 8, AF-front 5 / 3. Tables: [arena_v2_s1.md](arena_v2_s1.md),
[arena_old_s1.md](arena_old_s1.md), [held460_v2_s1.md](held460_v2_s1.md), [held460_old_s1.md](held460_old_s1.md); per-window
[arena_windows_v2_s1.csv.gz](arena_windows_v2_s1.csv.gz) / [old](arena_windows_old_s1.csv.gz). Easy-item table not built (the
lane stopped before the D1N read it was registered for).

## Reading

- P1N is an NTV0 / D3-type model on nuPlan: it launches well and is better than B2 on held 460, but it does not hold still in static
  windows. STAND, as carried over, is inert under the static-row rule: the rule removes exactly the rows STAND works on.
- A next attempt has to give STAND its rows from this lineage: keep the practice-start static rows for STAND only (agent hinge off on
  them, which decision 377 named as the alternative fix for the lateral push), or B2's STAND-R rolling rows. Not run; needs the user.
- Cost: 35.7 GPU job-h + 1.4 CPU job-h (practice roll 46 blocks, B2 re-reads 2 x 24 blocks, P1N training and 24 read blocks); disk
  about 160 GB kept (roll dumps, on1_CBR0-6 / on1_JTPR, reads with ASLs).
