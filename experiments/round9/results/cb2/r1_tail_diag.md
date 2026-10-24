# CB2: the R1 (a) tail, diagnosed (2026-10-25)

Diagnostic only: no training, no gate change, no pool or lane file touched, nothing adopted. Script
[../../scripts/r1tail.py](../../scripts/r1tail.py) (reads pools / tokens / dumps / checkpoints, writes only
`$DATA_DIR/runs/round9/cb2/r1tail/`); 8 GPU pool jobs, about 15 job-min in total. Context: [README.md](README.md) amendments 4-7,
[r1diag_*.json](.), plan [../../plans/2026-10-25-cb2.md](../../plans/2026-10-25-cb2.md) amendments 4-7, decision 482.

**Question.** Gate R1 (a) compares the trainer's forward of the serving checkpoint (cached tokens, batch 128) with the served plan on the
first 15 plan points. In P1N''s pools a few rows differ by up to 0.0625 m at points inside 64 m (2-7 rows per practice shard, 42 on
L18CBPB), while most rows agree to ~1e-3 m. Why, and is this new to the P1N' lineage?

## Answer

1. **Cause: the served policy pass is `torch.compile`d; the trainer's is eager. Both are fp16** (the brief's "fp32 trainer forward" is
   not right: `pp_train.PModel` runs the Cinque graph in fp16 on both sides, so neither side is a reference). The two give plan outputs that
   differ by a *relative* error of about one fp16 epsilon. That error is the same on every pool and every checkpoint. Measured on the
   offending rows themselves: the same cached tokens through Core.compile's policy pass at batch 1 (the served call) reproduce the served
   plan to a median 0.004-0.012 m and at most 0.031 m. **0 of the 70 offending rows** (L18CBPB 42, CBPB4 4, L18RB0 22, PN0 2) stay over
   the floor. The eager pass at batch 1 keeps the tail (29 / 42, 2 / 4, 15 / 22, 2 / 2 still over), so batch shape is not the cause.
   Ego rows are bit-equal to the dumps. One-step token jitter and 1e-5 route-memory jitter each move the plan by at most 0.031 m.
   The PN6 end-to-end check below (the only pool whose dumps still have the slot frames) confirms this: the compiled policy pass removes the
   tail on eager and on compiled tokens alike. The encoder contributes at most ~1-2 steps. No ego, route-memory, saturation or branch
   effect is involved: the offending rows are ordinary fast cruising rows (6-15 m/s), not stop / route-branch states.
2. **Where the tail sits.** Every offending point is a longitudinal x at plan points 12-14 (the far end of the 15 checked, about 1.5-2 s)
   at |x| = 9-32 m, on rows driving 6-15 m/s. Over 16-32 m one fp16 step is 1/64 m, so 0.039 / 0.047 / 0.0625 m are 2.5-4 steps; in
   relative terms 2-3e-3 (fp16 eps = 9.8e-4). The per-pool relative-error distribution on points >= 4 m is the same everywhere: median
   ~0.7e-3, q99 2.2-2.7e-3, q99.99 3.4-4.3e-3. A row exceeds 0.03126 m when the large-relative-error tail meets a point beyond ~10-15 m,
   so the count follows how fast the policy drives, not the lineage.
3. **Not new to this lineage.** Rows with any point over 0.03126 m (= the "rows over floor" of r1diag, which matches exactly):

| pool (roller) | rows | rows with a point at 16-32 m | rows over 0.03126 m | per 1 000 rows | per 1 000 exposed rows | max diff m | rows at 0.0625 | rel. error q99.99 / max (points >= 4 m) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| L18CBPB (P1N', onpol-train) | 17 653 | 3 957 | 42 | 2.38 | 10.6 | 0.0625 | 3 | 3.7e-3 / 3.9e-3 |
| L18RB0 (RB0-F-s0) | 17 429 | 2 431 | 22 | 1.26 | 9.1 | 0.0469 | 0 | 3.9e-3 / 4.6e-3 |
| L18RTB1 (RTB1-F-s0) | 17 489 | 2 914 | 30 | 1.72 | 10.3 | 0.0469 | 0 | 3.9e-3 / 4.3e-3 |
| L18D3 (NT4D3-F-s0) | 17 590 | 2 746 | 13 | 0.74 | 4.7 | 0.0469 | 0 | 3.4e-3 / 4.3e-3 |
| CBPB0 / 2 / 4 (P1N', practice) | 6 216 / 8 797 / 6 256 | 254 / 636 / 535 | 3 / 7 / 4 | 0.5-0.8 | 7.5-11.8 | 0.047 / 0.047 / 0.0625 | 0 / 0 / 1 | 3.4-3.8e-3 |
| CBR0 / 4 (RB0-F-s0, practice) | 6 208 / 6 213 | 194 / 361 | 1 / 2 | 0.2-0.3 | 5.2-5.5 | 0.039 / 0.047 | 0 | 3.4-3.8e-3 |
| PN0 / 3 / 5 (JTN-L18-s0, practice; in B2's on1_JTX) | 6 216 / 6 800 / 8 989 | 197 / 393 / 270 | 2 / 6 / 3 | 0.3-0.9 | 10.2-15.3 | 0.047 / 0.047 / 0.039 | 0 | 3.7-4.3e-3 |
| B2's own DAgger pools NDP + ND001-024 (STANDFMXH-L18-s1; logs only) | 17 725 | n/a | 26 | 1.47 | n/a | 0.0469 | 0 | n/a |

B2's own nuplan_dag pools had the same tail: 26 rows over one 0.03126 m step in 17 725 rows (1.47 / 1 000), a per-pool max of 0.047 m
(3 steps at 16-32 m). Their tokens and rows are deleted, so these numbers come from the stored `nd-tok-*` R1 log lines. The nuplan_dag
amendment-2 cap of 0.047 m was set as "the largest already in the pool", i.e. at this tail's observed max, not from a model of it.
L18CBPB has the most rows with points at 16-32 m (3 957 vs 2 431-2 914: P1N' drives faster on onpol-train, median speed 5.8 vs 4.8-5.2
m/s). Its rate per exposed row (10.6 / 1 000) is the same as RB0's / RTB1's / the PN shards'. The three 0.0625 m rows have relative
errors 2.5-2.9e-3, inside every other pool's range (L18RB0 has larger ones, up to 4.6e-3, that landed on nearer points and showed as
0.047 m). So 0.0625 vs 0.047 is the same error on a point a few metres further out. It was always present; P1N' just drew a few more of them.

## Encoder vs policy (PN6, full served path from the dump frames)

PN6 (JTN-L18-s0, 3 528 rows; failed R1 (a) in practice_nuplan, so it was never merged and its dumps were not stripped). I rebuilt the tokens from
the dumped slot frames twice: with the eager encoder (pp_prep.encoder, as on_rows) and with Core.compile's encoder pass (8 pairs, as
served). I crossed each with the three policy passes and compared against the served mu (first 15 points x / y):

| tokens + policy pass | identical rows | median max diff m | max diff m | rows within 0.03126 m | rows over the floor |
|:--|--:|--:|--:|--:|--:|
| eager + eager b128 (= R1) | 0 | 0.0059 | 0.0469 | 0.9989 | 4 |
| eager + eager b1 | 0 | 0.0059 | 0.0547 | 0.9991 | 3 |
| compiled + eager b128 | 0 | 0.0059 | 0.0469 | 0.9994 | 2 |
| compiled + eager b1 | 0 | 0.0059 | 0.0469 | 0.9989 | 4 |
| eager + compiled b1 | 0.001 | 0.0039 | 0.0313 | 1.0000 | 0 |
| **compiled + compiled b1 (the served path)** | 0.027 | **0.0020** | **0.0313** | **1.0000** | **0** |

- **The policy pass carries the tail.** With any eager policy pass the tail is there (2-4 rows over the floor, up to 0.055 m). With the
  compiled policy pass it is gone (0 rows, max 0.031 m), even on eager tokens.
- **The encoder pass matters element-wise but hardly at the plan.** Compiled tokens differ from eager ones on 83 % of the entries (max
  |diff| 0.32, q99 relative 5 %). Swapping them still only moves the plan median from 0.0039 to 0.0020 m and leaves the max at 0.031 m.
- **A fresh compile is not bit-identical to the serving process** (2.7 % identical rows): kernel selection in another process and at
  another time differs at the rounding level. So no offline check can be bit-exact against the served plan, and any R1 (a) has to be a
  tolerance.

## Does it matter for training labels?

No, not materially:
- **The labels do not pass through this forward.** The DAgger label (b3 teacher, nd_teach / cb2 dlabel) is built from the *served*
  poses and the map. The imitation target and the S hinges also use the stored served poses. R1 (a) only checks that the cached
  training input (eager tokens + fed ego / route memory) reproduces the served decision. The tail shows that it does, to 2-4e-3
  relative on 0.1-0.2 % of the rows.
- **Size against the label scale.** On P1N''s practice shards the b3 label minus served at 4 s (lateral) is p50 0.29 m, p90 1.76 m;
  |d0| p50 0.13 m ([label_CBPB0.md](.) etc. on the box). On the offending rows labelled so far (CBPB0 / CBPB2, 10 rows) the label
  correction (max over the 8 poses) is 0.05-2.9 m, median ~0.33 m. The wrong-checkpoint control (CB2P1N-F-s0) differs from the served
  poses by a median 0.32 m on L18CBPB. The tail (<= 0.0625 m, at 1.5-2 s, longitudinal) is 5-10x below the median correction and
  about 5x below the gap between two neighbouring checkpoints. In relative terms it is below the fp16 resolution the trainer itself
  works at (its own forward has ~1e-3 relative noise against the served one on every row).
- One labelled offending row (CBPB0) has a correction of 0.048 m, the size of the tail. On such a row the "correction" is itself
  inside the fp16 noise: the label is right, but it carries no signal. The same holds for any row whose correction is that small,
  on every pool.

## A principled R1 (a) (recommendation for the user; not applied)

The check wants "the cached input reproduces the served decision up to serving numerics". Serving numerics are a **relative** fp16
error of the compiled-vs-eager pass, so the tolerance should be relative to each point's magnitude, with an absolute floor for points
near zero:

> per point |trainer - served| <= max(0.03 m, k * eps16 * |x|) with eps16 = 2^-10 and k = 8 (relative 7.8e-3), on >= 0.99 of the rows;
> and no row above 2x that bound.

- k = 8 is about 2x the largest relative error measured on any of the 12 pools / 5 checkpoints here (4.6e-3 = 4.7 eps). The floor
  0.03 m is the registered line and covers the near-zero points (lateral y ~ 0), where amendment 6's per-point step (1e-4 m) is far
  below the ~1e-3 m absolute noise of the network.
- At the points that matter: at |x| = 25 m the bound is 0.195 m, still 1.6x below the 0.32 m median wrong-checkpoint distance, so
  clause (c) keeps its power. Clause (c) stays the actual test that the right checkpoint and inputs are fed.
- Stronger alternative, if the user prefers a bit-level check over a tolerance: compute R1 (a) through the served call itself (Core's
  compiled policy at batch 1 on the cached tokens, as `r1tail.py loc` does). On the 70 tail rows this brings every row within 0.031 m
  and makes the tolerance question mostly disappear. The residual is the encoder pass (eager tokens are cached; the served tokens are
  not dumped).
- Under either line, every pool measured here would pass (max relative 4.6e-3 < 7.8e-3; all 70 tail rows <= 0.031 m through the served
  call), including CBPB4 and L18CBPB. B2's own pools (max 0.047 m) would pass the relative line for any point beyond 6 m, which is where
  every tail point sits on the 12 pools measured; their points cannot be re-checked (deleted).
- Whatever line is chosen, register it before the next R1 run. Applying it to CB2's existing pools is a new amendment the user decides.

Raw outputs (box): `$DATA_DIR/runs/round9/cb2/r1tail/{stat,loc,full}_*.json`, `stat_*.npz` (trainer plans, per-row max diff).
