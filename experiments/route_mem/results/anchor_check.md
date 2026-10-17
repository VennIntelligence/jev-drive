# ANCHOR-CHECK: the anchored route memory on the audit's reads and training rows

2026-10-17. An input-definition change behind a new flag (`rm_train --rm-anchor`, sidecar head `order: true, anchor: true`): the user-approved
fix of the alignment audit ([align_audit.md](align_audit.md), "Minimal fix" C + D). Nothing is trained on it yet and no existing checkpoint
serves it. Code: `lib/rmem.py` (`merge_a`, `memory_a`, `Tracker(anchor=True)`, `head_anchor`), `scripts/rm_build.py anchor`,
`scripts/rm_train.py --rm-anchor` (its ANCHOR block), the drivers (`sh30_core.py` / `sh30_driver.py` / `pai_driver.py`: `SERVE_ANCHOR`
from the sidecar), `scripts/rm_align.py anchor` (tables), `scripts/rm_ident.py` (flags-off identity). Generated tables:
[anchor/anchor.md](anchor/anchor.md) (+ `anchor.json`); identity: `anchor/identity_*.json`. Box: `$DATA_DIR/runs/route_mem/anchor/`.

## The rule

- **Merge (fix C).** `merge_a` appends a newer batch from the arc of the memory's end on that batch: the end's nearest point on the
  batch's segments that run within 90 deg of the memory's last segment (within 2 m), then the batch's points more than 1 m further along.
  A batch that starts beyond the end (a gap) is appended whole; one the end lies beyond adds nothing, and so does one with no segment
  running with the memory's end (it lies on the approach of a loop the memory has already passed: without this test 3 NAVSIM rows folded
  by 26-36 m). `merge` decides "beyond the end" by the last segment's direction, which at a loop points at the newer batch's first points,
  so the whole batch was appended.
- **Anchor (fix D).** Per message, C_k = arc of its first waypoint on the merged polyline - L0 (the ordered layout's progress anchor =
  AlpaSim's own projection of the car). AlpaSim resamples 20 points every 80 / 19 m from that projection and drops those whose *chord*
  sum is under 40 m, so on a sharp corner close ahead m more points go: the first waypoint lies m x 4.21 m further out and the message
  holds 10 - m points. The waypoint count is the signal: in the held reads all 7 982 messages with 10 points had no extra trim, and the
  one-point-short ones were trims (15 of 23) or a far end cut by a route end or a bad segment. A short message holds the hypotheses
  C_k - m x 4.21 m, m = 0 .. 10 - n; the first message with waypoints takes m = 0; every later one takes the hypothesis nearest the
  previous anchor + the chord the car drove since if it explains that within A_TRIM = 1.5 m, else it keeps the previous count (B). A
  full message is C_k itself. At the decision: + the chord driven since the last message. The 40 slots (-6 .. +72 m), the point features
  and the 3 scalars are the ordered layout's (DIM_O = 203), so RouteHead(order=True) reads it.
- **Why not the proposed cap** s_k = min(C_k, s_{k-1} + chord + 1 m). Route arc legitimately outruns the driven chord where the car is
  inside a turn: in the held reads by up to 7.6 m in one 0.5 s step (p99.9 of all steps 3.9 m; 8.1 m in the pool rollouts). The cap
  lags there, and it cannot fix the trims that start with the rollout (most of them). Table A4: on RSG held it leaves 58 decisions off by
  more than 1 m (36 of them behind, down to -5.0 m), against 24 for the rule.
- **Why m = 0 at the first message.** A far end cut by a route end leaves the same short message as a trim. Taking m = 1 for a one-short
  first message put 81 public decisions 4.2 m behind (9 of RSG's 13 one-short first messages on public 1491 were cuts) and stays wrong
  until a full message arrives; m = 0 is the ordered layout's assumption, and its error is the audit's (+4.2 m on the far slots).

## Verdict

1. **The slots sit at their true distance.** First-slot lead median / p90 / p99 is 0.00 / 0.00 / 0.00 m on every read set and on the
   training rollouts, 0.00 / 0.00 / 0.01 m on the NAVSIM rows (the served projected layout: p90 7.1-10.3 m, p99 26-30 m). Decisions off
   by more than 1 m, projected -> ordered -> anchored: RSG held 1 985 -> 58 -> 24, V2 held 1 970 -> 64 -> 30, RSG public 4 525 -> 48 -> 25,
   V2 public 4 502 -> 47 -> 27, arm a public 4 506 -> 44 -> 25, RB0-F-s0's training rollouts (pool on1_L18RB0) 2 999 -> 123 -> 70, NAVSIM
   rows 32 555 -> 1 626 -> 313. A near slot (<= 20 m) shown 10 m or more ahead: 0 on the reads and the pool (projected 332-609, ordered
   11-30), 8 of 101 313 NAVSIM rows (projected 4 417, ordered 1 200). At decisions with a route turn within 42 m the anchored p99 is
   0.01-0.05 m (projected 35-52 m); in U-turn rollouts it is up to 4.21 m (projected 72-82 m): the trims that start with the rollout (3).
2. **The folded merges are gone.** merge_a folds on no decision of the reads and the pool rollouts (old merge: 16-32 per set), and no
   anchored decision at an old fold is off by more than 1 m (ordered: 11-30 per set, the anchor 46-66 m ahead). On the NAVSIM rows it
   still shows 65 "folds" (old merge: 1 347), all 65 in the 92 rows whose route revisits a place (passes within 2.5 m of itself), where a
   point has two stations and the fold metric itself is ambiguous.
3. **AlpaSim's trim is tracked where it can be told.** Of the offset decisions (first waypoint 4.2 / 8.4 m further than L0: 24-49 per
   set), the rule took the extra spacings on 7-10 per read set and 36 of 49 in the pool rollouts, and never took one without an offset
   (0 false trims in all 73 762 read and pool decisions). Still off by > 1 m there: 15-17 per set (ordered: 19-49). On the reads all of
   them are trim runs that start with the rollout's first message, where the short message is indistinguishable from a far end cut by a
   route end; their error is the ordered layout's (+4.2 m, on the far slots: the near range is still empty then). In the pool rollouts 10
   of 17 are such runs; the rest are tracked trims left with point 4's chord error (-1.1 m) or misses at a sharp corner where the car's
   projection jumped against its chord (+4.2 m, as in the ordered layout).
4. **What remains on the reads is the 4.2 m sampling.** Behind by 1-1.7 m: 5-8 decisions per read set, 53 in the pool rollouts (the
   plain variant has 49): where the merged polyline's 4.2 m chords cut a sharp corner between a slot and the newest first waypoint, arc
   measured along the chords is short by up to the cut. The ordered layout has the same.
5. **Training rows.** The NAVSIM rows go through the same function (rm_build anchor; rm_align nav rebuilds them with the same messages).
   Without the 92 route-revisit rows: anchored 292 off by > 1 m (123 ahead / 169 behind), ordered 1 548 (1 391 / 157). What is left there:
   rows where the logged car drives against the route (its station falls while it moves 2-3 m per step forward: the progress anchor sits
   at the memory's start in every layout, ordered and plain show the same 24-57 m); rows with a broken route (messages of 1-2 points, the
   progress anchor stuck while the car drives); trims that start with the row's first message (+4.2 m); and coincidences of a legit
   projection jump of about one spacing with a far-end cut at the route's end (-4.2 m), which the band (B) limits.
6. **Flags off, nothing moves** (table below): old vs new code byte-identical on every feature and served-memory array (same hash seed),
   60-step training identical (RSG-L18's on_hash_60 ea8cacc3e451c577 reproduced), and no existing checkpoint carries `anchor`.
7. **The proposed cap is worse than no cap** (A4): on every set it adds 26-86 decisions behind (to -6.4 m) and fixes no trim that starts
   with the rollout. merge_a with the plain progress anchor (no trim tracking) is the first column: 28-102 off by > 1 m.

**B. The trim band.** |lead| > 1 m (NAVSIM: ahead / behind) for the nearest-hypothesis rule and two bands (same code otherwise, the same
NAVSIM routes: PYTHONHASHSEED 0; tables [anchor/band_nearest.md](anchor/band_nearest.md), [anchor/anchor.md](anchor/anchor.md),
[anchor/band_1.0.md](anchor/band_1.0.md)):

| rule | RSG held | V2 held | RSG public | V2 public | arm a public | pool rollouts | NAVSIM rows |
|:--|--:|--:|--:|--:|--:|--:|--:|
| nearest hypothesis (no band) | 21 | 27 | 21 | 23 | 21 | 69 | 315 (115 / 200) |
| A_TRIM 1.5 m (the rule) | 24 | 30 | 25 | 27 | 25 | 70 | 313 (127 / 186) |
| A_TRIM 1.0 m | 25 | 30 | 25 | 27 | 25 | 70 | 311 (133 / 178) |

The band keeps the count where the step is ambiguous. What it gives up on the reads are trim runs that start with the rollout and that
the nearest rule happened to correct when the car's projection jumped by more than half a spacing at a U-turn (RSG held: 5943b58c
decisions 4-6); what it removes are errors behind that the ordered layout does not make (NAVSIM: 200 -> 186). 1.0 and 1.5 m differ by one
read decision and two NAVSIM rows.

## What to look at

- [anchor/anchor.md](anchor/anchor.md) A1: the anchored column is 0.00 / 0.00 / 0.00 everywhere; the counts > 1 m are about half the
  ordered layout's and 40-200x fewer than the served layout's; the min / max column bounds what is left on the reads (-1.7 .. +4.8 m).
- A2: the turn and U-turn strata, the audit's problem rows; the last column (near slots >= 10 m ahead) is 0 for the anchored layout on
  every read set.
- A3: the two defects the audit listed for the ordered layout: folds 0 and `anchored` 0 at the old folds; at the offsets, `tracked` vs
  `anchored` (still off) and `false trims` (0). The NAVSIM line: merge_a's 65 "folds" are all route-revisit rows (the note below A3).
- A4: plain / cap / anchored side by side: the cap's negative counts are the lag at turns.

## Flags-off identity

Old code = 4ff092b2 (the commit before the change, a worktree on the box); new = the final code (14fb2bcc), every new flag off. The two
training runs were made at de3e94c7; `rm_train.py`, `lib/retain_train.py` and the three drivers have not changed since (git diff
de3e94c7..14fb2bcc on them is empty), and the later rmem changes touch only `merge_a` / `memory_a`, which no flags-off path calls.

| check | result | file |
|:--|:--|:--|
| training: the RSG-L18 command, 60 steps, eager (JEV_RM_EAGER=1) | identical: 89 scalars, every weight (max diff 0), dev ADE 0.529079 both, on_hash_60 ea8cacc3e451c577 both = RSG-L18-s0's | [anchor/identity_rsg.json](anchor/identity_rsg.json) |
| training: V2's base command (RTB1AR-F-s0 = RB0 + the RETAIN / D2 terms: the path of the `retain_train` change), 60 steps, eager | identical: 56 scalars, weights, dev ADE 1.502367 both | [anchor/identity_base.json](anchor/identity_base.json) |
| offline features, same PYTHONHASHSEED: the projected and ordered rows of 277 NAVSIM rows (24 logs of navtrain shard 0: features, ok, k, messages, fut_d, jdist, turn40, trav) and 416 pool rows (24 rollouts of on1_L18RB0: features, route0_d, anchors, route labels) | byte-identical, every array | [anchor/identity_features_final.json](anchor/identity_features_final.json) (de3e94c7: identity_features_h0.json) |
| the served Tracker (projected, ordered, and `Tracker()` as the drivers construct it) at every decision of RSG and V2 held 460 @18 (2 x 8 129) on the regenerated messages | byte-identical | same file |
| the same with each process's own random hash seed | everything byte-identical except 1 of the 277 NAVSIM rows (0.05 m) | [anchor/identity_features.json](anchor/identity_features.json) |
| why: the new code against itself under PYTHONHASHSEED 1 and 0 | 6 of the 277 NAVSIM rows differ (up to 72 m: another lane). AlpaSim's map route search depends on the hash seed, which is also why the stored training files match a rebuild on only 97.8-99.6 % of the NAVSIM rows (the audit's 99.4 %) | [anchor/identity_features_seed.json](anchor/identity_features_seed.json) |
| served = offline: the memory logged by RSG's held read (old code) against the new code's offline rebuild | 8 129 decisions, max 2.7e-4 m, masks 100 % equal (the audit's gate) | [anchor/gate_rsg_held.json](anchor/gate_rsg_held.json) |
| sidecars: `head.anchor` on the box's 274 route-memory checkpoints (at the final run) | none except the smoke run ANCSMK-F-s0, so `head_anchor` is False and every driver serves what it served | identity_features*.json `sidecars` |

The flag on: the anchored layout as the drivers serve it (`SERVE_ANCHOR`, `Tracker()`) equals `rm_build.feats_of_a` byte for byte at all 73
762 decisions of the six sets, reads and pool rollouts ([anchor/identity_served_anchor.json](anchor/identity_served_anchor.json)); the
NAVSIM rows `rm_build anchor` wrote (shard 0, float16, PYTHONHASHSEED 0) equal rm_align's rebuild within 0.05 m on all 8 468 rows, masks 100
% equal; a 20-step `rm_train --rm-anchor` smoke (ANCSMK-F-s0: shard 0 + its yaw-rate rows, V2's base terms with `--kd-lam` on `on1_WAIT.a`,
7 820 rows; the rule before the band) ran, wrote the sidecar head `order: true, anchor: true`, and `rmem.load_pmodel` loads it as
RouteHead(order, anchor) with 203 inputs.

## Build commands for a base retrain

Built so far, for the checks only (the final rule, PYTHONHASHSEED 0): `feat/navtrain_full.s0of12.a.npz`, `yr1_navtrain_full.s0of12.a.npz`,
`idx_waitcap_tr{A,B,C}.a.npz`, `on1_WAIT.a.npz`, and `runs/route_mem/anchor_sample/on1_L18RB0.a.npz` (24 rollouts). Everything a base retrain
on the anchored layout reads (box, repo root; one CPU pool job, about 10 min; `--force` rebuilds the check files too):

```
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
.venv/bin/python -m jevdrive.cl submit --name rm-anchor-build --vram 0.1 --cpu 64 --ram 128 --timeout-h 2 --env PYTHONHASHSEED=0 -- \
  $AP experiments/route_mem/scripts/rm_build.py anchor --force --workers 64 \
  --data navtrain_full.s0of12 navtrain_full.s1of12 navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12 navtrain_full.s5of12 \
         navtrain_full.s6of12 navtrain_full.s7of12 navtrain_full.s8of12 navtrain_full.s9of12 navtrain_full.s10of12 navtrain_full.s11of12 \
         idx_waitcap_trA idx_waitcap_trB idx_waitcap_trC idx_selfiter_trA idx_selfiter_trB idx_selfiter_trC \
  --pool L18RB0 L18RTB1 \
  --cat on1_WAIT=idx_waitcap_trA,idx_waitcap_trB,idx_waitcap_trC on1_NR1TV=idx_selfiter_trA,idx_selfiter_trB,idx_selfiter_trC \
  --merge on1_L18NR1=on1_L18RB0:rb0,on1_L18RTB1:rtb1
```

This writes `feat/<name>.a.npz` for the 12 navtrain shards and their `yr1_` rows, the KD pool `on1_WAIT`, the trainval pool `on1_NR1TV`, and
the L18 pools `on1_L18RB0` / `on1_L18RTB1` / `on1_L18NR1` (`--cat` / `--merge` check every pool row's old-layout row against its source).
Then each stage of the retrain adds `--rm-anchor` to its rm_train command (RB0-F -> RTB1AR-F -> L18; the L18 stage's `--init` is the
anchored base, so the layout also comes from its sidecar). The PAI submission image is untouched; a PAI read mounts rmem.py and
pai_driver.py, which serve the layout from the sidecar.

## Read tools that would need the anchored layout

Not rewritten (they belong to other lanes); each feeds the old layout or refuses an ordered head:

- `experiments/wait_cap/scripts/wc.py plans`, `experiments/cap_ledger/scripts/led.py`, `experiments/desire_cap/scripts/dc.py`: load
  `feat/<data>.npz` (projected) and attach the head without `order` (they assert it is absent), so they refuse an anchored checkpoint;
  they need `feat/<data>.a.npz` and the head as the sidecar gives it.
- `experiments/wait_cap/scripts/nx.py arrival`: feeds RSG-L18-s0's logged 200-number memory (its static run) to every tag; an anchored
  tag needs the 203-number memory of its own static run.
- `experiments/wait_cap/scripts/nx_turns.py replay`: sets `SERVE_ORDER` from the core but not `SERVE_ANCHOR` (one line:
  `RM.SERVE_ANCHOR = bool(getattr(core, "rm_anchor", False))`); `scripts/rm_align.py replay` asserts a projected checkpoint.
- `scripts/rm_build.py served` (gate G-serve): rebuilds the ordered layout for a 203-number log; an anchored read needs `feats_of_a`.
- `experiments/defect_fix/scripts/df.py v2plans` (`feat/<pool>.npz`), `experiments/route_follow/scripts/rf_v2gate.py` / `rf_appgate.py`
  (old / ordered features), and the pool builders with their own memory code (`rf_pait.py`, `rf_selfpool.py`, `rf_pai_pool.py`, `rf_cfpool.py`:
  `Tracker()` / `memory_o`), for pools built from an anchored checkpoint's rollouts.
- Through `rm_train.build` (follow the sidecar, need only the `.a` files): `nx.py hist`, `wt.py devplans` / `devall`, `rm_train rely`.

## Not covered

- PAI: its messages come at 10 Hz with the same generator settings (10 points from 42.1 m in 96 % of the messages of a V2 PAI chunk), and
  the rule has no time constant, but there is no truth route for PAI here, so no lead table.
- No anchored checkpoint has been trained beyond the 20-step smoke or served in closed loop; G-serve for one needs `rm_build served` with
  `feats_of_a` (above).
- Trim runs that start with the first message stay +4.2 m (point 3); a sharper rule would need the near route geometry, which no message
  carries.

## Reproduce

```
# box, repo root, AlpaSim venv ($AP), CPU pool jobs; O=$DATA_DIR/runs/route_mem/anchor
$AP experiments/route_mem/scripts/rm_align.py extract --set rsg_held=$DATA_DIR/runs/route_mem/read/RSG-L18-s0/c*/run --out $O --workers 16 --dump
#   likewise v2_held / rsg_pub / v2_pub / a_pub / pool_rb0 (rm_align.py SETRUNS; pool_rb0 = runs/onpol_long/roll/RB0-F-s0/c*/run)
PYTHONHASHSEED=0 $AP experiments/route_mem/scripts/rm_build.py anchor --force --workers 32 --data navtrain_full.s0of12 idx_waitcap_trA \
    idx_waitcap_trB idx_waitcap_trC --cat on1_WAIT=idx_waitcap_trA,idx_waitcap_trB,idx_waitcap_trC          # the check files
PYTHONHASHSEED=0 $AP experiments/route_mem/scripts/rm_align.py nav --out $O/nav_final --workers 32        # NAVSIM rows, all layouts
$AP experiments/route_mem/scripts/rm_align.py anchor --dir $O --nav-dir $O/nav_final --res $O/res_final --workers 16   # -> anchor.md / .json
#   the band table: the same two commands with --a-trim 1e9 / 1.5 / 1.0 into nav_<v> / res_<v> (analysis only)
# identity (W = a worktree at 4ff092b2)
PYTHONHASHSEED=0 JEV_REPO=$W $AP experiments/route_mem/scripts/rm_ident.py feats --out $O/ident_old_f.npz
PYTHONHASHSEED=0 $AP experiments/route_mem/scripts/rm_ident.py feats --out $O/ident_new_f.npz
$AP experiments/route_mem/scripts/rm_ident.py cmp --a $O/ident_old_f.npz --b $O/ident_new_f.npz --out identity_features_final --res $O/res_final
$AP experiments/route_mem/scripts/rm_ident.py served --out identity_served_anchor --res $O/res_final --sets rsg_held v2_held rsg_pub v2_pub a_pub pool_rb0
# op-train env, GPU pool jobs with --env JEV_RM_EAGER=1: the RSG-L18 command (and V2's base command) at 60 steps from both checkouts, then
$TR experiments/route_mem/scripts/rm_ident.py bitcmp --a ANCGG-RSG-old --b ANCGG-RSG-new --out identity_rsg --res $O/res --hash ea8cacc3e451c577
```
