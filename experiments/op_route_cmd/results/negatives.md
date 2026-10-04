# Negative route polylines: spec, generator, screens, sidecar

Generator `lib/route_neg.py` (`make`, then `screen` against the nuPlan map), builder `scripts/route_neg_sample.py`. Two stages: stage 2 (a 158-row hand-check sample, further down) and
stage 3 (2026-10-05, this section: map-screened sidecar that can enter training without manual review).

## Stage 3: screened sidecar (navtrain, split `navsim/navtrain`, CPU, 1078 logs, ~2 min on the box)

Training use is unchanged: the polyline is replaced by the negative, the target stays the logged path (or the original model's plan when distilling). The rule behind every screen:
a negative is used only if the map says the route is impossible; anything ambiguous is dropped, not kept.

| kind | candidate frames | screen (all numbers from `lib/route_neg.py`, constants at the top of the screens block) | rule decided by |
|---|---|---|---|
| N1 exit | status `branch`, v >= 1, path >= dist + 30 | tier A (no lane of the ego roadblock has the exit) -> used; tier B (another lane has it) -> **not used, emitted as `lane_change_needed`** (aux file); tier-A missing **straight** (ramps / loops, trivially easy) is capped at 15 % of the N1 rows used, flag `trivial_straight` | map (decision-93 inventory classes + roadblock union) |
| N2 side | status `no_junction_30m` or `branch`, v >= 3, path >= 90 | **exit leg** = the negative from `s_free + 8 m` over 40 m (after the 90 deg arc, clear of the road it leaves). Used only if its distance to every drivable polygon is > 6 m: vector layers lane, connector, intersection, roadblock, roadblock connector, carpark **and** the 0.1 m raster `DRIVABLE_AREA` (adds generic drivable areas without a vector object). Up to 4 redraws of turn position / side / radius, the first that passes is kept | map |
| N3 oncoming | same, path >= 100 | the shifted path (4-7 m, after its 20 m ramp, one point per 5 m) is classified per point: on a lane / connector polygon with heading within 60 deg of the path (same), > 120 deg (opposite), in between (cross road), on another drivable layer without a lane (area), off everything. Used only if same <= 10 % **and** opposite + off >= 50 %; else `rej_same_direction_lane` / `rej_ambiguous_area_or_cross`. Up to 4 offset redraws | map (lane travel direction) |
| N4 U-turn | same, path >= 60 | same exit-leg screen as N2 (leg = after the 180 deg arc). Up to 4 redraws | map |

Counts (rows = (frame, kind); train membership = all navtrain, hold-out logs are not removed here, `log` is in the sidecar so the trainer can):

| kind | candidate rows | used | dropped (reason) | frames / logs used |
|---|---:|---:|---|---|
| N1 exit tier A | 10 499 built | **9 661** (left 3 444, right 4 768, straight 1 449 after the cap) | 838 straight over the cap | 9 661 / 777 |
| N1 exit tier B | 4 343 | 0 | 4 343 `lane_change_needed` (aux file) | |
| N2 side | 31 014 | **23 877** | 7 099 drivable area within 6 m (after 4 redraws), 38 not built | 23 877 / 939 |
| N3 oncoming | 30 417 | **13 148** | 16 977 same-direction lane > 10 %, 292 ambiguous | 13 148 / 640 |
| N4 U-turn | 33 257 | **17 007** | 16 218 drivable within 6 m, 32 not built | 17 007 / 876 |

Total used 63 693 rows on 32 754 distinct frames (a token can carry several kinds, 12 262 one, 11 085 two, 8 367 three, 1 040 four). 1 383 N1 candidates were not built (no ego lane or
no missing class). `negatives_counts.csv` has the table; redraws: 41 % of the used N2, 20 % of N3, 53 % of N4 rows are not the first draw (`attempt` column).

What the survivors are, honestly:
- N2: 88 % of the survivors have a clearance >= 8 m, 12 % sit within 6-8 m (borderline set). The screen cannot see driveways that are in no map layer (the raster is the only
  source of unstructured drivable area); a driveway inside a not-drivable polygon region is invisible to it. Residual risk, not measured. Image screen (VLM) skipped.
- N4: the leg of a 180 deg U-turn with R 5-8 m lies 10-16 m beside the ego road, so a survivor is a U-turn that overshoots the road width (clearance > 6 m from any road), 40 % of them
  below 8 m. U-turns on narrow one-way streets survive, U-turns across a two-lane road mostly do not (the oncoming lanes are within 6 m): 49 % of the N4 draws are rejected.
- N3: 38 % of the survivor points are off every drivable layer, 52 % on opposite-direction lanes, 5 % on non-lane drivable area, 2 % on cross lanes, 2 % on same-direction lanes (the
  <= 10 % allowance). A point on a parking lane / shoulder that is in no lane polygon but inside the raster counts as `area`, not as opposite / off.
- N1 tier A: the rule is "no lane of the roadblock has this exit class", not "no road there". For the 9 661 used rows the exit leg is on or within 6 m of a drivable polygon in
  left 1 238 of 3 444 (+ 237 NaN), right 3 269 of 4 768 (+ 480 NaN), i.e. mostly turns that are prohibited or missing for the lane while the cross road exists
  (`clear_m` <= 6). The ones with `clear_m` > 6 are "no road there" (T-junction / ramp). All tier-A straight rows are on a drivable area (ramp / loop mouths: `clear_m` = 0),
  hence the cap. A trainer that wants only "no road" negatives filters `clear_m > 6`.

Files (`$DATA_DIR/processed/op_route_cmd/navtrain/`, built by `python experiments/op_route_cmd/scripts/route_neg_sample.py`, rerun resumes nothing: it rewrites both in ~2 min):
- `route_neg.npz`: **only rows to be used**. Keys: `id` (navtrain token, same as route.npz) + `kind` (N1_exit / N2_side / N3_wrong / N4_uturn; one row per (id, kind)), `tier` (N1: A),
  `missing`, `poly` / `pmask` (the negative, 16 vertices at 10 m, ego frame, same format as route.npz `poly`), `poly_pos` / `pmask_pos` (the logged route of the frame),
  `trivial_straight` (N1 straight), `clear_m` (leg clearance, NaN for N3), `same` / `opp` / `cross` / `area` / `off` (N3 shares), `margin` (normalised distance to the threshold, small =
  borderline; NaN for N1), `attempt`, `s_turn` / `angle` / `radius`, `v0`, `log`, `row` (index into route.npz), `label` (= ok).
- `route_neg_aux.npz`: every other built row, same keys, `label` in {`lane_change_needed`, `cap_straight`, `rej_drivable_near_leg`, `rej_same_direction_lane`, `rej_ambiguous_area_or_cross`}
  (45 767 rows). Not for training; `lane_change_needed` is the tier-B set for a later "lane change needed" arm.
- `drivable_raster/*.npy`: the nuPlan drivable-area rasters (memory-mapped by the workers; 4 maps, ~2 GB).
Rows are reproducible: `build_one(..., attempt)` seeds `default_rng([row, kind index, attempt])`.
Target share (~10-15 %) is the trainer's draw from `route_neg.npz` per batch; per-kind weights should not follow the row counts (N2 + N4 are 64 % of the rows).

Sheets for the spot check (Mac, `figs/`): [negatives_screened_sheet.png](../figs/negatives_screened_sheet.png) = 25 random survivors (5 N1-A, 7 N2, 6 N3, 7 N4) then 10 random rejects with
the reason in the title; [negatives_borderline_sheet.png](../figs/negatives_borderline_sheet.png) = 12 survivors closest to a threshold (4 N2, 3 N4, 3 N3, 2 N1-A with clearance < 20 m), listed
in `negatives_borderline.csv`. Look at: dark red = the screened leg / path; does it run along a road or driveway that the blue / grey underlay lacks (a survivor that should
have been rejected); are the rejects really legal routes. Underlay is not model input.
Run `route_neg_sample.py --sheet` to redraw with other seeds (the sheet re-derives each negative from the log).

## Stage 2 (2026-10-04): first sample and the open issues that stage 3 closes

Generator `lib/route_neg.py` (`make(kind, driven_path, rng, ...)`), sampler `scripts/route_neg_sample.py`, sample `negatives_sample.{npz,csv}`, sheet
[`../figs/negatives_sample.png`](../figs/negatives_sample.png) (grey = nuPlan lanes / connectors for the human check only, green = logged route, red = the negative).
Training use: the polyline is replaced by the negative, the target stays the **logged path** (hindsight future of the frame; or the original model's plan when distilling,
as in op_img_cmd's q3NA negatives). The model must learn "a route that does not fit the scene is not executed". A weaker variant (target = slow down) is a loss-side choice and
needs no new geometry.

## Kinds

| kind | construction | "nonexistent" is decided by | needs a visual check |
|---|---|---|---|
| N1 exit | logged path up to the branching point (decision-93 `dist`), then a 90 deg arc (R 9-15 m) into an exit class the ego lane lacks, or, for a missing straight exit, the heading kept straight | map: class not among the ego lane's exits. **Tier A**: no lane of the ego roadblock has it. **Tier B**: another lane has it (a lane change would make it legal) | tier B and every missing-straight (straight is "heading change < 25 deg", ramps and loops make it unreliable) |
| N2 side | on a no-junction stretch (no connector / intersection on the 150 m driven path, v >= 3 m/s, full path) a 90 deg turn at 20-60 m | map has no connector; driveways, parking-lot mouths and unmapped side streets are not in nuPlan | **all** |
| N3 wrong | the logged path moved 4-7 m to the oncoming side (left in right-hand traffic, right in Singapore) with a 20 m ramp | geometry only (wrong way / off the road) | all (a wide road, a parking lane or a median can make 4-7 m legal) |
| N4 uturn | 180 deg arc (R 5-8 m) at 15-45 m on a through road with no junction | map has no connector | all (U-turns at mid-block openings, median breaks) |

## Sample (158 rows, navtrain, <= 2 per log and kind, seed 0)

| kind | tier | missing | n | needs visual check |
|---|---|---|---:|---:|
| N1 | A | left / right / straight | 7 / 16 / 7 | 7 (the straight ones) |
| N1 | B | left / right / straight | 8 / 8 / 12 | 28 |
| N2 | - | side road | 40 | 40 |
| N3 | - | oncoming lane | 30 | 30 |
| N4 | - | u-turn | 30 | 30 |

135 of 158 are flagged; only 23 N1 tier-A left / right rows are map-certain (the ego roadblock has no such exit). The first look at the sheet: tier A rows are plausible (T / ramp
junctions), tier B rows show a cross road whose exit exists for the neighbouring lane (these are legal routes from another lane, keep them out of a strict negative set or relabel
them as "lane change needed"), N2 / N4 geometry is clean but only a person can say whether a driveway sits there, N3 offsets of 4-7 m often leave the drawn lanes. The ramp / loop cases
(first row of the sheet, missing "straight" at a ramp mouth) look wrong: the continuing straight line leaves the ramp road, which is a negative but a trivially easy one.

Fields of the npz: `id` (token), `kind`, `tier`, `poly` / `pmask` (negative), `poly_pos` / `pmask_pos` (the logged route of the same frame), `needs_visual`; the csv adds log, missing class, turn
position / angle / radius, speed and path lengths. Tokens are navtrain tokens of the decision-93 inventory (join to the route.npz / sample tables by `id`).

## Open points

- Human check of the flagged rows is the gate before any of N1-B / N2-N4 enters a training mix; N1-A left / right are usable as is. A cheap extra automatic filter for N2 is the camera
  (a driveway is visible on the image), which a person or a VLM would have to judge.
- N1 only exists at junction approaches of navtrain (11 310 decision-93 frames, 22 932 branch frames); WOD has no map and gets only N3 / N4-type negatives without a
  nonexistence check (not generated).
- The target for a negative on a frame whose logged path does turn at that junction is still the logged turn; that teaches "ignore an impossible route" but also "turn without a route",
  which the positives also support; keep the negative share small (q3NA used a minority share).
