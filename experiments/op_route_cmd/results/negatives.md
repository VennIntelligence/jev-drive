# Negative route polylines: spec, generator, sample (stage 2)

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
