# The AlpaSim corridor rule, and why the arena's static windows zero our cars (2026-10-18)

Code `scripts/corridor_rule.py` (`extract`: per-step scorer metrics of stored reads; `gtend`: the human's last GT segment from the raw
log; `report`). Full tables: [corridor_rule/tables.md](corridor_rule/tables.md). Per-window CSVs on the box under
`$DATA_DIR/runs/alpasim_proxy/arena/corridor/` (`ar_<tag>.csv.gz` arena, `ws_<tag>.csv.gz` wait set, `gtend_*.csv.gz`, `hand.csv.gz`).

## The rule, from the scorer source

Read in `third_party/alpasim/src/eval` (`scorers/ground_truth.py`, `aggregation/main.py`, `aggregation/modifiers.py`,
`aggregation/scene_score.py`); `max_dist_to_gt_trajectory = 4.0` (`wizard/configs/base_config.yaml`, `e2e_challenge/*.yaml`).

1. **Human path.** The GT polyline is the human's recorded track of the scored scene only: the 5.5 s trajdata window (56 poses at
   10 Hz, in the ego AABB-centre frame). Nothing after the window enters.
2. **Distances, per metric step** (0.5 s apart here: 0, 0.017, 0.517, ..., 5.017 s; policy-relevant from 1.017 s). `d` = distance from the
   ego polygon centroid to the GT polyline (shapely, clamped to the end point). `lateral` = `d`, except when the centroid projects onto
   the GT's far end (and the GT has length > 0 and a non-zero last segment): then only the offset perpendicular to the heading of the
   GT's **last segment** counts. Overshooting the end of the human's path along its final heading is free.
3. **Time cutoff.** Every step after the first with `d >= 4.0 m` is dropped (`RemoveTimestepsAfterEvent`). That first step itself is kept.
4. **Zero.** `left_corridor_laterally` = the max over the kept steps of `[lateral >= 4.0]`. If it is 1, the scene scores 0. In
   practice: zero if and only if `lateral >= 4 m` at the first step where `d >= 4 m`. Width 4 m on each side, no tolerance at the start.
   At the far end there is no longitudinal limit, only the perpendicular one.

**Consequence.** It all depends on the human's last GT segment.
- **Moving at the end.** The segment has a real heading along the lane, so a forward roll past the end of the path reads as
  lateral ~ 0. The corridor is open-ended forward.
- **Still at the end.** The human stands still to the end of the window. In the arena static windows the last segment is
  0.01-0.07 mm of pose jitter in a random direction. A forward point then projects inside the jittered tangle, not onto its end, so
  `lateral = d`. The corridor becomes a **4 m disc around the human's rest point**. Rolling 4 m in any direction zeroes the scene.

## Evidence (stored reads, every decision's metrics)

"Still" means the last 10 Hz GT segment is < 1 mm. "Moving" means >= 1 mm with cosine >= 0.9 to the final heading. "Tripped" means some
metric step had `d >= 4 m`.

| read | windows | end state | tripped | corridor zeros | zero given tripped |
|---|--:|---|--:|--:|--:|
| RSG-L18-s0, arena held, static stratum | 828 | still | 382 | 379 | 0.99 |
| RSG-L18-s0, arena held, static stratum | 48 | moving | 19 | 1 | 0.05 |
| RSG-L18-s0, arena held, other strata | 56 / 1228 | still / moving | 17 / 87 | 17 / 8 | 1.00 / 0.09 |
| RSG-L18-s0, wait set a (d344) | 146 | **all moving** | 73 | 3 | 0.04 |
| YR / NR1F / APY, arena held static, still end | 828 each | still | 394 / 346 / 647 | 389 / 342 / 638 | 0.99 |

- **Below 4 m it never fires.** On RSG's 828 held still-end static windows, 446 never reached `d = 4 m`: 0 corridor zeros. Of the 382
  that did, 379 are corridor zeros, and the other 3 are zeros of another type that came first. The smallest `d` at any corridor zero
  is 4.00-4.01 m for every tag. Binned by the largest `d`: (3.5, 4] m 0 / 14, (4, 5] m 221 / 222, (5, 8] m 158 / 160.
- **The scorer's GT length** tells the same story. Arena static windows have a median of 0.003 m, and with that the zero rate given a
  trip is 0.95-0.97. Pooled over all tags, the zero rate given a trip falls as the GT length grows: 0.72 for 0.05-0.25 m, 0.56 for
  0.5-1 m, and 0.31 above 2 m.
- **Every policy hits it, not just ours.** YR, NR1F and APY all show 0.99. The only thing that separates the policies is how often
  they roll 4 m.
- **Size on held.** RSG has 382 zeros on the 828 still-end static windows, worth 0.539 mean score, against 0.938 on the 48
  moving-end static windows. Its held static stratum overall is 878 windows, a 0.44 zero rate and a 0.56 mean score. 396 of RSG's 412
  held corridor zeros are in still-end windows.

## Why the waiting set (decision 344) found few zeros

**The cause is window selection.** Variant a puts the window's end 0.5 s before the human's movement onset, where onset means the
first frame > 0.25 m from the rest point. In all 146 windows the human is already creeping at that point: the last 10 Hz segment is
8-20 mm along the heading (0.1-0.2 m/s), and the median GT length is 0.037 m against the arena's 0.003 m. So the scorer saw a
forward-pointing path end, and rolls past it were free.

The other candidate causes do not hold:
- **Decision count.** Variant a uses the same 5.5 s / 10-decision windows.
- **Human-path extent.** The GT is the window in both sets.

Same model, same 4 m trip: RSG tripped on 73 windows there and was zeroed on 3. On the arena's still-end static windows it tripped
on 382 and was zeroed on 379. Decision 344's "rolling 3-4 m rarely zeroes" holds only for windows where the human creeps at the end.
Its follow-up claim, "waiting is a small lever", rests on that selection artifact.

## Allowed roll before a corridor zero

Write `xy_end` for the human's last GT pose and `h` for the heading of the human's last GT segment.
- **Still end** (human stands still to the window's end; last segment < 1 mm): the ego may stay up to **< 4.0 m** from the human's
  rest point at every 0.5 s metric step. Measure it from the ego AABB centre to the human's AABB-centre GT position; in any direction,
  forward included. The ego starts there because the 0.5 s force-GT puts it on the human, so this equals a roll of < 4 m.
- **Moving end, path length L along heading h:**
  - Inside the path (projection < L): < 4 m from the polyline.
  - Past its end: only the offset perpendicular to `h` through `xy_end` must stay < 4 m. Forward overshoot is unlimited.
  - So the allowed roll is not set by L at all, only by whether the final segment is non-degenerate.
- **In between.** A path whose last segment points off the heading (the human turns or stops at an angle, 68 held windows) behaves
  like the still case at 0.88-1.00.

**Implications.**
- **The lever is "don't move when the human doesn't".** Our arena static zeros are not lane errors: the cars roll 4-5 m with a human
  who stays put.
- **A 3 m leash is enough for the scorer:** it never fires below 4 m.
- **The private suite** (uniform windows, 30 % static) should behave like the arena, not like the waiting set. That assumes the private
  scorer is this one with the same 4 m setting.
