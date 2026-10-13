# Pre-registration: RSGboth variants (zeros back to RSG's level, keep the turn completion), 2026-10-13

Context: [results/rsgboth/summary.md](../results/rsgboth/summary.md): RSGboth-L18 (RSG-L18's command with `--on-mode both`) lifts held mean
(+0.037) and turn completion (66.7 -> 91.7 % on s0), but costs zeros (held +7 / +12, public guard +12). Cause named by
[research/route-follow/loss-audit.md](../../../research/route-follow/loss-audit.md): the imitation term on the on-policy rows carries speed,
the road / agent hinges are shape-only (no along-heading gradient). Written, committed and pushed before any variant is trained.
Amendments are appended, never edited in place.

## Amendment 1: diagnosis, then three variants (seed 0 only)

**Seen before this amendment** (rf_rsgbv.py diag on the stored held 460 @18 rollouts of RSGboth-L18-s{0,1} and RSG-L18-s{0,1}; zeros_mech
collect; [results/rsgboth_v/diag.md](../results/rsgboth_v/diag.md)):
- Extra zeros (zero in RSGboth, scored in RSG, same seed): 14 (s0) / 19 (s1), against 7 / 7 the other way. None is in a turn scene of
  rf_follow; by road class at the event 4 / 5 straight, 8 / 12 curve (route heading change >= 15 deg over 60 m), 2 / 2 junction; by type
  5 / 7 offroad, 4 / 8 collision, 5 / 4 corridor. Collisions mostly rear-end (struck object ahead, heading within 12 deg) with a lead in
  the corridor in the last 3 decisions; corridor zeros where the log was nearly stopped (v_log 0.1 - 0.3 m/s) and the ego crept on.
- Speed: RSGboth drives at the log's speed on average (A - log +0.02 straight, +0.20 curve, +0.06 junction m/s) where RSG drove about
  1 m/s below it; A - B = +1.03 [+0.91, +1.16] straight, +0.92 curve, +0.54 junction (both seeds alike). The extra speed is everywhere,
  slightly less in junctions; the extra zeros come where the ego is faster than the human (10 of the 14 s0 extras) on non-turn road.
- Reading: the human-speed imitation target on non-turn on-policy rows is the cause; the shape-only agent hinge cannot brake for a lead.

**Arms** (each = RSGboth-L18-s0's exact command, pool job, `--init RB0-F-s0`, 2500 steps, `--on-mode both`, seed 0, plus ONE change;
chain [rf_rsgbv.sh](../scripts/rf_rsgbv.sh), pools [rf_rsgbv.py](../scripts/rf_rsgbv.py) `pool`):
- **RSGbA30-L18-s0**: `--agent-lam 30` (stronger agent hinge). Predicted weak: most extra collisions are rear-ends, which a shape-only hinge
  does not see along the heading.
- **RSGbW-L18-s0**: `--on on1_L18RB0t --on-weight rclass=straight:0.3,rclass=curve:0.3`: non-turn pool rows drawn at 0.3x (turn rows
  16 % of the pool -> about 40 % of the draws). on1_L18RB0t = on1_L18RB0 hard-linked plus the tab key `rclass` (turn / curve / straight /
  none from the route label's first 40 m and the target's 4 s yaw); every target unchanged. The trainer has no per-term weight of the
  on-policy imitation, so the draw rate is the knob.
- **RSGbC2-L18-s0**: `--on on1_L18RB0c2`: on non-turn rows (straight AND curve, since the diagnosis puts most extras on curves) with
  v0 >= 2 m/s the imitation target keeps its path and its arc length is capped at max(served plan arc, v0 t): never more progress than
  the row's own speed holds; it still slows where the human slowed. Turn rows and launches keep the human's target (8 015 of 17 429 rows
  changed, median 4 s arc cut 2.0 m). Gate R1 (`on1_L18RB0c2.r1.json`): PASS (other files same inode, other arrays byte-equal, unchanged
  rows byte-equal, path deviation 1e-5 m, arcs never longer). on1_L18RB0t's R1 also PASS (no target changed).

**Reads** (each arm; references RSG-L18-s0 and RSGboth-L18-s0 from their stored reads): held 460 @18 (held.py common scenes, ol_confirm
paired), rf_follow turn completion (24 held turns), rf_inside, public 1491 @10 guard (ol_confirm), zeros_mech collect + diag.

**Lines** (all must hold for an arm to "meet the lines"):
1. held zeros <= RSG-L18-s0 + 3 (<= 37 on the 460 common scenes);
2. public guard zeros <= RSG-L18-s0 + 5 (<= 51);
3. held mean >= RSG-L18-s0 (0.8656) and guard mean >= RSG-L18-s0 (0.9480) (point estimates; paired CIs reported);
4. keep the turn completion: >= 19 / 24 (79 %) and "short" outcomes <= 2.
One seed: an arm that meets the lines is a candidate for a second seed, not a result.
