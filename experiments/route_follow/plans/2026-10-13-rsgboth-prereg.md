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

## Amendment 2: side-bias fix 1, the intervention test (RSGbothSB, seeds 0 and 1)

**Seen before this amendment**: [research/route-follow/side-bias.md](../../../research/route-follow/side-bias.md) (RSGboth's right-turn inside
cut comes from the pool's state mix: in on1_L18RB0 the right-turn rows outside / inside the human path by > 0.5 m are 321 : 77, left 325 : 278;
the `both` imitation push at 2 s is -0.38 m on right turns vs -0.14 m on left; held plan lat 2 s RSGboth - RSG on right turns -0.33 m
[-0.66, -0.15], left +0.06 [-0.07, +0.19]). Nothing of the new arm has been trained or read.

**Pool** on1_L18RB0s ([rf_sidetag.py](../scripts/rf_sidetag.py) pool): on1_L18RB0 hard-linked, tab gains `tside` (L / R / none), `pside`
(in / on / out / none) and `rside` = `<tside>_<pside>` from rf_sidebias's own classification; nothing else changed. Gate R1 PASS (other files
same inode, every source tab array byte-equal, feature file byte-equal, counts reproduce side-bias.md: R_out 321, R_in 77, L_out 325, L_in 278).
Dry run of `--on-weight rside=R_in:2,rside=R_out:0.48` with rm_train's parsing and generator (1000 batches x 13 rows): drawn R_out / R_in
107 / 98 (1.09 : 1; uniform 245 / 56 = 4.4 : 1), every other class at its uniform rate; weighted mean right-turn push at 2 s -0.38 -> -0.22 m
(left -0.14, unchanged). The registered weights make right-turn out : in exactly 1 : 1 in expectation (77 x 2 = 321 x 0.48).

**Arm** RSGbothSB-L18-s{0,1} = RSGboth-L18-s{k}'s exact command plus `--on on1_L18RB0s --on-weight rside=R_in:2,rside=R_out:0.48`; pool jobs on
cards 1-5, 60-step preflight smoke; chain [rf_rsgbsb.sh](../scripts/rf_rsgbsb.sh). The arm is orthogonal to Amendment 1's arms (they change the
agent weight, the non-turn draw rate or the non-turn target; this one only the right-turn draw rate by path side).

**Reads** (as RSGboth's): held 460 @18 (held.py, ol_confirm paired vs RSG-L18-s{k} and RSGboth-L18-s{k} and the seed means), rf_follow turn
completion (24 held turns), rf_inside, rf_tax extract + rf_sidebias held -> rf_sidetag.py read (plan / driven lateral at 2 s per side, paired by
scene), rf_ledger per-side peaks (loose L / R: inside peak on right turns, outside peak on left turns vs the human path), public 1491 @10 guard.

**Lines** (seed mean where both seeds exist; per seed reported):
1. right-turn inside shift vs RSG gone: held plan lat 2 s on right turns, RSGbothSB - RSG paired by scene, 95 % CI includes 0 (supporting: ledger
   loose-R inside peak RSGbothSB - RSG CI includes 0, RSGbothSB - RSGboth > 0);
2. turn completion kept: >= 85 % of the 24 held turns (rf_follow);
3. held zeros <= RSG-L18 + 3, or at least < RSGboth-L18 (same seed);
4. public guard: zeros <= RSG-L18 + 5 and mean >= RSG-L18 (point estimates, paired CIs reported).
Line 1 is the intervention's answer to side-bias.md's inference; lines 2-4 decide whether the fix is usable. If line 1 holds and left turns
also stay unchanged (left plan lat 2 s CI vs RSGboth includes 0), the side-bias mechanism is confirmed causally.
