# SPIN v9 pre-registration (2026-10-15, written before any v9 closed-loop read)

Line B of research/next-round/turn-night.md section 8. Serve side only, no training. The user: the earlier spin fix did not fix it; account
for the new situation (APP1X-L18-s0, decision 303: 29 handoff spins vs 2 for RSG-L18-s0 on PAI 281 at gains 20) and fix the controller-side
protection properly. **If v9 passes the lines below on both checkpoints it is adopted as the PAI serve default (the user's decision).**

Background: decision 285 (the handoff spin is the controller's first step after the 1.7 s force-GT handoff when the plan is shorter than the
speed implies; AlpaSim's MPC tracks positions 1-2 s ahead and loses distance it cannot brake away by steering), decision 291 (v8 =
`JEV_TRACK=gate` + PP check + tilted-pose anchor guard, experiments/alpasim/lib/track_ref.py; RSG spin set 26 -> 1, the one left a simulator
pose fault; not adopted on the letter of its lines), decision 292 (holdlead), decision 303 (APP1X).

Fact found before writing (driver.log of both stored reads: `JEV_TRACK 0`): **v8 was never served in the APP1X read nor in the r281/rsg
identity.** v8 stayed a switch (not adopted), so APP1X's 29 spins say nothing yet about v8; they are the old control path.

## Spin detector, sets, checkpoints, identity

- Spin = `experiments/steer_long/scripts/sf_detect.py`, any kind (handoff |vx wz| > 6 m/s^2 in (1.7, 3.2] s; fast; slow), the d291 detector,
  unchanged. rf_pai281's handoff spin (the 29 / 2 of d303) is reported alongside.
- `results/spin_v9/app1x_spin29.txt`: the 29 APP1X-L18-s0 handoff-spin scenes of its stored PAI 281 read (rf_pai281 rule, sv9_diag.py).
- `experiments/steer_long/results/spin_fix/spin_set.txt`: d291's 38 spin scenes (RSG-class); `ctrl_all.txt`: d291's 46 controls
  (fast 20, slow 16, brake 10; clean in the RSG identity).
- `results/spin_v9/union90.txt`: the three deduplicated, 90 scenes (13 of the 29 are in the 38; 10 of the 29 are in the 46 controls: the
  controls were chosen clean for RSG, APP1X spins on them).
- Per-checkpoint sets. RSG-L18-s0: spin set = the 38, controls = the 46. APP1X-L18-s0: spin set = the 29, controls =
  `results/spin_v9/ctrl_app1x.txt` = the 46 minus the 13 scenes where the APP1X identity has any sf_detect spin (33; a control measures harm
  where the identity does not spin); all 46 reported.
- Checkpoints (Tokyo copies sha256-equal to the box, checked 2026-10-15): APP1X-L18-s0 64e0bbc7..., RSG-L18-s0 fbe7ac9f....
- Identity = the stored gains-20 reads, which cover all 90 scenes for both: APP1X `/data/runs/route_night/v2pai/APP1X-L18-s0`, RSG
  `/data/runs/route_night/r281/rsg` (both `JEV_VCONT 1, JEV_LEAD 1, JEV_TRACK 0`, gains 20). No identity rerun.

## Arms (Tokyo, gains 20, `JEV_VCONT=1.0 JEV_LEAD=1`, one stack per card, `scripts/sv9.sh`)

- **v9a** `JEV_TRACK=v9`: v8 (gate + PP + tilt guard, unchanged) + the handoff guard: while the session is younger than H_T = 2.7 s (first
  drive call = 0; the first 10 policy decisions after the 1.7 s handoff) and v0 >= H_V = 5 m/s, the reference's deceleration bound is
  H_AMIN = -2.5 m/s^2 instead of -5 (re-timing from the car's speed along the plan's own path, jerk from the measured acceleration as in v8);
  the lateral side stays v8's (gate + PP: the curvature-capped full reference where the plan or the arc onto it exceeds the cap); not when
  the lead limit is cutting the served speed (then -5 stays: a real lead is never braked for more softly). APP1X and RSG on the 90 scenes,
  one card each.
- **v9b**, at most one more variant, run only if v9a leaves > 2 spins on a spin set and at least one of them starts in the handoff window
  (onset <= 3.2 s): H_AMIN = 0 (re-time to the fed speed, hold it through the window) and H_FULL = 1 (full, curvature-capped reference on
  every guarded call). Same scenes, both checkpoints. Nothing is tuned on the controls.
- **v8 diagnostic** (not a variant, not gated): APP1X-L18-s0 with `JEV_TRACK=gate` on the 29 only, to answer whether v8 alone already
  removes the new regime. RSG's v8 numbers are d291's read (`r281/spin_read2` + `v8_ec1d`).

## Gates before the full read

- Offline: `sv9_check.py same`: v9 with the guard off, outside its window or below H_V equals v8 bit for bit (3000 random plans x 3 cases).
  Run 2026-10-15: 0 / 9000 differ. With `JEV_TRACK` unset the driver's code path is the unchanged else-branch.
- 2-scene smoke (`sv9.sh smoke`, APP1X, the first two of the 29): switch off -> served plans identical to the stored identity on every
  common decision; v9 on -> guard records (`track.hand`) on k 17 .. 26 only.

## Lines (each checkpoint, `sv9_read.py`)

1. Spins (sf_detect, any kind) on the checkpoint's spin set <= 2 (the RSG level of d303).
2. Controls: zeros not up, i.e. the paired bootstrap 97.5 % bound of the summed zero difference (scenes resampled, 10 000 draws, seed 0)
   <= +2, and the paired mean score difference >= -0.01.
3. No new offroad zero whose event (the controller's last row: a zero ends the rollout) is <= 4.7 s (3 s after the handoff), over the 90.

Pass = all three on both checkpoints -> v9 is the PAI serve default. Reported, not gated: zeros / mean on the spin sets, at-fault / front /
rear collisions, handoff spins, all 46 controls for APP1X, the union, per scene.

Arithmetic of line 2, stated before the read: with the percentile bootstrap a +1 net (2 new, 1 recovered) reads [-2, +5] and 1 new / 0
recovered reads [0, +3], so line 2 holds in practice only with no new control zero that is not offset. d291's v8 read on the RSG controls
(2 new: 268949d2 gate at low speed, 09528162 curv re-timing; 1 recovered) reads +1 [-2, +5] under this reader. The guard does not act below
5 m/s nor after 2.7 s, so **prediction: RSG line 2 fails as under v8 if those two zeros reproduce (closed-loop reruns of one configuration
were identical per scene on PAI); lines 1 and 3 pass on both checkpoints.** Not tuned away; reported.

## Amendment 1 (mechanism, from the stored logs, before any v9 read; `sv9_diag.py`, `sv9_check.py replay`, results/spin_v9/diag_*)

Handoff decisions k 17 .. 27 of both stored reads, CPU only.
- All 29 APP1X spins start 0.1-0.2 s after the handoff (onset 1.8 s in 27, 1.9 s in 2), at 22-36 m/s (median 27). The steering command
  in 1.7-2.2 s is 0.69-1.05 rad (median 0.78) against 0.00-0.04 in the non-spinning rollouts.
- The plan at k 17 is nearly straight (|y| at 2 s median 0.44 m; v8's gate ratio median 0.21, opens at k 17 in 3 / 29), so the lateral side
  is not the cause. The longitudinal gap is: fed speed minus the model plan's speed over 0.5-1 s is 7.3-11.5 m/s (median 8.4) in the spins,
  2.6-6.2 (median 5.1) in APP1X's non-spinning rollouts at >= 15 m/s, 0.6-3.5 for RSG; at >= 15 m/s every APP1X rollout with a gap >= 7
  m/s spins at the handoff except 2 (29 / 31); RSG has 3 such rollouts, 2 of them its handoff spins (38 m/s, gap 7.5-8.2 m/s). The served
  (JEV_VCONT) 2 s arc is 12.5-20 m (median 14.5) short of 2 x the fed speed in the spins, 4.9-11 m in the non-spinning fast rollouts.
- Why the gap becomes a spin: JEV_VCONT blends from the fed speed into the plan's speed over exactly 1 s (serve_fix.py), i.e. a deceleration
  demand of gap / 1 s = 7-12 m/s^2 at the MPC's -9 m/s^2 bound before the 1-2 s tracked window; the MPC loses the remaining distance by
  steering (decision 285's mechanism, at gaps RSG almost never makes: APP1X learned to be slow before turns and the force-GT state is the
  log's speed).
- v8 replayed offline on the logged plans (same speed / acceleration): its re-timing bounds the reference's speed drop at 1.5 s to 7.4 m/s
  and the 2 s deficit to 10.3 m in every spin rollout (the gate itself rarely opens: 3 / 29 at k 17); v9a halves that (3.7 m/s, 5.1 m).
  The non-spinning APP1X rollouts at >= 15 m/s carried served deficits up to 11 m without spinning, so v8 alone sits at that edge and v9a
  well inside it; the v8 diagnostic arm reads it. The guard fires on 10 decisions per rollout at >= 5 m/s; 976 of 1 750 such APP1X calls
  (726 of 2 019 RSG) have the lead limit active and keep -5.
- Not supported: a ratio trigger (served arc < 0.7 x 2 x v0) at k 17: the spins read 0.64-0.78 there (only about a third below 0.7), so the
  guard bounds the deceleration in the whole window instead of triggering (the bound only binds when the plan asks for more).

## Amendment 2 (gates and launch, 2026-10-15 14:50 JST, before any v9 read)

- Offline: `sv9_check.py same` 0 / 9000 differ from v8 (commit acb84875).
- Smoke (Tokyo card 0, `/data/runs/route_night/spin_v9/smoke_{off,v9}`, 05855d57 and 0e05cb0c): switch off -> 398 / 398 served plans
  identical to the stored APP1X identity (max |diff| 0). v9 on -> guard records on k 0 .. 27 of both sessions (k 27 is inside: t_s is
  2.6999.. in floating point, so the window holds the 11 policy decisions k 17 .. 27, not 10; kept as run), both handoff spins gone
  (0.457 and 0.379 vs 0 / 0). Two scenes say nothing about the lines.
- Launch: v9a APP1X-L18-s0 on card 0, then the v8 diagnostic (APP1X, 29 scenes) on card 0; v9a RSG-L18-s0 on card 1 (freed by the
  rule-expert lane's smoke). Both at gains 20 (wizard_cmd checked), `JEV_VCONT 1, JEV_LEAD 1`, CHUNK 24, RUNPFX v9-. A first card-0-only
  chain was stopped after 8 min (no rollout finished) to use card 1, and restarted from scratch.
