# Pre-registration: turning school P2, branch ceiling (log to the junction, then the policy), 2026-10-13

Context: [turning-school-checks.md](../../../research/turning-school-checks.md) (route C of 2.2: the log-following driver hands over to the
policy at decision k; PAI turn scenes: hard 42 of 46 on 281), [zero-taxonomy](../../../research/zero-taxonomy.md) (PAI RSG-L18-s0: straight at
the junction 11, turned too little 11; turns are 16 % of the 281 scenes and 44 % of the zeros). Written, committed and pushed before any pilot
rollout is submitted. Amendments are appended, never edited in place.

**Question.** When the ego is brought to the junction by the human log (right lane, right speed, model fed and warm the whole way) and the
policy takes over at a fixed distance before the turn entry, does the policy complete the turn? If it does from some handover point in most
scenes, branch demonstrations from good entry states are worth building (route (b): exact branching from the policy's own states). If it
does not even from the entry, the policy cannot execute the turn and the school must teach how to turn (source 1 / 2 rows), not only when.

## Instrument

- Driver: `experiments/onpol_pai/lib/onp_log_driver.py` with `ONP_SWITCH` (a file `<scene> <k>`): up to decision k-1 it returns the log
  (MPC tracks it, `send_recording_ground_truth=true`); from k on it returns the model's plan exactly as `pai_driver.py` (same serving
  switches, same record). Decision k = the session's k-th `drive` call (10 Hz). A k below 17 hands over at the end of the dev preset's 1.7 s
  force-GT (k_eff = max(k, 17); k_eff 17 is the plain policy rollout).
- Stack: `experiments/route_follow/scripts/rf_branch.sh` -> `onp_pass2.sh` (MODE=driver) -> `pai_run.sh`, Tokyo, one free card, CONC 4.
  Checkpoint **RSG-L18-s0**, serving `JEV_VCONT=1.0 JEV_LEAD=1` (the 281 reads), wizard `controller.gains.rel_front_steering_angle_weight=20`,
  image `jev-alpasim:p2h10-f-s0-6f3d05d1`, HARMONIZER 0, full 20 s rollouts.
- Smoke gates before the pilot (1 scene, `f9e361aa`): (S1) no `ONP_SWITCH`: decision-0 plan identical to the committed driver before the
  change and the ego track within the d241 non-determinism (median position difference < 0.1 m); (S2) `ONP_SWITCH=0`: decision-0 plan
  identical to the stored RSG-L18-s0 281 rollout of that scene (pai_driver, gains 20; `/data/runs/route_night/r281/rsg`), the rest within
  d241's spread. The smoke also measures OFF, the log time of decision 0 (`log_t` of record k 0); if it is not 0.0 +- 0.05 s every k
  below is shifted by round(10 OFF) (mechanical, recorded in an amendment, no other change).

## Scenes (12, fixed now): hard-list turn scenes (>= 60 deg within 30 m, turning-school-checks.md 1)

Chosen from the 42 by: direction (6 left, 6 right), RSG-L18-s0 taxonomy class first "straight at the junction" (`cor_straight_at_turn`) or
"turned too little" (`cor_turn_under`), then other turn failures; the log must reach exit + 10 m inside the rollout (t_target <= 18.1 s,
which drops b680a689, bd97de34, cdf95fd1, 52003cee, 68e53160, 300465f7); among the rest those whose -40 m point lies after the 1.7 s
force-GT (four distinct branches) are preferred. U-turns (> 125 deg) are avoided except 302c5c99 (the only remaining left
"straight at the junction" with a reachable target).

| scene | side | log turn | RSG class (281) | entry (m along the log) | t_target (s) |
|:--|:--|--:|:--|--:|--:|
| e5e47fd2 | L | 97 | cor_straight_at_turn | 51 | 12.8 |
| 302c5c99 | L | 183 | cor_straight_at_turn | 29 | 12.6 |
| e37aa159 | L | 74 | cor_turn_under | 34 | 17.6 |
| fd01e9ab | L | 92 | cor_turn_under | 2 | 11.0 |
| ce161092 | L | 82 | off_curve_outside | 50 | 18.1 |
| 382a7247 | L | 72 | coll_veh_oncoming | 21 | 9.4 |
| f9e361aa | R | -77 | cor_straight_at_turn | 53 | 11.8 |
| eaba3f6a | R | -87 | cor_straight_at_turn | 51 | 15.2 |
| a174faed | R | -96 | cor_straight_at_turn | 42 | 14.8 |
| 2f227f2c | R | -68 | cor_straight_at_turn | 24 | 14.1 |
| 3ce13eed | R | -98 | cor_turn_under | 17 | 12.4 |
| 94c1d722 | R | -81 | off_junc_outside | 76 | 12.8 |

(entry / t_target from the turning-school scratch read; the authoritative values and the k of every branch are those `rf_branch.py plan`
writes from the logged rig path into `plan.json`, committed with the results.)

## Arms

Per scene 4 branches, handover at the log point entry - 40 / -20 / -10 / 0 m (arms d40, d20, d10, d0): k = round(10 * t_log(entry - d)).
Where entry - d is before the ego leaves the force-GT, the branch is the plain policy rollout (k_eff 17); such branches are reported as
such (fd01e9ab: d40, d20, d10 all k_eff 17), not dropped. Reference column k17 = the stored 281 rollout of the same checkpoint
(`/data/runs/route_night/r281/rsg`, pai_driver from the start). 48 rollouts, about 1 h on one card.

## Measures (rf_branch.py read; definitions fixed here)

- entry / exit / target on the log: entry = first point of the first 30 m window reaching 60 deg that departs 5 deg from the window start;
  exit = first point within 5 deg of the largest heading change from the entry inside entry + 40 m; target = exit + 10 m (arc along the log).
- **turn completed (primary)**: the ego's station along the log (monotone windowed projection of the controller trace) reaches the target
  while the ego is within 2.5 m of the route (AlpaSim RouteGenerator MAP, rebuilt at the ego's 0.5 s poses as rf_tax.extract), before its
  zero event (rf_tax.classify: the earliest flag that zeroed it) and before the rollout ends (rear-end truncation included).
- zero type (collision / offroad / corridor / other, rf_tax.classify), score.
- corridor distance at exit: the ego's distance to the log when its station first reaches the exit (none if never).
- handover state: the ego's offset from the log and its fed speed at the handover decision (drive.jsonl `log_now`, `ego`).

## Lines

- **Ceiling** = share of the 12 scenes with at least one completing branch among d40 / d20 / d10 / d0.
- Ceiling >= 70 % (>= 9 / 12): from the right entry state the policy turns; build branch demonstrations, route (b) (exact branching from
  the policy's own states) next.
- Ceiling <= 40 % (<= 4 / 12): the policy cannot turn even from the right entry state at the right speed; the school must teach how
  (source 1 / 2 rows), not only when.
- 5 - 8 / 12: no call from the ceiling alone; the dependence on handover distance and the failure types decide what the report recommends,
  stated as weaker evidence.
- Secondary (descriptive, no line): completion per arm (expected to rise as the handover moves towards the entry if the deficit is "when";
  flat if it is "how"); zero types per arm; corridor distance at exit per arm; left vs right.

## Not claimed

12 scenes, one seed, one checkpoint: the ceiling is a gate for the next build, not an estimate of a rate. The handover is near the log, not
from the policy's own states (route C, not A); a completion here says the policy can execute the turn from a human entry state.
