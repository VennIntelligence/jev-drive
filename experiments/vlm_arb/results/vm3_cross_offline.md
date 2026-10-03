# vm3 cross-traffic release check: zero-shot Qwen3-VL-4B, offline on the vmerge-family frames

Script: `experiments/vlm_arb/scripts/vm3_cross_offline.py` (label / score / report; pre-registration in its docstring). Frames: every saved light-request frame pair (wide + road) of the vmerge, vmerge2, vmj, vmnobyp, vmnocusum, vmnor1 and dbg vm* units. Scoring = the server's path (forced `ANSWER:`, first-token logits of the two options, softmax, r1153, whole model); P = softmax probability of the positive option.

Label (privileged.jsonl, last snapshot <= t_q): positive = a vehicle (not ego, > 1 m/s, heading > 30 deg off the ego's) whose constant-velocity centre track within 3.0 s is inside the ego corridor (0-20 m ahead of the front bumper, |lat| <= 2.5 m). Main set = ego v < 0.5 m/s and junction window (rules R2 / R3, or 0 <= junc_dist <= 10 m). Routes alternate dev / test by id.

## Label prevalence

| set | frames | positive | prevalence |
|:--|--:|--:|--:|
| all saved frames | 23403 | 4787 | 0.205 |
| main (junction stop) | 5723 | 1332 | 0.233 |
| main dev | 3374 | 682 | 0.202 |
| main test | 2349 | 650 | 0.277 |

Positive ttc (main set): median 1.0 s, 217 of 1332 at ttc 0 (already in the corridor).

## Dev: prompt choice (main set, 1200 frames, 254 positive, 9 routes)

| prompt | dev AUC [route bootstrap 95%] | recall / FPR at argmax |
|:--|--:|--:|
| cross | 0.652 [0.478, 0.806] | 0.79 / 0.48 |
| go | 0.717 [0.570, 0.871] | 1.00 / 1.00 |
| side | 0.747 [0.599, 0.839] | 0.64 / 0.25 |

Chosen: **side** (highest dev AUC). Dev threshold (dev FPR = 0.10): logit margin >= 1.230, i.e. P(positive) >= 0.7738; argmax = margin >= 0.

Prompt (exact string):

```
The two images were taken at the same instant by the front cameras of a car (the ego vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera.
Vehicles approaching the ego vehicle path from the side. Options:
  side_vehicle_approaching: a moving vehicle coming from the left or the right, or an oncoming vehicle turning, will cross the lane directly ahead of the ego vehicle within the next few seconds
  no_side_vehicle: no vehicle will cross the lane directly ahead of the ego vehicle
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: side_vehicle_approaching, no_side_vehicle. Reply with that line only.
```

Options: positive `side_vehicle_approaching`, negative `no_side_vehicle`.

## Test (main set, the chosen prompt only)

| set | n | n pos | AUC [95%] | recall @dev thr | FPR @dev thr | recall @argmax | FPR @argmax |
|:--|--:|--:|--:|--:|--:|--:|--:|
| test main | 1206 | 340 | 0.875 [0.739, 0.909] | 0.93 | 0.66 | 0.95 | 0.74 |
| dev main | 1200 | 254 | 0.747 [0.599, 0.839] | 0.49 | 0.10 | 0.64 | 0.25 |
| all scored frames (secondary) | 2973 | 690 | 0.758 [0.615, 0.855] | 0.70 | 0.37 | 0.78 | 0.50 |
| all scored, ego moving (secondary) | 255 | 50 | 0.742 [0.571, 0.846] | 0.70 | 0.36 | 0.76 | 0.50 |

Unchosen `cross` on test main (for the record, not used): AUC 0.703.

Unchosen `go` on test main (for the record, not used): AUC 0.794.

**Pre-registered line** (test main, dev threshold: recall >= 0.6 and FPR <= 0.15): recall 0.93, FPR 0.66 (n pos 340, n neg 866) -> **not usable**.

Test main positives by time-to-corridor: <=0.5 s: 113/113, 0.5-1.5 s: 136/138, 1.5-3 s: 68/89

## Collision episodes: answers in the 3 s before contact

Contact time = scenario clock (contacts.jsonl frame -> privileged frame). P = P(side_vehicle_approaching); flag = P >= dev threshold.

| unit | route | contact t | other | frame t | ego v | label (ttc) | P | flag |
|:--|:--|--:|:--|--:|--:|:--|--:|:--|
| v2-vmerge-s1 | 27297 | 33.80 | chevrolet.impala | 31.05 | 1.11 | neg | 0.773 | - |
| v2-vmerge-s1 | 27297 | 33.80 | chevrolet.impala | 31.55 | 1.83 | neg | 0.811 | STOP |
| v2-vmerge-s1 | 27297 | 33.80 | chevrolet.impala | 32.05 | 3.20 | neg | 0.843 | STOP |
| v2-vmerge-s1 | 27297 | 33.80 | chevrolet.impala | 32.55 | 4.08 | pos (1.6 s) | 0.837 | STOP |
| v2-vmerge-s1 | 27297 | 33.80 | chevrolet.impala | 33.05 | 4.67 | pos (0.4 s) | 0.984 | STOP |
| v2-vmerge-s1 | 27297 | 33.80 | chevrolet.impala | 33.55 | 4.24 | pos (0.1 s) | 0.979 | STOP |
| v2-vmerge-s1 | 17280 | 17.85 | nissan.patrol_2021 | 15.05 | 4.30 | neg | 0.910 | STOP |
| v2-vmerge-s1 | 17280 | 17.85 | nissan.patrol_2021 | 15.55 | 4.69 | neg | 1.000 | STOP |
| v2-vmerge-s1 | 17280 | 17.85 | nissan.patrol_2021 | 16.05 | 4.53 | pos (2.7 s) | 0.999 | STOP |
| v2-vmerge-s1 | 17280 | 17.85 | nissan.patrol_2021 | 16.55 | 4.22 | pos (2.8 s) | 1.000 | STOP |
| v2-vmerge-s1 | 17280 | 17.85 | nissan.patrol_2021 | 17.05 | 4.32 | pos (1.1 s) | 0.999 | STOP |
| v2-vmerge-s1 | 17280 | 17.85 | nissan.patrol_2021 | 17.55 | 4.40 | neg | 0.861 | STOP |
| v2-vmerge2-s1 | 17280 | 17.50 | nissan.patrol_2021 | 14.55 | 1.96 | neg | 0.854 | STOP |
| v2-vmerge2-s1 | 17280 | 17.50 | nissan.patrol_2021 | 15.05 | 3.53 | neg | 0.899 | STOP |
| v2-vmerge2-s1 | 17280 | 17.50 | nissan.patrol_2021 | 15.55 | 4.47 | neg | 0.943 | STOP |
| v2-vmerge2-s1 | 17280 | 17.50 | nissan.patrol_2021 | 16.05 | 4.60 | pos (2.9 s) | 0.999 | STOP |
| v2-vmerge2-s1 | 17280 | 17.50 | nissan.patrol_2021 | 16.55 | 4.53 | pos (3.0 s) | 1.000 | STOP |
| v2-vmerge2-s1 | 17280 | 17.50 | nissan.patrol_2021 | 17.05 | 3.91 | pos (0.7 s) | 1.000 | STOP |
| v2-vmerge2-s2 | 17280 | 17.70 | nissan.patrol_2021 | 15.05 | 4.25 | neg | 0.968 | STOP |
| v2-vmerge2-s2 | 17280 | 17.70 | nissan.patrol_2021 | 15.55 | 4.69 | neg | 0.992 | STOP |
| v2-vmerge2-s2 | 17280 | 17.70 | nissan.patrol_2021 | 16.05 | 4.53 | pos (2.6 s) | 1.000 | STOP |
| v2-vmerge2-s2 | 17280 | 17.70 | nissan.patrol_2021 | 16.55 | 4.32 | pos (2.7 s) | 1.000 | STOP |
| v2-vmerge2-s2 | 17280 | 17.70 | nissan.patrol_2021 | 17.05 | 4.31 | pos (0.5 s) | 0.999 | STOP |
| v2-vmerge2-s2 | 17280 | 17.70 | nissan.patrol_2021 | 17.55 | 4.13 | pos (0.0 s) | 0.952 | STOP |
| v2-vmerge2-s3 | 17280 | 18.20 | nissan.patrol_2021 | 15.55 | 4.18 | neg | 0.905 | STOP |
| v2-vmerge2-s3 | 17280 | 18.20 | nissan.patrol_2021 | 16.05 | 4.65 | neg | 0.999 | STOP |
| v2-vmerge2-s3 | 17280 | 18.20 | nissan.patrol_2021 | 16.55 | 4.48 | neg | 0.991 | STOP |
| v2-vmerge2-s3 | 17280 | 18.20 | nissan.patrol_2021 | 17.05 | 3.88 | pos (2.6 s) | 1.000 | STOP |
| v2-vmerge2-s3 | 17280 | 18.20 | nissan.patrol_2021 | 17.55 | 4.16 | pos (1.5 s) | 0.994 | STOP |
| v2-vmerge2-s3 | 17280 | 18.20 | nissan.patrol_2021 | 18.05 | 4.19 | pos (0.3 s) | 0.994 | STOP |

## Per-route readout of the chosen prompt (main set, dev threshold margin 1.230)

| part | route | n | n pos | FPR | recall | median margin of negatives |
|:--|:--|--:|--:|--:|--:|--:|
| dev | 9196 | 175 | 5 | 0.14 | 0.80 | -0.23 |
| dev | 15483 | 279 | 43 | 0.00 | 0.70 | -2.69 |
| dev | 16390 | 67 | 0 | 0.00 | n/a | -4.41 |
| dev | 16529 | 262 | 97 | 0.25 | 0.51 | 0.39 |
| dev | 24944 | 240 | 44 | 0.11 | 0.09 | -4.59 |
| dev | 27043 | 61 | 12 | 0.12 | 1.00 | -0.18 |
| dev | 27870 | 111 | 48 | 0.05 | 0.50 | -5.27 |
| test | 15102 | 491 | 202 | 0.93 | 1.00 | 3.16 |
| test | 15612 | 224 | 45 | 0.54 | 0.76 | 1.66 |
| test | 16508 | 81 | 10 | 0.00 | 0.00 | -3.48 |
| test | 17280 | 175 | 10 | 0.67 | 1.00 | 3.01 |
| test | 27297 | 153 | 73 | 0.74 | 0.97 | 1.92 |
| test | 28147 | 82 | 0 | 0.43 | n/a | 0.85 |

(dev 334 and 37969 have 1 and 4 frames.)

## Reading

- The question ranks frames within a route (test AUC 0.875) but its level is set by the scene: on the busy test junctions (15102,
  17280, 27297, 15612) the median negative already sits above the dev threshold, so FPR at the dev threshold is 0.66 and at argmax
  0.74. The model answers "is there cross traffic at this junction", not "will a vehicle enter my path within 3 s". A route-independent
  threshold does not exist on these data; the pre-registered line fails on FPR (recall passes).
- Collision episodes: in all five the ego was moving at 1-4.7 m/s in the 3 s before contact (scenario clock from the contact frame
  number); the "standing at the stop sign" reading of 17280 in vmerge_collisions.md came from the world clock, which runs about 1 s
  ahead. The question flags STOP on 29 of 30 frames before contact, but it also flags the negative frames before them at the same
  level, consistent with the scene-level bias above; at 27297 it was already at P 0.77-0.84 right after the cusum release (31.05-32.05,
  label negative).
- As a release check it would hold the ego on most stops at busy junctions (FPR 0.66 on test junction-stop frames), i.e. it would
  trade the crossing collisions for stalls / timeouts. Not usable as is.
