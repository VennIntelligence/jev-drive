# PAI exam split (v2 with the hard carve-out, 2026-10-13; v1 frozen in [v1/](v1/))

User decision 2026-10-13: the PAI track becomes the exam. Rule: over the scenes every reference read covers, **hard** = scored zero in any
reference; **control** = a random half (seed 0) of the never-zero scenes; **trainable** = the other never-zero half (PAI training-eligible).
Infrastructure failures (a zero whose failure_reason is not offroad / left_corridor_laterally / collision_at_fault, e.g. an RPC crash before
any drive call) are not zeros; such scenes are excluded from all three lists and listed in `infra_excluded.txt`.
Built by `experiments/alpasim/scripts/pai_exam_split.py`; counts and per-reference zeros in `split.json`.

**Provisional.** Only the 281 curated PAI test scenes (onpol_pai `test281`) have reads by every reference, so the split covers those 281,
not all PAI scenes with rollouts (the 160 `train160` scenes have one read only, RB0-F-s0's re-roll). References, serving `JEV_VCONT=1.0 JEV_LEAD=1`:

| reference | checkpoint | `rel_front_steering_angle_weight` | zeros (offroad / corridor / at-fault) | run root (Tokyo) |
|:--|:--|--:|:--|:--|
| ybase | YR10m10-F-s0 | **5 (default)** | 110 (37 / 59 / 14) | /data/runs/onpol_long/pai281/ybase |
| ys0 | YS-L18-s0 | **5 (default)** | 96 (26 / 57 / 13) | /data/runs/onpol_long/pai281/ys0 |
| ys1 | YS-L18-s1 | **5 (default)** | 89 (22 / 55 / 12) | /data/runs/onpol_long/pai281/ys1 |
| rsg | RSG-L18-s0 | 20 | 70 (14 / 47 / 9) | /data/runs/route_night/r281/rsg |

Counts: covered 281, infrastructure-excluded 0, **hard 131, control 75, trainable 75**.
Planned v2: ybase and ys0 re-read at weight 20 (`r281/ybase_g20`, `r281/ys0_g20`) and RBSG2-L18-s0 (weight 20) added; the membership
diff against v1 will be recorded here. The weight-5 references include the handoff spins that weight 20 removes (steer_long gains pilot),
so v1's hard half holds some scenes that are hard only at weight 5.

## Rule from v2 on (approved 2026-10-13)

v1 is frozen. From v2 on, membership only ever moves into **hard**: a v1 control / trainable scene that scores zero in any reference read
(non-infrastructure) moves to hard; every other scene keeps its v1 list. A scene outside v1 goes to hard if it is zero anywhere, else to
control or trainable by a stable hash (`sha1("<seed>:<scene id>")`, even first byte = control). `pai_exam_split.py --freeze <v1 dir>`.
The exam reads queued on Tokyo (`/data/runs/route_night/exam206.txt`) use v1 hard + control (206 scenes).

## v2: hard carve-out (user decision 2026-10-13)

The hard half is split in two by a stable per-scene hash: **hard_train** (`hard_train.txt`, training-eligible, like trainable) and
**hard_exam** (`hard_exam.txt`, never trained on, never used for checkpoint or recipe selection). Rule: u = first 8 bytes of
`sha1("1013:<scene id>")` read big-endian / 2^64; hard_train if u < 0.5. Hash seed **1013** (fixed before the split was drawn; not to
be changed to rebalance). Being per scene, the rule also places any scene that moves into hard later.

Built with `pai_exam_split.py --freeze experiments/alpasim/results/pai_exam/v1 --hard-train-frac 0.5 --hash-seed 1013 --refs <the four v1
refs> --tally turn48=... turn46=...`. Membership of hard / control / trainable is v1's exactly (same refs; the planned weight-20 refs
and RBSG2 are not added yet; when they are, the v2 rule above applies and the carve-out follows by hash). v1's lists and split.json are
frozen copies in `v1/`.

| list | scenes | route-follow 2.4 turn scenes (48) | logged turns >= 60 deg / 30 m (46) |
|:--|--:|--:|--:|
| hard | 131 | 47 | 42 |
| **hard_train** | **59** | 21 | 19 |
| **hard_exam** | **72** | 26 | 23 |
| control | 75 | 0 | 0 |
| trainable | 75 | 1 | 4 |

Turn lists: [research/turning-school-checks.md](../../../../research/turning-school-checks.md) (48 = zero rollouts failing at a
60-125 deg route turn; 46 = logged ego track turns >= 60 deg within 30 m in the 20 s window); their union is
`experiments/route_follow/results/taxonomy/p1_turn_scenes.txt`. The 59 / 72 imbalance is the hash's draw (binomial sd about 5.7 at
n = 131), kept as drawn. Training-eligible PAI scenes from v2 on: train160 + trainable 75 + hard_train 59. The exam reads are
hard_exam 72 + control 75; the queued v1 exam206 reads stay valid and are re-sliced to hard_exam for any training arm that saw hard_train.
