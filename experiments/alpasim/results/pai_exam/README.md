# PAI exam split (provisional v1, 2026-10-13)

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
