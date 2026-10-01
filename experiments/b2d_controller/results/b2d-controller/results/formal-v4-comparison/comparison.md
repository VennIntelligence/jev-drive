# Controller comparison

Partial snapshots retain all requested routes; missing results are not success.

| group | final / requested | mean completion observed | true completed | blocked observed | collisions observed | attempt wall s |
|---|---:|---:|---:|---:|---:|---:|
| formal-v4/carla-seed0 | 10 / 10 | 94.7 | 9 | 0 (10/10 known) | 11 (10/10 known) | 346.8 |
| formal-v4/carla-seed1 | 10 / 10 | 90.8 | 8 | 0 (10/10 known) | 12 (10/10 known) | 398.0 |
| formal-v4/tcp-seed0 | 10 / 10 | 85.2 | 8 | 0 (10/10 known) | 8 (10/10 known) | 369.3 |
| formal-v4/tcp-seed1 | 10 / 10 | 85.2 | 8 | 0 (10/10 known) | 8 (10/10 known) | 365.5 |
| formal-v4/pursuit-seed0 | 10 / 10 | 92.6 | 8 | 0 (10/10 known) | 9 (10/10 known) | 357.9 |
| formal-v4/pursuit-seed1 | 10 / 10 | 92.6 | 8 | 0 (10/10 known) | 9 (10/10 known) | 357.3 |

Speed errors use the controller trajectory derivative, including its origin-to-first-point bridge. A fixed cruise setting of 8 m/s is not independent truth for this metric; route offset can inflate initial reference/command speed. G2 independent cruise metrics remain separate.

Vendor TickRuntime failures stay in the completion denominator; effective caps and artificial harness caps are distinct.

Pairs are right minus left; partial pairs cannot select a default.

| left | right | paired / expected | completion delta |
|---|---|---:|---:|
| formal-v4/carla-seed0 | formal-v4/tcp-seed0 | 10 / 10 | -9.5 |
| formal-v4/carla-seed0 | formal-v4/pursuit-seed0 | 10 / 10 | -2.1 |
| formal-v4/carla-seed1 | formal-v4/tcp-seed1 | 10 / 10 | -5.6 |
| formal-v4/carla-seed1 | formal-v4/pursuit-seed1 | 10 / 10 | 1.8 |
| formal-v4/tcp-seed0 | formal-v4/pursuit-seed0 | 10 / 10 | 7.4 |
| formal-v4/tcp-seed1 | formal-v4/pursuit-seed1 | 10 / 10 | 7.4 |
