# Controller comparison

Partial snapshots retain all requested routes; missing results are not success.

| group | final / requested | mean completion observed | true completed | blocked observed | collisions observed | attempt wall s |
|---|---:|---:|---:|---:|---:|---:|
| dev10/carla-seed0 | 10 / 10 | 94.7 | 9 | 0 (10/10 known) | 9 (10/10 known) | 339.1 |
| dev10/tcp-seed0 | 10 / 10 | 94.9 | 9 | 0 (10/10 known) | 13 (10/10 known) | 367.9 |
| dev10/pursuit-seed0 | 0 / 10 | - | 0 | - (0/10 known) | - (0/10 known) | 0.0 |

Speed errors use the controller trajectory derivative, including its origin-to-first-point bridge. A fixed cruise setting of 8 m/s is not independent truth for this metric; route offset can inflate initial reference/command speed. G2 independent cruise metrics remain separate.

Vendor TickRuntime failures stay in the completion denominator; effective caps and artificial harness caps are distinct.

Pairs are right minus left; partial pairs cannot select a default.

| left | right | paired / expected | completion delta |
|---|---|---:|---:|
| dev10/carla-seed0 | dev10/tcp-seed0 | 10 / 10 | 0.2 |
| dev10/carla-seed0 | dev10/pursuit-seed0 | 0 / 10 | - |
| dev10/tcp-seed0 | dev10/pursuit-seed0 | 0 / 10 | - |
