# Recovery rows: pools RECA,RECB,RECC

33250 rows, 3692 episodes (rows k0 - 4 .. k0 + 4), 676 scenes.

| | turn | curve | straight |
|:--|--:|--:|--:|
| episodes | 473 | 621 | 2598 |
| rows | 4263 | 5592 | 23395 |

Episodes by type: {'lat': 1870, 'both': 527, 'head': 1295}; |e0| (m): {'1.5': 522, '0.5': 1061, '1.0': 814, '0.0': 1295}; |psi0| (deg): {'0.0': 1870, '3.0': 962, '10.0': 243, '6.0': 617}.

| log speed at t0 (m/s) | episodes | mean abs e0 (m) | mean abs psi0 (deg) |
|:--|--:|--:|--:|
| 2-5 | 358 | 0.41 | 4.5 |
| 5-10 | 1517 | 0.48 | 3.3 |
| 10-20 | 1263 | 0.62 | 1.7 |
| >=20 | 554 | 0.86 | 0.4 |

Realised vs intended offset (the MPC tracking the perturbed reference): |e_real - E| median 0.021 m, p90 0.062 m; at t0 with |E| >= 0.4 m e_real / E median 0.9809727668762207; heading error median 0.32 deg.

Dropped at select: A: {'collision': 27}; B: {'collision': 44, 'missing_scene': 18}; C: {'collision': 61, 'missing_scene': 9}
