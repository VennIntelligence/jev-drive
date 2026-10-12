# Inside-of-turn offroad zeros, held 460, all route-night arms (rf_offroad.py held)

Same classifier as research/route-follow-offroad.md table 1 (turn = route heading change >= 20 deg over +-20 m at the zero point;
inside = ego on the turn side of the route). The first eight labels are offroad/held.parquet (2026-10-13 02:28 CST, the diagnosis run);
RSG-L18 / RBSGc-L18 were run with the same script on 2026-10-13 ~04:17 JST into `$DATA_DIR/runs/route_follow/offroad_a3/`
(`rf_offroad.py held --runs RSG-L18-s0=... RSG-L18-s1=... RBSGc-L18-s0=... RBSGc-L18-s1=... --workers 12`).

| label | offroad zeros | junction / approach | turn zeros | inside right | inside left |
|:--|--:|:--|--:|--:|--:|
| RB0-F-s0 | 29 | 5 / 4 | 19 | 1 | 7 |
| RB0-F-s1 | 37 | 13 / 8 | 28 | 2 | 10 |
| RNONEG-L18-s0 | 31 | 10 / 2 | 21 | 1 | 9 |
| RB-F-s0 | 49 | 19 / 12 | 39 | 17 | 14 |
| RBS-L18-s0 | 71 | 18 / 19 | 50 | 20 | 18 |
| RBS-L18-s1 | 64 | 20 / 16 | 48 | 21 | 19 |
| RBSG-L18-s0 | 40 | 20 / 9 | 31 | 19 | 8 |
| RBSG-L18-s1 | 43 | 18 / 10 | 36 | 21 | 7 |
| RSG-L18-s0 (no route hinge) | 17 | 8 / 3 | 11 | 1 | 6 |
| RSG-L18-s1 (no route hinge) | 13 | 7 / 2 | 8 | 1 | 5 |
| RBSGc-L18-s0 (hinge on cov20 rows only) | 27 | 9 / 6 | 20 | 5 | 9 |
| RBSGc-L18-s1 (hinge on cov20 rows only) | 30 | 11 / 6 | 23 | 10 | 6 |

Drop-one reading: removing the route hinge from RBSG (RSG) brings inside-right offroad zeros from 19 / 21 back to 1 / 1 (the base's 1 / 2);
keeping it only on rows whose memory covers +20 m (RBSGc) halves them (5 / 10).
