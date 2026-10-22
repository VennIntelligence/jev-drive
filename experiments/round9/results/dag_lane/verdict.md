# DAG-LANE verdict (dl.py verdict)

Common scenes: 279. Lane-keeping = cor_drift, cor_lane_change, off_straight, off_junc_straight; high speed = the human's start speed >= 15 m/s; turn inside = off_junc_inside, off_curve_inside; other zeros = every zero outside lane-keeping.

| read | mean | zeros | lane-keep zeros | high-speed | other zeros | turn inside | collisions | straight at turn | lane-keep loss | turn-inside loss | 26004ca8 | 4cb05d65 | 7dc59d0d | 0e05cb0c |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 0.743 | 59 | 12 | 8 | 47 | 3 | 3 | 11 | 12.0 | 3.0 | 0.00 | 1.00 | 0.00 | 0.00 |
| NTV0-agH | 0.745 | 58 | 13 | 8 | 45 | 5 | 5 | 5 | 13.0 | 5.0 | 0.00 | 0.00 | 1.00 | 0.00 |
| R6DAG-s0-ag | 0.775 | 53 | 12 | 8 | 41 | 10 | 6 | 2 | 12.0 | 10.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0-agH | 0.792 | 46 | 8 | 4 | 38 | 6 | 5 | 2 | 8.0 | 6.0 | 0.00 | 0.00 | 1.00 | 1.00 |
| R6DAG-s1-ag | 0.815 | 43 | 8 | 6 | 35 | 7 | 6 | 2 | 8.0 | 7.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1-agH | 0.780 | 49 | 10 | 6 | 39 | 8 | 6 | 2 | 10.0 | 8.0 | 0.00 | 0.00 | 0.00 | 1.00 |
| R6DAG-s2-ag | 0.791 | 51 | 12 | 9 | 39 | 8 | 6 | 2 | 12.0 | 8.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s2-agH | 0.780 | 48 | 11 | 6 | 37 | 6 | 7 | 2 | 11.0 | 6.0 | 0.00 | 0.00 | 1.00 | 1.00 |
| R9DL-s0-ag | 0.786 | 51 | 10 | 7 | 41 | 9 | 5 | 2 | 10.0 | 9.0 | 0.00 | 0.00 | 1.00 | 0.00 |
| R9DL-s0-agH | 0.773 | 53 | 10 | 5 | 43 | 9 | 6 | 3 | 10.0 | 9.0 | 0.00 | 0.00 | 1.00 | 1.00 |
| R9DL-s1-ag | 0.800 | 47 | 7 | 4 | 40 | 7 | 7 | 2 | 7.0 | 7.0 | 0.00 | 0.00 | 1.00 | 0.00 |
| R9DL-s1-agH | 0.765 | 54 | 11 | 6 | 43 | 10 | 7 | 2 | 11.0 | 10.0 | 0.00 | 0.00 | 1.00 | 1.00 |

R6DAG 3-seed means of (OFF + ON) / 2: mean 0.7890, high-speed lane-keeping zeros 6.50, other zeros 38.17.

- R9DL-s0: (OFF + ON) / 2 0.7795 (fail); high-speed lane-keeping zeros 6.0 (pass); other zeros 42.0 (fail).
- R9DL-s1: (OFF + ON) / 2 0.7828 (fail); high-speed lane-keeping zeros 5.0 (pass); other zeros 41.5 (fail).

**Adoption rule (plan section 6): NOT ADOPTED.**
