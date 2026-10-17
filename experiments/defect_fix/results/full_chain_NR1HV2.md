# NR1HV2 (on-k 64, 10000 steps, lateral 1.0v2) held summary against RSG / same-seed NR1F / NR1HV

| tag | read | scenes | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| NR1HV2-L18-s0 | held | 460 | 51 | 0.8520 | 6 | 26 | 40 | 3 / 3 / 27 | 0.829 | 11 / 0 | vs RSG-L18-s0 -0.0136 [-0.0336, +0.0164] (zeros 51 / 34); vs NR1F-L18-s0 -0.0280 [-0.0460, -0.0157] (zeros 51 / 39); vs NR1HV-L18-s0 -0.0282 [-0.0463, -0.0136] (zeros 51 / 37) |
| NR1HV2-L18-s1 | held | 460 | 52 | 0.8528 | 7 | 23 | 40 | 4 / 3 / 28 | 0.830 | 8 / 0 | vs RSG-L18-s0 -0.0128 [-0.0316, +0.0153] (zeros 52 / 34); vs NR1F-L18-s1 -0.0258 [-0.0548, -0.0019] (zeros 52 / 41); vs NR1HV-L18-s1 -0.0322 [-0.0520, -0.0073] (zeros 52 / 37) |
| RSG-L18-s0 | held | 460 | 34 | 0.8656 | 4 | 17 | 22 | 4 / 0 / 38 | 0.788 | 7 / 0 |  |
| RSG-L18-s0 | pub | 1491 | 58 | 0.9397 | 8 | 31 | 40 | 6 / 2 / 18 | 0.850 | 17 / 0 |  |
| NR1F-L18-s0 | held | 460 | 39 | 0.8800 | 5 | 19 | 28 | 4 / 1 / 29 | 0.832 | 7 / 0 | vs RSG-L18-s0 +0.0144 [-0.0021, +0.0415] (zeros 39 / 34) |
| NR1F-L18-s0 | pub | 1491 | 57 | 0.9429 | 2 | 31 | 45 | 1 / 1 / 14 | 0.870 | 17 / 1 | vs RSG-L18-s0 +0.0032 [-0.0052, +0.0113] (zeros 57 / 58) |
| NR1F-L18-s1 | held | 460 | 41 | 0.8785 | 5 | 20 | 28 | 5 / 0 / 27 | 0.836 | 8 / 0 | vs RSG-L18-s0 +0.0130 [-0.0075, +0.0404] (zeros 41 / 34) |
| NR1F-L18-s1 | pub | 1491 | 57 | 0.9436 | 1 | 31 | 45 | 1 / 0 / 16 | 0.872 | 18 / 2 | vs RSG-L18-s0 +0.0039 [-0.0044, +0.0127] (zeros 57 / 58) |
| NR1HV-L18-s0 | held | 460 | 37 | 0.8802 | 5 | 16 | 25 | 4 / 1 / 31 | 0.831 | 9 / 0 | vs RSG-L18-s0 +0.0146 [+0.0006, +0.0408] (zeros 37 / 34) |
| NR1HV-L18-s1 | held | 460 | 37 | 0.8849 | 6 | 14 | 24 | 5 / 1 / 25 | 0.835 | 9 / 0 | vs RSG-L18-s0 +0.0194 [+0.0057, +0.0374] (zeros 37 / 34) |

| tag | static scene | W | G |
|---|---|--:|--:|
| NR1HV2-L18-s0 | 1.14 m, no contact | 0.03916666666666667 | 1.0 |
| NR1HV2-L18-s1 | 0.79 m, no contact | 0.037083333333333336 | 1.0 |
| RSG-L18-s0 | - | 0.005 | 1.0 |
| NR1F-L18-s0 | 0.85 m, no contact | 0.07916666666666666 | 0.9979123173277662 |
| NR1F-L18-s1 | 1.08 m, no contact | 0.06791666666666667 | 0.9979123173277662 |
| NR1HV-L18-s0 | 1.33 m, no contact | 0.04541666666666667 | 1.0 |
| NR1HV-L18-s1 | 2.38 m, contact | 0.03875 | 1.0 |

gate (supplement 7: both seeds held zeros <= 29, paired CI lower bound vs RSG > 0, static scene no contact):
- NR1HV2-L18-s0: {'zeros_le_29': False, 'ci_lo_gt0': False, 'static_no_contact': True}
- NR1HV2-L18-s1: {'zeros_le_29': False, 'ci_lo_gt0': False, 'static_no_contact': True}
gate passed: False
