# Held turns: entry state per label (rf_appheld.py)

Decisions before the first zero; lateral + toward the turn's inside; q25 / median / q75.

| label | turns | completed | v_entry m/s | lat_entry m | lat at entry - 20 m | lat at decision 0 | entry - station at decision 0 (m) |
|:--|--:|--:|:--|:--|:--|:--|:--|
| RSG-L18-s0-XP-L12-E46@18 | 24 | 22 | 3.37 / 4.27 / 4.74 (n 24) | 0.05 / 0.26 / 0.42 (n 23) | nan / nan / nan | -0.13 / 0.05 / 0.29 | 3.75 / 5.00 / 6.96 |
| RSG-L18-s0@18 | 24 | 16 | 3.47 / 4.30 / 4.75 (n 24) | 0.15 / 0.34 / 0.77 (n 23) | nan / nan / nan | -0.13 / 0.05 / 0.29 | 3.75 / 5.00 / 6.96 |
| RSG-L18-s1@18 | 24 | 14 | 3.38 / 4.25 / 4.70 (n 24) | 0.07 / 0.34 / 0.80 (n 23) | nan / nan / nan | -0.13 / 0.05 / 0.29 | 3.75 / 5.00 / 6.96 |

| pair | read | n | mean A - B [95 % CI] | median diff |
|:--|:--|--:|:--|--:|
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s0@18 | v_entry | 24 | -0.01 [-0.21, +0.17] | -0.00 |
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s0@18 | lat_entry | 23 | -0.13 [-0.28, -0.00] | -0.14 |
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s0@18 | lat_m20 | 0 | - | - |
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s0@18 | lat_k0 | 24 | +0.00 [+0.00, +0.00] | +0.00 |
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s1@18 | v_entry | 24 | +0.03 [-0.17, +0.21] | +0.04 |
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s1@18 | lat_entry | 23 | -0.11 [-0.25, +0.01] | -0.03 |
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s1@18 | lat_m20 | 0 | - | - |
| RSG-L18-s0-XP-L12-E46@18:RSG-L18-s1@18 | lat_k0 | 24 | +0.00 [+0.00, +0.00] | +0.00 |
