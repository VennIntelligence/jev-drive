# Held turns: entry state per label (rf_appheld.py)

Decisions before the first zero; lateral + toward the turn's inside; q25 / median / q75.

| label | turns | completed | v_entry m/s | lat_entry m | lat at entry - 20 m | lat at decision 0 | entry - station at decision 0 (m) |
|:--|--:|--:|:--|:--|:--|:--|:--|
| APP2W05-L18-s0@18 | 24 | 14 | 3.56 / 4.38 / 4.95 (n 24) | 0.28 / 0.58 / 1.00 (n 23) | nan / nan / nan | -0.13 / 0.05 / 0.29 | 3.75 / 5.00 / 6.96 |
| RSG-L18-s0@18 | 24 | 16 | 3.47 / 4.30 / 4.75 (n 24) | 0.15 / 0.34 / 0.77 (n 23) | nan / nan / nan | -0.13 / 0.05 / 0.29 | 3.75 / 5.00 / 6.96 |

| pair | read | n | mean A - B [95 % CI] | median diff |
|:--|:--|--:|:--|--:|
| APP2W05-L18-s0@18:RSG-L18-s0@18 | v_entry | 24 | +0.10 [+0.04, +0.16] | +0.07 |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | lat_entry | 23 | +0.16 [+0.08, +0.23] | +0.06 |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | lat_m20 | 0 | - | - |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | lat_k0 | 24 | +0.00 [+0.00, +0.00] | +0.00 |
