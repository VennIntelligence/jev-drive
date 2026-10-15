# Held turns: entry state per label (rf_appheld.py)

Decisions before the first zero; lateral + toward the turn's inside; q25 / median / q75.

| label | turns | completed | v_entry m/s | lat_entry m | lat at entry - 20 m | lat at decision 0 | entry - station at decision 0 (m) |
|:--|--:|--:|:--|:--|:--|:--|:--|
| APP1X-L18-s0@18 | 24 | 8 | 3.15 / 4.36 / 4.89 (n 24) | 0.24 / 0.57 / 1.02 (n 23) | nan / nan / nan | -0.13 / 0.05 / 0.29 | 3.75 / 5.00 / 6.96 |
| RSG-L18-s0@18 | 24 | 16 | 3.47 / 4.30 / 4.75 (n 24) | 0.15 / 0.34 / 0.77 (n 23) | nan / nan / nan | -0.13 / 0.05 / 0.29 | 3.75 / 5.00 / 6.96 |

| pair | read | n | mean A - B [95 % CI] | median diff |
|:--|:--|--:|:--|--:|
| APP1X-L18-s0@18:RSG-L18-s0@18 | v_entry | 24 | -0.02 [-0.18, +0.10] | +0.04 |
| APP1X-L18-s0@18:RSG-L18-s0@18 | lat_entry | 23 | +0.17 [+0.09, +0.25] | +0.05 |
| APP1X-L18-s0@18:RSG-L18-s0@18 | lat_m20 | 0 | - | - |
| APP1X-L18-s0@18:RSG-L18-s0@18 | lat_k0 | 24 | +0.00 [+0.00, +0.00] | +0.00 |
