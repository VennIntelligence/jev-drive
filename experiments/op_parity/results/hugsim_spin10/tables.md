## T1 per arm on the 10 spinner scenarios (one run each, PR #57 controller, 400-step cap)

| arm | n | HD-Score | RC | spins >= 60 deg | launch stalls (v_max first 40 steps < 1.6 m/s) | complete | stuck | fg coll | bg coll | off_route | max heading err median (deg) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Cinque rerun (cinque-fixed-base, 2026-10-03) | 10 | 0.315 | 0.416 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 161 |
| P0 shipped, through the parity path | 10 | 0.352 | 0.456 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 154 |
| P1 fine-tuned, no inputs | 10 | 0.330 | 0.438 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 148 |
| P2 + ego / pose / command | 10 | 0.265 | 0.392 | 7 | 0 | 1 | 0 | 1 | 1 | 0 | 98 |
| P3 + side / rear cameras | 10 | 0.321 | 0.484 | 8 | 0 | 1 | 0 | 0 | 1 | 0 | 115 |
| Cinque PR #57 (cinque-fixed, 2026-09-25) | 10 | 0.350 | 0.467 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 150 |
| WA-JEPA | 10 | 0.593 | 0.670 | 1 | 0 | 4 | 0 | 3 | 0 | 2 | 7 |

## T2 per scenario: HD-Score / class / max heading error (deg) / v_max over the first 40 steps (m/s)

| scenario | Cinque rerun (cinque-fixed-base, 2026-10-03) | P0 shipped, through the parity path | P1 fine-tuned, no inputs | P2 + ego / pose / command | P3 + side / rear cameras | Cinque PR #57 (cinque-fixed, 2026-09-25) | WA-JEPA |
|---|---|---|---|---|---|---|---|
| scene-0013-medium-00 | 0.055 spin 171 4.5 | 0.059 spin 173 4.5 | 0.053 spin 169 4.5 | 0.064 spin 148 5.3 | 0.063 spin 140 5.2 | 0.055 spin 171 4.5 | 0.870 complete 3 8.6 |
| scene-0528-medium-00 | 0.045 spin 113 4.0 | 0.043 spin 128 3.9 | 0.042 spin 114 4.1 | 0.050 spin 118 4.5 | 0.033 spin 120 4.3 | 0.045 spin 114 3.9 | 0.100 fg_coll 5 8.5 |
| scene-0254-extreme-00 | 0.818 spin 178 7.5 | 0.835 spin 177 6.8 | 0.796 spin 177 4.1 | 0.069 fg_coll 12 6.4 | 0.759 spin 176 7.9 | 0.821 spin 178 7.4 | 0.062 fg_coll 13 2.5 |
| scene-102751446607-medium-01 | 0.141 spin 177 3.9 | 0.148 spin 178 3.7 | 0.154 spin 177 3.9 | 0.054 spin 180 3.4 | 0.050 spin 179 3.5 | 0.150 spin 180 3.7 | 0.996 complete 9 3.4 |
| scene-152217047339-medium-00 | 0.083 spin 151 3.0 | 0.083 spin 149 3.1 | 0.079 spin 167 3.2 | 0.143 spin 113 4.6 | 0.120 spin 109 4.5 | 0.083 spin 151 3.0 | 0.251 off_route 56 5.2 |
| scene-570_770-easy-00 | 0.225 bg_coll 15 13.2 | 0.224 bg_coll 17 13.1 | 0.226 bg_coll 17 13.5 | 0.212 bg_coll 22 14.6 | 0.196 bg_coll 23 13.8 | 0.225 spin 63 13.1 | 0.951 complete 4 6.4 |
| scene-5980_6180-easy-00 | 0.304 bg_coll 49 5.4 | 0.322 bg_coll 42 5.4 | 0.276 spin 70 5.6 | 0.785 complete 12 9.4 | 0.776 complete 23 8.8 | 0.325 spin 70 5.4 | 0.302 off_route 53 7.4 |
| scene-8440_8640-easy-00 | 0.098 spin 177 3.3 | 0.563 spin 160 3.3 | 0.545 bg_coll 37 3.6 | 0.487 spin 139 8.2 | 0.509 spin 105 7.9 | 0.557 spin 150 3.3 | 0.760 spin 64 6.4 |
| scene-040-easy-00 | 0.945 spin 89 2.5 | 0.813 spin 98 2.5 | 0.656 spin 130 2.6 | 0.665 spin 83 7.9 | 0.621 spin 110 6.7 | 0.806 spin 130 2.5 | 0.906 complete 3 7.3 |
| scene-053-medium-02 | 0.431 spin 179 6.1 | 0.432 spin 179 6.0 | 0.475 spin 178 6.0 | 0.120 spin 80 6.5 | 0.085 spin 163 4.4 | 0.432 spin 179 6.1 | 0.734 fg_coll 1 5.5 |

## T3 parity path per arm (means over runs)

| arm | parity steps | bias rms | bias server round trip median (ms) |
|---|--:|--:|--:|
| P0 shipped, through the parity path | 1225 | 0.0000 | 27.0 |
| P1 fine-tuned, no inputs | 1066 | 0.0000 | 0.6 |
| P2 + ego / pose / command | 358 | 0.4991 | 4.8 |
| P3 + side / rear cameras | 408 | 0.5168 | 25.9 |
