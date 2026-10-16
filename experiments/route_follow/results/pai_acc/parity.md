# Cross-host parity (descriptive): box recovery_envelope/pai/RSG-L18-s0 (native, gains 20, v8 gate, b7bcd3d3) vs Tokyo v2pai/RSG-L18-s0-gate (docker, gains 20, v8 gate, 2026-10-15), same 8 chunk lists

rf_pai281.py --rows box=... tokyo=... --ref box,tokyo --lists all exam control --vs box (Mac CPU, 2026-10-16). Per scene: 162 / 281 identical score, 265 within 0.05, zero agreement 276 / 281 (both 69, box only 3, Tokyo only 2), score correlation 0.961.

Scenes scored by every label: 281; turn scenes (a scoring reference label turned 60-200 deg): 17 (left 11, right 6)

| label | mean | zeros | collision | offroad | corridor | handoff spins | turn done left | turn done right |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| box | 0.5806 | 72 | 12 | 12 | 48 | 0 | 10/11 | 6/6 |
| tokyo | 0.5860 | 71 | 12 | 12 | 47 | 0 | 11/11 | 5/6 |

## all: 281 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs box (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| box | 0.5806 | 72 | 12 | 12 | 48 | 72 | 0 | +nan | +0.0000 [+0.0000, +0.0000] |
| tokyo | 0.5860 | 71 | 12 | 12 | 47 | 72 | 0 | +nan | +0.0054 [-0.0078, +0.0190] |

## exam: 72 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs box (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| box | 0.3815 | 37 | 7 | 7 | 23 | 8 | 0 | +nan | +0.0000 [+0.0000, +0.0000] |
| tokyo | 0.3806 | 37 | 6 | 8 | 23 | 9 | 0 | +nan | -0.0009 [-0.0226, +0.0213] |

## control: 75 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs box (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| box | 0.7850 | 0 | 0 | 0 | 0 | 31 | 0 | +nan | +0.0000 [+0.0000, +0.0000] |
| tokyo | 0.7865 | 0 | 0 | 0 | 0 | 30 | 0 | +nan | +0.0014 [-0.0028, +0.0055] |
