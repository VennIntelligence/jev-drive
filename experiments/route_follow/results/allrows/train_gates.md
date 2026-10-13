# Training gates (dev rows; runs/route_mem/rely/<tag>.json, rm_train logs)

| tag | dev ADE final (RSG-L18 line: 0.528 / 0.527 + 0.01) | G-rely ΔADE junction | junction − straight [95 % CI] | G-rely |
|:--|--:|--:|:--|:--|
| RSGall-L18-s0 | 0.5283 | +0.0250 | +0.0260 [+0.0130, +0.0386] | pass |
| RSGall-L18-s1 | 0.5285 | +0.0255 | +0.0267 [+0.0132, +0.0397] | pass |
| RSGallV-L18-s0 | 0.5379 (at the line) | +0.0211 | +0.0220 [+0.0089, +0.0346] | pass |
| RSGallV-L18-s1 | 0.5395 (over by 0.0025) | +0.0255 | +0.0265 [+0.0146, +0.0386] | pass |

dev ADE trajectory of the speed-drop arm: 0.5276 (step 0) -> 0.555 (500) -> 0.547 -> 0.542 / 0.547 -> 0.539 / 0.541 -> 0.538 / 0.540 (2500);
without the drop 0.537 / 0.540 at 500, 0.528 / 0.529 at 2500.
