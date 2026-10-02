# Q-light sweep: ego traffic light state at junction approaches, model x prompt

**Exploratory**: every prompt is read on the same 233 frames (21 routes, 41 segments; ego red or yellow 85, ego green 68, no light 80), all variants are listed, and no result here unlocks a closed-loop arm (plan D16). Cells: estimate [95% route-cluster CI] hits/frames; raw counts alone where fewer than 3 routes carry the cell. Yellow counts as red; a reply without a parseable answer is no answer and stays in the denominator.

| model | prompt | ego red: answered red | ego red: answered green | ego green: answered green | no light: answered red | unparsed | latency p50 ms |
|:--|:--|:--|:--|:--|:--|--:|:--|
| openjev-logged | as run (4 questions, in the drive) | 74% [59, 87] 63/85 | 15% [6, 27] 13/85 | 91% [84, 98] 62/68 | 0% [0, 0] 0/80 | 0 |  |
| openjev | four | 76% [61, 89] 65/85 | 11% [3, 21] 9/85 | 78% [64, 90] 53/68 | 0% [0, 0] 0/80 | 0 | 256 (quiet box) |
| openjev | four_road | 47% [31, 62] 40/85 | 8% [0, 22] 7/85 | 66% [38, 88] 45/68 | 0% [0, 0] 0/80 | 0 | 98 (quiet box) |
| openjev | four_crop | 84% [69, 94] 71/85 | 6% [1, 12] 5/85 | 88% [76, 98] 60/68 | 0% [0, 0] 0/80 | 0 | 193 (quiet box) |
| openjev | position | 68% [55, 79] 58/85 | 2% [0, 6] 2/85 | 82% [65, 95] 56/68 | 0% [0, 0] 0/80 | 0 | 126 (quiet box) |
| dgemma-chat | four | 76% [62, 89] 65/85 | 13% [2, 26] 11/85 | 74% [63, 85] 50/68 | 1% [0, 4] 1/80 | 5 | 514 (quiet box) |
| dgemma-chat | plain | 89% [78, 98] 76/85 | 8% [0, 21] 7/85 | 75% [57, 90] 51/68 | 0% [0, 0] 0/80 | 1 | 350 (quiet box) |
| dgemma-chat | describe | 91% [82, 97] 77/85 | 1% [0, 3] 1/85 | 40% [17, 64] 27/68 | 10% [0, 31] 8/80 | 0 | 407 (quiet box) |
| dgemma-chat | convention | 85% [79, 91] 72/85 | 2% [0, 6] 2/85 | 79% [70, 89] 54/68 | 14% [2, 31] 11/80 | 16 | 419 (quiet box) |
| dgemma-chat | position | 92% [85, 98] 78/85 | 4% [0, 7] 3/85 | 69% [46, 89] 47/68 | 21% [6, 44] 17/80 | 3 | 394 (quiet box) |
| dgemma-chat | plain_road | 58% [42, 72] 49/85 | 4% [0, 13] 3/85 | 69% [43, 90] 47/68 | 0% [0, 0] 0/80 | 0 | 272 (quiet box) |
| dgemma-chat | plain_wide | 62% [41, 81] 53/85 | 9% [3, 17] 8/85 | 60% [42, 76] 41/68 | 0% [0, 0] 0/80 | 0 | 272 (quiet box) |
| dgemma-chat | plain_crop | 78% [68, 87] 66/85 | 6% [0, 13] 5/85 | 75% [61, 88] 51/68 | 2% [0, 9] 2/80 | 18 | 487 (quiet box) |
| qwen3-vl-4b | four | 93% [88, 99] 79/85 | 5% [0, 10] 4/85 | 97% [91, 100] 66/68 | 0% [0, 0] 0/80 | 0 | 869 (quiet box) |
| qwen3-vl-4b | plain | 85% [70, 97] 72/85 | 13% [1, 29] 11/85 | 97% [93, 100] 66/68 | 0% [0, 0] 0/80 | 0 | 802 (quiet box) |
| qwen3-vl-4b | describe | 94% [89, 100] 80/85 | 4% [0, 8] 3/85 | 99% [95, 100] 67/68 | 0% [0, 0] 0/80 | 0 | 1062 (quiet box) |
| qwen3-vl-4b | convention | 87% [72, 97] 74/85 | 13% [3, 28] 11/85 | 99% [95, 100] 67/68 | 4% [0, 10] 3/80 | 0 | 816 (quiet box) |
| qwen3-vl-4b | position | 93% [88, 99] 79/85 | 5% [0, 10] 4/85 | 82% [67, 95] 56/68 | 0% [0, 0] 0/80 | 0 | 803 (quiet box) |
| qwen3-vl-4b | plain_road | 58% [42, 72] 49/85 | 9% [1, 22] 8/85 | 76% [50, 96] 52/68 | 1% [0, 4] 1/80 | 0 | 432 (quiet box) |
| qwen3-vl-4b | plain_wide | 58% [38, 76] 49/85 | 32% [14, 51] 27/85 | 63% [46, 80] 43/68 | 0% [0, 0] 0/80 | 0 | 430 (quiet box) |
| qwen3-vl-4b | plain_crop | 74% [52, 94] 63/85 | 24% [5, 45] 20/85 | 93% [86, 98] 63/68 | 0% [0, 0] 0/80 | 0 | 1135 (quiet box) |
| cosmos-reason1-7b | four | 86% [75, 97] 73/85 | 7% [0, 18] 6/85 | 97% [92, 100] 66/68 | 0% [0, 0] 0/80 | 0 | 1536 (quiet box) |
| cosmos-reason1-7b | plain | 88% [75, 99] 75/85 | 11% [1, 24] 9/85 | 100% [100, 100] 68/68 | 40% [6, 67] 32/80 | 0 | 1467 (quiet box) |
| cosmos-reason1-7b | describe | 27% [16, 38] 23/85 | 8% [3, 14] 7/85 | 34% [21, 46] 23/68 | 12% [4, 24] 10/80 | 151 | 2148 (quiet box) |
| cosmos-reason1-7b | convention | 88% [75, 99] 75/85 | 11% [1, 24] 9/85 | 100% [100, 100] 68/68 | 28% [0, 55] 22/80 | 1 | 1458 (quiet box) |
| cosmos-reason1-7b | position | 72% [55, 89] 61/85 | 12% [1, 27] 10/85 | 72% [43, 95] 49/68 | 0% [0, 0] 0/80 | 88 | 1486 (quiet box) |
| cosmos-reason1-7b | plain_road | 68% [52, 83] 58/85 | 16% [4, 32] 14/85 | 79% [50, 100] 54/68 | 38% [8, 60] 30/80 | 0 | 732 (quiet box) |
| cosmos-reason1-7b | plain_wide | 51% [31, 70] 43/85 | 47% [27, 68] 40/85 | 79% [57, 96] 54/68 | 32% [7, 53] 26/80 | 0 | 744 (quiet box) |
| cosmos-reason1-7b | plain_crop | 73% [50, 94] 62/85 | 27% [6, 50] 23/85 | 99% [95, 100] 67/68 | 39% [5, 64] 31/80 | 0 | 2154 (quiet box) |
| qwen-drive-1.0-4b | four | 79% [57, 95] 67/85 | 11% [0, 29] 9/85 | 81% [61, 95] 55/68 | 0% [0, 0] 0/80 | 17 | 2257 (quiet box) |
| qwen-drive-1.0-4b | plain | 89% [77, 100] 76/85 | 8% [0, 22] 7/85 | 93% [82, 100] 63/68 | 0% [0, 0] 0/80 | 0 | 1158 (quiet box) |
| qwen-drive-1.0-4b | describe | 89% [76, 100] 76/85 | 7% [0, 18] 6/85 | 94% [82, 100] 64/68 | 5% [0, 13] 4/80 | 5 | 3230 (quiet box) |
| qwen-drive-1.0-4b | convention | 56% [38, 71] 48/85 | 0% [0, 0] 0/85 | 44% [28, 61] 30/68 | 0% [0, 0] 0/80 | 70 | 2563 (quiet box) |
| qwen-drive-1.0-4b | position | 75% [61, 89] 64/85 | 12% [0, 26] 10/85 | 65% [39, 86] 44/68 | 0% [0, 0] 0/80 | 32 | 1711 (quiet box) |
| qwen-drive-1.0-4b | plain_road | 66% [53, 80] 56/85 | 0% [0, 0] 0/85 | 72% [45, 95] 49/68 | 0% [0, 0] 0/80 | 1 | 779 (quiet box) |
| qwen-drive-1.0-4b | plain_wide | 66% [44, 86] 56/85 | 16% [2, 39] 14/85 | 82% [64, 97] 56/68 | 0% [0, 0] 0/80 | 0 | 785 (quiet box) |
| qwen-drive-1.0-4b | plain_crop | 80% [61, 96] 68/85 | 8% [1, 18] 7/85 | 91% [79, 100] 62/68 | 0% [0, 0] 0/80 | 14 | 1661 (quiet box) |
| internvl2-1b | four | 46% [29, 61] 39/85 | 54% [39, 71] 46/85 | 100% [100, 100] 68/68 | 19% [2, 46] 15/80 | 0 | 462 (quiet box) |
| internvl2-1b | plain | 0% [0, 0] 0/85 | 100% [100, 100] 85/85 | 100% [100, 100] 68/68 | 0% [0, 0] 0/80 | 0 | 415 (quiet box) |
| internvl2-1b | describe | 46% [31, 62] 39/85 | 46% [31, 61] 39/85 | 88% [74, 100] 60/68 | 57% [40, 78] 46/80 | 13 | 464 (quiet box) |
| internvl2-1b | convention | 1% [0, 4] 1/85 | 99% [96, 100] 84/85 | 100% [100, 100] 68/68 | 0% [0, 0] 0/80 | 0 | 420 (quiet box) |
| internvl2-1b | position | 4% [0, 8] 3/85 | 96% [92, 100] 82/85 | 100% [100, 100] 68/68 | 0% [0, 0] 0/80 | 0 | 423 (quiet box) |
| internvl2-1b | plain_road | 0% [0, 0] 0/85 | 100% [100, 100] 85/85 | 100% [100, 100] 68/68 | 0% [0, 0] 0/80 | 0 | 191 (quiet box) |
| internvl2-1b | plain_wide | 0% [0, 0] 0/85 | 100% [100, 100] 85/85 | 100% [100, 100] 68/68 | 0% [0, 0] 0/80 | 0 | 191 (quiet box) |
| internvl2-1b | plain_crop | 0% [0, 0] 0/85 | 100% [100, 100] 85/85 | 100% [100, 100] 68/68 | 0% [0, 0] 0/80 | 0 | 571 (quiet box) |

## Ego red by what the other approaches show, and route 27043

Cells: answered red / answered green.

| model | prompt | another approach green | no other light green | route 27043 | without 27043 |
|:--|:--|:--|:--|:--|:--|
| openjev-logged | as run (4 questions, in the drive) | 71% [52, 89] 40/56 / 18% [4, 32] 10/56 | 79% [65, 95] 23/29 / 10% [0, 25] 3/29 | 3/5 / 2/5 | 75% [59, 89] 60/80 / 14% [4, 26] 11/80 |
| openjev | four | 73% [53, 89] 41/56 / 12% [2, 25] 7/56 | 83% [67, 100] 24/29 / 7% [0, 20] 2/29 | 2/5 / 3/5 | 79% [63, 91] 63/80 / 8% [2, 15] 6/80 |
| openjev | four_road | 64% [41, 89] 36/56 / 11% [0, 29] 6/56 | 14% [0, 30] 4/29 / 3% [0, 14] 1/29 | 2/5 / 2/5 | 48% [30, 63] 38/80 / 6% [0, 19] 5/80 |
| openjev | four_crop | 80% [59, 94] 45/56 / 9% [2, 18] 5/56 | 90% [73, 100] 26/29 / 0% [0, 0] 0/29 | 3/5 / 2/5 | 85% [69, 96] 68/80 / 4% [0, 8] 3/80 |
| openjev | position | 75% [55, 92] 42/56 / 2% [0, 7] 1/56 | 55% [39, 73] 16/29 / 3% [0, 14] 1/29 | 4/5 / 0/5 | 68% [53, 79] 54/80 / 2% [0, 7] 2/80 |
| dgemma-chat | four | 71% [52, 87] 40/56 / 20% [4, 37] 11/56 | 86% [70, 100] 25/29 / 0% [0, 0] 0/29 | 4/5 / 0/5 | 76% [61, 89] 61/80 / 14% [3, 29] 11/80 |
| dgemma-chat | plain | 88% [71, 100] 49/56 / 12% [0, 29] 7/56 | 93% [82, 100] 27/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 89% [77, 98] 71/80 / 9% [0, 23] 7/80 |
| dgemma-chat | describe | 89% [76, 98] 50/56 / 2% [0, 5] 1/56 | 93% [82, 100] 27/29 / 0% [0, 0] 0/29 | 4/5 / 0/5 | 91% [82, 99] 73/80 / 1% [0, 4] 1/80 |
| dgemma-chat | convention | 82% [71, 91] 46/56 / 4% [0, 8] 2/56 | 90% [79, 100] 26/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 84% [78, 91] 67/80 / 2% [0, 6] 2/80 |
| dgemma-chat | position | 93% [86, 98] 52/56 / 5% [0, 12] 3/56 | 90% [73, 100] 26/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 91% [85, 98] 73/80 / 4% [0, 8] 3/80 |
| dgemma-chat | plain_road | 77% [58, 95] 43/56 / 5% [0, 18] 3/56 | 21% [0, 41] 6/29 / 0% [0, 0] 0/29 | 4/5 / 0/5 | 56% [40, 71] 45/80 / 4% [0, 13] 3/80 |
| dgemma-chat | plain_wide | 54% [23, 77] 30/56 / 14% [6, 27] 8/56 | 79% [59, 100] 23/29 / 0% [0, 0] 0/29 | 1/5 / 2/5 | 65% [43, 83] 52/80 / 8% [2, 14] 6/80 |
| dgemma-chat | plain_crop | 77% [64, 89] 43/56 / 9% [0, 19] 5/56 | 79% [70, 90] 23/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 76% [66, 85] 61/80 / 6% [0, 14] 5/80 |
| qwen3-vl-4b | four | 93% [86, 100] 52/56 / 7% [0, 14] 4/56 | 93% [82, 100] 27/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 92% [87, 98] 74/80 / 5% [0, 11] 4/80 |
| qwen3-vl-4b | plain | 80% [60, 98] 45/56 / 20% [2, 40] 11/56 | 93% [82, 100] 27/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 84% [68, 97] 67/80 / 14% [1, 31] 11/80 |
| qwen3-vl-4b | describe | 95% [89, 100] 53/56 / 5% [0, 11] 3/56 | 93% [82, 100] 27/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 94% [88, 100] 75/80 / 4% [0, 9] 3/80 |
| qwen3-vl-4b | convention | 80% [60, 96] 45/56 / 20% [4, 40] 11/56 | 100% [100, 100] 29/29 / 0% [0, 0] 0/29 | 4/5 / 1/5 | 88% [71, 99] 70/80 / 12% [1, 29] 10/80 |
| qwen3-vl-4b | position | 93% [86, 100] 52/56 / 7% [0, 14] 4/56 | 93% [82, 100] 27/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 92% [87, 98] 74/80 / 5% [0, 11] 4/80 |
| qwen3-vl-4b | plain_road | 75% [56, 94] 42/56 / 12% [0, 29] 7/56 | 24% [0, 50] 7/29 / 3% [0, 11] 1/29 | 3/5 / 2/5 | 57% [41, 73] 46/80 / 8% [0, 20] 6/80 |
| qwen3-vl-4b | plain_wide | 41% [20, 68] 23/56 / 46% [21, 68] 26/56 | 90% [78, 100] 26/29 / 3% [0, 14] 1/29 | 4/5 / 1/5 | 56% [34, 76] 45/80 / 32% [13, 52] 26/80 |
| qwen3-vl-4b | plain_crop | 61% [34, 91] 34/56 / 36% [9, 61] 20/56 | 100% [100, 100] 29/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 72% [48, 94] 58/80 / 25% [6, 48] 20/80 |
| cosmos-reason1-7b | four | 88% [73, 100] 49/56 / 11% [0, 25] 6/56 | 83% [70, 100] 24/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 85% [74, 96] 68/80 / 8% [0, 20] 6/80 |
| cosmos-reason1-7b | plain | 86% [69, 100] 48/56 / 14% [0, 31] 8/56 | 93% [82, 100] 27/29 / 3% [0, 9] 1/29 | 5/5 / 0/5 | 88% [74, 98] 70/80 / 11% [1, 25] 9/80 |
| cosmos-reason1-7b | describe | 25% [12, 37] 14/56 / 12% [4, 21] 7/56 | 31% [11, 48] 9/29 / 0% [0, 0] 0/29 | 0/5 / 0/5 | 29% [18, 39] 23/80 / 9% [3, 15] 7/80 |
| cosmos-reason1-7b | convention | 86% [69, 100] 48/56 / 14% [0, 31] 8/56 | 93% [82, 100] 27/29 / 3% [0, 9] 1/29 | 5/5 / 0/5 | 88% [74, 98] 70/80 / 11% [1, 25] 9/80 |
| cosmos-reason1-7b | position | 71% [48, 94] 40/56 / 18% [2, 37] 10/56 | 72% [43, 100] 21/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 70% [53, 87] 56/80 / 12% [1, 29] 10/80 |
| cosmos-reason1-7b | plain_road | 86% [69, 100] 48/56 / 7% [0, 18] 4/56 | 34% [0, 67] 10/29 / 34% [7, 67] 10/29 | 5/5 / 0/5 | 66% [50, 82] 53/80 / 18% [5, 34] 14/80 |
| cosmos-reason1-7b | plain_wide | 32% [11, 57] 18/56 / 68% [42, 89] 38/56 | 86% [69, 100] 25/29 / 7% [0, 17] 2/29 | 5/5 / 0/5 | 48% [26, 67] 38/80 / 50% [29, 74] 40/80 |
| cosmos-reason1-7b | plain_crop | 59% [31, 90] 33/56 / 41% [10, 69] 23/56 | 100% [100, 100] 29/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 71% [47, 93] 57/80 / 29% [7, 53] 23/80 |
| qwen-drive-1.0-4b | four | 75% [48, 98] 42/56 / 16% [0, 39] 9/56 | 86% [72, 100] 25/29 / 0% [0, 0] 0/29 | 4/5 / 0/5 | 79% [56, 96] 63/80 / 11% [0, 30] 9/80 |
| qwen-drive-1.0-4b | plain | 88% [70, 100] 49/56 / 12% [0, 30] 7/56 | 93% [82, 100] 27/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 89% [75, 100] 71/80 / 9% [0, 23] 7/80 |
| qwen-drive-1.0-4b | describe | 84% [69, 100] 47/56 / 11% [0, 25] 6/56 | 100% [100, 100] 29/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 89% [75, 100] 71/80 / 8% [0, 19] 6/80 |
| qwen-drive-1.0-4b | convention | 62% [43, 80] 35/56 / 0% [0, 0] 0/56 | 45% [20, 69] 13/29 / 0% [0, 0] 0/29 | 3/5 / 0/5 | 56% [37, 72] 45/80 / 0% [0, 0] 0/80 |
| qwen-drive-1.0-4b | position | 75% [58, 92] 42/56 / 18% [0, 36] 10/56 | 76% [56, 100] 22/29 / 0% [0, 0] 0/29 | 4/5 / 0/5 | 75% [59, 90] 60/80 / 12% [0, 29] 10/80 |
| qwen-drive-1.0-4b | plain_road | 84% [67, 98] 47/56 / 0% [0, 0] 0/56 | 31% [6, 57] 9/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 64% [51, 78] 51/80 / 0% [0, 0] 0/80 |
| qwen-drive-1.0-4b | plain_wide | 54% [29, 82] 30/56 / 25% [4, 52] 14/56 | 90% [73, 100] 26/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 64% [40, 84] 51/80 / 18% [2, 41] 14/80 |
| qwen-drive-1.0-4b | plain_crop | 75% [51, 98] 42/56 / 12% [2, 24] 7/56 | 90% [79, 100] 26/29 / 0% [0, 0] 0/29 | 5/5 / 0/5 | 79% [59, 96] 63/80 / 9% [1, 19] 7/80 |
| internvl2-1b | four | 45% [25, 64] 25/56 / 55% [36, 75] 31/56 | 48% [16, 77] 14/29 / 52% [23, 84] 15/29 | 0/5 / 5/5 | 49% [32, 64] 39/80 / 51% [36, 68] 41/80 |
| internvl2-1b | plain | 0% [0, 0] 0/56 / 100% [100, 100] 56/56 | 0% [0, 0] 0/29 / 100% [100, 100] 29/29 | 0/5 / 5/5 | 0% [0, 0] 0/80 / 100% [100, 100] 80/80 |
| internvl2-1b | describe | 46% [27, 67] 26/56 / 50% [32, 67] 28/56 | 45% [17, 68] 13/29 / 38% [17, 57] 11/29 | 3/5 / 2/5 | 45% [29, 62] 36/80 / 46% [31, 62] 37/80 |
| internvl2-1b | convention | 2% [0, 6] 1/56 / 98% [94, 100] 55/56 | 0% [0, 0] 0/29 / 100% [100, 100] 29/29 | 0/5 / 5/5 | 1% [0, 4] 1/80 / 99% [96, 100] 79/80 |
| internvl2-1b | position | 2% [0, 6] 1/56 / 98% [94, 100] 55/56 | 7% [0, 23] 2/29 / 93% [77, 100] 27/29 | 0/5 / 5/5 | 4% [0, 8] 3/80 / 96% [92, 100] 77/80 |
| internvl2-1b | plain_road | 0% [0, 0] 0/56 / 100% [100, 100] 56/56 | 0% [0, 0] 0/29 / 100% [100, 100] 29/29 | 0/5 / 5/5 | 0% [0, 0] 0/80 / 100% [100, 100] 80/80 |
| internvl2-1b | plain_wide | 0% [0, 0] 0/56 / 100% [100, 100] 56/56 | 0% [0, 0] 0/29 / 100% [100, 100] 29/29 | 0/5 / 5/5 | 0% [0, 0] 0/80 / 100% [100, 100] 80/80 |
| internvl2-1b | plain_crop | 0% [0, 0] 0/56 / 100% [100, 100] 56/56 | 0% [0, 0] 0/29 / 100% [100, 100] 29/29 | 0/5 / 5/5 | 0% [0, 0] 0/80 / 100% [100, 100] 80/80 |

## Guard against a lucky prompt: chosen on one half of the routes, read on the other

Routes sorted by id, alternating halves A / B. Score = red recall - red answered green + green recall - no-light false alarm.

| model | chosen on | prompt | read on | ego red: answered red | ego red: answered green | ego green: answered green | no light: answered red |
|:--|:--|:--|:--|:--|:--|:--|:--|
| openjev | A | four_crop | B | 72% [30, 97] 21/29 | 10% [0, 28] 3/29 | 86% [60, 100] 24/28 | 0% [0, 0] 0/50 |
| openjev | B | four_crop | A | 89% [82, 98] 50/56 | 4% [0, 9] 2/56 | 90% [78, 100] 36/40 | 0% [0, 0] 0/30 |
| dgemma-chat | A | plain | B | 86% [52, 100] 25/29 | 14% [0, 48] 4/29 | 79% [55, 92] 22/28 | 0% [0, 0] 0/50 |
| dgemma-chat | B | convention | A | 82% [76, 92] 46/56 | 2% [0, 5] 1/56 | 75% [65, 88] 30/40 | 10% [0, 30] 3/30 |
| qwen3-vl-4b | A | describe | B | 100% [100, 100] 29/29 | 0% [0, 0] 0/29 | 96% [88, 100] 27/28 | 0% [0, 0] 0/50 |
| qwen3-vl-4b | B | describe | A | 91% [85, 100] 51/56 | 5% [0, 12] 3/56 | 100% [100, 100] 40/40 | 0% [0, 0] 0/30 |
| cosmos-reason1-7b | A | four | B | 93% [73, 100] 27/29 | 0% [0, 0] 0/29 | 96% [88, 100] 27/28 | 0% [0, 0] 0/50 |
| cosmos-reason1-7b | B | four | A | 82% [69, 96] 46/56 | 11% [0, 28] 6/56 | 98% [90, 100] 39/40 | 0% [0, 0] 0/30 |
| qwen-drive-1.0-4b | A | describe | B | 90% [64, 100] 26/29 | 0% [0, 0] 0/29 | 86% [60, 100] 24/28 | 8% [0, 20] 4/50 |
| qwen-drive-1.0-4b | B | plain | A | 84% [66, 100] 47/56 | 12% [0, 33] 7/56 | 98% [93, 100] 39/40 | 0% [0, 0] 0/30 |
| internvl2-1b | A | four | B | 66% [33, 89] 19/29 | 34% [11, 67] 10/29 | 100% [100, 100] 28/28 | 10% [0, 28] 5/50 |
| internvl2-1b | B | four | A | 36% [16, 51] 20/56 | 64% [49, 84] 36/56 | 100% [100, 100] 40/40 | 33% [0, 90] 10/30 |

## Route 27043, ego red within 50 m: what is lit in the frame, by the logged answer

Lamps = blobs of very bright saturated pixels in the upper 55% of the frame, named by hue (red < 20 or >= 330 deg, amber < 70, green < 190). A heuristic: it also picks up tail lights and signs and does not know which head serves which approach.

| logged answer   |   frames | truth: another approach green   | wide: frames with a green lamp   | wide: frames with a red or amber lamp   | wide: mean lamps red / amber / green   | road: frames with a green lamp   | road: frames with a red or amber lamp   | road: mean lamps red / amber / green   |
|:----------------|---------:|:--------------------------------|:---------------------------------|:----------------------------------------|:---------------------------------------|:---------------------------------|:----------------------------------------|:---------------------------------------|
| green           |       22 | 22/22                           | 0/22                             | 22/22                                   | 0.4 / 1.5 / 0.0                        | 0/22                             | 22/22                                   | 0.5 / 7.3 / 0.0                        |
| red             |        4 | 4/4                             | 0/4                              | 4/4                                     | 0.0 / 2.8 / 0.0                        | 0/4                              | 4/4                                     | 0.5 / 6.2 / 0.0                        |

Detector check on 60 ego-green frames within 50 m of the other routes: a green lamp is found in the wide frame of 14, in the road frame of 35.

## Junction structure on these routes (truth labels, every answered request with an ego light)

Rank 1 = the ego light is the nearest light actor; a higher rank = it is mounted beyond other approaches' lights, on the far side.

|   route |   ego_light |   frames |   lights within 60 m (median) |   green at once (median) |   green at once (max) |   ego light distance rank (median) |   ego light is the nearest (share) |
|--------:|------------:|---------:|------------------------------:|-------------------------:|----------------------:|-----------------------------------:|-----------------------------------:|
|   15102 |        1656 |       19 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   15483 |        1462 |       68 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   15612 |         881 |       52 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   16390 |        1285 |       36 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   16508 |        4348 |       18 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   16508 |        4416 |       18 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   16529 |        4848 |       19 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   16529 |        4849 |       19 |                             4 |                        1 |                     1 |                                  3 |                               0    |
|   24944 |          25 |       71 |                             9 |                        2 |                     4 |                                  4 |                               0.14 |
|   24944 |          39 |      108 |                             1 |                        0 |                     2 |                                  1 |                               0.81 |
|   27043 |         104 |       44 |                             4 |                        1 |                     2 |                                  1 |                               1    |
|   27297 |        4316 |       32 |                             4 |                        2 |                     2 |                                  3 |                               0    |
|   27787 |         504 |       20 |                             4 |                        2 |                     2 |                                  3 |                               0    |
|   27870 |         104 |       19 |                             5 |                        2 |                     3 |                                  3 |                               0    |
|   28147 |        5806 |       41 |                             7 |                        2 |                     3 |                                  2 |                               0    |
|     334 |         122 |       38 |                             4 |                        1 |                     2 |                                  4 |                               0    |
|    9196 |        1312 |       56 |                             4 |                        1 |                     1 |                                  3 |                               0    |

## Prompts

`four`:

```
The two images were taken at the same instant by the front cameras of a car (the ego vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera.
Traffic light status controlling the ego vehicle lane ahead. Options:
  no_light: no traffic light controlling ego lane ahead
  red_or_yellow_for_ego: red or yellow traffic light controlling ego lane ahead
  green_for_ego: green traffic light controlling ego lane ahead
  light_for_other_lane: traffic light visible but controlling another lane, not ego lane
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: no_light, red_or_yellow_for_ego, green_for_ego, light_for_other_lane. Reply with that line only.
```

`plain`:

```
The two images were taken at the same instant by the front cameras of a car (the ego vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera.
What colour is the traffic light that applies to you (the ego vehicle) right now: red, yellow, green, or none if no traffic light applies to you?
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: red, yellow, green, none. Reply with that line only.
```

`describe`:

```
The two images were taken at the same instant by the front cameras of a car (the ego vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera.
First list every traffic light head visible, one per line, with its position (left / centre / right of the image, near or far side of the junction) and the colour of its lit lamp. Then pick the one that faces the ego vehicle across the junction it is approaching and give its colour: red, yellow, green, or none if no traffic light applies to the ego vehicle.
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: red, yellow, green, none.
```

`convention`:

```
The two images were taken at the same instant by the front cameras of a car (the ego vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera.
Convention of this road network: the traffic lights of a junction are mounted on its far side, one light per approach. The light that applies to the ego vehicle is the one facing it from across the junction, straight ahead beyond the stop line. The lights of the other approaches are visible too, from the side or at an angle, and may show a different colour: ignore them. What colour is the light that applies to the ego vehicle: red, yellow, green, or none if the ego vehicle is not approaching a traffic light?
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: red, yellow, green, none. Reply with that line only.
```

`position`:

```
The two images were taken at the same instant by the front cameras of a car (the ego vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera.
Find the traffic light that faces the ego vehicle at the junction ahead; ignore lights seen from the side that serve cross traffic. Decide by which lamp of its vertical housing is lit, not by the hue: top lamp lit = red, middle lamp lit = yellow, bottom lamp lit = green. Answer red, yellow, green, or none if no traffic light faces the ego vehicle.
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: red, yellow, green, none. Reply with that line only.
```

`plain_road`:

```
The image was taken by the narrow front camera of a car (the ego vehicle).
What colour is the traffic light that applies to you (the ego vehicle) right now: red, yellow, green, or none if no traffic light applies to you?
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: red, yellow, green, none. Reply with that line only.
```

`plain_wide`:

```
The image was taken by the wide-angle front camera of a car (the ego vehicle).
What colour is the traffic light that applies to you (the ego vehicle) right now: red, yellow, green, or none if no traffic light applies to you?
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: red, yellow, green, none. Reply with that line only.
```

`plain_crop`:

```
The three images were taken at the same instant by the front cameras of a car (the ego vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera, image 3 is an enlarged crop of the upper centre of image 1.
What colour is the traffic light that applies to you (the ego vehicle) right now: red, yellow, green, or none if no traffic light applies to you?
End your reply with one line of the form `ANSWER: <option>`, <option> being one of: red, yellow, green, none. Reply with that line only.
```

`openjev` rows use the System One endpoint with Q_light alone and its own variants: `four` = base, `four_road` = road, `four_crop` = crop, `position` = pos (lib/vlm_protocol.VARIANTS).

Frames: `results/lightsweep/frames.csv`; raw replies: `results/lightsweep/<model>.jsonl`.
