| task | old steps | old tokens | new steps | new tokens | ratio |
|---|---|---|---|---|---|
| *lookup* | | | | | |
| op-adapt L: which script produced the result, what was the conclusion | 4 | 8,540 | 2 | 405 | 21.1x |
| default CARLA worker profile and its evidence | 2 | 7,815 | 2 | 806 | 9.7x |
| where is the nq3 Q1 table | 4 | 3,537 | 2 | 45 | 78.6x |
| which controller experiments exist, which are superseded | 3 | 5,511 | 1 | 1,322 | 4.2x |
| status and claim of decision 55 | 2 | 1,277 | 1 | 1,208 | 1.1x |
| list every decision still pending (待定) | 1 | 5,467 | 1 | 1,730 | 3.2x |
| which file implements the S_jev rule scorer, what is it | 2 | 312 | 2 | 89 | 3.5x |
| WL-2 verdict, results and figures | 4 | 7,355 | 3 | 429 | 17.1x |
| how to run the HUGSIM zero-shot exam | 3 | 7,426 | 3 | 750 | 9.9x |
| what did the zero-shot Bench2Drive exam conclude | 3 | 8,576 | 2 | 373 | 23.0x |
| which shared module reads WOD-E2E frames | 2 | 859 | 2 | 562 | 1.5x |
| night queue 4 G (ghost test): conclusion and code | 5 | 9,124 | 2 | 396 | 23.0x |
| NAVSIM skill pack N3 score and its run script | 3 | 3,458 | 3 | 663 | 5.2x |
| TFv6 controller campaign: conclusion and report | 3 | 6,349 | 2 | 373 | 17.0x |
| which experiments are live right now | 3 | 5,529 | 1 | 88 | 62.8x |
| **lookup: 15 tasks** | | **81,135** | | **9,239** | **8.8x** |
| *cold start (CLAUDE.md included)* | | | | | |
| what rules apply when I add an experiment | 4 | 10,071 | 2 | 1,134 | 8.9x |
| where do I put a new result and its figure | 2 | 2,958 | 2 | 1,134 | 2.6x |
| which experiments are live | 3 | 7,256 | 2 | 1,003 | 7.2x |
| what is the CARLA default worker profile | 2 | 6,399 | 2 | 942 | 6.8x |
| how do I start a long job on the box | 2 | 3,594 | 2 | 3,138 | 1.1x |
| which shared library gives run dirs and splits | 5 | 8,311 | 2 | 2,751 | 3.0x |
| open the op-adapt L experiment page | 3 | 11,994 | 3 | 1,320 | 9.1x |
| how do I log in to the GPU box | 2 | 2,982 | 2 | 2,004 | 1.5x |
| what are the writing rules for research notes | 2 | 2,958 | 2 | 1,441 | 2.1x |
| which experiments touched NAVSIM | 2 | 3,627 | 2 | 1,066 | 3.4x |
| **cold start (CLAUDE.md included): 10 tasks** | | **60,150** | | **15,933** | **3.8x** |
| **all 25 tasks** | | **141,285** | | **25,172** | **5.6x** |

CLAUDE.md, read every session: 1,898 -> 915 tokens. Directory listings: scripts/ 518 -> 49 entries, jevdrive/ 155 -> 72; research/decisions.md 121,852 -> 1,730 tokens.

| file | old tokens | new tokens |
|---|---|---|
| CLAUDE.md | 1,898 | 915 |
| README.md | 3,314 | 293 |
| docs/README.md | 746 | 525 |
| research/README.md | 1,060 | 526 |
| experiments/INDEX.md | 0 | 1,322 |
| research/decisions.md | 121,852 | 1,730 |
| experiments/TEMPLATE.md | 0 | 218 |
| experiments/*/README.md (40) | - | avg 318, max 377 |

Cold start to any topic README (CLAUDE.md + one INDEX.md grep line <= 39 tok + README): <= 1,331 tok; reading the whole INDEX.md instead: <= 2,614 tok.
