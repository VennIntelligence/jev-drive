# TV-AUDIT census (658206 eligible trainval frames, 1250 logs, 658206 with 4 history CAM_F0)

Cells: frames / events (runs of consecutive frames) / logs. Rules: stand: v0 < 0.3 and 4 s displacement < 1.0 m; line: first stop line ahead on the hindsight route, front bumper -2.0..10.0 m; red: light line, red at t0, on the route connector; no_vehicle_before_line: lead_any > line distance + 2 m; events: runs of consecutive 2 Hz frames of one log

| class | navtrain tokens | trainval clean train logs, not navtrain, CAM_F0 history |   of which already in selfiter_tv (NR1TV / JT rows) |   of which new | trainval clean hold logs, not navtrain, CAM_F0 history |
|:--|--:|--:|--:|--:|--:|
| all eligible | 102373 / 17883 / 1190 | 493955 / 17254 / 1100 | 82594 / 5415 / 847 | 411361 / 11839 / 932 | 51454 / 1765 / 118 |
| class stopped | 82 / 67 / 49 | 191414 / 3947 / 961 | 34053 / 631 / 147 | 157361 / 3316 / 816 | 18244 / 414 / 108 |
| class starting | 19053 / 4906 / 1064 | 22971 / 6656 / 1017 | 3883 / 1379 / 403 | 19088 / 5277 / 861 | 2445 / 729 / 114 |
| class turning | 24065 / 4892 / 885 | 23600 / 5098 / 839 | 4891 / 1711 / 536 | 18709 / 3387 / 702 | 2561 / 534 / 90 |
| class braking | 31810 / 5029 / 1038 | 55590 / 8644 / 1051 | 8614 / 1771 / 480 | 46976 / 6873 / 890 | 6355 / 985 / 116 |
| class cruising | 27363 / 9337 / 1122 | 200380 / 17728 / 1093 | 31153 / 3807 / 680 | 169227 / 13921 / 926 | 21849 / 1939 / 118 |
| standing (v0 < 0.3, 4 s < 1 m) | 23 / 20 / 16 | 188697 / 3985 / 961 | 33597 / 637 / 147 | 155100 / 3348 / 816 | 17932 / 418 / 108 |
| standing, reason visible (stopped lead <= 15 m | stop sign | red light, line -2..10 m) | 9 / 9 / 7 | 120373 / 2565 / 730 | 21909 / 429 / 107 | 98464 / 2136 / 624 | 13124 / 265 / 86 |
| standing, no visible reason | 14 / 11 / 11 | 68324 / 1938 / 742 | 11688 / 295 / 117 | 56636 / 1643 / 626 | 4808 / 191 / 71 |
| standing, stopped lead <= 15 m (W_L) | 8 / 8 / 6 | 99732 / 2271 / 670 | 19528 / 391 / 99 | 80204 / 1880 / 572 | 10070 / 235 / 82 |
| standing at red line (-2..10 m), any lead | 1 / 1 / 1 | 45233 / 716 / 385 | 5582 / 92 / 49 | 39651 / 624 / 336 | 5025 / 80 / 37 |
| standing at red line, no stopped lead <= 15 m | 1 / 1 / 1 | 20432 / 307 / 193 | 2356 / 39 / 25 | 18076 / 268 / 168 | 3045 / 32 / 16 |
| standing at red line, no vehicle before the line (d372 cue 1) | 1 / 1 / 1 | 20990 / 314 / 196 | 2451 / 46 / 29 | 18539 / 268 / 167 | 3078 / 31 / 16 |
| standing at red line, no vehicle before the line, line 0..5 m | 1 / 1 / 1 | 15925 / 228 / 147 | 1647 / 34 / 22 | 14278 / 194 / 125 | 2465 / 26 / 13 |
| standing at stop sign line | 0 / 0 / 0 | 216 / 24 / 21 | 26 / 2 / 2 | 190 / 22 / 19 | 10 / 2 / 2 |
| launch (G_L | G_N) | 5181 / 2660 / 908 | 5397 / 2631 / 847 | 876 / 438 / 156 | 4521 / 2193 / 713 | 560 / 275 / 92 |
| lead start (G_L: human launches, lead <= 15 m, v < 3; d372 cue 2) | 2225 / 1272 / 558 | 2734 / 1403 / 557 | 459 / 236 / 87 | 2275 / 1167 / 475 | 263 / 142 / 69 |
| lead start, lead stopped at t0 (G_L, lead v < 0.5) | 126 / 94 / 81 | 1152 / 712 / 380 | 199 / 115 / 53 | 953 / 597 / 327 | 112 / 75 / 45 |
| launch, no lead (G_N) | 2956 / 1393 / 642 | 2663 / 1239 / 593 | 417 / 203 / 112 | 2246 / 1036 / 498 | 297 / 133 / 57 |
| launch at a light line (G_* with a light line -2..10 m) | 661 / 395 / 267 | 870 / 443 / 298 | 119 / 61 / 39 | 751 / 382 / 260 | 107 / 52 / 31 |
| stop behind lead (S_L) | 3912 / 1211 / 566 | 773 / 393 / 284 | 219 / 141 / 104 | 554 / 252 / 203 | 95 / 48 / 27 |
| stop at red (d331 S_R) | 971 / 379 / 241 | 2347 / 578 / 323 | 355 / 103 / 74 | 1992 / 475 / 269 | 246 / 56 / 33 |
| go on green (d331 GO_G) | 208 / 87 / 71 | 4524 / 1330 / 441 | 626 / 186 / 68 | 3898 / 1144 / 375 | 508 / 152 / 45 |
| yield pedestrian (Y_P) | 577 / 196 / 138 | 1423 / 455 / 256 | 255 / 93 / 59 | 1168 / 362 / 210 | 131 / 44 / 24 |
| yield vehicle (Y_V) | 203 / 108 / 87 | 1073 / 436 / 267 | 149 / 67 / 47 | 924 / 369 / 229 | 108 / 46 / 31 |
| follow moving lead (F_L) | 10483 / 3154 / 742 | 66340 / 5293 / 837 | 10858 / 1301 / 394 | 55482 / 3992 / 714 | 7751 / 610 / 96 |
| junction | 67775 / 13974 / 1162 | 251957 / 16960 / 1083 | 42294 / 4672 / 799 | 209663 / 12288 / 919 | 27687 / 1780 / 117 |
