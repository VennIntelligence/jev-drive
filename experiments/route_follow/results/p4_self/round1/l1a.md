# L1a: PAI identity completion on the 27 P4 train turns (P2's rule)

Row turns (scenes of the pool's branches): 8; other turns: 19.

| label | scenes read | completed | on row turns | on other turns | zeros | mean score |
|:--|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0@P4 | 27 | 9 | 4 / 8 | 5 / 19 | 17 | 0.3467 |
| RSGCTL-L18-s0 | 27 | 9 | 4 / 8 | 5 / 19 | 17 | 0.3467 |
| RSGSELF-L18-s0 | 27 | 8 | 3 / 8 | 5 / 19 | 18 | 0.3116 |
| RSGSELF-L18-s1 | 27 | 9 | 4 / 8 | 5 / 19 | 18 | 0.2997 |

Per scene (C = completed, z = zero score with its kind, . = neither; * = row turn):

| scene | RSG-L18-s0@P4 | RSGCTL-L18-s0 | RSGSELF-L18-s0 | RSGSELF-L18-s1 |
|:--|--:|--:|--:|--:|
| 0d10d2e3 | z collision | z collision | z collision | z collision |
| 1594e89e | z collision | z collision | z collision | z collision |
| 1e8e939e* | z collision | z collision | z collision | z collision |
| 213dfdac | z offroad | z offroad | z offroad | z corridor |
| 232d240e* | z offroad | z offroad | C | C |
| 268949d2* | C | C | z collision | z collision |
| 3232fdb7 | z corridor | z corridor | z corridor | z corridor |
| 33c8976b | z offroad | z offroad | z offroad | z offroad |
| 37d660b1 | z corridor | z corridor | C | C |
| 403d558a | z corridor | z corridor | z corridor | z corridor |
| 77fdd75c | C | C | C | C |
| 8a365ac2* | z offroad | z offroad | z offroad | z offroad |
| 96da4f1a | C | C | z offroad | z offroad |
| 96eb4700 | C | C | C | C |
| 98694f91 | z corridor | z corridor | z corridor | C |
| a174faed | z corridor | z corridor | z corridor | z corridor |
| a4be61d6* | C | C | C | C |
| a4d2f7cf | . | . | . | z corridor |
| b40f4729 | C | C | C | C |
| c6c01d25* | C | C | C | C |
| d0b9a872 | z offroad | z offroad | z offroad | z offroad |
| dc1966f4 | C | C | C | z offroad |
| ddc3e8df* | z corridor | z corridor | z corridor | C |
| e5e47fd2 | z corridor | z corridor | z corridor | z corridor |
| ea468790* | C | C | z corridor | z corridor |
| eaba3f6a | z corridor | z corridor | z corridor | z corridor |
| f9e361aa | z corridor | z corridor | z corridor | z corridor |
