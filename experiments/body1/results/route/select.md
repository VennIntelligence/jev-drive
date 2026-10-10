| Validation part, shards s2 + s3 | `P2H10R-Pb15-s0` (B = 1.5 m) | `P2H10R-Pb25-s0` (B = 2.5 m) | `P2H10S-P-s0` | `P2H10-P-s0` |
|:--|--:|--:|--:|--:|
| dev ADE at step 3 000 (m; limit switch-off + 0.01) | 0.6051 | 0.5949 |  | 0.5913 |
| dev_drift_off (limit 0.30) | 0.059 | 0.052 |  |  |
| own-plan agent-contact rate, relative fall (>= 30 %) | 0.0246, 31.2 % | 0.0241, 32.6 % |  |  |
| own-plan boundary rate, relative fall (>= 25 %) | 0.0249, 37.5 % | 0.0254, 36.3 % |  |  |
| route hinge non-zero, imitation rows, steps 2 701-3 000 (<= 5 %) | 3.16 % | 0.68 % |  |  |
| route hinge non-zero, hinge-only rows (<= 20 %) | 9.67 % | 3.44 % |  |  |
| eligible | **no** | yes |  |  |
| W2, turn rows > 45 deg, four families (n = 490) | 49 | 55 | 65 | 66 |
| W2, on-log turn rows (reported) | 10 | 10 | 11 | 10 |
| mean lateral at 4 s, + = outside (m; reported) | +0.369 | +0.442 | +0.435 | +0.262 |

Selected: `P2H10R-Pb25-s0`
