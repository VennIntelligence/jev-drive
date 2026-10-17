# V2AC = V2AC-L18-s0 against V2 = RTB1ARO-L18-s0, RSG = RSG-L18-s0 (stored reads, one seed each)

## Held 460 @18

| arm | scenes | zeros | mean | paired vs V2AC | paired vs V2 | paired vs RSG | at-fault | offroad | progress_rel | front / side / rear | turns /24 | static scene | stop17 zeros | creep | no-launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| V2AC | 460 | 39 | 0.8816 | - | +0.009 [-0.006, +0.035] | +0.016 [-0.007, +0.053] | 4 | 17 | 0.830 | 4 / 0 / 31 | 19 / 24 | 0.84 m, no contact | 16 / 17 | 8 | 0 |
| V2 | 460 | 39 | 0.8729 | -0.009 [-0.035, +0.006] | - | +0.007 [-0.005, +0.025] | 7 | 16 | 0.817 | 6 / 1 / 31 | 16 / 24 | 0.86 m, no contact | 15 / 17 | 8 | 0 |
| RSG | 460 | 34 | 0.8656 | -0.016 [-0.053, +0.007] | -0.007 [-0.025, +0.005] | - | 4 | 17 | 0.788 | 4 / 0 / 38 | 16 / 24 | 2.61 m, contact | 16 / 17 | 7 | 0 |

## Public 1491 @10

| arm | scenes | zeros | mean | paired vs V2AC | paired vs V2 | paired vs RSG | at-fault | offroad | progress_rel | front / side / rear | creep | no-launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| V2AC | 1491 | 63 | 0.9391 | - | +0.001 [-0.004, +0.006] | -0.001 [-0.010, +0.008] | 6 | 33 | 0.868 | 5 / 1 / 15 | 17 | 2 |
| V2 | 1491 | 65 | 0.9378 | -0.001 [-0.006, +0.004] | - | -0.002 [-0.011, +0.007] | 5 | 36 | 0.864 | 3 / 2 / 17 | 17 | 0 |
| RSG | 1491 | 58 | 0.9397 | +0.001 [-0.008, +0.010] | +0.002 [-0.007, +0.011] | - | 8 | 31 | 0.850 | 6 / 2 / 18 | 17 | 0 |

## Lines of prereg plans/2026-10-17-v2ac-prereg.md (V2AC against V2)

| line | V2AC | V2 | threshold | met |
|---|--:|--:|:-:|:-:|
| 1 held zeros | 39 | 39 | <= ref + 2 | yes |
| 2 held progress_rel | 0.830 | 0.817 | > ref | yes |
| 3 public paired mean vs ref, CI lower bound | -0.004 | - | >= -0.005 | yes |
| 4 public rear-ended scenes | 15 | 17 | <= ref | yes |
| 5 static scene distance (m), no contact | 0.842 | 0.865 | <= 1.0 and no contact | yes |

All five lines met: yes.

V2AC: arrival 4 s arc @k=1 1.467 m, dev ADE 0.549.
V2: arrival 4 s arc @k=1 1.494 m, dev ADE 0.545.
RSG: arrival 4 s arc @k=1 2.902 m, dev ADE 0.528.
