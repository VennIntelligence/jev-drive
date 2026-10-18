# RTJA dose selection (NR1 prereg supplement 12)

Rule: per lead class (other_lead / red_queue) wait-set v2 variant a wait rate >= 0.78 / 0.67 (V2 - 0.05), variant b median launch delay (car minus human, launched windows) within +-0.15 s and launched >= 95%; held 460 @18 zeros <= 37. Among the passing arms the fewest held zeros (within 1: the smaller lambda).

| arm | kd-lam | a wait other_lead / red_queue | b launched other_lead / red_queue | b delay med s | a zeros | b mean | held zeros | held mean | held vs RSG [CI] | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| RTJA0-L18-s0 | 0 | 0.917 / 0.923 | 0.833 / 0.743 | +0.52 / +0.77 | 3 | 0.728 | 33 | 0.8780 | +0.0124 [-0.0087, +0.0312] | no (delay_other_lead, delay_red_queue, launch_other_lead, launch_red_queue) |
| RTJA05-L18-s0 | 0.5 | 0.889 / 0.897 | 0.875 / 0.743 | +0.44 / +0.37 | 2 | 0.779 | 34 | 0.8722 | +0.0066 [-0.0113, +0.0292] | no (delay_other_lead, delay_red_queue, launch_other_lead, launch_red_queue) |
| RTJA1-L18-s0 | 1 | 0.889 / 0.872 | 0.875 / 0.743 | +0.37 / +0.32 | 2 | 0.789 | 33 | 0.8776 | +0.0121 [-0.0095, +0.0283] | no (delay_other_lead, delay_red_queue, launch_other_lead, launch_red_queue) |
| RTB1ARO-L18-s0 (stored) | - | 0.833 / 0.718 | 1.000 / 0.971 | -0.03 / -0.08 | 6 | 0.927 | - | - | - | - |
| JTN-L18-s0 (stored) | - | 0.889 / 0.923 | 0.812 / 0.800 | +0.37 / +0.32 | 2 | 0.679 | - | - | - | - |
| RSG-L18-s0 (stored) | - | 0.500 / 0.385 | 0.979 / 0.943 | -0.23 / -0.13 | 7 | 0.938 | - | - | - | - |

**No arm passes: stop (supplement 12), the winner's reads do not run.**
