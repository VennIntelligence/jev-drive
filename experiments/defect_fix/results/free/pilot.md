# NR1FREE pilot (NR1FREE1K-L18-s0 against NTVN1K-L18-s0; NR1 prereg supplement 15)

| arm | a wait other_lead / red_queue | b launched other_lead / red_queue | b delay med s | a zeros | b mean | held zeros | held mean | held front coll | held vs control [CI] | held vs RSG [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NR1FREE1K-L18-s0 | 0.583 / 0.487 | 0.938 / 0.943 | -0.23 / -0.03 | 6 | 0.929 | 47 | 0.8885 | 12 | +0.0378 [+0.0195, +0.0615] | +0.0229 [+0.0051, +0.0542] |
| NTVN1K-L18-s0 | 0.861 / 0.872 | 0.792 / 0.743 | +0.34 / +0.32 | 3 | 0.731 | 37 | 0.8507 | 5 | - | -0.0149 [-0.0331, +0.0088] |
| JTN-L18-s0 (stored) | 0.889 / 0.923 | 0.812 / 0.800 | +0.37 / +0.32 | 2 | 0.679 | - | - | - | - | - |
| JTNS1K-L18-s0 (stored) | 0.944 / 0.923 | 0.729 / 0.657 | +0.57 / +0.77 | 3 | 0.654 | - | - | - | - | - |
| LDNTV1K-L18-s0 (stored) | 0.306 / 0.282 | 0.979 / 0.943 | -0.33 / -0.28 | 8 | 0.949 | - | - | - | - | - |
| RTB1ARO-L18-s0 (stored) | 0.833 / 0.718 | 1.000 / 0.971 | -0.03 / -0.08 | 6 | 0.927 | - | - | - | - | - |

Gates (supplement 15): held_zeros FAIL, held_front FAIL, launched_other_lead ok, launched_red_queue ok, delay_other_lead ok, delay_red_queue ok, wait_other_lead FAIL, wait_red_queue FAIL
Effect (a lead-class delay at least 0.10 s earlier than the control): yes

**Verdict: fail: stop the line**
