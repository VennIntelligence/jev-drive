# image-command fine-tune Q3: q3SB-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.44 [0.42, 0.47] (b 0.45, n 356) | -0.01 [-0.02, +0.00] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| sky | 225 | +6.49 | +6.49 [+5.27, +7.80] | 0.23 [0.20, 0.26] (b 0.00) | +0.23 [+0.20, +0.26] | 0.68 [0.64, 0.71] (b 0.45, n 356) | +0.23 [+0.19, +0.27] | +0.19 / +2.26 / +7.73 | -0.90 | -0.29 [-0.51, -0.09] |

Guards: {"none_drift_median_junction": 0.1879173219203949, "sky_disc_move_median": 0.3107162272373921, "sky_disc_abs_dy4_mean": 2.4908781436766305, "sky_disc_move_median_ref": 0.24228987798510915, "sky_disc_abs_dy4_mean_ref": 0.2796735839161208, "sky_disc_n": 225, "sky_wrong_move_median": 0.43723179993729444, "sky_wrong_abs_dy4_mean": 5.620224553585614, "sky_wrong_move_median_ref": 0.2688277182613713, "sky_wrong_abs_dy4_mean_ref": 0.29156829488924196, "sky_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": 0.14250532122553128, "lo": 0.0030942566508944374, "hi": 0.2893566048705577, "mean_a": 3.7701125214235454, "mean_b": 3.6276072001980135}}
