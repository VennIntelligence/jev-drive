# image-command fine-tune Q3: q3SA-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.43 [0.41, 0.45] (b 0.45, n 368) | -0.02 [-0.04, -0.01] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 225 | +9.25 | +9.25 [+8.08, +10.50] | 0.39 [0.37, 0.42] (b 0.00) | +0.39 [+0.36, +0.42] | 0.78 [0.74, 0.82] (b 0.45, n 368) | +0.33 [+0.29, +0.38] | +1.65 / +3.34 / +9.34 | -1.04 | -0.43 [-0.66, -0.21] |

Guards: {"none_drift_median_junction": 0.2706422507762909, "sky_disc_move_median": 0.4342068649837944, "sky_disc_abs_dy4_mean": 3.0260801445278482, "sky_disc_move_median_ref": 0.24228987798510915, "sky_disc_abs_dy4_mean_ref": 0.2796735839161208, "sky_disc_n": 225, "sky_wrong_move_median": 0.8474208138496118, "sky_wrong_abs_dy4_mean": 7.521780945858886, "sky_wrong_move_median_ref": 0.2688277182613713, "sky_wrong_abs_dy4_mean_ref": 0.29156829488924196, "sky_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": -0.6293039141063883, "lo": -1.0646544488981147, "hi": -0.22264881683998652, "mean_a": 2.9983032860916254, "mean_b": 3.6276072001980135}}
