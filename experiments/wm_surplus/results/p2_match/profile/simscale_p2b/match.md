| board       | metric   | driver       | reference   |   driver mean |   ref mean |   diff |     lo |     hi |   n |    drc |    dnc |   ddac |   dttc |     dc |
|:------------|:---------|:-------------|:------------|--------------:|-----------:|-------:|-------:|-------:|----:|-------:|-------:|-------:|-------:|-------:|
| hugsim:exam | HD-Score | SimScale-LTF | WA-JEPA     |         0.340 |      0.458 | -0.118 | -0.190 | -0.050 |  64 | -0.104 | -0.153 | -0.008 | -0.151 |  0.173 |
| hugsim:exam | HD-Score | SimScale-DD  | WA-JEPA     |         0.308 |      0.458 | -0.150 | -0.222 | -0.086 |  64 | -0.130 | -0.080 |  0.015 | -0.087 | -0.156 |

CI: percentile bootstrap

driver - reference, paired per unit (hugsim: HD-Score, navtest: EPDMS x 100, navhard: two-stage EPDMS); per-board tables beside this file.
