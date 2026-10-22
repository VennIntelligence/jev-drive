| board       | metric   | driver       | reference   |   driver mean |   ref mean |   diff |     lo |     hi |   n |    drc |    dnc |   ddac |   dttc |     dc |
|:------------|:---------|:-------------|:------------|--------------:|-----------:|-------:|-------:|-------:|----:|-------:|-------:|-------:|-------:|-------:|
| hugsim:exam | HD-Score | SimScale-LTF | WA-JEPA     |         0.342 |      0.484 | -0.142 | -0.215 | -0.070 |  64 | -0.131 | -0.164 | -0.006 | -0.167 |  0.148 |
| hugsim:exam | HD-Score | SimScale-DD  | WA-JEPA     |         0.307 |      0.484 | -0.177 | -0.249 | -0.111 |  64 | -0.152 | -0.094 |  0.013 | -0.113 | -0.188 |
| hugsim:exam | HD-Score | P0           | WA-JEPA     |         0.271 |      0.484 | -0.213 | -0.323 | -0.105 |  64 | -0.262 |  0.086 | -0.046 |  0.087 |  0.081 |

CI: percentile bootstrap

driver - reference, paired per unit (hugsim: HD-Score, navtest: EPDMS x 100, navhard: two-stage EPDMS); per-board tables beside this file.
