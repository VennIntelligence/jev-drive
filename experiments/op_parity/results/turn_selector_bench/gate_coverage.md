| variant   |   seed |   gate share % |   precision % (logged turn | gate) |   recall % (gate | logged turn) |   moved % of all |   moved % of gated |   moved % of logged turns |   moved % of straight |
|:----------|-------:|---------------:|-----------------------------------:|--------------------------------:|-----------------:|-------------------:|--------------------------:|----------------------:|
| B         |      0 |          26.10 |                              93.22 |                           93.69 |            19.76 |              75.71 |                     70.64 |                  0.19 |
| B         |      1 |          26.07 |                              93.27 |                           93.63 |            20.02 |              76.82 |                     71.50 |                  0.21 |
| A         |      0 |         100.00 |                              25.97 |                          100.00 |            74.10 |              74.10 |                     76.00 |                 75.34 |
| A         |      1 |         100.00 |                              25.97 |                          100.00 |            74.42 |              74.42 |                     76.92 |                 75.17 |

navtest, 12 146 tokens; logged turn = |dyaw| >= 20 deg (privileged bucket); gate B = the model's own exported 4 s heading >= 20 deg
