| path                                                   |    n |   LLM tokens |   preprocess ms (median) |   model ms (median) |   model ms (p95) |   peak VRAM GB |
|:-------------------------------------------------------|-----:|-------------:|-------------------------:|--------------------:|-----------------:|---------------:|
| Q1: 3 cameras, 1 frame                                 |   40 |     3060.000 |                  106.701 |             215.188 |          216.061 |          8.320 |
| QV / QL: 3 cameras x 4 frames (video path)             |   40 |     6120.000 |                  385.736 |             443.305 |          522.683 |         13.292 |
| head (linear, PCA folded into the weights; CPU, numpy) | 1000 |      nan     |                  nan     |               0.014 |          nan     |        nan     |
