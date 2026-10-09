| domain                | arm   | policy                                                           |   frames on base speed |   score |    - A |     lo |     hi |    - S |   S lo |   S hi |
|:----------------------|:------|:-----------------------------------------------------------------|-----------------------:|--------:|-------:|-------:|-------:|-------:|-------:|-------:|
| navtest (no-EC EPDMS) | P2H10 | AS everywhere (adapter path, base speed profile)                 |                  1.000 |  85.906 | -3.294 | -3.871 | -2.773 |  3.517 |  2.637 |  4.457 |
| navtest (no-EC EPDMS) | P2H10 | base speed on lead frames                                        |                  0.487 |  87.056 | -2.144 | -2.624 | -1.695 |  4.667 |  3.753 |  5.641 |
| navtest (no-EC EPDMS) | P2H10 | base speed on closing-lead frames                                |                  0.258 |  88.959 | -0.241 | -0.448 | -0.061 |  6.570 |  5.653 |  7.533 |
| navtest (no-EC EPDMS) | P2H10 | base speed when moving (v0 >= 0.5), adapter launch at standstill |                  0.930 |  87.242 | -1.958 | -2.404 | -1.529 |  4.853 |  3.914 |  5.858 |
| navtest (no-EC EPDMS) | P2H10 | base speed on no-lead frames                                     |                  0.513 |  88.051 | -1.150 | -1.525 | -0.792 |  5.662 |  4.746 |  6.636 |
| navtest (no-EC EPDMS) | P2H10 | oracle max(A, AS) per frame (privileged upper bound)             |                nan     |  91.303 |  2.103 |  1.847 |  2.363 |  8.914 |  7.946 |  9.894 |
| navtest (no-EC EPDMS) | SH30  | AS everywhere (adapter path, base speed profile)                 |                  1.000 |  86.713 | -3.353 | -3.909 | -2.856 |  4.324 |  3.389 |  5.345 |
| navtest (no-EC EPDMS) | SH30  | base speed on lead frames                                        |                  0.487 |  87.863 | -2.202 | -2.684 | -1.755 |  5.474 |  4.484 |  6.516 |
| navtest (no-EC EPDMS) | SH30  | base speed on closing-lead frames                                |                  0.258 |  89.828 | -0.237 | -0.446 | -0.059 |  7.439 |  6.483 |  8.462 |
| navtest (no-EC EPDMS) | SH30  | base speed when moving (v0 >= 0.5), adapter launch at standstill |                  0.930 |  88.098 | -1.967 | -2.412 | -1.549 |  5.709 |  4.708 |  6.764 |
| navtest (no-EC EPDMS) | SH30  | base speed on no-lead frames                                     |                  0.513 |  88.914 | -1.151 | -1.500 | -0.832 |  6.525 |  5.580 |  7.539 |
| navtest (no-EC EPDMS) | SH30  | oracle max(A, AS) per frame (privileged upper bound)             |                nan     |  92.075 |  2.010 |  1.759 |  2.255 |  9.686 |  8.677 | 10.715 |
| WOD val (RFS)         | P2H10 | AS everywhere (adapter path, base speed profile)                 |                  1.000 |   8.067 |  0.359 |  0.180 |  0.546 |  0.062 | -0.023 |  0.152 |
| WOD val (RFS)         | P2H10 | base speed on lead frames                                        |                  0.344 |   7.859 |  0.150 |  0.055 |  0.253 | -0.146 | -0.325 |  0.033 |
| WOD val (RFS)         | P2H10 | base speed on closing-lead frames                                |                  0.044 |   7.704 | -0.004 | -0.036 |  0.024 | -0.301 | -0.508 | -0.100 |
| WOD val (RFS)         | P2H10 | base speed when moving (v0 >= 0.5), adapter launch at standstill |                  0.749 |   7.907 |  0.199 |  0.073 |  0.333 | -0.097 | -0.264 |  0.064 |
| WOD val (RFS)         | P2H10 | base speed on no-lead frames                                     |                  0.656 |   7.916 |  0.208 |  0.056 |  0.371 | -0.089 | -0.222 |  0.047 |
| WOD val (RFS)         | P2H10 | oracle max(A, AS) per frame (privileged upper bound)             |                nan     |   8.443 |  0.735 |  0.604 |  0.875 |  0.438 |  0.313 |  0.572 |
| WOD val (RFS)         | SH30  | AS everywhere (adapter path, base speed profile)                 |                  1.000 |   8.077 |  0.344 |  0.169 |  0.527 |  0.072 | -0.013 |  0.163 |
| WOD val (RFS)         | SH30  | base speed on lead frames                                        |                  0.344 |   7.881 |  0.147 |  0.057 |  0.242 | -0.124 | -0.308 |  0.056 |
| WOD val (RFS)         | SH30  | base speed on closing-lead frames                                |                  0.044 |   7.731 | -0.003 | -0.032 |  0.021 | -0.274 | -0.477 | -0.075 |
| WOD val (RFS)         | SH30  | base speed when moving (v0 >= 0.5), adapter launch at standstill |                  0.749 |   7.922 |  0.188 |  0.060 |  0.324 | -0.083 | -0.248 |  0.076 |
| WOD val (RFS)         | SH30  | base speed on no-lead frames                                     |                  0.656 |   7.931 |  0.197 |  0.038 |  0.364 | -0.074 | -0.203 |  0.060 |
| WOD val (RFS)         | SH30  | oracle max(A, AS) per frame (privileged upper bound)             |                nan     |   8.469 |  0.735 |  0.606 |  0.874 |  0.464 |  0.340 |  0.602 |

CI: percentile bootstrap
