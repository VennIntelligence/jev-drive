| score                             | AUC                  |
|:----------------------------------|:---------------------|
| base rate % of token-seeds        | 15.3                 |
| - identity map margin (4 s), PRIV | 0.706 [0.670, 0.747] |
| - identity own-edge margin (4 s)  | 0.546 [0.506, 0.578] |
| ridge on the 0 / 1 label: E       | 0.692 [0.660, 0.720] |
| ... E + map margins, PRIV         | 0.758 [0.733, 0.787] |
| ... E + own-edge margins          | 0.693 [0.663, 0.722] |
| ... E + hidden state (32 PCs)     | 0.686 [0.655, 0.716] |
| P1 head: max predicted gain       | 0.619 [0.570, 0.659] |
| P2 head: max predicted gain       | 0.651 [0.604, 0.698] |
| P3 rule: - identity margin        | 0.706 [0.670, 0.747] |
| N1 head: max predicted gain       | 0.501 [0.460, 0.538] |
| N2 head: max predicted gain       | 0.450 [0.396, 0.498] |
| N3 rule: - identity margin        | 0.546 [0.506, 0.578] |
| N4 head: max predicted gain       | 0.510 [0.475, 0.545] |
| N5 head: max predicted gain       | 0.506 [0.462, 0.547] |
| N6 head: max predicted gain       | 0.545 [0.498, 0.589] |
| N7 head: max predicted gain       | 0.518 [0.477, 0.558] |

out-of-fold AUC of 'F19 x pc has a candidate with gain > 0' per (token, seed), > 20 deg; log-cluster bootstrap B 1000. Decision 186 main arm E: 0.692 [0.660, 0.720].
