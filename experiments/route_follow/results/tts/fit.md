# TTS weight fit (prereg plans/2026-10-16-tts-prereg.md section 3)

Rows: on1_L18RSG own states kept by C2 (t0 not in contact, both labels), K = 16 candidates (own, lat-1, lat-0.5, lat+0.5, lat+1, g0.8, g1.2, g1.4, s0.9, s1.1, g0.8s0.9, g0.8s1.1, g1.2s0.9, g1.2s1.1, g1.4s0.9, g1.4s1.1). Grid 400 sets, feasible 348 (mean deviation <= 0.15 m, switch share <= 0.25, route distance change <= 0.0 m). Candidate-0 labels equal C2's on 1.0000 of the bins.

**Frozen weights**: w_road 1.0, w_agent 0.5, w_route 0.1 / m, w_dev 0.1 / m, margin 0.02. Train contact 0.0987 (own plan) -> 0.0628 (selected); stop line (< 0.005 lower): **not hit, reads go ahead**. Unconstrained grid minimum 0.0616 (D 0.074 m, switch 0.113).

| split | rows / logs | contact own | contact selected | d contact [log CI] | d agent | d road | switch | mean dev (m) | d route (m) | oracle |
|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|
| train | 13698 / 24 | 0.0987 | 0.0628 | -0.0359 [-0.0433, -0.0270] | -0.0138 [-0.0181, -0.0096] | -0.0248 [-0.0322, -0.0165] | 0.101 | 0.061 | -0.020 | 0.0497 |
| inner | 455 / 1 | 0.0791 | 0.0769 | -0.0022 [-0.0022, -0.0022] | +0.0000 [+0.0000, +0.0000] | -0.0022 [-0.0022, -0.0022] | 0.099 | 0.056 | -0.009 | 0.0308 |
| dev | 3618 / 4 | 0.1031 | 0.0987 | -0.0044 [-0.0075, +0.0010] | +0.0008 [-0.0009, +0.0025] | -0.0025 [-0.0069, +0.0097] | 0.086 | 0.051 | -0.005 | 0.0484 |

Picked candidates and switches by segment (switched / decisions):

- train: picked {'own': 12313, 'lat-1': 240, 'lat-0.5': 269, 'lat+0.5': 252, 'lat+1': 184, 'g0.8': 147, 'g1.2': 84, 'g1.4': 60, 's0.9': 1, 's1.1': 2, 'g0.8s0.9': 56, 'g0.8s1.1': 23, 'g1.2s0.9': 10, 'g1.2s1.1': 11, 'g1.4s0.9': 20, 'g1.4s1.1': 26}; by segment {'approach': [556, 3409], 'junction': [12, 39], 'straight': [797, 10184], 'none': [20, 66]}
- inner: picked {'own': 410, 'lat-1': 5, 'lat-0.5': 9, 'lat+0.5': 14, 'lat+1': 6, 'g0.8': 8, 'g1.4': 1, 'g1.4s0.9': 2}; by segment {'approach': [21, 105], 'junction': [3, 5], 'straight': [21, 345], 'none': [0, 0]}
- dev: picked {'own': 3308, 'lat-1': 39, 'lat-0.5': 62, 'lat+0.5': 63, 'lat+1': 33, 'g0.8': 37, 'g1.2': 18, 'g1.4': 11, 's0.9': 5, 's1.1': 2, 'g0.8s0.9': 13, 'g0.8s1.1': 6, 'g1.2s0.9': 4, 'g1.2s1.1': 3, 'g1.4s0.9': 8, 'g1.4s1.1': 6}; by segment {'approach': [158, 1053], 'junction': [0, 2], 'straight': [151, 2548], 'none': [1, 15]}

Contact rate by candidate (train): own 0.099, lat-1 0.365, lat-0.5 0.164, lat+0.5 0.154, lat+1 0.265, g0.8 0.124, g1.2 0.129, g1.4 0.189, s0.9 0.098, s1.1 0.104, g0.8s0.9 0.116, g0.8s1.1 0.141, g1.2s0.9 0.122, g1.2s1.1 0.136, g1.4s0.9 0.171, g1.4s1.1 0.200

train = the head's own training rows (its probabilities are optimistic there); inner = train logs held out of the head's gradient; dev = held-out logs (4 logs). Only train chose the weights.
