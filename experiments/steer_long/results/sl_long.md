# STEER-LONG longitudinal read (sections 3.6 + amendment 1, 3.13 amendment 3)

PAI proxy check on nuPlan (along-track gap, proxy vs true): median |err| 0.00 m, p90 3.57 m, sign right (true gap > 1 m) 100.0 %, n 48774

## Speed ratio v_ego / v_log (position-aligned, log moving >= 2 m/s, before any flag)

| track / label | n | median | 0-10 | 10-25 | 25-50 | 50-100 | > 100 m | slope log ratio / 100 m [95 % CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| nuplan / YR10m10-F-s0 | 6596 | 0.92 | 1.00 | 0.90 | 0.78 | 0.85 | 1.01 | -0.571 [-0.904, -0.331] farther = slower |
| nuplan / YR10m10-F-s1 | 6670 | 0.92 | 1.00 | 0.89 | 0.79 | 0.89 | 1.02 | -0.426 [-0.725, -0.202] farther = slower |
| nuplan / YS-L18-s0 | 6772 | 0.90 | 1.00 | 0.89 | 0.75 | 0.87 | 0.99 | -0.587 [-0.948, -0.337] farther = slower |
| nuplan / YS-L18-s1 | 6766 | 0.91 | 1.00 | 0.88 | 0.76 | 0.87 | 0.98 | -0.563 [-0.919, -0.318] farther = slower |
| nuplan / YS2G-L18-s0 | 6770 | 0.90 | 1.00 | 0.88 | 0.75 | 0.86 | 0.98 | -0.642 [-1.012, -0.383] farther = slower |
| nuplan / YS2G-L18-s1 | 6741 | 0.91 | 1.00 | 0.89 | 0.76 | 0.86 | 1.00 | -0.612 [-0.974, -0.359] farther = slower |
| pai / ybase | 6804 | 0.81 | 0.87 | 0.96 | 0.87 | 0.75 | 0.73 | -0.064 [-0.128, -0.017] farther = slower |
| pai / ys0 | 7019 | 0.80 | 0.90 | 0.97 | 0.88 | 0.70 | 0.73 | -0.079 [-0.144, -0.027] farther = slower |
| pai / ys1 | 7131 | 0.80 | 0.90 | 0.96 | 0.88 | 0.71 | 0.73 | -0.070 [-0.132, -0.025] farther = slower |

By time since the last stop (or start): median ratio

| track / label | 0-2 | 2-4 | 4-6 | 6-9 | 9-20 s |
|:--|--:|--:|--:|--:|--:|
| nuplan / YR10m10-F-s0 | 0.99 | 0.90 | 0.82 | 0.72 | - |
| nuplan / YR10m10-F-s1 | 0.99 | 0.91 | 0.83 | 0.72 | - |
| nuplan / YS-L18-s0 | 0.99 | 0.90 | 0.80 | 0.69 | - |
| nuplan / YS-L18-s1 | 0.99 | 0.90 | 0.81 | 0.69 | - |
| nuplan / YS2G-L18-s0 | 0.99 | 0.90 | 0.81 | 0.69 | - |
| nuplan / YS2G-L18-s1 | 0.99 | 0.91 | 0.82 | 0.69 | - |
| pai / ybase | 1.00 | 0.89 | 0.79 | 0.74 | 0.70 |
| pai / ys0 | 1.00 | 0.89 | 0.79 | 0.74 | 0.65 |
| pai / ys1 | 1.00 | 0.90 | 0.79 | 0.74 | 0.65 |

Decomposition (means, m/s; v_plan = model plan's first 1 s)

| track / label | v_log | v_plan | v_ego | total | model part | execution part | model share |
|:--|--:|--:|--:|--:|--:|--:|--:|
| nuplan / YR10m10-F-s0 | 6.99 | 5.78 | 5.85 | +1.14 | +1.21 | -0.07 | 1.07 |
| nuplan / YR10m10-F-s1 | 6.96 | 5.83 | 5.89 | +1.06 | +1.13 | -0.06 | 1.06 |
| nuplan / YS-L18-s0 | 6.96 | 5.63 | 5.71 | +1.25 | +1.33 | -0.08 | 1.06 |
| nuplan / YS-L18-s1 | 6.96 | 5.67 | 5.74 | +1.22 | +1.29 | -0.08 | 1.06 |
| nuplan / YS2G-L18-s0 | 6.99 | 5.66 | 5.74 | +1.25 | +1.34 | -0.08 | 1.06 |
| nuplan / YS2G-L18-s1 | 6.99 | 5.69 | 5.77 | +1.22 | +1.30 | -0.08 | 1.07 |
| pai / ybase | 14.40 | 10.54 | 10.59 | +3.82 | +3.86 | -0.05 | 1.01 |
| pai / ys0 | 14.04 | 10.22 | 10.25 | +3.80 | +3.82 | -0.02 | 1.01 |
| pai / ys1 | 14.35 | 10.39 | 10.43 | +3.92 | +3.96 | -0.04 | 1.01 |

## Stops (v < 1 m/s >= 1 s before any flag); unexplained = log moving >= 2 m/s here, nothing within 10 m, not past the log end

| track / label | stops | at start | unexplained | P | L | V | M | other |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| nuplan / YR10m10-F-s0 | 121 | 72 | 15 | 4 | 0 | 0 | 11 | 0 |
| nuplan / YR10m10-F-s1 | 127 | 72 | 17 | 2 | 0 | 0 | 15 | 0 |
| nuplan / YS-L18-s0 | 127 | 73 | 21 | 5 | 0 | 0 | 16 | 0 |
| nuplan / YS-L18-s1 | 129 | 73 | 25 | 4 | 0 | 0 | 21 | 0 |
| nuplan / YS2G-L18-s0 | 121 | 72 | 20 | 3 | 0 | 0 | 17 | 0 |
| nuplan / YS2G-L18-s1 | 127 | 72 | 25 | 5 | 0 | 0 | 20 | 0 |
| pai / ybase | 75 | 37 | 3 | 1 | 0 | 0 | 2 | 0 |
| pai / ys0 | 65 | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| pai / ys1 | 72 | 37 | 3 | 2 | 0 | 0 | 1 | 0 |

## Slowdown onsets (v >= 5, drop to <= 0.7 v and >= 2 m/s within 3 s, before any flag)

| track / label | onsets | per 100 rollouts | log did not slow (>= 0.85) | P | L (phantom) | V | M | P flag | L flag | median recover s |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| nuplan / YR10m10-F-s0 | 204 | 45.7 | 127 | 103 | 0 (0) | 0 | 24 | 103 | 0 | 1.4 |
| nuplan / YR10m10-F-s1 | 195 | 43.7 | 114 | 104 | 0 (0) | 0 | 10 | 104 | 0 | nan |
| nuplan / YS-L18-s0 | 212 | 47.5 | 136 | 117 | 0 (0) | 0 | 19 | 117 | 0 | 1.9 |
| nuplan / YS-L18-s1 | 202 | 45.3 | 124 | 107 | 0 (0) | 0 | 17 | 107 | 0 | nan |
| nuplan / YS2G-L18-s0 | 202 | 45.3 | 129 | 110 | 0 (0) | 0 | 19 | 110 | 0 | nan |
| nuplan / YS2G-L18-s1 | 203 | 45.5 | 125 | 100 | 0 (0) | 0 | 25 | 100 | 0 | nan |
| pai / ybase | 157 | 55.9 | 111 | 107 | 1 (0) | 0 | 3 | 107 | 48 | 5.5 |
| pai / ys0 | 161 | 57.3 | 114 | 107 | 0 (0) | 0 | 7 | 107 | 38 | 5.5 |
| pai / ys1 | 167 | 59.4 | 119 | 112 | 2 (0) | 0 | 5 | 112 | 35 | 7.0 |
- nuplan: unexplained onsets 755, largest class P 84.9 % -> >= 50 %
- pai: unexplained onsets 344, largest class P 94.8 % -> >= 50 %

Onsets followed by the rollout's zero within 5 s: nuplan 26 / 1218, pai 23 / 485

## Rear impacts (collision_rear; never at-fault by the scorer)

- nuplan: rollouts with a rear impact 252 (first event 230), ego speed median 1.8 m/s, v_ego / v_log median 0.29, ego < 1 m/s 33, mean score 0.786, at-fault flag also set 0
- pai: rollouts with a rear impact 313 (first event 206), ego speed median 4.6 m/s, v_ego / v_log median 0.47, ego < 1 m/s 31, mean score 0.566, at-fault flag also set 0

## Corridor zeros: where the ego was along the log at the event

- nuplan: 141 corridor zeros; lateral at event median 4.38 m; time-aligned along-track gap median 4.0 m, > 4 m 70; behind the log pose by > 4 m 34, past the log end 38
- pai: 171 corridor zeros; lateral at event median 4.18 m; time-aligned along-track gap median 11.8 m, > 4 m 140; 
