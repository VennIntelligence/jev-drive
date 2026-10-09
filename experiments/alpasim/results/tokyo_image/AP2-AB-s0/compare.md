# default

containerised: n 48, mean scene score 0.9521, zeros 2, at 1 41
driver: 480 drive calls, 0 inference errors, total p50 / p90 / p99 / max 55.9 / 72.1 / 129.0 / 156.8 ms (encode p50 21.0, frames p50 25.3, wait p50 0.0)
native reference: n 48, mean 0.9320, zeros 3, at 1 41
scenes that differ: 5 of 48 (max |diff| 0.9719)

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| 2021.05.25.14.16.10_veh-35_00083_00485-c9b12b21fa7c57fd | 0.8124 | 0.8185 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-368cb65e8fef57b7 | 0.9719 | 0.0000 | - / collision_at_fault |
| 2021.05.25.14.16.10_veh-35_01100_01664-6fbe0e06902e5304 | 0.9978 | 0.9989 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-82cd122751085a80 | 0.9616 | 0.9616 | - / - |
| 2021.05.25.14.16.10_veh-35_01690_02183-54e1cb577c0a5f7e | 0.9592 | 0.9578 | - / - |
