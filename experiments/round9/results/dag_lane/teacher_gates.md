# DAG-LANE teacher gates on human states and offroad events (dl.py tcmp; dag.py t1 / t2 with DAG_TV=b3 and b3L)

T1 human states (both teachers labelled): 18745. Median |error| at 2 / 4 s: b3 0.075 / 0.089 m, b3L 0.057 / 0.068 m. States where b3 misses the human by > 1.75 m at 4 s: 2172 (b3 median 2.529, b3L 0.2045 m).
G-L2 (b3L no worse than b3 + 0.01 m; on those states b3L <= 0.5 m) -> **PASS**.

T2 in canvas: b3 teacher inside 62 / 65 (95%); b3L 62 / 65 (95%). G-L3 (b3L >= 90 % and >= b3 - 1) -> **PASS**.
