# six_rsg

containerised: n 6, mean scene score 0.0876, zeros 5, at 1 0
driver: 992 drive calls, 0 inference errors, total p50 / p90 / p99 / max 52.8 / 71.4 / 140.2 / 186.5 ms (encode p50 31.4, frames p50 None, wait p50 0.0)
native reference: n 6, mean 0.2543, zeros 4, at 1 1
scenes that differ: 1 of 6 (max |diff| 1.0); paired mean diff -0.1667, within 0.01: 5, within 0.05: 5, zero / non-zero agree: 5 of 6

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-49bcab6f-d6a9-48de-a7c3-47f91059fc66 | 0.0000 | 1.0000 | <AioRpcError of RPC that terminated with:
	status = StatusCo / - |
