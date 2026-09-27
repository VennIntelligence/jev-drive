### ab6.jsonl

| round | arm | starts | crashed at start (RenderThread / other) | threads per server | aggregate FPS | ms/tick mean (median) | server cores | server core-s per tick | VRAM GB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | base | 6 | 3 (3 / 0) | 301 | 26.3 | 115 (98) | 2.54 | 0.292 | 8.5 |
| 1 | reduced | 6 | 1 (0 / 1) | 109 | 26.1 | 193 (150) | 1.79 | 0.345 | 8.9 |
| 2 | base | 6 | 2 (2 / 0) | 301 | 28.8 | 140 (107) | 2.16 | 0.302 | 8.5 |
| 3 | reduced | 6 | 1 (1 / 0) | 109 | 30.2 | 167 (120) | 1.85 | 0.308 | 8.7 |
| 4 | base | 6 | 0 (0 / 0) | 301 | 32.2 | 188 (147) | 1.63 | 0.304 | 8.9 |
| 5 | reduced | 6 | 1 (1 / 0) | 109 | 30.4 | 166 (128) | 1.88 | 0.312 | 8.6 |

- base: 18 starts, 5 RenderThread-timeout crashes, 0 other failures
- reduced: 18 starts, 2 RenderThread-timeout crashes, 1 other failures

### e2-single.jsonl

| round | arm | starts | crashed at start (RenderThread / other) | threads per server | aggregate FPS | ms/tick mean (median) | server cores | server core-s per tick | VRAM GB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | base | 1 | 0 (0 / 0) | 301 | 9.5 | 105 (104) | 3.23 | 0.339 | 9.8 |
| 1 | reduced | 1 | 0 (0 / 0) | 109 | 9.8 | 102 (102) | 3.24 | 0.331 | 9.7 |
| 2 | base | 1 | 0 (0 / 0) | 301 | 10.2 | 98 (103) | 3.24 | 0.316 | 9.9 |
| 3 | reduced | 1 | 0 (0 / 0) | 109 | 9.4 | 106 (106) | 3.24 | 0.343 | 9.8 |
| 4 | base | 1 | 0 (0 / 0) | 301 | 9.4 | 106 (105) | 3.26 | 0.347 | 9.8 |
| 5 | reduced | 1 | 0 (0 / 0) | 109 | 9.4 | 106 (105) | 3.23 | 0.342 | 9.8 |

- base: 3 starts, 0 RenderThread-timeout crashes, 0 other failures
- reduced: 3 starts, 0 RenderThread-timeout crashes, 0 other failures

### e3-cpu32-CONTAMINATED.jsonl

| round | arm | starts | crashed at start (RenderThread / other) | threads per server | aggregate FPS | ms/tick mean (median) | server cores | server core-s per tick | VRAM GB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | base | 6 | 1 (1 / 0) | 361 | 34.8 | 146 (122) | 2.07 | 0.301 | 8.4 |
| 1 | reduced | 6 | 1 (1 / 0) | 169 | 36.1 | 142 (125) | 2.13 | 0.299 | 8.3 |
| 2 | base | 6 | 0 (0 / 0) | 361 | 32.9 | 197 (160) | 1.59 | 0.295 | 9.0 |
| 3 | reduced | 6 | 1 (0 / 1) | 169 | 25.8 | 194 (145) | 1.88 | 0.366 | 9.8 |

- base: 12 starts, 1 RenderThread-timeout crashes, 0 other failures
- reduced: 12 starts, 1 RenderThread-timeout crashes, 1 other failures

