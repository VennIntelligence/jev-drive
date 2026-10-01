**Summary.** CARLA 0.9.15 runs headless on our Blackwell cards and renders on the GPU; real Bench2Drive routes run
end to end on Town12. Hard blocker was Vulkan in the container: install `libegl1` (not a loader rebuild) and pin
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`; check with `vulkaninfo --summary`. Start/stop with
`scripts/carla_server.sh start 0` (rpc port 2000 + 50i); GPU via `-graphicsadapter=<rank>`, never
`CUDA_VISIBLE_DEVICES`. Key traps: TM-port reuse, RenderThread 60 s timeout then Signal 11, pids.max 20480 thread cap
(`-RPCThreads=4 -StreamingThreads=4 -SecondaryThreads=4` takes a server from 301 to 109 threads), no `pkill -f`.
Python client: PyPI wheel in a 3.8 venv `$DATA_DIR/envs/carla`. Sizing and capacity numbers are superseded by
docs/closed-loop-runbook.md and docs/bench2drive-cost.md; the "Numbers" section is the older server-side evidence.

**Sections.**
- Vulkan in the container: the one hard blocker, and its fix - libegl1 fix, packages, ICD pin
- Traps - ports, crashes, flags, sensors, shutdown, determinism
- Threads per server - thread pools, reduced-pool flags, equivalence tests
- Town12 and Large Maps - sensor-dormancy crash, 11 failed routes, leaderboard guidance
- Python client - cp38 wheel venv install
- Starting and stopping a server - carla_server.sh, render check via carla_bench.py
- Numbers, and where each came from - bench vs leaderboard FPS, saturation corrections
- Prerequisites for a real evaluation - downloads, AdditionalMaps import, 220 routes by town
