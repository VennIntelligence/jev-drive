**Summary.** Tokyo box = the machine with a monitor, for looking at CARLA (window, manual drive, screenshot,
recording): `ssh ujs@100.108.238.8` (only user `ujs`; physically in Shanghai, clock JST). Ryzen 9 9950X, 60 GB,
one RTX 3090 24 GB at CUDA index 0 / PCI 02:00.0; use CUDA index 0 directly and CARLA `-graphicsadapter=0`.
Stock CARLA at `/data/third_party/carla/CARLA_0.9.15`, Bench2Drive branch 0.0.4 at `/data/third_party/Bench2Drive`,
Python 3.8 env `/data/envs/carla`; DISPLAY=:0 needs XAUTHORITY from the Xwayland `-auth` arg. Network is Clash global
mode (port 7890): git over SSH works only with the selector on DIRECT, pip via official PyPI through the proxy.
Long runs go in tmux `jev`. Later sections are single-3090 windowed controller diagnostics, not GPU-box numbers;
windowed timings must not be mixed with off-screen ones.

**Sections.**
- What it is - login, hardware table, paths
- GPU selection (measured 2026-09-23) - one 3090, graphicsadapter mapping, UUID recheck
- Looking at CARLA - windowed launch, screenshots, recording
- Everything long runs in tmux - tmux sessions jev and dl
- Network: Clash in global mode, and nodes that decay - selector, node benchmarks, git/pip routing
- CARLA and Bench2Drive on this box - installs, runner command, map import, setup history
- Official agents and single-window preview (2026-09-22 update) - TCP agent, viewer, preview, timings
### Preview layout / preprocessing follow-up - viewer layout, parallel preprocessing check
- Controller diagnostic campaign and server reuse (2026-09-22) - Dev10 controller runs, reuse
### Reuse the process, keep recovery explicit - server reuse rules
### Interpretation and validation traps found in this iteration - validation pitfalls
### Controller feedback iteration: compass dropout and completed reuse audit - compass dropout, reuse audit
