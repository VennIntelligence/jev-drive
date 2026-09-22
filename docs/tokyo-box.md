# Tokyo box: the box with eyes

Read this when you need to **see** CARLA rather than measure it: a window on a real monitor, a
manual drive, a screenshot or a screen recording of a route. Experiments still belong on the GPU
box ([remote-box.md](remote-box.md)); this one exists so a human can look at what the numbers
describe.

## What it is

- Login: `ssh ujs@100.108.238.8` (Tailscale peer `tokyo-devbox`). Only the user `ujs` is accepted;
  `gaochengzhi1999`, `snail` and `root` are all refused. No `~/.ssh/config` alias yet.
- The name says Tokyo, the machine is in **Shanghai** (China Mobile). Its clock runs **JST
  (+0900)**, so every timestamp in a log there is one hour ahead of this Mac.
- It has a physical monitor and a real desktop session, which is the whole point of it.

| | |
|---|---|
| OS / kernel | Ubuntu 24.04.3, 6.17.0-35-generic |
| CPU / RAM | Ryzen 9 9950X, 16 cores / 32 threads; 60 GB |
| GPU | 2x RTX 3090 24 GB, driver 580.159.03 — **only GPU 1 is usable, see below** |
| Disks | `/` 1.8 TB (1.4 TB free), `/data` 3.6 TB (nearly empty) |
| Desktop | Xorg + GNOME on `:0`, seat0, 4384x2466 |
| Code | `~/mycode/jev-drive` (clean clone, `origin` is plain `github.com`) |
| Data | `/data` — treat it as `DATA_DIR`, same layout as the GPU box |

## GPU 0 is broken. Use GPU 1, and the rank is inverted

**GPU 0 must not be used.** Pin everything to GPU 1: `CUDA_VISIBLE_DEVICES=1` for PyTorch.

CARLA ignores `CUDA_VISIBLE_DEVICES` and takes `-graphicsadapter=<rank>`, a Vulkan physical-device
index. [carla.md](carla.md) warns that the rank need not match `nvidia-smi`. **On this box it is
inverted**, measured by starting a server at each rank and reading back the GPU UUID:

| flag | lands on `nvidia-smi` index | PCI |
|---|---|---|
| `-graphicsadapter=0` | **1** (the good card) | `0000:03:00.0` |
| `-graphicsadapter=1` | 0 (the broken card) | `0000:01:00.0` |

So CARLA on this box wants **`-graphicsadapter=0`**. Never assume it; confirm with
`nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv` against
`nvidia-smi --query-gpu=index,uuid --format=csv` after the server comes up. Started with no flag at
all, CARLA happened to pick GPU 1, but that is luck, not a guarantee.

## Looking at CARLA

X access from an SSH shell works with nothing but `DISPLAY=:0` — no `XAUTHORITY`, no `xhost`:

```bash
ssh ujs@100.108.238.8
cd /data/third_party/carla/CARLA_0.9.15
DISPLAY=:0 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  ./CarlaUE4.sh -windowed -ResX=1280 -ResY=720 -nosound \
                -carla-rpc-port=3000 -quality-level=Epic -graphicsadapter=0
```

A `CarlaUE4` window appears on the physical monitor and renders at once; the spectator camera is
driven with the mouse and WASD on that machine. Everything [carla.md](carla.md) says about CARLA
still applies here — silence is not a hang, an open RPC port is not a ready server, never
`pgrep -f` a pattern that matches your own command line (it matched three times while this page was
being written).

**Getting the picture back to the Mac.** ImageMagick's `import` and `xwd` are installed;
`scrot`, `gnome-screenshot`, `xdotool` and `wmctrl` are not.

```bash
# on the box
DISPLAY=:0 import -window root -resize 1400x /tmp/shot.png
# from the Mac
scp ujs@100.108.238.8:/tmp/shot.png .
```

`ffmpeg` and `x11vnc` are both installed, so a route can be recorded
(`DISPLAY=:0 ffmpeg -f x11grab -framerate 20 -i :0 out.mp4`) or watched live over VNC. Neither has
been used in anger yet.

## Everything long runs in tmux

Same rule as the GPU box ([long-runs.md](long-runs.md)), and here it has a second purpose: the
person sitting at the monitor sees the same pane you do. Sessions in use: `jev` for project work,
`dl` for downloads and installs.

## Network: Clash in global mode, and nodes that decay

Clash Verge runs on the desktop: mixed port **7890**, **mode `global`**, external controller on
`127.0.0.1:9097`. The controller secret is the `secret:` line of
`~/.local/share/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml`; it is not written down
here and must not be committed.

Three things about global mode that cost an hour here:

- **In global mode there is no direct traffic.** Everything, including a plain `curl` with no
  proxy variables set, goes through whatever node the `GLOBAL` selector points at, because Clash
  runs a TUN interface (`utun1024`) and hijacks DNS to fake IPs in `198.18.0.0/16`. A dead node
  therefore presents as `SSL_ERROR_SYSCALL` on an unrelated host, which reads exactly like a
  GFW reset. Check the selector before diagnosing anything else:

  ```bash
  curl -s --noproxy '*' -H "Authorization: Bearer $SECRET" \
    http://127.0.0.1:9097/proxies/GLOBAL | python3 -c 'import json,sys;print(json.load(sys.stdin)["now"])'
  ```

  Switch it with `PUT /proxies/GLOBAL`, body `{"name": "<node>"}`.

- **A node that benchmarks fast dies within a minute or two.** Measured 2026-09-22 against the
  CARLA release mirror, 12 s per node:

  | Node | MB/s |
  |---|---|
  | AWS Japan 09 | 9.29 |
  | Tokyo 07, 0.1x "dedicated line" | 6.87 |
  | `DIRECT` | 6.41 |
  | San Jose 01 | 5.89 |
  | Hong Kong HKT hy2 | 0.47 |
  | US "good for downloading" 0.01x | 0.03-0.41 |

  The winner of that table, AWS Japan 09, then stalled at 1 MB per 45 s and never recovered, while
  `DIRECT` — fourth-ranked — pulled the remaining 7.1 GB in one unbroken stream at ~10 MB/s. So a
  12 s benchmark ranks nothing that matters. **Make the downloader rotate the node on every stall**
  (`--speed-limit 300000 --speed-time 15` plus a `PUT` to the selector on each retry) instead of
  picking a winner up front. The "0.01x, good for downloading" nodes were the slowest on offer.

- **GitHub over SSH works only on `DIRECT`.** Every proxy node tried closes port 22
  (`Connection closed by 198.18.0.7 port 22` — the fake IP gives away that it never left the
  tunnel). So `git pull` here means setting the selector to `DIRECT` first, where it works fine.
  Python packages are the opposite case: install from **official PyPI through the proxy**, because
  in global mode a domestic mirror is routed abroad too and `mirrors.aliyun.com` just times out.

## CARLA and Bench2Drive on this box

| Path | What | Use it? |
|---|---|---|
| `/data/third_party/carla/CARLA_0.9.15` | stock CARLA 0.9.15 + all additional maps | **yes** |
| `~/mycode/Bench2Drive` | full repo, branch `main` @ `21d85ee`, `bench2drive220.xml` and Dev10 present | yes, but see below |
| `~/mycode/CARLA_Truck_Platoon_0.0.1` | a 0.9.15 **fork**, `VERSION` says `0.9.15-dirty`, Town01-07 + Town10HD only | no |
| `~/mycode/Carla-0.10.0` | CARLA 0.10.0, UE5 | no — Bench2Drive needs 0.9.15 |

The stock install was added on 2026-09-22 because neither pre-existing build could run a
Bench2Drive route:

- the truck-platoon fork is a modified build, so nothing measured on it could be quoted; and
- **the additional maps were never imported.** The official
  `AdditionalMaps_0.9.15.tar.gz` (7375946087 bytes, byte-identical in size to the GPU box's copy,
  and it does contain Town11/12/13/15) was sitting unused in that fork's `Import/`, so the box had
  only Town01-07 and Town10HD — that is **165 of the 220 Bench2Drive routes missing**, Town12's 104
  among them. Having the tarball on disk is not having the maps; `ImportAssets.sh` has to run, and
  it consumes the tarball, so feed it a copy.

**`ImportAssets.sh` prints alarming `tar` errors and exits non-zero. Ignore them.** It ends with
`Unexpected inconsistency when making directory` on directories that already exist, then
`Exiting with failure status due to previous errors`, which under `set -e` aborts whatever called
it. The import had in fact finished. The way to settle that is a file-by-file diff against the GPU
box's working install, not the exit code:

```bash
ssh autodl 'cd $DATA_DIR/third_party/carla/CARLA_0.9.15 && find . -type f | sort' > a.txt
ssh ujs@100.108.238.8 'cd /data/third_party/carla/CARLA_0.9.15 && find . -type f | sort' > t.txt
LC_ALL=C comm -23 <(LC_ALL=C sort a.txt) <(LC_ALL=C sort t.txt)    # LC_ALL matters: the two boxes collate differently
```

44712 files here against 44721 there, and the nine extras were all `__pycache__/*.pyc` that autodl
generated by running Python. Identical install.

The Bench2Drive checkout at `~/mycode/Bench2Drive` is on `main`, while the GPU box uses branch
`0.0.4`. Match the branch before comparing anything measured on the two boxes.

Still missing, and needed before a route can actually be watched here:

- a Python 3.8 client venv (`uv` is at `~/.local/bin/uv`; system Python is 3.12). Install it from
  **official PyPI through the proxy**, not from a domestic mirror: in global mode the mirror's
  traffic is routed through a Japan node too, and `mirrors.aliyun.com` simply times out.
- a way to point the leaderboard at an already-running windowed server. Bench2Drive's
  `leaderboard_evaluator.py` starts CARLA itself with `-RenderOffScreen`, which is exactly the
  thing this box exists to avoid.
- a working way to pick the map. `./CarlaUE4.sh /Game/Carla/Maps/Town12` is **silently ignored** —
  the server comes up on Town10HD and says nothing about it. Use a client `load_world`, and read
  the map back rather than trusting the launch line.

Last verified: 2026-09-22
