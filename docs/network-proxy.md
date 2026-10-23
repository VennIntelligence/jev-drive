# Network proxy

Read this when a download fails or is slow on the box.
This page is about the GPU box. The Tokyo box runs Clash in *global* mode, where none of the advice
below holds — see the network section of [tokyo-box.md](tokyo-box.md).

## Default: direct

- Shells start with no proxy. Turn one on only for the command that needs it, then turn it off.
- Scripts set their own proxy and never rely on the caller's shell.
- Domestic traffic (Aliyun Drive, ModelScope, pip/uv mirrors) must go direct. A proxy only slows it down.
- Clash may keep running in the background. Only processes with `http(s)_proxy` set use it (no TUN mode).

## When a download is slow

Try in this order. Never use both in one shell.
1. `source /etc/network_turbo`: AutoDL's built-in proxy, GitHub and HuggingFace only.
   Undo: `unset http_proxy https_proxy`.
2. `proxy_on`: Clash in global mode, auto-picks a Japan/Singapore node; use a 0.1x node first. Undo: `proxy_off`.

## Picking a Clash node

Node names carry a traffic multiplier (1x, 0.1x) and sometimes a label such as "dedicated line".
Neither predicts throughput, so measure before committing a large transfer. Measured 2026-09-20
against a live download: 1x AWS Singapore 8.44 MB/s, 0.1x Tokyo-01 11.96, 0.1x US-02 11.79,
0.1x Tokyo-06 "high-speed dedicated line" 9.73. The cheapest node was also the fastest.

The multiplier is what the subscription bills, not what you transfer: on a 0.1x node, 1.2 TB of
data costs about 120 GB of quota. Check the remaining quota before a bulk transfer, because
running out also breaks Google OAuth token refresh, which only works through the proxy, and that
stops a Waymo download from resuming at all.

## Turbo is broken for the HF CDN, and a 60 s test lied to me again (2026-09-22)

Two separate findings; the second one is a warning about this page's own advice.

**1. `source /etc/network_turbo` is not slow for HuggingFace, it is failing.** Measured against one file
(`MiniMaxAI/MiniMax-H3` transformer shard 2), 60 s: **0.15 MB/s** (9 MB), with repeated
`SSL handshake timed out` and `peer closed connection without sending complete message body`.
`proxy_on` on the same file in the same minute: **25.9 MB/s**. So for HuggingFace, skip step 1 and go
straight to `proxy_on`; the "turbo first" order above still holds for GitHub.

**2. That 25.9 MB/s was a burst, and I believed it.** Sustained over 90 s, every route settles at the
same **3 MB/s**:

| how | 60 s burst | 90 s sustained |
|---|---:|---:|
| `proxy_on`, one `curl` on `resolve/main` | **25.9 MB/s** | — |
| `proxy_on`, four parallel `curl`s | — | **3 MB/s** |
| `proxy_on`, `snapshot_download` (Xet path) | — | 3 MB/s |
| `proxy_on`, `snapshot_download` + `HF_HUB_ENABLE_HF_TRANSFER=1` | — | 3 MB/s |

**So the downloader is not the bottleneck and the link really is about 3 MB/s under load.** I briefly
concluded the opposite from the 60 s number and started rewriting the download tooling around it. The
paragraph three sections down — *"A one-minute test overstates free headroom... for anything running for
hours, sample at least 90 s after the change has settled"* — already said exactly this. **Read it before
quoting a throughput number, including one you measured yourself.**

What does follow: size a budget from a 90 s sample, and do not bother swapping `snapshot_download` for
curl or `hf_transfer` to go faster, because it will not.

## Sharing the link between jobs

The box's egress tops out at about 18 MB/s, and neither route nor Clash node changes that ceiling,
only how it is shared. The link is shared per TCP stream, not per job: a job with 8 concurrent
downloads and a job with 2 do not get half each. On 2026-09-20 the NAVSIM download held 8 streams
and got 3-4 MB/s while the other two jobs took ~9 MB/s. So raise the stream count of the job that
matters, and keep the total near the ceiling instead of above it.

**A one-minute test overstates free headroom.** New streams are still ramping and the existing ones
have not backed off yet, so the burst reads as headroom. The same day, a 60 s test said 6 extra
streams gained 6.1 MB/s with no cost to the other jobs; a 90 s steady-state sample after the change
showed the total flat at 12.3 MB/s and most of the gain taken from a job we had agreed not to slow
down. For anything running for hours, sample at least 90 s after the change has settled, and measure
every competing job in the same window, not only your own.

**At busy hours the link is zero-sum, and slightly worse.** Measured the same day across a change:
one job gained 1.9 MB/s, the other lost 2.9, and the total fell from 12.6 to 11.6 MB/s. Past some
point extra streams cost aggregate throughput, so the 18 MB/s ceiling is not reachable by adding
streams. Decide who should win and say so; do not expect to find spare capacity.

**A stream is not a stream.** Per-stream yield depends on the route: 16 proxy streams each carried
less than 8 direct ones, so the direct job held nearly twice the share at half the stream count.
Compare per-stream yield by route before changing counts.

How to measure (while everything keeps running; `<my dir>` per job, one sample per job):
```bash
a=$(awk '/eth0/{print $2}' /proc/net/dev); s0=$(du -sb <my dir> | cut -f1); sleep 60
b=$(awk '/eth0/{print $2}' /proc/net/dev); s1=$(du -sb <my dir> | cut -f1)
echo "box $(( (b-a)/60000000 )) MB/s, mine $(( (s1-s0)/60000000 )) MB/s"  # repeat over 90 s+
```

Notes:
- HF downloads through either proxy fail with `CAS Client Error ... 401 Unauthorized` (Xet storage).
  Set `HF_HUB_DISABLE_XET=1`. `scripts/download_models.sh` already does.
- `proxy_on` starts Clash if needed. Also available: `clash-start`, `clash-stop`, `clash-status`.
- Clash does not survive an instance restart. Run `proxy_on` again.
- Config lives in `~/data/clash/` and contains node credentials. Never put it in the repo.
- If Clash hangs at start with "Can't find GeoSite.dat": turn on turbo, download `geosite.dat` from the
  MetaCubeX/meta-rules-dat releases, and save it as `~/data/clash/GeoSite.dat`.
- New nodes: regenerate the config locally from the subscription (kept outside the repo),
  then `scp -C` it to `~/data/clash/config.yaml`.

Last verified: 2026-09-22

## Big weights: split the files across two sources (2026-09-28)

Wan2.1-VACE-14B, 69 GB in 7 shards plus an 11 GB T5. `modelscope download` (direct, one connection per file, 4 files
at a time) ran at 1.5-2 MB/s per file, about 7 MB/s in total, so a 2.5 h ETA. Starting a second download of only the
missing files from HuggingFace in parallel (`hf download --max-workers 8 --include ...` under `source /etc/network_turbo`,
into its own `--local-dir`) did not slow the ModelScope side: sampled over 6 min, ModelScope ran at 5.4 MB/s and HF at
2.6-3.9 MB/s per file, about 12 MB/s in total. The two routes have separate bottlenecks, so the gain comes from using
both, not from more connections on one (see the table above: four parallel curls on one route stayed at 3 MB/s).
Take each file from whichever side finishes it first and check its size against the repo listing. `aria2c` is not
installed on the box.

## Clash dead for the day, hf-mirror.com direct instead (2026-10-18)

NAVSIM trainval download stalled at 05:00 CST: through Clash, TLS to huggingface.co died with `SSL unexpected eof`.
Cause: the subscription had hit its daily traffic cap. Fetching it returns only three placeholder nodes ("too much traffic
today", "limit lifted at 12:00 midnight"), while `subscription-userinfo` still shows quota left (317 of 1000 GB), so a
refreshed or recharged subscription does not help until the next midnight. The box config keeps the 71 old node entries,
which the provider rejects server side; do not overwrite it with the placeholder list. Check the fetched node count
before replacing `~/data/clash/config.yaml`.
Fallback that needs no proxy: direct `hf-mirror.com` (huggingface.co direct is unreachable). 90 s sustained, MB/s total over N
range streams of one archive: 2 -> 3.7, 4 -> 7.2, 8 -> 11.8, 16 -> 10.8 (the 16 test was probably capped by the file size).
Relaunched as `ROUTE=direct HF_ENDPOINT=https://hf-mirror.com RANGES=8 JOBS=5`: 8 MB/s into the archive dir, box 9 MB/s.
15:00 CST re-tune of the direct hf-mirror.com run (box rx counter, 90 s after ~2 min settling): the 40-stream run (JOBS=5 RANGES=8) gave ~5 MB/s;
JOBS=2 RANGES=4 (8 streams) 0.7 MB/s (mirror also slow at that moment); JOBS=4 RANGES=4 (16 streams) 4.1 MB/s, kept. The ranged_get
chunk layout (`<part>.d/<offset>`, 32 MB) does not depend on RANGES, so changing it resumes cleanly. At ~4 MB/s the ~430 GB left is ~30 h; the mirror rate varies over the day.

## hf-mirror 429 circuit breaker for the trainval camera download (2026-10-18 night)

Route decision: direct hf-mirror.com only (Clash is rate-limited for the day and out of scope). Probe at 23:20 CST, one
40 MB range of `openscene_sensor_trainval_camera_133.tgz`: direct hf-mirror.com HTTP 429 (0.5 KB/s, mirror throttling while
16 streams were already running), Clash to hf-mirror.com and to huggingface.co: connection failed, huggingface.co direct:
unreachable (20 s timeout). Box rx before the restart: ~1 MB/s total (earlier archives averaged 0.4-0.8 MB/s each).

`scripts/ranged_get.sh` now has a shared breaker (state in `_archives/state/.gate/`, all archives and chunks share it):
any chunk that does not return 206 with the full byte count (429, 5xx, stall, reset) halves the allowed number of
concurrent range requests (min 2) and sets a cool-down of 30 s doubling per consecutive failure (cap 900 s, plus jitter);
failures already in flight when it trips do not stack; 15 clean chunks in a row raise the limit by 1 up to `MAXCONN`
(default 16). Trips and recoveries are logged to `.gate/events.log`. `scripts/dl_progress.sh [interval_s] [total]`
appends `done N/183, box MB/s, archives MB/s, gate_limit` to `$DATA_DIR/runs/download/navsim/progress.log`.
Launch (the env must go through `env`, because `tmux_run.sh` does not pass the caller's environment):
`scripts/tmux_run.sh navdl env PIGZ=-p4 RANGES=4 CAMS=CAM_F0 SKIP_LIST=... ROUTE=direct HF_ENDPOINT=https://hf-mirror.com JOBS=4 MAXCONN=16 scripts/download_navsim.sh trainvalall`
(.part files and chunk dirs are reused; kill the job by Ctrl-C in the tmux window, not by pattern).
