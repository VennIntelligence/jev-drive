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
2. `proxy_on`: Clash in global mode, auto-picks a Japan/Singapore node. Undo: `proxy_off`.

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
