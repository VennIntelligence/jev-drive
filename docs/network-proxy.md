# Network proxy

Read this when a download fails or is slow on the box.

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

Last verified: 2026-09-19
