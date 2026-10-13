# phone_wheel: the phone as steering wheel and pedals

status: live
decisions: none
index: phone page -> Tokyo daemon -> UDP reader; loopback rtt < 1 ms, phone rtt pending

**Question.** Can a phone (Galaxy Z Fold 6, open square screen and folded cover screen) serve as wheel + pedals for CARLA,
with low enough latency for manual driving on the Tokyo box?

**Conclusion.** Tooling, not a result. Headless Chrome on Tokyo checks the steering math at screen angles 0 / 90 / 270
(dead zone, expo, clamp exact), two-thumb pedals and the end-to-end path to `wheel_client` (loopback rtt 0.6 ms).
On-phone latency and the physical steering sign are measured on the phone.

**Next.** A Bench2Drive HumanAgent that reads `wheel_client` (someone else); the phone rtt and sign check.

## How it works

- `web/index.html` (single file, no build): steering = rotation of the gravity vector in the screen plane
  (`DeviceMotionEvent.accelerationIncludingGravity` rotated by `screen.orientation.angle`, so any orientation works;
  iOS sign flipped). "Center" stores the zero; dead zone 2 deg, +-90 deg = steer +-1, linear or mild expo; settings
  (lock angle, dead zone, curve, invert, pedal hold-on-release, landscape lock, fullscreen on Start) live in
  localStorage. Pedals (default "one pedal", Tesla style): one tall track at the right thumb, neutral detent at
  mid-height (+-6 %), up = throttle 0..1, down = brake 0..1, release springs to neutral (coast). Setting "two pedals":
  left brake / right throttle tracks, absolute position (bottom 0, top 1), spring back to 0 unless "hold" is set. Both
  modes: multi-touch, 36 px grab tolerance; throttle rises at most 0.8 /s (setting 0.3-3.0) and falls at once, brake is
  never limited; the track shows the limited value and the raw thumb value (`raw_throttle`). "Stop" = handbrake (sends
  throttle 0, brake 1, `handbrake: true`), "Reverse" a flag. Sends `{steer, throttle, brake, raw_throttle, reverse,
  handbrake, ts, seq, rtt}` at 60 Hz over a WebSocket; shows rtt, send rate, daemon rate, drops and the number of local consumers.
  A viewport aspect-class change or a > 1.4x area jump (fold / unfold / rotate) asks for a re-center. Layouts:
  square (open) = tracks at both edges, big wheel middle; wide (cover, landscape) = same, narrower tracks; portrait =
  wheel above, tracks side by side below. "Full" = `requestFullscreen` (optional landscape lock); `manifest.webmanifest`
  (`display: fullscreen`) + a pass-through `sw.js` make "Add to Home screen" / "Install app" launch it with no browser UI.
- `scripts/wheel_server.py` (aiohttp): serves the page and `/ws` on 127.0.0.1:8765 (optional plain-http `--lan-port`),
  keeps the latest frame, echoes `{ts}`, pushes stats every 5 s, logs connections / rate / drops / rtt, and fans every
  frame out as one JSON datagram to each UDP subscriber of 127.0.0.1:47800.
- `lib/wheel_client.py` (stdlib, Python 3.8): `WheelClient().latest() -> (steer, throttle, brake, age_s)`, non-blocking;
  `state()` adds reverse / handbrake / stale. It subscribes by sending a datagram every 0.5 s, so any number of
  consumers can read at once. No frame for 0.3 s -> stale: steer 0, throttle 0, brake 1, handbrake on.
  Steer +1 = right.

## Run (Tokyo)

```bash
# once: env (pip goes through Clash)
python3 -m venv /data/envs/wheel && /data/envs/wheel/bin/pip install aiohttp
# once, needs the sudo password: let user ujs drive tailscale serve / cert
sudo tailscale set --operator=ujs

cd ~/mycode/jev-drive
mkdir -p /data/tmp/wheel && tmux new-session -d -s wheel -n server \
  "/data/envs/wheel/bin/python experiments/phone_wheel/scripts/wheel_server.py --lan-port 8766 2>&1 | tee -a /data/tmp/wheel/log.txt"
tailscale serve --bg http://127.0.0.1:8765      # https://tokyo-devbox.tailcab48d.ts.net/ ; persists across reboots
python experiments/phone_wheel/lib/wheel_client.py   # 10 Hz console readout (any Python >= 3.8)
```

Phone: open `https://tokyo-devbox.tailcab48d.ts.net/` in Chrome (Tailscale app on), tap Start, hold straight, tap Center.
Fallback without HTTPS: `http://100.108.238.8:8766/` plus chrome://flags "Insecure origins treated as secure" = that
origin, otherwise Chrome blocks the motion sensors. Undo: `tailscale serve reset`, `tmux kill-session -t wheel`.
If the steering sign is mirrored on a device, tick "Invert steering" in Settings (a 90 deg axis error is absorbed by Center).

<!-- files:begin -->
## Files

- `wheel_server.py` (scripts): Phone steering wheel daemon
- `wheel_client.py` (lib): Non-blocking reader for the phone wheel …
- `index.html` (web): 
- `manifest.webmanifest` (web): 

[lib/](lib/) 1 library · [scripts/](scripts/) 1 entry points
<!-- files:end -->
