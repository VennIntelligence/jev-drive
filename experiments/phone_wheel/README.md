# phone_wheel: the phone as steering wheel and pedals

status: live
decisions: none
index: phone page -> Tokyo daemon -> UDP reader -> human_agent on Bench2Drive; route 24721 driven by hand: DS 100, RC 100

**Question.** Can a phone (Galaxy Z Fold 6, open square screen and folded cover screen) serve as wheel + pedals for CARLA,
with low enough latency for manual driving on the Tokyo box?

**Conclusion.** Tooling, not a result. Headless Chrome on Tokyo checks the steering math at screen angles 0 / 90 / 270
(dead zone, expo, clamp exact), two-thumb pedals and the end-to-end path to `wheel_client` (loopback rtt 0.6 ms).
On-phone latency and the physical steering sign are measured on the phone.

Human drive (below): the user drove Bench2Drive route 24721 (Town01, HardBreakRoute) in openpilot's wide model frame, DS 100,
RC 100, IS 1.00 (20 min-speed entries, unpenalised), 2026-10-13 run `1013-174600`.

**Next.** The user drives the openpilot loop's 11 routes, so the human's scores sit next to the model's on the same exam.

## How it works

- `web/index.html` (single file, no build): steering = rotation of the gravity vector in the screen plane
  (`DeviceMotionEvent.accelerationIncludingGravity` rotated by `screen.orientation.angle`, so any orientation works;
  iOS sign flipped). "Center" stores the zero; dead zone 2 deg, linear or mild expo. The main screen holds only Center,
  Settings and two sliders: "Steering range" (tilt for full lock, 20-90 deg, default 60) and "Pedal gain" (0.5-3.0,
  default 1.3, multiplies throttle and brake on the page before sending, clamped to 1). Settings (dead zone, curve, invert, pedal hold-on-release, landscape lock, fullscreen on Start) live in
  localStorage. Pedals (default "one pedal", Tesla style): one tall track at the right thumb, neutral detent at
  mid-height (+-6 %), up = throttle 0..1, down = brake 0..1, release springs to neutral (coast). Setting "two pedals":
  left brake / right throttle tracks, absolute position (bottom 0, top 1), spring back to 0 unless "hold" is set. Both
  modes: multi-touch, 36 px grab tolerance; throttle rises at most 0.8 /s (setting 0.3-3.0) and falls at once, brake is
  never limited; the track shows the limited value and the raw thumb value (`raw_throttle`). In Settings: "Stop" = handbrake
  (sends throttle 0, brake 1, `handbrake: true`), "Reverse" a flag, "Full" fullscreen. Sends `{steer, throttle, brake, raw_throttle, reverse,
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

## Human drive: a person takes the Bench2Drive exam (Tokyo)

`lib/human_agent.py` is a Bench2Drive leaderboard agent whose control is the phone (`WheelClient.state()` every tick ->
`carla.VehicleControl(steer, throttle, brake, hand_brake, reverse)`; stale phone -> steer 0, brake 1, handbrake). Its only
sensor is openpilot's wide camera exactly as the openpilot agents mount and render it (`scripts/zeroshot_rigs`, 1928 x 1208,
focal 567 px, windshield top), and the viewer shows it warped as modeld warps it for the model (512 x 256 frame, at 2x), so the
human sees the model's input. The road camera is not rendered (never displayed; config `"road": true` adds it).

- Pedals: throttle = min(1, 1.3 * t^0.6) (`throttle_gain`, `throttle_gamma`; after the first drive the linear map felt dull),
  brake = min(1, 1.0 * b^1.0) (`brake_gain`, `brake_gamma`; the MKZ brake is already sharp: 0.05 gives about -3.2 m/s^2), on
  top of the page's pedal gain. Steer: `steer_gain` 1.0, `steer_expo` 0 (the page has its own curve). Config JSON via
  `HUMAN_CONFIG=<file>`.
- Real time: the leaderboard ticks synchronously at 20 Hz sim; the agent sleeps so sim time never runs ahead of wall time
  (`realtime` 1.0). The world is held at its first tick until the first throttle press (at most 240 s, under the
  leaderboard's 300 s agent watchdog), so the route clock starts when the driver does. Measured on route 24721 (Town01, card 0
  shared with the openpilot monitor loop and its model server): unpaced 0.62 wall s per sim s, so paced runs are exactly real
  time and no sensor had to be dropped.
- Overlays use the hero's true pose (display only; control is the phone's): route command banner, benchmark route ahead in blue
  with 10/20/40/80 m dots, ego panel, phone panel (applied steer / throttle / brake bars, input age, stale), route progress and
  sim speed. A large orange "DRIVER: HUMAN (phone)" tag tells it apart from the model's viewer window.

```bash
# Tokyo; wheel daemon running (tmux wheel), phone on the page and centred
bash ~/mycode/jev-drive/experiments/phone_wheel/scripts/human_drive.sh            # the openpilot loop's 11 routes, its order
bash ~/mycode/jev-drive/experiments/phone_wheel/scripts/human_drive.sh 24721 334  # chosen routes
```

It refuses to start with less than 8 GB free on the card (`HUMAN_MIN_FREE_MB`) or when its port is taken. tmux session `human`:
`viewer` = window "human drive (phone wheel)" on DISPLAY=:0 (`scripts/human_viewer.py`, preview dir
/data/runs/human_drive/live), `drive` = `scripts/b2d_run.py` with its own off-screen CARLA on card 0 (`HUMAN_GPU`), server
index 4 = rpc 2200 / tm 8200 (`HUMAN_SERVER_INDEX`; the openpilot monitor runs use 0 and 2); the launcher checks in nvidia-smi
that the server landed on that card and stops the run otherwise (`gpu_check.txt`). Never card 1 while its PAI chain runs.
Results: `/data/runs/human_drive/<stamp>/results.md` (and `results.json`): DS, RC, IS, status and infraction counts per route,
mean over finished routes, routes never finished; per attempt `attempts/<id>/<n>/human.jsonl` (every tick: phone input,
applied control, input age, speed, route index, sleep) and `human_summary.json` (sim s, wall per sim s, unpaced, stale ticks).
`scripts/human_results.py <run>` rebuilds the table.

<!-- files:begin -->
## Files

- `wheel_server.py` (scripts): Phone steering wheel daemon
- `wheel_client.py` (lib): Non-blocking reader for the phone wheel …
- `human_agent.py` (lib): Bench2Drive agent driven by the phone, openpilot wide camera view
- `human_drive.sh` (scripts): launcher: own CARLA, tmux `human`, results table
- `human_viewer.py` (scripts): the "human drive (phone wheel)" window
- `human_results.py` (scripts): per-route DS / RC / infractions of a run
- `index.html` (web): 
- `manifest.webmanifest` (web): 

[lib/](lib/) 2 libraries · [scripts/](scripts/) 4 entry points
<!-- files:end -->
