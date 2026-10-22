#!/usr/bin/env bash
# Desktop entry of the Bench2Drive human exam panel (Tokyo): one panel at a time; the panel starts the wheel daemon and the
# viewer when they are missing. Install the icon: bash experiments/phone_wheel/scripts/human_panel.sh --install
# Web panel (default): the server (scripts/human_panel.py, lib/exam_web.py) on 127.0.0.1:8771 and <tailscale IP>:8771, one instance
# (flock held by the server), opened as a Chrome app window left of the viewer; clicking the icon again only opens a window.
#   --restart  stop the running server first (after a code update)    --tk  the old Tk panel instead
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
export DISPLAY=${DISPLAY:-:0} XAUTHORITY=${XAUTHORITY:-/run/user/1001/gdm/Xauthority} DATA_DIR=/data
export CARLA_ROOT=/data/third_party/carla/CARLA_0.9.15 BENCH2DRIVE_ROOT=/data/third_party/Bench2Drive
if [[ ${1:-} == --install ]]; then
  f=b2d-human-exam.desktop
  cat > ~/.local/share/applications/$f <<EOD
[Desktop Entry]
Type=Application
Name=B2D human exam
Comment=Bench2Drive by hand with the phone wheel: batches, resume, pause, route list
Exec=bash $REPO/experiments/phone_wheel/scripts/human_panel.sh
Icon=$REPO/experiments/phone_wheel/web/icon-192.png
Terminal=false
Categories=Utility;
EOD
  cp ~/.local/share/applications/$f ~/Desktop/$f
  chmod +x ~/Desktop/$f ~/.local/share/applications/$f
  gio set ~/Desktop/$f metadata::trusted true 2>/dev/null
  echo "installed ~/Desktop/$f and ~/.local/share/applications/$f"; exit 0
fi
mkdir -p /data/runs/human_drive
LOG=/data/runs/human_drive/panel.log PORT=${EXAM_PANEL_PORT:-8771}
PANEL="experiments/phone_wheel/scripts/human_panel.py"
cd "$REPO"
if [[ ${1:-} == --tk ]]; then
  exec 9>/data/runs/human_drive/.panel.lock
  flock -n 9 || { echo "panel already running"; exit 0; }
  exec /usr/bin/python3 $PANEL --tk >> $LOG 2>&1
fi
if [[ ${1:-} == --restart ]]; then
  pkill -f "$PANEL --port" && sleep 1
fi
exec 9>/data/runs/human_drive/.panel.lock
if flock -n 9; then        # no server yet: start one; it inherits fd 9 and so holds the lock for its whole life
  setsid /usr/bin/python3 $PANEL --port $PORT >> $LOG 2>&1 < /dev/null &
fi
exec 9>&-
for _ in $(seq 50); do curl -sf -o /dev/null http://127.0.0.1:$PORT/api/meta && break; sleep 0.2; done
if ! curl -sf -o /dev/null http://127.0.0.1:$PORT/api/meta; then     # e.g. a Tk panel (--tk) holds the lock
  msg="exam panel: no server on port $PORT (is the Tk panel still open? close it, then click again). Log: $LOG"
  echo "$msg" | tee -a $LOG; notify-send "$msg" 2>/dev/null; exit 1
fi
curl -sf -X POST -d '{}' http://127.0.0.1:$PORT/api/ensure_side > /dev/null     # wheel daemon + viewer when missing
# Chrome sizes windows in device-independent px (the desktop runs Xft.dpi 192 = scale 2): a fixed scale for the panel's own profile,
# and the window size in screen px (EXAM_PANEL_SIZE) divided by it, so it fills x 20..1320 left of the viewer (x 1340).
SCALE=${EXAM_PANEL_SCALE:-1.4} SIZE=${EXAM_PANEL_SIZE:-1300,1800}
DIP=$(awk -v s=$SCALE -v wh=$SIZE 'BEGIN { split(wh, a, ","); printf "%d,%d", a[1] / s, a[2] / s }')
mkdir -p /data/tmp/exam_panel_chrome
exec google-chrome --user-data-dir=/data/tmp/exam_panel_chrome --no-first-run --no-default-browser-check \
  --simulate-outdated-no-au='Tue, 31 Dec 2099 23:59:59 GMT' --force-device-scale-factor=$SCALE \
  --app=http://127.0.0.1:$PORT/ --window-position=20,40 --window-size=$DIP >> $LOG 2>&1
