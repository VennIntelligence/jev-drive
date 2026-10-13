#!/usr/bin/env bash
# Desktop entry of the Bench2Drive human exam panel (Tokyo): one panel at a time; the panel starts the wheel daemon and the
# viewer when they are missing. Install the icon: bash experiments/phone_wheel/scripts/human_panel.sh --install
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
exec 9>/data/runs/human_drive/.panel.lock
flock -n 9 || { echo "panel already running"; exit 0; }
cd "$REPO" && exec /usr/bin/python3 experiments/phone_wheel/scripts/human_panel.py >> /data/runs/human_drive/panel.log 2>&1
