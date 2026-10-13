#!/usr/bin/env bash
# Desktop entry of the AlpaSim human exam panel (Tokyo): one panel at a time; the panel starts the wheel daemon and the AlpaSim
# viewer when they are missing. Install the icon: bash experiments/alpasim_human/scripts/human_panel.sh --install
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
export DISPLAY=${DISPLAY:-:0} XAUTHORITY=${XAUTHORITY:-/run/user/1001/gdm/Xauthority} DATA_DIR=/data
if [[ ${1:-} == --install ]]; then
  f=alpasim-human-exam.desktop
  cat > ~/.local/share/applications/$f <<EOD
[Desktop Entry]
Type=Application
Name=AlpaSim human exam
Comment=AlpaSim scenes by hand with the phone wheel: PAI / nuPlan sets, resume, pause, scene list, scores next to the model
Exec=bash $REPO/experiments/alpasim_human/scripts/human_panel.sh
Icon=$REPO/experiments/phone_wheel/web/icon-192.png
Terminal=false
Categories=Utility;
EOD
  cp ~/.local/share/applications/$f ~/Desktop/$f
  chmod +x ~/Desktop/$f ~/.local/share/applications/$f
  gio set ~/Desktop/$f metadata::trusted true 2>/dev/null
  echo "installed ~/Desktop/$f and ~/.local/share/applications/$f"; exit 0
fi
mkdir -p /data/runs/alpasim_human
exec 9>/data/runs/alpasim_human/.panel.lock
flock -n 9 || { echo "panel already running"; exit 0; }
cd "$REPO" && exec /usr/bin/python3 experiments/alpasim_human/scripts/human_panel.py >> /data/runs/alpasim_human/panel.log 2>&1
