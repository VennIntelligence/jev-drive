#!/usr/bin/env bash
# Monitor viewer: the openpilot wide model frame with overlays (preview from $V/live) and the route picker, top right of the
# 3840x2160 Tokyo screen. usage: viewer.sh   env: SDL_VIDEO_WINDOW_POS (1340,40), VIEWER_SIZE ("2480 1736")
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export SDL_VIDEO_WINDOW_POS=${SDL_VIDEO_WINDOW_POS:-1340,40}
exec $PY experiments/b2d_monitor/scripts/b2d_viewer.py --dir $V/live --window-size ${VIEWER_SIZE:-2480 1736} --routes-xml $XML
