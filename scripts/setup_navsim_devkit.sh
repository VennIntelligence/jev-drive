#!/usr/bin/env bash
# Install the official NAVSIM devkits (third-party, run as shipped) into two venvs on the box:
#   $DATA_DIR/envs/navsim2  navsim main @ 0a380a9 (v2.2 + fixes): EPDMS, navtest one-stage and navhard two-stage
#   $DATA_DIR/envs/navsim1  navsim v1.1: PDMS (NAVSIM v1 navtest leaderboard)
# Both share nuplan-devkit v1.2 (the tag navsim's requirements.txt pins). Sources are fetched as codeload
# tarballs into $DATA_DIR/third_party (git clone over this link stalls): navsim/, navsim-v1.1/, nuplan-devkit/.
# Python 3.10 (navsim asks for 3.9; numpy 1.23.4 / torch 2.0.1 pins also hold on 3.10). Packages from the pip mirror.
# Usage (on the box): scripts/setup_navsim_devkit.sh [navsim2|navsim1 ...]   (default: both)
set -euo pipefail
tp=$DATA_DIR/third_party
export UV_INDEX_URL=${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}  # uv ignores pip.conf; pypi.org times out here
targets=("$@")
[[ $# -eq 0 ]] && targets=(navsim2 navsim1)

for t in "${targets[@]}"; do
  src=$tp/$([[ $t == navsim2 ]] && echo navsim || echo navsim-v1.1)
  env=$DATA_DIR/envs/$t
  echo "== $t: $src -> $env"
  [[ -x $env/bin/python ]] || uv venv -q --python 3.10 "$env"
  # navsim's requirements minus the nuplan git pin (installed from the local tarball below)
  grep -v '^nuplan-devkit' "$src/requirements.txt" > "$env/requirements.navsim.txt"
  VIRTUAL_ENV=$env uv pip install -q -r "$env/requirements.navsim.txt"
  VIRTUAL_ENV=$env uv pip install -q --no-deps -e "$tp/nuplan-devkit"
  VIRTUAL_ENV=$env uv pip install -q --no-deps -e "$src"
  "$env/bin/python" -c "import navsim, nuplan, torch; print('$t ok', navsim.__file__, torch.__version__)"
done
