#!/usr/bin/env bash
# Per-host throughput of Clash GLOBAL nodes on the Tokyo box (global mode, so the selector decides every route). Runs ON Tokyo.
#   tokyo_nodebench.sh screen [secs]            every geographic node + DIRECT, torch wheel (download-r2.pytorch.org) and a pythonhosted wheel, secs each (default 6)
#   tokyo_nodebench.sh full "<node>" ... [secs] those nodes against every build host, secs each (default 25), a real ranged GET of up to 300 MB
# Output: TSV rows `node host MB/s` on stdout and in $OUT. Leaves the selector on the node it started with. Secret read from the Clash file, never printed.
set -uo pipefail
OUT=${OUT:-/data/runs/alpasim/tokyo_setup/nodebench.tsv}
CFG=~/.local/share/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml
sec=$(grep -m1 '^secret:' "$CFG" | sed "s/secret: *//;s/[\"']//g")
api() { curl -s --noproxy '*' -X "$1" -H "Authorization: Bearer $sec" -H 'Content-Type: application/json' ${3:+-d "$3"} "http://127.0.0.1:9097$2"; }
now() { api GET /proxies/GLOBAL | python3 -c 'import json,sys;print(json.load(sys.stdin)["now"])'; }
setn() { api PUT /proxies/GLOBAL "$(python3 -c 'import json,sys;print(json.dumps({"name":sys.argv[1]}))' "$1")" >/dev/null; sleep 1; }
declare -A URL=(
  [torch_r2]='https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
  [torch_dl]='https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
  [pythonhosted]='https://files.pythonhosted.org/packages/65/e4/c5a205d48ff00ed8b27882bb45d338d8138e976c76385f328a326bbfaeda/nvidia_cudnn_cu12-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl'
  [pypi_nvidia]='https://pypi.nvidia.com/nvidia-cudnn-cu12/nvidia_cudnn_cu12-9.9.0.52-py3-none-manylinux_2_27_x86_64.whl'
  [github_codeload]='https://codeload.github.com/pytorch/pytorch/tar.gz/refs/tags/v2.5.0'
  [github_release]='https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.12.7+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz'
  [huggingface]='https://huggingface.co/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track/resolve/main/MTGS_asset/navtest/assets/part001.tar.gz'
)
probe() {  # node host secs -> MB/s (bytes received / elapsed, ranged GET capped at 300 MB)
  local r; r=$(curl -sL -r 0-299999999 -o /dev/null -m "$3" -w '%{size_download} %{time_total}' "${URL[$2]}" 2>/dev/null)
  python3 -c 'import sys;b,t=map(float,sys.argv[1:3]);print("%.2f"%(b/1e6/max(t,0.001)))' ${r:-0 1}; }
start=$(now); trap 'setn "$start"' EXIT
mode=${1:-}; shift || true
case $mode in
  screen) secs=${1:-6}
    mapfile -t nodes < <(api GET /proxies/GLOBAL | python3 -c 'import json,sys;print("\n".join(n for n in json.load(sys.stdin)["all"] if n=="DIRECT" or any(k in n for k in ("Hong Kong","Macao","Taiwan","Singapore","Malaysia","Thailand","Philippines","Vietnam","Indonesia","Australia","New Zealand","Japan","Korea","United States","Canada","Germany","United Kingdom","France","Netherlands"))))')
    for n in "${nodes[@]}"; do setn "$n"; echo -e "$n\ttorch_r2\t$(probe "$n" torch_r2 "$secs")\tpythonhosted\t$(probe "$n" pythonhosted "$secs")" | tee -a "$OUT"; done ;;
  full) secs=25; args=("$@"); [[ ${args[-1]} =~ ^[0-9]+$ ]] && { secs=${args[-1]}; unset 'args[-1]'; }
    for n in "${args[@]}"; do setn "$n"; for h in "${!URL[@]}"; do echo -e "$n\t$h\t$(probe "$n" "$h" "$secs")" | tee -a "$OUT"; done; done ;;
  *) sed -n 2,6p "$0"; exit 2 ;;
esac
