#!/usr/bin/env bash
# Pull AlpaSim runtime files GPU box -> Tokyo box, restartable, with a stall watchdog that rotates the Clash GLOBAL node.
# Runs ON the Tokyo box (tmux session `dl`), mirrors the GPU box's relative paths under /data (= DATA_DIR there), appends every
# item to $SETUP/MANIFEST.md and touches $SETUP/DONE_<item> (ERROR_<item> + ERROR on failure).
#   tokyo_pull.sh ckpt <tag>...   adapter checkpoint(s) runs/op_parity/runs/<tag>/ckpt-final.pt, sha256-checked against the box
#   tokyo_pull.sh base            openpilot model file models/openpilot/cinque.ort.onnx
#   tokyo_pull.sh lists           the 48-scene list
#   tokyo_pull.sh scenes [list]   per-scene assets of a scene list (default: the 48-scene list), 8 parallel rsyncs, checksum verify
#   tokyo_pull.sh hf              trajdata cache nuplan_test + wizard configs from Hugging Face (tarball, sha256 from the tree API)
#   tokyo_pull.sh hfassets [N]    part001 assets tarball from Hugging Face in N (16) parallel ranges into _dl/ (the box link is capped
#                                 at ~3.4 MB/s whatever the stream count; HF scales to ~8 MB/s at 8 streams), then sha256 + extract only the
#                                 listed scenes without overwriting; `scenes` afterwards repairs / verifies against the box
#   tokyo_pull.sh ref             native reference results (small files, staged by hand in BOX_REF=/tmp/bn_native_ref on the box)
# Some Clash nodes close the box's ssh port; box-bound commands rotate to one that passes (DIRECT as last resort, measured to work) and put
# the selector back when they end. New tag later: from the Mac `ssh-add ~/.ssh/id_ed25519; ssh -A ujs@<tokyo> 'bash ~/mycode/jev-drive/experiments/alpasim/scripts/tokyo_pull.sh ckpt <tag>'`.
# Needs: box login via the forwarded agent (SSH_AUTH_SOCK, default /tmp/bn_agent.sock), $SETUP/box.env or the env (BOX_HOST, BOX_PORT, BOX_USER;
# not in git), Clash secret file as in docs/tokyo-box.md. Never prints the secret.
set -uo pipefail
D=${DATA_DIR:-/data}; SETUP=$D/runs/alpasim/tokyo_setup; mkdir -p "$SETUP"; [[ -f $SETUP/box.env ]] && source "$SETUP/box.env"   # or export BOX_HOST BOX_PORT BOX_USER
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/tmp/bn_agent.sock}
BOXD=${BOX_DATA:-/root/autodl-tmp/ujs}                  # DATA_DIR on the GPU box (~/data resolves to it)
SSHO=(-o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -c aes128-gcm@openssh.com -p "$BOX_PORT")
BOX=$BOX_USER@$BOX_HOST
RSH="ssh ${SSHO[*]}"; RPATH="nice -n 19 ionice -c3 rsync"
NUPLAN=datasets/alpasim_nuplan; HFREPO=https://huggingface.co/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track/resolve/main
HFAPI=https://huggingface.co/api/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track/tree/main
log() { echo "$(date '+%F %T') $*" | tee -a "$SETUP/log.txt"; }
rbox() { ssh "${SSHO[@]}" "$BOX" "$@"; }

# ---- Clash node rotation ----
SECRET_FILE=~/.local/share/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml
api() {  # METHOD PATH [json body]
  local s; s=$(grep -m1 '^secret:' "$SECRET_FILE" | sed "s/secret: *//;s/[\"']//g")
  curl -s --noproxy '*' -X "$1" -H "Authorization: Bearer $s" -H 'Content-Type: application/json' ${3:+-d "$3"} "http://127.0.0.1:9097$2"; }
reach() {  # the four hosts the image agent's pull / build needs
  local u c; for u in https://huggingface.co https://nvcr.io/v2/ https://registry-1.docker.io/v2/ https://github.com; do
    c=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$u"); [[ $c != 000 ]] || return 1; done; }
rotate() {  # next real node (not a group, not DIRECT/REJECT) that passes the checks; box-bound commands need the box ssh port open, others the four hosts
  local cur nodes n i k; cur=$(api GET /proxies/GLOBAL | python3 -c 'import json,sys;print(json.load(sys.stdin)["now"])')
  mapfile -t nodes < <(python3 - "$(api GET /proxies)" <<'PY'
import json, sys
P = json.loads(sys.argv[1])["proxies"]
bad = {"Selector", "URLTest", "Fallback", "LoadBalance", "Direct", "Reject", "RejectDrop", "Pass", "Compatible"}
print("\n".join(n for n in P["GLOBAL"]["all"] if n in P and P[n]["type"] not in bad and n != "DIRECT"))
PY
)
  # $SETUP/nodes.pref (one node per line, best first, from tokyo_nodebench.sh) is tried first when present
  [[ -s $SETUP/nodes.pref ]] && mapfile -t nodes < <(cat "$SETUP/nodes.pref" <(printf '%s\n' "${nodes[@]}") | awk '!seen[$0]++')
  [[ -n ${NEED_BOX:-} ]] && nodes+=(DIRECT)   # last resort for the box port: DIRECT works (measured); keep such runs short, the selector returns afterwards
  for i in "${!nodes[@]}"; do [[ ${nodes[i]} == "$cur" ]] && break; done
  for k in $(seq 1 ${#nodes[@]}); do
    n=${nodes[$(( (i + k) % ${#nodes[@]} ))]}
    api PUT /proxies/GLOBAL "$(python3 -c 'import json,sys;print(json.dumps({"name":sys.argv[1]}))' "$n")" >/dev/null
    # some nodes close the box's ssh port ("Connection closed by 198.18.x.x"); the selector returns to its start node when the command ends
    if [[ -n ${NEED_BOX:-} ]]; then ssh "${SSHO[@]}" -o ConnectTimeout=10 "$BOX" true 2>/dev/null && { log "node -> $n (box ssh ok)"; [[ $n == DIRECT ]] && echo "$(date '+%F %T') selector DIRECT for a box transfer (tokyo_pull.sh), returns when it ends" >> "$SETUP/STATUS"; return 0; }
    elif reach; then log "node -> $n"; return 0; fi
  done; log "no usable node"; return 1; }

# run a command until it succeeds; rotate the node after each failure (a stalled rsync dies on --timeout)
retry() { local n=0; until "$@"; do n=$((n+1)); log "retry $n: $1 ${2:-}"; (( n > 40 )) && return 1; (( n % 2 == 0 )) || [[ -n ${NEED_BOX:-} ]] && rotate; sleep 3; done; }

done_item() { touch "$SETUP/DONE_$1"; rm -f "$SETUP/ERROR_$1"; }
fail_item() { echo "$2" > "$SETUP/ERROR_$1"; echo "$1: $2" >> "$SETUP/ERROR"; log "ERROR $1: $2"; }
manifest() {  # item path source seconds   (size + sha256 of a file, or size + file count of a dir)
  local p=$D/$2 size sha
  if [[ -d $p ]]; then size=$(du -sb --apparent-size "$p" | cut -f1); sha="$(find "$p" -type f | wc -l) files"
  else size=$(stat -c%s "$p"); sha=$(sha256sum "$p" | cut -d' ' -f1); fi
  [[ -f $SETUP/MANIFEST.md ]] || printf '| item | path under /data | bytes | sha256 / files | source | seconds | date |\n|---|---|---|---|---|---|---|\n' > "$SETUP/MANIFEST.md"
  printf '| %s | %s | %s | %s | %s | %s | %s |\n' "$1" "$2" "$size" "$sha" "$3" "$4" "$(date '+%F %T')" >> "$SETUP/MANIFEST.md"; }

pull_file() {  # relative path under the data dir; sha256 compared with the box
  local rel=$1 t0=$SECONDS want got
  mkdir -p "$D/$(dirname "$rel")"
  retry rsync -a --partial --timeout=40 --rsync-path="$RPATH" -e "$RSH" "$BOX:$BOXD/$rel" "$D/$rel" || return 1
  want=$(rbox "sha256sum $BOXD/$rel" | cut -d' ' -f1); got=$(sha256sum "$D/$rel" | cut -d' ' -f1)
  [[ $want == "$got" ]] || { rm -f "$D/$rel"; return 2; }
  manifest "$3" "$rel" "GPU box via ssh" $((SECONDS - t0)); }

cmd=${1:-}; shift || true
if [[ $cmd != hf && $cmd != hfassets && $cmd != rotate ]]; then   # box-bound: rotation requires box reachability, and the selector goes back afterwards
  export NEED_BOX=1; START_NODE=$(api GET /proxies/GLOBAL | python3 -c 'import json,sys;print(json.load(sys.stdin)["now"])')
  trap 'api PUT /proxies/GLOBAL "$(python3 -c "import json,sys;print(json.dumps({\"name\":sys.argv[1]}))" "$START_NODE")" >/dev/null' EXIT
fi
case $cmd in
  ckpt) for tag in "$@"; do
          pull_file "runs/op_parity/runs/$tag/ckpt-final.pt" x "ckpt_$tag" && done_item "ckpt_$tag" || fail_item "ckpt_$tag" "pull or sha256 mismatch"; done ;;
  base) pull_file models/openpilot/cinque.ort.onnx x openpilot_cinque_ort && done_item openpilot_cinque_ort || fail_item openpilot_cinque_ort "pull or sha256 mismatch" ;;
  lists) mkdir -p "$D/runs/alpasim"; t0=$SECONDS
        retry rsync -a --timeout=40 -e "$RSH" "$BOX:$BOXD/runs/alpasim/scenes_navtest_full_part001_48.txt" "$D/runs/alpasim/" \
          && manifest scene_list runs/alpasim/scenes_navtest_full_part001_48.txt "GPU box via ssh" $((SECONDS - t0)) && done_item scene_list ;;
  scenes) list=${1:-$D/runs/alpasim/scenes_navtest_full_part001_48.txt}; t0=$SECONDS; root=$NUPLAN/navtest/assets; mkdir -p "$D/$root"
        one() { retry rsync -a $CK --partial --timeout=40 --rsync-path="$RPATH" -e "$RSH" "$BOX:$BOXD/$root/$1/" "$D/$root/$1/"; }
        export CK=; export -f one retry rotate reach api log; export D BOX BOXD root RSH RPATH SETUP SECRET_FILE
        xargs -a "$list" -P 8 -I{} bash -c 'one {}' || { fail_item scenes48 "rsync failed"; exit 1; }
        # verify: a second pass with checksums on both sides must transfer nothing
        chk() { xargs -a "$list" -P 8 -I{} bash -c 'rsync -anc --out-format=%n --rsync-path="$RPATH" -e "$RSH" "$BOX:$BOXD/$root/{}/" "$D/$root/{}/"' | wc -l; }
        bad=$(chk)
        if (( bad > 0 )); then log "$bad files differ (partials from an interrupted pass or the HF copy): repairing with checksums"
          CK=-c xargs -a "$list" -P 8 -I{} bash -c 'one {}'; bad=$(chk); fi
        if (( bad == 0 )); then mkdir -p "$D/$root"; manifest scenes48 "$root" "GPU box via ssh (rsync -c verified, $(wc -l < "$list") scenes)" $((SECONDS - t0)); done_item scenes48
        else fail_item scenes48 "$bad files differ after checksum pass"; fi ;;
  hf) t0=$SECONDS; mkdir -p "$D/$NUPLAN/_dl" "$D/$NUPLAN/navtest"
      for rel in MTGS_asset/navtest/configs.tar.gz trajdata_cache/nuplan_test.tar.gz; do
        f=$D/$NUPLAN/_dl/$(basename "$rel"); item=$(basename "$rel" .tar.gz)
        [[ -e $D/$NUPLAN/.done.$item ]] && continue
        until curl -sfL -C - --speed-limit 300000 --speed-time 15 -o "$f" "$HFREPO/$rel"; do log "hf stall/err $rel at $(stat -c%s "$f" 2>/dev/null)"; rotate; sleep 2; done
        want=$(curl -sf -m 60 "$HFAPI/$(dirname "$rel")" | python3 -c "import json,sys;print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']==sys.argv[1]))" "$rel")
        got=$(sha256sum "$f" | cut -d' ' -f1)
        [[ $want == "$got" ]] || { rm -f "$f"; fail_item "hf_$item" "sha256 mismatch $got != $want"; exit 1; }
        pigz -dc "$f" | tar -x -C "$D/$NUPLAN" && touch "$D/$NUPLAN/.done.$item"
        # both tarballs are rooted at the dataset dir: navtest/configs/ and nuplan_test/
        manifest "hf_$item" "$NUPLAN/$([[ $item == configs ]] && echo navtest/configs || echo nuplan_test)" "Hugging Face, tarball sha256 $got" $((SECONDS - t0)) && done_item "hf_$item"
      done ;;
  ref) t0=$SECONDS; mkdir -p "$D/runs/alpasim/native_ref"
      retry rsync -a --timeout=40 -e "$RSH" "$BOX:${BOX_REF:-/tmp/bn_native_ref}/" "$D/runs/alpasim/native_ref/" \
        && manifest native_ref runs/alpasim/native_ref "GPU box via ssh" $((SECONDS - t0)) && done_item native_ref ;;
  hfassets) N=${1:-16}; t0=$SECONDS; rel=MTGS_asset/navtest/assets/part001.tar.gz; dl=$D/$NUPLAN/_dl; mkdir -p "$dl"
      list=$D/runs/alpasim/scenes_navtest_full_part001_48.txt
      size=$(curl -sIL -m 30 "$HFREPO/$rel" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}'); chunk=$(( (size + N - 1) / N ))
      chunkdl() {  # one range, resumed from the bytes on disk; a stalled connection (< 300 kB/s for 20 s) is cut and the node rotated
        local i=$1 a=$(( $1 * chunk )) f=$dl/part001.$1 b code; b=$(( a + chunk - 1 )); (( b >= size )) && b=$(( size - 1 ))
        until (( $(stat -c%s "$f" 2>/dev/null || echo 0) == b - a + 1 )); do
          code=$(curl -sL -o "$f.new" -w '%{http_code}' --speed-limit 300000 --speed-time 20 --max-time 300 -r $(( a + $(stat -c%s "$f" 2>/dev/null || echo 0) ))-$b "$HFREPO/$rel")
          [[ $code == 206 ]] && head -c $(( b - a + 1 - $(stat -c%s "$f" 2>/dev/null || echo 0) )) "$f.new" >> "$f"; rm -f "$f.new"; sleep 1; done; }
      export -f chunkdl; export dl chunk size HFREPO rel
      # the watchdog: if all chunks together made no progress in 3 min, rotate the node
      ( last=0; while sleep 180; do cur=$(cat "$dl"/part001.* 2>/dev/null | wc -c); (( cur == last )) && rotate; last=$cur; [[ -e $dl/.hf_stop ]] && break; done ) &
      wd=$!; seq 0 $((N - 1)) | xargs -P "$N" -I{} bash -c 'chunkdl {}'; touch "$dl/.hf_stop"; kill $wd 2>/dev/null
      want=$(curl -sf -m 60 "$HFAPI/$(dirname "$rel")" | python3 -c "import json,sys;print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']==sys.argv[1]))" "$rel")
      got=$(for i in $(seq 0 $((N - 1))); do cat "$dl/part001.$i"; done | sha256sum | cut -d' ' -f1)
      [[ $want == "$got" ]] || { fail_item hf_part001 "sha256 mismatch $got != $want"; exit 1; }
      log "part001 tarball ok in $((SECONDS - t0)) s, extracting listed scenes"
      for i in $(seq 0 $((N - 1))); do cat "$dl/part001.$i"; done | pigz -dc | tar -x --skip-old-files -C "$D/$NUPLAN" $(sed 's#^#navtest/assets/#' "$list" | tr '\n' ' ')
      rm -f "$dl"/part001.*; echo "$((SECONDS - t0))" > "$SETUP/hf_part001.seconds"; log "hfassets extracted; run scenes to verify against the box" ;;
  rotate) rotate ;;
  *) sed -n 2,13p "$0"; exit 2 ;;
esac
