# Sourced helper: parallel ranged download that resumes and never writes the same bytes twice (the fixed pattern of fetch_data.sh).
#   ranged_get <url> <dest> <size> <n ranges> [auth command printing one "Header: value" line, run before every connection]
# Env: RG_PROXY (curl -x), RG_CONN_S (cut and resume a connection after this many seconds, default 120), RG_STOP_FREE_GB (abort below this
# many free GB on the dest filesystem, default 400). Range files <dest>.<i> hold the bytes so far; a range file only grows on HTTP 206,
# a body of any other status goes to a side file, so an error page can never be appended as data. One writer per dest (flock).
ranged_get() {
  local url=$1 dest=$2 size=$3 n=$4 auth=${5:-} chunk i pids=() lk rc=0
  [[ -s $dest ]] && return 0
  exec {lk}>"$dest.lock"; flock -n "$lk" || { echo "$dest is being fetched by another run" >&2; return 1; }
  chunk=$(( (size + n - 1) / n ))
  for i in $(seq 0 $((n - 1))); do
    (
      a=$((i * chunk)); b=$((a + chunk - 1)); ((b >= size)) && b=$((size - 1)); f=$dest.$i; want=$((b - a + 1)); tries=0
      sz() { stat -c%s "$1" 2>/dev/null || echo 0; }
      until (( $(sz "$f") == want )); do
        (( $(df -B1G --output=avail "$(dirname "$dest")" | tail -1) > ${RG_STOP_FREE_GB:-400} )) || { echo "free disk low" >&2; exit 1; }
        (( ++tries > 3000 )) && exit 1
        h=(); [[ -n $auth ]] && h=(-H "$(eval "$auth")")
        code=$(curl -sL ${RG_PROXY:+-x "$RG_PROXY"} -o "$f.new" -w '%{http_code}' --speed-limit 20000 --speed-time 40 --max-time "${RG_CONN_S:-120}" \
          "${h[@]}" -r $((a + $(sz "$f")))-$b "$url" || true)
        [[ $code == 206 ]] && head -c $((want - $(sz "$f"))) "$f.new" >> "$f" || sleep 2
        rm -f "$f.new"
      done
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || return 1
  for i in $(seq 0 $((n - 1))); do cat "$dest.$i"; done > "$dest.tmp" && [[ $(stat -c%s "$dest.tmp") == "$size" ]] || { echo "size mismatch $dest" >&2; rm -f "$dest.tmp"; return 1; }
  mv "$dest.tmp" "$dest"; rm -f "$dest".[0-9]*
}
