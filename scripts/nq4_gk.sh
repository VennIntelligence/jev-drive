#!/usr/bin/env bash
# Night queue 4: the G + K + X closed-loop chain (todos/2026-09-26-night-queue-4.md, sections G / K / X, the [F] entries).
# One chain script, started once in tmux jev:nq4-gk; it waits for its gate and then runs unattended.
#
#   scripts/nq4_gk.sh chain        gate (runs/nq3/b/DONE, lane B's CARLA servers gone) -> queue by priority -> DONE
#   scripts/nq4_gk.sh step <cand> <variant> <seeds> <routeset> [est_h]    one queue step (resumable), for smoke / debug
#   scripts/nq4_gk.sh report       the tables (runs/nq4/gk/results/{g,k,x}/)
#   scripts/nq4_gk.sh plan         print the queue with its current worker-hour estimate, apply no cut
#
# Step = examinee x world variant x TM seeds x route set. The seeds of a step run side by side, seed i on the GPUs
# i, i + n_seeds, ... (one b2d_run runner per GPU, $WORKERS CARLA servers each, the runners of a seed share its out dir).
#   examinees  pdm (PDM-Lite, SimLingo's tree: it plans on the registry), tfv6 / bridgedrive / simlingo / blue (author
#              executors, scripts/nq3_b_cl10.sh as is), cinque (openpilot Cinque native plan -> P7, lane B's CL2),
#              mc / q2 (our heads -> P7, cross-fitted: each route driven by the fold that never saw it), x (Q2 mode head ->
#              geometric path -> P7), k0..k3 (K ladder, K's recipe, scripts/nq4_k_recipes.sh when it exists),
#              k3seen (K3 driven by the fold that DID see the route: G's positive control)
#   variants   orig ghost shift swap (G, runs/nq4/gk/g_routes.xml) | k (the official 220, K)
#   routesets  g (the 80 G routes / 50 for swap), gob (the 40 obstacle routes), grec (G routes recorded in K's
#              training data), k220 (all 220), krec (the 220 routes recorded in K's training data)
# Budget: after every step the whole queue is re-estimated from measured worker-minutes per route (prior until an
# examinee has >= 5 finished routes). Projected total > $BUDGET_WH worker-hours (600) -> the todo's cut: first drop every
# swap step, then keep shift only for tfv6 / bridgedrive / blue / mc. Each step stops at twice its estimate (ERROR).
# Resources: GPUs 0-5 x <= 6 servers, cores 60-149. Server index i binds RPC 2000 + 50 i (+1, +2) and TM 8000 + 50 i, i.e.
# the RPC port of index i + 120, so a block must keep i, i + 120 and i - 120 clear of every other lane's indices (lane B
# 300-479, lane A 600-689, K 170-179): GPU g uses [60 + 18 g, 78 + 18 g), 60-167 in all (i + 120 in 180-287, i - 120 <= 47).
# Before every step each GPU's block is checked against the listening sockets (/proc/net/tcp): fewer than WORKERS + 2
# clean indices -> wait (2 min polls, 30 min at most), then ERROR.
# Only processes whose PIDs this script recorded are ever killed. Hand-offs: runs/nq4/gk/{STATUS.md,ERROR,DONE,events.jsonl}.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
REPO=$(pwd)
G=$DATA_DIR/runs/nq4/gk
NQ3=$DATA_DIR/runs/nq3
K=$DATA_DIR/runs/nq4/k
mkdir -p "$G/srv" "$G/cfg" "$G/arms" "$G/arms_k" "$G/steps"
GPUS=${GPUS:-0 1 2 3 4 5}
WORKERS=${WORKERS:-6}
CPUS=${CPUS:-60-149}
SIDX0=${SIDX0:-60}          # GPU g: server indices [60 + 18 g, 60 + 18 g + 18), i.e. 60-167 on GPUs 0-5 ([F] entry)
SPAN=${SPAN:-18}
BUDGET_WH=${BUDGET_WH:-600}
CUR_OUTS=""
export B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000} B2D_NQ4_TRACE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
SIM=$DATA_DIR/third_party/simlingo
XML_K=$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml
XML_G=$G/g_routes.xml
P7=$REPO/todos/2026-09-23-tfv6-controller/controller-eval/P7.json
SPLIT=$K/route_split.json
PY_CARLA=$DATA_DIR/envs/carla/bin/python PY_TCP=$DATA_DIR/envs/b2d-tcp/bin/python
PY_SCOUT=$DATA_DIR/envs/scout-tfv6/bin/python PY_OP=$DATA_DIR/envs/openpilot/bin/python
PY_VENV=$REPO/.venv/bin/python PY_SL=$DATA_DIR/envs/simlingo/bin/python
# worker-minutes per route before an examinee is measured (lane B's CL0 profiling: CL2 7.4, CL3 9.8, CL4 22, CL10 14.7)
declare -A PRIOR=([pdm]=4 [cinque]=7.5 [mc]=22 [q2]=10 [x]=10 [k0]=10 [k1]=10 [k2]=10 [k3]=10 [k3seen]=10
                  [tfv6]=15 [bridgedrive]=15 [simlingo]=15 [blue]=15)

ev() { printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N | cut -c1-14)" "$1" "${2:+, $2}" >> "$G/events.jsonl"; }
log() { echo "$(date '+%F %T') $*" | tee -a "$G/log.txt" >&2; }
error() {
    {
        echo "# nq4-gk ERROR $(date '+%F %T %Z')"; echo; echo "reason: $1"; echo
        echo "current step: $(cat "$G/CURRENT" 2>/dev/null)"; echo
        [[ -n ${2:-} && -f $2 ]] && { echo '```'; tail -50 "$2"; echo '```'; }
    } > "$G/ERROR"
    ev error "\"reason\": \"$1\""
    log "ERROR: $1"
    exit 1
}

# ---------------------------------------------------------------- model servers (own session, PID recorded)
srv_start() {  # srv_start <name> <gpu> <cmd ...>
    local name=$1 gpu=$2; shift 2
    srv_alive "$name" && return 0
    local ready=$G/srv/$name.ready
    rm -f "$ready" "$G/srv/$name.sock"
    (
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com setsid taskset -c "$CPUS" \
            "$@" --socket "$G/srv/$name.sock" --ready-file "$ready" >> "$G/srv/$name.log" 2>&1 &
        echo $! > "$G/srv/$name.pid"
        wait $!
        echo "$(date '+%F %T') server $name exited rc=$?" >> "$G/srv/$name.log"
    ) &
    until [[ -s $G/srv/$name.pid ]]; do sleep 0.2; done
    local t0=$SECONDS
    until [[ -e $ready ]]; do
        srv_alive "$name" || { log "server $name died at start-up"; return 1; }
        (( SECONDS - t0 > 900 )) && { log "server $name not ready after 15 min"; return 1; }
        sleep 5
    done
    ev server_ready "\"name\": \"$name\", \"pid\": $(cat "$G/srv/$name.pid")"
}
srv_alive() { local p; p=$(cat "$G/srv/$1.pid" 2>/dev/null) && [[ -n $p ]] && kill -0 "$p" 2>/dev/null; }
srv_stop() { local p; p=$(cat "$G/srv/$1.pid" 2>/dev/null) && [[ -n $p ]] && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; }; rm -f "$G/srv/$1.pid"; }
srv_stop_all() { local f; for f in "$G"/srv/*.pid; do [[ -e $f ]] && srv_stop "$(basename "$f" .pid)"; done; }

server_names() {  # the model servers an examinee needs on one GPU
    case $1 in
        cinque) echo op-cinque-g$2 ;;
        mc) echo qwen-g$2 headq-g$2 ;;
        q2|x) echo head-g$2 ;;
        k*) declare -F k_server_names >/dev/null && k_server_names "$1" "$2" ;;
    esac
}
servers_for() {
    local c=$1 g=$2
    case $c in
        cinque) srv_start op-cinque-g$g "$g" "$PY_OP" scripts/zeroshot_policy_server.py cinque --pool "$WORKERS" ;;
        mc) srv_start qwen-g$g "$g" "$PY_VENV" scripts/nq3_feat_server.py qwen || return 1
            srv_start headq-g$g "$g" "$PY_OP" scripts/nq3_cl_server.py --pool "$WORKERS" --qwen "$G/srv/qwen-g$g.sock" ;;
        q2|x) srv_start head-g$g "$g" "$PY_OP" scripts/nq3_cl_server.py --pool "$WORKERS" ;;
        k*) k_servers "$c" "$g" ;;
        *) return 0 ;;
    esac
}

# ---------------------------------------------------------------- examinee configs and runners
fold_json() {  # fold_json <key> <R1 path> <R2 path> <pick>: the cross-fit block of the nq4 agent config
    echo "\"folds\": {\"split\": \"$SPLIT\", \"key\": \"$1\", \"R1\": \"$2\", \"R2\": \"$3\", \"pick\": \"$4\"}"
}
cfg_for() {  # cfg_for <cand> <gpu> <seed> -> agent config path (P7 examinees)
    local c=$1 g=$2 seed=$3 f=$G/cfg/$1-g$2-s$3.json
    local ctl="\"controller\": \"fixed\", \"controller_preset\": \"pursuit\", \"controller_config\": \"$P7\""
    local head="\"model\": \"head\", \"warmup_s\": 5.0, \"desire\": true, \"head_cam_tick\": 0.0"
    local QX=$G/heads_xfit Q2N=${Q2NAME:-q2} D=${DUMP_EVERY:-0}   # Q2NAME=q2_placeholder, DUMP_EVERY=1: smoke / rule 8 only
    case $c in
        cinque) echo "{\"model\": \"cinque\", \"socket\": \"$G/srv/op-cinque-g$g.sock\", \"plan_every\": 1, \"ctl_every\": 4, \"op_camera_tick\": 0.05, \"plan_origin\": \"rear\", \"warmup_s\": 5.0, \"desire\": true, $ctl, \"seed\": $seed, \"dump_every\": 0}" ;;
        mc) echo "{$head, \"arm\": \"mc\", \"socket\": \"$G/srv/headq-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $D, $(fold_json heads "$QX/mc/R1/heads.npz" "$QX/mc/R2/heads.npz" unseen)}" ;;
        q2) echo "{$head, \"arm\": \"q2\", \"socket\": \"$G/srv/head-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $D, $(fold_json q2_dir "$QX/$Q2N/R1" "$QX/$Q2N/R2" unseen)}" ;;
        x) echo "{$head, \"arm\": \"q2\", \"x\": true, \"socket\": \"$G/srv/head-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $D, $(fold_json q2_dir "$QX/$Q2N/R1" "$QX/$Q2N/R2" unseen)}" ;;
        k*) k_cfg "$c" "$g" "$seed" ;;
    esac > "$f"
    echo "$f"
}
launch() {  # launch <cand> <gpu> <seed> <xml> <ids> <out>: one b2d_run runner in the background, its PID on stdout
    local c=$1 g=$2 seed=$3 xml=$4 ids=$5 out=$6 idx=$((SIDX0 + SPAN * g))
    local common=(--route-ids "$ids" --out "$out" --workers "$WORKERS" --server-index "$idx" --index-span "$SPAN"
                  --gpu-rank "$g" --tm-seed "$seed" --no-spectator --no-reap --client-threads 8 --max-attempts 3
                  --stall-s 480)
    case $c in
        pdm)
            BENCH2DRIVE_ROOT=$SIM/Bench2Drive WORK_DIR=$SIM taskset -c "$CPUS" "$PY_CARLA" scripts/b2d_run.py \
                --routes "$xml" "${common[@]}" --route-timeout-s 3600 --python "$PY_SL" \
                --agent scripts/b2d_expert_agent.py --agent-config "expert+nq4" >> "$out/runner-g$g.log" 2>&1 & ;;
        tfv6|bridgedrive|simlingo|blue)
            CL10_ROUTES=$xml INDEX_SPAN=$SPAN B_CPUS=$CPUS scripts/nq3_b_cl10.sh "$c" "$g" "$WORKERS" "$idx" "$seed" "$out" \
                "$ids" >> "$out/runner-g$g.log" 2>&1 & ;;
        *)
            local py=$PY_SCOUT ag=scripts/nq4_x_agent.py
            [[ $c == cinque ]] && { py=$PY_TCP; ag=scripts/b2d_zeroshot_agent.py; }
            [[ $c == k* ]] && declare -F k_agent >/dev/null && read -r py ag <<< "$(k_agent "$c")"
            B2D_SENSOR_TICK=1 taskset -c "$CPUS" "$PY_CARLA" scripts/b2d_run.py --routes "$xml" "${common[@]}" \
                --route-timeout-s 3600 --python "$py" --agent "$ag" --agent-config "$(cfg_for "$c" "$g" "$seed")" \
                --fast-copy --cache-lights >> "$out/runner-g$g.log" 2>&1 & ;;
    esac
    echo $!
}

kill_runs() {  # the runners, route processes and CARLA servers recorded under one out dir
    local out=$1 p f
    for p in $(cat "$out/runner.pids" 2>/dev/null); do kill "$p" 2>/dev/null; pkill -P "$p" 2>/dev/null; done
    sleep 5
    for f in "$out"/attempts/*/*/route.pid; do
        p=$(cat "$f" 2>/dev/null) || continue
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qF "$out/" && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; }
    done
    for f in "$out"/servers/carla-*.pid; do       # the CarlaUE4.sh wrapper's process group: its shipping child survives
        p=$(cat "$f" 2>/dev/null) || continue      # the wrapper when a runner dies by a signal
        pkill -P "$p" 2>/dev/null; kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null
    done
}

# ---------------------------------------------------------------- route sets and bookkeeping (python helpers)
ids_for() {  # ids_for <cand> <variant> <seed> <routeset>: comma list still to run (orig: minus night-queue-3 reuse)
    "$PY_VENV" - "$@" <<'EOF'
import json, sys
from pathlib import Path
import pandas as pd
from jevdrive import nq4_g as NG
cand, variant, seed, rs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
if rs.startswith("ids:"):                      # smoke / debug: explicit ids
    print(rs[4:]); sys.exit()
t = pd.read_csv(NG.root() / "routes.csv", dtype={"base": str})
split = json.loads((NG.data_dir() / "runs/nq4/k/route_split.json").read_text())["routes"] if rs in ("grec", "krec") else {}
if variant == "k":
    import xml.etree.ElementTree as ET
    ids = [r.get("id") for r in ET.parse(NG.data_dir() / NG.B2D_XML).getroot().iter("route")]
    if rs == "krec":
        ids = [i for i in ids if split.get(i, {}).get("recorded")]
    print(",".join(ids)); sys.exit()
v = pd.read_csv(NG.root() / "variants.csv", dtype={"id": str, "base": str})
v = v[v.variant == variant].merge(t[["base", "obstacle"]], on="base")
if rs == "gob":
    v = v[v.obstacle]
elif rs == "grec":
    v = v[v.base.map(lambda b: bool(split.get(b, {}).get("recorded")))]
if variant == "orig":
    have = set()
    for d in NG.reuse_dirs().get((cand, seed), []):
        have |= {p.stem for p in (d / "done").glob("*.json")} if (d / "done").exists() else set()
    v = v[~v.base.isin(have)]
print(",".join(v.id))
EOF
}
measured_wmin() {  # measured worker-minutes per route of one examinee (median over its finished routes), or empty
    "$PY_VENV" - "$G" "$1" <<'EOF'
import json, sys
from pathlib import Path
G, cand = Path(sys.argv[1]), sys.argv[2]
w = []
for base in ("arms", "arms_k"):
    for f in (G / base).glob(f"{cand}*/**/done/*.json") if base == "arms_k" else (G / base / cand).glob("*/s*/done/*.json"):
        try:
            w.append(float(json.loads(f.read_text())["wall_s"]) / 60.0)
        except Exception:
            pass
if len(w) >= 5:
    w.sort()
    print(round(w[len(w) // 2] * 1.15, 2))     # +15 %: server start-up and retries, not in wall_s
EOF
}
wmin() { local m; m=$(measured_wmin "${1%%_*}"); echo "${m:-${PRIOR[${1%%_*}]:-10}}"; }
n_ids() { local s=$1; [[ -z $s ]] && { echo 0; return; }; echo $(( $(tr -cd , <<< "$s" | wc -c) + 1 )); }

ports_ok() {  # ports_ok <first index>: >= WORKERS + 2 indices of the block with RPC (+1, +2) and TM ports not listening
    local t0=$SECONDS n
    while :; do
        n=$(python3 - "$1" "$SPAN" <<'EOF'
import sys
i0, span = int(sys.argv[1]), int(sys.argv[2])
busy = set()
for f in ("/proc/net/tcp", "/proc/net/tcp6"):
    for l in open(f).readlines()[1:]:
        x = l.split()
        if x[3] == "0A":                                # LISTEN
            busy.add(int(x[1].split(":")[1], 16))
print(sum(1 for i in range(i0, i0 + span)
          if not busy & {2000 + 50 * i, 2001 + 50 * i, 2002 + 50 * i, 8000 + 50 * i, 8001 + 50 * i}))
EOF
)
        (( n >= (WORKERS + 2 < SPAN ? WORKERS + 2 : SPAN) )) && return 0
        (( SECONDS - t0 > 1800 )) && return 1
        log "server block from index $1: only $n clean indices; waiting"; sleep 120
    done
}

# ---------------------------------------------------------------- one step
arm_dir() {  # arm_dir <cand> <variant> <seed>
    if [[ $2 == k ]]; then echo "$G/${ARMS:-arms}_k/$1/s$3"; else echo "$G/${ARMS:-arms}/$1/$2/s$3"; fi
}
run_step() {  # run_step <cand> <variant> <seeds a,b,c> <routeset> [est_h]
    local c=$1 v=$2 seeds=(${3//,/ }) rs=$4 est=${5:-}
    local xml=$XML_G; [[ $v == k ]] && xml=$XML_K
    local cname=${c%%_*} ns=${#seeds[@]} gl=($GPUS) t0=$SECONDS s i g
    declare -A IDS OUT
    local total=0
    for s in "${seeds[@]}"; do
        OUT[$s]=$(arm_dir "$c" "$v" "$s"); mkdir -p "${OUT[$s]}"
        if [[ -f ${OUT[$s]}/requested.json ]]; then
            IDS[$s]=$(python3 -c "import json;print(','.join(json.load(open('${OUT[$s]}/requested.json'))))")
        else
            IDS[$s]=$(ids_for "$c" "$v" "$s" "$rs") || error "route list failed for $c $v"
            python3 -c "import json,sys;print(json.dumps([i for i in sys.argv[1].split(',') if i]))" "${IDS[$s]}" > "${OUT[$s]}/requested.json"
        fi
        total=$(( total + $(n_ids "${IDS[$s]}") ))
    done
    (( total == 0 )) && { log "step $c $v $3 $rs: nothing to run (reused)"; return 0; }
    local w; w=$(wmin "$c")
    [[ -z $est ]] && est=$(python3 -c "print(max(0.3, round($total * $w / 60 / (${#gl[@]} * $WORKERS) * 1.3, 2)))")
    echo "$c $v $3 $rs" > "$G/CURRENT"
    CUR_OUTS="${OUT[*]}"
    ev step_start "\"cand\": \"$c\", \"variant\": \"$v\", \"seeds\": \"$3\", \"routes\": $total, \"est_h\": $est, \"wmin\": $w"
    log "start $c $v seeds $3 ($rs, $total routes, $w worker-min/route, estimate $est h)"
    for g in "${gl[@]}"; do ports_ok $((SIDX0 + SPAN * g)) || error "server block of GPU $g has fewer than $((WORKERS + 2)) free indices for 30 min" "$G/log.txt"; done
    for i in "${!gl[@]}"; do servers_for "$c" "${gl[$i]}" || error "server start failed for $c on GPU ${gl[$i]}" "$G/log.txt"; done
    local pids=()
    for i in "${!gl[@]}"; do
        g=${gl[$i]}; s=${seeds[$(( i % ns ))]}
        [[ -z ${IDS[$s]} ]] && continue
        local p; p=$(launch "$c" "$g" "$s" "$xml" "${IDS[$s]}" "${OUT[$s]}")
        pids+=("$p"); echo "$p" >> "${OUT[$s]}/runner.pids"
        sleep 10
    done
    local limit; limit=$(python3 -c "print(int(2 * $est * 3600))")
    while :; do
        local alive=0 p
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        (( alive )) || break
        if (( SECONDS - t0 > limit )); then
            for s in "${seeds[@]}"; do kill_runs "${OUT[$s]}"; done
            error "$c $v seeds $3 exceeded twice its estimate ($est h)" "${OUT[${seeds[0]}]}/runner-g${gl[0]}.log"
        fi
        for g in "${gl[@]}"; do
            for n in $(server_names "$c" "$g"); do
                srv_alive "$n" || { log "server $n died; restarting"; ev server_died "\"name\": \"$n\""
                                    servers_for "$c" "$g" || error "server restart failed: $n" "$G/srv/$n.log"; }
            done
        done
        sleep 30
    done
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
    local bad=0 req done_ wall
    wall=$(python3 -c "print(round(($SECONDS - $t0) / 3600, 3))")
    for s in "${seeds[@]}"; do
        req=$(n_ids "${IDS[$s]}"); done_=$(ls "${OUT[$s]}/done" 2>/dev/null | wc -l)
        printf '{"cand": "%s", "variant": "%s", "seed": %s, "done": %s, "requested": %s, "wall_h": %s, "finished": "%s"}\n' \
            "$c" "$v" "$s" "$done_" "$req" "$wall" "$(date '+%F %T')" > "${OUT[$s]}/DONE"
        python3 -c "import sys; sys.exit(0 if ($req - $done_) / max($req, 1) <= 0.10 else 1)" || bad=1
    done
    if [[ $c == simlingo || $c == blue ]]; then          # the author agents' debug images: output only, ~60 MB a route
        for s in "${seeds[@]}"; do rm -rf "${OUT[$s]}/viz"; done
    fi
    ev step_end "\"cand\": \"$c\", \"variant\": \"$v\", \"seeds\": \"$3\", \"wall_h\": $wall"
    log "done $c $v seeds $3 in $wall h"
    report_now
    (( bad )) && error "$c $v seeds $3: more than 10% of the routes never finished" "${OUT[${seeds[0]}]}/runner-g${gl[0]}.log"
    return 0
}
report_now() { taskset -c "$CPUS" "$PY_VENV" -m jevdrive.nq4_g report >> "$G/report.log" 2>&1 || log "report failed (report.log)"; }

# ---------------------------------------------------------------- queue, budget, cuts
queue() {  # the registered order (todo "时间表": ghost first, then K0 / K3, the other examinees, X, K1 / K2, shift / swap)
    local S=0,1,2
    cat <<EOF
pdm ghost $S g
tfv6 ghost $S g
bridgedrive ghost $S g
blue ghost $S g
simlingo ghost $S g
mc ghost $S g
k3seen ghost $S grec
k0_unseen k $S k220
k3_unseen k $S k220
k0_seen k $S krec
k3_seen k $S krec
cinque ghost $S g
q2 ghost $S gob
pdm orig 1,2 g
tfv6 orig $S g
bridgedrive orig $S g
blue orig $S g
simlingo orig $S g
cinque orig $S g
mc orig $S g
q2 orig $S gob
k3seen orig $S grec
x orig $S gob
k1_unseen k $S k220
k2_unseen k $S k220
EOF
    local c
    for c in tfv6 bridgedrive blue mc pdm simlingo cinque k3seen; do echo "$c shift 0 $( [[ $c == k3seen ]] && echo grec || echo g)"; done
    echo "q2 shift 0 gob"
    for c in tfv6 bridgedrive blue mc pdm simlingo cinque k3seen; do echo "$c swap 0 $( [[ $c == k3seen ]] && echo grec || echo g)"; done
    echo "q2 swap 0 gob"
}
step_done() {  # every seed dir of the step has DONE
    local c=$1 v=$2 s
    for s in ${3//,/ }; do [[ -e $(arm_dir "$c" "$v" "$s")/DONE ]] || return 1; done
}
ready_for() {  # an examinee's inputs exist (heads exported, K's READY); else the step waits for a later pass
    case $1 in
        mc) [[ -e $G/heads_xfit/mc/READY ]] ;;
        q2|x) [[ -e $G/heads_xfit/q2/READY ]] && return 0
              [[ -e $NQ3/q2/closed_loop_head/READY ]] || return 1
              log "lane C's head is READY: exporting the cross-fitted Q2 head (GPU ${GPUS%% *})"
              CUDA_VISIBLE_DEVICES=${GPUS%% *} taskset -c "$CPUS" "$PY_VENV" -m jevdrive.nq4_x export-q2 >> "$G/export_q2.log" 2>&1 \
                  || error "cross-fitted Q2 export failed" "$G/export_q2.log"
              ev export '"what": "q2 cross-fit"'; [[ -e $G/heads_xfit/q2/READY ]] ;;
        k*) [[ -e $K/READY ]] && declare -F k_cfg >/dev/null ;;
        *) return 0 ;;
    esac
}
project() {  # project <queue file>: projected worker-hours of every step (done steps at their measured cost)
    local tot=0 line c v s rs n w
    while read -r c v s rs; do
        n=0
        for sd in ${s//,/ }; do
            if [[ -e $(arm_dir "$c" "$v" "$sd")/DONE ]]; then continue; fi
            local cf=$G/counts/$c.$v.$sd.$rs
            [[ -s $cf ]] || { mkdir -p "$G/counts"; n_ids "$(ids_for "$c" "$v" "$sd" "$rs")" > "$cf"; }
            n=$(( n + $(cat "$cf") ))
        done
        w=$(wmin "$c")
        tot=$(python3 -c "print($tot + $n * $w / 60)")
        printf '%-12s %-6s %-6s %-5s %4d routes x %5s wmin = %6.1f worker-h\n' "$c" "$v" "$s" "$rs" "$n" "$w" "$(python3 -c "print($n * $w / 60)")"
    done < "$1"
    echo "TOTAL_REMAINING_WH $tot"
}
apply_cuts() {  # apply_cuts <queue file>: the todo's cut rule, once the projection exceeds the budget
    local qf=$1 rem done_wh total
    rem=$(project "$qf" | awk '/^TOTAL_REMAINING_WH/ {print $2}')
    done_wh=$(python3 - "$G" <<'EOF'
import json, sys
from pathlib import Path
print(round(sum(float(json.loads(f.read_text())["wall_s"]) for f in Path(sys.argv[1]).glob("arms*/**/done/*.json")) / 3600 * 1.15, 1))
EOF
)
    total=$(python3 -c "print(round($rem + $done_wh, 1))")
    echo "$total" > "$G/PROJECTED_WH"
    if python3 -c "import sys; sys.exit(0 if $total > $BUDGET_WH else 1)"; then
        if grep -q ' swap ' "$qf"; then
            grep -v ' swap ' "$qf" > "$qf.tmp" && mv "$qf.tmp" "$qf"
            ev cut "\"what\": \"swap\", \"projected_wh\": $total"; log "projected $total worker-h > $BUDGET_WH: swap steps cut"
            apply_cuts "$qf"; return
        fi
        if grep -E ' shift ' "$qf" | grep -qvE '^(tfv6|bridgedrive|blue|mc) '; then
            grep -vE '^(pdm|simlingo|cinque|q2|k3seen) shift ' "$qf" > "$qf.tmp" && mv "$qf.tmp" "$qf"
            ev cut "\"what\": \"shift to tfv6 / bridgedrive / blue / mc\", \"projected_wh\": $total"
            log "projected $total worker-h > $BUDGET_WH: shift cut to tfv6 / bridgedrive / blue / mc"
            apply_cuts "$qf"; return
        fi
        [[ -e $G/OVER_BUDGET ]] || { echo "$(date '+%F %T') projected $total worker-h after both cuts" > "$G/OVER_BUDGET"
                                     ev over_budget "\"projected_wh\": $total"; log "still over budget after both cuts ($total worker-h); the queue runs in priority order"; }
    fi
}

status_loop() {
    while sleep 600; do
        {
            echo "# nq4-gk status $(date '+%F %T %Z')"; echo
            echo "- current: $(cat "$G/CURRENT" 2>/dev/null)"
            echo "- projected total worker-h: $(cat "$G/PROJECTED_WH" 2>/dev/null) (budget $BUDGET_WH)$( [[ -e $G/OVER_BUDGET ]] && echo ', OVER BUDGET after both cuts')"
            echo "- steps left: $(grep -c . "$G/QUEUE" 2>/dev/null)"
            echo "- GPUs $GPUS, workers/GPU $WORKERS, cores $CPUS; pids.current $(cat /sys/fs/cgroup/pids.current), load $(cut -d' ' -f1-3 /proc/loadavg)"
            echo; echo '```'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo '```'
            echo; echo "finished seeds:"
            for f in "$G"/arms*/*/*/DONE "$G"/arms/*/*/*/DONE; do [[ -e $f ]] && echo "- $(cat "$f")"; done | sort -u
        } > "$G/STATUS.md.tmp" && mv "$G/STATUS.md.tmp" "$G/STATUS.md"
    done
}
gate() {  # runs/nq3/b/DONE and no CARLA server left on the chain's GPUs
    local g n
    until [[ -e $NQ3/b/DONE ]]; do sleep 120; done
    while :; do
        n=0
        for g in $GPUS; do n=$(( n + $(nvidia-smi -i "$g" --query-compute-apps=process_name --format=csv,noheader | grep -c CarlaUE4) )); done
        (( n == 0 )) && break
        sleep 120
    done
    ev gate_open; log "gate open: runs/nq3/b/DONE and no CARLA server on GPUs $GPUS"
}

chain() {
    echo "waiting for runs/nq3/b/DONE" > "$G/CURRENT"
    status_loop & local st=$!
    trap 'kill $st 2>/dev/null; for o in $CUR_OUTS; do kill_runs "$o"; done; srv_stop_all' EXIT
    trap 'exit 129' HUP INT TERM
    gate
    [[ -e $G/prep/DONE ]] || error "G-prep has not passed (runs/nq4/gk/prep/DONE missing)"
    [[ -f scripts/nq4_k_recipes.sh ]] && source scripts/nq4_k_recipes.sh
    [[ -f $G/QUEUE ]] || queue > "$G/QUEUE"
    apply_cuts "$G/QUEUE"
    local line c v s rs pass=0 ran
    while :; do
        ran=0
        while read -r c v s rs; do
            step_done "$c" "$v" "$s" && continue
            ready_for "$c" || continue
            ran=1
            run_step "$c" "$v" "$s" "$rs" < /dev/null
            srv_stop_all
            apply_cuts "$G/QUEUE"
            break                                    # re-read the (possibly cut) queue from the top
        done < "$G/QUEUE"
        (( ran )) && continue
        local left; left=$(while read -r c v s rs; do step_done "$c" "$v" "$s" || echo x; done < "$G/QUEUE" | wc -l)
        (( left == 0 )) && break
        (( pass++ == 0 )) && log "$left steps wait for their inputs (fold heads / K READY); polling every 10 min"
        sleep 600
    done
    date '+%F %T' > "$G/DONE"
    ev end '"what": "nq4 G + K + X queue done"'
    log "queue done"
}

case ${1:-} in
    chain) chain ;;
    step) shift; trap 'for o in $CUR_OUTS; do kill_runs "$o"; done; srv_stop_all' EXIT; trap 'exit 129' HUP INT TERM
          [[ -f scripts/nq4_k_recipes.sh ]] && source scripts/nq4_k_recipes.sh; run_step "$@" ;;
    report) report_now ;;
    plan) [[ -f scripts/nq4_k_recipes.sh ]] && source scripts/nq4_k_recipes.sh; queue > "$G/plan.tmp"; project "$G/plan.tmp" ;;
    *) sed -n 2,26p "$0"; exit 1 ;;
esac
