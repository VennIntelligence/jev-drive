#!/usr/bin/env bash
# Night queue 4: the G + K + X closed-loop chain (todos/2026-09-26-night-queue-4.md, sections G / K / X, the [F] entries).
# One chain script, started once in tmux jev:nq4-gk; it runs unattended from there.
#
#   scripts/nq4_gk.sh chain        pilot loop on the validation card + the batch on the GO cards -> DONE
#   scripts/nq4_gk.sh step <cand> <variant> <seeds> <routeset> [est_h]   one queue step (pilot first), for debugging
#   scripts/nq4_gk.sh pilot <cand> <variant> <seed> <routeset>             one staged pilot, for debugging
#   scripts/nq4_gk.sh report       the tables (runs/nq4/gk/results/{g,k,x}/)
#   scripts/nq4_gk.sh plan         print the queue with its current worker-hour estimate, apply no cut
#
# Capacity (first come, first served; no wait for other lanes' DONE): the batch runs on the cards of the GO file
# runs/nq4/gk/GO (SCH's table runs/sched/table.tsv; shell vars GPUS, WORKERS, IDX0, IDX_SPAN, NQ4GK_CPUS; the i-th card of GPUS uses server indices
# [IDX0 + i IDX_SPAN, IDX0 + (i + 1) IDX_SPAN)), written by SCH / Codex once the pilots pass; it is re-read before every step.
# The pilots run on the validation card of runs/sched/nq4-gk.pilot (PILOT_GPU, PILOT_WORKERS, PILOT_IDX0, PILOT_SPAN,
# PILOT_CPUS). No card starts more CARLA servers than fit its 6 (other lanes' counted).
# Staged launch (CLAUDE.md "Before a long run"): every examinee x world type runs 1 route, is checked, then 10 routes, is
# checked against the written checklist (jevdrive.nq4_g.pilot_check: completion, crash, stalled / blocked, moving,
# plans / trace present, PDM-Lite ghost stays in lane, ghost actors really absent, shift / swap really applied, K DS near
# night queue 3's), and only then the full step. A failed check blocks that examinee (runs/nq4/gk/ERROR.<cand>,
# runs/nq4/gk/blocked/<cand>); the rest of the queue goes on. Pilot routes are seed-0 routes of the step and count.
#
# Step = examinee x world variant x TM seeds x route set. The seeds of a step run side by side, seed i on the i-th,
# (i + n)-th ... card (one b2d_run runner per card; the runners of a seed share its out dir); fewer cards than seeds ->
# one seed after the other.
#   examinees  pdm (PDM-Lite, SimLingo's tree: it plans on the registry), tfv6 / bridgedrive / simlingo / blue (author
#              executors, scripts/nq3_b_cl10.sh as is), cinque (openpilot Cinque native plan -> P7, lane B's CL2),
#              mc / q2 (our heads -> P7, cross-fitted: each route driven by the fold that never saw it), x (Q2 mode head ->
#              geometric path -> P7), k0..k3 (K ladder, K's wiring in b2d_zeroshot_agent; waits for runs/nq4/k/READY),
#              k3seen (K3 driven by the fold that DID see the route: G's positive control)
#   variants   orig ghost shift swap (G, runs/nq4/gk/g_routes.xml) | k (the official 220, K)
#   routesets  g (the 80 G routes / 50 for swap), gob (the 40 obstacle routes), grec (G routes recorded in K's
#              training data), k220 (all 220), krec (the 220 routes recorded in K's training data)
# Budget: after every step the whole queue is re-estimated from measured worker-minutes per route (median x 1.15, prior
# until an examinee has >= 5 finished routes; the first 20 G routes of the queue are that profiling). G projected (done +
# remaining) > $BUDGET_WH worker-hours (600) -> the todo's cut: first drop every swap step, then keep shift only for
# tfv6 / bridgedrive / blue / mc; still over -> OVER_BUDGET, the queue runs on in priority order. K and X are projected
# and shown (PROJECTED_WH) but not cut. Each step stops at twice its estimate (ERROR).
# Ports: server index i binds RPC 2000 + 50 i (+1, +2) and TM 8000 + 50 i, i.e. the RPC port of index i + 120, so a block
# must keep i, i + 120 and i - 120 clear of every other lane's indices (defaults without a GO file: IDX0 60, span 18,
# 60-167 for six cards). Before every execution each card's block is checked against the listening sockets
# (/proc/net/tcp): fewer than min(WORKERS + 2, span) clean indices -> wait (2 min polls, 30 min at most), then ERROR.
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
PIL=$G/${ARMS:-arms}_pilot BLK=$G/${ARMS:-arms}_blocked ERRP=$G/ERROR${ARMS:+.$ARMS}   # smoke runs (ARMS=...) keep their own
export B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000} B2D_NQ4_TRACE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
SIM=$DATA_DIR/third_party/simlingo
XML_K=$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml
XML_G=$G/g_routes.xml
P7=$REPO/todos/2026-09-23-tfv6-controller/controller-eval/P7.json
SPLIT=$K/route_split.json
SCHED=$DATA_DIR/runs/sched
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
srv_stop_gpus() { local g f; for g in $1; do for f in "$G"/srv/*-g$g.pid; do [[ -e $f ]] && srv_stop "$(basename "$f" .pid)"; done; done; }

server_names() {  # the model servers an examinee needs on one GPU
    case $1 in
        cinque) echo op-cinque-g$2 ;;
        mc) echo qwen-g$2 headq-g$2 ;;
        q2|x) echo head-g$2 ;;
        k*) k_server_names "$1" "$2" ;;
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

# K's closed-loop wiring (scripts/nq4_k.sh cl): scripts/b2d_zeroshot_agent.py "model": "head", "arm": "k0".."k3",
# "k_view" unseen / seen, "k_split"; one head server (the same as q2 / x). Names: k<L>_unseen, k<L>_seen, k3seen (G).
k_server_names() { echo head-g$2; }
k_servers() { srv_start head-g$2 "$2" "$PY_OP" scripts/nq3_cl_server.py --pool "$WORKERS"; }
k_agent() { echo "$PY_SCOUT scripts/b2d_zeroshot_agent.py"; }
k_cfg() {  # k_cfg <cand> <gpu> <seed>
    local arm=${1:0:2} view=unseen
    [[ $1 == *seen && $1 != *unseen ]] && view=seen
    echo "{\"model\": \"head\", \"warmup_s\": 5.0, \"desire\": true, \"head_cam_tick\": 0.0, \"arm\": \"$arm\", \"k_view\": \"$view\", \"k_split\": \"$SPLIT\", \"socket\": \"$G/srv/head-g$2.sock\", \"controller\": \"fixed\", \"controller_preset\": \"pursuit\", \"controller_config\": \"$P7\", \"seed\": $3, \"dump_every\": 0}"
}

# ---------------------------------------------------------------- examinee configs and runners
fold_json() {  # fold_json <key> <R1 path> <R2 path> <pick> <rule>: the cross-fit block of the nq4 agent config
    echo "\"folds\": {\"split\": \"$SPLIT\", \"key\": \"$1\", \"R1\": \"$2\", \"R2\": \"$3\", \"pick\": \"$4\", \"rule\": \"$5\"}"
}
cfg_for() {  # cfg_for <cand> <gpu> <seed> -> agent config path (P7 examinees)
    local c=$1 g=$2 seed=$3 f=$G/cfg/$1-g$2-s$3.json
    local ctl="\"controller\": \"fixed\", \"controller_preset\": \"pursuit\", \"controller_config\": \"$P7\""
    local head="\"model\": \"head\", \"warmup_s\": 5.0, \"desire\": true, \"head_cam_tick\": 0.0"
    local QX=$G/heads_xfit Q2N=${Q2NAME:-q2} D=${DUMP_EVERY:-0}   # Q2NAME=q2_placeholder, DUMP_EVERY=1: smoke / rule 8 only
    case $c in
        cinque) echo "{\"model\": \"cinque\", \"socket\": \"$G/srv/op-cinque-g$g.sock\", \"plan_every\": 1, \"ctl_every\": 4, \"op_camera_tick\": 0.05, \"plan_origin\": \"rear\", \"warmup_s\": 5.0, \"desire\": true, $ctl, \"seed\": $seed, \"dump_every\": 0}" ;;
        mc) echo "{$head, \"arm\": \"mc\", \"socket\": \"$G/srv/headq-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $D, $(fold_json heads "$QX/mc/R1/heads.npz" "$QX/mc/R2/heads.npz" unseen k)}" ;;
        q2) echo "{$head, \"arm\": \"q2\", \"socket\": \"$G/srv/head-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $D, $(fold_json q2_dir "$QX/$Q2N/R1" "$QX/$Q2N/R2" unseen label)}" ;;
        x) echo "{$head, \"arm\": \"q2\", \"x\": true, \"socket\": \"$G/srv/head-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $D, $(fold_json q2_dir "$QX/$Q2N/R1" "$QX/$Q2N/R2" unseen label)}" ;;
        k*) k_cfg "$c" "$g" "$seed" ;;
    esac > "$f"
    echo "$f"
}
launch() {  # launch <cand> <gpu> <seed> <xml> <ids> <out> <workers> <first index>: one b2d_run runner, PID on stdout
    local c=$1 g=$2 seed=$3 xml=$4 ids=$5 out=$6 nw=$7 idx=$8
    mkdir -p "$out"
    local common=(--route-ids "$ids" --out "$out" --workers "$nw" --server-index "$idx" --index-span "$SPAN"
                  --gpu-rank "$g" --tm-seed "$seed" --no-spectator --no-reap --client-threads 8 --max-attempts 3
                  --stall-s 480)
    case $c in
        pdm)
            BENCH2DRIVE_ROOT=$SIM/Bench2Drive WORK_DIR=$SIM taskset -c "$CPUS" "$PY_CARLA" scripts/b2d_run.py \
                --routes "$xml" "${common[@]}" --route-timeout-s 3600 --python "$PY_SL" \
                --agent scripts/b2d_expert_agent.py --agent-config "expert+nq4" >> "$out/runner-g$g.log" 2>&1 & ;;
        tfv6|bridgedrive|simlingo|blue)
            CL10_ROUTES=$xml INDEX_SPAN=$SPAN B_CPUS=$CPUS scripts/nq3_b_cl10.sh "$c" "$g" "$nw" "$idx" "$seed" "$out" \
                "$ids" >> "$out/runner-g$g.log" 2>&1 & ;;
        *)
            local py=$PY_SCOUT ag=scripts/nq4_x_agent.py
            [[ $c == cinque ]] && { py=$PY_TCP; ag=scripts/b2d_zeroshot_agent.py; }
            [[ $c == k* ]] && read -r py ag <<< "$(k_agent "$c")"
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
    local left=()
    for f in "$out"/attempts/*/*/route.pid; do
        p=$(cat "$f" 2>/dev/null) || continue
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qF "$out/" && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; left+=("$p"); }
    done
    sleep 10                                        # a route process can sit in the evaluator's SIGTERM handler: KILL it
    for p in "${left[@]}"; do
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qF "$out/" && { kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; }
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
    town = {r.get("id"): r.get("town") for r in ET.parse(NG.data_dir() / NG.B2D_XML).getroot().iter("route")}
    ids = list(town)
    if rs == "krec":
        ids = [i for i in ids if split.get(i, {}).get("recorded")]
    print(",".join(sorted(ids, key=lambda i: (town[i], int(i))))); sys.exit()   # same town back to back (world reuse)
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
v = v.merge(t[["base", "town"]], on="base").sort_values(["town", "id"])      # same town back to back (world reuse)
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
    for f in (G / base).glob(f"{cand}_*/s*/done/*.json") if base == "arms_k" else (G / base / cand).glob("*/s*/done/*.json"):
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

# ---------------------------------------------------------------- capacity (GO file, pilot card), execution of one route set
load_go() {  # the batch cards: runs/nq4/gk/GO (SCH's table, runs/sched/table.tsv; runs/sched/nq4-gk.go also read), shell
             # vars GPUS, WORKERS, IDX0, IDX_SPAN and the lane's cores (NQ4GK_CPUS / NQ4_GK_CPUS / GK_CPUS / CPUS)
    local f; for f in "$G/GO" "$SCHED/nq4-gk.go"; do [[ -f $f ]] && break; done
    [[ -f $f ]] || return 1
    local GPUS_= WORKERS_= IDX0_= IDX_SPAN_= CPUS_=
    eval "$(set +u; source "$f"; echo "GPUS_='${GPUS//,/ }' WORKERS_='$WORKERS' IDX0_='$IDX0' IDX_SPAN_='$IDX_SPAN' CPUS_='${NQ4GK_CPUS:-${NQ4_GK_CPUS:-${GK_CPUS:-$CPUS}}}'")"
    [[ -n $GPUS_ ]] || return 1
    GPUS=$GPUS_; WORKERS=${WORKERS_:-6}; SIDX0=${IDX0_:-60}; SPAN=${IDX_SPAN_:-18}; CPUS=${CPUS_:-60-149}; CTX=batch
}
load_pilot() {  # the validation card: $SCHED/nq4-gk.pilot (PILOT_GPU, PILOT_WORKERS, PILOT_IDX0, PILOT_SPAN, PILOT_CPUS)
    [[ -f $SCHED/nq4-gk.pilot ]] || return 1
    local P_= W_= I_= S_= C_=
    eval "$(set +u; source "$SCHED/nq4-gk.pilot"; echo "P_='$PILOT_GPU' W_='$PILOT_WORKERS' I_='$PILOT_IDX0' S_='$PILOT_SPAN' C_='$PILOT_CPUS'")"
    [[ -n $P_ ]] || return 1
    GPUS=$P_; WORKERS=${W_:-2}; SIDX0=${I_:-150}; SPAN=${S_:-10}; CPUS=${C_:-110-113}; CTX=pilot
}
block_of() { echo $(( SIDX0 + SPAN * $1 )); }         # <position of the GPU in $GPUS> -> its first server index
free_slots() {  # free_slots <gpu>: CARLA servers that still fit on a batch card (<= 6 a card, other lanes' counted); on the
               # validation card SCH's table already splits the slots between lanes, so there it is PILOT_WORKERS as granted
    [[ ${CTX:-batch} == pilot ]] && { echo "$WORKERS"; return; }
    local n; n=$(nvidia-smi -i "$1" --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c CarlaUE4)
    echo $(( 6 - n > WORKERS ? WORKERS : (6 - n > 0 ? 6 - n : 0) ))
}

execute() {  # execute <cand> <variant> <est_h> <cap> <seed>=<ids> ...: run route sets on the context's cards (GPUS,
             # WORKERS, SIDX0, SPAN, CPUS), at most <cap> workers in all; seeds side by side when there are enough cards
    local c=$1 v=$2 est=$3 cap=$4; shift 4
    local xml=$XML_G; [[ $v == k ]] && xml=$XML_K
    local gl=($GPUS) sets=("$@") t0=$SECONDS i g s ids n pids=() outs=() used=0
    (( ${#gl[@]} < ${#sets[@]} && ${#sets[@]} > 1 )) && { for s in "${sets[@]}"; do execute "$c" "$v" "$est" "$cap" "$s" || return 1; done; return 0; }
    for i in "${!gl[@]}"; do ports_ok "$(block_of "$i")" || { log "server block of GPU ${gl[$i]} has no free indices for 30 min"; return 1; }; done
    for i in "${!gl[@]}"; do servers_for "$c" "${gl[$i]}" || { log "server start failed for $c on GPU ${gl[$i]}"; return 1; }; done
    for i in "${!gl[@]}"; do
        g=${gl[$i]}; s=${sets[$(( i % ${#sets[@]} ))]}; ids=${s#*=}; s=${s%%=*}
        [[ -z $ids ]] && continue
        n=$(free_slots "$g"); (( n > cap - used )) && n=$(( cap - used ))
        (( n <= 0 )) && continue
        local out; out=$(arm_dir "$c" "$v" "$s"); outs+=("$out")
        local p; p=$(launch "$c" "$g" "$s" "$xml" "$ids" "$out" "$n" "$(block_of "$i")")
        pids+=("$p"); echo "$p" >> "$out/runner.pids"; used=$(( used + n ))
        sleep 10
    done
    (( ${#pids[@]} )) || { log "no free CARLA slot on GPUs $GPUS"; srv_stop_gpus "$GPUS"; return 3; }
    CUR_OUTS="${outs[*]}"
    local limit; limit=$(python3 -c "print(int(2 * $est * 3600))")
    while :; do
        local alive=0 p
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        (( alive )) || break
        if (( SECONDS - t0 > limit )); then
            for s in "${outs[@]}"; do kill_runs "$s"; done
            log "$c $v exceeded twice its estimate ($est h)"; return 2
        fi
        for g in "${gl[@]}"; do
            for n in $(server_names "$c" "$g"); do
                srv_alive "$n" || { log "server $n died; restarting"; ev server_died "\"name\": \"$n\""
                                    servers_for "$c" "$g" || { log "server restart failed: $n"; return 1; }; }
            done
        done
        sleep 30
    done
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
    CUR_OUTS=""
    return 0
}

# ---------------------------------------------------------------- staged launch (1 route, then 10, the checklist, then all)
pilot() {  # pilot <cand> <variant> <seed> <ids>: 0 = passed (now or before), 1 = examinee blocked, 2 = not run (busy elsewhere)
    local c=$1 v=$2 sd=$3 ids=$4 d=$PIL/$1.$2 st r
    [[ -e $d/PASS ]] && return 0
    [[ -z $ids ]] && return 0                              # nothing to run for this seed (all reused)
    [[ -e $BLK/$c ]] && return 1
    mkdir -p "$PIL"; mkdir "$d" 2>/dev/null || return 2          # another loop is piloting it
    local one=${ids%%,*} ten since; ten=$(tr , '\n' <<< "$ids" | head -10 | paste -sd,); since=$(date +%s)
    for st in 1 2; do
        local sel=$one; [[ $st == 2 ]] && sel=$ten
        log "pilot $c $v stage $st: $(n_ids "$sel") route(s) on GPUs $GPUS"
        ev pilot_start "\"cand\": \"$c\", \"variant\": \"$v\", \"stage\": $st"
        while :; do                                    # no free slot is not a pilot failure: wait for one
            execute "$c" "$v" "$( [[ $st == 1 ]] && echo 0.5 || echo 1.5)" "$( [[ $st == 1 ]] && echo 1 || echo 10)" "$sd=$sel"; r=$?
            (( r == 3 )) || break
            sleep 120
        done
        srv_stop_gpus "$GPUS"
        r=$(taskset -c "$CPUS" "$PY_VENV" -m jevdrive.nq4_g pilot-check --cand "$c" --variant "$v" --out "$(arm_dir "$c" "$v" "$sd")" --ids "$sel" --since "$since" 2>> "$d/check.err")
        echo "$r" > "$d/stage$st.json"
        if ! python3 -c "import json,sys; sys.exit(0 if json.loads(sys.argv[1])['pass'] else 1)" "$r" 2>/dev/null; then
            mkdir -p "$BLK"; echo "$r" > "$BLK/$c"
            { echo "# nq4-gk pilot FAILED: $c $v stage $st ($(date '+%F %T %Z'))"; echo; echo "$r"; } > "$ERRP.$c"
            ev pilot_failed "\"cand\": \"$c\", \"variant\": \"$v\", \"stage\": $st"
            log "pilot $c $v stage $st FAILED: examinee $c blocked ($ERRP.$c)"; return 1
        fi
    done
    date '+%F %T' > "$d/PASS"; ev pilot_pass "\"cand\": \"$c\", \"variant\": \"$v\""; log "pilot $c $v passed"
    return 0
}

# ---------------------------------------------------------------- one step
arm_dir() {  # arm_dir <cand> <variant> <seed>
    if [[ $2 == k ]]; then echo "$G/${ARMS:-arms}_k/$1/s$3"; else echo "$G/${ARMS:-arms}/$1/$2/s$3"; fi
}
step_ids() {  # step_ids <cand> <variant> <seed> <routeset>: the seed's route list (requested.json once written)
    local out; out=$(arm_dir "$1" "$2" "$3"); mkdir -p "$out"
    if [[ ! -f $out/requested.json ]]; then
        local ids; ids=$(ids_for "$@") || return 1
        python3 -c "import json,sys;print(json.dumps([i for i in sys.argv[1].split(',') if i]))" "$ids" > "$out/requested.json"
    fi
    python3 -c "import json;print(','.join(json.load(open('$out/requested.json'))))"
}
run_step() {  # run_step <cand> <variant> <seeds a,b,c> <routeset> [est_h]: pilot (if not passed) then the full step
    local c=$1 v=$2 seeds=(${3//,/ }) rs=$4 est=${5:-} s total=0 sets=() t0=$SECONDS
    for s in "${seeds[@]}"; do
        local ids; ids=$(step_ids "$c" "$v" "$s" "$rs") || error "route list failed for $c $v"
        sets+=("$s=$ids"); total=$(( total + $(n_ids "$ids") ))
    done
    (( total == 0 )) && { for s in "${seeds[@]}"; do echo '{"reused": true}' > "$(arm_dir "$c" "$v" "$s")/DONE"; done
                          log "step $c $v $3 $rs: nothing to run (reused)"; return 0; }
    local s0ids; s0ids=$(step_ids "$c" "$v" "${seeds[0]}" "$rs")
    if [[ -n $s0ids && -z ${NO_PILOT:-} ]]; then
        pilot "$c" "$v" "${seeds[0]}" "$s0ids"; local r=$?
        (( r == 1 )) && return 1
        (( r == 2 )) && return 2                          # being piloted elsewhere: come back later
    fi
    local nw=0 g; for g in $GPUS; do nw=$(( nw + $(free_slots "$g") )); done
    (( nw == 0 )) && { log "no free CARLA slot on GPUs $GPUS for $c $v; waiting"; return 2; }
    local w; w=$(wmin "$c")
    [[ -z $est ]] && est=$(python3 -c "print(max(0.3, round($total * $w / 60 / $nw * 1.3, 2)))")
    echo "$c $v $3 $rs" > "$G/CURRENT"
    ev step_start "\"cand\": \"$c\", \"variant\": \"$v\", \"seeds\": \"$3\", \"routes\": $total, \"est_h\": $est, \"wmin\": $w, \"workers\": $nw"
    log "start $c $v seeds $3 ($rs, $total routes, $w worker-min/route, $nw workers, estimate $est h)"
    execute "$c" "$v" "$est" 999 "${sets[@]}"; local r=$?
    srv_stop_gpus "$GPUS"
    (( r == 3 )) && return 2                          # no free slot right now: the chain comes back to it
    (( r == 2 )) && error "$c $v seeds $3 exceeded twice its estimate ($est h)" "$G/log.txt"
    (( r == 1 )) && error "$c $v seeds $3: servers / ports failed (see log.txt)" "$G/log.txt"
    local bad=0 req done_ wall out
    wall=$(python3 -c "print(round(($SECONDS - $t0) / 3600, 3))")
    for s in "${seeds[@]}"; do
        out=$(arm_dir "$c" "$v" "$s"); req=$(python3 -c "import json;print(len(json.load(open('$out/requested.json'))))")
        done_=$(ls "$out/done" 2>/dev/null | wc -l)
        printf '{"cand": "%s", "variant": "%s", "seed": %s, "done": %s, "requested": %s, "wall_h": %s, "finished": "%s"}\n' \
            "$c" "$v" "$s" "$done_" "$req" "$wall" "$(date '+%F %T')" > "$out/DONE"
        python3 -c "import sys; sys.exit(0 if ($req - $done_) / max($req, 1) <= 0.10 else 1)" || bad=1
        [[ $c == simlingo || $c == blue ]] && rm -rf "$out/viz"     # the author agents' debug images: output only
    done
    ev step_end "\"cand\": \"$c\", \"variant\": \"$v\", \"seeds\": \"$3\", \"wall_h\": $wall"
    log "done $c $v seeds $3 in $wall h"
    report_now
    (( bad )) && error "$c $v seeds $3: more than 10% of the routes never finished" "$G/log.txt"
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
              mkdir "$G/heads_xfit/q2.lock" 2>/dev/null || return 1          # the other loop is exporting it
              log "lane C's head is READY: exporting the cross-fitted Q2 head (GPU ${GPUS%% *})"
              CUDA_VISIBLE_DEVICES=${GPUS%% *} taskset -c "$CPUS" "$PY_VENV" -m jevdrive.nq4_x export-q2 >> "$G/export_q2.log" 2>&1 \
                  || error "cross-fitted Q2 export failed" "$G/export_q2.log"
              ev export '"what": "q2 cross-fit"'; [[ -e $G/heads_xfit/q2/READY ]] ;;
        k*) [[ -e $K/READY ]] ;;
        *) return 0 ;;
    esac
}
project() {  # project <queue file>: remaining worker-hours per step and per part (G / K / X)
    local c v s rs n w sd cf wh g=0 k=0 x=0
    while read -r c v s rs; do
        n=0
        for sd in ${s//,/ }; do
            [[ -e $(arm_dir "$c" "$v" "$sd")/DONE ]] && continue
            cf=$G/counts/$c.$v.$sd.$rs
            [[ -s $cf ]] || { mkdir -p "$G/counts"; n_ids "$(ids_for "$c" "$v" "$sd" "$rs")" > "$cf"; }
            n=$(( n + $(cat "$cf") ))
        done
        w=$(wmin "$c")
        wh=$(python3 -c "print(round($n * $w / 60, 2))")
        case $v:$c in k:*) k=$(python3 -c "print($k + $wh)") ;; *:x) x=$(python3 -c "print($x + $wh)") ;; *) g=$(python3 -c "print($g + $wh)") ;; esac
        printf '%-12s %-6s %-6s %-5s %4d routes x %5s wmin = %6.1f worker-h\n' "$c" "$v" "$s" "$rs" "$n" "$w" "$wh"
    done < "$1"
    echo "REMAINING_WH G $g K $k X $x"
}
done_wh() {  # worker-hours already spent: G, K
    python3 - "$G" <<'EOF'
import json, sys
from pathlib import Path
G = Path(sys.argv[1])
f = lambda pat: round(sum(float(json.loads(p.read_text())["wall_s"]) for p in G.glob(pat)) / 3600 * 1.15, 1)
print(f("arms/*/*/s*/done/*.json"), f("arms_k/*/s*/done/*.json"))
EOF
}
apply_cuts() {  # apply_cuts <queue file>: the todo's G cut rule (G = orig / ghost / shift / swap without X), budget BUDGET_WH
    local qf=$1 rem dg dk g k x total nw
    rem=$(project "$qf" | awk '/^REMAINING_WH/ {print $3, $5, $7}')
    read -r g k x <<< "$rem"
    read -r dg dk <<< "$(done_wh)"
    total=$(python3 -c "print(round($g + $dg, 1))")
    nw=$(( $(wc -w <<< "$GPUS") * WORKERS ))
    printf 'G %s (done %s) | K %s (done %s) | X %s worker-h remaining; all remaining at %d workers: %.1f h\n' \
        "$total" "$dg" "$k" "$dk" "$x" "$nw" "$(python3 -c "print(($g + $k + $x) / $nw)")" > "$G/PROJECTED_WH"
    # cut only on measured costs: every G examinee left in the queue has >= 5 finished routes, or the queue has reached
    # the shift / swap steps (by then every examinee has run its ghost step)
    local unmeasured="" cc
    for cc in $(awk '$2 != "k" && $1 != "x" {print $1}' "$qf" | sort -u); do [[ -z $(measured_wmin "$cc") ]] && unmeasured+=" $cc"; done
    if [[ -n $unmeasured && ${2:-} != force ]]; then
        echo "(priors for:$unmeasured; no cut before they are measured)" >> "$G/PROJECTED_WH"
        return 0
    fi
    if python3 -c "import sys; sys.exit(0 if $total > $BUDGET_WH else 1)"; then
        if grep -q ' swap ' "$qf"; then
            grep -v ' swap ' "$qf" > "$qf.tmp" && mv "$qf.tmp" "$qf"
            ev cut "\"what\": \"swap\", \"projected_G_wh\": $total"; log "projected G $total worker-h > $BUDGET_WH: swap steps cut"
            apply_cuts "$qf" force; return
        fi
        if grep -E ' shift ' "$qf" | grep -qvE '^(tfv6|bridgedrive|blue|mc) '; then
            grep -vE '^(pdm|simlingo|cinque|q2|k3seen) shift ' "$qf" > "$qf.tmp" && mv "$qf.tmp" "$qf"
            ev cut "\"what\": \"shift to tfv6 / bridgedrive / blue / mc\", \"projected_G_wh\": $total"
            log "projected G $total worker-h > $BUDGET_WH: shift cut to tfv6 / bridgedrive / blue / mc"
            apply_cuts "$qf" force; return
        fi
        [[ -e $G/OVER_BUDGET ]] || { echo "$(date '+%F %T') projected G $total worker-h after both cuts" > "$G/OVER_BUDGET"
                                     ev over_budget "\"projected_G_wh\": $total"
                                     log "G still over budget after both cuts ($total worker-h): the queue runs on in priority order; main decides"; }
    fi
}

status_loop() {
    while sleep 600; do
        {
            echo "# nq4-gk status $(date '+%F %T %Z')"; echo
            echo "- batch: $( [[ -f $G/GO ]] && echo "GO ($(tr '\n' ' ' < "$G/GO"))" || echo 'waiting for runs/nq4/gk/GO')"
            echo "- pilot card: $( [[ -f $SCHED/nq4-gk.pilot ]] && tr '\n' ' ' < "$SCHED/nq4-gk.pilot" || echo 'waiting for runs/sched/nq4-gk.pilot')"
            echo "- current batch step: $(cat "$G/CURRENT" 2>/dev/null)"
            echo "- pilots passed: $(ls "$PIL"/*/PASS 2>/dev/null | wc -l); blocked examinees: $(ls "$BLK" 2>/dev/null | tr '\n' ' ')"
            echo "- projection: $(cat "$G/PROJECTED_WH" 2>/dev/null) (G budget $BUDGET_WH)$( [[ -e $G/OVER_BUDGET ]] && echo '; G OVER BUDGET after both cuts')"
            echo "- steps left: $(grep -c . "$G/QUEUE" 2>/dev/null); pids.current $(cat /sys/fs/cgroup/pids.current), load $(cut -d' ' -f1-3 /proc/loadavg)"
            echo; echo '```'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo '```'
            echo; echo "finished seeds:"
            for f in "$G"/arms_k/*/*/DONE "$G"/arms/*/*/*/DONE; do [[ -e $f ]] && echo "- $(cat "$f")"; done
        } > "$G/STATUS.md.tmp" && mv "$G/STATUS.md.tmp" "$G/STATUS.md"
    done
}
step_skip() {  # a step is settled: every seed dir has DONE, or its examinee is blocked
    [[ -e $BLK/$1 ]] || step_done "$1" "$2" "$3"
}
pilot_loop() {  # on the validation card: the staged pilot of every examinee x world, in queue order, ahead of the batch
    trap 'for o in $CUR_OUTS; do kill_runs "$o"; done; srv_stop_gpus "$GPUS"' EXIT
    trap 'exit 129' HUP INT TERM
    until load_pilot; do [[ -e $G/DONE ]] && return 0; sleep 120; done
    log "pilot loop on GPU $GPUS ($WORKERS workers, indices from $SIDX0)"
    local c v s rs pending ids
    while :; do
        pending=0
        while read -r c v s rs; do
            [[ -e $PIL/$c.$v/PASS || -e $BLK/$c || -d $PIL/$c.$v ]] && continue
            ready_for "$c" || { pending=1; continue; }
            ids=$(step_ids "$c" "$v" "${s%%,*}" "$rs") || continue
            [[ -z $ids ]] && continue
            load_pilot
            pilot "$c" "$v" "${s%%,*}" "$ids" < /dev/null
            pending=1; break
        done < "$G/QUEUE"
        (( pending )) || { log "pilot loop: every examinee x world is piloted"; return 0; }
        [[ -e $G/DONE ]] && return 0
        sleep 60
    done
}

chain() {
    echo "waiting for runs/nq4/gk/GO" > "$G/CURRENT"
    status_loop & local st=$!
    [[ -e $G/prep/DONE ]] || error "G-prep has not passed (runs/nq4/gk/prep/DONE missing)"
    [[ -f $G/QUEUE ]] || queue > "$G/QUEUE"
    pilot_loop & local pl=$!
    echo "$pl" > "$G/pilot_loop.pid"
    trap 'kill $st 2>/dev/null; kill $pl 2>/dev/null; for o in $CUR_OUTS; do kill_runs "$o"; done; srv_stop_gpus "$GPUS"' EXIT
    trap 'exit 129' HUP INT TERM
    local said=0
    until load_go; do (( said++ == 0 )) && log "waiting for the GO file $G/GO"; sleep 120; done
    ev go "\"gpus\": \"$GPUS\", \"workers\": $WORKERS, \"idx0\": $SIDX0, \"span\": $SPAN, \"cpus\": \"$CPUS\""
    log "GO: GPUs $GPUS, $WORKERS workers each, server indices from $SIDX0 (span $SPAN), cores $CPUS"
    apply_cuts "$G/QUEUE"
    local c v s rs pass=0 ran r
    while :; do
        load_go
        ran=0
        while read -r c v s rs; do
            step_skip "$c" "$v" "$s" && continue
            ready_for "$c" || continue
            if [[ $v == shift || $v == swap ]] && [[ ! -e $G/CUT_CHECKED ]]; then   # last chance for the cut rule
                touch "$G/CUT_CHECKED"; apply_cuts "$G/QUEUE" force; ran=1; break
            fi
            run_step "$c" "$v" "$s" "$rs" < /dev/null; r=$?
            (( r == 2 )) && continue                         # being piloted on the validation card / no slot: next step
            ran=1
            apply_cuts "$G/QUEUE"
            break                                            # re-read the (possibly cut) queue from the top
        done < "$G/QUEUE"
        (( ran )) && continue
        local left; left=$(while read -r c v s rs; do step_skip "$c" "$v" "$s" || echo x; done < "$G/QUEUE" | wc -l)
        (( left == 0 )) && break
        (( pass++ == 0 )) && log "$left steps wait for inputs (fold heads / K READY) or for their pilot; polling every 5 min"
        sleep 300
    done
    date '+%F %T' > "$G/DONE"
    ev end '"what": "nq4 G + K + X queue done"'
    log "queue done"
}

case ${1:-} in
    chain) chain ;;
    step) shift; trap 'for o in $CUR_OUTS; do kill_runs "$o"; done; srv_stop_gpus "$GPUS"' EXIT; trap 'exit 129' HUP INT TERM
          run_step "$@" ;;
    pilot) shift; trap 'for o in $CUR_OUTS; do kill_runs "$o"; done; srv_stop_gpus "$GPUS"' EXIT; trap 'exit 129' HUP INT TERM
           pilot "$1" "$2" "$3" "$(step_ids "$1" "$2" "$3" "$4")" ;;
    report) report_now ;;
    plan) queue > "$G/plan.tmp"; project "$G/plan.tmp" ;;
    *) sed -n 2,26p "$0"; exit 1 ;;
esac
