#!/usr/bin/env bash
# Superseded by jevdrive.cl (2026-10-01): scripts/lanes/cl_worker_profile.py --arg stage=old reproduces this lane on the
# library (20/20 routes, DS differences only on the known flaky routes 1956 / 3564 / 17563; todos/2026-10-01-cl-lib.md).
# Kept for the record; new runs go through `python -m jevdrive.cl run`.
# Route-level equivalence of reduced CARLA thread pools (docs/carla.md, "Threads per server"): PDM-Lite (SimLingo's
# expert, as the 2026-09-25 controller acceptance ran it: runs/infra-accept/b2d-ctl/expert-a, expert-b) on the same 20
# routes, once per arm, one arm after another on the same card and cores. Arms: "base" (stock pools) or "reduced"
# (-RPCThreads=4 -StreamingThreads=4 -SecondaryThreads=4). Compare with `carla_threads.py compare`.
#   scripts/carla_threads_routes.sh <out-root> <gpu> <server-index> <cpus> <arm>[:<name>] ...
#   e.g. scripts/carla_threads_routes.sh $DATA_DIR/runs/infra/carla-threads/routes 1 210 110-117,200-207 base:base-c reduced:red-a
set -euo pipefail
(( $# >= 5 )) || { sed -n 2,7p "$0"; exit 1; }
root=$1 gpu=$2 idx=$3 cpus=$4; shift 4
SIM=$DATA_DIR/third_party/simlingo
ROUTES=2390,24211,1711,2373,3564,1833,1852,1956,2668,4183,11381,1825,2084,2086,2091,2115,2286,24330,17563,26458
for spec in "$@"; do
    arm=${spec%%:*} name=${spec#*:}
    case $arm in base) sargs="" ;; reduced) sargs="-RPCThreads=4 -StreamingThreads=4 -SecondaryThreads=4" ;;
                 *) echo "unknown arm $arm" >&2; exit 1 ;; esac
    echo "== $name ($arm) $(date +%T)"
    BENCH2DRIVE_ROOT=$SIM/Bench2Drive WORK_DIR=$SIM taskset -c "$cpus" "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py \
        --routes "$SIM/leaderboard/data/bench2drive220.xml" --route-ids "$ROUTES" --out "$root/$name" --workers 5 \
        --server-index "$idx" --index-span 10 --gpu-rank "$gpu" --tm-seed 0 --no-spectator --no-reap --client-threads 8 \
        --max-attempts 2 --stall-s 480 --route-timeout-s 3600 --python "$DATA_DIR/envs/simlingo/bin/python" \
        --agent scripts/b2d_expert_agent.py --agent-config "expert+$name" --server-args "$sargs"
done
echo "== all arms done $(date +%T)"
