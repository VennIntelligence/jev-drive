#!/bin/bash
# Training throughput probe (docs/speed.md, "Headroom"): rm_train on the RSG-L18 recipe (wait_cap/scripts/nx.sh $RSG), short runs timed per
# step by train_probe.py. Pool job bodies:
#   train_probe.sh seq <steps>       one card, one run after another: base twice (determinism), --fetch-workers 4 --prefetch 8, nosync (a
#                                    patched copy: losses logged without a host sync per term, row indices uploaded from pinned memory),
#                                    profiled base, torch.compile of the training forward (patched copy), --batch 256; then bitcmp of the
#                                    identity candidates against base-a. Checkpoints are deleted at the end (the summaries stay).
#   train_probe.sh multi <n> <steps> n copies of base at once on one card
# Out: $DATA_DIR/runs/speed/train_probe/<mode>-<time>/ (<tag>.json per run, smi.csv = card util / memory every 2 s, bitcmp_*.json).
set -u
cd "$(dirname "$0")/../../.." || exit 1
R=$PWD; TR=$DATA_DIR/envs/op-train/bin/python; RT=experiments/route_mem/scripts/rm_train.py; PR=$R/experiments/cl_infra/scripts/train_probe.py
RSG="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 \
--on-split alpasim_nuplan/onpol-train --lr-scale 0.1 --on on1_L18RB0 --on-k 13 --on-mode self"
MODE=${1:?seq|multi}; O=$DATA_DIR/runs/speed/train_probe/$MODE-$(date +%m%d-%H%M%S); mkdir -p "$O"; P=SPEEDP-$(date +%H%M%S)
( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}"),$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | wc -l)"; sleep 2; done >> "$O/smi.csv" ) &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT
run() {   # run <tree> <tag> <wrapper flags> [rm_train flags ...]
  local tree=$1 tag=$2 wf=$3; shift 3
  $TR "$PR" --out "$O/$tag.json" $wf -- "$tree/$RT" train --tag "$tag" $RSG --steps "$N" --seed 0 "$@" > "$O/$tag.log" 2>&1
  echo "$tag rc $? $(cat "$O/$tag.json" 2>/dev/null | tr -d '\n ' | cut -c1-200)"
}
patched() {   # patched <dir> <kind>: a worktree of HEAD with rm_train edited (nosync | compile)
  [[ -d $1 ]] || git worktree add --detach "$1" HEAD > /dev/null || exit 1
  $TR - "$1/$RT" "$2" <<'EOF'
import sys, pathlib
p, kind = pathlib.Path(sys.argv[1]), sys.argv[2]
s = p.read_text()
rep = {"nosync": [("hist.append({q: float(v) for q, v in Ls.items()})",
                   "hist.append({q: v.detach() if torch.is_tensor(v) else v for q, v in Ls.items()})"),
                  ("mm = {q: float(np.mean([h[q] for h in hist if q in h])) for q in keys}",
                   "mm = {q: float(np.mean([float(h[q]) for h in hist if q in h])) for q in keys}"),
                  ("rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)",
                   "rows, anchor = (torch.from_numpy(np.ascontiguousarray(x)).pin_memory().to(dev, non_blocking=True) for x in (r, an))"),
                  ("keep = torch.as_tensor(mrng.random(len(r)) >= a.rm_drop, device=dev) | is_pool[rows]",
                   "keep = torch.from_numpy(mrng.random(len(r)) >= a.rm_drop).pin_memory().to(dev, non_blocking=True) | is_pool[rows]")],
       "compile": [("out = model(front, ego, S.tc[rows], None, None, nv=None if ns.nv is None else ns.nv[rows], desire=dq)",
                    "out = CMP(front, ego, S.tc[rows], None, None, nv=None if ns.nv is None else ns.nv[rows], desire=dq)"),
                   ("        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)\n",
                    "        CMP = torch.compile(model, dynamic=False)\n        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)\n")]}[kind]
for x, y in rep:
    assert s.count(x) == 1, (kind, x)
    s = s.replace(x, y)
p.write_text(s)
EOF
}
case $MODE in
seq)
  N=${2:-300}; WN=$DATA_DIR/tmp/speedp_nosync_$P; WC=$DATA_DIR/tmp/speedp_compile_$P
  patched "$WN" nosync; patched "$WC" compile
  run "$R" $P-base-a ""; run "$R" $P-base-b ""; run "$R" $P-fw4 "" --fetch-workers 4 --prefetch 8; run "$WN" $P-nosync ""
  run "$R" $P-prof "--prof"; run "$WC" $P-compile ""; run "$R" $P-bs256 "" --batch 256
  for b in base-b fw4 nosync compile; do
    $TR experiments/wait_cap/scripts/nx.py bitcmp --a $P-base-a --b $P-$b --out speedp_bitcmp_$b > "$O/bitcmp_$b.log" 2>&1; echo "bitcmp base-a vs $b rc $?"
    cp "$DATA_DIR/runs/wait_cap/next/speedp_bitcmp_$b.json" "$O/" 2>/dev/null
  done
  git worktree remove --force "$WN"; git worktree remove --force "$WC" ;;
multi)
  K=${2:?n}; N=${3:-300}; pids=()
  for i in $(seq 1 "$K"); do run "$R" $P-m$K-$i "" & pids+=($!); done
  wait "${pids[@]}" ;;
esac
for d in "$DATA_DIR"/runs/op_parity/runs/$P-*; do rm -f "$d"/ckpt-final.pt "$d"/adapter.pt; done
echo "out $O"
