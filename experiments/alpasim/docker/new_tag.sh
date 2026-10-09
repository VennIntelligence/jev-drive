#!/usr/bin/env bash
# One command from "a new checkpoint exists" to "a tested submission image": thin-layer rebuild -> constraint tests -> 48-scene smoke in
# the official containerised stack -> comparison with the native reference, as one self-advancing chain. Run it on the Docker host
# (Tokyo box) after the checkpoint is there (experiments/alpasim/scripts/tokyo_pull.sh ckpt <tag>; `ref` brings the native results):
#   scripts/tmux_run.sh img-<tag> env DATA_DIR=/data experiments/alpasim/docker/new_tag.sh <tag> [native results-summary.json]
# <tag> = a run tag under $DATA_DIR/runs/op_parity/runs (tagA+tagB = the ensemble). The driver family comes from the checkpoint
# (serve.py); DRIVER=sh30|ap2|ens forces it. A tag outside build.sh's stable set goes into the thin extra layer, and the image's default
# becomes <tag>: the image this prints is the one to push. Nothing here pushes, logs in or submits.
# State: $DATA_DIR/runs/alpasim/tokyo_image/<tag>-<ts>/{STATUS, DONE | ERROR, log.txt, report.md, IMAGE, build/, test/, smoke/}.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
TAG=$1 REF=${2:-}
: "${DATA_DIR:?DATA_DIR}"
O=${OUT:-$DATA_DIR/runs/alpasim/tokyo_image/$(echo "$TAG" | tr '+' '_')-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$O"; rm -f "$O"/{DONE,ERROR}; exec > >(tee -a "$O/log.txt") 2>&1
st() { echo "$(date '+%F %T') $TAG: $*" | tee "$O/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
stable=" ${TAGS:-P2H10-F-s0 AP2-AB-s0} "; extra=
for t in ${TAG//+/ }; do
  [[ -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "no checkpoint for $t on this host: tokyo_pull.sh ckpt $t"
  [[ $stable == *" $t "* ]] || extra+="$t "
done
NR=$DATA_DIR/runs/alpasim/native_ref/$TAG/aggregate/results-summary.json; [[ -z $REF && -f $NR ]] && REF=$NR   # native 48-scene run of this tag, when one was brought

st "build (extra layer: ${extra:-none})"
EXTRA="${EXTRA:-} $extra" TAG=$TAG OUT=$O/build bash "$here/build.sh" || die "build"
IMG=$(cat "$O/build/IMAGE"); echo "$IMG" > "$O/IMAGE"
st "constraint tests: $IMG"
bash "$here/test.sh" "$IMG" "$O/test" "" "${TGPU:-1}" || die "constraint tests, see test/result.json"
st "48-scene smoke in the official stack${REF:+, reference $REF}"
bash "$here/smoke.sh" "$IMG" "$O/smoke" "" "$REF" || die "smoke, see smoke/log.txt"
python3 - "$O" "$IMG" <<'PY' > "$O/report.md" || die "report"
import json, sys
o, img = sys.argv[1:]
t, s = json.load(open(f"{o}/test/result.json")), json.load(open(f"{o}/smoke/smoke.json"))
p, d, r = t["probe"]["drive_ms"], s.get("driver", {}), s.get("ref")
ok = lambda b: "pass" if b else "FAIL"
print(f"# {s['label']}: {img}\n")
print("| check | value | limit | |\n|---|--:|--:|---|")
print(f"| image size | {t['image_bytes'] / 2**30:.2f} GiB | 40 GiB | {ok(t['image_bytes'] < 40 * 2**30)} |")
print(f"| read-only root, no network, uid 10001 | {t['read_only_errors']} write errors, {t['docker_diff_lines']} changed files | 0 | {ok(not t['read_only_errors'] and not t['docker_diff_lines'])} |")
print(f"| peak GPU memory, 2 concurrent rollouts | {t['gpu_mib']['driver_peak'] / 1024:.2f} GiB | 16 GiB | {ok(t['gpu_mib']['driver_peak'] < 16384)} |")
print(f"| /tmp high-water (probe / 48-scene smoke) | {t['tmp_peak_bytes'] / 2**20:.1f} / {int(s.get('tmp_peak_bytes', 0)) / 2**20:.1f} MiB | 2048 MiB | {ok(t['tmp_peak_bytes'] < 2**31)} |")
print(f"| /run high-water | {t['run_peak_bytes'] / 2**20:.2f} MiB | 64 MiB | {ok(t['run_peak_bytes'] < 2**26)} |")
print(f"| cold start until the service answers | {t['cold_start_s']} s | - | |")
print(f"| `drive`, 2 concurrent synthetic rollouts ({t['gpu']}) | p50 {p['p50']} / p90 {p['p90']} / p99 {p['p99']} / max {p['max']} ms | target 100 ms | |")
if d:
    print(f"| `drive`, 48-scene smoke at 8 concurrent rollouts (server side) | p50 {d['total_ms']['p50']} / p90 {d['total_ms']['p90']} / max {d['total_ms']['max']} ms | | {d['inference_errors']} inference errors |")
print(f"| 48 scenes, containerised | mean {s['run']['mean']:.4f}, zeros {s['run']['zeros']}, at 1 {s['run']['ones']} (n {s['run']['n']}) | | |")
if r:
    print(f"| 48 scenes, native reference | mean {r['mean']:.4f}, zeros {r['zeros']}, at 1 {r['ones']} | | {s['differ']} of {s['common']} scenes differ, max {s['max_abs_diff']} |")
print("\n" + open(f"{o}/smoke/compare.md").read())
PY
cat "$O/report.md"; date > "$O/DONE"; st "done: $IMG, report $O/report.md"
