"""Staged-launch checklist of an ONPOL-PAI rollout stage (prereg section 6): every scene has a result, no driver error, 190 to 201 decisions
per session, a dump per second and a rollout.asl per rolled scene. Reads completion only, no scores. Plain python3, Tokyo box.

  onp_pass_check.py <stage dir of pai_eval.sh> <scenes expected>        exit 1 if any item fails
"""
import glob
import json
import sys

d, n = sys.argv[1], int(sys.argv[2])
bad, scenes, close = [], set(), []
for r in sorted(glob.glob(d + "/runs/*/")):
    if not glob.glob(r + "DONE"):
        continue
    scenes |= {x["clipgt_id"] for x in json.load(open(r + "sim/aggregate/results-summary.json"))["rollouts"]}
    close += [json.loads(line) for line in open(r + "driver/drive.jsonl") if '"close"' in line]
    rolled = len(glob.glob(r + "sim/rollouts/*/"))
    dumps, asl = len(glob.glob(r + "driver/dump/*.npz")), len({p.split("/rollouts/")[1].split("/")[0] for p in glob.glob(r + "sim/rollouts/*/*/rollout.asl")})
    if dumps < 19 * rolled:
        bad.append(f"{r}: {dumps} dumps for {rolled} scenes")
    if asl < rolled:
        bad.append(f"{r}: rollout.asl for {asl} of {rolled} scenes")
err = sum(c["inference_error"] + c["input_error"] for c in close)
short = [c["scene"] for c in close if not 190 <= c["drive"] <= 201]
if len(scenes) != n:
    bad.append(f"{len(scenes)} of {n} scenes with a result")
if err:
    bad.append(f"{err} driver errors")
print(f"{d}: {len(scenes)}/{n} results, {len(close)} sessions, driver errors {err}, sessions outside 190..201 decisions: {len(short)} {short[:5]}, problems: {bad}")
sys.exit(1 if bad else 0)
