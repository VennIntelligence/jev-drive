"""P3 insertion readout (amendment 2, option 3): stage the insertion items as a P3 set and run the registered readout.

Every item of runs/nq4/p3/insert_batch/<key> with a chosen donor becomes one scene per variant v in processed/nq4_p3_insert/
scenes/<key><v>: real = the log, plus = the variant (ins2 / ins3 / ins4 = donor on the lane centre at TTR 2 / 3 / 4 s,
null = donor 4.5 m off the lane; <v>f = the grounded versions), minus = the donor hidden. Frames are the 10 Hz renders
thinned to 5 Hz on the t* grid (t* - 3 s .. t* + 2 s, 26 frames, as the deletion pairs), images linked, not copied.
Then the registered chain with P3_SET = P5_SET = nq4_p3_insert: index -> openpilot temporal -> finalize -> exam
(`ridge_late` Cinque / Lebowski at the I3 tau). The summary splits pair flips (plus vs minus) by variant: for ins* they
are hits (the donor is in the lane), for null* they are false flips (insertion null gate: pooled <= 7 %).

  CUDA_VISIBLE_DEVICES=6 python3 scripts/p3/insert_readout.py [--stage-only]
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

DATA = Path(os.environ["DATA_DIR"])
REPO = Path(__file__).resolve().parents[2]
I = DATA / "runs/nq4/p3/insert_batch"
SET = "nq4_p3_insert"
OUT = DATA / "processed" / SET
VARS = ("ins2", "ins3", "ins4", "null", "ins2f", "ins3f", "ins4f", "nullf")


def stage():
    n = 0
    for md in sorted(I.glob("p3_*/meta.json")):
        m = json.loads(md.read_text())
        if not m.get("chosen"):
            continue
        key, ts, src = m["scene"], m["t_star"], md.parent
        ref = json.loads((DATA / "processed/nq4_p3/scenes" / key / "meta.json").read_text())   # calibration of the scene
        keep = [t for t in m["frames"] if (t - ts) % 2 == 0]
        for v in VARS:
            if not (src / v / "frames.jsonl").exists():
                continue
            sd = OUT / "scenes" / f"{key}{v}"
            for w, from_ in (("real", "real"), ("plus", v), ("minus", "minus")):
                (sd / w).mkdir(parents=True, exist_ok=True)
                if not (sd / w / "cams").exists():
                    (sd / w / "cams").symlink_to(src / from_ / "cams")
                rows = [json.loads(x) for x in open(src / from_ / "frames.jsonl")]
                (sd / w / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["frame"] // 2 in keep))
            meta = {"scene": int(key[3:]), "key": f"{key}{v}", "segment": m["segment"], "f0": ts, "variant": v,
                    "calib": ref["calib"], "extrinsics_cam_to_ego": ref["extrinsics_cam_to_ego"], "donor": m["chosen"],
                    "diagnostics": m.get("diagnostics", {})}
            (sd / "meta.json").write_text(json.dumps(meta, indent=1))
            n += 1
    print(json.dumps({"staged_scenes": n}))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    a = ap.parse_args()
    if not stage() or a.stage_only:
        return
    env = dict(os.environ, P3_SET=SET, P5_SET=SET, P3_HEAD_TOL=os.environ.get("P3_HEAD_TOL", "0.5"))   # new box, see nq4_p3.exam
    jv, op = str(DATA / "envs/jevdrive/bin/python"), str(DATA / "envs/openpilot/bin/python")
    for argv in ([jv, "-m", "jevdrive.nq4_p3", "index", "--processed-root", str(DATA / "processed/waymo_ds/training")],
                 [op, "scripts/p5_openpilot.py", "--models", "cinque", "lebowski", "--workers", "1"],
                 [jv, "-m", "jevdrive.p5_openpilot", "finalize", "--models", "cinque,lebowski"],
                 [jv, "-m", "jevdrive.nq4_p3", "exam"]):
        rc = subprocess.call(argv, cwd=REPO, env=env)
        if rc:
            raise SystemExit(f"{argv[1:4]} rc={rc}")


if __name__ == "__main__":
    main()
