"""R3D2 sensitivity check of the pedestrian dose-response exam (fc65452:todos/2026-09-28-ped-dose-response.md, "R3D2 敏感性检查",
registered before any number): does harmonising the inserted pedestrian with R3D2-big change openpilot's readouts?

  select   (jevdrive env, CPU) 10 inserted cells from the stage-2 readout: sorted by the native-plan dv2 = v2(x+) - v2(x-)
           at t* (Cinque, the primary examinee), the cell nearest each of 10 equally spaced quantiles (strongest reaction
           and no reaction included), at most 2 per scene; -> runs/nq4/p3/dose/r3d2/selected.json
  render   (r3d2 env, GPU) per selected cell and frame: the front camera of (b) (the dose render) through R3D2-big on the
           2x-upsampled 1080-row window, pasted back inside the edit region only (the actor box, from |x+ - x-| > 15 on any
           channel, widened by 0.8 h left / right, 0.45 h down, 0.1 h up; 4 px feather) as experiments/p3_ped_exam/archive/ins_tools.r3d2, the 1080-row
           window centred on each frame's region.
           Side cameras and x- are untouched.  -> r3d2/items/<scene_state>/<cell>/plus/{cams,frames.jsonl}
  stage    (CPU) the P3 set nq4_p3_dose_r3d2: same scene keys, meta, log and x- as nq4_p3_dose, plus = the R3D2 version
  compare  (jevdrive env, CPU) paired readouts, Delta = readout((b)+R3D2) - readout((b)), and the registered reading:
           |Delta v2| >= tau in <= (appearance-null false-flip rate on these scenes + 10 pp) of the cells and the stop
           call agreeing in >= 9 / 10 -> "shadow and realism do not drive the result"; otherwise reported as a finding.
           -> r3d2/compare.csv, r3d2/verdict.json

The readout chain between stage and compare is the registered one (index -> openpilot -> finalize -> exam -> ped_dose), run
by runs/nq4/p3/dose/r3d2/run.sh (written by `chain`).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(os.environ["DATA_DIR"])
DO = DATA / "runs/nq4/p3/dose"
RD = Path(os.environ.get("DOSE_R3D2_DIR", DO / "r3d2"))    # override for a dry run
THR, W, H = 15, 960, 640
N_SEL, PER_SCENE = 10, 2


def select(a):
    import pandas as pd
    w = pd.read_csv(a.cells)
    w = w[(w.t_rel_f0.round(1) == 0.0)].dropna(subset=["cinque|dv2"]).sort_values("cinque|dv2").reset_index(drop=True)
    w["scene_state"] = w.base_id.str.rsplit("_", n=3).str[0]
    qs = np.quantile(w["cinque|dv2"], np.linspace(0, 1, a.n))
    picked, per = [], {}
    for q in qs:
        order = (w["cinque|dv2"] - q).abs().sort_values().index
        for i in order:
            r = w.loc[i]
            if r.base_id in {p["key"] for p in picked} or per.get(r.scene, 0) >= PER_SCENE:
                continue
            picked.append({"key": r.base_id, "scene": int(r.scene), "state": r.state, "cell": r.cell, "quantile_target": float(q),
                           "cinque_dv2": float(r["cinque|dv2"]), "lebowski_dv2": float(r["lebowski|dv2"])})
            per[int(r.scene)] = per.get(int(r.scene), 0) + 1
            break
    RD.mkdir(parents=True, exist_ok=True)
    (RD / "selected.json").write_text(json.dumps(picked, indent=1))
    print(json.dumps({"selected": len(picked), "dv2": [round(p["cinque_dv2"], 2) for p in picked]}))


def _src(key):
    """(item dir of the scene x state, cell dir) of a staged key p3_<k>_<state>_<cell>."""
    k, st, cell = key.split("_", 3)[1], key.split("_", 3)[2], key.split("_", 3)[3]
    sd = DO / "items" / f"p3_{k}_{st}"
    return sd, sd / cell


def render(a):
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image
    from scipy.ndimage import binary_opening, gaussian_filter
    sel = json.loads((RD / "selected.json").read_text())
    mp = str(DATA / "models/r3d2" / a.model)
    pipe = DiffusionPipeline.from_pretrained(mp, custom_pipeline=mp, trust_remote_code=True, torch_dtype=torch.float32).to("cuda")

    def soft(reg):
        m = np.zeros((H, W), np.float32)
        x0, y0, x1, y1 = reg
        m[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = 1
        return np.clip(gaussian_filter(m, 4.0), 0, 1)[..., None]

    for p in sel:
        sd, cd = _src(p["key"])
        od = RD / "items" / sd.name / cd.name / "plus"
        if (od / "frames.jsonl").exists():
            continue
        t0, n_edit = time.time(), 0
        for c in ("front_left", "front_right"):
            (od / "cams").mkdir(parents=True, exist_ok=True)
            if not (od / "cams" / c).exists():
                (od / "cams" / c).symlink_to(cd / "plus/cams" / c)
        (od / "cams/front").mkdir(parents=True, exist_ok=True)
        for f in sorted((cd / "plus/cams/front").glob("*.jpg")):
            inp = np.asarray(Image.open(f).convert("RGB"))
            mi = np.asarray(Image.open(sd / "minus/cams/front" / f.name).convert("RGB"))
            m = binary_opening(np.abs(inp.astype(int) - mi.astype(int)).max(-1) > THR, iterations=1)
            out = inp
            if m.sum() >= 30:
                ys, xs = np.nonzero(m)
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                h = y1 - y0
                reg = (max(int(x0 - 0.8 * h), 0), max(int(y0 - 0.1 * h), 0), min(int(x1 + 0.8 * h) + 1, W), min(int(y1 + 0.45 * h) + 1, H))
                up = np.asarray(Image.fromarray(inp).resize((2 * W, 2 * H), Image.BICUBIC))
                wy0 = int(np.clip(reg[1] + reg[3] - 540, 0, 2 * H - 1080))
                with torch.no_grad():
                    y = pipe(Image.fromarray(up[wy0:wy0 + 1080])).images[0]
                full = up.copy()
                full[wy0:wy0 + 1080] = np.asarray(y.convert("RGB").resize((2 * W, 1080), Image.BICUBIC))
                raw = np.asarray(Image.fromarray(full).resize((W, H), Image.BICUBIC)).astype(np.float32)
                wgt = soft(reg)
                out = np.clip(inp * (1 - wgt) + raw * wgt + 0.5, 0, 255).astype(np.uint8)
                n_edit += 1
            Image.fromarray(out).save(od / "cams/front" / f.name, quality=95)
        (od / "frames.jsonl").write_text((cd / "plus/frames.jsonl").read_text())
        print(json.dumps({"key": p["key"], "frames_edited": n_edit, "s": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"done": len(sel)}))


def stage(a):
    """processed/nq4_p3_dose_r3d2/scenes/<key>: meta and frames of the nq4_p3_dose item, plus -> the R3D2 version."""
    sel = json.loads((RD / "selected.json").read_text())
    OUT = DATA / "processed/nq4_p3_dose_r3d2/scenes"
    for p in sel:
        sd, cd = _src(p["key"])
        dst = OUT / p["key"]
        for w, src in (("real", sd / "real"), ("plus", RD / "items" / sd.name / cd.name / "plus"), ("minus", sd / "minus")):
            (dst / w).mkdir(parents=True, exist_ok=True)
            if not (dst / w / "cams").exists():
                (dst / w / "cams").symlink_to(src / "cams")
            (dst / w / "frames.jsonl").write_text((src / "frames.jsonl").read_text())
        ref = DATA / "processed/nq4_p3_dose/scenes" / p["key"] / "meta.json"
        (dst / "meta.json").write_text(ref.read_text())
    print(json.dumps({"staged": len(sel)}))


def compare(a):
    import pandas as pd
    from jevdrive.common import data_dir
    from experiments.p3_ped_exam.archive.nq4_p3 import I3_EXAM
    b = pd.read_csv(a.cells)
    r = pd.read_csv(RD / "readout/cells.csv")
    keys = ["base_id", "t_rel_f0"]
    b["t_rel_f0"], r["t_rel_f0"] = b.t_rel_f0.round(1), r.t_rel_f0.round(1)
    j = r.merge(b, on=keys, suffixes=("_r", "_b"))
    taus = pd.read_csv(data_dir() / I3_EXAM / "flip_rates.csv").query("scope == 'pooled'").set_index("examinee").tau_model
    rows, verdict = [], {}
    for m in ("cinque", "lebowski"):
        tau = float(taus[f"ridge_late op-{m} temporal"])
        d = pd.DataFrame({"base_id": j.base_id, "t_rel_f0": j.t_rel_f0, "model": m,
                          "dv2": j[f"{m}|v2_plus_r"] - j[f"{m}|v2_plus_b"], "dvmin": j[f"{m}|vmin_plus_r"] - j[f"{m}|vmin_plus_b"],
                          "dlead_p": j[f"{m}|lead_p_plus_r"] - j[f"{m}|lead_p_plus_b"],
                          "d_rl_v2": j[f"rl_{m}|v2_plus_r"] - j[f"rl_{m}|v2_plus_b"],
                          "stop_b": j[f"{m}|stop_b"], "stop_r": j[f"{m}|stop_r"], "react_b": j[f"{m}|react_b"], "react_r": j[f"{m}|react_r"],
                          "lead_b": j[f"{m}|lead_b"], "lead_r": j[f"{m}|lead_r"]})
        rows.append(d)
        d0 = d[d.t_rel_f0 == 0.0]
        scenes = set(j[j.t_rel_f0 == 0.0].base_id.str.rsplit("_", n=3).str[0])
        bs = b[(b.t_rel_f0 == 0.0) & b.base_id.str.rsplit("_", n=3).str[0].isin(scenes)]
        null = float(bs.groupby(bs.base_id.str.rsplit("_", n=3).str[0])[f"{m}|null_flip"].first().mean())
        moved = float((d0.dv2.abs() >= tau).mean())
        agree = int((d0.stop_b.astype(bool) == d0.stop_r.astype(bool)).sum())
        verdict[m] = {"tau": tau, "n": len(d0), "moved_frac": moved, "null_false_flip": null, "stop_agree": agree,
                      "react_agree": int((d0.react_b.astype(bool) == d0.react_r.astype(bool)).sum()),
                      "median_abs_dv2": float(d0.dv2.abs().median()), "max_abs_dv2": float(d0.dv2.abs().max()),
                      "passes": bool(moved <= null + 0.10 and agree >= 9)}
    pd.concat(rows).to_csv(RD / "compare.csv", index=False)
    (RD / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


CHAIN = r"""#!/usr/bin/env bash
# R3D2 sensitivity chain (experiments/p3_ped_exam/archive/dose_r3d2.py): waits for the stage-2 dose readout, then select -> R3D2 -> stage ->
# index -> openpilot -> finalize -> exam -> ped_dose -> compare. Writes r3d2/DONE or r3d2/ERROR.
set -euo pipefail
cd ~/data/jev-drive
RD=$DATA_DIR/runs/nq4/p3/dose/r3d2; DO=$DATA_DIR/runs/nq4/p3/dose
trap 'echo "failed at line $LINENO" > $RD/ERROR' ERR
rm -f $RD/DONE $RD/ERROR
until [ -f $DO/STAGE2_DONE ] && [ -f $DO/readout_stage2/cells.csv ]; do [ -f $DO/STAGE2_ERROR ] && { echo "stage 2 failed" > $RD/ERROR; exit 1; }; sleep 120; done
PY=$DATA_DIR/envs/drivestudio/bin/python; JV=$DATA_DIR/envs/jevdrive/bin/python; OP=$DATA_DIR/envs/openpilot/bin/python
RP=$DATA_DIR/envs/r3d2/bin/python
CELLS=$DO/readout_stage2/cells.csv
$JV experiments/p3_ped_exam/archive/dose_r3d2.py select --cells $CELLS
until [ $(nvidia-smi -i 6 --query-gpu=memory.free --format=csv,noheader,nounits) -ge 30000 ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 taskset -c 168,169 $RP experiments/p3_ped_exam/archive/dose_r3d2.py render
export P3_SET=nq4_p3_dose_r3d2 P5_SET=nq4_p3_dose_r3d2 CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 P3_HEAD_TOL=0.5
rm -rf $DATA_DIR/processed/nq4_p3_dose_r3d2
$JV experiments/p3_ped_exam/archive/dose_r3d2.py stage
taskset -c 168,169 $JV -m experiments.p3_ped_exam.archive.nq4_p3 index --processed-root $DATA_DIR/processed/waymo_ds/training
taskset -c 168,169 $OP scripts/p5_openpilot.py --models cinque lebowski --workers 1 --arrays temporal plan lead lead_prob
$JV -m jevdrive.p5_openpilot finalize --models cinque,lebowski --arrays temporal,plan,lead,lead_prob
taskset -c 168,169 $JV -m experiments.p3_ped_exam.archive.nq4_p3 exam
taskset -c 168,169 $JV -m experiments.p3_ped_exam.archive.ped_dose --exam-dir $(ls -td $DATA_DIR/runs/nq4/nq4_p3_dose_r3d2-exam/*/ | head -1) --out $RD/readout
$JV experiments/p3_ped_exam/archive/dose_r3d2.py compare --cells $CELLS
touch $RD/DONE
"""


def chain(a):
    RD.mkdir(parents=True, exist_ok=True)
    (RD / "run.sh").write_text(CHAIN)
    (RD / "run.sh").chmod(0o755)
    print(RD / "run.sh")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("select", "compare"):
        sp.add_parser(c).add_argument("--cells", type=Path, default=DO / "readout_stage2/cells.csv")
    sp.choices["select"].add_argument("--n", type=int, default=N_SEL)
    sp.add_parser("render").add_argument("--model", default="R3D2-big")
    sp.add_parser("stage")
    sp.add_parser("chain")
    a = ap.parse_args()
    if a.cmd == "compare":
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    {"select": select, "render": render, "stage": stage, "compare": compare, "chain": chain}[a.cmd](a)


if __name__ == "__main__":
    main()
