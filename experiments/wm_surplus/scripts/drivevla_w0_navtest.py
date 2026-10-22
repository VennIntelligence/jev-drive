"""navtest poses of DriveVLA-W0 (Emu3 world-model VLA, flow-matching action expert, checkpoint
Emu3_Flow_Matching_Action_Expert_PDMS_87.2), run as shipped: the bench `cmd` runner of jevdrive/bench/agents/DriveVLA-W0.json,
env envs/p0-drivevla-w0.

Their inference (inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh) is torchrun over
inference_action_navsim_flow_matching_vava.py: dataset = the shipped test meta pickle (one entry per navtest token, in the order of
navtest.yaml, VQ codes of CAM_F0), output = one `<token>.json` per token (key `action`: (8, 3) relative (dx, dy, dyaw) per 0.5 s
step, each in the previous step's frame). This file restricts the run to a chunk by writing the chunk's entries of that pickle and
the matching token list (same order) to a scratch dir and passing them as --test_data_pkl / --token_yaml, runs the entry with
`torchrun --standalone --nproc_per_node 1`, then converts each JSON exactly as their navsim agent does
(navsim/agents/emu_vla_agent.py, anchor_based=False: convert_relative_to_absolute_se2_trajectory) into NAVSIM poses
(x forward, y left, heading at 0.5 .. 4 s). Flow-matching noise is unseeded torch.randn, as shipped.

Path-only settings (no code changed): their env-var paths, VLA_PATH_REPLACEMENTS (pickle paths -> our unzipped VQ codes),
PYTHONPATH += <repo>/utils (setup_paths_early expects that directory under its old name `train/`).

    python drivevla_w0_navtest.py --tokens chunk.tsv --out chunk.npz
    python drivevla_w0_navtest.py --compare chunk.npz          # max |our pose - pose from their shipped JSON| on the same tokens
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
ROOT = D / "third_party/drivevla-w0"
W = D / "models/drivevla-w0"
HUB = W / "Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
PKL = W / "navsim_emu_vla_256_144_test_pre_1s.pkl"
YAML = ROOT / "inference/navsim/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
SHIPPED = W / "shipped_json/json_output_step_10"      # their released per-token outputs (flow_matching_json_output_PDMS_87.2.zip)


def rel_to_abs(rel) -> np.ndarray:
    """emu_vla_agent.convert_relative_to_absolute_se2_trajectory, verbatim arithmetic (float64)."""
    rel = np.asarray(rel, np.float64)
    out = np.zeros_like(rel)
    x = y = yaw = 0.0
    for i, (dx, dy, dyaw) in enumerate(rel):
        x += np.cos(yaw) * dx - np.sin(yaw) * dy
        y += np.sin(yaw) * dx + np.cos(yaw) * dy
        yaw = (yaw + dyaw + np.pi) % (2 * np.pi) - np.pi
        out[i] = [x, y, yaw]
    return out


def poses_of(d: Path, toks) -> np.ndarray:
    return np.stack([rel_to_abs(json.loads((d / f"{t}.json").read_text())["action"]) for t in toks]).astype(np.float32)


def compare(npz: str) -> None:
    z = np.load(npz)
    toks = [t for t in z["tokens"].tolist() if (SHIPPED / f"{t}.json").exists()]
    idx = [z["tokens"].tolist().index(t) for t in toks]
    diff = np.abs(z["poses"][idx] - poses_of(SHIPPED, toks))
    print(json.dumps(dict(n=len(toks), of=len(z["tokens"]), max_abs=float(diff.max()), max_abs_xy=float(diff[..., :2].max()),
                          mean_abs_xy=float(diff[..., :2].mean()), max_abs_heading=float(diff[..., 2].max()),
                          mean_final_l2=float(np.linalg.norm(diff[:, -1, :2], axis=-1).mean()))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens")
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    out = Path(a.out)
    if out.exists():
        return
    want = {ln.split("\t")[0] for ln in Path(a.tokens).read_text().split("\n") if ln.strip()}
    allt = yaml.safe_load(YAML.read_text())["tokens"]
    data = pickle.load(open(PKL, "rb"))
    assert len(allt) == len(data), (len(allt), len(data))
    idx = [i for i, t in enumerate(allt) if t in want]
    if len(idx) != len(want):
        sys.exit(f"{len(want) - len(idx)} chunk tokens are not in their navtest list")
    toks = [allt[i] for i in idx]
    wd = out.with_name(f".{out.stem}_w0")
    shutil.rmtree(wd, ignore_errors=True)
    (wd / "json").mkdir(parents=True)
    pickle.dump([data[i] for i in idx], open(wd / "test.pkl", "wb"))
    (wd / "tokens.yaml").write_text(yaml.safe_dump({"tokens": toks}))
    env = dict(os.environ, DRIVEVLA_ROOT=str(ROOT), HF_HUB_OFFLINE="1",
               PYTHONPATH=f"{ROOT}/inference/navsim/navsim:{ROOT}:{ROOT}/utils",
               VLA_ACTION_TOKENIZER=str(ROOT / "configs/fast"), VLA_VLM_MODEL=str(W / "Emu3_NuPlan_Pretrain_Cktps"),
               VLA_NORM_STATS=str(ROOT / "configs/normalizer_navsim_trainval/norm_stats.json"), VLA_TOKEN_YAML=str(wd / "tokens.yaml"),
               VLA_NUM_WORKERS=str(a.workers), VLA_BATCH_SIZE="1",
               VLA_ANCHOR_CLUSTER_PATH=str(W / "cluster_centers_8192.npy"),
               VLA_ANCHOR_METRIC_SCORE_PATH=str(W / "formatted_pdm_score_8192.npy"),   # listed by the script, unused by flow matching
               VLA_PATH_REPLACEMENTS=f"/mnt/vdb1/yingyan.li/repo/VLA/:{W}/")
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
                    "inference/vla/inference_action_navsim_flow_matching_vava.py", "--emu_hub", str(HUB),
                    "--output_dir", str(wd / "json"), "--test_data_pkl", str(wd / "test.pkl")], cwd=ROOT, env=env, check=True)
    wall = time.time() - t0
    P = poses_of(wd / "json", toks)
    assert P.shape == (len(toks), 8, 3) and np.isfinite(P).all()
    print(f"{len(toks)} tokens, {wall:.0f} s incl. load, {wall / len(toks):.2f} s/token", flush=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.npz")
    np.savez(tmp, tokens=np.asarray(toks), poses=P, wall_s=wall)
    os.replace(tmp, out)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main()
