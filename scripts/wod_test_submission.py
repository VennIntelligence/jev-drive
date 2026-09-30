"""Build WOD-E2E challenge submissions from the zero-shot openpilot test-split predictions
(todos/2026-09-24-zeroshot-exam/wod-e2e.md, "test 集提交包"). Does NOT upload anything: it only writes and
validates the tar.gz, using the official proto compiled from the waymo-open-dataset repo
(jevdrive.waymo.write_submission / read_submission).

  python scripts/wod_test_submission.py --model cinque --out ~/data/runs/zeroshot-exam/wod-test/cinque.tar.gz
  python scripts/wod_test_submission.py --model cinque_valcal --out ~/data/runs/zeroshot-exam/wod-test/cinque.valcal.tar.gz
  python scripts/wod_test_submission.py --model lebowski --out ~/data/runs/zeroshot-exam/wod-test/lebowski.tar.gz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import waymo as W  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402

MODEL_META = {
    "cinque": dict(unique_method_name="openpilot_cinque_v3_zeroshot",
                   description="Zero-shot transfer of the publicly released openpilot Cinque v3 (382M) "
                                "planning model to WOD-E2E. FRONT, FRONT_LEFT and FRONT_RIGHT images are "
                                "reprojected into the openpilot camera view; the pretrained model runs "
                                "with 10 s of causal history. Its mean plan is converted to 20 ego-frame "
                                "waypoints at 4 Hz. No WOD-E2E training, fine-tuning or fitted calibration "
                                "was used for this submission.",
                   public_model_names=["openpilot Cinque v3"], num_model_parameters="382M"),
    "cinque_valcal": dict(unique_method_name="openpilot_cinque_v3_valcal_x1p06",
                          description="Frozen openpilot Cinque v3 (382M) planning model transferred to "
                                      "WOD-E2E without training or fine-tuning its weights on WOD. FRONT, FRONT_LEFT "
                                      "and FRONT_RIGHT images are reprojected into the openpilot camera "
                                      "view; the model uses 10 s of causal history. Its mean plan is "
                                      "converted to 20 ego-frame waypoints at 4 Hz. Forward x positions "
                                      "are multiplied by 1.06, selected by two-fold sequence cross-fit "
                                      "on the labeled WOD-E2E validation set. No test labels or "
                                      "leaderboard scores were used to select the calibration.",
                          public_model_names=["openpilot Cinque v3"], num_model_parameters="382M"),
    "lebowski": dict(unique_method_name="openpilot_lebowski_zeroshot",
                     description="Zero-shot transfer of the publicly released openpilot Lebowski (877M, "
                                  "0.11.2 big model) planning model to WOD-E2E via the same camera "
                                  "adapter as Cinque v3. No WOD-E2E training, fine-tuning or fitted "
                                  "calibration was used for this submission.",
                     public_model_names=["openpilot Lebowski"], num_model_parameters="877M"),
}
COMMON_META = dict(account_name="gaochengzhi1999@gmail.com", authors=["Chengzhi Gao"], affiliation="Southeast University",
                   method_link="", uses_public_model_pretraining=False)


def build(model: str, out: Path) -> dict:
    sets = Z.load_sets()
    names = [str(n) for n in sets["test"]["name"]]
    source_model = "cinque" if model == "cinque_valcal" else model
    outdir = Z.root("preds", f"op_{source_model}")
    missing = [n for n in names if not (outdir / f"{n}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} of {len(names)} predictions missing for {model}, "
                            f"e.g. {missing[:5]} -- rerun scripts/wod_zeroshot_openpilot.py --set test")
    traj = np.stack([np.load(outdir / f"{n}.npz")["wod"] for n in names])
    if model == "cinque_valcal":
        traj = traj.copy()
        traj[..., 0] *= 1.06
    meta = {**COMMON_META, **MODEL_META[model]}
    path = W.write_submission(names, traj, out, meta)
    got_names, got_traj, got_meta = W.read_submission(path)
    assert got_names == names, "round-trip: frame name order/content changed"
    assert np.allclose(got_traj, traj, atol=1e-5), "round-trip: trajectory values changed"
    assert got_meta["unique_method_name"] == meta["unique_method_name"], "round-trip: metadata changed"
    want = set(W.submission_frames())
    return {"model": model, "frames": len(names), "covers_all_submission_frames": want == set(names),
            "bytes": path.stat().st_size, "path": str(path), "roundtrip_ok": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_META))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    print(build(a.model, a.out))


if __name__ == "__main__":
    main()
