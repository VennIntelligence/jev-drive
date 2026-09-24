"""Build WOD-E2E challenge submissions from the zero-shot openpilot test-split predictions
(todos/2026-09-24-zeroshot-exam/wod-e2e.md, "test 集提交包"). Does NOT upload anything: it only writes and
validates the tar.gz, using the official proto compiled from the waymo-open-dataset repo
(jevdrive.waymo.write_submission / read_submission).

  python scripts/wod_test_submission.py --model cinque --out ~/data/runs/zeroshot-exam/wod-test/cinque.tar.gz
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
                   description="openpilot Cinque v3 (382M), run zero-shot (no fine-tuning, no fitted "
                                "parameters) on WOD-E2E via a rendered calib-frame adapter from the FRONT/"
                                "FRONT_LEFT/FRONT_RIGHT cameras; see todos/2026-09-24-zeroshot-exam/wod-e2e.md.",
                   public_model_names=["openpilot Cinque v3"], num_model_parameters="382000000"),
    "lebowski": dict(unique_method_name="openpilot_lebowski_zeroshot",
                     description="openpilot Lebowski (877M, 0.11.2 big model), run zero-shot on WOD-E2E via "
                                  "the same adapter as Cinque v3; see todos/2026-09-24-zeroshot-exam/wod-e2e.md.",
                     public_model_names=["openpilot Lebowski"], num_model_parameters="877000000"),
}
COMMON_META = dict(account_name="liuziyue6991@gmail.com", authors=["Gaochengzhi"], affiliation="VennIntelligence",
                   method_link="", uses_public_model_pretraining=True)


def build(model: str, out: Path) -> dict:
    sets = Z.load_sets()
    names = [str(n) for n in sets["test"]["name"]]
    outdir = Z.root("preds", f"op_{model}")
    missing = [n for n in names if not (outdir / f"{n}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} of {len(names)} predictions missing for {model}, "
                            f"e.g. {missing[:5]} -- rerun scripts/wod_zeroshot_openpilot.py --set test")
    traj = np.stack([np.load(outdir / f"{n}.npz")["wod"] for n in names])
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
