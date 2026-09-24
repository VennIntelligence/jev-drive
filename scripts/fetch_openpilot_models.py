"""Fetch openpilot driving-model ONNX files (no login needed), sha256-checked.

openpilot keeps its LFS objects on huggingface.co/commaai/openpilot-lfs; a pointer (oid, size) is resolved
through the LFS batch API. Cinque is published separately on HF and is fetched through hf-mirror.com
(domestic, direct) with parallel range requests, which is ~10x faster on the box than one proxied stream.

  small     driving_supercombo.onnx on master (30M params, on-device model)
  lebowski  big_driving_supercombo.onnx from PR #38268 (877M params, shipped as 0.11.2 on chestnut)
  cinque    big_driving_supercombo.onnx "Cinque Terre v3" (382M params, master big model since PR #38932),
            HF commaai/openpilot_driving_models, checkpoint f78ed37d
"""
import argparse, re, sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.openpilot.dl import download, hf_url

LFS = "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs/objects/batch"
RAW = "https://raw.githubusercontent.com/commaai/openpilot/master/openpilot/selfdrive/modeld/models/"
MODELS = {
    "small": dict(pointer=RAW + "driving_supercombo.onnx"),
    "lebowski": dict(oid="a501760a9d1d5fef0eab2b8c5d122d06124fc26dc8e0782e0aa94b82a208f0ff", size=1757355221),
    "cinque": dict(oid="404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4", size=766354845,
                   href=hf_url("commaai/openpilot_driving_models",
                               "f78ed37d-afad-4dbc-8050-40ea885eedde/12864/big_driving_supercombo.onnx")),
}


def resolve(spec):
    if "oid" in spec:
        return spec["oid"], spec["size"]
    txt = requests.get(spec["pointer"], timeout=30).text
    return re.search(r"sha256:([0-9a-f]{64})", txt)[1], int(re.search(r"size (\d+)", txt)[1])


def lfs_href(oid, size):
    req = {"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}]}
    r = requests.post(LFS, json=req, headers={"Accept": "application/vnd.git-lfs+json"}, timeout=60)
    return r.json()["objects"][0]["actions"]["download"]["href"]


def fetch(name, out_dir, streams):
    oid, size = resolve(MODELS[name])
    href = MODELS[name].get("href") or lfs_href(oid, size)
    dst = download(href, out_dir / f"{name}.onnx", size, oid, streams)
    print(f"{name}: {size / 1e6:.0f} MB -> {dst} (sha256 ok, oid {oid[:12]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "data/models/openpilot")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("names", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    for n in a.names:
        fetch(n, a.out, a.streams)
