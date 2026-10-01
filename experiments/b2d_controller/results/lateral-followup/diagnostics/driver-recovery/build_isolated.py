#!/usr/bin/env python3
"""Build private NVIDIA userspace from an already extracted official runfile.

Copies files and reads ELF metadata only; never invokes GPU libraries/installers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
from datetime import datetime, timezone

VERSION = "580.159.03"
DEFAULT_OUTPUT = "/data/tools/nvidia-userspace-580.159.03/isolated"
OFFICIAL_URL = f"https://download.nvidia.com/XFree86/Linux-x86_64/{VERSION}/NVIDIA-Linux-x86_64-{VERSION}.run"
OFFICIAL_SHA256 = "32c85d99b0f640c9501f61b39ddad208fd0288d015c4fbc5fd0435c07783fa77"
VENDOR = re.compile(r"^(?:libnvidia-[\w-]+|libGLX_nvidia|libEGL_nvidia|libGLESv[12](?:_CM)?_nvidia|libcuda|libnvcuvid)\.so(?:\..+)?$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def elf64_x86(path: Path) -> bool:
    with path.open("rb") as f:
        header = f.read(20)
    return (len(header) == 20 and header[:6] == b"\x7fELF\x02\x01"
            and struct.unpack("<H", header[18:20])[0] == 62)


def metadata(path: Path) -> dict:
    proc = subprocess.run(["readelf", "-d", str(path)], check=True,
                          capture_output=True, text=True)
    values = lambda tag: re.findall(r"\(" + tag + r"\).*?\[(.*?)\]", proc.stdout)
    sonames = values("SONAME")
    if len(sonames) != 1 or Path(sonames[0]).name != sonames[0]:
        raise ValueError(f"Expected one plain SONAME: {path}: {sonames}")
    return {"soname": sonames[0], "needed": values("NEEDED"),
            "rpath": values("RPATH"), "runpath": values("RUNPATH")}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    tool = manifest["nvidia_smi"]
    tool_link = output / "bin" / "nvidia-smi"
    if (not tool_link.is_symlink() or str(tool_link.readlink()) != tool["source"]
            or not elf64_x86(tool_link) or sha256(tool_link) != tool["sha256"]):
        raise ValueError("Private nvidia-smi source/link/hash mismatch")
    for row in manifest["libraries"]:
        path = output / "lib" / row["filename"]
        if not elf64_x86(path) or sha256(path) != row["sha256"]:
            raise ValueError(f"Library hash/ELF mismatch: {path}")
        if metadata(path) != row["elf"]:
            raise ValueError(f"ELF metadata mismatch: {path}")
    for name, target in manifest["symlinks"].items():
        path = output / "lib" / name
        if not path.is_symlink() or path.readlink().as_posix() != target:
            raise ValueError(f"Symlink mismatch: {path}")
    expected_names = {r["filename"] for r in manifest["libraries"]} | set(manifest["symlinks"])
    actual_names = {p.name for p in (output / "lib").iterdir()}
    if expected_names != actual_names:
        raise ValueError("Unexpected/missing files in private library directory")
    for name, digest in manifest["generated_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Generated file mismatch: {name}")
    return {"status": "static_verified", "libraries": len(manifest["libraries"]),
            "soname_links": len(manifest["symlinks"]),
            "manifest_sha256": sha256(output / "manifest.json"),
            "gpu_or_library_initialization_performed": False}


def build(source: Path, output: Path, icd_template: Path | None) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing existing output directory: {output}")
    if not source.is_dir():
        raise ValueError(f"Missing extracted input directory: {source}")
    smi = source / "nvidia-smi"
    if not smi.is_file() or smi.is_symlink() or not elf64_x86(smi):
        raise ValueError("Expected top-level regular ELF64 x86-64 nvidia-smi")
    smi_dynamic = subprocess.run(["readelf", "-d", str(smi)], check=True,
                                 capture_output=True, text=True).stdout
    smi_record = {"source": str(smi), "sha256": sha256(smi),
                  "bytes": smi.stat().st_size,
                  "needed": re.findall(r"\(NEEDED\).*?\[(.*?)\]", smi_dynamic)}
    libraries, ignored = [], []
    for path in sorted(source.iterdir()):
        if not VENDOR.fullmatch(path.name) or not path.is_file():
            continue
        if path.is_symlink():
            ignored.append({"name": path.name, "reason": "source_symlink_not_copied"})
            continue
        if not elf64_x86(path):
            ignored.append({"name": path.name, "reason": "not_ELF64_little_endian_x86_64"})
            continue
        if re.search(r"\.\d{3}\.\d+\.\d+$", path.name) and not path.name.endswith("." + VERSION):
            raise ValueError(f"Different NVIDIA driver version in source: {path.name}")
        libraries.append({"filename": path.name, "source": str(path),
                          "bytes": path.stat().st_size, "sha256": sha256(path),
                          "elf": metadata(path)})
    if not libraries:
        raise ValueError("No top-level NVIDIA ELF64 vendor libraries found")
    filenames = {r["filename"] for r in libraries}
    sonames, links = {}, {}
    for row in libraries:
        soname, filename = row["elf"]["soname"], row["filename"]
        if soname in sonames and sonames[soname] != filename:
            raise ValueError(f"Ambiguous SONAME {soname}")
        sonames[soname] = filename
        if soname != filename:
            if soname in filenames:
                raise ValueError(f"SONAME collides with source filename: {soname}")
            links[soname] = filename
    for required in ("libnvidia-ml.so.1", "libGLX_nvidia.so.0", "libEGL_nvidia.so.0"):
        if required not in sonames:
            raise ValueError(f"Required vendor library missing: {required}")
    available = filenames | set(sonames)
    missing = sorted({dep for r in libraries for dep in r["elf"]["needed"]
                      if VENDOR.fullmatch(dep) and dep not in available})
    if missing:
        raise ValueError(f"Private NVIDIA DT_NEEDED closure incomplete: {missing}")
    # Keep the archive's own API declaration; do not manufacture a Vulkan version.
    template = (icd_template or source / "nvidia_icd.json").resolve()
    icd = json.loads(template.read_text())
    if not isinstance(icd.get("ICD", {}).get("api_version"), str):
        raise ValueError("Extracted Vulkan ICD template lacks api_version")
    icd["ICD"]["library_path"] = str(output / "lib" / "libGLX_nvidia.so.0")
    egl = {"file_format_version": "1.0.0",
           "ICD": {"library_path": str(output / "lib" / "libEGL_nvidia.so.0")}}
    # All preflight above is read-only. An incomplete build is kept for inspection.
    output.mkdir(parents=True, exist_ok=False)
    (output / "lib").mkdir()
    (output / "bin").mkdir()
    (output / "bin" / "nvidia-smi").symlink_to(smi)
    for row in libraries:
        dest = output / "lib" / row["filename"]
        shutil.copy2(row["source"], dest)
        if sha256(dest) != row["sha256"]:
            raise ValueError(f"Source changed during copying: {row['source']}")
    for soname, filename in sorted(links.items()):
        (output / "lib" / soname).symlink_to(filename)
    write_json(output / "nvidia_icd.json", icd)
    write_json(output / "10_nvidia.json", egl)
    env = ("# Source in the shell launching the server; no system files are changed.\n"
           f"export PATH={shlex.quote(str(output / 'bin'))}:\"$PATH\"\n"
           f"export LD_LIBRARY_PATH={shlex.quote(str(output / 'lib'))}"
           '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n'
           f"export VK_DRIVER_FILES={shlex.quote(str(output / 'nvidia_icd.json'))}\n"
           f"export VK_ICD_FILENAMES={shlex.quote(str(output / 'nvidia_icd.json'))}\n"
           f"export __EGL_VENDOR_LIBRARY_FILENAMES={shlex.quote(str(output / '10_nvidia.json'))}\n")
    (output / "env.sh").write_text(env)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "driver_version": VERSION, "source_directory": str(source),
        "output_directory": str(output), "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__)),
        "official_runfile_url": OFFICIAL_URL, "official_runfile_sha256": OFFICIAL_SHA256,
        "runfile_hash_verified_by_this_script": False,
        "icd_template": str(template), "icd_template_sha256": sha256(template),
        "libraries": libraries, "symlinks": links, "ignored_vendor_entries": ignored,
        "nvidia_smi": smi_record,
        "system_dependencies": sorted({dep for r in libraries for dep in r["elf"]["needed"]
                                       if dep not in available}),
        "private_vendor_DT_NEEDED_missing": missing,
        "generated_sha256": {name: sha256(output / name) for name in
                             ("env.sh", "nvidia_icd.json", "10_nvidia.json")},
        "scope": "Static ELF/copy audit only; no GPU, installer, linker-cache or system mutation.",
        "limitations": ["DT_NEEDED closure does not enumerate every runtime dlopen.",
                        "Neutral GLVND/Vulkan loader and non-NVIDIA dependencies remain system-provided.",
                        "Actual server maps must establish which vendor files were loaded."]}
    write_json(output / "manifest.json", manifest)
    result = verify(output)
    write_json(output / "static-verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Already extracted runfile top-level directory")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--icd-template", type=Path, help="Optional extracted nvidia_icd.json path")
    parser.add_argument("--verify-only", action="store_true", help="Read-only hash/ELF/link verification")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.out.resolve())
    else:
        if args.source is None:
            parser.error("--source is required for building")
        result = build(args.source.resolve(), args.out.absolute(), args.icd_template)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
