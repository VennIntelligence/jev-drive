"""Real frames for the HUGSIM Waymo / PandaSet / KITTI-360 scenes (real-vs-render extension, same protocol as nuScenes).
Run on the box with $DATA_DIR/envs/waymo/bin/python (needs remotezip) and the proxy on for the commands that say so:
  pandaset   range-reads only the three front cameras of the HUGSIM sequences out of the 44 GB HF zip -> datasets/rvr_real/pandaset/<seq>/<cam>/NN.jpg
  kitti360   range-reads image_00 / image_01 data_rect PNGs of the HUGSIM frame ranges from the official S3 zip -> datasets/rvr_real/kitti360/<cam>/<frame>.png
  waymo      downloads the 22 HUGSIM segments (v1.4.3 training/validation tfrecords) via gcloud -> datasets/rvr_real/waymo/<segment>.tfrecord
"""
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
HS = D / "datasets/hugsim/scenes"
OUT = D / "datasets/rvr_real"
PANDA_URL = "https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip"
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync_image_0{c}.zip"


def open_zip(url, tries=12):
    import time
    from remotezip import RemoteZip
    for i in range(tries):
        try:
            return RemoteZip(url)
        except Exception as e:  # flaky proxy / S3: retry
            print("open_zip retry", i, str(e)[:80], flush=True)
            time.sleep(5 + 5 * i)
    raise RuntimeError(url)


def scenes(ds):
    return sorted(p.name for p in (HS / ds).iterdir() if p.is_dir() and p.name != ds)


def pandaset():
    from remotezip import RemoteZip
    z0 = open_zip(PANDA_URL)
    pat = re.compile(r"pandaset/(\d+)/camera/(front_camera|front_left_camera|front_right_camera)/[^/]+$")
    want = set(scenes("pandaset"))
    names = [n for n in z0.namelist() if (m := pat.match(n)) and m.group(1) in want]
    print("pandaset members", len(names), flush=True)

    import threading, time
    tl = threading.local()

    def get(n):
        dst = OUT / "pandaset" / n.split("/", 1)[1].replace("/camera/", "/")
        for attempt in range(8):
            if dst.exists():
                break
            try:
                if getattr(tl, "z", None) is None:
                    tl.z = open_zip(PANDA_URL)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tl.z.read(n))
            except Exception as e:  # flaky proxy: reopen and retry
                tl.z = None
                time.sleep(2 + attempt * 3)
                if attempt == 7:
                    raise
        return n
    with ThreadPoolExecutor(12) as ex:
        for i, _ in enumerate(ex.map(get, names)):
            if i % 200 == 0:
                print("pandaset", i, flush=True)


def kitti360():
    import threading, time
    want = set()
    for s in scenes("kitti360"):
        a, b = map(int, s.split("_")[1:3])
        want |= set(range(a, b + 1))
    tl = threading.local()
    for c in (0, 1):
        url = KITTI_URL.format(c=c)
        z = open_zip(url)
        names = {int(re.search(r"(\d+)\.png$", n).group(1)): n for n in z.namelist() if "data_rect" in n and n.endswith(".png")}
        print("kitti360 cam", c, "members", len(names), "wanted", len(want), "missing", len(want - set(names)), flush=True)
        (OUT / "kitti360" / f"cam_{c}").mkdir(parents=True, exist_ok=True)

        def get(f):
            dst = OUT / "kitti360" / f"cam_{c}" / f"{f:010d}.png"
            for attempt in range(8):
                if dst.exists():
                    break
                try:
                    if getattr(tl, "u", None) != url:
                        tl.z, tl.u = open_zip(url), url
                    dst.write_bytes(tl.z.read(names[f]))
                except Exception:
                    tl.u = None
                    time.sleep(2 + attempt * 3)
                    if attempt == 7:
                        raise
            return f
        with ThreadPoolExecutor(8) as ex:
            for i, _ in enumerate(ex.map(get, sorted(want & set(names)))):
                if i % 500 == 0:
                    print("kitti360", c, i, flush=True)


def waymo():
    env = dict(os.environ)
    ids = sorted({s.split("_")[0] for s in scenes("waymo")})
    (OUT / "waymo").mkdir(parents=True, exist_ok=True)
    todo = {}
    for i in ids:
        for split in ("training", "validation"):
            r = subprocess.run(["gcloud", "storage", "ls", f"gs://waymo_open_dataset_v_1_4_3/individual_files/{split}/segment-{i}*"],
                               capture_output=True, text=True, env=env)
            if r.stdout.strip():
                todo[i] = r.stdout.split()[0]
                break
        print(i, todo.get(i), flush=True)
    json.dump(todo, open(OUT / "waymo" / "segments.json", "w"), indent=1)

    def cp(kv):
        i, u = kv
        dst = OUT / "waymo" / u.rsplit("/", 1)[1]
        if not dst.exists():
            subprocess.run(["gcloud", "storage", "cp", u, str(dst) + ".tmp"], check=True, env=env)
            os.replace(str(dst) + ".tmp", dst)
        print("got", i, flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(cp, todo.items()))


if __name__ == "__main__":
    {"pandaset": pandaset, "kitti360": kitti360, "waymo": waymo}[sys.argv[1]]()
