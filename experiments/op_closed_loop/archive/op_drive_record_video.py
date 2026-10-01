"""Stream BGRA chase frames to H.264, then label them from frame-aligned control logs."""
import argparse
import bisect
import json
from pathlib import Path
import re
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1280, 720, 10


def output(path):
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.codec_context.thread_count = 2
    stream.options = {"crf": "24", "preset": "veryfast"}
    return container, stream


def write(container, stream, frame):
    for packet in stream.encode(frame):
        container.mux(packet)


def finish(container, stream):
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def encode(path):
    container, stream = output(path)
    size = WIDTH * HEIGHT * 4
    source = sys.stdin.buffer
    try:
        while True:
            raw = source.read(size)
            if not raw:
                break
            if len(raw) != size:
                raise ValueError("Incomplete BGRA frame")
            pixels = np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH, 4)
            write(container, stream, av.VideoFrame.from_ndarray(pixels, format="bgra"))
    finally:
        finish(container, stream)


def annotate(attempt):
    root = Path(attempt)
    summary = json.loads((root / "video_summary.json").read_text())
    if summary["error"] or summary["dropped"] or summary["frames"] < 10:
        raise RuntimeError("Recording failed QA: " + json.dumps(summary))
    rows = lambda name: [json.loads(line) for line in (root / name).open()]
    ticks, plans, frames = rows("ticks.jsonl"), rows("plans.jsonl"), rows("video_frames.jsonl")
    tick_ids, plan_ids = [r["frame"] for r in ticks], [r["frame"] for r in plans]
    source = av.open(str(root / "chase_raw.mp4"))
    container, stream = output(root / "chase.mp4")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    route = root.parent.name
    run_config = json.loads((root / "route_result.json").read_text()).get("config", {})
    seed = run_config.get("tm_seed", 0)
    signals = {0: "GREEN", 1: "YELLOW", 2: "RED"}
    count = 0
    pixel_stds = []
    for count, frame in enumerate(source.decode(video=0), 1):
        fid = frames[count - 1]["frame"]
        tick = ticks[max(bisect.bisect_right(tick_ids, fid) - 1, 0)]
        plan = plans[max(bisect.bisect_right(plan_ids, fid) - 1, 0)]
        pixels = frame.to_ndarray(format="rgb24")
        if count % 50 == 1:
            pixel_stds.append(float(pixels.std()))
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, WIDTH, 82), fill=(12, 16, 24))
        owner = "OPENPILOT" if plan["lat"] == "op" else "ROUTE"
        why = plan.get("lat_why") or "lane follow"
        context = tick.get("ctx", {})
        light = signals.get(context.get("tl"), "--")
        title = "Route %s | seed %s | t=%.1fs | %.1f km/h" % (route, seed, tick["t"], tick["v"] * 3.6)
        status = "Steering: %s (%s) | speed limit: %s | light: %s | release: %s" % (
            owner, why, plan["src"], light, plan.get("rel") or "--")
        draw.text((14, 9), title, font=font, fill="white")
        draw.text((14, 43), status, font=font, fill=(110, 220, 255))
        write(container, stream, av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24"))
        if count == 1 or count == len(frames) // 2:
            image.save(root / ("preview-%d.jpg" % count))
    finish(container, stream)
    source.close()
    if count != len(frames) or min(pixel_stds) < 5:
        raise RuntimeError("Invalid video frame count or uniform render")
    result = {"frames": count, "video_s": count / FPS, "sim_s": ticks[-1]["t"],
              "pixel_std_samples": pixel_stds, "bytes": (root / "chase.mp4").stat().st_size}
    (root / "video_qa.json").write_text(json.dumps(result, indent=2))
    make_gif(root, ticks, frames)
    print(json.dumps(result), flush=True)


def make_gif(root, ticks, frames):
    """Locate a representative infraction approximately from its reported position."""
    record = json.loads((root / "results.json").read_text())["_checkpoint"]["records"][0]
    messages = [(kind, message) for kind in ("collisions_vehicle", "collisions_pedestrian",
                "collisions_layout", "red_light", "outside_route_lanes")
                for message in record["infractions"].get(kind, [])]
    center = ticks[len(ticks) // 2]["t"]
    reason = "route midpoint; no spatially localized infraction"
    for kind, message in messages:
        match = re.search(r"x=([-\d.]+), y=([-\d.]+)", message)
        if not match:
            continue
        point = np.array([float(x) for x in match.groups()])
        distances = np.linalg.norm(np.array([r["truth"][:2] for r in ticks]) - point, axis=1)
        # First arrival near the infraction avoids centering on later stationary ticks.
        index = int(np.flatnonzero(distances <= distances.min() + .5)[0])
        center, reason = ticks[index]["t"], kind + ": " + message
        break
    start = max(0.0, center - 14.0)
    end = min(ticks[-1]["t"], start + 20.0)
    tick_ids = [r["frame"] for r in ticks]
    images = []
    with av.open(str(root / "chase.mp4")) as source:
        for index, frame in enumerate(source.decode(video=0)):
            row = ticks[max(bisect.bisect_right(tick_ids, frames[index]["frame"]) - 1, 0)]
            if index % 2 == 0 and start <= row["t"] <= end:
                image = Image.fromarray(frame.to_ndarray(format="rgb24")).resize((640, 360))
                images.append(image.convert("P", palette=Image.ADAPTIVE, colors=128))
    if not images:
        raise RuntimeError("No frames selected for diagnostic GIF")
    images[0].save(root / "event.gif", save_all=True, append_images=images[1:],
                   duration=200, loop=0, optimize=True, disposal=2)
    (root / "event_clip.json").write_text(json.dumps({"center_s": center, "start_s": start,
        "end_s": end, "frames": len(images), "reason": reason,
        "timing": "approximate, inferred from infraction position and actual ego trajectory",
        "bytes": (root / "event.gif").stat().st_size}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["encode", "annotate"])
    parser.add_argument("path")
    args = parser.parse_args()
    (encode if args.mode == "encode" else annotate)(args.path)
